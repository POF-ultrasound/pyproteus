"""MATLAB k-Wave <-> k-wave-python 0.6.2 ``kspaceFirstOrder()`` conventions.

Measured by the tracer bullet (BUGS.md B-10, docs/dev/LOG.md 2026-11-20)
against MATLAB k-Wave 1.4.1 ``kspaceFirstOrder3DC`` on the same C++ binary:

=========  ===================  ==========================  ==========================
backend    source rows in       time-varying source scaling  ``result["p"]``
=========  ===================  ==========================  ==========================
``cpp``    MATLAB find (F)      none: caller must scale      ``(Nt, n_sensor)``, columns F
``python`` C (row-major)        applied internally           ``(n_sensor, Nt)``, rows C
MATLAB     find (F)             applied by MATLAB front end  ``(n_sensor, Nt)``, rows F
=========  ===================  ==========================  ==========================

With these mappings both backends reproduce MATLAB's sensor data to < 1e-6
relative (single precision); the ``cpp`` HDF5 source inputs are bit-identical
to MATLAB's. The functions below take and return MATLAB/PROTEUS conventions:
rows in ``find(mask)`` order, unscaled source signals, ``(n_sensor, Nt)``.

Meant to be lifted into ``pyproteus.backends.kwave`` (M3). Pin k-wave-python
to exactly 0.6.2: these conventions are not a stable upstream contract.
"""

from __future__ import annotations

import numpy as np

try:
    from . import ordering
except ImportError:  # pragma: no cover
    import ordering  # type: ignore[no-redef]

__all__ = [
    "additive_source_scale",
    "medium_for_kwp",
    "sensor_rows_from_kwp",
    "source_rows_to_kwp",
]

BACKENDS = ("cpp", "python")


def additive_source_scale(
    kind: str, mask: np.ndarray, sound_speed, dt: float, spacing
) -> np.ndarray:
    """Per-row factor k-Wave applies to a time-varying *additive* source, rows in find order.

    Not taken from k-Wave MATLAB source. Sources: the k-Wave User Manual 1.1
    (Treeby, Jaros, Cox, 2016), chapter on time-varying sources (source terms
    are mass/momentum injection rates: a pressure source is split over the
    ``ndim`` density components and converted with ``2*dt/(c0*dx)``), and
    k-wave-python's documented python-solver scaling (``kspace_solver.py``
    docstring, citing Cox et al., IEEE IUS 2018):

    * pressure (``kind="p"``): ``p_in = p * 2*dt / (ndim * c0 * dx)``
    * velocity (``kind="ux"/"uy"/"uz"``): ``u_in = u * 2*c0*dt / d``, with
      ``d`` = dx, dy, dz respectively

    ``c0`` is the sound speed at each source voxel and ``spacing`` the grid
    spacing per axis (``(dx, dy, dz)``; ``ndim = len(spacing)``). For the
    pressure source with an anisotropic grid k-Wave differs per density
    component, which a single factor cannot express: refused. MATLAB k-Wave applies
    this before writing the C++ input file; k-wave-python 0.6.2's ``cpp``
    backend does not, so the adapter multiplies the signals by this factor.
    Compute in float64 and let k-wave-python cast to float32 (as MATLAB does):
    the tracer bullet found the resulting HDF5 inputs bit-identical.
    """
    mask = np.asarray(mask, dtype=bool)
    spacing = tuple(float(d) for d in np.atleast_1d(spacing))
    if len(spacing) != mask.ndim:
        raise ValueError(f"spacing {spacing} does not match a {mask.ndim}-D mask")
    c = np.broadcast_to(np.asarray(sound_speed, dtype=np.float64), mask.shape)
    c_rows = c.ravel(order="F")[np.flatnonzero(mask.ravel(order="F"))]
    if kind == "p":
        if len(set(spacing)) != 1:
            raise ValueError("pressure-source scaling on an anisotropic grid is not supported")
        return 2.0 * dt / (len(spacing) * c_rows * spacing[0])
    axis = {"ux": 0, "uy": 1, "uz": 2}.get(kind)
    if axis is None or axis >= len(spacing):
        raise ValueError(f"unknown source kind {kind!r} for a {len(spacing)}-D grid")
    return 2.0 * c_rows * dt / spacing[axis]


def medium_for_kwp(medium: dict, backend: str) -> dict:
    """Medium fields for ``kspaceFirstOrder``, mirroring what MATLAB ``kspaceFirstOrder3DC`` does.

    PROTEUS (``define_medium.m``) sets ``alpha_mode='no_dispersion'`` when
    ``|alpha_power - 1| < 0.1``. The C++ input format has no field for it, so
    MATLAB's 3DC path silently runs with full absorption + dispersion
    (BUGS.md B-14). k-wave-python's cpp backend raises instead; the adapter
    drops ``alpha_mode`` for cpp to reproduce MATLAB. The python backend would
    honour it and so differ from MATLAB 3DC: refused.
    """
    out = dict(medium)
    mode = out.pop("alpha_mode", None)
    if mode is not None and backend == "python":
        raise ValueError(
            f"alpha_mode={mode!r}: MATLAB 3DC ignores it but the python backend applies it (B-14)"
        )
    return out


def source_rows_to_kwp(
    rows_find: np.ndarray,
    mask: np.ndarray,
    backend: str,
    *,
    kind: str,
    sound_speed,
    dt: float,
    spacing,
) -> np.ndarray:
    """MATLAB-convention source signals -> what ``kspaceFirstOrder(backend=...)`` expects.

    ``rows_find``: unscaled ``(n_source, M)`` signals, rows in ``find(mask)`` order
    (``source.p``, ``source.ux``... as PROTEUS builds them). Additive mode only
    (the only mode PROTEUS uses).
    """
    rows_find = np.asarray(rows_find, dtype=np.float64)
    n = int(np.count_nonzero(mask))
    if rows_find.ndim != 2 or rows_find.shape[0] != n:
        raise ValueError(
            f"expected (n_source, M) rows for {n} source voxels, got {rows_find.shape}"
        )
    if backend == "cpp":
        return rows_find * additive_source_scale(kind, mask, sound_speed, dt, spacing)[:, None]
    if backend == "python":
        return ordering.rows_f_to_c(rows_find, mask)
    raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")


def sensor_rows_from_kwp(
    p: np.ndarray, sensor_mask: np.ndarray, Nt: int, backend: str
) -> np.ndarray:
    """``result["p"]`` of ``kspaceFirstOrder(backend=...)`` -> ``(n_sensor, Nt)``, find order.

    For ``cpp``, k-wave-python 0.6.2 returns ``(Nt, n_sensor)`` with columns in
    F order: its own F->C reordering only fires when ``shape[0] == n_sensor``.
    When ``Nt == n_sensor`` it does fire, permuting the *time* axis with the
    sensor-mask permutation; that known permutation is undone here.
    """
    p = np.asarray(p)
    n = int(np.count_nonzero(sensor_mask))
    if backend == "cpp":
        if p.shape != (Nt, n):
            raise ValueError(f"expected cpp sensor data of shape {(Nt, n)}, got {p.shape}")
        if n == Nt and n > 1:  # undo k-wave-python's p[perm] on the time axis (B-10)
            perm = ordering.f_to_c_rows(sensor_mask)
            q = np.empty_like(p)
            q[perm] = p
            p = q
        return p.T
    if backend == "python":
        if p.shape != (n, Nt):
            raise ValueError(f"expected python sensor data of shape {(n, Nt)}, got {p.shape}")
        return ordering.rows_c_to_f(p, sensor_mask)
    raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
