"""MATLAB k-Wave <-> k-wave-python 0.6.2 ``kspaceFirstOrder()`` conventions.

Measured by the tracer bullet (``tracer-bullet-findings.md``, BUGS.md B-10)
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

__all__ = ["additive_source_scale", "sensor_rows_from_kwp", "source_rows_to_kwp"]

BACKENDS = ("cpp", "python")


def additive_source_scale(
    kind: str, mask: np.ndarray, sound_speed, dt: float, dx: float, ndim: int = 3
) -> np.ndarray:
    """Per-row factor k-Wave applies to a time-varying *additive* source, rows in find order.

    From the k-Wave manual / Treeby et al. (not from k-Wave MATLAB source):

    * pressure (``kind="p"``):            ``p_in = p * 2*dt / (ndim * c0 * dx)``
    * velocity (``kind="ux"/"uy"/"uz"``): ``u_in = u * 2*c0*dt / dx``

    ``c0`` is the sound speed at each source voxel and ``dx`` the grid spacing
    along the relevant axis (PROTEUS grids are isotropic). MATLAB k-Wave applies
    this before writing the C++ input file; k-wave-python 0.6.2's ``cpp``
    backend does not, so the adapter multiplies the signals by this factor.
    Compute in float64 and let k-wave-python cast to float32 (as MATLAB does):
    the tracer bullet found the resulting HDF5 inputs bit-identical.
    """
    mask = np.asarray(mask, dtype=bool)
    c = np.broadcast_to(np.asarray(sound_speed, dtype=np.float64), mask.shape)
    c_rows = c.ravel(order="F")[np.flatnonzero(mask.ravel(order="F"))]
    if kind == "p":
        return 2.0 * dt / (ndim * c_rows * dx)
    if kind in ("ux", "uy", "uz"):
        return 2.0 * c_rows * dt / dx
    raise ValueError(f"unknown source kind {kind!r}")


def source_rows_to_kwp(
    rows_find: np.ndarray,
    mask: np.ndarray,
    backend: str,
    *,
    kind: str,
    sound_speed,
    dt: float,
    dx: float,
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
        return rows_find * additive_source_scale(kind, mask, sound_speed, dt, dx)[:, None]
    if backend == "python":
        return ordering.rows_f_to_c(rows_find, mask)
    raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")


def sensor_rows_from_kwp(
    p: np.ndarray, sensor_mask: np.ndarray, Nt: int, backend: str
) -> np.ndarray:
    """``result["p"]`` of ``kspaceFirstOrder(backend=...)`` -> ``(n_sensor, Nt)``, find order.

    For ``cpp``, k-wave-python 0.6.2 returns ``(Nt, n_sensor)`` with columns in
    F order: its own C-order reordering only fires when ``shape[0] == n_sensor``,
    which then permutes the *time* axis. ``Nt == n_sensor`` is therefore refused.
    """
    p = np.asarray(p)
    n = int(np.count_nonzero(sensor_mask))
    if backend == "cpp":
        if n == Nt:
            raise ValueError(
                "Nt == n_sensor: k-wave-python 0.6.2 cpp output order is ambiguous (B-10)"
            )
        if p.shape != (Nt, n):
            raise ValueError(f"expected cpp sensor data of shape {(Nt, n)}, got {p.shape}")
        return p.T
    if backend == "python":
        if p.shape != (n, Nt):
            raise ValueError(f"expected python sensor data of shape {(n, Nt)}, got {p.shape}")
        return ordering.rows_c_to_f(p, sensor_mask)
    raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
