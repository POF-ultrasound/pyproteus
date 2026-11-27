"""k-Wave backend: a thin adapter over k-wave-python 0.6.2 ``kspaceFirstOrder()``.

Takes and returns MATLAB k-Wave / PROTEUS conventions, so ported PROTEUS code
(``run_simulation.m`` and its callers) can stay close to the MATLAB original:

* per-voxel rows (``source.p``, ``source.ux``/``uy``/``uz``, recorded
  ``sensor_data.p``) are in MATLAB ``find(mask)`` order (column-major);
* time-varying sources are additive and **unscaled**, exactly as passed to
  MATLAB ``kspaceFirstOrder3DC`` (k-Wave applies the source scaling);
* time series come back as ``(n_sensor, Nt)``, aggregates (``p_max``,
  ``p_min``) as ``(n_sensor,)``, ``float32`` (``DataCast='single'``).

Arrays are indexed ``[ix, iy, iz]`` throughout (same element positions as in
MATLAB); only the flattening order of the rows differs between front ends.

Why an adapter is needed (BUGS.md B-10, measured by the M0 tracer bullet
against MATLAB k-Wave 1.4.1 ``kspaceFirstOrder3DC`` on the same binary):

=========  ===================  ==========================  ==========================
front end  source rows in       time-varying source scaling  sensor data out
=========  ===================  ==========================  ==========================
MATLAB     find (F)             applied by MATLAB front end  ``(n_sensor, Nt)``, rows F
``cpp``    find (F)             none: caller must scale      ``(Nt, n_sensor)``, columns F
``python`` C (row-major)        applied internally           ``(n_sensor, Nt)``, rows C
=========  ===================  ==========================  ==========================

Backends of :func:`kspace_first_order`:

``"cpp"`` (parity target, = MATLAB ``kspaceFirstOrder3DC``/``3DG``)
    k-wave-python writes the HDF5 input file (``kspaceFirstOrder(...,
    save_only=True)``); the adapter then runs the k-Wave C++ binary itself with
    the output flags MATLAB derives from ``sensor.record`` (k-wave-python 0.6.2
    ignores ``sensor.record`` for ``cpp`` and always records ``p``) and reads
    the output file directly (which also avoids k-wave-python's output
    reordering, B-10).
``"python"`` (NumPy, or CuPy with ``device="gpu"``; in-process, for self-sense runs)
    k-wave-python's own solver, rows reordered F <-> C.

``alpha_mode`` is dropped on every backend (D-9, QUIRKS.md Q-1): MATLAB 3DC/3DG
ignore it, so PROTEUS runs full absorption with dispersion.

Pin k-wave-python to exactly 0.6.2: these conventions are not a stable upstream
contract. k-Wave and k-wave-python (LGPL) are called, never copied or
translated; the source scaling factors are documented in
:func:`additive_source_scale`.
"""

from __future__ import annotations

import contextlib
import functools
import os
import subprocess
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "BACKENDS",
    "RECORD_FLAGS",
    "KGrid",
    "additive_source_scale",
    "binary_command",
    "binary_path",
    "c_to_f_rows",
    "f_to_c_rows",
    "filter_time_series",
    "get_win",
    "kspace_first_order",
    "matlab_find",
    "medium_for_kwp",
    "read_cpp_output",
    "rows_c_to_f",
    "rows_f_to_c",
    "sensor_rows_from_kwp",
    "source_rows_to_kwp",
]

KWAVE_PYTHON_VERSION = "0.6.2"
BACKENDS = ("cpp", "python")
DEVICES = ("cpu", "gpu")
#: ``sensor.record`` entries supported (those PROTEUS uses), with the k-Wave C++
#: binary flag MATLAB ``kspaceFirstOrder3DC`` passes for each, in its order.
RECORD_FLAGS = {"p": "--p_raw", "p_max": "--p_max", "p_min": "--p_min"}
_TIME_SERIES = ("p",)
VELOCITY_KINDS = ("ux", "uy", "uz")
OMP_NAME = "kspaceFirstOrder-OMP"
CUDA_NAME = "kspaceFirstOrder-CUDA"
#: Environment variable overriding the k-Wave binary (e.g. the macOS wrapper
#: ``tools/tracer_bullet/bin/kspaceFirstOrder-OMP``).
BINARY_ENV = "PYPROTEUS_KWAVE_BINARY"


# ============================================================================ row order


def matlab_find(mask: np.ndarray) -> np.ndarray:
    """MATLAB ``find(mask)``: **1-based** column-major linear indices, ascending.

    (``pyproteus.numerics.find`` is the 0-based equivalent.)
    """
    return np.flatnonzero(np.asarray(mask, dtype=bool).ravel(order="F")) + 1


def _perm(mask: np.ndarray, src: str) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    shape = mask.shape
    dst = "C" if src == "F" else "F"
    dst_lin = np.flatnonzero(mask.ravel(order=dst))  # sorted, = rows of the destination
    src_lin = np.flatnonzero(mask.ravel(order=src))  # sorted, = rows of the source
    # source-order linear index of each voxel, listed in destination order
    src_of_dst = np.ravel_multi_index(np.unravel_index(dst_lin, shape, order=dst), shape, order=src)
    perm = np.searchsorted(src_lin, src_of_dst)  # its row in the source array
    if not np.array_equal(src_lin[perm], src_of_dst):  # pragma: no cover  (invariant)
        raise AssertionError("row permutation is not a bijection")
    return perm


def f_to_c_rows(mask: np.ndarray) -> np.ndarray:
    """Permutation ``perm`` with ``rows_c = rows_f[perm]``.

    ``rows_f``: one row per active voxel of ``mask`` in MATLAB ``find``
    (Fortran) order; ``rows_c``: the same rows in C (row-major) order. Any
    number of dimensions.
    """
    return _perm(mask, "F")


def c_to_f_rows(mask: np.ndarray) -> np.ndarray:
    """Permutation ``perm`` with ``rows_f = rows_c[perm]`` (inverse of :func:`f_to_c_rows`)."""
    return _perm(mask, "C")


def _reorder(rows: np.ndarray, perm: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows)
    if rows.shape[0] != perm.size:
        raise ValueError(f"{rows.shape[0]} rows for a mask with {perm.size} active voxels")
    return rows[perm]


def rows_f_to_c(rows_f: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Reorder per-voxel rows (axis 0) from MATLAB ``find`` order to C order."""
    return _reorder(rows_f, f_to_c_rows(mask))


def rows_c_to_f(rows_c: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Reorder per-voxel rows (axis 0) from C order to MATLAB ``find`` order."""
    return _reorder(rows_c, c_to_f_rows(mask))


# ============================================================================ conventions


def additive_source_scale(
    kind: str, mask: np.ndarray, sound_speed: Any, dt: float, spacing: Any
) -> np.ndarray:
    """Per-row factor k-Wave applies to a time-varying *additive* source, rows in find order.

    Not taken from k-Wave MATLAB source. Sources: the k-Wave User Manual 1.1
    (Treeby, Jaros, Cox, 2016), section on time-varying sources (source terms
    are mass/momentum injection rates: a pressure source is split over the
    ``ndim`` density components and converted with ``2*dt/(c0*dx)``), and
    k-wave-python's documented python-solver scaling (``kspace_solver.py``
    docstring, citing Cox et al., IEEE IUS 2018):

    * pressure (``kind="p"``): ``p_in = p * 2*dt / (ndim * c0 * dx)``
    * velocity (``kind="ux"/"uy"/"uz"``): ``u_in = u * 2*c0*dt / d``, with
      ``d`` = dx, dy, dz respectively

    ``c0`` is the sound speed at each source voxel, ``spacing`` the grid spacing
    per axis (``ndim = len(spacing)``). An anisotropic grid with a pressure
    source would need one factor per density component: refused (PROTEUS grids
    are isotropic). MATLAB k-Wave applies this before writing the C++ input
    file; k-wave-python 0.6.2's ``cpp`` backend does not. Computed in float64
    and cast to float32 when the input file is written; the resulting HDF5
    source datasets agree with MATLAB k-Wave's within float32 noise
    (``tests/oracle/test_kwave_backend.py``).
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


def medium_for_kwp(medium: Mapping[str, Any]) -> dict[str, Any]:
    """Medium fields for ``kspaceFirstOrder``, as MATLAB ``kspaceFirstOrder3DC`` sees them.

    PROTEUS (``define_medium.m``) sets ``alpha_mode='no_dispersion'`` when
    ``|alpha_power - 1| < 0.1``. The C++ input format has no field for it, so
    MATLAB 3DC/3DG silently run with full absorption and dispersion (B-14).
    k-wave-python's ``cpp`` backend would raise and its ``python`` backend would
    honour it; ``alpha_mode`` is dropped for both (D-9, QUIRKS.md Q-1).
    """
    out = dict(medium)
    out.pop("alpha_mode", None)
    return out


def _check_backend(backend: str) -> None:
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")


def source_rows_to_kwp(
    rows_find: np.ndarray,
    mask: np.ndarray,
    backend: str,
    *,
    kind: str,
    sound_speed: Any,
    dt: float,
    spacing: Any,
) -> np.ndarray:
    """MATLAB-convention source signals -> what ``kspaceFirstOrder(backend=...)`` expects.

    ``rows_find``: unscaled ``(n_source, M)`` signals, rows in ``find(mask)``
    order (``source.p``, ``source.ux``... as PROTEUS builds them). Additive mode
    only (the only mode PROTEUS uses). ``cpp``: scaled, find order;
    ``python``: unscaled (scaled internally), C order.
    """
    _check_backend(backend)
    rows_find = np.asarray(rows_find, dtype=np.float64)
    n = int(np.count_nonzero(mask))
    if rows_find.ndim != 2 or rows_find.shape[0] != n:
        raise ValueError(
            f"expected (n_source, M) {kind} rows for {n} source voxels, got {rows_find.shape}"
        )
    if backend == "cpp":
        return rows_find * additive_source_scale(kind, mask, sound_speed, dt, spacing)[:, None]
    return rows_f_to_c(rows_find, mask)


def sensor_rows_from_kwp(
    p: np.ndarray, sensor_mask: np.ndarray, Nt: int, backend: str
) -> np.ndarray:
    """``kspaceFirstOrder(backend=...)["p"]`` -> ``(n_sensor, Nt)``, find order.

    For callers of ``kspaceFirstOrder()`` itself (the tracer bullet);
    :func:`kspace_first_order` reads the ``cpp`` output file directly instead.
    For ``cpp``, k-wave-python 0.6.2 returns ``(Nt, n_sensor)`` with columns in
    F order: its own F->C reordering only fires when ``shape[0] == n_sensor``.
    When ``Nt == n_sensor`` it does fire, permuting the *time* axis with the
    sensor-mask permutation; that known permutation is undone here.
    """
    _check_backend(backend)
    p = np.asarray(p)
    n = int(np.count_nonzero(sensor_mask))
    if backend == "cpp":
        if p.shape != (Nt, n):
            raise ValueError(f"expected cpp sensor data of shape {(Nt, n)}, got {p.shape}")
        if n == Nt and n > 1:  # undo k-wave-python's p[perm] on the time axis (B-10)
            perm = f_to_c_rows(sensor_mask)
            q = np.empty_like(p)
            q[perm] = p
            p = q
        return p.T
    if p.shape != (n, Nt):
        raise ValueError(f"expected python sensor data of shape {(n, Nt)}, got {p.shape}")
    return rows_c_to_f(p, sensor_mask)


# ============================================================================ grid


@dataclass(frozen=True)
class KGrid:
    """The parts of a MATLAB ``kWaveGrid`` PROTEUS uses.

    ``kgrid = kWaveGrid(Nx, dx, Ny, dy, Nz, dz); kgrid.dt = dt; kgrid.Nt = Nt``
    (or ``kgrid.setTime(Nt, dt)``) is ``KGrid((Nx, Ny, Nz), (dx, dy, dz), dt, Nt)``.
    """

    shape: tuple[int, ...]
    spacing: tuple[float, ...]
    dt: float
    Nt: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(int(n) for n in self.shape))
        object.__setattr__(self, "spacing", tuple(float(d) for d in self.spacing))
        object.__setattr__(self, "dt", float(self.dt))
        object.__setattr__(self, "Nt", int(self.Nt))
        if len(self.shape) != len(self.spacing) or not 1 <= len(self.shape) <= 3:
            raise ValueError(f"shape {self.shape} and spacing {self.spacing} must be 1-3-D")

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def to_kwp(self):
        """k-wave-python ``kWaveGrid`` with the time array set."""
        from kwave.kgrid import kWaveGrid

        g = kWaveGrid(list(self.shape), list(self.spacing))
        g.setTime(self.Nt, self.dt)
        return g

    @functools.cached_property
    def k_max(self) -> float:
        """MATLAB ``kgrid.k_max``: largest wavenumber supported in every direction [rad/m]."""
        from kwave.kgrid import kWaveGrid

        return float(kWaveGrid(list(self.shape), list(self.spacing)).k_max_all)


# ============================================================================ binary


def binary_path(device: str = "cpu", path: str | os.PathLike | None = None) -> Path:
    """The k-Wave C++ binary to run.

    ``path`` if given, else ``$PYPROTEUS_KWAVE_BINARY``, else the binary
    bundled with k-wave-python (``kspaceFirstOrder-OMP`` for ``device="cpu"``,
    ``kspaceFirstOrder-CUDA`` for ``"gpu"``). ``path`` may be a file or, like
    MATLAB's ``BinaryPath`` (PROTEUS ``PATHS.BinaryPath``), a folder holding
    the binary under its default name. On macOS the bundled binary links
    Homebrew libraries; without them, point this at the wrapper
    ``tools/tracer_bullet/bin/kspaceFirstOrder-OMP`` (D-10).
    """
    if device not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}, got {device!r}")
    if path is None:
        path = os.environ.get(BINARY_ENV) or None
    if path is None:
        import kwave

        name = CUDA_NAME if device == "gpu" else OMP_NAME
        path = Path(kwave.BINARY_PATH) / name
    path = Path(path)
    if path.is_dir():
        path = path / (CUDA_NAME if device == "gpu" else OMP_NAME)
    if not path.is_file():
        raise FileNotFoundError(f"k-Wave binary not found: {path}")
    return path


# ============================================================================ solver


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _as_record(record: Any) -> tuple[str, ...]:
    if record is None:
        return ("p",)  # MATLAB default without sensor.record: pressure time series
    if isinstance(record, str):
        record = (record,)
    record = tuple(str(r) for r in record)
    bad = [r for r in record if r not in RECORD_FLAGS]
    if bad:
        raise NotImplementedError(f"sensor.record {bad} not supported (only {tuple(RECORD_FLAGS)})")
    if len(set(record)) != len(record):
        raise ValueError(f"duplicate entries in sensor.record {record}")
    return record


def _sources(source: Any, kgrid: KGrid) -> dict[str, tuple[np.ndarray, dict[str, np.ndarray]]]:
    """``{"p_mask": (mask, {"p": rows}), "u_mask": (mask, {"ux": rows, ...})}`` (find order)."""
    out: dict[str, tuple[np.ndarray, dict[str, np.ndarray]]] = {}
    for mask_key, kinds, mode_key in (
        ("p_mask", ("p",), "p_mode"),
        ("u_mask", VELOCITY_KINDS[: kgrid.ndim], "u_mode"),
    ):
        mask = _field(source, mask_key)
        signals = {k: _field(source, k) for k in kinds if _field(source, k) is not None}
        if mask is None:
            if signals:
                raise ValueError(f"source has {sorted(signals)} but no {mask_key}")
            continue
        mask = np.asarray(mask)
        if mask.shape != kgrid.shape:
            raise ValueError(f"source.{mask_key} has shape {mask.shape}, grid is {kgrid.shape}")
        if not np.isin(mask, (0, 1)).all():
            raise ValueError(f"source.{mask_key} must be binary")
        mask = mask.astype(bool)
        if not signals:
            raise ValueError(f"source.{mask_key} given without {kinds} signals")
        if not mask.any():
            raise ValueError(f"source.{mask_key} has no active voxels (k-Wave rejects it)")
        mode = _field(source, mode_key, "additive")
        if mode != "additive":
            raise NotImplementedError(f"source.{mode_key}={mode!r}: only 'additive' is supported")
        rows = {}
        for k, v in signals.items():
            v = np.asarray(v, dtype=np.float64)
            if v.ndim == 1:
                v = v[None, :]
            if v.ndim != 2 or v.shape[0] != int(mask.sum()):
                raise ValueError(
                    f"source.{k} must be (n_source, M) with one row per voxel of "
                    f"source.{mask_key} ({int(mask.sum())}), got {v.shape}"
                )
            if v.shape[1] > kgrid.Nt:
                raise NotImplementedError(
                    f"source.{k} has {v.shape[1]} samples > Nt = {kgrid.Nt}: not compared "
                    "with MATLAB yet"
                )
            rows[k] = v
        out[mask_key] = (mask, rows)
    for key in ("p0", "sxx", "syy", "szz", "sxy", "sxz", "syz"):
        if _field(source, key) is not None:
            raise NotImplementedError(f"source.{key} is not supported (PROTEUS does not use it)")
    if not out:
        raise ValueError("source needs p_mask + p or u_mask + ux/uy/uz")
    return out


def _sensor_mask(sensor: Any, kgrid: KGrid) -> np.ndarray:
    mask = _field(sensor, "mask")
    if mask is None:
        raise ValueError("sensor.mask is required")
    mask = np.asarray(mask)
    if mask.shape != kgrid.shape or not np.isin(mask, (0, 1)).all():
        raise ValueError(f"sensor.mask must be a binary mask of shape {kgrid.shape}")
    mask = mask.astype(bool)
    if not mask.any():
        # run_simulation.m returns zeros(0, Nt) itself before calling k-Wave.
        raise ValueError("sensor.mask has no active voxels")
    return mask


def _kwp_inputs(kgrid: KGrid, medium: Any, source: Any, sensor_mask: np.ndarray, backend: str):
    from kwave.kmedium import kWaveMedium
    from kwave.ksensor import kSensor
    from kwave.ksource import kSource

    fields = ("sound_speed", "density", "BonA", "alpha_coeff", "alpha_power", "alpha_mode")
    m = {k: _field(medium, k) for k in fields if _field(medium, k) is not None}
    if "sound_speed" not in m:
        raise ValueError("medium.sound_speed is required")
    m = medium_for_kwp(m)
    for k in ("sound_speed", "density", "BonA", "alpha_coeff"):
        if k in m:
            m[k] = np.asarray(m[k], dtype=np.float64)
    src = kSource()
    for mask_key, (mask, rows) in _sources(source, kgrid).items():
        setattr(src, mask_key, mask.copy())
        for kind, v in rows.items():
            setattr(
                src,
                kind,
                source_rows_to_kwp(
                    v,
                    mask,
                    backend,
                    kind=kind,
                    sound_speed=m["sound_speed"],
                    dt=kgrid.dt,
                    spacing=kgrid.spacing,
                ),
            )
    return kWaveMedium(**m), src, kSensor(mask=sensor_mask.copy())


def _pml(value: Any, ndim: int, name: str) -> tuple:
    t = tuple(np.atleast_1d(np.asarray(value)).tolist())
    if len(t) == 1:
        t = t * ndim
    if len(t) != ndim:
        raise ValueError(f"{name} must be a scalar or have {ndim} entries, got {value!r}")
    return t


def kspace_first_order(
    kgrid: KGrid,
    medium: Any,
    source: Any,
    sensor: Any,
    *,
    pml_size: Any,
    pml_alpha: Any = 2.0,
    pml_inside: bool = False,
    smooth: bool = False,
    data_cast: str = "single",
    backend: str = "cpp",
    device: str = "cpu",
    device_num: int | None = None,
    num_threads: int | None = None,
    binary: str | os.PathLike | None = None,
    binary_env: Mapping[str, str] | None = None,
    data_path: str | os.PathLike | None = None,
    delete_data: bool = True,
) -> dict[str, np.ndarray]:
    """Run k-Wave as PROTEUS ``run_simulation.m`` calls ``kspaceFirstOrder3DC``/``3DG``.

    ``medium``, ``source`` and ``sensor`` are mappings (or attribute objects)
    with MATLAB field names: ``medium.sound_speed/density/BonA/alpha_coeff/
    alpha_power[/alpha_mode]``; ``source.p_mask`` + ``source.p`` and/or
    ``source.u_mask`` + ``source.ux/uy/uz`` (rows in ``find`` order, unscaled,
    additive, one row per source voxel, length ``<= Nt``); ``sensor.mask``
    (binary) and ``sensor.record`` (default ``{'p'}``; also ``p_max``,
    ``p_min``).

    Options as in ``run_simulation.m``: ``PMLSize`` (``pml_size``),
    ``PMLAlpha``, ``PMLInside=false``, ``Smooth=false``, ``DataCast`` ``'single'``
    (``'gpuArray-single'`` is accepted as its 3DG spelling), ``DeviceNum``
    (``device_num``), ``BinaryPath`` (``binary``, see :func:`binary_path`),
    ``BinaryPath`` may be a folder, as in MATLAB. ``DataPath`` (``data_path``,
    default: the system temporary folder) and ``DeleteData`` (``delete_data``):
    each call writes its HDF5 files into a new uniquely named subfolder of
    ``data_path`` (so concurrent runs sharing it do not collide, like MATLAB's
    date-stamped names) and deletes it afterwards unless ``delete_data=False``.
    ``device="gpu"`` selects the CUDA binary (``cpp``) or CuPy (``python``).
    ``binary_env`` adds environment variables for the binary (e.g.
    ``KWAVE_LIBS`` for the macOS wrapper).

    Returns ``{"p": (n_sensor, Nt), "p_max": (n_sensor,), ...}`` as float32,
    rows in ``find(sensor.mask)`` order.

    Differences from ``run_simulation.m`` left to its port: an empty sensor
    mask raises here, while ``run_simulation.m`` returns ``zeros(0, Nt)`` with a
    warning before calling k-Wave; source signals longer than ``Nt`` raise.
    """
    _check_backend(backend)
    if device not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}, got {device!r}")
    if pml_inside:
        raise NotImplementedError("PMLInside=true: PROTEUS always uses PMLInside=false")
    if smooth:
        raise NotImplementedError("Smooth=true: PROTEUS always uses Smooth=false")
    if data_cast not in ("single", "gpuArray-single"):
        raise NotImplementedError(f"DataCast={data_cast!r}: PROTEUS uses 'single'")
    if kgrid.ndim != 3:
        raise NotImplementedError("only 3-D grids (kspaceFirstOrder3DC/3DG) are supported")
    record = _as_record(_field(sensor, "record"))
    sensor_mask = _sensor_mask(sensor, kgrid)
    pml_size = tuple(int(v) for v in _pml(pml_size, kgrid.ndim, "pml_size"))
    pml_alpha = tuple(float(v) for v in _pml(pml_alpha, kgrid.ndim, "pml_alpha"))
    kw_medium, kw_source, kw_sensor = _kwp_inputs(kgrid, medium, source, sensor_mask, backend)
    # PMLInside=false: expand the grid by the PML here with k-wave-python's own
    # helper (what kspaceFirstOrder(pml_inside=False) does internally), then run
    # with pml_inside=True. Same input file and result, but it skips k-wave-python's
    # check 2*pml_size < N on the *unexpanded* grid, which MATLAB does not make
    # and which PROTEUS' cropped self-sense grids can fail (B-61). The shift keeps
    # the relative order of mask voxels, so the F/C row mappings are unchanged.
    from kwave.kspaceFirstOrder import _expand_for_pml_outside

    kw_grid, kw_medium, kw_source, kw_sensor = _expand_for_pml_outside(
        kgrid.to_kwp(), kw_medium, kw_source, kw_sensor, pml_size
    )
    common = dict(
        pml_size=pml_size, pml_alpha=pml_alpha, pml_inside=True, smooth_p0=False, quiet=True
    )
    if backend == "python":
        return _run_python(
            kgrid, kw_grid, kw_medium, kw_source, kw_sensor, sensor_mask, record, device,
            device_num, common,
        )  # fmt: skip
    exe = binary_path(device, binary)
    root = None if data_path is None else Path(data_path)
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
    if not delete_data:
        run_dir = Path(tempfile.mkdtemp(prefix="pyproteus_kwave_", dir=root))
        return _run_cpp(
            kgrid, kw_grid, kw_medium, kw_source, kw_sensor, record, exe, binary_env,
            run_dir, device, device_num, num_threads, common,
        )  # fmt: skip
    with tempfile.TemporaryDirectory(prefix="pyproteus_kwave_", dir=root) as tmp:
        return _run_cpp(
            kgrid, kw_grid, kw_medium, kw_source, kw_sensor, record, exe, binary_env,
            Path(tmp), device, device_num, num_threads, common,
        )  # fmt: skip


def _write_cpp_input(
    kw_grid, medium, source, sensor, device, data_dir: Path, common
) -> tuple[str, str]:
    """The k-Wave C++ HDF5 input file, written by k-wave-python (the pinned input preparation).

    The only place that uses ``kspaceFirstOrder(..., save_only=True)``.
    Returns ``(input_file, output_file)``.
    """
    from kwave.kspaceFirstOrder import kspaceFirstOrder

    data_dir.mkdir(parents=True, exist_ok=True)
    for name in ("kwave_input.h5", "kwave_output.h5"):  # the HDF5 writer appends otherwise
        (data_dir / name).unlink(missing_ok=True)
    files = kspaceFirstOrder(
        kw_grid, medium, source, sensor, backend="cpp", device=device,
        save_only=True, data_path=str(data_dir), **common,
    )  # fmt: skip
    return files["input_file"], files["output_file"]


def binary_command(
    exe: Path,
    input_file: str,
    output_file: str,
    record: Iterable[str],
    *,
    num_threads: int | None = None,
    device_num: int | None = None,
) -> list[str]:
    """Command line for the k-Wave C++ binary, as MATLAB ``kspaceFirstOrder3DC`` builds it.

    Captured from MATLAB k-Wave 1.4.1 (``tests/oracle/test_kwave_backend.py``):
    ``-i <in> -o <out>``, ``-t n`` for ``NumThreads``, then ``--p_raw`` /
    ``--p_max`` / ``--p_min`` in that order for the requested ``sensor.record``
    entries. ``-g n`` for ``DeviceNum`` (3DG; same flag as k-wave-python's own
    runner, placed next to ``-t``; not checked against MATLAB: no CUDA here).
    """
    record = set(record)
    cmd = [str(exe), "-i", str(input_file), "-o", str(output_file)]
    if num_threads is not None:
        cmd += ["-t", str(int(num_threads))]
    if device_num is not None:
        cmd += ["-g", str(int(device_num))]
    cmd += [flag for key, flag in RECORD_FLAGS.items() if key in record]
    return cmd


def read_cpp_output(output_file: str, record: Iterable[str], n_sensor: int, Nt: int):
    """Read the binary's HDF5 output: ``p`` -> ``(n_sensor, Nt)``, aggregates -> ``(n_sensor,)``.

    k-Wave stores datasets in column-major order, so h5py shows the reversed
    MATLAB size: a MATLAB ``n_sensor x Nt`` time series is ``(1, Nt, n_sensor)``,
    an ``n_sensor x 1`` aggregate ``(1, 1, n_sensor)``. Rows stay in
    ``find(sensor.mask)`` order (k-Wave's ``sensor_mask_index``).
    """
    import h5py

    out = {}
    with h5py.File(output_file, "r") as f:
        for key in record:
            a = np.asarray(f[key][()], dtype=np.float32)
            expected = (1, Nt, n_sensor) if key in _TIME_SERIES else (1, 1, n_sensor)
            if a.shape != expected:
                raise RuntimeError(
                    f"k-Wave output {key!r} has shape {a.shape}, expected {expected}"
                )
            out[key] = np.ascontiguousarray(a[0].T) if key in _TIME_SERIES else a.reshape(n_sensor)
    return out


def _run_cpp(
    kgrid, kw_grid, medium, source, sensor, record, exe, binary_env, data_dir, device, device_num,
    num_threads, common,
) -> dict[str, np.ndarray]:  # fmt: skip
    input_file, output_file = _write_cpp_input(
        kw_grid, medium, source, sensor, device, data_dir.resolve(), common
    )
    cmd = binary_command(
        exe, input_file, output_file, record, num_threads=num_threads, device_num=device_num
    )
    env = {**os.environ, **(binary_env or {})}
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"k-Wave binary failed ({r.returncode}): {' '.join(cmd)}\n{r.stdout}\n{r.stderr}"
        )
    return read_cpp_output(output_file, record, int(np.count_nonzero(sensor.mask)), kgrid.Nt)


def _run_python(
    kgrid, kw_grid, medium, source, sensor, sensor_mask, record, device, device_num, common
):
    from kwave.kspaceFirstOrder import kspaceFirstOrder

    sensor.record = list(record)
    ctx: Any = contextlib.nullcontext()
    if device == "gpu" and device_num is not None:
        import cupy

        ctx = cupy.cuda.Device(int(device_num))
    with ctx:
        res = kspaceFirstOrder(
            kw_grid, medium, source, sensor, backend="python", device=device,
            dtype=np.float32, **common,
        )  # fmt: skip
    out = {}
    for key in record:
        a = res[key]
        a = np.asarray(a.get() if hasattr(a, "get") else a, dtype=np.float32)
        out[key] = np.ascontiguousarray(rows_c_to_f(a, sensor_mask))
    return out


# ============================================================================ signal tools


def filter_time_series(
    dt: float,
    k_max: float,
    sound_speed: Any,
    signal: np.ndarray,
    *,
    zero_phase: bool = False,
    transition_width: float = 0.1,
    ppw: float = 3,
    rppw: float = 0,
    stop_band_atten: float = 60,
) -> np.ndarray:
    """k-Wave ``filterTimeSeries`` through k-wave-python's ``filter_time_series``.

    PROTEUS calls it with structs instead of a grid and a medium
    (``preprocess_transmit.m``, ``compute_bubble_mass_source.m``)::

        Filter.dt = 1/fs; Filter.t_array = []; Filter.k_max = kgrid.k_max;
        Filter.sound_speed = c_min;
        filterTimeSeries(Filter, Filter, x, 'ZeroPhase', true, ...
                         'TransitionWidth', Filter.TW, 'PPW', 2)

    which is ``filter_time_series(Filter.dt, Filter.k_max, Filter.sound_speed,
    x, zero_phase=True, transition_width=Filter.TW, ppw=2)``. ``signal`` is a
    vector (1-D, row or column); the result has the same shape.
    """
    from types import SimpleNamespace

    from kwave.utils.filters import filter_time_series as _kwp_filter

    x = np.asarray(signal, dtype=np.float64)
    shape = x.shape
    if x.ndim > 2 or (x.ndim == 2 and min(shape) != 1):
        raise ValueError(f"signal must be a vector, got shape {shape}")
    row = x.reshape(1, -1)
    grid = SimpleNamespace(dt=float(dt), t_array=np.array([]), k_max_all=float(k_max))
    medium = SimpleNamespace(sound_speed=np.asarray(sound_speed, dtype=np.float64))
    if rppw != 0:
        raise NotImplementedError("rppw (start-up ramp) is not supported; PROTEUS uses 0")

    def causal(v: np.ndarray, atten: float) -> np.ndarray:
        y = _kwp_filter(
            grid, medium, v, ppw=ppw, rppw=0, stop_band_atten=atten,
            transition_width=transition_width, zerophase=False,
        )  # fmt: skip
        return np.asarray(y, dtype=np.float64).reshape(1, -1)

    if not zero_phase:
        return causal(row, stop_band_atten).reshape(shape)
    # k-wave-python 0.6.2's zerophase=True crashes (B-60). Zero-phase filtering
    # is forward-backward filtering: run the causal FIR filter F, time-reverse,
    # run F again, reverse back: y = flip(F(flip(F(x)))). The cascade squares the
    # magnitude response, so its stop-band attenuation in dB is twice that of F;
    # F is therefore designed for half the requested attenuation. k-Wave's
    # ``filterTimeSeries`` help text only says the filter is applied "as either
    # a zero phase or linear phase filter" with a given stop-band attenuation;
    # forward-backward filtering is the standard construction of a zero-phase
    # filter from a causal one (as in SciPy's ``filtfilt``). No k-Wave MATLAB
    # source (filterTimeSeries.m, applyFilter.m) was read; the result was checked
    # against MATLAB k-Wave ``filterTimeSeries`` (float class, 1e-9 rel)
    # (``tests/oracle/test_kwave_signal.py``).
    half = stop_band_atten / 2
    y = causal(causal(row, half)[:, ::-1].copy(), half)[:, ::-1]
    return np.ascontiguousarray(y).reshape(shape)


def get_win(
    N: int | Iterable[int],
    type_: str,
    *,
    param: float | None = None,
    rotation: bool = False,
    symmetric: bool = True,
    square: bool = False,
) -> tuple[np.ndarray, float]:
    """k-Wave ``getWin(N, type, 'Param', param, ...)`` through k-wave-python's ``get_win``.

    Returns ``(win, cg)``; for a scalar ``N`` the window is a column vector
    ``(N, 1)`` like MATLAB's. PROTEUS uses ``getWin(N, 'Tukey', 'Param', 0.5)``
    (``get_transducer_integration_points.m``, disabled branch).
    """
    from kwave.utils.signals import get_win as _kwp_get_win

    win, cg = _kwp_get_win(
        N, type_, param=param, rotation=rotation, symmetric=symmetric, square=square
    )
    return np.asarray(win, dtype=np.float64), float(cg)
