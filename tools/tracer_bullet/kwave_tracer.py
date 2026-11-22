"""Run tracer-bullet cases through MATLAB k-Wave and k-wave-python, and compare.

Runners (the first three on the same k-Wave C++ OpenMP binary, k-wave-python's
v1.4.1 macOS arm64 build, through ``bin/kspaceFirstOrder-OMP``):

* ``run_matlab``      MATLAB k-Wave 1.4.1 ``kspaceFirstOrder3DC`` via the engine
                      (writes the HDF5 input file with k-Wave's own MATLAB code)
* ``run_kwp``         k-wave-python ``kspaceFirstOrder(..., backend="cpp")``
                      (the >= 0.6.1 API; writes its own HDF5 input file), or
                      ``backend="python"`` (NumPy solver, no binary)
* ``run_kwp_legacy``  k-wave-python ``kspaceFirstOrder3D`` (MATLAB-style
                      legacy API, ``kWaveSimulation`` + ``save_to_disk``)

All runners take and return MATLAB/PROTEUS conventions (rows in
``find(mask)`` order, ``(n_sensor, Nt)``); ``pyproteus.backends.kwave`` does
the mapping.
Because the C++ solver binary is shared, the comparison isolates what the M3
adapter controls: input preparation (source scaling, staggered density,
absorption conversion, PML expansion) and row order. ``compare_inputs``
diffs the HDF5 input files dataset by dataset to pin down any difference.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

import h5py
import numpy as np

from pyproteus.backends import kwave as kwb  # row order + source scaling (M3)

try:  # package-style import (tests) or script-style (python tools/tracer_bullet/...)
    from .cases import Case
except ImportError:  # pragma: no cover
    from cases import Case  # type: ignore[no-redef]

HERE = Path(__file__).resolve().parent
WRAPPER_DIR = HERE / "bin"
WRAPPER = WRAPPER_DIR / "kspaceFirstOrder-OMP"
LIBS = HERE / ".kwave-libs" / "lib"
KWAVE_MATLAB = Path(
    os.environ.get(
        "KWAVE_MATLAB_PATH",
        "/Users/theodoremichel/Repos/TWENTE/Ultrasound/PROTEUS-cleaning/PROTEUS/external/k-wave/k-Wave",
    )
)


def binary_env() -> dict[str, str]:
    """Environment for the wrapper: which binary to exec, where its libraries are."""
    import kwave

    return {
        "KWAVE_BINARY": str(Path(kwave.BINARY_PATH) / "kspaceFirstOrder-OMP"),
        "KWAVE_LIBS": str(LIBS),
    }


def binary_available() -> bool:
    """True if the wrapped k-Wave binary starts (runtime libraries found)."""
    import subprocess

    try:
        r = subprocess.run(
            [str(WRAPPER), "--version"],
            env={**os.environ, **binary_env()},
            capture_output=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return r.returncode == 0 and b"kspaceFirstOrder" in r.stdout


def kwave_binary_version() -> str:
    import kwave

    return str(kwave.BINARY_VERSION)


# ----------------------------------------------------------------------------- MATLAB


def preload_before_engine() -> None:
    """Load conda's shared libraries before MATLAB's (B-1 libcrypto, B-11 libexpat)."""
    import ssl  # noqa: F401
    import xml.parsers.expat  # noqa: F401

    import kwave.kspaceFirstOrder
    import kwave.kspaceFirstOrder3D  # noqa: F401  (pulls matplotlib -> plistlib -> pyexpat)


def start_engine():
    preload_before_engine()
    import matlab.engine

    eng = matlab.engine.start_matlab("-nodisplay")
    prepare_engine(eng)
    return eng


def prepare_engine(eng) -> None:
    eng.addpath(str(KWAVE_MATLAB), nargout=0)
    eng.addpath(str(HERE), nargout=0)
    for k, v in binary_env().items():
        eng.setenv(k, v, nargout=0)


def _to_matlab_struct(case: Case) -> dict:
    import matlab

    def dbl(a):
        return matlab.double(np.asarray(a, dtype=np.float64))

    s = {
        "Nx": float(case.shape[0]),
        "Ny": float(case.shape[1]),
        "Nz": float(case.shape[2]),
        "dx": case.dx,
        "dt": case.dt,
        "Nt": float(case.Nt),
        "pml_size": dbl(np.asarray(case.pml_size, float).reshape(1, -1)),
        "pml_alpha": case.pml_alpha,
        "sensor_mask": matlab.logical(case.sensor_mask),
        "alpha_power": float(case.medium["alpha_power"]),
        "record": list(case.record),
    }
    if "alpha_mode" in case.medium:
        s["alpha_mode"] = str(case.medium["alpha_mode"])
    for k in ("sound_speed", "density", "BonA", "alpha_coeff"):
        s[k] = dbl(case.medium[k])
    for k, v in case.source.items():
        s[k] = matlab.logical(v) if k.endswith("mask") else dbl(v)
    return s


def run_matlab_fields(
    eng, case: Case, data_dir: Path, binary_dir: Path | None = None, **extra
) -> dict[str, np.ndarray]:
    """All recorded fields (``case.record``) from MATLAB k-Wave, rows in find(sensor.mask) order.

    Time series as ``(n_sensor, Nt)``, aggregates (``p_max``...) as ``(n_sensor,)``.
    ``binary_dir`` replaces the folder holding ``kspaceFirstOrder-OMP``;
    ``extra`` adds fields to the MATLAB struct (``num_threads`` -> ``NumThreads``).
    """
    data_dir = Path(data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    s = _to_matlab_struct(case) | extra  # e.g. num_threads=2.0
    bdir = str(binary_dir or WRAPPER_DIR) + "/"
    out = eng.run_kwave_matlab(s, bdir, str(data_dir) + "/", f"matlab_{case.name}")
    res = {}
    for k, v in out.items():
        a = np.asarray(v, dtype=np.float64)
        res[k] = a if k == "p" else a.reshape(-1)
    return res


def run_matlab(eng, case: Case, data_dir: Path) -> np.ndarray:
    """Pressure sensor data from MATLAB k-Wave, rows in find(sensor.mask) order."""
    return run_matlab_fields(eng, case, data_dir)["p"]


def check_engine_layout(eng) -> None:
    """Guard: numpy -> matlab.double keeps [ix,iy,iz] element positions."""
    import matlab

    a = np.zeros((3, 4, 5))
    a[2, 1, 4] = 1.0
    idx = eng.find(matlab.double(a))
    expected = kwb.matlab_find(a)
    if int(idx) != int(expected[0]):
        raise RuntimeError(f"engine layout mismatch: MATLAB find={idx}, expected {expected}")


# ----------------------------------------------------------------------------- k-wave-python


def _fresh_dir(path: Path) -> Path:
    """Empty data folder (k-wave-python's HDF5 writer appends to existing files)."""
    import shutil

    path = Path(path).resolve()
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


@contextlib.contextmanager
def _binary_environ():
    """Set the wrapper's variables for the duration of a run, then restore os.environ."""
    old = {k: os.environ.get(k) for k in binary_env()}
    os.environ.update(binary_env())
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _kwp_objects(case: Case, backend: str = "cpp"):
    from kwave.kgrid import kWaveGrid
    from kwave.kmedium import kWaveMedium
    from kwave.ksensor import kSensor

    kgrid = kWaveGrid(list(case.shape), [case.dx] * 3)
    kgrid.setTime(case.Nt, case.dt)
    m = kwb.medium_for_kwp(case.medium)  # drops alpha_mode on every backend (D-9)
    medium = kWaveMedium(**m)
    sensor = kSensor(mask=case.sensor_mask.copy())
    return kgrid, medium, sensor


def _kind_signals(case: Case) -> tuple[str, dict[str, np.ndarray]]:
    if "p_mask" in case.source:
        return "p_mask", {"p": case.source["p"]}
    return "u_mask", {k: case.source[k] for k in ("ux", "uy")}


def _kwp_source(case: Case, backend: str, *, raw_rows: str | None = None, unscaled=False):
    """kSource for ``kspaceFirstOrder``; ``raw_rows`` ('f'/'c') skips the adapter's reordering."""
    from kwave.ksource import kSource

    mask_key, signals = _kind_signals(case)
    mask = case.source[mask_key]
    src = kSource()
    setattr(src, mask_key, mask.copy())
    for k, rows in signals.items():
        if raw_rows is None:
            v = kwb.source_rows_to_kwp(
                rows,
                mask,
                backend,
                kind=k,
                sound_speed=case.medium["sound_speed"],
                dt=case.dt,
                spacing=(case.dx,) * 3,
            )
        else:
            v = rows
            if backend == "cpp" and not unscaled:  # the python backend scales internally
                v = (
                    kwb.additive_source_scale(
                        k, mask, case.medium["sound_speed"], case.dt, (case.dx,) * 3
                    )[:, None]
                    * v
                )
            if raw_rows == "c":
                v = kwb.rows_f_to_c(v, mask)
        setattr(src, k, v)
    return src


def run_kwp(
    case: Case,
    data_dir: Path | None,
    backend: str = "cpp",
    *,
    raw_rows: str | None = None,
    unscaled: bool = False,
) -> np.ndarray:
    """k-wave-python ``kspaceFirstOrder()``; sensor data as ``(n_sensor, Nt)``, rows in find order.

    ``raw_rows`` ('f' or 'c') feeds the source rows in that order without the
    adapter's reordering (scaling still applied for cpp): used to show that a
    row-order mix-up is detected. ``unscaled=True`` (with ``raw_rows``) also
    skips the cpp source scaling, to show that it is needed.
    """
    from kwave.kspaceFirstOrder import kspaceFirstOrder

    kgrid, medium, sensor = _kwp_objects(case, backend)
    src = _kwp_source(case, backend, raw_rows=raw_rows, unscaled=unscaled)
    kw = {}
    if backend == "cpp":
        kw = dict(
            binary_path=str(WRAPPER),
            data_path=str(
                _fresh_dir(
                    Path(data_dir)
                    / (
                        f"kwp_cpp_{case.name}"
                        + (f"_raw_{raw_rows}{'_u' * unscaled}" if raw_rows else "")
                    )
                )
            ),
        )
    else:
        kw = dict(dtype=np.float32)
    with _binary_environ():
        res = kspaceFirstOrder(
            kgrid,
            medium,
            src,
            sensor,
            pml_size=case.pml_size,
            pml_alpha=case.pml_alpha,
            pml_inside=False,
            smooth_p0=False,
            backend=backend,
            device="cpu",
            quiet=True,
            **kw,
        )
    return kwb.sensor_rows_from_kwp(
        np.asarray(res["p"], dtype=np.float64), case.sensor_mask, case.Nt, backend
    )


def run_kwp_legacy(case: Case, data_dir: Path) -> np.ndarray:
    """k-wave-python legacy ``kspaceFirstOrder3D`` (MATLAB-style API); rows in find order.

    Source rows are passed in find order and unscaled, exactly as to MATLAB.
    """
    from kwave.ksource import kSource
    from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
    from kwave.options.simulation_execution_options import SimulationExecutionOptions
    from kwave.options.simulation_options import SimulationOptions

    kgrid, medium, sensor = _kwp_objects(case)
    sensor.record = ["p"]
    src = kSource()
    mask_key, signals = _kind_signals(case)
    setattr(src, mask_key, case.source[mask_key].copy())
    for k, v in signals.items():
        setattr(src, k, v)
    data_dir = _fresh_dir(Path(data_dir) / f"kwp_legacy_{case.name}")
    sim_opts = SimulationOptions(
        pml_inside=False,
        pml_size=list(case.pml_size),
        pml_alpha=case.pml_alpha,
        data_cast="single",
        smooth_p0=False,
        smooth_c0=False,
        smooth_rho0=False,
        save_to_disk=True,
        data_path=str(data_dir),
        input_filename="input.h5",
        output_filename="output.h5",
    )
    exe_opts = SimulationExecutionOptions(
        is_gpu_simulation=False,
        binary_path=WRAPPER,
        delete_data=False,
        show_sim_log=False,
    )
    with _binary_environ():
        kspaceFirstOrder3D(
            kgrid, src, sensor, medium, simulation_options=sim_opts, execution_options=exe_opts
        )
    # read the binary's output directly: (1, Nt, n_sensor) in h5py, columns in find order
    with h5py.File(data_dir / "output.h5", "r") as f:
        return np.asarray(f["p"][0], dtype=np.float64).T


# ----------------------------------------------------------------------------- comparison


def rel_err(a: np.ndarray, ref: np.ndarray) -> float:
    """max |a - ref| / max |ref| over the whole array."""
    return float(np.max(np.abs(a - ref)) / np.max(np.abs(ref)))


def rel_err_rows(a: np.ndarray, ref: np.ndarray, floor: float = 1e-3) -> float:
    """max over rows of ||a_i - ref_i|| / ||ref_i||, for rows whose norm exceeds
    ``floor`` x the largest row norm (weak sensors are not hidden by strong ones)."""
    num = np.linalg.norm(a - ref, axis=1)
    den = np.linalg.norm(ref, axis=1)
    keep = den > floor * den.max()
    return float(np.max(num[keep] / den[keep]))


def read_h5(path: Path) -> dict[str, np.ndarray]:
    out = {}
    with h5py.File(path, "r") as f:
        for k in f:
            out[k] = f[k][()]
    return out


INPUT_REL = 1e-6  # float32 datasets of the k-Wave input file (D-13: k-Wave class, input side)


def assert_inputs_close(a: dict, b: dict, skip=("rho0_sg",)) -> None:
    """Two k-Wave C++ input files (``read_h5`` dicts) carry the same data.

    Integer datasets (sizes, flags, 1-based mask indices: the documented file
    format) must be equal; float datasets within ``INPUT_REL`` of the maximum
    magnitude of the reference ``a`` (float32). Datasets whose name starts with ``skip`` are left out.
    """
    assert set(a) == set(b), sorted(set(a) ^ set(b))
    for key in sorted(a):
        if key.startswith(skip):
            continue
        x, y = np.asarray(a[key]), np.asarray(b[key])
        assert x.shape == y.shape, key
        if x.dtype.kind in "iu":
            assert np.array_equal(x, y), key
        else:
            xd, yd = x.astype(np.float64), y.astype(np.float64)
            den = max(float(np.max(np.abs(xd), initial=0.0)), 1e-300)
            assert float(np.max(np.abs(xd - yd), initial=0.0)) <= INPUT_REL * den, key


def compare_inputs(a_path: Path, b_path: Path) -> dict[str, str]:
    """Dataset-by-dataset diff of two k-Wave C++ input files."""
    a, b = read_h5(a_path), read_h5(b_path)
    report = {}
    for k in sorted(set(a) | set(b)):
        if k not in a or k not in b:
            report[k] = "only in " + ("first" if k in a else "second")
            continue
        x, y = np.asarray(a[k]), np.asarray(b[k])
        if x.shape != y.shape:
            report[k] = f"shape {x.shape} vs {y.shape}"
            continue
        if x.size == 0:
            report[k] = "equal (empty)"
            continue
        if np.array_equal(x, y):
            report[k] = "equal"
            continue
        xd, yd = x.astype(np.float64), y.astype(np.float64)
        den = np.max(np.abs(xd)) or 1.0
        report[k] = (
            f"max rel diff {np.max(np.abs(xd - yd)) / den:.3e} (dtype {x.dtype} vs {y.dtype})"
        )
    return report
