"""CuPy path vs NumPy path on a real GPU (M2 "Done when": CuPy matches NumPy).

Skipped without CuPy or a CUDA device (the development Mac and CI); run on the
university cluster (D-16). ``tests/unit/test_homogeneous_backend.py`` checks the
array-module switch with a CuPy stand-in; this file checks the real thing: cuFFT,
device transfers, device selection, the host-accumulation and RF paths.

Chain (the GPU part of ``hybrid_simulator``'s last stage, no MATLAB needed):
bubble mass sources (synthetic: the bubble ODE runs on the CPU in both paths, so
it cannot differ) → ``run_simulation_homogeneous`` (gridded and exact) to the
integration points of a 96-element linear array → ``compute_RF_data`` with
sparse sensor weights.

Classes (D-13, D-18): double precision ≤ 1e-9 × max|x| (float class; cuFFT and
pocketfft differ at ~1e-15), single precision ≤ 1e-6 × max|x| (D-18 single class;
the kernels work in float64, so single results differ by float32 rounding only).

Run ``pytest tests/gpu -s`` to also print CPU/GPU timings (not asserted).
"""

from __future__ import annotations

import time

import numpy as np
import pytest
import scipy.sparse as sp

from pyproteus.acoustic import compute_RF_data, run_simulation_homogeneous

cupy = pytest.importorskip("cupy")
try:
    _N_DEV = cupy.cuda.runtime.getDeviceCount()
except Exception:  # no driver / no device
    _N_DEV = 0
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(_N_DEV == 0, reason="no CUDA device"),
]

FS = 62.5e6  # PROTEUS-like sampling rate
DT = 1 / FS
C0 = 1540.0
MEDIUM = {"density": 1000.0, "sound_speed": C0, "alpha_coeff": 0.5, "alpha_power": 1.1}
REL = {"double": 1e-9, "single": 1e-6}


def _gather(x):
    return x.get() if isinstance(x, cupy.ndarray) else np.asarray(x)


def _max_rel(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def make_problem(n_bubbles: int = 300, nt: int = 1500, n_el: int = 96, n_int: int = 5, seed=0):
    """Bubbles in a 4 × 2 × 1 cm box in front of a linear array at x = 0."""
    rng = np.random.default_rng(seed)
    t = np.arange(nt) * DT
    t0 = rng.uniform(2e-6, 0.5 * nt * DT, (n_bubbles, 1))  # arrival of the pulse
    f0 = rng.uniform(2e6, 4e6, (n_bubbles, 1))
    env = np.exp(-(((t - t0) / 4e-7) ** 2))
    ms = rng.uniform(0.2, 2.0, (n_bubbles, 1)) * 1e-12 * env * np.sin(2 * np.pi * f0 * (t - t0))
    pts = np.column_stack(
        [
            rng.uniform(5e-3, 45e-3, n_bubbles),
            rng.uniform(-10e-3, 10e-3, n_bubbles),
            rng.uniform(-0.5e-3, 0.5e-3, n_bubbles),
        ]
    )
    pitch = 0.3e-3
    y_el = (np.arange(n_el) - (n_el - 1) / 2) * pitch
    z_int = np.linspace(-2e-3, 2e-3, n_int)  # integration points along the elevation
    ip = np.zeros((n_el, n_int, 3))
    ip[:, :, 1] = y_el[:, None]
    ip[:, :, 2] = z_int[None, :]
    T = {
        "integration_points": ip,
        "integration_receive_apodization": np.ones((n_el, n_int)),
        "integration_receive_delays": rng.uniform(0, 3 * DT, (n_el, n_int)),  # lens
        "integration_weights": 1.0,
        "ReceiveImpulseResponse": np.hanning(33)
        * np.sin(2 * np.pi * 2.5e6 * np.arange(33) / 250e6),
        "SamplingRate": 250e6,
    }
    sensor_points = ip.reshape(n_el * n_int, 3, order="F")  # MATLAB column-major order
    return nt, ms, pts, sensor_points, T


def run_chain(cast: str, gpu: bool, gridded: bool, problem, **extra):
    nt, ms, pts, sensor_points, T = problem
    dtype = np.float32 if cast == "single" else np.float64
    data_cast = f"gpuArray-{cast}" if gpu else cast
    rp = {"gridded": gridded, "DATA_CAST": data_cast, "DATA_CAST_RF": data_cast, **extra}
    t0 = time.perf_counter()
    sd = run_simulation_homogeneous(
        rp,
        {"dt": DT, "Nt": nt},
        MEDIUM,
        {"points": pts, "mass_source": ms.astype(dtype)},
        {"points": sensor_points},
    )
    if gpu:
        cupy.cuda.Device().synchronize()
    t1 = time.perf_counter()
    n = sensor_points.shape[0]
    rf, rp_out = compute_RF_data(T, sd, sp.identity(n, format="csr"), {"dt": DT}, rp)
    t2 = time.perf_counter()
    return sd["p"], rf, rp_out, (t1 - t0, t2 - t1)


@pytest.fixture(scope="module")
def problem():
    return make_problem()


def test_cupy_device_visible() -> None:
    props = cupy.cuda.runtime.getDeviceProperties(0)
    name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
    print(
        f"\nCuPy {cupy.__version__}, {_N_DEV} device(s), device 0: {name}, "
        f"CC {props['major']}.{props['minor']}, "
        f"{props['totalGlobalMem'] / 2**30:.1f} GiB"
    )
    x = cupy.arange(10, dtype=cupy.float64)
    assert float(cupy.fft.irfft(cupy.fft.rfft(x), n=10).get()[3]) == pytest.approx(3.0)


@pytest.mark.parametrize("gridded", [True, False], ids=["gridded", "exact"])
@pytest.mark.parametrize("cast", ["double", "single"])
def test_cupy_matches_numpy(problem, cast: str, gridded: bool) -> None:
    p_cpu, rf_cpu, _, t_cpu = run_chain(cast, False, gridded, problem)
    p_gpu, rf_gpu, rp_out, t_gpu = run_chain(cast, True, gridded, problem)
    dtype = np.float32 if cast == "single" else np.float64

    assert isinstance(p_gpu, cupy.ndarray)  # small output stays on the device (MaxGpuElements)
    assert p_gpu.dtype == dtype and rf_gpu.dtype == dtype
    assert type(rf_gpu) is np.ndarray  # RF gathered to the host, as MATLAB
    assert rp_out["DATA_CAST_RF"] == f"gpuArray-{cast}"  # no OOM fallback happened
    assert np.all(np.isfinite(rf_gpu)) and np.max(np.abs(rf_cpu)) > 0

    e_p, e_rf = _max_rel(_gather(p_gpu), p_cpu), _max_rel(rf_gpu, rf_cpu)
    print(
        f"\n[{cast}, {'gridded' if gridded else 'exact'}] max rel diff: p {e_p:.2e}, RF {e_rf:.2e}"
        f" | propagation CPU {t_cpu[0]:.2f}s GPU {t_gpu[0]:.2f}s"
        f" | RF CPU {t_cpu[1]:.2f}s GPU {t_gpu[1]:.2f}s"
    )
    assert e_p <= REL[cast]
    assert e_rf <= REL[cast]


def test_cupy_large_output_accumulated_on_host(problem) -> None:
    """Above ``MaxGpuElements`` the sensor data is accumulated in a host array (as PROTEUS)."""
    p_cpu, rf_cpu, _, _ = run_chain("double", False, True, problem)
    p_gpu, rf_gpu, _, _ = run_chain("double", True, True, problem, MaxGpuElements=1000)
    assert type(p_gpu) is np.ndarray
    assert _max_rel(p_gpu, p_cpu) <= REL["double"]
    assert _max_rel(rf_gpu, rf_cpu) <= REL["double"]


def test_cupy_small_blocks_match(problem) -> None:
    """Block sizes (``MaxArrayElements``) change only rounding, on the GPU too."""
    p_ref, rf_ref, _, _ = run_chain("double", True, False, problem)
    p_blk, rf_blk, _, _ = run_chain("double", True, False, problem, MaxArrayElements=2**16)
    assert _max_rel(_gather(p_blk), _gather(p_ref)) <= REL["double"]
    assert _max_rel(rf_blk, rf_ref) <= REL["double"]


@pytest.mark.skipif(_N_DEV < 2, reason="needs two CUDA devices")
def test_cupy_device_num_selects_device(problem) -> None:
    nt, ms, pts, sensor_points, _ = problem
    sd = run_simulation_homogeneous(
        {"gridded": True, "DATA_CAST": "gpuArray-double", "DEVICE_NUM": 1},
        {"dt": DT, "Nt": nt},
        MEDIUM,
        {"points": pts, "mass_source": ms},
        {"points": sensor_points},
    )
    assert sd["p"].device.id == 1


@pytest.mark.slow
def test_cupy_benchmark() -> None:
    """Timing on a larger case (1000 bubbles, 96 × 9 points); prints, asserts parity only."""
    prob = make_problem(n_bubbles=1000, nt=2000, n_int=9, seed=1)
    for gridded in (True, False):
        _, rf_cpu, _, t_cpu = run_chain("single", False, gridded, prob)
        _, rf_gpu, _, t_gpu = run_chain("single", True, gridded, prob)
        print(
            f"\n[benchmark single, {'gridded' if gridded else 'exact'}] "
            f"propagation CPU {t_cpu[0]:.2f}s GPU {t_gpu[0]:.2f}s "
            f"(x{t_cpu[0] / max(t_gpu[0], 1e-9):.1f}); RF CPU {t_cpu[1]:.2f}s GPU {t_gpu[1]:.2f}s"
        )
        assert _max_rel(rf_gpu, rf_cpu) <= REL["single"]
