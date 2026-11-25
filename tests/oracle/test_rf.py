"""Element RF data vs MATLAB PROTEUS: ``compute_RF``, ``compute_RF_data``.

The transducer (integration points, lens delays, weights, impulse response)
is built by MATLAB PROTEUS (``get_transducer_properties``,
``get_transducer_integration_points``, ``get_transducer_integration_delays``)
and handed to both sides, so only ``compute_RF`` is compared. Cases: linear
(P4-1) and curved (C1-6D) arrays with odd element counts, finite and infinite
elevation focus, odd/even signal lengths, random receive apodization,
per-point quadrature weights, dense and sparse sensor weights, single and
double ``DATA_CAST_RF``, ``RF_type = 'pressure'``.

Classes (D-13): double: float, ``|Δ| ≤ 1e-9 max|RF|``; single: ``1e-5``
relative to max|RF| (as k-Wave single data); output sizes exactly equal.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp

from pyproteus.acoustic import compute_RF, compute_RF_data, rf_signal_length
from tests.oracle.helpers import get, put

REL = 1e-9
SINGLE_REL = 1e-5
DT = 1 / 50e6

FIELDS = (
    "integration_points",
    "integration_receive_apodization",
    "integration_receive_delays",
    "integration_weights",
    "ReceiveImpulseResponse",
    "SamplingRate",
)


def rel_err(a: np.ndarray, e: np.ndarray) -> float:
    a, e = np.asarray(a, dtype=float), np.asarray(e, dtype=float)
    assert a.shape == e.shape, f"shape {a.shape} != {e.shape}"
    err = float(np.max(np.abs(a - e)) / np.max(np.abs(e)))
    if os.environ.get("PYPROTEUS_REPORT"):
        print(f"  rel err {err:.2e}", end="")
    return err


def build_transducer(eng: Any, kind: str, n_el: int, focus: float | None) -> dict[str, Any]:
    """MATLAB transducer ``pyp_T`` with integration points and lens delays; fields → Python."""
    put(eng, pyp_n=float(n_el))
    eng.eval(
        f"pyp_T = struct('Type', '{kind}'); pyp_T = get_transducer_properties(pyp_T);"
        "if strcmp(pyp_T.ImpulseResponseType, 'Estimate'), pyp_T.SamplingRate = 250e6;"
        " pyp_T = estimate_impulse_response(pyp_T); end;"
        "pyp_T.NumberOfElements = pyp_n; pyp_T.IntegrationDensity = 1;"
        + (f"pyp_T.ElevationFocus = {focus!r};" if focus is not None else "")
        + "pyp_G = struct('dy', 1.1e-4, 'dz', 1.3e-4, 'dt', 2e-8);"
        "pyp_T = get_transducer_integration_points(pyp_T, pyp_G);"
        "pyp_T = get_transducer_integration_delays(pyp_T, struct('SpeedOfSound', 1540));",
        nargout=0,
    )
    out = {}
    for f in FIELDS:
        eng.eval(f"pyp_tmp = pyp_T.{f};", nargout=0)
        v = get(eng, "pyp_tmp")
        out[f] = float(v[0, 0]) if v.size == 1 else v
    out["integration_points"] = out["integration_points"].reshape(n_el, -1, 3)
    out["ReceiveImpulseResponse"] = np.asarray(out["ReceiveImpulseResponse"]).reshape(-1)
    eng.eval("clear pyp_tmp pyp_n pyp_G", nargout=0)
    return out


def matlab_rf(
    eng: Any, T: dict, p: np.ndarray, W: np.ndarray, sparse: bool, run_param: dict
) -> np.ndarray:
    put(
        eng,
        pyp_p=p,
        pyp_W=W,
        pyp_apod=T["integration_receive_apodization"],
        pyp_w=np.atleast_2d(np.asarray(T["integration_weights"], dtype=float)),
        pyp_dt=DT,
    )
    rp = ", ".join(f"'{k}', '{v}'" for k, v in run_param.items())
    eng.eval(
        "pyp_T.integration_receive_apodization = pyp_apod; pyp_T.integration_weights = pyp_w;"
        + ("pyp_W = sparse(pyp_W);" if sparse else "")
        + f"pyp_rp = struct({rp});"
        "evalc(['[pyp_rf, pyp_rp2] = compute_RF_data(pyp_T, struct(''p'', pyp_p), pyp_W,"
        " struct(''dt'', pyp_dt), pyp_rp);']);",
        nargout=0,
    )
    rf = get(eng, "pyp_rf")
    eng.eval("clear pyp_p pyp_W pyp_apod pyp_w pyp_dt pyp_rp pyp_rp2 pyp_rf", nargout=0)
    return rf


def sensor_data(n_sensor: int, m: int, seed: int) -> np.ndarray:
    """Band-limited echoes at random times, some arriving at the very end of the record."""
    rng = np.random.default_rng(seed)
    t = np.arange(m) * DT
    t0 = rng.uniform(0, t[-1], (n_sensor, 1))
    t0[0] = t[-1]  # truncated by the end of the record
    return np.sin(2 * np.pi * 2.5e6 * (t - t0)) * np.exp(-(((t - t0) / 6e-7) ** 2))


def sensor_weights(n_rows: int, n_sensor: int, seed: int) -> np.ndarray:
    """Sparse-like interpolation weights (a few sensors per integration point)."""
    rng = np.random.default_rng(seed)
    W = rng.random((n_rows, n_sensor))
    W[W < 0.8] = 0
    return W


CASES = [
    # id, transducer, n_el, focus, M, cast, RF_type, weights, sparse
    ("linear-odd-M-double", "P4-1", 5, None, 301, "double", None, "scalar", True),
    ("linear-even-M-single", "P4-1", 5, None, 400, "single", None, "scalar", False),
    ("linear-inf-focus", "P4-1", 3, float("inf"), 257, "double", None, "scalar", True),
    ("linear-pressure", "P4-1", 7, 20e-3, 333, "double", "pressure", "scalar", True),
    ("linear-pressure-single", "P4-1", 7, 20e-3, 222, "single", "pressure", "scalar", True),
    ("curved-odd", "C1-6D", 7, None, 351, "double", None, "scalar", True),
    ("curved-single", "C1-6D", 9, None, 290, "single", None, "scalar", True),
    ("linear-point-weights", "L22-14v", 5, None, 512, "double", None, "matrix", False),
]


@pytest.mark.parametrize(
    ("kind", "n_el", "focus", "m", "cast", "rf_type", "weights", "sparse"),
    [c[1:] for c in CASES],
    ids=[c[0] for c in CASES],
)
def test_compute_RF(matlab_engine, kind, n_el, focus, m, cast, rf_type, weights, sparse) -> None:
    T = build_transducer(matlab_engine, kind, n_el, focus)
    n_int = T["integration_points"].shape[1]
    rng = np.random.default_rng(n_el * m)
    T["integration_receive_apodization"] = rng.uniform(0.2, 1.0, (n_el, n_int))
    if weights == "matrix":
        T["integration_weights"] = rng.uniform(0.5, 2.0, (n_el, n_int))
    n_sensor = 23
    p = sensor_data(n_sensor, m, seed=m)
    if cast == "single":
        p = p.astype(np.float32)  # k-Wave sensor data are single
    W = sensor_weights(n_el * n_int, n_sensor, seed=n_el)
    run_param = {"DATA_CAST_RF": cast}
    if rf_type:
        run_param["RF_type"] = rf_type
    expected = matlab_rf(matlab_engine, T, p, W, sparse, run_param)
    W_py = sp.csr_matrix(W) if sparse else W
    actual, rp = compute_RF_data(T, {"p": p}, W_py, {"dt": DT}, run_param)
    assert rp is run_param
    assert actual.dtype == (np.float32 if cast == "single" else np.float64)
    assert actual.shape == expected.shape
    assert rel_err(actual, expected) <= (SINGLE_REL if cast == "single" else REL)
    if focus is None or np.isfinite(focus):
        assert np.max(T["integration_receive_delays"]) > 0  # delays exercised


def test_rf_signal_length(matlab_engine) -> None:
    """``N = M + ceil(max(delays)/dt)``, then ``optimize_grid_size`` (max prime 5)."""
    put(matlab_engine, pyp_dt=DT)
    for m in (1, 7, 100, 211, 997, 4096, 10007):
        for delay in (0.0, 3.3e-7, 40 * DT, 1.234567e-6):
            matlab_engine.eval(
                f"pyp_N = {m} + ceil({delay!r}/pyp_dt);"
                "pyp_N = optimize_grid_size(pyp_N, [0 max(10, round(pyp_N*0.05))], 5);",
                nargout=0,
            )
            assert rf_signal_length(m, delay, DT) == int(get(matlab_engine, "pyp_N")[0, 0])
    matlab_engine.eval("clear pyp_dt pyp_N", nargout=0)


def test_compute_RF_python_transducer_chain(matlab_engine) -> None:
    """Same RF from pyproteus's own integration points and delays (M1 ports) as from MATLAB's."""
    from pyproteus.acoustic import (
        get_transducer_integration_delays,
        get_transducer_integration_points,
    )
    from pyproteus.config import Transducer
    from pyproteus.transducer import get_transducer_properties

    T_m = build_transducer(matlab_engine, "C1-6D", 5, None)
    T = get_transducer_properties(Transducer(Type="C1-6D"))
    T.NumberOfElements = 5.0
    T.IntegrationDensity = 1.0
    grid = type("G", (), {"dy": 1.1e-4, "dz": 1.3e-4})()
    T = get_transducer_integration_delays(
        get_transducer_integration_points(T, grid), {"SpeedOfSound": 1540.0}
    )
    n_int = T_m["integration_points"].shape[1]
    W = sensor_weights(5 * n_int, 11, seed=3)
    p = sensor_data(11, 300, seed=4)
    rf_m = compute_RF(T_m, {"p": p}, W, {"dt": DT}, {"DATA_CAST_RF": "double"})
    rf_p = compute_RF(T, {"p": p}, W, {"dt": DT}, {"DATA_CAST_RF": "double"})
    assert rel_err(rf_p, rf_m) <= REL
