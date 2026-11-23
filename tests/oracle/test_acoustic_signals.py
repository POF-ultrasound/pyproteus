"""Oracle tests: ``evaluate_delta_function``, ``sinc_interpolation``, ``resample_signal``.

D-13 float class: ≤ 1e-9 relative, with an absolute floor near zero of
1e-12 × max|reference| (sinc and delta values at sample points are zero up to
rounding: ``sin(pi*k)`` is ~1e-16, not 0, in both environments).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pyproteus import acoustic
from tests.oracle.helpers import ORACLE_SETTINGS, batch, col, row
from tests.oracle.test_acoustic_grid import assert_float


def assert_signal(actual: Any, expected: Any, err_msg: str = "") -> None:
    e = np.asarray(expected, dtype=float)
    scale = float(np.max(np.abs(e[np.isfinite(e)]), initial=0.0))
    assert_float(actual, expected, atol=1e-12 * max(scale, 1.0), err_msg=err_msg)


# --------------------------------------------------------- delta function


def _delta_case(rs: np.random.RandomState, d: int, m: int) -> tuple:
    dx = rs.uniform(0.5e-4, 2e-4, size=d)
    N = rs.randint(2, 40, size=d).astype(float)
    xi = rs.uniform(-1e-3, 1e-3, size=d)
    # grid points around xi, some exactly at xi along one or more axes
    k = rs.randint(-6, 7, size=(d, m)).astype(float)
    x = xi[:, None] + k * dx[:, None] + rs.choice([0, 0.5, 0.3], size=(d, m)) * dx[:, None]
    x[:, : max(1, m // 5)] = xi[:, None]  # x == xi exactly → 1
    return x, xi, dx, N


def test_evaluate_delta_function_cases(matlab_engine: Any) -> None:
    rs = np.random.RandomState(3)
    cases = [_delta_case(rs, d, m) for d in (1, 2, 3) for m in (1, 7, 50)]
    # odd and even sizes on the same axis set, and N = 1
    x = np.array([[0.0, 1e-4, 2.5e-4, -3e-4], [0.0, 0.0, 1e-4, 5e-5], [1e-4, 0.0, 0.0, 2e-4]])
    for N in ([4, 5, 6], [3, 3, 3], [1, 2, 7], [128, 75, 1]):
        cases.append(
            (x, np.array([0.0, 0.0, 0.0]), np.array([1e-4, 1e-4, 1e-4]), np.asarray(N, float))
        )
    out = batch(
        matlab_engine,
        "@evaluate_delta_function",
        [c[0] for c in cases],
        [col(c[1]) for c in cases],
        [col(c[2]) for c in cases],
        [col(c[3]) for c in cases],
    )[0]
    for case, ref in zip(cases, out, strict=True):
        mine = acoustic.evaluate_delta_function(*case)
        assert ref.shape == (1, case[0].shape[1])
        assert_signal(mine, ref.reshape(-1), err_msg=f"N={case[3]}")


@ORACLE_SETTINGS
@given(st.integers(0, 2**31 - 1), st.integers(1, 3), st.integers(1, 30))
def test_evaluate_delta_function_random(matlab_engine: Any, seed: int, d: int, m: int) -> None:
    case = _delta_case(np.random.RandomState(seed), d, m)
    (ref,) = batch(
        matlab_engine,
        "@evaluate_delta_function",
        [case[0]],
        [col(case[1])],
        [col(case[2])],
        [col(case[3])],
    )[0]
    assert_signal(acoustic.evaluate_delta_function(*case), ref.reshape(-1))


# ------------------------------------------------------- sinc interpolation


def test_sinc_interpolation_shapes(matlab_engine: Any) -> None:
    rs = np.random.RandomState(7)
    X = np.arange(20) * 1e-8 + 3e-7
    Xq_in = X + 0.5e-8  # half-sample offsets (tie-like)
    Xq_out = np.concatenate(
        [X[:3] - 5e-8, X[-3:] + 7e-8, X[[0, 5, 19]]]
    )  # out of range + on samples
    V = rs.standard_normal(20)
    M = rs.standard_normal((20, 3))
    cases = [
        (row(X), row(V), row(Xq_in)),
        (row(X), col(V), row(Xq_in)),
        (col(X), row(V), col(Xq_out)),
        (row(X), M, row(Xq_out)),
        (row(X), M[:, :1], row(Xq_in)),
        (row(X[:1]), row(V[:1]), row(X[:1])),  # one sample: mean(diff) is NaN in both
    ]
    out = batch(matlab_engine, "@sinc_interpolation", *[list(c) for c in zip(*cases, strict=True)])[
        0
    ]
    py_args = [
        (X, V, Xq_in),
        (X, V.reshape(-1, 1), Xq_in),
        (X, V, Xq_out),
        (X, M, Xq_out),
        (X, M[:, :1], Xq_in),
        (X[:1], V[:1], X[:1]),
    ]
    for k, (args, ref) in enumerate(zip(py_args, out, strict=True)):
        mine = acoustic.sinc_interpolation(*args)
        if ref.shape[0] == 1:  # MATLAB row ↔ Python 1-D
            ref = ref.reshape(-1)
        assert_signal(mine, ref, err_msg=f"case {k}")


@ORACLE_SETTINGS
@given(
    st.integers(0, 2**31 - 1),
    st.integers(2, 60),
    st.integers(1, 80),
    st.floats(0.25, 3.0),
    st.integers(1, 4),
)
def test_sinc_interpolation_random(
    matlab_engine: Any, seed: int, n: int, nq: int, span: float, k: int
) -> None:
    rs = np.random.RandomState(seed)
    dt = rs.uniform(1e-9, 1e-7)
    X = rs.uniform(-1, 1) * 1e-6 + np.arange(n) * dt
    # queries inside and outside [X(1), X(end)], some on samples and at half samples
    Xq = X[0] + (rs.uniform(-0.5, span, nq) * (n - 1)) * dt
    Xq[: nq // 3] = X[rs.randint(0, n, nq // 3)]
    V = rs.standard_normal((n, k))
    (ref,) = batch(matlab_engine, "@sinc_interpolation", [row(X)], [V], [row(Xq)])[0]
    mine = acoustic.sinc_interpolation(X, V, Xq)
    if k == 1:
        ref = ref.reshape(-1)
    assert_signal(mine, ref)


# ------------------------------------------------------------ resampling

RESAMPLE_CASES = [
    (25, 250e6, 100e6),  # downsampling (preprocess_transmit-like)
    (40, 50e6, 125e6),  # upsampling
    (31, 33e6, 33e6),  # same rate: identity
    (1, 10e6, 20e6),  # one sample → one sample
    (11, 10.0, 25.0),  # t_old(end)*fs_new = 25 exactly: floor on an integer
    (7, 3.0, 7.0),  # (N-1)/fs_old*fs_new = 14: exact tie for floor
    (64, 62.5e6, 15e6),
    (100, 1e9, 33.3e6),
]


@pytest.mark.parametrize("orient", ["row", "col"])
def test_resample_signal(matlab_engine: Any, orient: str) -> None:
    rs = np.random.RandomState(11)
    sigs = [rs.standard_normal(n) * 1e5 for n, _, _ in RESAMPLE_CASES]
    wrap = row if orient == "row" else col
    out = batch(
        matlab_engine,
        "@(p, a, b) resample_signal(p, a, b, false)",
        [wrap(s) for s in sigs],
        [float(c[1]) for c in RESAMPLE_CASES],
        [float(c[2]) for c in RESAMPLE_CASES],
    )[0]
    for (n, fo, fn), s, ref in zip(RESAMPLE_CASES, sigs, out, strict=True):
        assert ref.shape[0] == 1  # MATLAB returns a row
        mine = acoustic.resample_signal(s, fo, fn)
        assert_signal(mine, ref.reshape(-1), err_msg=f"{n} {fo} {fn}")


@ORACLE_SETTINGS
@given(st.integers(0, 2**31 - 1), st.integers(1, 120), st.floats(1e6, 1e9), st.floats(1e6, 1e9))
def test_resample_signal_random(
    matlab_engine: Any, seed: int, n: int, fo: float, fn: float
) -> None:
    s = np.random.RandomState(seed).standard_normal(n)
    (ref,) = batch(
        matlab_engine, "@(p, a, b) resample_signal(p, a, b, false)", [row(s)], [fo], [fn]
    )[0]
    assert_signal(acoustic.resample_signal(s, fo, fn), ref.reshape(-1))
