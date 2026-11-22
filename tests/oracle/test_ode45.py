"""Oracle tests: ``pyproteus.numerics.ode45`` vs MATLAB R2025a ``ode45``, on solutions.

Parity policy: pyproteus reproduces PROTEUS' results within a tolerance; it
does not reproduce MATLAB's internal step sequence. Both solvers are
Dormand–Prince 5(4) with error control at the same ``RelTol``/``AbsTol``, so
their solutions agree to roughly the tolerance. Class D-13 (PR #10, bubble
ODEs at about 10·RelTol rel + AbsTol abs of the solver settings): per component,

    max_t |y_py - y_matlab| <= 10 * RelTol * max_t |y_matlab| + AbsTol

compared at ``tspan`` points (many-point ``tspan``) or at the final time
(2-point ``tspan``, where the output times are each solver's own steps).
Each case also checks both solvers against a tight reference solution
(``RelTol=1e-12``) with the same bound.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pyproteus.numerics import ode45, odeset
from tests.oracle.helpers import col, from_matlab, row, to_matlab

LAM9 = np.array([0.5, 1, 2, 3, 5, 7, 10, 13, 17], dtype=float)

RP_BASE = dict(
    rho=1000.0, P0=101325.0, c=1500.0, kappa=1.07, nu=2e-3, sig=0.02, Ks=1e-8, Pa=150e3, n=24.0
)


def rp_params(R0: np.ndarray, tau1: float = 25.0) -> dict[str, Any]:
    R0 = np.asarray(R0, dtype=float)
    T = float(np.median(np.sqrt(RP_BASE["rho"] * R0**2 / RP_BASE["P0"])))
    return dict(RP_BASE, R0=R0, T=T, tau1=tau1)


def rp_rhs(tau: float, vec: np.ndarray, p: dict[str, Any]) -> np.ndarray:
    """Python twin of ``tests/oracle/shims/ode45_rp_rhs.m``."""
    R0 = p["R0"]
    v = vec.reshape(2, R0.size, order="F")
    x, xd = v[0], v[1]
    u = tau / p["tau1"]
    if 0 < u < 1:
        z = 2 * u - 1
        tm, tc = 1.0, z
        for _ in range(2, int(p["n"]) + 1):
            tm, tc = tc, 2 * z * tc - tm
        P = p["Pa"] * (16 * u * u * (1 - u) * (1 - u)) * tc
    else:
        P = 0.0
    P0, rho, T, kappa = p["P0"], p["rho"], p["T"], p["kappa"]
    opx = 1 + x
    bracket = (
        (1 + 2 * p["sig"] / (R0 * P0))
        * opx ** (-3 * kappa)
        * (1 - 3 * kappa / p["c"] / T * R0 * xd)
        - 1
        - 2 * p["sig"] / (R0 * P0) / opx
        - 4 * p["nu"] / P0 / T * xd / opx
        - 4 * p["Ks"] / P0 / R0 / T * xd / (opx * opx)
        - P / P0
    )
    xdd = 1 / opx * (-1.5 * xd * xd + P0 / rho / (R0 * R0) * (T * T) * bracket)
    return np.stack([xd, xdd]).reshape(-1, order="F")


def py_rhs(name: str, p: dict[str, Any]):
    if name == "decay":
        lam = np.asarray(p["lam"], dtype=float).ravel()
        return lambda t, y: -lam * y
    if name == "rot":
        return lambda t, y: np.array([-p["w"] * y[1], p["w"] * y[0]])
    if name == "vdp":
        return lambda t, y: np.array([y[1], p["mu"] * (1 - y[0] ** 2) * y[1] - y[0]])
    if name == "scales":
        return lambda t, y: np.array(
            [-1e3 * (y[0] - np.cos(t)), 1e-6 * y[0] - 0.1 * y[1], y[0] * y[1] * 1e4]
        )
    if name == "rp":
        return lambda t, y: rp_rhs(t, y, p)
    raise KeyError(name)


def _mparams(p: dict[str, Any]) -> dict[str, Any]:
    return {k: to_matlab(row(v)) if isinstance(v, np.ndarray) else float(v) for k, v in p.items()}


def _mopts(opts: dict[str, Any]) -> list[Any]:
    cell: list[Any] = []
    for k, v in opts.items():
        cell.append(k)
        if isinstance(v, np.ndarray | list):
            cell.append(to_matlab(col(np.asarray(v, dtype=float))))
        else:
            cell.append(v if isinstance(v, str) else float(v))
    return cell


def matlab_solve(eng, rhs, tspan, y0, opts, p):
    t, y = eng.ode45_run(
        rhs,
        to_matlab(row(np.asarray(tspan, dtype=float))),
        to_matlab(col(np.asarray(y0, dtype=float))),
        _mopts(opts),
        _mparams(p),
        nargout=2,
    )
    return from_matlab(t).ravel(), from_matlab(y).reshape(-1, len(y0))


def assert_within_tol(actual, expected, rtol, atol, what) -> None:
    """Per component: max |a - e| <= 10*rtol*max|e| + atol."""
    actual = np.atleast_2d(actual)
    expected = np.atleast_2d(expected)
    assert actual.shape == expected.shape, f"{what}: shape {actual.shape} != {expected.shape}"
    atol = np.broadcast_to(np.asarray(atol, dtype=float), (expected.shape[1],))
    err = np.max(np.abs(actual - expected), axis=0)
    bound = 10 * rtol * np.max(np.abs(expected), axis=0) + atol
    bad = err > bound
    assert not bad.any(), (
        f"{what}: error {err[bad]} > bound {bound[bad]} (components {np.nonzero(bad)[0]})"
    )


def _rp_case(n_bubbles: int, seed: int = 3):
    rng = np.random.default_rng(seed)
    R0 = np.sort(rng.uniform(1.0e-6, 3.0e-6, n_bubbles), kind="stable")
    p = rp_params(R0)
    tau = np.linspace(0.0, 40.0, 1201)
    # PROTEUS calcBubbleResponse options ('BDF' does not apply to ode45).
    opts = {"BDF": "on", "AbsTol": 1e-6, "RelTol": 1e-6, "InitialStep": 1e-12}
    return p, tau, np.zeros(2 * n_bubbles), opts


# (id, rhs, tspan, y0, options, params)
CASES = [
    ("decay-default", "decay", [0, 1], [1.0], {}, {"lam": np.array([1.0])}),
    ("decay9-refine8", "decay", [0, 1], np.ones(9), {"Refine": 8, "RelTol": 1e-4}, {"lam": LAM9}),
    (
        "decay9-tspan",
        "decay",
        np.linspace(0, 3, 301),
        np.ones(9),
        {"RelTol": 1e-6, "AbsTol": 1e-6, "InitialStep": 1e-12},
        {"lam": LAM9},
    ),
    (
        "decay9-backward",
        "decay",
        np.linspace(2, -1, 41),
        np.ones(9),
        {"RelTol": 1e-5, "AbsTol": 1e-8},
        {"lam": LAM9},
    ),
    ("decay3-backward-2pt", "decay", [1, -2], [1, 2, 3.0], {"RelTol": 1e-4}, {"lam": LAM9[:3]}),
    (
        "maxstep-abstol-vector",
        "decay",
        np.linspace(0, 4, 81),
        [1, 1e-3, 5],
        {"MaxStep": 0.05, "AbsTol": [1e-6, 1e-9, 1e-4], "RelTol": 1e-5},
        {"lam": LAM9[:3]},
    ),
    (
        "normcontrol",
        "decay",
        np.linspace(0, 4, 41),
        [1, 1e-3, 5],
        {"NormControl": "on", "RelTol": 1e-5},
        {"lam": LAM9[:3]},
    ),
    (
        "rotation",
        "rot",
        np.linspace(0, 1, 977),
        [1.0, 0.0],
        {"RelTol": 1e-6, "AbsTol": 1e-6},
        {"w": 30.0},
    ),
    ("tspan-sparse", "rot", [0, 1e-3, 0.5, 3], [1.0, 0.0], {"RelTol": 1e-6}, {"w": 3.0}),
    (
        "vdp-mu1",
        "vdp",
        np.linspace(0, 20, 401),
        [2.0, 0.0],
        {"RelTol": 1e-6, "AbsTol": 1e-8},
        {"mu": 1.0},
    ),
    ("vdp-mu5-2pt", "vdp", [0, 6], [2.0, 0.0], {"RelTol": 1e-7, "AbsTol": 1e-9}, {"mu": 5.0}),
    (
        "scales",
        "scales",
        np.linspace(0, 3, 301),
        [0, 1e-3, 1e-5],
        {"RelTol": 1e-6, "AbsTol": 1e-6, "InitialStep": 1e-12},
        {},
    ),
]


def _check(r_t, r_y, tm, ym, tspan, opts, label) -> None:
    rtol = opts.get("RelTol", 1e-3)
    atol = opts.get("AbsTol", 1e-6)
    if len(tspan) > 2:
        np.testing.assert_allclose(r_t, tm, rtol=1e-15, atol=0)
        assert_within_tol(r_y, ym, rtol, atol, label)
    else:
        assert r_t[-1] == tm[-1] == tspan[-1]
        assert_within_tol(r_y[-1:], ym[-1:], rtol, atol, label + " (final value)")


@pytest.mark.parametrize(
    ("rhs", "tspan", "y0", "opts", "p"), [c[1:] for c in CASES], ids=[c[0] for c in CASES]
)
def test_solution_matches_matlab(matlab_engine, rhs, tspan, y0, opts, p) -> None:
    tm, ym = matlab_solve(matlab_engine, rhs, tspan, y0, opts, p)
    fun = py_rhs(rhs, p)
    r = ode45(fun, tspan, y0, odeset(**opts))
    _check(r.t, r.y, tm, ym, tspan, opts, "pyproteus vs MATLAB")
    ref = ode45(fun, tspan, y0, odeset(**(opts | {"RelTol": 1e-12, "AbsTol": 1e-14})))
    _check(r.t, r.y, ref.t, ref.y, tspan, opts, "pyproteus vs reference")
    _check(tm, ym, ref.t, ref.y, tspan, opts, "MATLAB vs reference")


@pytest.mark.parametrize("n_bubbles", [1, 4, 16])
def test_rayleigh_plesset_batch(matlab_engine, n_bubbles) -> None:
    """Stacked RP-like batch with PROTEUS' ode45 options."""
    p, tau, y0, opts = _rp_case(n_bubbles)
    tm, ym = matlab_solve(matlab_engine, "rp", tau, y0, opts, p)
    r = ode45(lambda t, y: rp_rhs(t, y, p), tau, y0, odeset(**opts))
    assert r.stats.nsteps > 200, "case too easy"
    _check(r.t, r.y, tm, ym, tau, opts, f"RP batch of {n_bubbles}")


def test_batch_dependence_similar(matlab_engine) -> None:
    """Q-50: a bubble's response depends on its batch, in MATLAB and pyproteus, within the class."""
    p, tau, y0, opts = _rp_case(3)
    p1 = dict(p, R0=p["R0"][:1])
    _, m_stacked = matlab_solve(matlab_engine, "rp", tau, y0, opts, p)
    _, m_single = matlab_solve(matlab_engine, "rp", tau, y0[:2], opts, p1)
    py_stacked = ode45(lambda t, y: rp_rhs(t, y, p), tau, y0, odeset(**opts)).y
    py_single = ode45(lambda t, y: rp_rhs(t, y, p1), tau, y0[:2], odeset(**opts)).y

    def rel(a, b):
        return np.max(np.abs(a[:, 0] - b[:, 0])) / np.max(np.abs(b[:, 0]))

    assert rel(m_single, m_stacked) > 0
    assert rel(py_single, py_stacked) > 0
    assert rel(py_single, py_stacked) < 10 * opts["RelTol"]
    assert rel(m_single, m_stacked) < 10 * opts["RelTol"]
