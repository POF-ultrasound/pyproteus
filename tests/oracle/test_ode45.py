"""Oracle tests: clean-room ``ode45`` against MATLAB R2025a ``ode45``.

Black-box comparison only (no MathWorks source): the shim
``tests/oracle/shims/ode45_probe.m`` logs every right-hand-side call
``(t, y, f)`` that MATLAB's ode45 makes, plus ``[t, y]`` and ``sol``.

What is asserted:

- **Step sequences, bit for bit.** Every RHS call (time and state) is
  identical to MATLAB's, so the accepted steps, the rejected steps and the
  solution at the steps are identical. For nonlinear RHS this is checked in
  *replay* mode (the Python RHS returns MATLAB's own ``f`` after checking that
  its inputs are bit-identical), which isolates the solver from differences in
  RHS arithmetic; for RHS made of plain IEEE operations it is also checked
  with the Python RHS itself.
- **Dense output** (``Refine`` points, ``tspan`` points inside a step): within
  1e-15 of the solution scale. One-equation systems are bit-identical; with
  several equations MATLAB's evaluation order is not pinned down yet (B-50).
- **Rayleigh–Plesset-like batch** (PROTEUS options): solutions at ``tspan``
  within the bubble class of ROADMAP §4 (1e-8 rel), with both an fdlibm-exact
  RHS and a plain-NumPy RHS (whose ``**`` differs from MATLAB's by an ulp).
- MATLAB's scalar ``x^0.2`` is fdlibm's ``pow`` (used by the step-size rule).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from pyproteus.numerics import ode45, odeset
from pyproteus.numerics.ode45 import _fdlibm_pow, _pow5
from tests.oracle.helpers import assert_rel, assert_same_float, col, from_matlab, row, to_matlab

LAM9 = np.array([0.5, 1, 2, 3, 5, 7, 10, 13, 17], dtype=float)

# Rayleigh-Plesset-like parameters (SI), PROTEUS-style nondimensionalisation.
RP_BASE = dict(
    rho=1000.0,
    P0=101325.0,
    c=1500.0,
    kappa=1.07,
    nu=2e-3,
    sig=0.02,
    Ks=1e-8,
    Pa=150e3,
    n=24.0,
)


def rp_params(R0: np.ndarray, tau1: float = 25.0) -> dict[str, Any]:
    R0 = np.asarray(R0, dtype=float)
    T = float(np.median(np.sqrt(RP_BASE["rho"] * R0**2 / RP_BASE["P0"])))
    return dict(RP_BASE, R0=R0, T=T, tau1=tau1)


def rp_rhs(tau: float, vec: np.ndarray, p: dict[str, Any], power=None) -> np.ndarray:
    """Python twin of ``ode45_rp_rhs.m`` (same IEEE operations in the same order)."""
    if power is None:

        def power(b, e):
            return b**e

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
        * power(opx, -3 * kappa)
        * (1 - 3 * kappa / p["c"] / T * R0 * xd)
        - 1
        - 2 * p["sig"] / (R0 * P0) * (1 / opx)
        - 4 * p["nu"] / P0 / T * xd / opx
        - 4 * p["Ks"] / P0 / R0 / T * xd / (opx * opx)
        - P / P0
    )
    xdd = 1 / opx * (-3 / 2 * xd * xd + P0 / rho / (R0 * R0) * (T * T) * bracket)
    return np.stack([xd, xdd]).reshape(-1, order="F")


def fdlibm_power(b: np.ndarray, e: float) -> np.ndarray:
    return np.array([_fdlibm_pow(float(v), e) for v in np.ravel(b)]).reshape(np.shape(b))


def py_rhs(name: str, p: dict[str, Any]):
    if name == "decay":
        lam = np.asarray(p["lam"], dtype=float).ravel()
        return lambda t, y: -lam * y
    if name == "rot":
        w = p["w"]
        return lambda t, y: np.array([-w * y[1], w * y[0]])
    if name == "vdp":
        mu = p["mu"]
        # y[0]*y[0], not y[0]**2: NumPy's scalar ** is not correctly rounded (B-51).
        return lambda t, y: np.array([y[1], mu * (1 - y[0] * y[0]) * y[1] - y[0]])
    if name == "scales":
        return lambda t, y: np.array(
            [-1e3 * (y[0] - 1 + t * t), 1e-6 * y[0] - 0.1 * y[1], y[0] * y[1] * 1e4]
        )
    if name == "rp":
        return lambda t, y: rp_rhs(t, y, p, fdlibm_power)
    raise KeyError(name)


def _mparams(p: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in p.items():
        if isinstance(v, np.ndarray):
            out[k] = to_matlab(row(v)) if v.ndim == 1 else to_matlab(v)
        else:
            out[k] = float(v)
    return out


def _mopts(opts: dict[str, Any]) -> list[Any]:
    cell: list[Any] = []
    for k, v in opts.items():
        cell.append(k)
        if isinstance(v, np.ndarray | list):
            cell.append(to_matlab(col(np.asarray(v, dtype=float))))
        elif isinstance(v, str):
            cell.append(v)
        else:
            cell.append(float(v))
    return cell


def matlab_probe(eng, rhs: str, tspan, y0, opts: dict[str, Any], p: dict[str, Any]) -> dict:
    out = eng.ode45_probe(
        rhs,
        to_matlab(row(np.asarray(tspan, dtype=float))),
        to_matlab(col(np.asarray(y0, dtype=float))),
        _mopts(opts),
        _mparams(p),
        nargout=1,
    )
    res = {k: from_matlab(v) for k, v in out.items()}
    res["calls_t"] = res["calls_t"].ravel()
    res["t"] = res["t"].ravel()
    res["sol_x"] = res["sol_x"].ravel()
    res["stats"] = res["stats"].ravel().astype(int)
    return res


class Replay:
    """RHS that checks each call against MATLAB's log and returns MATLAB's ``f``."""

    def __init__(self, m: dict) -> None:
        self.m = m
        self.k = 0

    def __call__(self, t: float, y: np.ndarray) -> np.ndarray:
        k = self.k
        m = self.m
        assert k < m["calls_t"].size, f"more RHS calls than MATLAB ({m['calls_t'].size})"
        assert t == m["calls_t"][k], f"call {k}: t {t!r} != MATLAB {m['calls_t'][k]!r}"
        assert np.array_equal(y, m["calls_y"][k]), (
            f"call {k}: y differs from MATLAB by {np.max(np.abs(y - m['calls_y'][k])):.3g}"
        )
        self.k += 1
        return m["calls_f"][k].copy()


def check_steps(r, m: dict, calls: list | None = None) -> None:
    """Bit-identical RHS calls, stats and step sequence."""
    if calls is not None:
        ct = np.array([c[0] for c in calls])
        cy = np.array([c[1] for c in calls])
        assert ct.size == m["calls_t"].size, f"{ct.size} RHS calls, MATLAB {m['calls_t'].size}"
        assert_same_float(ct, m["calls_t"], "RHS call times")
        assert_same_float(cy, m["calls_y"].reshape(cy.shape), "RHS call states")
    assert (r.stats.nsteps, r.stats.nfailed, r.stats.nfevals) == tuple(m["stats"])
    assert_same_float(r.t_steps, m["sol_x"], "accepted steps")
    assert_same_float(r.y_steps, m["sol_y"].reshape(r.y_steps.shape), "solution at steps")


def check_output(r, m: dict, exact: bool = False) -> None:
    assert_same_float(r.t, m["t"], "output times")
    my = m["y"].reshape(r.y.shape)
    if exact:
        assert_same_float(r.y, my, "output")
    else:
        assert_rel(r.y, my, rel=1e-15, err_msg="dense output")


# (id, rhs, tspan, y0, options, params, 1-equation dense output exact?)
CASES = [
    ("decay-default", "decay", [0, 1], [1.0], {}, {"lam": np.array([1.0])}, True),
    ("decay-refine1", "decay", [0, 1], [1.0], {"Refine": 1}, {"lam": np.array([1.0])}, True),
    (
        "decay-refine7",
        "decay",
        [0, 3],
        [2.0],
        {"Refine": 7, "RelTol": 1e-6},
        {"lam": np.array([1.3])},
        False,
    ),
    (
        "decay9-refine8",
        "decay",
        [0, 1],
        np.ones(9),
        {"Refine": 8, "RelTol": 1e-2, "AbsTol": 1e-2},
        {"lam": LAM9},
        False,
    ),
    (
        "decay9-tspan",
        "decay",
        np.arange(301) * 0.01,
        np.ones(9),
        {"RelTol": 1e-6, "AbsTol": 1e-6, "InitialStep": 1e-12},
        {"lam": LAM9},
        False,
    ),
    (
        "decay9-backward",
        "decay",
        np.arange(41) * -0.075 + 2,
        np.ones(9),
        {"RelTol": 1e-5, "AbsTol": 1e-8},
        {"lam": LAM9},
        False,
    ),
    (
        "decay3-backward-2pt",
        "decay",
        [1, -2],
        [1, 2, 3.0],
        {"RelTol": 1e-4},
        {"lam": LAM9[:3]},
        False,
    ),
    ("maxstep-default", "decay", [0, 100], [1.0], {"Refine": 1}, {"lam": np.array([0.01])}, True),
    (
        "maxstep-abstol-vector",
        "decay",
        [0, 4],
        [1, 1e-3, 5],
        {"MaxStep": 0.05, "AbsTol": [1e-6, 1e-9, 1e-4], "RelTol": 1e-5, "Refine": 2},
        {"lam": LAM9[:3]},
        False,
    ),
    (
        "normcontrol",
        "decay",
        [0, 4],
        [1, 1e-3, 5],
        {"NormControl": "on", "RelTol": 1e-5, "Refine": 1},
        {"lam": LAM9[:3]},
        True,
    ),
    (
        "rotation-auto-h0",
        "rot",
        np.arange(977) / 976,
        [1.0, 0.0],
        {"RelTol": 1e-6, "AbsTol": 1e-6},
        {"w": 30.0},
        False,
    ),
    ("tspan-sparse", "rot", [0, 1e-3, 0.5, 3], [1.0, 0.0], {"RelTol": 1e-6}, {"w": 3.0}, False),
    (
        "tspan-dense-pts",
        "decay",
        np.arange(2001) * 5e-4,
        [1.0, 2.0],
        {"RelTol": 1e-3},
        {"lam": np.array([1.0, 4.0])},
        False,
    ),
    ("tspan-2pts-vs-3pts", "decay", [0, 0.5, 1], [1.0], {}, {"lam": np.array([1.0])}, True),
    ("vdp-rejections", "vdp", [0, 20], [2.0, 0.0], {"Refine": 1}, {"mu": 5.0}, False),
    (
        "vdp-proteus-opts",
        "vdp",
        np.arange(301) / 15,
        [2.0, 0.0],
        {"RelTol": 1e-6, "AbsTol": 1e-6, "InitialStep": 1e-12},
        {"mu": 5.0},
        False,
    ),
    (
        "scales",
        "scales",
        [0, 3],
        [0, 1e-3, 1e-5],
        {"RelTol": 1e-6, "AbsTol": 1e-6, "InitialStep": 1e-12, "Refine": 1},
        {},
        False,
    ),
]


@pytest.mark.parametrize(
    ("rhs", "tspan", "y0", "opts", "p", "exact"), [c[1:] for c in CASES], ids=[c[0] for c in CASES]
)
def test_step_sequence_bit_identical(matlab_engine, rhs, tspan, y0, opts, p, exact) -> None:
    m = matlab_probe(matlab_engine, rhs, tspan, y0, opts, p)
    # Replay: solver alone, independent of RHS arithmetic.
    rep = Replay(m)
    r = ode45(rep, tspan, y0, odeset(**opts))
    assert rep.k == m["calls_t"].size
    check_steps(r, m)
    check_output(r, m, exact=exact)
    # Python's own RHS (plain IEEE operations): same calls as MATLAB.
    calls: list = []
    r2 = ode45(py_rhs(rhs, p), tspan, y0, odeset(**opts), record_calls=calls)
    check_steps(r2, m, calls)
    check_output(r2, m, exact=exact)


def _rp_case(n_bubbles: int, seed: int = 3):
    rng = np.random.default_rng(seed)
    R0 = np.sort(rng.uniform(1.0e-6, 3.0e-6, n_bubbles), kind="stable")
    p = rp_params(R0)
    tau = np.arange(1201) * (40.0 / 1200)
    y0 = np.zeros(2 * n_bubbles)
    # PROTEUS calcBubbleResponse options ('BDF' is ignored by ode45).
    opts = {"BDF": "on", "AbsTol": 1e-6, "RelTol": 1e-6, "InitialStep": 1e-12}
    return p, tau, y0, opts


@pytest.mark.parametrize("n_bubbles", [1, 4])
def test_rayleigh_plesset_batch(matlab_engine, n_bubbles) -> None:
    p, tau, y0, opts = _rp_case(n_bubbles)
    m = matlab_probe(matlab_engine, "rp", tau, y0, opts, p)
    assert m["stats"][0] > 200, "case too easy"
    rep = Replay(m)
    r = ode45(rep, tau, y0, odeset(**opts))
    check_steps(r, m)
    check_output(r, m)
    # fdlibm-exact Python RHS: bit-identical step sequence end to end.
    calls: list = []
    r2 = ode45(
        lambda t, y: rp_rhs(t, y, p, fdlibm_power), tau, y0, odeset(**opts), record_calls=calls
    )
    check_steps(r2, m, calls)
    check_output(r2, m)
    # Plain NumPy RHS (libm pow): steps drift by ulps; bubble class 1e-8 rel.
    r3 = ode45(lambda t, y: rp_rhs(t, y, p), tau, y0, odeset(**opts))
    my = m["y"].reshape(r3.y.shape)
    for i in range(n_bubbles):  # per bubble, radius x and velocity xdot
        for c in (2 * i, 2 * i + 1):
            assert_rel(r3.y[:, c], my[:, c], rel=1e-8, err_msg=f"bubble {i} comp {c}")


def test_batch_dependence_quirk(matlab_engine) -> None:
    """Q-2: a bubble's response depends on the batch it is integrated in (MATLAB and Python)."""
    p, tau, y0, opts = _rp_case(3)
    stacked = matlab_probe(matlab_engine, "rp", tau, y0, opts, p)["y"]
    r = ode45(lambda t, y: rp_rhs(t, y, p, fdlibm_power), tau, y0, odeset(**opts))
    assert_rel(r.y, stacked.reshape(r.y.shape), rel=1e-15)
    p1 = dict(p, R0=p["R0"][:1])
    single = matlab_probe(matlab_engine, "rp", tau, y0[:2], opts, p1)["y"]
    r1 = ode45(lambda t, y: rp_rhs(t, y, p1, fdlibm_power), tau, y0[:2], odeset(**opts))
    assert_rel(r1.y, single.reshape(r1.y.shape), rel=1e-15)
    diff = np.max(np.abs(single[:, 0] - stacked[:, 0])) / np.max(np.abs(single[:, 0]))
    assert diff > 1e-9, f"batch dependence expected in MATLAB, got {diff:.2g}"


def _recording_outputfcn(plog: list, stop_at: int):
    count = 0

    def ofcn(t, y, flag):
        nonlocal count
        plog.append((flag, np.asarray(t, dtype=float).ravel(), tuple(np.shape(y))))
        if flag == "":
            count += 1
            return stop_at > 0 and count == stop_at
        return False

    return ofcn


def test_outputfcn_protocol(matlab_engine) -> None:
    """Flags, arguments and stop status of OutputFcn match MATLAB."""
    for tspan, refine, stop_at in (
        ([0, 1e-3, 0.5, 3], None, 0),
        ([0, 3], 3, 0),
        ([0, 3], None, 4),
        (np.arange(31) * 0.1, None, 5),
    ):
        optcell = ["RelTol", 1e-6] + ([] if refine is None else ["Refine", float(refine)])
        flags, times, sizes, mt, my = matlab_engine.ode45_outputfcn_probe(
            to_matlab(row(np.asarray(tspan, dtype=float))),
            to_matlab(col(np.array([1.0, 2.0]))),
            optcell,
            float(stop_at),
            nargout=5,
        )
        sizes = from_matlab(sizes).astype(int)
        mlog = []
        for flag, t, sz in zip(flags, times, sizes, strict=True):
            sz = tuple(sz)
            if flag == "init":
                sz = (sz[0],)  # MATLAB y0 column; Python passes y0 as a 1-D array
            flag = flag if isinstance(flag, str) else ""  # empty char arrives as []
            mlog.append((flag, np.asarray(t, dtype=float).ravel(), sz))
        plog: list = []
        ofcn = _recording_outputfcn(plog, stop_at)
        opts = {"RelTol": 1e-6, "OutputFcn": ofcn}
        if refine is not None:
            opts["Refine"] = refine
        r = ode45(lambda t, y: -y, tspan, [1.0, 2.0], odeset(**opts))
        assert [e[0] for e in plog] == [e[0] for e in mlog]
        for (_pf, pt, ps), (mf, mt_, ms) in zip(plog, mlog, strict=True):
            assert_same_float(pt, mt_, f"OutputFcn t, flag {mf!r}")
            assert ps == ms, f"OutputFcn y size {ps} != {ms} (flag {mf!r})"
        assert_same_float(r.t, from_matlab(mt).ravel())
        assert_rel(r.y, from_matlab(my).reshape(r.y.shape), rel=1e-15)
        assert r.stopped == (stop_at > 0)


def test_step_rule_power_is_fdlibm(matlab_engine) -> None:
    """MATLAB x^0.2 (scalar and array) equals the fdlibm port, incl. subnormals and extremes."""
    rng = np.random.default_rng(11)
    x = np.concatenate(
        [
            10.0 ** rng.uniform(-300, 300, 20000),
            10.0 ** rng.uniform(-8, 2, 20000),
            rng.uniform(0.5, 2.0, 5000),
            [5e-324, 1e-310, 2.2250738585072014e-308, 1.0, 2.0, 0.2, 1.7976931348623157e308],
        ]
    )
    matlab_engine.workspace["pyp_x"] = to_matlab(row(x))
    matlab_engine.eval("pyp_p = pyp_x.^0.2; pyp_s = arrayfun(@(v) v^0.2, pyp_x);", nargout=0)
    mp = from_matlab(matlab_engine.workspace["pyp_p"]).ravel()
    ms = from_matlab(matlab_engine.workspace["pyp_s"]).ravel()
    matlab_engine.eval("clear pyp_x pyp_p pyp_s", nargout=0)
    assert_same_float(mp, ms, "MATLAB array vs scalar power")
    ours = np.array([_pow5(v) for v in x])
    assert_same_float(ours, mp, "fdlibm pow port")
    assert np.mean(x**0.2 != mp) > 0.02  # libm pow really differs (why the port exists)
