"""Unit tests for the clean-room ode45 (no MATLAB needed).

Pinned values come from MATLAB R2025a runs recorded by the oracle tests
(``tests/oracle/test_ode45.py``), so CI catches regressions in step control.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyproteus.numerics import ode45, odeset
from pyproteus.numerics.ode45 import _fdlibm_pow, _pow5


def vdp(t, y):
    return np.array([y[1], 5 * (1 - y[0] * y[0]) * y[1] - y[0]])


def test_matlab_pinned_step_sequence_vdp() -> None:
    """van der Pol mu=5, [0 20], Refine 1: MATLAB R2025a stats and steps, bit for bit."""
    r = ode45(vdp, [0, 20], [2.0, 0.0], odeset(Refine=1))
    assert (r.stats.nsteps, r.stats.nfailed, r.stats.nfevals) == (109, 15, 745)
    assert r.t_steps.size == 110
    expected = {
        1: "0x1.a56cb36416f7cp-14",
        2: "0x1.3c11868b1139ep-11",
        10: "0x1.b7b067274b74fp-2",
        50: "0x1.19223bb2ad6bdp+3",
        108: "0x1.3b9b88d43d4f9p+4",
        109: "0x1.4000000000000p+4",
    }
    for i, h in expected.items():
        assert r.t_steps[i] == float.fromhex(h), i
    assert r.y_steps[-1, 0] == float.fromhex("-0x1.99e3aa183444cp+0")
    assert r.y_steps[-1, 1] == float.fromhex("0x1.9653660100596p-3")
    np.testing.assert_array_equal(r.t, r.t_steps)


def test_matlab_pinned_tspan_output() -> None:
    lam = np.array([0.5, 1, 2, 3, 5, 7, 10, 13, 17], dtype=float)
    tspan = np.arange(301) * 0.01
    r = ode45(
        lambda t, y: -lam * y,
        tspan,
        np.ones(9),
        odeset(RelTol=1e-6, AbsTol=1e-6, InitialStep=1e-12),
    )
    assert (r.stats.nsteps, r.stats.nfailed, r.stats.nfevals) == (56, 0, 337)
    assert r.t_steps[-2] == float.fromhex("0x1.70ce456b30b05p+1")
    np.testing.assert_array_equal(r.t, tspan)
    expected = [
        0.22313016103943567,
        0.04978708374468299,
        0.0024788224931813667,
        0.00012346601547957,
        3.1164833191149024e-07,
        9.722746864029714e-10,
        2.090319252530042e-12,
        6.817878281112521e-13,
        2.495029281990317e-11,
    ]
    np.testing.assert_allclose(r.y[-1], expected, rtol=0, atol=2e-16)


@pytest.mark.parametrize(
    ("x", "expected"),
    [  # MATLAB x^0.2 where libm x**0.2 differs by an ulp
        ("0x1.2f0bb5fb25cc8p-26", "0x1.cd04887f5caeep-6"),
        ("0x1.4b89e480b221cp+2", "0x1.63b8d914fc552p+0"),
        ("0x1.fc13c893a0789p+2", "0x1.836d524f1f396p+0"),
        ("0x1.0512118d479e6p-18", "0x1.531f268abec88p-4"),
        ("0x1.c48669f7fd2fdp-17", "0x1.b2d9595d60b2ap-4"),
        ("0x1.cf455614ef087p-17", "0x1.b4e5176761260p-4"),
    ],
)
def test_pow5_matches_matlab(x, expected) -> None:
    assert _pow5(float.fromhex(x)) == float.fromhex(expected)


def test_pow5_close_to_libm_and_special_values() -> None:
    rng = np.random.default_rng(0)
    x = np.concatenate([10.0 ** rng.uniform(-307, 307, 5000), [5e-324, 1e-310, 1.0, 32.0]])
    ours = np.array([_pow5(v) for v in x])
    ulps = np.abs(ours - x**0.2) / np.spacing(x**0.2)
    assert ulps.max() <= 1
    assert _pow5(32.0) == 2.0
    assert _pow5(1.0) == 1.0
    assert _pow5(0.0) == 0.0
    assert _pow5(math.inf) == math.inf
    assert math.isnan(_pow5(math.nan))
    assert _fdlibm_pow(2.0, -3.3) == pytest.approx(2.0**-3.3, rel=2e-16)


def test_accuracy_exponential_decay() -> None:
    for rtol in (1e-3, 1e-6, 1e-9):
        r = ode45(lambda t, y: -y, [0, 2], [1.0], odeset(RelTol=rtol, AbsTol=1e-12))
        err = np.max(np.abs(r.y[:, 0] - np.exp(-r.t)))
        assert err < 10 * rtol


def test_fsal_and_refine_counts() -> None:
    r = ode45(vdp, [0, 20], [2.0, 0.0])
    assert r.stats.nfevals == 6 * (r.stats.nsteps + r.stats.nfailed) + 1
    assert r.t.size == 4 * r.stats.nsteps + 1  # default Refine = 4
    for refine in (1, 2, 7):
        rr = ode45(vdp, [0, 20], [2.0, 0.0], odeset(Refine=refine))
        assert rr.t.size == refine * rr.stats.nsteps + 1
        np.testing.assert_array_equal(rr.t[::refine], rr.t_steps)
        np.testing.assert_array_equal(rr.y[::refine], rr.y_steps)


def test_many_point_tspan_ignores_refine_and_hits_points() -> None:
    tspan = [0.0, 0.3, 0.7, 2.0]
    r = ode45(lambda t, y: -y, tspan, [1.0], odeset(Refine=8))
    np.testing.assert_array_equal(r.t, tspan)
    assert r.y.shape == (4, 1)
    np.testing.assert_allclose(r.y[:, 0], np.exp(-np.array(tspan)), rtol=1e-3)


def test_backward_integration() -> None:
    r = ode45(lambda t, y: -y, [1.0, -1.0], [1.0], odeset(RelTol=1e-8, AbsTol=1e-12))
    assert r.t[0] == 1.0 and r.t[-1] == -1.0
    assert np.all(np.diff(r.t) < 0)
    np.testing.assert_allclose(r.y[:, 0], np.exp(1.0 - r.t), rtol=1e-6)


def test_max_step() -> None:
    r = ode45(lambda t, y: -0.01 * y, [0, 100], [1.0], odeset(Refine=1))
    np.testing.assert_array_equal(r.t_steps, np.arange(11) * 10.0)  # default 0.1 * span
    r = ode45(lambda t, y: -0.01 * y, [0, 1], [1.0], odeset(MaxStep=0.03))
    assert np.max(np.diff(r.t_steps)) <= 0.03 * 1.1


def test_initial_step_capped_by_first_tspan_interval() -> None:
    calls: list = []
    ode45(lambda t, y: -y, [0, 1e-3, 1.0], [1.0], record_calls=calls)
    assert calls[1][0] == pytest.approx(0.2e-3)  # stage 2 at t0 + h/5, h = first interval


def test_vectorised_batch_shares_steps() -> None:
    """Q-2: stacking decoupled systems changes the steps of each (MATLAB does the same)."""
    lam = np.array([1.0, 30.0])
    both = ode45(lambda t, y: -lam * y, [0, 1], [1.0, 1.0], odeset(Refine=1))
    one = ode45(lambda t, y: -lam[:1] * y, [0, 1], [1.0], odeset(Refine=1))
    assert both.t_steps.size != one.t_steps.size
    assert both.y_steps[-1, 0] != one.y_steps[-1, 0]


def test_outputfcn_stop_and_protocol() -> None:
    log = []

    def ofcn(t, y, flag):
        log.append((flag, np.array(t), np.shape(y)))
        return flag == "" and sum(e[0] == "" for e in log) == 3

    r = ode45(
        lambda t, y: -y, [0, 0.5, 1, 2, 3, 4, 5], [1.0, 2.0], odeset(OutputFcn=ofcn, MaxStep=0.6)
    )
    assert r.stopped
    assert log[0][0] == "init"
    np.testing.assert_array_equal(log[0][1], [0, 5])
    assert log[-1][0] == "done" and log[-1][2] == (0, 0)
    assert sum(e[0] == "" for e in log) == 3
    assert r.t[-1] < 5


def test_y0_shape_and_outputs() -> None:
    r = ode45(lambda t, y: -y, [0, 1], np.ones((2, 3)))
    assert r.y.shape[1] == 6
    t, y = r
    assert t is r.t and y is r.y


@pytest.mark.parametrize(
    ("kwargs", "msg"),
    [
        (dict(tspan=[0]), "at least 2"),
        (dict(tspan=[1, 1]), "differ"),
        (dict(tspan=[0, 2, 1]), "strictly"),
        (dict(options={"RelTol": -1}), "positive"),
        (dict(options={"AbsTol": [1e-6, 1e-6, 1e-6]}), "AbsTol"),
        (dict(options={"MaxStep": 0}), "MaxStep"),
        (dict(options={"InitialStep": -1}), "InitialStep"),
        (dict(options={"Events": None}), "unsupported"),
    ],
)
def test_argument_errors(kwargs, msg) -> None:
    args = dict(tspan=[0, 1], options=None) | kwargs
    with pytest.raises(ValueError, match=msg):
        ode45(lambda t, y: -y, args["tspan"], [1.0, 2.0], args["options"])


def test_rhs_size_checked() -> None:
    with pytest.raises(ValueError, match="expected 2"):
        ode45(lambda t, y: np.zeros(3), [0, 1], [1.0, 2.0])


def test_odeset_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        odeset(reltol=1e-3)
    assert odeset(RelTol=1e-3, BDF="on") == {"RelTol": 1e-3, "BDF": "on"}


def test_step_size_failure_matches_matlab() -> None:
    """y' = y^2 blows up at t = 1: MATLAB R2025a stops at the same step with this warning."""
    with pytest.warns(RuntimeWarning, match=r"Failure at t=9.999694e-01\.  Unable to meet"):
        r = ode45(lambda t, y: y * y, [0, 2], [1.0])
    assert r.stopped
    assert (r.stats.nsteps, r.stats.nfailed, r.stats.nfevals) == (66, 59, 751)
    assert r.t_steps[-1] == float.fromhex("0x1.fffbfe33a860fp-1")
    assert r.t.size == 265
    assert "(1.776357e-15)" in r.message
