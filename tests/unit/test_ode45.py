"""Unit tests for ``pyproteus.numerics.ode45`` against analytical solutions (no MATLAB)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from pyproteus.numerics import ode45, odeset
from pyproteus.numerics.ode45 import _A, _B, _C, _D, _E


def bound(y_exact, rtol, atol):
    """Accuracy class used throughout (D-13): 10*RelTol*max|y| + AbsTol per component."""
    return 10 * rtol * np.max(np.abs(y_exact), axis=0) + atol


def test_tableau_consistency() -> None:
    np.testing.assert_allclose(_A.sum(axis=1), _C, atol=1e-15)  # row sums = nodes
    assert _B.sum() == pytest.approx(1.0, abs=1e-15)
    assert _E.sum() == pytest.approx(0.0, abs=1e-15)  # b5 - b4 sums to zero
    assert _D.sum() == pytest.approx(0.0, abs=1e-14)
    # 5th-order conditions (a few): sum b c^k = 1/(k+1)
    for k in range(5):
        assert _B @ _C**k == pytest.approx(1 / (k + 1), abs=1e-14)
    # embedded 4th-order weights b4 = b5 - E (Dormand & Prince 1980)
    b4 = _B - _E
    published = [5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40]
    np.testing.assert_allclose(b4, published, rtol=0, atol=1e-16)
    for k in range(4):
        assert b4 @ _C**k == pytest.approx(1 / (k + 1), abs=1e-14)
    assert b4 @ _C**4 != pytest.approx(1 / 5, abs=1e-6)  # really only 4th order


def _one_step(f, t, y, h):
    k = np.zeros((y.size, 7))
    k[:, 0] = f(t, y)
    for i in range(1, 7):
        k[:, i] = f(t + _C[i] * h, y + h * (k[:, :i] @ _A[i, :i]))
    return y + h * (k @ _B), h * (k @ _E)


@pytest.mark.parametrize("h", [0.4, 0.2, 0.1])
def test_error_estimate_tracks_true_local_error(h) -> None:
    """|h f@E| = |y5 - y4| matches the true local error of the 4th-order solution."""
    f = lambda t, y: np.array([np.cos(t) * y[0], -2.0 * y[1]])  # noqa: E731
    exact = np.array([np.exp(np.sin(h)), np.exp(-2.0 * h)])
    y5, est = _one_step(f, 0.0, np.array([1.0, 1.0]), h)
    true4 = np.abs((y5 - est) - exact)
    ratio = np.abs(est) / true4
    assert np.all((ratio > 0.5) & (ratio < 2.0)), ratio


def test_steps_scale_with_tolerance() -> None:
    """Error control of a 5(4) pair: nsteps grows like RelTol^(-1/5)."""

    def n(rtol):
        osc = lambda t, y: np.array([y[1], -y[0]])  # noqa: E731
        return ode45(osc, [0, 20], [1.0, 0.0], odeset(RelTol=rtol, AbsTol=1e-14)).stats.nsteps

    ratio = n(1e-10) / n(1e-5)
    assert 7 < ratio < 15, ratio  # 10 expected


@pytest.mark.parametrize("rtol", [1e-3, 1e-6, 1e-9])
def test_exponential_decay(rtol) -> None:
    atol = 1e-12
    tspan = np.linspace(0, 5, 51)
    r = ode45(lambda t, y: -2.0 * y, tspan, [3.0], odeset(RelTol=rtol, AbsTol=atol))
    exact = 3.0 * np.exp(-2.0 * tspan)[:, None]
    assert np.all(np.max(np.abs(r.y - exact), axis=0) <= bound(exact, rtol, atol))


def test_harmonic_oscillator() -> None:
    """Within the class over ~5 periods; global error then grows ~linearly in time."""
    w, rtol, atol = 7.0, 1e-8, 1e-10

    def run(T):
        tspan = np.linspace(0, T, 1001)
        r = ode45(
            lambda t, y: np.array([y[1], -w * w * y[0]]),
            tspan,
            [1.0, 0.0],
            odeset(RelTol=rtol, AbsTol=atol),
        )
        exact = np.stack([np.cos(w * tspan), -w * np.sin(w * tspan)], axis=1)
        return np.max(np.abs(r.y - exact), axis=0), exact

    err5, exact = run(5.0)
    assert np.all(err5 <= bound(exact, rtol, atol))
    err20, _ = run(20.0)
    assert np.all(err20 <= 6 * err5)  # 4x longer: linear growth, not exponential


def test_linear_system_widely_different_scales() -> None:
    M = np.array([[-1.0, 0.0, 0.0], [0.5, -30.0, 0.0], [0.0, 1e-3, -0.2]])
    y0 = np.array([1.0, 1e-4, 1e3])
    rtol, atol = 1e-7, np.array([1e-9, 1e-13, 1e-6])
    tspan = np.linspace(0, 3, 61)
    r = ode45(lambda t, y: M @ y, tspan, y0, odeset(RelTol=rtol, AbsTol=atol))
    exact = np.array([expm(M * t) @ y0 for t in tspan])
    assert np.all(np.max(np.abs(r.y - exact), axis=0) <= bound(exact, rtol, atol))


def test_logistic_nonlinear() -> None:
    rtol, atol = 1e-6, 1e-9
    tspan = np.linspace(0, 10, 101)
    r = ode45(lambda t, y: 1.5 * y * (1 - y), tspan, [0.01], odeset(RelTol=rtol, AbsTol=atol))
    exact = (1 / (1 + 99 * np.exp(-1.5 * tspan)))[:, None]
    assert np.all(np.max(np.abs(r.y - exact), axis=0) <= bound(exact, rtol, atol))


def test_backward_integration() -> None:
    rtol, atol = 1e-8, 1e-12
    r = ode45(lambda t, y: -y, [1.0, -1.0], [1.0], odeset(RelTol=rtol, AbsTol=atol))
    assert r.t[0] == 1.0 and r.t[-1] == -1.0 and np.all(np.diff(r.t) < 0)
    exact = np.exp(1.0 - r.t)[:, None]
    assert np.all(np.max(np.abs(r.y - exact), axis=0) <= bound(exact, rtol, atol))


@pytest.mark.parametrize("refine", [1, 2, 4, 7])
def test_refine_output_and_dense_accuracy(refine) -> None:
    rtol, atol = 1e-6, 1e-10
    r = ode45(
        lambda t, y: np.array([y[1], -y[0]]),
        [0, 10],
        [0.0, 1.0],
        odeset(Refine=refine, RelTol=rtol, AbsTol=atol),
    )
    assert r.t.size == refine * r.stats.nsteps + 1
    np.testing.assert_array_equal(r.t[::refine], r.t_steps)
    exact = np.stack([np.sin(r.t), np.cos(r.t)], axis=1)
    assert np.all(np.max(np.abs(r.y - exact), axis=0) <= bound(exact, rtol, atol))


def test_dense_output_is_continuous_and_4th_order() -> None:
    """Interpolant matches the step endpoints and has O(h^5) local error."""
    errs = []
    for h in (0.2, 0.1):
        r = ode45(
            lambda t, y: np.cos(t) * y,
            [0, h],
            [1.0],
            odeset(InitialStep=h, RelTol=1, AbsTol=1, Refine=5),
        )
        exact = np.exp(np.sin(r.t))
        errs.append(np.max(np.abs(r.y[:, 0] - exact)))
    assert errs[0] / errs[1] > 2**4  # local error of the interpolant at least O(h^5)


def test_many_point_tspan() -> None:
    tspan = [0.0, 1e-4, 0.3, 0.30001, 2.0]
    r = ode45(lambda t, y: -y, tspan, [1.0], odeset(Refine=8, RelTol=1e-6, AbsTol=1e-12))
    np.testing.assert_array_equal(r.t, tspan)
    exact = np.exp(-np.array(tspan))[:, None]
    assert np.all(np.max(np.abs(r.y - exact), axis=0) <= bound(exact, 1e-6, 1e-12))


def test_counts_fsal() -> None:
    r = ode45(lambda t, y: -y, [0, 1], [1.0], odeset(InitialStep=0.01))
    assert r.stats.nfevals == 6 * (r.stats.nsteps + r.stats.nfailed) + 1
    r = ode45(lambda t, y: -y, [0, 1], [1.0])
    assert r.stats.nfevals == 6 * (r.stats.nsteps + r.stats.nfailed) + 2  # + starting-step probe


def test_max_step_and_default() -> None:
    r = ode45(lambda t, y: -0.01 * y, [0, 100], [1.0], odeset(Refine=1))
    assert np.max(np.diff(r.t_steps)) <= 10.0 * (1 + 1e-12)  # default |tf - t0|/10
    r = ode45(lambda t, y: -y, [0, 1], [1.0], odeset(MaxStep=0.03))
    assert np.max(np.diff(r.t_steps)) <= 0.03 * (1 + 1e-12)
    # span just over MaxStep: never one stretched step longer than MaxStep
    r = ode45(lambda t, y: 0 * y, [0, 1.005], [1.0], odeset(MaxStep=1.0, InitialStep=1.0, Refine=1))
    np.testing.assert_allclose(np.diff(r.t_steps), [1.0, 0.005], rtol=1e-12)
    # within MaxStep the last step is stretched (no 1e-3-long sliver)
    r = ode45(lambda t, y: 0 * y, [0, 1.005], [1.0], odeset(MaxStep=2.0, InitialStep=1.0, Refine=1))
    np.testing.assert_array_equal(r.t_steps, [0.0, 1.005])


def test_initial_step_option() -> None:
    r = ode45(lambda t, y: -y, [0, 1], [1.0], odeset(InitialStep=1e-5, Refine=1))
    assert r.t_steps[1] == pytest.approx(1e-5)


def test_stacked_batch_shares_steps() -> None:
    """Q-50: stacking decoupled systems gives one step sequence for all of them."""
    lam = np.array([1.0, 30.0])
    both = ode45(lambda t, y: -lam * y, [0, 1], [1.0, 1.0], odeset(Refine=1))
    one = ode45(lambda t, y: -lam[:1] * y, [0, 1], [1.0], odeset(Refine=1))
    assert both.t_steps.size > one.t_steps.size
    assert both.y_steps[-1, 0] != one.y_steps[-1, 0]
    assert abs(both.y_steps[-1, 0] - one.y_steps[-1, 0]) < 1e-3 * np.exp(-1)


def test_outputfcn_protocol_and_stop() -> None:
    log = []

    def ofcn(t, y, flag):
        log.append((flag, np.array(t), np.shape(y)))
        return flag == "" and sum(e[0] == "" for e in log) == 3

    tspan = [0, 0.5, 1, 2, 3, 4, 5]
    r = ode45(lambda t, y: -y, tspan, [1.0, 2.0], odeset(OutputFcn=ofcn, MaxStep=0.6))
    assert r.stopped and r.t[-1] < 5
    assert log[0][0] == "init"
    np.testing.assert_array_equal(log[0][1], tspan)
    assert log[-1][0] == "done"
    steps = [e for e in log if e[0] == ""]
    assert len(steps) == 3 and all(e[2][0] == 2 for e in steps)


@pytest.mark.parametrize("rtol", [1e-3, 1e-6, 1e-9])
def test_step_size_failure_warns_and_stops(rtol) -> None:
    with pytest.warns(RuntimeWarning, match="error test fails"):
        r = ode45(lambda t, y: y * y, [0, 2], [1.0], odeset(RelTol=rtol))  # blow-up at t = 1
    assert r.stopped and abs(r.t[-1] - 1.0) < 1e-3


def test_nonfinite_rhs() -> None:
    times = []

    def nan_rhs(t, y):
        times.append(t)
        return np.full(1, np.nan)

    with pytest.warns(RuntimeWarning, match="error test fails"):
        r = ode45(nan_rhs, [0, 1], [1.0])
    assert r.stopped and r.t[-1] == 0.0
    assert np.all(np.isfinite(times))  # never evaluated at a NaN time
    with pytest.warns(RuntimeWarning, match="error test fails"):
        r = ode45(lambda t, y: np.array([np.nan if t > 0.5 else -y[0]]), [0, 1], [1.0])
    assert r.stopped and 0.45 < r.t[-1] <= 0.5 + 1e-12
    assert np.all(np.isfinite(r.y))


def test_normcontrol_changes_control() -> None:
    """NormControl 'on': error relative to ||y||, so a tiny component stops driving steps."""
    f = lambda t, y: np.array([-y[0], -50.0 * y[1]])  # noqa: E731
    y0, rtol, atol = [1.0, 1e-6], 1e-6, 1e-14
    tspan = np.linspace(0, 1, 11)
    off = ode45(f, tspan, y0, odeset(RelTol=rtol, AbsTol=atol))
    on = ode45(f, tspan, y0, odeset(RelTol=rtol, AbsTol=atol, NormControl="on"))
    assert on.stats.nsteps < 0.6 * off.stats.nsteps
    exact = np.stack([np.exp(-tspan), 1e-6 * np.exp(-50 * tspan)], axis=1)
    err = np.linalg.norm(on.y - exact, axis=1)
    assert np.all(err <= 10 * rtol * np.linalg.norm(exact, axis=1) + atol)


def test_abstol_per_component() -> None:
    """A tight AbsTol on a small fast component makes it drive the steps; a loose one does not."""
    f = lambda t, y: np.array([-y[0], -20.0 * y[1]])  # noqa: E731
    y0, rtol = [1.0, 1e-8], 1e-6
    tspan = np.linspace(0, 1, 11)
    tight = ode45(f, tspan, y0, odeset(RelTol=rtol, AbsTol=[1e-10, 1e-18]))
    loose = ode45(f, tspan, y0, odeset(RelTol=rtol, AbsTol=[1e-10, 1e-3]))
    assert tight.stats.nsteps > 1.5 * loose.stats.nsteps
    exact2 = 1e-8 * np.exp(-20 * tspan)
    assert np.max(np.abs(tight.y[:, 1] - exact2)) <= 10 * rtol * 1e-8 + 1e-18


def test_rejected_landing_step_is_retried() -> None:
    """A stretched last step that fails is retried with a smaller step, not forced onto tfinal."""
    from scipy.special import erf

    a, c, w = 200.0, 0.98, 0.005
    f = lambda t, y: -a * np.exp(-(((t - c) / w) ** 2)) * y  # noqa: E731
    rtol, atol = 1e-8, 1e-12
    r = ode45(f, [0, 1], [1.0], odeset(InitialStep=1.0, MaxStep=1.0, RelTol=rtol, AbsTol=atol))
    assert r.stats.nfailed > 0 and not r.stopped and r.t[-1] == 1.0
    integral = a * w * np.sqrt(np.pi) / 2 * (erf((1 - c) / w) - erf(-c / w))
    assert abs(r.y[-1, 0] - np.exp(-integral)) <= 10 * rtol + atol


def test_outputfcn_stop_on_last_step_is_not_early() -> None:
    def ofcn(t, y, flag):
        return flag == "" and t.size and t[-1] == 1.0

    r = ode45(lambda t, y: -y, [0, 1], [1.0], odeset(OutputFcn=ofcn))
    assert not r.stopped and r.t[-1] == 1.0


def test_refine_points_equally_spaced_within_steps() -> None:
    r = ode45(lambda t, y: np.array([y[1], -y[0]]), [0, 10], [0.0, 1.0], odeset(Refine=5))
    for k in range(r.stats.nsteps):
        seg = r.t[5 * k : 5 * k + 6]
        np.testing.assert_allclose(np.diff(seg), (seg[-1] - seg[0]) / 5, rtol=1e-9)


def _steps_with_calls(f, tspan, y0, opts):
    """Accepted steps and the RHS calls each one took (via OutputFcn, Refine=1)."""
    calls = [0]
    marks: list[int] = []

    def counted(t, y):
        calls[0] += 1
        return f(t, y)

    def ofcn(t, y, flag):
        if flag == "":
            marks.append(calls[0])
        return False

    r = ode45(counted, tspan, y0, odeset(**opts, Refine=1, OutputFcn=ofcn))
    per_step = np.diff([marks[0] - 6, *marks]) if marks else np.array([])
    return r, np.diff(r.t_steps), per_step


def test_no_growth_right_after_rejection() -> None:
    """The step after a rejected-then-accepted step is never larger (HNW: facmax = 1)."""
    a, c, w = 200.0, 0.5, 0.01
    f = lambda t, y: -a * np.exp(-(((t - c) / w) ** 2)) * y  # noqa: E731
    r, h, calls = _steps_with_calls(f, [0, 1], [1.0], {"InitialStep": 0.3, "RelTol": 1e-6})
    had_reject = np.nonzero(calls[:-1] > 6)[0]
    assert had_reject.size > 0 and r.stats.nfailed > 0
    for i in had_reject:
        assert h[i + 1] <= h[i] * (1 + 1e-12), (i, h[i], h[i + 1])


def test_growth_factor_bounded_by_facmax() -> None:
    """Zero error estimate: steps grow by exactly facmax = 10 until MaxStep."""
    r = ode45(lambda t, y: 0 * y, [0, 1], [1.0], odeset(InitialStep=1e-6, MaxStep=1.0, Refine=1))
    h = np.diff(r.t_steps)
    np.testing.assert_allclose(h[1:5] / h[:4], 10.0, rtol=1e-12)


def test_error_test_uses_max_norm() -> None:
    """Componentwise (documented) test: zero padding changes nothing; an RMS norm would."""
    osc = lambda t, y: np.concatenate([[y[1], -y[0]], np.zeros(y.size - 2)])  # noqa: E731
    opts = odeset(RelTol=1e-6, AbsTol=1e-9, InitialStep=0.01)
    base = ode45(osc, [0, 10], [1.0, 0.0], opts)
    padded = ode45(osc, [0, 10], np.r_[1.0, 0.0, np.zeros(8)], opts)
    np.testing.assert_array_equal(base.t_steps, padded.t_steps)


@pytest.mark.parametrize(("target", "rejects"), [(0.99, False), (1.01, True)])
def test_accept_threshold_is_reltol(target, rejects) -> None:
    """A first attempt whose scaled error is 0.99*RelTol is accepted, 1.01*RelTol rejected."""
    f = lambda t, y: np.array([np.cos(3 * t) * y[0]])  # noqa: E731
    rtol, atol = 1e-6, 1e-12
    # scaled error of one step grows like h^5: solve for h giving the target
    h_ref = 0.1
    e_ref = _scaled_err_for_test(f, np.array([1.0]), h_ref, rtol, atol)
    h = h_ref * (target * rtol / e_ref) ** 0.2
    for _ in range(20):  # refine (h^5 law is only asymptotic)
        h *= (target * rtol / _scaled_err_for_test(f, np.array([1.0]), h, rtol, atol)) ** 0.2
    r = ode45(f, [0, 1], [1.0], odeset(InitialStep=h, RelTol=rtol, AbsTol=atol, MaxStep=1.0))
    first_ok = r.t_steps[1] == pytest.approx(h, rel=1e-12)
    assert first_ok != rejects


def _scaled_err_for_test(f, y, h, rtol, atol):
    y5, est = _one_step(f, 0.0, y, h)
    return np.max(np.abs(est) / np.maximum(np.maximum(np.abs(y), np.abs(y5)), atol / rtol))


def _first_err(h, rtol=1e-6, atol=1e-12):
    f = lambda t, y: np.array([np.cos(3 * t) * y[0]])  # noqa: E731
    return f, _scaled_err_for_test(f, np.array([1.0]), h, rtol, atol) / rtol


def test_step_update_uses_safety_09() -> None:
    """After an accepted first step: h_next = h * 0.9 (RelTol/err)^(1/5) (within facmin..facmax)."""
    h = 0.05
    f, e = _first_err(h)
    r = ode45(
        f, [0, 1], [1.0], odeset(InitialStep=h, RelTol=1e-6, AbsTol=1e-12, Refine=1, MaxStep=1.0)
    )
    assert r.t_steps[1] == pytest.approx(h, rel=1e-14)
    expected = h * min(10.0, max(0.2, 0.9 * e**-0.2))
    assert r.t_steps[2] - r.t_steps[1] == pytest.approx(expected, rel=1e-9)


def test_rejection_shrink_bounded_by_facmin() -> None:
    """A first attempt with err ~ 2500*RelTol is retried with exactly 0.2*h."""
    h = 0.05
    for _ in range(30):  # scale h so the first attempt's scaled error is ~2500
        h *= (2500 / _first_err(h)[1]) ** 0.2
    f, e = _first_err(h)
    assert 0.9 * e**-0.2 < 0.2 and _first_err(0.2 * h)[1] < 1
    r = ode45(
        f, [0, 1], [1.0], odeset(InitialStep=h, RelTol=1e-6, AbsTol=1e-12, Refine=1, MaxStep=1.0)
    )
    assert r.stats.nfailed >= 1
    assert r.t_steps[1] == pytest.approx(0.2 * h, rel=1e-12)


def test_landing_rule_is_1_01() -> None:
    """Remaining distance 1.05*step is not stretched (1.01 rule): two steps, the last 0.05."""
    r = ode45(lambda t, y: 0 * y, [0, 1.05], [1.0], odeset(InitialStep=1.0, MaxStep=2.0, Refine=1))
    np.testing.assert_allclose(r.t_steps, [0.0, 1.0, 1.05], rtol=1e-14)
    r = ode45(lambda t, y: 0 * y, [0, 1.005], [1.0], odeset(InitialStep=1.0, MaxStep=2.0, Refine=1))
    np.testing.assert_array_equal(r.t_steps, [0.0, 1.005])


@pytest.mark.parametrize(("t0", "span"), [(0.0, 1.0), (1e6, 1.0), (0.0, 1e-3)])
def test_minimum_step(t0, span) -> None:
    """The failure warning fires at the documented smallest step 10 eps max(|t|, span)."""
    f = lambda t, y: np.array([np.nan if t > t0 else -y[0]])  # noqa: E731
    with pytest.warns(RuntimeWarning, match="error test fails"):
        r = ode45(f, [t0, t0 + span], [1.0])
    hmin = 10 * np.finfo(float).eps * max(abs(t0), span)
    assert f"({hmin:.3e})" in r.message
    assert r.stopped and r.t[-1] == t0 and np.all(np.isfinite(r.y))


def test_reltol_just_above_floor_accepted() -> None:
    r = ode45(lambda t, y: -y, [0, 0.01], [1.0], odeset(RelTol=5e-15, AbsTol=1e-20))
    assert r.y[-1, 0] == pytest.approx(np.exp(-0.01), rel=1e-12)


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
        (dict(options={"Refine": 0}), "Refine"),
        (dict(options={"RelTol": 1e-17}), "10\\*eps"),
        (dict(options={"RelTol": 2e-15}), "10\\*eps"),
        (dict(tspan=[1e6, 1e6 + 1], options={"MaxStep": 1e-12}), "MaxStep is below"),
        (dict(tspan=[0, np.nan, 1]), "finite"),
        (dict(tspan=[0, np.inf]), "finite"),
        (dict(options={"NormControl": "on", "AbsTol": [1e-6, 1e-7]}), "NormControl"),
        (dict(options={"Events": None}), "unsupported"),
    ],
)
def test_argument_errors(kwargs, msg) -> None:
    args = dict(tspan=[0, 1], options=None) | kwargs
    with pytest.raises(ValueError, match=msg):
        ode45(lambda t, y: -y, args["tspan"], [1.0, 2.0], args["options"])


def test_odeset() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        odeset(reltol=1e-3)
    assert odeset(RelTol=1e-3, BDF="on") == {"RelTol": 1e-3, "BDF": "on"}
