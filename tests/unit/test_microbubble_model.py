"""Unit tests for the microbubble model (no MATLAB): physics and accuracy checks,
vectorisation invariants, timer (Q-122), batching (Q-50/Q-121)."""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from pyproteus import config
from pyproteus import microbubble as mb
from pyproteus.acoustic import mass_source as ms
from pyproteus.numerics import pchip


def pulse(t, f0, amp, n=3.0):
    dur = n / f0
    return amp * np.where(t < dur, np.sin(np.pi * t / dur) ** 2, 0.0) * np.sin(2 * np.pi * f0 * t)


@pytest.fixture
def setup():
    liquid, gas = mb.get_material_properties()
    R0 = np.array([1.5e-6, 2.5e-6, 2.5e-6])
    f0 = 2.5e6
    t = np.arange(0, 4e-6, 1 / 250e6)
    p = np.vstack([pulse(t, f0, a) for a in (60e3, 150e3, 20e3)])
    return liquid, gas, R0, mb.Pulse(t=t, p=p, tq=t, w=2 * np.pi * f0)


def test_surface_tension_regimes() -> None:
    liquid, _ = mb.get_material_properties()
    shell = mb.get_shell_properties(2e-6, "Marmottant", 0.02, liquid)
    sig = mb.calc_surface_tension(2e-6 * np.array([0.5, 1.0, 3.0, np.nan]), shell)
    assert sig[0] == 0 and sig[2] == liquid.sig and np.isnan(sig[3])
    np.testing.assert_allclose(sig[1], 0.02, rtol=1e-14)
    tab = mb.get_shell_properties(2e-6, "SegersTable", 0.02, liquid)
    np.testing.assert_allclose(mb.calc_surface_tension(2e-6, tab), 0.02, rtol=1e-12)


def test_eq_param_vectorised_equals_scalar() -> None:
    liquid, gas = mb.get_material_properties()
    R0 = np.array([1e-6, 2e-6, 3.5e-6])
    for model in ("Marmottant", "SegersTable"):
        sh = mb.get_shell_properties(R0, model, 0.02, liquid)
        eq = mb.get_eq_param(liquid, gas, sh, R0, 2 * np.pi * 2e6)
        for i, r in enumerate(R0):
            e1 = mb.get_eq_param(
                liquid, gas, mb.get_shell_properties(r, model, 0.02, liquid), r, 2 * np.pi * 2e6
            )
            np.testing.assert_allclose(eq.omega_0[i], e1.omega_0, rtol=1e-14)
            np.testing.assert_allclose(eq.delta[i], e1.delta, rtol=1e-14)


def test_minnaert_limit() -> None:
    """No shell, isothermal, large bubble: omega_0 -> Minnaert sqrt(3 kappa P0/rho)/R0."""
    liquid, gas = mb.get_material_properties()
    liquid.ThermalModel = "Isothermal"
    R0 = 1e-3
    shell = mb.Shell(model="Marmottant", sig_0=0.0, chi=0.0, Rb=R0, sig_l=0.0, Ks=0.0)
    eq = mb.get_eq_param(liquid, gas, shell, R0, 1.0)
    np.testing.assert_allclose(eq.omega_0, np.sqrt(3 * liquid.P0 / liquid.rho) / R0, rtol=1e-14)


def test_pchip_batch_matches_pchip() -> None:
    rng = np.random.default_rng(0)
    t = np.cumsum(rng.uniform(0.5, 1.5, 40))
    p = rng.normal(size=(3, 40))
    P = mb.PchipBatch(t, p)
    for tq in [t[0] - 1, t[0], 3.3, t[17], t[-1], t[-1] + 2]:
        np.testing.assert_allclose(
            P(tq), [pchip(t, p[i], tq) for i in range(3)], rtol=1e-12, atol=1e-13
        )


def test_simple_rp_batch_equals_single(setup) -> None:
    liquid, gas, R0, pl = setup
    sh = mb.get_shell_properties(R0, "Marmottant", 0.02, liquid)
    eq = mb.get_eq_param(liquid, gas, sh, R0, pl.w)
    P = mb.PchipBatch(pl.t, pl.p)
    v = np.array([0.1, -0.3, -0.05, 0.2, 0.3, 1.0])
    full = mb.simple_rp(2.0, v, liquid, sh, eq, R0, P, 1e-7)
    for i in range(3):
        one = mb.simple_rp(
            2.0,
            v[2 * i : 2 * i + 2],
            liquid,
            sh.take([i]),
            eq_take(eq, i),
            R0[i : i + 1],
            mb.PchipBatch(pl.t, pl.p[i : i + 1]),
            1e-7,
        )
        np.testing.assert_allclose(full[2 * i : 2 * i + 2], one, rtol=1e-15)


def eq_take(eq, i):
    return mb.EqParam(**{k: np.asarray(v)[i : i + 1] for k, v in vars(eq).items()})


@pytest.mark.parametrize("model", ["Marmottant", "SegersTable"])
def test_response_accuracy_vs_reference(setup, model) -> None:
    """Accuracy against DOP853 at rtol 1e-13 (Phase B physics check): within 1e-3 of the
    excursion with PROTEUS' tolerances (MATLAB measured <= 1.2e-4, docs/PARITY.md)."""
    liquid, gas, R0, pl = setup
    sh = mb.get_shell_properties(R0, model, 0.02, liquid)
    resp, eq = mb.calc_bubble_response(liquid, gas, sh, R0, pl)
    T = float(np.median(np.sqrt(liquid.rho * R0**2 / liquid.P0)))
    P = mb.PchipBatch(pl.t, pl.p)
    ref = solve_ivp(
        lambda s, v: mb.simple_rp(s, v, liquid, sh, eq, R0, P, T),
        (0, pl.t[-1] / T),
        np.zeros(6),
        method="DOP853",
        rtol=1e-13,
        atol=1e-14,
        t_eval=pl.t / T,
        max_step=2 * pl.t[1] / T,
    )
    x_ref = ref.y.T.reshape(-1, 3, 2)[:, :, 0].T
    x = resp.R / R0[:, None] - 1
    err = np.max(np.abs(x - x_ref), axis=1) / np.max(np.abs(x_ref), axis=1)
    assert np.all(err < 1e-3), err


def test_linear_regime_scales_with_pressure() -> None:
    liquid, gas = mb.get_material_properties()
    liquid.ThermalModel = "Adiabatic"
    R0 = np.array([2e-6, 2e-6])
    # elastic regime throughout (no buckling/rupture threshold reached)
    sh = mb.Shell(
        model="Marmottant", sig_0=1e-3, chi=0.5, Rb=R0 / np.sqrt(1 + 1e-3 / 0.5), sig_l=1.0, Ks=1e-9
    )
    t = np.arange(0, 3e-6, 1 / 250e6)
    p = np.vstack([pulse(t, 2e6, 100.0), pulse(t, 2e6, 200.0)])
    r, _ = mb.calc_bubble_response(liquid, gas, sh, R0, mb.Pulse(t=t, p=p, tq=t, w=4e6 * np.pi))
    x = r.R / R0[:, None] - 1
    np.testing.assert_allclose(np.max(np.abs(x[1])) / np.max(np.abs(x[0])), 2.0, rtol=2e-2)


def test_proteus_options_pinned() -> None:
    """The options are what holds parity (calcBubbleResponse.m); a 10x typo would fit the class."""
    from pyproteus.microbubble import dynamics

    assert (dynamics.REL_TOL, dynamics.ABS_TOL, dynamics.INITIAL_STEP) == (1e-6, 1e-6, 1e-12)


def test_time_scale(setup) -> None:
    liquid, gas, R0, pl = setup
    sh = mb.get_shell_properties(R0, "Marmottant", 0.02, liquid)
    T = np.sqrt(liquid.rho * R0**2 / liquid.P0)
    r, _ = mb.calc_bubble_response(liquid, gas, sh, R0, pl)
    np.testing.assert_array_equal(r.T, T)  # independent: own time scale
    r, _ = mb.calc_bubble_response(liquid, gas, sh, R0, pl, step_control="shared")
    assert r.T[0] == np.median(T)  # PROTEUS: median over the batch


def test_independent_batch_equals_single_bitwise(setup) -> None:
    """D-22: a bubble's response does not depend on its batch (fixes Q-50/Q-121)."""
    liquid, gas, R0, pl = setup
    sh = mb.get_shell_properties(R0, "SegersTable", 0.02, liquid)
    r, _ = mb.calc_bubble_response(liquid, gas, sh, R0, pl)
    for order in ([2, 0, 1], [1]):
        o = np.array(order)
        p2 = mb.Pulse(t=pl.t, p=pl.p[o], tq=pl.tq, w=pl.w)
        r2, _ = mb.calc_bubble_response(liquid, gas, sh.take(o), R0[o], p2)
        np.testing.assert_array_equal(r2.R, r.R[o])
        np.testing.assert_array_equal(r2.Rdot, r.Rdot[o])


def test_shared_mode_depends_on_batch(setup) -> None:
    liquid, gas, R0, pl = setup
    sh = mb.get_shell_properties(R0, "Marmottant", 0.02, liquid)
    r, _ = mb.calc_bubble_response(liquid, gas, sh, R0, pl, step_control="shared")
    p1 = mb.Pulse(t=pl.t, p=pl.p[:1], tq=pl.tq, w=pl.w)
    r1, _ = mb.calc_bubble_response(liquid, gas, sh.take([0]), R0[:1], p1, step_control="shared")
    assert not np.array_equal(r1.R[0], r.R[0])
    np.testing.assert_allclose(r1.R[0], r.R[0], rtol=1e-4)


def test_blow_up_reported_others_finish(setup) -> None:
    liquid, gas, R0, pl = setup
    sh = mb.get_shell_properties(R0, "Marmottant", 0.02, liquid)
    with pytest.raises(mb.BubbleIntegrationError, match=r"bubble 1:.*blow-up") as exc:
        mb.calc_bubble_response(liquid, gas, sh, R0, pl, max_radius_ratio=1.15)
    e = exc.value
    assert [f.index for f in e.failures] == [1] and 0 < e.failures[0].t < pl.tq[-1]
    assert "max_radius_ratio=1.15" in str(e)
    R = e.response.R
    assert np.isnan(R[1, -1]) and np.all(np.isfinite(R[[0, 2]]))
    ok, _ = mb.calc_bubble_response(liquid, gas, sh, R0, pl)
    np.testing.assert_array_equal(R[[0, 2]], ok.R[[0, 2]])  # unaffected by the failure


def test_non_finite_pressure_reported(setup) -> None:
    liquid, gas, R0, pl = setup
    p = pl.p.copy()
    p[2, 300:] = np.nan
    sh = mb.get_shell_properties(R0, "Marmottant", 0.02, liquid)
    with pytest.raises(mb.BubbleIntegrationError, match=r"bubble 2:.*non-finite") as exc:
        mb.calc_bubble_response(liquid, gas, sh, R0, mb.Pulse(t=pl.t, p=p, tq=pl.tq, w=pl.w))
    assert [f.index for f in exc.value.failures] == [2]
    assert np.all(np.isfinite(exc.value.response.R[:2]))


def test_step_budget_and_wall_clock(setup) -> None:
    liquid, gas, R0, pl = setup
    sh = mb.get_shell_properties(R0, "Marmottant", 0.02, liquid)
    with pytest.raises(mb.BubbleIntegrationError, match="more than 20 steps") as exc:
        mb.calc_bubble_response(liquid, gas, sh, R0, pl, max_steps=20)
    assert len(exc.value.failures) == 3
    with pytest.raises(mb.BubbleIntegrationError, match="wall-clock"):
        mb.calc_bubble_response(liquid, gas, sh, R0, pl, wall_clock_limit=0.0)
    with pytest.raises(mb.BubbleIntegrationError, match="shared steps: whole batch"):
        mb.calc_bubble_response(liquid, gas, sh, R0, pl, step_control="shared", max_steps=20)


def test_mass_source_failure_lists_global_indices() -> None:
    med, mbs = config.reset_medium(), config.reset_microbubble()
    tt = np.arange(0, 2e-6, 1 / 40e6)
    sp = np.vstack([pulse(tt, 2.5e6, a) for a in (5e4, 1e5, 8e4)])
    sp[1, 20:] = np.nan
    kg = SimpleNamespace(dt=1 / 40e6, k_max=np.pi / 1e-4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(mb.BubbleIntegrationError) as exc:
            ms.compute_bubble_mass_source(
                sp,
                [3e-6, 2e-6, 1e-6],
                kg,
                med,
                mbs,
                SimpleNamespace(CenterFrequency=2.5e6),
                batch_size=2,
            )
    assert [f.index for f in exc.value.failures] == [1]
    m = exc.value.mass_source
    assert np.all(np.isnan(m[1])) and np.all(np.isfinite(m[[0, 2]]))


def test_mass_source_independent_of_batch_size() -> None:
    med, mbs = config.reset_medium(), config.reset_microbubble()
    tt = np.arange(0, 2e-6, 1 / 40e6)
    sp = np.vstack([pulse(tt, 2.5e6, a) for a in (5e4, 1e5, 8e4, 2e4)])
    kg = SimpleNamespace(dt=1 / 40e6, k_max=np.pi / 1e-4)
    r = [2e-6, 1.5e-6, 2e-6, 3e-6]
    tx = SimpleNamespace(CenterFrequency=2.5e6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m1 = ms.compute_bubble_mass_source(sp, r, kg, med, mbs, tx, batch_size=1)
        m4 = ms.compute_bubble_mass_source(sp, r, kg, med, mbs, tx, batch_size=4)
    np.testing.assert_array_equal(m1, m4)


def test_batch_indices_stable_ties() -> None:
    # > 16 elements: NumPy's default sort is not stable there (insertion sort below)
    r = np.random.default_rng(1).choice([1.0, 2.0, 3.0], 40)
    order = np.concatenate([ms.get_batch_indices(k, 40, 7, r) for k in range(6)])
    expected = sorted(range(40), key=lambda i: r[i])  # Python's sort is stable
    np.testing.assert_array_equal(order, expected)


def test_mass_source_float32_and_shape() -> None:
    med, mbs = config.reset_medium(), config.reset_microbubble()
    sp = np.vstack([pulse(np.arange(0, 2e-6, 1 / 40e6), 2.5e6, a) for a in (5e4, 1e5)])
    kg = SimpleNamespace(dt=1 / 40e6, k_max=np.pi / 1e-4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m32 = ms.compute_bubble_mass_source(
            sp.astype(np.float32),
            [2e-6, 3e-6],
            kg,
            med,
            mbs,
            SimpleNamespace(CenterFrequency=2.5e6),
        )
        m64 = ms.compute_bubble_mass_source(
            sp, [2e-6, 3e-6], kg, med, mbs, SimpleNamespace(CenterFrequency=2.5e6)
        )
    assert m32.dtype == np.float32 and m64.dtype == np.float64 and m64.shape == sp.shape
    np.testing.assert_allclose(m32, m64, atol=1e-3 * np.abs(m64).max())
