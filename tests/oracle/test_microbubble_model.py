"""Oracle tests: microbubble model (``microbubble-simulator/``), M2.

Classes (D-13):

- parameters (material, surface tension, thermal damping, Segers fit roots,
  shell properties, compression, equation parameters) and the RHS
  ``simple_RP``: float, 1e-9 rel (small atol near zero);
- ``calcBubbleResponse``: ODE class on the solver's own (nondimensional)
  states, per bubble ``max|dx| <= 10 RelTol max|x| + AbsTol`` for
  ``x = R/R0 - 1`` and ``xdot = Rdot T/R0`` (RelTol = AbsTol = 1e-6, PROTEUS'
  options). MATLAB's step sequence is not compared (D-12).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pyproteus import data
from pyproteus import microbubble as mb
from tests.oracle.helpers import ORACLE_SETTINGS, from_matlab, put, row, vec

RTOL = 1e-9
#: Bubble-response class (D-22, amended 2026-11-25): per bubble
#: max|dy| <= CLASS_REL max|y| + CLASS_ABS for y = x and xdot. The D-13 bound
#: 10 RelTol max|y| + AbsTol is missed by MATLAB against itself (batch vs alone,
#: 17-34x; B-121). The absolute term is 3e-5, not 1e-5, because quiet bubbles that
#: just cross the Marmottant buckling kink are integrated less accurately by both
#: solvers (B-123: up to 0.82 (MATLAB) / 1.16 (pyproteus) of the 1e-5 class against
#: a DOP853 reference); kink events are a 1.x item.
CLASS_REL, CLASS_ABS = 1e-3, 3e-5
#: B-120: PROTEUS' 'Segers' polynomial (degree 10, monomial basis, |c| up to 9e10) is
#: ill-conditioned: evaluation noise ~eps*sum|c| = 8e-5 N/m and root positions to
#: ~1e-3 rel differ between any two correct implementations (NumPy vs MATLAB).
SEGERS_RTOL = 1e-3
SEGERS_SIG_ATOL = 2e-4
ODE_REL, ODE_ABS = 1e-6, 1e-6


def assert_float(a: Any, e: Any, atol: float = 0.0, rtol: float = RTOL, err_msg: str = "") -> None:
    a, e = np.asarray(a, dtype=float), np.asarray(e, dtype=float)
    assert a.size == e.size, f"size {a.shape} != {e.shape} {err_msg}"
    np.testing.assert_allclose(
        a.reshape(-1), e.reshape(-1), rtol=rtol, atol=atol, equal_nan=True, err_msg=err_msg
    )


def g17(v: float) -> str:
    return f"{float(v):.17g}"


def put_materials(eng: Any, liquid: mb.Liquid, gas: mb.GasProps) -> None:
    """Assign MATLAB ``liquid`` and ``gas`` structs with the same values."""
    cmds = ["clear liquid gas;"]
    for k in ("k", "rho", "cp", "nu", "c", "T0", "P0", "sig"):
        v = getattr(liquid, k)
        if v is not None:
            cmds.append(f"liquid.{k} = {g17(v)};")
    cmds.append(f"liquid.ThermalModel = '{liquid.ThermalModel}';")
    for k in ("k", "rho", "Mg", "gam", "cp"):
        cmds.append(f"gas.{k} = {g17(getattr(gas, k))};")
    eng.eval(" ".join(cmds), nargout=0)


def put_shells(eng: Any, R0: np.ndarray, model: str, sig_0: float) -> None:
    """MATLAB struct arrays ``bubble``/``shell`` via ``getShellProperties`` (per bubble)."""
    put(eng, pyp_R0=row(R0))
    eng.eval(
        "clear shell bubble; warning('off','all');"
        "for i = numel(pyp_R0):-1:1, bubble(i).R0 = pyp_R0(i);"
        f" s = struct('model','{model}','sig_0',{g17(sig_0)});"
        " shell(i) = getShellProperties(bubble(i), s, liquid); end; warning('on','all');",
        nargout=0,
    )


def materials(thermal: str = "Prosperetti") -> tuple[mb.Liquid, mb.GasProps]:
    liquid, gas = mb.get_material_properties()
    liquid.ThermalModel = thermal
    return liquid, gas


# ---------------------------------------------------------------- parameters


def test_get_material_properties(matlab_engine) -> None:
    eng = matlab_engine
    eng.eval("[pyp_l, pyp_g] = getMaterialProperties();", nargout=0)
    liquid, gas = mb.get_material_properties()
    for k in ("k", "rho", "cp", "nu", "c", "sig", "T0", "P0", "a", "b", "BA", "beta"):
        assert getattr(liquid, k) == eng.eval(f"pyp_l.{k}"), k
    for k in ("k", "rho", "Mg", "gam", "cp"):
        assert getattr(gas, k) == eng.eval(f"pyp_g.{k}"), k


@pytest.mark.parametrize("model", ["Marmottant", "Segers", "SegersTable"])
@pytest.mark.parametrize("sig_0", [0.0, 0.01, 0.04, 0.072])
def test_shell_properties_and_surface_tension(matlab_engine, model: str, sig_0: float) -> None:
    """``getShellProperties`` + ``calc_surface_tension`` over the buckled, elastic and
    ruptured regimes (radii 0.5-2 R0), NaN, and the exact table ends."""
    eng = matlab_engine
    if model != "Marmottant" and sig_0 in (0.0, 0.072):
        pytest.skip("no Segers normalised area for sig_0 at the ends of the fit")
    liquid, gas = materials()
    put_materials(eng, liquid, gas)
    R0 = np.array([0.8e-6, 1.0e-6, 2.0e-6, 3.3e-6, 4.0e-6, 4.5e-6])  # incl. warning range
    put_shells(eng, R0, model, sig_0)
    with pytest.warns(UserWarning, match="uncertain"):
        shell = mb.get_shell_properties(R0, model, sig_0, liquid)
    fields = {
        "Marmottant": ("sig_l", "Ks", "chi", "Rb"),
        "Segers": ("sig_l", "Ks", "A_m1", "A_m2", "A_N"),
        "SegersTable": ("sig_l", "Ks", "A_m1", "A_m2", "A_N"),
    }[model]
    for f in fields:
        ref = vec(eng.eval(f"[shell.{f}]"))
        # B-120: A_m1 is a root in an ill-conditioned cluster of the Segers polynomial
        rtol = SEGERS_RTOL if model == "Segers" and f.startswith("A_") else RTOL
        assert_float(np.broadcast_to(getattr(shell, f), R0.shape), ref, rtol=rtol, err_msg=f)
    if model == "Segers":
        assert_float(shell.coeff, vec(eng.eval("shell(1).coeff")), rtol=0)
    # surface tension of each bubble on 0.5..2 R0 (+ NaN)
    scale = np.concatenate([np.linspace(0.5, 2, 301), [np.nan]])
    for i in range(R0.size):
        R = R0[i] * scale
        put(eng, pyp_R=row(R))
        ref = vec(eng.eval(f"calc_surface_tension(pyp_R, shell({i + 1}))"))
        sh_i = shell.take(i) if shell.size > 1 else shell
        got = mb.calc_surface_tension(R, sh_i)
        if model == "Segers":
            # B-120: between the two A_m1 values one side gives 0, the other the fit
            # value there (|sig| < 1e-4 N/m); elsewhere the float class holds.
            # ...and next to the domain ends one side may already be clamped.
            A_m = 4 * np.pi * R**2 / float(np.asarray(sh_i.A_N).reshape(-1)[0])
            band = np.zeros(R.shape, bool)
            for f in ("A_m1", "A_m2"):
                lo, hi = sorted([float(getattr(sh_i, f)), float(eng.eval(f"shell(1).{f}"))])
                band |= (A_m >= lo * (1 - 1e-3)) & (A_m <= hi * (1 + 1e-3))
            assert np.all(np.abs(got[band] - ref[band]) < 2e-3)
            assert_float(got[~band], ref[~band], atol=SEGERS_SIG_ATOL, rtol=0, err_msg="Segers")
            continue
        assert_float(got, ref, atol=1e-15, err_msg=f"{model} bubble {i}")
        assert np.any(ref == 0) and np.any(ref == np.max(ref[~np.isnan(ref)]))


def test_surface_tension_table_ends(matlab_engine) -> None:
    """SegersTable: a normalised area exactly at A_m1/A_m2 gives 0 (Q-120)."""
    eng = matlab_engine
    liquid, gas = materials()
    put_materials(eng, liquid, gas)
    R0 = np.array([2e-6])
    put_shells(eng, R0, "SegersTable", 0.02)
    shell = mb.get_shell_properties(R0, "SegersTable", 0.02, liquid)
    A_N = float(shell.A_N.reshape(-1)[0])
    A = np.array([float(shell.A_m1), float(shell.A_m2)])
    R = np.sqrt(A * A_N / (4 * np.pi))
    # radii whose normalised area is exactly the table ends in IEEE arithmetic
    A_back = 4 * np.pi * R**2 / A_N
    put(eng, pyp_R=row(R))
    ref = vec(eng.eval("calc_surface_tension(pyp_R, shell(1))"))
    assert_float(mb.calc_surface_tension(R, shell), ref, atol=1e-15)
    exact = A_back == A
    assert np.all(mb.calc_surface_tension(R, shell)[exact] == 0)


@given(
    R0=st.floats(0.3e-6, 10e-6),
    sig_0=st.floats(0.0, 0.072),
    f=st.floats(0.1e6, 20e6),
)
@ORACLE_SETTINGS
def test_calc_thermal_damp(matlab_engine, R0: float, sig_0: float, f: float) -> None:
    eng = matlab_engine
    liquid, gas = materials()
    put_materials(eng, liquid, gas)
    w = 2 * np.pi * f
    nuth, kappa = eng.eval(
        f"calc_thermal_damp(liquid, gas, struct('R0',{g17(R0)}), struct('sig_0',{g17(sig_0)}),"
        f" {g17(w)})",
        nargout=2,
    )
    got = mb.calc_thermal_damp(liquid, gas, R0, sig_0, w)
    assert_float(got[0], nuth)
    assert_float(got[1], kappa)


def test_segers_fit_roots(matlab_engine) -> None:
    eng = matlab_engine
    coeff = np.asarray(data.load("fit_SigmaR_04-08-2017.mat")["fit"]["coeff"], float).ravel()
    put(eng, pyp_c=row(coeff))
    a1, a2 = eng.eval("find_domain_boundaries(pyp_c)", nargout=2)
    A_m1, A_m2 = mb.find_domain_boundaries(coeff)
    assert_float(A_m1, a1, rtol=SEGERS_RTOL)  # B-120
    assert_float(A_m2, a2, rtol=SEGERS_RTOL)
    for sig_0 in (1e-4, 0.01, 0.03, 0.06, 0.07):
        ref = eng.eval(
            f"find_A_m0(struct('coeff',pyp_c,'A_m1',{g17(a1)},'A_m2',{g17(a2)},"
            f"'sig_0',{g17(sig_0)}))"
        )
        # same bounds as MATLAB, so only the polynomial's conditioning differs
        assert_float(mb.find_A_m0(coeff, a1, a2, sig_0), ref, rtol=SEGERS_RTOL, err_msg=str(sig_0))
    # above the fit's maximum MATLAB returns [] (fails later); the port raises at once
    assert (
        from_matlab(
            eng.eval(
                f"find_A_m0(struct('coeff',pyp_c,'A_m1',{g17(a1)},'A_m2',{g17(a2)},'sig_0',0.08))"
            )
        ).size
        == 0
    )
    with pytest.raises(ValueError, match="sig_0"):
        mb.find_A_m0(coeff, a1, a2, 0.08)


@pytest.mark.parametrize("thermal", ["Prosperetti", "Adiabatic", "Isothermal"])
@pytest.mark.parametrize("model", ["Marmottant", "Segers", "SegersTable"])
def test_get_eq_param(matlab_engine, thermal: str, model: str) -> None:
    eng = matlab_engine
    liquid, gas = materials(thermal)
    put_materials(eng, liquid, gas)
    R0 = np.array([1.1e-6, 2e-6, 2e-6, 3.9e-6])
    sig_0 = 0.02
    put_shells(eng, R0, model, sig_0)
    w = 2 * np.pi * 2.5e6
    eng.eval(
        f"clear pyp_eq; for i = numel(bubble):-1:1, pyp_eq(i) = getEqParam(liquid, gas,"
        f" shell(i), bubble(i), struct('w',{g17(w)})); end",
        nargout=0,
    )
    shell = mb.get_shell_properties(R0, model, sig_0, liquid)
    eq = mb.get_eq_param(liquid, gas, shell, R0, w)
    for f in (
        "nu_th",
        "kappa",
        "nu_vis",
        "nu",
        "nu_rad",
        "chi",
        "omega_0",
        "delta_vis",
        "delta_shell",
        "delta_th",
        "delta_rad",
        "delta",
    ):
        # Segers: chi is a finite difference of the noisy polynomial (B-120) and
        # feeds omega_0 and the deltas; the other models stay at the float class.
        rtol = (
            1e-2
            if model == "Segers" and f not in ("nu_th", "kappa", "nu_vis", "nu", "nu_rad")
            else RTOL
        )
        assert_float(
            getattr(eq, f), vec(eng.eval(f"[pyp_eq.{f}]")), atol=1e-300, rtol=rtol, err_msg=f
        )


@pytest.mark.parametrize(
    ("model", "R0", "P1"),
    [
        ("Marmottant", 2e-6, 1e3),
        ("Marmottant", 2e-6, -5e3),
        ("Marmottant", 1.2e-6, 2e4),
        ("SegersTable", 2e-6, 1e3),
        ("SegersTable", 3e-6, 1e4),
        ("Segers", 2e-6, 5e3),
    ],
)
def test_compress_bubble(matlab_engine, model: str, R0: float, P1: float) -> None:
    eng = matlab_engine
    liquid, gas = materials()
    put_materials(eng, liquid, gas)
    put_shells(eng, np.array([R0]), model, 0.02)
    eng.eval(
        f"warning('off','all'); [pyp_l, pyp_g, pyp_b, pyp_s] = compress_bubble(liquid, gas,"
        f" bubble(1), shell(1), {g17(P1)}); warning('on','all');",
        nargout=0,
    )
    shell = mb.get_shell_properties(R0, model, 0.02, liquid)
    lq, gs, R_new, sh = mb.compress_bubble(liquid, gas, R0, shell, P1)
    assert_float(lq.P0, eng.eval("pyp_l.P0"))
    assert_float(gs.rho, eng.eval("pyp_g.rho"))
    seg = model == "Segers"  # B-120
    assert_float(R_new, eng.eval("pyp_b.R0"), rtol=SEGERS_RTOL if seg else RTOL)
    assert_float(sh.sig_0, eng.eval("pyp_s.sig_0"), atol=SEGERS_SIG_ATOL if seg else 1e-15)


def test_compress_bubble_out_of_range(matlab_engine) -> None:
    liquid, gas = materials()
    shell = mb.get_shell_properties(2e-6, "Marmottant", 0.02, liquid)
    with pytest.raises(ValueError, match="Overpressure"):
        mb.compress_bubble(liquid, gas, 2e-6, shell, 1e8)


# ---------------------------------------------------------------- dynamics


def pulse_signal(t: np.ndarray, f0: float, amp: float, n_cycles: float = 3.0) -> np.ndarray:
    dur = n_cycles / f0
    env = np.where(t < dur, np.sin(np.pi * t / dur) ** 2, 0.0)
    return amp * env * np.sin(2 * np.pi * f0 * t)


def put_pulse(eng: Any, t: np.ndarray, p: np.ndarray, tq: np.ndarray, w: float) -> None:
    put(eng, pyp_t=row(t), pyp_p=np.atleast_2d(p), pyp_tq=row(tq))
    eng.eval(
        f"pulse = struct('t', pyp_t, 'p', pyp_p, 'tq', pyp_tq, 'w', {g17(w)},"
        " 'dispProgress', false);",
        nargout=0,
    )


@pytest.mark.parametrize("model", ["Marmottant", "SegersTable"])
def test_simple_rp(matlab_engine, model: str) -> None:
    """RHS at random states and times (incl. off-node, outside the pulse), float class."""
    eng = matlab_engine
    liquid, gas = materials()
    put_materials(eng, liquid, gas)
    R0 = np.array([1.3e-6, 2e-6, 2e-6, 3.1e-6, 5e-6])
    put_shells(eng, R0, model, 0.015)
    f0 = 2.5e6
    w = 2 * np.pi * f0
    t = np.arange(0, 3e-6, 1 / 100e6)
    p = np.vstack([pulse_signal(t, f0, a) for a in (50e3, 150e3, -80e3, 300e3, 1e3)])
    put_pulse(eng, t, p, t, w)
    eng.eval(
        "clear pyp_eq P_acc; for i = numel(bubble):-1:1, pyp_eq(i) = getEqParam(liquid, gas,"
        " shell(i), bubble(i), pulse); P_acc(i).p = griddedInterpolant(pulse.t, pulse.p(i,:),"
        "'pchip'); end; pyp_T = median(sqrt(liquid.rho.*[bubble.R0].^2/liquid.P0));",
        nargout=0,
    )
    T = float(eng.eval("pyp_T"))
    shell = mb.get_shell_properties(R0, model, 0.015, liquid)
    eq = mb.get_eq_param(liquid, gas, shell, R0, w)
    P_acc = mb.PchipBatch(t, p)
    rng = np.random.default_rng(3)
    for tau in [0.0, 1e-3, 3.7, t[57] / T, t[-1] / T, 1.01 * t[-1] / T, 2 * t[-1] / T]:
        v = np.zeros(2 * R0.size)
        v[0::2] = rng.uniform(-0.5, 1.0, R0.size)
        v[1::2] = rng.uniform(-2, 2, R0.size)
        put(eng, pyp_v=v.reshape(-1, 1))
        ref = vec(
            eng.eval(f"simple_RP({g17(tau)}, pyp_v, liquid, shell, pyp_eq, bubble, P_acc, pyp_T)")
        )
        got = mb.simple_rp(tau, v, liquid, shell, eq, R0, P_acc, T)
        assert_float(got, ref, atol=1e-12 * np.max(np.abs(ref)), err_msg=f"tau={tau}")


def run_matlab_response(eng: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eng.eval(
        "[pyp_resp, ~] = calcBubbleResponse(liquid, gas, shell, bubble, pulse);"
        "pyp_R = [pyp_resp.R]; pyp_Rd = [pyp_resp.Rdot]; pyp_tt = pyp_resp(1).t;"
        "stop(timerfindall); delete(timerfindall);",
        nargout=0,
    )
    return (
        from_matlab(eng.eval("pyp_R")).T,
        from_matlab(eng.eval("pyp_Rd")).T,
        vec(eng.eval("pyp_tt")),
    )


def ode_errors(R0, T, R, Rd, R_ref, Rd_ref, rel=CLASS_REL, abs_=CLASS_ABS) -> np.ndarray:
    """Per bubble ``max|dy| / (rel max|y| + abs_)`` for y = x and xdot (<= 1 passes)."""
    out = []
    for y, y_ref in (
        (R / R0[:, None] - 1, R_ref / R0[:, None] - 1),
        (Rd * T / R0[:, None], Rd_ref * T / R0[:, None]),
    ):
        bound = rel * np.max(np.abs(y_ref), axis=1) + abs_
        out.append(np.max(np.abs(y - y_ref), axis=1) / bound)
    return np.array(out)


CASES = {
    # name: (model, sig_0, radii, amplitudes, f0)
    "single_elastic": ("Marmottant", 0.02, [2e-6], [50e3], 2.5e6),
    "single_buckled": ("Marmottant", 0.0, [2e-6], [30e3], 2.5e6),
    "single_ruptured": ("Marmottant", 0.072, [2e-6], [200e3], 2.5e6),
    "tiny": ("Marmottant", 0.02, [0.5e-6], [100e3], 5e6),
    "large": ("Marmottant", 0.02, [6e-6], [100e3], 1e6),
    "batch_mixed": (
        "Marmottant",
        0.01,
        [1.2e-6, 2.1e-6, 3.5e-6, 4.4e-6],
        [80e3, 20e3, 150e3, 300e3],
        2e6,
    ),
    "batch_equal_radii": ("Marmottant", 0.01, [2e-6, 2e-6, 2e-6], [10e3, 100e3, 250e3], 2e6),
    "segers_table": ("SegersTable", 0.02, [1.5e-6, 2.5e-6, 3e-6], [40e3, 120e3, 200e3], 2.5e6),
    "segers_poly": ("Segers", 0.02, [2e-6, 2.6e-6], [60e3, 150e3], 2.5e6),
}


@pytest.mark.parametrize("step_control", ["independent", "shared"])
@pytest.mark.parametrize("name", list(CASES))
def test_calc_bubble_response(matlab_engine, name: str, step_control: str) -> None:
    """Both step controls (D-22) against MATLAB's stacked solve. A single bubble gets the
    same algorithm either way; in batches "independent" differs from MATLAB by MATLAB's
    own batch dependence (Q-50), which is within the class."""
    eng = matlab_engine
    model, sig_0, radii, amps, f0 = CASES[name]
    R0 = np.array(radii)
    liquid, gas = materials()
    put_materials(eng, liquid, gas)
    put_shells(eng, R0, model, sig_0)
    fs = 250e6
    t = np.arange(0, 5e-6, 1 / fs)
    p = np.vstack([pulse_signal(t, f0, a) for a in amps])
    w = 2 * np.pi * f0
    put_pulse(eng, t, p, t, w)
    R_ref, Rd_ref, t_ref = run_matlab_response(eng)
    T = float(eng.eval("median(sqrt(liquid.rho.*[bubble.R0].^2/liquid.P0))"))
    R_bat, Rd_bat = R_ref, Rd_ref
    if step_control == "independent" and R0.size > 1:
        # D-22: independent steps = MATLAB integrating each bubble alone
        rows = []
        for i in range(R0.size):
            put_shells(eng, R0[i : i + 1], model, sig_0)
            put_pulse(eng, t, p[i : i + 1], t, w)
            rows.append(run_matlab_response(eng)[:2])
        R_ref = np.vstack([r[0] for r in rows])
        Rd_ref = np.vstack([r[1] for r in rows])
    with __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore")
        shell = mb.get_shell_properties(R0, model, sig_0, liquid)
    pulse = mb.Pulse(t=t, p=p, tq=t, w=w)
    resp, _ = mb.calc_bubble_response(liquid, gas, shell, R0, pulse, step_control=step_control)
    assert not resp.failures
    assert_float(resp.t, t_ref, atol=1e-20)
    if step_control == "shared":
        assert_float(resp.T, T, rtol=1e-15)  # PROTEUS' batch time scale
    else:
        assert_float(resp.T, np.sqrt(liquid.rho * R0**2 / liquid.P0), rtol=0)
    err = ode_errors(R0, T, resp.R, resp.Rdot, R_ref, Rd_ref)
    d13 = ode_errors(R0, T, resp.R, resp.Rdot, R_ref, Rd_ref, 10 * ODE_REL, ODE_ABS)
    vs_batch = ode_errors(R0, T, resp.R, resp.Rdot, R_bat, Rd_bat).max()
    print(f"{name} [{step_control}]: vs MATLAB's batched solve {vs_batch:.3f} of the class")
    print(
        f"{name}: fraction of the class {err.max():.3f} (of the D-13 bound {d13.max():.1f}); "
        f"max R/R0 {np.max(R_ref / R0[:, None], axis=1)}"
    )
    assert np.all(err <= 1), err


def test_matlab_batch_dependence(matlab_engine) -> None:
    """Measured, not a parity check: MATLAB's own response of one bubble alone vs in a batch.

    The batch-dependence of PROTEUS (Q-50/Q-121) is of the same size as the
    pyproteus-vs-MATLAB differences, so both are reported together.
    """
    eng = matlab_engine
    liquid, gas = materials()
    put_materials(eng, liquid, gas)
    R0 = np.array([1.2e-6, 2.1e-6, 3.5e-6, 4.4e-6])
    fs, f0 = 250e6, 2e6
    t = np.arange(0, 5e-6, 1 / fs)
    p = np.vstack([pulse_signal(t, f0, a) for a in (80e3, 20e3, 150e3, 300e3)])
    put_shells(eng, R0, "Marmottant", 0.01)
    put_pulse(eng, t, p, t, 2 * np.pi * f0)
    R_b, Rd_b, _ = run_matlab_response(eng)
    put_shells(eng, R0[1:2], "Marmottant", 0.01)
    put_pulse(eng, t, p[1:2], t, 2 * np.pi * f0)
    R_1, Rd_1, _ = run_matlab_response(eng)
    T1 = float(np.sqrt(liquid.rho * R0[1] ** 2 / liquid.P0))
    err = ode_errors(R0[1:2], T1, R_b[1:2], Rd_b[1:2], R_1, Rd_1)
    d13 = ode_errors(R0[1:2], T1, R_b[1:2], Rd_b[1:2], R_1, Rd_1, 10 * ODE_REL, ODE_ABS)
    print(f"MATLAB batch vs alone: fraction of the class {err.ravel()}, of D-13 {d13.ravel()}")
    assert d13.max() > 1  # documents that D-13 is not met by MATLAB itself
    assert np.all(err <= 1)
