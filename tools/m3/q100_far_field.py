"""Q-100: time offsets of the simulated RF chain, point scatterers at several depths.

Decides whether DAS's ``timeToPeak = I/Fs`` (one transmit sample after the
pulse peak ``(I-1)/Fs``) is absorbed by the k-Wave correction or is an
off-by-one. pyproteus only (its hybrid/full paths match MATLAB ``main_RF``,
tests/oracle/test_main_rf.py); MATLAB builds the settings once.

Medium: homogeneous, lossless, linear (Inhomogeneity 0, BonA 0, AttenuationA 0).
Array: P4-1 shrunk to 8 elements (2.4 mm), element height 1 mm, no elevation
lens; plane wave. Scatterers on the axis at 5..23 mm (the aperture's near field
ends at ~2.4 mm): far field.

Stages, each against an exact reference (homogeneous Green's function,
``run_simulation_homogeneous`` with ``gridded=False``, analytic delays):

A. Transmit, k-Wave (both paths): pressure at the scatterers vs the field of
   the integration points emitting ``PressureSignal`` (unshifted, t = 0 at
   sample 1) as monopoles: lag ``tau_A``.
B. Receive, hybrid: point source at the scatterer, gridded propagation to the
   transducer voxels + ``compute_RF`` with the sensor weights vs exact
   propagation to the integration points: lag ``tau_B_hyb``.
C. Receive, full: the same point source in k-Wave (``update_source``) recorded
   by the transducer sensor + ``compute_RF``: lag ``tau_B_full``.

PROTEUS's DAS time axis assumes the simulated chain is early by
``kwave = dx/(2c) + dt`` (hybrid) or ``dx/(2c) + 3/2 dt`` (full), and the echo
peak at ``timeToPeak`` after the time origin. With the RF time axis 0-based
throughout (QUIRKS Q-100), the correct peak time is ``(I-1)/Fs``; ``I/Fs`` is
right only if the chain is early by ``kwave + 1/Fs_tx``.

D. DAS: ideal linear reflector (mass source ``∫ p_inc dt``, so the echo is a
   delayed copy of the incident pressure) through the hybrid path and
   ``compute_RF``; DAS along the axis with ``das_time_axis`` (I/Fs) and with
   ``(I-1)/Fs``; envelope-peak depth minus true depth.

Usage (MATLAB shared session 'pyp' for the settings, then pyproteus):
    PYTHONPATH=src:. python tools/m3/q100_far_field.py /tmp/claude/m3/q100 [ppw]
"""

from __future__ import annotations

import dataclasses
import json
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.signal import hilbert

from pyproteus.acoustic import (
    compute_RF_data,
    define_sensor_MB,
    define_source_transducer,
    run_simulation,
    run_simulation_homogeneous,
    update_source,
)
from pyproteus.acoustic.main import _nt, _with_seq_pulse, prepare_simulation
from pyproteus.acoustic.sensor import _find
from pyproteus.acoustic.setup import setup_run
from pyproteus.numerics import MatlabRNG, interp1, matlab_round
from pyproteus.reconstruction import das_time_axis, time_to_peak

DEPTHS = np.array([5.0, 8.0, 11.0, 14.0, 17.0, 20.0, 23.0]) * 1e-3
UP = 64  # cross-correlation upsampling


def make_settings(root: Path, ppw: float) -> Path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tests.oracle.main_rf_case import Case, matlab_settings

    from m3.run_case import ORACLE, engine

    eng = engine()
    case = Case(
        name="q100",
        n_el=8,
        element_height=1e-3,
        ppw=ppw,
        start_depth=4e-3,
        diagonal=(21e-3, 1e-3, 1e-3),
        inhomogeneity=0.0,
        bonA=0.0,
        n_interactions=0,
    )
    matlab_settings(eng, ORACLE, case)
    eng.eval(
        "S.Transducer.ElevationFocus = Inf; S.Medium.AttenuationA = 0;"
        "S.Geometry = compute_simulation_domain(S.Geometry, S.Transducer, S.Transmit);",
        nargout=0,
    )
    f = root / f"q100_ppw{ppw:g}.mat"
    eng.eval(f"save('{f}', '-struct', 'S'); clear S", nargout=0)
    return f


def lag(x: np.ndarray, ref: np.ndarray, dt: float) -> float:
    """Delay of ``x`` relative to ``ref`` [s]: peak of the upsampled cross-correlation."""
    n = 1 << int(np.ceil(np.log2(2 * max(x.size, ref.size))))
    X, R = np.fft.rfft(x, n), np.fft.rfft(ref, n)
    cc = np.fft.irfft(X * np.conj(R), n * UP)
    k = int(np.argmax(cc))
    if k > cc.size // 2:
        k -= cc.size
    return k * dt / UP


def env_peak(t: np.ndarray, v: np.ndarray) -> float:
    e = np.abs(hilbert(v))
    i = int(np.argmax(e))
    if 0 < i < e.size - 1:  # parabolic sub-sample fit
        a, b, c = e[i - 1], e[i], e[i + 1]
        return float(t[i] + 0.5 * (a - c) / (a - 2 * b + c) * (t[1] - t[0]))
    return float(t[i])


def main(root: Path, ppw: float) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    sfile = root / f"q100_ppw{ppw:g}.mat"
    if not sfile.is_file():
        sfile = make_settings(root, ppw)
    from tracer_bullet import kwave_tracer as kt

    kw = {"binary_env": kt.binary_env()}
    setup = setup_run(sfile, root, root / "out", binary_path=kt.WRAPPER, check_ground_truth=False)
    setup.settings.Medium.Save = False
    S = setup.settings
    prep = prepare_simulation(setup, rng=MatlabRNG(0))
    grid, rp, T = prep.grid, dict(prep.run_param), prep.transducer
    dt, c = grid.dt, float(S.Medium.SpeedOfSound)
    kgrid = dataclasses.replace(prep.kgrid, Nt=_nt(rp["tr"][2], dt))
    nt = kgrid.Nt
    pts = np.column_stack([DEPTHS, np.zeros_like(DEPTHS), np.zeros_like(DEPTHS)])
    sensor_mb, w_mb, mb, _ = define_sensor_MB(
        grid, {"points": pts, "radii": np.ones((len(pts), 1))}
    )
    assert mb["points"].shape[0] == len(DEPTHS), "scatterers outside the grid"

    # ---------------- A: transmit
    src = define_source_transducer(
        T,
        _with_seq_pulse(prep.transmit, "pulse"),
        S.Medium,
        grid,
        prep.sensor_weights.T,
        prep.mask_idx_trans,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sd = run_simulation(rp, kgrid, prep.medium, src, sensor_mb, **kw)
    p_inc = np.asarray(w_mb @ np.asarray(sd["p"], dtype=float))

    ip = np.asarray(T.integration_points)
    n_el, n_int = ip.shape[:2]
    points_src = ip.reshape(n_el * n_int, 3, order="F")
    s = np.asarray(prep.transmit.PressureSignal, dtype=float).reshape(-1)
    s_pad = np.zeros(nt)
    s_pad[: min(nt, s.size)] = s[:nt]
    w_int = np.asarray(T.integration_weights, dtype=float).reshape(-1, order="F")
    delays = (
        np.asarray(T.integration_transmit_delays, float)
        + np.asarray(S.Transmit.Delays, float).reshape(-1, 1)
    ).reshape(-1, order="F")
    assert np.allclose(delays, delays[0]), "study assumes no transmit/lens delays"
    w_int = np.broadcast_to(w_int, (points_src.shape[0],)) if w_int.size == 1 else w_int
    ms = w_int[:, None] * s_pad[None, :]
    hom = {
        "density": float(S.Medium.Density),
        "sound_speed": c,
        "alpha_coeff": 0.0,
        "alpha_power": float(S.Medium.AttenuationB),
    }
    ref = run_simulation_homogeneous(
        {"gridded": False, "DATA_CAST": "double"},
        {"dt": dt, "Nt": nt},
        hom,
        {"points": points_src, "mass_source": ms},
        {"points": mb["points"]},
    )["p"]
    tau_A = np.array([lag(p_inc[i], np.asarray(ref[i]), dt) for i in range(len(DEPTHS))])
    shape_A = [
        float(np.corrcoef(p_inc[i], np.roll(ref[i], int(matlab_round(tau_A[i] / dt))))[0, 1])
        for i in range(len(DEPTHS))
    ]

    # ---------------- B: receive, hybrid (gridded to voxel centres) vs exact
    pulse = np.asarray(ref, dtype=float)  # any band-limited pulse; reuse the incident field
    idx = prep.mask_idx_trans
    sub = np.unravel_index(idx, grid.shape, order="F")
    vox = np.column_stack([grid.x[sub[0]], grid.y[sub[1]], grid.z[sub[2]]])
    rpd = {"DATA_CAST": "double", "DATA_CAST_RF": "double"}
    tau_Bh, tau_Bf = [], []
    for i in range(len(DEPTHS)):
        q = pulse[i : i + 1]
        one = {"points": mb["points"][i : i + 1], "mass_source": q}
        g = run_simulation_homogeneous(
            {**rpd, "gridded": True}, {"dt": dt, "Nt": nt}, hom, one, {"points": vox}
        )
        rf_g, _ = compute_RF_data(T, g, prep.sensor_weights, grid, rpd)
        e = run_simulation_homogeneous(
            {**rpd, "gridded": False}, {"dt": dt, "Nt": nt}, hom, one, {"points": points_src}
        )
        import scipy.sparse as sp

        rf_e, _ = compute_RF_data(T, e, sp.identity(points_src.shape[0], format="csr"), grid, rpd)
        rf_g, rf_e = np.asarray(rf_g), np.asarray(rf_e)
        tau_Bh.append(np.mean([lag(rf_g[n], rf_e[n], dt) for n in range(n_el)]))
        # C: same source in k-Wave
        w1 = w_mb[i : i + 1]
        mask_i = np.asarray(w1.indices)
        idx_i = _find(sensor_mb.mask)[mask_i]
        src_k = update_source(None, q, w1[:, mask_i].T, idx_i, grid, prep.medium)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sk = run_simulation(rp, kgrid, prep.medium, src_k, prep.sensor_transducer, **kw)
        rf_k, _ = compute_RF_data(T, sk, prep.sensor_weights, grid, rpd)
        rf_k = np.asarray(rf_k)
        tau_Bf.append(np.mean([lag(rf_k[n], rf_e[n], dt) for n in range(n_el)]))
    tau_Bh, tau_Bf = np.array(tau_Bh), np.array(tau_Bf)

    # ---------------- D: DAS along the axis, ideal reflector through the hybrid path
    q_ideal = np.cumsum(p_inc, axis=1) * dt  # echo = delayed copy of the incident pressure
    g = run_simulation_homogeneous(
        {**rpd, "gridded": True},
        {"dt": dt, "Nt": nt},
        hom,
        {"points": mb["points"], "mass_source": q_ideal},
        {"points": vox},
    )
    das_res = {"I": [], "I-1": []}
    fs_tx = float(S.Transmit.SamplingRate)
    y_el = np.mean(ip[:, :, 1], axis=1)
    for i in range(len(DEPTHS)):
        gi = run_simulation_homogeneous(
            {**rpd, "gridded": True},
            {"dt": dt, "Nt": nt},
            hom,
            {"points": mb["points"][i : i + 1], "mass_source": q_ideal[i : i + 1]},
            {"points": vox},
        )
        rf, _ = compute_RF_data(T, gi, prep.sensor_weights, grid, rpd)
        rf = np.asarray(rf)
        t_axis = das_time_axis(rf.shape[1], S.SimulationParameters, T, S.Transmit, c)
        x_true = mb["points"][i, 0]
        xs = x_true + np.linspace(-0.6e-3, 0.6e-3, 1201)
        for key, shift in (("I", 0.0), ("I-1", 1 / fs_tx)):
            t_ax = t_axis + shift  # (I-1)/Fs: one transmit sample less subtracted
            line = np.zeros(xs.size)
            for n in range(n_el):
                tof = xs / c + np.hypot(xs, y_el[n]) / c
                line += interp1(t_ax, np.abs(hilbert(rf[n])), tof, "linear", 0.0)
            j = int(np.argmax(line))
            a, b, cc = line[j - 1], line[j], line[j + 1]
            xpk = xs[j] + 0.5 * (a - cc) / (a - 2 * b + cc) * (xs[1] - xs[0])
            das_res[key].append(float(xpk - x_true))
    del g

    sim = S.SimulationParameters
    dx = float(sim.GridSize)
    out = {
        "ppw": ppw,
        "dx": dx,
        "dt": dt,
        "fs_tx": fs_tx,
        "c": c,
        "depths_mm": (mb["points"][:, 0] * 1e3).tolist(),
        "kwave_hybrid": dx / (2 * c) + dt,
        "kwave_full": dx / (2 * c) + 1.5 * dt,
        "tau_A": tau_A.tolist(),
        "shape_corr_A": shape_A,
        "tau_B_hybrid": tau_Bh.tolist(),
        "tau_B_full": tau_Bf.tolist(),
        "time_to_peak_I": time_to_peak(S.Transmit, T),
        "das_depth_error_I_um": [1e6 * v for v in das_res["I"]],
        "das_depth_error_Iminus1_um": [1e6 * v for v in das_res["I-1"]],
    }
    (root / f"q100_ppw{ppw:g}.json").write_text(json.dumps(out, indent=1))
    ns = 1e9
    print(f"ppw {ppw}: dx {dx * 1e6:.1f} um, dt {dt * ns:.2f} ns, 1/Fs_tx {ns / fs_tx:.2f} ns")
    print(f"kwave hybrid {out['kwave_hybrid'] * ns:.2f} ns, full {out['kwave_full'] * ns:.2f} ns")
    for i, d in enumerate(out["depths_mm"]):
        print(
            f"{d:6.2f} mm  tau_A {tau_A[i] * ns:7.2f}  (shape r {shape_A[i]:.5f})  "
            f"tau_B hyb {tau_Bh[i] * ns:6.2f}  full {tau_Bf[i] * ns:6.2f}  "
            f"DAS err I {das_res['I'][i] * 1e6:6.2f} um, I-1 {das_res['I-1'][i] * 1e6:6.2f} um"
        )
    return out


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main(Path(sys.argv[1]), float(sys.argv[2]) if len(sys.argv) > 2 else 6.0)
    _ = SimpleNamespace
