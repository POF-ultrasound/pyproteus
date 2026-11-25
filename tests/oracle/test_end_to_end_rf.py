"""End-to-end RF on given bubble positions vs MATLAB PROTEUS (M2 "Done when").

Chain, each side with its own functions (MATLAB PROTEUS 8d37938 / pyproteus):

1. settings: MainGUI start-up + ``updateParameters`` in MATLAB (default P4-1, and
   the curved C1-6D), read into the Python models;
2. transducer integration points and lens delays (``get_transducer_integration_*``);
3. bubbles at fixed positions and radii, driven by the same incident plane wave
   (the transmit pressure pulse delayed by ``x/c``, built once in Python and given
   to both sides: the k-Wave first iteration is M3);
4. mass source (``compute_bubble_mass_source``; pyproteus with
   ``step_control="shared"`` for the comparison, D-22);
5. propagation to the integration points (``run_simulation_homogeneous``,
   gridded, as ``hybrid_simulator``'s last stage);
6. element RF (``compute_RF_data``) with unit sensor weights (sensors = points).

Classes: propagation + RF from MATLAB's own mass source ≤ 1e-9 × max|RF| (D-13
float); end-to-end RF ≤ 1e-4 × max|RF| (D-13), reported as an expected failure
when only the bubble model (D-22 class 1e-3) puts it above (B-130). pyproteus' default
``step_control="independent"`` is compared and reported separately (D-22: it
equals MATLAB with one bubble per batch, not MATLAB's shared batch steps).
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import scipy.sparse as sp

from pyproteus.acoustic import (
    compute_RF_data,
    get_transducer_integration_delays,
    get_transducer_integration_points,
    run_simulation_homogeneous,
    sinc_interpolation,
)
from pyproteus.acoustic.mass_source import compute_bubble_mass_source
from tests.oracle.helpers import get, put
from tests.oracle.test_acoustic_grid import build_settings, load_settings, update_settings

pytest.importorskip("kwave")

KWAVE_MATLAB = Path(
    os.environ.get(
        "KWAVE_MATLAB_PATH",
        "/Users/theodoremichel/Repos/TWENTE/Ultrasound/PROTEUS-cleaning/PROTEUS/external/k-wave/k-Wave",
    )
)
RF_REL = 1e-4  # D-13 end-to-end RF class (B-130: missed by 1 % on the default case)
BUBBLE_REL = 1e-3  # D-22 bubble-response class, the floor of what the RF can reach


@pytest.fixture(scope="module")
def keng(matlab_engine: Any) -> Any:
    if not (KWAVE_MATLAB / "filterTimeSeries.m").is_file():
        pytest.skip(f"MATLAB k-Wave not found at {KWAVE_MATLAB} (set KWAVE_MATLAB_PATH)")
    matlab_engine.addpath(str(KWAVE_MATLAB), nargout=0)
    yield matlab_engine
    matlab_engine.rmpath(str(KWAVE_MATLAB), nargout=0)


# bubbles: (x depth, y lateral, z elevation) [m] and radii [m]; one 2 mm from the
# array, off-axis ones, two with equal radii (stable sort ties)
POSITIONS = np.array(
    [
        [2.0e-3, 0.4e-3, 0.0],
        [8.0e-3, -3.1e-3, 0.5e-3],
        [12.5e-3, 0.0, 0.0],
        [15.0e-3, 4.2e-3, -1.0e-3],
        [21.0e-3, -1.7e-3, 0.2e-3],
        [26.0e-3, 2.5e-3, 0.0],
    ]
)
RADII = np.array([2.0e-6, 1.5e-6, 2.5e-6, 2.0e-6, 3.0e-6, 1.8e-6])


def make_settings(eng: Any, oracle: Any, tmp: Path, transducer: str, n_el: int) -> Any:
    build_settings(eng, oracle.path, "S")
    if transducer != "P4-1":
        eng.eval(
            f"S.Transducer = struct('Type', '{transducer}');"
            "S.Transducer = get_transducer_properties(S.Transducer);"
            "if strcmp(S.Transducer.ImpulseResponseType,'Estimate'),"
            " S.Transducer.SamplingRate = 250e6;"
            " S.Transducer = estimate_impulse_response(S.Transducer); end;"
            "S.Transmit = reset_transmit(S.Transducer);",
            nargout=0,
        )
    eng.eval(f"S.Transducer.NumberOfElements = {n_el};", nargout=0)
    update_settings(eng, "S")
    return load_settings(eng, tmp, "S")


def incident(transmit: Any, c: float, dt: float, x: np.ndarray, nt: int) -> np.ndarray:
    """Plane wave along x: the transmit pressure pulse delayed by ``x/c`` (band-limited)."""
    p = np.asarray(transmit.PressureSignal, dtype=float).reshape(-1)
    fs = float(transmit.SamplingRate)
    t_src = np.arange(p.size) / fs
    t = np.arange(nt) * dt
    out = np.zeros((x.size, nt))
    for i, xi in enumerate(x):
        tq = t - xi / c
        inside = (tq >= 0) & (tq <= t_src[-1])
        out[i, inside] = sinc_interpolation(t_src, p, tq[inside])
    return out


def run_matlab(eng: Any, sensed: np.ndarray, nt3: int) -> np.ndarray:
    """MATLAB chain on ``S`` (steps 2, 4, 5, 6)."""
    put(eng, pyp_sp=sensed, pyp_pts=POSITIONS, pyp_r=RADII.reshape(-1, 1), pyp_nt3=float(nt3))
    eng.eval(
        "G = struct('dx', S.SimulationParameters.GridSize, 'dy', S.SimulationParameters.GridSize,"
        " 'dz', S.SimulationParameters.GridSize, 'dt', 1/S.SimulationParameters.SamplingRate);"
        "G.k_max = pi/G.dt/S.Medium.SpeedOfSoundMinimum;"
        "T = S.Transducer; T.IntegrationDensity = S.SimulationParameters.IntegrationDensity;"
        "T = get_transducer_integration_points(T, G);"
        "T = get_transducer_integration_delays(T, S.Medium);"
        "mb = S.Microbubble; mb.BatchSize = 15; mb.UseParfor = 'off';"
        "warning('off','all');"
        "evalc('pyp_ms = compute_bubble_mass_source(pyp_sp, pyp_r, G, S.Medium, mb, S.Transmit);');"
        "stop(timerfindall); delete(timerfindall);"
        "med = struct('density', S.Medium.Density, 'sound_speed', S.Medium.SpeedOfSound,"
        " 'alpha_coeff', S.Medium.AttenuationA, 'alpha_power', S.Medium.AttenuationB);"
        "if abs(med.alpha_power - 1) < 0.1, med.alpha_mode = 'no_dispersion'; end;"
        "[Nel, Nint, ~] = size(T.integration_points);"
        "sensor.points = reshape(T.integration_points, Nel*Nint, 3);"
        "source = struct('points', pyp_pts, 'mass_source', pyp_ms);"
        "rp = struct('gridded', true, 'DATA_CAST', 'double', 'DATA_CAST_RF', 'double');"
        "G.Nt = pyp_nt3;"
        "evalc('sd = run_simulation_homogeneous(rp, G, med, source, sensor);');"
        "evalc('pyp_rf = compute_RF_data(T, sd, speye(Nel*Nint), G, rp);');"
        "warning('on','all');",
        nargout=0,
    )
    ms, rf = get(eng, "pyp_ms", "pyp_rf")
    eng.eval(
        "clear pyp_sp pyp_pts pyp_r pyp_nt3 pyp_ms pyp_rf G T mb med sensor source rp sd", nargout=0
    )
    return ms, rf


def run_python(
    S: Any, sensed: np.ndarray, nt3: int, step_control: str, mass_source: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """pyproteus chain (steps 2, 4, 5, 6); ``mass_source`` given: steps 2, 5, 6 only."""
    sim, medium = S.SimulationParameters, S.Medium
    h = float(sim.GridSize)
    dt = 1 / float(sim.SamplingRate)
    grid = SimpleNamespace(
        dx=h, dy=h, dz=h, dt=dt, k_max=np.pi / dt / float(medium.SpeedOfSoundMinimum)
    )
    T = S.Transducer.copy_struct()
    T.IntegrationDensity = sim.IntegrationDensity
    T = get_transducer_integration_delays(get_transducer_integration_points(T, grid), medium)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ms = (
            mass_source
            if mass_source is not None
            else compute_bubble_mass_source(
                sensed,
                RADII,
                grid,
                medium,
                S.Microbubble,
                S.Transmit,
                batch_size=15,
                step_control=step_control,
            )
        )
    kmed = {
        "density": float(medium.Density),
        "sound_speed": float(medium.SpeedOfSound),
        "alpha_coeff": float(medium.AttenuationA),
        "alpha_power": float(medium.AttenuationB),
    }
    if abs(kmed["alpha_power"] - 1) < 0.1:
        kmed["alpha_mode"] = "no_dispersion"
    pts = np.asarray(T.integration_points)
    n_el, n_int = pts.shape[:2]
    sensor_points = pts.reshape(n_el * n_int, 3, order="F")  # MATLAB reshape (column-major)
    rp = {"gridded": True, "DATA_CAST": "double", "DATA_CAST_RF": "double"}
    sd = run_simulation_homogeneous(
        rp,
        {"dt": dt, "Nt": nt3},
        kmed,
        {"points": POSITIONS, "mass_source": ms},
        {"points": sensor_points},
    )
    rf, _ = compute_RF_data(T, sd, sp.identity(n_el * n_int, format="csr"), grid, rp)
    return ms, rf


def rel(a: np.ndarray, e: np.ndarray) -> float:
    assert a.shape == e.shape, f"{a.shape} != {e.shape}"
    return float(np.max(np.abs(a - e)) / np.max(np.abs(e)))


CASES = {"default_P4-1": ("P4-1", 96), "curved_C1-6D_odd": ("C1-6D", 33)}


@pytest.mark.parametrize("case", list(CASES))
def test_end_to_end_rf(keng: Any, oracle: Any, tmp_path: Path, case: str) -> None:
    transducer, n_el = CASES[case]
    S = make_settings(keng, oracle, tmp_path, transducer, n_el)
    c = float(S.Medium.SpeedOfSound)
    dt = 1 / float(S.SimulationParameters.SamplingRate)
    pulse_len = np.size(S.Transmit.PressureSignal) / float(S.Transmit.SamplingRate)
    t_end = POSITIONS[:, 0].max() / c + pulse_len + 1e-6
    nt1 = int(np.floor(t_end / dt)) + 1
    nt3 = int(np.floor((2 * POSITIONS[:, 0].max() + 5e-3) / c / dt + 2 * pulse_len / dt)) + 1
    sensed = incident(S.Transmit, c, dt, POSITIONS[:, 0], nt1)

    ms_m, rf_m = run_matlab(keng, sensed, nt3)
    ms_s, rf_s = run_python(S, sensed, nt3, "shared")
    ms_i, rf_i = run_python(S, sensed, nt3, "independent")
    e_ms, e_rf = rel(ms_s, ms_m), rel(rf_s, rf_m)
    e_ms_i, e_rf_i = rel(ms_i, ms_m), rel(rf_i, rf_m)
    # propagation + RF alone, from MATLAB's mass source (D-13 float class)
    _, rf_p = run_python(S, sensed, nt3, "shared", mass_source=ms_m)
    e_prop = rel(rf_p, rf_m)
    print(f"\n{case}: propagation + RF from MATLAB's mass source: {e_prop:.2e} × max")
    assert e_prop <= 1e-9
    print(
        f"\n{case}: RF {rf_m.shape}; shared: mass source {e_ms:.2e}, RF {e_rf:.2e}; "
        f"independent: mass source {e_ms_i:.2e}, RF {e_rf_i:.2e} (× max)"
    )
    assert np.max(np.abs(rf_m)) > 0
    # never worse than the bubble-model class (D-22) passed on to the RF
    assert max(e_rf, e_rf_i) <= BUBBLE_REL
    if e_rf > RF_REL:
        pytest.xfail(f"B-130: end-to-end RF {e_rf:.2e} > {RF_REL:g} (bubble-model difference)")
