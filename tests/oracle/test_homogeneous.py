"""Homogeneous propagation vs MATLAB PROTEUS.

Functions: ``calc_scatter_attenuated``, ``run_simulation_homogeneous``.

Same inputs through the MATLAB Engine and Python. The mass sources include
bubble responses computed by MATLAB PROTEUS (``compute_bubble_mass_source``
with the default microbubble, medium and transmit), so this does not wait for
the Python microbubble model.

Classes (D-13):
- double-precision input: float, ``|Δp| ≤ 1e-9 max|p|`` per case;
- single-precision input (PROTEUS default ``DATA_CAST = 'single'``): MATLAB
  computes the phase ``2 pi r f / c`` in single precision; pyproteus works in
  float64 (docs/PARITY.md). Compared at ``SINGLE_REL`` relative to max|p|,
  plus a check that pyproteus is closer to MATLAB's double-precision result than
  MATLAB's own single-precision result is;
- gridded distances: indices after ``round`` (statistical class): the grid
  and indices are compared directly, the field at the float class.

MATLAB k-Wave must be on the path (``powerLawKramersKronig``, ``db2neper``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pyproteus.acoustic import homogeneous as hom
from tests.oracle.helpers import col, get, put

KWAVE_MATLAB = Path(
    os.environ.get(
        "KWAVE_MATLAB_PATH",
        "/Users/theodoremichel/Repos/TWENTE/Ultrasound/PROTEUS-cleaning/PROTEUS/external/k-wave/k-Wave",
    )
)
REL = 1e-9
SINGLE_REL = 1e-5  # k-Wave single class; measured ≤ 8.1e-7 (docs/PARITY.md)


@pytest.fixture(scope="module")
def keng(matlab_engine: Any) -> Any:
    """Engine with MATLAB k-Wave on the path (the shared fixture does not add it)."""
    if not (KWAVE_MATLAB / "powerLawKramersKronig.m").is_file():
        pytest.skip(f"MATLAB k-Wave not found at {KWAVE_MATLAB} (set KWAVE_MATLAB_PATH)")
    matlab_engine.addpath(str(KWAVE_MATLAB), nargout=0)
    yield matlab_engine
    matlab_engine.rmpath(str(KWAVE_MATLAB), nargout=0)


def rel_err(a: np.ndarray, e: np.ndarray) -> float:
    a, e = np.asarray(a, dtype=float), np.asarray(e, dtype=float)
    assert a.shape == e.shape, f"shape {a.shape} != {e.shape}"
    nan = np.isnan(e)
    assert np.array_equal(np.isnan(a), nan), "NaN positions differ"
    if nan.all():
        return 0.0
    scale = np.max(np.abs(e[~nan]))
    err = float(np.max(np.abs(a[~nan] - e[~nan])) / max(scale, np.finfo(float).tiny))
    if os.environ.get("PYPROTEUS_REPORT"):
        print(f"\n  rel err {err:.2e}", end="")
    return err


# ---------------------------------------------------------------- inputs

DT = 1 / 40e6


def gauss_sources(n_src: int, n: int, dt: float = DT, f0: float = 3e6) -> np.ndarray:
    t = np.arange(n) * dt
    pulse = np.sin(2 * np.pi * f0 * t) * np.exp(-(((t - 1e-6) / 4e-7) ** 2))
    return pulse[None, :] * np.arange(1, n_src + 1)[:, None] * 1e-12


@pytest.fixture(scope="module")
def bubble_mass_source(keng: Any) -> dict[str, np.ndarray]:
    """Mass sources of three bubbles from MATLAB PROTEUS (default settings).

    As the oracle's ``regression_cases.m`` (``case_bubble_mass_source``): the
    transmit pulse resampled to the grid rate and scaled per bubble, through
    ``compute_bubble_mass_source``.
    """
    keng.eval(
        "pyp_S.Microbubble = reset_microbubble(); pyp_S.Medium = reset_medium();"
        "pyp_S.Transducer = reset_transducer(); pyp_S.Transmit = reset_transmit(pyp_S.Transducer);"
        "pyp_S.SimulationParameters = reset_simulation_parameters();"
        "pyp_S.SimulationParameters = update_simulation_parameters(pyp_S.SimulationParameters,"
        " pyp_S.Medium, pyp_S.Transmit.CenterFrequency);"
        "pyp_S.SimulationParameters.GridSize = pyp_S.Medium.SpeedOfSound/"
        "(pyp_S.Transmit.CenterFrequency*pyp_S.SimulationParameters.PointsPerWavelength);"
        "pyp_mb = pyp_S.Microbubble; pyp_mb.BatchSize = 15; pyp_mb.UseParfor = 'off';"
        "pyp_dt = 1/pyp_S.SimulationParameters.SamplingRate;"
        "pyp_kg = struct('dt', pyp_dt, 'k_max', pi/pyp_S.SimulationParameters.GridSize);"
        "pyp_p = pyp_S.Transmit.PressureSignal;"
        "pyp_t0 = (0:numel(pyp_p)-1)/pyp_S.Transmit.SamplingRate;"
        "pyp_t = 0:pyp_dt:(pyp_t0(end) + 4e-6);"
        "pyp_p = interp1(pyp_t0, pyp_p, pyp_t, 'linear', 0);"
        "evalc(['pyp_ms = compute_bubble_mass_source([1; 0.5; 2]*pyp_p, "
        "[1.5e-6; 2e-6; 2.5e-6], pyp_kg, pyp_S.Medium, pyp_mb, pyp_S.Transmit);']);"
        "pyp_c = pyp_S.Medium.SpeedOfSound;",
        nargout=0,
    )
    ms, dt, c = get(keng, "pyp_ms", "pyp_dt", "pyp_c")
    keng.eval("clear pyp_S pyp_mb pyp_dt pyp_kg pyp_p pyp_t0 pyp_t pyp_ms pyp_c", nargout=0)
    return {"mass_source": ms, "dt": float(dt[0, 0]), "c": float(c[0, 0])}


def matlab_scatter(
    eng: Any, ms: np.ndarray, medium: dict, dt: float, nt: int, r: np.ndarray
) -> np.ndarray:
    put(eng, pyp_ms=ms, pyp_r=r, pyp_dt=dt, pyp_nt=float(nt))
    put(eng, pyp_c=medium["c"], pyp_a=medium["a"], pyp_b=medium["b"])
    mode = medium.get("alpha_mode")
    eng.eval(
        "pyp_med = struct('c', pyp_c, 'a', pyp_a, 'b', pyp_b);"
        + (f"pyp_med.alpha_mode = '{mode}';" if mode else "")
        + "pyp_kg = struct('dt', pyp_dt, 'Nt', pyp_nt);"
        "pyp_out = calc_scatter_attenuated(pyp_ms, pyp_med, pyp_kg, pyp_r);",
        nargout=0,
    )
    out = get(eng, "pyp_out")
    eng.eval("clear pyp_ms pyp_r pyp_dt pyp_nt pyp_c pyp_a pyp_b pyp_med pyp_kg pyp_out", nargout=0)
    n_sen, n_src = r.shape
    return out.reshape(n_sen, nt, n_src)  # MATLAB drops a trailing singleton


def distances(n_sen: int, n_src: int, c: float, dt: float, nt: int, seed: int) -> np.ndarray:
    """Near (< 1 wavelength), typical, after the window (Nt..2Nt) and wrapping (> 2Nt) distances."""
    rng = np.random.default_rng(seed)
    r = rng.uniform(1e-3, 0.6 * nt * dt * c, size=(n_sen, n_src))
    r[0, 0] = 5e-4 * rng.uniform(0.5, 1.0)  # bubble very close to the sensor
    if n_sen > 2:
        r[1, -1] = 1.4 * nt * dt * c  # arrives after the window: cropped
        r[2, 0] = 2.3 * nt * dt * c  # beyond 2 Nt: MATLAB wraps (Q-131), pyproteus not (D-23)
    if n_sen > 3 and n_src > 1:
        r[3, 1] = 0.0  # source on the sensor: silent
    return r


WRAP_ROW = 2  # row of distances() whose arrival MATLAB wraps into the window
FAR_ROWS = [1, 2]  # rows beyond Nt*dt*c0: pyproteus lengthens its FFT (D-23)


#: Class of the non-wrapping rows when a far source lengthens pyproteus's FFT
#: (D-23): MATLAB's 2*Nt periodic FFT leaks the band-limited (sinc) delay tails
#: of sources with an abrupt onset (these test pulses start at 2e-3 of their
#: peak) around the period; a longer period moves that leakage. Measured
#: ≤ 4.1e-6 × max; end-to-end RF class. Unreachable with PROTEUS's own Nt.
LONG_FFT_REL = 1e-4


def compare_without_wrap(actual: np.ndarray, expected: np.ndarray, rel: float) -> None:
    """MATLAB parity on the near rows; the far rows (arrivals after the window) are silent here.

    MATLAB wraps the far arrivals into the window (Q-131: always the ``2.3 Nt``
    row, the ``1.4 Nt`` row too when the pulse is long compared with ``Nt``);
    pyproteus does not (D-23). The far sources lengthen pyproteus's FFT for the
    whole case, so the near rows are compared at ``max(rel, LONG_FFT_REL)``; the
    strict class is checked on the same cases without the far sources.
    """
    keep = np.setdiff1d(np.arange(actual.shape[0]), FAR_ROWS)
    assert rel_err(actual[keep], expected[keep]) <= max(rel, LONG_FFT_REL)
    scale = np.max(np.abs(expected[keep]))
    # band-limited leakage of the abrupt source onset only (measured ≤ 4e-7 × max)
    assert np.max(np.abs(actual[1, :, -1])) <= 1e-5 * scale  # FAR_ROWS[0] is source -1
    assert np.max(np.abs(actual[WRAP_ROW, :, 0])) <= 1e-5 * scale
    assert np.max(np.abs(expected[WRAP_ROW, :, 0])) > 1e-3 * scale  # MATLAB wraps here


MEDIA = {
    "tissue_b1.5": {"c": 1540.0, "a": 0.75, "b": 1.5},
    "water_b2": {"c": 1480.0, "a": 0.002, "b": 2.0},
    "lossless": {"c": 1540.0, "a": 0.0, "b": 1.5},
    "b1.05_no_dispersion": {"c": 1450.0, "a": 0.6, "b": 1.05, "alpha_mode": "no_dispersion"},
    "b1.2_no_dispersion": {"c": 1575.0, "a": 0.15, "b": 1.2, "alpha_mode": "no_dispersion"},
    "b2.5": {"c": 1540.0, "a": 0.5, "b": 2.5},
}

# (n_src, N, Nt): N == Nt even/odd, N < Nt (padding), N > Nt (cropping), one source
SHAPES = [
    (3, 200, 200),
    (2, 151, 151),
    (4, 120, 257),
    (2, 300, 211),
    (1, 96, 96),
]  # sources end near zero (Q-132)


# ---------------------------------------------------------------- calc_scatter_attenuated


@pytest.mark.parametrize("medium", list(MEDIA), ids=list(MEDIA))
@pytest.mark.parametrize("shape", SHAPES, ids=[f"src{s}-N{n}-Nt{t}" for s, n, t in SHAPES])
def test_calc_scatter_attenuated_double(keng: Any, medium: str, shape: tuple) -> None:
    n_src, n, nt = shape
    med = MEDIA[medium]
    ms = gauss_sources(n_src, n)
    r = distances(7, n_src, med["c"], DT, nt, seed=n + nt)
    expected = matlab_scatter(keng, ms, med, DT, nt, r)
    actual = hom.calc_scatter_attenuated(ms, med, {"dt": DT, "Nt": nt}, r)
    assert actual.dtype == np.float64
    compare_without_wrap(actual, expected, REL)
    # without the far sources the FFT length is MATLAB's 2*Nt: all rows at the float class
    near = np.delete(r, FAR_ROWS, axis=0)
    exp_near = matlab_scatter(keng, ms, med, DT, nt, near)
    assert (
        rel_err(hom.calc_scatter_attenuated(ms, med, {"dt": DT, "Nt": nt}, near), exp_near) <= REL
    )
    if n_src > 1:
        assert np.all(actual[3, :, 1] == 0)


def test_calc_scatter_attenuated_single(keng: Any) -> None:
    """Single-precision mass source (PROTEUS default): looser class.

    pyproteus (float64 inside) must also be closer to the double result.
    """
    med = MEDIA["tissue_b1.5"]
    nt = 400
    ms = gauss_sources(3, nt)
    r = distances(9, 3, med["c"], DT, nt, seed=3)
    exp_single = matlab_scatter(keng, ms.astype(np.float32), med, DT, nt, r.astype(np.float32))
    exp_double = matlab_scatter(keng, ms, med, DT, nt, r)
    actual = hom.calc_scatter_attenuated(ms.astype(np.float32), med, {"dt": DT, "Nt": nt}, r)
    assert actual.dtype == np.float32
    compare_without_wrap(actual, exp_single, SINGLE_REL)
    # without the far sources (MATLAB's FFT length): single class, and pyproteus (float64
    # inside) at least as close to MATLAB's double result as MATLAB single is
    near = np.delete(r, FAR_ROWS, axis=0)
    exp_single = matlab_scatter(keng, ms.astype(np.float32), med, DT, nt, near.astype(np.float32))
    exp_double = matlab_scatter(keng, ms, med, DT, nt, near)
    actual = hom.calc_scatter_attenuated(ms.astype(np.float32), med, {"dt": DT, "Nt": nt}, near)
    assert rel_err(actual, exp_single) <= SINGLE_REL
    assert rel_err(actual, exp_double) <= rel_err(exp_single, exp_double)


def test_calc_scatter_attenuated_bubble_response(keng: Any, bubble_mass_source: dict) -> None:
    """MATLAB PROTEUS bubble responses (default settings) as the mass sources."""
    ms = bubble_mass_source["mass_source"]
    dt, c = bubble_mass_source["dt"], bubble_mass_source["c"]
    nt = ms.shape[1] + 37  # zero-padded to the window, as hybrid_simulator's later stages
    med = {"c": c, "a": 0.75, "b": 1.5}
    r = distances(6, ms.shape[0], c, dt, nt, seed=11)
    expected = matlab_scatter(keng, ms, med, dt, nt, r)
    actual = hom.calc_scatter_attenuated(ms, med, {"dt": dt, "Nt": nt}, r)
    compare_without_wrap(actual, expected, REL)


@pytest.mark.parametrize("b", [0.5, 1.0], ids=["b0.5", "b1.0"])
def test_calc_scatter_attenuated_singular_dispersion(keng: Any, b: float) -> None:
    """``b <= 1`` with dispersion: MATLAB gives NaN (Q-130); pyproteus a finite field (D-23)."""
    med = {"c": 1540.0, "a": 0.5, "b": b}
    ms = gauss_sources(2, 100)
    r = np.random.default_rng(5).uniform(1e-3, 2e-3, (4, 2))
    expected = matlab_scatter(keng, ms, med, DT, 100, r)
    assert np.isnan(expected).all()
    with pytest.raises(ValueError, match="reference"):
        hom.calc_scatter_attenuated(ms, med, {"dt": DT, "Nt": 100}, r)
    actual = hom.calc_scatter_attenuated(
        ms, {**med, "reference_frequency": 3e6}, {"dt": DT, "Nt": 100}, r
    )
    assert np.isfinite(actual).all() and np.abs(actual).max() > 0


def test_power_law_kramers_kronig_and_db2neper(keng: Any) -> None:
    """The clean-room Kramers–Kronig speed and dB→Np conversion vs MATLAB k-Wave."""
    w = 2 * np.pi * np.linspace(0, 30e6, 301)
    for y in (0.5, 1.0, 1.05, 1.5, 2.0, 2.7):
        for w0 in (0.0, 2 * np.pi * 1e6):
            for a_db in (0.0, 0.75):
                put(keng, pyp_w=col(w), pyp_a=a_db, pyp_y=y, pyp_w0=w0)
                keng.eval(
                    "pyp_a0 = db2neper(pyp_a, pyp_y);"
                    "pyp_c = powerLawKramersKronig(pyp_w, pyp_w0, 1540, pyp_a0, pyp_y);",
                    nargout=0,
                )
                a0, c = get(keng, "pyp_a0", "pyp_c")
                assert abs(hom.db2neper(a_db, y) - a0[0, 0]) <= REL * abs(a0[0, 0])
                c_py = hom.power_law_kramers_kronig(w, w0, 1540.0, float(hom.db2neper(a_db, y)), y)
                fin = np.isfinite(c[:, 0])
                np.testing.assert_array_equal(np.isfinite(c_py), fin)
                np.testing.assert_allclose(c_py[fin], c[fin, 0], rtol=REL, atol=0)
    keng.eval("clear pyp_w pyp_a pyp_y pyp_w0 pyp_a0 pyp_c", nargout=0)


# ---------------------------------------------------------------- run_simulation_homogeneous


def matlab_run(
    eng: Any, run_param: dict, kgrid: dict, medium: dict, source: dict, sensor: dict
) -> np.ndarray:
    put(eng, pyp_spts=source["points"], pyp_ms=source["mass_source"], pyp_sen=sensor["points"])
    put(eng, pyp_dt=kgrid["dt"], pyp_nt=float(kgrid["Nt"]))
    for k in ("density", "sound_speed", "alpha_coeff", "alpha_power"):
        put(eng, **{f"pyp_{k}": np.asarray(medium[k], dtype=float)})
    rp = ", ".join(
        f"'{k}', {v!r}" if isinstance(v, str) else f"'{k}', {float(v)!r}"
        for k, v in run_param.items()
    )
    eng.eval(
        "pyp_med = struct('density', pyp_density, 'sound_speed', pyp_sound_speed,"
        " 'alpha_coeff', pyp_alpha_coeff, 'alpha_power', pyp_alpha_power);"
        + (f"pyp_med.alpha_mode = '{medium['alpha_mode']}';" if "alpha_mode" in medium else "")
        + f"pyp_rp = struct({rp}); pyp_rp.gridded = logical(pyp_rp.gridded);"
        "pyp_kg = struct('dt', pyp_dt, 'Nt', pyp_nt);"
        "evalc(['pyp_out = run_simulation_homogeneous(pyp_rp, pyp_kg,"
        " pyp_med, struct(''points'', pyp_spts, ''mass_source'', pyp_ms),"
        " struct(''points'', pyp_sen));']); pyp_out = pyp_out.p;",
        nargout=0,
    )
    out = get(eng, "pyp_out")
    eng.eval(
        "clear pyp_spts pyp_ms pyp_sen pyp_dt pyp_nt pyp_density pyp_sound_speed"
        " pyp_alpha_coeff pyp_alpha_power pyp_med pyp_rp pyp_kg pyp_out",
        nargout=0,
    )
    return out


def linear_array_points(n_el: int, pitch: float, n_z: int, height: float) -> np.ndarray:
    """Sensor points on a linear array face (x = 0): elements along y, points along z."""
    y = (np.arange(n_el) - (n_el - 1) / 2) * pitch
    z = (np.arange(n_z) - (n_z - 1) / 2) * height / max(n_z - 1, 1)
    Y, Z = np.meshgrid(y, z, indexing="ij")
    return np.column_stack([np.zeros(Y.size), Y.ravel(), Z.ravel()])


def curved_array_points(n_el: int, radius: float, pitch: float) -> np.ndarray:
    """Sensor points on a convex arc (centre at x = -radius)."""
    theta = (np.arange(n_el) - (n_el - 1) / 2) * pitch / radius
    return np.column_stack(
        [radius * np.cos(theta) - radius, radius * np.sin(theta), np.zeros(n_el)]
    )


def medium_maps(c: float, a: float, b: float, seed: int, shape=(3, 4, 5)) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "density": 1000 + 10 * rng.random(shape),
        "sound_speed": c + 5 * rng.random(shape),
        "alpha_coeff": a * np.ones(shape),
        "alpha_power": b,
    }


ARRAYS = {
    "linear_odd": lambda: linear_array_points(7, 3e-4, 3, 4e-3),
    "curved_odd": lambda: curved_array_points(9, 40e-3, 5e-4),
}


@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["double", "single"])
@pytest.mark.parametrize("max_elements", [3000, 2**26], ids=["chunked", "default"])
@pytest.mark.parametrize("array", list(ARRAYS), ids=list(ARRAYS))
def test_run_simulation_homogeneous(keng: Any, array: str, max_elements: int, dtype: Any) -> None:
    """Sensors on linear/curved arrays; bubbles on and off axis, one near the array."""
    rng = np.random.default_rng(1)
    sen = ARRAYS[array]()
    n_src, nt = 5, 240
    src = np.column_stack(
        [
            rng.uniform(2e-3, 6e-3, n_src),
            rng.uniform(-3e-3, 3e-3, n_src),
            rng.uniform(-1e-3, 1e-3, n_src),
        ]
    )
    src[0] = [2e-4, 1e-5, 0.0]  # next to the array
    src[1, 1] = 0.0  # on axis
    ms = gauss_sources(n_src, 180).astype(dtype)
    medium = medium_maps(1540.0, 0.75, 1.5, seed=2)
    run_param = {
        "DATA_CAST": "single" if dtype == np.float32 else "double",
        "gridded": 0.0,
        "MaxArrayElements": max_elements,
    }
    kgrid = {"dt": DT, "Nt": nt}
    source = {"points": src, "mass_source": ms}
    expected = matlab_run(keng, run_param, kgrid, medium, source, {"points": sen})
    actual = hom.run_simulation_homogeneous(run_param, kgrid, medium, source, {"points": sen})["p"]
    assert actual.dtype == dtype
    assert rel_err(actual, expected) <= (REL if dtype == np.float64 else SINGLE_REL)


@pytest.mark.parametrize("dtype", [np.float64, np.float32], ids=["double", "single"])
def test_run_simulation_homogeneous_gridded(keng: Any, dtype: Any) -> None:
    """Gridded distances (transducer sensors in ``hybrid_simulator``), one sensor on a source."""
    rng = np.random.default_rng(4)
    sen = linear_array_points(11, 2e-4, 5, 3e-3)
    n_src, nt = 4, 300
    src = np.column_stack(
        [
            rng.uniform(1e-3, 5e-3, n_src),
            rng.uniform(-2e-3, 2e-3, n_src),
            rng.uniform(-1e-3, 1e-3, n_src),
        ]
    )
    src[2] = sen[7]  # a bubble on a sensor point: that sensor gets nothing from it
    ms = gauss_sources(n_src, nt).astype(dtype)
    medium = medium_maps(1540.0, 0.5, 1.3, seed=6)
    run_param = {
        "DATA_CAST": "single" if dtype == np.float32 else "double",
        "gridded": 1.0,
        "MaxArrayElements": 2000.0,
    }
    kgrid = {"dt": DT, "Nt": nt}
    source = {"points": src, "mass_source": ms}
    expected = matlab_run(keng, run_param, kgrid, medium, source, {"points": sen})
    actual = hom.run_simulation_homogeneous(run_param, kgrid, medium, source, {"points": sen})["p"]
    assert rel_err(actual, expected) <= (REL if dtype == np.float64 else SINGLE_REL)


def test_run_simulation_homogeneous_bubbles_no_dispersion(
    keng: Any, bubble_mass_source: dict
) -> None:
    """MATLAB bubble responses, bubble to bubble (``hybrid_simulator``), ``no_dispersion``."""
    ms = bubble_mass_source["mass_source"]
    dt, c = bubble_mass_source["dt"], bubble_mass_source["c"]
    rng = np.random.default_rng(9)
    pts = rng.uniform(-1e-3, 1e-3, (ms.shape[0], 3)) + np.array([4e-3, 0, 0])
    medium = medium_maps(c, 0.6, 1.05, seed=8)
    medium["alpha_mode"] = "no_dispersion"
    run_param = {"DATA_CAST": "double", "gridded": 0.0}
    kgrid = {"dt": dt, "Nt": ms.shape[1] + 20}
    source = {"points": pts, "mass_source": ms}
    expected = matlab_run(keng, run_param, kgrid, medium, source, {"points": pts})
    actual = hom.run_simulation_homogeneous(run_param, kgrid, medium, source, {"points": pts})["p"]
    assert rel_err(actual, expected) <= REL


def test_gridded_distance_matches_matlab(keng: Any) -> None:
    """Grid and indices of ``gridded_distance`` (subfunction, reproduced through its effect)."""
    rng = np.random.default_rng(12)
    for n in (1, 2, 17, 400):
        d = rng.uniform(1e-3, 9e-3, n)
        dx = 1540 * DT
        d_grid, i_grid = hom.gridded_distance(d, dx)
        # MATLAB's expression, evaluated in MATLAB (the subfunction is private to the file)
        put(keng, pyp_d=col(d), pyp_dx=dx)
        keng.eval(
            "pyp_i = round((pyp_d - mean(pyp_d))/pyp_dx);"
            "pyp_g = transpose((min(pyp_i):max(pyp_i))*pyp_dx + mean(pyp_d));"
            "pyp_i = pyp_i - min(pyp_i) + 1;",
            nargout=0,
        )
        g, i = get(keng, "pyp_g", "pyp_i")
        np.testing.assert_array_equal(i_grid, i[:, 0].astype(int) - 1)
        np.testing.assert_allclose(d_grid, g.reshape(-1), rtol=REL)
    keng.eval("clear pyp_d pyp_dx pyp_i pyp_g", nargout=0)


def test_average_medium(keng: Any) -> None:
    """Mean of the medium maps (homogeneous approximation), through the propagated field."""
    medium = medium_maps(1500.0, 0.4, 1.7, seed=13, shape=(2, 3, 4))
    medium["alpha_power"] = 1.7 * np.ones((2, 3, 4))
    avg = hom.average_medium(medium)
    assert avg["c"] == pytest.approx(float(np.mean(medium["sound_speed"])), rel=1e-15)
    sen = np.array([[3e-3, 0, 0], [4e-3, 1e-3, 0]])  # within Nt*dt*c0 (MATLAB FFT length)
    src = np.array([[0.0, 0, 0]])
    source = {"points": src, "mass_source": gauss_sources(1, 128)}
    run_param = {"DATA_CAST": "double", "gridded": 0.0}
    expected = matlab_run(keng, run_param, {"dt": DT, "Nt": 128}, medium, source, {"points": sen})
    actual = hom.run_simulation_homogeneous(
        run_param, {"dt": DT, "Nt": 128}, medium, source, {"points": sen}
    )["p"]
    assert rel_err(actual, expected) <= REL
