"""k-Wave signal tools vs MATLAB k-Wave 1.4.x: ``filterTimeSeries``, ``getWin``, ``kgrid.k_max``.

``pyproteus.backends.kwave.filter_time_series`` / ``get_win`` wrap
k-wave-python; MATLAB k-Wave (LGPL) is only called through the engine. The
MATLAB calls are the ones PROTEUS makes (``preprocess_transmit.m``,
``compute_bubble_mass_source.m``, ``get_transducer_integration_points.m``).
Comparison class: float, 1e-9 rel (ROADMAP §4, D-13). No k-Wave binary needed.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import pytest

from pyproteus.backends import kwave as kw
from tests.oracle.helpers import from_matlab, put, row

pytest.importorskip("kwave")

KWAVE_MATLAB = Path(
    os.environ.get(
        "KWAVE_MATLAB_PATH",
        "/Users/theodoremichel/Repos/TWENTE/Ultrasound/PROTEUS-cleaning/PROTEUS/external/k-wave/k-Wave",
    )
)
REL = 1e-9


@pytest.fixture(scope="module")
def keng(matlab_engine):
    """Engine with MATLAB k-Wave on the path (the shared fixture does not add it)."""
    if not (KWAVE_MATLAB / "filterTimeSeries.m").is_file():
        pytest.skip(f"MATLAB k-Wave not found at {KWAVE_MATLAB} (set KWAVE_MATLAB_PATH)")
    matlab_engine.addpath(str(KWAVE_MATLAB), nargout=0)
    yield matlab_engine
    matlab_engine.rmpath(str(KWAVE_MATLAB), nargout=0)


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


# Grids as PROTEUS builds them: odd/even sizes, isotropic and not.
GRIDS = [
    ((128, 96, 64), (5e-5, 5e-5, 5e-5)),
    ((37, 29, 23), (1e-4, 1e-4, 1e-4)),
    ((40, 33, 21), (6e-5, 7e-5, 5e-5)),
    ((25, 64, 31), (1e-4, 5e-5, 8e-5)),
]


@pytest.mark.parametrize(("shape", "spacing"), GRIDS)
def test_kgrid_k_max_matches_matlab(keng, shape, spacing) -> None:
    keng.eval(
        "kg = kWaveGrid({}, {!r}, {}, {!r}, {}, {!r}); km = kg.k_max;".format(
            *[v for pair in zip(shape, spacing, strict=True) for v in pair]
        ),
        nargout=0,
        stdout=io.StringIO(),
    )
    km = float(keng.workspace["km"])
    assert kw.KGrid(shape, spacing, 1e-8, 10).k_max == pytest.approx(km, rel=REL)


def _signals(rng: np.random.Generator) -> dict[str, np.ndarray]:
    t = np.arange(1001) / 250e6
    burst = np.sin(2 * np.pi * 5e6 * t) * np.exp(-(((t - 1e-6) / 3e-7) ** 2))
    chirp = np.sin(2 * np.pi * (2e6 + 6e12 * t) * t) * np.hanning(t.size)
    return {
        "burst": burst,  # transmit pulse at Transmit.SamplingRate = 250 MHz
        "chirp_even": chirp[:800],
        "noise_odd": rng.standard_normal(777),  # broadband, like a mass source
        "short": burst[230:281],  # shorter than the filter order
        "step": np.r_[np.zeros(100), np.ones(203)],
    }


# (fs, kgrid dt, c_min, grid) as in PROTEUS: fs = 250 MHz, TW = 0.1/(fs*dt)
SETUPS = [
    (250e6, 2e-8, 1480.0, GRIDS[0]),
    (250e6, 1.2345e-8, 1450.5, GRIDS[1]),
    (250e6, 4e-9, 1500.0, GRIDS[2]),  # fs*dt = 1 -> TW = 0.1 (default width)
    (100e6, 3.3e-8, 1520.0, GRIDS[3]),
]


@pytest.mark.parametrize("setup", SETUPS, ids=lambda s: f"fs{s[0]:.0e}_dt{s[1]:.3e}")
def test_filter_time_series_proteus_call_matches_matlab(keng, setup) -> None:
    """PROTEUS' call.

    ``filterTimeSeries(F, F, x, 'ZeroPhase', true, 'TransitionWidth', TW, 'PPW', 2)``.
    """
    fs, dt, c_min, (shape, spacing) = setup
    k_max = kw.KGrid(shape, spacing, dt, 10).k_max
    tw = 0.1 / (fs * dt)
    for name, x in _signals(np.random.default_rng(int(dt * 1e12))).items():
        put(keng, x=row(x), fdt=1 / fs, kmax=k_max, cmin=c_min, tw=tw)
        keng.eval(
            "Filter.dt = fdt; Filter.t_array = []; Filter.k_max = kmax; "
            "Filter.TW = tw; Filter.sound_speed = cmin; "
            "y = filterTimeSeries(Filter, Filter, x, 'ZeroPhase', true, "
            "'TransitionWidth', Filter.TW, 'PPW', 2);",
            nargout=0,
            stdout=io.StringIO(),
        )
        y_mat = from_matlab(keng.workspace["y"]).ravel()
        y = kw.filter_time_series(
            1 / fs, k_max, c_min, x, zero_phase=True, transition_width=tw, ppw=2
        )
        assert y.shape == x.shape
        assert _rel(y, y_mat) <= REL, name
        # row and column vectors keep their orientation, as in MATLAB
        y_col = kw.filter_time_series(
            1 / fs, k_max, c_min, x[:, None], zero_phase=True, transition_width=tw, ppw=2
        )
        assert y_col.shape == (x.size, 1)
        assert np.array_equal(y_col.ravel(), y)


@pytest.mark.parametrize(
    "opts",
    [
        dict(),  # defaults: causal, PPW 3, TransitionWidth 0.1, StopBandAtten 60
        dict(zero_phase=True),
        dict(ppw=4, stop_band_atten=80, transition_width=0.05),
        dict(zero_phase=True, ppw=2.5, stop_band_atten=45, transition_width=0.2),
    ],
)
def test_filter_time_series_options_match_matlab(keng, opts) -> None:
    fs, dt, c_min = 250e6, 1.5e-8, 1490.0
    shape, spacing = GRIDS[1]
    k_max = kw.KGrid(shape, spacing, dt, 10).k_max
    x = _signals(np.random.default_rng(3))["noise_odd"]
    names = {
        "zero_phase": "ZeroPhase",
        "ppw": "PPW",
        "stop_band_atten": "StopBandAtten",
        "transition_width": "TransitionWidth",
    }
    args = ", ".join(
        f"'{names[k]}', {str(v).lower() if isinstance(v, bool) else repr(v)}"
        for k, v in opts.items()
    )
    put(keng, x=row(x), fdt=1 / fs, kmax=k_max, cmin=c_min)
    keng.eval(
        "F.dt = fdt; F.t_array = []; F.k_max = kmax; F.sound_speed = cmin; "
        f"y = filterTimeSeries(F, F, x{', ' + args if args else ''});",
        nargout=0,
        stdout=io.StringIO(),
    )
    y_mat = from_matlab(keng.workspace["y"]).ravel()
    y = kw.filter_time_series(1 / fs, k_max, c_min, x, **opts)
    assert _rel(y, y_mat) <= REL


@pytest.mark.parametrize("n", [1, 2, 5, 8, 9, 16, 33])
@pytest.mark.parametrize("param", [0.5, 0.0, 1.0, 0.25])
def test_get_win_tukey_matches_matlab(keng, n, param) -> None:
    """``getWin(N, 'Tukey', 'Param', 0.5, 'Plot', false)`` (get_transducer_integration_points.m)."""
    put(keng, n=float(n), prm=param)
    keng.eval(
        "[w, cg] = getWin(n, 'Tukey', 'Param', prm, 'Plot', false);",
        nargout=0,
        stdout=io.StringIO(),
    )
    w_mat = from_matlab(keng.workspace["w"])
    cg_mat = float(keng.workspace["cg"])
    with np.errstate(invalid="ignore", divide="ignore"):  # Param 0: NaN on both sides
        w, cg = kw.get_win(n, "Tukey", param=param)
    assert w.shape == w_mat.shape
    np.testing.assert_allclose(w, w_mat, rtol=REL, atol=1e-15)  # NaN positions must agree
    np.testing.assert_allclose(cg, cg_mat, rtol=REL, atol=1e-15)
