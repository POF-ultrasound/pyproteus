"""Homogeneous propagation on its own terms.

Analytical Green's function, attenuation, dispersion, batching, gridding.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyproteus.acoustic import homogeneous as hom

DT = 1 / 50e6
F0 = 3e6
SIGMA = 3e-7


def pulse(t: np.ndarray, t0: float = 2e-6) -> np.ndarray:
    """Gaussian-modulated sine mass source [kg/s] (band-limited well below Nyquist)."""
    u = t - t0
    return np.sin(2 * np.pi * F0 * u) * np.exp(-((u / SIGMA) ** 2))


def pulse_dot(t: np.ndarray, t0: float = 2e-6) -> np.ndarray:
    u = t - t0
    g = np.exp(-((u / SIGMA) ** 2))
    return (
        2 * np.pi * F0 * np.cos(2 * np.pi * F0 * u) - 2 * u / SIGMA**2 * np.sin(2 * np.pi * F0 * u)
    ) * g


LOSSLESS = {"c": 1540.0, "a": 0.0, "b": 1.5}


def test_lossless_free_field_green_function() -> None:
    """``p(r,t) = m'(t - r/c)/(4 pi r)`` (delays off the sample grid)."""
    nt = 600
    t = np.arange(nt) * DT
    r = np.array([[1.0e-3], [2.345e-3], [4.4321e-3]])
    p = hom.calc_scatter_attenuated(pulse(t), LOSSLESS, {"dt": DT, "Nt": nt}, r)[:, :, 0]
    exact = pulse_dot(t[None, :] - r / LOSSLESS["c"]) / (4 * np.pi * r)
    err = np.max(np.abs(p - exact), axis=1) / np.max(np.abs(exact), axis=1)
    assert np.all(err < 1e-9)


def test_attenuation_spectrum_and_kramers_kronig_phase() -> None:
    """Transfer function between two distances: ``exp(-(r2-r1)(alpha(f) + 2 pi i f/c(f)))``."""
    med = {"c": 1500.0, "a": 0.75, "b": 1.5}
    nt = 2000
    t = np.arange(nt) * DT
    r1, r2 = 2e-3, 6e-3
    p = hom.calc_scatter_attenuated(pulse(t), med, {"dt": DT, "Nt": nt}, np.array([[r1], [r2]]))[
        :, :, 0
    ]
    f = np.fft.rfftfreq(2 * nt, DT)
    P = np.fft.rfft(p, n=2 * nt, axis=1)
    band = (f > 1.5e6) & (f < 4.5e6)
    H = (P[1, band] * r2) / (P[0, band] * r1)
    a_np = 0.75 * math.log(10) * 100 / 20
    alpha = a_np * (f[band] / 1e6) ** 1.5
    c = hom.power_law_kramers_kronig(
        2 * np.pi * f[band], 0.0, 1500.0, float(hom.db2neper(0.75, 1.5)), 1.5
    )
    expected = np.exp(-(r2 - r1) * (alpha + 2j * np.pi * f[band] / c))
    np.testing.assert_allclose(H, expected, rtol=1e-6)
    assert np.all(np.diff(c) > 0)  # 1 < y < 3: phase speed increases with frequency


def test_no_dispersion_uses_c0() -> None:
    nt = 1000
    t = np.arange(nt) * DT
    med = {"c": 1540.0, "a": 0.5, "b": 1.05, "alpha_mode": "no_dispersion"}
    r = np.array([[3e-3], [5e-3]])
    p = hom.calc_scatter_attenuated(pulse(t), med, {"dt": DT, "Nt": nt}, r)[:, :, 0]
    P = np.fft.rfft(p, n=2 * nt, axis=1)
    f = np.fft.rfftfreq(2 * nt, DT)
    k = np.argmin(np.abs(f - F0))
    phase = np.angle(P[1, k] / P[0, k] * np.exp(2j * np.pi * f[k] * 2e-3 / 1540.0))
    assert abs(phase) < 1e-9


def test_kramers_kronig_reference_and_cases() -> None:
    w = 2 * np.pi * np.array([0.5e6, 1e6, 5e6])
    w0 = 2 * np.pi * 1e6
    for y in (0.5, 1.0, 1.5, 2.0):
        c = hom.power_law_kramers_kronig(w, w0, 1540.0, 1e-5, y)
        assert c[1] == pytest.approx(1540.0, rel=1e-15)
    np.testing.assert_array_equal(hom.power_law_kramers_kronig(w, w0, 1540.0, 1e-5, 3.5), 1540.0)
    # y = 2: tan(pi) = 0 up to rounding -> no dispersion
    np.testing.assert_allclose(
        hom.power_law_kramers_kronig(w, 0.0, 1540.0, 1e-5, 2.0), 1540.0, rtol=1e-9
    )


def test_db2neper() -> None:
    # 1 dB/(MHz cm) = 100/8.686 Np/(MHz m); per (rad/s): / (2 pi 1e6)
    assert hom.db2neper(1.0, 1.0) == pytest.approx(
        100 / (20 * math.log10(math.e)) / (2 * math.pi * 1e6), rel=1e-15
    )
    assert hom.db2neper(0.0, 1.3) == 0.0


@pytest.mark.parametrize("b", [0.5, 1.0])
def test_singular_dispersion_warns_and_gives_nan(b: float) -> None:
    med = {"c": 1540.0, "a": 0.5, "b": b}
    with pytest.warns(UserWarning, match="Q-130"):
        p = hom.calc_scatter_attenuated(
            np.ones((1, 16)), med, {"dt": DT, "Nt": 16}, np.array([[1e-3]])
        )
    assert np.isnan(p).all()


def test_zero_distance_is_silent_and_lengths() -> None:
    ms = np.vstack([pulse(np.arange(50) * DT), pulse(np.arange(50) * DT)])
    r = np.array([[0.0, 1e-3], [2e-3, 0.0]])
    for nt in (31, 50, 77):  # crop, equal, pad
        p = hom.calc_scatter_attenuated(ms, LOSSLESS, {"dt": DT, "Nt": nt}, r)
        assert p.shape == (2, nt, 2)
        assert not p[0, :, 0].any() and not p[1, :, 1].any()
        assert np.isfinite(p).all()


def test_dtype_follows_mass_source() -> None:
    ms = pulse(np.arange(64) * DT)
    r = np.array([[1e-3]])
    assert (
        hom.calc_scatter_attenuated(ms.astype(np.float32), LOSSLESS, {"dt": DT, "Nt": 64}, r).dtype
        == np.float32
    )
    assert hom.calc_scatter_attenuated(ms, LOSSLESS, {"dt": DT, "Nt": 64}, r).dtype == np.float64


def test_arrival_after_extended_window_wraps() -> None:
    """Q-131: delays beyond ``2*Nt*dt`` wrap around (periodic FFT), as in PROTEUS."""
    nt = 400
    t = np.arange(nt) * DT
    c = LOSSLESS["c"]
    tau = 2 * nt * DT + 1.5e-6  # wraps to 1.5 us
    r = np.array([[tau * c], [1.5 * nt * DT * c]])  # wrapped; arrives after the window
    p = hom.calc_scatter_attenuated(pulse(t), LOSSLESS, {"dt": DT, "Nt": nt}, r)[:, :, 0]
    exact = pulse_dot(t - 1.5e-6) / (4 * np.pi * r[0, 0])
    assert np.max(np.abs(p[0] - exact)) < 1e-6 * np.max(np.abs(exact))
    assert np.max(np.abs(p[1])) < 1e-6 * np.max(np.abs(exact))  # cropped


def problem(seed: int = 0, n_src: int = 5, n_sen: int = 17, nt: int = 150):
    rng = np.random.default_rng(seed)
    t = np.arange(nt) * DT
    ms = pulse(t)[None, :] * rng.uniform(0.5, 2, (n_src, 1))
    src = 1e-3 * rng.random((n_src, 3))
    sen = 1e-3 * (rng.random((n_sen, 3)) + np.array([3, 0, 0]))
    return ms, src, sen, nt


def test_propagate_sums_calc_scatter_and_is_batch_independent() -> None:
    ms, src, sen, nt = problem()
    med = {"c": 1540.0, "a": 0.75, "b": 1.5}
    r = np.linalg.norm(sen[:, None] - src[None], axis=2)
    ref = hom.calc_scatter_attenuated(ms, med, {"dt": DT, "Nt": nt}, r).sum(axis=2)
    for max_el in (1, 500, 5000, 60000, 2**26):  # one-by-one, chunks, batches, all at once
        p = hom.propagate_homogeneous(
            ms, src, sen, dt=DT, nt=nt, medium=med, max_array_elements=max_el
        )
        assert np.max(np.abs(p - ref)) <= 1e-12 * np.max(np.abs(ref))


def test_gridded_mode() -> None:
    ms, src, sen, nt = problem(1)
    sen[4] = src[2]  # coincident sensor
    med = {"c": 1540.0, "a": 0.75, "b": 1.5}
    exact = hom.propagate_homogeneous(ms, src, sen, dt=DT, nt=nt, medium=med)
    g1 = hom.propagate_homogeneous(ms, src, sen, dt=DT, nt=nt, medium=med, gridded=True)
    g2 = hom.propagate_homogeneous(
        ms, src, sen, dt=DT, nt=nt, medium=med, gridded=True, max_array_elements=300
    )
    np.testing.assert_array_equal(g1, g2)
    # sensor 4 does not hear the source on it
    only2 = hom.propagate_homogeneous(
        ms[[2]], src[[2]], sen, dt=DT, nt=nt, medium=med, gridded=True
    )
    assert not only2[4].any() and only2[3].any()
    # snapping by at most half a sample of path: the field stays close to the exact one
    assert np.max(np.abs(g1 - exact)) < 0.5 * np.max(np.abs(exact))


def test_gridded_distance() -> None:
    d = np.array([1.0, 1.24, 1.26, 2.0])
    g, i = hom.gridded_distance(d, 0.5)
    np.testing.assert_allclose(g[i], [0.875, 1.375, 1.375, 1.875])  # mean 1.375
    assert i.min() == 0 and g.size == i.max() + 1
    g, i = hom.gridded_distance([3.0], 0.1)
    np.testing.assert_array_equal(g, [3.0])
    np.testing.assert_array_equal(i, [0])
    # halves round away from zero (MATLAB round)
    g, i = hom.gridded_distance([0.0, 1.0, 2.0, 3.0], 1.0)  # (d - 1.5) = -1.5, -0.5, 0.5, 1.5
    np.testing.assert_array_equal(i, [0, 1, 3, 4])  # round(+-0.5) = +-1, round(+-1.5) = +-2
    np.testing.assert_allclose(g, [-0.5, 0.5, 1.5, 2.5, 3.5])


def test_run_simulation_homogeneous_struct_api() -> None:
    ms, src, sen, nt = problem(2)
    medium = {
        "density": 1000 * np.ones((2, 2, 2)),
        "sound_speed": np.full((2, 2, 2), 1540.0),
        "alpha_coeff": 0.75 * np.ones((2, 2, 2)),
        "alpha_power": 1.5,
    }
    out = hom.run_simulation_homogeneous(
        {"DATA_CAST": "single", "gridded": False},
        {"dt": DT, "Nt": nt},
        medium,
        {"points": src, "mass_source": ms.astype(np.float32)},
        {"points": sen},
    )
    ref = hom.propagate_homogeneous(ms, src, sen, dt=DT, nt=nt, medium=hom.average_medium(medium))
    assert out["p"].dtype == np.float32 and out["p"].shape == (len(sen), nt)
    assert np.max(np.abs(out["p"] - ref)) <= 1e-6 * np.max(np.abs(ref))
    assert (
        hom.average_medium({**medium, "alpha_mode": "no_dispersion"})["alpha_mode"]
        == "no_dispersion"
    )


def test_gpu_cast_without_cupy() -> None:
    pytest.importorskip("numpy")
    try:
        import cupy  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="CuPy"):
            hom.array_module("gpuArray-single")
    assert hom.array_module("single") is np
    assert hom.array_module(None) is np


def test_memory_is_bounded_by_output_and_budget() -> None:
    """Single precision: peak ≈ float32 output + inputs + O(max_array_elements), no float64 copy."""
    import tracemalloc

    rng = np.random.default_rng(3)
    n_sen, n_src, nt = 4000, 6, 1000
    ms = rng.standard_normal((n_src, nt)).astype(np.float32)
    src = 1e-3 * rng.random((n_src, 3))
    sen = 1e-2 * rng.random((n_sen, 3))
    out_bytes = n_sen * nt * 4
    budget = 2**18
    for gridded in (False, True):
        tracemalloc.start()
        p = hom.propagate_homogeneous(
            ms,
            src,
            sen,
            dt=DT,
            nt=nt,
            medium=LOSSLESS,
            gridded=gridded,
            max_array_elements=budget,
            dtype=np.float32,
        )
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        assert p.dtype == np.float32
        # float32 output + two budget-sized complex128 blocks + small change; a float64
        # output (measured 35-40 MB here) or a full-size temporary would exceed it
        assert peak < out_bytes + 2 * 16 * budget + 6_000_000, (gridded, peak)
