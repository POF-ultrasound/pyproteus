"""Linear sample interpolation in the DAS matrices (D-20).

Not compared with MATLAB (PROTEUS has nearest-sample DAS only): checked
against analytical references, i.e. the continuous RF signal evaluated at the
exact delays.
"""

from __future__ import annotations

import numpy as np
import pytest

from pyproteus.reconstruction import compute_das_matrix, compute_das_matrix_general

C, FS, F0 = 1540.0, 40e6, 2.5e6
SIGMA = 0.4e-6


def pulse(t):
    """Gaussian-modulated cosine (continuous RF; peak 1 at t = 0)."""
    return np.cos(2 * np.pi * F0 * t) * np.exp(-((t / SIGMA) ** 2))


# Derivative bounds of the pulse (|f'| <= w*(1+..), |f''| <= w^2*(1+..)):
W = 2 * np.pi * F0
D1 = W + 2 / SIGMA  # crude upper bounds, valid for all t
D2 = (W + 2 / SIGMA) ** 2 + 2 / SIGMA**2


def single_element(x, t, apodization="binary"):
    recon = dict(x=x, y=np.array([0.0]), c=C, t=t, Fs=FS, alpha_th=np.pi / 3,
                 apodization=apodization)  # fmt: skip
    tr = dict(x_el=np.zeros(1), y_el=np.zeros(1), azimuth=np.zeros(1))
    return recon, tr


def test_linear_is_an_interpolator_with_textbook_error() -> None:
    """One element, pixels on axis: M @ rf is the RF interpolated at t_del = 2x/c."""
    rng = np.random.default_rng(0)
    t = np.arange(800) / FS - 2e-6
    x = np.sort(rng.uniform(0.5e-3, 12e-3, 400), kind="stable")
    recon, tr = single_element(x, t)
    t_del = 2 * x / C
    rf = pulse(t - 8e-6)  # pulse centred inside the record
    exact = pulse(t_del - 8e-6)
    lin = compute_das_matrix_general(recon, tr, {}, interpolation="linear") @ rf
    near = compute_das_matrix_general(recon, tr, {}) @ rf
    dt = 1 / FS
    err_lin = np.max(np.abs(lin - exact))
    err_near = np.max(np.abs(near - exact))
    assert err_lin <= D2 * dt**2 / 8  # linear interpolation bound
    assert err_near <= D1 * dt / 2
    assert err_lin < err_near / 4


def test_linear_equals_nearest_on_sample_delays() -> None:
    """c = Fs = 1, integer delays: linear has one entry per pair, equal to nearest."""
    t = np.arange(12, dtype=float)
    x = np.arange(0, 6) * 1.0  # t_del = 2x: integer samples
    recon = dict(x=x, y=np.array([0.0]), c=1.0, t=t, Fs=1.0, alpha_th=np.pi / 3,
                 apodization="binary")  # fmt: skip
    tr = dict(x_el=np.zeros(1), y_el=np.zeros(1), azimuth=np.zeros(1))
    A = compute_das_matrix_general(recon, tr, {}, interpolation="linear")
    B = compute_das_matrix_general(recon, tr, {})
    assert (A != B).nnz == 0
    A = compute_das_matrix(t, [0.0], x, [0.0], 1.0, 1.0, np.inf, interpolation="linear")
    B = compute_das_matrix(t, [0.0], x, [0.0], 1.0, 1.0, np.inf)
    assert (A != B).nnz == 0


def test_linear_weights_split_and_stay_in_the_element() -> None:
    rng = np.random.default_rng(1)
    nelem, Nt = 6, 300
    y_el = (np.arange(nelem) - 2.5) * 0.3e-3
    recon = dict(x=np.linspace(0.5e-3, 6e-3, 17), y=np.linspace(-2e-3, 2e-3, 13), c=C,
                 t=np.arange(Nt) / FS + rng.uniform(0, 1e-6), Fs=FS, alpha_th=np.pi / 3,
                 apodization="cosine")  # fmt: skip
    tr = dict(x_el=np.zeros(nelem), y_el=y_el, azimuth=np.zeros(nelem))
    L = compute_das_matrix_general(recon, tr, {"steer": 0.1}, interpolation="linear").tocoo()
    N = compute_das_matrix_general(recon, tr, {"steer": 0.1}).tocoo()
    el = L.col // Nt
    # Per (pixel, element): at most two adjacent samples, weights summing to
    # the cosine apodization of that pair (nearest's weight where both exist).
    key = L.row * nelem + el
    sums = np.bincount(key, weights=L.data, minlength=L.shape[0] * nelem)
    nkey = N.row * nelem + N.col // Nt
    both = np.isin(nkey, key)
    np.testing.assert_allclose(sums[nkey[both]], N.data[both], rtol=1e-14)
    counts = np.bincount(key)
    assert counts.max() <= 2
    for k in np.flatnonzero(counts == 2):
        cols = np.sort(L.col[key == k], kind="stable")
        assert cols[1] - cols[0] == 1 and cols[0] // Nt == cols[1] // Nt
    assert np.all(L.data > 0)


def test_linear_record_edges() -> None:
    """Pairs need both samples inside the record: 0 <= s <= Nt-1."""
    t = np.arange(5, dtype=float)
    x = np.array([-0.25, 0.0, 1.0, 1.75, 2.0, 2.25])  # plane wave, c = Fs = 1: s = |x| + x
    M = compute_das_matrix(t, [0.0], x, [0.0], 1.0, 1.0, np.inf, interpolation="linear")
    dense = M.toarray()
    # x = -0.25: X < 0 keeps the element (Q-102) but s = |x| + x = 0 -> sample 0.
    np.testing.assert_array_equal(dense[0], [1, 0, 0, 0, 0])
    np.testing.assert_array_equal(dense[1], [1, 0, 0, 0, 0])
    np.testing.assert_array_equal(dense[2], [0, 0, 1, 0, 0])
    np.testing.assert_allclose(dense[3], [0, 0, 0, 0.5, 0.5])  # s = 3.5
    np.testing.assert_array_equal(dense[4], [0, 0, 0, 0, 1])  # s = Nt-1 exactly
    np.testing.assert_array_equal(dense[5], [0, 0, 0, 0, 0])  # s = 4.5 > Nt-1


def test_linear_psf_closer_to_continuous_das() -> None:
    """Point scatterer, 32 elements, steered plane wave: the linear image is much
    closer to DAS on the continuous RF (exact delays) than the nearest image."""
    steer = 0.12
    y_el = (np.arange(32) - 15.5) * 0.3e-3
    tr = dict(x_el=np.zeros(32), y_el=y_el, azimuth=np.zeros(32))
    S = np.array([10e-3, 1.3e-3])
    t_tx = (S[1] * np.sin(steer) + S[0] * np.cos(steer) - np.min(y_el * np.sin(steer))) / C
    t_echo = t_tx + np.hypot(S[0], S[1] - y_el) / C
    t = np.arange(int(1.6 * t_echo.max() * FS)) / FS
    rf = pulse(t[None, :] - t_echo[:, None])  # Nelem x Nt
    pix = 0.02e-3
    x = S[0] + np.arange(-30, 31) * pix
    y = S[1] + np.arange(-25, 26) * pix
    recon = dict(x=x, y=y, c=C, t=t, Fs=FS, alpha_th=np.pi / 3, apodization="cosine")
    tx = {"focus": np.inf, "steer": steer}

    # Reference: same weights, RF evaluated at the exact delays.
    from pyproteus.reconstruction import das_transmit_distance

    X, Y = np.meshgrid(x, y, indexing="ij")
    Xf, Yf = X.ravel(order="F")[:, None], Y.ravel(order="F")[:, None]
    d_in = das_transmit_distance(Xf, Yf, dict(tr, center=-np.inf), tx)
    t_del = (np.hypot(Yf - y_el, Xf) + d_in) / C
    alpha = np.arctan2(Yf - y_el, Xf)
    v = np.where(np.abs(alpha) > np.pi / 3, 0, np.cos(np.pi / 2 * alpha / (np.pi / 3)))
    ref = np.sum(v * pulse(t_del - t_echo[None, :]), axis=1)

    img = {
        k: compute_das_matrix_general(recon, tr, tx, interpolation=k) @ rf.reshape(-1)
        for k in ("nearest", "linear")
    }
    scale = np.max(np.abs(ref))
    err = {k: np.max(np.abs(v_ - ref)) / scale for k, v_ in img.items()}
    assert err["linear"] < 0.02
    assert err["linear"] < err["nearest"] / 4
    # Peak at the scatterer for both:
    for k in img:
        ix, iy = np.unravel_index(np.argmax(img[k].reshape(y.size, x.size).T), (x.size, y.size))
        assert abs(x[ix] - S[0]) <= 2 * pix and abs(y[iy] - S[1]) <= 3 * pix


def test_unknown_interpolation() -> None:
    with pytest.raises(ValueError, match="interpolation"):
        compute_das_matrix([0.0, 1.0], [0.0], [1.0], [0.0], 1.0, 1.0, np.inf, interpolation="sinc")
    recon, tr = single_element(np.array([1e-3]), np.arange(10) / FS)
    with pytest.raises(ValueError, match="interpolation"):
        compute_das_matrix_general(recon, tr, {}, interpolation="cubic")


def test_beamformers_accept_linear(tmp_path) -> None:
    from pyproteus.reconstruction import das_reconstruction, das_reconstruction_general
    from tests.unit.test_reconstruction import general_settings, write_case

    (tmp_path / "ref").mkdir()
    (tmp_path / "gen").mkdir()
    data, settings, S = write_case(tmp_path / "ref")
    for interp in ("nearest", "linear"):
        IMG, x, y = das_reconstruction(data, settings, interpolation=interp)
        ix, iy = np.unravel_index(np.argmax(IMG[:, :, 0]), IMG.shape[:2])
        pix = C / F0 / 5
        assert abs(x[ix] - S[0]) < 4 * pix and abs(y[iy] - S[1]) < 2 * pix
    data, Sg = general_settings(tmp_path / "gen", config="Linear")
    a = das_reconstruction_general(data, Sg, interpolation="linear")[0]
    b = das_reconstruction_general(data, Sg)[0]
    assert a.shape == b.shape and np.isfinite(a).any() and not np.allclose(a, b)


def test_linear_lower_edge_and_element_blocks() -> None:
    """t(1) > 0: a pair with -1 < s < 0 is dropped (floor(s) = -1 would read the
    previous element's last sample), for every element."""
    t = np.arange(5) + 0.25  # c = Fs = 1, plane wave: s = |x| + x - 0.25
    x = np.array([0.0, 0.0625, 0.125, 0.5])  # s = -0.25, -0.125, 0, 0.75
    M = compute_das_matrix(t, [0.0], x, [0.0, 0.0], 1.0, 1.0, np.inf, interpolation="linear")
    dense = M.toarray()
    np.testing.assert_array_equal(dense[0], 0)
    np.testing.assert_array_equal(dense[1], 0)
    np.testing.assert_array_equal(dense[2], [1, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    np.testing.assert_allclose(dense[3], [0.25, 0.75, 0, 0, 0, 0.25, 0.75, 0, 0, 0])
    assert M.has_sorted_indices


def test_linear_matches_its_definition() -> None:
    """Exact reference: rf[i0]*(1-w) + rf[i0+1]*w at s = (t_del - t(1))*Fs."""
    rng = np.random.default_rng(3)
    Nt = 400
    t = np.arange(Nt) / FS + 1e-7
    x = rng.uniform(0.05e-3, 7e-3, 300)
    recon, tr = single_element(x, t)
    rf = rng.standard_normal(Nt)
    M = compute_das_matrix_general(recon, tr, {}, interpolation="linear")
    assert M.has_sorted_indices
    s = (2 * x / C - t[0]) * FS
    ok = (s >= 0) & (s <= Nt - 1)
    i0 = np.floor(s[ok]).astype(int)
    w = s[ok] - i0
    expected = np.zeros(x.size)
    expected[ok] = rf[i0] * (1 - w) + rf[np.minimum(i0 + 1, Nt - 1)] * w
    np.testing.assert_allclose(M @ rf, expected, rtol=1e-13, atol=1e-13)
    assert ok.sum() > 200 and (~ok).sum() > 0


@pytest.fixture(autouse=True)
def _fresh_das_cache():
    """Tests must not depend on DAS matrices cached by other tests."""
    from pyproteus.reconstruction import clear_das_matrix_cache

    clear_das_matrix_cache()
    yield
    clear_das_matrix_cache()
