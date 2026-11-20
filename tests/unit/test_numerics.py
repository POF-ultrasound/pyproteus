"""Pure-Python tests for pyproteus.numerics (no MATLAB needed).

Expected values marked "MATLAB" were taken from MATLAB R2025a; the oracle
tests in tests/oracle/ check the same behaviour live.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from pyproteus import numerics as nm

finite = st.floats(allow_nan=False, allow_infinity=False, width=64)


# --- round -------------------------------------------------------------------


def test_round_ties_away_from_zero() -> None:
    x = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, -3.5])
    np.testing.assert_array_equal(nm.matlab_round(x), [-3, -2, -1, 1, 2, 3, 4, -4])


def test_round_specials() -> None:
    x = np.array([0.49999999999999994, -0.49999999999999994, 4503599627370495.5, np.nan])
    out = nm.matlab_round(x)
    np.testing.assert_array_equal(out, [0, -0, 4503599627370496, np.nan])
    assert np.signbit(nm.matlab_round(-0.4))  # MATLAB: round(-0.4) is -0
    assert np.signbit(nm.matlab_round(-0.0))
    np.testing.assert_array_equal(nm.matlab_round([np.inf, -np.inf]), [np.inf, -np.inf])


def test_round_dtypes() -> None:
    assert nm.matlab_round(np.float32(2.5)).dtype == np.float32
    assert nm.matlab_round(np.float32(2.5)) == 3
    np.testing.assert_array_equal(nm.matlab_round(np.array([1, -2])), [1, -2])
    np.testing.assert_array_equal(nm.matlab_round([1.5 - 2.5j]), [2 - 3j])


@given(hnp.arrays(np.float64, hnp.array_shapes(max_dims=3), elements=finite))
def test_round_matches_definition(x) -> None:
    out = nm.matlab_round(x)
    # Nearest integer; at exact halves, the one farther from zero.
    assert np.all(np.abs(out - x) <= 0.5)
    halves = np.abs(x - np.trunc(x)) == 0.5
    assert np.all(np.abs(out[halves]) > np.abs(x[halves]))
    assert np.all(np.mod(out, 1) == 0)


# --- sort --------------------------------------------------------------------


def test_sort_matlab_example() -> None:
    x = [3, np.nan, 1, np.nan, 1, -0.0, 0.0, -np.inf]
    s, i = nm.sort(x)  # MATLAB: I = [8 6 7 3 5 1 2 4]
    np.testing.assert_array_equal(i + 1, [8, 6, 7, 3, 5, 1, 2, 4])
    s, i = nm.sort(x, direction="descend")  # MATLAB: I = [2 4 1 3 5 6 7 8]
    np.testing.assert_array_equal(i + 1, [2, 4, 1, 3, 5, 6, 7, 8])
    np.testing.assert_array_equal(s, np.asarray(x)[i])


def test_sort_default_axis() -> None:
    row = np.array([[3.0, 1.0, 2.0]])
    _s, i = nm.sort(row)  # 1×3: sorts along axis 1
    np.testing.assert_array_equal(i, [[1, 2, 0]])
    m = np.array([[3.0, 1.0], [1.0, 2.0]])
    np.testing.assert_array_equal(nm.argsort(m), [[1, 0], [0, 1]])
    np.testing.assert_array_equal(nm.argsort(m, axis=1), [[1, 0], [0, 1]])


@given(
    hnp.arrays(np.float64, st.integers(0, 30), elements=st.sampled_from([0.0, 1.0, -1.0, np.nan]))
)
def test_sort_stable_both_directions(x) -> None:
    for direction in ("ascend", "descend"):
        i = nm.argsort(x, direction=direction)
        s = x[i]
        finite_part = s[~np.isnan(s)]
        step = np.diff(finite_part)
        assert np.all(step >= 0) if direction == "ascend" else np.all(step <= 0)
        nan_pos = np.flatnonzero(np.isnan(s))
        n_nan = nan_pos.size
        expected = np.arange(len(s) - n_nan, len(s)) if direction == "ascend" else np.arange(n_nan)
        np.testing.assert_array_equal(nan_pos, expected)
        # Stability: within each group of equal keys, original indices increase.
        for key in (0.0, 1.0, -1.0):
            assert np.all(np.diff(i[s == key]) > 0)
        assert np.all(np.diff(i[np.isnan(s)]) > 0)


# --- indexing ----------------------------------------------------------------


def test_find_column_major() -> None:
    m = np.array([[0, 1, 0], [1, 0, np.nan]])
    np.testing.assert_array_equal(nm.find(m), [1, 2, 5])  # MATLAB: [2 3 6]
    np.testing.assert_array_equal(nm.find(m, 2), [1, 2])
    np.testing.assert_array_equal(nm.find(m, 2, "last"), [2, 5])
    assert nm.find(np.zeros((2, 2))).size == 0


def test_sub2ind_ind2sub() -> None:
    shape = (2, 3, 4)
    assert nm.sub2ind(shape, 1, 2, 3) == 1 + 2 * 2 + 3 * 6  # MATLAB sub2ind(...,2,3,4)=24
    i, j, k = nm.ind2sub(shape, 23)
    assert (i, j, k) == (1, 2, 3)
    with pytest.raises(ValueError):
        nm.sub2ind(shape, 2, 0, 0)
    with pytest.raises(ValueError):
        nm.sub2ind(shape, 0.5, 0, 0)


@given(st.lists(st.integers(1, 5), min_size=1, max_size=4), st.data())
def test_sub2ind_roundtrip(shape, data) -> None:
    n = int(np.prod(shape))
    ind = np.array(data.draw(st.lists(st.integers(0, n - 1), max_size=20)), dtype=np.intp)
    subs = nm.ind2sub(shape, ind)
    np.testing.assert_array_equal(nm.sub2ind(shape, *subs), ind)
    a = np.arange(n).reshape(shape, order="F")
    np.testing.assert_array_equal(a[subs], ind)


# --- interpolation -----------------------------------------------------------


def test_interp1_outside_is_nan() -> None:
    out = nm.interp1([1, 2, 3], [10, 20, 40], [0, 1, 3, 4, 2.5, np.nan])
    np.testing.assert_array_equal(out, [np.nan, 10, 40, np.nan, 30, np.nan])


def test_interp1_options() -> None:
    np.testing.assert_array_equal(nm.interp1([3, 2, 1], [40, 20, 10], [1.5, 2.5]), [15, 30])
    np.testing.assert_array_equal(
        nm.interp1([1, 2, 3], [10, 20, 40], [0, 4], extrap="extrap"), [0, 60]
    )
    np.testing.assert_array_equal(nm.interp1([1, 2, 3], [10, 20, 40], [0, 4], extrap=0), [0, 0])
    np.testing.assert_array_equal(
        nm.interp1([1, 2, 3], [10, 20, 40], [1.5, 2.5], "nearest"), [20, 40]
    )
    with pytest.raises(ValueError, match="unique"):
        nm.interp1([1, 2, 2], [1, 2, 3], 1.5)


def test_interp1_matrix_values_and_shape() -> None:
    v = np.array([[0.0, 1.0], [10.0, 11.0], [20.0, 21.0]])
    # MATLAB: a column or row xq with matrix v gives numel(xq)×size(v,2)
    for q in (np.array([[0.5], [1.5]]), np.array([[0.5, 1.5]]), np.array([0.5, 1.5])):
        np.testing.assert_array_equal(nm.interp1([0, 1, 2], v, q), [[5, 6], [15, 16]])
    assert nm.interp1([0, 1, 2], v, np.ones((2, 2))).shape == (2, 2, 2)


@given(
    st.lists(finite.filter(lambda t: abs(t) < 1e6), min_size=2, max_size=12, unique=True),
    st.data(),
)
def test_interp1_exact_at_nodes(xs, data) -> None:
    x = np.array(xs)
    v = np.array(
        data.draw(st.lists(finite.filter(lambda t: abs(t) < 1e6), min_size=x.size, max_size=x.size))
    )
    np.testing.assert_array_equal(nm.interp1(x, v, x), v)


def test_pchip_matlab_example() -> None:
    # MATLAB: interp1([1 2 3 4],[10 20 40 30],[0 5],'pchip') = [13.3333 -20]
    np.testing.assert_allclose(
        nm.interp1([1, 2, 3, 4], [10, 20, 40, 30], [0, 5], "pchip"), [40 / 3, -20], rtol=1e-14
    )


def test_pchip_shape_preserving_and_two_points() -> None:
    x = np.array([0.0, 1, 2, 3, 4])
    y = np.array([0.0, 0, 1, 1, 1])  # flat, rise, flat
    q = np.linspace(0, 4, 401)
    out = nm.pchip(x, y, q)
    assert np.all(np.diff(out) >= -1e-15)
    assert np.all((out >= 0) & (out <= 1))
    np.testing.assert_allclose(nm.pchip([0, 2], [1, 5], [1, 3]), [3, 7])


def test_pchip_matches_scipy() -> None:
    from scipy.interpolate import PchipInterpolator

    rs = np.random.RandomState(0)
    x = np.cumsum(rs.random_sample(30) + 0.1)
    y = rs.standard_normal(30)
    q = np.linspace(x[0], x[-1], 1000)
    np.testing.assert_allclose(
        nm.pchip(x, y, q), PchipInterpolator(x, y)(q), rtol=1e-12, atol=1e-12
    )


def test_gridded_interpolant() -> None:
    f = nm.gridded_interpolant([1, 2, 3], [10, 20, 40])
    np.testing.assert_array_equal(f([0, 4, 2.5]), [0, 60, 30])  # linear extrapolation
    g = nm.gridded_interpolant([1, 2, 3], [10, 20, 40], extrap="none")
    np.testing.assert_array_equal(g([0, 2]), [np.nan, 20])
    with pytest.raises(ValueError, match="ascending"):
        nm.gridded_interpolant([3, 2, 1], [1, 2, 3])


# --- spectral ----------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 7, 8, 33])
def test_ifft_symmetric_real_signal(n) -> None:
    rs = np.random.RandomState(n)
    x = rs.standard_normal((3, n))
    X = np.fft.fft(x, axis=1)
    np.testing.assert_allclose(nm.ifft_symmetric(X, axis=1), x, atol=1e-12)
    # Corrupting the upper half changes nothing.
    X[:, n // 2 + 1 :] = 99.0
    np.testing.assert_allclose(nm.ifft_symmetric(X, axis=1), x, atol=1e-12)


def test_ifft_symmetric_n() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    X = np.fft.fft(x, 8)
    np.testing.assert_allclose(nm.ifft_symmetric(X, 8), np.r_[x, 0, 0, 0, 0], atol=1e-14)


def test_hilbert_axis_rules() -> None:
    from scipy.signal import hilbert as sp_hilbert

    rs = np.random.RandomState(1)
    m = rs.standard_normal((16, 3))
    np.testing.assert_allclose(nm.hilbert(m), sp_hilbert(m, axis=0), atol=1e-12)
    r = rs.standard_normal((1, 9))
    np.testing.assert_allclose(nm.hilbert(r), sp_hilbert(r, axis=1), atol=1e-12)
    v = rs.standard_normal(10)
    np.testing.assert_allclose(nm.hilbert(v).real, v, atol=1e-12)


# --- convolution -------------------------------------------------------------


def test_conv_same_even_kernels() -> None:
    # MATLAB conv(...,'same') results
    np.testing.assert_array_equal(nm.conv(np.arange(1, 6), [1, 10], "same"), [12, 23, 34, 45, 50])
    np.testing.assert_array_equal(
        nm.conv(np.arange(1, 6), [1, 10, 100, 1000], "same"), [123, 1234, 2345, 3450, 4500]
    )
    np.testing.assert_array_equal(
        nm.conv(np.arange(1, 7), [1, 10, 100], "same"), [12, 123, 234, 345, 456, 560]
    )
    np.testing.assert_array_equal(nm.conv([1, 2, 3], [1, 10]), [1, 12, 23, 30])
    np.testing.assert_array_equal(nm.conv([1, 2, 3], [1, 10], "valid"), [12, 23])


def test_convn_trailing_singletons() -> None:
    a = np.arange(24.0).reshape(2, 3, 4)
    k = np.ones((2, 2))
    out = nm.convn(a, k, "same")
    assert out.shape == a.shape
    ref = nm.convn(a, k[:, :, None], "same")
    np.testing.assert_array_equal(out, ref)
    assert nm.convn(a, np.ones((3, 4, 5)), "valid").shape == (0, 0, 0)


# --- random ------------------------------------------------------------------


def test_rng_seed0_is_matlab_default() -> None:
    # MATLAB: rng(0,'twister'); rand(1,3) = 0.8147 0.9058 0.1270
    np.testing.assert_allclose(
        nm.rng(0).rand(1, 3), [[0.81472369, 0.90579194, 0.12698682]], atol=5e-9
    )


def test_rand_column_major_and_sizes() -> None:
    r = nm.rng(42).rand(2, 3)
    flat = np.random.RandomState(42).random_sample(6)
    np.testing.assert_array_equal(r, flat.reshape((2, 3), order="F"))
    assert nm.rng(1).rand(3).shape == (3, 3)  # MATLAB rand(n) is n×n
    assert nm.rng(1).rand().shape == (1, 1)
    assert nm.rng(1).rand([2, 3, 4]).shape == (2, 3, 4)


def test_randi_and_interleaving() -> None:
    g = nm.rng(42)
    np.testing.assert_array_equal(g.randi(10, 1, 8), [[4, 10, 8, 6, 2, 2, 1, 9]])  # MATLAB
    np.testing.assert_array_equal(nm.rng(42).randi([3, 7], 1, 8), [[4, 7, 6, 5, 3, 3, 3, 7]])
    g = nm.rng(42)
    g.randi(10, 1, 2)
    assert g.rand().item() == np.random.RandomState(42).random_sample(3)[2]


def test_interp1_flat_and_nonfinite_neighbours() -> None:
    # MATLAB R2025a: flat segments give v0 exactly; nodes give the node value.
    x = np.array([5.96, 9.08, 10.08, 10.58, 13.70])
    v = np.array([1.0, 2.0, 2.0, 3.0, 3.0])
    q = np.linspace(x[0], x[-1], 997)
    out = nm.interp1(x, v, q)
    flat = q >= x[3]
    assert np.all(out[flat] == 3.0)
    np.testing.assert_array_equal(
        nm.interp1([0, 1, 2], [1, np.inf, 3], [0, 0.5, 1, 1.5, 2]), [1, np.inf, np.inf, np.inf, 3]
    )
    np.testing.assert_array_equal(
        nm.interp1([0, 1, 2], [1, np.nan, 3], [0, 0.5, 2]), [1, np.nan, 3]
    )
    np.testing.assert_array_equal(
        nm.interp1([0, 1, 2], [1, np.inf, 3], [-1, 3], extrap="extrap"), [-np.inf, -np.inf]
    )


def test_interp1_pchip_drops_nan_values() -> None:
    # MATLAB: interp1([0 1 2 3],[1 NaN 3 2],q,'pchip') == pchip([0 2 3],[1 3 2],q)
    q = np.array([0, 0.5, 1, 1.5, 2])
    np.testing.assert_array_equal(
        nm.interp1([0, 1, 2, 3], [1, np.nan, 3, 2], q, "pchip"), nm.pchip([0, 2, 3], [1, 3, 2], q)
    )


def test_sort_trailing_singleton_axis() -> None:
    a = np.array([[3.0, 1.0], [2.0, 4.0]])
    s, i = nm.sort(a, axis=2)  # MATLAB sort(A, 3)
    np.testing.assert_array_equal(s, a)
    np.testing.assert_array_equal(i, np.zeros((2, 2)))
    with pytest.raises(ValueError):
        nm.argsort(a, axis=-3)


def test_ifft_symmetric_default_axis() -> None:
    x = np.random.RandomState(0).standard_normal((6, 2))
    X = np.fft.fft(x, axis=0)
    np.testing.assert_allclose(nm.ifft_symmetric(X), x, atol=1e-12)  # MATLAB: along columns


def test_random_size_rules() -> None:
    assert nm.rng(1).rand(-2, 3).shape == (0, 3)
    assert nm.rng(1).rand(2, 3, 1).shape == (2, 3)
    assert nm.rng(1).rand(2, 1, 3).shape == (2, 1, 3)
    with pytest.raises(ValueError):
        nm.rng(1).randi(3.5, 2, 2)
    with pytest.raises(ValueError):
        nm.rng(1).rand(2.5)


def test_convn_rejects_empty() -> None:
    with pytest.raises(ValueError):
        nm.convn(np.zeros((0, 3)), np.ones((2, 2)))
