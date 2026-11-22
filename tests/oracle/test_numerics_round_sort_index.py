"""Oracle tests: round, sort, find, sub2ind, ind2sub.

Comparison class: exact. These helpers return integers (rounded values,
indices) or permutations of exact inputs, so equality is documented semantics,
not float noise (D-13). Values compare with NaN == NaN; the sign of zero is not
checked (D-12).
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from pyproteus import numerics as nm
from tests.oracle.helpers import ORACLE_SETTINGS, batch, col, get, put, row, vec

ORACLE = ORACLE_SETTINGS


def assert_exact(actual, expected, err_msg: str = "") -> None:
    """Same shape and values (NaN == NaN; -0 == +0)."""
    a, e = np.asarray(actual), np.asarray(expected)
    assert a.shape == e.shape, f"shape {a.shape} != {e.shape} {err_msg}"
    np.testing.assert_array_equal(a, e, err_msg=err_msg)


TIES = np.concatenate(
    [
        np.arange(-10, 10) + 0.5,
        [0.49999999999999994, -0.49999999999999994, 0.5000000000000001, -0.5000000000000001],
        [2.0**52 - 0.5, -(2.0**52) + 0.5, 2.0**52 + 1, 2.0**53, -(2.0**53) - 2],
        [0.0, -0.0, 1e-300, -1e-300, 5e-324, -5e-324],
        [np.nan, np.inf, -np.inf],
        np.arange(-3, 3, 0.25),
    ]
)


def test_round_ties_and_specials(matlab_engine) -> None:
    put(matlab_engine, x=row(TIES))
    matlab_engine.eval("y = round(x); ys = double(round(single(x)));", nargout=0)
    y, ys = get(matlab_engine, "y", "ys")
    assert_exact(nm.matlab_round(TIES).reshape(1, -1), y)
    assert_exact(nm.matlab_round(TIES.astype(np.float32)).astype(np.float64), ys.ravel())


def test_round_nd_shape(matlab_engine) -> None:
    x = (np.arange(2 * 3 * 5).reshape(2, 3, 5) - 15) / 2.0
    put(matlab_engine, x=x)
    matlab_engine.eval("y = round(x);", nargout=0)
    assert_exact(nm.matlab_round(x), get(matlab_engine, "y"))


@ORACLE
@given(
    st.lists(
        st.one_of(
            st.floats(allow_nan=True, allow_infinity=True),
            st.integers(-(2**20), 2**20).map(lambda k: k / 2),  # exact halves
            st.tuples(st.integers(-(2**20), 2**20), st.sampled_from([-1.0, 1.0])).map(
                lambda t: np.nextafter(t[0] / 2, t[1] * np.inf)  # one ulp off a half
            ),
        ),
        min_size=1,
        max_size=200,
    )
)
def test_round_random(matlab_engine, xs) -> None:
    x = np.array(xs, dtype=np.float64)
    put(matlab_engine, x=row(x))
    matlab_engine.eval("y = round(x);", nargout=0)
    assert_exact(nm.matlab_round(x), vec(get(matlab_engine, "y")))


# --- sort ----------------------------------------------------------------------

SORT_VALUES = st.sampled_from([0.0, -0.0, 1.0, -1.0, 2.5, np.nan, np.inf, -np.inf])


def _sort_cases():
    yield np.array([3, np.nan, 1, np.nan, 1, -0.0, 0.0, -np.inf])
    yield np.array([1.0, 1.0, 1.0])
    yield np.array([np.nan, np.nan])
    yield np.array([5.0])
    yield np.array([2.0, 1.0, 2.0, 1.0, 2.0])


@pytest.mark.parametrize("direction", ["ascend", "descend"])
@pytest.mark.parametrize("orient", [row, col])
def test_sort_vectors(matlab_engine, direction, orient) -> None:
    cases = list(_sort_cases())
    S, I = batch(
        matlab_engine, f"@(x) sort(x, '{direction}')", [orient(c) for c in cases], nargout=2
    )
    for c, s_m, i_m in zip(cases, S, I, strict=True):
        s, i = nm.sort(orient(c).a, direction=direction)
        assert_exact(s, s_m, err_msg=str(c))
        np.testing.assert_array_equal(i + 1, i_m, err_msg=str(c))


@ORACLE
@given(
    st.lists(
        hnp.arrays(
            np.float64, hnp.array_shapes(min_dims=2, max_dims=3, max_side=5), elements=SORT_VALUES
        ),
        min_size=1,
        max_size=10,
    ),
    st.sampled_from(["ascend", "descend"]),
    st.sampled_from([None, 1, 2, 3]),
)
def test_sort_random_nd(matlab_engine, arrays, direction, dim) -> None:
    fn = f"@(x) sort(x, '{direction}')" if dim is None else f"@(x) sort(x, {dim}, '{direction}')"
    S, I = batch(matlab_engine, fn, arrays, nargout=2)
    for a, s_m, i_m in zip(arrays, S, I, strict=True):
        axis = None if dim is None else dim - 1  # dim > ndim: trailing singleton
        s, i = nm.sort(a, axis=axis, direction=direction)
        assert_exact(s, s_m.reshape(s.shape), err_msg=f"{a} {dim}")
        np.testing.assert_array_equal(i + 1, i_m.reshape(i.shape))


# --- find / sub2ind / ind2sub ----------------------------------------------------


@ORACLE
@given(
    st.lists(
        hnp.arrays(
            np.float64,
            hnp.array_shapes(min_dims=2, max_dims=4, max_side=5),
            elements=st.sampled_from([0.0, 0.0, 1.0, -2.0, np.nan, -0.0]),
        ),
        min_size=1,
        max_size=10,
    ),
    st.integers(1, 6),
)
def test_find(matlab_engine, arrays, n) -> None:
    (F,) = batch(matlab_engine, "@(x) find(x)", arrays)
    (Ff,) = batch(matlab_engine, f"@(x) find(x, {n})", arrays)
    (Fl,) = batch(matlab_engine, f"@(x) find(x, {n}, 'last')", arrays)
    for a, f, ff, fl in zip(arrays, F, Ff, Fl, strict=True):
        np.testing.assert_array_equal(nm.find(a) + 1, f.ravel())
        np.testing.assert_array_equal(nm.find(a, n) + 1, ff.ravel())
        np.testing.assert_array_equal(nm.find(a, n, "last") + 1, fl.ravel())


@ORACLE
@given(st.lists(st.integers(1, 7), min_size=2, max_size=4), st.data())
def test_sub2ind_ind2sub(matlab_engine, shape, data) -> None:
    n = int(np.prod(shape))
    ind = np.array(data.draw(st.lists(st.integers(0, n - 1), min_size=1, max_size=50)))
    subs = nm.ind2sub(shape, ind)
    put(matlab_engine, sz=row(np.array(shape, dtype=float)), ind=row(ind + 1.0))
    outs = ", ".join(f"s{k}" for k in range(len(shape)))
    matlab_engine.eval(f"[{outs}] = ind2sub(sz, ind);", nargout=0)
    for k in range(len(shape)):
        np.testing.assert_array_equal(subs[k] + 1, vec(get(matlab_engine, f"s{k}")))
    args = ", ".join(f"s{k}" for k in range(len(shape)))
    matlab_engine.eval(f"back = sub2ind(sz, {args});", nargout=0)
    np.testing.assert_array_equal(nm.sub2ind(shape, *subs) + 1, vec(get(matlab_engine, "back")))


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.uint8])
def test_sort_integer_dtypes(matlab_engine, dtype) -> None:
    a = np.array([[3, 1, 2, 1, 0, 3], [0, 0, 5, 1, 1, 2]], dtype=dtype)
    S, I = batch(matlab_engine, "@(x) sort(x, 2, 'descend')", [a], nargout=2)
    s, i = nm.sort(a, axis=1, direction="descend")
    np.testing.assert_array_equal(s, S[0])
    assert S[0].dtype == np.dtype(dtype)
    np.testing.assert_array_equal(i + 1, I[0])
