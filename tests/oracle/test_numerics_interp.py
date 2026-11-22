"""Oracle tests: interp1, griddedInterpolant, pchip.

Comparison class (D-13): floats ≤ 1e-9 relative, with an absolute floor of
1e-12 × the largest finite reference value (near zero); NaN and ±Inf positions
and Inf signs must match. ``'nearest'`` only selects sample values, so it is
compared exactly. Linear interpolation on an interval with an Inf/NaN end value
may give NaN where MATLAB gives a number (B-70, accepted under D-14); those
queries are checked to equal MATLAB or be NaN.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pyproteus import numerics as nm
from tests.oracle.helpers import ORACLE_SETTINGS, batch, col, get, put, row, vec

ORACLE = ORACLE_SETTINGS
REL = 1e-9  # D-13 float class
ATOL_FRACTION = 1e-12  # absolute floor, relative to the largest finite reference value


def assert_float_class(actual, expected, err_msg: str = "", rel: float = REL) -> None:
    """D-13 float class: same shape, NaN and ±Inf positions; finite values within rel."""
    a = np.asarray(actual)
    e = np.asarray(expected)
    assert a.shape == e.shape, f"shape {a.shape} != {e.shape} {err_msg}"
    assert np.array_equal(np.isnan(a), np.isnan(e)), f"NaN positions differ {err_msg}"
    inf = np.isinf(e)
    assert np.array_equal(np.isinf(a), inf), f"Inf positions differ {err_msg}"
    np.testing.assert_array_equal(a[inf], e[inf], err_msg=f"Inf signs differ {err_msg}")
    fin = np.isfinite(e)
    if not fin.any():
        return
    atol = ATOL_FRACTION * np.max(np.abs(e[fin]))
    err = np.abs(a[fin] - e[fin])
    bound = rel * np.abs(e[fin]) + atol
    worst = int(np.argmax(err - bound))
    assert np.all(err <= bound), (
        f"error {err[worst]:.3g} > bound {bound[worst]:.3g} at {e[fin][worst]!r} {err_msg}"
    )


def assert_exact(actual, expected, err_msg: str = "") -> None:
    """Same shape and values (NaN == NaN; -0 == +0)."""
    a, e = np.asarray(actual), np.asarray(expected)
    assert a.shape == e.shape, f"shape {a.shape} != {e.shape} {err_msg}"
    np.testing.assert_array_equal(a, e, err_msg=err_msg)


def _b70_queries(x, v, q):
    """Queries where B-70 allows NaN: on a node, on a flat segment, or at ±Inf, of an
    interval with an Inf/NaN end value (the blend's ``0*Inf`` or ``Inf - Inf``)."""
    xs, vs = (x, v) if x[0] < x[-1] else (x[::-1], v[::-1])
    k = np.clip(np.searchsorted(xs, q, side="right") - 1, 0, xs.size - 2)
    v0, v1 = vs[k], vs[k + 1]
    nonfinite = ~(np.isfinite(v0) & np.isfinite(v1))
    on_node = (q == xs[k]) | (q == xs[k + 1])
    return nonfinite & (on_node | (v0 == v1) | np.isinf(q))


def assert_linear(actual, expected, x, v, q, err_msg: str = "") -> None:
    """Float class; on B-70 queries, equal to MATLAB (Inf/NaN aware) or NaN."""
    a, e = np.asarray(actual), np.asarray(expected)
    b70 = _b70_queries(x, v, q)
    same = (a == e) | (np.isnan(a) & np.isnan(e))
    bad = b70 & ~same & ~np.isnan(a)
    assert not bad.any(), f"B-70 query neither MATLAB's value nor NaN: q={q[bad]} {err_msg}"
    assert_float_class(a[~b70], e[~b70], err_msg)


def _grid(draw, n):
    steps = np.array(draw(st.lists(st.floats(1e-3, 10.0), min_size=n - 1, max_size=n - 1)))
    x0 = draw(st.floats(-100, 100))
    return x0 + np.concatenate([[0.0], np.cumsum(steps)])


@st.composite
def interp_case(draw):
    n = draw(st.integers(2, 25))
    x = _grid(draw, n)
    # Small integer values make flat segments (v0 == v1) and zeros common.
    value = st.one_of(st.floats(-1e3, 1e3), st.integers(-2, 2).map(float))
    v = np.array(draw(st.lists(value, min_size=n, max_size=n)))
    span = x[-1] - x[0]
    q = list(draw(st.lists(st.floats(x[0] - span, x[-1] + span), min_size=1, max_size=60)))
    q += [*x, x[0], x[-1], np.nan, np.inf, -np.inf, *((x[:-1] + x[1:]) / 2)]
    return x, v, np.array(q)


def _matlab(eng, expr, **variables):
    put(eng, **variables)
    eng.eval(f"pyp_r = {expr};", nargout=0)
    return vec(get(eng, "pyp_r"))


@ORACLE
@given(interp_case(), st.booleans())
def test_interp1_linear(matlab_engine, case, flip) -> None:
    x, v, q = case
    if flip:  # MATLAB sorts unsorted sample points
        x, v = x[::-1].copy(), v[::-1].copy()
    kw = {"x": row(x), "v": row(v), "q": row(q)}
    assert_float_class(nm.interp1(x, v, q), _matlab(matlab_engine, "interp1(x, v, q)", **kw))
    assert_float_class(
        nm.interp1(x, v, q, extrap="extrap"),
        _matlab(matlab_engine, "interp1(x, v, q, 'linear', 'extrap')", **kw),
    )
    assert_float_class(
        nm.interp1(x, v, q, extrap=-7.0),
        _matlab(matlab_engine, "interp1(x, v, q, 'linear', -7)", **kw),
    )


@ORACLE
@given(interp_case())
def test_interp1_nearest(matlab_engine, case) -> None:
    x, v, q = case
    out = _matlab(matlab_engine, "interp1(x, v, q, 'nearest')", x=row(x), v=row(v), q=row(q))
    assert_exact(nm.interp1(x, v, q, "nearest"), out)


@ORACLE
@given(interp_case())
def test_gridded_interpolant_linear(matlab_engine, case) -> None:
    x, v, q = case
    put(matlab_engine, x=row(x), v=row(v), q=row(q))
    matlab_engine.eval("pyp_F = griddedInterpolant(x, v); pyp_r = pyp_F(q);", nargout=0)
    out = vec(get(matlab_engine, "pyp_r"))
    assert_float_class(nm.gridded_interpolant(x, v)(q), out)


@ORACLE
@given(interp_case())
def test_pchip(matlab_engine, case) -> None:
    x, v, q = case
    put(matlab_engine, x=row(x), v=row(v), q=row(q))
    matlab_engine.eval(
        "pyp_a = pchip(x, v, q); pyp_b = interp1(x, v, q, 'pchip');"
        "pyp_F = griddedInterpolant(x, v, 'pchip'); pyp_c = pyp_F(q);",
        nargout=0,
    )
    a, b, c = (vec(get(matlab_engine, n)) for n in ("pyp_a", "pyp_b", "pyp_c"))
    mine = nm.pchip(x, v, q)
    assert_float_class(mine, a)  # pchip(x,v,q) gives NaN for NaN queries too
    assert_float_class(nm.interp1(x, v, q, "pchip"), b)
    assert_float_class(nm.gridded_interpolant(x, v, "pchip")(q), c)
    assert np.all(np.isnan(mine[np.isnan(q)]))


def test_pchip_bubble_pulse(matlab_engine, record_property) -> None:
    """Fixed bubble-driving-pulse-like case (dense queries, zero crossings, extrapolation)."""
    rs = np.random.RandomState(7)
    t = np.linspace(0, 2e-6, 401)
    p = 1e5 * np.sin(2 * np.pi * 2.5e6 * t) * np.exp(-(((t - 1e-6) / 4e-7) ** 2))
    p += rs.standard_normal(t.size)
    q = np.sort(rs.random_sample(5000), kind="stable") * 2.2e-6 - 0.1e-6
    put(matlab_engine, t=row(t), p=row(p), q=row(q))
    matlab_engine.eval("pyp_F = griddedInterpolant(t, p, 'pchip'); pyp_r = pyp_F(q);", nargout=0)
    ref = vec(get(matlab_engine, "pyp_r"))
    mine = nm.gridded_interpolant(t, p, "pchip")(q)
    err = float(np.max(np.abs(mine - ref)) / np.max(np.abs(ref)))
    record_property("pchip_max_err_over_max_ref", err)
    assert_float_class(mine, ref)


def test_special_cases(matlab_engine) -> None:
    # compress_bubble: two points, decreasing x, query 0
    x, v = np.array([0.3, -0.2]), np.array([2e-6, 3e-6])
    out = _matlab(matlab_engine, "interp1(x, v, 0)", x=row(x), v=row(v))
    assert_float_class(np.atleast_1d(nm.interp1(x, v, 0.0)), out)
    # Column vectors and matrix-valued v
    vm = np.arange(12.0).reshape(4, 3) ** 2
    xs = np.array([0.0, 1.0, 2.5, 4.0])
    qs = np.array([-1.0, 0.5, 2.5, 3.9, 5.0])
    put(matlab_engine, xs=xs.reshape(-1, 1), vm=vm, qs=qs.reshape(-1, 1))
    matlab_engine.eval(
        "pyp_r = interp1(xs, vm, qs); pyp_p = interp1(xs, vm, qs, 'pchip');", nargout=0
    )
    assert_float_class(nm.interp1(xs, vm, qs), get(matlab_engine, "pyp_r"))
    assert_float_class(nm.interp1(xs, vm, qs, "pchip"), get(matlab_engine, "pyp_p"))


@pytest.mark.parametrize("bad", [[1.0, 2.0, 2.0], [1.0, np.nan, 3.0]])
def test_errors_like_matlab(matlab_engine, bad) -> None:
    import matlab.engine

    put(matlab_engine, x=row(bad), v=row([1.0, 2.0, 3.0]))
    with pytest.raises(matlab.engine.MatlabExecutionError):
        matlab_engine.eval("interp1(x, v, 1.5);", nargout=0)
    with pytest.raises(ValueError):
        nm.interp1(bad, [1.0, 2.0, 3.0], 1.5)


FLAT_X = np.array([5.96, 9.08, 10.08, 10.58, 13.70, 14.1, 17.3, 18.0])
FLAT_V = np.array([1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0])
NONFINITE = [
    (np.array([0.0, 1.0, 2.0]), np.array([1.0, np.inf, 3.0])),
    (np.array([0.0, 1.0, 2.0]), np.array([1.0, np.nan, 3.0])),
    (np.array([0.0, 1.0, 2.0]), np.array([-np.inf, np.inf, np.inf])),
    (np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.1, 0.1, -np.inf, 0.1])),
    (FLAT_X, FLAT_V * 0.1),
    (np.array([0.0, 1.0]), np.array([0.0, -0.0])),
    (np.array([0.0, 1.0, 2.0]), np.array([0.0, -0.0, 0.0])),
    (np.array([0.0, 1.0, 2.0]), np.array([-0.0, 0.0, np.nan])),
]


@pytest.mark.parametrize(("x", "v"), [(FLAT_X, FLAT_V), *NONFINITE])
def test_linear_fixed_cases(matlab_engine, x, v) -> None:
    span = x[-1] - x[0]
    q = np.concatenate([np.linspace(x[0] - span, x[-1] + span, 2001), x, [np.inf, -np.inf]])
    put(matlab_engine, x=row(x), v=row(v), q=row(q))
    matlab_engine.eval(
        "pyp_a = interp1(x, v, q); pyp_b = interp1(x, v, q, 'linear', 'extrap');"
        "pyp_F = griddedInterpolant(x, v); pyp_c = pyp_F(q);",
        nargout=0,
    )
    a, b, c = (vec(get(matlab_engine, n)) for n in ("pyp_a", "pyp_b", "pyp_c"))
    assert_linear(nm.interp1(x, v, q), a, x, v, q)
    assert_linear(nm.interp1(x, v, q, extrap="extrap"), b, x, v, q)
    assert_linear(nm.gridded_interpolant(x, v)(q), c, x, v, q)


def test_pchip_nan_values(matlab_engine) -> None:
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.5])
    v = np.array([1.0, np.nan, 3.0, 2.0, 2.5])
    q = np.linspace(-1, 5.5, 131)
    put(matlab_engine, x=row(x), v=row(v), q=row(q))
    matlab_engine.eval("pyp_a = interp1(x, v, q, 'pchip');", nargout=0)
    assert_float_class(nm.interp1(x, v, q, "pchip"), vec(get(matlab_engine, "pyp_a")))


@pytest.mark.parametrize("q_shape", [(1, 5), (5, 1), (5,)])
def test_matrix_values_vector_query_shape(matlab_engine, q_shape) -> None:
    xs = np.array([0.0, 1.0, 2.5, 4.0])
    vm = np.arange(12.0).reshape(4, 3) ** 2
    qs = np.array([-1.0, 0.5, 2.5, 3.9, 5.0])
    q = qs.reshape(q_shape)
    put(matlab_engine, xs=col(xs), vm=vm, qs=q if q.ndim == 2 else row(q))
    matlab_engine.eval("pyp_r = interp1(xs, vm, qs);", nargout=0)
    assert_float_class(nm.interp1(xs, vm, q), get(matlab_engine, "pyp_r"))


SPECIAL = [1.0, -0.0, 0.0, np.inf, -np.inf, np.nan, 2.0]


@pytest.mark.parametrize("n", [2, 3])
def test_linear_special_value_combinations(matlab_engine, n) -> None:
    """Every combination of special values at nodes, midpoints and outside."""
    x = np.arange(float(n))
    q = np.array([-np.inf, -1.0, *np.arange(0, n - 0.5, 0.5), n - 1.0, float(n), np.inf])
    cases = [np.array(c) for c in itertools.product(SPECIAL, repeat=n)]
    put(matlab_engine, x=row(x), q=row(q))
    rows = [row(c) for c in cases]
    (A,) = batch(matlab_engine, "@(v) interp1(x, v, q)", rows)
    (B,) = batch(matlab_engine, "@(v) interp1(x, v, q, 'linear', 'extrap')", rows)
    (C,) = batch(
        matlab_engine, "@(v) subsref(griddedInterpolant(x, v), substruct('()', {q}))", rows
    )
    for c, a, b, g in zip(cases, A, B, C, strict=True):
        assert_linear(nm.interp1(x, c, q), vec(a), x, c, q, err_msg=str(c))
        assert_linear(nm.interp1(x, c, q, extrap="extrap"), vec(b), x, c, q, err_msg=str(c))
        assert_linear(nm.gridded_interpolant(x, c)(q), vec(g), x, c, q, err_msg=str(c))


def assert_value_or_nan(actual, expected, err_msg: str = "") -> None:
    """B-71: where pyproteus gives NaN, MATLAB may give a number; elsewhere float class."""
    a, e = np.asarray(actual), np.asarray(expected)
    ours_nan = np.isnan(a)
    assert np.all(ours_nan[np.isnan(e)]), f"MATLAB NaN but pyproteus a number {err_msg}"
    assert_float_class(a[~ours_nan], e[~ours_nan], err_msg)


PCHIP_NONFINITE = [
    np.array([np.nan, 1.0, 2.0, 3.0]),
    np.array([np.nan, 5.0, 4.0, 3.0]),
    np.array([1.0, np.inf, 3.0, 4.0]),
    np.array([1.0, np.nan, 3.0, 4.0]),
    np.array([-np.inf, 0.0, 1.0, np.inf]),
    np.array([1.0, 2.0, 2.0, np.nan]),
    np.array([0.0, 0.0, 0.0, 0.0]),
    np.array([1.0, -1.0, 1.0, -1.0]),
]


@pytest.mark.parametrize("v", PCHIP_NONFINITE, ids=str)
def test_pchip_nonfinite_values(matlab_engine, v) -> None:
    """NaN samples dropped by interp1/pchip; propagated by griddedInterpolant (B-71); ±Inf."""
    x = np.array([0.0, 1.0, 2.0, 3.5])
    q = np.array([-np.inf, -1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 2.75, 3.5, 5.0, np.inf, np.nan])
    put(matlab_engine, x=row(x), v=row(v), q=row(q))
    matlab_engine.eval(
        "pyp_a = interp1(x, v, q, 'pchip'); pyp_b = pchip(x, v, q);"
        "pyp_F = griddedInterpolant(x, v, 'pchip'); pyp_c = pyp_F(q);",
        nargout=0,
    )
    a, b, c = (vec(get(matlab_engine, n)) for n in ("pyp_a", "pyp_b", "pyp_c"))
    assert_float_class(nm.interp1(x, v, q, "pchip"), a)
    assert_float_class(nm.pchip(x, v, q), b)
    assert_value_or_nan(nm.gridded_interpolant(x, v, "pchip")(q), c)
    if not np.isnan(v).any():  # no NaN: strict float class
        assert_float_class(nm.gridded_interpolant(x, v, "pchip")(q), c)


def test_pchip_matrix_values_with_nan(matlab_engine) -> None:
    """interp1 'pchip' drops NaN rows column by column."""
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.5])
    vm = np.array([[1, 1, 2], [np.nan, 2, 0], [2, np.nan, 1], [4, 5, 3], [3, 4, np.nan]])
    q = np.array([-1.0, 0.5, 1.5, 2.5, 4.0, 6.0])
    put(matlab_engine, x=col(x), vm=vm, q=col(q))
    matlab_engine.eval("pyp_r = interp1(x, vm, q, 'pchip');", nargout=0)
    assert_float_class(nm.interp1(x, vm, q, "pchip"), get(matlab_engine, "pyp_r"))


def test_pchip_complex_values(matlab_engine) -> None:
    x = np.array([0.0, 1.0, 2.0, 3.0])
    v = np.array([1 + 1j, 2.0, 3j, -1 - 2j])
    q = np.array([-0.5, 0.5, 1.5, 2.2, 3.5])
    put(matlab_engine, x=row(x), v=row(v), q=row(q))
    matlab_engine.eval("pyp_r = interp1(x, v, q, 'pchip');", nargout=0)
    ref = vec(get(matlab_engine, "pyp_r"))
    mine = nm.interp1(x, v, q, "pchip")
    assert_float_class(mine.real, ref.real)
    assert_float_class(mine.imag, ref.imag)


@pytest.mark.parametrize(
    "v",
    [np.array([1 + 1j, 2 + 3j, 3.0]), np.array([1 + 1j, 2 + 1j, 3 + 2j])],  # 2nd: flat imag start
    ids=str,
)
def test_linear_complex_infinite_queries(matlab_engine, v) -> None:
    x = np.array([0.0, 1.0, 2.0])
    q = np.array([-np.inf, -1.0, 0.5, 1.5, 3.0, np.inf])
    put(matlab_engine, x=row(x), v=row(v), q=row(q))
    matlab_engine.eval(
        "pyp_a = interp1(x, v, q, 'linear', 'extrap');"
        "pyp_F = griddedInterpolant(x, v); pyp_b = pyp_F(q);",
        nargout=0,
    )
    for mine, name in [
        (nm.interp1(x, v, q, extrap="extrap"), "pyp_a"),
        (nm.gridded_interpolant(x, v)(q), "pyp_b"),
    ]:
        ref = vec(get(matlab_engine, name)).astype(complex)
        assert_float_class(mine.real, ref.real, name)
        assert_float_class(mine.imag, ref.imag, name)
