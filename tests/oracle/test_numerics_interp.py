"""Oracle tests: interp1, griddedInterpolant, pchip.

Linear, nearest and pchip were bit-identical to MATLAB R2025a, so bit identity
is asserted (stronger than the ≤ 1e-12 class of ROADMAP §4).
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pyproteus import numerics as nm
from tests.oracle.helpers import (
    ORACLE_SETTINGS,
    assert_same_float,
    batch,
    col,
    get,
    put,
    row,
    vec,
)

ORACLE = ORACLE_SETTINGS


def _grid(draw, n, decreasing=False):
    steps = np.array(draw(st.lists(st.floats(1e-3, 10.0), min_size=n - 1, max_size=n - 1)))
    x0 = draw(st.floats(-100, 100))
    x = x0 + np.concatenate([[0.0], np.cumsum(steps)])
    return x[::-1].copy() if decreasing else x


@st.composite
def interp_case(draw):
    n = draw(st.integers(2, 25))
    x = _grid(draw, n)
    # Small integer values make flat segments (v0 == v1) common.
    value = st.one_of(st.floats(-1e3, 1e3), st.integers(-2, 2).map(float))
    v = np.array(draw(st.lists(value, min_size=n, max_size=n)))
    span = x[-1] - x[0]
    q = list(draw(st.lists(st.floats(x[0] - span, x[-1] + span), min_size=1, max_size=60)))
    q += [*x, x[0], x[-1], np.nan, *((x[:-1] + x[1:]) / 2)]  # nodes, midpoints
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
    assert_same_float(nm.interp1(x, v, q), _matlab(matlab_engine, "interp1(x, v, q)", **kw))
    assert_same_float(
        nm.interp1(x, v, q, extrap="extrap"),
        _matlab(matlab_engine, "interp1(x, v, q, 'linear', 'extrap')", **kw),
    )
    assert_same_float(
        nm.interp1(x, v, q, extrap=-7.0),
        _matlab(matlab_engine, "interp1(x, v, q, 'linear', -7)", **kw),
    )


@ORACLE
@given(interp_case())
def test_interp1_nearest(matlab_engine, case) -> None:
    x, v, q = case
    out = _matlab(matlab_engine, "interp1(x, v, q, 'nearest')", x=row(x), v=row(v), q=row(q))
    assert_same_float(nm.interp1(x, v, q, "nearest"), out)


@ORACLE
@given(interp_case())
def test_gridded_interpolant_linear(matlab_engine, case) -> None:
    x, v, q = case
    put(matlab_engine, x=row(x), v=row(v), q=row(q))
    matlab_engine.eval("pyp_F = griddedInterpolant(x, v); pyp_r = pyp_F(q);", nargout=0)
    out = vec(get(matlab_engine, "pyp_r"))
    assert_same_float(nm.gridded_interpolant(x, v)(q), out)


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
    q_ok = ~np.isnan(q)
    mine = nm.pchip(x, v, q)
    # pchip(x,v,q) returns NaN for NaN queries as well
    assert_same_float(mine, a)
    assert_same_float(nm.interp1(x, v, q, "pchip"), b)
    assert_same_float(nm.gridded_interpolant(x, v, "pchip")(q), c)
    assert np.all(np.isnan(mine[~q_ok]))


def test_pchip_ulp_report(matlab_engine, record_property) -> None:
    """Measure the ulp distance to MATLAB on a fixed, bubble-pulse-like case."""
    rs = np.random.RandomState(7)
    t = np.linspace(0, 2e-6, 401)
    p = 1e5 * np.sin(2 * np.pi * 2.5e6 * t) * np.exp(-(((t - 1e-6) / 4e-7) ** 2))
    p += rs.standard_normal(t.size)
    q = np.sort(rs.random_sample(5000), kind="stable") * 2.2e-6 - 0.1e-6
    put(matlab_engine, t=row(t), p=row(p), q=row(q))
    matlab_engine.eval("pyp_F = griddedInterpolant(t, p, 'pchip'); pyp_r = pyp_F(q);", nargout=0)
    ref = vec(get(matlab_engine, "pyp_r"))
    mine = nm.gridded_interpolant(t, p, "pchip")(q)
    ulps = np.abs(mine - ref) / np.spacing(np.abs(ref))
    record_property("pchip_max_ulp", float(ulps.max()))
    record_property("pchip_exact_fraction", float(np.mean(mine == ref)))
    print(f"pchip vs MATLAB: max {ulps.max():.1f} ulp, exact {np.mean(mine == ref):.3f}")
    assert_same_float(mine, ref)


def test_special_cases(matlab_engine) -> None:
    # compress_bubble: two points, decreasing x, query 0
    x, v = np.array([0.3, -0.2]), np.array([2e-6, 3e-6])
    out = _matlab(matlab_engine, "interp1(x, v, 0)", x=row(x), v=row(v))
    assert_same_float(np.atleast_1d(nm.interp1(x, v, 0.0)), out)
    # Column vectors and matrix-valued v
    vm = np.arange(12.0).reshape(4, 3) ** 2
    xs = np.array([0.0, 1.0, 2.5, 4.0])
    qs = np.array([-1.0, 0.5, 2.5, 3.9, 5.0])
    put(matlab_engine, xs=xs.reshape(-1, 1), vm=vm, qs=qs.reshape(-1, 1))
    matlab_engine.eval("pyp_r = interp1(xs, vm, qs);", nargout=0)
    assert_same_float(nm.interp1(xs, vm, qs), get(matlab_engine, "pyp_r"))


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
    q = np.concatenate([np.linspace(x[0] - span, x[-1] + span, 2001), x])
    put(matlab_engine, x=row(x), v=row(v), q=row(q))
    matlab_engine.eval(
        "pyp_a = interp1(x, v, q); pyp_b = interp1(x, v, q, 'linear', 'extrap');"
        "pyp_F = griddedInterpolant(x, v); pyp_c = pyp_F(q);",
        nargout=0,
    )
    a, b, c = (vec(get(matlab_engine, n)) for n in ("pyp_a", "pyp_b", "pyp_c"))
    assert_same_float(nm.interp1(x, v, q), a)
    assert_same_float(nm.interp1(x, v, q, extrap="extrap"), b)
    assert_same_float(nm.gridded_interpolant(x, v)(q), c)


def test_pchip_nan_values(matlab_engine) -> None:
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.5])
    v = np.array([1.0, np.nan, 3.0, 2.0, 2.5])
    q = np.linspace(-1, 5.5, 131)
    put(matlab_engine, x=row(x), v=row(v), q=row(q))
    matlab_engine.eval("pyp_a = interp1(x, v, q, 'pchip');", nargout=0)
    assert_same_float(nm.interp1(x, v, q, "pchip"), vec(get(matlab_engine, "pyp_a")))


@pytest.mark.parametrize("q_shape", [(1, 5), (5, 1), (5,)])
def test_matrix_values_vector_query_shape(matlab_engine, q_shape) -> None:
    xs = np.array([0.0, 1.0, 2.5, 4.0])
    vm = np.arange(12.0).reshape(4, 3) ** 2
    qs = np.array([-1.0, 0.5, 2.5, 3.9, 5.0])
    q = qs.reshape(q_shape)
    put(matlab_engine, xs=col(xs), vm=vm, qs=q if q.ndim == 2 else row(q))
    matlab_engine.eval("pyp_r = interp1(xs, vm, qs);", nargout=0)
    assert_same_float(nm.interp1(xs, vm, q), get(matlab_engine, "pyp_r"))


SPECIAL = [1.0, -0.0, 0.0, np.inf, -np.inf, np.nan, 2.0]


@pytest.mark.parametrize("n", [2, 3])
def test_linear_special_value_combinations(matlab_engine, n) -> None:
    """Every combination of special values at nodes, midpoints and outside."""
    import itertools

    x = np.arange(float(n))
    q = np.array([-1.0, *np.arange(0, n - 0.5, 0.5), n - 1.0, float(n)])
    cases = [np.array(c) for c in itertools.product(SPECIAL, repeat=n)]
    put(matlab_engine, x=row(x), q=row(q))
    rows = [row(c) for c in cases]
    (A,) = batch(matlab_engine, "@(v) interp1(x, v, q)", rows)
    (B,) = batch(matlab_engine, "@(v) interp1(x, v, q, 'linear', 'extrap')", rows)
    (C,) = batch(
        matlab_engine, "@(v) subsref(griddedInterpolant(x, v), substruct('()', {q}))", rows
    )
    for c, a, b, g in zip(cases, A, B, C, strict=True):
        assert_same_float(nm.interp1(x, c, q), vec(a), err_msg=str(c))
        assert_same_float(nm.interp1(x, c, q, extrap="extrap"), vec(b), err_msg=str(c))
        assert_same_float(nm.gridded_interpolant(x, c)(q), vec(g), err_msg=str(c))
