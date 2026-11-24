"""Oracle tests: ``compute_gaussian_distribution`` against MATLAB.

D-13 float class: ``|a - e| <= 1e-9*|e| + 1e-14*max|e|`` (``R`` from ``linspace``,
``P`` from ``exp`` and ``sum``); NaN positions equal (PDI = 0, Q-84).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pyproteus import microbubble as pmb
from pyproteus.config.models import Distribution, Microbubble
from tests.oracle.helpers import ORACLE_SETTINGS, from_matlab


def assert_float(actual: Any, expected: Any, err_msg: str) -> None:
    a, e = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    assert a.shape == e.shape, f"shape {a.shape} != {e.shape} {err_msg}"
    assert np.array_equal(np.isnan(a), np.isnan(e)), f"NaN positions differ {err_msg}"
    fin = np.isfinite(e)
    if not fin.any():
        return
    err = np.abs(a[fin] - e[fin])
    bound = 1e-9 * np.abs(e[fin]) + 1e-14 * np.max(np.abs(e[fin]))
    assert np.all(err <= bound), f"max excess {np.max(err - bound):.3g} {err_msg}"


def check(eng: Any, cases: list[tuple[float, float]]) -> None:
    eng.workspace["pyp_c__"] = [
        {"Distribution": {"MeanRadius": float(r0), "PDI": float(pdi)}} for r0, pdi in cases
    ]
    eng.eval(
        "[pyp_R__, pyp_P__] = cellfun(@compute_gaussian_distribution, pyp_c__,"
        " 'UniformOutput', false);",
        nargout=0,
    )
    rs, ps = eng.workspace["pyp_R__"], eng.workspace["pyp_P__"]
    eng.eval("clear pyp_c__ pyp_R__ pyp_P__", nargout=0)
    for (r0, pdi), r, p in zip(cases, rs, ps, strict=True):
        mb = Microbubble(Distribution=Distribution(MeanRadius=r0, PDI=pdi))
        ra, pa = pmb.compute_gaussian_distribution(mb)
        assert_float(ra, from_matlab(r), f"R r0={r0} pdi={pdi}")
        assert_float(pa, from_matlab(p), f"P r0={r0} pdi={pdi}")


# Mean radius on a grid node (k = 499: 0.5e-6 + 499*5.5e-6/999), at the ends,
# outside the range (tails underflow), tiny; PDI typical, tiny, huge, negative, zero.
R0 = [1e-6, 2.5e-6, 0.5e-6 + 499 * 5.5e-6 / 999, 0.5e-6, 6e-6, 10e-6, 0.1e-6, 3.3e-6]
PDI = [5.0, 10.0, 37.5, 100.0, 0.1, -10.0, 0.0, 1e4]


@pytest.mark.parametrize("r0", R0)
def test_fixed_cases(matlab_engine: Any, r0: float) -> None:
    check(matlab_engine, [(r0, pdi) for pdi in PDI])


def test_default_settings(matlab_engine: Any) -> None:
    eng = matlab_engine
    eng.eval("[pyp_R__, pyp_P__] = compute_gaussian_distribution(reset_microbubble());", nargout=0)
    r, p = from_matlab(eng.workspace["pyp_R__"]), from_matlab(eng.workspace["pyp_P__"])
    eng.eval("clear pyp_R__ pyp_P__", nargout=0)
    from pyproteus import config

    ra, pa = pmb.compute_gaussian_distribution(config.reset_microbubble())
    assert_float(ra, r, "R default")
    assert_float(pa, p, "P default")
    assert pa.shape == (1, 1000)


@ORACLE_SETTINGS
@given(st.lists(st.tuples(st.floats(0.2e-6, 8e-6), st.floats(0.5, 200)), min_size=1, max_size=20))
def test_random(matlab_engine: Any, cases: list[tuple[float, float]]) -> None:
    check(matlab_engine, cases)
