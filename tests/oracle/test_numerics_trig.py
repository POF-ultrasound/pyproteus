"""Oracle tests: numerics.sind/cosd against MATLAB sind/cosd.

Integer multiples of 90 degrees: exact (documented MATLAB behaviour; sign of zero
not compared). Other angles: D-13 float class, 1e-9 rel + 1e-15 abs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from pyproteus.numerics import cosd, sind
from tests.oracle.helpers import ORACLE_SETTINGS, from_matlab, row, to_matlab


def matlab(eng: Any, fn: str, x: np.ndarray) -> np.ndarray:
    eng.workspace["pyp_x__"] = to_matlab(row(x))
    out = from_matlab(eng.eval(f"{fn}(pyp_x__)", nargout=1)).ravel()
    eng.eval("clear pyp_x__", nargout=0)
    return out


def check(eng: Any, x: np.ndarray) -> None:
    for fn, py in (("sind", sind), ("cosd", cosd)):
        e, a = matlab(eng, fn, x), py(x)
        assert np.array_equal(np.isnan(a), np.isnan(e)), fn
        ok = ~np.isnan(e)
        exact = ok & (np.mod(x, 90.0) == 0)
        np.testing.assert_array_equal(a[exact], e[exact], err_msg=fn)
        np.testing.assert_allclose(a[ok], e[ok], rtol=1e-9, atol=1e-15, err_msg=fn)


def test_multiples_of_90(matlab_engine: Any) -> None:
    x = 90.0 * np.concatenate([np.arange(-20, 21), [1e6, -1e6 + 1, 2.0**40, 1e15]])
    check(matlab_engine, x)


def test_special_and_generic(matlab_engine: Any) -> None:
    x = np.array(
        [
            -0.0,
            30,
            45,
            60,
            135,
            -30,
            1e-300,
            89.999999999,
            90.000000001,
            12.5,
            1e20,
            -1e20,
            np.inf,
            -np.inf,
            np.nan,
            360 * 1e10 + 30,
        ]
    )
    check(matlab_engine, x)


@ORACLE_SETTINGS
@given(st.lists(st.floats(-1e6, 1e6, allow_nan=False), min_size=1, max_size=50))
def test_random(matlab_engine: Any, x: list[float]) -> None:
    check(matlab_engine, np.array(x))
