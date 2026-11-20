"""The NumPy <-> MATLAB conversion helpers keep shape, order, dtype and specials."""

from __future__ import annotations

import numpy as np
import pytest

from tests.oracle.helpers import col, from_matlab, get, put, row, to_matlab, vec


def test_oracle_commit_recorded(oracle) -> None:
    assert len(oracle.commit) == 40


def test_engine_version(matlab_engine, oracle) -> None:
    assert "R2025a" in oracle.matlab_version


def test_row_col_orientation(matlab_engine) -> None:
    x = np.array([3.0, 1.0, 2.0])
    assert from_matlab(matlab_engine.size(to_matlab(row(x)))).tolist() == [[1, 3]]
    assert from_matlab(matlab_engine.size(to_matlab(col(x)))).tolist() == [[3, 1]]
    with pytest.raises(ValueError, match="ambiguous"):
        to_matlab(x)


def test_nd_index_order(matlab_engine) -> None:
    a = np.arange(2 * 3 * 5, dtype=float).reshape(2, 3, 5)
    put(matlab_engine, a=a)
    assert matlab_engine.eval("a(2,3,4)") == a[1, 2, 3]
    # column-major linear order on the MATLAB side
    lin = vec(matlab_engine.eval("a(:)"))
    np.testing.assert_array_equal(lin, a.ravel(order="F"))
    np.testing.assert_array_equal(get(matlab_engine, "a"), a)
    # Fortran-ordered input arrays are handled the same way
    put(matlab_engine, b=np.asfortranarray(a))
    assert matlab_engine.eval("isequal(a,b)")


@pytest.mark.parametrize(
    ("dtype", "cls"),
    [
        (np.float64, "double"),
        (np.float32, "single"),
        (np.int32, "int32"),
        (np.int64, "int64"),
        (np.uint8, "uint8"),
        (np.bool_, "logical"),
    ],
)
def test_dtypes_roundtrip(matlab_engine, dtype, cls) -> None:
    a = np.array([[1, 0, 2], [3, 4, 0]]).astype(dtype)
    put(matlab_engine, a=a)
    assert matlab_engine.eval("class(a)") == cls
    back = get(matlab_engine, "a")
    assert back.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(back, a)


def test_complex_and_specials(matlab_engine) -> None:
    a = np.array([[1 + 2j, np.nan, -np.inf], [np.inf, -0.0, 3j]])
    put(matlab_engine, a=a)
    assert matlab_engine.eval("isreal(a)") is False
    back = get(matlab_engine, "a")
    np.testing.assert_array_equal(back, a)
    assert np.signbit(back[1, 1].real)


def test_empty(matlab_engine) -> None:
    put(matlab_engine, e=np.zeros((0, 3)))
    assert from_matlab(matlab_engine.size(matlab_engine.workspace["e"])).tolist() == [[0, 3]]


def test_scalars(matlab_engine) -> None:
    put(matlab_engine, s=np.float64(2.5), i=np.int32(7))
    assert matlab_engine.eval("class(s)") == "double"
    assert matlab_engine.eval("class(i)") == "int32"
    assert matlab_engine.eval("isequal(size(s),[1 1])")
