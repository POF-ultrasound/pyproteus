"""Convert arrays between NumPy and the MATLAB Engine without shape surprises.

MATLAB has no 1-D arrays. Passing a 1-D NumPy array silently becomes a 1×N
row, which changes the behaviour of shape-sensitive MATLAB functions
(``sort``, ``find``, ``hilbert``, ``interp1`` output orientation...). So
:func:`to_matlab` refuses 1-D input: wrap it in :func:`row` or :func:`col`.

Indexing is preserved both ways: ``A[i, j, k]`` in NumPy is ``A(i+1, j+1, k+1)``
in MATLAB (the engine copies memory in the right order). Values coming back
keep MATLAB's size (a vector stays 2-D); use :func:`vec` to flatten a vector.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_CONVERTERS = {
    np.dtype(np.float64): "double",
    np.dtype(np.float32): "single",
    np.dtype(np.int8): "int8",
    np.dtype(np.int16): "int16",
    np.dtype(np.int32): "int32",
    np.dtype(np.int64): "int64",
    np.dtype(np.uint8): "uint8",
    np.dtype(np.uint16): "uint16",
    np.dtype(np.uint32): "uint32",
    np.dtype(np.uint64): "uint64",
    np.dtype(np.bool_): "logical",
    np.dtype(np.complex128): "double",
    np.dtype(np.complex64): "single",
}


def _as_array(a: Any) -> np.ndarray:
    """np.asarray, but Python ints/lists of ints become double (MATLAB literals are double).

    Integer *NumPy* arrays keep their integer class: that is an explicit choice.
    """
    if isinstance(a, np.ndarray | np.generic):
        return np.asarray(a)
    arr = np.asarray(a)
    if arr.dtype.kind in "iu":
        arr = arr.astype(np.float64)
    return arr


def _vector(a: Any) -> np.ndarray:
    arr = _as_array(a)
    if arr.ndim > 1:
        raise ValueError(f"row()/col() expect a 1-D array or scalar, got shape {arr.shape}")
    return arr


class Row:
    """Marks a 1-D array to be sent as a MATLAB 1×N row vector."""

    def __init__(self, a: Any) -> None:
        self.a = _vector(a).reshape(1, -1)


class Col:
    """Marks a 1-D array to be sent as a MATLAB N×1 column vector."""

    def __init__(self, a: Any) -> None:
        self.a = _vector(a).reshape(-1, 1)


def row(a: Any) -> Row:
    return Row(a)


def col(a: Any) -> Col:
    return Col(a)


def to_matlab(a: Any) -> Any:
    """Convert a NumPy array (≥ 2-D, or wrapped in row/col) to a MATLAB array.

    Python float, bool and str pass through (double, logical, char). Python
    ints and lists of ints are sent as **double**, like MATLAB literals (the
    engine would otherwise send int64, and ``int64(3)/2`` is 2 in MATLAB).
    NumPy scalars and 0-d arrays become 1×1 MATLAB arrays of the same class;
    integer NumPy arrays keep their integer class.
    """
    import matlab

    if isinstance(a, Row | Col):
        arr = a.a
    elif isinstance(a, bool | float | str):
        return a
    elif isinstance(a, int):
        return float(a)
    else:
        arr = _as_array(a)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            raise ValueError("1-D array is ambiguous in MATLAB: wrap it in row(...) or col(...)")
    try:
        cls = getattr(matlab, _CONVERTERS[arr.dtype])
    except KeyError:
        raise TypeError(f"no MATLAB class for dtype {arr.dtype}") from None
    if arr.size == 0:
        # The engine cannot build empty N-D arrays from a buffer; use size=.
        return cls(size=arr.shape, is_complex=np.iscomplexobj(arr))
    return cls(np.ascontiguousarray(arr), is_complex=np.iscomplexobj(arr))


def from_matlab(x: Any, dtype: Any = None) -> np.ndarray:
    """Convert a MATLAB value to a NumPy array with MATLAB's size.

    Exception: the engine returns 1×1 MATLAB values as Python scalars, which
    become 0-d arrays here. ``dtype`` must not drop an imaginary part.
    """
    out = np.asarray(x)
    if dtype is not None:
        if np.iscomplexobj(out) and not np.issubdtype(np.dtype(dtype), np.complexfloating):
            raise TypeError("refusing to drop the imaginary part")
        out = out.astype(dtype, copy=False)
    return out


def vec(x: Any) -> np.ndarray:
    """MATLAB vector (1×N, N×1, or scalar) → 1-D NumPy array. Fails on matrices."""
    out = from_matlab(x)
    if out.ndim > 2 or (out.ndim == 2 and min(out.shape) > 1):
        raise ValueError(f"not a vector: MATLAB size {out.shape}")
    return out.reshape(-1)


def put(eng: Any, **variables: Any) -> None:
    """Assign variables in the MATLAB base workspace (converted with to_matlab)."""
    for name, value in variables.items():
        eng.workspace[name] = to_matlab(value)


def get(eng: Any, *names: str) -> np.ndarray | tuple[np.ndarray, ...]:
    """Read variables from the MATLAB base workspace as NumPy arrays."""
    out = tuple(from_matlab(eng.workspace[n]) for n in names)
    return out[0] if len(out) == 1 else out
