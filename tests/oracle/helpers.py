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

import os
from typing import Any

import numpy as np
from hypothesis import settings

#: Hypothesis settings for oracle tests (each example costs engine calls).
#: ``PYPROTEUS_ORACLE_EXAMPLES=500 pytest -m oracle`` for a deeper sweep.
ORACLE_SETTINGS = settings(
    max_examples=int(os.environ.get("PYPROTEUS_ORACLE_EXAMPLES", "25")), deadline=None
)

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

    The engine returns 1×1 MATLAB values as Python scalars; they become 1×1
    arrays here, so sizes always match MATLAB. ``dtype`` must not drop an
    imaginary part.
    """
    out = np.asarray(x)
    if out.ndim == 0 and isinstance(x, bool | int | float | complex):
        out = out.reshape(1, 1)
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


def batch(eng: Any, fn: str, *arglists: list[Any], nargout: int = 1) -> list[list[np.ndarray]]:
    """Evaluate ``fn`` on many cases in one engine call (``cellfun``, non-uniform output).

    ``fn`` is a MATLAB function handle expression, e.g. ``"@(x) sort(x,'descend')"``.
    Each ``arglists[j]`` holds the j-th argument of every case (converted with
    :func:`to_matlab`). Returns ``nargout`` lists with one NumPy array per case.
    """
    n = {len(a) for a in arglists}
    if len(n) != 1:
        raise ValueError("all argument lists must have the same length")
    names = []
    for j, args in enumerate(arglists):
        name = f"pyp_in{j}__"
        eng.workspace[name] = [to_matlab(a) for a in args]
        names.append(name)
    outs = [f"pyp_out{k}__" for k in range(nargout)]
    eng.eval(
        f"[{', '.join(outs)}] = cellfun({fn}, {', '.join(names)}, 'UniformOutput', false);",
        nargout=0,
    )
    result = [[from_matlab(v) for v in eng.workspace[o]] for o in outs]
    eng.eval(f"clear {' '.join(names + outs)};", nargout=0)
    return result


def assert_same_float(actual: Any, expected: Any, err_msg: str = "") -> None:
    """Bitwise float equality: same shape, same values, NaN == NaN, same sign of zero."""
    a = np.asarray(actual)
    e = np.asarray(expected)
    assert a.shape == e.shape, f"shape {a.shape} != {e.shape} {err_msg}"
    np.testing.assert_array_equal(a, e, err_msg=err_msg)
    if a.dtype.kind == "f":
        zeros = (a == 0) & (e == 0)
        assert np.array_equal(np.signbit(a[zeros]), np.signbit(e[zeros])), f"-0 vs +0 {err_msg}"


def assert_rel(actual: Any, expected: Any, rel: float = 1e-12, err_msg: str = "") -> None:
    """|a - e| <= rel * max|e| (ROADMAP §4 class for signals), NaN positions equal."""
    a = np.asarray(actual)
    e = np.asarray(expected)
    assert a.shape == e.shape, f"shape {a.shape} != {e.shape} {err_msg}"
    nan_a, nan_e = np.isnan(a), np.isnan(e)
    assert np.array_equal(nan_a, nan_e), f"NaN positions differ {err_msg}"
    ok = ~nan_e
    if not ok.any():
        return
    scale = np.max(np.abs(e[ok]))
    err = np.max(np.abs(a[ok] - e[ok]))
    assert err <= rel * max(scale, np.finfo(float).tiny), (
        f"max abs error {err:.3g} > {rel:g} * {scale:.3g} {err_msg}"
    )
