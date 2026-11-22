"""Rebuild MATLAB values from ``pyp_flatten`` records (see ``matlab/pyp_flatten.m``).

Pure Python (no MATLAB): the records are what the MATLAB Engine returns for
``pyp_flatten(value)``, i.e. a list of dicts with keys ``path``, ``class``,
``size``, ``fields``, ``sparse``, ``data``. The engine turns 1×1 arrays into
Python scalars and blurs cell/struct shapes, so the exact class and size come
from each record and are enforced here.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import numpy as np

from pyproteus.io.reference_format import (
    DTYPE_OF_CLASS,
    FunctionHandle,
    MatlabString,
    StructArray,
)


class RecordError(ValueError):
    """Records are inconsistent (wrong order, path, size or class)."""


def _size(rec: Mapping[str, Any]) -> tuple[int, ...]:
    size = np.asarray(rec["size"], dtype=float).reshape(-1)
    if size.size < 2 or np.any(size < 0) or np.any(size != np.floor(size)):
        raise RecordError(f"bad size {rec['size']!r} at {rec['path']}")
    return tuple(int(s) for s in size)


def _strlist(x: Any) -> list[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    if hasattr(x, "size") and getattr(x, "size", 1) == 0:  # empty matlab.double
        return []
    return [str(s) for s in x]


def _array(rec: Mapping[str, Any], size: tuple[int, ...]) -> np.ndarray:
    cls = rec["class"]
    dtype = DTYPE_OF_CLASS[cls]
    data = rec["data"]
    arr = np.asarray(data)
    complex_ = np.iscomplexobj(arr)
    if complex_ and cls not in ("double", "single"):
        raise RecordError(f"complex {cls} at {rec['path']}")
    target = np.result_type(dtype, np.complex64) if complex_ else dtype
    n = int(np.prod(size))
    if arr.shape != size:
        # Only scalars (engine returns Python scalars) and empties may be reshaped;
        # anything else would reorder elements silently.
        if arr.size == n and n <= 1:
            arr = arr.reshape(size)
        else:
            raise RecordError(f"data shape {arr.shape} != MATLAB size {size} at {rec['path']}")
    if arr.dtype != target:
        if arr.size and not np.can_cast(arr.dtype, target, casting="same_kind"):
            raise RecordError(f"data dtype {arr.dtype} cannot hold class {cls} at {rec['path']}")
        cast = arr.astype(target)
        if arr.size and not np.array_equal(cast, arr, equal_nan=arr.dtype.kind in "fc"):
            raise RecordError(f"casting {arr.dtype} to {cls} changed values at {rec['path']}")
        arr = cast
    if rec.get("sparse"):
        import scipy.sparse as sp

        if arr.ndim != 2:
            raise RecordError(f"sparse value must be 2-D at {rec['path']}")
        return sp.csc_array(arr)
    return np.ascontiguousarray(arr)


def _build(it: Iterator[Mapping[str, Any]], path: list[str]) -> Any:
    try:
        rec = next(it)
    except StopIteration:
        raise RecordError(f"records end early, expected node at {path}") from None
    if _strlist(rec["path"]) != path:
        raise RecordError(f"expected node at {path}, got {rec['path']}")
    cls = rec["class"]
    size = _size(rec)
    n = int(np.prod(size))
    if cls == "struct":
        fields = _strlist(rec.get("fields"))
        if size == (1, 1):
            return {f: _build(it, [*path, f]) for f in fields}
        els = np.empty(n, dtype=object)
        for k in range(n):
            el = _build(it, [*path, str(k)])
            if not isinstance(el, dict) or list(el) != fields:
                raise RecordError(f"struct array element {k} at {path} has wrong fields")
            els[k] = el
        return StructArray(els.reshape(size, order="F"), tuple(fields))
    if cls == "cell":
        els = np.empty(n, dtype=object)
        for k in range(n):
            els[k] = _build(it, [*path, str(k)])
        return els.reshape(size, order="F")
    if cls in ("char", "string", "function_handle"):
        data = rec["data"]
        text = data if isinstance(data, str) else ""
        if not isinstance(data, str) and np.asarray(data).size:
            raise RecordError(f"text node without str data at {path}")
        if cls == "char":
            if size != ((1, len(text)) if text else size) or (not text and n != 0):
                raise RecordError(f"char size {size} does not match {text!r} at {path}")
            return text
        return (MatlabString if cls == "string" else FunctionHandle)(text)
    if cls in DTYPE_OF_CLASS:
        return _array(rec, size)
    raise RecordError(f"unsupported class {cls!r} at {path}")


def records_to_value(records: Sequence[Mapping[str, Any]]) -> Any:
    """Rebuild the value flattened by ``pyp_flatten`` (Python representation of
    :mod:`pyproteus.io.reference_format`)."""
    it = iter(records)
    value = _build(it, [])
    rest = sum(1 for _ in it)
    if rest:
        raise RecordError(f"{rest} records left over")
    return value
