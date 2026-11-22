"""JSON encoding of MATLAB value trees and settings (lossless).

Readable where possible, tagged where needed, always lossless:

=====================================  =======================================
MATLAB value (tree)                    JSON
=====================================  =======================================
1×1 struct (``dict``)                  object
char (``str``)                         string
1×1 double                             number (``Infinity``/``NaN`` literals)
1×1 logical                            ``true`` / ``false``
real double M×N, M ≥ 1, N ≥ 1          list of rows: ``[[1.0, 2.0]]`` is 1×2,
                                       ``[[1.0], [2.0]]`` is 2×1
0×0 double (``[]``)                    ``[]``
anything else (other classes, complex, ``{"__matlab__": class, "size": [...],
N-D, other empty sizes, logical          "data": [...]}``: data in MATLAB
arrays, cells, struct arrays)            column-major order; complex as
                                         ``[re, im]`` pairs
char matrix (> 1 row)                  ``{"__matlab__": "char", "rows": [...]}``
=====================================  =======================================

Doubles are written with Python's shortest round-trip ``repr``, so values
survive bit for bit. ``Infinity``/``NaN`` are not standard JSON but are read by
Python's :mod:`json` (``LateralFocus = Inf`` is a default setting).

Settings files hold ``{"format": "pyproteus-settings", "version": 1,
"Transducer": {...}, ...}``.
"""

from __future__ import annotations

import json
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from .matfile import StructArray
from .models import Settings

__all__ = [
    "FORMAT",
    "VERSION",
    "dumps",
    "load_settings_json",
    "save_settings_json",
    "settings_from_json",
    "settings_to_json",
    "tree_from_jsonable",
    "tree_to_jsonable",
]

FORMAT = "pyproteus-settings"
VERSION = 1
_TAG = "__matlab__"

_CLASS_OF = {
    np.dtype(np.float64): "double",
    np.dtype(np.float32): "single",
    np.dtype(np.complex128): "double",
    np.dtype(np.complex64): "single",
    np.dtype(np.bool_): "logical",
    np.dtype(np.int8): "int8",
    np.dtype(np.int16): "int16",
    np.dtype(np.int32): "int32",
    np.dtype(np.int64): "int64",
    np.dtype(np.uint8): "uint8",
    np.dtype(np.uint16): "uint16",
    np.dtype(np.uint32): "uint32",
    np.dtype(np.uint64): "uint64",
}
_DTYPE_OF = {
    "double": np.float64,
    "single": np.float32,
    "logical": np.bool_,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
}


# ------------------------------------------------------------------ tree codec


def tree_to_jsonable(v: Any) -> Any:
    """MATLAB value tree → JSON-serialisable Python value (see module docstring)."""
    if isinstance(v, dict):
        return {str(k): tree_to_jsonable(x) for k, x in v.items()}
    if isinstance(v, str):
        return v
    if isinstance(v, bool | np.bool_):
        return bool(v)
    if isinstance(v, int | float) and not isinstance(v, bool):
        return float(v)
    if isinstance(v, StructArray):
        data = [tree_to_jsonable(e) for e in v.elements.reshape(-1, order="F")]
        return {_TAG: "struct", "size": list(v.shape), "fields": v.fields, "data": data}
    if not isinstance(v, np.ndarray):
        raise TypeError(f"not a MATLAB value tree leaf: {type(v)!r}")
    if v.dtype.kind == "U":  # char matrix rows
        return {_TAG: "char", "rows": [str(s) for s in v.reshape(-1)]}
    if v.ndim < 2:
        raise ValueError(f"MATLAB arrays are ≥ 2-D, got shape {v.shape}")
    size = [int(n) for n in v.shape]
    if v.dtype == object:
        data = [tree_to_jsonable(e) for e in v.reshape(-1, order="F")]
        return {_TAG: "cell", "size": size, "data": data}
    cls = _CLASS_OF.get(v.dtype)
    if cls is None:
        raise TypeError(f"no MATLAB class for dtype {v.dtype}")
    if v.dtype == np.float64 and v.ndim == 2:
        if v.shape == (1, 1):
            return float(v[0, 0])
        if v.shape == (0, 0):
            return []
        if v.shape[0] >= 1 and v.shape[1] >= 1:
            return [[float(x) for x in r] for r in v]
    if v.dtype == np.bool_ and v.shape == (1, 1):
        return bool(v[0, 0])
    flat = v.reshape(-1, order="F")
    if np.iscomplexobj(v):
        data = [[float(z.real), float(z.imag)] for z in flat]
        return {_TAG: cls, "complex": True, "size": size, "data": data}
    if v.dtype.kind == "f":
        data = [float(x) for x in flat]
    elif v.dtype == np.bool_:
        data = [bool(x) for x in flat]
    else:
        data = [int(x) for x in flat]
    return {_TAG: cls, "size": size, "data": data}


def tree_from_jsonable(v: Any) -> Any:
    """Inverse of :func:`tree_to_jsonable`."""
    if isinstance(v, str):
        return v
    if isinstance(v, bool):
        return np.array([[v]], dtype=np.bool_)
    if isinstance(v, int | float):
        return np.array([[float(v)]])
    if isinstance(v, list):
        if len(v) == 0:
            return np.zeros((0, 0))
        if not all(isinstance(r, list) for r in v):
            raise ValueError("a MATLAB matrix is a list of rows (lists of numbers)")
        if len({len(r) for r in v}) != 1 or len(v[0]) == 0:
            raise ValueError("matrix rows must be non-empty and of equal length")
        for r in v:
            for x in r:
                if isinstance(x, bool) or not isinstance(x, int | float):
                    raise ValueError(f"matrix elements must be numbers, got {x!r}")
        return np.array(v, dtype=np.float64)
    if isinstance(v, dict):
        if _TAG not in v:
            return {k: tree_from_jsonable(x) for k, x in v.items()}
        return _decode_tagged(v)
    if v is None:
        raise ValueError("null is not a MATLAB value (absent fields are left out)")
    raise TypeError(f"unexpected JSON value {v!r}")


def _decode_tagged(v: dict[str, Any]) -> Any:
    cls = v[_TAG]
    if cls == "char":
        return np.array(v["rows"])
    size = tuple(int(n) for n in v["size"])
    data = v["data"]
    if cls in ("cell", "struct"):
        out = np.empty(len(data), dtype=object)
        for i, e in enumerate(data):
            out[i] = tree_from_jsonable(e)
        out = out.reshape(size, order="F")
        return StructArray(out, list(v["fields"])) if cls == "struct" else out
    dtype = _DTYPE_OF[cls]
    if v.get("complex"):
        ctype = np.complex64 if cls == "single" else np.complex128
        arr = np.array([complex(re, im) for re, im in data], dtype=ctype)
    else:
        arr = np.array(data, dtype=dtype)
    return np.ascontiguousarray(arr.reshape(size, order="F"))


def dumps(obj: Any, indent: int | None = 2) -> str:
    """``json.dumps`` with objects indented and arrays (lists) kept on one line."""
    if indent is None:
        return json.dumps(obj, allow_nan=True)
    return _dumps(obj, indent, 0)


def _dumps(obj: Any, indent: int, level: int) -> str:
    if isinstance(obj, dict) and obj:
        pad = " " * (indent * (level + 1))
        items = [f"{pad}{json.dumps(k)}: {_dumps(v, indent, level + 1)}" for k, v in obj.items()]
        return "{\n" + ",\n".join(items) + "\n" + " " * (indent * level) + "}"
    return json.dumps(obj, allow_nan=True, separators=(", ", ": "))


# -------------------------------------------------------------------- settings


def settings_to_json(settings: Settings, indent: int | None = 2) -> str:
    """Settings → JSON text (lossless; see module docstring)."""
    doc: dict[str, Any] = {"format": FORMAT, "version": VERSION}
    doc.update(tree_to_jsonable(settings.to_tree()))
    return dumps(doc, indent=indent)


def settings_from_json(text: str) -> Settings:
    """JSON text written by :func:`settings_to_json` → Settings."""
    doc = json.loads(text)
    if not isinstance(doc, dict) or doc.get("format") != FORMAT:
        raise ValueError(f"not a {FORMAT} JSON document")
    if doc.get("version") != VERSION:
        raise ValueError(f"unsupported {FORMAT} version {doc.get('version')!r}")
    unknown = set(doc) - {"format", "version", *Settings.GROUPS}
    if unknown:
        raise ValueError(f"unknown settings groups: {sorted(unknown)}")
    groups = {k: tree_from_jsonable(v) for k, v in doc.items() if k in Settings.GROUPS}
    return Settings.from_tree(groups)


def save_settings_json(settings: Settings, path: str | PathLike[str]) -> None:
    """Write settings to a JSON file (UTF-8)."""
    Path(path).write_text(settings_to_json(settings) + "\n", encoding="utf-8")


def load_settings_json(path: str | PathLike[str]) -> Settings:
    """Read settings from a JSON file written by :func:`save_settings_json`."""
    return settings_from_json(Path(path).read_text(encoding="utf-8"))
