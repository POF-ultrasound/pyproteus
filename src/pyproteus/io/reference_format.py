"""Reference HDF5 files: MATLAB values written and read with MATLAB's shapes.

Format specification: ``docs/REFERENCE_DATA.md``. This module only needs
h5py; the MATLAB side (export through the MATLAB Engine) lives in
``tools/export_reference/``.

Python representation of MATLAB values (what :func:`read_reference` returns
and :func:`write_reference` accepts):

=============================  =============================================
MATLAB                         Python
=============================  =============================================
numeric / logical array        ``np.ndarray`` with ``shape == size(x)``
                               (always >= 2-D; a scalar is ``(1, 1)``) and
                               ``a[i, j, k] == x(i+1, j+1, k+1)``
sparse matrix                  ``scipy.sparse.csc_array``
char row vector                ``str`` (any empty char becomes ``''``, 0×0)
string scalar                  :class:`MatlabString`
function handle                :class:`FunctionHandle` (``func2str`` text)
scalar struct                  ``dict`` (field order kept)
struct array (not 1×1)         :class:`StructArray`
cell array                     ``np.ndarray`` of ``dtype=object``
=============================  =============================================

Values are never adjusted: 1-based MATLAB indices (``find`` output...) stay
1-based, and a MATLAB linear index ``k`` addresses ``a.ravel(order="F")[k-1]``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

#: Value of the root attribute ``format``.
FORMAT_NAME = "pyproteus-reference"
#: Value of the root attribute ``format_version``; bump on incompatible changes.
FORMAT_VERSION = 1
#: Value of the root attribute ``array_layout`` (see docs/REFERENCE_DATA.md).
ARRAY_LAYOUT = "matlab-size"

#: Root attributes every reference file must carry (provenance).
REQUIRED_ATTRS: dict[str, type | tuple[type, ...]] = {
    "name": str,
    "kind": str,  # "module" or "full_run"
    "module": str,
    "function": str,
    "description": str,
    "call": str,
    "matlab_version": str,
    "proteus_version": str,
    "oracle_commit": str,
    "oracle_dirty": bool,
    "kwave_matlab_version": str,  # "" when k-Wave is not used
    "kwave_matlab_path": str,
    "kwave_solver": str,
    "kwave_binary_path": str,
    "kwave_binary_version": str,
    "kwave_binary_sha256": str,
    "seeds": dict,  # stored as a JSON object string
    "matlab_path_extra": list,  # non-PROTEUS folders put on the MATLAB path (shims...)
    "created_utc": str,
    "exporter_version": str,
    "exporter_commit": str,
    "platform": str,
}
KINDS = ("module", "full_run")

_CLASS_OF_DTYPE = {
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
#: MATLAB numeric/logical class -> NumPy dtype (real).
DTYPE_OF_CLASS = {
    "double": np.dtype(np.float64),
    "single": np.dtype(np.float32),
    "logical": np.dtype(np.bool_),
    "int8": np.dtype(np.int8),
    "int16": np.dtype(np.int16),
    "int32": np.dtype(np.int32),
    "int64": np.dtype(np.int64),
    "uint8": np.dtype(np.uint8),
    "uint16": np.dtype(np.uint16),
    "uint32": np.dtype(np.uint32),
    "uint64": np.dtype(np.uint64),
}
_TEXT_CLASSES = ("char", "string", "function_handle")
_STR = h5py.string_dtype("utf-8")


def utf16_len(text: str) -> int:
    """Length of a MATLAB char (UTF-16 code units) holding ``text``."""
    return len(text.encode("utf-16-le")) // 2


class ReferenceFormatError(ValueError):
    """A file or value does not follow the reference format."""


class MatlabString(str):
    """A MATLAB ``string`` scalar (as opposed to a char vector)."""

    __slots__ = ()


class FunctionHandle(str):
    """A MATLAB function handle, kept as its ``func2str`` text."""

    __slots__ = ()


@dataclass(eq=False)
class StructArray:
    """A MATLAB struct array whose size is not 1×1.

    ``elements`` is an object array of shape ``size`` whose entries are dicts
    with exactly the keys ``fields`` (in that order).
    """

    elements: np.ndarray
    fields: tuple[str, ...]

    def __post_init__(self) -> None:
        self.elements = np.asarray(self.elements, dtype=object)
        self.fields = tuple(self.fields)
        if self.elements.ndim < 2:
            raise ReferenceFormatError("StructArray elements must be >= 2-D (MATLAB size)")
        if self.elements.shape == (1, 1):
            raise ReferenceFormatError("a 1x1 struct is a dict, not a StructArray")
        for el in self.elements.ravel(order="F"):
            if not isinstance(el, Mapping) or tuple(el) != self.fields:
                raise ReferenceFormatError(f"struct array element fields != {self.fields}")

    @property
    def shape(self) -> tuple[int, ...]:
        return self.elements.shape


@dataclass
class Reference:
    """Content of one reference file."""

    path: Path
    attrs: dict[str, Any]
    inputs: dict[str, Any]
    outputs: dict[str, Any]
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.attrs["name"]


# ----------------------------------------------------------------------------- write


def _size_attr(shape: tuple[int, ...]) -> np.ndarray:
    return np.asarray(shape, dtype=np.int64)


def _write_value(parent: h5py.Group, name: str, value: Any) -> None:
    if isinstance(value, StructArray):
        g = parent.create_group(name, track_order=True)
        g.attrs["matlab_class"] = "struct"
        g.attrs["matlab_size"] = _size_attr(value.shape)
        g.attrs["matlab_fields"] = np.asarray(value.fields, dtype=_STR)
        for k, el in enumerate(value.elements.ravel(order="F")):
            _write_value(g, str(k), dict(el))
        return
    if isinstance(value, Mapping):
        g = parent.create_group(name, track_order=True)
        g.attrs["matlab_class"] = "struct"
        g.attrs["matlab_size"] = _size_attr((1, 1))
        keys = list(value)
        for key in keys:
            if not isinstance(key, str) or not key or not key[0].isalpha():
                raise ReferenceFormatError(f"invalid MATLAB field name {key!r}")
        g.attrs["matlab_fields"] = np.asarray(keys, dtype=_STR)
        for key in keys:
            _write_value(g, key, value[key])
        return
    if isinstance(value, str):
        cls = (
            "function_handle"
            if isinstance(value, FunctionHandle)
            else "string"
            if isinstance(value, MatlabString)
            else "char"
        )
        d = parent.create_dataset(name, data=str(value), dtype=_STR)
        d.attrs["matlab_class"] = cls
        size = ((1, utf16_len(value)) if value else (0, 0)) if cls == "char" else (1, 1)
        d.attrs["matlab_size"] = _size_attr(size)
        return
    if _is_sparse(value):
        dense = np.asarray(value.toarray())
        if dense.ndim != 2:
            raise ReferenceFormatError("sparse values must be 2-D")
        _write_array(parent, name, dense, sparse=True)
        return
    if isinstance(value, np.ndarray) and value.dtype == object:
        if value.ndim < 2:
            raise ReferenceFormatError(f"{name}: cell array must be >= 2-D (MATLAB size)")
        g = parent.create_group(name, track_order=True)
        g.attrs["matlab_class"] = "cell"
        g.attrs["matlab_size"] = _size_attr(value.shape)
        for k, el in enumerate(value.ravel(order="F")):
            _write_value(g, str(k), el)
        return
    if isinstance(value, np.ndarray):
        _write_array(parent, name, value, sparse=False)
        return
    raise ReferenceFormatError(
        f"{name}: cannot store {type(value).__name__}; use a >= 2-D ndarray, str, dict, "
        "StructArray, object ndarray (cell) or scipy.sparse"
    )


def _is_sparse(value: Any) -> bool:
    try:
        import scipy.sparse as sp
    except ImportError:  # pragma: no cover
        return False
    return sp.issparse(value)


def _write_array(parent: h5py.Group, name: str, arr: np.ndarray, *, sparse: bool) -> None:
    if arr.ndim < 2:
        raise ReferenceFormatError(
            f"{name}: arrays must have MATLAB's size (>= 2-D), got shape {arr.shape}"
        )
    if arr.ndim > 2 and arr.shape[-1] == 1:
        raise ReferenceFormatError(
            f"{name}: shape {arr.shape} has a trailing singleton; MATLAB size never does"
        )
    try:
        cls = _CLASS_OF_DTYPE[arr.dtype]
    except KeyError:
        raise ReferenceFormatError(f"{name}: no MATLAB class for dtype {arr.dtype}") from None
    if sparse and cls not in ("double", "logical"):
        raise ReferenceFormatError(f"{name}: MATLAB sparse is double or logical only")
    # C-contiguous with dims == MATLAB size: h5py reads back the same shape and
    # a[i, j, k] == x(i+1, j+1, k+1). Zero-size datasets are allowed.
    d = parent.create_dataset(name, data=np.ascontiguousarray(arr))
    d.attrs["matlab_class"] = cls
    d.attrs["matlab_size"] = _size_attr(arr.shape)
    if sparse:
        d.attrs["matlab_sparse"] = True


def _check_attrs(attrs: Mapping[str, Any]) -> None:
    missing = [k for k in REQUIRED_ATTRS if k not in attrs]
    if missing:
        raise ReferenceFormatError(f"missing provenance attributes: {missing}")
    for k, t in REQUIRED_ATTRS.items():
        if not isinstance(attrs[k], t):
            raise ReferenceFormatError(f"attribute {k!r} must be {t}, got {type(attrs[k])}")
    if attrs["kind"] not in KINDS:
        raise ReferenceFormatError(f"kind must be one of {KINDS}")
    if not attrs["name"] or "/" in attrs["name"] or os.sep in attrs["name"]:
        raise ReferenceFormatError("name must be a non-empty flat file stem")
    for k, v in attrs["seeds"].items():
        if not isinstance(k, str) or isinstance(v, bool) or not isinstance(v, int):
            raise ReferenceFormatError("seeds must map names to integers")
    if not all(isinstance(p, str) for p in attrs["matlab_path_extra"]):
        raise ReferenceFormatError("matlab_path_extra must be a list of str")
    if attrs["kind"] == "full_run" and not attrs["kwave_binary_path"]:
        # D-10: every k-Wave reference records the binary it ran.
        raise ReferenceFormatError("full_run references must record kwave_binary_path")


def write_reference(
    path: str | os.PathLike,
    attrs: Mapping[str, Any],
    inputs: Mapping[str, Any],
    outputs: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write one reference file (overwrites). ``attrs`` must hold :data:`REQUIRED_ATTRS`.

    ``inputs`` are the MATLAB function's arguments in call order (as MATLAB
    received them), ``outputs`` its results; ``extra`` holds anything else
    worth keeping (intermediate values of a full run...).
    """
    _check_attrs(attrs)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    try:
        _write_file(tmp, attrs, inputs, outputs, extra)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)
    return path


def _write_file(
    tmp: Path,
    attrs: Mapping[str, Any],
    inputs: Mapping[str, Any],
    outputs: Mapping[str, Any],
    extra: Mapping[str, Any] | None,
) -> None:
    with h5py.File(tmp, "w", track_order=True) as f:
        f.attrs["format"] = FORMAT_NAME
        f.attrs["format_version"] = np.int64(FORMAT_VERSION)
        f.attrs["array_layout"] = ARRAY_LAYOUT
        for k, v in attrs.items():
            if k in ("format", "format_version", "array_layout"):
                raise ReferenceFormatError(f"attribute {k!r} is reserved")
            if isinstance(v, dict):
                f.attrs[k] = json.dumps(v, sort_keys=True)
            elif isinstance(v, list):
                f.attrs[k] = json.dumps(v)
            elif isinstance(v, bool):
                f.attrs[k] = np.bool_(v)
            elif isinstance(v, str | int | float):
                f.attrs[k] = v
            else:
                raise ReferenceFormatError(f"attribute {k!r}: unsupported type {type(v)}")
        _write_value(f, "inputs", dict(inputs))
        _write_value(f, "outputs", dict(outputs))
        if extra:
            _write_value(f, "extra", dict(extra))


# ----------------------------------------------------------------------------- read


def _text(x: Any) -> str:
    return x.decode("utf-8") if isinstance(x, bytes) else str(x)


def _read_value(node: h5py.Group | h5py.Dataset) -> Any:
    try:
        cls = _text(node.attrs["matlab_class"])
        size = tuple(int(s) for s in node.attrs["matlab_size"])
    except KeyError:
        raise ReferenceFormatError(f"{node.name}: missing matlab_class/matlab_size") from None
    if isinstance(node, h5py.Group):
        if cls == "struct":
            fields = tuple(_text(s) for s in node.attrs["matlab_fields"])
            if size == (1, 1) and set(node) == set(fields):
                return {k: _read_value(node[k]) for k in fields}
            n = int(np.prod(size))
            if set(node) != {str(k) for k in range(n)}:
                raise ReferenceFormatError(f"{node.name}: struct array elements missing")
            els = np.empty(n, dtype=object)
            for k in range(n):
                els[k] = _read_value(node[str(k)])
            return StructArray(els.reshape(size, order="F"), fields)
        if cls == "cell":
            n = int(np.prod(size))
            if set(node) != {str(k) for k in range(n)}:
                raise ReferenceFormatError(f"{node.name}: cell elements missing")
            els = np.empty(n, dtype=object)
            for k in range(n):
                els[k] = _read_value(node[str(k)])
            return els.reshape(size, order="F")
        raise ReferenceFormatError(f"{node.name}: group with class {cls!r}")
    if cls in _TEXT_CLASSES:
        s = _text(node[()])
        want = ((1, utf16_len(s)) if s else (0, 0)) if cls == "char" else (1, 1)
        if size != want:
            raise ReferenceFormatError(f"{node.name}: matlab_size {size} != text size {want}")
        return {"char": str, "string": MatlabString, "function_handle": FunctionHandle}[cls](s)
    arr = node[()]
    if arr.shape != size:
        raise ReferenceFormatError(f"{node.name}: dataset shape {arr.shape} != matlab_size {size}")
    expected = DTYPE_OF_CLASS.get(cls)
    allowed = {expected}
    if cls in ("double", "single"):
        allowed.add(np.result_type(expected, np.complex64))
    if expected is None or arr.dtype not in allowed:
        raise ReferenceFormatError(f"{node.name}: dtype {arr.dtype} does not match class {cls}")
    if node.attrs.get("matlab_sparse", False):
        import scipy.sparse as sp

        return sp.csc_array(arr)
    return arr


def _read_attrs(f: h5py.File) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for k, v in f.attrs.items():
        if isinstance(v, bytes):
            v = v.decode("utf-8")
        if isinstance(v, np.bool_):
            v = bool(v)
        elif isinstance(v, np.integer):
            v = int(v)
        elif isinstance(v, np.floating):
            v = float(v)
        if k in ("seeds", "matlab_path_extra") and isinstance(v, str):
            v = json.loads(v)
        attrs[k] = v
    return attrs


def read_reference(path: str | os.PathLike) -> Reference:
    """Read a reference file written by :func:`write_reference` (checks format and provenance)."""
    path = Path(path)
    with h5py.File(path, "r") as f:
        attrs = _read_attrs(f)
        if attrs.get("format") != FORMAT_NAME:
            raise ReferenceFormatError(f"{path}: not a pyproteus reference file")
        if attrs.get("format_version") != FORMAT_VERSION:
            raise ReferenceFormatError(
                f"{path}: format_version {attrs.get('format_version')} "
                f"not supported (expected {FORMAT_VERSION})"
            )
        if attrs.get("array_layout") != ARRAY_LAYOUT:
            raise ReferenceFormatError(f"{path}: unknown array_layout {attrs.get('array_layout')}")
        _check_attrs(attrs)
        inputs = _read_value(f["inputs"])
        outputs = _read_value(f["outputs"])
        extra = _read_value(f["extra"]) if "extra" in f else {}
    return Reference(path=path, attrs=attrs, inputs=inputs, outputs=outputs, extra=extra)
