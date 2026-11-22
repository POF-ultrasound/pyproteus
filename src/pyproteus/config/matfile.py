"""Read MATLAB ``.mat`` files into plain Python values with MATLAB's sizes.

Every ``.mat`` file PROTEUS reads or writes goes through :func:`loadmat`. It
returns a *MATLAB value tree*:

=====================  ==========================================================
MATLAB                 Python
=====================  ==========================================================
1×1 struct             ``dict`` (field order kept)
struct array (≠ 1×1)   :class:`StructArray` (object array of ``dict`` + field names)
cell array             ``numpy`` object array, MATLAB size
char row vector        ``str`` (``''`` of any empty size becomes ``""``)
char matrix (> 1 row)  ``numpy`` array of ``str``, one per row
numeric, logical       ``numpy`` array, MATLAB size (never squeezed, ≥ 2-D),
                       MATLAB class kept (double → float64, logical → bool, ...)
=====================  ==========================================================

Scalars stay 1×1 arrays: :mod:`pyproteus.config.models` decides which fields
are scalars. Traps handled here (each covered by a test):

* MATLAB v5 files store integer-valued doubles as small integer types; SciPy
  returns the *storage* type unless ``mat_dtype=True`` (``10`` → ``uint8``).
* ``mat_dtype=True`` silently drops the imaginary part of complex arrays
  (SciPy 1.15, B-30); complex arrays are taken from a second, raw read.
* ``squeeze_me`` is never used: it turns 1×N rows, N×1 columns and 1×1 structs
  into indistinguishable shapes.
* MATLAB (older versions) writes ``[]`` inside cells and struct fields as a
  0-byte matrix element, which SciPy reads as 1×0; MATLAB loads it as 0×0.
  :func:`_zero_byte_elements` finds these elements in the file (B-33).
* v7.3 files (HDF5) are read with h5py: dimensions are reversed, chars are
  UTF-16 codes, logicals are ``uint8`` with a ``MATLAB_class`` attribute, empty
  arrays store their size as data. MATLAB loads v7.3 variables in alphabetical
  order (HDF5 link order); so does :func:`loadmat`.

MATLAB ``string`` objects, function handles, other objects and sparse matrices
are not supported (PROTEUS never stores them) and raise ``TypeError``.
"""

from __future__ import annotations

import warnings
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["StructArray", "is_struct", "loadmat", "whosmat"]

_SKIP = ("__header__", "__version__", "__globals__")


class StructArray:
    """A MATLAB struct array that is not 1×1 (including empty ones).

    ``elements`` is an object array of ``dict`` with MATLAB's size; ``fields``
    keeps the field names (also when the array is empty).
    """

    __slots__ = ("elements", "fields")

    def __init__(self, elements: np.ndarray, fields: list[str]) -> None:
        if not isinstance(elements, np.ndarray) or elements.dtype != object:
            raise TypeError("elements must be an object array of dicts")
        if elements.ndim < 2:
            raise ValueError("MATLAB arrays are ≥ 2-D")
        self.elements = elements
        self.fields = list(fields)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.elements.shape

    def __repr__(self) -> str:
        return f"StructArray(shape={self.shape}, fields={self.fields})"


def is_struct(value: Any) -> bool:
    """True for a 1×1 struct in a MATLAB value tree."""
    return isinstance(value, dict)


def loadmat(path: str | PathLike[str], variable_names: list[str] | None = None) -> dict[str, Any]:
    """Load the variables of a ``.mat`` file (v4/v5/v7 or v7.3) as a MATLAB value tree.

    ``variable_names`` restricts the variables read, like ``load(file, names...)``.
    Missing names are silently skipped, as in SciPy (MATLAB warns).
    """
    path = Path(path)
    import h5py

    if h5py.is_hdf5(path):
        return _load_v73(path, variable_names)
    return _load_v5(path, variable_names)


def whosmat(path: str | PathLike[str]) -> list[str]:
    """Names of the variables in a ``.mat`` file (``whos('-file', ...)``), in file order."""
    path = Path(path)
    import h5py

    if h5py.is_hdf5(path):
        with h5py.File(path, "r") as f:
            return [k for k in f if not k.startswith("#")]
    from scipy.io import whosmat as _whosmat

    return [name for name, _, _ in _whosmat(path)]


# --------------------------------------------------------------------------- v5


def _load_v5(path: Path, names: list[str] | None) -> dict[str, Any]:
    from scipy.io import loadmat as _scipy_loadmat

    kw = {
        "squeeze_me": False,
        "struct_as_record": True,
        "chars_as_strings": True,
        "variable_names": names,
    }
    with warnings.catch_warnings():
        # B-30: the dropped imaginary part is restored from the raw read below.
        warnings.simplefilter("ignore", getattr(np, "exceptions", np).ComplexWarning)
        typed = _scipy_loadmat(path, mat_dtype=True, **kw)
    raw = _scipy_loadmat(path, mat_dtype=False, **kw)
    out = {k: _from_scipy(typed[k], raw[k]) for k in typed if k not in _SKIP}
    for var, steps in _zero_byte_elements(path):
        if var in out:
            out[var] = _replace_at(out[var], steps, np.zeros((0, 0)))
    return out


def _from_scipy(typed: Any, raw: Any) -> Any:
    from scipy.io.matlab import MatlabFunction, MatlabOpaque

    if isinstance(typed, MatlabOpaque | MatlabFunction):
        raise TypeError("MATLAB objects (string, function handle, classdef) are not supported")
    if not isinstance(typed, np.ndarray):
        import scipy.sparse

        if scipy.sparse.issparse(typed):
            raise TypeError("sparse matrices are not supported")
        raise TypeError(f"unexpected value from scipy.io.loadmat: {type(typed)!r}")
    if typed.dtype.names is not None:  # struct (record array)
        names = typed.dtype.names
        out = np.empty(typed.shape, dtype=object)
        for idx in np.ndindex(typed.shape):
            out[idx] = {n: _from_scipy(typed[idx][n], raw[idx][n]) for n in names}
        if typed.size == 1:
            return out.reshape(-1)[0]
        return StructArray(out, list(names))
    if typed.dtype == object:  # cell array (or empty struct field)
        out = np.empty(typed.shape, dtype=object)
        for idx in np.ndindex(typed.shape):
            out[idx] = _from_scipy(typed[idx], raw[idx])
        return out
    if typed.dtype.kind == "U":  # char
        if typed.size == 0:
            return ""
        if typed.size == 1:
            return str(typed.reshape(-1)[0])
        return np.array([str(s) for s in typed.reshape(-1)])  # char matrix rows
    if np.iscomplexobj(raw) and not np.iscomplexobj(typed):
        # B-30: mat_dtype=True drops the imaginary part. Take the raw values,
        # cast to the MATLAB class (single stays single).
        ctype = np.complex64 if typed.dtype == np.float32 else np.complex128
        return np.asarray(raw, dtype=ctype)
    return np.asarray(typed)


# B-33: 0-byte matrix elements (MATLAB's legacy encoding of [] in cells/structs)

_MI_MATRIX = 14
_MI_COMPRESSED = 15
_MX_CELL = 1
_MX_STRUCT = 2
_MX_OBJECT = 3


def _zero_byte_elements(path: Path) -> list[tuple[str, list[tuple[Any, ...]]]]:
    """``(variable, steps)`` of every 0-byte matrix element in a v5 file.

    ``steps`` lead from the variable to the element: ``("cell", k)`` (k-th
    element, column-major) or ``("field", k, name)`` (field of the k-th struct).
    """
    import zlib

    buf = path.read_bytes()
    if len(buf) < 128:
        return []
    endian = "<" if buf[126:128] == b"IM" else ">"
    found: list[tuple[str, list[tuple[Any, ...]]]] = []

    def tag(b: bytes, pos: int) -> tuple[int, int, int, int]:
        """(mdtype, nbytes, data start, next element position)."""
        first = int.from_bytes(b[pos : pos + 4], "little" if endian == "<" else "big")
        if first >> 16:  # small data element: type and size packed in 4 bytes
            return first & 0xFFFF, first >> 16, pos + 4, pos + 8
        n = int.from_bytes(b[pos + 4 : pos + 8], "little" if endian == "<" else "big")
        nxt = pos + 8 + n if first == _MI_COMPRESSED else pos + 8 + ((n + 7) // 8) * 8
        return first, n, pos + 8, nxt

    def ints(b: bytes, start: int, n: int, size: int) -> list[int]:
        order = "little" if endian == "<" else "big"
        return [
            int.from_bytes(b[i : i + size], order, signed=True)
            for i in range(start, start + n, size)
        ]

    def matrix(b: bytes, start: int, nbytes: int, steps: list[tuple[Any, ...]], var: list[str]):
        if nbytes == 0:
            found.append((var[0], list(steps)))
            return
        _, _, d, p = tag(b, start)  # array flags
        cls = b[d] if endian == "<" else b[d + 3]
        _, n, d, p2 = tag(b, p)  # dimensions
        dims = ints(b, d, n, 4)
        _, n, d, p = tag(b, p2)  # name
        if not var:
            var.append(b[d : d + n].decode("latin-1"))
        count = int(np.prod(dims))
        if cls == _MX_OBJECT:
            _, _, _, p = tag(b, p)  # class name
        if cls == _MX_CELL:
            for k in range(count):
                _, n, d, p = tag(b, p)
                matrix(b, d, n, [*steps, ("cell", k)], var)
        elif cls in (_MX_STRUCT, _MX_OBJECT):
            _, n, d, p = tag(b, p)
            width = ints(b, d, n, 4)[0]
            _, n, d, p = tag(b, p)
            raw = b[d : d + n]
            names = [
                raw[i : i + width].split(b"\0")[0].decode("latin-1") for i in range(0, n, width)
            ]
            for k in range(count):
                for name in names:
                    _, n, d, p = tag(b, p)
                    matrix(b, d, n, [*steps, ("field", k, name)], var)

    pos = 128
    while pos + 8 <= len(buf):
        mdtype, n, d, nxt = tag(buf, pos)
        if mdtype == _MI_COMPRESSED:
            inner = zlib.decompress(buf[d : d + n])
            imt, inn, idata, _ = tag(inner, 0)
            if imt == _MI_MATRIX:
                matrix(inner, idata, inn, [], [])
        elif mdtype == _MI_MATRIX:
            matrix(buf, d, n, [], [])
        pos = nxt
    return found


def _replace_at(tree: Any, steps: list[tuple[Any, ...]], value: Any) -> Any:
    if not steps:
        return value
    step, rest = steps[0], steps[1:]
    if step[0] == "cell":
        idx = np.unravel_index(step[1], tree.shape, order="F")
        tree[idx] = _replace_at(tree[idx], rest, value)
        return tree
    _, k, name = step
    if isinstance(tree, dict):
        tree[name] = _replace_at(tree[name], rest, value)
        return tree
    idx = np.unravel_index(k, tree.shape, order="F")
    elem = tree.elements[idx]
    elem[name] = _replace_at(elem[name], rest, value)
    return tree


# ------------------------------------------------------------------------- v7.3

_H5_NUMERIC = {
    "double": np.float64,
    "single": np.float32,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
    "logical": np.bool_,
}


def _load_v73(path: Path, names: list[str] | None) -> dict[str, Any]:
    import h5py

    out: dict[str, Any] = {}
    with h5py.File(path, "r") as f:
        keys = [k for k in f if not k.startswith("#")]
        for k in keys:
            if names is not None and k not in names:
                continue
            out[k] = _from_h5(f[k], f)
    return out


def _attr_str(obj: Any, name: str) -> str | None:
    v = obj.attrs.get(name)
    if v is None:
        return None
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.ndarray):
        return b"".join(v.astype("S1").reshape(-1)).decode()
    return str(v)


def _matlab_shape(h5shape: tuple[int, ...]) -> tuple[int, ...]:
    shape = tuple(reversed(h5shape))
    return shape if len(shape) >= 2 else (*shape, 1)


def _from_h5(obj: Any, f: Any) -> Any:
    import h5py

    cls = _attr_str(obj, "MATLAB_class")
    if isinstance(obj, h5py.Group):
        if cls != "struct":
            raise TypeError(f"unsupported MATLAB class {cls!r} in v7.3 file ({obj.name})")
        return _h5_struct(obj, f)
    empty = bool(obj.attrs.get("MATLAB_empty", 0))
    if empty:
        dims = tuple(int(d) for d in np.asarray(obj[()]).reshape(-1))
        if cls == "char":
            return ""
        if cls == "cell":
            return np.empty(dims, dtype=object)
        if cls == "struct":
            order = obj.attrs.get("MATLAB_fields")
            fields = (
                []
                if order is None
                else [b"".join(np.asarray(n).astype("S1")).decode() for n in order]
            )
            return StructArray(np.empty(dims, dtype=object), fields)
        return np.zeros(dims, dtype=_H5_NUMERIC.get(cls, np.float64))
    data = obj[()]
    if cls == "cell":
        cell = np.empty(data.shape, dtype=object)
        for idx in np.ndindex(data.shape):
            cell[idx] = _from_h5(f[data[idx]], f)
        return _to_matlab_order(cell)
    if cls == "char":
        codes = _to_matlab_order(np.asarray(data, dtype=np.uint16))
        rows = ["".join(chr(c) for c in r) for r in codes.reshape(codes.shape[0], -1)]
        return rows[0] if len(rows) == 1 else np.array(rows)
    if cls in _H5_NUMERIC:
        arr = np.asarray(data)
        if arr.dtype.names and {"real", "imag"} <= set(arr.dtype.names):
            ctype = np.complex64 if cls == "single" else np.complex128
            arr = arr["real"] + 1j * arr["imag"]
            arr = arr.astype(ctype)
        else:
            arr = arr.astype(_H5_NUMERIC[cls])
        return _to_matlab_order(arr)
    raise TypeError(f"unsupported MATLAB class {cls!r} in v7.3 file ({obj.name})")


def _to_matlab_order(arr: np.ndarray) -> np.ndarray:
    """h5py array (reversed MATLAB dims) → MATLAB size, same element positions."""
    out = np.transpose(arr)
    if out.ndim < 2:
        out = out.reshape(*out.shape, 1)
    return np.ascontiguousarray(out)


def _h5_struct(group: Any, f: Any) -> Any:
    import h5py

    order = group.attrs.get("MATLAB_fields")
    if order is not None:
        fields = [b"".join(np.asarray(n).astype("S1")).decode() for n in order]
    else:
        fields = list(group.keys())
    members = {n: group[n] for n in fields}
    # A struct array stores each field as a dataset of references (no class attr).
    is_array = any(
        isinstance(m, h5py.Dataset)
        and m.dtype == h5py.ref_dtype
        and _attr_str(m, "MATLAB_class") is None
        for m in members.values()
    )
    if not is_array:
        return {n: _from_h5(m, f) for n, m in members.items()}
    shape = next(m.shape for m in members.values())
    out = np.empty(shape, dtype=object)
    for idx in np.ndindex(shape):
        out[idx] = {n: _from_h5(f[m[idx]], f) for n, m in members.items()}
    out = _to_matlab_order(out)
    return out.reshape(-1)[0] if out.size == 1 else StructArray(out, fields)
