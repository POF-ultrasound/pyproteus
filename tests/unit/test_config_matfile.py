"""Unit tests for ``pyproteus.config.matfile`` (no MATLAB needed)."""

from __future__ import annotations

import struct
from pathlib import Path

import h5py
import numpy as np
import pytest
import scipy.io as sio

from pyproteus.config.matfile import StructArray, loadmat, whosmat


def test_v5_classes_and_shapes(tmp_path: Path) -> None:
    f = tmp_path / "a.mat"
    sio.savemat(
        f,
        {
            "row": np.array([[1.0, 2.0, 3.0]]),
            "col": np.array([[1.0], [2.0]]),
            "flag": np.array([[True]]),
            "i8": np.array([[-3, 4]], dtype=np.int8),
            "z": np.array([[1 + 2j, -0.5j]]),
            "s": "hello",
            "empty": np.zeros((0, 0)),
            "empty13": np.zeros((1, 0)),
            "st": {"a": 1.0, "b": "x", "c": np.zeros((0, 0))},
        },
    )
    d = loadmat(f)
    assert d["row"].shape == (1, 3) and d["col"].shape == (2, 1)
    assert d["flag"].dtype == np.bool_
    assert d["i8"].dtype == np.int8
    assert d["z"].dtype == np.complex128
    np.testing.assert_array_equal(d["z"], [[1 + 2j, -0.5j]])  # B-30: imaginary kept
    assert d["s"] == "hello"
    assert d["empty"].shape == (0, 0) and d["empty13"].shape == (1, 0)
    assert list(d["st"]) == ["a", "b", "c"]
    assert d["st"]["b"] == "x"
    assert d["st"]["a"].shape == (1, 1)


def test_v5_struct_array_vs_cell_of_structs(tmp_path: Path) -> None:
    f = tmp_path / "a.mat"
    sa = np.zeros((1, 2), dtype=[("p", object), ("q", object)])
    sa[0, 0] = (1.0, "a")
    sa[0, 1] = (2.0, "b")
    cell = np.empty((1, 2), dtype=object)
    cell[0, 0] = {"p": 1.0}
    cell[0, 1] = {"p": 2.0}
    sio.savemat(f, {"sa": sa, "cell": cell})
    d = loadmat(f)
    assert isinstance(d["sa"], StructArray)
    assert d["sa"].shape == (1, 2) and d["sa"].fields == ["p", "q"]
    assert d["sa"].elements[0, 1]["q"] == "b"
    assert isinstance(d["cell"], np.ndarray) and d["cell"].dtype == object
    assert isinstance(d["cell"][0, 1], dict)


def test_char_matrix(tmp_path: Path) -> None:
    f = tmp_path / "a.mat"
    sio.savemat(f, {"cm": np.array(["ab", "cd"])})
    out = loadmat(f)["cm"]
    assert isinstance(out, np.ndarray) and list(out) == ["ab", "cd"]


def _v5_element(mdtype: int, payload: bytes) -> bytes:
    pad = (-len(payload)) % 8
    return struct.pack("<II", mdtype, len(payload)) + payload + b"\0" * pad


def _v5_cell_with_zero_byte_element() -> bytes:
    """A v5 file with ``c = {[], 5}`` where [] is a 0-byte miMATRIX (old MATLAB)."""
    header = b"MATLAB 5.0 MAT-file, test".ljust(116, b" ") + b"\0" * 8
    header += struct.pack("<H", 0x0100) + b"IM"
    flags_double = _v5_element(6, struct.pack("<II", 6, 0))  # mxDOUBLE_CLASS
    dims11 = _v5_element(5, struct.pack("<ii", 1, 1))
    noname = _v5_element(1, b"")
    five = _v5_element(9, struct.pack("<d", 5.0))
    elem5 = _v5_element(14, flags_double + dims11 + noname + five)
    zero = struct.pack("<II", 14, 0)
    flags_cell = _v5_element(6, struct.pack("<II", 1, 0))  # mxCELL_CLASS
    dims12 = _v5_element(5, struct.pack("<ii", 1, 2))
    name = _v5_element(1, b"c")
    cell = _v5_element(14, flags_cell + dims12 + name + zero + elem5)
    return header + cell


def test_zero_byte_empty_element(tmp_path: Path) -> None:
    """B-33: MATLAB loads a 0-byte matrix element as 0×0 (scipy alone gives 1×0)."""
    f = tmp_path / "z.mat"
    f.write_bytes(_v5_cell_with_zero_byte_element())
    raw = sio.loadmat(f)["c"][0, 0]
    assert raw.shape == (1, 0)  # the scipy behaviour corrected below
    c = loadmat(f)["c"]
    assert c.shape == (1, 2)
    assert c[0, 0].shape == (0, 0) and c[0, 0].dtype == np.float64
    assert c[0, 1][0, 0] == 5.0


# --------------------------------------------------------------------- v7.3


def _attr(obj: h5py.HLObject, cls: str) -> None:
    obj.attrs["MATLAB_class"] = np.bytes_(cls)


def _write_v73(path: Path) -> None:
    """A MATLAB-style v7.3 file built with h5py (layout as written by MATLAB)."""
    with h5py.File(path, "w", userblock_size=512) as f:
        d = f.create_dataset("m", data=np.arange(6.0).reshape(3, 2))  # MATLAB 2×3
        _attr(d, "double")
        d = f.create_dataset("flag", data=np.array([[1]], dtype=np.uint8))
        _attr(d, "logical")
        d = f.create_dataset("s", data=np.array([[ord(c)] for c in "hi"], dtype=np.uint16))
        _attr(d, "char")
        d = f.create_dataset("e", data=np.array([0, 0], dtype=np.uint64))
        _attr(d, "double")
        d.attrs["MATLAB_empty"] = np.uint8(1)
        cplx = np.zeros((2, 1), dtype=[("real", "<f8"), ("imag", "<f8")])
        cplx["real"] = [[1.0], [2.0]]
        cplx["imag"] = [[-1.0], [0.5]]
        d = f.create_dataset("z", data=cplx)
        _attr(d, "double")
        g = f.create_group("st")
        _attr(g, "struct")
        x = g.create_dataset("x", data=np.array([[3.0]]))
        _attr(x, "double")
        refs = f.create_group("#refs#")
        r1 = refs.create_dataset("a", data=np.array([[7.0]]))
        _attr(r1, "double")
        r2 = refs.create_dataset("b", data=np.array([[ord("q")]], dtype=np.uint16))
        _attr(r2, "char")
        c = f.create_dataset("c", data=np.array([[r1.ref], [r2.ref]], dtype=h5py.ref_dtype))
        _attr(c, "cell")


def test_v73(tmp_path: Path) -> None:
    f = tmp_path / "b.mat"
    _write_v73(f)
    d = loadmat(f)
    np.testing.assert_array_equal(d["m"], np.arange(6.0).reshape(3, 2).T)
    assert d["flag"].dtype == np.bool_ and d["flag"].shape == (1, 1)
    assert d["s"] == "hi"
    assert d["e"].shape == (0, 0)
    np.testing.assert_array_equal(d["z"], [[1 - 1j, 2 + 0.5j]])
    assert d["st"]["x"][0, 0] == 3.0
    assert d["c"].shape == (1, 2) and d["c"][0, 1] == "q"
    assert sorted(whosmat(f)) == ["c", "e", "flag", "m", "s", "st", "z"]
    assert list(loadmat(f, ["m"])) == ["m"]


def test_unsupported_objects(tmp_path: Path) -> None:
    import scipy.sparse

    f = tmp_path / "sp.mat"
    sio.savemat(f, {"sp": scipy.sparse.eye(3).tocsc()})
    with pytest.raises(TypeError, match="sparse"):
        loadmat(f)
