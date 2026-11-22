"""Reference HDF5 format: round trip, MATLAB-size layout, validation (no MATLAB)."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import scipy.sparse as sp

from pyproteus.io.reference_format import (
    ARRAY_LAYOUT,
    FORMAT_NAME,
    FORMAT_VERSION,
    REQUIRED_ATTRS,
    FunctionHandle,
    MatlabString,
    ReferenceFormatError,
    StructArray,
    read_reference,
    write_reference,
)


def attrs(**over):
    a = {
        "name": "mod.fn",
        "kind": "module",
        "module": "GUIfunctions",
        "function": "fn",
        "description": "test",
        "call": "[pyp_out.y] = fn(pyp_in.x);",
        "matlab_version": "25.1",
        "proteus_version": "1.1.0",
        "oracle_commit": "8d37938f5e4fe5e408ae707e792dad463303b98d",
        "oracle_dirty": False,
        "kwave_matlab_version": "",
        "kwave_matlab_path": "",
        "kwave_solver": "",
        "kwave_binary_path": "",
        "kwave_binary_version": "",
        "kwave_binary_sha256": "",
        "seeds": {"matlab_rng_twister": 0},
        "matlab_path_extra": ["tools/export_reference/matlab"],
        "created_utc": "2026-11-22T00:00:00+00:00",
        "exporter_version": "0.0.1",
        "exporter_commit": "abc",
        "platform": "test",
    }
    a.update(over)
    return a


def assert_same(a, b, where="value"):
    """Deep equality incl. type, dtype, shape and sign of zero."""
    assert type(a) is type(b), f"{where}: {type(a)} != {type(b)}"
    if isinstance(a, dict):
        assert list(a) == list(b), where
        for k in a:
            assert_same(a[k], b[k], f"{where}.{k}")
    elif isinstance(a, StructArray):
        assert a.fields == b.fields and a.shape == b.shape, where
        for idx in np.ndindex(a.shape):
            assert_same(a.elements[idx], b.elements[idx], f"{where}{idx}")
    elif sp.issparse(a):
        assert a.shape == b.shape and a.dtype == b.dtype, where
        assert (a != b).nnz == 0, where
    elif isinstance(a, np.ndarray) and a.dtype == object:
        assert a.shape == b.shape, where
        for idx in np.ndindex(a.shape):
            assert_same(a[idx], b[idx], f"{where}{{{idx}}}")
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape, f"{where}: {a.dtype}{a.shape}"
        np.testing.assert_array_equal(a, b, err_msg=where)
        if a.dtype.kind == "f":
            assert np.array_equal(np.signbit(a), np.signbit(b)), where
        if a.dtype.kind == "c":
            assert np.array_equal(np.signbit(a.real), np.signbit(b.real)), where
            assert np.array_equal(np.signbit(a.imag), np.signbit(b.imag)), where
    else:
        assert a == b, where


def every_kind():
    rng = np.random.default_rng(0)
    cell = np.empty((2, 3), dtype=object)
    for k, idx in enumerate(np.ndindex(2, 3)):
        cell[idx] = np.full((1, k + 1), float(k))
    cell[1, 2] = "text"
    empty_cell = np.empty((0, 0), dtype=object)
    sa = np.empty((2, 2), dtype=object)
    for k, idx in enumerate(np.ndindex(2, 2)):
        sa[idx] = {"a": np.array([[float(k)]]), "b": f"el{k}"}
    return {
        "scalar": np.array([[2.5]]),
        "neg_zero": np.array([[-0.0, 0.0, np.nan, np.inf, -np.inf]]),
        "row": rng.random((1, 7)),
        "col": rng.random((7, 1)),
        "nonsquare3d": rng.random((2, 3, 5)),
        "fourd": rng.random((2, 1, 3, 2)),
        "single": rng.random((3, 2)).astype(np.float32),
        "complex": (rng.random((2, 3)) + 1j * rng.random((2, 3))),
        "complex_single": (rng.random((2, 2)) + 1j).astype(np.complex64),
        "complex_neg_zero": np.array([[complex(-0.0, 0.0), complex(0.0, -0.0)]]),
        "non_bmp": "a\U0001d11e",  # MATLAB: 3 UTF-16 code units
        "logical": rng.random((3, 3)) > 0.5,
        "int8": np.array([[-128, 127]], dtype=np.int8),
        "uint16": np.array([[0], [65535]], dtype=np.uint16),
        "int32": np.arange(6, dtype=np.int32).reshape(2, 3),
        "int64": np.array([[2**62]], dtype=np.int64),
        "uint64": np.array([[2**64 - 1]], dtype=np.uint64),
        "empty_0x3": np.zeros((0, 3)),
        "empty_0x0": np.zeros((0, 0)),
        "sparse": sp.csc_array(np.array([[0.0, 1.5, 0.0], [2.0, 0.0, 0.0]])),
        "sparse_logical": sp.csc_array(np.array([[True, False], [False, False]])),
        "char": "Three-level",
        "char_empty": "",
        "unicode": "µm ±",
        "string": MatlabString("a string"),
        "string_empty": MatlabString(""),
        "handle": FunctionHandle("@(x)x.^2"),
        "nested": {"inner": {"x": np.array([[1.0]]), "s": "y"}, "empty_struct": {}},
        "cell": cell,
        "empty_cell": empty_cell,
        "struct_array": StructArray(sa, ("a", "b")),
        "struct_array_row": StructArray(
            np.array([[{"q": np.array([[1.0]])}, {"q": np.array([[2.0]])}]], dtype=object),
            ("q",),
        ),
    }


def test_round_trip_every_kind(tmp_path):
    inputs = every_kind()
    outputs = {"y": np.arange(24.0).reshape(2, 3, 4)}
    extra = {"note": "x"}
    p = write_reference(tmp_path / "mod.fn.h5", attrs(), inputs, outputs, extra)
    r = read_reference(p)
    assert_same(r.inputs, inputs, "inputs")
    assert_same(r.outputs, outputs, "outputs")
    assert_same(r.extra, extra, "extra")
    assert r.attrs["seeds"] == {"matlab_rng_twister": 0}
    assert r.attrs["matlab_path_extra"] == ["tools/export_reference/matlab"]
    assert r.attrs["oracle_dirty"] is False
    assert r.attrs["format"] == FORMAT_NAME
    assert r.attrs["format_version"] == FORMAT_VERSION
    assert r.attrs["array_layout"] == ARRAY_LAYOUT
    assert r.name == "mod.fn"
    for k in REQUIRED_ATTRS:
        assert k in r.attrs


def test_layout_dims_equal_matlab_size(tmp_path):
    """The stored dataset has dims == MATLAB size, independent of this reader.

    MATLAB ``reshape(1:24, 2, 3, 4)`` numbers elements column-major, so element
    (i, j, k) (0-based) holds 1 + i + 2j + 6k. That array is written as it
    would come from the engine; raw h5py must see shape (2, 3, 4) with the same
    element positions, and ravel(order='F') must reproduce MATLAB's x(:).
    """
    i, j, k = np.indices((2, 3, 4))
    a = (1 + i + 2 * j + 6 * k).astype(np.float64)
    p = write_reference(tmp_path / "mod.fn.h5", attrs(), {}, {"y": a})
    with h5py.File(p, "r") as f:
        d = f["outputs/y"]
        assert d.shape == (2, 3, 4)
        assert list(d.attrs["matlab_size"]) == [2, 3, 4]
        assert d.attrs["matlab_class"] == "double"
        raw = d[()]
    assert raw[1, 2, 3] == 1 + 1 + 4 + 18
    np.testing.assert_array_equal(raw.ravel(order="F"), np.arange(1.0, 25.0))
    # A Fortran-ordered input array must give the same file content.
    write_reference(tmp_path / "f.h5", attrs(name="f"), {}, {"y": np.asfortranarray(a)})
    with h5py.File(tmp_path / "f.h5", "r") as f:
        np.testing.assert_array_equal(f["outputs/y"][()], raw)


def test_cell_and_struct_array_column_major(tmp_path):
    """Cell elements are stored by MATLAB linear index (column-major)."""
    c = np.empty((2, 3), dtype=object)
    for idx in np.ndindex(2, 3):
        c[idx] = f"{idx[0]}{idx[1]}"
    write_reference(tmp_path / "mod.fn.h5", attrs(), {}, {"c": c})
    with h5py.File(tmp_path / "mod.fn.h5", "r") as f:
        g = f["outputs/c"]
        assert g.attrs["matlab_class"] == "cell"
        order = [g[str(n)][()].decode() for n in range(6)]
    assert order == ["00", "10", "01", "11", "02", "12"]


@pytest.mark.parametrize(
    "bad",
    [
        np.arange(3.0),  # 1-D: MATLAB size unknown
        np.float64(1.0),  # NumPy scalar: not >= 2-D
        1.0,  # Python scalar
        [1.0, 2.0],
        np.array([["a"]]),  # unicode array has no MATLAB class
        np.empty(3, dtype=object),  # 1-D cell
        {"1bad": np.ones((1, 1))},  # invalid field name
        sp.csc_array(np.ones((2, 2), dtype=np.float32)),  # sparse single
        np.ones((2, 3, 1)),  # trailing singleton: MATLAB size never ends in 1 after dim 2
        np.empty((2, 3, 1), dtype=object),  # same for cells
    ],
)
def test_rejects_ambiguous_values(tmp_path, bad):
    with pytest.raises(ReferenceFormatError):
        write_reference(tmp_path / "mod.fn.h5", attrs(), {"x": bad}, {})
    assert not list(tmp_path.iterdir()), "no file or partial file left behind"


def test_struct_array_validation():
    with pytest.raises(ReferenceFormatError):
        StructArray(np.array([[{"a": 1}], [{"b": 1}]], dtype=object), ("a",))
    with pytest.raises(ReferenceFormatError):
        StructArray(np.array([[{"a": 1}]], dtype=object), ("a",))  # 1x1 -> dict
    with pytest.raises(ReferenceFormatError):
        StructArray(np.array([{"a": 1}, {"a": 2}], dtype=object), ("a",))  # 1-D


@pytest.mark.parametrize(
    "over",
    [
        {"kind": "other"},
        {"name": ""},
        {"name": "a/b"},
        {"oracle_dirty": "no"},
        {"seeds": {"s": 1.5}},
        {"seeds": {"s": True}},
        {"matlab_path_extra": [1]},
        {"kind": "full_run"},  # full run without the k-Wave binary path (D-10)
        {"format": "x"},  # reserved
    ],
)
def test_rejects_bad_provenance(tmp_path, over):
    with pytest.raises(ReferenceFormatError):
        write_reference(tmp_path / "x.h5", attrs(**over), {}, {})


def test_missing_provenance(tmp_path):
    a = attrs()
    del a["oracle_commit"]
    with pytest.raises(ReferenceFormatError, match="oracle_commit"):
        write_reference(tmp_path / "x.h5", a, {}, {})


def test_full_run_with_binary_ok(tmp_path):
    a = attrs(kind="full_run", kwave_binary_path="/bin/kspaceFirstOrder-OMP")
    write_reference(tmp_path / "x.h5", a, {}, {})


def _tamper(path, fn):
    with h5py.File(path, "r+") as f:
        fn(f)


@pytest.mark.parametrize(
    "tamper, match",
    [
        (lambda f: f.attrs.__setitem__("format_version", 99), "format_version"),
        (lambda f: f.attrs.__setitem__("format", "other"), "not a pyproteus"),
        (lambda f: f.attrs.__setitem__("array_layout", "c-order"), "array_layout"),
        (lambda f: f.attrs.__delitem__("oracle_commit"), "oracle_commit"),
        (
            lambda f: f["outputs/y"].attrs.__setitem__("matlab_size", np.array([3, 2])),
            "matlab_size",
        ),
        (lambda f: f["outputs/y"].attrs.__setitem__("matlab_class", "int32"), "class"),
        (lambda f: f["outputs/y"].attrs.__setitem__("matlab_class", "logical"), "class"),
        (lambda f: f["outputs/y"].attrs.__delitem__("matlab_size"), "matlab_size"),
        (lambda f: f["outputs/c"].__delitem__("1"), "cell"),
        (
            lambda f: f["outputs/c/0"].attrs.__setitem__("matlab_size", np.array([1, 5])),
            "text size",
        ),
    ],
)
def test_reader_rejects_tampered(tmp_path, tamper, match):
    c = np.empty((1, 2), dtype=object)
    c[0, 0], c[0, 1] = "a", "b"
    p = write_reference(tmp_path / "mod.fn.h5", attrs(), {}, {"y": np.ones((2, 3)), "c": c})
    _tamper(p, tamper)
    with pytest.raises(ReferenceFormatError, match=match):
        read_reference(p)


def test_overwrite_is_atomic(tmp_path):
    p = write_reference(tmp_path / "mod.fn.h5", attrs(), {}, {"y": np.ones((1, 1))})
    with pytest.raises(ReferenceFormatError):
        write_reference(p, attrs(), {"x": np.arange(2.0)}, {})
    assert read_reference(p).outputs["y"].shape == (1, 1)
    assert sorted(x.name for x in tmp_path.iterdir()) == ["mod.fn.h5"]


def test_char_size_counts_utf16_units(tmp_path):
    p = write_reference(tmp_path / "mod.fn.h5", attrs(), {}, {"s": "a\U0001d11e"})
    with h5py.File(p, "r") as f:
        assert list(f["outputs/s"].attrs["matlab_size"]) == [1, 3]
