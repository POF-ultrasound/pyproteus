"""Export tool pieces that need no MATLAB: record decoding, specs, k-Wave version text."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from tools.export_reference.records import RecordError, records_to_value
from tools.export_reference.specs import SPECS, ExportSpec, M

from pyproteus.io.reference_format import FunctionHandle, MatlabString, StructArray


def rec(path, cls, size, data=None, fields=(), sparse=False):
    """A record as the MATLAB Engine returns it for pyp_flatten output."""
    return {
        "path": list(path),
        "class": cls,
        "size": [float(s) for s in size],
        "fields": list(fields),
        "sparse": sparse,
        "data": data,
    }


def test_scalars_become_matlab_size():
    # The engine returns 1x1 values as Python scalars.
    assert records_to_value([rec([], "double", (1, 1), 2.5)]).shape == (1, 1)
    v = records_to_value([rec([], "logical", (1, 1), True)])
    assert v.dtype == np.bool_ and v.shape == (1, 1) and v[0, 0]
    v = records_to_value([rec([], "int32", (1, 1), 7)])
    assert v.dtype == np.int32
    # The engine returns 1x1 unsigned/64-bit integers as Python int.
    for cls, val in (("uint8", 5), ("uint16", 65535), ("uint32", 7), ("uint64", 2**64 - 1)):
        v = records_to_value([rec([], cls, (1, 1), val)])
        assert v.dtype == np.dtype(cls) and v[0, 0] == val
    with pytest.raises(RecordError, match="range"):
        records_to_value([rec([], "uint8", (1, 1), 256)])
    with pytest.raises(RecordError, match="range"):
        records_to_value([rec([], "uint8", (1, 1), -1)])
    v = records_to_value([rec([], "double", (1, 1), 1 + 2j)])
    assert v.dtype == np.complex128
    v = records_to_value([rec([], "single", (0, 3), np.zeros((0, 3)))])
    assert v.shape == (0, 3) and v.dtype == np.float32


def test_array_shape_must_match():
    a = np.arange(6.0).reshape(2, 3)
    np.testing.assert_array_equal(records_to_value([rec([], "double", (2, 3), a)]), a)
    with pytest.raises(RecordError, match="shape"):  # would reorder silently
        records_to_value([rec([], "double", (3, 2), a)])


def test_lossy_class_cast_rejected():
    with pytest.raises(RecordError):
        records_to_value([rec([], "int32", (1, 2), np.array([[1.5, 2.0]]))])
    with pytest.raises(RecordError):
        records_to_value([rec([], "logical", (1, 1), 1 + 1j)])


def test_struct_cell_and_text():
    records = [
        rec([], "struct", (1, 1), fields=("a", "c", "s", "h", "e")),
        rec(["a"], "struct", (1, 2), fields=("q",)),
        rec(["a", "0"], "struct", (1, 1), fields=("q",)),
        rec(["a", "0", "q"], "double", (1, 1), 1.0),
        rec(["a", "1"], "struct", (1, 1), fields=("q",)),
        rec(["a", "1", "q"], "double", (1, 1), 2.0),
        rec(["c"], "cell", (2, 2)),
        rec(["c", "0"], "char", (1, 2), "00"),
        rec(["c", "1"], "char", (1, 2), "10"),
        rec(["c", "2"], "char", (1, 2), "01"),
        rec(["c", "3"], "double", (2, 2), np.array([[0.0, 1.0], [0.0, 0.0]]), sparse=True),
        rec(["s"], "string", (1, 1), "str"),
        rec(["h"], "function_handle", (1, 1), "@(x)x+1"),
        rec(["e"], "char", (0, 0), ""),
    ]
    v = records_to_value(records)
    assert list(v) == ["a", "c", "s", "h", "e"]
    assert isinstance(v["a"], StructArray) and v["a"].shape == (1, 2)
    assert v["a"].elements[0, 1]["q"][0, 0] == 2.0
    c = v["c"]
    assert c.shape == (2, 2) and c[0, 0] == "00" and c[1, 0] == "10" and c[0, 1] == "01"
    assert sp.issparse(c[1, 1]) and c[1, 1][0, 1] == 1.0
    assert type(v["s"]) is MatlabString and type(v["h"]) is FunctionHandle
    assert v["e"] == ""


@pytest.mark.parametrize(
    "records",
    [
        [rec([], "cell", (1, 2)), rec(["0"], "char", (1, 1), "a")],  # missing element
        [rec([], "char", (1, 1), "a"), rec([], "char", (1, 1), "b")],  # left over
        [rec([], "struct", (1, 1), fields=("a",)), rec(["b"], "double", (1, 1), 1.0)],  # path
        [rec([], "char", (1, 3), "ab")],  # char size mismatch
        [rec([], "char", (1, 2), "a\U0001d11e")],  # non-BMP: 3 UTF-16 units in MATLAB
        [rec([], "table", (1, 1))],
        [rec([], "double", (1,), 1.0)],  # size must be >= 2 entries
        [
            rec([], "struct", (2, 1), fields=("a",)),
            rec(["0"], "struct", (1, 1), fields=("a",)),
            rec(["0", "a"], "double", (1, 1), 1.0),
            rec(["1"], "struct", (1, 1), fields=("b",)),
            rec(["1", "b"], "double", (1, 1), 1.0),
        ],
    ],
)
def test_bad_records(records):
    with pytest.raises(RecordError):
        records_to_value(records)


def test_registered_specs():
    assert {"GUIfunctions.reset_acquisition", "GUIfunctions.reset_transmit"} <= set(SPECS)
    assert {"full_run.tiny", "full_run.small", "full_run.medium"} <= set(SPECS)
    for name, spec in SPECS.items():
        assert spec.name == name
        assert spec.statement(list(spec.inputs()))
        assert "matlab_rng_twister" in spec.all_seeds()
        if spec.kind == "full_run":
            assert spec.needs_kwave and "pyp_out" in spec.statement([])
    assert not SPECS["full_run.tiny"].slow
    assert SPECS["full_run.medium"].slow


def test_spec_statement_and_validation():
    s = ExportSpec(
        name="m.f",
        module="m",
        function="f",
        description="",
        inputs=lambda: {"a": M("1"), "b": 2.0},
        outputs=("x", "y"),
    )
    assert s.statement(["a", "b"]) == "[pyp_out.x, pyp_out.y] = f(pyp_in.a, pyp_in.b);"
    for bad in (
        {"name": "a/b"},
        {"name": "x.h5"},
        {"function": "f(1)"},
        {"outputs": ()},
        {"outputs": ("1x",)},
        {"kind": "other"},
        {"kind": "full_run"},  # full run without k-Wave
    ):
        kw = {"name": "m.f", "module": "m", "function": "f", "description": "", "outputs": ("x",)}
        kw.update(bad)
        with pytest.raises(ValueError):
            ExportSpec(**kw)


def test_binary_version_text():
    from tools.export_reference.engine import binary_version_text

    out = (
        "┌──────┐\n│   kspaceFirstOrder-OMP v1.3   │\n├──────┤\n"
        "│ Build number:     kspaceFirstOrder v2.17   │\n"
        "│ Build date:       May 17 2026   │\n│ Contact email: x │\n└──────┘\n"
    )
    assert binary_version_text(out) == (
        "kspaceFirstOrder-OMP v1.3; Build number: kspaceFirstOrder v2.17; Build date: May 17 2026"
    )
    with pytest.raises(ValueError):
        binary_version_text("garbage")


def test_non_bmp_char_record():
    assert records_to_value([rec([], "char", (1, 3), "a\U0001d11e")]) == "a\U0001d11e"
