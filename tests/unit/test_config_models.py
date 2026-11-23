"""Unit tests for settings models, JSON I/O, defaults and lookup functions (no MATLAB)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio
from pydantic import ValidationError

from pyproteus import config, data
from pyproteus import medium as pmed
from pyproteus import microbubble as pmb
from pyproteus import transducer as ptr
from pyproteus.config.json_io import tree_from_jsonable, tree_to_jsonable
from pyproteus.config.matfile import StructArray
from pyproteus.config.models import Gas, Medium, Settings, Shell, Transducer, Transmit

# ------------------------------------------------------------------- models


def test_scalars_and_arrays() -> None:
    t = Transducer(Pitch=np.array([[3e-4]]), NumberOfElements=96, calibrated=np.array([[0.0]]))
    assert t.Pitch == 3e-4 and isinstance(t.NumberOfElements, float)
    assert t.calibrated == 0.0 and not isinstance(t.calibrated, bool)
    t.calibrated = True
    assert t.calibrated is True  # Flag keeps the class it is given
    t.TransmitImpulseResponse = np.array([[1, 2, 3]])
    assert t.TransmitImpulseResponse.dtype == np.float64
    assert t.TransmitImpulseResponse.shape == (1, 3)
    with pytest.raises(ValidationError, match="1-D"):
        t.TransmitImpulseResponse = np.array([1.0, 2.0])
    with pytest.raises(ValidationError):
        t.Pitch = np.array([[1.0, 2.0]])
    with pytest.raises(ValidationError):
        t.NumberOfElements = True  # logical is not a double
    assert Shell(FitFile=[]).FitFile.shape == (0, 0)
    assert Shell(FitFile="fit.mat").FitFile == "fit.mat"


def test_field_order_isfield_rmfield_extras() -> None:
    t = Transducer(Pitch=1.0, Type="P4-1")
    t.Radius = 0.0
    t.integration_points = np.zeros((2, 3, 4))
    assert t.present_fields() == ["Pitch", "Type", "Radius", "integration_points"]
    assert t.isfield("integration_points") and not t.isfield("Configuration")
    t.rmfield("Type")
    t.rmfield("integration_points")
    assert list(t.to_tree()) == ["Pitch", "Radius"]
    with pytest.raises(KeyError):
        t.rmfield("Type")
    t.Type = "C1-6D"  # MATLAB appends a re-created field at the end
    assert t.present_fields() == ["Pitch", "Radius", "Type"]


def test_copy_struct_is_deep() -> None:
    m = config.reset_microbubble()
    c = m.copy_struct()
    c.Distribution.Radii[0, 0] = -1.0
    c.Gas.Type = "Air"
    assert m.Distribution.Radii[0, 0] == 0.5e-6 and m.Gas.Type == "Sulfur hexafluoride"


def test_tree_round_trip_keeps_classes() -> None:
    tree = {
        "Pitch": np.array([[1.0]]),
        "calibrated": np.array([[True]]),
        "Type": "x",
        "extra": np.array([[1, 2]], dtype=np.int16),
    }
    t = Transducer.from_tree(tree)
    out = t.to_tree()
    assert list(out) == list(tree)
    assert out["calibrated"].dtype == np.bool_ and out["extra"].dtype == np.int16


# --------------------------------------------------------------------- JSON


def _tree() -> dict:
    cell = np.empty((2, 1), dtype=object)
    cell[0, 0] = "a"
    cell[1, 0] = np.zeros((0, 3))
    sa = np.empty((1, 2), dtype=object)
    sa[0, 0] = {"p": np.array([[1.0]])}
    sa[0, 1] = {"p": "b"}
    return {
        "x": np.array([[math.inf, -0.0, math.nan, 5e-324, 0.1]]),
        "col": np.array([[1.0], [2.0]]),
        "nd": np.arange(24.0).reshape(2, 3, 4),
        "e": np.zeros((0, 0)),
        "e10": np.zeros((1, 0)),
        "b": np.array([[True, False]]),
        "b1": np.array([[False]]),
        "i": np.array([[-3, 7]], dtype=np.int8),
        "u": np.array([[2**63]], dtype=np.uint64),
        "z": np.array([[1 - 2j]], dtype=np.complex64),
        "s": "text",
        "es": "",
        "cm": np.array(["ab", "cd"]),
        "cell": cell,
        "sa": StructArray(sa, ["p"]),
        "esa": StructArray(np.empty((0, 0), dtype=object), ["q", "r"]),
        "st": {"inner": {"v": np.array([[2.0]])}},
    }


def _same(a: object, b: object) -> None:
    assert type(a) is type(b), (type(a), type(b))
    if isinstance(a, dict):
        assert list(a) == list(b)
        for k in a:
            _same(a[k], b[k])
    elif isinstance(a, StructArray):
        assert a.fields == b.fields and a.shape == b.shape
        for x, y in zip(a.elements.flat, b.elements.flat, strict=True):
            _same(x, y)
    elif isinstance(a, np.ndarray) and a.dtype == object:
        assert a.shape == b.shape
        for x, y in zip(a.flat, b.flat, strict=True):
            _same(x, y)
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape
        assert np.array_equal(a, b, equal_nan=a.dtype.kind in "fc")
        if a.dtype.kind == "f":
            assert np.array_equal(np.signbit(a), np.signbit(b))
    else:
        assert a == b


def test_json_tree_round_trip_is_lossless() -> None:
    import json

    tree = _tree()
    text = json.dumps(tree_to_jsonable(tree))
    _same(tree_from_jsonable(json.loads(text)), tree)


def test_json_readable_forms() -> None:
    j = tree_to_jsonable(_tree())
    assert j["col"] == [[1.0], [2.0]] and j["e"] == [] and j["b1"] is False
    assert j["s"] == "text" and j["st"] == {"inner": {"v": 2.0}}


def test_settings_json_file_round_trip(tmp_path: Path) -> None:
    s = config.default_settings()
    s.Transducer.integration_points = np.arange(6.0).reshape(1, 2, 3)
    f = tmp_path / "s.json"
    config.save_settings_json(s, f)
    s2 = config.load_settings_json(f)
    for g in Settings.GROUPS:
        _same(getattr(s2, g).to_tree(), getattr(s, g).to_tree())
    assert config.settings_to_json(s2) == config.settings_to_json(s)
    assert s2.Transmit.LateralFocus == math.inf


@pytest.mark.parametrize(
    "text",
    [
        "{}",
        '{"format": "pyproteus-settings", "version": 2}',
        '{"format": "pyproteus-settings", "version": 1, "Bogus": {}}',
        '{"format": "pyproteus-settings", "version": 1, "Medium": {"Tissue": null}}',
        '{"format": "pyproteus-settings", "version": 1, "Medium": {"X": [1, 2]}}',
    ],
)
def test_settings_json_rejects_bad_documents(text: str) -> None:
    with pytest.raises((ValueError, ValidationError)):
        config.settings_from_json(text)


# ----------------------------------------------------------------- defaults


def test_defaults_spot_values() -> None:
    """Values read off the reset_*.m sources (the oracle test compares all fields)."""
    s = config.default_settings()
    assert s.Medium.SpeedOfSoundMinimum == 1540 * (1 - 0.02 * 2)
    assert s.Medium.SpeedOfSoundMaximum == 1540 * (1 + 0.02 * 2)
    assert s.Medium.Vessel.Density == 1.055 * 1000
    assert s.Medium.Save is True and s.Acquisition.Precompute is False
    assert s.Geometry.Visualization.ShowTransducer == 1.0
    assert not isinstance(s.Geometry.Visualization.ShowTransducer, bool)
    assert s.Geometry.EmbedVessel is True
    assert s.Geometry.BoundingBox.Center.shape == (3, 1)
    assert s.Transducer.calibrated == 0.0 and s.Transducer.ImpulseResponseType == "Estimate"
    assert s.Transducer.SamplingRate == 250e6
    assert s.Transmit.CenterFrequency == 2.5e6 and s.Transmit.DelayType == "Compute delays"
    assert s.Microbubble.Shell.FitFile.shape == (0, 0)
    assert s.Microbubble.Distribution.Radii.shape == (1, 1001)
    assert s.Acquisition.Folder == ""
    for group, fields in config.PENDING_FIELDS.items():
        for f in fields:
            assert not getattr(s, group).isfield(f)


@pytest.mark.parametrize(
    ("configuration", "delay"),
    [
        ("Linear", "Compute delays"),
        ("Curved", "Compute delays"),
        ("RCA", "No delays"),
        ("Custom", "No delays"),
        (None, "Compute delays"),
        ("Other", None),
    ],
)
def test_reset_transmit_delay_type(configuration: str | None, delay: str | None) -> None:
    t = Transducer(CenterFrequency=3e6, SamplingRate=250e6, TransmitImpulseResponse=np.ones((1, 3)))
    if configuration is not None:
        t.Configuration = configuration
    tx = config.reset_transmit(t)
    assert tx.DelayType == delay
    assert not tx.isfield("DelayType") if delay is None else True


def test_reset_geometry_mouse_brain() -> None:
    g = config.reset_geometry(config.Geometry(Folder="mouse_brain"))
    assert g.STLfile == "lpa_mouse_t10.stl" and g.Visualization.Image == "mouse_brain.png"
    assert list(g.to_tree())[:2] == ["Folder", "STLfile"]


# -------------------------------------------------------- lookup functions


def test_tissue_lookup_quirks() -> None:
    m = pmed.assign_medium_properties(Medium(Tissue="Muscle"))
    tissues = data.load("tissue_properties.mat")
    row = [str(n) for n in tissues["tissue_names"].reshape(-1)].index("Muscle (along the fibers)")
    assert m.SpeedOfSound == tissues["tissue_properties"][row, 1]
    for bad in ("", "muscle", "Nope"):
        with pytest.raises(ValueError, match="Tissue not found"):
            pmed.assign_medium_properties(Medium(Tissue=bad))
    c = pmed.assign_medium_properties(Medium(Tissue="Custom", SpeedOfSound=1.0))
    assert c.SpeedOfSound == 1.0 and c.present_fields() == ["Tissue", "SpeedOfSound"]


def test_gas_and_liquid() -> None:
    assert pmb.get_gas_properties(Gas(Type="Perfluorobutane")).Density == 9.64
    with pytest.raises(ValueError, match="Gas not found"):
        pmb.get_gas_properties(Gas(Type="Air"))
    v = pmed.assign_liquid_properties(config.models.Vessel(Tissue="Blood"), "Plasma")
    assert v.present_fields() == ["Tissue"]


def test_sound_speed_extrema_nan_is_omitted() -> None:
    m = Medium(
        SpeedOfSound=1500.0,
        Inhomogeneity=0.1,
        InhomogeneityCutoff=1.0,
        Vessel=config.models.Vessel(SpeedOfSound=math.nan),
    )
    out = pmed.compute_sound_speed_extrema(m)
    assert out.SpeedOfSoundMinimum == 1500 * (1 - 0.1) and out.SpeedOfSoundMaximum == 1500 * 1.1


def test_catalogue_and_shipped_model() -> None:
    c = ptr.get_transducer_properties(Transducer(Type="C1-6D"))
    model = data.load("C1-6D.mat")["Transducer"]
    assert c.ImpulseResponseType == "Calibrated" and c.calibrated == 1.0
    assert c.Pitch == model["Pitch"][0, 0]
    np.testing.assert_array_equal(c.TransmitImpulseResponse, model["TransmitImpulseResponse"])
    unknown = ptr.get_transducer_properties(Transducer(Type="Mystery"))
    assert unknown.present_fields() == ["Type"]
    custom = ptr.get_transducer_properties(
        Transducer(Type="Custom linear", ImpulseResponseType="Calibrated", Configuration="Curved")
    )
    assert custom.ImpulseResponseType == "Load file" and custom.Configuration == "Linear"


def test_load_transducer_model_formats(tmp_path: Path) -> None:
    sio.savemat(tmp_path / "P4-1.mat", {"Trans": {"IR1wy": np.array([[0.25], [0.5]])}})
    t = ptr.load_transducer_model(Transducer(Type="P4-1"), tmp_path)
    np.testing.assert_array_equal(t.TransmitImpulseResponse, [[0.25 * 250e6, 0.5 * 250e6]])
    assert t.SamplingRate == 250e6 and t.calibrated == 1.0
    sio.savemat(
        tmp_path / "X.mat",
        {
            "Transducer": {
                "IR_trans": np.array([[1.0], [2.0]]),
                "IR_rec": np.array([[3.0]]),
                "Fs_trans": 1e8,
                "Fs_rec": 2e8,
            }
        },
    )
    with pytest.raises(ValueError, match="samplingRate"):
        ptr.load_transducer_model(Transducer(Type="X"), tmp_path)
    assert ptr.load_transducer_model(Transducer(Type="none"), tmp_path).present_fields() == ["Type"]


def test_linspace_endpoints() -> None:
    assert pmb.linspace(0.5, 6, 1001)[0, -1] == 6.0
    assert pmb.linspace(1, 2, 1).tolist() == [[2.0]]
    assert pmb.linspace(1, 2, 0).shape == (1, 0)


# --------------------------------------------------------------------- data


def test_package_data() -> None:
    names = data.names()
    assert "tissue_properties.mat" in names and "TransmitTransferFunctionFit.mat" in names
    assert data.geometry_folders() == ["mouse_brain", "renal_tree"]
    assert data.transducer_models() == ["C1-6D"]
    t = data.load("TwoWayTransferFunctionFit.mat")
    assert t["Vfit"].dtype == np.complex128 and t["Vfit"].shape == (1, 50000)
    assert t["Fs"].dtype == np.float64 and t["Fs"][0, 0] == 250e6
    a = data.load("fit_SigmaR_evaluated.mat")
    a["A_m"][0, 0] = -1
    assert data.load("fit_SigmaR_evaluated.mat")["A_m"][0, 0] == 0.92  # fresh copy
    assert list(data.load("fit_SigmaR_evaluated.mat", ["sig"])) == ["sig"]
    tp = data.load("tissue_properties.mat")
    assert tp["properties_name"][1, 4].shape == (0, 0)  # B-33
    info = data.manifest()
    assert len(info["oracle_commit"]) == 40
    assert all(len(f["sha256"]) == 64 for f in info["files"].values())
    with pytest.raises(FileNotFoundError):
        data.load("beta_coeff.mat")


def test_transmit_model_has_all_matlab_fields() -> None:
    """Every field the MATLAB code assigns to Transmit is declared (GUI builds on it)."""
    declared = set(Transmit.model_fields)
    assert {"Delays", "Apodization", "VoltageSignal", "PressureSignal", "MechanicalIndex"} <= (
        declared
    )


def test_json_1x1_struct_array_decodes_to_dict() -> None:
    doc = {"__matlab__": "struct", "size": [1, 1], "fields": ["a"], "data": [{"a": 1.0}]}
    out = tree_from_jsonable(doc)
    assert isinstance(out, dict) and out["a"][0, 0] == 1.0


def test_verasonics_integer_ir_saturates(tmp_path: Path) -> None:
    """Q-32: MATLAB computes IR1wy*250e6 in the integer class (round, saturate)."""
    sio.savemat(tmp_path / "P4-1.mat", {"Trans": {"IR1wy": np.array([[1, -2, 0]], dtype=np.int16)}})
    t = ptr.load_transducer_model(Transducer(Type="P4-1"), tmp_path)
    np.testing.assert_array_equal(t.TransmitImpulseResponse, [[32767.0, -32768.0, 0.0]])


def test_missing_type_raises() -> None:
    with pytest.raises(ValueError, match="Type"):
        ptr.get_transducer_properties(Transducer())
    with pytest.raises(ValueError, match="Type"):
        ptr.load_transducer_model(Transducer())
