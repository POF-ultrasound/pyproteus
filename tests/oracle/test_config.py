"""Oracle tests for ``pyproteus.config``: defaults, lookup leaves, .mat readers.

MATLAB values are compared as *flattened trees*: one entry per node with its
MATLAB class, size, field names and values (real and imaginary parts as
doubles, so integer classes and logicals compare exactly). The MATLAB side is
``pyp_flatten`` (written to a temporary folder at session start), the Python
side :func:`flatten_tree`. Comparison is bit-exact (sign of zero included)
unless a path is listed with a relative tolerance.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pyproteus import config
from pyproteus.config import properties as prop
from pyproteus.config.matfile import StructArray, loadmat
from pyproteus.config.models import Gas, Geometry, Medium, Transducer, Vessel
from tests.oracle.conftest import oracle_path
from tests.oracle.helpers import assert_rel, assert_same_float

PYP_FLATTEN = textwrap.dedent(
    r"""
    function out = pyp_flatten(v, prefix)
    %PYP_FLATTEN One row {path, class, size, re, im, text} per node of v.
    if nargin < 2, prefix = ''; end
    out = {};
    sz = double(size(v));
    if isstruct(v)
        f = fieldnames(v);
        out{end+1} = {prefix, 'struct', sz, zeros(1,0), zeros(1,0), strjoin(f', ',')};
        if isscalar(v)
            for k = 1:numel(f)
                out = [out, pyp_flatten(v.(f{k}), [prefix '.' f{k}])]; %#ok<AGROW>
            end
        else
            for i = 1:numel(v)
                out = [out, pyp_flatten(v(i), sprintf('%s(%d)', prefix, i))]; %#ok<AGROW>
            end
        end
    elseif iscell(v)
        out{end+1} = {prefix, 'cell', sz, zeros(1,0), zeros(1,0), ''};
        for i = 1:numel(v)
            out = [out, pyp_flatten(v{i}, sprintf('%s{%d}', prefix, i))]; %#ok<AGROW>
        end
    elseif ischar(v)
        if isempty(v)
            t = '';
        else
            t = strjoin(num2cell(v, 2)', newline);
        end
        out{end+1} = {prefix, 'char', sz, zeros(1,0), zeros(1,0), t};
    elseif isnumeric(v) || islogical(v)
        x = reshape(v, 1, []);
        out{end+1} = {prefix, class(v), sz, double(real(x)), double(imag(x)), ''};
    else
        error('pyp_flatten:class', 'unsupported class %s at %s', class(v), prefix);
    end
    end
    """
)

_CLASS = {
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

Entry = tuple[str, tuple[int, ...] | None, np.ndarray, np.ndarray, str]


@pytest.fixture(scope="session")
def meng(matlab_engine: Any, tmp_path_factory: pytest.TempPathFactory) -> Any:
    """The shared engine with ``pyp_flatten`` on the path."""
    folder = tmp_path_factory.mktemp("pyp_flatten")
    (folder / "pyp_flatten.m").write_text(PYP_FLATTEN)
    matlab_engine.addpath(str(folder), nargout=0)
    return matlab_engine


def _row(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64).reshape(-1)


def matlab_flatten(eng: Any, expr: str) -> dict[str, Entry]:
    """Evaluate ``expr`` in MATLAB and flatten it. Paths start at '' (the root)."""
    eng.eval(f"pyp_v__ = {expr}; pyp_f__ = pyp_flatten(pyp_v__);", nargout=0)
    rows = eng.workspace["pyp_f__"]
    eng.eval("clear pyp_v__ pyp_f__", nargout=0)
    out: dict[str, Entry] = {}
    for path, cls, size, re, im, text in rows:
        # The engine returns an empty char ('') as an empty double.
        path = path if isinstance(path, str) else ""
        text = text if isinstance(text, str) else ""
        shape = tuple(int(n) for n in _row(size))
        if cls == "char" and text == "":
            shape = None  # '' of any empty size is "" in Python
        out[path] = (cls, shape, _row(re), _row(im), text)
    return out


def flatten_tree(v: Any, prefix: str = "") -> dict[str, Entry]:
    """Python counterpart of ``pyp_flatten`` for a MATLAB value tree."""
    empty = np.zeros(0)
    out: dict[str, Entry] = {}
    if isinstance(v, dict):
        out[prefix] = ("struct", (1, 1), empty, empty, ",".join(v))
        for k, x in v.items():
            out.update(flatten_tree(x, f"{prefix}.{k}"))
    elif isinstance(v, str):
        out[prefix] = ("char", (1, len(v)) if v else None, empty, empty, v)
    elif isinstance(v, StructArray):
        out[prefix] = ("struct", v.shape, empty, empty, ",".join(v.fields))
        for i, e in enumerate(v.elements.reshape(-1, order="F")):
            assert list(e) == v.fields, f"{prefix}({i + 1}): fields {list(e)} != {v.fields}"
            out.update(flatten_tree(e, f"{prefix}({i + 1})"))
    elif isinstance(v, np.ndarray) and v.dtype == object:
        out[prefix] = ("cell", v.shape, empty, empty, "")
        for i, e in enumerate(v.reshape(-1, order="F")):
            out.update(flatten_tree(e, f"{prefix}{{{i + 1}}}"))
    elif isinstance(v, np.ndarray) and v.dtype.kind == "U":
        rows = [str(s) for s in v.reshape(-1)]
        width = max(len(s) for s in rows)
        out[prefix] = ("char", (len(rows), width), empty, empty, "\n".join(rows))
    elif isinstance(v, np.ndarray):
        assert v.ndim >= 2, f"{prefix}: tree arrays must be ≥ 2-D, got {v.shape}"
        x = v.reshape(-1, order="F")
        out[prefix] = (
            _CLASS[v.dtype],
            v.shape,
            np.real(x).astype(np.float64),
            np.imag(x).astype(np.float64) if np.iscomplexobj(x) else np.zeros(x.size),
            "",
        )
    else:
        raise TypeError(f"{prefix}: not a tree value: {type(v)!r}")
    return out


def assert_same_tree(
    actual: dict[str, Entry],
    expected: dict[str, Entry],
    *,
    rel: dict[str, float] | None = None,
    field_order: bool = True,
    skip: tuple[str, ...] = (),
) -> None:
    """Compare flattened trees: same nodes, classes, sizes, field names, values.

    ``rel`` maps paths to a relative tolerance (else bit-exact). With
    ``field_order=False`` struct field names are compared as sets. Paths in
    ``skip`` (and their children) are removed from both sides first.
    """
    rel = rel or {}

    def keep(p: str) -> bool:
        return not any(p == s or p.startswith(s + ".") for s in skip)

    a = {p: e for p, e in actual.items() if keep(p)}
    e = {p: x for p, x in expected.items() if keep(p)}
    assert set(a) == set(e), (
        f"nodes differ: only in Python {sorted(set(a) - set(e))}, "
        f"only in MATLAB {sorted(set(e) - set(a))}"
    )
    for p in e:
        ca, sa, ra, ia, ta = a[p]
        ce, se, re_, ie, te = e[p]
        assert ca == ce, f"{p}: class {ca} != MATLAB {ce}"
        assert sa == se, f"{p}: size {sa} != MATLAB {se}"
        if ca == "struct":
            fa = [f for f in ta.split(",") if keep(f"{p}.{f}")] if ta else []
            fe = [f for f in te.split(",") if keep(f"{p}.{f}")] if te else []
            if field_order:
                assert fa == fe, f"{p}: fields {fa} != MATLAB {fe}"
            else:
                assert sorted(fa) == sorted(fe), f"{p}: fields {fa} != MATLAB {fe}"
            continue
        assert ta == te, f"{p}: text {ta!r} != MATLAB {te!r}"
        if p in rel:
            assert_rel(ra, re_, rel=rel[p], err_msg=p)
            assert_rel(ia, ie, rel=rel[p], err_msg=p)
        else:
            assert_same_float(ra, re_, err_msg=p)
            assert_same_float(ia, ie, err_msg=p)


def py_struct(model: Any) -> dict[str, Entry]:
    return flatten_tree(model.to_tree())


# ----------------------------------------------------------------- defaults

#: Probabilities = R.^2.*exp(-a*R)/sum(...): MATLAB exp/sum differ by ulps (B-31, B-32).
PROB_REL = {".Distribution.Probabilities": 1e-12}


def _pending(group: str) -> tuple[str, ...]:
    return tuple(f".{f}" for f in config.PENDING_FIELDS.get(group, ()))


def test_reset_acquisition(meng: Any) -> None:
    assert_same_tree(
        py_struct(config.reset_acquisition()),
        matlab_flatten(meng, "reset_acquisition()"),
    )


def test_reset_simulation_parameters(meng: Any) -> None:
    assert_same_tree(
        py_struct(config.reset_simulation_parameters()),
        matlab_flatten(meng, "reset_simulation_parameters()"),
    )


def test_reset_microbubble(meng: Any) -> None:
    assert_same_tree(
        py_struct(config.reset_microbubble()),
        matlab_flatten(meng, "reset_microbubble()"),
        rel=PROB_REL,
    )


def test_reset_medium(meng: Any) -> None:
    assert_same_tree(
        py_struct(config.reset_medium()),
        matlab_flatten(meng, "reset_medium()"),
    )


def test_reset_transducer(meng: Any) -> None:
    assert_same_tree(
        py_struct(config.reset_transducer()),
        matlab_flatten(meng, "reset_transducer()"),
        skip=_pending("Transducer"),
    )


CATALOGUE = ["P4-1", "C1-6D", "L22-14v", "9L-D", "RC6gV"]


@pytest.mark.parametrize("ttype", [*CATALOGUE, "no-configuration"])
def test_reset_transmit(meng: Any, ttype: str) -> None:
    mtype = "P4-1" if ttype == "no-configuration" else ttype
    meng.eval(
        f"pyp_T__ = get_transducer_properties(struct('Type','{mtype}'));"
        "if strcmp(pyp_T__.ImpulseResponseType,'Estimate'), pyp_T__.SamplingRate = 250e6;"
        " pyp_T__ = estimate_impulse_response(pyp_T__); end",
        nargout=0,
    )
    T = prop.get_transducer_properties(Transducer(Type=mtype))
    if ttype == "no-configuration":
        meng.eval("pyp_T__ = rmfield(pyp_T__, 'Configuration');", nargout=0)
        T.rmfield("Configuration")
    expected = matlab_flatten(meng, "reset_transmit(pyp_T__)")
    meng.eval("clear pyp_T__", nargout=0)
    assert_same_tree(
        py_struct(config.reset_transmit(T)),
        expected,
        skip=_pending("Transmit"),
    )


def _paths_struct(oracle_path: Path) -> str:
    return f"struct('GeometriesPath', '{oracle_path / 'geometry_data'}')"


@pytest.mark.parametrize("folder", ["renal_tree", "mouse_brain", None])
def test_reset_geometry(meng: Any, oracle: Any, folder: str | None) -> None:
    paths = _paths_struct(oracle.path)
    if folder is None:  # no Folder field: MATLAB picks renal_tree
        expected = matlab_flatten(meng, f"reset_geometry(struct(), {paths})")
        geometry = Geometry()
    else:
        expected = matlab_flatten(meng, f"reset_geometry(struct('Folder','{folder}'), {paths})")
        geometry = Geometry(Folder=folder)
    py = config.reset_geometry(geometry)
    assert_same_tree(py_struct(py), expected)
    # Same result when reading the oracle's GeometryProperties.mat directly:
    py_file = config.reset_geometry(geometry, oracle.path / "geometry_data")
    assert_same_tree(py_struct(py_file), expected)


def test_reset_geometry_keeps_other_fields(meng: Any, oracle: Any) -> None:
    paths = _paths_struct(oracle.path)
    meng.eval(
        "pyp_G__ = struct('Folder','mouse_brain','startDepth',1,'Extra',[1 2 3]);"
        "pyp_G__.Domain.Xmin = -0.5; pyp_G__.Domain.Manual = true;"
        "pyp_G__.BoundingBox.Other = 'x';",
        nargout=0,
    )
    expected = matlab_flatten(meng, f"reset_geometry(pyp_G__, {paths})")
    meng.eval("clear pyp_G__", nargout=0)
    G = Geometry(Folder="mouse_brain", startDepth=1.0, Extra=np.array([[1.0, 2.0, 3.0]]))
    G.Domain = config.models.Domain(Xmin=-0.5, Manual=True)
    G.BoundingBox = config.models.BoundingBox(Other="x")
    assert_same_tree(py_struct(config.reset_geometry(G)), expected)


# --------------------------------------------------------- lookup functions


@pytest.mark.parametrize("ttype", [*CATALOGUE, "Unknown type"])
def test_get_transducer_properties(meng: Any, ttype: str) -> None:
    expected = matlab_flatten(meng, f"get_transducer_properties(struct('Type','{ttype}'))")
    actual = py_struct(prop.get_transducer_properties(Transducer(Type=ttype)))
    assert_same_tree(actual, expected)


@pytest.mark.parametrize("irtype", ["Calibrated", "Estimate", None])
def test_get_transducer_properties_custom_linear(meng: Any, irtype: str | None) -> None:
    fields = "'Type','Custom linear','Configuration','Curved','Radius',0.05,'calibrated',1,"
    fields += "'NumberOfElements',64,'Pitch',1e-4"
    py = Transducer(
        Type="Custom linear",
        Configuration="Curved",
        Radius=0.05,
        calibrated=1.0,
        NumberOfElements=64.0,
        Pitch=1e-4,
    )
    if irtype is not None:
        fields += f",'ImpulseResponseType','{irtype}'"
        py.ImpulseResponseType = irtype
    expected = matlab_flatten(meng, f"get_transducer_properties(struct({fields}))")
    assert_same_tree(py_struct(prop.get_transducer_properties(py)), expected)


# Model files written by MATLAB into a temporary folder: (name, MATLAB code
# defining the variables, variables to save, save option).
MODEL_FILES = {
    "verasonics": ("Trans.IR1wy = [0 0.25 0.5 0.25 0]';", "Trans", "-v7"),
    "verasonics_single": ("Trans.IR1wy = single([0.1; -0.3; 0.7]); Trans.x = 1;", "Trans", "-v7"),
    "proteus_new": (
        "Transducer.TransmitImpulseResponse = [1;2;3]; Transducer.ReceiveImpulseResponse = "
        "[4 5; 6 7]; Transducer.SamplingRate = 1e8; Transducer.Pitch = 3e-4; "
        "Transducer.Configuration = 'Linear'; Transducer.NumberOfElements = 7;",
        "Transducer",
        "-v7",
    ),
    "proteus_old": (
        "Transducer.IR_trans = int16([1 -2 3]); Transducer.IR_rec = [0.5 0.25];"
        "Transducer.Fs_trans = 2e8; Transducer.Fs_rec = 2e8; Transducer.Radius = 0.04;"
        "Transducer.BandwidthLow = 1e6;",
        "Transducer",
        "-v7",
    ),
    "proteus_v73": (
        "Transducer.IR_trans = [1 -2 3]; Transducer.IR_rec = [0.5; 0.25];"
        "Transducer.Fs_trans = 2e8; Transducer.Fs_rec = 2e8; Transducer.ElementHeight = 0.01;",
        "Transducer",
        "-v7.3",
    ),
    "proteus_both_forms": (
        "Transducer.TransmitImpulseResponse = [1 2]; Transducer.ReceiveImpulseResponse = [3 4];"
        "Transducer.SamplingRate = 1e8; Transducer.IR_trans = [9 9]; Transducer.IR_rec = [9 9];"
        "Transducer.Fs_trans = 1; Transducer.Fs_rec = 2;",
        "Transducer",
        "-v7",
    ),
}

MODEL_ERRORS = {
    "fs_mismatch": (
        "Transducer.IR_trans = [1 2]; Transducer.IR_rec = [1 2]; Transducer.Fs_trans = 1e8;"
        "Transducer.Fs_rec = 2e8;",
        "Transducer",
        "load_transducer_model:samplingRate",
    ),
    "no_ir": ("Transducer.Pitch = 1;", "Transducer", "load_transducer_model:noImpulseResponse"),
    "trans_no_ir": ("Trans.Other = 1;", "Trans", "load_transducer_model:noImpulseResponse"),
    "unknown": ("Other = 1;", "Other", "load_transducer_model:unknownFormat"),
}


def _write_model(eng: Any, folder: Path, ttype: str, code: str, var: str, opt: str) -> None:
    file = folder / f"{ttype}.mat"
    eng.eval(f"{code} save('{file}', '{var}', '{opt}'); clear {var}", nargout=0)


@pytest.mark.parametrize("case", sorted(MODEL_FILES))
def test_load_transducer_model(meng: Any, tmp_path: Path, case: str) -> None:
    code, var, opt = MODEL_FILES[case]
    _write_model(meng, tmp_path, "P4-1", code, var, opt)
    meng.eval("pyp_T__ = get_transducer_properties(struct('Type','P4-1'));", nargout=0)
    expected = matlab_flatten(meng, f"load_transducer_model(pyp_T__, '{tmp_path}')")
    meng.eval("clear pyp_T__", nargout=0)
    T = prop.get_transducer_properties(Transducer(Type="P4-1"))
    actual = py_struct(prop.load_transducer_model(T, tmp_path))
    assert_same_tree(actual, expected)


def test_load_transducer_model_no_file(meng: Any, tmp_path: Path) -> None:
    expected = matlab_flatten(meng, f"load_transducer_model(struct('Type','P4-1'), '{tmp_path}')")
    actual = py_struct(prop.load_transducer_model(Transducer(Type="P4-1"), tmp_path))
    assert_same_tree(actual, expected)


@pytest.mark.parametrize("case", sorted(MODEL_ERRORS))
def test_load_transducer_model_errors(meng: Any, tmp_path: Path, case: str) -> None:
    import matlab.engine

    code, var, ident = MODEL_ERRORS[case]
    _write_model(meng, tmp_path, "P4-1", code, var, "-v7")
    with pytest.raises(matlab.engine.MatlabExecutionError):
        meng.eval(f"load_transducer_model(struct('Type','P4-1'), '{tmp_path}');", nargout=0)
    meng.eval(
        f"try, load_transducer_model(struct('Type','P4-1'), '{tmp_path}'); "
        "pyp_id__ = ''; catch e, pyp_id__ = e.identifier; end",
        nargout=0,
    )
    assert meng.workspace["pyp_id__"] == ident
    with pytest.raises(ValueError, match=ident):
        prop.load_transducer_model(Transducer(Type="P4-1"), tmp_path)


@pytest.mark.parametrize("gas", ["Sulfur hexafluoride", "Perfluoropropane", "Perfluorobutane"])
def test_get_gas_properties(meng: Any, gas: str) -> None:
    expected = matlab_flatten(meng, f"get_gas_properties(struct('Type','{gas}'))")
    assert_same_tree(py_struct(prop.get_gas_properties(Gas(Type=gas))), expected)


def test_get_gas_properties_unknown(meng: Any) -> None:
    import matlab.engine

    with pytest.raises(matlab.engine.MatlabExecutionError, match="Gas not found"):
        meng.eval("get_gas_properties(struct('Type','Air'));", nargout=0)
    with pytest.raises(ValueError, match="Gas not found"):
        prop.get_gas_properties(Gas(Type="Air"))


def _tissue_names() -> list[str]:
    from pyproteus import data

    return [str(s) for s in data.load("tissue_properties.mat")["tissue_names"].reshape(-1)]


# Substrings select the first matching row (Q-30); 'ate' is in 'Water'.
TISSUES = [
    *_tissue_names(),
    "General tissue",
    "Muscle",
    "Eye",
    "Bone",
    "Teeth",
    "ate",
    "(",
]


@pytest.mark.parametrize("tissue", TISSUES)
def test_assign_medium_properties(meng: Any, tissue: str) -> None:
    expected = matlab_flatten(meng, f"assign_medium_properties(struct('Tissue','{tissue}'))")
    actual = py_struct(prop.assign_medium_properties(Medium(Tissue=tissue)))
    assert_same_tree(actual, expected)


def test_assign_medium_properties_custom(meng: Any) -> None:
    expr = "assign_medium_properties(struct('Tissue','Custom','SpeedOfSound',1234,'X',1))"
    expected = matlab_flatten(meng, expr)
    actual = prop.assign_medium_properties(Medium(Tissue="Custom", SpeedOfSound=1234.0, X=1.0))
    assert_same_tree(py_struct(actual), expected)


@pytest.mark.parametrize("tissue", ["", "blood", "Nope"])
def test_assign_medium_properties_not_found(meng: Any, tissue: str) -> None:
    import matlab.engine

    # strfind(name, '') is empty: an empty name matches no tissue.
    with pytest.raises(matlab.engine.MatlabExecutionError, match="Tissue not found"):
        meng.eval(f"assign_medium_properties(struct('Tissue','{tissue}'));", nargout=0)
    with pytest.raises(ValueError, match="Tissue not found"):
        prop.assign_medium_properties(Medium(Tissue=tissue))


@pytest.mark.parametrize("liquid", ["Blood", "Water", "Plasma"])
def test_assign_liquid_properties(meng: Any, liquid: str) -> None:
    expected = matlab_flatten(
        meng, f"assign_liquid_properties(struct('Tissue','Blood','Density',1), '{liquid}')"
    )
    actual = prop.assign_liquid_properties(Vessel(Tissue="Blood", Density=1.0), liquid)
    assert_same_tree(py_struct(actual), expected)


@pytest.mark.parametrize(
    ("c", "inhom", "cutoff", "cv"),
    [
        (1540, 0.02, 2, 1575),
        (1540, 0.0, 2, 1540),
        (1500, 0.1, 3, 1400),
        (1500, 0.1, 3, float("nan")),
        (float("nan"), 0.1, 3, 1400),
        (1600, 0.05, float("inf"), 1500),
    ],
)
def test_compute_sound_speed_extrema(
    meng: Any, c: float, inhom: float, cutoff: float, cv: float
) -> None:
    def m(x: float) -> str:
        return repr(x).replace("nan", "NaN").replace("inf", "Inf")

    expected = matlab_flatten(
        meng,
        f"compute_sound_speed_extrema(struct('SpeedOfSound',{m(c)},'Inhomogeneity',{m(inhom)},"
        f"'InhomogeneityCutoff',{m(cutoff)},'Vessel',struct('SpeedOfSound',{m(cv)})))",
    )
    medium = Medium(
        SpeedOfSound=c,
        Inhomogeneity=inhom,
        InhomogeneityCutoff=cutoff,
        Vessel=Vessel(SpeedOfSound=cv),
    )
    assert_same_tree(py_struct(prop.compute_sound_speed_extrema(medium)), expected)


def test_compute_polydisperse_distribution(meng: Any) -> None:
    meng.eval("[pyp_R__, pyp_P__] = compute_polydisperse_distribution();", nargout=0)
    R = np.asarray(meng.workspace["pyp_R__"])
    P = np.asarray(meng.workspace["pyp_P__"])
    meng.eval("clear pyp_R__ pyp_P__", nargout=0)
    r, p = prop.compute_polydisperse_distribution()
    assert_same_float(r, R)
    assert_rel(p, P, rel=1e-12)  # B-31/B-32: exp and sum differ by ulps
    assert p.shape == P.shape == (1, 1001)


@pytest.mark.parametrize(
    ("d1", "d2", "n"),
    [(0.5, 6, 1001), (0.5e-6, 6e-6, 1000), (0, 1, 7), (-3, 2.5, 12), (1, 1, 5), (2, 1, 4)],
)
def test_linspace(meng: Any, d1: float, d2: float, n: int) -> None:
    expected = np.asarray(meng.eval(f"linspace({d1!r}, {d2!r}, {n})"))
    assert_same_float(prop.linspace(d1, d2, n), expected.reshape(1, -1))


# ---------------------------------------------------------------- readers


def _build_settings(eng: Any, oracle_path: Path, var: str = "S") -> None:
    """MainGUI start-up + updateParameters, plus extra fields, in MATLAB variable ``var``."""
    paths = _paths_struct(oracle_path)
    eng.eval(
        f"{var}.Microbubble = reset_microbubble(); {var}.Medium = reset_medium();"
        f"{var}.Acquisition = reset_acquisition(); {var}.Transducer = reset_transducer();"
        f"{var}.Transmit = reset_transmit({var}.Transducer);"
        f"{var}.SimulationParameters = reset_simulation_parameters();"
        f"{var}.Geometry = reset_geometry(struct('Folder','renal_tree'), {paths});"
        f"{var}.Geometry.Domain.Margin = 2*{var}.Medium.SpeedOfSound/"
        f"{var}.Transmit.CenterFrequency;"
        f"{var}.Geometry = compute_simulation_domain({var}.Geometry, {var}.Transducer,"
        f" {var}.Transmit);"
        f"{var}.SimulationParameters = update_simulation_parameters("
        f"{var}.SimulationParameters, {var}.Medium, {var}.Transmit.CenterFrequency);"
        f"{var}.Transducer.integration_points = reshape(1:24, 2, 4, 3);"
        f"{var}.Microbubble.Shell.FitFile = '/some/fit file.mat';"
        f"{var}.Transducer.cellfield = {{'a', [1 2; 3 4]; int8(-3), {{}}}};"
        f"{var}.Medium.Vessel.sarray = struct('x', {{1, 'b', []}});",
        nargout=0,
    )


@pytest.mark.parametrize("version", ["-v7", "-v7.3", "-v6"])
def test_load_settings_mat(meng: Any, oracle: Any, tmp_path: Path, version: str) -> None:
    file = tmp_path / "settings.mat"
    _build_settings(meng, oracle.path)
    meng.eval(
        f"write_to_file(S.Microbubble, S.SimulationParameters, S.Geometry, S.Transducer,"
        f" S.Acquisition, S.Medium, S.Transmit, struct(), '{file}');"
        f"pyp_L__ = load('{file}'); save('{file}', '-struct', 'pyp_L__', '{version}');"
        "clear S",
        nargout=0,
    )
    settings = config.load_settings_mat(file)
    for group in config.SETTINGS_GROUPS:
        expected = matlab_flatten(meng, f"pyp_L__.{group}")
        assert_same_tree(py_struct(getattr(settings, group)), expected)
    # The raw reader agrees too (no model involved):
    raw = loadmat(file)
    order = meng.eval(f"fieldnames(load('{file}'))'")  # v7.3: MATLAB sorts by name
    assert list(raw) == list(order)
    meng.eval("clear pyp_L__", nargout=0)
    # JSON round trip of a real settings file is lossless:
    again = config.settings_from_json(config.settings_to_json(settings))
    assert config.settings_to_json(again) == config.settings_to_json(settings)
    for group in config.SETTINGS_GROUPS:
        assert_same_tree(py_struct(getattr(again, group)), py_struct(getattr(settings, group)))


@pytest.mark.parametrize("version", ["-v7", "-v7.3"])
def test_ground_truth_frames(meng: Any, tmp_path: Path, version: str) -> None:
    # Same format as generate_streamlines.m (and the oracle's
    # write_synthetic_ground_truth): 10 frames → Frame_01 ... Frame_10.
    meng.eval(
        "pyp_s__ = RandStream('mt19937ar','Seed',3); N = 5;"
        "FlowSimulationParameters = struct('TimeBtwPulse',1e-4,'FrameRate',500,"
        "'NBPulses',2,'NMicrobubbles',N,'NumberOfFrames',10);"
        "FlowSimulationParameters.Microbubble.Distribution.Probabilities = [0.25 0.75];"
        "FlowSimulationParameters.Microbubble.Distribution.Radii = [1e-6 2e-6];"
        f"save(fullfile('{tmp_path}','FlowSimulationParameters.mat'),"
        f"'FlowSimulationParameters','{version}');"
        "for m = 1:10, Frame = struct(); for n = 1:2, p = ['Pulse' num2str(n)];"
        "Frame.(p).Points = rand(pyp_s__, N, 3); Frame.(p).Velocity = randn(pyp_s__, N, 3);"
        "Frame.(p).Radius = rand(pyp_s__, N, 1); Frame.(p).StreamNumber = (1:N)'; end;"
        f"save(fullfile('{tmp_path}',['Frame_' num2str(m,'%02i') '.mat']),'Frame',"
        f"'{version}'); end;"
        "clear pyp_s__ N Frame FlowSimulationParameters m n p",
        nargout=0,
    )
    expected = matlab_flatten(
        meng,
        f"getfield(load(fullfile('{tmp_path}','FlowSimulationParameters.mat')),"
        "'FlowSimulationParameters')",
    )
    assert_same_tree(flatten_tree(config.load_flow_simulation_parameters(tmp_path)), expected)
    for frame in (1, 7, 10):
        name = config.frame_file_name(frame, 10)
        expected = matlab_flatten(meng, f"getfield(load(fullfile('{tmp_path}','{name}')), 'Frame')")
        assert_same_tree(flatten_tree(config.load_frame(tmp_path, frame, 10)), expected)


@pytest.mark.parametrize(("frame", "n"), [(3, 10), (10, 10), (1, 9), (42, 100), (7, 1000)])
def test_frame_file_name(meng: Any, frame: int, n: int) -> None:
    expected = meng.eval(
        f"['Frame_', num2str({frame},['%0',num2str(length(num2str({n}))),'i']),'.mat']"
    )
    assert config.frame_file_name(frame, n) == expected


def _shipped_mat_files(oracle_path: Path) -> list[str]:
    folders = [
        "example_custom_data",
        "Transducer-models",
        "GUIfunctions",
        "microbubble-simulator/functions",
        "geometry_data/renal_tree",
        "geometry_data/mouse_brain",
        "tests/baseline",
    ]
    return sorted(
        str(f.relative_to(oracle_path)) for d in folders for f in (oracle_path / d).glob("*.mat")
    )


SHIPPED_MAT = _shipped_mat_files(oracle_path()) if oracle_path().is_dir() else []


@pytest.mark.parametrize("rel", SHIPPED_MAT)
def test_loadmat_shipped_files(meng: Any, oracle: Any, tmp_path: Path, rel: str) -> None:
    """Every shipped .mat: Python loadmat == MATLAB load, also after a v7.3 re-save."""
    src = oracle.path / rel
    expected = matlab_flatten(meng, f"load('{src}')")
    assert_same_tree(flatten_tree(loadmat(src)), expected)
    v73 = tmp_path / "v73.mat"
    meng.eval(f"pyp_L__ = load('{src}'); save('{v73}', '-struct', 'pyp_L__', '-v7.3');", nargout=0)
    meng.eval("clear pyp_L__", nargout=0)
    assert_same_tree(flatten_tree(loadmat(v73)), matlab_flatten(meng, f"load('{v73}')"))
