"""Reference export through the MATLAB Engine: orientation, classes, provenance.

The probe values are built in MATLAB with known column-major numbering, so the
checks do not depend on the exporter's own conversion code. Values read back
from the HDF5 file are also sent back to MATLAB and compared there with
``isequaln`` plus ``class``/``size``.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import scipy.sparse as sp
from tools.export_reference.engine import KWAVE_BINARY_NAME, ExportConfig, ExportSession
from tools.export_reference.export import export_spec
from tools.export_reference.specs import SPECS, ExportSpec

from pyproteus.io.reference_format import (
    FunctionHandle,
    MatlabString,
    StructArray,
    read_reference,
)
from tests.oracle.helpers import from_matlab, to_matlab

PROBE = """
pyp_out.a3 = reshape(1:24, 2, 3, 4);
pyp_out.a4 = reshape(single(1:24), 2, 1, 3, 4);
pyp_out.row = 1:5;
pyp_out.col = (1:5)';
pyp_out.scalar = pi;
pyp_out.negzero = [-0 0 NaN Inf -Inf];
pyp_out.cplx = reshape(1:6, 2, 3) + 1i*reshape(7:12, 2, 3);
pyp_out.cplx_real_part_only = complex(1, 0);
pyp_out.logic = logical([1 0 1; 0 0 1]);
pyp_out.i8 = int8([-128 127]);
pyp_out.u16 = uint16([0; 65535]);
pyp_out.i64 = int64(2)^62;
pyp_out.u8s = uint8(5);
pyp_out.u64s = intmax('uint64');
pyp_out.i32s = int32(-7);
pyp_out.empty03 = zeros(0, 3);
pyp_out.empty = [];
pyp_out.sp = sparse([1 2 2], [3 1 3], [1.5 -2 4], 2, 4);
pyp_out.splog = sparse(logical([0 1; 0 0]));
pyp_out.txt = 'Three-level';
pyp_out.emptychar = '';
pyp_out.str = "a string";
pyp_out.fh = @(x) x.^2 + 1;
pyp_out.nested.inner.x = 1;
pyp_out.nested.empty = struct();
pyp_out.c = {1, 'b'; [1 2 3], {2, "s"}; int32(5), []};
pyp_out.sa = struct('a', {1, 2; 3, 4}, 'b', {'p', 'q'; 'r', 's'});
pyp_out.sarow = struct('q', {10, 20, 30});
"""


def probe_spec() -> ExportSpec:
    return ExportSpec(
        name="probe.layout",
        module="(test)",
        function="pyp_flatten",
        description="orientation probe",
        call=PROBE,
    )


@pytest.fixture(scope="module")
def session(matlab_engine, oracle):
    cfg = ExportConfig.from_env()
    cfg.oracle = oracle.path
    with ExportSession(cfg, engine=matlab_engine) as s:
        yield s


def test_probe_orientation_and_classes(session, tmp_path):
    path = export_spec(session, probe_spec(), tmp_path)
    o = read_reference(path).outputs

    i, j, k = np.indices((2, 3, 4))
    np.testing.assert_array_equal(o["a3"], 1.0 + i + 2 * j + 6 * k)
    assert o["a3"].dtype == np.float64
    i, j, k, m = np.indices((2, 1, 3, 4))
    np.testing.assert_array_equal(o["a4"], (1 + i + 2 * j + 2 * k + 6 * m).astype(np.float32))
    assert o["a4"].dtype == np.float32
    assert o["row"].shape == (1, 5) and o["col"].shape == (5, 1)
    assert o["scalar"].shape == (1, 1) and o["scalar"][0, 0] == np.pi
    nz = o["negzero"]
    assert np.signbit(nz[0, 0]) and not np.signbit(nz[0, 1]) and np.isnan(nz[0, 2])
    np.testing.assert_array_equal(
        o["cplx"], np.arange(1, 7).reshape(3, 2).T + 1j * np.arange(7, 13).reshape(3, 2).T
    )
    assert np.iscomplexobj(o["cplx_real_part_only"])
    np.testing.assert_array_equal(o["logic"], [[True, False, True], [False, False, True]])
    assert o["logic"].dtype == np.bool_
    assert o["i8"].dtype == np.int8 and o["u16"].shape == (2, 1) and o["u16"][1, 0] == 65535
    assert o["u16"].dtype == np.uint16
    assert o["u8s"].dtype == np.uint8 and o["u8s"].shape == (1, 1) and o["u8s"][0, 0] == 5
    assert o["u64s"].dtype == np.uint64 and o["u64s"][0, 0] == 2**64 - 1
    assert o["i32s"].dtype == np.int32 and o["i32s"][0, 0] == -7
    assert o["i64"].dtype == np.int64 and o["i64"][0, 0] == 2**62
    assert o["empty03"].shape == (0, 3) and o["empty"].shape == (0, 0)
    assert sp.issparse(o["sp"]) and o["sp"].shape == (2, 4) and o["sp"].dtype == np.float64
    np.testing.assert_array_equal(o["sp"].toarray(), [[0, 0, 1.5, 0], [-2, 0, 4, 0]])
    assert o["splog"].dtype == np.bool_
    assert o["txt"] == "Three-level" and o["emptychar"] == ""
    assert type(o["str"]) is MatlabString and o["str"] == "a string"
    assert type(o["fh"]) is FunctionHandle and o["fh"].startswith("@(x)")
    assert list(o["nested"]) == ["inner", "empty"] and o["nested"]["empty"] == {}
    assert o["nested"]["inner"]["x"].shape == (1, 1) and o["nested"]["inner"]["x"][0, 0] == 1
    c = o["c"]
    assert c.shape == (3, 2)
    assert c[0, 0][0, 0] == 1 and c[0, 1] == "b"
    np.testing.assert_array_equal(c[1, 0], [[1.0, 2.0, 3.0]])
    assert c[1, 1].shape == (1, 2) and type(c[1, 1][0, 1]) is MatlabString
    assert c[2, 0].dtype == np.int32 and c[2, 1].shape == (0, 0)
    sa = o["sa"]
    assert isinstance(sa, StructArray) and sa.shape == (2, 2) and sa.fields == ("a", "b")
    assert sa.elements[1, 0]["a"][0, 0] == 3 and sa.elements[0, 1]["b"] == "q"
    assert o["sarow"].shape == (1, 3) and o["sarow"].elements[0, 2]["q"][0, 0] == 30


def send(eng, name, value):
    """Rebuild a value read from a reference file as MATLAB variable ``name``."""
    if isinstance(value, dict):
        eng.eval(f"{name} = struct();", nargout=0)
        for k, v in value.items():
            send(eng, f"{name}.{k}", v)
    elif isinstance(value, StructArray):
        size = ",".join(str(n) for n in value.shape)
        fields = ",".join(f"'{f}',{{[]}}" for f in value.fields) or ""
        eng.eval(f"{name} = repmat(struct({fields}), {size});", nargout=0)
        if not value.fields:
            eng.eval(f"{name} = repmat(struct(), {size});", nargout=0)
        for k, el in enumerate(value.elements.ravel(order="F")):
            for f, v in el.items():
                send(eng, f"{name}({k + 1}).{f}", v)
    elif isinstance(value, np.ndarray) and value.dtype == object:
        size = ",".join(str(n) for n in value.shape)
        eng.eval(f"{name} = cell({size});", nargout=0)
        for k, el in enumerate(value.ravel(order="F")):
            send(eng, f"{name}{{{k + 1}}}", el)
    elif isinstance(value, FunctionHandle):
        eng.workspace["pyp_txt__"] = str(value)
        eng.eval(f"{name} = str2func(pyp_txt__); clear pyp_txt__;", nargout=0)
    elif isinstance(value, MatlabString):
        eng.workspace["pyp_txt__"] = str(value)
        eng.eval(f"{name} = string(pyp_txt__); clear pyp_txt__;", nargout=0)
    elif isinstance(value, str):
        eng.workspace["pyp_txt__"] = value
        eng.eval(f"{name} = char(pyp_txt__); clear pyp_txt__;", nargout=0)
        if not value:
            eng.eval(f"{name} = '';", nargout=0)
    elif sp.issparse(value):
        eng.workspace["pyp_full__"] = to_matlab(value.toarray())
        eng.eval(f"{name} = sparse(pyp_full__); clear pyp_full__;", nargout=0)
    else:
        eng.workspace["pyp_arr__"] = to_matlab(value)
        eng.eval(f"{name} = pyp_arr__; clear pyp_arr__;", nargout=0)


def test_every_value_round_trips_to_matlab(session, tmp_path):
    """Everything read from the file, rebuilt in MATLAB, equals MATLAB's original.

    Compared in MATLAB through pyp_flatten records, which hold class, size,
    sparsity and data of every node (isequaln alone ignores class).
    """
    path = export_spec(session, probe_spec(), tmp_path)
    o = read_reference(path).outputs
    eng = session.eng
    eng.eval(PROBE.replace("pyp_out", "pyp_ref__"), nargout=0)
    try:
        for name, value in o.items():
            if name == "fh":  # handle text is compared in Python (str2func re-parses)
                continue
            send(eng, "pyp_back__", value)
            same = eng.eval(
                f"isequaln(pyp_flatten(pyp_back__), pyp_flatten(pyp_ref__.{name}))", nargout=1
            )
            assert same, name
            eng.eval("clear pyp_back__;", nargout=0)
        assert o["fh"] == eng.eval("func2str(pyp_ref__.fh)", nargout=1)
    finally:
        eng.eval("clear pyp_back__ pyp_ref__;", nargout=0)


def test_reused_engine_state_restored(matlab_engine, oracle, tmp_path):
    eng = matlab_engine
    eng.eval("rng(123,'twister'); pyp_rng_before__ = rng; pyp_path_before__ = path;", nargout=0)
    cwd = eng.pwd(nargout=1)
    env = {v: eng.getenv(v, nargout=1) for v in ("PROTEUS_LOCAL_PATHS", "KWAVE_LIBS")}
    cfg = ExportConfig.from_env()
    cfg.oracle = oracle.path
    with ExportSession(cfg, engine=matlab_engine) as s:
        export_spec(s, SPECS["GUIfunctions.reset_acquisition"], tmp_path)
        assert eng.pwd(nargout=1) != cwd
    try:
        assert eng.eval("isequal(rng, pyp_rng_before__)", nargout=1)
        assert eng.eval("isequal(path, pyp_path_before__)", nargout=1)
        assert eng.pwd(nargout=1) == cwd
        assert {v: eng.getenv(v, nargout=1) for v in env} == env
    finally:
        eng.eval("clear pyp_rng_before__ pyp_path_before__; rng(0,'twister');", nargout=0)


def test_reset_specs_match_direct_calls(session, tmp_path):
    eng = session.eng
    for name, direct in (
        ("GUIfunctions.reset_acquisition", lambda: eng.reset_acquisition(nargout=1)),
        (
            "GUIfunctions.reset_transmit",
            lambda: eng.reset_transmit(eng.reset_transducer(nargout=1), nargout=1),
        ),
    ):
        r = read_reference(export_spec(session, SPECS[name], tmp_path))
        out = next(iter(r.outputs.values()))
        ref = direct()
        assert list(out) == list(ref), name
        for k, v in ref.items():
            if isinstance(v, str):
                assert type(out[k]) is str and out[k] == v, (name, k)
            else:
                expected = from_matlab(v)
                assert out[k].shape == expected.shape, (name, k)
                assert out[k].dtype == expected.dtype, (name, k)
                np.testing.assert_array_equal(out[k], expected, err_msg=k)
        a = r.attrs
        assert a["oracle_commit"].startswith("8d37938")
        assert a["proteus_version"] == "1.1.0"
        assert a["matlab_version"] == str(eng.version(nargout=1))
        assert a["seeds"] == {"matlab_rng_twister": 0}
        assert a["kind"] == "module" and a["function"] == SPECS[name].function
        assert a["call"].endswith(";") and a["created_utc"].endswith("+00:00")


def eng_type(session):
    return session.eng.eval("getfield(reset_transducer(), 'Type')", nargout=1)


def test_transmit_inputs_recorded(session, tmp_path):
    r = read_reference(export_spec(session, SPECS["GUIfunctions.reset_transmit"], tmp_path))
    t = r.inputs["Transducer"]
    assert t["Type"] == eng_type(session)
    sig = r.outputs["Transmit"]["PressureSignal"]
    assert sig.ndim == 2 and sig.shape[0] == 1  # MATLAB row vector stays 1xN


# ----------------------------------------------------------------------------- full run


def _kwave_ready() -> ExportConfig:
    cfg = ExportConfig.from_env()
    if not (cfg.kwave_matlab_path / "kWaveGrid.m").is_file():
        pytest.skip(f"MATLAB k-Wave not found at {cfg.kwave_matlab_path} (KWAVE_MATLAB_PATH)")
    if not (cfg.kwave_binary_dir / KWAVE_BINARY_NAME).is_file():
        pytest.skip("no k-Wave binary (PYPROTEUS_KWAVE_BINARY_DIR)")
    if cfg.kwave_libs is None and not os.environ.get("KWAVE_LIBS"):
        default = cfg.kwave_binary_dir.parent / ".kwave-libs" / "lib"
        if not default.is_dir() and not os.path.isdir("/opt/homebrew/opt/fftw/lib"):
            pytest.skip("k-Wave binary runtime libraries missing (PYPROTEUS_KWAVE_LIBS)")
    return cfg


@pytest.mark.slow
def test_full_run_tiny(matlab_engine, oracle, tmp_path):
    cfg = _kwave_ready()
    cfg.oracle = oracle.path
    cfg.workdir = tmp_path / "work"
    with ExportSession(cfg, kwave=True, engine=matlab_engine) as s:
        path = export_spec(s, SPECS["full_run.tiny"], tmp_path)
    r = read_reference(path)
    a = r.attrs
    assert a["kind"] == "full_run" and a["kwave_solver"] == "kspaceFirstOrder3DC"
    assert a["kwave_matlab_version"].startswith("1.4")
    assert len(a["kwave_binary_sha256"]) == 64 and "kspaceFirstOrder" in a["kwave_binary_version"]
    assert a["seeds"]["ground_truth_mt19937ar"] == 1
    rf = r.outputs["RF"]
    s = r.inputs["settings"]
    n_frames = int(s["Acquisition"]["NumberOfFrames"][0, 0])
    n_el = int(s["Transducer"]["NumberOfElements"][0, 0])
    assert rf.shape == (1, n_frames)
    frame = rf[0, 0]
    assert frame.shape == (1, 1)  # one pulse ('Standard')
    assert frame[0, 0].shape[0] == n_el and frame[0, 0].dtype == np.float32
    assert np.all(np.isfinite(frame[0, 0])) and np.abs(frame[0, 0]).max() > 0
    assert len(r.inputs["ground_truth"]["Frames"].ravel()) == n_frames
    assert r.extra["medium"]["sound_speed"].ndim == 3
