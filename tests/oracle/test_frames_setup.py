"""Oracle tests: run setup and run data I/O (M3 wave 1, item B).

``load_microbubbles``, ``check_ground_truth_data``, ``input_handling``,
``sim_setup`` (with a temporary ``path_setup`` stand-in, so PROTEUS never
creates folders in the oracle), MATLAB RF ``Frame_N.mat`` files and
``load_RF_data``, and MATLAB's ``medium.mat`` (v7.3).

Classes (D-13): bubble positions/velocities float (≤ 1e-9 rel); loaded data,
messages, flags and integer settings exact.
"""

from __future__ import annotations

import textwrap
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pyproteus.acoustic import setup as su
from pyproteus.config import default_settings
from pyproteus.io import frames as fr
from pyproteus.reconstruction import load_rf_data
from tests.oracle.helpers import assert_rel, from_matlab

# ------------------------------------------------------------- ground truth

GT_CODE = """
pyp_s__ = RandStream('mt19937ar','Seed',{seed}); N = {n}; NF = {nf}; NP = {npulse};
FlowSimulationParameters = struct('TimeBtwPulse',1e-4,'FrameRate',{rate},'NBPulses',NP, ...
    'NMicrobubbles',N,'NumberOfFrames',NF);
FlowSimulationParameters.Microbubble.Distribution.Probabilities = [0.25 0.5 0.25];
FlowSimulationParameters.Microbubble.Distribution.Radii = [1e-6 2e-6 3e-6];
save(fullfile('{folder}','FlowSimulationParameters.mat'),'FlowSimulationParameters','{ver}');
pad = num2str(length(num2str(NF)));
for m = 1:NF
    Frame = struct();
    for k = 1:NP
        p = ['Pulse' num2str(k)];
        Frame.(p).Points = 0.01*(rand(pyp_s__, N, 3) - 0.5);
        Frame.(p).Velocity = 1e-3*randn(pyp_s__, N, 3);
        Frame.(p).Radius = 1e-6*(1 + 2*rand(pyp_s__, N, 1));
        Frame.(p).StreamNumber = (1:N)';
    end
    save(fullfile('{folder}',['Frame_' num2str(m,['%0' pad 'i']) '.mat']),'Frame','{ver}');
end
[Q, Rq] = qr(randn(pyp_s__, 3));
pyp_G__ = struct();
pyp_G__.Rotation = Q*diag(sign(diag(Rq)));
pyp_G__.BoundingBox.Center = [0.008; 0.007; 0.011];
pyp_G__.Center = [0.03; 0.001; -0.002];
clear pyp_s__ N NF NP pad m k p Frame FlowSimulationParameters Q Rq
"""


def write_gt(
    eng: Any, folder: Path, *, n=7, nf=10, npulse=3, seed=4, ver="-v7", rate=500.0
) -> dict:
    code = GT_CODE.format(
        seed=seed, n=n, nf=nf, npulse=npulse, folder=folder, ver=ver, rate=repr(rate)
    )
    eng.eval(textwrap.dedent(code), nargout=0)
    G = eng.workspace["pyp_G__"]
    return {
        "Rotation": from_matlab(G["Rotation"]),
        "BoundingBox": {"Center": from_matlab(G["BoundingBox"]["Center"])},
        "Center": from_matlab(G["Center"]),
    }


@pytest.mark.parametrize(
    ("n", "nf", "ver"),
    [(7, 10, "-v7"), (7, 10, "-v7.3"), (1, 9, "-v7"), (13, 100, "-v7")],
)
def test_load_microbubbles(matlab_engine: Any, tmp_path: Path, n: int, nf: int, ver: str) -> None:
    eng = matlab_engine
    G = write_gt(eng, tmp_path, n=n, nf=nf, ver=ver)
    for frame in sorted({1, 7 % nf + 1, nf}):
        tree = su.read_frame_tree(tmp_path, frame, nf)
        for k in (1, 2, 3):
            eng.eval(
                f"pyp_MB__ = load_microbubbles('{tmp_path}', {frame}, {k}, pyp_G__, {nf});",
                nargout=0,
            )
            ref = eng.workspace["pyp_MB__"]
            mb = su.load_microbubbles(tmp_path, frame, k, G, nf)
            again = su.load_microbubbles(tmp_path, frame, k, G, nf, frame_tree=tree)
            for key in ("points", "radii", "velocities"):
                expected = from_matlab(ref[key])
                assert mb[key].shape == expected.shape, key
                assert_rel(mb[key], expected, rel=1e-9, err_msg=f"{key} frame {frame} pulse {k}")
                np.testing.assert_array_equal(again[key], mb[key])
            np.testing.assert_array_equal(mb["radii"], from_matlab(ref["radii"]))  # loaded data
    eng.eval("clear pyp_MB__ pyp_G__", nargout=0)


# -------------------------------------------------- check_ground_truth_data

CHECK_CASES = [
    # (name, Acquisition overrides, Microbubble overrides)
    ("match", {}, {}),
    ("frames", {"NumberOfFrames": 11.0}, {}),
    ("pulses", {"NumberOfPulses": 2.0}, {}),
    ("bubbles", {}, {"Number": 8.0}),
    ("all", {"NumberOfFrames": 3.0, "NumberOfPulses": 1.0, "FrameRate": 400.0}, {"Number": 1.0}),
    ("rate within 6 digits", {"FrameRate": 500.0004}, {}),
    ("rate beyond 6 digits", {"FrameRate": 500.0006}, {}),
    ("rate near tie", {"FrameRate": 500.0005}, {}),
    # Exact binary ties at the 6th digit (500.0625, 500.1875): half to even or away?
    ("rate exact tie down", {"FrameRate": 500.0625, "gt_rate": 500.062}, {}),
    ("rate exact tie up", {"FrameRate": 500.1875, "gt_rate": 500.188}, {}),
    ("rate exact tie away", {"FrameRate": 500.0625, "gt_rate": 500.063}, {}),
    ("time within 6 digits", {"TimeBetweenPulses": 1.0000049e-4}, {}),
    ("time beyond 6 digits", {"TimeBetweenPulses": 1.0000051e-4}, {}),
    ("distribution values", {}, {"Radii": [1e-6, 2e-6, 3.00001e-6]}),
    ("distribution tiny", {}, {"Radii": [1e-6, 2e-6, 3.000001e-6]}),
    ("distribution length", {}, {"Radii": [1e-6, 2e-6], "Probabilities": [0.5, 0.5]}),
]


@pytest.mark.parametrize(
    ("case", "acq_over", "mb_over"), CHECK_CASES, ids=[c[0] for c in CHECK_CASES]
)
def test_check_ground_truth_data(
    matlab_engine: Any, tmp_path: Path, case: str, acq_over: dict, mb_over: dict
) -> None:
    import matlab

    eng = matlab_engine
    gt = tmp_path / "gt"
    gt.mkdir()
    write_gt(eng, gt, nf=10, npulse=3)
    acq = {
        "NumberOfFrames": 10.0,
        "FrameRate": 500.0,
        "NumberOfPulses": 3.0,
        "TimeBetweenPulses": 1e-4,
        **acq_over,
    }
    dist = {"Probabilities": [0.25, 0.5, 0.25], "Radii": [1e-6, 2e-6, 3e-6]}
    dist.update({k: v for k, v in mb_over.items() if k in dist})
    mb = {
        "Number": mb_over.get("Number", 7.0),
        "Distribution": {k: np.array(v)[None, :] for k, v in dist.items()},
    }
    eng.workspace["pyp_A__"] = acq
    eng.workspace["pyp_M__"] = {
        "Number": mb["Number"],
        "Distribution": {k: matlab.double(v.tolist()) for k, v in mb["Distribution"].items()},
    }
    save = tmp_path / "save_m"
    eng.eval(
        "lastwarn(''); pyp_err__ = '';"
        f"try, check_ground_truth_data('{gt}', pyp_A__, pyp_M__, '{save}');"
        "catch pyp_e__, pyp_err__ = pyp_e__.message; end;"
        "pyp_warn__ = lastwarn();",
        nargout=0,
    )
    m_err = eng.workspace["pyp_err__"]
    m_warn = eng.workspace["pyp_warn__"]
    eng.eval("clear pyp_A__ pyp_M__ pyp_err__ pyp_warn__ pyp_e__ pyp_G__", nargout=0)

    p_err, p_warn = "", ""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            su.check_ground_truth_data(gt, acq, mb, savedir=tmp_path / "save_p")
        except su.GroundTruthMismatch as exc:
            p_err = str(exc)
    p_warn = "".join(str(w.message) for w in rec if w.category is su.GroundTruthDistributionWarning)
    assert p_err == m_err, case
    assert p_warn == m_warn, case
    # MATLAB saves the parameters when the check passes; pyproteus copies them on request.
    assert (save / "FlowSimulationParameters.mat").is_file() == (not m_err)
    assert (tmp_path / "save_p" / "FlowSimulationParameters.mat").is_file() == (not p_err)
    if not m_err:
        a = fr.read_flow_simulation_parameters(save)
        b = fr.read_flow_simulation_parameters(tmp_path / "save_p")
        assert repr(a.to_tree()) == repr(b.to_tree())


# ------------------------------------------------------------ input_handling

INPUT_CASES = [
    ("{}", {}),
    ("{true}", {"continue_": True}),
    ("{false, 2}", {"continue_": False, "start_frame": 2}),
    ("{[], 3, 50}", {"start_frame": 3, "end_frame": 50}),
    ("{[], [], 4}", {"end_frame": 4}),
    ("{[], 2.0, 10}", {"start_frame": 2.0, "end_frame": 10}),
    ("{1}", {"continue_": 1}),
    ("{[], 0}", {"start_frame": 0}),
    ("{[], 2.5}", {"start_frame": 2.5}),
    ("{[], true}", {"start_frame": True}),
    ("{[], [], -1}", {"end_frame": -1}),
]


@pytest.mark.parametrize(("cell", "kwargs"), INPUT_CASES, ids=[c[0] for c in INPUT_CASES])
def test_input_handling(matlab_engine: Any, cell: str, kwargs: dict) -> None:
    eng = matlab_engine
    s = default_settings()
    eng.workspace["pyp_A__"] = {"NumberOfFrames": 10.0}
    eng.workspace["pyp_S__"] = {"DeviceNumber": 0.0}
    eng.eval(
        "pyp_err__ = false;"
        f"try, [pyp_A__, pyp_S__] = input_handling(pyp_A__, pyp_S__, {cell});"
        "catch, pyp_err__ = true; end",
        nargout=0,
    )
    m_err = bool(eng.workspace["pyp_err__"])
    acq = s.Acquisition
    acq.NumberOfFrames = 10.0
    try:
        a, _ = su.input_handling(acq, s.SimulationParameters, **kwargs)
        p_err = False
    except ValueError:
        p_err = True
    assert p_err == m_err
    if not m_err:
        ref = eng.workspace["pyp_A__"]
        assert bool(ref["Continue"]) is a.Continue
        assert float(ref["StartFrame"]) == a.StartFrame
        assert float(ref["EndFrame"]) == a.EndFrame
    eng.eval("clear pyp_A__ pyp_S__ pyp_err__", nargout=0)


def test_input_handling_start_after_end(matlab_engine: Any) -> None:
    """MATLAB accepts StartFrame > EndFrame (main_RF then runs no frame); pyproteus raises."""
    eng = matlab_engine
    eng.workspace["pyp_A__"] = {"NumberOfFrames": 10.0}
    eng.eval("pyp_A__ = input_handling(pyp_A__, struct(), {[], 5, 3});", nargout=0)
    assert float(eng.workspace["pyp_A__"]["StartFrame"]) == 5.0
    eng.eval("clear pyp_A__", nargout=0)
    s = default_settings()
    with pytest.raises(ValueError, match="after the end frame"):
        su.input_handling(s.Acquisition, s.SimulationParameters, start_frame=5, end_frame=3)


# ------------------------------------------------------------------ sim_setup

PATH_SETUP_SHIM = """function PATHS = path_setup()
root = '{root}';
names = {{'SettingsPath','GroundTruthPath','ResultsPath','DataPath','BinaryPath', ...
    'VoxelisationPath','kWavePath','GeometriesPath','MicrobubblePath'}};
for n = 1:numel(names)
    PATHS.(names{{n}}) = fullfile(root, names{{n}});
end
end
"""


@pytest.mark.parametrize("solver", ["3D", "3DC", "3DG"])
def test_sim_setup(matlab_engine: Any, tmp_path: Path, solver: str) -> None:
    eng = matlab_engine
    shim = tmp_path / "shim"
    shim.mkdir()
    (shim / "path_setup.m").write_text(PATH_SETUP_SHIM.format(root=tmp_path))
    for name in ("VoxelisationPath", "kWavePath", "BinaryPath", "DataPath"):
        (tmp_path / name).mkdir()
    (tmp_path / "BinaryPath" / "kspaceFirstOrder-OMP").write_text("")
    (tmp_path / "BinaryPath" / "kspaceFirstOrder-CUDA").write_text("")
    sp = {"Solver": solver, "NumberOfInteractions": 2.0, "DeviceNumber": 1.0}
    saved = eng.path(nargout=1)
    try:
        eng.addpath(str(shim), nargout=0)
        eng.eval("clear path_setup", nargout=0)
        eng.workspace["pyp_SP__"] = sp
        ref = eng.eval("sim_setup(pyp_SP__)", nargout=1)
    finally:
        eng.path(saved, nargout=0)
        eng.eval("clear pyp_SP__ path_setup", nargout=0)
    cfg = su.make_run_config(
        sp, binary_path=tmp_path / "BinaryPath", data_path=tmp_path / "DataPath"
    )
    rp = cfg.run_param()
    for key in ("solver", "DATA_CAST", "DATA_CAST_RF", "MicrobubblesUseParfor"):
        assert rp[key] == ref[key], key
    assert rp["record_movie"] is bool(ref["record_movie"])
    assert rp["N_interactions"] == float(ref["N_interactions"])
    assert rp["MicrobubblesBatchSize"] == float(ref["MicrobubblesBatchSize"])
    assert ("DEVICE_NUM" in rp) == ("DEVICE_NUM" in ref)
    if "DEVICE_NUM" in ref:
        assert rp["DEVICE_NUM"] == float(ref["DEVICE_NUM"])
    assert ("BINARY_PATH" in rp) == ("BINARY_PATH" in ref)
    if "BINARY_PATH" in ref:
        # MATLAB keeps the folder; pyproteus resolves the binary inside it.
        assert Path(rp["BINARY_PATH"]).parent == Path(ref["BINARY_PATH"])
        assert Path(rp["DATA_PATH"]) == Path(ref["DATA_PATH"])


# -------------------------------------------------------- MATLAB RF frames


RF_CODE = """
pyp_s__ = RandStream('mt19937ar','Seed',{seed});
RF = cell(1,3); Frame = cell(1,3);
for k = 1:3
    RF{{k}} = single(randn(pyp_s__, 5, 9));
    N = 2 + k;
    MB = struct();
    MB.points = rand(pyp_s__, N, 3); MB.nodes = rand(pyp_s__, N, 3);
    MB.idx = (1:N)'; MB.radii = rand(pyp_s__, N, 1); MB.velocities = rand(pyp_s__, N, 3);
    Frame{{k}} = MB;
end
dt = 1/{seed}e7;
save(fullfile('{folder}','{name}'), 'RF', 'dt', 'Frame', '{ver}');
clear pyp_s__ RF Frame MB N k dt
"""


@pytest.mark.parametrize("ver", ["-v7", "-v7.3"])
def test_matlab_rf_frames(matlab_engine: Any, tmp_path: Path, ver: str) -> None:
    eng = matlab_engine
    folder = tmp_path / "res"
    folder.mkdir()
    for frame in (12, 3, 1):
        name = fr.frame_file_name(frame, 12)
        eng.eval(
            textwrap.dedent(RF_CODE.format(seed=frame, folder=folder, name=name, ver=ver)),
            nargout=0,
        )
    s = default_settings()
    w = fr.RFWriter(tmp_path / "rf.h5", settings=s, sequence=("odd", "even", "all"), num_frames=12)
    for frame in (1, 3, 12):
        file = folder / fr.frame_file_name(frame, 12)
        got = fr.read_matlab_rf_frame(file)
        ref = eng.eval(f"load('{file}')", nargout=1)
        assert got.frame == frame and got.dt == float(ref["dt"])
        for k in range(3):
            expected = from_matlab(ref["RF"][k])
            assert got.rf[k].dtype == np.float32
            np.testing.assert_array_equal(got.rf[k], expected)
            for key in ("points", "nodes", "idx", "radii", "velocities"):
                np.testing.assert_array_equal(
                    got.bubbles[k][key], from_matlab(ref["Frame"][k][key])
                )
        w.write_frame(frame, got.rf, got.dt, got.bubbles)
    for scheme in su.PULSE_SEQUENCES:
        expected = from_matlab(eng.eval(f"load_RF_data('{folder}', '{scheme}')", nargout=1))
        for source in (folder, tmp_path / "rf.h5"):
            got = load_rf_data(source, scheme)
            assert got.dtype == np.float32
            np.testing.assert_array_equal(got, expected, err_msg=f"{scheme} {source}")


# ------------------------------------------------------------- medium.mat


def test_matlab_medium_cache(matlab_engine: Any, tmp_path: Path) -> None:
    eng = matlab_engine
    path = tmp_path / "medium.mat"
    eng.eval(
        "pyp_s__ = RandStream('mt19937ar','Seed',2);"
        "medium = struct(); medium.sound_speed = 1540 + rand(pyp_s__, 4, 5, 3);"
        "medium.density = 1000 + rand(pyp_s__, 4, 5, 3); medium.BonA = 6*ones(4,5,3);"
        "medium.alpha_coeff = rand(pyp_s__, 4, 5, 3); medium.alpha_power = 1.05;"
        "medium.alpha_mode = 'no_dispersion';"
        "vessel_grid = rand(pyp_s__, 4, 5, 3) > 0.5;"
        "Grid = struct('Nx', 4, 'Ny', 5, 'Nz', 3, 'dx', 1e-4, 'x', linspace(0, 1e-3, 4));"
        f"save('{path}', 'medium', 'vessel_grid', 'Grid', '-v7.3');"
        "clear pyp_s__ medium vessel_grid Grid",
        nargout=0,
    )
    got = fr.load_medium_cache(path)
    ref = eng.eval(f"load('{path}')", nargout=1)
    for key in ("sound_speed", "density", "BonA", "alpha_coeff"):
        np.testing.assert_array_equal(got.medium[key], from_matlab(ref["medium"][key]))
    assert got.medium["alpha_power"] == 1.05
    assert got.medium["alpha_mode"] == "no_dispersion"
    np.testing.assert_array_equal(got.vessel_grid, from_matlab(ref["vessel_grid"]))
    assert got.vessel_grid.dtype == np.bool_
    np.testing.assert_array_equal(got.grid["x"], from_matlab(ref["Grid"]["x"]))
