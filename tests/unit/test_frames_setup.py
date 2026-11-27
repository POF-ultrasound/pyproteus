"""Unit tests for ``pyproteus.io.frames`` and ``pyproteus.acoustic.setup`` (no MATLAB)."""

from __future__ import annotations

import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest
import scipy.io

from pyproteus.acoustic import setup as su
from pyproteus.acoustic.grid import PML, Grid
from pyproteus.config import default_settings, save_settings_json
from pyproteus.io import frames as fr
from pyproteus.reconstruction import das_reconstruction, iter_rf_frames, load_rf_data

# ------------------------------------------------------------------ fixtures


@pytest.fixture(scope="module")
def settings():
    s = default_settings()
    s.Geometry.Center = np.array([[0.03], [0.001], [-0.002]])
    s.Acquisition.NumberOfFrames = 12.0
    s.Acquisition.NumberOfPulses = 2.0
    s.Microbubble.Number = 7.0
    return s


def rotation(seed: int = 0) -> np.ndarray:
    q, r = np.linalg.qr(np.random.default_rng(seed).standard_normal((3, 3)))
    return q * np.sign(np.diag(r))


def write_ground_truth(folder: Path, s, seed: int = 1, n_frames=None, fsp_overrides=None):
    """Ground truth in the format of ``generate_streamlines.m`` (scipy v5 writer)."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    acq, mb = s.Acquisition, s.Microbubble
    nf = int(acq.NumberOfFrames if n_frames is None else n_frames)
    fsp = {
        "TimeBtwPulse": acq.TimeBetweenPulses,
        "FrameRate": acq.FrameRate,
        "NBPulses": acq.NumberOfPulses,
        "NMicrobubbles": mb.Number,
        "NumberOfFrames": float(nf),
        "Microbubble": {
            "Distribution": {
                "Probabilities": mb.Distribution.Probabilities,
                "Radii": mb.Distribution.Radii,
            }
        },
    }
    fsp.update(fsp_overrides or {})
    scipy.io.savemat(folder / "FlowSimulationParameters.mat", {"FlowSimulationParameters": fsp})
    n = int(mb.Number)
    frames = {}
    for m in range(1, nf + 1):
        frame = {}
        for k in range(1, int(acq.NumberOfPulses) + 1):
            frame[f"Pulse{k}"] = {
                "Points": rng.uniform(-0.01, 0.01, (n, 3)),
                "Velocity": 1e-3 * rng.standard_normal((n, 3)),
                "Radius": 1e-6 * (1 + 2 * rng.random((n, 1))),
                "StreamNumber": np.arange(1.0, n + 1).reshape(-1, 1),
            }
        frames[m] = frame
        scipy.io.savemat(folder / fr.frame_file_name(m, nf), {"Frame": frame})
    return frames


# ------------------------------------------------------------ ground truth


def test_ground_truth_read_and_transform(tmp_path, settings) -> None:
    frames = write_ground_truth(tmp_path, settings)
    assert (tmp_path / "Frame_07.mat").is_file()
    fsp = fr.read_flow_simulation_parameters(tmp_path)
    assert fsp.NumberOfFrames == 12 and fsp.NBPulses == 2 and fsp.NMicrobubbles == 7
    p, _r = fsp.distribution()
    np.testing.assert_array_equal(p, settings.Microbubble.Distribution.Probabilities)
    pulse = fr.read_ground_truth_pulse(tmp_path, 7, 2, 12)
    np.testing.assert_array_equal(pulse["Points"], frames[7]["Pulse2"]["Points"])
    assert pulse["StreamNumber"].shape == (7, 1)

    G = settings.Geometry.copy_struct()
    G.Rotation = rotation()
    mb = su.load_microbubbles(tmp_path, 7, 2, G, 12)
    src = frames[7]["Pulse2"]
    R, c0, c1 = G.Rotation, G.BoundingBox.Center, G.Center
    expected = (R @ (src["Points"].T - c0) + c1).T
    np.testing.assert_allclose(mb["points"], expected, rtol=0, atol=1e-15)
    np.testing.assert_allclose(mb["velocities"], src["Velocity"] @ R.T, rtol=1e-15)
    np.testing.assert_array_equal(mb["radii"], src["Radius"])
    assert mb["radii"].shape == (7, 1)
    # Reading the frame once gives the same result.
    tree = su.read_frame_tree(tmp_path, 7, 12)
    again = su.load_microbubbles(tmp_path, 7, 2, G, 12, frame_tree=tree)
    np.testing.assert_array_equal(again["points"], mb["points"])
    with pytest.raises(KeyError, match="Pulse3"):
        su.load_microbubbles(tmp_path, 7, 3, G, 12)
    with pytest.raises(FileNotFoundError):
        su.load_microbubbles(tmp_path, 13, 1, G, 12)


def test_ground_truth_validation(tmp_path, settings) -> None:
    frame = {"Pulse1": {"Points": np.zeros((3, 2)), "Velocity": np.zeros((3, 3)), "Radius": [1]}}
    with pytest.raises(ValueError, match="N×3"):
        fr.read_ground_truth_pulse(tmp_path, 1, 1, 1, frame_tree=frame)
    frame["Pulse1"]["Points"] = np.zeros((3, 3))
    with pytest.raises(ValueError, match="Radius"):
        fr.read_ground_truth_pulse(tmp_path, 1, 1, 1, frame_tree=frame)
    with pytest.raises(KeyError, match="NumberOfFrames"):
        fr.FlowSimulationParameters.from_tree({"TimeBtwPulse": 1})


def test_check_ground_truth_data(tmp_path, settings) -> None:
    write_ground_truth(tmp_path / "gt", settings, n_frames=1)
    acq = settings.Acquisition.copy_struct()
    acq.NumberOfFrames = 1.0
    mb = settings.Microbubble
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fsp = su.check_ground_truth_data(tmp_path / "gt", acq, mb)
    assert fsp.NMicrobubbles == 7
    assert not (tmp_path / "out").exists()
    su.check_ground_truth_data(tmp_path / "gt", acq, mb, savedir=tmp_path / "out")
    assert (tmp_path / "out" / "FlowSimulationParameters.mat").read_bytes() == (
        tmp_path / "gt" / "FlowSimulationParameters.mat"
    ).read_bytes()

    # 6 significant digits: 500.0004 equals 500, 500.004 does not.
    a2 = acq.copy_struct()
    a2.FrameRate = acq.FrameRate * (1 + 8e-7)
    su.check_ground_truth_data(tmp_path / "gt", a2, mb)
    a2.FrameRate = acq.FrameRate * (1 + 8e-6)
    with pytest.raises(su.GroundTruthMismatch, match="frame rate"):
        su.check_ground_truth_data(tmp_path / "gt", a2, mb)
    # The message names the last failing check (MATLAB); all are listed.
    a3 = acq.copy_struct()
    a3.NumberOfPulses = 3.0
    a3.NumberOfFrames = 2.0
    with pytest.raises(su.GroundTruthMismatch, match="number of frames") as err:
        su.check_ground_truth_data(tmp_path / "gt", a3, mb)
    assert err.value.parameters == ["number of pulses", "number of frames"]
    # Time between pulses only matters with more than one pulse.
    write_ground_truth(tmp_path / "gt1", settings, n_frames=1, fsp_overrides={"NBPulses": 1.0})
    a4 = acq.copy_struct()
    a4.NumberOfPulses = 1.0
    a4.TimeBetweenPulses = 1.0
    su.check_ground_truth_data(tmp_path / "gt1", a4, mb)
    a4.NumberOfPulses = 2.0
    with pytest.raises(su.GroundTruthMismatch, match="number of pulses"):
        su.check_ground_truth_data(tmp_path / "gt1", a4, mb)
    # Size distribution: warning only.
    mb2 = mb.copy_struct()
    mb2.Distribution.Radii = mb.Distribution.Radii * (1 + 1e-5)
    with pytest.warns(su.GroundTruthDistributionWarning):
        su.check_ground_truth_data(tmp_path / "gt", acq, mb2)
    mb2.Distribution.Radii = mb.Distribution.Radii[:, :-1]
    with pytest.warns(su.GroundTruthDistributionWarning):
        su.check_ground_truth_data(tmp_path / "gt", acq, mb2)


# --------------------------------------------------------------- run setup


def test_sim_setup_solvers(tmp_path) -> None:
    binary = tmp_path / "kspaceFirstOrder-OMP"
    binary.write_text("")
    sp = {"Solver": "3DC", "NumberOfInteractions": 2.0, "DeviceNumber": 1.0}
    rp = su.sim_setup(sp, binary_path=binary, data_path=tmp_path)
    assert rp["solver"] == "kspaceFirstOrder3DC" and rp["DATA_CAST"] == "single"
    assert rp["DATA_CAST_RF"] == "single" and rp["record_movie"] is False
    assert rp["BINARY_PATH"] == str(binary) and rp["DATA_PATH"] == str(tmp_path)
    assert "DEVICE_NUM" not in rp and rp["N_interactions"] == 2
    assert rp["MicrobubblesBatchSize"] == 15 and rp["MicrobubblesUseParfor"] == "auto"
    assert (rp["backend"], rp["device"]) == ("cpp", "cpu")
    # A folder holding the binary works too (MATLAB BinaryPath).
    assert su.sim_setup(sp, binary_path=tmp_path)["BINARY_PATH"] == str(binary)

    cfg = su.make_run_config({**sp, "Solver": "3DG"}, resolve_binary=False)
    assert cfg.data_cast == "gpuArray-single" and cfg.device_number == 1 and cfg.device == "gpu"
    assert cfg.run_param()["DEVICE_NUM"] == 1
    cfg = su.make_run_config({**sp, "Solver": "3D"})
    assert cfg.backend == "python" and cfg.record_movie and cfg.binary_path is None
    assert "BINARY_PATH" not in cfg.run_param()
    with pytest.raises(ValueError, match="Solver"):
        su.sim_setup({**sp, "Solver": "3DX"})
    with pytest.raises(FileNotFoundError):
        su.sim_setup(sp, binary_path=tmp_path / "missing")
    with pytest.raises(ValueError, match="integer"):
        su.sim_setup({**sp, "NumberOfInteractions": 1.5}, binary_path=binary)


def test_input_handling(settings) -> None:
    acq, sp = settings.Acquisition, settings.SimulationParameters
    a, s = su.input_handling(acq, sp)
    assert (a.Continue, a.StartFrame, a.EndFrame) == (False, 1.0, 12.0)
    assert not acq.isfield("Continue")  # input not modified
    a, s = su.input_handling(acq, sp, continue_=True, start_frame=3, end_frame=40.0)
    assert (a.Continue, a.StartFrame, a.EndFrame) == (True, 3.0, 12.0)
    a, s = su.input_handling(acq, sp, device_number=2)
    assert s.DeviceNumber == 2.0 and sp.DeviceNumber == 0.0
    for kwargs, msg in [
        ({"continue_": 1}, "boolean"),
        ({"start_frame": 0}, "positive integer"),
        ({"start_frame": 2.5}, "positive integer"),
        ({"start_frame": True}, "positive integer"),
        ({"end_frame": -1}, "positive integer"),
        ({"device_number": 1.5}, "integer"),
        ({"device_number": -1}, "from zero"),
        ({"start_frame": 5, "end_frame": 4}, "after the end frame"),
    ]:
        with pytest.raises(ValueError, match=msg):
            su.input_handling(acq, sp, **kwargs)


def test_sim_startup(tmp_path) -> None:
    sdir = tmp_path / "settings"
    sdir.mkdir()
    (sdir / "case.mat").write_bytes(b"")
    p = su.sim_startup("case", "gt", "out", settings_dir=sdir, ground_truth_dir=tmp_path / "G")
    assert p.settings_file == sdir / "case.mat"
    assert p.ground_truth == tmp_path / "G" / "gt" and p.results == Path("out")
    assert p.rf_file == Path("out/rf.h5") and p.medium_cache == Path("out/medium.h5")
    other = tmp_path / "other.mat"
    other.write_bytes(b"")
    assert su.sim_startup(other, "gt", "out", settings_dir=sdir).settings_file == other
    with pytest.raises(FileNotFoundError, match="not found"):
        su.sim_startup("nothing", "gt", "out", settings_dir=sdir)
    assert not (tmp_path / "G").exists()


def test_setup_run(tmp_path, settings) -> None:
    s = settings.model_copy(deep=True)
    s.Acquisition.NumberOfFrames = 3.0
    write_ground_truth(tmp_path / "gt", s)
    save_settings_json(s, tmp_path / "case.json")
    run = su.setup_run(
        tmp_path / "case.json",
        tmp_path / "gt",
        tmp_path / "res",
        start_frame=2,
        resolve_binary=False,
    )
    assert run.sequence == ("pulse",) and run.frames() == [2, 3]
    assert run.rf_writer_mode == "resume"
    assert run.settings.Transducer.OnGrid is False and run.settings.Medium.Save is True
    assert run.settings.Microbubble.BatchSize == 15 and run.flow_simulation_parameters
    assert su.frames_to_simulate(run, completed=[2]) == [2, 3]  # no Continue: recompute
    run2 = su.setup_run(s, tmp_path / "gt", tmp_path / "res", continue_=True, resolve_binary=False)
    assert su.frames_to_simulate(run2, completed=[1, 3]) == [2]
    s.Acquisition.NumberOfFrames = 4.0
    with pytest.raises(su.GroundTruthMismatch):
        su.setup_run(s, tmp_path / "gt", tmp_path / "res", resolve_binary=False)
    assert not (tmp_path / "res").exists()
    assert su.pulse_sequence("Amplitude modulation") == ("odd", "even", "all")
    with pytest.raises(ValueError, match="PulsingScheme"):
        su.pulse_sequence("Harmonic")


# ------------------------------------------------------------------ RF files


def rf_pulses(n_pulses, shape=(5, 9), dtype=np.float32, seed=0):
    rng = np.random.default_rng(seed)
    return [rng.standard_normal(shape).astype(dtype) for _ in range(n_pulses)]


def bubbles(n, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "points": rng.random((n, 3)),
        "radii": rng.random((n, 1)),
        "velocities": rng.random((n, 3)),
        "idx": np.arange(1, n + 1, dtype=np.float64).reshape(-1, 1),
    }


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_rf_writer_round_trip(tmp_path, settings, dtype) -> None:
    s = settings.model_copy(deep=True)
    s.Acquisition.PulsingScheme = "Amplitude modulation"
    seq = su.pulse_sequence("Amplitude modulation")
    fsp = fr.FlowSimulationParameters(1e-4, 500.0, 2.0, 7.0, 12.0, {"Distribution": {}})
    w = fr.RFWriter(
        tmp_path / "rf.h5", settings=s, sequence=seq, num_frames=12, flow_simulation_parameters=fsp
    )
    written = {}
    for m in (3, 1, 12):
        rf = rf_pulses(3, dtype=dtype, seed=m)
        mbs = [bubbles(4 + k, seed=m + k) for k in range(3)]
        w.write_frame(m, rf, 1 / 40e6, mbs)
        written[m] = (rf, mbs)
    with pytest.warns(RuntimeWarning, match="label"):
        w.write_frame(2, rf_pulses(3, dtype=dtype), 1e-8, [{**bubbles(2), "label": "x"}] * 3)
    assert set(fr.RFFile(tmp_path / "rf.h5").read_frame(2).bubbles[0]) == {
        "points",
        "radii",
        "velocities",
        "idx",
    }
    with h5py.File(tmp_path / "rf.h5", "a") as h:
        del h["frames/Frame_02"]
    assert w.completed_frames() == [1, 3, 12]
    assert w.remaining_frames(1, 4) == [2, 4]

    f = fr.RFFile(tmp_path / "rf.h5")
    assert f.frames == [1, 3, 12] and len(f) == 3 and f.sequence == seq
    assert f.pulsing_scheme == "Amplitude modulation"
    from pyproteus.config import settings_to_json

    assert settings_to_json(f.settings) == settings_to_json(s)
    assert f.flow_simulation_parameters.NMicrobubbles == 7
    for frame in f:
        rf, mbs = written[frame.frame]
        assert frame.dt == 1 / 40e6
        for a, b in zip(frame.rf, rf, strict=True):
            assert a.dtype == dtype
            np.testing.assert_array_equal(a, b)
        for got, mb in zip(frame.bubbles, mbs, strict=True):
            assert set(got) == {"points", "radii", "velocities", "idx"}
            for key in got:
                np.testing.assert_array_equal(got[key], mb[key])
                assert got[key].shape == np.asarray(mb[key]).shape
    with h5py.File(tmp_path / "rf.h5") as h:
        assert sorted(h["frames"]) == ["Frame_01", "Frame_03", "Frame_12"]
        assert h["frames/Frame_03/rf"].shape == (3, 5, 9)
    # Pulsing scheme from the file:
    rf = load_rf_data(tmp_path / "rf.h5")
    r3 = written[3][0]
    np.testing.assert_array_equal(rf[:, :, 1], r3[2] - r3[0] - r3[1])
    assert rf.shape == (5, 9, 3) and rf.dtype == dtype


def test_rf_writer_errors_and_resume(tmp_path, settings) -> None:
    path = tmp_path / "rf.h5"
    w = fr.RFWriter(path, settings=settings, sequence=("pulse",), num_frames=4)
    w.write_frame(1, rf_pulses(1), 1e-8)
    with pytest.raises(FileExistsError):
        fr.RFWriter(path, settings=settings, sequence=("pulse",), num_frames=4)
    with pytest.raises(FileExistsError, match="already"):
        w.write_frame(1, rf_pulses(1), 1e-8)
    w.write_frame(1, rf_pulses(1, seed=5), 2e-8, replace=True)
    assert fr.RFFile(path).read_frame(1).dt == 2e-8
    for args, msg in [
        ((5, rf_pulses(1), 1e-8), "outside"),
        ((2, rf_pulses(2), 1e-8), "2 RF arrays"),
        ((2, [np.zeros(4, np.float32)], 1e-8), "Nelem-by-Nt"),
    ]:
        with pytest.raises(ValueError, match=msg):
            w.write_frame(*args)
    with pytest.raises(ValueError, match="same class"):
        fr.RFWriter(
            tmp_path / "b.h5", settings=settings, sequence=("a", "b"), num_frames=1
        ).write_frame(1, [np.zeros((2, 2)), np.zeros((2, 2), np.float32)], 1.0)

    # Simulate an interrupted frame: a partial group is left behind.
    with h5py.File(path, "a") as h:
        h["frames"].create_group(".partial_Frame_2").attrs["frame"] = 2
    assert fr.RFFile(path).frames == [1]
    w2 = fr.RFWriter(path, settings=settings, sequence=("pulse",), num_frames=4, mode="resume")
    with h5py.File(path) as h:
        assert list(h["frames"]) == ["Frame_1"]
    assert w2.remaining_frames() == [2, 3, 4]
    # Resume refuses other settings / sequences / frame counts.
    other = settings.model_copy(deep=True)
    other.Medium.SpeedOfSound = 1500.0
    for kw, msg in [
        ({"settings": other}, "settings"),
        ({"sequence": ("plus", "minus")}, "pulse sequence"),
        ({"num_frames": 5}, "number of frames"),
    ]:
        args = {"settings": settings, "sequence": ("pulse",), "num_frames": 4, **kw}
        with pytest.raises(ValueError, match=msg):
            fr.RFWriter(path, mode="resume", **args)
    # Resume on a missing file creates it; overwrite replaces.
    fr.RFWriter(
        tmp_path / "new.h5", settings=settings, sequence=("p",), num_frames=1, mode="resume"
    )
    fr.RFWriter(path, settings=settings, sequence=("pulse",), num_frames=4, mode="overwrite")
    assert fr.RFFile(path).frames == []
    with pytest.raises(ValueError, match="mode"):
        fr.RFWriter(path, settings=settings, sequence=("p",), num_frames=1, mode="append")
    (tmp_path / "x.h5").write_bytes(b"")
    with h5py.File(tmp_path / "x.h5", "w") as h:
        h.attrs["format"] = "other"
    with pytest.raises(ValueError, match="not a pyproteus-rf"):
        fr.RFFile(tmp_path / "x.h5")


def save_matlab_rf_frame(path: Path, rf, dt, mbs=None) -> None:
    cell = np.empty((1, len(rf)), dtype=object)
    for k, a in enumerate(rf):
        cell[0, k] = a
    out = {"RF": cell, "dt": dt}
    if mbs is not None:
        fcell = np.empty((1, len(mbs)), dtype=object)
        for k, mb in enumerate(mbs):
            fcell[0, k] = dict(mb)
        out["Frame"] = fcell
    scipy.io.savemat(path, out)


def test_matlab_rf_frames_and_hdf5_agree(tmp_path, settings) -> None:
    """The same run as MATLAB Frame_N.mat files and as one HDF5 file reads the same."""
    s = settings.model_copy(deep=True)
    s.Acquisition.PulsingScheme = "Pulse inversion"
    folder = tmp_path / "matlab"
    folder.mkdir()
    w = fr.RFWriter(tmp_path / "rf.h5", settings=s, sequence=("plus", "minus"), num_frames=12)
    for m in (2, 10, 1):
        rf = rf_pulses(2, seed=m)
        mbs = [bubbles(3, seed=m), bubbles(5, seed=m + 1)]
        save_matlab_rf_frame(folder / fr.frame_file_name(m, 12), rf, 1e-8, mbs)
        w.write_frame(m, rf, 1e-8, mbs)
    one = fr.read_matlab_rf_frame(folder / "Frame_10.mat")
    assert one.frame == 10 and one.dt == 1e-8 and len(one.rf) == 2
    assert one.bubbles[1]["points"].shape == (5, 3)
    a = list(fr.iter_rf_source(folder, bubbles=True))
    b = list(fr.iter_rf_source(tmp_path / "rf.h5", bubbles=True))
    assert [x.frame for x in a] == [x.frame for x in b] == [1, 2, 10]
    for x, y in zip(a, b, strict=True):
        for p, q in zip(x.rf, y.rf, strict=True):
            np.testing.assert_array_equal(p, q)
        for p, q in zip(x.bubbles, y.bubbles, strict=True):
            for key in q:
                np.testing.assert_array_equal(p[key], q[key])
    assert all(x.bubbles is None for x in fr.iter_rf_source(folder))
    np.testing.assert_array_equal(
        load_rf_data(folder, "Pulse inversion"), load_rf_data(tmp_path / "rf.h5")
    )
    names = [str(p) for p, _ in iter_rf_frames(tmp_path / "rf.h5")]
    assert Path(names[0]).parts[-3:] == ("rf.h5", "frames", "Frame_01")
    with pytest.raises(ValueError, match="pulsing_scheme"):
        load_rf_data(folder)
    with pytest.raises(FileNotFoundError):
        list(fr.iter_rf_source(tmp_path / "nothing"))


def test_das_reconstruction_reads_hdf5(tmp_path) -> None:
    """DAS on an RF HDF5 file (settings from the file) equals DAS on the MATLAB folder."""
    from pyproteus.config.readers import load_settings_mat
    from tests.unit.test_reconstruction import write_case

    data, settings_path, _ = write_case(tmp_path)
    S = load_settings_mat(settings_path)
    w = fr.RFWriter(tmp_path / "rf.h5", settings=S, sequence=("pulse",), num_frames=2)
    for frame in fr.iter_rf_source(data):
        w.write_frame(frame.frame, frame.rf, frame.dt)
    img_mat, x, _y = das_reconstruction(data, settings_path, cache=False)
    img_h5, x2, _y2 = das_reconstruction(tmp_path / "rf.h5", cache=False)
    np.testing.assert_array_equal(img_mat, img_h5)
    np.testing.assert_array_equal(x, x2)
    with pytest.raises(ValueError, match="filename_parameters"):
        das_reconstruction(data)


# -------------------------------------------------------------- medium cache


def tiny_grid() -> Grid:
    return Grid(
        x=np.linspace(0, 1e-3, 4),
        y=np.linspace(-1e-3, 1e-3, 5),
        z=np.linspace(-5e-4, 5e-4, 3),
        Nx=4,
        Ny=5,
        Nz=3,
        dx=1e-4,
        dy=1e-4,
        dz=1e-4,
        dt=1e-8,
        full_size=(24, 25, 23),
        PML=PML(10, 10, 10, 2.0),
        sensor_on_grid=1.0,
    )


def test_medium_cache_round_trip(tmp_path, settings) -> None:
    rng = np.random.default_rng(0)
    medium = {
        "sound_speed": 1540 + rng.random((4, 5, 3)),
        "density": (1000 + rng.random((4, 5, 3))).astype(np.float32),
        "BonA": np.full((4, 5, 3), 6.0),
        "alpha_coeff": rng.random((4, 5, 3)),
        "alpha_power": 1.05,
        "alpha_mode": "no_dispersion",
    }
    vessel = rng.random((4, 5, 3)) > 0.5
    key = fr.medium_fingerprint(settings, seed=0)
    path = tmp_path / "res" / "medium.h5"
    fr.save_medium_cache(path, medium, vessel, tiny_grid(), fingerprint=key)
    assert not path.with_name("medium.h5.tmp").exists()
    got = fr.load_medium_cache(path, fingerprint=key)
    assert got.fingerprint == key
    for k, v in medium.items():
        if isinstance(v, np.ndarray):
            np.testing.assert_array_equal(got.medium[k], v)
            assert got.medium[k].dtype == v.dtype
        else:
            assert got.medium[k] == v
    np.testing.assert_array_equal(got.vessel_grid, vessel)
    g, ref = got.grid, tiny_grid()
    for name in ("x", "y", "z"):
        np.testing.assert_array_equal(getattr(g, name), getattr(ref, name))
    assert (g.shape, g.dt, g.full_size, g.PML, g.sensor_on_grid) == (
        ref.shape,
        ref.dt,
        ref.full_size,
        ref.PML,
        1.0,
    )
    other = settings.model_copy(deep=True)
    other.Medium.Inhomogeneity = 0.123
    assert fr.medium_fingerprint(other, seed=0) != key
    assert fr.medium_fingerprint(settings, seed=1) != key
    assert fr.medium_fingerprint(settings.model_copy(deep=True), seed=0) == key
    with pytest.raises(fr.MediumCacheMismatch):
        fr.load_medium_cache(path, fingerprint=fr.medium_fingerprint(other, seed=0))
    # No vessel; boolean sensor_on_grid.
    import dataclasses

    fr.save_medium_cache(
        path, medium, None, dataclasses.replace(tiny_grid(), sensor_on_grid=False), fingerprint="k"
    )
    got = fr.load_medium_cache(path)
    assert got.vessel_grid is None and got.grid.sensor_on_grid is False


def test_medium_cache_reads_matlab_v73(tmp_path) -> None:
    """MATLAB's medium.mat is v7.3 (HDF5): dimensions reversed, MATLAB_class attributes."""
    path = tmp_path / "medium.mat"
    arr = np.arange(24, dtype=float).reshape(2, 3, 4)
    with h5py.File(path, "w", userblock_size=512) as h:
        m = h.create_group("medium")
        m.attrs["MATLAB_class"] = np.bytes_("struct")
        d = m.create_dataset("sound_speed", data=arr.transpose())
        d.attrs["MATLAB_class"] = np.bytes_("double")
        d = m.create_dataset("alpha_power", data=np.array([[1.5]]))
        d.attrs["MATLAB_class"] = np.bytes_("double")
        d = h.create_dataset("vessel_grid", data=(arr > 10).astype(np.uint8).transpose())
        d.attrs["MATLAB_class"] = np.bytes_("logical")
    with open(path, "r+b") as fh:
        fh.write(b"MATLAB 7.3 MAT-file".ljust(116) + b"\x00" * 8 + b"\x00\x02IM")
    got = fr.load_medium_cache(path)
    np.testing.assert_array_equal(got.medium["sound_speed"], arr)
    assert got.medium["alpha_power"] == 1.5 and got.fingerprint is None
    np.testing.assert_array_equal(got.vessel_grid, arr > 10)


def test_resume_through_setup_run(tmp_path, settings) -> None:
    """A run stopped after frame 1 resumes with Continue/StartFrame/another GPU (D-25)."""
    s = settings.model_copy(deep=True)
    s.Acquisition.NumberOfFrames = 3.0
    write_ground_truth(tmp_path / "gt", s)
    common = {"resolve_binary": False}
    run = su.setup_run(s, tmp_path / "gt", tmp_path / "res", **common)
    assert run.rf_writer_mode == "resume" and run.replace_frames
    w = fr.RFWriter(
        run.paths.rf_file, settings=run.settings, sequence=run.sequence, num_frames=3,
        mode=run.rf_writer_mode, flow_simulation_parameters=run.flow_simulation_parameters,
    )  # fmt: skip
    w.write_frame(1, rf_pulses(1), 1e-8, replace=run.replace_frames)
    w.write_frame(3, rf_pulses(1), 1e-8, replace=run.replace_frames)
    key = fr.medium_fingerprint(run.settings, seed=0)

    run2 = su.setup_run(
        s, tmp_path / "gt", tmp_path / "res", continue_=True, start_frame=2, device_number=1,
        **common,
    )  # fmt: skip
    run2.settings.Medium.Save = False  # main_RF does this on Continue
    w2 = fr.RFWriter(
        run2.paths.rf_file, settings=run2.settings, sequence=run2.sequence, num_frames=3,
        mode=run2.rf_writer_mode,
    )  # fmt: skip
    assert su.frames_to_simulate(run2, w2.completed_frames()) == [2]
    assert not run2.replace_frames
    # Medium cache key ignores run control and the solver.
    assert fr.medium_fingerprint(run2.settings, seed=0) == key
    s3 = run2.settings.model_copy(deep=True)
    s3.SimulationParameters.Solver = "3DG"
    s3.SimulationParameters.NumberOfInteractions = 3.0
    assert fr.medium_fingerprint(s3, seed=0) == key
    s3.SimulationParameters.CFL = 0.2
    assert fr.medium_fingerprint(s3, seed=0) != key
    # Re-running frames 1..2 without Continue keeps frame 3 and replaces 1.
    run3 = su.setup_run(s, tmp_path / "gt", tmp_path / "res", end_frame=2, **common)
    w3 = fr.RFWriter(
        run3.paths.rf_file, settings=run3.settings, sequence=run3.sequence, num_frames=3,
        mode=run3.rf_writer_mode,
    )  # fmt: skip
    assert w3.completed_frames() == [1, 3] and su.frames_to_simulate(run3, [1, 3]) == [1, 2]
    w3.write_frame(1, rf_pulses(1, seed=9), 2e-8, replace=run3.replace_frames)
    assert fr.RFFile(run3.paths.rf_file).read_frame(1).dt == 2e-8
    # Physics settings still refuse a resume.
    s4 = s.model_copy(deep=True)
    s4.Medium.SpeedOfSound = 1500.0
    run4 = su.setup_run(s4, tmp_path / "gt", tmp_path / "res", continue_=True, **common)
    with pytest.raises(ValueError, match="settings"):
        fr.RFWriter(
            run4.paths.rf_file, settings=run4.settings, sequence=run4.sequence, num_frames=3,
            mode="resume",
        )  # fmt: skip
    assert run.paths.settings_file is None


def test_end_frame_inf_and_sequences(settings) -> None:
    a, _ = su.input_handling(settings.Acquisition, settings.SimulationParameters, end_frame=np.inf)
    assert a.EndFrame == settings.Acquisition.NumberOfFrames
    assert su.PULSE_SEQUENCES == {  # main_RF.m
        "Amplitude modulation": ("odd", "even", "all"),
        "Pulse inversion": ("plus", "minus"),
        "Standard": ("pulse",),
        "Amplitude modulation with pulse inversion": ("odd", "even", "minus"),
    }


def test_write_waits_for_a_reader(tmp_path, settings) -> None:
    """A reader holding the file (HDF5 lock) delays a write instead of losing the frame."""
    import subprocess
    import sys
    import time

    path = tmp_path / "rf.h5"
    w = fr.RFWriter(path, settings=settings, sequence=("pulse",), num_frames=2, lock_timeout=30)
    code = (
        "import h5py, sys, time; f = h5py.File(sys.argv[1], 'r'); print('open', flush=True);"
        " time.sleep(1.5); f.close()"
    )
    proc = subprocess.Popen([sys.executable, "-c", code, str(path)], stdout=subprocess.PIPE)
    assert proc.stdout.readline().strip() == b"open"
    t0 = time.monotonic()
    w.write_frame(1, rf_pulses(1), 1e-8)
    proc.wait(timeout=30)
    assert w.completed_frames() == [1]
    blocked = time.monotonic() - t0 > 0.5
    if blocked:  # file locking active (default HDF5 build): the write waited
        assert fr.RFFile(path).frames == [1]
    w2 = fr.RFWriter(path, settings=settings, sequence=("pulse",), num_frames=2, mode="resume",
                     lock_timeout=0.2)  # fmt: skip
    holder = subprocess.Popen([sys.executable, "-c", code, str(path)], stdout=subprocess.PIPE)
    assert holder.stdout.readline().strip() == b"open"
    if blocked:
        with pytest.raises(OSError, match="lock"):
            w2.write_frame(2, rf_pulses(1), 1e-8)
    holder.wait(timeout=30)
