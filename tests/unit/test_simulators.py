"""Hybrid/full simulators and the ``main_RF`` / ``main_pressure_field`` entry points.

Wiring checks with stubs (what each stage receives), plus end-to-end runs on a
tiny case with k-wave-python's in-process ``python`` backend (Solver ``'3D'``,
no binary needed). MATLAB parity is in ``tests/oracle/test_main_rf.py``.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.io
import scipy.sparse as sp

from pyproteus.acoustic import simulators
from pyproteus.acoustic.simulators import full_simulator, hybrid_simulator

DATA = Path(__file__).parent / "data"
TINY = DATA / "main_rf_tiny.json"

# ----------------------------------------------------------------------------- stubs


def _grid(n=(6, 5, 4), dx=1e-4, dt=2e-8):
    return SimpleNamespace(
        Nx=n[0], Ny=n[1], Nz=n[2], dx=dx, dy=dx, dz=dx, dt=dt,
        x=np.arange(n[0]) * dx, y=np.arange(n[1]) * dx, z=np.arange(n[2]) * dx,
    )  # fmt: skip


SETTINGS = dict(
    Medium=SimpleNamespace(SpeedOfSound=1500.0, SpeedOfSoundMinimum=1450.0),
    Microbubble=SimpleNamespace(),
    Transmit=SimpleNamespace(CenterFrequency=2.5e6),
)


def _run_param(n_int, dt=2e-8):
    # tr chosen so that Nt1 = 11, Nt3 = 21; interaction window Nt2 = 16
    return {"tr": np.array([10.5 * dt, 0.0, 20.5 * dt]), "max_trans_dist": 15.5 * dt * 1500.0,
            "max_dist": 0.0, "pulse_length": 0.0, "N_interactions": n_int,
            "DATA_CAST": "double"}  # fmt: skip


def test_hybrid_simulator_wiring(monkeypatch) -> None:
    calls = []

    def mass_source(sensed_p, radii, kgrid, *a, **k):
        calls.append(("ms", np.array(sensed_p), kgrid.Nt, kgrid.k_max, k["step_control"]))
        return np.asarray(sensed_p) * 2.0

    def homogeneous(rp, kgrid, medium, source, sensor, *, reference_frequency=None):
        calls.append(("hom", rp["gridded"], kgrid.Nt, reference_frequency,
                      np.asarray(sensor["points"])))  # fmt: skip
        n = np.asarray(sensor["points"]).shape[0]
        return {"p": np.full((n, kgrid.Nt), 0.5)}

    monkeypatch.setattr(simulators, "compute_bubble_mass_source", mass_source)
    monkeypatch.setattr(simulators, "run_simulation_homogeneous", homogeneous)
    grid = _grid()
    mb = {"points": np.zeros((2, 3)), "radii": np.ones((2, 1))}
    sensed = np.arange(22, dtype=float).reshape(2, 11)
    idx = np.array([0, 7, 33])
    out = hybrid_simulator(idx, sensed, mb, grid, {}, _run_param(2), **SETTINGS,
                           step_control="shared")  # fmt: skip
    kinds = [c[0] for c in calls]
    assert kinds == ["ms", "hom", "ms", "hom", "ms", "hom"]
    k_max = np.pi / grid.dt / 1450.0
    # interactions: exact distances, interaction window, f0 as KK reference (D-23)
    for c in (calls[1], calls[3]):
        assert c[1] is False and c[2] == 16 and c[3] == 2.5e6
    assert calls[0][2] == 16 and calls[0][3] == pytest.approx(k_max) and calls[0][4] == "shared"
    np.testing.assert_array_equal(calls[0][1], sensed)
    # second interaction senses p_bubbles[:, :Nt1] + sensed_1iter
    np.testing.assert_array_equal(calls[2][1], sensed + 0.5)
    # receive stage: gridded, Nt3, sensors at the transducer voxel centres
    assert calls[5][1] is True and calls[5][2] == 21 and calls[5][3] == 2.5e6
    i, j, k = np.unravel_index(idx, (6, 5, 4), order="F")
    np.testing.assert_allclose(calls[5][4], np.column_stack([i, j, k]) * 1e-4)
    assert out["p"].shape == (3, 21)


def test_hybrid_simulator_no_interaction(monkeypatch) -> None:
    seen = []
    monkeypatch.setattr(simulators, "compute_bubble_mass_source",
                        lambda s, *a, **k: seen.append(np.array(s)) or np.asarray(s))  # fmt: skip
    monkeypatch.setattr(simulators, "run_simulation_homogeneous",
                        lambda rp, kg, m, src, sen, **k: {"p": np.zeros((1, kg.Nt))})  # fmt: skip
    sensed = np.ones((2, 11))
    hybrid_simulator([0], sensed, {"points": np.zeros((2, 3)), "radii": np.ones(2)}, _grid(),
                     {}, _run_param(0), **SETTINGS)  # fmt: skip
    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0], sensed)


@pytest.mark.parametrize(
    ("mode", "impulse", "homogeneous", "cached"),
    [("strict", False, False, False), ("impulse_response", True, False, True),
     ("green", True, True, True)],
)  # fmt: skip
def test_full_simulator_wiring(monkeypatch, mode, impulse, homogeneous, cached) -> None:
    from pyproteus.backends.kwave import KGrid

    log = []
    monkeypatch.setattr(
        simulators,
        "compute_bubble_mass_source",
        lambda s, r, kg, *a, **k: np.ones((2, kg.Nt), dtype=np.float32),
    )
    monkeypatch.setattr(
        simulators, "update_source", lambda src, ms, w, idx, g, m: {"p": ms, "src": src}
    )
    monkeypatch.setattr(
        simulators,
        "run_simulation",
        lambda rp, kg, m, src, sen, **k: (
            log.append(("k", kg.Nt, sen)) or {"p": np.ones((3, kg.Nt), np.float32)}
        ),
    )

    def self_sense(kgrid, grid, idx, pts, ms, medium, n_sup, rp, **k):
        log.append(("ss", k["impulse_response"], k["homogeneous"], k["cache"], n_sup))
        return np.full((2, kgrid.Nt), 0.25)

    monkeypatch.setattr(simulators, "compute_self_sense_pressure", self_sense)
    grid = _grid()
    kgrid = KGrid((6, 5, 4), (1e-4,) * 3, grid.dt, 1)
    w = sp.csr_matrix(np.ones((2, 3)))
    out = full_simulator({"p_mask": 1}, "sensor_trans", "sensor_frame", w, [0, 1, 2],
                         np.ones((2, 11), np.float32), {"idx": [0, 1], "points": np.zeros((2, 3)),
                         "radii": np.ones(2)}, kgrid, grid, {}, _run_param(2), **SETTINGS,
                         self_sense=mode)  # fmt: skip
    runs = [e for e in log if e[0] == "k"]
    ss = [e for e in log if e[0] == "ss"]
    assert [r[2] for r in runs] == ["sensor_frame", "sensor_frame", "sensor_trans"]
    assert [r[1] for r in runs] == [16, 16, 21]
    assert len(ss) == 2 and all(e[1] is impulse and e[2] is homogeneous for e in ss)
    assert (ss[0][3] is not None and ss[0][3] is ss[1][3]) if cached else ss[0][3] is None
    assert ss[0][4] == int(np.ceil(4 * 1500.0 / 2.5e6 / 1e-4))  # N_sup
    assert out["p"].shape == (3, 21)


def test_full_simulator_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="self_sense"):
        full_simulator(None, None, None, sp.csr_matrix((0, 0)), [], np.zeros((0, 1)), {}, None,
                       _grid(), {}, _run_param(0), **SETTINGS, self_sense="exact")  # fmt: skip


# ------------------------------------------------------------ end to end (python backend)

pytest.importorskip("kwave")


def _write_ground_truth(folder: Path, settings, n_frames: int, n_mb: int, seed: int = 3) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    acq, mbs = settings.Acquisition, settings.Microbubble
    dist = mbs.Distribution
    fsp = {"TimeBtwPulse": float(acq.TimeBetweenPulses), "FrameRate": float(acq.FrameRate),
           "NBPulses": 1.0, "NMicrobubbles": float(mbs.Number), "NumberOfFrames": float(n_frames),
           "Microbubble": {"Distribution": {"Probabilities": np.asarray(dist.Probabilities),
                                            "Radii": np.asarray(dist.Radii)}}}  # fmt: skip
    scipy.io.savemat(folder / "FlowSimulationParameters.mat", {"FlowSimulationParameters": fsp})
    rng = np.random.default_rng(seed)
    D = np.asarray(settings.Geometry.BoundingBox.Diagonal, float).reshape(-1)
    for m in range(1, n_frames + 1):
        pulse = {
            "Points": 0.8 * (rng.random((n_mb, 3)) - 0.5) * D,
            "Radius": 1e-6 + 2e-6 * rng.random((n_mb, 1)),
            "Velocity": np.zeros((n_mb, 3)),
            "StreamNumber": np.arange(1.0, n_mb + 1)[:, None],
        }
        scipy.io.savemat(folder / f"Frame_{m}.mat", {"Frame": {"Pulse1": pulse}})  # fmt: skip


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from pyproteus.config.json_io import load_settings_json

    root = tmp_path_factory.mktemp("tiny")
    s = load_settings_json(TINY)
    _write_ground_truth(
        root / "gt", s, int(s.Acquisition.NumberOfFrames), int(s.Microbubble.Number)
    )
    return root, s


def _main(root, s, out, **kw):
    from pyproteus.acoustic import main_RF

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return main_RF(s, root / "gt", root / out, verbose=False, **kw)


@pytest.mark.slow
def test_main_rf_hybrid_resume(tiny, monkeypatch) -> None:
    """Whole run == first frame + Continue; Continue reads the medium cache, skips done frames."""
    from pyproteus.acoustic import main as main_mod
    from pyproteus.io.frames import RFFile

    root, s = tiny
    whole = list(RFFile(_main(root, s, "whole")))
    assert [f.frame for f in whole] == [1, 2]
    n_el = int(s.Transducer.NumberOfElements)
    for f in whole:
        assert len(f.rf) == 1 and f.rf[0].shape[0] == n_el and f.rf[0].dtype == np.float32
        assert np.all(np.isfinite(f.rf[0])) and np.abs(f.rf[0]).max() > 0
        assert f.bubbles[0]["radii"].size == int(s.Microbubble.Number)

    _main(root, s, "split", end_frame=1)
    assert (root / "split" / "medium.h5").is_file()

    def no_medium(*a, **k):
        raise AssertionError("Continue must read the medium cache")

    monkeypatch.setattr(main_mod, "define_medium", no_medium)
    split = list(RFFile(_main(root, s, "split", continue_=True)))
    assert [f.frame for f in split] == [1, 2]
    for a, b in zip(whole, split, strict=True):
        np.testing.assert_array_equal(a.rf[0], b.rf[0])

    # everything written: returns without any simulation
    monkeypatch.setattr(main_mod, "prepare_simulation", no_medium)
    _main(root, s, "split", continue_=True)


@pytest.mark.slow
def test_main_pressure_field_tiny(tiny) -> None:
    import h5py

    from pyproteus.acoustic import main_pressure_field

    root, s = tiny
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        maps = main_pressure_field(s, root / "pf")
    g = maps.grid
    assert maps.xy_p_max.shape == (g.Nx, g.Ny) and maps.xz_p_max.shape == (g.Nx, g.Nz)
    assert np.all(maps.xy_p_max >= maps.xy_p_min) and maps.xy_p_max.max() > 0 > maps.xy_p_min.min()
    # the planes share the line (:, iy, iz)
    np.testing.assert_array_equal(maps.xy_p_max[:, maps.iy], maps.xz_p_max[:, maps.iz])
    assert maps.iy == int(np.argmin(np.abs(g.y))) and maps.iz == int(np.argmin(np.abs(g.z)))
    with h5py.File(root / "pf" / "pressure_maps.h5") as f:
        np.testing.assert_array_equal(f["xz/p_min"][()], maps.xz_p_min)


@pytest.mark.parametrize(
    ("cast", "given", "expected"),
    [("double", None, True), ("gpuArray-single", None, False), ("gpuArray-single", True, True),
     ("single", False, False)],
)  # fmt: skip
def test_hybrid_receive_gridded_default(monkeypatch, cast, given, expected) -> None:
    """D-31: exact receive propagation by default on the GPU, gridded on the CPU."""
    seen = []
    monkeypatch.setattr(simulators, "compute_bubble_mass_source", lambda s, *a, **k: np.asarray(s))
    monkeypatch.setattr(simulators, "run_simulation_homogeneous",
                        lambda rp, kg, m, src, sen, **k: seen.append(rp["gridded"])
                        or {"p": np.zeros((1, kg.Nt))})  # fmt: skip
    rp = {**_run_param(0), "DATA_CAST": cast}
    hybrid_simulator([0], np.ones((1, 11)), {"points": np.zeros((1, 3)), "radii": np.ones(1)},
                     _grid(), {}, rp, **SETTINGS, gridded_receive=given)  # fmt: skip
    assert seen == [expected]
