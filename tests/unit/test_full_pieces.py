"""Unit tests for ``acoustic.simulation`` and ``acoustic.full`` (no MATLAB).

MATLAB parity is in ``tests/oracle/test_full_pieces.py``. Runs of k-wave-python's
``python`` backend here are tiny; nothing needs the C++ binary.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import scipy.sparse as sp

from pyproteus.acoustic import full, simulation
from pyproteus.acoustic.full import (
    compute_self_sense_pressure,
    self_sense_grid_size,
    update_source,
)
from pyproteus.acoustic.simulation import run_simulation, solver_backend
from pyproteus.backends.kwave import KGrid

DX, DT = 1e-4, 2e-8


def _grid(shape, **extra):
    return {"Nx": shape[0], "Ny": shape[1], "Nz": shape[2], "dx": DX, "dy": DX, "dz": DX,
            "dt": DT, **extra}  # fmt: skip


def _rp(solver="3DC", **extra):
    return {"solver": solver, "DATA_CAST": "single",
            "PML": {"X_SIZE": 10, "Y_SIZE": 12, "Z_SIZE": 9, "Alpha": 2.0}, **extra}  # fmt: skip


# ----------------------------------------------------------------------------- run_simulation


@pytest.mark.parametrize(
    ("solver", "expected"),
    [
        ("kspaceFirstOrder3D", ("python", "cpu")),
        ("kspaceFirstOrder3DC", ("cpp", "cpu")),
        ("kspaceFirstOrder3DG", ("cpp", "gpu")),
        ("3D", ("python", "cpu")),
        ("3DC", ("cpp", "cpu")),
        ("3DG", ("cpp", "gpu")),
    ],
)
def test_solver_backend(solver, expected) -> None:
    assert solver_backend(solver) == expected


@pytest.mark.parametrize("solver", ["kspaceFirstOrder2D", "3DX", "", "cpp"])
def test_solver_backend_rejects_unknown(solver) -> None:
    with pytest.raises(ValueError, match="unknown solver"):
        solver_backend(solver)


@pytest.fixture
def captured(monkeypatch):
    calls = []

    def fake(kgrid, medium, source, sensor, **kw):
        calls.append(kw)
        return {"p": np.zeros((int(np.count_nonzero(sensor["mask"])), kgrid.Nt), np.float32)}

    monkeypatch.setattr(simulation, "kspace_first_order", fake)
    return calls


def test_run_simulation_options(captured) -> None:
    kgrid = KGrid((4, 5, 6), (DX,) * 3, DT, 7)
    mask = np.zeros(kgrid.shape, bool)
    mask[1, 2, 3] = True
    rp = _rp("kspaceFirstOrder3DG", DATA_CAST="gpuArray-single", DEVICE_NUM=1.0,
             DATA_PATH="/tmp/x", BINARY_PATH="/opt/kwave/")  # fmt: skip
    out = run_simulation(rp, kgrid, {}, {}, {"mask": mask})
    assert out["p"].shape == (1, 7)
    (kw,) = captured
    assert kw["pml_size"] == (10, 12, 9) and kw["pml_alpha"] == 2.0
    assert kw["pml_inside"] is False and kw["smooth"] is False
    assert kw["data_cast"] == "gpuArray-single"
    assert (kw["backend"], kw["device"], kw["device_num"]) == ("cpp", "gpu", 1)
    assert kw["binary"] == "/opt/kwave/" and kw["data_path"] == "/tmp/x"


def test_run_simulation_python_backend_ignores_binary_and_warns_movie(captured) -> None:
    kgrid = KGrid((4, 5, 6), (DX,) * 3, DT, 7)
    mask = np.ones(kgrid.shape, bool)
    with pytest.warns(UserWarning, match="record_movie"):
        run_simulation(_rp("3D", record_movie=True, BINARY_PATH="/x"), kgrid, {}, {},
                       {"mask": mask})  # fmt: skip
    (kw,) = captured
    assert kw["backend"] == "python" and kw["binary"] is None and kw["data_path"] is None


def test_run_simulation_empty_mask(captured) -> None:
    kgrid = KGrid((4, 5, 6), (DX,) * 3, DT, 9)
    with pytest.warns(UserWarning, match="No nonzero elements in sensor mask"):
        out = run_simulation(_rp(), kgrid, {}, {}, {"mask": np.zeros(kgrid.shape)})
    assert out["p"].shape == (0, 9) and not captured


def test_run_simulation_needs_pml() -> None:
    kgrid = KGrid((4, 5, 6), (DX,) * 3, DT, 9)
    with pytest.raises(ValueError, match="PML"):
        run_simulation({"solver": "3DC"}, kgrid, {}, {}, {"mask": np.ones(kgrid.shape)})


# ----------------------------------------------------------------------------- update_source


def _f(shape, *subs):
    return np.ravel_multi_index(subs, shape, order="F")


def test_update_source_merges_by_find_order() -> None:
    shape = (4, 3, 2)
    old_mask = np.zeros(shape, bool)
    old_mask[1, 0, 0] = old_mask[2, 2, 1] = True  # find order: (1,0,0), (2,2,1)
    p_old = np.array([[1.0, 2.0], [3.0, 4.0]])
    idx_new = np.array([_f(shape, 0, 1, 0), _f(shape, 2, 2, 1)])  # one new, one shared
    w = np.array([[1.0, 0.0], [0.5, 2.0]])
    ms = np.array([[1.0, 1.0, 1.0], [0.0, 2.0, 4.0]]) * DX**3
    c = np.full(shape, 1500.0)
    out = update_source(
        {"p_mask": old_mask, "p": p_old}, ms, w, idx_new, _grid(shape), {"sound_speed": c}
    )
    k = 1500.0 * DX / 2
    new = (w @ (ms / DX**3)) * k
    # find order of the merged mask: (1,0,0), (0,1,0), (2,2,1)
    expected = np.array([[1.0, 2.0, 0.0], new[0], [3.0 + new[1, 0], 4.0 + new[1, 1], new[1, 2]]])
    np.testing.assert_allclose(out["p"], expected, rtol=1e-14)
    assert out["p_mask"].sum() == 3 and out["p_mask"][0, 1, 0]
    assert old_mask.sum() == 2 and out["p"] is not p_old  # inputs untouched


def test_update_source_sparse_equals_dense_and_scalar_sound_speed() -> None:
    shape = (5, 4, 3)
    rng = np.random.default_rng(0)
    idx_new = np.sort(rng.choice(60, 7, replace=False), kind="stable")
    w = rng.random((7, 2)) * (rng.random((7, 2)) > 0.4)
    ms = rng.standard_normal((2, 11)).astype(np.float32)
    a = update_source(None, ms, sp.csr_matrix(w), idx_new, _grid(shape), {"sound_speed": 1540.0})
    b = update_source(None, ms, w, idx_new, _grid(shape), {"sound_speed": np.full(shape, 1540.0)})
    np.testing.assert_allclose(a["p"], b["p"], rtol=1e-14)  # sparse vs dense summation
    assert a["p"].dtype == np.float64 and a["p"].shape == (7, 11)


def test_update_source_pads_shorter_signal() -> None:
    shape = (3, 3, 3)
    mask = np.zeros(shape, bool)
    mask[0, 0, 0] = True
    out = update_source({"p_mask": mask, "p": np.ones((1, 2))}, np.ones((1, 5)), np.ones((1, 1)),
                        [_f(shape, 2, 2, 2)], _grid(shape), {"sound_speed": 1500.0})  # fmt: skip
    assert out["p"].shape == (2, 5)
    np.testing.assert_array_equal(out["p"][0], [1, 1, 0, 0, 0])


def test_update_source_keeps_velocity_source() -> None:
    shape = (3, 3, 3)
    src = {"u_mask": np.ones(shape, bool), "ux": np.ones((27, 4))}
    out = update_source(src, np.ones((1, 4)), np.ones((1, 1)), [5], _grid(shape),
                        {"sound_speed": 1500.0})  # fmt: skip
    assert out["u_mask"] is src["u_mask"] and "p_mask" not in src
    assert out["p"].shape == (1, 4)


# ----------------------------------------------------------------------------- self-sense


@pytest.mark.parametrize(("n_sup", "core"), [(12, 12), (25, 25), (13, 16), (5, 5)])
def test_self_sense_grid_size(n_sup, core) -> None:
    assert self_sense_grid_size(n_sup) == core


def test_point_sensor_off_grid_weights() -> None:
    shape = (12, 12, 12)
    coords = tuple(DX * np.arange(n) for n in shape)
    point = np.array([coords[0][6] + 0.3 * DX, coords[1][1] - 0.2 * DX, coords[2][4]])
    idx = int(_f(shape, 6, 1, 4))
    mask, w = full._point_sensor(point, idx, shape, coords, np.full(3, DX),
                                 np.array([33.0, 32.0, 31.0]), False)  # fmt: skip
    # x: 9 points; y: cropped at 0 -> 1..5 = 6 points; z: on grid -> 1 point
    assert mask.sum() == w.size == 9 * 6 * 1
    assert np.all(mask[2:11, 0:6, 4]) and mask[:, :, 3].sum() == 0
    assert 0.9 < w.sum() < 1.1  # band-limited delta, truncated


def test_point_sensor_on_grid() -> None:
    shape = (6, 6, 6)
    coords = tuple(DX * np.arange(n) for n in shape)
    mask, w = full._point_sensor(
        np.zeros(3), 7, shape, coords, np.full(3, DX), np.full(3, 26.0), True
    )
    assert mask.sum() == 1 and mask.ravel(order="F")[7] and w.tolist() == [1.0]


def test_self_sense_local_grids(monkeypatch) -> None:
    """Cropping, centring, local medium and run_param as compute_self_sense_pressure.m."""
    shape = (20, 15, 14)
    rng = np.random.default_rng(1)
    medium = {"sound_speed": 1500 + rng.random(shape), "density": 1000 + rng.random(shape),
              "BonA": 6.0, "alpha_coeff": rng.random(shape), "alpha_power": 1.5}  # fmt: skip
    coords = [DX * np.arange(n) for n in shape]
    subs = np.array([[10, 7, 7], [0, 14, 3]])
    idx = np.ravel_multi_index(subs.T, shape, order="F")
    points = np.array([[coords[d][s[d]] for d in range(3)] for s in subs])
    grid = _grid(shape, x=coords[0], y=coords[1], z=coords[2], full_size=(50, 45, 44),
                 sensor_on_grid=True)  # fmt: skip
    calls = []

    def fake(rp, kgrid, med, src, sensor, **kw):
        calls.append((rp, kgrid, med, src, sensor))
        return {"p": np.ones((int(sensor["mask"].sum()), kgrid.Nt), np.float32)}

    monkeypatch.setattr(full, "run_simulation", fake)
    ms = np.ones((2, 9), np.float32)
    out = compute_self_sense_pressure(KGrid(shape, (DX,) * 3, DT, 9), grid, idx, points, ms,
                                      medium, 12, _rp())  # fmt: skip
    np.testing.assert_array_equal(out, np.ones((2, 9)))
    (rp0, kg0, med0, _, sen0), (_, kg1, med1, src1, sen1) = calls
    assert rp0["PML"] == full.SELF_SENSE_PML and rp0["solver"] == "3DC"
    # bubble 0: I = 4..15 (12), J = 1..12, K = 1..12 -> centre (6, 6, 6)
    assert kg0.shape == (12, 12, 12) and kg0.Nt == 9 and kg0.dt == DT
    np.testing.assert_array_equal(med0["sound_speed"], medium["sound_speed"][4:16, 1:13, 1:13])
    assert med0["BonA"] == 6.0 and med0["alpha_power"] == 1.5
    assert np.flatnonzero(sen0["mask"].ravel(order="F")).tolist() == [6 + 12 * 6 + 144 * 6]
    # bubble 1 at (0, 14, 3): I = 0..5 (6), J = 8..14 (7), K = 0..8 (9) -> centre (0, 6, 3)
    assert kg1.shape == (6, 7, 9)
    np.testing.assert_array_equal(med1["density"], medium["density"][0:6, 8:15, 0:9])
    assert sen1["mask"][0, 6, 3] and sen1["mask"].sum() == 1
    # p = ms / V * c0 * dx / 2 at the bubble voxel
    np.testing.assert_allclose(
        src1["p"][0, 0], 1 / DX**3 * medium["sound_speed"][0, 14, 3] * DX / 2
    )


def test_self_sense_python_backend_translation_invariant() -> None:
    """Homogeneous medium: two interior on-grid bubbles sense the same pressure."""
    shape = (24, 22, 20)
    medium = {"sound_speed": np.full(shape, 1500.0), "density": np.full(shape, 1000.0),
              "BonA": np.full(shape, 6.0), "alpha_coeff": np.full(shape, 0.5),
              "alpha_power": 1.5}  # fmt: skip
    coords = [DX * np.arange(n) for n in shape]
    subs = np.array([[8, 9, 9], [15, 12, 10]])
    idx = np.ravel_multi_index(subs.T, shape, order="F")
    points = np.array([[coords[d][s[d]] for d in range(3)] for s in subs])
    grid = _grid(shape, x=coords[0], y=coords[1], z=coords[2], full_size=(44, 42, 40),
                 sensor_on_grid=False)  # fmt: skip
    nt = 60
    t = np.arange(nt) * DT - 3e-7
    ms = 1e-8 * np.exp(-((t / 1e-7) ** 2)) * np.sin(2 * np.pi * 3e6 * t)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = compute_self_sense_pressure(KGrid(shape, (DX,) * 3, DT, nt), grid, idx, points,
                                          np.vstack([ms, ms]).astype(np.float32), medium, 8,
                                          _rp("3D"))  # fmt: skip
    assert np.abs(out[0]).max() > 1.0
    np.testing.assert_allclose(out[0], out[1], rtol=0, atol=1e-5 * np.abs(out[0]).max())


def test_self_sense_n_jobs_and_solver_override(monkeypatch) -> None:
    """Bubble-parallel runs keep bubble order and split the cores; solver override applies."""
    shape = (16, 16, 16)
    coords = [DX * np.arange(n) for n in shape]
    subs = np.array([[4, 5, 6], [8, 8, 8], [11, 3, 9], [6, 12, 2], [9, 9, 13]])
    idx = np.ravel_multi_index(subs.T, shape, order="F")
    points = np.array([[coords[d][s[d]] for d in range(3)] for s in subs])
    grid = _grid(shape, x=coords[0], y=coords[1], z=coords[2], full_size=(36, 36, 36),
                 sensor_on_grid=True)  # fmt: skip
    seen = []

    def fake(rp, kgrid, med, src, sensor, **kw):
        seen.append((rp["solver"], kw["num_threads"]))
        # echo the source amplitude so each row identifies its bubble
        return {"p": np.asarray(src["p"], np.float32)}

    monkeypatch.setattr(full, "run_simulation", fake)
    monkeypatch.setattr(full.os, "cpu_count", lambda: 12)
    ms = (np.arange(5)[:, None] + 1.0) * np.ones((5, 4))
    medium = {"sound_speed": 1500.0, "density": 1000.0, "alpha_power": 1.5}
    kg = KGrid(shape, (DX,) * 3, DT, 4)
    seq = compute_self_sense_pressure(kg, grid, idx, points, ms, medium, 8, _rp())
    assert {s for s in seen} == {("3DC", None)}
    seen.clear()
    par = compute_self_sense_pressure(kg, grid, idx, points, ms, medium, 8, _rp(), n_jobs=3,
                                      solver="3D")  # fmt: skip
    np.testing.assert_array_equal(par, seq)
    assert np.all(np.diff(seq[:, 0]) > 0)  # rows in bubble order
    assert {s for s in seen} == {("3D", 4)} and len(seen) == 5


def test_update_source_rejects_row_mismatch() -> None:
    shape = (3, 3, 3)
    mask = np.zeros(shape, bool)
    mask[0, 0, :2] = True
    with pytest.raises(ValueError, match="rows"):
        update_source({"p_mask": mask, "p": np.ones((1, 3))}, np.ones((1, 3)), np.ones((1, 1)),
                      [5], _grid(shape), {"sound_speed": 1500.0})  # fmt: skip


def test_self_sense_accepts_1d_mass_source_for_one_bubble(monkeypatch) -> None:
    shape = (10, 10, 10)
    coords = [DX * np.arange(n) for n in shape]
    grid = _grid(shape, x=coords[0], y=coords[1], z=coords[2], full_size=(30, 30, 30),
                 sensor_on_grid=True)  # fmt: skip
    monkeypatch.setattr(
        full,
        "run_simulation",
        lambda rp, kg, med, src, sen, **kw: {"p": np.asarray(src["p"], np.float32)},
    )
    point = [[coords[0][5], coords[1][5], coords[2][5]]]
    kg = KGrid(shape, (DX,) * 3, DT, 6)
    ms = np.arange(1.0, 7.0)
    out = compute_self_sense_pressure(kg, grid, [555], point, ms, {"sound_speed": 1500.0}, 8, _rp())
    assert out.shape == (1, 6) and np.all(out[0] > 0) and np.all(np.diff(out[0]) > 0)
