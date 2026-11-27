"""Full-simulator k-Wave pieces vs MATLAB PROTEUS + MATLAB k-Wave 1.4.1.

``update_source`` (deterministic float class, 1e-9), ``update_source`` +
``run_simulation`` and ``compute_self_sense_pressure`` (k-Wave class, 1e-5)
against the oracle ``update_source.m``, ``run_simulation.m`` and
``compute_self_sense_pressure.m`` (with its own ``update_sensor.m``), all run
through ``tests/oracle/shims/full_pieces_oracle.m``. MATLAB runs
``kspaceFirstOrder3DC`` on the D-10 binary; pyproteus runs the ``cpp`` backend
(solver ``3DC``) and k-wave-python's ``python`` backend (solver ``3D``).

Cases (tiny grids): heterogeneous nonlinear absorbing medium; a transducer
pressure source whose signals are shorter than ``Nt`` and bubble sources that
partly overlap it (the merge path of ``update_source.m``); self-sense bubbles
off-grid, cropped at the global grid edges, on-grid in one dimension and
on-grid in all three, plus ``sensor_on_grid = true``.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from pyproteus.acoustic.full import compute_self_sense_pressure, update_source
from pyproteus.acoustic.simulation import run_simulation
from pyproteus.backends.kwave import KGrid
from tests.oracle.helpers import col as _col
from tests.oracle.helpers import to_matlab


def col(a):
    return to_matlab(_col(a))


TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))

try:  # load conda's shared libraries before matlab.engine (B-1, B-11)
    from tracer_bullet import kwave_tracer as kt

    kt.preload_before_engine()
    HAVE_KWAVE = True
except ImportError:
    HAVE_KWAVE = False

KWAVE_TOL = 1e-5  # k-Wave fields (D-13)
ROW_TOL = 2e-5  # per row, rows >= 1e-3 of the peak
FLOAT_TOL = 1e-9  # deterministic floats (D-13)
F32_TOL = 2.0**-23  # one float32 ulp (rounding of float32 inputs)
DX = 1e-4
DT = 2e-8


# ----------------------------------------------------------------------------- cases


def _medium(shape, rng) -> dict:
    return {
        "sound_speed": 1500 + 80 * rng.random(shape),
        "density": 1000 + 60 * rng.random(shape),
        "BonA": 6 + rng.random(shape),
        "alpha_coeff": 0.5 + 0.5 * rng.random(shape),
        "alpha_power": 1.5,
    }


def _pulse(nt: int, delay: float, f0: float = 3e6, width: float = 2.5e-7) -> np.ndarray:
    t = np.arange(nt) * DT - delay
    return np.exp(-((t / width) ** 2)) * np.sin(2 * np.pi * f0 * t)


def _f_idx(shape, *subs) -> np.ndarray:
    return np.ravel_multi_index(tuple(np.asarray(s) for s in subs), shape, order="F")


def full_case(seed: int = 3) -> dict:
    """Transducer pressure source + 4 bubbles (one overlapping the transducer)."""
    rng = np.random.default_rng(seed)
    shape, nt = (30, 26, 22), 220
    p_mask = np.zeros(shape, dtype=bool)
    p_mask[2, 5:20, 6:15] = True
    p_mask[2, 5:9, 6:8] = False  # asymmetric
    n_tr = int(p_mask.sum())
    p = np.array([(1 + 0.01 * i) * 2e3 * _pulse(150, 2e-7 + 3e-9 * i) for i in range(n_tr)])
    bubbles = [(2, 10, 10), (15, 13, 11), (21, 7, 16), (9, 20, 4)]
    idx_new, wi, wj, wv = [], [], [], []
    for b in bubbles:  # 3x3x3 neighbourhood of each bubble voxel
        g = np.meshgrid(*(np.arange(c - 1, c + 2) for c in b), indexing="ij")
        idx_new.append(_f_idx(shape, *(a.ravel() for a in g)))
    idx_new = np.unique(np.concatenate(idx_new))
    for n, b in enumerate(bubbles):
        g = np.meshgrid(*(np.arange(c - 1, c + 2) for c in b), indexing="ij")
        rows = np.searchsorted(idx_new, _f_idx(shape, *(a.ravel() for a in g)))
        wi += list(rows)
        wj += [n] * rows.size
        wv += list(0.2 + 0.8 * rng.random(rows.size))
    mass = np.array([1e-8 * (1 + 0.3 * n) * _pulse(nt, 6e-7 + 1e-7 * n) for n in range(4)])
    flat = p_mask.ravel(order="F").copy()
    flat[idx_new] = True
    return {
        "shape": shape, "Nt": nt, "medium": _medium(shape, rng), "p_mask": p_mask, "p": p,
        "idx_new": idx_new, "wi": np.array(wi), "wj": np.array(wj), "wv": np.array(wv),
        "mass_source": mass.astype(np.float32), "n_mb": len(bubbles),
        "sensor_mask": flat.reshape(shape, order="F"),
    }  # fmt: skip


def self_sense_case(sensor_on_grid: bool, seed: int = 5) -> dict:
    rng = np.random.default_rng(seed)
    shape, nt = (28, 24, 20), 200
    coords = [(-1.3e-3 + DX * np.arange(n)) for n in shape]
    subs = [(14, 12, 10), (1, 22, 3), (8, 5, 15), (20, 18, 6)]
    offsets = [(0.3, -0.2, 0.45), (-0.4, 0.35, 0.1), (0.25, 0.0, -0.3), (0.0, 0.0, 0.0)]
    points = np.array(
        [
            [coords[d][s[d]] + o[d] * DX for d in range(3)]
            for s, o in zip(subs, offsets, strict=True)
        ]
    )
    mb_idx = _f_idx(shape, *np.array(subs).T)
    mass = np.array([1e-8 * (1 + 0.2 * n) * _pulse(nt, 5e-7 + 8e-8 * n) for n in range(4)])
    return {
        "shape": shape, "Nt": nt, "medium": _medium(shape, rng), "coords": coords,
        "full_size": np.array([49.0, 44.0, 41.0]), "sensor_on_grid": sensor_on_grid,
        "MB_idx": mb_idx, "MB_points": points, "mass_source": mass.astype(np.float32),
        "N_sup": 12,
    }  # fmt: skip


def _grid(c) -> dict:
    nx, ny, nz = c["shape"]
    g = {"Nx": nx, "Ny": ny, "Nz": nz, "dx": DX, "dy": DX, "dz": DX, "dt": DT}
    if "coords" in c:
        g |= {"x": c["coords"][0], "y": c["coords"][1], "z": c["coords"][2],
              "full_size": c["full_size"], "sensor_on_grid": c["sensor_on_grid"]}  # fmt: skip
    return g


def _run_param(solver: str, data_dir: Path | None = None) -> dict:
    rp = {
        "solver": solver,
        "DATA_CAST": "single",
        "record_movie": False,
        "PML": {"X_SIZE": 10.0, "Y_SIZE": 12.0, "Z_SIZE": 9.0, "Alpha": 2.0},
    }
    if solver.endswith("3DC"):
        rp["BINARY_PATH"] = str(kt.WRAPPER_DIR) + "/"
        if data_dir is not None:
            data_dir.mkdir(parents=True, exist_ok=True)  # MATLAB k-Wave needs it to exist
            rp["DATA_PATH"] = str(data_dir) + "/"
    return rp


def _matlab_struct(c: dict, run_param: dict | None = None) -> dict:
    nx, ny, nz = c["shape"]
    s = {"Nx": float(nx), "Ny": float(ny), "Nz": float(nz), "dx": DX, "dt": DT,
         "Nt": float(c["Nt"]), "mass_source": to_matlab(c["mass_source"])}  # fmt: skip
    for k, v in c["medium"].items():
        s[k] = to_matlab(np.asarray(v, dtype=np.float64)) if np.ndim(v) else float(v)
    if "idx_new" in c:
        s |= {"wi": col(c["wi"] + 1.0), "wj": col(c["wj"] + 1.0), "wv": col(c["wv"]),
              "n_new": float(c["idx_new"].size), "n_mb": float(c["n_mb"]),
              "idx_new": col(c["idx_new"] + 1.0)}  # fmt: skip
        for k in ("p_mask", "p", "u_mask", "ux", "sensor_mask"):
            if k in c:
                s[k] = to_matlab(c[k])
    if "coords" in c:
        s |= {"x": col(c["coords"][0]), "y": col(c["coords"][1]), "z": col(c["coords"][2]),
              "full_size": col(c["full_size"]), "sensor_on_grid": bool(c["sensor_on_grid"]),
              "MB_idx": col(c["MB_idx"] + 1.0), "MB_points": to_matlab(c["MB_points"]),
              "N_sup": float(c["N_sup"])}  # fmt: skip
    if run_param is not None:
        s["run_param"] = run_param
    return s


def _weights(c) -> sp.csr_matrix:
    return sp.csr_matrix((c["wv"], (c["wi"], c["wj"])), shape=(c["idx_new"].size, c["n_mb"]))


def rel_err(a, ref) -> float:
    return float(np.max(np.abs(np.asarray(a) - ref)) / np.max(np.abs(ref)))


def rel_err_rows(a, ref, floor=1e-3) -> float:
    a, ref = np.asarray(a, dtype=np.float64), np.asarray(ref, dtype=np.float64)
    peak = np.max(np.abs(ref), axis=1)
    keep = peak >= floor * peak.max()
    return float(np.max(np.max(np.abs(a - ref), axis=1)[keep] / peak[keep]))


# ----------------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def eng(matlab_engine):
    if not HAVE_KWAVE:
        pytest.skip("k-wave-python (and scipy, h5py) needed")
    matlab_engine.addpath(str(kt.KWAVE_MATLAB), nargout=0)  # for kWaveGrid
    yield matlab_engine
    matlab_engine.rmpath(str(kt.KWAVE_MATLAB), nargout=0)


@pytest.fixture(scope="module")
def keng(eng):
    if not (kt.KWAVE_MATLAB / "kspaceFirstOrder3DC.m").is_file():
        pytest.skip(f"MATLAB k-Wave not found at {kt.KWAVE_MATLAB} (set KWAVE_MATLAB_PATH)")
    if not kt.binary_available():
        pytest.skip("k-Wave binary does not start: run tools/tracer_bullet/setup_libs.sh")
    kt.prepare_engine(eng)
    kt.check_engine_layout(eng)
    yield eng
    eng.rmpath(str(kt.HERE), nargout=0)


def _matlab(eng, c, what, run_param=None) -> dict:
    out = eng.full_pieces_oracle(_matlab_struct(c, run_param), what)
    return {
        k: (v if isinstance(v, str) else np.asarray(v, dtype=np.float64)) for k, v in out.items()
    }


# ----------------------------------------------------------------------------- update_source


def _source_variants():
    c = full_case()
    yield "transducer_p", c, {"p_mask": c["p_mask"], "p": c["p"]}
    yield "empty", c, None
    longer = dict(c, mass_source=c["mass_source"][:, :120].astype(np.float64))  # Nt1 > Nt2
    yield "old_longer_double", longer, {"p_mask": c["p_mask"].astype(np.float64), "p": c["p"]}
    u_mask = np.zeros(c["shape"], dtype=bool)
    u_mask[3, 4:12, 5:9] = True
    ux = np.ones((int(u_mask.sum()), 50))
    yield "velocity_only", c, {"u_mask": u_mask, "ux": ux}


@pytest.mark.parametrize("name", ["transducer_p", "empty", "old_longer_double", "velocity_only"])
def test_update_source_matches_matlab(eng, name) -> None:
    variants = {n: (c, s) for n, c, s in _source_variants()}
    c, src = variants[name]
    cm = dict(c)
    for k in ("p_mask", "p", "u_mask", "ux"):
        cm.pop(k, None)
    if src is not None:
        cm.update(src)
    m = _matlab(eng, cm, "update_source")
    ours = update_source(src, c["mass_source"], _weights(c), c["idx_new"], _grid(c), c["medium"])
    assert m["p_class"] == "double" and ours["p"].dtype == np.float64
    np.testing.assert_array_equal(np.asarray(ours["p_mask"], dtype=np.float64), m["p_mask"])
    assert ours["p"].shape == m["p"].shape
    # float32 mass source: MATLAB divides by the volume in single (one float32 ulp)
    tol = FLOAT_TOL if c["mass_source"].dtype == np.float64 else F32_TOL
    np.testing.assert_allclose(ours["p"], m["p"], rtol=tol, atol=FLOAT_TOL * np.abs(m["p"]).max())
    if src is not None:  # value semantics: the caller's source is untouched
        assert set(src) <= set(ours)
        if "u_mask" in src:
            assert ours["u_mask"] is src["u_mask"] and "p" not in src


def test_update_source_no_new_voxels_returns_source(eng) -> None:
    c = full_case()
    src = {"p_mask": c["p_mask"], "p": c["p"]}
    out = update_source(src, c["mass_source"][:0], np.zeros((0, 0)), [], _grid(c), c["medium"])
    assert out is not src and out["p"] is src["p"]
    assert (
        update_source(None, c["mass_source"][:0], np.zeros((0, 0)), [], _grid(c), c["medium"])
        is None
    )


# ----------------------------------------------------------------------------- run_simulation


@pytest.fixture(scope="module")
def full_run(keng, tmp_path_factory):
    c = full_case()
    data = tmp_path_factory.mktemp("full_run")
    m = _matlab(keng, c, "run_full", _run_param("kspaceFirstOrder3DC", data / "matlab"))
    src = update_source(
        {"p_mask": c["p_mask"], "p": c["p"]}, c["mass_source"], _weights(c), c["idx_new"],
        _grid(c), c["medium"],
    )  # fmt: skip
    kgrid = KGrid(c["shape"], (DX,) * 3, DT, c["Nt"])
    sensor = {"mask": c["sensor_mask"], "record": ["p"]}
    out = {}
    for solver in ("kspaceFirstOrder3DC", "3D"):
        out[solver] = run_simulation(
            _run_param(solver, data / "ours"), kgrid, c["medium"], src, sensor,
            binary_env=kt.binary_env(),
        )["p"]  # fmt: skip
    return c, m["p"], out


@pytest.mark.parametrize("solver", ["kspaceFirstOrder3DC", "3D"])
def test_run_simulation_full_source_matches_matlab(full_run, solver) -> None:
    c, ref, out = full_run
    p = out[solver]
    assert p.dtype == np.float32 and p.shape == ref.shape == (int(c["sensor_mask"].sum()), c["Nt"])
    assert np.abs(ref).max() > 100  # the waves reached the sensors
    assert rel_err(p, ref) <= KWAVE_TOL
    assert rel_err_rows(p, ref) <= ROW_TOL


def test_run_simulation_source_scaling_is_checked(full_run) -> None:
    """A 1 % error in the bubble-source scaling alone is caught at the k-Wave tolerance."""
    c, ref, _ = full_run
    src = update_source(
        {"p_mask": c["p_mask"], "p": c["p"]}, 1.01 * c["mass_source"], _weights(c), c["idx_new"],
        _grid(c), c["medium"],
    )  # fmt: skip
    p = run_simulation(
        _run_param("kspaceFirstOrder3DC"), KGrid(c["shape"], (DX,) * 3, DT, c["Nt"]),
        c["medium"], src, {"mask": c["sensor_mask"]}, binary_env=kt.binary_env(),
    )["p"]  # fmt: skip
    assert rel_err_rows(p, ref) > 100 * ROW_TOL


def test_run_simulation_empty_sensor_matches_matlab(keng) -> None:
    c = full_case()
    c = dict(c, sensor_mask=np.zeros(c["shape"], dtype=bool))
    m = _matlab(keng, c, "run_full", _run_param("kspaceFirstOrder3DC"))
    src = update_source(None, c["mass_source"], _weights(c), c["idx_new"], _grid(c), c["medium"])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = run_simulation(
            _run_param("kspaceFirstOrder3DC"), KGrid(c["shape"], (DX,) * 3, DT, c["Nt"]),
            c["medium"], src, {"mask": c["sensor_mask"]},
        )  # fmt: skip
    assert any("No nonzero elements in sensor mask" in str(x.message) for x in w)
    assert out["p"].shape == (0, c["Nt"]) and m["p"].size == 0


# ----------------------------------------------------------------------------- self-sense


@pytest.fixture(scope="module", params=[False, True], ids=["off_grid", "on_grid"])
def self_sense(request, keng, tmp_path_factory):
    c = self_sense_case(request.param)
    data = tmp_path_factory.mktemp("self_sense")
    m = _matlab(keng, c, "self_sense", _run_param("kspaceFirstOrder3DC", data / "m"))
    kgrid = KGrid(c["shape"], (DX,) * 3, DT, c["Nt"])
    out = {}
    for solver in ("kspaceFirstOrder3DC", "3D"):
        out[solver] = compute_self_sense_pressure(
            kgrid, _grid(c), c["MB_idx"], c["MB_points"], c["mass_source"], c["medium"],
            c["N_sup"], _run_param(solver, data / "ours"), binary_env=kt.binary_env(),
        )  # fmt: skip
    out["parallel"] = compute_self_sense_pressure(
        kgrid, _grid(c), c["MB_idx"], c["MB_points"], c["mass_source"], c["medium"],
        c["N_sup"], _run_param("kspaceFirstOrder3DC"), binary_env=kt.binary_env(), n_jobs=3,
    )  # fmt: skip
    return c, m, out


@pytest.mark.parametrize("solver", ["kspaceFirstOrder3DC", "3D", "parallel"])
def test_self_sense_matches_matlab(self_sense, solver) -> None:
    c, m, out = self_sense
    ref, p = m["p"], out[solver]
    assert m["p_class"] == "double" and p.dtype == np.float64
    assert p.shape == ref.shape == (4, c["Nt"])
    assert np.all(np.abs(ref).max(axis=1) > 1.0)  # every bubble senses itself
    assert np.array_equal(p.astype(np.float32).astype(np.float64), p)  # float32-rounded
    assert rel_err(p, ref) <= KWAVE_TOL
    assert rel_err_rows(p, ref) <= ROW_TOL


def test_self_sense_cases_differ(self_sense) -> None:
    """Rows differ enough that a bubble mix-up or a missing crop would be caught."""
    _, m, _ = self_sense
    ref = m["p"] / np.abs(m["p"]).max(axis=1, keepdims=True)
    for i in range(4):
        for j in range(i + 1, 4):
            assert np.max(np.abs(ref[i] - ref[j])) > 1e-2
