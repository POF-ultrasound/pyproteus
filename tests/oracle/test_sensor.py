"""Oracle tests: k-Wave sensors (``update_sensor``/``update_sensor_fast``, ``define_sensor_*``,
``extract_sensor_data``).

Classes (D-13): masks, voxel indices and sparsity patterns are compared exactly
(they come from IEEE rounding and exact comparisons on the same inputs); the
band-limited delta weights, positions and distances at the float class
(≤ 1e-9 rel, absolute floor 1e-12 × max). MATLAB sensor weights are compared as
``full()`` matrices; MATLAB linear indices are 1-based, pyproteus' 0-based.

Inputs hit the traps: odd/even/non-cubic grids (and odd/even ``full_size``, which
switch the delta function between its two forms), points exactly on nodes, on
one or two grid lines, on voxel boundaries (ties at .5, power-of-two spacing so
the halves are exact), within the truncation half-width of the domain edge, on
the domain boundary, coincident points, points outside (dropped by
``voxelize_media_points``), and a non-empty input mask.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from pyproteus import acoustic
from pyproteus.acoustic import grid as grid_mod
from pyproteus.medium import voxelize_media_points
from tests.oracle.helpers import col, from_matlab, get, put, row
from tests.oracle.test_acoustic_grid import (
    assert_float,
    build_settings,
    load_settings,
    matlab_value,
    update_settings,
)

# ------------------------------------------------------------------ helpers


def make_grid(
    n: tuple[int, int, int],
    d: float,
    start: tuple[int, int, int],
    full: tuple[int, int, int],
    on_grid: bool,
) -> grid_mod.Grid:
    x, y, z = (np.arange(s, s + k, dtype=float) * d for s, k in zip(start, n, strict=True))
    return grid_mod.Grid(
        x=x, y=y, z=z, Nx=n[0], Ny=n[1], Nz=n[2], dx=d, dy=d, dz=d, dt=1e-8,
        full_size=full, PML=grid_mod.PML(1, 1, 1), sensor_on_grid=on_grid,
    )  # fmt: skip


def put_grid(eng: Any, g: grid_mod.Grid, var: str = "pys_G__") -> None:
    """The same grid as a MATLAB struct (coordinates copied bit for bit)."""
    put(
        eng,
        pys_x__=row(g.x), pys_y__=row(g.y), pys_z__=row(g.z),
        pys_fs__=col(np.asarray(g.full_size, dtype=float)),
    )  # fmt: skip
    eng.eval(
        f"{var} = struct('x', pys_x__, 'y', pys_y__, 'z', pys_z__, 'Nx', {g.Nx}, 'Ny', {g.Ny},"
        f" 'Nz', {g.Nz}, 'dx', {g.dx!r}, 'dy', {g.dy!r}, 'dz', {g.dz!r}, 'dt', {g.dt!r},"
        f" 'full_size', pys_fs__, 'sensor_on_grid', {str(bool(g.sensor_on_grid)).lower()});"
        "clear pys_x__ pys_y__ pys_z__ pys_fs__",
        nargout=0,
    )


def trap_points(g: grid_mod.Grid, rs: np.random.RandomState, n_random: int = 12) -> np.ndarray:
    """Random interior points plus every placement trap (some outside the grid)."""
    lo = np.array([g.x[0], g.y[0], g.z[0]])
    hi = np.array([g.x[-1], g.y[-1], g.z[-1]])
    d = g.dx
    P = [lo + rs.random_sample(3) * (hi - lo) for _ in range(n_random)]
    P.append(P[0].copy())  # coincident with the first
    P.append(np.array([g.x[2], g.y[3], g.z[1]]))  # exactly on a node
    P.append(np.array([g.x[3], g.y[1] + 0.3 * d, g.z[2] + 0.7 * d]))  # on one grid line (x)
    P.append(np.array([g.x[1] + 0.4 * d, g.y[2], g.z[2]]))  # on two grid lines (y, z)
    P.append(np.array([g.x[2] + 0.5 * d, g.y[2] + 0.5 * d, g.z[1] + 0.5 * d]))  # ties at .5
    P.append(np.array([g.x[-2] + 0.5 * d, g.y[0] + 0.25 * d, g.z[-1] - 0.5 * d]))  # ties near edge
    P.append(lo.copy())  # domain corner (min)
    P.append(hi.copy())  # domain corner (max)
    P.append(np.array([g.x[0] + 0.1 * d, g.y[-1] - 0.2 * d, g.z[1] + 0.6 * d]))  # edge, truncated
    P.append(lo - np.array([0.0, 0.5 * d, 0.0]))  # outside (y)
    P.append(hi + np.array([1e-3 * d, 0.0, 0.0]))  # just outside (x)
    return np.array(P)


def matlab_sensor(eng: Any, fn: str, mask_only: bool, tmp: Path, mask0: str = "pys_m0__"):
    """Run ``fn`` (update_sensor or update_sensor_fast) on the workspace inputs."""
    eng.eval(
        f"s0 = struct('mask', {mask0});"
        f"[pys_s__, pys_w__] = {fn}(s0, pys_P__, pys_idx__, pys_G__, {str(mask_only).lower()});"
        "pys_mask__ = logical(pys_s__.mask); pys_wf__ = full(pys_w__); clear s0",
        nargout=0,
    )
    mask, w = get(eng, "pys_mask__", "pys_wf__")
    return mask.astype(bool), w


def compare_weights(w_py: Any, w_ref: np.ndarray, n_points: int, n_mask: int, label: str) -> None:
    """pyproteus N × nnz(mask) vs MATLAB max(i) × max(j) (padded with zero columns)."""
    assert w_py.shape == (n_points, n_mask), label
    dense = w_py.toarray()
    ref = np.zeros_like(dense)
    if w_ref.size:
        ref[: w_ref.shape[0], : w_ref.shape[1]] = w_ref
    np.testing.assert_array_equal(dense != 0, ref != 0, err_msg=f"sparsity {label}")
    assert_float(dense, ref, atol=1e-12 * max(np.max(np.abs(ref), initial=0.0), 1.0), err_msg=label)


# ------------------------------------------------------------------ update_sensor

GRIDS = [
    # (N, spacing, start index, full size)
    ((13, 10, 7), 2.0**-13, (0, -5, -3), (45, 40, 37)),  # odd/even, power-of-two spacing
    ((11, 9, 12), 1e-4, (3, -4, -6), (40, 39, 42)),  # decimal spacing, even/odd full size
    ((6, 15, 9), 2.0**-12, (-2, -7, -4), (36, 45, 39)),  # short x axis (< 2 th + 1)
]


@pytest.mark.parametrize("gcase", GRIDS, ids=["pow2", "decimal", "short"])
@pytest.mark.parametrize("on_grid", [False, True])
@pytest.mark.parametrize("mask_only", [False, True])
def test_update_sensor(
    matlab_engine: Any, tmp_path: Path, gcase: tuple, on_grid: bool, mask_only: bool
) -> None:
    eng = matlab_engine
    g = make_grid(*gcase, on_grid=on_grid)
    put_grid(eng, g)
    rs = np.random.RandomState(GRIDS.index(gcase))
    P = trap_points(g, rs)
    vox = voxelize_media_points(P, g)
    assert vox.exclude.sum() == 2
    put(eng, pys_P__=vox.points, pys_idx__=col(vox.index.astype(float) + 1))
    eng.eval("pys_m0__ = zeros(pys_G__.Nx, pys_G__.Ny, pys_G__.Nz);", nargout=0)
    for fn in ("update_sensor", "update_sensor_fast"):
        mask_ref, w_ref = matlab_sensor(eng, fn, mask_only, tmp_path)
        s, w = getattr(acoustic, fn)(
            acoustic.Sensor(np.zeros(g.shape)), vox.points, vox.index, g, mask_only
        )
        np.testing.assert_array_equal(s.mask.astype(bool), mask_ref, err_msg=fn)
        if mask_only:
            assert w is None
            assert w_ref.size == 0
        else:
            compare_weights(w, w_ref, len(vox.points), int(mask_ref.sum()), fn)
            if not on_grid:  # each row integrates to ~1 near the centre (sanity)
                assert np.all(np.asarray(w.sum(axis=1)).ravel() > 0.5)
    eng.eval("clear pys_*", nargout=0)


def test_update_sensor_existing_mask(matlab_engine: Any, tmp_path: Path) -> None:
    """Points added to a non-empty mask (``define_sensor_MB_all`` accumulates masks).

    Masks: both MATLAB versions and pyproteus agree. Weights: pyproteus follows
    ``update_sensor.m`` (columns index the whole updated mask); ``update_sensor_fast.m``
    indexes only the new points' nodes (B-150), shown here to differ.
    """
    eng = matlab_engine
    g = make_grid(*GRIDS[0], on_grid=False)
    put_grid(eng, g)
    rs = np.random.RandomState(5)
    m0 = rs.random_sample(g.shape) < 0.05
    P = trap_points(g, rs, n_random=4)
    vox = voxelize_media_points(P, g)
    put(eng, pys_P__=vox.points, pys_idx__=col(vox.index.astype(float) + 1), pys_m0__=m0)
    s, w = acoustic.update_sensor(acoustic.Sensor(m0.copy()), vox.points, vox.index, g, False)
    n_mask = int(s.mask.sum())
    mask_ref, w_ref = matlab_sensor(eng, "update_sensor", False, tmp_path)
    np.testing.assert_array_equal(s.mask, mask_ref)
    compare_weights(w, w_ref, len(vox.points), n_mask, "existing mask")
    mask_fast, w_fast = matlab_sensor(eng, "update_sensor_fast", False, tmp_path)
    np.testing.assert_array_equal(mask_fast, mask_ref)
    assert w_fast.shape[1] < n_mask  # B-150: fast's columns skip the pre-existing mask points
    _, w_mask = acoustic.update_sensor(acoustic.Sensor(m0.copy()), vox.points, vox.index, g, True)
    assert w_mask is None
    eng.eval("clear pys_*", nargout=0)


def test_update_sensor_threshold(matlab_engine: Any, tmp_path: Path) -> None:
    """``Grid.threshold`` (read by ``update_sensor_fast.m`` only) changes the truncation."""
    eng = matlab_engine
    g = make_grid(*GRIDS[1], on_grid=False)
    put_grid(eng, g)
    eng.eval("pys_G__.threshold = 2;", nargout=0)
    P = trap_points(g, np.random.RandomState(9), n_random=5)
    vox = voxelize_media_points(P, g)
    put(eng, pys_P__=vox.points, pys_idx__=col(vox.index.astype(float) + 1))
    eng.eval("pys_m0__ = false(pys_G__.Nx, pys_G__.Ny, pys_G__.Nz);", nargout=0)
    mask_ref, w_ref = matlab_sensor(eng, "update_sensor_fast", False, tmp_path)
    s, w = acoustic.update_sensor(
        acoustic.Sensor(np.zeros(g.shape, bool)), vox.points, vox.index, g, threshold=2
    )
    np.testing.assert_array_equal(s.mask, mask_ref)
    compare_weights(w, w_ref, len(vox.points), int(mask_ref.sum()), "threshold 2")
    eng.eval("clear pys_*", nargout=0)


def test_update_sensor_tiny_settings_grid(matlab_engine: Any, oracle: Any, tmp_path: Path) -> None:
    """PROTEUS's own ``UpdateSensorTest`` set-up on the default settings' ``define_grid`` grid."""
    eng = matlab_engine
    build_settings(eng, oracle.path, "pys_S__")
    S = load_settings(eng, tmp_path, "pys_S__")
    for on_grid in (False, True):
        S.SimulationParameters.SensorOnGrid = on_grid
        eng.eval(f"pys_S__.SimulationParameters.SensorOnGrid = {str(on_grid).lower()};", nargout=0)
        eng.eval(
            "pys_G__ = define_grid(pys_S__.SimulationParameters, pys_S__.Geometry);", nargout=0
        )
        g = acoustic.define_grid(S.SimulationParameters, S.Geometry)
        ref_x = from_matlab(eng.eval("pys_G__.x", nargout=1)).ravel()
        np.testing.assert_array_equal(g.x, ref_x)  # same grid, bit for bit
        rs = np.random.RandomState(11)
        lo = np.array([g.x[0], g.y[0], g.z[0]])
        hi = np.array([g.x[-1], g.y[-1], g.z[-1]])
        P = lo + (0.1 + 0.8 * rs.random_sample((25, 3))) * (hi - lo)
        P[1] = P[0]
        P[2] = [g.x[4], g.y[4], g.z[4]]
        vox = voxelize_media_points(P, g)
        put(eng, pys_P__=vox.points, pys_idx__=col(vox.index.astype(float) + 1))
        eng.eval("pys_m0__ = zeros(pys_G__.Nx, pys_G__.Ny, pys_G__.Nz);", nargout=0)
        for fn in ("update_sensor", "update_sensor_fast"):
            mask_ref, w_ref = matlab_sensor(eng, fn, False, tmp_path)
            s, w = acoustic.update_sensor(
                acoustic.Sensor(np.zeros(g.shape)), vox.points, vox.index, g
            )
            np.testing.assert_array_equal(s.mask.astype(bool), mask_ref)
            compare_weights(w, w_ref, 25, int(mask_ref.sum()), f"{fn} on_grid={on_grid}")
    eng.eval("clear pys_*", nargout=0)


# ------------------------------------------------------------------ define_sensor_transducer


def transducer_case(eng: Any, oracle: Any, tmp: Path, kind: str, n_el: int, density: float):
    """MATLAB settings, grid and transducer with integration points (as main_RF builds them)."""
    build_settings(eng, oracle.path, "pys_S__")
    if kind == "C1-6D":
        eng.eval(
            "pys_S__.Transducer = struct('Type', 'C1-6D');"
            "pys_S__.Transducer = get_transducer_properties(pys_S__.Transducer);"
            "if strcmp(pys_S__.Transducer.ImpulseResponseType,'Estimate'),"
            " pys_S__.Transducer.SamplingRate = 250e6;"
            " pys_S__.Transducer = estimate_impulse_response(pys_S__.Transducer); end;"
            "pys_S__.Transmit = reset_transmit(pys_S__.Transducer);",
            nargout=0,
        )
    eng.eval(
        f"pys_S__.Transducer.NumberOfElements = {n_el};"
        f"pys_S__.Transducer.IntegrationDensity = {density!r};"
        "pys_S__.Transmit.Angle = 7;",  # steered: varied delays for the source tests
        nargout=0,
    )
    update_settings(eng, "pys_S__")
    S = load_settings(eng, tmp, "pys_S__")
    eng.eval(
        "pys_G__ = define_grid(pys_S__.SimulationParameters, pys_S__.Geometry);"
        "pys_T__ = get_transducer_integration_points(pys_S__.Transducer, pys_G__);"
        "pys_T__ = get_transducer_integration_delays(pys_T__, pys_S__.Medium);",
        nargout=0,
    )
    g = acoustic.define_grid(S.SimulationParameters, S.Geometry)
    np.testing.assert_array_equal(g.x, from_matlab(eng.eval("pys_G__.x", nargout=1)).ravel())
    T = matlab_value(eng, "pys_T__", tmp)
    return S, g, T


TRANSDUCERS = [("P4-1", 6, 1.0), ("P4-1", 5, 2.3), ("C1-6D", 7, 1.0)]


@pytest.mark.parametrize("tcase", TRANSDUCERS, ids=["P4-1", "P4-1-dense", "C1-6D"])
@pytest.mark.parametrize("on_grid", [False, True])
def test_define_sensor_transducer(
    matlab_engine: Any, oracle: Any, tmp_path: Path, tcase: tuple, on_grid: bool
) -> None:
    eng = matlab_engine
    _, g, T = transducer_case(eng, oracle, tmp_path, *tcase)
    T["OnGrid"] = on_grid
    eng.eval(
        f"pys_T__.OnGrid = {str(on_grid).lower()};"
        "[pys_s__, pys_w__] = define_sensor_transducer(pys_T__, pys_G__);"
        "pys_mask__ = pys_s__.mask; pys_wf__ = full(pys_w__); pys_rec__ = pys_s__.record{1};",
        nargout=0,
    )
    mask_ref, w_ref = get(eng, "pys_mask__", "pys_wf__")
    s, w = acoustic.define_sensor_transducer(T, g)
    assert s.mask.dtype == bool
    np.testing.assert_array_equal(s.mask, mask_ref.astype(bool))
    n_points = T["integration_points"].shape[0] * T["integration_points"].shape[1]
    compare_weights(w, w_ref, n_points, int(mask_ref.sum()), f"{tcase} on_grid={on_grid}")
    assert s.record == ("p",) and eng.eval("pys_rec__", nargout=1) == "p"
    eng.eval("clear pys_*", nargout=0)


def test_define_sensor_transducer_outside(matlab_engine: Any, oracle: Any, tmp_path: Path) -> None:
    import matlab.engine

    eng = matlab_engine
    _, g, T = transducer_case(eng, oracle, tmp_path, "P4-1", 4, 1.0)
    T["OnGrid"] = False
    T["integration_points"] = T["integration_points"].copy()
    T["integration_points"][0, 0, 0] -= 1.0  # one point 1 m behind the array
    eng.eval("pys_T__.OnGrid = false;"
             " pys_T__.integration_points(1,1,1) = pys_T__.integration_points(1,1,1) - 1;",
             nargout=0)  # fmt: skip
    with pytest.raises(matlab.engine.MatlabExecutionError, match="outside the grid"):
        eng.eval("define_sensor_transducer(pys_T__, pys_G__);", nargout=0)
    with pytest.raises(ValueError, match="outside the grid"):
        acoustic.define_sensor_transducer(T, g)
    eng.eval("clear pys_*", nargout=0)


# ------------------------------------------------------------------ define_sensor_MB(_all)


def bubbles(g: grid_mod.Grid, rs: np.random.RandomState) -> dict[str, np.ndarray]:
    P = trap_points(g, rs, n_random=8)
    n = len(P)
    return {
        "points": P,
        "radii": rs.uniform(1e-6, 3e-6, (n, 1)),
        "velocities": rs.normal(size=(n, 3)) * 1e-2,
    }


@pytest.mark.parametrize("on_grid", [False, True])
def test_define_sensor_MB(matlab_engine: Any, tmp_path: Path, on_grid: bool) -> None:
    eng = matlab_engine
    g = make_grid(*GRIDS[0], on_grid=on_grid)
    put_grid(eng, g)
    mb = bubbles(g, np.random.RandomState(21))
    put(eng, pys_P__=mb["points"], pys_R__=mb["radii"], pys_V__=mb["velocities"])
    eng.eval(
        "pys_MB__ = struct('points', pys_P__, 'radii', pys_R__, 'velocities', pys_V__);"
        "[pys_s__, pys_w__, pys_MB2__, pys_md__] = define_sensor_MB(pys_G__, pys_MB__);"
        "pys_mask__ = pys_s__.mask; pys_wf__ = full(pys_w__);",
        nargout=0,
    )
    s, w, mb2, max_dist = acoustic.define_sensor_MB(g, mb)
    ref = matlab_value(eng, "pys_MB2__", tmp_path)
    mask_ref, w_ref, md_ref = get(eng, "pys_mask__", "pys_wf__", "pys_md__")
    np.testing.assert_array_equal(s.mask, mask_ref)
    compare_weights(w, w_ref, len(mb2["points"]), int(mask_ref.sum()), "MB")
    np.testing.assert_array_equal(mb2["points"], ref["points"])
    np.testing.assert_array_equal(mb2["nodes"] + 1, ref["nodes"])
    np.testing.assert_array_equal(mb2["idx"] + 1, ref["idx"].ravel())
    np.testing.assert_array_equal(mb2["radii"], ref["radii"])
    np.testing.assert_array_equal(mb2["velocities"], ref["velocities"])
    assert_float(max_dist, md_ref.item())
    assert len(mb["points"]) == len(mb2["points"]) + 2  # input not modified, 2 dropped
    eng.eval("clear pys_*", nargout=0)


def test_define_sensor_MB_no_bubbles_inside(matlab_engine: Any) -> None:
    eng = matlab_engine
    g = make_grid(*GRIDS[1], on_grid=False)
    put_grid(eng, g)
    P = np.array([[g.x[0] - 1e-3, g.y[0], g.z[0]], [g.x[-1], g.y[-1] + 1e-3, g.z[0]]])
    put(eng, pys_P__=P, pys_R__=np.ones((2, 1)), pys_V__=np.zeros((2, 3)))
    eng.eval(
        "pys_MB__ = struct('points', pys_P__, 'radii', pys_R__, 'velocities', pys_V__);"
        "[pys_s__, pys_w__, pys_MB2__, pys_md__] = define_sensor_MB(pys_G__, pys_MB__);"
        "pys_n__ = nnz(pys_s__.mask); pys_sz__ = size(pys_w__); pys_np__ = size(pys_MB2__.points);",
        nargout=0,
    )
    s, w, mb2, max_dist = acoustic.define_sensor_MB(
        g, {"points": P, "radii": np.ones((2, 1)), "velocities": np.zeros((2, 3))}
    )
    n, sz, npts, md = get(eng, "pys_n__", "pys_sz__", "pys_np__", "pys_md__")
    assert int(n.item()) == int(s.mask.sum()) == 0
    assert tuple(sz.ravel()) == w.shape == (0, 0)
    assert npts.ravel()[0] == mb2["points"].shape[0] == 0
    assert md.item() == max_dist == 0
    eng.eval("clear pys_*", nargout=0)


@pytest.mark.parametrize("on_grid", [False, True])
def test_define_sensor_MB_all(matlab_engine: Any, tmp_path: Path, on_grid: bool) -> None:
    """Frames written as ground-truth files for MATLAB, passed as loaded structs to pyproteus.

    ``Geometry`` is the identity transform (centre 0, no rotation), so
    ``load_microbubbles`` returns the stored points unchanged.
    """
    eng = matlab_engine
    g = make_grid(*GRIDS[2], on_grid=on_grid)
    put_grid(eng, g)
    rs = np.random.RandomState(31)
    start, end, n_frames, n_seq = 2, 4, 12, 3  # two-digit frame numbers (zero padding)
    frames = []
    for frame in range(start, end + 1):
        for k in range(1, n_seq + 1):
            mb = bubbles(g, rs) if (frame, k) != (3, 2) else {
                "points": np.array([[g.x[0] - 1.0, 0.0, 0.0]]),  # all outside
                "radii": np.ones((1, 1)), "velocities": np.zeros((1, 3)),
            }  # fmt: skip
            if (frame, k) == (4, 1):  # more bubbles: sets max_mb
                extra = bubbles(g, rs)
                mb = {key: np.concatenate([mb[key], extra[key]]) for key in mb}
            frames.append(mb)
            put(eng, pys_P__=mb["points"], pys_R__=mb["radii"], pys_V__=mb["velocities"])
            eng.eval(
                f"Frame.Pulse{k} = struct('Points', pys_P__, 'Radius', pys_R__,"
                " 'Velocity', pys_V__);",
                nargout=0,
            )
        file = tmp_path / f"Frame_{frame:02d}.mat"
        eng.eval(f"save('{file}', 'Frame'); clear Frame", nargout=0)
    eng.eval(
        f"pys_Acq__ = struct('StartFrame', {start}, 'EndFrame', {end},"
        f" 'NumberOfFrames', {n_frames});"
        "pys_Geo__.BoundingBox.Center = zeros(3,1); pys_Geo__.Rotation = eye(3);"
        "pys_Geo__.Center = zeros(3,1);"
        f"[pys_s__, pys_idx__, pys_max__] = define_sensor_MB_all(pys_G__, '{tmp_path}',"
        f" pys_Acq__, {n_seq}, pys_Geo__);"
        "pys_mask__ = pys_s__.mask;",
        nargout=0,
    )
    mask_ref, idx_ref, max_ref = get(eng, "pys_mask__", "pys_idx__", "pys_max__")
    s, idx, max_mb = acoustic.define_sensor_MB_all(g, iter(frames))  # one pass, generator-safe
    np.testing.assert_array_equal(s.mask, mask_ref)
    np.testing.assert_array_equal(idx + 1, idx_ref.ravel())
    assert max_mb == int(max_ref.item())
    assert s.record == ("p",)
    # arrays of points work too; no frames at all gives max_mb = 1, as MATLAB's initial value
    _, idx2, _ = acoustic.define_sensor_MB_all(g, [f["points"] for f in frames])
    np.testing.assert_array_equal(idx2, idx)
    assert acoustic.define_sensor_MB_all(g, [])[2] == 1
    eng.eval("clear pys_*", nargout=0)


# ------------------------------------------------------------------ extract_sensor_data


@pytest.mark.parametrize("single", [False, True])
def test_extract_sensor_data(matlab_engine: Any, single: bool) -> None:
    eng = matlab_engine
    rs = np.random.RandomState(41)
    mask_idx = np.unique(rs.randint(0, 5000, 60))
    n, nt = mask_idx.size, 37
    p = rs.normal(size=(n, nt)).astype(np.float32 if single else np.float64)
    mb = np.concatenate([rs.choice(mask_idx, 9, replace=False), [5001, 5003]])  # 2 absent
    tr = rs.choice(mask_idx, 14, replace=False)[::-1].copy()  # unsorted
    dt = 1.0 / 3.0e7
    tr1 = 20 * dt  # floor(tr1/dt) lands on the boundary
    for t_end in (tr1, 20.5 * dt, (nt - 1) * dt):
        put(
            eng, pys_p__=p, pys_mi__=col(mask_idx + 1.0), pys_mt__=col(tr + 1.0),
            pys_mb__=col(mb + 1.0), pys_tr__=row([t_end, 2 * t_end, 3 * t_end]),
        )  # fmt: skip
        eng.eval(
            f"[pys_a__, pys_b__] = extract_sensor_data(struct('p', pys_p__), pys_mi__,"
            f" pys_mt__, pys_mb__, struct('tr', pys_tr__), struct('dt', {dt!r}));"
            "pys_a__ = pys_a__.p; pys_b__ = pys_b__.p;",
            nargout=0,
        )
        a_ref, b_ref = get(eng, "pys_a__", "pys_b__")
        a, b = acoustic.extract_sensor_data(
            {"p": p}, mask_idx, tr, mb, {"tr": [t_end, 2 * t_end, 3 * t_end]}, {"dt": dt}
        )
        assert a["p"].dtype == p.dtype and b["p"].dtype == p.dtype
        np.testing.assert_array_equal(a["p"], a_ref.reshape(a["p"].shape))
        np.testing.assert_array_equal(b["p"], b_ref)
    # too few samples: MATLAB index error, pyproteus IndexError
    import matlab.engine

    put(eng, pys_tr__=row([nt * dt, 0.0, 0.0]))
    with pytest.raises(matlab.engine.MatlabExecutionError):
        eng.eval(
            f"extract_sensor_data(struct('p', pys_p__), pys_mi__, pys_mt__, pys_mb__,"
            f" struct('tr', pys_tr__), struct('dt', {dt!r}));",
            nargout=0,
        )
    with pytest.raises(IndexError):
        acoustic.extract_sensor_data(
            SimpleNamespace(p=p), mask_idx, tr, mb, {"tr": [nt * dt]}, SimpleNamespace(dt=dt)
        )
    eng.eval("clear pys_*", nargout=0)
