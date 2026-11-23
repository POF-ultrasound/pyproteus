"""Oracle tests: ``voxelize_media_points``.

Class (D-13): voxel subscripts/indices are integers after rounding →
statistical, ≥ 99.9 % equal with the differences reported. With identical
inputs the operations are IEEE subtraction, division and MATLAB rounding, so
the measured agreement is expected to be 100 %; the assertion still uses the
statistical class and prints any differences. The kept points and the
exclusion mask are compared exactly (comparisons on identical doubles).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pyproteus import acoustic, medium
from pyproteus.config.models import Domain, Geometry, SimulationParameters
from tests.oracle.helpers import ORACLE_SETTINGS, put
from tests.oracle.test_acoustic_grid import matlab_value

THRESHOLD = 0.999


def _grid(eng: Any, dx: float, dom: dict[str, float]) -> acoustic.Grid:
    put(eng, **{f"pyp_{k}__": float(v) for k, v in dom.items()}, pyp_dx__=dx)
    fields = ",".join(f"'{k}',pyp_{k}__" for k in dom)
    eng.eval(
        "pyp_SP__ = struct('GridSize', pyp_dx__, 'SamplingRate', 30e6, 'SensorOnGrid', false);"
        f"pyp_Grid__ = define_grid(pyp_SP__, struct('Domain', struct({fields})));",
        nargout=0,
    )
    return acoustic.define_grid(
        SimulationParameters(GridSize=dx, SamplingRate=30e6, SensorOnGrid=False),
        Geometry(Domain=Domain(**dom)),
    )


def _compare(eng: Any, grid: acoustic.Grid, pts: np.ndarray, tmp: Path, label: str) -> None:
    put(eng, pyp_P__=pts.reshape(-1, 3) if pts.size else np.zeros((0, 3)))
    eng.eval(
        "[pyp_o1__, pyp_o2__, pyp_o3__, pyp_o4__] = voxelize_media_points(pyp_P__, pyp_Grid__);",
        nargout=0,
    )
    ref = matlab_value(
        eng,
        "struct('MP', pyp_o1__, 'nodes', pyp_o2__, 'idx', pyp_o3__, 'excl', pyp_o4__)",
        tmp,
    )
    out = medium.voxelize_media_points(pts, grid)
    np.testing.assert_array_equal(out.exclude, ref["excl"].reshape(-1), err_msg=label)
    np.testing.assert_array_equal(out.points, ref["MP"].reshape(-1, 3), err_msg=label)
    n = out.points.shape[0]
    ref_nodes = ref["nodes"].reshape(-1, 3) - 1 if n else np.zeros((0, 3))
    ref_idx = ref["idx"].reshape(-1) - 1 if n else np.zeros(0)
    assert out.nodes.shape == (n, 3) and out.index.shape == (n,)
    if n == 0:
        return
    same = np.all(out.nodes == ref_nodes, axis=1) & (out.index == ref_idx)
    frac = same.mean()
    if not same.all():
        bad = np.flatnonzero(~same)[:10]
        print(f"{label}: {np.count_nonzero(~same)} of {n} voxels differ, e.g. rows {bad}")
    assert frac >= THRESHOLD, f"{label}: only {frac:.4%} of voxel indices equal"


DOMAIN = dict(Xmin=1e-3, Xmax=6e-3, Ymin=-2e-3, Ymax=2.5e-3, Zmin=-1e-3, Zmax=1.5e-3)


def test_tie_points(matlab_engine: Any, tmp_path: Path) -> None:
    eng = matlab_engine
    grid = _grid(eng, 1e-4, DOMAIN)
    lo = np.array([grid.x[0], grid.y[0], grid.z[0]])
    hi = np.array([grid.x[-1], grid.y[-1], grid.z[-1]])
    rs = np.random.RandomState(0)
    k = rs.randint(0, 20, size=(200, 3))
    halves = lo + (k + 0.5) * grid.dx  # exact .5 ties (up to rounding of the products)
    on_nodes = lo + k * grid.dx
    boundary = np.vstack([lo, hi, [lo[0], hi[1], lo[2]], hi - 0.5 * grid.dx])
    outside = np.vstack(
        [lo - 1e-9, hi + 1e-9, [lo[0] - grid.dx, 0, 0], [0, hi[1] * 1.01, 0], [0, 0, lo[2] - 1]]
    )
    pts = np.vstack(
        [halves, on_nodes, boundary, outside, rs.uniform(lo - 2e-4, hi + 2e-4, (500, 3))]
    )
    _compare(eng, grid, pts, tmp_path, "ties")


@pytest.mark.parametrize("n", [0, 1])
def test_empty_and_single(matlab_engine: Any, tmp_path: Path, n: int) -> None:
    grid = _grid(matlab_engine, 1.3e-4, DOMAIN)
    pts = np.array([[3e-3, 0.1e-3, 0.2e-3]])[:n]
    _compare(matlab_engine, grid, pts, tmp_path, f"n={n}")


def test_all_outside(matlab_engine: Any, tmp_path: Path) -> None:
    grid = _grid(matlab_engine, 1.3e-4, DOMAIN)
    _compare(matlab_engine, grid, np.array([[0.0, 0, 0], [1.0, 0, 0]]), tmp_path, "outside")


@ORACLE_SETTINGS
@given(st.integers(0, 2**31 - 1), st.floats(5e-5, 3e-4), st.integers(1, 300))
def test_random(matlab_engine: Any, seed: int, dx: float, n: int) -> None:
    import tempfile

    rs = np.random.RandomState(seed)
    lo = rs.uniform(-3e-3, 3e-3, 3)
    dom = dict(zip(("Xmin", "Ymin", "Zmin"), lo, strict=True))
    dom.update(zip(("Xmax", "Ymax", "Zmax"), lo + rs.uniform(4 * dx, 4e-3, 3), strict=True))
    order = ["Xmin", "Xmax", "Ymin", "Ymax", "Zmin", "Zmax"]
    grid = _grid(matlab_engine, dx, {k: dom[k] for k in order})
    a = np.array([grid.x[0], grid.y[0], grid.z[0]])
    b = np.array([grid.x[-1], grid.y[-1], grid.z[-1]])
    pts = rs.uniform(a - dx, b + dx, (n, 3))
    m = n // 3
    pts[:m] = a + (rs.randint(0, 5, (m, 3)) + 0.5) * dx  # ties
    with tempfile.TemporaryDirectory() as tmp:
        _compare(matlab_engine, grid, pts, Path(tmp), f"seed {seed}")
