"""Oracle tests: ``pyproteus.geometry`` vs ``READ_stl``/``VOXELISE``/``define_medium``.

Class D-13 "integers after rounding" for masks, with the 99.9 % measured
against the voxels inside either mask rather than the whole grid (a vessel is
< 0.1 % of the grid, so a whole-grid rate would accept any vessel): at most
0.1 % of them differ, every difference reported and required to sit on the
surface (a differing voxel touches both inside and outside voxels of MATLAB's
mask). Measured: 100 % on every case.
The port uses the same ray/facet formulas as ``VOXELISE.m``, so the synthetic
cases are expected to agree exactly and are asserted exactly. ``READ_stl`` is
exact (data read from files).

Shipped geometries (renal tree, mouse brain; Zenodo 10573662) are not in the
oracle checkout: set ``PYPROTEUS_GEOMETRIES_PATH`` to a folder holding
``renal_tree/maa.stl`` and ``mouse_brain/lpa_mouse_t10.stl`` (the
``GeometryProperties`` come from the oracle's ``geometry_data``). Those tests
skip otherwise.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy import ndimage

from pyproteus.geometry import (
    EmptyVesselWarning,
    place_vessel_mesh,
    read_stl,
    vessel_mask,
    voxelise,
)
from tests.oracle.helpers import from_matlab, put, row, to_matlab

SAMPLE_STL = Path("mesh-voxelisation/Mesh_voxelisation/sample.stl")
SHIPPED = {"renal_tree": "maa.stl", "mouse_brain": "lpa_mouse_t10.stl"}
MIN_EQUAL = 0.999  # D-13 default for integer classes


# ---------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def eng(matlab_engine: Any, oracle: Any) -> Any:
    """Engine with the oracle's ``tests/helpers`` (``make_default_settings``) on the path."""
    helpers = str(oracle.path / "tests" / "helpers")
    matlab_engine.addpath(helpers, nargout=0)
    yield matlab_engine
    matlab_engine.rmpath(helpers, nargout=0)


@pytest.fixture(scope="module")
def geometries_path(oracle: Any, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Oracle ``geometry_data`` plus the shipped STLs, in a scratch folder."""
    src = os.environ.get("PYPROTEUS_GEOMETRIES_PATH")
    if not src:
        pytest.skip("PYPROTEUS_GEOMETRIES_PATH not set (shipped STLs: Zenodo 10573662)")
    root = tmp_path_factory.mktemp("geometry_data")
    for folder, stl in SHIPPED.items():
        stl_file = Path(src) / folder / stl
        if not stl_file.is_file():
            pytest.skip(f"{stl_file} not found")
        (root / folder).mkdir()
        shutil.copy(
            oracle.path / "geometry_data" / folder / "GeometryProperties.mat", root / folder
        )
        (root / folder / stl).symlink_to(stl_file.resolve())
    return root


# ---------------------------------------------------------------- helpers


def matlab_voxelise(
    eng: Any, gx: np.ndarray, gy: np.ndarray, gz: np.ndarray, mesh_expr: str, ray: str = "xyz"
) -> np.ndarray:
    """``VOXELISE(gx, gy, gz, <mesh_expr>, ray)``; the mask comes back as indices."""
    put(eng, pyp_gx=row(gx), pyp_gy=row(gy), pyp_gz=row(gz))
    eng.eval(
        f"pyp_idx = uint32(find(VOXELISE(pyp_gx, pyp_gy, pyp_gz, {mesh_expr}, '{ray}')));",
        nargout=0,
    )
    return _from_indices(eng, (gx.size, gy.size, gz.size))


def _from_indices(eng: Any, shape: tuple[int, ...]) -> np.ndarray:
    idx = from_matlab(eng.workspace["pyp_idx"]).reshape(-1).astype(np.int64) - 1
    flat = np.zeros(int(np.prod(shape)), bool)
    flat[idx] = True
    return flat.reshape(shape, order="F")


def compare_masks(
    actual: np.ndarray, expected: np.ndarray, label: str, exact: bool = False
) -> None:
    """D-13 integer class with the differences reported and checked to be on the surface."""
    assert actual.dtype == bool and actual.shape == expected.shape, label
    diff = actual != expected
    n = int(diff.sum())
    frac = 1 - n / diff.size
    print(
        f"{label}: {expected.sum()} voxels inside (MATLAB), {actual.sum()} (Python), "
        f"{n} differ ({frac:.6%} equal) of {diff.size}"
    )
    if exact:
        assert n == 0, f"{label}: {n} voxels differ"
        return
    # D-13 integer class, measured against the voxels inside either mask (not the
    # whole grid: thin vessels are a tiny fraction of it and entirely surface).
    union = int((actual | expected).sum())
    assert n <= (1 - MIN_EQUAL) * union, f"{label}: {n} of {union} inside voxels differ"
    if n:
        k = np.ones((3, 3, 3), bool)
        near_in = ndimage.binary_dilation(expected, k)
        near_out = ndimage.binary_dilation(~expected, k)
        off_surface = diff & ~(near_in & near_out)
        assert not off_surface.any(), f"{label}: {off_surface.sum()} differences off the surface"


def odd_grid(
    lo: np.ndarray, hi: np.ndarray, counts: tuple[int, int, int], margin: float = 0.07
) -> tuple[np.ndarray, ...]:
    """Non-cubic grid with odd counts, unequal spacings and an unaligned origin."""
    out = []
    for a, b, n, shift in zip(lo, hi, counts, (0.37, 0.61, 0.13), strict=True):
        span = (b - a) * (1 + 2 * margin)
        d = span / (n - 1)
        out.append(a - margin * (b - a) + shift * d + d * np.arange(n))
    return tuple(out)


def box_mesh(lo: np.ndarray, hi: np.ndarray, skip_face: int | None = None) -> np.ndarray:
    """Closed, outward-oriented box as (N, 3, 3) facets; ``skip_face`` opens one side."""
    c = np.array(
        [
            [lo[0], lo[1], lo[2]],
            [hi[0], lo[1], lo[2]],
            [hi[0], hi[1], lo[2]],
            [lo[0], hi[1], lo[2]],
            [lo[0], lo[1], hi[2]],
            [hi[0], lo[1], hi[2]],
            [hi[0], hi[1], hi[2]],
            [lo[0], hi[1], hi[2]],
        ]
    )
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 4, 7, 3)]
    tris = []
    for q, (a, b, cc, d) in enumerate(quads):
        if q != skip_face:
            tris += [(a, b, cc), (a, cc, d)]
    return np.stack([c[list(t)].T for t in tris])


def pyramid_apex_on_ray() -> np.ndarray:
    """Square pyramid, apex (5, 5, 7.7) on a grid ray: its 4 side facets all face +z, so
    the vertex counts as one crossing (vertex branch, same-sign normals)."""
    b = np.array([[2.3, 2.6, 1.4], [7.6, 2.6, 1.4], [7.6, 7.3, 1.4], [2.3, 7.3, 1.4]])
    a = np.array([5.0, 5.0, 7.7])
    tris = [(b[0], b[2], b[1]), (b[0], b[3], b[2])] + [(b[i], b[(i + 1) % 4], a) for i in range(4)]
    return np.stack([np.array(t).T for t in tris])


def _quad(p0: Any, p1: Any, p2: Any, p3: Any) -> list[np.ndarray]:
    return [np.array([p0, p1, p2], float).T, np.array([p0, p2, p3], float).T]


def box_with_holed_top(
    lo: np.ndarray, hi: np.ndarray, xs: list[float], ys: list[float], hole: tuple[int, int]
) -> np.ndarray:
    """Box without its top face, the top tessellated on the ``xs`` x ``ys`` cells minus
    ``hole``: rays through the hole cross once (odd) and are filled from neighbours."""
    c = np.array(
        [
            [lo[0], lo[1], lo[2]],
            [hi[0], lo[1], lo[2]],
            [hi[0], hi[1], lo[2]],
            [lo[0], hi[1], lo[2]],
            [lo[0], lo[1], hi[2]],
            [hi[0], lo[1], hi[2]],
            [hi[0], hi[1], hi[2]],
            [lo[0], hi[1], hi[2]],
        ]
    )
    tris = []
    for a, b, cc, d in [(0, 3, 2, 1), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 4, 7, 3)]:
        tris += _quad(c[a], c[b], c[cc], c[d])
    z = hi[2]
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            if (i, j) != hole:
                tris += _quad(
                    [xs[i], ys[j], z],
                    [xs[i + 1], ys[j], z],
                    [xs[i + 1], ys[j + 1], z],
                    [xs[i], ys[j + 1], z],
                )
    return np.stack(tris)


def octahedron(center: Any, r: float, h: float) -> list[np.ndarray]:
    cx, cy, cz = center
    eq = [(cx - r, cy, cz), (cx, cy - r, cz), (cx + r, cy, cz), (cx, cy + r, cz)]
    top, bottom = (cx, cy, cz + h), (cx, cy, cz - h)
    tris = []
    for k in range(4):
        a, b = eq[k], eq[(k + 1) % 4]
        tris += [np.array([a, b, top], float).T, np.array([b, a, bottom], float).T]
    return tris


def _matlab_geometry(eng: Any, folder: str, geometries_path: Path) -> dict[str, Any]:
    """Default settings and grid as MainGUI + write_to_file + define_grid build them."""
    eng.eval(
        f"pyp_S = make_default_settings(struct('GeometriesPath','{geometries_path}'),'{folder}');"
        "pyp_S.SimulationParameters.GridSize = pyp_S.Medium.SpeedOfSound/"
        "(pyp_S.Transmit.CenterFrequency*pyp_S.SimulationParameters.PointsPerWavelength);"
        "pyp_Grid = define_grid(pyp_S.SimulationParameters, pyp_S.Geometry);"
        f"pyp_S.Geometry.GeometriesPath = '{geometries_path}';"
        "pyp_G = pyp_S.Geometry;",
        nargout=0,
    )
    G = eng.workspace["pyp_G"]
    return {
        "Folder": folder,
        "STLfile": G["STLfile"],
        "STLunit": float(G["STLunit"]),
        "EmbedVessel": bool(G["EmbedVessel"]),
        "Rotation": from_matlab(G["Rotation"]),
        "Center": from_matlab(G["Center"]),
        "BoundingBox": {"Center": from_matlab(G["BoundingBox"]["Center"])},
        "grid": tuple(from_matlab(eng.eval(f"pyp_Grid.{a}", nargout=1)).reshape(-1) for a in "xyz"),
    }


_PLACE_IN_MATLAB = (
    "pyp_V = READ_stl(fullfile(pyp_G.GeometriesPath, pyp_G.Folder, pyp_G.STLfile));"
    "pyp_V = pyp_V*pyp_G.STLunit - transpose(pyp_G.BoundingBox.Center);"
    "for pyp_i = 1:3,"
    " pyp_V(:,:,pyp_i) = transpose(pyp_G.Rotation*transpose(pyp_V(:,:,pyp_i))); end;"
    "pyp_V = pyp_V + transpose(pyp_G.Center);"
)


# ---------------------------------------------------------------- READ_stl


def test_read_stl_sample_ascii(eng: Any, oracle: Any) -> None:
    path = oracle.path / SAMPLE_STL
    eng.eval(f"[pyp_V, pyp_N] = READ_stl('{path}');", nargout=0)
    V, N = read_stl(path, return_normals=True)
    np.testing.assert_array_equal(V, from_matlab(eng.workspace["pyp_V"]))
    np.testing.assert_array_equal(N, from_matlab(eng.workspace["pyp_N"]))


def test_read_stl_binary_roundtrip(eng: Any, oracle: Any, tmp_path: Path) -> None:
    """Binary file whose header starts with 'solid' (format detection by the last 80 bytes)."""
    V = read_stl(oracle.path / SAMPLE_STL).astype(np.float32)
    rec = np.zeros(V.shape[0], dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"] = np.arange(V.shape[0] * 3, dtype=np.float32).reshape(-1, 3)
    rec["v"] = V.transpose(0, 2, 1)
    path = tmp_path / "solid_header.stl"
    path.write_bytes(
        b"solid but binary".ljust(80, b" ") + np.int32(V.shape[0]).tobytes() + rec.tobytes()
    )
    eng.eval(f"[pyp_V, pyp_N] = READ_stl('{path}');", nargout=0)
    Vp, Np = read_stl(path, return_normals=True)
    np.testing.assert_array_equal(Vp, from_matlab(eng.workspace["pyp_V"]))
    np.testing.assert_array_equal(Np, from_matlab(eng.workspace["pyp_N"]))


@pytest.mark.parametrize("folder", SHIPPED)
def test_read_stl_shipped(eng: Any, geometries_path: Path, folder: str) -> None:
    path = geometries_path / folder / SHIPPED[folder]
    eng.eval(f"pyp_V = READ_stl('{path}'); pyp_sz = size(pyp_V);", nargout=0)
    V = read_stl(path)
    assert tuple(from_matlab(eng.workspace["pyp_sz"]).astype(int).ravel()) == V.shape
    # compare in slices (the mouse brain mesh is ~62 MB of doubles)
    for s in range(0, V.shape[0], 200_000):
        e = min(s + 200_000, V.shape[0])
        got = from_matlab(eng.eval(f"pyp_V({s + 1}:{e},:,:)", nargout=1))
        np.testing.assert_array_equal(V[s:e], got.reshape(e - s, 3, 3))


# ---------------------------------------------------------------- VOXELISE, synthetic


def test_sample_stl_odd_grid_all_directions(eng: Any, oracle: Any) -> None:
    path = oracle.path / SAMPLE_STL
    mesh = read_stl(path)
    gx, gy, gz = odd_grid(mesh.min(axis=(0, 2)), mesh.max(axis=(0, 2)), (37, 23, 51))
    eng.eval(f"pyp_M = READ_stl('{path}');", nargout=0)
    for ray in ("xyz", "zyx", "x", "y", "z", "xy", "zx"):
        compare_masks(
            voxelise(gx, gy, gz, mesh, ray),
            matlab_voxelise(eng, gx, gy, gz, "pyp_M", ray),
            f"sample.stl {ray}",
            exact=True,
        )


def test_vertices_on_rays_and_open_mesh(eng: Any) -> None:
    """Box vertices exactly on grid rays (vertex branch, mixed normals -> correction),
    a pyramid apex on a ray (vertex branch, crossing counted), an open box (odd crossings,
    neighbour corrections, also at the grid border)."""
    g = np.arange(11, dtype=float)
    cases = {
        "closed box, vertices on rays": box_mesh(
            np.array([2.0, 3.0, 1.0]), np.array([7.0, 8.0, 6.0])
        ),
        # bounds off the .5 ties (B-90), unequal sides
        "open box": box_mesh(np.array([2.4, 3.7, 1.2]), np.array([7.3, 8.6, 6.9]), skip_face=4),
        "open box at the border": box_mesh(
            np.array([0.0, 2.3, 1.6]), np.array([5.4, 7.7, 6.2]), skip_face=5
        ),
        # faces' diagonals through grid rays: edge hits, odd crossings, corrections
        "cube, diagonals on rays": box_mesh(np.array([1.2, 2.2, 3.2]), np.array([6.2, 7.2, 8.2])),
        "pyramid, apex on a ray": pyramid_apex_on_ray(),
    }
    R = np.array([[0.8, -0.6, 0.0], [0.6, 0.8, 0.0], [0.0, 0.0, 1.0]])
    cases["tilted box"] = np.einsum("ij,njv->niv", R, cases["closed box, vertices on rays"] - 5) + 5
    for label, mesh in cases.items():
        put(eng, pyp_M=mesh)
        for ray in ("xyz", "z"):
            compare_masks(
                voxelise(g, g, g, mesh, ray),
                matlab_voxelise(eng, g, g, g, "pyp_M", ray),
                f"{label} {ray}",
                exact=True,
            )


def test_correction_order_border_and_vertex_groups(eng: Any) -> None:
    """Cases where the correction order (sequential, y outer / x inner), the zero
    padding at the grid border and the vertex grouping change the result."""
    g = np.arange(11, dtype=float)
    lo, hi = np.array([1.3, 1.4, 1.2]), np.array([8.7, 8.6, 6.8])
    cases = {
        # hole in the middle of the top: corrected rays feed later corrections
        "holed top": box_with_holed_top(
            lo, hi, [1.3, 3.45, 6.55, 8.7], [1.4, 3.35, 6.6, 8.6], (1, 1)
        ),
        # box beyond the grid in y, hole at the y = 0 edge: corrections use padding
        "holed top at the border": box_with_holed_top(
            np.array([1.3, -0.6, 1.2]),
            np.array([8.7, 8.6, 6.8]),
            [1.3, 3.45, 6.55, 8.7],
            [-0.6, 3.35, 6.6, 8.6],
            (1, 0),
        ),
        # one-ray hole at the grid corner: 3 in-grid neighbours inside, the 5 outside
        # the grid count as empty (MATLAB's zero border), so the ray stays empty
        "holed top at the corner": box_with_holed_top(
            np.array([-0.6, -0.6, 1.2]),
            np.array([8.7, 8.6, 6.8]),
            [-0.6, 0.5, 4.45, 8.7],
            [-0.6, 0.5, 4.35, 8.6],
            (0, 0),
        ),
        # ray (5, 5) touches two octahedra at an equator vertex: facets around each
        # vertex face +z and -z, so the ray must be corrected, not counted twice
        "vertex on silhouettes": np.stack(
            octahedron((7.2, 5.0, 2.3), 2.2, 1.7) + octahedron((7.2, 5.0, 6.4), 2.2, 1.7)
        ),
        # degenerate (zero-area, NaN normal) facet at a vertex on a ray
        "degenerate facet at a vertex": np.concatenate(
            [
                pyramid_apex_on_ray(),
                np.array([[5.0, 5.0, 7.7], [5.0, 5.0, 7.7], [2.3, 2.6, 1.4]]).T[None],
            ]
        ),
    }
    for label, mesh in cases.items():
        put(eng, pyp_M=mesh)
        for ray in ("z", "xyz"):
            compare_masks(
                voxelise(g, g, g, mesh, ray),
                matlab_voxelise(eng, g, g, g, "pyp_M", ray),
                f"{label} {ray}",
                exact=True,
            )


def test_mesh_larger_than_grid(eng: Any) -> None:
    """The grid is extended along the first traced axis needing it, then cropped (B-110)."""
    mesh = box_mesh(np.array([-1.3, 1.2, 2.1]), np.array([12.4, 7.7, 13.9]))
    gx = np.linspace(0, 10, 13)
    gy = np.linspace(0, 9, 17)
    gz = np.linspace(0, 11, 9)
    put(eng, pyp_M=mesh)
    for ray in ("xyz", "yz", "z"):
        compare_masks(
            voxelise(gx, gy, gz, mesh, ray),
            matlab_voxelise(eng, gx, gy, gz, "pyp_M", ray),
            f"mesh beyond grid {ray}",
            exact=True,
        )


def test_mesh_bound_halfway_between_grid_lines(eng: Any) -> None:
    """B-90: MATLAB fails when a mesh bound is equidistant from two grid lines; Python
    returns the box (D-15). Python matches MATLAB once the tie is broken by 1e-9."""
    g = np.arange(12, dtype=float)
    lo, hi = np.array([2.5, 3.2, 1.3]), np.array([7.4, 8.6, 6.8])  # x bound 2.5: tie
    mesh = box_mesh(lo, hi)
    put(eng, pyp_M=mesh)
    with pytest.raises(Exception, match=r"(?i)colon|scalar"):
        matlab_voxelise(eng, g, g, g, "pyp_M")
    ours = voxelise(g, g, g, mesh)
    inside = [(g > a) & (g < b) for a, b in zip(lo, hi, strict=True)]
    np.testing.assert_array_equal(
        ours, inside[0][:, None, None] & inside[1][None, :, None] & inside[2][None, None, :]
    )
    shifted = box_mesh(lo + np.array([1e-9, 0, 0]), hi)
    put(eng, pyp_M=shifted)
    compare_masks(ours, matlab_voxelise(eng, g, g, g, "pyp_M"), "tie broken by 1e-9", exact=True)


# ---------------------------------------------------------------- shipped geometries


@pytest.mark.parametrize("folder", SHIPPED)
def test_shipped_geometry_default_grid(eng: Any, geometries_path: Path, folder: str) -> None:
    """Default settings and grid: the mask define_medium computes."""
    G = _matlab_geometry(eng, folder, geometries_path)
    gx, gy, gz = G["grid"]
    eng.eval(_PLACE_IN_MATLAB, nargout=0)
    expected = matlab_voxelise(eng, gx, gy, gz, "pyp_V")
    if expected.any():
        actual = vessel_mask(gx, gy, gz, G, geometries_path)
    else:  # mouse brain: vessel thinner than dx (B-112) -> warning (D-15)
        with pytest.warns(EmptyVesselWarning, match="thinner than the grid spacing"):
            actual = vessel_mask(gx, gy, gz, G, geometries_path)
    compare_masks(actual, expected, f"{folder} default grid {actual.shape}")


@pytest.mark.parametrize(
    ("folder", "counts"), [("renal_tree", (151, 113, 207)), ("mouse_brain", (101, 113, 389))]
)
def test_shipped_geometry_fine_odd_grid(
    eng: Any, geometries_path: Path, folder: str, counts: tuple[int, int, int]
) -> None:
    """Grid resolving the vessels (the default grid is coarser than the mouse brain
    arteriole: its mask is empty), odd non-cubic sizes, unaligned origin."""
    G = _matlab_geometry(eng, folder, geometries_path)
    eng.eval(_PLACE_IN_MATLAB, nargout=0)
    mesh = place_vessel_mesh(read_stl(geometries_path / folder / SHIPPED[folder]), G)
    gx, gy, gz = odd_grid(mesh.min(axis=(0, 2)), mesh.max(axis=(0, 2)), counts)
    expected = matlab_voxelise(eng, gx, gy, gz, "pyp_V")
    assert expected.sum() > 1000  # the case must exercise the voxeliser
    compare_masks(voxelise(gx, gy, gz, mesh), expected, f"{folder} fine grid {counts}")


def test_define_medium_downstream(eng: Any, geometries_path: Path) -> None:
    """MATLAB define_medium with the vessel vs MATLAB background + the Python mask.

    Same rng seed for both MATLAB calls; the Python mask replaces VOXELISE's.
    sound_speed, density and BonA must agree at the float class (D-13) on the
    voxels where the masks agree, and the masks at the integer class.
    """
    G = _matlab_geometry(eng, "renal_tree", geometries_path)
    gx, gy, gz = G["grid"]
    mask = vessel_mask(gx, gy, gz, G, geometries_path)
    eng.workspace["pyp_pyidx"] = to_matlab(
        np.flatnonzero(mask.ravel(order="F")).astype(np.uint32)[:, None] + 1
    )
    eng.eval(
        "pyp_rng = rng; rng(7,'twister');"
        "[pyp_m, pyp_v] = define_medium(pyp_Grid, pyp_S.Medium, pyp_G);"
        "pyp_G0 = pyp_G; pyp_G0.EmbedVessel = false; rng(7,'twister');"
        "pyp_m0 = define_medium(pyp_Grid, pyp_S.Medium, pyp_G0); rng(pyp_rng);"
        "pyp_mask = false(size(pyp_v)); pyp_mask(pyp_pyidx) = true;"
        "pyp_m0.sound_speed(pyp_mask) = pyp_S.Medium.Vessel.SpeedOfSound;"
        "pyp_m0.density(pyp_mask) = pyp_S.Medium.Vessel.Density;"
        "pyp_m0.BonA(pyp_mask) = pyp_S.Medium.Vessel.BonA;"
        "pyp_ok = pyp_v == pyp_mask;"
        "pyp_rel = @(a, b) max(abs(a(pyp_ok) - b(pyp_ok)) ./ abs(a(pyp_ok)));"
        "pyp_err = [pyp_rel(pyp_m.sound_speed, pyp_m0.sound_speed),"
        " pyp_rel(pyp_m.density, pyp_m0.density), pyp_rel(pyp_m.BonA, pyp_m0.BonA)];"
        "pyp_vdiff = nnz(pyp_v ~= pyp_mask); pyp_nv = nnz(pyp_v);"
        "clear pyp_m pyp_m0 pyp_v pyp_mask pyp_ok pyp_rel;",
        nargout=0,
    )
    err = from_matlab(eng.workspace["pyp_err"]).ravel()
    vdiff = int(eng.workspace["pyp_vdiff"])
    print(
        f"define_medium renal_tree default grid: {int(eng.workspace['pyp_nv'])} vessel voxels, "
        f"{vdiff} differ; max rel err c, rho, BonA = {err}"
    )
    # integer class on the mask (relative to the vessel), float class elsewhere
    nv = int(eng.workspace["pyp_nv"])
    assert nv > 1000 and mask.sum() > 1000
    assert vdiff <= (1 - MIN_EQUAL) * nv
    assert np.all(err <= 1e-9), err


def python_default_pipeline(folder: str, geometries_path: Path, rng_seed: int = 1):
    """Default settings to medium in Python only, as MainGUI + write_to_file + define_grid +
    define_medium (``update_derived_settings``: Margin, compute_simulation_domain)."""
    from pyproteus import acoustic, medium
    from pyproteus import numerics as nm
    from pyproteus.config.defaults import default_settings
    from pyproteus.transducer.domain import compute_simulation_domain

    S = default_settings(folder)
    c, f0 = float(S.Medium.SpeedOfSound), float(S.Transmit.CenterFrequency)
    S.Geometry.Domain.Margin = 2 * c / f0
    S.Geometry = compute_simulation_domain(S.Geometry, S.Transducer, S.Transmit)
    sp = S.SimulationParameters
    sp.GridSize = c / (f0 * float(sp.PointsPerWavelength))
    grid = acoustic.define_grid(sp, S.Geometry)
    kw, vessel = medium.define_medium(
        grid,
        S.Medium,
        S.Geometry,
        S.Transducer,
        rng=nm.rng(rng_seed),
        geometries_path=geometries_path,
    )
    return S, grid, kw, vessel


def test_default_pipeline_end_to_end_renal_tree(eng: Any, geometries_path: Path) -> None:
    """Default settings run end to end in Python (settings, domain, grid, define_medium
    reading and voxelising the STL); the vessel equals MATLAB's define_medium output.

    Python's grid keeps the last domain point MATLAB's odd PML padding drops (D-19), so
    MATLAB's grid is Python's without its last index per axis; the vessel is far from it.
    """
    S, grid, kw, vessel = python_default_pipeline("renal_tree", geometries_path)
    _matlab_geometry(eng, "renal_tree", geometries_path)
    eng.eval(
        "rng(1,'twister'); [~, pyp_v] = define_medium(pyp_Grid, pyp_S.Medium, pyp_G);"
        "pyp_idx = uint32(find(pyp_v)); pyp_sz = size(pyp_v); clear pyp_v;",
        nargout=0,
    )
    shape = tuple(from_matlab(eng.workspace["pyp_sz"]).astype(int).ravel())
    expected = _from_indices(eng, shape)
    gm = [from_matlab(eng.eval(f"pyp_Grid.{a}", nargout=1)).reshape(-1) for a in "xyz"]
    common = tuple(slice(0, n) for n in shape)
    for a, m in zip((grid.x, grid.y, grid.z), gm, strict=True):
        np.testing.assert_allclose(np.asarray(a).reshape(-1)[: m.size], m, rtol=0, atol=1e-15)
    outside = vessel.copy()
    outside[common] = False
    assert not outside.any()
    compare_masks(vessel[common], expected, "default pipeline renal_tree (Python vs MATLAB)")
    assert expected.sum() > 1000
    vc = float(S.Medium.Vessel.SpeedOfSound)
    np.testing.assert_array_equal(kw["sound_speed"][vessel], vc)
