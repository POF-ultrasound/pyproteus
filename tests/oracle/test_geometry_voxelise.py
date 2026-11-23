"""Oracle tests: ``pyproteus.geometry`` vs ``READ_stl``/``VOXELISE``/``define_medium``.

Class D-13 "integers after rounding" for masks: at least 99.9 % of voxels
equal, every difference reported and required to sit on the surface (a
differing voxel touches both inside and outside voxels of MATLAB's mask).
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

from pyproteus.geometry import place_vessel_mesh, read_stl, vessel_mask, voxelise
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
    assert frac >= MIN_EQUAL, label
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
    # integer class on the mask, float class on every voxel where the masks agree
    assert vdiff <= (1 - MIN_EQUAL) * mask.size
    assert np.all(err <= 1e-9), err
