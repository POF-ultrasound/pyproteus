"""Unit tests for pyproteus.geometry (no MATLAB): STL I/O, voxelisation, placement."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyproteus.geometry import (
    identify_stl_format,
    place_vessel_mesh,
    read_stl,
    stl_path,
    vessel_mask,
    voxelise,
)
from pyproteus.geometry import voxelisation as voxelise_module


def box_mesh(lo, hi) -> np.ndarray:
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
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
    tris = [t for a, b, cc, d in quads for t in ((a, b, cc), (a, cc, d))]
    return np.stack([c[list(t)].T for t in tris])


def icosphere(radius: float, center, levels: int = 3) -> np.ndarray:
    t = (1 + 5**0.5) / 2
    v = [
        (-1, t, 0),
        (1, t, 0),
        (-1, -t, 0),
        (1, -t, 0),
        (0, -1, t),
        (0, 1, t),
        (0, -1, -t),
        (0, 1, -t),
        (t, 0, -1),
        (t, 0, 1),
        (-t, 0, -1),
        (-t, 0, 1),
    ]
    f = [
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        (4, 9, 5),
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1),
    ]
    tri = np.array([[v[i] for i in face] for face in f], float)
    for _ in range(levels):
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        tri = np.concatenate(
            [np.stack(x, 1) for x in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))]
        )
    tri = tri / np.linalg.norm(tri, axis=2, keepdims=True) * radius + np.asarray(center)
    return tri.transpose(0, 2, 1)


def write_binary(path: Path, mesh: np.ndarray, header: bytes = b"binary") -> None:
    rec = np.zeros(mesh.shape[0], dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["v"] = mesh.transpose(0, 2, 1)
    path.write_bytes(header.ljust(80, b" ") + np.int32(mesh.shape[0]).tobytes() + rec.tobytes())


def write_ascii(path: Path, mesh: np.ndarray) -> None:
    lines = ["solid test"]
    for tri in mesh:
        lines += ["facet normal 0 0 1", "  outer loop"]
        lines += [f"    vertex {float(x)!r} {float(y)!r} {float(z)!r}" for x, y, z in tri.T]
        lines += ["  endloop", "endfacet"]
    path.write_text("\n\n".join(lines) + "\nendsolid test\n")


# ---------------------------------------------------------------- STL


def test_read_binary_and_ascii(tmp_path: Path) -> None:
    mesh = box_mesh([0.5, -1.25, 2], [3, 1.5, 4.75])
    write_binary(tmp_path / "b.stl", mesh)
    write_ascii(tmp_path / "a.stl", mesh)
    assert identify_stl_format(tmp_path / "b.stl") == "binary"
    assert identify_stl_format(tmp_path / "a.stl") == "ascii"
    for name in ("a.stl", "b.stl"):
        got = read_stl(tmp_path / name)
        assert got.dtype == np.float64 and got.shape == (12, 3, 3)
        np.testing.assert_array_equal(got, mesh)
    _, normals = read_stl(tmp_path / "a.stl", return_normals=True)
    np.testing.assert_array_equal(normals, np.tile([0.0, 0.0, 1.0], (12, 1)))


def test_binary_with_solid_header_is_binary(tmp_path: Path) -> None:
    mesh = box_mesh([0, 0, 0], [1, 2, 3])
    write_binary(tmp_path / "s.stl", mesh, header=b"solid looks like ascii")
    assert identify_stl_format(tmp_path / "s.stl") == "binary"
    np.testing.assert_array_equal(read_stl(tmp_path / "s.stl"), mesh)
    np.testing.assert_array_equal(read_stl(tmp_path / "s.stl", "binary"), mesh)


def test_float32_values_widen_exactly(tmp_path: Path) -> None:
    mesh = box_mesh([0.1, 0.2, 0.3], [1.1, 1.7, 2.3])
    write_binary(tmp_path / "f.stl", mesh)
    np.testing.assert_array_equal(
        read_stl(tmp_path / "f.stl"), mesh.astype(np.float32).astype(np.float64)
    )


# ---------------------------------------------------------------- voxelise


def test_box_mask_orientation_and_strict_interior() -> None:
    """Asymmetric box on a non-cubic grid: mask[i, j, k] is (x[i], y[j], z[k]); grid points
    exactly on a face are outside (strict inequality between crossings)."""
    gx, gy, gz = np.arange(9.0), np.arange(0.0, 13.0, 1.0) * 0.5, np.linspace(-1, 4, 7)
    lo, hi = np.array([1.5, 1.0, -0.2]), np.array([6.2, 4.3, 2.4])
    want = (
        ((gx > lo[0]) & (gx < hi[0]))[:, None, None]
        & ((gy > lo[1]) & (gy < hi[1]))[None, :, None]
        & ((gz > lo[2]) & (gz < hi[2]))[None, None, :]
    )
    for ray in ("xyz", "x", "y", "z", "yz"):
        got = voxelise(gx, gy, gz, box_mesh(lo, hi), ray)
        assert got.dtype == bool and got.shape == (9, 13, 7)
        np.testing.assert_array_equal(got, want, err_msg=ray)


def test_sphere_volume_and_symmetry() -> None:
    g = np.linspace(-1.2, 1.2, 49)
    mask = voxelise(g, g, g, icosphere(1.0, [0, 0, 0], levels=4))
    vol = mask.sum() * (g[1] - g[0]) ** 3
    assert abs(vol - 4 / 3 * np.pi) / (4 / 3 * np.pi) < 0.03
    np.testing.assert_array_equal(mask, mask[::-1, ::-1, ::-1])


def test_chunking_does_not_change_result(monkeypatch: pytest.MonkeyPatch) -> None:
    g = np.linspace(-1.1, 1.3, 41)
    mesh = icosphere(1.0, [0.1, 0.05, -0.1], levels=3)
    ref = voxelise(g, g, g, mesh)
    monkeypatch.setattr(voxelise_module, "_CHUNK_PAIRS", 7)
    np.testing.assert_array_equal(voxelise(g, g, g, mesh), ref)


def test_open_mesh_rays_are_corrected_from_neighbours() -> None:
    """A box with its top removed: z-rays see one crossing (odd) and are filled from
    neighbours; x and y rays still close the box, so the vote keeps it solid."""
    # unequal sides: no face diagonal passes exactly through a grid ray
    full = box_mesh([1.4, 1.7, 1.2], [6.3, 6.6, 6.9])
    open_top = np.delete(full, [2, 3], axis=0)
    g = np.arange(9.0)
    np.testing.assert_array_equal(voxelise(g, g, g, open_top), voxelise(g, g, g, full))
    # z alone: every ray inside the footprint is odd; corrections need >= 4 inside neighbours
    assert not voxelise(g, g, g, open_top, "z").any()


def test_mesh_bound_halfway_between_grid_lines() -> None:
    """MATLAB fails on this tie (x bound 2.5, B-90); the port has no tie to break."""
    g = np.arange(12.0)
    got = voxelise(g, g, g, box_mesh([2.5, 3.2, 1.3], [7.4, 8.6, 6.8]))
    assert got.sum() == 5 * 5 * 5


def test_mesh_outside_grid_is_empty() -> None:
    g = np.arange(5.0)
    assert not voxelise(g, g, g, box_mesh([10, 10, 10], [12, 12, 12])).any()


def test_input_validation() -> None:
    g = np.arange(5.0)
    mesh = box_mesh([1, 1, 1], [3, 3, 3])
    with pytest.raises(ValueError, match="ray_direction"):
        voxelise(g, g, g, mesh, "xx")
    with pytest.raises(ValueError, match="at least two"):
        voxelise(np.array([3.0]), g, g, mesh)
    with pytest.raises(ValueError, match="non-decreasing"):
        voxelise(g[::-1], g, g, mesh)
    with pytest.raises(ValueError, match="shape"):
        voxelise(g, g, g, mesh[:, :2])
    assert voxelise(g[None, :], g[:, None], g, mesh).shape == (5, 5, 5)


# ---------------------------------------------------------------- placement


def geometry(**over):
    G = {
        "Folder": "box",
        "STLfile": "box.stl",
        "STLunit": 1e-3,
        "EmbedVessel": True,
        "Rotation": np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        "BoundingBox": {"Center": np.array([[2e-3], [3e-3], [4e-3]])},  # metres
        "Center": np.array([[0.02], [0.0], [0.001]]),
    }
    G.update(over)
    return G


def test_place_vessel_mesh_matches_define_medium() -> None:
    mesh = box_mesh([1, 2, 3], [3, 4, 5])
    G = geometry()
    got = place_vessel_mesh(mesh, G)
    for k in range(3):  # per vertex: R @ (v*unit - c_bb) + c, as rotate_stl
        v = mesh[:, :, k] * 1e-3 - G["BoundingBox"]["Center"].T
        np.testing.assert_allclose(
            got[:, :, k], (G["Rotation"] @ v.T).T + G["Center"].T, rtol=0, atol=1e-18
        )


def test_vessel_mask_reads_and_places(tmp_path: Path) -> None:
    (tmp_path / "box").mkdir()
    mesh = box_mesh([1.0, 2.0, 3.0], [3.0, 4.0, 5.0])  # mm, bounding box centre (2, 3, 4)
    write_binary(tmp_path / "box" / "box.stl", mesh)
    G = geometry()
    assert stl_path(G, tmp_path) == tmp_path / "box" / "box.stl"
    gx = np.linspace(0.017, 0.023, 13)
    gy = np.linspace(-0.0015, 0.0015, 7)
    gz = np.linspace(-0.0005, 0.0025, 11)
    got = vessel_mask(gx, gy, gz, G, tmp_path)
    want = voxelise(gx, gy, gz, place_vessel_mesh(read_stl(tmp_path / "box" / "box.stl"), G))
    np.testing.assert_array_equal(got, want)
    assert got.any() and not got.all()
    # rotated box: x extent comes from the z extent (2 mm), z from y (2 mm), y from x
    assert got.shape == (13, 7, 11)


def test_vessel_mask_without_embedding_reads_nothing(tmp_path: Path) -> None:
    got = vessel_mask(
        np.arange(4.0), np.arange(3.0), np.arange(5.0), geometry(EmbedVessel=False), tmp_path
    )
    assert got.dtype == bool and got.shape == (4, 3, 5) and not got.any()


def test_vertex_on_ray_counts_once_when_facets_agree() -> None:
    """Pyramid apex on the ray (5, 5): one crossing at the apex, so the ray is filled
    between the base (z = 1.4) and the apex (z = 7.7)."""
    b = np.array([[2.3, 2.6, 1.4], [7.6, 2.6, 1.4], [7.6, 7.3, 1.4], [2.3, 7.3, 1.4]])
    a = np.array([5.0, 5.0, 7.7])
    tris = [(b[0], b[2], b[1]), (b[0], b[3], b[2])] + [(b[i], b[(i + 1) % 4], a) for i in range(4)]
    mesh = np.stack([np.array(t).T for t in tris])
    g = np.arange(11.0)
    np.testing.assert_array_equal(voxelise(g, g, g, mesh, "z")[5, 5], (g > 1.4) & (g < 7.7))


def test_vertex_on_ray_with_mixed_normals_is_corrected() -> None:
    """Box faces on grid planes: rays along its faces and edges hit vertices whose facets
    face different ways, go to the neighbour correction and stay mostly empty. The counts
    are MATLAB VOXELISE's (oracle test ``test_vertices_on_rays_and_open_mesh``)."""
    g = np.arange(11.0)
    mesh = box_mesh([2, 3, 1], [7, 8, 6])
    assert voxelise(g, g, g, mesh, "z").sum() == 48
    assert voxelise(g, g, g, mesh).sum() == 56
