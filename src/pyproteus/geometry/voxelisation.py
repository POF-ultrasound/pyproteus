"""Mesh voxelisation: port of ``VOXELISE.m`` (mesh voxelisation toolbox).

Ported from the MATLAB "Mesh voxelisation" toolbox by Adam H. Aitkenhead
(The Christie NHS Foundation Trust), shipped with PROTEUS in
``mesh-voxelisation/``. Copyright (c) 2013, Adam H. Aitkenhead. All rights
reserved. Redistributed under the BSD-3-Clause licence (non-endorsement
clause naming The Christie NHS Foundation Trust); full text and disclaimer
in ``LICENSES/mesh-voxelisation-BSD.txt``.

Method (Patil S and Ravi B, "Voxel-based representation, display and
thickness analysis of intricate shapes", CAD/CG 2005): rays through the grid
points along one axis cross the mesh; grid points between the 1st and 2nd,
3rd and 4th, ... crossings are inside. Rays with an odd number of crossings,
or through a mesh vertex with mixed facet orientations, are filled from
their 8 neighbouring rays. Tracing along x, y and z and taking a majority
vote removes most artefacts.

The ray/facet tests use the same formulas as ``VOXELISE.m``, but run
vectorised over all (facet, ray) candidate pairs instead of looping over rays;
only the rare vertex hits and the neighbour corrections are sequential.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from pyproteus.numerics import matlab_round

__all__ = ["voxelise"]

_DIRECTIONS = frozenset(
    {"x", "y", "z", "xy", "xz", "yx", "yz", "zx", "zy", "xyz", "xzy", "yxz", "yzx", "zxy", "zyx"}
)
#: Upper bound on (facet, ray) candidate pairs processed at once (memory).
_CHUNK_PAIRS = 2_000_000


def voxelise(
    grid_x: ArrayLike,
    grid_y: ArrayLike,
    grid_z: ArrayLike,
    mesh_xyz: ArrayLike,
    ray_direction: str = "xyz",
) -> np.ndarray:
    """Voxelise a closed triangle mesh on a rectilinear grid (``VOXELISE.m``).

    Parameters
    ----------
    grid_x, grid_y, grid_z
        Grid coordinates along x, y, z (non-decreasing). A single value is a
        one-point axis; MATLAB would read a scalar as a voxel count (PROTEUS
        always passes coordinates, PARITY.md).
    mesh_xyz
        ``(N, 3, 3)`` facet vertices, ``mesh_xyz[facet, coord, vertex]`` (the
        output of :func:`pyproteus.geometry.read_stl`).
    ray_direction
        Axes to trace along, any ordering of a subset of ``'xyz'``. With more
        than one axis a grid point is inside when at least half the
        directions say so (``'xyz'``: 2 of 3; two axes: either).

    Returns
    -------
    numpy.ndarray
        Boolean mask of shape ``(len(grid_x), len(grid_y), len(grid_z))``;
        ``mask[i, j, k]`` is grid point ``(grid_x[i], grid_y[j], grid_z[k])``,
        as MATLAB's ``gridOUTPUT(i+1, j+1, k+1)``.
    """
    direction = ray_direction.lower()
    if direction not in _DIRECTIONS:
        raise ValueError(
            f"ray_direction must be a combination of 'x', 'y', 'z', got {ray_direction!r}"
        )
    gx, gy, gz = (
        _grid_vector(g, name) for g, name in ((grid_x, "x"), (grid_y, "y"), (grid_z, "z"))
    )
    mesh = np.asarray(mesh_xyz, dtype=np.float64)
    if mesh.ndim != 3 or mesh.shape[1:] != (3, 3):
        raise ValueError(f"mesh_xyz must have shape (N, 3, 3), got {mesh.shape}")

    lo = mesh.min(axis=(0, 2))
    hi = mesh.max(axis=(0, 2))

    # VOXELISE.m extends the grid to cover the mesh, but only along the first
    # traced axis that needs it (if/elseif chain), and crops the result again.
    # This only changes which neighbouring rays the corrections see. B-110.
    grids = [gx, gy, gz]
    pad = [(0, 0), (0, 0), (0, 0)]
    for axis, name in enumerate("xyz"):
        g = grids[axis]
        if name in direction and (g.min() > lo[axis] or g.max() < hi[axis]):
            before = int(g.min() > lo[axis])
            after = int(g.max() < hi[axis])
            grids[axis] = np.concatenate([[lo[axis]] * before, g, [hi[axis]] * after])
            pad[axis] = (before, after)
            break
    gx, gy, gz = grids
    if any(np.any(np.diff(g) < 0) for g in grids):
        raise ValueError("grid coordinates must be non-decreasing")

    votes = np.zeros((gx.size, gy.size, gz.size), dtype=np.uint8)
    if "x" in direction:
        votes += _voxelise_internal(gy, gz, gx, mesh[:, [1, 2, 0], :]).transpose(2, 0, 1)
    if "y" in direction:
        votes += _voxelise_internal(gz, gx, gy, mesh[:, [2, 0, 1], :]).transpose(1, 2, 0)
    if "z" in direction:
        votes += _voxelise_internal(gx, gy, gz, mesh)
    mask = votes >= len(direction) / 2

    crop = tuple(slice(b, n - a) for (b, a), n in zip(pad, mask.shape, strict=True))
    return np.ascontiguousarray(mask[crop])


def _grid_vector(g: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(g, dtype=np.float64)
    if arr.ndim > 1 and sum(s > 1 for s in arr.shape) > 1:
        raise ValueError(f"grid_{name} must be a vector, got shape {arr.shape}")
    arr = arr.reshape(-1)
    if arr.size < 1:
        raise ValueError(f"grid_{name} is empty")
    return arr


def _voxelise_internal(
    gx: np.ndarray, gy: np.ndarray, gz: np.ndarray, mesh: np.ndarray
) -> np.ndarray:
    """Trace rays along the 3rd axis (``VOXELISEinternal``); returns a uint8 mask."""
    nx, ny, nz = gx.size, gy.size, gz.size
    zmin = mesh[:, 2, :].min()
    zmax = mesh[:, 2, :].max()
    fmin = mesh.min(axis=2)
    fmax = mesh.max(axis=2)

    # Candidate rays per facet: grid points inside the facet's xy bounding
    # box, bounds included (meshXYZmin <= g <= meshXYZmax).
    ix0 = np.searchsorted(gx, fmin[:, 0], "left")
    cx = np.maximum(np.searchsorted(gx, fmax[:, 0], "right") - ix0, 0)
    iy0 = np.searchsorted(gy, fmin[:, 1], "left")
    cy = np.maximum(np.searchsorted(gy, fmax[:, 1], "right") - iy0, 0)
    count = cx * cy

    keys, zs, vertex_pairs = [], [], []
    facets = np.flatnonzero(count)
    csum = np.cumsum(count[facets])
    start = 0
    while start < facets.size:
        base = csum[start - 1] if start else 0
        stop = max(int(np.searchsorted(csum, base + _CHUNK_PAIRS, "right")), start + 1)
        chunk = facets[start:stop]
        start = stop
        n = count[chunk]
        f = np.repeat(chunk, n)
        local = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
        ix = ix0[f] + local % cx[f]
        iy = iy0[f] + local // cx[f]
        xr, yr = gx[ix], gy[iy]
        tri = mesh[f]
        hit = _vertex_hit(tri, xr, yr)
        if hit.any():
            vertex_pairs.append((f[hit], ix[hit], iy[hit]))
        cross = _ray_crosses_facet(tri, xr, yr)
        keys.append(iy[cross] * nx + ix[cross])
        zs.append(_plane_z(tri[cross], xr[cross], yr[cross]))

    key = np.concatenate(keys) if keys else np.zeros(0, np.int64)
    z = np.concatenate(zs) if zs else np.zeros(0)
    correction = np.zeros(0, np.int64)

    if vertex_pairs:
        extra_key, extra_z, vertex_corr = _vertex_rays(
            mesh, gx, gy, nx, *map(np.concatenate, zip(*vertex_pairs, strict=True))
        )
        drop = np.isin(key, vertex_corr)
        key = np.concatenate([key[~drop], extra_key])
        z = np.concatenate([z[~drop], extra_z])
        correction = vertex_corr

    # Keep crossings inside the mesh z-range (1e-12 margin), round to 1e-12
    # and keep the unique values per ray.
    keep = (z >= zmin - 1e-12) & (z <= zmax + 1e-12)
    key, z = key[keep], matlab_round(z[keep] * 1e12) / 1e12
    order = np.lexsort((z, key))
    key, z = key[order], z[order]
    first = np.ones(key.size, bool)
    first[1:] = (key[1:] != key[:-1]) | (z[1:] != z[:-1])
    key, z = key[first], z[first]

    rays, n_cross = np.unique(key, return_counts=True)
    even = np.repeat(n_cross % 2 == 0, n_cross)
    correction = np.union1d(correction, rays[n_cross % 2 == 1])

    # Fill grid points strictly between crossings 1-2, 3-4, ... of even rays.
    k_even, z_even = key[even], z[even]
    k_pair = k_even[0::2]
    i0 = np.searchsorted(gz, z_even[0::2], "right")
    i1 = np.searchsorted(gz, z_even[1::2], "left")
    nonempty = i1 > i0
    edges = np.zeros((nx * ny, nz + 1), dtype=np.int8)
    np.add.at(edges, (k_pair[nonempty], i0[nonempty]), 1)
    np.add.at(edges, (k_pair[nonempty], i1[nonempty]), -1)
    out = np.cumsum(edges[:, :nz], axis=1, dtype=np.int8).astype(np.uint8)
    # rays are keyed iy*nx + ix: rows are (iy, ix)
    out = out.reshape(ny, nx, nz).transpose(1, 0, 2)

    if correction.size:
        out = _correct_rays(out, correction, nx)
    return out


def _vertex_hit(tri: np.ndarray, xr: np.ndarray, yr: np.ndarray) -> np.ndarray:
    return np.any((tri[:, 0, :] == xr[:, None]) & (tri[:, 1, :] == yr[:, None]), axis=1)


def _ray_crosses_facet(tri: np.ndarray, xr: np.ndarray, yr: np.ndarray) -> np.ndarray:
    """Ray strictly on the inner side of all three edges (VOXELISE.m formulas)."""
    x, y = tri[:, 0, :], tri[:, 1, :]
    ok = np.ones(xr.size, bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        # (edge start b, edge end c, opposite vertex a): edges 2-3, 3-1, 1-2
        for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
            slope = y[:, b] - y[:, c]
            dx = x[:, b] - x[:, c]
            y_opp = y[:, b] - (slope * (x[:, b] - x[:, a]) / dx)
            y_ray = y[:, b] - (slope * (x[:, b] - xr) / dx)
            ok &= ((y_opp > y[:, a]) & (y_ray > yr)) | ((y_opp < y[:, a]) & (y_ray < yr))
    return ok


def _plane_z(tri: np.ndarray, xr: np.ndarray, yr: np.ndarray) -> np.ndarray:
    """z where the ray meets the facet plane Ax + By + Cz + D = 0."""
    (x1, x2, x3), (y1, y2, y3), (z1, z2, z3) = (tri[:, i, :].T for i in range(3))
    A = y1 * (z2 - z3) + y2 * (z3 - z1) + y3 * (z1 - z2)
    B = z1 * (x2 - x3) + z2 * (x3 - x1) + z3 * (x1 - x2)
    C = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
    D = -x1 * (y2 * z3 - y3 * z2) - x2 * (y3 * z1 - y1 * z3) - x3 * (y1 * z2 - y2 * z1)
    # Absolute threshold in the mesh's length unit squared (B-111).
    C = np.where(np.abs(C) < 1e-14, 0.0, C)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (-D - A * xr - B * yr) / C


def _vertex_rays(
    mesh: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    nx: int,
    f: np.ndarray,
    ix: np.ndarray,
    iy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rays through mesh vertices (VOXELISE.m step 1).

    Facets sharing a vertex with each crossed-vertex facet are grouped; if
    their normals all point the same way along the ray, the facet counts as
    crossed, otherwise the whole ray is left for the neighbour correction.
    Returns the extra crossings (ray keys, z) and the rays to correct.
    """
    keys = iy * nx + ix
    extra_k, extra_z, corrections = [], [], []
    for k in np.unique(keys):
        sel = keys == k
        cand = np.sort(f[sel], kind="stable")
        tri = mesh[cand]
        xr, yr = gx[ix[sel][0]], gy[iy[sel][0]]
        flat = tri.transpose(0, 2, 1).reshape(-1, 3)
        _, vid = np.unique(flat, axis=0, return_inverse=True)
        vid = vid.reshape(-1, 3)
        normal_z = _unit_normal_z(tri)
        checked = np.zeros(cand.size, bool)
        crossed = []
        while not checked.all():
            i = int(np.flatnonzero(~checked)[0])
            adjacent = np.isin(vid, vid[i]).any(axis=1)
            checked |= adjacent
            checked[i] = True
            nzs = normal_z[adjacent]
            nzs = nzs[~np.isnan(nzs)]  # MATLAB max/min omit NaN
            if nzs.size and (nzs.max() < 0 or nzs.min() > 0):
                crossed.append(i)
            else:
                corrections.append(k)
                crossed = []
                break
        if crossed:
            extra_k.append(np.full(len(crossed), k))
            extra_z.append(
                _plane_z(tri[crossed], np.full(len(crossed), xr), np.full(len(crossed), yr))
            )
    return _cat(extra_k, np.int64), _cat(extra_z, np.float64), np.asarray(corrections, np.int64)


def _cat(arrays: list[np.ndarray], dtype: type) -> np.ndarray:
    return np.concatenate(arrays) if arrays else np.zeros(0, dtype)


def _unit_normal_z(tri: np.ndarray) -> np.ndarray:
    ab = tri[:, :, 1] - tri[:, :, 0]
    ac = tri[:, :, 2] - tri[:, :, 0]
    n = np.cross(ab, ac)
    with np.errstate(divide="ignore", invalid="ignore"):
        return n[:, 2] / np.linalg.norm(n, axis=1)


def _correct_rays(out: np.ndarray, correction: np.ndarray, nx: int) -> np.ndarray:
    """Fill unresolved rays from their 8 neighbours (>= 4 inside), in ray order.

    Corrections run sequentially in MATLAB's loop order (y outer, x inner), so
    a corrected ray feeds later ones. Out-of-grid neighbours count as outside.
    """
    padded = np.pad(out, ((1, 1), (1, 1), (0, 0)))
    for k in np.sort(correction, kind="stable"):
        i, j = int(k % nx) + 1, int(k // nx) + 1
        block = padded[i - 1 : i + 2, j - 1 : j + 2, :].astype(np.int16)
        votes = block.sum(axis=(0, 1)) - block[1, 1]
        padded[i, j, votes >= 4] = 1
    return padded[1:-1, 1:-1, :]
