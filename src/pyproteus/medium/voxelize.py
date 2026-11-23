"""Snap points to the grid (port of ``voxelize_media_points.m``)."""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
from numpy.typing import ArrayLike

from pyproteus.numerics import matlab_round, sub2ind

__all__ = ["VoxelizedPoints", "voxelize_media_points"]


class VoxelizedPoints(NamedTuple):
    """Result of :func:`voxelize_media_points` (MATLAB ``[MP, MP_nodes, MB_idx, idx_exclude]``)."""

    points: np.ndarray
    """Points inside the grid, ``M × 3`` [m] (MATLAB ``MP``)."""
    nodes: np.ndarray
    """0-based grid subscripts, ``M × 3`` int (MATLAB ``MP_nodes`` - 1)."""
    index: np.ndarray
    """0-based column-major linear indices into ``(Nx, Ny, Nz)`` (MATLAB ``MB_idx`` - 1)."""
    exclude: np.ndarray
    """Boolean mask over the input rows: points outside the grid (MATLAB ``idx_exclude``)."""


def voxelize_media_points(points: ArrayLike, grid: Any) -> VoxelizedPoints:
    """Port of ``voxelize_media_points.m``.

    Points outside ``[min(x), max(x)] × [min(y), max(y)] × [min(z), max(z)]``
    are dropped; the others go to the nearest node,
    ``round((p - min)/d)`` with MATLAB rounding (halves away from zero).
    Indices are 0-based (pyproteus convention); add 1 for MATLAB's values.
    """
    P = np.asarray(points, dtype=float).reshape(-1, 3)
    axes = (np.asarray(grid.x), np.asarray(grid.y), np.asarray(grid.z))
    spacing = (grid.dx, grid.dy, grid.dz)
    lo = np.array([a.min() for a in axes])
    hi = np.array([a.max() for a in axes])

    exclude = np.any(lo > P, axis=1) | np.any(hi < P, axis=1)
    kept = P[~exclude]
    if not np.all(np.isfinite(kept)):
        raise ValueError("points must be finite (MATLAB sub2ind fails on NaN)")
    nodes = np.column_stack(
        [matlab_round((kept[:, k] - lo[k]) / spacing[k]) for k in range(3)]
    ).astype(np.intp)
    nodes = nodes.reshape(-1, 3)
    index = sub2ind((grid.Nx, grid.Ny, grid.Nz), nodes[:, 0], nodes[:, 1], nodes[:, 2])
    return VoxelizedPoints(kept, nodes, np.asarray(index, dtype=np.intp), exclude)
