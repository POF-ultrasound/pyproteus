"""Vessel geometry: STL reading, placement and voxelisation.

:func:`read_stl` and :func:`voxelise` port ``READ_stl.m`` and ``VOXELISE.m``
from the BSD-licensed mesh voxelisation toolbox by Adam H. Aitkenhead
(``LICENSES/mesh-voxelisation-BSD.txt``). :func:`place_vessel_mesh` and
:func:`vessel_mask` port the vessel block of PROTEUS ``define_medium.m``
(MIT): read the STL, scale to metres, centre on the bounding box, rotate,
move to ``Geometry.Center`` and voxelise on the simulation grid.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from pyproteus.geometry.stl import identify_stl_format, read_stl
from pyproteus.geometry.voxelisation import voxelise

__all__ = [
    "EmptyVesselWarning",
    "identify_stl_format",
    "place_vessel_mesh",
    "read_stl",
    "stl_path",
    "vessel_mask",
    "voxelise",
]


class EmptyVesselWarning(UserWarning):
    """An embedded vessel covers no grid point (vessel thinner than the grid spacing)."""


def _get(obj: Any, name: str) -> Any:
    return obj[name] if isinstance(obj, Mapping) else getattr(obj, name)


def stl_path(geometry: Any, geometries_path: str | PathLike[str]) -> Path:
    """``GeometriesPath/Folder/STLfile`` (MATLAB ``PATHS.GeometriesPath``)."""
    return Path(geometries_path) / _get(geometry, "Folder") / _get(geometry, "STLfile")


def place_vessel_mesh(mesh_xyz: ArrayLike, geometry: Any) -> np.ndarray:
    """Scale, centre, rotate and position STL facets as ``define_medium.m``.

    ``V*STLunit - BoundingBox.Center``, rotated by ``Geometry.Rotation``
    (``B = R*A`` for column vectors), then ``+ Geometry.Center``. ``geometry``
    is a :class:`pyproteus.config.models.Geometry` or a mapping with MATLAB
    field names. Returns a new ``(N, 3, 3)`` array.
    """
    mesh = np.asarray(mesh_xyz, dtype=np.float64)
    bbox = _get(geometry, "BoundingBox")
    bb_center = np.asarray(_get(bbox, "Center"), dtype=np.float64).reshape(1, 3, 1)
    center = np.asarray(_get(geometry, "Center"), dtype=np.float64).reshape(1, 3, 1)
    rotation = np.asarray(_get(geometry, "Rotation"), dtype=np.float64).reshape(3, 3)
    v = mesh * float(_get(geometry, "STLunit")) - bb_center
    v = np.einsum("ij,njv->niv", rotation, v)
    return v + center


def vessel_mask(
    grid_x: ArrayLike,
    grid_y: ArrayLike,
    grid_z: ArrayLike,
    geometry: Any,
    geometries_path: str | PathLike[str],
    *,
    _stacklevel: int = 2,
) -> np.ndarray:
    """Vessel mask of ``define_medium.m`` on the grid ``(Grid.x, Grid.y, Grid.z)``.

    Returns a boolean ``(Nx, Ny, Nz)`` array, ``mask[i, j, k]`` for grid point
    ``(x[i], y[j], z[k])`` (MATLAB ``vessel(i+1, j+1, k+1)``). With
    ``Geometry.EmbedVessel`` false the mask is all false and no file is read.

    Warns (:class:`EmptyVesselWarning`) when the embedded vessel covers no grid
    point, e.g. the mouse brain arteriole on PROTEUS's default grid (B-112);
    MATLAB silently simulates without a vessel.
    """
    shape = tuple(np.asarray(g).size for g in (grid_x, grid_y, grid_z))
    if not bool(np.asarray(_get(geometry, "EmbedVessel")).reshape(-1)[0]):
        return np.zeros(shape, dtype=bool)
    mesh = place_vessel_mesh(read_stl(stl_path(geometry, geometries_path)), geometry)
    mask = voxelise(grid_x, grid_y, grid_z, mesh)
    if not mask.any():
        axes = [np.asarray(g, dtype=float).reshape(-1) for g in (grid_x, grid_y, grid_z)]
        lo, hi = mesh.min(axis=(0, 2)), mesh.max(axis=(0, 2))
        name = f"{_get(geometry, 'Folder')}/{_get(geometry, 'STLfile')}"
        if any(h < a.min() or l > a.max() for a, l, h in zip(axes, lo, hi, strict=True)):
            msg = (
                f"The vessel ({name}) lies outside the grid: check Geometry.Center, "
                "Rotation and STLunit. The medium has no vessel."
            )
        else:
            steps = [np.diff(a) for a in axes]
            dx = max((float(d.max()) for d in steps if d.size), default=float("nan"))
            msg = (
                f"The vessel ({name}) covers no grid point: it is thinner than the grid "
                f"spacing (dx = {dx * 1e6:.1f} µm). The medium has no vessel; refine the grid "
                "(PointsPerWavelength) to resolve it."
            )
        warnings.warn(msg, EmptyVesselWarning, stacklevel=_stacklevel)
    return mask
