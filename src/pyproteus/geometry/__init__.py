"""Vessel geometry: STL reading, placement and voxelisation.

:func:`read_stl` and :func:`voxelise` port ``READ_stl.m`` and ``VOXELISE.m``
from the BSD-licensed mesh voxelisation toolbox by Adam H. Aitkenhead
(``LICENSES/mesh-voxelisation-BSD.txt``). :func:`place_vessel_mesh` and
:func:`vessel_mask` port the vessel block of PROTEUS ``define_medium.m``
(MIT): read the STL, scale to metres, centre on the bounding box, rotate,
move to ``Geometry.Center`` and voxelise on the simulation grid.
"""

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from pyproteus.geometry.stl import identify_stl_format, read_stl
from pyproteus.geometry.voxelisation import voxelise

__all__ = [
    "identify_stl_format",
    "place_vessel_mesh",
    "read_stl",
    "stl_path",
    "vessel_mask",
    "voxelise",
]


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
) -> np.ndarray:
    """Vessel mask of ``define_medium.m`` on the grid ``(Grid.x, Grid.y, Grid.z)``.

    Returns a boolean ``(Nx, Ny, Nz)`` array, ``mask[i, j, k]`` for grid point
    ``(x[i], y[j], z[k])`` (MATLAB ``vessel(i+1, j+1, k+1)``). With
    ``Geometry.EmbedVessel`` false the mask is all false and no file is read.
    """
    shape = tuple(np.asarray(g).size for g in (grid_x, grid_y, grid_z))
    if not bool(_get(geometry, "EmbedVessel")):
        return np.zeros(shape, dtype=bool)
    mesh = place_vessel_mesh(read_stl(stl_path(geometry, geometries_path)), geometry)
    return voxelise(grid_x, grid_y, grid_z, mesh)
