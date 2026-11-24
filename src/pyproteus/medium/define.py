"""Voxel medium for k-Wave (port of ``define_medium.m``)."""

from __future__ import annotations

from os import PathLike
from typing import Any

import numpy as np

from pyproteus.acoustic._struct import get as _get
from pyproteus.acoustic._struct import scalar as _scalar
from pyproteus.numerics import MatlabRNG

from .random import random_truncated_normal

__all__ = ["define_medium"]


def define_medium(
    grid: Any,
    medium: Any,
    geometry: Any,
    transducer: Any = None,
    *,
    rng: MatlabRNG,
    vessel: np.ndarray | None = None,
    geometries_path: str | PathLike[str] | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Port of ``define_medium.m``: k-Wave medium maps and the vessel mask.

    Returns ``(kwave_medium, vessel)``. ``kwave_medium`` is a dict with
    ``sound_speed``, ``density``, ``BonA``, ``alpha_coeff`` (``Nx × Ny × Nz``),
    ``alpha_power`` (scalar) and, when ``|alpha_power - 1| < 0.1``,
    ``alpha_mode = 'no_dispersion'`` (ignored by the k-Wave C++ path, Q-1).

    Background: ``1 + Inhomogeneity * random_truncated_normal(InhomogeneityCutoff,
    [Nx Ny Nz])`` scales ``SpeedOfSound`` and ``Density``. The draws come from
    ``rng`` (MATLAB: the global stream). With a curved ``transducer``, the
    background is 1 inside the circle of curvature (x-y plane), the coupling
    region behind the array. Vessel voxels take ``Medium.Vessel`` sound speed,
    density and B/A.

    Vessel (``Geometry.EmbedVessel``): the STL file
    ``GeometriesPath/Folder/STLfile`` is read, placed and voxelised on the grid
    (:func:`pyproteus.geometry.vessel_mask`, ports of ``READ_stl``/``VOXELISE``).
    ``geometries_path`` is MATLAB's ``Geometry.GeometriesPath`` (used when not
    given). A precomputed mask can be passed as ``vessel`` (bool
    ``Nx × Ny × Nz``) instead; then no file is read. An empty embedded vessel
    warns (:class:`pyproteus.geometry.EmptyVesselWarning`, B-112). With
    ``EmbedVessel`` false, ``vessel`` must be None and the mask is all false.

    Divergences (D-15, docs/PARITY.md): ``alpha_coeff`` is a read-only
    broadcast view of one value (MATLAB: a full array of a constant), and
    ``BonA`` too when the vessel is empty; values and indexing are the same, the
    memory is not. ``Geometry.GeometriesPath``/``Folder`` are not needed when
    no STL file is read (MATLAB builds the path first and errors if absent).
    """
    shape = (int(grid.Nx), int(grid.Ny), int(grid.Nz))
    cutoff = _get(medium, "InhomogeneityCutoff")
    background = 1 + _scalar(_get(medium, "Inhomogeneity")) * random_truncated_normal(
        cutoff, list(shape), rng
    ).reshape(shape)  # MATLAB rand drops a trailing singleton (Nz = 1); keep 3-D

    if transducer is not None:
        from pyproteus.transducer import get_element_geometry, is_curved

        if is_curved(transducer):
            el = get_element_geometry(transducer)
            x = np.asarray(grid.x, dtype=float).reshape(-1, 1)
            y = np.asarray(grid.y, dtype=float).reshape(1, -1)
            coupling = (x - el.center) ** 2 + y**2 <= el.radius**2
            background[coupling, :] = 1

    embed = bool(np.asarray(_get(geometry, "EmbedVessel", False)).reshape(-1)[0])
    if embed:
        if vessel is None:
            from pyproteus.geometry import vessel_mask

            path = geometries_path
            if path is None:
                path = _get(geometry, "GeometriesPath", None)
            if path is None:
                raise ValueError(
                    "Geometry.EmbedVessel needs the STL folder: pass geometries_path= "
                    "(MATLAB Geometry.GeometriesPath) or a precomputed vessel= mask"
                )
            vessel = vessel_mask(grid.x, grid.y, grid.z, geometry, str(path))
        vessel = np.asarray(vessel)
        if vessel.shape != shape or vessel.dtype != bool:
            raise ValueError(f"vessel must be a bool array of shape {shape}")
    else:
        if vessel is not None:
            raise ValueError("vessel given but Geometry.EmbedVessel is false")
        vessel = np.zeros(shape, dtype=bool)

    v = _get(medium, "Vessel")
    sound_speed = _scalar(_get(medium, "SpeedOfSound")) * background
    density = _scalar(_get(medium, "Density")) * background
    bona_value = _scalar(_get(medium, "BonA"))
    if vessel.any():
        sound_speed[vessel] = _scalar(_get(v, "SpeedOfSound"))
        density[vessel] = _scalar(_get(v, "Density"))
        bona = np.full(shape, bona_value)
        bona[vessel] = _scalar(_get(v, "BonA"))
    else:
        bona = np.broadcast_to(bona_value, shape)

    out: dict[str, Any] = {
        "sound_speed": sound_speed,
        "density": density,
        "BonA": bona,
        "alpha_coeff": np.broadcast_to(_scalar(_get(medium, "AttenuationA")), shape),
        "alpha_power": _scalar(_get(medium, "AttenuationB")),
    }
    if abs(out["alpha_power"] - 1) < 0.1:
        out["alpha_mode"] = "no_dispersion"
    return out, vessel
