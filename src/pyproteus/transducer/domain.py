"""Simulation domain (port of ``GUIfunctions/compute_simulation_domain.m``).

The domain is the axis-aligned box that holds the transducer surface and,
unless the domain is set manually, the beam projected to the end depth of the
rotated geometry bounding box, plus a margin.
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np

from pyproteus.config.models import Domain, Geometry

from ._fields import configuration_is, field, scalar
from ._trig import cosd, sind
from .geometry import get_element_geometry, is_curved

__all__ = [
    "GeometryBehindTransducerWarning",
    "compute_end_depth",
    "compute_simulation_domain",
    "compute_transducer_projection",
    "compute_transducer_vertices",
]

# Vertices of a cube with unit sides, centred on the origin (3×8).
_UNIT_CUBE = (
    np.array(
        [
            [-1, -1, -1, -1, 1, 1, 1, 1],
            [-1, -1, 1, 1, -1, -1, 1, 1],
            [-1, 1, -1, 1, -1, 1, -1, 1],
        ],
        dtype=np.float64,
    )
    / 2
)


class GeometryBehindTransducerWarning(UserWarning):
    """The geometry starts behind the surface of a curved array
    (MATLAB ``compute_simulation_domain:geometryBehindTransducer``)."""


class _Halved:
    """A transducer view with ``NumberOfElements`` replaced (RCA: columns only)."""

    def __init__(self, transducer: Any, n: float) -> None:
        self._t, self.NumberOfElements = transducer, n

    def __getattr__(self, name: str) -> Any:
        return field(self._t, name)


def _width(transducer: Any) -> float:
    """Aperture ``p*(N-1) + w`` of a flat array [m]."""
    p = scalar(transducer, "Pitch")
    w = scalar(transducer, "ElementWidth")
    return p * (scalar(transducer, "NumberOfElements") - 1) + w


def compute_transducer_vertices(transducer: Any) -> np.ndarray:
    """Vertices of the transducer surface: 4×3, or 4N×3 for a curved array.

    Flat arrays: the rectangle x = 0, |y| <= W/2, |z| <= H/2, rows ordered
    (right, top), (left, top), (left, bottom), (right, bottom). Curved arrays:
    the flat quadrilateral through the edges of each element, element by
    element in the same order. ``'Custom'`` arrays: the bounding rectangle of
    ``Transducer.integration_points`` at its smallest x.
    """
    h = scalar(transducer, "ElementHeight")
    if is_curved(transducer):
        el = get_element_geometry(transducer)
        r = el.radius
        theta_l = el.theta - el.width / (2 * r)  # angles of the left edges
        theta_r = el.theta + el.width / (2 * r)  # angles of the right edges
        xl, yl = el.center + r * np.cos(theta_l), r * np.sin(theta_l)
        xr, yr = el.center + r * np.cos(theta_r), r * np.sin(theta_r)
        zt = np.full(el.n_elements, h / 2)
        zb = np.full(el.n_elements, -h / 2)
        x = np.stack([xr, xl, xl, xr], axis=1).reshape(-1)
        y = np.stack([yr, yl, yl, yr], axis=1).reshape(-1)
        z = np.stack([zt, zt, zb, zb], axis=1).reshape(-1)
        return np.stack([x, y, z], axis=1)

    if configuration_is(transducer, "Custom"):
        pts = np.asarray(field(transducer, "integration_points"), dtype=np.float64)
        pts = pts.reshape(-1, 3, order="F")  # MATLAB reshape(points, [], 3)
        x0 = np.fmin.reduce(pts[:, 0])
        yl, yr = np.fmin.reduce(pts[:, 1]), np.fmax.reduce(pts[:, 1])
        zb, zt = np.fmin.reduce(pts[:, 2]), np.fmax.reduce(pts[:, 2])
    else:
        w = _width(transducer)
        x0, yl, yr, zb, zt = 0.0, -w / 2, w / 2, -h / 2, h / 2

    return np.array([[x0, yr, zt], [x0, yl, zt], [x0, yl, zb], [x0, yr, zb]], dtype=np.float64)


def compute_transducer_projection(transducer: Any, transmit: Any, d: float) -> np.ndarray:
    """Vertices (4×3) of the beam's footprint on the plane x = ``d``.

    Intersection of two pinhole projections of the transducer surface: through
    the elevation focus line (x = ``ElevationFocus``, z = 0) and through the
    lateral focus line (``LateralFocus`` at ``Angle`` degrees; unfocused: the
    aperture shifted along the steering direction). A curved array uses the
    chord through its outer edges, and an infinite focus becomes its centre
    of curvature (a diverging wave). ``'Custom'`` arrays: the surface moved to
    x = ``d``.
    """
    if configuration_is(transducer, "Custom"):
        out = compute_transducer_vertices(transducer)
        out[:, 0] = d
        return out

    f_e = scalar(transducer, "ElevationFocus")
    f_l = scalar(transmit, "LateralFocus", "Transmit")
    theta = scalar(transmit, "Angle", "Transmit")
    w = _width(transducer)
    h = scalar(transducer, "ElementHeight")

    if is_curved(transducer):
        el = get_element_geometry(transducer)
        w = 2 * el.radius * math.sin(el.theta_max)
        if not math.isfinite(f_l):
            f_l = el.center

    with np.errstate(divide="ignore", invalid="ignore"):
        d_ = np.float64(d)
        zt = h / 2 * (1 - d_ / f_e)  # top line
        zb = -h / 2 * (1 - d_ / f_e)  # bottom line
        if abs(f_l) < math.inf:  # focused beam
            fx = np.float64(f_l * float(cosd(theta)))  # axial focus coordinate
            fy = f_l * float(sind(theta))  # lateral focus coordinate
            yl = -w / 2 * (1 - d_ / fx) + d_ * fy / fx  # left line
            yr = w / 2 * (1 - d_ / fx) + d_ * fy / fx  # right line
        else:  # unfocused
            s = float(sind(theta))
            yl = -w / 2 + d_ * s
            yr = w / 2 + d_ * s
    return np.array([[d, yr, zt], [d, yl, zt], [d, yl, zb], [d, yr, zb]], dtype=np.float64)


def compute_end_depth(geometry: Geometry) -> float:
    """Start depth plus the axial extent of the rotated geometry bounding box [m]."""
    diagonal = np.asarray(geometry.BoundingBox.Diagonal, dtype=np.float64).reshape(-1, 1)
    bb = np.asarray(geometry.Rotation, dtype=np.float64) @ (diagonal * _UNIT_CUBE)
    return float(geometry.startDepth) + np.max(bb[0]) - np.min(bb[0])


def compute_simulation_domain(geometry: Geometry, transducer: Any, transmit: Any) -> Geometry:
    """Port of ``compute_simulation_domain.m``. Returns an updated copy of ``geometry``.

    Sets ``Domain.TransducerSurface``, ``Domain.TransducerProjection``,
    ``Domain.Xmin..Zmax``, ``Domain.Vertices`` (8×3) and ``Geometry.Center``
    (3×1, centre of the rotated bounding box on the x axis). Uses
    ``Domain.Manual`` and ``Domain.Margin`` (and ``Domain.Xmin..Zmax`` when
    manual: the box is then grown to hold the transducer). RCA arrays use their
    first half of elements (the columns); an odd RCA count raises ``ValueError``.
    Warns (:class:`GeometryBehindTransducerWarning`) when the geometry starts
    behind the apex of a curved array.
    """
    if configuration_is(transducer, "RCA"):
        n = scalar(transducer, "NumberOfElements")
        if math.fmod(n, 2):
            raise ValueError("Number of elements in RCA must be even.")
        transducer = _Halved(transducer, n / 2)

    start_depth = float(geometry.startDepth)
    end_depth = compute_end_depth(geometry)

    if is_curved(transducer):
        el = get_element_geometry(transducer)
        if start_depth < el.apex:
            warnings.warn(
                f"The geometry starts at {start_depth * 1e3:.1f} mm, behind the surface of the "
                f"curved array (apex at {el.apex * 1e3:.1f} mm). Increase the start depth.",
                GeometryBehindTransducerWarning,
                stacklevel=2,
            )

    surface = compute_transducer_vertices(transducer)
    domain: Domain = geometry.Domain.copy_struct()
    manual = bool(domain.Manual)
    margin = scalar(domain, "Margin", "Geometry.Domain")

    if manual:
        v = surface
    else:
        projection = compute_transducer_projection(transducer, transmit, end_depth)
        v = np.concatenate([surface, projection])

    # MATLAB max/min along columns ignore NaN
    xmax, ymax, zmax = np.fmax.reduce(v, axis=0) + margin
    xmin, ymin, zmin = np.fmin.reduce(v, axis=0) - margin

    if manual:
        xmin, xmax = np.fmin(domain.Xmin, xmin), np.fmax(domain.Xmax, xmax)
        ymin, ymax = np.fmin(domain.Ymin, ymin), np.fmax(domain.Ymax, ymax)
        zmin, zmax = np.fmin(domain.Zmin, zmin), np.fmax(domain.Zmax, zmax)
        projection = compute_transducer_projection(transducer, transmit, float(xmax))

    x = [xmin, xmin, xmin, xmin, xmax, xmax, xmax, xmax]
    y = [ymax, ymax, ymin, ymin, ymax, ymax, ymin, ymin]
    z = [zmax, zmin, zmin, zmax, zmax, zmin, zmin, zmax]

    domain.TransducerSurface = surface
    domain.TransducerProjection = projection
    domain.Xmax, domain.Ymax, domain.Zmax = float(xmax), float(ymax), float(zmax)
    domain.Xmin, domain.Ymin, domain.Zmin = float(xmin), float(ymin), float(zmin)
    domain.Vertices = np.array([x, y, z], dtype=np.float64).T

    out = geometry.copy_struct()
    out.Domain = domain
    out.Center = np.array([[(start_depth + end_depth) / 2], [0.0], [0.0]])
    return out
