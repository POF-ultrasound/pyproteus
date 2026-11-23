"""Element geometry of linear and curved arrays (port of ``get_element_geometry.m``).

This is the single definition of the array geometry. The transmit delays, the
simulation domain, the integration points, the medium and the delay-and-sum
reconstruction all build on it.

Conventions (x axial, y lateral, z elevation), as in PROTEUS:

* Linear array: the elements lie in the plane x = 0, centred about y = 0.
* Curved array (``Transducer.Configuration == 'Curved'``): the elements are a
  uniform subdivision of an arc of radius R, without kerf. Element i (1-based)
  has its normal at ``theta_i = p*(i - (N+1)/2)/R`` from the x axis and spans
  ``theta_i -/+ p/(2R)``, p the pitch. ``Transducer.ElementWidth`` is not used.
  The two outer edges of the array lie at x = 0, so the centre of curvature is
  at ``x = -R*cos(N*p/(2R))`` and the apex (centre of the array surface) at
  ``x = R - R*cos(N*p/(2R))``.
* Element 1 has the most negative lateral coordinate.

Any other ``Configuration`` (``'Linear'``, ``'RCA'``, absent) gives the linear
geometry, as in MATLAB (exact, case-sensitive ``strcmp`` with ``'Curved'``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["ElementGeometry", "get_element_geometry"]


@dataclass(frozen=True)
class ElementGeometry:
    """Element positions of an array (MATLAB struct ``El``).

    Attributes (MATLAB field in brackets):

    x, y, theta
        Axial and lateral coordinates of the element centres [m] and angle of
        the element normals with the x axis [rad]; 1-D arrays of length N
        (``El.x``, ``El.y``, ``El.theta``, N×1 in MATLAB).
    width
        Width of each element along the array surface [m] (``El.width``).
    radius
        Radius of curvature, ``inf`` for a linear array [m] (``El.Radius``).
    apex
        Axial coordinate of the centre of the array surface [m] (``El.Apex``).
    center
        Axial coordinate of the centre of curvature, ``-inf`` for a linear
        array [m] (``El.Center``).
    theta_max
        Half the angular aperture of a curved array, 0 for a linear array
        [rad] (``El.ThetaMax``).
    """

    x: np.ndarray
    y: np.ndarray
    theta: np.ndarray
    width: float
    radius: float
    apex: float
    center: float
    theta_max: float

    @property
    def curved(self) -> bool:
        """True for a curved array (finite radius)."""
        return math.isfinite(self.radius)

    @property
    def n_elements(self) -> int:
        return int(self.x.size)

    def to_struct(self) -> dict[str, Any]:
        """The MATLAB struct ``El``: MATLAB field names, N×1 column vectors."""
        return {
            "theta": self.theta.reshape(-1, 1),
            "x": self.x.reshape(-1, 1),
            "y": self.y.reshape(-1, 1),
            "width": self.width,
            "Radius": self.radius,
            "Apex": self.apex,
            "Center": self.center,
            "ThetaMax": self.theta_max,
        }


def _field(transducer: Any, name: str) -> Any:
    """``Transducer.(name)``, or None if the field is absent."""
    if isinstance(transducer, Mapping):
        return transducer.get(name)
    return getattr(transducer, name, None)


def _required_scalar(transducer: Any, name: str) -> float:
    value = _field(transducer, name)
    if value is None:
        raise ValueError(f"Transducer has no field {name}")  # MATLAB: reference error
    arr = np.asarray(value, dtype=np.float64)
    if arr.size != 1:
        raise ValueError(f"Transducer.{name} must be a scalar, got size {arr.shape}")
    return float(arr.reshape(()))


def _is_curved(transducer: Any) -> bool:
    return _field(transducer, "Configuration") == "Curved"


def get_element_geometry(transducer: Any) -> ElementGeometry:
    """Port of ``get_element_geometry.m``: element positions of a linear or curved array.

    ``transducer`` is a :class:`pyproteus.config.Transducer` or a mapping with
    the MATLAB field names. Uses ``NumberOfElements``, ``Pitch``,
    ``Configuration``, ``Radius`` (curved) and ``ElementWidth`` (linear).

    Raises ``ValueError`` where MATLAB raises an error: a curved array without
    a positive finite scalar ``Radius`` (``get_element_geometry:invalidRadius``),
    an arc ``N*Pitch`` not shorter than ``pi*Radius``, or a missing field.
    ``NumberOfElements`` must be a non-negative integer (see docs/PARITY.md).
    """
    n_value = _required_scalar(transducer, "NumberOfElements")
    if not (n_value >= 0 and n_value == math.floor(n_value)):
        raise ValueError(
            f"Transducer.NumberOfElements must be a non-negative integer, got {n_value!r}"
        )
    n = int(n_value)
    p = _required_scalar(transducer, "Pitch")
    i = np.arange(1, n + 1, dtype=np.float64)
    offset = i - (n + 1) / 2  # element index relative to the array centre

    if _is_curved(transducer):
        r_value = _field(transducer, "Radius")
        if r_value is None:
            raise ValueError("Transducer has no field Radius")  # MATLAB: reference error
        r_arr = np.asarray(r_value, dtype=np.float64)
        if not (r_arr.size == 1 and np.isfinite(r_arr).all() and (r_arr > 0).all()):
            raise ValueError("A curved array needs a positive finite Transducer.Radius.")
        r = float(r_arr.reshape(()))
        theta_max = n * p / (2 * r)
        if theta_max >= math.pi / 2:
            raise ValueError(
                "The arc of the curved array (N*Pitch) must be shorter than pi*Radius."
            )
        xc = -r * math.cos(theta_max)  # axial position of the centre of curvature
        theta = p * offset / r
        return ElementGeometry(
            x=xc + r * np.cos(theta),
            y=r * np.sin(theta),
            theta=theta,
            width=p,
            radius=r,
            apex=xc + r,
            center=xc,
            theta_max=theta_max,
        )

    return ElementGeometry(
        x=np.zeros(n),
        y=offset * p,
        theta=np.zeros(n),
        width=_required_scalar(transducer, "ElementWidth"),
        radius=math.inf,
        apex=0.0,
        center=-math.inf,
        theta_max=0.0,
    )
