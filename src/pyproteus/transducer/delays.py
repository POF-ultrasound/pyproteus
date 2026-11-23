"""Transmit delays (port of ``GUIfunctions/compute_delays.m``)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from pyproteus.config.models import Transmit

from ._fields import element_count, required, scalar
from ._trig import cosd, sind
from .geometry import get_element_geometry

__all__ = ["compute_delays", "transmit_delays"]


def transmit_delays(transducer: Any, lateral_focus: float, angle: float, c: float) -> np.ndarray:
    """Transmit delays [s] of every element, as a 1-D array (min delay 0).

    ``lateral_focus`` [m] is the focus (> 0) or virtual source (< 0) distance
    from the origin in the direction ``angle`` [deg] from the x axis; ±Inf (or
    NaN) is a plane wave for linear arrays and a diverging wave from the centre
    of curvature for curved arrays. RCA arrays get the same delays for columns
    and rows (length ``NumberOfElements``).
    """
    configuration = required(transducer, "Configuration", "Transducer")
    f, theta = float(lateral_focus), float(angle)

    if configuration == "Curved":
        el = get_element_geometry(transducer)
        if not math.isfinite(f):
            f = el.center  # centre of curvature (negative axial coordinate)
        f_x = f * float(sind(theta))  # lateral component of the focal point
        f_y = f * float(cosd(theta))  # axial component of the focal point
        delays = np.sqrt((f_x - el.y) ** 2 + (f_y - el.x) ** 2) / c
        if f > 0:
            delays = -delays  # reverse time for a positive focus
        return delays - np.min(delays) if delays.size else delays

    if configuration == "Linear":
        n = element_count(transducer)
    elif configuration == "RCA":
        n_total = element_count(transducer)
        if n_total % 2:
            raise ValueError("Number of elements in RCA must be even.")
        n = n_total // 2
    else:
        raise ValueError("Compute delays not supported for this transducer configuration")

    # MATLAB (-(N-1)/2:(N-1)/2)*Pitch: offsets i - (N+1)/2, i = 1..N
    x = (np.arange(1, n + 1, dtype=np.float64) - (n + 1) / 2) * scalar(transducer, "Pitch")
    f_x = f * float(sind(theta))  # lateral component of the focal point
    f_y = f * float(cosd(theta))  # axial component of the focal point
    if abs(f) < math.inf:
        delays = np.sqrt((f_x - x) ** 2 + f_y**2) / c  # focused
        if f > 0:
            delays = -delays
    else:
        delays = x * float(sind(theta)) / c  # unfocused (plane wave)
    if delays.size:
        delays = delays - np.min(delays)
    if configuration == "RCA":
        delays = np.concatenate([delays, delays])
    return delays


def compute_delays(transmit: Transmit, transducer: Any, medium: Any) -> Transmit:
    """Port of ``compute_delays.m``: ``Transmit.Delays`` (1×N row) from the focus and angle.

    Uses ``Transmit.LateralFocus``, ``Transmit.Angle`` [deg] and
    ``Medium.SpeedOfSound``. Returns a copy of ``transmit``. See
    :func:`transmit_delays` for the geometry. Raises ``ValueError`` for an
    unsupported ``Configuration`` (anything but ``'Linear'``, ``'Curved'``,
    ``'RCA'``), an odd RCA element count or a non-integer element count.
    """
    c = scalar(medium, "SpeedOfSound", "Medium")
    f = scalar(transmit, "LateralFocus", "Transmit")
    theta = scalar(transmit, "Angle", "Transmit")
    out = transmit.copy_struct()
    out.Delays = transmit_delays(transducer, f, theta, c).reshape(1, -1)
    return out
