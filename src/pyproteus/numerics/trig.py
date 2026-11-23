"""Trigonometry in degrees (MATLAB ``sind``, ``cosd``).

Documented MATLAB behaviour: exact results at integer multiples of 90 degrees
(``sind(180) == 0``, ``cosd(90) == 0``), unlike ``sin(deg2rad(x))``. This
matters where PROTEUS divides by ``cosd(theta)`` (beam projections) or tests
for zero. ±Inf and NaN give NaN. Implemented by exact
reduction modulo 360 and to the nearest multiple of 90; exact zeros are +0.
Not tuned to MATLAB's last bit elsewhere (D-12).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .rounding import matlab_round

__all__ = ["cosd", "sind"]


def _reduce(x: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """``x = 90*q + r`` with ``q`` in 0..3 (mod 4) and ``|r| <= 45`` (NaN for ±Inf/NaN)."""
    a = np.asarray(x, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        m = np.fmod(a, 360.0)  # exact
        n = matlab_round(m / 90.0)
        r = m - 90.0 * n
        q = np.mod(np.nan_to_num(n), 4.0)
    return q, np.deg2rad(r)


def _assemble(q: np.ndarray, s: np.ndarray, c: np.ndarray, shift: int) -> np.ndarray:
    k = np.mod(q + shift, 4.0)
    out = np.select([k == 0, k == 1, k == 2], [s, c, -s], default=-c)
    return out + 0.0  # -0.0 -> +0.0


def sind(x: ArrayLike) -> np.ndarray:
    """Sine of ``x`` in degrees, exact at multiples of 90."""
    q, r = _reduce(x)
    return _assemble(q, np.sin(r), np.cos(r), 0)


def cosd(x: ArrayLike) -> np.ndarray:
    """Cosine of ``x`` in degrees, exact at multiples of 90."""
    q, r = _reduce(x)
    return _assemble(q, np.sin(r), np.cos(r), 1)
