"""MATLAB colon operator ``a:d:b`` (documented semantics)."""

from __future__ import annotations

import numpy as np

__all__ = ["colon"]

_EPS = np.finfo(np.float64).eps


def colon(a: float, d: float, b: float) -> np.ndarray:
    """MATLAB ``a:d:b`` for real scalars: ``[a, a+d, ..., a+m*d]``, ``m = fix((b-a)/d)``.

    Documented MATLAB semantics; empty when the range is empty (``d > 0`` and
    ``b < a``, or ``d < 0`` and ``b > a``). ``d = 0`` gives an empty array, as in
    MATLAB.

    Count rule (own choice, not derived from MATLAB): ``q = (b-a)/d`` carries a
    rounding error of a few ulp of ``max(|q|, max(|a|,|b|)/|d|)``. If ``q`` lies
    below an integer by less than ``4*eps`` times that scale, the integer is
    used, so a range meant to end on ``b`` keeps its last element despite that
    error (``0:0.1:0.3`` has 4 elements). Anything further below is a genuine
    shortfall (``0:1:3-1e-14`` has 3 elements). Values are ``a + k*d``; MATLAB
    may differ in the last bits (D-13 float class).
    """
    a, d, b = float(a), float(d), float(b)
    if not (np.isfinite(a) and np.isfinite(d) and np.isfinite(b)):
        raise ValueError("colon: finite a, d and b expected")
    if d == 0:
        return np.zeros(0)
    q = (b - a) / d
    if q < 0:
        return np.zeros(0)
    scale = max(abs(q), max(abs(a), abs(b)) / abs(d), 1.0)
    m = np.floor(q)
    if np.ceil(q) - q <= 4 * _EPS * scale:
        m = np.ceil(q)
    return a + np.arange(int(m) + 1) * d
