"""MATLAB ``round``."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def matlab_round(x: ArrayLike) -> np.ndarray:
    """Round to the nearest integer, halves away from zero (MATLAB ``round(x)``).

    Differs from ``np.round``/builtin ``round`` (halves to even) and from
    ``np.floor(x + 0.5)`` (wrong for 0.49999999999999994, where ``x + 0.5``
    rounds up to 1.0, and for negative halves).

    - ``round(-2.5) = -3``, ``round(2.5) = 3``, ``round(-0.4) = -0.0``
      (the sign of zero is kept, as in MATLAB).
    - NaN and ±Inf are returned unchanged; integer arrays are returned as is.
    - Complex input: real and imaginary parts are rounded separately.

    Returns an array of the input's floating dtype (MATLAB keeps the class).
    """
    a = np.asarray(x)
    if np.iscomplexobj(a):
        return matlab_round(a.real) + 1j * matlab_round(a.imag)
    if a.dtype.kind in "iub":
        return a.copy()
    if a.dtype.kind != "f":
        a = a.astype(np.float64)
    with np.errstate(invalid="ignore"):
        whole = np.trunc(a)
        frac = a - whole  # exact in binary floating point
        step = np.where(np.abs(frac) >= 0.5, np.sign(a), 0).astype(a.dtype)
        out = whole + step
    # whole + 0 turns -0.0 into +0.0; MATLAB keeps the sign of x.
    return np.copysign(out, a)
