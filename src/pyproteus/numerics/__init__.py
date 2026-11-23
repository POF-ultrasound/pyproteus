"""MATLAB semantics helpers (ROADMAP §5).

Every operation where NumPy/SciPy silently differ from MATLAB goes through
these helpers. Conventions:

- Indices are **0-based**; linear indices follow MATLAB's **column-major** order.
- ``axis=None`` means MATLAB's default dimension (first non-singleton axis)
  where MATLAB has one.
- Functions keep MATLAB names in snake_case.

See ``docs/PARITY.md`` for the oracle status of each helper.
"""

from .colon import colon
from .convolution import conv, convn
from .indexing import find, ind2sub, sub2ind
from .interpolation import GriddedInterpolant, gridded_interpolant, interp1, pchip
from .ode45 import ODEResult, ODEStats, ode45, odeset
from .random import MatlabRNG, rng
from .rounding import matlab_round
from .sorting import argsort, first_nonsingleton_axis, sort
from .spectral import hilbert, ifft_symmetric

__all__ = [
    "GriddedInterpolant",
    "MatlabRNG",
    "ODEResult",
    "ODEStats",
    "argsort",
    "colon",
    "conv",
    "convn",
    "find",
    "first_nonsingleton_axis",
    "gridded_interpolant",
    "hilbert",
    "ifft_symmetric",
    "ind2sub",
    "interp1",
    "matlab_round",
    "ode45",
    "odeset",
    "pchip",
    "rng",
    "sort",
    "sub2ind",
]
