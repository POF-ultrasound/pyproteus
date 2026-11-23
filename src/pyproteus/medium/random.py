"""Truncated standard normal draws (port of ``random_truncated_normal.m``)."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.special import erfc, erfcinv

from pyproteus.numerics import MatlabRNG

__all__ = ["random_truncated_normal"]

_SQRT2 = np.sqrt(2.0)


def _eps(x: float) -> float:
    """MATLAB ``eps(x)``: spacing from ``|x|`` to the next larger double (documented)."""
    return float(np.spacing(abs(x)))


def _normcdf(z: Any) -> Any:
    return 0.5 * erfc(-z / _SQRT2)


def random_truncated_normal(cutoff: float, sz: Any, rng: MatlabRNG) -> np.ndarray:
    """Standard normal draws truncated at ``±cutoff``, by inverse transform sampling.

    Port of PROTEUS ``random_truncated_normal.m``: ``p`` uniform in
    ``[Φ(lower), Φ(cutoff)]`` from ``rng.rand(sz)`` (MATLAB ``rand(sz)`` on the
    global stream; here the stream is explicit), ``x = -√2·erfcinv(2p)``,
    clipped to ``[-cutoff, cutoff]``. ``lower`` is the double just below
    ``-cutoff``, as in the PROTEUS code. ``erfc``/``erfcinv`` are SciPy's
    (D-13 float class against MATLAB).

    ``sz`` follows MATLAB size arguments (``[Nx, Ny, Nz]``; a scalar ``n`` gives
    ``n × n``). The output is filled in column-major order like MATLAB.

    Raises ``ValueError`` unless ``cutoff`` is a positive scalar (MATLAB:
    ``random_truncated_normal:invalidCutoff``). An infinite ``cutoff`` (MATLAB:
    untruncated ``randn``) raises ``NotImplementedError``: MATLAB's ``randn``
    algorithm is not documented, so its stream is not reproduced (ROADMAP §5;
    the default cutoff is finite).
    """
    c = np.asarray(cutoff)
    if not (c.size == 1 and c.ndim <= 2 and np.issubdtype(c.dtype, np.number) and c > 0):
        raise ValueError("The cutoff must be a positive scalar.")
    cutoff = float(c.reshape(-1)[0])
    if np.isinf(cutoff):
        raise NotImplementedError(
            "an infinite cutoff needs MATLAB randn, which pyproteus does not reproduce"
        )

    lower = -cutoff - _eps(-cutoff)
    if lower + _eps(lower) < -cutoff:
        lower = lower + _eps(lower)
    p_lower = _normcdf(lower)
    p_upper = _normcdf(cutoff)

    size = tuple(sz) if isinstance(sz, tuple | list | np.ndarray) else sz
    p = p_lower + rng.rand(size) * (p_upper - p_lower)
    x = -_SQRT2 * erfcinv(2 * p)
    return np.minimum(np.maximum(x, -cutoff), cutoff)
