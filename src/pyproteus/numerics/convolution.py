"""MATLAB ``conv`` and ``convn`` with ``'full'``, ``'same'`` and ``'valid'``.

``'same'`` keeps the central part of the full convolution starting at
``floor(size(B)/2)`` (0-based) along each axis. For even kernel lengths this
is one sample later than ``scipy.signal.convolve(..., mode='same')``.

The full convolution is computed by direct summation (``method='direct'``),
so small values are not swamped by FFT round-off. Pass ``method='fft'`` for
large inputs where FFT round-off (relative to the largest output) is acceptable.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy import signal

Shape = Literal["full", "same", "valid"]


def _matlab_ndim(a: np.ndarray, ndim: int) -> np.ndarray:
    """Append trailing singleton axes (MATLAB arrays have implicit trailing 1s)."""
    return a.reshape(a.shape + (1,) * (ndim - a.ndim))


def convn(
    a: ArrayLike,
    b: ArrayLike,
    shape: Shape = "full",
    method: Literal["direct", "fft"] = "direct",
) -> np.ndarray:
    """MATLAB ``convn(A, B, shape)`` for N-D arrays.

    Arrays of different dimensionality are padded with *trailing* singleton
    axes (MATLAB semantics), so a 2-D kernel acts on the first two axes of a
    3-D array. **A 1-D NumPy array therefore counts as a MATLAB column**: for
    MATLAB ``convn(p, IR)`` with a row ``IR`` (``compute_RF``), pass ``IR`` with
    shape ``(1, n)``. Inputs are converted to float64 (or complex128).
    """
    aa = np.asarray(a)
    bb = np.asarray(b)
    dtype = np.result_type(aa.dtype, bb.dtype, np.float64)
    nd = max(aa.ndim, bb.ndim, 1)
    aa = _matlab_ndim(aa.astype(dtype, copy=False), nd)
    bb = _matlab_ndim(bb.astype(dtype, copy=False), nd)
    if shape not in ("full", "same", "valid"):
        raise ValueError(f"shape must be 'full', 'same' or 'valid', not {shape!r}")
    if aa.size == 0 or bb.size == 0:
        # MATLAB's sizes for empty convn inputs are irregular (conv errors);
        # PROTEUS never convolves empties, so refuse rather than guess.
        raise ValueError("convn of an empty array is not supported")
    full = signal.convolve(aa, bb, mode="full", method=method)
    if shape == "full":
        return full
    slices = []
    for na, nb in zip(aa.shape, bb.shape, strict=True):
        if shape == "same":
            start = nb // 2
            slices.append(slice(start, start + na))
        else:  # valid
            length = max(na - nb + 1, 0)
            slices.append(slice(nb - 1, nb - 1 + length))
    out = full[tuple(slices)]
    if shape == "valid":
        out = out.reshape(tuple(max(p - q + 1, 0) for p, q in zip(aa.shape, bb.shape, strict=True)))
    return out


def conv(u: ArrayLike, v: ArrayLike, shape: Shape = "full") -> np.ndarray:
    """MATLAB ``conv(u, v, shape)`` for vectors; returns a 1-D array.

    MATLAB's output orientation (row/column) has no 1-D NumPy equivalent;
    reshape the result if needed.
    """
    uu = np.asarray(u).reshape(-1)
    vv = np.asarray(v).reshape(-1)
    return convn(uu, vv, shape)
