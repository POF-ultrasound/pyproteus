"""MATLAB ``ifft(X, [], dim, 'symmetric')`` and ``hilbert``.

Uses NumPy's pocketfft; MATLAB uses FFTW. Results agree to rounding error,
checked at the D-13 float class (≤ 1e-9 relative to the signal maximum).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from .sorting import first_nonsingleton_axis


def _axis(a: np.ndarray, axis: int | None) -> int:
    if axis is None:
        return first_nonsingleton_axis(a)
    if not -a.ndim <= axis < a.ndim:
        raise ValueError(f"axis {axis} out of range for {a.ndim}-D array")
    return axis % a.ndim


def ifft_symmetric(x: ArrayLike, n: int | None = None, axis: int | None = None) -> np.ndarray:
    """MATLAB ``ifft(X, n, dim, 'symmetric')``: real inverse FFT of a conjugate-symmetric X.

    Only the first ``n//2 + 1`` bins along ``axis`` are used (the rest are taken
    as the conjugate mirror); the imaginary parts of the DC bin and, for even
    ``n``, of the Nyquist bin are ignored. ``n`` defaults to ``X.shape[axis]``;
    a smaller ``n`` truncates and a larger one zero-pads X before the transform,
    as MATLAB does. ``axis=None`` is MATLAB's default dimension (first
    non-singleton axis). The MATLAB call ``ifft(X,[],2,'symmetric')`` is
    ``ifft_symmetric(X, axis=1)``.
    """
    a = np.asarray(x)
    if a.ndim == 0:
        a = a.reshape(1)
    ax = _axis(a, axis)
    m = a.shape[ax]
    if n is None:
        n = m
    if n < 1:
        raise ValueError("n must be positive")
    if n != m:
        if n < m:
            a = np.take(a, np.arange(n), axis=ax)
        else:
            pad = [(0, 0)] * a.ndim
            pad[ax] = (0, n - m)
            a = np.pad(a, pad)
    half = np.take(a, np.arange(n // 2 + 1), axis=ax)
    return np.fft.irfft(half, n=n, axis=ax)


def hilbert(x: ArrayLike, axis: int | None = None) -> np.ndarray:
    """MATLAB ``hilbert(x)``: analytic signal ``x + i*H{x}``.

    ``axis=None`` follows MATLAB: along the first non-singleton axis, i.e.
    along the length of a vector (row or column) and down the columns
    (axis 0) of a matrix. SciPy's ``scipy.signal.hilbert`` defaults to the last
    axis instead. Complex input: MATLAB uses only the real part (with a
    warning); so does this function.
    """
    a = np.asarray(x)
    if np.iscomplexobj(a):
        a = a.real
    a = a.astype(np.float64, copy=False)
    if a.ndim == 0:
        a = a.reshape(1)
    ax = _axis(a, axis)
    n = a.shape[ax]
    if n == 0:
        return a.astype(np.complex128)
    spectrum = np.fft.fft(a, axis=ax)
    h = np.zeros(n)
    if n % 2 == 0:
        h[0] = h[n // 2] = 1.0
        h[1 : n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (n + 1) // 2] = 2.0
    shape = [1] * a.ndim
    shape[ax] = n
    return np.fft.ifft(spectrum * h.reshape(shape), axis=ax)
