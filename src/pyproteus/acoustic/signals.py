"""Band-limited interpolation and the band-limited delta function.

Ports of ``evaluate_delta_function.m``, ``sinc_interpolation.m`` and
``resample_signal.m``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "evaluate_delta_function",
    "resample_signal",
    "sinc_interpolation",
    "sinc_matrix",
]

#: Elements of the dense sinc matrix built at once (2^22 doubles = 32 MB).
_CHUNK_ELEMENTS = 1 << 22


def evaluate_delta_function(x: ArrayLike, xi: ArrayLike, dx: ArrayLike, N: ArrayLike) -> np.ndarray:
    """Port of ``evaluate_delta_function.m``: band-limited delta centred at ``xi``.

    Wise, Cox, Jaros & Treeby (2019), JASA 146(1) 278-288, Eqs. 10, 12, 13.

    ``x`` is ``d × M`` (grid coordinates). ``xi``, ``dx`` and ``N`` hold one
    value per dimension (``d``-vector or ``d × 1``) or, with MATLAB implicit
    expansion, ``d × M`` (one centre per column, as ``update_sensor_fast.m``
    passes). Returns a 1-D array of ``M`` values (MATLAB: ``1 × M``).
    Per dimension: odd ``N`` uses the
    ``sin``-denominator form (Eq. 10), even ``N`` the ``tan`` form (Eq. 12);
    ``x == xi`` gives 1. As in MATLAB, a distance that is a nonzero multiple
    of ``N*dx`` divides by zero (Inf/NaN).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    d = x.shape[0]

    def per_dim(v: ArrayLike) -> np.ndarray:
        a = np.asarray(v, dtype=float)
        if a.ndim == 2 and a.shape[0] == d and a.shape[1] > 1:
            return a  # d × M: implicit expansion against x
        if d == 1 and a.ndim == 1 and a.size > 1:
            return a.reshape(1, -1)  # d = 1: a row of centres
        a = a.reshape(-1)
        if a.size == 1:
            a = np.repeat(a, d)
        if a.size != d:
            raise ValueError(f"expected {d} values (one per dimension) or d × M, got {a.size}")
        return a.reshape(-1, 1)

    xi_, dx_, N_ = per_dim(xi), per_dim(dx), per_dim(N)
    x, xi_ = np.broadcast_arrays(x, xi_)
    r = x - xi_
    even = np.mod(N_, 2) == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        num = np.sin(np.pi * r / dx_)
        arg = np.pi * r / (N_ * dx_)
        den = N_ * np.where(even, np.tan(arg), np.sin(arg))
        b = num / den
    b = np.array(b, copy=True)
    b[x == xi_] = 1.0
    return np.prod(b, axis=0)


def sinc_matrix(
    t_query: np.ndarray, t: np.ndarray, *, scale: float = 1.0, spacing: float = 1.0
) -> np.ndarray:
    """``sin(pi*u)/(pi*u)``, 1 where ``u == 0``.

    ``u = (t_query[:, None] - t[None, :]) * scale / spacing``.
    """
    u = (t_query[:, None] - t[None, :]) * scale / spacing
    return np.sinc(u)


def _apply_sinc(t_query: np.ndarray, t: np.ndarray, V: np.ndarray, **kw: float) -> np.ndarray:
    """``sinc_matrix(t_query, t, **kw) @ V``, in row chunks to bound memory."""
    out = np.empty((t_query.size, *V.shape[1:]), dtype=np.result_type(V, float))
    rows = max(1, _CHUNK_ELEMENTS // max(t.size, 1))
    # errstate: Apple Accelerate's matmul raises spurious FP flags on finite data.
    with np.errstate(all="ignore"):
        for start in range(0, t_query.size, rows):
            stop = start + rows
            out[start:stop] = sinc_matrix(t_query[start:stop], t, **kw) @ V
    return out


def _vector(a: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(a)
    if arr.ndim > 1 and sum(s > 1 for s in arr.shape) > 1:
        raise ValueError(f"{name} must be a vector, got shape {arr.shape}")
    return arr.reshape(-1)


def sinc_interpolation(X: ArrayLike, V: ArrayLike, Xq: ArrayLike) -> np.ndarray:
    """Port of ``sinc_interpolation.m``: band-limited interpolation of ``V = F(X)`` at ``Xq``.

    The sample spacing is ``mean(diff(X))``. Shapes:

    - ``V`` 1-D or a ``1 × n`` row: returns a 1-D array of ``len(Xq)`` values
      (MATLAB returns a row).
    - ``V`` an ``len(X) × k`` matrix, ``k ≥ 1`` (one signal per column, as
      PROTEUS passes ``transpose(sensed_p)``, also for a single bubble):
      returns ``k × len(Xq)``, as MATLAB.
    - dtype: ``float32`` ``V`` (k-Wave sensor data) gives ``float32`` output, as
      MATLAB's double × single product; the sum is accumulated in double and
      rounded once (MATLAB accumulates in single), so the two agree at the
      single-precision class, not 1e-9 (docs/PARITY.md).

    Divergence (D-15): the ``len(Xq) × len(X)`` sinc matrix is built in row
    blocks of at most 2^22 elements instead of at once (same values, bounded memory).
    """
    X = _vector(X, "X").astype(float)
    Xq = _vector(Xq, "Xq").astype(float)
    V = np.asarray(V)
    out_dtype = np.float32 if V.dtype == np.float32 else None
    # one sample: mean of an empty diff is NaN, as in MATLAB (output NaN)
    spacing = float(np.mean(np.diff(X))) if X.size > 1 else np.nan
    if V.ndim <= 1 or (V.ndim == 2 and V.shape[0] == 1):  # vector or MATLAB row
        out = _apply_sinc(Xq, X, V.reshape(-1, 1), spacing=spacing)[:, 0]
    elif V.ndim == 2:  # len(X) × k, also k = 1 (one bubble): k × len(Xq)
        out = _apply_sinc(Xq, X, V, spacing=spacing).T
    else:
        raise ValueError(f"V must be a vector or a matrix, got shape {V.shape}")
    return out.astype(out_dtype) if out_dtype is not None else out


def resample_signal(
    pressure_old: ArrayLike, fs_old: float, fs_new: float, disp_fig: bool = False
) -> np.ndarray:
    """Port of ``resample_signal.m``: sinc resampling of a signal from ``fs_old`` to ``fs_new``.

    Output times are ``(0:floor(t_old(end)*fs_new))/fs_new`` with
    ``t_old = (0:N-1)/fs_old``. Returns a 1-D array (MATLAB returns a row).

    ``disp_fig=True`` (MATLAB plots the signals and spectra) raises
    ``NotImplementedError``: plotting is not part of the library (D-15).
    """
    if disp_fig:
        raise NotImplementedError("resample_signal does not plot (disp_fig must be False)")
    p = _vector(pressure_old, "pressure_old")
    fs_old, fs_new = float(fs_old), float(fs_new)
    t_old = np.arange(p.size, dtype=float) / fs_old
    t_new = np.arange(np.floor(t_old[-1] * fs_new) + 1, dtype=float) / fs_new
    # MATLAB: x = (Ts - T)*fs_old; sin(pi*x)/(pi*x) with x == 0 -> 1.
    return _apply_sinc(t_new, t_old, p.reshape(-1, 1), scale=fs_old)[:, 0]
