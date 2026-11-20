"""MATLAB ``interp1``, ``griddedInterpolant`` (1-D) and ``pchip``.

Clean-room implementations from the MATLAB documentation (behaviour), the
Fritsch–Carlson (1980) and Fritsch–Butland (1984) papers, and Moler's
*Numerical Computing with MATLAB* (2004), ch. 3 (shape-preserving end slopes),
and de Boor, *A Practical Guide to Splines* (Hermite data → piecewise polynomial).
No MathWorks source was consulted. Bit-level agreement with MATLAB was found by
black-box comparison (``tests/oracle/test_numerics_interp.py``).

Linear interpolation uses ``v0*(1-t) + v1*t`` with ``t = (xq-x0)/(x1-x0)``:
this matched MATLAB R2025a ``interp1`` and ``griddedInterpolant`` on every
tested query, and it is exact at the nodes.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

Method = Literal["linear", "nearest", "pchip"]


def _intervals(x: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """Index k of the interval [x[k], x[k+1]] used for each query (clipped to the ends).

    NaN queries map to 0; callers mask them afterwards.
    """
    k = np.searchsorted(x, xq, side="right") - 1
    return np.clip(k, 0, x.size - 2)


def _linear(x: np.ndarray, v: np.ndarray, xq: np.ndarray) -> np.ndarray:
    k = _intervals(x, xq)
    x0, x1 = x[k], x[k + 1]
    v0, v1 = v[k], v[k + 1]
    t = (xq - x0) / (x1 - x0)
    if v.ndim > 1:
        t = t.reshape(t.shape + (1,) * (v.ndim - 1))
    return v0 * (1 - t) + v1 * t


def _nearest(x: np.ndarray, v: np.ndarray, xq: np.ndarray) -> np.ndarray:
    k = _intervals(x, xq)
    # Ties at the midpoint go to the upper point (MATLAB interp1 'nearest').
    upper = (xq - x[k]) >= (x[k + 1] - xq)
    return v[np.where(upper, k + 1, k)]


def pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Node derivatives of the shape-preserving piecewise cubic Hermite interpolant.

    Interior: weighted harmonic mean of the neighbouring secant slopes when they
    have the same sign, else 0 (Fritsch & Butland 1984). Ends: three-point
    non-centred formula, limited to preserve shape (Moler 2004, §3.4).
    ``y`` may have trailing dimensions (interpolated independently).
    """
    h = np.diff(x)
    if y.ndim > 1:
        h = h.reshape(h.shape + (1,) * (y.ndim - 1))
    delta = np.diff(y, axis=0) / h
    n = x.size
    d = np.zeros_like(delta, shape=y.shape)
    if n == 2:
        d[0] = delta[0]
        d[1] = delta[0]
        return d
    h0, h1 = h[:-1], h[1:]
    d0, d1 = delta[:-1], delta[1:]
    w1 = 2 * h1 + h0
    w2 = h1 + 2 * h0
    same = np.sign(d0) * np.sign(d1) > 0
    # Tiny secants can overflow w/δ to inf; the harmonic mean is then 0 (MATLAB agrees).
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        interior = (w1 + w2) / (w1 / d0 + w2 / d1)
    d[1:-1] = np.where(same, interior, 0.0)
    d[0] = _end_slope(h[0], h[1], delta[0], delta[1])
    d[-1] = _end_slope(h[-1], h[-2], delta[-1], delta[-2])
    return d


def _end_slope(h0, h1, del0, del1):
    d = ((2 * h0 + h1) * del0 - h0 * del1) / (h0 + h1)
    d = np.where(np.sign(d) != np.sign(del0), 0.0, d)
    limit = (np.sign(del0) != np.sign(del1)) & (np.abs(d) > np.abs(3 * del0))
    return np.where(limit, 3 * del0, d)


def _hermite(x: np.ndarray, y: np.ndarray, d: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """Evaluate the cubic Hermite interpolant (local power form, Horner)."""
    k = _intervals(x, xq)
    h = np.diff(x)
    if y.ndim > 1:
        h = h.reshape(h.shape + (1,) * (y.ndim - 1))
    delta = np.diff(y, axis=0) / h
    # Coefficients of y_k + d_k s + c_k s^2 + b_k s^3, s = xq - x_k, from the
    # divided differences of the Hermite data (de Boor, A Practical Guide to
    # Splines, ch. IV). This grouping of operations reproduced MATLAB's pchip
    # coefficients bit for bit; the algebraically equal (3δ - 2d0 - d1)/h did not.
    dzzdx = (delta - d[:-1]) / h
    dzdxdx = (d[1:] - delta) / h
    c = 2 * dzzdx - dzdxdx
    b = (dzdxdx - dzzdx) / h
    s = xq - x[k]
    if y.ndim > 1:
        s = s.reshape(s.shape + (1,) * (y.ndim - 1))
    return y[k] + s * (d[k] + s * (c[k] + s * b[k]))


def _prepare(x: ArrayLike, v: ArrayLike, sort: bool) -> tuple[np.ndarray, np.ndarray]:
    xa = np.asarray(x, dtype=np.float64)
    va = np.asarray(v)
    if va.dtype.kind not in "fc":
        va = va.astype(np.float64)
    if xa.ndim != 1:
        if xa.ndim == 2 and 1 in xa.shape:
            xa = xa.reshape(-1)
        else:
            raise ValueError("sample points x must be a vector")
    if va.ndim == 2 and 1 in va.shape and va.size == xa.size:
        va = va.reshape(-1)
    if va.shape[0] != xa.size:
        raise ValueError(f"v has {va.shape[0]} rows, x has {xa.size} points")
    if xa.size < 2:
        raise ValueError("at least two sample points are required")
    if np.any(np.isnan(xa)):
        raise ValueError("sample points must not be NaN")
    if sort:
        order = np.argsort(xa, kind="stable")
        xa, va = xa[order], va[order]
    if np.any(np.diff(xa) <= 0):
        if sort:
            raise ValueError("sample points must be unique")
        raise ValueError("sample points must be sorted in ascending order and unique")
    return xa, va


def _evaluate(
    xa: np.ndarray, va: np.ndarray, xq: np.ndarray, method: str, extrap: str | float | None
) -> np.ndarray:
    xq_flat = np.asarray(xq, dtype=np.float64).reshape(-1)
    if method == "linear":
        out = _linear(xa, va, xq_flat)
    elif method == "nearest":
        out = _nearest(xa, va, xq_flat)
    elif method == "pchip":
        out = _hermite(xa, va, pchip_slopes(xa, va), xq_flat)
    else:
        raise ValueError(f"unsupported method {method!r}")
    outside = (xq_flat < xa[0]) | (xq_flat > xa[-1])
    if extrap is None:
        out[outside] = np.nan
    elif extrap != "extrap":
        out[outside] = extrap
    out[np.isnan(xq_flat)] = np.nan
    return out


def interp1(
    x: ArrayLike,
    v: ArrayLike,
    xq: ArrayLike,
    method: Method = "linear",
    extrap: Literal["extrap"] | float | None = None,
) -> np.ndarray:
    """MATLAB ``interp1(x, v, xq, method, extrapolation)``.

    - Outside ``[min(x), max(x)]``: NaN by default for ``'linear'`` and
      ``'nearest'``; ``'pchip'`` extrapolates by default (as MATLAB).
      ``extrap='extrap'`` extrapolates with the method; a number fills.
    - ``x`` need not be sorted (sorted together with ``v``); duplicate points raise.
    - ``v`` may be a vector or have trailing dimensions (interpolated along axis 0).
    - ``'nearest'``: a query exactly halfway takes the upper point.
    - Output shape: ``xq.shape`` for vector ``v``, else ``xq.shape + v.shape[1:]``.
    """
    if extrap is None and method == "pchip":
        extrap = "extrap"
    xa, va = _prepare(x, v, sort=True)
    xq_arr = np.asarray(xq, dtype=np.float64)
    out = _evaluate(xa, va, xq_arr, method, extrap)
    return out.reshape(xq_arr.shape + va.shape[1:])


def pchip(x: ArrayLike, y: ArrayLike, xq: ArrayLike) -> np.ndarray:
    """MATLAB ``pchip(x, y, xq)``: shape-preserving cubic, extrapolates outside the range."""
    return interp1(x, y, xq, method="pchip", extrap="extrap")


class GriddedInterpolant:
    """MATLAB ``griddedInterpolant(x, v, method, extrapolation)`` on a 1-D grid.

    - ``x`` must be strictly increasing (MATLAB errors otherwise; no sorting).
    - Default extrapolation is the interpolation method itself: linear
      extrapolation for ``'linear'``, pchip for ``'pchip'``. ``extrap='none'``
      gives NaN outside the grid; a number fills.
    - Call the object with query points: ``F(xq)``.
    """

    def __init__(
        self,
        x: ArrayLike,
        v: ArrayLike,
        method: Method = "linear",
        extrap: str | float | None = None,
    ) -> None:
        if method not in ("linear", "nearest", "pchip"):
            raise ValueError(f"unsupported method {method!r}")
        self.grid, self.values = _prepare(x, v, sort=False)
        self.method = method
        if extrap is None or extrap == method:
            self._extrap: str | float | None = "extrap"
        elif extrap == "none":
            self._extrap = None
        elif isinstance(extrap, int | float):
            self._extrap = float(extrap)
        else:
            raise ValueError(f"unsupported extrapolation {extrap!r}")

    def __call__(self, xq: ArrayLike) -> np.ndarray:
        xq_arr = np.asarray(xq, dtype=np.float64)
        out = _evaluate(self.grid, self.values, xq_arr, self.method, self._extrap)
        return out.reshape(xq_arr.shape + self.values.shape[1:])


def gridded_interpolant(
    x: ArrayLike, v: ArrayLike, method: Method = "linear", extrap: str | float | None = None
) -> GriddedInterpolant:
    """Factory mirroring MATLAB's ``griddedInterpolant`` call."""
    return GriddedInterpolant(x, v, method, extrap)
