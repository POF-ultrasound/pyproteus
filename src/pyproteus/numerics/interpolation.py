"""MATLAB ``interp1``, ``griddedInterpolant`` (1-D) and ``pchip``.

Clean-room implementations from the MATLAB documentation (behaviour), the
Fritsch–Carlson (1980) and Fritsch–Butland (1984) papers, and Moler's
*Numerical Computing with MATLAB* (2004), ch. 3 (shape-preserving end slopes),
and de Boor, *A Practical Guide to Splines* (Hermite data → piecewise polynomial).
No MathWorks source was consulted. Where several algebraically equal forms
exist, the one used here was chosen by black-box comparison with MATLAB outputs
(``tests/oracle/test_numerics_interp.py``).

Linear interpolation uses ``v0*(1-t) + v1*t`` with ``t = (xq-x0)/(x1-x0)``,
with two corrections found by black-box comparison: flat segments
(``v0 == v1``) return ``v0``, and on a node a NaN caused only by a zero weight
times Inf/NaN is replaced by the node value. This matched MATLAB R2025a
``interp1`` and ``griddedInterpolant`` on every tested query.
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
    at_left, at_right = xq == x0, xq == x1
    if v.ndim > 1:
        expand = (1,) * (v.ndim - 1)
        t = t.reshape(t.shape + expand)
        at_left = at_left.reshape(at_left.shape + expand)
        at_right = at_right.reshape(at_right.shape + expand)
    with np.errstate(invalid="ignore"):
        out = v0 * (1 - t) + v1 * t
    # Matched MATLAB by black-box comparison (all {1, ±0, ±Inf, NaN, 2} triples):
    # - on a node, a NaN produced only by the zero weight (0 * Inf, 0 * NaN) is
    #   replaced by the node value; otherwise the blend stands (so the sign of a
    #   zero follows the blend, e.g. v = [1, -0] at the right node gives +0);
    # - flat segments (v0 == v1) return v0 exactly (the blend can be 1 ulp off).
    blend_nan = np.isnan(out)
    out = np.where(blend_nan & at_left, v0, out)
    out = np.where(blend_nan & at_right, v1, out)
    return np.where(v0 == v1, v0, out)


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
    """End derivative from the end interval (h0, del0) and its neighbour (h1, del1).

    Formula (Moler 2004, §3.4, text): the derivative at the end of the quadratic
    through the first three points, d = ((2 h0 + h1) del0 - h0 del1) / (h0 + h1),
    set to 0 if its sign differs from del0, and to 3 del0 if the secants change
    sign and |d| > |3 del0| (keeps the end interval monotone).
    """
    quadratic = ((2 * h0 + h1) * del0 - h0 * del1) / (h0 + h1)
    wrong_sign = np.sign(quadratic) != np.sign(del0)
    too_steep = (np.sign(del0) != np.sign(del1)) & (np.abs(quadratic) > np.abs(3 * del0))
    return np.where(wrong_sign, 0.0, np.where(too_steep, 3 * del0, quadratic))


def _hermite(x: np.ndarray, y: np.ndarray, d: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """Evaluate the cubic Hermite interpolant (local power form, Horner)."""
    k = _intervals(x, xq)
    h = np.diff(x)
    if y.ndim > 1:
        h = h.reshape(h.shape + (1,) * (y.ndim - 1))
    delta = np.diff(y, axis=0) / h
    # Newton form of the cubic with double nodes x_k, x_k, x_{k+1}, x_{k+1}
    # (de Boor, A Practical Guide to Splines, ch. I):
    #   p(s) = y_k + d_k s + [x_k,x_k,x_{k+1}] s^2 + [x_k,x_k,x_{k+1},x_{k+1}] s^2 (s - h)
    # with second divided differences left = (δ - d_k)/h, right = (d_{k+1} - δ)/h
    # and third divided difference (right - left)/h. Expanding gives the power-
    # form coefficients below. Black-box comparison with MATLAB's unmkpp(pchip)
    # showed this grouping is bit-identical, while the algebraically equal
    # (3δ - 2d_k - d_{k+1})/h is not. Clean-room status: docs/dev/DECISIONS.md D-8.
    dd2_left = (delta - d[:-1]) / h
    dd2_right = (d[1:] - delta) / h
    c = 2 * dd2_left - dd2_right
    b = (dd2_right - dd2_left) / h
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
    - ``'pchip'`` drops sample points whose value is NaN (MATLAB does, with a
      warning); ``'linear'``/``'nearest'`` keep them.
    - Output shape: ``xq.shape`` for vector ``v``, else ``xq.shape + v.shape[1:]``
      (a row/column ``xq`` counts as a vector, as in MATLAB).
    """
    if extrap is None and method == "pchip":
        extrap = "extrap"
    xa, va = _prepare(x, v, sort=True)
    if method == "pchip" and va.ndim == 1 and np.isnan(va).any():
        # MATLAB interp1 drops NaN sample values for 'pchip' (with a warning);
        # 'linear' and 'nearest' keep them.
        keep = ~np.isnan(va)
        if keep.sum() < 2:
            raise ValueError("fewer than two non-NaN sample values")
        xa, va = xa[keep], va[keep]
    xq_arr = np.asarray(xq, dtype=np.float64)
    if va.ndim > 1 and xq_arr.ndim == 2 and 1 in xq_arr.shape:
        xq_arr = xq_arr.reshape(-1)  # MATLAB: vector xq with matrix v gives numel(xq)×m
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
