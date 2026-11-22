"""MATLAB ``interp1``, ``griddedInterpolant`` (1-D) and ``pchip``.

Behaviour follows the MATLAB documentation (NaN outside the sample range by
default, ``'extrap'`` and fill values, sorting of unsorted sample points in
``interp1``, linear extrapolation by default in ``griddedInterpolant``). The
numerics are standard and self-contained (D-12, D-14):

- linear: the two-point Lagrange form ``v0*(1-t) + v1*t`` with
  ``t = (xq-x0)/(x1-x0)``, with no special cases. A query on a node next to an
  Inf or NaN value therefore gives NaN (``0*Inf``), where MATLAB returns the
  node value (B-70).
- pchip: the cubic Hermite interpolant in the textbook local power form, with node
  slopes from Fritsch & Butland (1984) (weighted harmonic mean of the secants,
  0 at a sign change) and the shape-preserving three-point end slopes of Moler,
  *Numerical Computing with MATLAB*, 2004, §3.4. These are the same rules as
  SciPy's ``PchipInterpolator``; the own implementation is kept because SciPy
  rejects non-finite and complex sample values, which MATLAB accepts (NaN/Inf
  propagate; real and imaginary parts are interpolated separately). A NaN
  secant makes its node slopes NaN, so ``griddedInterpolant`` can give NaN
  next to a NaN sample where MATLAB gives a number (B-71).
- ±Inf queries with extrapolation: linear gives the limit of the end line
  (its value if flat, else ±Inf by the sign of the slope); pchip evaluates the local power form by
  Horner's rule, so a flat or cancelling cubic gives NaN, as in MATLAB.

No MathWorks source was consulted and no formula was chosen by comparing
MATLAB's bits. Agreement with MATLAB is checked at the D-13 float class
(``tests/oracle/test_numerics_interp.py``).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

Method = Literal["linear", "nearest", "pchip"]


def _intervals(x: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """Index k of the interval [x[k], x[k+1]] used for each query (clipped to the ends).

    NaN queries map to the last interval; callers mask them afterwards.
    """
    k = np.searchsorted(x, xq, side="right") - 1
    return np.clip(k, 0, x.size - 2)


def _linear(x: np.ndarray, v: np.ndarray, xq: np.ndarray) -> np.ndarray:
    k = _intervals(x, xq)
    t = (xq - x[k]) / (x[k + 1] - x[k])
    if v.ndim > 1:
        t = t.reshape(t.shape + (1,) * (v.ndim - 1))
    with np.errstate(invalid="ignore"):
        return v[k] * (1 - t) + v[k + 1] * t


def _nearest(x: np.ndarray, v: np.ndarray, xq: np.ndarray) -> np.ndarray:
    k = _intervals(x, xq)
    # Ties at the midpoint go to the upper point (MATLAB interp1 'nearest').
    upper = (xq - x[k]) >= (x[k + 1] - xq)
    return v[np.where(upper, k + 1, k)]


def _trailing(a: np.ndarray, ndim: int) -> np.ndarray:
    """Append singleton axes so a per-query/per-interval array broadcasts against v."""
    return a.reshape(a.shape + (1,) * (ndim - 1))


def pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Node slopes of the shape-preserving piecewise cubic Hermite interpolant.

    Interior: weighted harmonic mean of the neighbouring secants when they have
    the same sign, else 0 (Fritsch & Butland 1984). Ends: three-point formula,
    limited to preserve shape (Moler 2004, §3.4). ``y`` may have trailing axes.
    A NaN secant makes the slopes at its nodes NaN (interior and end), so NaN
    propagates to the neighbouring intervals (B-71).
    """
    h = _trailing(np.diff(x), y.ndim)
    with np.errstate(invalid="ignore", over="ignore"):
        delta = np.diff(y, axis=0) / h
    d = np.zeros_like(delta, shape=y.shape)
    if x.size == 2:
        d[:] = delta[0]
        return d
    h0, h1 = h[:-1], h[1:]
    d0, d1 = delta[:-1], delta[1:]
    w1, w2 = 2 * h1 + h0, h1 + 2 * h0
    same = np.sign(d0) * np.sign(d1) > 0
    # Tiny secants can overflow w/δ to Inf; the harmonic mean is then 0.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        interior = (w1 + w2) / (w1 / d0 + w2 / d1)
    d[1:-1] = np.where(same, interior, 0.0)
    d[1:-1] = np.where(np.isnan(d0) | np.isnan(d1), np.nan, d[1:-1])  # NaN propagates (B-71)
    d[0] = _end_slope(h[0], h[1], delta[0], delta[1])
    d[-1] = _end_slope(h[-1], h[-2], delta[-1], delta[-2])
    return d


def _end_slope(h0, h1, del0, del1):
    """End slope: derivative of the quadratic through the three end points,
    ``((2 h0 + h1) del0 - h0 del1) / (h0 + h1)``, set to 0 if its sign differs
    from ``del0`` and to ``3 del0`` if the secants change sign and it exceeds
    ``3 |del0|`` in magnitude (Moler 2004, §3.4)."""
    with np.errstate(invalid="ignore", over="ignore"):
        e = ((2 * h0 + h1) * del0 - h0 * del1) / (h0 + h1)
        wrong_sign = np.sign(e) != np.sign(del0)
        too_steep = (np.sign(del0) != np.sign(del1)) & (np.abs(e) > np.abs(3 * del0))
    out = np.where(wrong_sign, 0.0, np.where(too_steep, 3 * del0, e))
    return np.where(np.isnan(e), np.nan, out)  # a NaN end slope propagates


def _pchip_coefficients(x: np.ndarray, v: np.ndarray) -> list[np.ndarray]:
    """Local power-form coefficients [c0, c1, c2, c3] per interval, in s = xq - x_k.

    Textbook cubic Hermite (e.g. Moler 2004, §3.4):
    c0 = y_k, c1 = d_k, c2 = (3δ - 2d_k - d_{k+1})/h, c3 = (d_k - 2δ + d_{k+1})/h².
    """
    d = pchip_slopes(x, v)
    h = _trailing(np.diff(x), v.ndim)
    with np.errstate(invalid="ignore", over="ignore"):
        delta = np.diff(v, axis=0) / h
        c2 = (3 * delta - 2 * d[:-1] - d[1:]) / h
        c3 = (d[:-1] - 2 * delta + d[1:]) / (h * h)
    return [v[:-1], d[:-1], c2, c3]


def _pchip(x: np.ndarray, v: np.ndarray, xq: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(v):  # MATLAB interpolates real and imaginary parts separately
        return _pchip(x, v.real, xq) + 1j * _pchip(x, v.imag, xq)
    k = _intervals(x, xq)
    c0, c1, c2, c3 = (c[k] for c in _pchip_coefficients(x, v))
    s = _trailing(xq - x[k], v.ndim)
    with np.errstate(invalid="ignore", over="ignore"):
        return c0 + s * (c1 + s * (c2 + s * c3))


def _infinite_queries(x, v, xq, out) -> None:
    """Linear extrapolation at ±Inf queries: the limit of the end line (its value
    if flat, else ±Inf by the sign of the slope), where the blend gives Inf - Inf.
    Complex values: real and imaginary parts separately."""
    if np.iscomplexobj(v):
        re, im = out.real.copy(), out.imag.copy()
        _infinite_queries(x, v.real, xq, re)
        _infinite_queries(x, v.imag, xq, im)
        out.real[...], out.imag[...] = re, im  # re + 1j*im would turn 0*Inf into NaN
        return
    for sign, k in ((1.0, x.size - 2), (-1.0, 0)):
        mask = xq == sign * np.inf
        if mask.any():
            with np.errstate(invalid="ignore", over="ignore"):
                rise = v[k + 1] - v[k]  # same sign as the slope (h > 0), cannot underflow
                limit = np.where(rise != 0, np.sign(rise) * sign * np.inf, v[k])  # NaN stays NaN
            out[mask] = limit


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
        _infinite_queries(xa, va, xq_flat, out)
    elif method == "nearest":
        out = _nearest(xa, va, xq_flat)
    elif method == "pchip":
        out = _pchip(xa, va, xq_flat)
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
    xq_arr = np.asarray(xq, dtype=np.float64)
    if va.ndim > 1 and xq_arr.ndim == 2 and 1 in xq_arr.shape:
        xq_arr = xq_arr.reshape(-1)  # MATLAB: vector xq with matrix v gives numel(xq)×m
    if method == "pchip" and np.isnan(va).any():
        # MATLAB interp1 drops NaN sample values for 'pchip' (with a warning),
        # column by column for matrix v; 'linear' and 'nearest' keep them.
        cols = va.reshape(va.shape[0], -1)
        out = np.empty((xq_arr.size, cols.shape[1]), dtype=np.result_type(va, np.float64))
        for j in range(cols.shape[1]):
            keep = ~np.isnan(cols[:, j])
            if keep.sum() < 2:
                raise ValueError("fewer than two non-NaN sample values")
            out[:, j] = _evaluate(xa[keep], cols[keep, j], xq_arr, method, extrap)
        return out.reshape(xq_arr.shape + va.shape[1:])
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
    - ``'pchip'`` keeps NaN sample values (they propagate); ``interp1``/``pchip``
      drop them instead.
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
