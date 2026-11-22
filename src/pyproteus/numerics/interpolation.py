"""MATLAB ``interp1``, ``griddedInterpolant`` (1-D) and ``pchip``.

Behaviour follows the MATLAB documentation (NaN outside the sample range by
default, ``'extrap'`` and fill values, sorting of unsorted sample points in
``interp1``, linear extrapolation by default in ``griddedInterpolant``). The
numerics are standard and self-contained (D-12, D-14):

- linear: the two-point Lagrange form ``v0*(1-t) + v1*t`` with
  ``t = (xq-x0)/(x1-x0)``, with no special cases. A query on a node next to an
  Inf or NaN value therefore gives NaN (``0*Inf``), where MATLAB returns the
  node value (B-70).
- pchip: SciPy's ``PchipInterpolator`` (Fritsch & Carlson 1980; Fritsch &
  Butland 1984 weighted harmonic mean for interior slopes; the shape-preserving
  three-point end slopes of Moler, *Numerical Computing with MATLAB*, 2004,
  §3.4).

No MathWorks source was consulted and no formula was chosen by comparing
MATLAB's bits. Agreement with MATLAB is checked at the D-13 float class
(``tests/oracle/test_numerics_interp.py``).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import PchipInterpolator

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


def _pchip(x: np.ndarray, v: np.ndarray, xq: np.ndarray) -> np.ndarray:
    if np.iscomplexobj(v):
        raise ValueError("pchip needs real sample values")
    # Tiny secants overflow w/δ in the harmonic mean; the slope is then 0 (correct).
    with np.errstate(over="ignore", divide="ignore"):
        return PchipInterpolator(x, v, axis=0, extrapolate=True)(xq)


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
