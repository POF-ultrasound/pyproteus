"""MATLAB ``sort`` with index output.

Indices are **0-based** (Python convention): ``s, i = sort(x)`` gives
``x[i] == s``, where MATLAB gives ``x(I) == S`` with ``I = i + 1``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike


def first_nonsingleton_axis(a: np.ndarray) -> int:
    """MATLAB's default dimension: the first axis whose length is not 1 (0 if none)."""
    for ax, n in enumerate(a.shape):
        if n != 1:
            return ax
    return 0


def _resolve_axis(a: np.ndarray, axis: int | None) -> int:
    if a.ndim == 0:
        return 0
    return first_nonsingleton_axis(a) if axis is None else axis % a.ndim


def argsort(
    x: ArrayLike,
    axis: int | None = None,
    direction: Literal["ascend", "descend"] = "ascend",
) -> np.ndarray:
    """0-based indices of MATLAB ``[~, I] = sort(x, dim, direction)``.

    - Stable in both directions: equal elements keep their original order
      (also for ``'descend'``, which is *not* a reversed ascending sort).
    - NaNs go last for ``'ascend'`` and first for ``'descend'``, each group
      in original order. ``-0.0`` and ``0.0`` compare equal.
    - ``axis=None`` uses MATLAB's default: the first non-singleton axis.
    """
    a = np.asarray(x)
    if np.iscomplexobj(a):
        raise TypeError("complex sort (MATLAB: by abs, then angle) is not implemented")
    if a.ndim == 0:
        return np.zeros((), dtype=np.intp)
    ax = _resolve_axis(a, axis)
    if direction == "ascend":
        return np.argsort(a, axis=ax, kind="stable")
    if direction != "descend":
        raise ValueError(f"direction must be 'ascend' or 'descend', not {direction!r}")
    # Stable ascending sort of the reversed array, reversed back: equal values
    # (and NaNs, which NumPy sorts last) end up in original order, NaNs first.
    n = a.shape[ax]
    rev = np.argsort(np.flip(a, axis=ax), axis=ax, kind="stable")
    return (n - 1) - np.flip(rev, axis=ax)


def sort(
    x: ArrayLike,
    axis: int | None = None,
    direction: Literal["ascend", "descend"] = "ascend",
) -> tuple[np.ndarray, np.ndarray]:
    """MATLAB ``[S, I] = sort(x, dim, direction)`` with 0-based ``I``. See :func:`argsort`."""
    a = np.asarray(x)
    idx = argsort(a, axis=axis, direction=direction)
    if a.ndim == 0:
        return a.copy(), idx
    return np.take_along_axis(a, idx, axis=_resolve_axis(a, axis)), idx
