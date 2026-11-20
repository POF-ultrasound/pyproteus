"""Column-major (Fortran-order) index helpers: MATLAB ``find``, ``sub2ind``, ``ind2sub``.

Convention: **all indices are 0-based** (Python), but linear indices follow
MATLAB's **column-major** order. A MATLAB linear index ``k`` corresponds to
``k - 1`` here, and ``A(k) == A.ravel(order="F")[k - 1]``. Subscripts are
0-based too: ``A(i, j, k) == A[i - 1, j - 1, k - 1]``.

Convert at the boundary only (reading/writing MATLAB files); keep 0-based
values inside pyproteus.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike


def _check_integral(a: np.ndarray, what: str) -> None:
    if a.dtype.kind in "iub" or a.size == 0:
        return
    if not np.all(np.isfinite(a)) or not np.all(np.mod(a, 1) == 0):
        raise ValueError(f"{what} must be integers")


def find(
    x: ArrayLike,
    n: int | None = None,
    direction: Literal["first", "last"] = "first",
) -> np.ndarray:
    """0-based column-major linear indices of the nonzero elements (MATLAB ``find``).

    NaN counts as nonzero, as in MATLAB. ``n`` and ``direction`` follow
    ``find(x, n, 'first'|'last')``; indices are always returned in ascending order.
    """
    a = np.asarray(x)
    idx = np.flatnonzero(a.ravel(order="F"))
    if n is not None:
        if n < 1 or int(n) != n:
            raise ValueError("n must be a positive integer (MATLAB find)")
        if direction == "first":
            idx = idx[:n]
        elif direction == "last":
            idx = idx[max(idx.size - n, 0) :]
        else:
            raise ValueError(f"direction must be 'first' or 'last', not {direction!r}")
    return idx


def sub2ind(shape: Sequence[int], *subs: ArrayLike) -> np.ndarray:
    """0-based subscripts → 0-based column-major linear index (MATLAB ``sub2ind``).

    Out-of-range subscripts raise ``ValueError`` (MATLAB: "Index exceeds array
    bounds"). Subscripts broadcast against each other.
    """
    if len(subs) != len(shape):
        raise ValueError(f"expected {len(shape)} subscript arrays, got {len(subs)}")
    arrays = np.broadcast_arrays(*(np.asarray(s) for s in subs))
    for s in arrays:
        _check_integral(s, "subscripts")
    ints = [s.astype(np.intp) for s in arrays]
    return np.ravel_multi_index(ints, tuple(shape), order="F")


def ind2sub(shape: Sequence[int], ind: ArrayLike) -> tuple[np.ndarray, ...]:
    """0-based column-major linear index → tuple of 0-based subscripts (MATLAB ``ind2sub``)."""
    a = np.asarray(ind)
    _check_integral(a, "linear indices")
    return np.unravel_index(a.astype(np.intp), tuple(shape), order="F")
