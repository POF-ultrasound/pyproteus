"""Row-order mapping between MATLAB k-Wave and k-wave-python (>= 0.6.1).

Both k-Wave front ends attach one row per active voxel of a binary mask to
the time-series arrays (``source.p``, ``source.ux``/``uy``/``uz`` and the
recorded ``sensor_data.p``). They disagree on the order of those rows:

* MATLAB (and PROTEUS, which builds rows from ``find(mask)`` /
  ``intersect(mask_idx, ...)``) lists active voxels in **column-major
  (Fortran) order**: x varies fastest, then y, then z.
* k-wave-python's ``kspaceFirstOrder()`` (>= 0.6.1) documents **row-major
  (C) order** (z fastest). In 0.6.2 only the ``python`` backend follows it
  (C-order source rows in, C-order sensor rows out). The ``cpp`` backend
  takes source rows in F order and returns sensor data as ``(Nt, n_sensor)``
  with F-order columns (BUGS.md B-10). ``adapter.py`` applies the right
  mapping per backend; this module only provides the permutations.

Arrays here are always indexed ``[ix, iy, iz]`` (the same element positions
as in MATLAB); only the flattening order differs.

This module is meant to be lifted into ``pyproteus.backends.kwave`` (M3).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "c_to_f_rows",
    "f_to_c_rows",
    "matlab_find",
    "rows_c_to_f",
    "rows_f_to_c",
]


def matlab_find(mask: np.ndarray) -> np.ndarray:
    """MATLAB ``find(mask)`` for a 3-D (or N-D) array: 1-based column-major linear indices.

    The result is sorted ascending, exactly like MATLAB's ``find``.
    """
    return np.flatnonzero(np.asarray(mask, dtype=bool).ravel(order="F")) + 1


def f_to_c_rows(mask: np.ndarray) -> np.ndarray:
    """Permutation ``perm`` with ``rows_c = rows_f[perm]``.

    ``rows_f`` has one row per active voxel of ``mask`` in MATLAB ``find``
    (Fortran) order; ``rows_c`` has the same rows in C (row-major) order.
    """
    mask = np.asarray(mask, dtype=bool)
    shape = mask.shape
    f_lin = np.flatnonzero(mask.ravel(order="F"))  # sorted F linear indices
    c_lin = np.flatnonzero(mask.ravel(order="C"))  # sorted C linear indices
    # F linear index of every voxel, listed in C order:
    f_of_c = np.ravel_multi_index(np.unravel_index(c_lin, shape, order="C"), shape, order="F")
    # Its row in the F-ordered array is its rank among the sorted F indices.
    perm = np.searchsorted(f_lin, f_of_c)
    assert np.array_equal(f_lin[perm], f_of_c)
    return perm


def c_to_f_rows(mask: np.ndarray) -> np.ndarray:
    """Permutation ``perm`` with ``rows_f = rows_c[perm]`` (inverse of :func:`f_to_c_rows`)."""
    mask = np.asarray(mask, dtype=bool)
    shape = mask.shape
    f_lin = np.flatnonzero(mask.ravel(order="F"))
    c_lin = np.flatnonzero(mask.ravel(order="C"))
    c_of_f = np.ravel_multi_index(np.unravel_index(f_lin, shape, order="F"), shape, order="C")
    perm = np.searchsorted(c_lin, c_of_f)
    assert np.array_equal(c_lin[perm], c_of_f)
    return perm


def rows_f_to_c(rows_f: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Reorder per-voxel rows from MATLAB ``find`` order to C order."""
    rows_f = np.asarray(rows_f)
    perm = f_to_c_rows(mask)
    if rows_f.shape[0] != perm.size:
        raise ValueError(f"{rows_f.shape[0]} rows for a mask with {perm.size} active voxels")
    return rows_f[perm]


def rows_c_to_f(rows_c: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Reorder per-voxel rows from C order to MATLAB ``find`` order."""
    rows_c = np.asarray(rows_c)
    perm = c_to_f_rows(mask)
    if rows_c.shape[0] != perm.size:
        raise ValueError(f"{rows_c.shape[0]} rows for a mask with {perm.size} active voxels")
    return rows_c[perm]
