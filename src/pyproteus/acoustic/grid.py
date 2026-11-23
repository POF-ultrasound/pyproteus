"""PROTEUS computational grid (ports of ``optimize_grid_size.m`` and ``define_grid.m``).

The grid is a small frozen dataclass instead of a MATLAB struct: coordinate
vectors are 1-D NumPy arrays (MATLAB 1×N rows) and the PML thicknesses live in
a :class:`PML` record. Field names are MATLAB's.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._struct import get as _get
from ._struct import scalar as _scalar

__all__ = ["PML", "Grid", "define_grid", "largest_prime_factor", "optimize_grid_size"]


def largest_prime_factor(n: int) -> int:
    """Largest prime factor of ``n`` (MATLAB ``max(factor(n))``).

    As MATLAB ``factor``: ``0`` → 0 and ``1`` → 1. Negative or non-integer
    ``n`` raise ``ValueError`` (MATLAB ``factor`` errors too).
    """
    if int(n) != n or n < 0:
        raise ValueError(f"n must be a nonnegative integer, got {n!r}")
    n = int(n)
    if n < 4:
        return n
    largest = 1
    p = 2
    while p * p <= n:
        while n % p == 0:
            largest = p
            n //= p
        p += 1 if p == 2 else 2
    return max(largest, n) if n > 1 else largest


def optimize_grid_size(n: int, expand_range: Any, max_prime: int) -> int:
    """Port of ``optimize_grid_size.m``.

    Searches the sizes ``n + (min(expand_range):max(expand_range))`` for the one
    with the smallest largest prime factor (first one on ties, as MATLAB
    ``min``). If even that factor exceeds ``max_prime``, the size grows from
    ``n + max(expand_range)`` until its largest prime factor is at most
    ``max_prime``.

    A scalar ``expand_range`` (as ``define_grid`` passes ``2*PMLmin``) gives a
    single candidate ``n + expand_range``.

    Divergence (D-15, docs/PARITY.md): ``max_prime < 2`` raises ``ValueError``;
    MATLAB loops forever (no integer ≥ 2 has all prime factors below 2).
    """
    r = np.atleast_1d(np.asarray(expand_range, dtype=float))
    lo, hi = float(r.min()), float(r.max())
    # MATLAB colon lo:hi = lo + (0:floor(hi - lo)); non-integer sizes raise in factors.
    candidates = [n + lo + k for k in range(int(np.floor(hi - lo)) + 1)]
    factors = [largest_prime_factor(m) for m in candidates]
    i = int(np.argmin(factors))  # first minimum, as MATLAB min
    best, fac = int(candidates[i]), factors[i]
    if fac > max_prime:
        if max_prime < 2:
            raise ValueError("max_prime must be at least 2 (MATLAB would loop forever)")
        best = int(candidates[-1])
        while largest_prime_factor(best) > max_prime:
            best += 1
    return best


@dataclass(frozen=True)
class PML:
    """Perfectly matched layer: thickness per axis [grid points] and absorption [Np/point]."""

    X_SIZE: int
    Y_SIZE: int
    Z_SIZE: int
    Alpha: float = 2.0


@dataclass(frozen=True, eq=False)
class Grid:
    """PROTEUS grid (``pgrid`` of ``define_grid.m``).

    ``x``, ``y``, ``z`` are 1-D coordinate vectors [m] of the core grid (PML
    excluded); ``full_size`` is ``(Mx, My, Mz)`` including the PML.
    """

    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    Nx: int
    Ny: int
    Nz: int
    dx: float
    dy: float
    dz: float
    dt: float
    full_size: tuple[int, int, int]
    PML: PML
    sensor_on_grid: bool | float = field(default=False)

    @property
    def shape(self) -> tuple[int, int, int]:
        """``(Nx, Ny, Nz)``."""
        return (self.Nx, self.Ny, self.Nz)


def define_grid(simulation_parameters: Any, geometry: Any) -> Grid:
    """Port of ``define_grid.m``.

    Grid points are integer multiples of ``GridSize`` that lie within or on the
    domain boundaries ``Geometry.Domain``. The full grid (core + PML) in each
    direction is ``optimize_grid_size(N, 2*PMLmin, MaxPrime)`` (defaults
    ``PMLmin = 15``, ``MaxPrime = 5``, used when the fields are absent), and
    the PML takes ``ceil((M - N)/2)`` points per side.

    Quirk Q-90: when ``M - N`` is odd the core grid shrinks by one point, so
    the last grid point inside the domain is dropped.
    """
    sp, D = simulation_parameters, _get(geometry, "Domain")
    dx = dy = dz = _scalar(_get(sp, "GridSize"))

    def bounds(lo: Any, hi: Any, d: float) -> tuple[int, int]:
        return int(np.ceil(_scalar(lo) / d)), int(np.floor(_scalar(hi) / d))

    X1, X2 = bounds(_get(D, "Xmin"), _get(D, "Xmax"), dx)
    Y1, Y2 = bounds(_get(D, "Ymin"), _get(D, "Ymax"), dy)
    Z1, Z2 = bounds(_get(D, "Zmin"), _get(D, "Zmax"), dz)
    Nx, Ny, Nz = X2 - X1 + 1, Y2 - Y1 + 1, Z2 - Z1 + 1

    max_prime = _scalar(_get(sp, "MaxPrime", 5))
    pml_min = _scalar(_get(sp, "PMLmin", 15))

    Mx = optimize_grid_size(Nx, 2 * pml_min, max_prime)
    My = optimize_grid_size(Ny, 2 * pml_min, max_prime)
    Mz = optimize_grid_size(Nz, 2 * pml_min, max_prime)

    pml = PML(
        X_SIZE=int(np.ceil((Mx - Nx) / 2)),
        Y_SIZE=int(np.ceil((My - Ny) / 2)),
        Z_SIZE=int(np.ceil((Mz - Nz) / 2)),
        Alpha=2.0,
    )
    Nx, Ny, Nz = Mx - 2 * pml.X_SIZE, My - 2 * pml.Y_SIZE, Mz - 2 * pml.Z_SIZE

    dt = 1 / _scalar(_get(sp, "SamplingRate"))
    sensor_on_grid = _get(sp, "SensorOnGrid")
    if sensor_on_grid is not None and not isinstance(sensor_on_grid, bool):
        sensor_on_grid = _scalar(sensor_on_grid)

    return Grid(
        x=np.arange(X1, X1 + Nx, dtype=float) * dx,
        y=np.arange(Y1, Y1 + Ny, dtype=float) * dy,
        z=np.arange(Z1, Z1 + Nz, dtype=float) * dz,
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        dx=dx,
        dy=dy,
        dz=dz,
        dt=dt,
        full_size=(Mx, My, Mz),
        PML=pml,
        sensor_on_grid=sensor_on_grid,
    )
