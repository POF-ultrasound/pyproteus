"""k-Wave pieces of PROTEUS' full simulator.

Ports of ``update_source.m`` and ``compute_self_sense_pressure.m`` (oracle
8d37938, ``acoustic-module``). ``full_simulator.m`` itself comes in M3 wave 2.

Index convention (pyproteus): linear grid indices are **0-based column-major**
(MATLAB ``find``/``MB.idx`` minus 1); bubble positions are ``(N_MB, 3)``.
Sparse weight matrices may be SciPy sparse or dense arrays.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from pyproteus.backends.kwave import KGrid
from pyproteus.numerics import ind2sub, sub2ind

from ._struct import get as _get
from .grid import optimize_grid_size
from .signals import evaluate_delta_function
from .simulation import run_simulation

__all__ = ["SELF_SENSE_PML", "compute_self_sense_pressure", "self_sense_grid_size", "update_source"]

#: PML of the local single-bubble grids (fixed in ``compute_self_sense_pressure.m``).
SELF_SENSE_PML = {"X_SIZE": 10, "Y_SIZE": 10, "Z_SIZE": 10, "Alpha": 2.0}
#: Half-width of the truncated band-limited delta function (``update_sensor.m``).
_DELTA_HALF_WIDTH = 4


def _shape(grid: Any) -> tuple[int, int, int]:
    return (int(_get(grid, "Nx")), int(_get(grid, "Ny")), int(_get(grid, "Nz")))


def _at(field: Any, idx: np.ndarray) -> np.ndarray:
    """``field(idx)`` with 0-based column-major ``idx``; a scalar field broadcasts."""
    a = np.asarray(field)
    if a.size == 1:
        return np.full(idx.shape, a.reshape(-1)[0], dtype=np.float64)
    return a.ravel(order="F")[idx]


def update_source(
    source: Mapping[str, Any] | None,
    mass_source: Any,
    source_weights: Any,
    source_mask_idx_new: Any,
    grid: Any,
    medium: Any,
) -> dict[str, Any] | None:
    """Port of ``update_source.m``: add bubble mass sources as pressure sources.

    ``source``: the transducer source (``p_mask``/``p`` and/or ``u_mask``/
    ``ux``...), or ``None`` (MATLAB ``[]``); not modified (MATLAB value
    semantics: a new dict is returned). ``mass_source``: ``(N_MB, Nt)`` [kg/s].
    ``source_weights``: ``(n_new, N_MB)`` (MATLAB ``transpose(sensor_weights)``),
    sparse or dense. ``source_mask_idx_new``: 0-based column-major indices of
    the new source voxels, row order of ``source_weights``.

    Mass source -> mass source density (divide by the voxel volume) -> k-Wave
    pressure source ``p = S_M * c0 * dx / 2`` (k-Wave manual 1.1, Eq. 2.19);
    k-Wave's additive scaling ``2 dt / (3 c0 dx)`` turns it back into a density
    source ``S_M dt / 3`` per component. The new rows are merged into
    ``source.p`` by ``find(p_mask)`` order, zero-padded to the longer signal,
    and added where a voxel already is a pressure source. Returns ``source``
    unchanged when there are no new voxels (k-Wave rejects empty masks).

    ``source.p`` is float64, as in MATLAB. Divergence (docs/PARITY.md): MATLAB
    divides a single ``mass_source`` by the volume in single precision before
    converting to double; here the division is done in float64 (differences of
    one float32 ulp, ~1e-7 relative, far below the k-Wave class).
    """
    idx_new = np.asarray(source_mask_idx_new, dtype=np.int64).reshape(-1)
    if idx_new.size == 0:
        return None if source is None else dict(source)
    shape = _shape(grid)
    out: dict[str, Any] = {} if source is None else dict(source)
    if out.get("p_mask") is not None:
        mask = np.array(out["p_mask"], copy=True)
        p_old = np.atleast_2d(np.asarray(out["p"]))
    else:
        mask = np.zeros(shape)
        p_old = np.zeros((0, 0))

    volume = float(_get(grid, "dx")) * float(_get(grid, "dy")) * float(_get(grid, "dz"))
    ms = np.atleast_2d(np.asarray(mass_source, dtype=np.float64) / volume)

    c0 = _at(_get(medium, "sound_speed"), idx_new)
    dx = float(_get(grid, "dx"))
    w = source_weights
    p_new = w @ ms if hasattr(w, "tocsr") else np.asarray(w, dtype=np.float64) @ ms
    p_new = np.asarray(p_new, dtype=np.float64) * (c0 * dx / 2.0)[:, None]

    flat = mask.ravel(order="F").copy()
    idx_old = np.flatnonzero(flat)
    flat[idx_new] = 1
    idx_full = np.flatnonzero(flat)
    mask = flat.reshape(shape, order="F")

    nt1, nt2 = (p_old.shape[1] if p_old.size else 0), p_new.shape[1]
    p = np.zeros((idx_full.size, max(nt1, nt2)))
    j_old = np.searchsorted(idx_full, idx_old)
    j_new = np.searchsorted(idx_full, idx_new)
    if idx_old.size:
        p[j_old, :nt1] = p_old
    # MATLAB A(j,:) = A(j,:) + B with repeated j: the last assignment wins.
    p[j_new, :nt2] = p[j_new, :nt2] + p_new
    out["p_mask"] = mask
    out["p"] = p
    return out


def self_sense_grid_size(n_sup: int) -> int:
    """Core size of the local self-sense grids: ``optimize_grid_size(N_sup, 20, 5) - 20``."""
    pml = SELF_SENSE_PML["X_SIZE"]
    return optimize_grid_size(int(n_sup), 2 * pml, 5) - 2 * pml


def _point_sensor(
    point: np.ndarray,
    idx: int,
    shape: tuple[int, int, int],
    coords: tuple[np.ndarray, np.ndarray, np.ndarray],
    spacing: np.ndarray,
    full_size: np.ndarray,
    sensor_on_grid: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """``update_sensor.m`` for a single point on an empty mask: ``(mask, weights)``.

    Private stand-in for the M3 ``update_sensor`` port (session A); replaced by
    it once merged. ``weights`` are in ``find(mask)`` order (the 1 x n_mask
    row of ``sensor_weights``).
    """
    mask = np.zeros(shape, dtype=bool)
    if sensor_on_grid:
        mask[ind2sub(shape, idx)] = True
        return mask, np.ones(1)
    sub = [int(s) for s in ind2sub(shape, idx)]
    ranges = []
    for d in range(3):
        s = sub[d]
        if point[d] == coords[d][s]:
            r = np.array([s])
        else:
            r = np.arange(s - _DELTA_HALF_WIDTH, s + _DELTA_HALF_WIDTH + 1)
        ranges.append(r[(r >= 0) & (r < shape[d])])
    gi, gj, gk = (g.ravel(order="F") for g in np.meshgrid(*ranges, indexing="ij"))
    local_idx = sub2ind(shape, gi, gj, gk)
    # ndgrid order with ascending ranges is ascending linear order, so the
    # intersect() in update_sensor.m keeps b in this order.
    assert np.all(np.diff(local_idx) > 0)
    b = evaluate_delta_function(
        np.vstack([coords[0][gi], coords[1][gj], coords[2][gk]]), point, spacing, full_size
    )
    mask[gi, gj, gk] = True
    return mask, np.asarray(b, dtype=np.float64).reshape(-1)


def _local_run_param(run_param: Any) -> dict[str, Any]:
    keys = ("solver", "DATA_CAST", "DATA_PATH", "DEVICE_NUM", "BINARY_PATH", "record_movie")
    rp = {k: _get(run_param, k) for k in keys}
    rp["PML"] = dict(SELF_SENSE_PML)
    return rp


def compute_self_sense_pressure(
    kgrid: KGrid,
    grid: Any,
    mb_idx: Any,
    mb_points: Any,
    mass_source: Any,
    medium: Any,
    n_sup: int,
    run_param: Any,
    *,
    binary_env: Mapping[str, str] | None = None,
    num_threads: int | None = None,
    n_jobs: int = 1,
    solver: str | None = None,
) -> np.ndarray:
    """Port of ``compute_self_sense_pressure.m`` (strict: one k-Wave run per bubble).

    For each bubble, the global medium is cropped to a cube of
    :func:`self_sense_grid_size` (``n_sup``) voxels centred on the bubble's
    voxel (clipped at the global grid edges), with a fixed 10-voxel PML
    (``PMLAlpha = 2``) outside; the bubble alone emits its mass source
    (:func:`update_source`) and senses the pressure through the band-limited
    delta function (``update_sensor.m``). Returns ``(N_MB, kgrid.Nt)`` float64
    holding float32-rounded values (MATLAB casts the sensed pressure to the
    class of the k-Wave output, then stores it in a double array).

    ``mb_idx``: 0-based column-major voxel index of each bubble in ``grid``;
    ``mb_points``: ``(N_MB, 3)`` positions [m]; ``mass_source``: ``(N_MB, Nt)``.
    ``run_param`` as :func:`~pyproteus.acoustic.simulation.run_simulation`
    (its ``PML`` is replaced). ``medium`` fields ``sound_speed``, ``density``,
    ``BonA``, ``alpha_coeff`` (3-D, or scalars) and ``alpha_power``.

    Options not in MATLAB (D-15, results within the k-Wave class):

    ``n_jobs``
        Bubbles simulated concurrently (threads; each run is a separate binary
        process or an in-process NumPy solve). With ``n_jobs > 1`` and no
        ``num_threads``, the CPU cores are split between the runs. Each bubble
        gets exactly the same inputs; only the binary's thread count changes
        (measured ≤ 6e-7 vs sequential; 2.7x faster with 4 x 4 threads on 18
        cores for 72³ local grids).
    ``solver``
        Overrides ``run_param.solver`` for the local runs, e.g. ``"3D"``
        (k-wave-python's NumPy solver in-process: faster than the binary on
        small local grids, slower on large ones; 4e-7 vs 3DC).

    Quirk reproduced (QUIRKS.md Q-170, BUGS.md B-172): the delta function uses the *global*
    grid's ``full_size`` for its period (``Grid_local.full_size =
    Grid.full_size``), not the local grid's.
    """
    mb_idx = np.asarray(mb_idx, dtype=np.int64).reshape(-1)
    n_mb = mb_idx.size
    points = np.asarray(mb_points, dtype=np.float64).reshape(n_mb, 3) if n_mb else np.zeros((0, 3))
    ms = np.asarray(mass_source)
    nt = kgrid.Nt
    out = np.zeros((n_mb, nt))
    if n_mb == 0:
        return out

    shape = _shape(grid)
    spacing = np.array([float(_get(grid, k)) for k in ("dx", "dy", "dz")])
    dt = float(_get(grid, "dt"))
    full_size = np.asarray(_get(grid, "full_size"), dtype=np.float64).reshape(-1)
    sensor_on_grid = bool(_get(grid, "sensor_on_grid", False))
    gx, gy, gz = (np.asarray(_get(grid, k), dtype=np.float64).reshape(-1) for k in "xyz")
    rp_local = _local_run_param(run_param)
    if solver is not None:
        rp_local["solver"] = solver
    n_jobs = max(1, min(int(n_jobs), n_mb))
    if n_jobs > 1 and num_threads is None:
        num_threads = max(1, (os.cpu_count() or 1) // n_jobs)

    n_core = self_sense_grid_size(n_sup)
    centre = n_core // 2  # 0-based: floor(NX/2) + 1 - 1
    fields = {k: _get(medium, k) for k in ("sound_speed", "density", "BonA", "alpha_coeff")}

    ax, ay, az = ind2sub(shape, mb_idx)

    def one(n: int) -> None:
        sel = []
        for a, size in zip((int(ax[n]), int(ay[n]), int(az[n])), shape, strict=True):
            glob = np.arange(n_core) + a - centre
            keep = (glob >= 0) & (glob < size)
            first_local = int(np.flatnonzero(keep)[0])
            sel.append((glob[keep], centre - first_local))
        (I, bx), (J, by), (K, bz) = sel
        local_shape = (I.size, J.size, K.size)
        idx_local = int(sub2ind(local_shape, bx, by, bz))
        kgrid_local = KGrid(local_shape, tuple(spacing), dt, nt)

        medium_local: dict[str, Any] = {"alpha_power": _get(medium, "alpha_power")}
        for k, v in fields.items():
            if v is None:
                continue
            a = np.asarray(v)
            medium_local[k] = a if a.size == 1 else a[np.ix_(I, J, K)]

        mask, weights = _point_sensor(
            points[n], idx_local, local_shape, (gx[I], gy[J], gz[K]), spacing, full_size,
            sensor_on_grid,
        )  # fmt: skip
        mask_idx = np.flatnonzero(mask.ravel(order="F"))
        grid_local = {"Nx": I.size, "Ny": J.size, "Nz": K.size,
                      "dx": spacing[0], "dy": spacing[1], "dz": spacing[2]}  # fmt: skip
        source_local = update_source(
            None, ms[n : n + 1], weights[:, None], mask_idx, grid_local, medium_local
        )
        sensor_data = run_simulation(
            rp_local, kgrid_local, medium_local, source_local,
            {"mask": mask, "record": ("p",)}, binary_env=binary_env, num_threads=num_threads,
        )  # fmt: skip
        p = np.asarray(sensor_data["p"])
        sensed = weights @ p.astype(np.float64)
        out[n] = sensed.astype(p.dtype)

    if n_jobs == 1:
        for n in range(n_mb):
            one(n)
    else:
        with ThreadPoolExecutor(n_jobs) as pool:
            list(pool.map(one, range(n_mb)))  # re-raises the first failure
    return out
