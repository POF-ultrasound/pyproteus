"""k-Wave pieces of PROTEUS' full simulator.

Ports of ``update_source.m`` and ``compute_self_sense_pressure.m`` (oracle
8d37938, ``acoustic-module``). ``full_simulator.m`` itself comes in M3 wave 2.

Index convention (pyproteus): linear grid indices are **0-based column-major**
(MATLAB ``find``/``MB.idx`` minus 1); bubble positions are ``(N_MB, 3)``.
Sparse weight matrices may be SciPy sparse or dense arrays.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pyproteus.backends.kwave import KGrid
from pyproteus.numerics import ind2sub, sub2ind

from ._struct import get as _get
from .grid import optimize_grid_size
from .signals import evaluate_delta_function
from .simulation import run_simulation

__all__ = [
    "IMPULSE_AMPLITUDE",
    "SELF_SENSE_PML",
    "SelfSenseCache",
    "compute_self_sense_pressure",
    "self_sense_grid_size",
    "update_source",
]

#: PML of the local single-bubble grids (fixed in ``compute_self_sense_pressure.m``).
SELF_SENSE_PML = {"X_SIZE": 10, "Y_SIZE": 10, "Z_SIZE": 10, "Alpha": 2.0}
#: Half-width of the truncated band-limited delta function (``update_sensor.m``).
_DELTA_HALF_WIDTH = 4
#: Mass-source amplitude [kg/s] of the unit-sample source of impulse-response runs:
#: small enough for the linear regime (self-sense peaks of ~0.1 Pa), far from float32 underflow.
IMPULSE_AMPLITUDE = 1e-9


@dataclass
class SelfSenseCache:
    """Self-sense impulse responses kept between calls (D-27).

    Create one per frame (fixed grid, medium, bubble positions, ``Nt`` and
    solver) and pass it to every :func:`compute_self_sense_pressure` call of
    the interaction loop: each bubble's local k-Wave run then happens once
    per frame instead of once per iteration. Entries are keyed by the
    bubble's voxel and position (and, for the homogeneous option, by the local
    medium values); the caller must not reuse a cache after changing the
    medium, grid, ``n_sup`` or solver. ``runs`` counts the k-Wave runs made.
    """

    responses: dict[Any, np.ndarray] = field(default_factory=dict)
    runs: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def get(self, key: Any, nt: int) -> np.ndarray | None:
        h = self.responses.get(key)
        return None if h is None or h.shape[-1] < nt else h[..., :nt]

    def put(self, key: Any, h: np.ndarray) -> None:
        with self._lock:
            self.responses[key] = h
            self.runs += 1

    def clear(self) -> None:
        with self._lock:
            self.responses.clear()
            self.runs = 0


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
        if p_old.shape[0] != int(np.count_nonzero(mask)):
            raise ValueError(
                f"source.p has {p_old.shape[0]} rows for {int(np.count_nonzero(mask))} "
                "p_mask voxels (MATLAB errors too)"
            )
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
    if not np.all(np.diff(local_idx) > 0):  # pragma: no cover - invariant
        raise AssertionError("truncated delta grid is not in ascending linear order")
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


def _causal_conv(h: np.ndarray, ms: np.ndarray, nt: int) -> np.ndarray:
    """First ``nt`` samples of the linear convolution of ``h`` with each row of ``ms``."""
    from scipy.signal import fftconvolve

    return fftconvolve(h, np.asarray(ms, dtype=np.float64), axes=-1)[..., :nt]


def _weights_1d(
    point: float, sub: int, coord: np.ndarray, d: float, n_full: float, on_grid: bool
) -> np.ndarray:
    """1-D factor of the truncated band-limited delta (``update_sensor.m``), not cropped.

    ``evaluate_delta_function`` is a product of 1-D factors, so the 3-D
    weights of :func:`_point_sensor` are the outer product of these.
    """
    if on_grid or point == coord[sub]:
        return np.ones(1)
    x = coord[sub - _DELTA_HALF_WIDTH : sub + _DELTA_HALF_WIDTH + 1]
    return np.asarray(evaluate_delta_function(x.reshape(1, -1), point, d, n_full)).reshape(-1)


def _autocorr(w: np.ndarray, hw: int) -> np.ndarray:
    """``a[hw + k] = sum_j w[j] w[j + k]`` for ``|k| <= hw``."""
    a = np.zeros(2 * hw + 1)
    c = np.correlate(w, w, mode="full")
    m = (c.size - 1) // 2
    a[hw - m : hw + m + 1] = c
    return a


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
    impulse_response: bool = True,
    cache: SelfSenseCache | None = None,
    homogeneous: bool = False,
    binary_env: Mapping[str, str] | None = None,
    num_threads: int | None = None,
    n_jobs: int = 1,
    solver: str | None = None,
) -> np.ndarray:
    """Port of ``compute_self_sense_pressure.m``: pressure each bubble senses from itself.

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

    Modes (D-27):

    ``impulse_response=True`` (default)
        Each bubble's local run uses a unit-sample source
        (:data:`IMPULSE_AMPLITUDE` at the first time step) instead of its mass
        source; the self-sensed pressure is that impulse response convolved
        with the mass source. k-Wave is linear and time invariant for
        ``BonA = 0``, so this equals the MATLAB result within the k-Wave class
        (measured 5.7e-7). With ``BonA > 0`` it drops the local nonlinearity:
        relative difference 4.5e-6 at a 5.9 kPa self-sense peak, 4.7e-5 at
        59 kPa (docs/PARITY.md). With a :class:`SelfSenseCache` the responses
        are reused across the interaction iterations of a frame.
    ``impulse_response=False``
        Strict port: one run per bubble with its own mass source (MATLAB);
        ``cache`` is not used.
    ``homogeneous=True`` (opt-in, changes results)
        Bubbles whose local cube is not cropped by the global grid edges use a
        homogeneous local medium (the medium values at the bubble's voxel): one
        k-Wave run per distinct set of values gives the Green's function
        ``G(Δ, t)`` between voxels ``Δ`` apart (``|Δ| <= 8`` per axis), and a
        bubble's response is ``Σ_Δ a(Δ) G(Δ, t)`` with ``a`` the
        autocorrelation of its delta weights (separable per axis). Cropped
        bubbles use the impulse-response path on their heterogeneous crop.
        Only pays off when many bubbles share local medium values (piecewise
        constant media, e.g. a vessel of uniform blood properties): with
        PROTEUS's per-voxel Inhomogeneity every bubble has its own values, so
        there is one tensor run per bubble (no fewer runs than the default
        mode) and each cached tensor holds 17³ x Nt float32 values (~98 MB at
        Nt = 5000). Measured against the heterogeneous crop: 4.8e-3 (Inhomogeneity 0.02),
        1.2e-2 (0.05). In a homogeneous medium it equals the per-bubble runs up
        to late-time PML reflections, which depend on where the source sits in
        the cube (3.8e-4 for a 20-voxel core, 5.2e-5 for 32). Needs a local core
        of at least 17 voxels.

    Other options not in MATLAB (D-15, results within the k-Wave class):

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
    ms = np.asarray(mass_source).reshape(n_mb, -1) if n_mb else np.asarray(mass_source)
    nt = kgrid.Nt
    out = np.zeros((n_mb, nt))
    if n_mb == 0:
        return out
    if homogeneous and not impulse_response:
        raise ValueError("homogeneous=True needs impulse_response=True")

    shape = _shape(grid)
    spacing = np.array([float(_get(grid, k)) for k in ("dx", "dy", "dz")])
    dt = float(_get(grid, "dt"))
    full_size = np.asarray(_get(grid, "full_size"), dtype=np.float64).reshape(-1)
    sensor_on_grid = bool(_get(grid, "sensor_on_grid", False))
    gxyz = tuple(np.asarray(_get(grid, k), dtype=np.float64).reshape(-1) for k in "xyz")
    rp_local = _local_run_param(run_param)
    if solver is not None:
        rp_local["solver"] = solver
    n_jobs = max(1, min(int(n_jobs), n_mb))
    if n_jobs > 1 and num_threads is None:
        num_threads = max(1, (os.cpu_count() or 1) // n_jobs)
    if cache is None:
        cache = SelfSenseCache()

    n_core = self_sense_grid_size(n_sup)
    centre = n_core // 2  # 0-based: floor(NX/2) + 1 - 1
    fields = {k: _get(medium, k) for k in ("sound_speed", "density", "BonA", "alpha_coeff")}
    alpha_power = _get(medium, "alpha_power")
    subs = np.stack(ind2sub(shape, mb_idx), axis=1).astype(int)
    unit = np.zeros((1, nt))
    unit[0, 0] = IMPULSE_AMPLITUDE

    def crop(n: int):
        sel = []
        for a, size in zip(subs[n], shape, strict=True):
            glob = np.arange(n_core) + a - centre
            keep = (glob >= 0) & (glob < size)
            sel.append((glob[keep], centre - int(np.flatnonzero(keep)[0])))
        return sel

    def run_local(medium_local, local_shape, mask, weights_row_sources, source_idx, sources):
        """One local run; returns the recorded rows (find order of ``mask``), float32."""
        grid_local = {"Nx": local_shape[0], "Ny": local_shape[1], "Nz": local_shape[2],
                      "dx": spacing[0], "dy": spacing[1], "dz": spacing[2]}  # fmt: skip
        source_local = update_source(
            None, sources, weights_row_sources, source_idx, grid_local, medium_local
        )
        sensor_data = run_simulation(
            rp_local, KGrid(local_shape, tuple(spacing), dt, nt), medium_local, source_local,
            {"mask": mask, "record": ("p",)}, binary_env=binary_env, num_threads=num_threads,
        )  # fmt: skip
        return np.asarray(sensor_data["p"])

    def bubble_run(n: int, sources: np.ndarray) -> np.ndarray:
        """Local heterogeneous run for bubble ``n``: sensed pressure (float64 of float32)."""
        (I, bx), (J, by), (K, bz) = crop(n)
        local_shape = (I.size, J.size, K.size)
        medium_local: dict[str, Any] = {"alpha_power": alpha_power}
        for k, v in fields.items():
            if v is not None:
                a = np.asarray(v)
                medium_local[k] = a if a.size == 1 else a[np.ix_(I, J, K)]
        mask, weights = _point_sensor(
            points[n], int(sub2ind(local_shape, bx, by, bz)), local_shape,
            (gxyz[0][I], gxyz[1][J], gxyz[2][K]), spacing, full_size, sensor_on_grid,
        )  # fmt: skip
        mask_idx = np.flatnonzero(mask.ravel(order="F"))
        p = run_local(medium_local, local_shape, mask, weights[:, None], mask_idx, sources)
        with np.errstate(all="ignore"):  # spurious BLAS FP flags on macOS (finite inputs)
            return (weights @ p.astype(np.float64)).astype(p.dtype).astype(np.float64)

    def green_tensor(values: tuple[float | None, ...]) -> np.ndarray:
        """``G[hw+dx, hw+dy, hw+dz, t]``: homogeneous cube, unit sample at the centre voxel."""
        hw = 2 * _DELTA_HALF_WIDTH
        if n_core < 2 * hw + 1:
            raise ValueError(
                f"homogeneous=True needs a local core of at least {2 * hw + 1} voxels, "
                f"got {n_core} (n_sup = {n_sup})"
            )
        cube = (n_core,) * 3
        medium_local: dict[str, Any] = {"alpha_power": alpha_power}
        for k, v in zip(fields, values, strict=True):
            if v is not None:
                medium_local[k] = np.full(cube, v)
        mask = np.zeros(cube, dtype=bool)
        r = slice(centre - hw, centre + hw + 1)
        mask[r, r, r] = True
        src = np.array([int(sub2ind(cube, centre, centre, centre))])
        p = run_local(medium_local, cube, mask, np.ones((1, 1)), src, unit)
        # float32 (the k-Wave output precision): 17³ x Nt values per tensor
        g = p.reshape((2 * hw + 1,) * 3 + (nt,), order="F")
        return (g / np.float32(IMPULSE_AMPLITUDE)).astype(np.float32)

    def uses_green(n: int) -> bool:
        cropped = any(
            s - centre < 0 or s - centre + n_core > size
            for s, size in zip(subs[n], shape, strict=True)
        )
        return homogeneous and not cropped

    def green_key(n: int) -> tuple:
        values = tuple(
            None if v is None
            else float(np.asarray(v).reshape(-1)[0] if np.size(v) == 1
                       else np.asarray(v).ravel(order="F")[mb_idx[n]])
            for v in fields.values()
        )  # fmt: skip
        return ("green", values)

    def green(key: tuple) -> np.ndarray:
        g = cache.get(key, nt)
        if g is None:
            g = green_tensor(key[1])
            cache.put(key, g)
        return g

    def response(n: int) -> np.ndarray:
        """Impulse response of bubble ``n`` (per unit mass source), cached."""
        if not uses_green(n):
            key = ("bubble", int(mb_idx[n]), tuple(points[n]))
            h = cache.get(key, nt)
            if h is None:
                h = bubble_run(n, unit) / IMPULSE_AMPLITUDE
                cache.put(key, h)
            return h
        g = green(green_key(n))
        hw = 2 * _DELTA_HALF_WIDTH
        a = [
            _autocorr(_weights_1d(points[n, d], subs[n, d], gxyz[d], spacing[d], full_size[d],
                                  sensor_on_grid), hw)
            for d in range(3)
        ]  # fmt: skip
        return np.einsum("i,j,k,ijkt->t", a[0], a[1], a[2], g.astype(np.float64))

    def one(n: int) -> None:
        if impulse_response:
            sensed = _causal_conv(response(n), ms[n], nt)
        else:
            sensed = bubble_run(n, ms[n : n + 1])
        out[n] = sensed.astype(np.float32)

    if homogeneous:  # one run per distinct local medium, before the per-bubble work
        keys = list(dict.fromkeys(green_key(n) for n in range(n_mb) if uses_green(n)))
        if n_jobs == 1 or len(keys) < 2:
            for key in keys:
                green(key)
        else:
            with ThreadPoolExecutor(n_jobs) as pool:
                list(pool.map(green, keys))
    if n_jobs == 1:
        for n in range(n_mb):
            one(n)
    else:
        with ThreadPoolExecutor(n_jobs) as pool:
            list(pool.map(one, range(n_mb)))  # re-raises the first failure
    return out
