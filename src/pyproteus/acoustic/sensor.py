"""k-Wave sensor definitions of PROTEUS.

Ports of ``update_sensor.m`` and ``update_sensor_fast.m`` (one implementation,
:func:`update_sensor`), ``define_sensor_transducer.m``, ``define_sensor_MB.m``,
``define_sensor_MB_all.m`` and ``extract_sensor_data.m``.

Conventions:

* Linear indices into the ``(Nx, Ny, Nz)`` grid are **0-based column-major**
  (MATLAB ``find`` order minus one), as :func:`pyproteus.medium.voxelize_media_points`
  returns them.
* Sensor masks are indexed ``[ix, iy, iz]`` like MATLAB's. The rows of recorded
  sensor data, and the columns of the sensor weights, follow MATLAB
  ``find(sensor.mask)`` order (column-major), which is what
  :func:`pyproteus.backends.kwave.kspace_first_order` takes and returns on every
  backend. Code that hands rows to k-wave-python directly (C order) must map
  them with :func:`pyproteus.backends.kwave.rows_f_to_c` (``sensor_weights[:, perm]``
  with ``perm = f_to_c_rows(mask)`` gives C-order columns).
* Sensor weights are a SciPy CSR matrix, ``N_points × nnz(mask)``: row ``n``
  holds the band-limited delta function of point ``n`` (Wise et al., JASA
  146(1) 278-288, 2019, Eq. 17) on the mask points around it, so
  ``sensor_weights @ sensor_data`` interpolates the recorded grid data at the
  points.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike

from pyproteus.medium.voxelize import voxelize_media_points
from pyproteus.numerics import ind2sub, sub2ind

from . import _struct
from .signals import evaluate_delta_function

__all__ = [
    "DEFAULT_THRESHOLD",
    "Sensor",
    "define_sensor_MB",
    "define_sensor_MB_all",
    "define_sensor_transducer",
    "extract_sensor_data",
    "update_sensor",
    "update_sensor_fast",
]

#: Half-width of the truncated band-limited delta function [grid points]
#: (``th = 4`` in ``update_sensor.m``; ``Grid.threshold`` overrides it in
#: ``update_sensor_fast.m``).
DEFAULT_THRESHOLD = 4

#: Elements of the per-chunk ``points × (2 th + 1)^3`` candidate arrays.
_CHUNK_ELEMENTS = 1 << 22


@dataclasses.dataclass
class Sensor:
    """MATLAB k-Wave ``sensor`` struct: a binary ``mask`` and what to ``record``."""

    mask: np.ndarray
    record: tuple[str, ...] | None = None


def _mask_of(sensor: Any) -> np.ndarray:
    mask = sensor.get("mask") if isinstance(sensor, Mapping) else getattr(sensor, "mask", None)
    if mask is None:
        raise ValueError("sensor.mask is required")
    return np.asarray(mask)


def _set_mask(mask: np.ndarray, idx: np.ndarray) -> None:
    """``mask(idx) = 1`` with 0-based column-major linear indices."""
    if idx.size:
        mask[np.unravel_index(idx, mask.shape, order="F")] = 1


def _find(mask: np.ndarray) -> np.ndarray:
    """0-based ``find(mask)`` (column-major, ascending)."""
    return np.flatnonzero(mask.ravel(order="F"))


def _sensor_on_grid(grid: Any) -> bool:
    v = _struct.get(grid, "sensor_on_grid")
    if v is None:
        raise ValueError("Grid.sensor_on_grid is required (SimulationParameters.SensorOnGrid)")
    return bool(np.asarray(v).reshape(-1)[0])


def _local_nodes(
    points: np.ndarray, points_idx: np.ndarray, grid: Any, th: int
) -> tuple[np.ndarray, np.ndarray]:
    """``(point, linear index)`` of every node of each point's truncated grid.

    The truncated grid of a point is the ``(2 th + 1)^3`` cube of nodes around
    its nearest node, collapsed to that node along each axis where the point
    lies exactly on a grid line (``point(d) == Grid.x(i)``), and cut at the
    domain edge. Pairs are sorted by point, then by linear index (MATLAB's
    ``ndgrid`` order within one cube).
    """
    shape = (grid.Nx, grid.Ny, grid.Nz)
    axes = (np.asarray(grid.x), np.asarray(grid.y), np.asarray(grid.z))
    subs = ind2sub(shape, points_idx)
    ongrid = np.column_stack([points[:, d] == axes[d][subs[d]] for d in range(3)])
    offsets = np.arange(-th, th + 1)
    n_cube = offsets.size
    oi, oj, ok = (o.ravel(order="F") for o in np.meshgrid(offsets, offsets, offsets, indexing="ij"))
    off = (oi, oj, ok)

    point_out: list[np.ndarray] = []
    idx_out: list[np.ndarray] = []
    chunk = max(1, _CHUNK_ELEMENTS // n_cube**3)
    for start in range(0, points_idx.size, chunk):
        sl = slice(start, start + chunk)
        include = np.ones((subs[0][sl].size, n_cube**3), dtype=bool)
        cube = []
        for d in range(3):
            s = subs[d][sl][:, None] + off[d][None, :]
            include &= (s >= 0) & (s < shape[d])
            include &= ~ongrid[sl, d][:, None] | (off[d] == 0)[None, :]
            cube.append(s)
        n, c = np.nonzero(include)  # row-major: by point, then cube position
        point_out.append(n + start)
        idx_out.append(sub2ind(shape, cube[0][n, c], cube[1][n, c], cube[2][n, c]))
    if not point_out:
        return np.zeros(0, dtype=np.intp), np.zeros(0, dtype=np.intp)
    return np.concatenate(point_out), np.concatenate(idx_out).astype(np.intp)


def update_sensor(
    sensor: Any,
    points: ArrayLike,
    points_idx: ArrayLike,
    grid: Any,
    mask_only: bool = False,
    *,
    threshold: int | None = None,
) -> tuple[Sensor, sp.csr_matrix | None]:
    """Port of ``update_sensor.m`` and ``update_sensor_fast.m`` (same results).

    Adds the sensor points ``points`` (``N × 3`` [m], inside the grid) with
    nearest-node indices ``points_idx`` (0-based column-major, from
    :func:`~pyproteus.medium.voxelize_media_points`) to ``sensor.mask`` and
    returns ``(sensor, sensor_weights)``; ``sensor`` is a new :class:`Sensor`
    (MATLAB value semantics; the mask keeps the input dtype; ``record`` is kept).

    * ``Grid.sensor_on_grid`` true: each point is its nearest node (weight 1).
    * otherwise: each point is spread over its truncated grid (half-width
      ``threshold``, default ``Grid.threshold`` if present, else 4 as
      ``update_sensor.m``) with the band-limited delta function
      (:func:`~pyproteus.acoustic.evaluate_delta_function`), collapsed along
      axes where the point lies exactly on a grid line.

    ``mask_only=True`` returns ``None`` weights. Otherwise the weights are CSR,
    ``N × nnz(mask)``, columns in ``find(mask)`` order of the *updated* mask
    (including points already in the input mask, as ``update_sensor.m``;
    ``update_sensor_fast.m`` indexes only the new points there, B-150).

    Divergences (D-15, docs/PARITY.md): one vectorised implementation in
    bounded-memory chunks of points instead of a loop / a 2 GiB-guarded
    vectorised version; the weight matrix shape is always ``N × nnz(mask)``
    (MATLAB: ``max(i) × max(j)``, the same whenever the input mask is empty,
    as in every PROTEUS call).
    """
    mask = _mask_of(sensor).copy()
    record = (
        sensor.get("record") if isinstance(sensor, Mapping) else getattr(sensor, "record", None)
    )
    P = np.asarray(points, dtype=float).reshape(-1, 3)
    idx = np.asarray(points_idx, dtype=np.intp).reshape(-1)
    if idx.size != P.shape[0]:
        raise ValueError(f"{P.shape[0]} points but {idx.size} indices")
    if mask.shape != (grid.Nx, grid.Ny, grid.Nz):
        raise ValueError(
            f"sensor.mask has shape {mask.shape}, grid is {(grid.Nx, grid.Ny, grid.Nz)}"
        )

    if _sensor_on_grid(grid):
        point_of, local_idx = np.arange(idx.size), idx
    else:
        th = (
            threshold
            if threshold is not None
            else _struct.get(grid, "threshold", DEFAULT_THRESHOLD)
        )
        point_of, local_idx = _local_nodes(P, idx, grid, int(th))
    _set_mask(mask, np.unique(local_idx))
    out = Sensor(mask=mask, record=record)
    if mask_only:
        return out, None

    mask_idx = _find(mask)
    cols = np.searchsorted(mask_idx, local_idx)
    if _sensor_on_grid(grid):
        values = np.ones(idx.size)
    else:
        i, j, k = ind2sub(mask.shape, local_idx)
        delta_grid = np.vstack(
            [np.asarray(grid.x)[i], np.asarray(grid.y)[j], np.asarray(grid.z)[k]]
        )
        values = evaluate_delta_function(
            delta_grid, P[point_of].T, [grid.dx, grid.dy, grid.dz], list(grid.full_size)
        ) if local_idx.size else np.zeros(0)  # fmt: skip
    weights = sp.csr_matrix((values, (point_of, cols)), shape=(idx.size, mask_idx.size))
    weights.eliminate_zeros()  # MATLAB sparse() drops zero values
    return out, weights


def update_sensor_fast(
    sensor: Any,
    points: ArrayLike,
    points_idx: ArrayLike,
    grid: Any,
    mask_only: bool = False,
    *,
    threshold: int | None = None,
) -> tuple[Sensor, sp.csr_matrix | None]:
    """Port of ``update_sensor_fast.m``: the same function as :func:`update_sensor`.

    PROTEUS's own ``tests/UpdateSensorTest.m`` requires the fast version to
    reproduce the reference; pyproteus has one vectorised implementation.
    """
    return update_sensor(sensor, points, points_idx, grid, mask_only, threshold=threshold)


def _with_sensor_on_grid(grid: Any, value: Any) -> Any:
    if dataclasses.is_dataclass(grid) and not isinstance(grid, type):
        return dataclasses.replace(grid, sensor_on_grid=value)
    if isinstance(grid, Mapping):
        return {**grid, "sensor_on_grid": value}
    import copy

    g = copy.copy(grid)
    g.sensor_on_grid = value
    return g


def _span(v: np.ndarray) -> str:
    return f"[{np.min(v) * 1e3:.2f}, {np.max(v) * 1e3:.2f}]"


def define_sensor_transducer(transducer: Any, grid: Any) -> tuple[Sensor, sp.csr_matrix]:
    """Port of ``define_sensor_transducer.m``: sensor mask and weights of the integration points.

    The ``N_el × N_int × 3`` ``Transducer.integration_points`` become
    ``N_el*N_int`` sensor points in MATLAB column-major order (point
    ``e + N_el*k`` is element ``e``, integration point ``k``), placed with
    ``Transducer.OnGrid`` as ``Grid.sensor_on_grid``. Returns ``(sensor,
    sensor_weights)``, ``sensor.record = ('p',)``, weights ``N_el*N_int ×
    nnz(mask)``.

    Raises ``ValueError`` (MATLAB ``define_sensor_transducer:pointsOutsideGrid``)
    if any point lies outside the grid: PROTEUS uses x axial (depth), y
    lateral, z elevation.
    """
    on_grid = _struct.get(transducer, "OnGrid")
    if on_grid is None:
        raise ValueError("Transducer.OnGrid is required (SimulationParameters.TransducerOnGrid)")
    grid = _with_sensor_on_grid(grid, on_grid)
    ip = np.asarray(_struct.get(transducer, "integration_points"), dtype=float)
    if ip.ndim == 2:  # MATLAB drops a trailing singleton: N_el × 3 is N_el × 1 × 3
        ip = ip[:, None, :] if ip.shape[1] == 3 else ip
    n_el, n_int, n_dim = ip.shape
    P = ip.reshape(n_el * n_int, n_dim, order="F")
    vox = voxelize_media_points(P, grid)
    if np.any(vox.exclude):
        raise ValueError(
            f"{int(np.count_nonzero(vox.exclude))} of {P.shape[0]} transducer integration points "
            f"lie outside the grid.\nPoints: x {_span(P[:, 0])}, y {_span(P[:, 1])}, "
            f"z {_span(P[:, 2])} mm. Grid: x {_span(grid.x)}, y {_span(grid.y)}, "
            f"z {_span(grid.z)} mm.\nPROTEUS uses x axial (depth), y lateral, z elevation "
            "(height): check that the transducer and the grid use this orientation."
        )
    sensor = Sensor(mask=np.zeros((grid.Nx, grid.Ny, grid.Nz), dtype=bool))
    sensor, weights = update_sensor(sensor, vox.points, vox.index, grid, False)
    sensor.record = ("p",)
    return sensor, weights


def _copy_mb(mb: Any) -> Any:
    if isinstance(mb, Mapping):
        return dict(mb)
    if hasattr(mb, "copy_struct"):
        return mb.copy_struct()
    import copy

    return copy.copy(mb)


def _max_pair_distance(P: np.ndarray) -> float:
    """``max(sqrt(dx.^2 + dy.^2 + dz.^2), [], 'all')`` over all pairs, in row blocks."""
    n = P.shape[0]
    if n == 0:
        return 0.0
    best = 0.0
    rows = max(1, _CHUNK_ELEMENTS // n)
    for s in range(0, n, rows):
        d = P[s : s + rows, None, :] - P[None, :, :]
        best = max(best, float(np.max(np.sqrt(d[..., 0] ** 2 + d[..., 1] ** 2 + d[..., 2] ** 2))))
    return best


def define_sensor_MB(grid: Any, mb: Any) -> tuple[Sensor, sp.csr_matrix, Any, float]:
    """Port of ``define_sensor_MB.m``: sensors at the microbubbles of one frame.

    ``mb`` is a microbubble struct (mapping or object) with ``points`` (``N × 3``
    [m]), ``radii`` and ``velocities`` (one row per bubble). Bubbles outside
    the grid are dropped from all three; ``nodes`` and ``idx`` (0-based
    subscripts / column-major linear indices, :func:`voxelize_media_points`)
    are added. Returns ``(sensor, sensor_weights, mb, max_dist)`` with
    ``max_dist`` the largest distance between two bubbles (0 without bubbles)
    and ``sensor.record = ('p',)``; weights as :func:`update_sensor`.

    Divergence (D-15): ``max_dist`` is computed in row blocks instead of the
    full ``N × N`` distance matrix (same value, bounded memory).
    """
    mb = _copy_mb(mb)
    vox = voxelize_media_points(_struct.get(mb, "points"), grid)
    _struct.set_field(mb, "points", vox.points)
    _struct.set_field(mb, "nodes", vox.nodes)
    _struct.set_field(mb, "idx", vox.index)
    for name in ("radii", "velocities"):
        v = _struct.get(mb, name)
        if v is not None:
            _struct.set_field(mb, name, np.asarray(v)[~vox.exclude])
    max_dist = _max_pair_distance(vox.points)
    sensor = Sensor(mask=np.zeros((grid.Nx, grid.Ny, grid.Nz)))
    sensor, weights = update_sensor(sensor, vox.points, vox.index, grid, False)
    sensor.record = ("p",)
    return sensor, weights, mb, max_dist


def _bubble_points(mb: Any) -> np.ndarray:
    if isinstance(mb, np.ndarray | list | tuple):
        return np.asarray(mb, dtype=float).reshape(-1, 3)
    return np.asarray(_struct.get(mb, "points"), dtype=float).reshape(-1, 3)


def define_sensor_MB_all(grid: Any, frames: Iterable[Any]) -> tuple[Sensor, np.ndarray, int]:
    """Port of ``define_sensor_MB_all.m``: one sensor mask for every bubble of every pulse.

    ``frames`` yields the microbubbles of each simulated frame and sequence pulse
    in MATLAB's loop order (frame ``StartFrame:EndFrame``, then pulse
    ``1:N_sequence``), each a microbubble struct with ``points`` or an
    ``N × 3`` array, e.g. ``load_microbubbles`` called in that order. MATLAB
    reads the frames itself (``folder``, ``Acquisition``, ``Geometry``);
    pyproteus takes them as input so frame reading stays in one place, and a
    generator keeps one frame in memory at a time.

    Returns ``(sensor, mb_idx_all, max_mb)``: the mask (mask only, no weights),
    ``sensor.record = ('p',)``; ``mb_idx_all`` = 0-based ``find(sensor.mask == 1)``;
    ``max_mb`` the largest number of bubbles inside the grid in one pulse, at
    least 1 (as MATLAB, also without bubbles).
    """
    sensor: Sensor = Sensor(mask=np.zeros((grid.Nx, grid.Ny, grid.Nz)))
    max_mb = 1
    for mb in frames:
        vox = voxelize_media_points(_bubble_points(mb), grid)
        max_mb = max(max_mb, vox.points.shape[0])
        sensor, _ = update_sensor(sensor, vox.points, vox.index, grid, True)
    sensor.record = ("p",)
    return sensor, _find(sensor.mask == 1), max_mb


def extract_sensor_data(
    sensor_data: Any,
    mask_idx: ArrayLike,
    mask_idx_trans: ArrayLike,
    mask_idx_mb: ArrayLike,
    param: Any,
    kgrid: Any,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Port of ``extract_sensor_data.m``: split first-iteration sensor data.

    ``sensor_data.p`` holds one row per point of the combined mask, in
    ``find`` order with 0-based linear indices ``mask_idx``. Rows whose index is
    in ``mask_idx_mb`` (resp. ``mask_idx_trans``) go to the microbubble
    (transducer) data, in ascending index order (MATLAB ``intersect``); indices
    absent from ``mask_idx`` are ignored. The microbubble data keep the first
    ``floor(param.tr(1)/kgrid.dt) + 1`` samples, the transducer data all.

    Returns ``({'p': p_mb}, {'p': p_trans})``. Raises ``IndexError`` when the
    data have fewer samples than needed (MATLAB index error).
    """
    p = np.asarray(sensor_data["p"] if isinstance(sensor_data, Mapping) else sensor_data.p)
    tr = np.asarray(_struct.get(param, "tr"), dtype=float).reshape(-1)
    nt = int(np.floor(tr[0] / float(_struct.get(kgrid, "dt")))) + 1
    if nt > p.shape[1]:
        raise IndexError(f"need {nt} samples of sensor data, have {p.shape[1]}")
    mask_idx = np.asarray(mask_idx).reshape(-1)
    _, rows_mb, _ = np.intersect1d(
        mask_idx, np.asarray(mask_idx_mb).reshape(-1), return_indices=True
    )
    _, rows_tr, _ = np.intersect1d(
        mask_idx, np.asarray(mask_idx_trans).reshape(-1), return_indices=True
    )
    return {"p": p[rows_mb, :nt]}, {"p": p[rows_tr, :]}
