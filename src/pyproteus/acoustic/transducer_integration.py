"""Transducer integration points, lens delays and simulation travel times.

Ports of ``get_transducer_integration_points.m``,
``get_transducer_integration_delays.m`` and ``compute_travel_times.m``.

Transducers are :class:`pyproteus.config.Transducer` models (or any struct-like
object); the functions return an updated copy with the run-time fields that
PROTEUS adds (``integration_points`` ``N × N_int × 3``,
``integration_transmit_delays`` / ``integration_receive_delays`` ``N × N_int``, ...).
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from ._struct import config_is as _config_is
from ._struct import copy as _copy
from ._struct import get as _get
from ._struct import isfield as _isfield
from ._struct import scalar as _scalar
from ._struct import set_field as _set

__all__ = [
    "compute_travel_times",
    "get_transducer_integration_delays",
    "get_transducer_integration_points",
]


def _columns(T: Any, grid: Any) -> None:
    """``get_columns`` of ``get_transducer_integration_points.m`` (in place on copy ``T``).

    Linear array: elements in the plane x = 0; curved array: points uniformly
    spaced along the arc of each element (``get_element_geometry``). Point
    ``k`` of an element is lateral index ``k % N_int_y`` and elevation index
    ``k // N_int_y`` (MATLAB ``ndgrid`` + ``(:)``: lateral fastest).

    Divergence (D-15): vectorised over elements instead of MATLAB's loop.
    """
    from pyproteus.transducer import get_element_geometry, is_curved

    n = int(_scalar(_get(T, "NumberOfElements")))
    p = _scalar(_get(T, "Pitch"))
    curved = is_curved(T)
    if curved:
        el = get_element_geometry(T)
        w = el.width
    else:
        w = _scalar(_get(T, "ElementWidth"))
    W = (n - 1) * p + w
    H = _scalar(_get(T, "ElementHeight"))
    y_min = -W / 2 + np.arange(n) * p

    density = _scalar(_get(T, "IntegrationDensity"))
    n_int_y = int(np.ceil(density * w / grid.dy))
    sp_y = w / n_int_y
    sp_z = grid.dz / density
    Z1 = int(np.ceil(-H / 2 / sp_z))
    Z2 = int(np.floor(H / 2 / sp_z))
    n_int_z = max(Z2 - Z1 + 1, 0)
    n_int = n_int_y * n_int_z

    k = np.arange(1, n_int_y + 1, dtype=float)
    z = np.arange(Z1, Z2 + 1, dtype=float) * sp_z
    if curved:
        s = (k - (n_int_y + 1) / 2) * sp_y
        phi = el.theta[:, None] + s[None, :] / el.radius  # n × n_int_y
        xl = el.center + el.radius * np.cos(phi)
        yl = el.radius * np.sin(phi)
    else:
        yl = y_min[:, None] + k[None, :] * sp_y - sp_y / 2
        xl = np.zeros_like(yl)
    pts = np.empty((n, n_int_y, n_int_z, 3))
    pts[..., 0] = xl[:, :, None]
    pts[..., 1] = yl[:, :, None]
    pts[..., 2] = z[None, None, :]
    # lateral index fastest (MATLAB column-major (:) of an n_int_y × n_int_z grid)
    _set(T, "integration_points", pts.transpose(0, 2, 1, 3).reshape(n, n_int, 3))
    _set(T, "integration_weights", sp_y * sp_z / (grid.dy * grid.dz))
    # apodization_Z is hard-coded false in MATLAB (getWin branch is dead code).
    apod = np.ones((n, n_int))
    _set(T, "integration_transmit_apodization", apod)
    _set(T, "integration_receive_apodization", apod.copy())


def get_transducer_integration_points(transducer: Any, grid: Any) -> Any:
    """Port of ``get_transducer_integration_points.m``.

    Adds ``integration_points`` (``N × N_int × 3`` [m]), ``integration_weights``
    (quadrature weight per point, relative to a grid cell face ``dy*dz``) and
    unit ``integration_transmit_apodization`` / ``integration_receive_apodization``
    (``N × N_int``). Uses ``NumberOfElements``, ``Pitch``, ``ElementWidth``
    (linear), ``ElementHeight``, ``IntegrationDensity`` and, for curved arrays,
    ``get_element_geometry``.

    - ``'RCA'``: ``N`` must be even (``ValueError`` as MATLAB's error); the first
      ``N/2`` elements are linear columns, the last ``N/2`` the same with the
      lateral and elevation coordinates swapped (rows).
    - ``'Custom'``: user ``integration_points`` are kept; weights are
      ``integration_area/(dy*dz)``; missing apodizations become ones; warns
      (``UserWarning``) when the aperture extends mostly along x.
    - otherwise: linear or curved array.
    """
    T = _copy(transducer)
    if _config_is(T, "RCA"):
        n = int(_scalar(_get(T, "NumberOfElements")))
        if n % 2:
            raise ValueError("Number of elements in RCA must be even.")
        _set(T, "NumberOfElements", float(n // 2))
        _columns(T, grid)
        columns = np.asarray(_get(T, "integration_points"))
        rows = columns[:, :, [0, 2, 1]]
        _set(T, "NumberOfElements", float(n))
        _set(T, "integration_points", np.concatenate([columns, rows], axis=0))
        for k in ("integration_transmit_apodization", "integration_receive_apodization"):
            _set(T, k, np.tile(np.asarray(_get(T, k)), (2, 1)))
    elif _config_is(T, "Custom"):
        pts = np.asarray(_get(T, "integration_points"), dtype=float)
        P = pts.reshape(-1, 3, order="F") if pts.ndim == 3 else pts.reshape(-1, 3)
        span = P.max(axis=0) - P.min(axis=0)
        if span[0] > max(span[1], span[2]):
            warnings.warn(
                f"The custom transducer extends {span[0] * 1e3:.1f} mm along x (axial) but "
                f"only {span[1] * 1e3:.1f} mm along y and {span[2] * 1e3:.1f} mm along z. "
                "PROTEUS uses x axial, y lateral, z elevation: check the orientation of "
                "the integration points.",
                UserWarning,
                stacklevel=2,
            )
        area = np.asarray(_get(T, "integration_area"), dtype=float)
        weights = area / (grid.dy * grid.dz)
        _set(T, "integration_weights", float(weights) if weights.size == 1 else weights)
        apod = np.ones(pts.shape[:2])
        for k in ("integration_transmit_apodization", "integration_receive_apodization"):
            if not _isfield(T, k):
                _set(T, k, apod.copy())
    else:
        _columns(T, grid)
    return T


def get_transducer_integration_delays(transducer: Any, medium: Any) -> Any:
    """Port of ``get_transducer_integration_delays.m``: lens delays per integration point.

    - ``'RCA'``: zero delays (shape of ``integration_transmit_apodization``).
    - ``'Custom'``: zero delays, only for the delay fields that are absent.
    - otherwise (linear, curved, no ``Configuration``): elevation lens focus
      ``ElevationFocus``: delays ``sqrt(z^2 + f^2)/c`` (negated for ``f > 0``),
      zero if ``f`` is not finite; then shifted so the minimum is 0.
    """
    T = _copy(transducer)
    if _config_is(T, "RCA"):
        delays = np.zeros(np.shape(_get(T, "integration_transmit_apodization")))
        _set(T, "integration_transmit_delays", delays)
        _set(T, "integration_receive_delays", delays.copy())
    elif _config_is(T, "Custom"):
        delays = np.zeros(np.shape(_get(T, "integration_transmit_apodization")))
        if not _isfield(T, "integration_transmit_delays"):
            _set(T, "integration_transmit_delays", delays)
        if not _isfield(T, "integration_receive_delays"):
            _set(T, "integration_receive_delays", delays.copy())
    else:
        f = _scalar(_get(T, "ElevationFocus"))
        z = np.asarray(_get(T, "integration_points"), dtype=float)[:, :, 2]
        if np.isfinite(f):
            delays = np.sqrt(z**2 + f**2) / _scalar(_get(medium, "SpeedOfSound"))
            if f > 0:
                delays = -delays
        else:
            delays = z * 0
        delays = delays - np.min(delays)
        _set(T, "integration_transmit_delays", delays)
        _set(T, "integration_receive_delays", delays.copy())
    return T


def _max_distance_domain_boundary(points: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Largest distance from each integration point (``N_el × N_int × 3``) to the vertices."""
    diff = points[:, :, None, :] - vertices[None, None, :, :]
    return np.max(np.sqrt(np.sum(diff**2, axis=3)), axis=2)


def compute_travel_times(
    run_param: dict[str, Any] | None,
    geometry: Any,
    medium: Any,
    transducer: Any,
    transmit: Any,
) -> dict[str, Any]:
    """Port of ``compute_travel_times.m``: simulation end times and distances.

    Returns a copy of ``run_param`` (a dict; ``None`` → new dict) with

    - ``tr``: ``[t_end_1, t_end_2, t_end_3]`` [s] (1-D, MATLAB 1×3),
    - ``pulse_length``: ``c * (N - 1)/fs`` [m], ``N = numel(PressureSignal)``,
    - ``max_trans_dist``: ``transmit_time * c`` [m].

    With a ``Configuration`` other than ``'Linear'`` the transmit/receive times
    use the largest distance from each integration point to the domain
    vertices, plus the lens and transmit delays; for ``'Linear'`` (or no
    ``Configuration``) the distances from the element centres to the far
    bottom corners of the domain in the x-y plane.

    ``Transmit.Delays`` is a vector with one delay per element (MATLAB row).
    """
    out: dict[str, Any] = dict(run_param or {})
    c = _scalar(_get(medium, "SpeedOfSound"))
    fs = _scalar(_get(transmit, "SamplingRate"))
    delays_tx = np.asarray(_get(transmit, "Delays"), dtype=float).reshape(-1)
    points = np.asarray(_get(transducer, "integration_points"), dtype=float)

    if _isfield(transducer, "Configuration") and _get(transducer, "Configuration") != "Linear":
        vertices = np.asarray(_get(_get(geometry, "Domain"), "Vertices"), dtype=float)
        d = _max_distance_domain_boundary(points, vertices)
        t_tx = d / c + np.asarray(_get(transducer, "integration_transmit_delays"), dtype=float)
        t_tx = t_tx + delays_tx[:, None]  # MATLAB: + transpose(Transmit.Delays)
        transmit_time = float(np.max(t_tx))
        t_rx = d / c + np.asarray(_get(transducer, "integration_receive_delays"), dtype=float)
        receive_time = float(np.max(t_rx))
    else:
        D = _get(geometry, "Domain")
        xmax, ymin, ymax = (_scalar(_get(D, k)) for k in ("Xmax", "Ymin", "Ymax"))
        py = points[:, :, 1]
        y = np.mean(py, axis=1)  # element centres
        y1, y2 = np.max(py), np.min(py)
        max_trans_dist = np.maximum(np.hypot(y - ymin, xmax), np.hypot(y - ymax, xmax))
        transmit_time = float(np.max(max_trans_dist / c + delays_tx))
        receive_time = max(float(np.hypot(y1 - ymin, xmax)), float(np.hypot(y2 - ymax, xmax))) / c

    diagonal = np.asarray(_get(_get(geometry, "BoundingBox"), "Diagonal"), dtype=float)
    max_mb_dist = float(np.linalg.norm(diagonal.reshape(-1)))

    n = np.size(_get(transmit, "PressureSignal"))
    pulse_duration = (n - 1) / fs
    out["tr"] = np.array(
        [
            2 * pulse_duration + transmit_time,
            2 * pulse_duration + transmit_time + max_mb_dist / c,
            2 * pulse_duration + transmit_time + receive_time,
        ]
    )
    out["pulse_length"] = c * pulse_duration
    out["max_trans_dist"] = transmit_time * c
    return out
