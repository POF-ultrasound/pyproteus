"""The two PROTEUS simulation paths: ports of ``hybrid_simulator.m`` and ``full_simulator.m``.

Both start from the pressure the bubbles sensed in the first k-Wave run
(``sensed_p``, transducer transmit only), iterate the bubble-bubble
interaction ``run_param.N_interactions`` times and return the pressure at the
transducer sensor.

* :func:`hybrid_simulator` (default path): the bubbles radiate through a
  homogeneous medium (:func:`~pyproteus.acoustic.run_simulation_homogeneous`,
  the k-Wave medium averaged); exact bubble-to-bubble distances during the
  interaction, distances rounded to the grid towards the transducer.
* :func:`full_simulator`: every iteration is a k-Wave run with the transducer
  source plus the bubble mass sources (:func:`~pyproteus.acoustic.full.update_source`),
  minus each bubble's self-sensed pressure
  (:func:`~pyproteus.acoustic.full.compute_self_sense_pressure`).

Differences with MATLAB (docs/PARITY.md): no ``addpath``/``rmpath`` of the
microbubble module; the Kramers–Kronig reference frequency of the homogeneous
propagation is the transmit centre frequency (D-23); the self-sense mode is
selectable (D-27, default: impulse responses cached over the iterations of a
frame); ``step_control`` of the bubble solver is passed through (D-22).
"""

from __future__ import annotations

import dataclasses
import math
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np

from pyproteus.backends.kwave import KGrid
from pyproteus.numerics import ind2sub

from . import _struct
from .full import SelfSenseCache, compute_self_sense_pressure, update_source
from .homogeneous import run_simulation_homogeneous, to_numpy
from .mass_source import compute_bubble_mass_source
from .simulation import run_simulation

__all__ = ["SELF_SENSE_MODES", "full_simulator", "hybrid_simulator", "self_sense_support"]

#: ``self_sense`` values of :func:`full_simulator` (D-27).
SELF_SENSE_MODES = ("impulse_response", "strict", "green")

SelfSenseMode = Literal["impulse_response", "strict", "green"]


def _nt(t_end: float, dt: float) -> int:
    """``floor(t_end / dt) + 1``."""
    return math.floor(t_end / dt) + 1


def _rp(run_param: Any, name: str) -> Any:
    v = _struct.get(run_param, name)
    if v is None:
        raise ValueError(f"run_param.{name} is required")
    return v


def _t_end_2(run_param: Any, c0: float) -> float:
    """Interaction time window (both simulators): ``(max_trans_dist + max_dist + 2 L) / c``."""
    return (
        float(_rp(run_param, "max_trans_dist"))
        + float(_rp(run_param, "max_dist"))
        + 2 * float(_rp(run_param, "pulse_length"))
    ) / c0


def _mass_source(
    sensed_p: np.ndarray,
    mb: Any,
    kgrid: Any,
    medium: Any,
    microbubble: Any,
    transmit: Any,
    step_control: str,
) -> np.ndarray:
    return compute_bubble_mass_source(
        sensed_p,
        _struct.get(mb, "radii"),
        kgrid,
        medium,
        microbubble,
        transmit,
        step_control=step_control,  # type: ignore[arg-type]
    )


def hybrid_simulator(
    sensor_mask_idx_trans: Any,
    sensed_p_1iter: np.ndarray,
    mb: Any,
    grid: Any,
    medium: Any,
    run_param: Any,
    Medium: Any,
    Microbubble: Any,
    Transmit: Any,
    *,
    step_control: str = "independent",
    gridded_receive: bool | None = None,
) -> dict[str, Any]:
    """Port of ``hybrid_simulator.m``.

    ``sensor_mask_idx_trans``: 0-based column-major indices of the transducer
    sensor voxels; ``sensed_p_1iter``: ``(N_MB, Nt1)`` pressure at the bubbles
    from the first k-Wave run, ``Nt1 = floor(tr(1)/dt) + 1``; ``mb``: the
    frame's bubbles (``points``, ``radii``, from
    :func:`~pyproteus.acoustic.define_sensor_MB`); ``grid``: the PROTEUS grid;
    ``medium``: the k-Wave medium (averaged by the homogeneous propagation);
    ``run_param``: ``tr``, ``max_trans_dist``, ``max_dist``, ``pulse_length``,
    ``N_interactions``, ``DATA_CAST``; ``Medium``/``Microbubble``/``Transmit``:
    the settings groups.

    Interaction ``n``: mass sources from ``sensed_p``
    (``k_max = π/(dt·c_min)``, as MATLAB's ``Grid.k_max``), radiated bubble to
    bubble over ``Nt2 = floor(t_end_2/dt)+1`` samples with exact distances;
    ``sensed_p[:, :Nt1] = p_bubbles[:, :Nt1] + sensed_p_1iter``. Last stage:
    mass sources radiated to the transducer voxel centres over
    ``Nt3 = floor(tr(3)/dt)+1`` samples with gridded distances. Returns
    ``{'p': (n_trans, Nt3)}`` (the transducer's own first-iteration data are
    added by the caller, as in ``main_RF.m``).

    ``gridded_receive``: distances of the last stage rounded to the time grid
    (``True``, PROTEUS) or exact (``False``); default (``None``) gridded on the
    CPU and exact when ``DATA_CAST`` is ``'gpuArray-*'``, where the exact kernel
    is faster (D-30).
    """
    tr = np.asarray(_rp(run_param, "tr"), dtype=float).reshape(-1)
    dt = float(grid.dt)
    c0 = _struct.scalar(_struct.get(Medium, "SpeedOfSound"))
    n_interactions = int(_struct.scalar(_rp(run_param, "N_interactions")))
    f0 = _struct.scalar(_struct.get(Transmit, "CenterFrequency"))
    nt1 = _nt(tr[0], dt)
    k_max = math.pi / dt / _struct.scalar(_struct.get(Medium, "SpeedOfSoundMinimum"))
    rp = dict(run_param) if isinstance(run_param, dict) else dict(vars(run_param))

    sensed_p_1iter = np.asarray(sensed_p_1iter)
    sensed_p = sensed_p_1iter.copy()
    points = np.asarray(_struct.get(mb, "points"), dtype=float).reshape(-1, 3)

    for _ in range(n_interactions):
        nt2 = _nt(_t_end_2(rp, c0), dt)
        bubble_grid = SimpleNamespace(dt=dt, k_max=k_max, Nt=nt2)
        ms = _mass_source(sensed_p, mb, bubble_grid, Medium, Microbubble, Transmit, step_control)
        rp["gridded"] = False
        sd = run_simulation_homogeneous(
            rp,
            bubble_grid,
            medium,
            {"mass_source": ms, "points": points},
            {"points": points},
            reference_frequency=f0,
        )
        p = to_numpy(sd["p"])
        if p.shape[1] < nt1:
            raise IndexError(  # MATLAB: index exceeds the array bounds
                f"interaction window ({p.shape[1]} samples) shorter than the "
                f"transmit window ({nt1})"
            )
        sensed_p[:, :nt1] = p[:, :nt1] + sensed_p_1iter

    nt3 = _nt(tr[2], dt)
    receive_grid = SimpleNamespace(dt=dt, k_max=k_max, Nt=nt3)
    ms = _mass_source(sensed_p, mb, receive_grid, Medium, Microbubble, Transmit, step_control)
    idx = np.asarray(sensor_mask_idx_trans, dtype=np.int64).reshape(-1)
    i, j, k = ind2sub((grid.Nx, grid.Ny, grid.Nz), idx)
    sensor_points = np.column_stack(
        [np.asarray(grid.x)[i], np.asarray(grid.y)[j], np.asarray(grid.z)[k]]
    )
    if gridded_receive is None:  # D-30: exact on the GPU, gridded (PROTEUS) on the CPU
        gridded_receive = not str(rp.get("DATA_CAST", "")).startswith("gpuArray")
    rp["gridded"] = bool(gridded_receive)
    return run_simulation_homogeneous(
        rp,
        receive_grid,
        medium,
        {"mass_source": ms, "points": points},
        {"points": sensor_points},
        reference_frequency=f0,
    )


def self_sense_support(Medium: Any, Transmit: Any, grid: Any) -> int:
    """``N_sup = ceil(2·2·c/f0/dx)``: half-width of the self-sense cube (``full_simulator.m``)."""
    c0 = _struct.scalar(_struct.get(Medium, "SpeedOfSound"))
    f0 = _struct.scalar(_struct.get(Transmit, "CenterFrequency"))
    return math.ceil(2 * 2 * c0 / f0 / float(_struct.get(grid, "dx")))


def full_simulator(
    source_transducer: Any,
    sensor_transducer: Any,
    sensor_frame: Any,
    sensor_weights_frame: Any,
    sensor_mask_idx_frame: Any,
    sensed_p: np.ndarray,
    mb: Any,
    kgrid: KGrid,
    grid: Any,
    medium: Any,
    run_param: Any,
    Medium: Any,
    Microbubble: Any,
    Transmit: Any,
    *,
    self_sense: SelfSenseMode = "impulse_response",
    step_control: str = "independent",
    binary_env: Any = None,
    num_threads: int | None = None,
    self_sense_jobs: int = 1,
) -> dict[str, Any]:
    """Port of ``full_simulator.m``.

    ``source_transducer``: the pulse's transducer source
    (:func:`~pyproteus.acoustic.define_source_transducer`, as a
    :class:`~pyproteus.acoustic.Source` or dict); ``sensor_transducer``: the
    transducer sensor; ``sensor_frame``/``sensor_weights_frame``/
    ``sensor_mask_idx_frame``: the frame's bubble sensor, weights
    (``N_MB × nnz``) and 0-based mask indices
    (:func:`~pyproteus.acoustic.define_sensor_MB`); ``sensed_p``: first-run
    pressure at the bubbles; ``kgrid``: :class:`~pyproteus.backends.kwave.KGrid`
    (its ``Nt`` is replaced); ``run_param`` as
    :func:`~pyproteus.acoustic.simulation.run_simulation` plus ``tr``,
    ``max_trans_dist``, ``max_dist``, ``pulse_length``, ``N_interactions``.

    Interaction: mass sources (``k_max`` of the k-Wave grid) added to the
    transducer source, one k-Wave run over ``Nt2`` samples recording at the
    bubbles, ``sensed_p = W·p − self_sense``. Last stage: one k-Wave run over
    ``Nt3`` samples recording at the transducer. Returns k-Wave's
    ``sensor_data`` (``{'p': (n_trans, Nt3)}``).

    ``self_sense`` (D-27): ``"impulse_response"`` (default; impulse responses
    computed once per call, i.e. per frame and pulse, and reused over the
    iterations), ``"strict"`` (MATLAB: one k-Wave run per bubble and
    iteration), ``"green"`` (opt-in homogeneous Green's function, changes
    results). ``self_sense_jobs``: bubbles run concurrently.
    """
    if self_sense not in SELF_SENSE_MODES:
        raise ValueError(f"self_sense must be one of {SELF_SENSE_MODES}, got {self_sense!r}")
    tr = np.asarray(_rp(run_param, "tr"), dtype=float).reshape(-1)
    c0 = _struct.scalar(_struct.get(Medium, "SpeedOfSound"))
    n_interactions = int(_struct.scalar(_rp(run_param, "N_interactions")))
    n_sup = self_sense_support(Medium, Transmit, grid)
    source_transducer = _as_source_dict(source_transducer)
    weights_t = sensor_weights_frame.T
    cache = SelfSenseCache() if self_sense != "strict" else None
    kw = {"binary_env": binary_env, "num_threads": num_threads}
    mb_idx = _struct.get(mb, "idx")
    mb_points = _struct.get(mb, "points")

    sensed_p = np.asarray(sensed_p)
    for _ in range(n_interactions):
        kgrid = dataclasses.replace(kgrid, Nt=_nt(_t_end_2(run_param, c0), kgrid.dt))
        ms = _mass_source(sensed_p, mb, kgrid, Medium, Microbubble, Transmit, step_control)
        source = update_source(
            source_transducer, ms, weights_t, sensor_mask_idx_frame, grid, medium
        )
        sd = run_simulation(run_param, kgrid, medium, source, sensor_frame, **kw)
        p = np.asarray(sd["p"])
        sensed = np.asarray(sensor_weights_frame @ p.astype(np.float64), dtype=np.float64)
        sensed = sensed.astype(p.dtype)
        ss = compute_self_sense_pressure(
            kgrid,
            grid,
            mb_idx,
            mb_points,
            ms,
            medium,
            n_sup,
            run_param,
            impulse_response=self_sense != "strict",
            cache=cache,
            homogeneous=self_sense == "green",
            n_jobs=self_sense_jobs,
            **kw,
        )
        # MATLAB single - double: the double operand is cast to single first
        sensed_p = sensed - ss.astype(sensed.dtype)

    kgrid = dataclasses.replace(kgrid, Nt=_nt(tr[2], kgrid.dt))
    ms = _mass_source(sensed_p, mb, kgrid, Medium, Microbubble, Transmit, step_control)
    source = update_source(source_transducer, ms, weights_t, sensor_mask_idx_frame, grid, medium)
    return run_simulation(run_param, kgrid, medium, source, sensor_transducer, **kw)


def _as_source_dict(source: Any) -> dict[str, Any] | None:
    if source is None or isinstance(source, dict):
        return source
    if dataclasses.is_dataclass(source):
        return {
            f.name: getattr(source, f.name)
            for f in dataclasses.fields(source)
            if getattr(source, f.name) is not None
        }
    return dict(vars(source))
