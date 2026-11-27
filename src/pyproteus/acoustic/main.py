"""Entry points: ports of ``main_RF.m`` and ``main_pressure_field.m``.

:func:`main_RF` simulates the element RF data of every frame and writes them
to one HDF5 file per run (D-25, :class:`pyproteus.io.RFWriter`);
:func:`main_pressure_field` the peak positive/negative pressure maps in the
lateral-axial and elevation-axial planes.

Differences with MATLAB (docs/PARITY.md):

* Settings may be a ``.mat``/``.json`` file or a :class:`~pyproteus.config.Settings`
  object; folders are explicit arguments (no ``path_setup``).
* RF output: ``<savefolder>/rf.h5`` instead of ``Frame_<m>.mat`` files; every
  run resumes the file; with ``Continue`` the frames already written are skipped
  (B-166) and the k-Wave medium is read from ``medium.h5`` after a fingerprint
  check (MATLAB ``medium.mat``, unchecked).
* The random medium draws come from ``rng`` (a
  :class:`~pyproteus.numerics.MatlabRNG`, default ``rng(0)``: MATLAB's start-up
  stream); MATLAB draws from whatever state its global stream is in.
* Ground-truth frames are read once per frame (MATLAB: once per pulse), and the
  sensor mask of the first k-Wave run covers only the frames still to compute.
* Options not in MATLAB: ``self_sense`` (D-27), ``step_control`` (D-22),
  ``binary_env``/``num_threads`` for the k-Wave binary.
"""

from __future__ import annotations

import dataclasses
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from pyproteus.backends.kwave import KGrid
from pyproteus.config.models import Settings
from pyproteus.io.frames import (
    RFWriter,
    load_medium_cache,
    medium_fingerprint,
    save_medium_cache,
)
from pyproteus.medium import define_medium
from pyproteus.numerics import MatlabRNG

from . import _struct
from .grid import Grid, define_grid
from .rf import compute_RF_data
from .sensor import (
    Sensor,
    _find,
    define_sensor_MB,
    define_sensor_MB_all,
    define_sensor_transducer,
    extract_sensor_data,
)
from .setup import (
    RunSetup,
    frames_to_simulate,
    load_microbubbles,
    read_frame_tree,
    setup_run,
)
from .simulation import run_simulation
from .simulators import SelfSenseMode, full_simulator, hybrid_simulator
from .source import define_source_transducer, preprocess_transmit
from .transducer_integration import (
    compute_travel_times,
    get_transducer_integration_delays,
    get_transducer_integration_points,
)

__all__ = ["Prepared", "PressureMaps", "main_RF", "main_pressure_field", "prepare_simulation"]


def _nt(t_end: float, dt: float) -> int:
    return int(np.floor(t_end / dt)) + 1


@dataclass
class Prepared:
    """State shared by both entry points after the transducer source is ready."""

    settings: Settings
    grid: Grid
    kgrid: KGrid
    medium: dict[str, Any]
    transducer: Any  # Transducer with integration points and delays
    transmit: Any  # Transmit after preprocess_transmit
    run_param: dict[str, Any]
    sensor_transducer: Sensor
    sensor_weights: Any  # CSR, N_points x nnz(mask)
    mask_idx_trans: np.ndarray


def _rng_state(rng: MatlabRNG) -> str:
    """Hash of the stream position, so an advanced stream gets another fingerprint."""
    import hashlib

    name, keys, pos, has_gauss, cached = rng._rs.get_state()
    h = hashlib.sha256(np.asarray(keys).tobytes())
    h.update(f"{name}{pos}{has_gauss}{cached!r}".encode())
    return h.hexdigest()[:16]


def _medium(
    setup: RunSetup, grid: Grid, rng: MatlabRNG, use_cache: bool, save: bool
) -> dict[str, Any]:
    s = setup.settings
    fp = medium_fingerprint(s, seed=rng.seed, state=_rng_state(rng))
    path = setup.paths.medium_cache
    if use_cache:
        if path.is_file():
            return load_medium_cache(path, fingerprint=fp).medium
        save = bool(_struct.get(s.Medium, "Save", True))  # build it once, then reuse it
        warnings.warn(f"Continue: no medium cache at {path}, creating the medium", stacklevel=3)
    medium, vessel = define_medium(
        grid,
        s.Medium,
        s.Geometry,
        s.Transducer,
        rng=rng,
        geometries_path=_struct.get(s.Geometry, "GeometriesPath"),
    )
    if save:
        save_medium_cache(path, medium, vessel, grid, fingerprint=fp)
    return medium


def prepare_simulation(
    setup: RunSetup,
    *,
    rng: MatlabRNG | None = None,
    use_medium_cache: bool = False,
    pml_rounding: str = "ceil",
) -> Prepared:
    """``main_RF.m`` from ``define_grid`` to ``define_sensor_transducer``.

    Grid → k-Wave grid → medium (cache) → transducer integration points and
    delays → travel times (``run_param.tr``) → ``preprocess_transmit`` →
    transducer sensor and weights.
    """
    s = setup.settings
    rng = MatlabRNG(0) if rng is None else rng
    grid = define_grid(s.SimulationParameters, s.Geometry, pml_rounding=pml_rounding)
    save = bool(_struct.get(s.Medium, "Save", True)) and not use_medium_cache
    medium = _medium(setup, grid, rng, use_medium_cache, save)
    T = get_transducer_integration_points(s.Transducer, grid)
    T = get_transducer_integration_delays(T, s.Medium)
    run_param = compute_travel_times(setup.run_param, s.Geometry, s.Medium, T, s.Transmit)
    run_param["PML"] = grid.PML
    kgrid = KGrid(
        grid.shape, (grid.dx, grid.dy, grid.dz), grid.dt, _nt(run_param["tr"][0], grid.dt)
    )
    transmit = preprocess_transmit(s.Transmit, s.Medium, kgrid)
    sensor_t, weights = define_sensor_transducer(T, grid)
    return Prepared(
        settings=s,
        grid=grid,
        kgrid=kgrid,
        medium=medium,
        transducer=T,
        transmit=transmit,
        run_param=run_param,
        sensor_transducer=sensor_t,
        sensor_weights=weights,
        mask_idx_trans=_find(sensor_t.mask),
    )


def _with_seq_pulse(transmit: Any, pulse: str) -> Any:
    out = _struct.copy(transmit)
    _struct.set_field(out, "SeqPulse", pulse)
    return out


def main_RF(
    settingsfile: str | PathLike[str] | Settings,
    groundtruthfolder: str | PathLike[str],
    savefolder: str | PathLike[str],
    *,
    continue_: bool | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
    device_number: int | None = None,
    rng: MatlabRNG | None = None,
    self_sense: SelfSenseMode = "impulse_response",
    step_control: str = "independent",
    gridded_receive: bool | None = None,
    binary_path: str | PathLike[str] | None = None,
    data_path: str | PathLike[str] | None = None,
    geometries_path: str | PathLike[str] | None = None,
    binary_env: Any = None,
    num_threads: int | None = None,
    self_sense_jobs: int = 1,
    pml_rounding: str = "ceil",
    verbose: bool = True,
) -> Path:
    """Port of ``main_RF.m``: simulate the RF data of frames ``StartFrame..EndFrame``.

    ``continue_``, ``start_frame``, ``end_frame``, ``device_number`` are
    MATLAB's ``varargin{1..4}`` (:func:`~pyproteus.acoustic.setup.input_handling`).
    Returns the RF file (``<savefolder>/rf.h5``, :class:`pyproteus.io.RFFile`
    reads it); each frame holds ``rf`` (``N_seq × N_el × Nt``), ``dt`` and the
    bubbles of each pulse (``points``, ``radii``, ``velocities``, ``nodes``
    and ``idx``, the latter two 0-based).

    Per frame and pulse: bubbles → sensor at the bubbles → first-run pressure
    at the bubbles → :func:`~pyproteus.acoustic.simulators.hybrid_simulator`
    (``SimulationParameters.HybridSimulation``) or
    :func:`~pyproteus.acoustic.simulators.full_simulator` →
    :func:`~pyproteus.acoustic.compute_RF_data`.

    ``gridded_receive``: see :func:`~pyproteus.acoustic.simulators.hybrid_simulator`
    (default: gridded on the CPU, exact on the GPU, D-30).
    """
    setup = setup_run(
        settingsfile,
        groundtruthfolder,
        savefolder,
        continue_=continue_,
        start_frame=start_frame,
        end_frame=end_frame,
        device_number=device_number,
        binary_path=binary_path,
        data_path=data_path,
        geometries_path=geometries_path,
    )
    s = setup.settings
    acq = s.Acquisition
    gt = setup.paths.ground_truth
    n_frames = int(_struct.scalar(acq.NumberOfFrames))
    say = print if verbose else (lambda *a, **k: None)
    say(f"RF data will be saved in: {setup.paths.rf_file}")

    writer = RFWriter(
        setup.paths.rf_file,
        settings=s,
        sequence=setup.sequence,
        num_frames=n_frames,
        flow_simulation_parameters=setup.flow_simulation_parameters,
        mode=setup.rf_writer_mode,
    )
    frames = frames_to_simulate(setup, writer.completed_frames())
    if not frames:
        say("All frames are already in the RF file.")
        return setup.paths.rf_file

    continue_run = bool(_struct.get(acq, "Continue", False))
    prep = prepare_simulation(
        setup, rng=rng, use_medium_cache=continue_run, pml_rounding=pml_rounding
    )
    grid, medium, run_param = prep.grid, prep.medium, prep.run_param
    hybrid = bool(_struct.get(s.SimulationParameters, "HybridSimulation"))
    sequence = setup.sequence
    geometry = s.Geometry

    def bubbles() -> Iterator[dict[str, np.ndarray]]:
        for frame in frames:
            tree = read_frame_tree(gt, frame, n_frames)
            for k in range(1, len(sequence) + 1):
                yield load_microbubbles(gt, frame, k, geometry, n_frames, frame_tree=tree)

    sensor_mb_all, mb_idx_all, _max_mb = define_sensor_MB_all(grid, bubbles())
    if np.intersect1d(mb_idx_all, prep.mask_idx_trans).size:
        warnings.warn("Microbubbles on transducer.", stacklevel=2)

    if hybrid:
        mask = np.logical_or(sensor_mb_all.mask, prep.sensor_transducer.mask)
        sensor = Sensor(mask=mask, record=sensor_mb_all.record)
        kgrid = dataclasses.replace(prep.kgrid, Nt=_nt(run_param["tr"][2], grid.dt))
    else:
        sensor = sensor_mb_all
        kgrid = dataclasses.replace(prep.kgrid, Nt=_nt(run_param["tr"][0], grid.dt))
    kw = {"binary_env": binary_env, "num_threads": num_threads}

    sources, first = [], []
    for pulse in sequence:
        say(f"Simulating transmit wave ({pulse}).")
        src = define_source_transducer(
            prep.transducer,
            _with_seq_pulse(prep.transmit, pulse),
            s.Medium,
            grid,
            prep.sensor_weights.T,
            prep.mask_idx_trans,
        )
        sources.append(src)
        first.append(run_simulation(run_param, kgrid, medium, src, sensor, **kw))
    transmit = _with_seq_pulse(prep.transmit, sequence[-1])  # MATLAB keeps the last SeqPulse
    mask_idx = _find(sensor.mask)

    for frame in frames:
        say(f"frame {frame}")
        tree = read_frame_tree(gt, frame, n_frames)
        rf, mbs = [], []
        for k in range(len(sequence)):
            mb = load_microbubbles(gt, frame, k + 1, geometry, n_frames, frame_tree=tree)
            sensor_frame, w_frame, mb, max_dist = define_sensor_MB(grid, mb)
            run_param["max_dist"] = max_dist
            mask_idx_frame = _find(sensor_frame.mask)
            data_mb, data_trans = extract_sensor_data(
                first[k], mask_idx, prep.mask_idx_trans, mask_idx_frame, run_param, kgrid
            )
            p_mb = data_mb["p"]
            sensed_p = np.asarray(w_frame @ p_mb.astype(np.float64)).astype(p_mb.dtype)
            if hybrid:
                sd = hybrid_simulator(
                    prep.mask_idx_trans,
                    sensed_p,
                    mb,
                    grid,
                    medium,
                    run_param,
                    s.Medium,
                    s.Microbubble,
                    transmit,
                    step_control=step_control,
                    gridded_receive=gridded_receive,
                )
                sensor_data = {"p": data_trans["p"] + np.asarray(sd["p"])}
            else:
                sensor_data = full_simulator(
                    sources[k],
                    prep.sensor_transducer,
                    sensor_frame,
                    w_frame,
                    mask_idx_frame,
                    sensed_p,
                    mb,
                    kgrid,
                    grid,
                    medium,
                    run_param,
                    s.Medium,
                    s.Microbubble,
                    transmit,
                    self_sense=self_sense,
                    step_control=step_control,
                    self_sense_jobs=self_sense_jobs,
                    **kw,
                )
            rf_k, run_param = compute_RF_data(
                prep.transducer, sensor_data, prep.sensor_weights, grid, run_param
            )
            rf.append(rf_k)
            mbs.append(mb)
        writer.write_frame(frame, rf, grid.dt, mbs, replace=setup.replace_frames)
    return setup.paths.rf_file


@dataclass
class PressureMaps:
    """Output of :func:`main_pressure_field` (MATLAB ``sensor_data_xy``/``sensor_data_xz``).

    ``xy_*`` are ``Nx × Ny`` maps in the plane ``z = Grid.z[iz]`` closest to 0,
    ``xz_*`` are ``Nx × Nz`` maps in the plane ``y = Grid.y[iy]`` closest to 0 [Pa].
    """

    xy_p_max: np.ndarray
    xy_p_min: np.ndarray
    xz_p_max: np.ndarray
    xz_p_min: np.ndarray
    grid: Grid
    iy: int
    iz: int

    def save(self, path: str | PathLike[str]) -> None:
        """HDF5: ``/xy/p_max`` ... ``/xz/p_min``, grid coordinates under ``/grid``."""
        with h5py.File(path, "w") as f:
            for plane in ("xy", "xz"):
                g = f.create_group(plane)
                for q in ("p_max", "p_min"):
                    g.create_dataset(q, data=getattr(self, f"{plane}_{q}"))
            g = f.create_group("grid")
            for name in ("x", "y", "z"):
                g.create_dataset(name, data=np.asarray(getattr(self.grid, name)))
            g.attrs["iy"], g.attrs["iz"] = self.iy, self.iz
            g.attrs["dt"] = self.grid.dt


def main_pressure_field(
    settingsfile: str | PathLike[str] | Settings,
    savefolder: str | PathLike[str],
    *,
    device_number: int | None = None,
    rng: MatlabRNG | None = None,
    binary_path: str | PathLike[str] | None = None,
    data_path: str | PathLike[str] | None = None,
    geometries_path: str | PathLike[str] | None = None,
    binary_env: Any = None,
    num_threads: int | None = None,
    pml_rounding: str = "ceil",
    save: bool = True,
) -> PressureMaps:
    """Port of ``main_pressure_field.m``: peak pressure maps of the full transmit.

    All elements transmit (``SeqPulse = 'full'``); one k-Wave run records
    ``p_max`` and ``p_min`` in the planes ``z ≈ 0`` and ``y ≈ 0`` (first grid
    index closest to 0). With ``save``, writes ``<savefolder>/pressure_maps.h5``
    (MATLAB ``pressure_maps.mat``) and ``medium.h5`` when ``Medium.Save``.
    """
    setup = setup_run(
        settingsfile,
        "",
        savefolder,
        device_number=device_number,
        binary_path=binary_path,
        data_path=data_path,
        geometries_path=geometries_path,
        check_ground_truth=False,
    )
    s = setup.settings
    if not save:
        s.Medium.Save = False
    prep = prepare_simulation(setup, rng=rng, pml_rounding=pml_rounding)
    grid = prep.grid
    src = define_source_transducer(
        prep.transducer,
        _with_seq_pulse(prep.transmit, "full"),
        s.Medium,
        grid,
        prep.sensor_weights.T,
        prep.mask_idx_trans,
    )
    iy = int(np.argmin(np.abs(np.asarray(grid.y))))
    iz = int(np.argmin(np.abs(np.asarray(grid.z))))
    mask = np.zeros(grid.shape, dtype=bool)
    mask[:, :, iz] = True
    mask[:, iy, :] = True
    sensor = Sensor(mask=mask, record=("p_max", "p_min"))
    sd = run_simulation(
        prep.run_param,
        prep.kgrid,
        prep.medium,
        src,
        sensor,
        binary_env=binary_env,
        num_threads=num_threads,
    )
    idx_all = _find(mask)
    xy = np.zeros(grid.shape, dtype=bool)
    xy[:, :, iz] = True
    xz = np.zeros(grid.shape, dtype=bool)
    xz[:, iy, :] = True
    rows_xy = np.searchsorted(idx_all, _find(xy))
    rows_xz = np.searchsorted(idx_all, _find(xz))

    def plane(q: str, rows: np.ndarray, n2: int) -> np.ndarray:
        return np.asarray(sd[q]).reshape(-1)[rows].reshape(grid.Nx, n2, order="F")

    maps = PressureMaps(
        xy_p_max=plane("p_max", rows_xy, grid.Ny),
        xy_p_min=plane("p_min", rows_xy, grid.Ny),
        xz_p_max=plane("p_max", rows_xz, grid.Nz),
        xz_p_min=plane("p_min", rows_xz, grid.Nz),
        grid=grid,
        iy=iy,
        iz=iz,
    )
    if save:
        out = Path(setup.paths.results)
        out.mkdir(parents=True, exist_ok=True)
        maps.save(out / "pressure_maps.h5")
    return maps
