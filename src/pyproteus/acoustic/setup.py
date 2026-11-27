"""Run setup for ``main_RF`` / ``main_pressure_field``.

Ports of ``sim_startup.m``, ``input_handling.m``, ``sim_setup.m``,
``check_ground_truth_data.m`` and ``load_microbubbles.m``
(``compute_travel_times.m`` is :func:`pyproteus.acoustic.compute_travel_times`).

Differences with MATLAB (D-15, ``docs/PARITY.md``):

* No MATLAB path handling (``path_setup``/``addpath``): folders are explicit
  arguments; the k-Wave binary comes from
  :func:`pyproteus.backends.kwave.binary_path`.
* The run configuration is a typed :class:`RunConfig`;
  :meth:`RunConfig.run_param` gives the MATLAB ``run_param`` struct (dict with
  MATLAB field names) that the ported functions read.
* :func:`check_ground_truth_data` raises/warns and returns the ground-truth
  parameters; it writes nothing unless ``savedir`` is given (MATLAB always
  creates the results folder and saves a copy there). With the HDF5 RF file
  (D-25) the parameters go into that file instead.
* :func:`input_handling` does not reset the GPU (MATLAB ``gpuDevice(n+1)``);
  the device number is passed to the k-Wave CUDA binary. A start frame after
  the end frame raises (MATLAB silently runs no frame).
"""

from __future__ import annotations

import shutil
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral, Real
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from pyproteus.acoustic import _struct
from pyproteus.acoustic.source import PULSE_SEQUENCES, pulse_sequence
from pyproteus.config.json_io import load_settings_json
from pyproteus.config.models import Settings
from pyproteus.config.readers import load_frame, load_settings_mat
from pyproteus.io.frames import (
    FlowSimulationParameters,
    read_flow_simulation_parameters,
    read_ground_truth_pulse,
)

__all__ = [
    "PULSE_SEQUENCES",
    "SOLVERS",
    "GroundTruthDistributionWarning",
    "GroundTruthMismatch",
    "RunConfig",
    "RunPaths",
    "RunSetup",
    "check_ground_truth_data",
    "frames_to_simulate",
    "input_handling",
    "load_microbubbles",
    "load_settings",
    "make_run_config",
    "pulse_sequence",
    "read_frame_tree",
    "setup_run",
    "sim_setup",
    "sim_startup",
    "transform_microbubbles",
]

# ================================================================= solvers


@dataclass(frozen=True)
class _Solver:
    name: str  # MATLAB k-Wave function
    backend: str  # k-wave-python backend
    device: str  # 'cpu' | 'gpu'
    data_cast: str
    record_movie: bool


#: ``SimulationParameters.Solver`` → k-Wave solver (``sim_setup.m``) and the
#: k-wave-python backend that replaces it.
SOLVERS: dict[str, _Solver] = {
    "3D": _Solver("kspaceFirstOrder3D", "python", "cpu", "single", True),
    "3DC": _Solver("kspaceFirstOrder3DC", "cpp", "cpu", "single", False),
    "3DG": _Solver("kspaceFirstOrder3DG", "cpp", "gpu", "gpuArray-single", False),
}


@dataclass(frozen=True)
class RunConfig:
    """Typed form of MATLAB ``run_param`` after ``sim_setup.m``.

    ``backend``/``device`` say how :mod:`pyproteus.backends.kwave` runs k-Wave;
    ``binary_path`` is the C++ binary (``None`` for the ``'3D'`` solver, which
    runs in-process); ``data_path`` the folder for k-Wave's HDF5 input/output
    files (``None``: a temporary folder).
    """

    solver: str
    backend: str
    device: str
    data_cast: str
    data_cast_rf: str
    record_movie: bool
    n_interactions: int
    device_number: int | None = None
    binary_path: Path | None = None
    data_path: Path | None = None
    geometries_path: Path | None = None
    microbubbles_batch_size: int = 15
    microbubbles_use_parfor: str | bool = "auto"

    def run_param(self) -> dict[str, Any]:
        """MATLAB ``run_param`` (field names of ``sim_setup.m``)."""
        rp: dict[str, Any] = {
            "solver": SOLVERS[self.solver].name,
            "DATA_CAST": self.data_cast,
            "record_movie": self.record_movie,
        }
        if self.device_number is not None:
            rp["DEVICE_NUM"] = self.device_number
        if self.solver != "3D":
            rp["DATA_PATH"] = None if self.data_path is None else str(self.data_path)
            rp["BINARY_PATH"] = None if self.binary_path is None else str(self.binary_path)
        rp["DATA_CAST_RF"] = self.data_cast_rf
        if self.geometries_path is not None:
            rp["GeometriesPath"] = str(self.geometries_path)
        rp["N_interactions"] = self.n_interactions
        rp["MicrobubblesBatchSize"] = self.microbubbles_batch_size
        rp["MicrobubblesUseParfor"] = self.microbubbles_use_parfor
        rp["backend"] = self.backend
        rp["device"] = self.device
        return rp


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool | np.bool_):
        raise ValueError(f"{name} must be an integer, got a logical")
    v = _struct.scalar(value)
    if not np.isfinite(v) or v != np.floor(v):
        raise ValueError(f"{name} must be an integer, got {v!r}")
    return int(v)


def make_run_config(
    simulation_parameters: Any,
    *,
    binary_path: str | PathLike[str] | None = None,
    data_path: str | PathLike[str] | None = None,
    geometries_path: str | PathLike[str] | None = None,
    resolve_binary: bool = True,
) -> RunConfig:
    """``sim_setup.m`` as a :class:`RunConfig`.

    ``binary_path`` (file or folder, MATLAB ``PATHS.BinaryPath``) is resolved
    with :func:`pyproteus.backends.kwave.binary_path` for the ``'3DC'`` (OMP)
    and ``'3DG'`` (CUDA) solvers; ``resolve_binary=False`` skips the lookup
    (configuration only, e.g. on a machine without the binary).
    """
    solver = _struct.get(simulation_parameters, "Solver")
    if solver not in SOLVERS:
        raise ValueError(
            f"SimulationParameters.Solver must be one of {list(SOLVERS)}, got {solver!r}"
        )
    spec = SOLVERS[solver]
    device_number = None
    if solver == "3DG":
        device_number = _integer(_struct.get(simulation_parameters, "DeviceNumber"), "DeviceNumber")
    binary = None
    if solver != "3D":
        if resolve_binary:
            from pyproteus.backends.kwave import binary_path as _binary_path

            binary = _binary_path(spec.device, binary_path)
        elif binary_path is not None:
            binary = Path(binary_path)
    n_int = _integer(
        _struct.get(simulation_parameters, "NumberOfInteractions"), "NumberOfInteractions"
    )
    return RunConfig(
        solver=solver,
        backend=spec.backend,
        device=spec.device,
        data_cast=spec.data_cast,
        data_cast_rf=spec.data_cast,
        record_movie=spec.record_movie,
        n_interactions=n_int,
        device_number=device_number,
        binary_path=binary,
        data_path=None if data_path is None else Path(data_path),
        geometries_path=None if geometries_path is None else Path(geometries_path),
    )


def sim_setup(simulation_parameters: Any, **kwargs: Any) -> dict[str, Any]:
    """Port of ``sim_setup.m``: ``run_param`` (dict). Keywords as :func:`make_run_config`."""
    return make_run_config(simulation_parameters, **kwargs).run_param()


# ================================================================= input_handling


def input_handling(
    acquisition: Any,
    simulation_parameters: Any,
    *,
    continue_: bool | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
    device_number: int | None = None,
) -> tuple[Any, Any]:
    """Port of ``input_handling.m`` (``main_RF``'s optional arguments ``varargin{1..4}``).

    Returns copies of ``acquisition`` with ``Continue`` (default ``False``),
    ``StartFrame`` (default 1) and ``EndFrame`` (default and upper limit
    ``NumberOfFrames``) set, and of ``simulation_parameters`` with
    ``DeviceNumber`` when ``device_number`` is given. ``None`` means "not
    given" (MATLAB ``[]``).
    """
    acq = _struct.copy(acquisition)
    sp = _struct.copy(simulation_parameters)
    n_frames = _integer(_struct.get(acq, "NumberOfFrames"), "Acquisition.NumberOfFrames")
    cont, start, end = False, 1, n_frames

    if continue_ is not None:
        if not isinstance(continue_, bool | np.bool_):
            raise ValueError("continue_ (varargin{1}) must be a boolean value.")
        cont = bool(continue_)
    if start_frame is not None:
        start = _positive_integer(start_frame, "start_frame (varargin{2})")
    if end_frame is not None:
        if isinstance(end_frame, float) and end_frame == np.inf:
            end = n_frames  # MATLAB accepts Inf (floor(Inf) == Inf): to the last frame
        else:
            end = min(_positive_integer(end_frame, "end_frame (varargin{3})"), n_frames)
    if device_number is not None:
        dev = _integer(device_number, "device_number (varargin{4})")
        if dev < 0:
            raise ValueError("device_number (varargin{4}) counts from zero.")
        _struct.set_field(sp, "DeviceNumber", float(dev))
    if start > end:
        raise ValueError(f"start frame {start} is after the end frame {end}: no frame to simulate")

    _struct.set_field(acq, "Continue", cont)
    _struct.set_field(acq, "StartFrame", float(start))
    _struct.set_field(acq, "EndFrame", float(end))
    return acq, sp


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool | np.bool_) or not isinstance(value, Integral | Real | np.ndarray):
        raise ValueError(f"{name} must be a positive integer.")
    try:
        v = _integer(value, name)
    except ValueError:
        raise ValueError(f"{name} must be a positive integer.") from None
    if v <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return v


# ================================================================= sim_startup


@dataclass(frozen=True)
class RunPaths:
    """Files and folders of one run (``sim_startup.m``), plus the D-25 output files."""

    settings_file: Path | None  # None when setup_run was given a Settings object
    ground_truth: Path
    results: Path

    @property
    def rf_file(self) -> Path:
        """HDF5 RF output (D-25), ``<results>/rf.h5``."""
        return self.results / "rf.h5"

    @property
    def medium_cache(self) -> Path:
        """k-Wave medium cache, ``<results>/medium.h5`` (MATLAB ``medium.mat``)."""
        return self.results / "medium.h5"


def sim_startup(
    settingsfile: str | PathLike[str],
    groundtruthfolder: str | PathLike[str],
    savefolder: str | PathLike[str],
    *,
    settings_dir: str | PathLike[str] | None = None,
    ground_truth_dir: str | PathLike[str] | None = None,
    results_dir: str | PathLike[str] | None = None,
) -> RunPaths:
    """Port of ``sim_startup.m``: full paths of the settings file and run folders.

    A settings name without extension gets ``.mat``. It is looked up in
    ``settings_dir`` (MATLAB ``PATHS.SettingsPath``) first and taken as given
    otherwise; a missing file raises ``FileNotFoundError``. The ground-truth
    and results folders are joined to ``ground_truth_dir``/``results_dir``
    when given (MATLAB ``PATHS.GroundTruthPath``/``ResultsPath``). Nothing is
    created.
    """
    settings = Path(settingsfile)
    if not settings.suffix:
        settings = settings.with_name(settings.name + ".mat")
    candidate = Path(settings_dir) / settings if settings_dir is not None else settings
    if not candidate.is_file():
        candidate = settings
    if not candidate.is_file():
        raise FileNotFoundError(f"File {candidate} not found")
    gt = (
        Path(ground_truth_dir) / groundtruthfolder
        if ground_truth_dir is not None
        else Path(groundtruthfolder)
    )
    res = Path(results_dir) / savefolder if results_dir is not None else Path(savefolder)
    return RunPaths(settings_file=candidate, ground_truth=gt, results=res)


def load_settings(path: str | PathLike[str]) -> Settings:
    """Settings from a MATLAB ``.mat`` or a pyproteus ``.json`` file."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        return load_settings_json(path)
    return load_settings_mat(path)


# ================================================================= ground truth


class GroundTruthMismatch(ValueError):
    """Ground-truth acquisition parameters differ from the simulation settings.

    ``parameters`` lists every mismatch; the message names the one MATLAB
    reports (the last check that fails).
    """

    def __init__(self, parameters: list[str]) -> None:
        self.parameters = parameters
        p = parameters[-1]
        super().__init__(
            f"The {p} in the ground truth data and the {p} in the simulation settings do not match."
        )


class GroundTruthDistributionWarning(UserWarning):
    """Ground-truth microbubble size distribution differs from the settings."""


def _digits6(x: float) -> float:
    """``str2double(num2str(x, 6))``: ``x`` rounded to 6 significant digits (``%.6g``)."""
    return float(f"{x:.6g}")


def check_ground_truth_data(
    folder: str | PathLike[str],
    acquisition: Any,
    microbubble: Any,
    savedir: str | PathLike[str] | None = None,
) -> FlowSimulationParameters:
    """Port of ``check_ground_truth_data.m``.

    Compares ``<folder>/FlowSimulationParameters.mat`` with the settings: time
    between pulses (to 6 significant digits, only with more than one pulse),
    number of pulses, number of microbubbles, frame rate (6 digits), number of
    frames; any mismatch raises :class:`GroundTruthMismatch`. A different size
    distribution (``max|ΔP|/max P`` or ``max|ΔR|/max R`` above 1e-6, or other
    lengths) only warns (:class:`GroundTruthDistributionWarning`).

    Returns the parameters. With ``savedir`` given, ``savedir`` is created if
    needed and ``FlowSimulationParameters.mat`` is copied there (MATLAB
    always does this).
    """
    fsp = read_flow_simulation_parameters(folder)
    s = _struct.scalar
    mismatches = []
    if fsp.NBPulses > 1 and _digits6(fsp.TimeBtwPulse) != _digits6(
        s(_struct.get(acquisition, "TimeBetweenPulses"))
    ):
        mismatches.append("time between pulses")
    if fsp.NBPulses != s(_struct.get(acquisition, "NumberOfPulses")):
        mismatches.append("number of pulses")
    if fsp.NMicrobubbles != s(_struct.get(microbubble, "Number")):
        mismatches.append("number of microbubbles")
    if _digits6(fsp.FrameRate) != _digits6(s(_struct.get(acquisition, "FrameRate"))):
        mismatches.append("frame rate")
    if fsp.NumberOfFrames != s(_struct.get(acquisition, "NumberOfFrames")):
        mismatches.append("number of frames")
    if mismatches:
        raise GroundTruthMismatch(mismatches)

    dist = _struct.get(microbubble, "Distribution")
    p1 = np.asarray(_struct.get(dist, "Probabilities"), dtype=float).reshape(-1, order="F")
    r1 = np.asarray(_struct.get(dist, "Radii"), dtype=float).reshape(-1, order="F")
    p2, r2 = (a.reshape(-1, order="F") for a in fsp.distribution())
    tol = 1e-6
    same = (
        p1.size == p2.size
        and r1.size == r2.size
        and not np.max(np.abs(p1 - p2)) / np.max(p1) > tol
        and not np.max(np.abs(r1 - r2)) / np.max(r1) > tol
    )
    if not same:
        warnings.warn(
            "The microbubble size distribution in the ground truth data and the microbubble "
            "size distribution in the simulation settings do not match.",
            GroundTruthDistributionWarning,
            stacklevel=2,
        )

    if savedir is not None:
        out = Path(savedir)
        out.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(
            Path(folder) / "FlowSimulationParameters.mat", out / "FlowSimulationParameters.mat"
        )
    return fsp


def transform_microbubbles(pulse: Mapping[str, Any], geometry: Any) -> dict[str, np.ndarray]:
    """Place ground-truth bubbles in the simulation frame (body of ``load_microbubbles.m``).

    ``points = R·(P - BoundingBox.Center) + Geometry.Center`` and
    ``velocities = V·Rᵀ`` with ``R = Geometry.Rotation``. Returns the MB dict
    ``{"points": N×3, "radii": N×1, "velocities": N×3}`` [m, m, m/s].
    """
    rot = np.asarray(_struct.get(geometry, "Rotation"), dtype=float)
    bb_center = np.asarray(_struct.get(_struct.get(geometry, "BoundingBox"), "Center"), dtype=float)
    center = np.asarray(_struct.get(geometry, "Center"), dtype=float)
    if rot.shape != (3, 3) or bb_center.size != 3 or center.size != 3:
        raise ValueError("Geometry.Rotation must be 3×3; BoundingBox.Center and Center 3-vectors")
    pts = np.asarray(pulse["Points"], dtype=float)
    vel = np.asarray(pulse["Velocity"], dtype=float)
    radii = np.asarray(pulse["Radius"], dtype=float)
    points = (pts - bb_center.reshape(1, 3)) @ rot.T + center.reshape(1, 3)
    return {
        "points": points,
        "radii": radii.reshape(-1, 1) if radii.ndim != 2 else radii,
        "velocities": vel @ rot.T,
    }


def load_microbubbles(
    folder: str | PathLike[str],
    frame: int,
    pulse_seq_idx: int,
    geometry: Any,
    num_frames: int,
    *,
    frame_tree: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """Port of ``load_microbubbles.m``: bubbles of pulse ``pulse_seq_idx`` in frame ``frame``.

    Reads ``Frame.Pulse<k>`` of ``<folder>/Frame_<frame>.mat`` (zero padded to
    the digits of ``num_frames``) and applies :func:`transform_microbubbles`.
    ``frame_tree`` (from :func:`pyproteus.config.load_frame`) skips the file
    read, so a frame is read once for all its pulses (MATLAB reads it per pulse).
    """
    pulse = read_ground_truth_pulse(folder, frame, pulse_seq_idx, num_frames, frame_tree=frame_tree)
    return transform_microbubbles(pulse, geometry)


# ================================================================= whole setup


@dataclass
class RunSetup:
    """Everything ``main_RF.m`` prepares before building the grid."""

    paths: RunPaths
    settings: Settings
    config: RunConfig
    run_param: dict[str, Any]
    sequence: tuple[str, ...]
    flow_simulation_parameters: FlowSimulationParameters | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def rf_writer_mode(self) -> str:
        """Always ``"resume"``: open the run's RF file if it exists (same settings up to
        run control, else ``ValueError``), create it otherwise. Frames outside
        ``StartFrame..EndFrame`` are kept, as MATLAB keeps the ``Frame_<m>.mat`` files it
        does not rewrite; nothing is truncated before a frame is computed."""
        return "resume"

    @property
    def replace_frames(self) -> bool:
        """Pass to :meth:`RFWriter.write_frame`: without ``Continue`` the frames of
        ``StartFrame..EndFrame`` are recomputed and replace earlier versions (MATLAB)."""
        return not bool(_struct.get(self.settings.Acquisition, "Continue", False))

    def frames(self) -> list[int]:
        """Frames ``StartFrame..EndFrame``."""
        acq = self.settings.Acquisition
        return list(
            range(int(_struct.scalar(acq.StartFrame)), int(_struct.scalar(acq.EndFrame)) + 1)
        )


def frames_to_simulate(setup: RunSetup, completed: list[int] | None = None) -> list[int]:
    """Frames to compute: ``StartFrame..EndFrame``, minus ``completed`` when ``Continue`` is set.

    Divergence (B-166): in MATLAB ``Continue`` only reuses the saved k-Wave medium and
    every frame of the range is recomputed; pyproteus also skips frames already in the
    RF file, so an interrupted run continues where it stopped. Without ``Continue`` all
    frames of the range are computed and written with ``replace=setup.replace_frames``.
    """
    frames = setup.frames()
    if completed and bool(_struct.get(setup.settings.Acquisition, "Continue", False)):
        done = set(completed)
        frames = [m for m in frames if m not in done]
    return frames


def setup_run(
    settingsfile: str | PathLike[str] | Settings,
    groundtruthfolder: str | PathLike[str],
    savefolder: str | PathLike[str],
    *,
    continue_: bool | None = None,
    start_frame: int | None = None,
    end_frame: int | None = None,
    device_number: int | None = None,
    settings_dir: str | PathLike[str] | None = None,
    ground_truth_dir: str | PathLike[str] | None = None,
    results_dir: str | PathLike[str] | None = None,
    binary_path: str | PathLike[str] | None = None,
    data_path: str | PathLike[str] | None = None,
    geometries_path: str | PathLike[str] | None = None,
    resolve_binary: bool = True,
    check_ground_truth: bool = True,
) -> RunSetup:
    """The preamble of ``main_RF.m`` up to ``check_ground_truth_data``.

    ``sim_startup`` → load settings → ``input_handling`` → ``sim_setup`` →
    ``Microbubble.BatchSize``/``UseParfor``, ``Transducer.OnGrid``/
    ``IntegrationDensity`` (defaults ``False``/1), ``Medium.Save`` (default
    ``True``), ``Geometry.GeometriesPath`` → ``check_ground_truth_data``.
    A :class:`~pyproteus.config.Settings` object may replace the settings file.
    """
    if isinstance(settingsfile, Settings):
        settings = settingsfile.model_copy(deep=True)
        gt = (
            Path(ground_truth_dir) / groundtruthfolder
            if ground_truth_dir is not None
            else Path(groundtruthfolder)
        )
        res = Path(results_dir) / savefolder if results_dir is not None else Path(savefolder)
        paths = RunPaths(settings_file=None, ground_truth=gt, results=res)
    else:
        paths = sim_startup(
            settingsfile,
            groundtruthfolder,
            savefolder,
            settings_dir=settings_dir,
            ground_truth_dir=ground_truth_dir,
            results_dir=results_dir,
        )
        settings = load_settings(paths.settings_file)
    missing = [g for g in Settings.GROUPS if getattr(settings, g) is None]
    if missing:
        raise ValueError(f"settings are missing {missing}")

    acq, sp = input_handling(
        settings.Acquisition,
        settings.SimulationParameters,
        continue_=continue_,
        start_frame=start_frame,
        end_frame=end_frame,
        device_number=device_number,
    )
    settings.Acquisition = acq
    settings.SimulationParameters = sp
    config = make_run_config(
        sp,
        binary_path=binary_path,
        data_path=data_path,
        geometries_path=geometries_path,
        resolve_binary=resolve_binary,
    )
    run_param = config.run_param()

    mb = settings.Microbubble
    mb.BatchSize = float(config.microbubbles_batch_size)
    mb.UseParfor = config.microbubbles_use_parfor
    tr = settings.Transducer
    tr.OnGrid = sp.TransducerOnGrid if sp.isfield("TransducerOnGrid") else False
    tr.IntegrationDensity = sp.IntegrationDensity if sp.isfield("IntegrationDensity") else 1.0
    if geometries_path is not None:
        settings.Geometry.GeometriesPath = str(geometries_path)
    if not settings.Medium.isfield("Save"):
        settings.Medium.Save = True

    fsp = None
    if check_ground_truth:
        fsp = check_ground_truth_data(paths.ground_truth, acq, mb)
    return RunSetup(
        paths=paths,
        settings=settings,
        config=config,
        run_param=run_param,
        sequence=pulse_sequence(acq.PulsingScheme),
        flow_simulation_parameters=fsp,
    )


def read_frame_tree(folder: str | PathLike[str], frame: int, num_frames: int) -> dict[str, Any]:
    """``Frame`` struct of a ground-truth file (pass to :func:`load_microbubbles`)."""
    return load_frame(folder, frame, num_frames)
