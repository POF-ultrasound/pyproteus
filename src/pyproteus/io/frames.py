"""Ground-truth frames, RF output and the k-Wave medium cache.

Three kinds of run data (``main_RF.m``, ``load_microbubbles.m``,
``check_ground_truth_data.m``, ``load_RF_data.m``):

**Ground truth (input, MATLAB files).** ``generate_streamlines.m`` writes
``FlowSimulationParameters.mat`` and one ``Frame_<m>.mat`` per frame (zero
padded to the digits of ``NumberOfFrames``) holding ``Frame.Pulse<n>`` with
``Points`` (N×3 [m]), ``Velocity`` (N×3 [m/s]), ``Radius`` (N×1 [m]) and
``StreamNumber`` (N×1). :func:`read_flow_simulation_parameters` and
:func:`read_ground_truth_pulse` read them (v5/v7 and v7.3) on top of
:mod:`pyproteus.config.readers`.

**RF output (D-25).** One HDF5 file per run instead of MATLAB's
``Frame_<m>.mat`` per frame. Layout (format ``pyproteus-rf``, version 1)::

    /                      attrs: format, format_version, pyproteus_version,
                                  created_utc, num_frames, pulsing_scheme,
                                  sequence (JSON list of pulse names)
    /settings              str: settings JSON (pyproteus.config, lossless)
    /flow_simulation_parameters   str: JSON value tree (optional)
    /frames/Frame_<m>      attrs: frame (1-based int), dt [s], written_utc
        rf                 (n_pulses, Nelem, Nt), class of the RF (single by default)
        bubbles/pulse_<k>  the frame's microbubbles for pulse k (1-based):
                           points (N×3), radii (N×1), velocities (N×3) and any
                           other numeric field of the MB struct

``RF{k}`` of MATLAB is ``rf[k-1]``; ``Frame{k}`` is ``bubbles/pulse_<k>``.
:class:`RFWriter` writes one frame group at a time: the group is built under a
temporary name and renamed when complete, and the file is closed after each
frame, so a run that stops leaves every finished frame readable and can resume
(``mode="resume"``; ``Acquisition.Continue``/``StartFrame``). :class:`RFFile`
reads it; :func:`read_matlab_rf_frame` reads MATLAB's ``Frame_<m>.mat`` (``RF``
cell, ``dt``, ``Frame`` cell); :func:`iter_rf_source` reads either.

**k-Wave medium cache.** MATLAB saves ``medium.mat`` (v7.3: ``medium``,
``vessel_grid``, ``Grid``) and reloads it when ``Continue`` is set, without
checking that it belongs to the settings. :func:`save_medium_cache` writes an
HDF5 file with a fingerprint of the settings that define the medium;
:func:`load_medium_cache` refuses a cache with another fingerprint and also
reads MATLAB's ``medium.mat``. See ``docs/PARITY.md``.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import tempfile
import time
import warnings
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from pyproteus.config.json_io import (
    settings_from_json,
    settings_to_json,
    tree_from_jsonable,
    tree_to_jsonable,
)
from pyproteus.config.matfile import loadmat
from pyproteus.config.models import Settings
from pyproteus.config.readers import frame_file_name, load_flow_simulation_parameters, load_frame

__all__ = [
    "MEDIUM_FORMAT",
    "RF_FORMAT",
    "RF_FORMAT_VERSION",
    "RUN_CONTROL_FIELDS",
    "FlowSimulationParameters",
    "MediumCache",
    "MediumCacheMismatch",
    "RFFile",
    "RFFrame",
    "RFWriter",
    "frame_file_name",
    "iter_rf_source",
    "load_medium_cache",
    "medium_fingerprint",
    "read_flow_simulation_parameters",
    "read_ground_truth_pulse",
    "read_matlab_rf_frame",
    "save_medium_cache",
]

RF_FORMAT = "pyproteus-rf"
RF_FORMAT_VERSION = 1
MEDIUM_FORMAT = "pyproteus-medium"
_PARTIAL = ".partial_"


def _utc_now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")


def _scalar(v: Any, name: str) -> float:
    a = np.asarray(v)
    if a.size != 1 or a.dtype.kind not in "biuf":
        raise ValueError(f"{name} must be a real scalar, got {a.dtype} of shape {a.shape}")
    return float(a.reshape(-1)[0])


# ============================================================== ground truth


_FSP_FIELDS = ("TimeBtwPulse", "FrameRate", "NBPulses", "NMicrobubbles", "NumberOfFrames")


@dataclass
class FlowSimulationParameters:
    """``FlowSimulationParameters`` struct of a ground-truth folder (MATLAB field names).

    ``Microbubble`` is the value tree of the ``Microbubble`` field (at least
    ``Distribution.Probabilities`` and ``Distribution.Radii``); ``extra``
    keeps any other fields.
    """

    TimeBtwPulse: float
    FrameRate: float
    NBPulses: float
    NMicrobubbles: float
    NumberOfFrames: float
    Microbubble: dict[str, Any]
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_tree(cls, tree: Mapping[str, Any]) -> FlowSimulationParameters:
        if not isinstance(tree, Mapping):
            raise TypeError("FlowSimulationParameters must be a 1×1 struct")
        missing = [k for k in (*_FSP_FIELDS, "Microbubble") if k not in tree]
        if missing:
            raise KeyError(f"FlowSimulationParameters has no field(s) {missing}")
        mb = tree["Microbubble"]
        if not isinstance(mb, Mapping):
            raise TypeError("FlowSimulationParameters.Microbubble must be a 1×1 struct")
        values = {k: _scalar(tree[k], f"FlowSimulationParameters.{k}") for k in _FSP_FIELDS}
        extra = {k: v for k, v in tree.items() if k not in (*_FSP_FIELDS, "Microbubble")}
        return cls(**values, Microbubble=dict(mb), extra=extra)

    def to_tree(self) -> dict[str, Any]:
        out: dict[str, Any] = {k: np.array([[getattr(self, k)]]) for k in _FSP_FIELDS}
        out["Microbubble"] = self.Microbubble
        out.update(self.extra)
        return out

    def distribution(self) -> tuple[np.ndarray, np.ndarray]:
        """``(Probabilities, Radii)`` of ``Microbubble.Distribution`` as float arrays."""
        dist = self.Microbubble.get("Distribution")
        if not isinstance(dist, Mapping) or "Probabilities" not in dist or "Radii" not in dist:
            raise KeyError("FlowSimulationParameters.Microbubble.Distribution is incomplete")
        return (
            np.asarray(dist["Probabilities"], dtype=float),
            np.asarray(dist["Radii"], dtype=float),
        )


def read_flow_simulation_parameters(folder: str | PathLike[str]) -> FlowSimulationParameters:
    """``<folder>/FlowSimulationParameters.mat`` as :class:`FlowSimulationParameters`."""
    file = Path(folder) / "FlowSimulationParameters.mat"
    if not file.is_file():
        raise FileNotFoundError(f"{file} not found")
    return FlowSimulationParameters.from_tree(load_flow_simulation_parameters(folder))


def read_ground_truth_pulse(
    folder: str | PathLike[str],
    frame: int,
    pulse_seq_idx: int,
    num_frames: int,
    *,
    frame_tree: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    """``Frame.Pulse<pulse_seq_idx>`` of ground-truth frame ``frame`` (both 1-based).

    Returns ``{"Points": N×3, "Velocity": N×3, "Radius": N×1, ...}`` (float64,
    MATLAB sizes; other numeric fields such as ``StreamNumber`` are kept).
    ``frame_tree`` (the ``Frame`` struct of :func:`pyproteus.config.load_frame`)
    avoids reading the file again for each pulse.
    """
    if frame_tree is None:
        file = Path(folder) / frame_file_name(frame, num_frames)
        if not file.is_file():
            raise FileNotFoundError(f"ground-truth frame {file} not found")
        tree = load_frame(folder, frame, num_frames)
    else:
        tree = frame_tree
    name = f"Pulse{int(pulse_seq_idx)}"
    if not isinstance(tree, Mapping) or name not in tree:
        raise KeyError(f"{frame_file_name(frame, num_frames)}: Frame has no field {name}")
    pulse = tree[name]
    if not isinstance(pulse, Mapping):
        raise TypeError(f"Frame.{name} must be a 1×1 struct")
    out: dict[str, np.ndarray] = {}
    for key in ("Points", "Velocity", "Radius"):
        if key not in pulse:
            raise KeyError(f"Frame.{name} has no field {key}")
        out[key] = np.asarray(pulse[key], dtype=float)
    n = out["Points"].shape[0]
    if out["Points"].shape != (n, 3) or out["Velocity"].shape != (n, 3):
        raise ValueError(
            f"Frame.{name}: Points and Velocity must be N×3, got "
            f"{out['Points'].shape} and {out['Velocity'].shape}"
        )
    if out["Radius"].size != n:
        raise ValueError(f"Frame.{name}: Radius has {out['Radius'].size} values for {n} points")
    for key, value in pulse.items():
        if key not in out and isinstance(value, np.ndarray) and value.dtype.kind in "biuf":
            out[key] = value
    return out


# ============================================================== RF frames


@dataclass
class RFFrame:
    """RF data of one frame: ``rf[k]`` is MATLAB ``RF{k+1}`` (``Nelem``-by-``Nt``).

    ``bubbles[k]`` is MATLAB ``Frame{k+1}`` (the MB struct as a dict), or
    ``None`` when the source does not store it. ``source`` names where the
    frame was read from (the ``.mat`` file, or ``<file.h5>/frames/<group>``).
    """

    frame: int
    rf: list[np.ndarray]
    dt: float
    bubbles: list[dict[str, Any]] | None = None
    source: str = ""

    def rf_cell(self) -> np.ndarray:
        """``RF`` as a MATLAB 1×n cell array (object array)."""
        cell = np.empty((1, len(self.rf)), dtype=object)
        for k, a in enumerate(self.rf):
            cell[0, k] = a
        return cell


_FRAME_NUMBER = re.compile(r"Frame_?0*(\d+)\.mat\Z")


def read_matlab_rf_frame(path: str | PathLike[str]) -> RFFrame:
    """MATLAB RF output ``Frame_<m>.mat`` (``main_RF.m``: ``RF``, ``dt``, ``Frame``).

    ``RF`` and ``Frame`` are cell arrays with one element per pulse of the
    sequence; elements are taken in column-major (linear index) order.
    """
    path = Path(path)
    tree = loadmat(path, ["RF", "dt", "Frame"])
    if "RF" not in tree:
        raise KeyError(f"{path}: no variable 'RF'")
    cell = tree["RF"]
    if not (isinstance(cell, np.ndarray) and cell.dtype == object):
        raise TypeError(f"{path}: 'RF' is not a cell array")
    rf = [np.asarray(a) for a in cell.reshape(-1, order="F")]
    for k, a in enumerate(rf, start=1):
        if a.dtype.kind not in "fiu":
            raise TypeError(f"{path}: RF{{{k}}} is not a numeric array")
    if "dt" not in tree:
        raise KeyError(f"{path}: no variable 'dt'")
    dt = _scalar(tree["dt"], "dt")
    bubbles = None
    if "Frame" in tree:
        fcell = tree["Frame"]
        if isinstance(fcell, np.ndarray) and fcell.dtype == object:
            bubbles = [dict(b) for b in fcell.reshape(-1, order="F")]
        elif isinstance(fcell, Mapping):  # one pulse saved as a struct
            bubbles = [dict(fcell)]
    m = _FRAME_NUMBER.search(path.name)
    frame = int(m.group(1)) if m else -1
    return RFFrame(frame=frame, rf=rf, dt=dt, bubbles=bubbles, source=str(path))


def _frame_group_name(frame: int, num_frames: int) -> str:
    return frame_file_name(frame, max(num_frames, frame))[: -len(".mat")]


#: Fields that ``setup_run``/``input_handling`` set for one invocation (run control,
#: not physics). They are ignored when a run resumes an RF file or checks a medium cache,
#: so ``Continue``, another start frame or another GPU does not refuse a valid file.
RUN_CONTROL_FIELDS: dict[str, tuple[str, ...]] = {
    "Acquisition": ("Continue", "StartFrame", "EndFrame"),
    "SimulationParameters": ("DeviceNumber",),
    "Microbubble": ("BatchSize", "UseParfor"),
    "Transducer": ("OnGrid", "IntegrationDensity"),
    "Geometry": ("GeometriesPath",),
    "Medium": ("Save",),
}

#: Further fields that do not change the grid or the k-Wave medium (medium cache key).
_NOT_MEDIUM_FIELDS: dict[str, tuple[str, ...]] = {
    "SimulationParameters": ("Solver", "NumberOfInteractions", "HybridSimulation"),
}


def _settings_doc(settings: Settings, drop: Mapping[str, tuple[str, ...]]) -> dict[str, Any]:
    doc = tree_to_jsonable(settings.to_tree())
    for group, names in drop.items():
        if isinstance(doc.get(group), dict):
            for name in names:
                doc[group].pop(name, None)
    return doc


def _canonical_settings(settings: Settings) -> str:
    return settings_to_json(settings, indent=None)


def _comparable_settings(settings: Settings) -> str:
    """Settings without :data:`RUN_CONTROL_FIELDS`, as canonical JSON."""
    return json.dumps(_settings_doc(settings, RUN_CONTROL_FIELDS), sort_keys=True, allow_nan=True)


def _write_str(group: h5py.Group, name: str, text: str) -> None:
    group.create_dataset(name, data=text, dtype=h5py.string_dtype("utf-8"))


def _read_str(group: h5py.Group, name: str) -> str:
    value = group[name][()]
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _to_numpy(a: Any) -> np.ndarray:
    """NumPy array from NumPy or CuPy input (CuPy arrays are copied to the host)."""
    if type(a).__module__.split(".")[0] == "cupy":
        return np.asarray(a.get())
    return np.asarray(a)


def _bubble_arrays(mb: Mapping[str, Any]) -> dict[str, np.ndarray]:
    out, dropped = {}, []
    for key, value in mb.items():
        if value is None:
            continue
        try:
            arr = _to_numpy(value)
        except (TypeError, ValueError):
            dropped.append(str(key))
            continue
        if arr.dtype.kind in "biuf":
            out[str(key)] = arr
        else:
            dropped.append(str(key))
    if dropped:
        warnings.warn(
            f"bubble fields {dropped} are not numeric arrays and are not stored",
            RuntimeWarning,
            stacklevel=3,
        )
    return out


def _open_locked(path: Path, mode: str, timeout: float) -> h5py.File:
    """``h5py.File(path, mode)``, retrying while another process holds the file lock.

    Raises the HDF5 lock error (:class:`OSError`) after ``timeout`` seconds.
    """
    deadline = time.monotonic() + timeout
    delay = 0.1
    while True:
        try:
            return h5py.File(path, mode)
        except OSError as exc:
            # POSIX: BlockingIOError (EAGAIN); Windows: OSError errno 0, Win32 error 33.
            locked = isinstance(exc, BlockingIOError) or "unable to lock file" in str(exc)
            if not locked or time.monotonic() >= deadline:
                raise
            time.sleep(delay)
            delay = min(2 * delay, 5.0)


class RFWriter:
    """Write RF frames to one HDF5 file per run (D-25).

    ``mode``:

    - ``"new"``: create the file; fail if it exists;
    - ``"overwrite"``: replace an existing file;
    - ``"resume"``: open an existing file written for the same ``settings``
      (ignoring :data:`RUN_CONTROL_FIELDS`), ``sequence`` and ``num_frames``
      (else ``ValueError``), dropping any frame left incomplete by an
      interrupted run; creates the file if absent. Frames already in the file
      are kept (MATLAB keeps the ``Frame_<m>.mat`` files it does not rewrite).

    Each :meth:`write_frame` opens the file, builds the frame group under a
    temporary name, renames it, and closes the file (HDF5 flush). While
    another process holds the file open (a reader, HDFView), HDF5's file lock
    refuses the write; :meth:`write_frame` retries for ``lock_timeout``
    seconds before raising, so a frame is not lost to a short read. Several
    processes writing one file is not supported. Deleted or replaced frames
    do not shrink the file (``h5repack`` does).
    """

    def __init__(
        self,
        path: str | PathLike[str],
        *,
        settings: Settings,
        sequence: Sequence[str],
        num_frames: int,
        flow_simulation_parameters: FlowSimulationParameters | None = None,
        mode: str = "new",
        lock_timeout: float = 600.0,
    ) -> None:
        self.lock_timeout = float(lock_timeout)
        if mode not in ("new", "overwrite", "resume"):
            raise ValueError(f"mode must be 'new', 'overwrite' or 'resume', got {mode!r}")
        if int(num_frames) != num_frames or num_frames < 1:
            raise ValueError("num_frames must be a positive integer")
        self.path = Path(path)
        self.sequence = tuple(str(s) for s in sequence)
        if not self.sequence:
            raise ValueError("sequence must name at least one pulse")
        self.num_frames = int(num_frames)
        settings_json = _canonical_settings(settings)
        self._comparable = _comparable_settings(settings)
        acq = settings.Acquisition
        scheme = getattr(acq, "PulsingScheme", None) if acq is not None else None

        if mode == "resume" and self.path.exists():
            with _open_locked(self.path, "a", self.lock_timeout) as f:
                self._check_compatible(f)
                frames = f["frames"]
                for name in [n for n in frames if n.startswith(_PARTIAL)]:
                    del frames[name]
            return
        if mode == "new" and self.path.exists():
            raise FileExistsError(f"{self.path} exists (use mode='overwrite' or 'resume')")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(self.path, "w") as f:
            from pyproteus import __version__

            f.attrs["format"] = RF_FORMAT
            f.attrs["format_version"] = RF_FORMAT_VERSION
            f.attrs["pyproteus_version"] = __version__
            f.attrs["created_utc"] = _utc_now()
            f.attrs["num_frames"] = self.num_frames
            f.attrs["pulsing_scheme"] = scheme or ""
            f.attrs["sequence"] = json.dumps(list(self.sequence))
            _write_str(f, "settings", settings_json)
            if flow_simulation_parameters is not None:
                fsp = json.dumps(tree_to_jsonable(flow_simulation_parameters.to_tree()))
                _write_str(f, "flow_simulation_parameters", fsp)
            f.create_group("frames")

    def _check_compatible(self, f: h5py.File) -> None:
        if f.attrs.get("format") != RF_FORMAT:
            raise ValueError(f"{self.path} is not a {RF_FORMAT} file")
        if int(f.attrs["format_version"]) != RF_FORMAT_VERSION:
            raise ValueError(
                f"cannot resume {self.path}: {RF_FORMAT} version {int(f.attrs['format_version'])}"
            )
        problems = []
        stored = settings_from_json(_read_str(f, "settings"))
        if _comparable_settings(stored) != self._comparable:
            problems.append("settings")
        if tuple(json.loads(f.attrs["sequence"])) != self.sequence:
            problems.append("pulse sequence")
        if int(f.attrs["num_frames"]) != self.num_frames:
            problems.append("number of frames")
        if problems:
            raise ValueError(
                f"cannot resume {self.path}: the {', '.join(problems)} differ from this run"
            )

    def completed_frames(self) -> list[int]:
        """Frame numbers already written (complete groups only), ascending."""
        with _open_locked(self.path, "r", self.lock_timeout) as f:
            return _complete_frames(f)

    def remaining_frames(self, start_frame: int = 1, end_frame: int | None = None) -> list[int]:
        """Frames in ``start_frame..end_frame`` not yet written."""
        end = self.num_frames if end_frame is None else int(end_frame)
        done = set(self.completed_frames())
        return [m for m in range(int(start_frame), end + 1) if m not in done]

    def write_frame(
        self,
        frame: int,
        rf: Sequence[np.ndarray],
        dt: float,
        bubbles: Sequence[Mapping[str, Any]] | None = None,
        *,
        replace: bool = False,
    ) -> None:
        """Write frame ``frame`` (1-based): ``rf[k]`` and ``bubbles[k]`` per pulse.

        All pulses must have the same ``Nelem``-by-``Nt`` size and class
        (NumPy or CuPy arrays). An existing frame raises unless ``replace=True``;
        a replaced frame stays readable until its new version is complete.
        """
        frame = int(frame)
        if not 1 <= frame <= self.num_frames:
            raise ValueError(f"frame {frame} outside 1..{self.num_frames}")
        if len(rf) != len(self.sequence):
            raise ValueError(f"{len(rf)} RF arrays for a {len(self.sequence)}-pulse sequence")
        arrays = [_to_numpy(a) for a in rf]
        if any(a.ndim != 2 or a.shape != arrays[0].shape for a in arrays):
            raise ValueError("every RF array must be Nelem-by-Nt with the same size")
        if any(a.dtype != arrays[0].dtype for a in arrays):
            raise ValueError("every RF array must have the same class")
        if bubbles is not None and len(bubbles) != len(self.sequence):
            raise ValueError(
                f"{len(bubbles)} bubble sets for a {len(self.sequence)}-pulse sequence"
            )
        name = _frame_group_name(frame, self.num_frames)
        stacked = np.stack(arrays)
        mbs = None if bubbles is None else [_bubble_arrays(mb) for mb in bubbles]
        with _open_locked(self.path, "a", self.lock_timeout) as f:
            frames = f["frames"]
            if name in frames and not replace:
                raise FileExistsError(f"frame {frame} is already in {self.path}")
            tmp = _PARTIAL + name
            if tmp in frames:
                del frames[tmp]
            g = frames.create_group(tmp)
            g.attrs["frame"] = frame
            g.attrs["dt"] = float(dt)
            g.attrs["written_utc"] = _utc_now()
            g.create_dataset("rf", data=stacked)
            if mbs is not None:
                bg = g.create_group("bubbles")
                for k, mb in enumerate(mbs, start=1):
                    pg = bg.create_group(f"pulse_{k}")
                    for key, arr in mb.items():
                        pg.create_dataset(key, data=arr)
            if name in frames:
                del frames[name]
            frames.move(tmp, name)
            f.flush()


def _complete_frames(f: h5py.File) -> list[int]:
    frames = f["frames"]
    return sorted(int(frames[n].attrs["frame"]) for n in frames if not n.startswith(_PARTIAL))


class RFFile:
    """Read an RF file written by :class:`RFWriter`."""

    def __init__(self, path: str | PathLike[str]) -> None:
        self.path = Path(path)
        with h5py.File(self.path, "r") as f:
            if f.attrs.get("format") != RF_FORMAT:
                raise ValueError(f"{self.path} is not a {RF_FORMAT} file")
            version = int(f.attrs["format_version"])
            if version > RF_FORMAT_VERSION:
                raise ValueError(f"{self.path}: {RF_FORMAT} version {version} is newer than this")
            self.num_frames = int(f.attrs["num_frames"])
            self.pulsing_scheme = str(f.attrs["pulsing_scheme"])
            self.sequence = tuple(json.loads(f.attrs["sequence"]))
            self.attrs = {k: f.attrs[k] for k in f.attrs}
            self._settings_json = _read_str(f, "settings")
            self._fsp_json = (
                _read_str(f, "flow_simulation_parameters")
                if "flow_simulation_parameters" in f
                else None
            )
            self.frames = _complete_frames(f)
            self._names = {
                int(f["frames"][n].attrs["frame"]): n
                for n in f["frames"]
                if not n.startswith(_PARTIAL)
            }

    @property
    def settings(self) -> Settings:
        return settings_from_json(self._settings_json)

    @property
    def flow_simulation_parameters(self) -> FlowSimulationParameters | None:
        if self._fsp_json is None:
            return None
        return FlowSimulationParameters.from_tree(tree_from_jsonable(json.loads(self._fsp_json)))

    def read_frame(self, frame: int, *, bubbles: bool = True) -> RFFrame:
        if frame not in self._names:
            raise KeyError(f"frame {frame} is not in {self.path}")
        name = self._names[frame]
        with h5py.File(self.path, "r") as f:
            g = f["frames"][name]
            data = g["rf"][()]
            mbs = None
            if bubbles and "bubbles" in g:
                bg = g["bubbles"]
                mbs = [
                    {k: bg[f"pulse_{i}"][k][()] for k in bg[f"pulse_{i}"]}
                    for i in range(1, len(bg) + 1)
                ]
            return RFFrame(
                frame=frame,
                rf=[data[k] for k in range(data.shape[0])],
                dt=float(g.attrs["dt"]),
                bubbles=mbs,
                source=f"{self.path}/frames/{name}",
            )

    def __iter__(self) -> Iterator[RFFrame]:
        for m in self.frames:
            yield self.read_frame(m)

    def __len__(self) -> int:
        return len(self.frames)


def iter_rf_source(source: str | PathLike[str], *, bubbles: bool = False) -> Iterator[RFFrame]:
    """RF frames, in frame order, from an :class:`RFWriter` HDF5 file or a MATLAB results folder.

    A folder is read like ``load_RF_data.m`` (``Frame*.mat`` sorted by frame
    number, see :func:`pyproteus.reconstruction.frame_files`).
    """
    path = Path(source)
    if path.is_file() and h5py.is_hdf5(path):
        rf_file = RFFile(path)
        for m in rf_file.frames:
            yield rf_file.read_frame(m, bubbles=bubbles)
        return
    if not path.is_dir():
        raise FileNotFoundError(f"{path} is neither an RF HDF5 file nor a results folder")
    from pyproteus.reconstruction.rf_data import frame_files

    for file in frame_files(path):
        fr = read_matlab_rf_frame(file)
        if not bubbles:
            fr.bubbles = None
        yield fr


# ============================================================== medium cache


class MediumCacheMismatch(ValueError):
    """The cached k-Wave medium was built from other settings."""


#: Settings groups that determine the grid and the k-Wave medium.
_MEDIUM_GROUPS = ("SimulationParameters", "Geometry", "Medium", "Transducer")


def medium_fingerprint(settings: Settings, **extra: Any) -> str:
    """SHA-256 of the settings that define the grid and medium (+ ``extra``, e.g. a seed).

    Groups ``SimulationParameters``, ``Geometry``, ``Medium``, ``Transducer``
    (lossless JSON), without the run-control fields (:data:`RUN_CONTROL_FIELDS`)
    and ``Solver``, ``NumberOfInteractions``, ``HybridSimulation``, which do not
    change the medium; ``extra`` values must be JSON-serialisable.
    """
    drop = {
        g: RUN_CONTROL_FIELDS.get(g, ()) + _NOT_MEDIUM_FIELDS.get(g, ()) for g in _MEDIUM_GROUPS
    }
    full = _settings_doc(settings, drop)
    doc = {g: full[g] for g in _MEDIUM_GROUPS if g in full}
    doc["extra"] = {k: extra[k] for k in sorted(extra)}
    return hashlib.sha256(json.dumps(doc, sort_keys=True, allow_nan=True).encode()).hexdigest()


@dataclass
class MediumCache:
    """Content of a medium cache: k-Wave ``medium`` maps, ``vessel_grid``, ``Grid``.

    ``grid`` is a :class:`pyproteus.acoustic.Grid` for pyproteus caches and
    the MATLAB ``Grid`` value tree (dict) for MATLAB ``medium.mat`` files.
    """

    medium: dict[str, Any]
    vessel_grid: np.ndarray | None
    grid: Any
    fingerprint: str | None = None


def save_medium_cache(
    path: str | PathLike[str],
    medium: Mapping[str, Any],
    vessel_grid: np.ndarray | None,
    grid: Any,
    *,
    fingerprint: str,
) -> None:
    """Write the k-Wave medium (``main_RF.m``: ``save medium.mat medium vessel_grid Grid``).

    Arrays go to ``/medium/<name>`` datasets, scalars and strings to
    attributes of ``/medium``; ``vessel_grid`` (bool, gzip); ``grid`` (a
    :class:`pyproteus.acoustic.Grid`) to ``/grid``. Written to a temporary file
    and renamed, so a stopped run never leaves a truncated cache.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    from pyproteus import __version__

    try:
        with h5py.File(tmp, "w") as f:
            f.attrs["format"] = MEDIUM_FORMAT
            f.attrs["format_version"] = 1
            f.attrs["pyproteus_version"] = __version__
            f.attrs["created_utc"] = _utc_now()
            f.attrs["fingerprint"] = fingerprint
            mg = f.create_group("medium")
            for key, value in medium.items():
                if value is None:
                    continue
                if isinstance(value, str):
                    mg.attrs[key] = value
                    continue
                if isinstance(value, Mapping):
                    raise TypeError(f"medium field {key!r}: nested structs are not supported")
                arr = np.asarray(value)
                if arr.ndim == 0:
                    mg.attrs[key] = arr.item()
                else:
                    mg.create_dataset(key, data=arr)
            if vessel_grid is not None:
                f.create_dataset("vessel_grid", data=np.asarray(vessel_grid), compression="gzip")
            _write_grid(f.create_group("grid"), grid)
            f.flush()
        with open(tmp, "rb+") as fh:
            os.fsync(fh.fileno())
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _write_grid(g: h5py.Group, grid: Any) -> None:
    for name in ("x", "y", "z"):
        g.create_dataset(name, data=np.asarray(getattr(grid, name), dtype=float))
    for name in ("Nx", "Ny", "Nz"):
        g.attrs[name] = int(getattr(grid, name))
    for name in ("dx", "dy", "dz", "dt"):
        g.attrs[name] = float(getattr(grid, name))
    g.attrs["full_size"] = np.asarray(grid.full_size, dtype=np.int64)
    sog = grid.sensor_on_grid
    g.attrs["sensor_on_grid"] = sog
    g.attrs["sensor_on_grid_is_bool"] = isinstance(sog, bool | np.bool_)
    pml = grid.PML
    for name in ("X_SIZE", "Y_SIZE", "Z_SIZE"):
        g.attrs[f"PML_{name}"] = int(getattr(pml, name))
    g.attrs["PML_Alpha"] = float(pml.Alpha)


def _read_grid(g: h5py.Group) -> Any:
    from pyproteus.acoustic.grid import PML, Grid

    sog = g.attrs["sensor_on_grid"]
    sog = bool(sog) if bool(g.attrs["sensor_on_grid_is_bool"]) else float(sog)
    return Grid(
        x=g["x"][()],
        y=g["y"][()],
        z=g["z"][()],
        Nx=int(g.attrs["Nx"]),
        Ny=int(g.attrs["Ny"]),
        Nz=int(g.attrs["Nz"]),
        dx=float(g.attrs["dx"]),
        dy=float(g.attrs["dy"]),
        dz=float(g.attrs["dz"]),
        dt=float(g.attrs["dt"]),
        full_size=tuple(int(n) for n in g.attrs["full_size"]),
        PML=PML(
            X_SIZE=int(g.attrs["PML_X_SIZE"]),
            Y_SIZE=int(g.attrs["PML_Y_SIZE"]),
            Z_SIZE=int(g.attrs["PML_Z_SIZE"]),
            Alpha=float(g.attrs["PML_Alpha"]),
        ),
        sensor_on_grid=sog,
    )


def _attr_value(v: Any) -> Any:
    if isinstance(v, bytes):
        return v.decode("utf-8")
    if isinstance(v, np.generic):
        return v.item()
    return v


def load_medium_cache(path: str | PathLike[str], *, fingerprint: str | None = None) -> MediumCache:
    """Read a medium cache written by :func:`save_medium_cache`, or MATLAB's ``medium.mat``.

    With ``fingerprint`` given, a pyproteus cache with another fingerprint
    raises :class:`MediumCacheMismatch`. MATLAB files carry no fingerprint and
    are returned as read (scalars as floats, arrays with MATLAB's size).
    """
    path = Path(path)
    if path.suffix == ".mat":
        tree = loadmat(path, ["medium", "vessel_grid", "Grid"])
        if "medium" not in tree:
            raise KeyError(f"{path}: no variable 'medium'")
        medium = {}
        for key, value in tree["medium"].items():
            if isinstance(value, np.ndarray) and value.size == 1 and value.dtype.kind in "biuf":
                medium[key] = value.reshape(-1)[0].item()
            else:
                medium[key] = value
        return MediumCache(medium, tree.get("vessel_grid"), tree.get("Grid"), None)
    with h5py.File(path, "r") as f:
        if f.attrs.get("format") != MEDIUM_FORMAT:
            raise ValueError(f"{path} is not a {MEDIUM_FORMAT} file")
        stored = str(_attr_value(f.attrs["fingerprint"]))
        if fingerprint is not None and stored != fingerprint:
            raise MediumCacheMismatch(
                f"{path} was built from other grid/medium settings; delete it or start "
                "without Continue"
            )
        mg = f["medium"]
        medium = {k: _attr_value(v) for k, v in mg.attrs.items()}
        medium.update({k: mg[k][()] for k in mg})
        vessel = f["vessel_grid"][()] if "vessel_grid" in f else None
        grid = _read_grid(f["grid"])
    return MediumCache(medium, vessel, grid, stored)
