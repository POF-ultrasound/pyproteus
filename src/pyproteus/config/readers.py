"""Readers for the MATLAB files PROTEUS writes: settings and ground-truth frames.

* Settings ``.mat`` (``write_to_file.m``; loaded by ``load_from_file.m`` and
  ``main_RF.m``): variables ``Microbubble``, ``SimulationParameters``,
  ``Geometry``, ``Transducer``, ``Acquisition``, ``Medium``, ``Transmit``.
* Ground truth (``generate_streamlines.m``; read by ``load_microbubbles.m``
  and ``check_ground_truth_data.m``): ``FlowSimulationParameters.mat`` and
  one ``Frame_<m>.mat`` per frame holding ``Frame.Pulse<n>`` with fields
  ``Points`` (N×3), ``Velocity`` (N×3), ``Radius`` (N×1), ``StreamNumber`` (N×1).

Transducer model files are read by :func:`pyproteus.config.properties.load_transducer_model`.
Values come back as MATLAB value trees (see :mod:`pyproteus.config.matfile`).
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any

from .matfile import loadmat
from .models import Settings

__all__ = [
    "frame_file_name",
    "load_flow_simulation_parameters",
    "load_frame",
    "load_settings_mat",
]


def load_settings_mat(path: str | PathLike[str]) -> Settings:
    """Read a PROTEUS settings ``.mat`` file (v5/v7 or v7.3) into :class:`Settings`.

    Variables other than the seven settings groups are ignored; missing groups
    stay ``None``.
    """
    return Settings.from_tree(loadmat(path, list(Settings.GROUPS)))


def frame_file_name(frame: int, num_frames: int) -> str:
    """``['Frame_' num2str(frame, ['%0' num2str(length(num2str(num_frames))) 'i']) '.mat']``.

    Frames are zero-padded to the number of digits of ``num_frames``
    (``Frame_03.mat`` for frame 3 of 10).
    """
    if int(frame) != frame or int(num_frames) != num_frames:
        raise ValueError("frame and num_frames must be integers")
    width = len(str(int(num_frames)))
    return f"Frame_{int(frame):0{width}d}.mat"


def load_frame(folder: str | PathLike[str], frame: int, num_frames: int) -> dict[str, Any]:
    """The ``Frame`` struct of ground-truth frame ``frame`` (1-based), as a dict.

    ``result['Pulse1']['Points']`` is the N×3 array of bubble positions [m].
    """
    out = loadmat(Path(folder) / frame_file_name(frame, num_frames), ["Frame"])
    if "Frame" not in out:
        raise KeyError(f"no variable Frame in {frame_file_name(frame, num_frames)}")
    return out["Frame"]


def load_flow_simulation_parameters(folder: str | PathLike[str]) -> dict[str, Any]:
    """``FlowSimulationParameters`` from ``<folder>/FlowSimulationParameters.mat``."""
    out = loadmat(Path(folder) / "FlowSimulationParameters.mat", ["FlowSimulationParameters"])
    if "FlowSimulationParameters" not in out:
        raise KeyError("no variable FlowSimulationParameters")
    return out["FlowSimulationParameters"]
