"""Read PROTEUS RF output (``load_RF_data.m``).

``main_RF.m`` saves one ``Frame_<n>.mat`` per frame (v5/v7 by default; v7.3
is read as well) with a cell array ``RF``: one ``Nelem``-by-``Nt`` matrix per
pulse of the pulsing scheme, in the class of ``run_param.DATA_CAST_RF``
(``single`` by default).

pyproteus writes one HDF5 file per run instead (D-25,
:class:`pyproteus.io.frames.RFWriter`); every function here takes either that
file or a MATLAB results folder.

Design (D-15, ``docs/PARITY.md``): :func:`iter_rf_frames` reads the frames one
at a time (each file once, only the ``RF`` variable), so a beamformer can
stream them; :func:`load_rf_data` stacks them like MATLAB. MATLAB reads the
first file twice and always holds the whole ``Nelem``-by-``Nt``-by-``Nframes``
array.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from os import PathLike
from pathlib import Path

import h5py
import numpy as np

from pyproteus.config.matfile import loadmat
from pyproteus.numerics import argsort

__all__ = ["PULSING_SCHEMES", "frame_files", "iter_rf_frames", "load_rf_data"]

#: Pulsing schemes and how the pulses are combined (``RF{k}`` is pulse ``k``).
PULSING_SCHEMES = (
    "Amplitude modulation",
    "Pulse inversion",
    "Amplitude modulation with pulse inversion",
    "Standard",
)

_NUMBER = re.compile(r"\s*[+-]?(\d+\.?\d*([eEdD][+-]?\d+)?|\.\d+([eEdD][+-]?\d+)?)\s*")


def _str2double(text: str) -> float:
    """``str2double`` for the plain decimal numbers of frame file names; NaN otherwise."""
    if not _NUMBER.fullmatch(text):
        return float("nan")
    return float(text.strip().replace("d", "e").replace("D", "e"))


def frame_files(results_folder: str | PathLike[str]) -> list[Path]:
    """``Frame*.mat`` files of a results folder, sorted by frame number.

    The frame number is ``str2double(name(7:end-4))`` (``Frame_0012.mat`` →
    12). Files are listed in name order, then stable-sorted by frame number;
    names without a number (NaN) go last, as in MATLAB ``sort``.
    """
    folder = Path(results_folder)
    names = sorted(p.name for p in folder.iterdir() if re.fullmatch(r"Frame.*\.mat", p.name))
    numbers = np.array([_str2double(n[6:-4]) for n in names])
    order = argsort(numbers) if names else np.zeros(0, dtype=np.intp)
    return [folder / names[i] for i in order]


def _pulse(cell: np.ndarray, k: int) -> np.ndarray:
    """``RF{k}`` (1-based linear index, column-major) as a numeric array."""
    flat = cell.reshape(-1, order="F")
    if k > flat.size:
        raise IndexError(f"RF{{{k}}}: the RF cell array has {flat.size} pulses")
    value = np.asarray(flat[k - 1])
    if value.dtype.kind not in "fiu":
        raise TypeError(f"RF{{{k}}} is not a numeric array (dtype {value.dtype})")
    return value


def _combine(terms: list[tuple[int, np.ndarray]]) -> np.ndarray:
    """Signed sum of pulses with MATLAB class rules (any ``single`` → ``single``)."""
    dtype = np.float32 if any(a.dtype == np.float32 for _, a in terms) else np.float64
    signed = [(a if sign > 0 else -a).astype(dtype, copy=False) for sign, a in terms]
    out = signed[0].copy()
    for a in signed[1:]:
        out = out + a
    return out


def _apply_scheme(cell: np.ndarray, pulsing_scheme: str) -> np.ndarray:
    if pulsing_scheme == "Amplitude modulation":
        return _combine([(1, _pulse(cell, 3)), (-1, _pulse(cell, 1)), (-1, _pulse(cell, 2))])
    if pulsing_scheme == "Pulse inversion":
        return _combine([(1, _pulse(cell, 1)), (1, _pulse(cell, 2))])
    if pulsing_scheme == "Amplitude modulation with pulse inversion":
        return _combine([(1, _pulse(cell, 3)), (1, _pulse(cell, 1)), (1, _pulse(cell, 2))])
    if pulsing_scheme == "Standard":
        return _pulse(cell, 1)
    # MATLAB leaves RF a cell and fails when storing it in the numeric array.
    raise ValueError(
        f"Unknown pulsing scheme {pulsing_scheme!r}; expected one of {PULSING_SCHEMES}"
    )


def _read_rf_cell(path: Path) -> np.ndarray:
    tree = loadmat(path, ["RF"])
    if "RF" not in tree:
        raise KeyError(f"{path}: no variable 'RF'")
    cell = tree["RF"]
    if not (isinstance(cell, np.ndarray) and cell.dtype == object):
        raise TypeError(f"{path}: 'RF' is not a cell array")
    return cell


def _is_rf_h5(source: str | PathLike[str]) -> bool:
    path = Path(source)
    return path.is_file() and h5py.is_hdf5(path)


def _iter_cells(source: str | PathLike[str]) -> Iterator[tuple[Path, np.ndarray]]:
    """``(name, RF cell)`` per frame from an RF HDF5 file or a MATLAB results folder."""
    if _is_rf_h5(source):
        from pyproteus.io.frames import RFFile

        rf_file = RFFile(source)
        for m in rf_file.frames:
            fr = rf_file.read_frame(m, bubbles=False)
            yield Path(fr.source), fr.rf_cell()
        return
    for path in frame_files(source):
        yield path, _read_rf_cell(path)


def _scheme(source: str | PathLike[str], pulsing_scheme: str | None) -> str:
    if pulsing_scheme is not None:
        return pulsing_scheme
    if _is_rf_h5(source):
        from pyproteus.io.frames import RFFile

        scheme = RFFile(source).pulsing_scheme
        if scheme:
            return scheme
        raise ValueError(f"{source} stores no pulsing scheme: pass pulsing_scheme")
    raise ValueError("pulsing_scheme is required for a MATLAB results folder")


def iter_rf_frames(
    results_folder: str | PathLike[str], pulsing_scheme: str | None = None
) -> Iterator[tuple[Path, np.ndarray]]:
    """Yield ``(source, RF)`` per frame, in frame order, with the pulsing scheme applied.

    ``results_folder`` is a MATLAB results folder (``Frame*.mat``) or a
    pyproteus RF HDF5 file (D-25); ``source`` is the ``.mat`` file or
    ``<file.h5>/frames/<group>``. ``pulsing_scheme`` defaults to the one
    stored in the HDF5 file (required for a folder). ``RF`` is
    ``Nelem``-by-``Nt`` in the class of the stored data (pulses of mixed
    classes combine as in MATLAB: ``single`` wins).
    """
    scheme = _scheme(results_folder, pulsing_scheme)
    for path, cell in _iter_cells(results_folder):
        yield path, _apply_scheme(cell, scheme)


def load_rf_data(
    results_folder: str | PathLike[str], pulsing_scheme: str | None = None
) -> np.ndarray:
    """``RF = load_RF_data(resultsFolder, pulsingScheme)``.

    ``results_folder`` may also be a pyproteus RF HDF5 file (see
    :func:`iter_rf_frames`).

    Returns the ``Nelem``-by-``Nt``-by-``Nframes`` RF array. Its class is that
    of ``RF{1}`` of the first frame (MATLAB ``zeros(..., class(RF))``); frames
    of another class are converted on assignment. All frames must have the
    size of the first one (MATLAB errors otherwise; so does this function,
    without NumPy broadcasting).
    """
    scheme = _scheme(results_folder, pulsing_scheme)
    if _is_rf_h5(results_folder):
        from pyproteus.io.frames import RFFile

        n_frames = len(RFFile(results_folder))
    else:
        n_frames = len(frame_files(results_folder))
    if not n_frames:
        raise FileNotFoundError(f"no RF frames in {results_folder}")
    out = None
    for iframe, (path, cell) in enumerate(_iter_cells(results_folder)):
        if out is None:
            first = _pulse(cell, 1)
            out = np.zeros((*first.shape[:2], n_frames), dtype=first.dtype)
        rf = _apply_scheme(cell, scheme)
        if rf.shape != out.shape[:2]:
            raise ValueError(
                f"{path.name}: RF is {rf.shape[0]}-by-{rf.shape[1]}, "
                f"expected {out.shape[0]}-by-{out.shape[1]}"
            )
        out[:, :, iframe] = rf
    return out
