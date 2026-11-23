"""Delay-and-sum reconstruction of PROTEUS RF data (``DAS_reconstruction.m``).

Shared pieces of ``DAS_reconstruction.m``, ``DAS_reconstruction_general.m``
and ``initialize_das_parameters.m``: pixel axes (MATLAB colon), time-axis
corrections (time to peak, lens, k-Wave staggered grid), time gain
compensation, and the beamforming itself (Hilbert transform along time,
DAS matrix, log compression).

Design (D-15, ``docs/PARITY.md``):

* frames are streamed from disk and beamformed in blocks
  (:func:`beamform_frames`); MATLAB loads all frames, then forms the
  complex ``Nt*Nelem``-by-``Nframes`` matrix at once;
* everything is computed in double precision. MATLAB keeps ``single`` RF data
  (the PROTEUS default) in ``single`` through the TGC and ``hilbert`` and
  converts to double afterwards; the difference is single-precision rounding
  of the input chain (see PARITY.md for the tolerance used).
"""

from __future__ import annotations

from collections.abc import Iterable
from os import PathLike
from typing import Any

import numpy as np
import scipy.sparse as sp

from pyproteus.config.models import Settings
from pyproteus.config.readers import load_settings_mat
from pyproteus.numerics import colon, conv, hilbert

from .das import _field, _required, compute_das_matrix
from .rf_data import iter_rf_frames

__all__ = [
    "beamform_frames",
    "das_reconstruction",
    "das_time_axis",
    "log_compress",
    "time_gain_compensation",
    "time_to_peak",
]


def time_to_peak(transmit: Any, transducer: Any) -> float:
    """Time to peak of the two-way pulse, as in ``DAS_reconstruction.m``.

    ``I/Transmit.SamplingRate`` with ``I`` the 1-based index of the maximum of
    ``|hilbert(conv(PressureSignal, ReceiveImpulseResponse))|`` (first one on
    ties). MATLAB uses the 1-based index, i.e. one sample later than the peak
    time ``(I-1)/Fs`` (quirk, ``docs/QUIRKS.md``).
    """
    pulse = _field(transmit, "PressureSignal")
    ir = _field(transducer, "ReceiveImpulseResponse")
    V_ref = conv(pulse, ir) / float(_field(transducer, "SamplingRate"))
    env = np.abs(hilbert(V_ref))
    # MATLAB max ignores NaN and returns the first maximum (index 1 if all NaN).
    I = 1 if np.isnan(env).all() else int(np.nanargmax(env)) + 1
    return I / float(_field(transmit, "SamplingRate"))


def das_time_axis(
    Nt: int, simulation_parameters: Any, transducer: Any, transmit: Any, c: float
) -> np.ndarray:
    """RF time axis relative to the transmit-pulse peak (``DAS_reconstruction.m``).

    ``t = (0:Nt-1)/Fs - timeToPeak - lensCorrection + kWaveCorrection`` with
    the two-way lens correction ``2*(sqrt((H/2)^2 + F^2) - F)/c`` (0 for an
    infinite elevation focus) and the k-Wave staggered-grid correction
    ``dx/(2c) + dt`` (hybrid) or ``dx/(2c) + 3/2*dt`` (full simulation).
    """
    sim = simulation_parameters
    Fs = float(_field(sim, "SamplingRate"))
    t = np.arange(Nt) / Fs
    ttp = time_to_peak(transmit, transducer)

    H = float(_field(transducer, "ElementHeight"))
    F = float(_field(transducer, "ElevationFocus"))
    lens = np.sqrt((H / 2) ** 2 + F**2) / c - F / c if np.isfinite(abs(F)) else 0.0
    lens = 2 * lens

    dt = 1 / float(_field(sim, "SamplingRate"))
    dx = float(_field(sim, "GridSize"))
    kwave = (
        dx / (2 * c) + dt
        if _required(sim, "HybridSimulation", "SimulationParameters")
        else dx / (2 * c) + dt * 3 / 2
    )
    return t - ttp - lens + kwave


def time_gain_compensation(t: np.ndarray, medium: Any, f0: float, c: float) -> np.ndarray:
    """TGC factor applied to the RF data (``DAS_reconstruction.m``), with Q-101 fixed (D-18).

    MATLAB: ``TGC = sqrt(t)/max(sqrt(t)).*10.^(att.*t.*c.*1e2./20./2)`` with
    ``att = AttenuationA*(f0*1e-6)^AttenuationB``, then ``TGC(t<0) = 0``.
    pyproteus normalises by ``sqrt(max(t))``, the largest real root. This is
    MATLAB's result whenever MATLAB's ``max`` (largest magnitude of the complex
    ``sqrt(t)``) picks that root, i.e. when ``max(t) > |min(t)|``. Otherwise
    MATLAB normalises by an imaginary number and its image is all NaN; pyproteus
    still gives the image (divergence D-18, ``docs/PARITY.md``). If no time is
    positive the factor is 0 (as MATLAB's real part).
    """
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    a, b = float(_field(medium, "AttenuationA")), float(_field(medium, "AttenuationB"))
    att = a * (float(f0) * 1e-6) ** b
    tmax = np.nanmax(t) if np.any(~np.isnan(t)) else 0.0  # MATLAB max ignores NaN
    if not tmax > 0:
        return np.zeros_like(t)
    gain = 10.0 ** (att * t * c * 1e2 / 20 / 2)
    tgc = np.sqrt(np.maximum(t, 0)) / np.sqrt(tmax) * gain
    tgc[t < 0] = 0
    return tgc


def log_compress(img: np.ndarray) -> np.ndarray:
    """``20*log10(|IMG| ./ max(|IMG|,[],[1 2]))`` per frame (last axis)."""
    env = np.abs(img)
    with np.errstate(divide="ignore", invalid="ignore"):
        peak = np.max(env, axis=(0, 1), keepdims=True)
        return 20 * np.log10(env / peak)


def beamform_frames(
    frames: Iterable[np.ndarray],
    tgc: np.ndarray,
    M: sp.sparray | sp.spmatrix,
    *,
    frames_per_block: int = 8,
) -> np.ndarray:
    """Complex DAS image ``M @ hilbert(RF.*TGC)`` for each frame.

    ``frames`` yields ``Nelem``-by-``Nt`` RF arrays; ``tgc`` has length ``Nt``.
    Returns an ``Npixels``-by-``Nframes`` complex array (pixel order of ``M``).
    The Hilbert transform runs along time for each element (MATLAB: columns
    of the permuted ``Nt``-by-``Nelem`` array), in double precision.
    """
    tgc = np.asarray(tgc, dtype=np.float64).reshape(1, -1)
    out: list[np.ndarray] = []
    block: list[np.ndarray] = []

    def flush() -> None:
        if not block:
            return
        rf = np.stack(block, axis=-1)  # Nelem x Nt x nb
        if np.iscomplexobj(rf):
            rf = rf.real  # MATLAB hilbert uses the real part
        rf = rf.astype(np.float64)
        rf = rf * tgc[:, :, None]
        a = hilbert(rf, axis=1)
        # C-order flattening of Nelem x Nt = column-major Nt x Nelem stacking.
        out.append(np.asarray(M @ a.reshape(-1, a.shape[-1])))
        block.clear()

    for rf in frames:
        rf = np.asarray(rf)
        if rf.ndim != 2 or rf.shape[1] != tgc.shape[1]:
            raise ValueError(f"RF frame of shape {rf.shape}, expected Nelem-by-{tgc.shape[1]}")
        block.append(rf)
        if len(block) >= frames_per_block:
            flush()
    flush()
    if not out:
        return np.zeros((M.shape[0], 0), dtype=np.complex128)
    return np.concatenate(out, axis=1)


def _settings(filename_parameters: str | PathLike[str] | Settings) -> Settings:
    if isinstance(filename_parameters, Settings):
        return filename_parameters
    return load_settings_mat(filename_parameters)


def _peek(frames):
    it = iter(frames)
    try:
        first = next(it)
    except StopIteration:
        return None, it

    def chain():
        yield first
        yield from it

    return first, chain()


def das_reconstruction(
    folder: str | PathLike[str],
    filename_parameters: str | PathLike[str] | Settings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``[IMG, x, y] = DAS_reconstruction(folder, filename_parameters)``.

    Reference beamformer for a linear array, plane wave without steering or
    diverging wave (``compute_das_matrix``). ``folder`` holds the
    ``Frame*.mat`` RF files; ``filename_parameters`` is the settings ``.mat``
    file (or a :class:`~pyproteus.config.models.Settings`).

    Returns the log-compressed image ``IMG`` (``Nx``-by-``Ny``-by-``Nframes``,
    dB, axial-by-lateral) and the axial ``x`` and lateral ``y`` pixel
    coordinates [m]. Pixel size ``lambda/5``, ``x = 0:pixelSize:Xmax``,
    ``y = -width/2:pixelSize:width/2``.
    """
    S = _settings(filename_parameters)
    frames = (rf for _, rf in iter_rf_frames(folder, S.Acquisition.PulsingScheme))
    first, frames = _peek(frames)
    if first is None:
        raise FileNotFoundError(f"no Frame*.mat files in {folder}")
    Nelem, Nt = first.shape

    Fs = float(S.SimulationParameters.SamplingRate)
    c = float(S.Medium.SpeedOfSound)
    f0 = float(S.Transmit.CenterFrequency)
    p = float(S.Transducer.Pitch)
    y_el = -p / 2 * (Nelem - 1) + np.arange(Nelem) * p
    focus = float(S.Transmit.LateralFocus)

    pixel_size = c / f0 / 5
    domain = S.Geometry.Domain
    width = float(domain.Ymax) - float(domain.Ymin)
    x = colon(0.0, pixel_size, float(domain.Xmax))
    y = colon(-width / 2, pixel_size, width / 2)

    t = das_time_axis(Nt, S.SimulationParameters, S.Transducer, S.Transmit, c)
    tgc = time_gain_compensation(t, S.Medium, f0, c)
    M = compute_das_matrix(t, y, x, y_el, c, Fs, focus)

    img = beamform_frames(frames, tgc, M)  # (Ny*Nx) x Nframes, lateral first
    img = img.reshape(x.size, y.size, -1)  # C order: [ix, iy] <- iy + Ny*ix
    return log_compress(img), x, y
