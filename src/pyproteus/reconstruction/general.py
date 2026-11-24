"""General delay-and-sum for linear and curved arrays.

Ports of ``initialize_das_parameters.m`` and ``DAS_reconstruction_general.m``
(PROTEUS v1.1.0, MIT). Element positions come from
:func:`pyproteus.transducer.get_element_geometry`.
"""

from __future__ import annotations

import warnings
from os import PathLike
from typing import Any

import numpy as np

from pyproteus.config.models import Settings
from pyproteus.transducer import get_element_geometry, is_curved

from .beamform import (
    _cache_key,
    _peek,
    _settings,
    beamform_frames,
    cached_das_matrix,
    das_time_axis,
    log_compress,
    pixel_axes,
    time_gain_compensation,
)
from .das import (
    DasRecon,
    DasTransducer,
    DasTransmit,
    _field,
    compute_das_matrix_general,
    das_transmit_distance,
)
from .rf_data import iter_rf_frames

__all__ = ["DasDelayMismatchWarning", "das_reconstruction_general", "initialize_das_parameters"]


class DasDelayMismatchWarning(UserWarning):
    """``initialize_das_parameters:delays``: the transmit delays differ from the DAS model."""


def initialize_das_parameters(
    medium: Any,
    simulation_parameters: Any,
    transducer: Any,
    transmit: Any,
    geometry: Any,
    Nelem: int,
    Nt: int,
    *,
    proteus_compat: bool = False,
) -> tuple[DasRecon, DasTransducer, DasTransmit]:
    """``[Recon, Trans, Tx] = initialize_das_parameters(Medium, SimulationParameters, ...)``.

    Inputs of :func:`compute_das_matrix_general` from the PROTEUS settings for
    RF data with ``Nelem`` elements and ``Nt`` samples per element:

    * ``Recon``: pixel axes ``x = Xmin:lambda/4:Xmax`` (axial) and
      ``y = Ymin:lambda/4:Ymax`` (lateral; D-21), or PROTEUS's
      ``-width/2:lambda/4:width/2`` with ``proteus_compat=True`` (PARITY
      D-21), ``c``, the corrected time axis (time to peak, lens, k-Wave
      grid; :func:`das_time_axis`), ``Fs``,
      ``alpha_th`` (``pi/6`` curved, ``pi/3`` linear), cosine apodization;
    * ``Trans``: element positions and normals (``get_element_geometry``),
      centre and radius of curvature;
    * ``Tx``: ``focus = LateralFocus``, ``steer = deg2rad(Angle)``, transmit
      apodization (``Transmit.Apodization`` or ones), ``axis_delay = 0``.

    Raises ``ValueError`` for a configuration other than ``'Linear'`` or
    ``'Curved'`` or an element count different from ``Nelem`` (MATLAB errors).
    Warns (:class:`DasDelayMismatchWarning`) when ``Transmit.Delays`` differ
    from the focus/angle model by more than half a sample.
    """
    Fs = float(_field(simulation_parameters, "SamplingRate"))
    c = float(_field(medium, "SpeedOfSound"))
    f0 = float(_field(transmit, "CenterFrequency"))
    pixel_size = c / f0 / 4

    domain = _field(geometry, "Domain")
    x, y = pixel_axes(domain, pixel_size, x_start_compat=None, proteus_compat=proteus_compat)

    configuration = _field(transducer, "Configuration")
    if configuration not in ("Linear", "Curved"):
        raise ValueError(f"Configuration '{configuration}' is not supported.")

    El = get_element_geometry(transducer)
    if El.y.size != Nelem:
        raise ValueError(f"The RF data has {Nelem} elements, the transducer {El.y.size}.")
    alpha_th = np.pi / 6 if is_curved(transducer) else np.pi / 3

    t = das_time_axis(Nt, simulation_parameters, transducer, transmit, c)

    recon = DasRecon(x=x, y=y, c=c, t=t, Fs=Fs, alpha_th=alpha_th, apodization="cosine")
    trans = DasTransducer(
        x_el=np.array(El.x),
        y_el=np.array(El.y),
        azimuth=np.array(El.theta),
        center=El.center,
        radius=El.radius,
        type=_field(transducer, "Type"),
        configuration=configuration,
    )
    apod = _field(transmit, "Apodization")
    apod = np.ones(Nelem) if apod is None else np.asarray(apod, dtype=np.float64).reshape(-1)
    tx = DasTransmit(
        focus=float(_field(transmit, "LateralFocus")),
        steer=np.pi / 180 * float(_field(transmit, "Angle")),
        apod=apod,
        axis_delay=0.0,
    )

    delays = _field(transmit, "Delays")
    if delays is not None and np.asarray(delays).size == Nelem:
        d = das_transmit_distance(El.x, El.y, trans, tx)
        diff = np.abs(d / c - np.asarray(delays, dtype=np.float64).reshape(-1, order="F"))
        if not np.isnan(diff).all():
            mismatch = float(np.nanmax(diff))  # MATLAB max ignores NaN
            if mismatch > 1 / (2 * Fs):
                warnings.warn(
                    "The transmit delays differ from the focus/angle model by up to "
                    f"{mismatch:.3g} s: the reconstruction will be misaligned.",
                    DasDelayMismatchWarning,
                    stacklevel=2,
                )
    return recon, trans, tx


def das_reconstruction_general(
    folder: str | PathLike[str],
    filename_parameters: str | PathLike[str] | Settings,
    *,
    interpolation: str = "nearest",
    proteus_compat: bool = False,
    cache: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``[IMG, x, y] = DAS_reconstruction_general(folder, filename_parameters)``.

    Same interface as :func:`das_reconstruction`, for linear and curved arrays
    with steering and focusing (:func:`initialize_das_parameters`,
    :func:`compute_das_matrix_general`). Differences with the reference: pixel
    size ``lambda/4``, image from ``Geometry.Domain.Xmin``, cosine apodization.
    Returns ``IMG`` (``Nx``-by-``Ny``-by-``Nframes``, dB) and ``x``, ``y`` [m].
    ``interpolation``: ``'nearest'`` (default, MATLAB) or ``'linear'`` (D-20).
    By default the axes span the domain and the transmit apodization weights
    the pixels (D-21); ``proteus_compat=True`` restores PROTEUS's behaviour
    (PARITY D-21). ``cache``: reuse the DAS matrix of an identical call.
    """
    S = _settings(filename_parameters)
    frames = (rf for _, rf in iter_rf_frames(folder, S.Acquisition.PulsingScheme))
    first, frames = _peek(frames)
    if first is None:
        raise FileNotFoundError(f"no Frame*.mat files in {folder}")
    Nelem, Nt = first.shape

    recon, trans, tx = initialize_das_parameters(
        S.Medium,
        S.SimulationParameters,
        S.Transducer,
        S.Transmit,
        S.Geometry,
        Nelem,
        Nt,
        proteus_compat=proteus_compat,
    )
    tgc = time_gain_compensation(recon.t, S.Medium, float(S.Transmit.CenterFrequency), recon.c)
    key = _cache_key(
        "general", recon.x, recon.y, recon.c, recon.t, recon.Fs, recon.alpha_th,
        recon.apodization, trans.x_el, trans.y_el, trans.azimuth, trans.center,
        trans.radius, tx.focus, tx.steer, tx.apod, tx.axis_delay, interpolation,
        proteus_compat,
    )  # fmt: skip
    M = cached_das_matrix(
        key,
        lambda: compute_das_matrix_general(
            recon, trans, tx, interpolation=interpolation, proteus_compat=proteus_compat
        ),
        cache,
    )

    img = beamform_frames(frames, tgc, M)  # (Nx*Ny) x Nframes, axial first
    img = img.reshape(recon.y.size, recon.x.size, -1).transpose(1, 0, 2)
    return log_compress(img), recon.x, recon.y
