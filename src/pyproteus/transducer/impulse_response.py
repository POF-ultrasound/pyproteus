"""Estimated transducer impulse response (port of ``estimate_impulse_response.m``).

PROTEUS approximates the impulse response of an uncalibrated transducer by a
third-order Butterworth bandpass filter between ``BandwidthLow`` and
``BandwidthHigh`` (-3 dB points), sampled at ``SamplingRate``, over
``round(5*Fs/bandwidth)`` samples, scaled by the sampling rate.

Divergence (D-15, B-80): the filter is designed and applied in second-order
sections (``scipy.signal.butter(..., output='sos')``, ``sosfilt``) instead of
the transfer-function (b, a) form PROTEUS uses. The (b, a) form of this
bandpass is ill-conditioned (poles near z = 1); the sections are accurate to
~1e-14 relative, while the (b, a) result loses up to 1.0e-8 for the catalogue
bands (1.5e-8 for other wide bands) and percent-level accuracy for narrow bands.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import signal

from pyproteus.numerics import matlab_round

from ._fields import scalar

__all__ = ["BUTTERWORTH_ORDER", "butterworth_impulse_response", "estimate_impulse_response"]

#: Order of the Butterworth prototype (the bandpass filter has order 6). PROTEUS:
#: "gives a result that looks most like experimentally determined impulse response
#: P4-1, with derivative zero at t = 0".
BUTTERWORTH_ORDER = 3


def butterworth_impulse_response(low: float, high: float, fs: float) -> np.ndarray:
    """Impulse response (1-D) of the Butterworth bandpass, multiplied by ``fs``.

    ``low``, ``high`` are the -3 dB band edges and ``fs`` the sampling rate,
    all in Hz. Length ``round(5*Fs/df)`` with ``Fs``, ``df`` in MHz (as PROTEUS).
    """
    fs_mhz = fs / 1e6
    band = np.array([low, high], dtype=np.float64) / 1e6
    n = int(matlab_round(5 * fs_mhz / (band[1] - band[0])))
    wn = band / (fs_mhz / 2)
    sos = signal.butter(BUTTERWORTH_ORDER, wn, btype="bandpass", output="sos")
    impulse = np.zeros(max(n, 0))
    if n > 0:
        impulse[0] = 1.0
    return signal.sosfilt(sos, impulse) * fs


def estimate_impulse_response(transducer: Any) -> Any:
    """Port of ``estimate_impulse_response.m``.

    Sets ``TransmitImpulseResponse`` and ``ReceiveImpulseResponse`` (1×N rows,
    identical) from ``BandwidthLow``, ``BandwidthHigh`` and ``SamplingRate``.
    Returns a copy of ``transducer`` (a :class:`pyproteus.config.Transducer`).
    """
    t = transducer.copy_struct()
    ir = butterworth_impulse_response(
        scalar(t, "BandwidthLow"), scalar(t, "BandwidthHigh"), scalar(t, "SamplingRate")
    ).reshape(1, -1)
    t.TransmitImpulseResponse = ir
    t.ReceiveImpulseResponse = ir.copy()
    return t
