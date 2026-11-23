"""Transmit pulses (ports of ``GUIfunctions/get_voltage_signal.m``, ``get_pressure_signal.m``).

Signals are 1×N rows (MATLAB size), sampled at ``Transmit.SamplingRate``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from pyproteus.config.models import Transmit
from pyproteus.numerics import conv, matlab_round

__all__ = [
    "compute_mechanical_index",
    "get_pressure_signal",
    "get_voltage_signal",
    "hann_pulse",
    "rect_pulse",
    "tri_level_signal",
]

#: Fraction of a half cycle at the high level in a three-level pulse.
ON_FRACTION = 0.67


def _scalar(s: Any, name: str, owner: str) -> float:
    value = getattr(s, name, None)
    if value is None:
        raise ValueError(f"{owner} has no field {name}")  # MATLAB: reference error
    return float(np.asarray(value, dtype=np.float64).reshape(()))


def tri_level_signal(fs: float, f: float, n_cycles: float) -> np.ndarray:
    """Pulse train of alternating positive and negative block pulses (1-D, values -1/0/1).

    ``round(Fs/f*Ncy)`` samples of ``sin(2*pi*f*t)``, thresholded at
    ``sin((1 - 0.67)*pi/2)``: high for 67 % of each half cycle.
    """
    n = int(matlab_round(fs / f * n_cycles))
    v = np.sin(2 * math.pi * f * np.arange(n, dtype=np.float64) / fs)
    v_th = math.sin((1 - ON_FRACTION) * math.pi / 2)
    return np.where(v >= v_th, 1.0, np.where(v <= -v_th, -1.0, 0.0))


def _time(fs: float, f: float, n_cycles: float) -> np.ndarray:
    n = int(matlab_round(n_cycles * fs / f))
    return np.arange(n, dtype=np.float64) / fs


def rect_pulse(fs: float, f: float, n_cycles: float, phi: float) -> np.ndarray:
    """``Ncy``-cycle cosine at ``f`` with phase ``phi`` and a rectangular window (1-D)."""
    t = _time(fs, f, n_cycles)
    return np.cos(2 * math.pi * f * t + phi)


def hann_pulse(fs: float, f: float, n_cycles: float, phi: float) -> np.ndarray:
    """``Ncy``-cycle cosine at ``f`` with phase ``phi`` and a Hann window (1-D)."""
    t = _time(fs, f, n_cycles)
    return np.cos(2 * math.pi * f * t + phi) * np.sin(2 * math.pi * f * t / n_cycles / 2) ** 2


def get_voltage_signal(transmit: Transmit) -> Transmit:
    """Port of ``get_voltage_signal.m``: ``Transmit.VoltageSignal`` (1×N row).

    ``Type`` ``'Three-level'``: :func:`tri_level_signal`; ``'Cosine envelope'``
    with ``Envelope`` ``'Rectangular'`` or ``'Hann'``: a sine-phase pulse
    (:func:`rect_pulse`, :func:`hann_pulse`, phase ``-pi/2``). Any other type
    or envelope leaves ``VoltageSignal`` unchanged (MATLAB ``switch`` without
    ``otherwise``). In ``AmplitudeMode`` ``'Voltage'`` the signal is scaled so
    that its largest absolute value is ``VoltageAmplitude``.
    Returns a copy of ``transmit``.
    """
    tx = transmit.copy_struct()
    ttype = tx.Type
    if ttype is None:
        raise ValueError("Transmit has no field Type")  # MATLAB: reference error

    def pulse_parameters() -> tuple[float, float, float]:
        return (
            _scalar(tx, "SamplingRate", "Transmit"),
            _scalar(tx, "CenterFrequency", "Transmit"),
            _scalar(tx, "NumberOfCycles", "Transmit"),
        )

    if ttype == "Three-level":
        tx.VoltageSignal = tri_level_signal(*pulse_parameters()).reshape(1, -1)
    elif ttype == "Cosine envelope":
        if tx.Envelope is None:
            raise ValueError("Transmit has no field Envelope")
        if tx.Envelope == "Rectangular":
            tx.VoltageSignal = rect_pulse(*pulse_parameters(), -math.pi / 2).reshape(1, -1)
        elif tx.Envelope == "Hann":
            tx.VoltageSignal = hann_pulse(*pulse_parameters(), -math.pi / 2).reshape(1, -1)

    if tx.AmplitudeMode is None:
        raise ValueError("Transmit has no field AmplitudeMode")
    if tx.AmplitudeMode == "Voltage":
        v = tx.VoltageSignal
        if v is None:
            raise ValueError("Transmit has no field VoltageSignal")
        v = np.asarray(v, dtype=np.float64)
        peak = np.fmax.reduce(np.abs(v), axis=None) if v.size else np.float64(np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):  # all-zero signal: NaN (Q-82)
            tx.VoltageSignal = v * (_scalar(tx, "VoltageAmplitude", "Transmit") / peak)
    return tx


def compute_mechanical_index(transmit: Transmit) -> float:
    """Mechanical index: peak negative pressure [MPa] / sqrt(centre frequency [MHz])."""
    f = _scalar(transmit, "CenterFrequency", "Transmit") / 1e6
    p = _scalar(transmit, "AcousticPressure", "Transmit") / 1e6
    return p / math.sqrt(f)


def get_pressure_signal(transmit: Transmit, transducer: Any) -> Transmit:
    """Port of ``get_pressure_signal.m``: ``Transmit.PressureSignal`` and ``MechanicalIndex``.

    The pressure is the driving voltage convolved with the transmit impulse
    response, ``conv(V, IR)/Transducer.SamplingRate`` (both assumed sampled at
    that rate; ``Transmit.SamplingRate`` is not checked, Q-81), or the given
    ``PressureSignal`` for ``Type`` ``'Custom (pressure)'``. ``AmplitudeMode``
    ``'Pressure'`` scales it to a peak negative pressure ``AcousticPressure``;
    ``'Voltage'`` sets ``AcousticPressure`` to its peak negative pressure.
    Returns a copy of ``transmit``.
    """
    tx = transmit.copy_struct()
    if tx.Type == "Custom (pressure)":
        if tx.PressureSignal is None:
            raise ValueError("Transmit has no field PressureSignal")
        p = np.asarray(tx.PressureSignal, dtype=np.float64)
    else:
        if tx.VoltageSignal is None:
            raise ValueError("Transmit has no field VoltageSignal")
        ir = getattr(transducer, "TransmitImpulseResponse", None)
        if ir is None:
            raise ValueError("Transducer has no field TransmitImpulseResponse")
        fs = _scalar(transducer, "SamplingRate", "Transducer")
        v = np.asarray(tx.VoltageSignal, dtype=np.float64)
        ir = np.asarray(ir, dtype=np.float64)
        p = (conv(v, ir) / fs).reshape(1, -1)  # always a row (PARITY.md)

    if tx.AmplitudeMode is None:
        raise ValueError("Transmit has no field AmplitudeMode")
    # Peak negative pressure (MATLAB min ignores NaN)
    pnp = abs(np.fmin.reduce(p, axis=None)) if p.size else np.float64(np.nan)
    if tx.AmplitudeMode == "Pressure":
        with np.errstate(divide="ignore", invalid="ignore"):  # pnp = 0: Inf/NaN (Q-82)
            p = p * (_scalar(tx, "AcousticPressure", "Transmit") / pnp)
    elif tx.AmplitudeMode == "Voltage":
        tx.AcousticPressure = float(pnp)
    tx.PressureSignal = p
    tx.MechanicalIndex = compute_mechanical_index(tx)
    return tx
