"""Unit tests (no MATLAB): transmit pulses, pressure signal, estimated impulse response."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import signal

from pyproteus import config
from pyproteus import transducer as ptr
from pyproteus import transmit as ptx
from pyproteus.config.models import Transducer, Transmit
from pyproteus.transmit.signals import hann_pulse, rect_pulse, tri_level_signal


def tx(**fields) -> Transmit:
    base = dict(
        Type="Three-level",
        CenterFrequency=2.5e6,
        NumberOfCycles=2.0,
        AcousticPressure=200e3,
        Envelope="Rectangular",
        SamplingRate=250e6,
        AmplitudeMode="Pressure",
        VoltageAmplitude=1.0,
    )
    base.update(fields)
    return Transmit(**base)


def test_tri_level_signal() -> None:
    v = tri_level_signal(250e6, 2.5e6, 2.0)
    assert v.shape == (200,) and set(np.unique(v)) == {-1.0, 0.0, 1.0}
    # high for 67 % of each half cycle, symmetric positive and negative
    assert np.sum(v == 1) == np.sum(v == -1)
    assert np.sum(v != 0) / v.size == pytest.approx(0.67, abs=0.02)
    assert v[0] == 0 and v[25] == 1 and v[75] == -1
    assert tri_level_signal(250e6, 250e6 / 82.5, 1.0).size == 83  # round half away from zero


def test_cosine_pulses() -> None:
    t = np.arange(100) / 250e6
    np.testing.assert_allclose(
        rect_pulse(250e6, 2.5e6, 1.0, -math.pi / 2), np.sin(2 * math.pi * 2.5e6 * t), atol=1e-15
    )
    h = hann_pulse(250e6, 2.5e6, 1.0, -math.pi / 2)
    assert h[0] == 0 and np.max(np.abs(h)) < 1


def test_get_voltage_signal_modes() -> None:
    out = ptx.get_voltage_signal(tx(AmplitudeMode="Voltage", VoltageAmplitude=7.5))
    assert out.VoltageSignal.shape == (1, 200) and np.max(np.abs(out.VoltageSignal)) == 7.5
    keep = np.array([[0.0, 1.0, -2.0]])
    assert np.array_equal(
        ptx.get_voltage_signal(tx(Type="Custom", VoltageSignal=keep)).VoltageSignal, keep
    )
    scaled = ptx.get_voltage_signal(
        tx(Type="Custom", VoltageSignal=keep, AmplitudeMode="Voltage", VoltageAmplitude=4.0)
    )
    np.testing.assert_array_equal(scaled.VoltageSignal, [[0.0, 2.0, -4.0]])
    hann = ptx.get_voltage_signal(tx(Type="Cosine envelope", Envelope="Hann"))
    assert hann.VoltageSignal.shape == (1, 200)
    # all-zero pulse in voltage mode: NaN, as MATLAB (Q-82)
    zero = ptx.get_voltage_signal(
        tx(
            CenterFrequency=26093751.0,
            NumberOfCycles=1.0,
            SamplingRate=62.5e6,
            AmplitudeMode="Voltage",
        )
    )
    assert np.all(np.isnan(zero.VoltageSignal))
    with pytest.raises(ValueError, match="VoltageSignal"):
        ptx.get_voltage_signal(tx(Type="Custom", AmplitudeMode="Voltage"))


def test_get_pressure_signal() -> None:
    t = Transducer(SamplingRate=1e6, TransmitImpulseResponse=np.array([[1.0, 2.0, 0.5]]))
    v = tx(Type="Custom", VoltageSignal=np.array([[1.0, 0.0, -1.0]]))
    raw = np.convolve([1.0, 0.0, -1.0], [1.0, 2.0, 0.5]) / 1e6
    out = ptx.get_pressure_signal(v, t)
    np.testing.assert_allclose(out.PressureSignal, [raw * 200e3 / abs(raw.min())])
    assert out.PressureSignal.min() == pytest.approx(-200e3)
    assert out.MechanicalIndex == pytest.approx(0.2 / math.sqrt(2.5))
    vm = v.copy_struct()
    vm.AmplitudeMode = "Voltage"
    outv = ptx.get_pressure_signal(vm, t)
    np.testing.assert_allclose(outv.PressureSignal, [raw])
    assert outv.AcousticPressure == pytest.approx(abs(raw.min()))
    custom = ptx.get_pressure_signal(
        tx(Type="Custom (pressure)", PressureSignal=np.array([[np.nan], [-2.0], [1.0]])), t
    )
    assert custom.PressureSignal.shape == (3, 1) and custom.PressureSignal[1, 0] == -200e3
    with pytest.raises(ValueError, match="TransmitImpulseResponse"):
        ptx.get_pressure_signal(v, Transducer(SamplingRate=1e6))


def test_impulse_response_matches_zpk_reference() -> None:
    lo, hi, fs = 1.5e6, 3.5e6, 250e6
    ir = ptr.butterworth_impulse_response(lo, hi, fs)
    assert ir.shape == (625,)  # round(5*250/2)
    z, p, k = signal.butter(3, [lo / (fs / 2), hi / (fs / 2)], "bandpass", output="zpk")
    r = np.array(
        [k * np.prod(1 - z / p[j]) / np.prod(1 - np.delete(p, j) / p[j]) for j in range(p.size)]
    )
    h = (r[None, :] * p[None, :] ** np.arange(625)[:, None]).sum(axis=1).real
    h[0] += (k * np.prod(z) / np.prod(p)).real
    np.testing.assert_allclose(ir, h * fs, atol=1e-12 * np.max(np.abs(h * fs)))
    # -3 dB at the band edges, unit gain at the geometric centre
    _, resp = signal.sosfreqz(
        signal.butter(3, [lo, hi], "bandpass", fs=fs, output="sos"),
        worN=[lo, hi, math.sqrt(lo * hi)],
        fs=fs,
    )
    np.testing.assert_allclose(np.abs(resp), [2**-0.5, 2**-0.5, 1.0], rtol=1e-9)


def test_estimate_impulse_response_and_defaults() -> None:
    t = ptr.estimate_impulse_response(
        Transducer(BandwidthLow=1.5e6, BandwidthHigh=3.5e6, SamplingRate=250e6)
    )
    assert t.TransmitImpulseResponse.shape == (1, 625)
    np.testing.assert_array_equal(t.TransmitImpulseResponse, t.ReceiveImpulseResponse)
    assert t.TransmitImpulseResponse is not t.ReceiveImpulseResponse
    d = config.reset_transducer()
    np.testing.assert_array_equal(d.TransmitImpulseResponse, t.TransmitImpulseResponse)
    tx_default = config.reset_transmit(d)
    assert tx_default.VoltageSignal.shape == (1, 200)
    assert tx_default.PressureSignal.shape == (1, 824)
    assert tx_default.PressureSignal.min() == pytest.approx(-200e3)
    assert tx_default.MechanicalIndex == pytest.approx(0.2 / math.sqrt(2.5))
    assert config.PENDING_FIELDS == {}
