"""Oracle tests: ``get_voltage_signal``, ``get_pressure_signal``, ``estimate_impulse_response``.

Classes (D-13):
- rectangular/Hann pulses, pressure signals: float, ``1e-9*|e| + 1e-14*max|e|``;
- three-level pulses (values -1/0/1 from thresholding a sine): statistical,
  same length and >= 99.9 % of samples equal, differences reported;
- impulse response: see ``IR_VS_MATLAB`` and B-80 (MATLAB's (b, a) + impz path is
  inaccurate). The realisation is checked against partial fractions of SciPy's zpk
  design, and the design itself against MATLAB's zpk run through MATLAB's
  ``zp2sos`` + ``sosfilt`` (D-17).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from scipy import signal

from pyproteus import transducer as ptr
from pyproteus import transmit as ptx
from pyproteus.config.models import Transducer, Transmit
from pyproteus.numerics import matlab_round
from tests.oracle.helpers import ORACLE_SETTINGS, from_matlab, to_matlab

REL = 1e-9
ATOL_FRACTION = 1e-14
#: Impulse response vs MATLAB, relative to max|IR| (D-17, B-80): MATLAB's butter (b, a)
#: + impz result is off by up to 1.5e-8 from the exact response for catalogue-like bands.
IR_VS_MATLAB = 1e-7
#: Impulse response vs the partial-fraction reference, relative to max|IR|.
IR_VS_REFERENCE = 1e-12  # D-17; measured <= 5e-14 for catalogue-like bands
#: Narrow bands (high Q at high sampling rates): the reference itself loses digits.
IR_VS_REFERENCE_NARROW = 1e-11  # measured <= 2.7e-12


def assert_float(
    actual: Any, expected: Any, err_msg: str, rel: float = REL, atol_fraction: float = ATOL_FRACTION
) -> None:
    a, e = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if a.ndim == 0 and e.shape == (1, 1):  # Python scalar vs MATLAB 1×1
        a = a.reshape(1, 1)
    assert a.shape == e.shape, f"shape {a.shape} != {e.shape} {err_msg}"
    assert np.array_equal(np.isnan(a), np.isnan(e)), f"NaN positions differ {err_msg}"
    inf = np.isinf(e)
    assert np.array_equal(np.isinf(a), inf), f"Inf positions differ {err_msg}"
    np.testing.assert_array_equal(a[inf], e[inf], err_msg=err_msg)
    fin = np.isfinite(e)
    if not fin.any():
        return
    err = np.abs(a[fin] - e[fin])
    bound = rel * np.abs(e[fin]) + atol_fraction * np.max(np.abs(e[fin]))
    assert np.all(err <= bound), f"max excess {np.max(err - bound):.3g} {err_msg}"


def to_engine(model: Any) -> dict[str, Any]:
    out = {}
    for k, v in model.to_tree().items():
        if isinstance(v, np.ndarray):
            out[k] = to_matlab(v if v.ndim >= 2 else v.reshape(1, -1))
        else:
            out[k] = float(v) if isinstance(v, int) else v
    return out


def from_engine(s: dict[str, Any]) -> dict[str, Any]:
    return {k: (v if isinstance(v, str) else from_matlab(v)) for k, v in s.items()}


def matlab_call(eng: Any, fn: str, *args: Any) -> list[dict[str, Any]]:
    """``fn`` on each tuple of models in ``args`` (lists of equal length), one engine call."""
    names = []
    for j, models in enumerate(args):
        eng.workspace[f"pyp_a{j}__"] = [to_engine(m) for m in models]
        names.append(f"pyp_a{j}__")
    eng.eval(f"pyp_o__ = cellfun(@{fn}, {', '.join(names)}, 'UniformOutput', false);", nargout=0)
    out = [from_engine(s) for s in eng.workspace["pyp_o__"]]
    eng.eval(f"clear pyp_o__ {' '.join(names)}", nargout=0)
    return out


# ------------------------------------------------------------- voltage


def tx(**fields: Any) -> Transmit:
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


def compare_voltage(actual: Transmit, expected: dict[str, Any], label: str) -> list[str]:
    """Returns a report line for three-level differences (statistical class)."""
    e = expected.get("VoltageSignal")
    if e is None:
        assert actual.VoltageSignal is None, label
        return []
    a = np.asarray(actual.VoltageSignal)
    if actual.Type == "Three-level":  # discrete levels (0, ±1 or ±VoltageAmplitude)
        assert a.shape == e.shape, f"length {a.shape} != {e.shape} {label}"
        diff = int(np.sum(~((a == e) | (np.isnan(a) & np.isnan(e)))))
        assert diff <= 1e-3 * e.size, f"{diff}/{e.size} samples differ {label}"
        return [f"{label}: {diff}/{e.size}"] if diff else []
    assert_float(a, e, label)
    return []


FREQS = [2.5e6, 3.4e6, 18e6, 250e6 / 82.5, 250e6 * 2 / 165.0 * 1.0000001]
CYCLES = [1.0, 2.0, 2.5, 0.5, 3.0, 1.0 / 3]


def voltage_cases() -> list[Transmit]:
    cases = []
    for f in FREQS:
        for ncy in CYCLES:
            for mode in ("Pressure", "Voltage"):
                cases.append(
                    tx(
                        CenterFrequency=f,
                        NumberOfCycles=ncy,
                        AmplitudeMode=mode,
                        VoltageAmplitude=7.5,
                    )
                )
                for env in ("Rectangular", "Hann"):
                    cases.append(
                        tx(
                            Type="Cosine envelope",
                            Envelope=env,
                            CenterFrequency=f,
                            NumberOfCycles=ncy,
                            AmplitudeMode=mode,
                            VoltageAmplitude=7.5,
                        )
                    )
    # no case in the switch: VoltageSignal kept
    keep = np.array([[0.0, 1.0, -2.0, 0.5]])
    cases.append(tx(Type="Custom (voltage)", VoltageSignal=keep))
    cases.append(tx(Type="Custom (voltage)", VoltageSignal=keep, AmplitudeMode="Voltage"))
    cases.append(tx(Type="Cosine envelope", Envelope="Gaussian", VoltageSignal=keep))
    cases.append(tx(Type="Three-level", SamplingRate=62.5e6, CenterFrequency=5.3e6))
    return cases


def test_voltage_signal(matlab_engine: Any) -> None:
    cases = voltage_cases()
    report = []
    for case, expected in zip(
        cases, matlab_call(matlab_engine, "get_voltage_signal", cases), strict=True
    ):
        label = f"{case.Type}/{case.Envelope} f={case.CenterFrequency} Ncy={case.NumberOfCycles}"
        report += compare_voltage(ptx.get_voltage_signal(case), expected, label)
    print("three-level differences:", report or "none")


@ORACLE_SETTINGS
@given(
    st.lists(
        st.tuples(
            st.sampled_from(
                [
                    ("Three-level", "Rectangular"),
                    ("Cosine envelope", "Rectangular"),
                    ("Cosine envelope", "Hann"),
                ]
            ),
            st.floats(0.5e6, 30e6),
            st.floats(0.25, 10),
            st.sampled_from([62.5e6, 125e6, 250e6, 1e9]),
            st.sampled_from(["Pressure", "Voltage"]),
        ),
        min_size=1,
        max_size=20,
    )
)
def test_voltage_signal_random(matlab_engine: Any, params: list) -> None:
    cases = [
        tx(
            Type=t,
            Envelope=e,
            CenterFrequency=f,
            NumberOfCycles=n,
            SamplingRate=fs,
            AmplitudeMode=m,
            VoltageAmplitude=3.0,
        )
        for (t, e), f, n, fs, m in params
    ]
    for case, expected in zip(
        cases, matlab_call(matlab_engine, "get_voltage_signal", cases), strict=True
    ):
        a = ptx.get_voltage_signal(case).VoltageSignal
        e = expected["VoltageSignal"]
        if case.Type == "Three-level":
            assert a.shape == e.shape
            assert_three_level_ties(a, e, case)
        else:
            assert_float(a, e, repr(case.to_tree()))


def assert_three_level_ties(a: np.ndarray, e: np.ndarray, case: Transmit) -> None:
    """Samples may differ only where the sine is within 1e-12 of the threshold (a tie
    decided by the last bits of ``sin``); the fixed-case test checks the overall rate."""
    same = (a == e) | (np.isnan(a) & np.isnan(e))
    differ = np.flatnonzero(~same.ravel())
    if differ.size == 0:
        return
    v = np.sin(2 * np.pi * case.CenterFrequency * differ / case.SamplingRate)
    gap = np.abs(np.abs(v) - np.sin((1 - ptx.signals.ON_FRACTION) * np.pi / 2))
    assert np.all(gap <= 1e-12), f"non-tie three-level differences at {differ} (gap {gap})"


# ------------------------------------------------------------- pressure


@pytest.fixture(scope="module")
def matlab_irs(matlab_engine: Any) -> dict[str, Transducer]:
    """Transducers with MATLAB's impulse responses (estimated or from model files)."""
    out = {}
    for ttype in ("P4-1", "C1-6D", "L22-14v"):
        matlab_engine.eval(
            f"pyp_T__ = get_transducer_properties(struct('Type','{ttype}'));"
            "if strcmp(pyp_T__.ImpulseResponseType,'Estimate'), pyp_T__.SamplingRate = 250e6;"
            " pyp_T__ = estimate_impulse_response(pyp_T__); end",
            nargout=0,
        )
        t = from_engine(matlab_engine.workspace["pyp_T__"])
        matlab_engine.eval("clear pyp_T__", nargout=0)
        out[ttype] = Transducer(**{k: v for k, v in t.items()})
    return out


def test_pressure_signal(matlab_engine: Any, matlab_irs: dict[str, Transducer]) -> None:
    transmits, transducers = [], []
    for t in matlab_irs.values():
        for case in voltage_cases()[:40]:
            v = ptx.get_voltage_signal(case)
            for mode in ("Pressure", "Voltage", "Other"):
                c = v.copy_struct()
                c.AmplitudeMode = mode
                transmits.append(c)
                transducers.append(t)
    # Custom pressure: kept as given (row, column, NaN), scaled
    t0 = next(iter(matlab_irs.values()))
    for p in (
        np.array([[0.1, -0.4, 0.3]]),
        np.array([[0.1], [-0.4], [0.3]]),
        np.array([[np.nan, -2.0, 1.0]]),
    ):
        for mode in ("Pressure", "Voltage"):
            transmits.append(tx(Type="Custom (pressure)", PressureSignal=p, AmplitudeMode=mode))
            transducers.append(t0)
    # short rows and a scalar impulse response (pyproteus always returns a row; MATLAB's
    # conv orientation for column inputs is not reproduced, docs/PARITY.md)
    for ir in (np.array([[1.0, 2.0, 0.5]]), np.array([[3.0]])):
        for v in (
            tx(VoltageSignal=np.array([[1.0, 0.0, -1.0]]), Type="Custom (voltage)"),
            tx(VoltageSignal=np.array([[2.0]]), Type="Custom (voltage)"),
        ):
            transmits.append(v)
            transducers.append(Transducer(SamplingRate=1e6, TransmitImpulseResponse=ir))

    expected = matlab_call(matlab_engine, "get_pressure_signal", transmits, transducers)
    for case, t, e in zip(transmits, transducers, expected, strict=True):
        a = ptx.get_pressure_signal(case, t)
        label = f"{t.Type} {case.Type} {case.AmplitudeMode} f={case.CenterFrequency}"
        assert_float(a.PressureSignal, e["PressureSignal"], f"PressureSignal {label}")
        assert_float(a.AcousticPressure, e["AcousticPressure"], f"AcousticPressure {label}")
        assert_float(a.MechanicalIndex, e["MechanicalIndex"], f"MI {label}")
        assert a.present_fields() == list(e), label


# ------------------------------------------------------------- impulse response


def reference_ir(low: float, high: float, fs: float) -> np.ndarray:
    """Partial fractions of the digital zpk: h[n] = c*delta[n] + sum_j r_j p_j^n (times fs)."""
    fs_mhz = fs / 1e6
    band = np.array([low, high]) / 1e6
    n = int(matlab_round(5 * fs_mhz / (band[1] - band[0])))
    z, p, k = signal.butter(3, band / (fs_mhz / 2), "bandpass", output="zpk")
    r = np.array(
        [k * np.prod(1 - z / p[j]) / np.prod(1 - np.delete(p, j) / p[j]) for j in range(p.size)]
    )
    h = (r[None, :] * p[None, :] ** np.arange(n)[:, None]).sum(axis=1).real
    h[0] += (k * np.prod(z) / np.prod(p)).real
    return h * fs


BANDS = [  # (low, high, fs): catalogue bands, odd rates
    (1.5e6, 3.5e6, 250e6),
    (1.5e6, 5.5e6, 250e6),
    (14e6, 22e6, 250e6),
    (3.3e6, 7.3e6, 250e6),
    (3e6, 9e6, 250e6),
    (2e6, 4e6, 62.5e6),
    (1e6, 1.9e6, 125e6),
    (5e6, 10e6, 100e6),
]
NARROW = [(0.5e6, 0.6e6, 250e6), (1e6, 2e6, 1e9)]  # MATLAB inaccurate by percent (B-80)


@pytest.mark.parametrize(("low", "high", "fs"), BANDS + NARROW)
def test_impulse_response_vs_reference(low: float, high: float, fs: float) -> None:
    a = ptr.butterworth_impulse_response(low, high, fs)
    h = reference_ir(low, high, fs)
    assert a.shape == h.shape
    bound = IR_VS_REFERENCE_NARROW if (low, high, fs) in NARROW else IR_VS_REFERENCE
    assert np.max(np.abs(a - h)) <= bound * np.max(np.abs(h))


def test_impulse_response_vs_matlab(matlab_engine: Any) -> None:
    cases = [
        Transducer(BandwidthLow=lo, BandwidthHigh=hi, SamplingRate=fs)
        for lo, hi, fs in BANDS + NARROW
    ]
    expected = matlab_call(matlab_engine, "estimate_impulse_response", cases)
    report = []
    for case, e in zip(cases, expected, strict=True):
        a = ptr.estimate_impulse_response(case)
        assert list(e) == a.present_fields()
        for k in ("TransmitImpulseResponse", "ReceiveImpulseResponse"):
            ea, ee = getattr(a, k), e[k]
            assert ea.shape == ee.shape, k
            err = np.max(np.abs(ea - ee)) / np.max(np.abs(ee))
            report.append(
                f"{case.BandwidthLow / 1e6}-{case.BandwidthHigh / 1e6} MHz "
                f"@ {case.SamplingRate / 1e6} MHz: {err:.2e}"
            )
            if (case.BandwidthLow, case.BandwidthHigh, case.SamplingRate) not in NARROW:
                assert err <= IR_VS_MATLAB, report[-1]
    print("IR vs MATLAB (rel. to max):", "; ".join(report[::2]))


def test_impulse_response_vs_matlab_zpk_sos(matlab_engine: Any) -> None:
    """Same filter design as MATLAB: MATLAB's butter zpk realised by MATLAB's zp2sos/sosfilt."""
    eng = matlab_engine
    for low, high, fs in BANDS + NARROW:
        band = np.array([low, high]) / 1e6
        n = int(matlab_round(5 * fs / 1e6 / (band[1] - band[0])))
        eng.eval(
            f"[pyp_z__,pyp_p__,pyp_k__] = butter(3,"
            f" [{low / 1e6!r} {high / 1e6!r}]/({fs / 1e6!r}/2), 'bandpass');"
            f" pyp_h__ = sosfilt(zp2sos(pyp_z__,pyp_p__,pyp_k__), [1 zeros(1,{n - 1})])"
            f" * {fs!r};",
            nargout=0,
        )
        e = from_matlab(eng.workspace["pyp_h__"]).ravel()
        eng.eval("clear pyp_z__ pyp_p__ pyp_k__ pyp_h__", nargout=0)
        a = ptr.butterworth_impulse_response(low, high, fs)
        bound = IR_VS_REFERENCE_NARROW if (low, high, fs) in NARROW else IR_VS_REFERENCE
        assert np.max(np.abs(a - e)) <= bound * np.max(np.abs(e)), (low, high, fs)
