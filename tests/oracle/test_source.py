"""Oracle tests: ``preprocess_transmit`` and ``define_source_transducer``.

Classes (D-13): source masks exactly; source signals (``p``, ``ux``, ``uy``) and
the preprocessed transmit signals at the float class, ≤ 1e-9 × max|signal|
(``assert_rel``; FFT libraries differ, FFTW vs pocketfft). Sampling rates and
signal lengths exactly.

Cases: linear (P4-1) and curved (C1-6D) arrays, steered (varied delays), integration
points off and on the grid, ``Transducer.SourceType`` absent / ``'dipole'`` /
``'monopole'``, every ``Transmit.SeqPulse`` value with odd element counts (so
``'odd'`` and ``'even'`` silence different numbers of elements), and an apodized
transmit.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pyproteus import acoustic
from pyproteus.backends.kwave import KGrid
from tests.oracle.helpers import assert_rel, from_matlab, get, put, row
from tests.oracle.test_acoustic_grid import build_settings, load_settings
from tests.oracle.test_sensor import transducer_case

REL = 1e-9
KWAVE_MATLAB = Path(
    os.environ.get(
        "KWAVE_MATLAB_PATH",
        "/Users/theodoremichel/Repos/TWENTE/Ultrasound/PROTEUS-cleaning/PROTEUS/external/k-wave/k-Wave",
    )
)


@pytest.fixture(scope="module")
def keng(matlab_engine: Any) -> Any:
    """Engine with MATLAB k-Wave on the path (``filterTimeSeries``, ``kWaveGrid``)."""
    pytest.importorskip("kwave")
    if not (KWAVE_MATLAB / "filterTimeSeries.m").is_file():
        pytest.skip(f"MATLAB k-Wave not found at {KWAVE_MATLAB} (set KWAVE_MATLAB_PATH)")
    matlab_engine.addpath(str(KWAVE_MATLAB), nargout=0)
    yield matlab_engine
    matlab_engine.rmpath(str(KWAVE_MATLAB), nargout=0)


# ------------------------------------------------------------------ preprocess_transmit


@pytest.mark.parametrize(
    ("shape", "spacing", "dt"),
    [
        ((64, 48, 32), (1e-4, 1e-4, 1e-4), 2e-8),  # PROTEUS-like: dt from CFL
        ((37, 29, 23), (1.3e-4, 1.3e-4, 1.3e-4), 3.3e-8),  # odd sizes, k-Wave rate not a divisor
    ],
)
def test_preprocess_transmit(
    keng: Any, oracle: Any, tmp_path: Path, shape: tuple, spacing: tuple, dt: float
) -> None:
    eng = keng
    build_settings(eng, oracle.path, "pys_S__")
    S = load_settings(eng, tmp_path, "pys_S__")
    eng.eval(
        "pys_kg__ = kWaveGrid({}, {!r}, {}, {!r}, {}, {!r}); pys_kg__.dt = {!r};"
        "pys_Tx__ = preprocess_transmit(pys_S__.Transmit, pys_S__.Medium, pys_kg__);"
        "pys_p__ = pys_Tx__.PressureSignal; pys_v__ = pys_Tx__.VoltageSignal;"
        "pys_fs__ = pys_Tx__.SamplingRate;".format(
            *[v for pair in zip(shape, spacing, strict=True) for v in pair], dt
        ),
        nargout=0,
    )
    p_ref, v_ref, fs_ref = get(eng, "pys_p__", "pys_v__", "pys_fs__")
    out = acoustic.preprocess_transmit(S.Transmit, S.Medium, KGrid(shape, spacing, dt, 1))
    assert p_ref.shape[0] == 1 and v_ref.shape[0] == 1  # MATLAB rows, as pyproteus
    assert_rel(out.PressureSignal, p_ref, REL, "PressureSignal")
    assert_rel(out.VoltageSignal, v_ref, REL, "VoltageSignal")
    assert out.SamplingRate == fs_ref.item()
    assert S.Transmit.SamplingRate != out.SamplingRate  # input left unchanged
    eng.eval("clear pys_kg__ pys_Tx__ pys_p__ pys_v__ pys_fs__ pys_S__", nargout=0)


# ------------------------------------------------------------------ define_source_transducer

TRANSDUCERS = [("P4-1", 7, 1.0), ("C1-6D", 5, 1.0)]
SOURCE_TYPES = [None, "dipole", "monopole"]


@pytest.fixture(scope="module", params=TRANSDUCERS, ids=["P4-1", "C1-6D"])
def tcase(request: Any, matlab_engine: Any, oracle: Any, tmp_path_factory: Any):
    tmp = tmp_path_factory.mktemp("src")
    S, g, T = transducer_case(matlab_engine, oracle, tmp, *request.param)
    matlab_engine.eval("pys_S0__ = pys_S__; pys_G0__ = pys_G__; pys_T0__ = pys_T__;", nargout=0)
    return S, g, T


def run_case(
    eng: Any, case: tuple, on_grid: bool, source_type: str | None, seq: str, apodize: bool
) -> None:
    S, g, T = case
    T = dict(T, OnGrid=on_grid)
    Tx = S.Transmit.copy_struct()
    Tx.SeqPulse = seq
    eng.eval(
        "pys_T__ = pys_T0__; pys_Tx__ = pys_S0__.Transmit;"
        f"pys_T__.OnGrid = {str(on_grid).lower()}; pys_Tx__.SeqPulse = '{seq}';",
        nargout=0,
    )
    if source_type is not None:
        T["SourceType"] = source_type
        eng.eval(f"pys_T__.SourceType = '{source_type}';", nargout=0)
    if apodize:
        n = int(S.Transducer.NumberOfElements)
        apod = np.linspace(0.3, 1.0, n)
        Tx.Apodization = apod.reshape(1, -1)
        put(eng, pys_ap__=row(apod))
        eng.eval("pys_Tx__.Apodization = pys_ap__;", nargout=0)

    sensor, w = acoustic.define_sensor_transducer(T, g)
    mask_idx = np.flatnonzero(sensor.mask.ravel(order="F"))
    eng.eval(
        "[pys_s__, pys_w__] = define_sensor_transducer(pys_T__, pys_G0__);"
        "pys_mi__ = find(logical(pys_s__.mask));"
        "pys_src__ = define_source_transducer(pys_T__, pys_Tx__, pys_S0__.Medium, pys_G0__,"
        " transpose(pys_w__), pys_mi__);",
        nargout=0,
    )
    np.testing.assert_array_equal(mask_idx + 1, from_matlab(eng.eval("pys_mi__")).ravel())
    src = acoustic.define_source_transducer(T, Tx, S.Medium, g, w.T, mask_idx)
    label = f"{T.get('Configuration')} on_grid={on_grid} {source_type} {seq} apod={apodize}"
    fields = eng.eval("fieldnames(pys_src__)", nargout=1)
    if source_type == "monopole":
        assert set(fields) == {"p_mask", "p"}, label
        assert src.u_mask is None and src.ux is None
        np.testing.assert_array_equal(src.p_mask, from_matlab(eng.eval("pys_src__.p_mask")))
        assert_rel(src.p, from_matlab(eng.eval("pys_src__.p")), REL, f"p {label}")
        return
    curved = T.get("Configuration") == "Curved"
    assert set(fields) == ({"u_mask", "ux", "uy"} if curved else {"u_mask", "ux"}), label
    assert src.p_mask is None and src.p is None
    np.testing.assert_array_equal(src.u_mask, from_matlab(eng.eval("pys_src__.u_mask")))
    assert_rel(src.ux, from_matlab(eng.eval("pys_src__.ux")), REL, f"ux {label}")
    if curved:
        assert_rel(src.uy, from_matlab(eng.eval("pys_src__.uy")), REL, f"uy {label}")
    else:
        assert src.uy is None
    if source_type is None:  # Q-150: no staggered-grid shift, mask = sensor mask
        np.testing.assert_array_equal(src.u_mask, sensor.mask)


@pytest.mark.parametrize("source_type", SOURCE_TYPES, ids=["absent", "dipole", "monopole"])
@pytest.mark.parametrize("on_grid", [False, True])
def test_define_source_transducer(
    matlab_engine: Any, tcase: tuple, on_grid: bool, source_type: str | None
) -> None:
    run_case(matlab_engine, tcase, on_grid, source_type, "pulse", apodize=False)


@pytest.mark.parametrize("seq", acoustic.SEQ_PULSES)
def test_define_source_transducer_seq_pulse(matlab_engine: Any, tcase: tuple, seq: str) -> None:
    run_case(matlab_engine, tcase, False, "dipole", seq, apodize=True)
    run_case(matlab_engine, tcase, False, "monopole", seq, apodize=False)


def test_seq_pulse_semantics(matlab_engine: Any, tcase: tuple) -> None:
    """'odd'/'even' split the elements (1-based odd/even fire), sum to 'all'; 'minus' = -'plus'."""
    S, g, T = tcase
    T = dict(T, OnGrid=False, SourceType="monopole")
    sensor, w = acoustic.define_sensor_transducer(T, g)
    idx = np.flatnonzero(sensor.mask.ravel(order="F"))
    out = {}
    for seq in ("odd", "even", "all", "plus", "minus"):
        Tx = S.Transmit.copy_struct()
        Tx.SeqPulse = seq
        out[seq] = acoustic.define_source_transducer(T, Tx, S.Medium, g, w.T, idx).p
    scale = np.max(np.abs(out["all"]))
    np.testing.assert_allclose(out["odd"] + out["even"], out["all"], atol=1e-12 * scale)
    np.testing.assert_array_equal(out["minus"], -out["plus"])
    assert not np.allclose(out["odd"], out["even"])
    Tx = S.Transmit.copy_struct()
    Tx.SeqPulse = "bogus"
    with pytest.raises(ValueError, match="SeqPulse"):
        acoustic.define_source_transducer(T, Tx, S.Medium, g, w.T, idx)


def test_pulsing_sequence() -> None:
    assert acoustic.pulsing_sequence("Amplitude modulation") == ("odd", "even", "all")
    assert acoustic.pulsing_sequence("Pulse inversion") == ("plus", "minus")
    assert acoustic.pulsing_sequence("Standard") == ("pulse",)
    assert acoustic.pulsing_sequence("Amplitude modulation with pulse inversion") == (
        "odd",
        "even",
        "minus",
    )
    with pytest.raises(ValueError, match="PulsingScheme"):
        acoustic.pulsing_sequence("Harmonic")
