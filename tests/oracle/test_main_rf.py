"""``main_RF`` end to end vs MATLAB PROTEUS (M3 "Done when": hybrid and full paths, CPU).

MATLAB runs ``main_RF.m`` unchanged (oracle 8d37938, k-Wave MATLAB 1.4.1,
``kspaceFirstOrder3DC`` on the D-10 binary) on settings and a ground truth it
writes itself (``tests/oracle/main_rf_case.py``); pyproteus runs
:func:`pyproteus.acoustic.main_RF` on the same files with ``MatlabRNG(seed)``
for the medium draws and reads its HDF5 output (D-25).

Class D-24: end-to-end RF ≤ 1e-3 × max|RF|. The element RF is dominated by the
transducer's own transmit (~1e3 × the bubble echoes), so the bubble part is
checked separately against the same class: ``RF − RF_transmit`` of each
implementation (MATLAB: ``shims/transmit_only_rf.m``), relative to its own
maximum. The bubble part inherits the D-22 bubble class (independent ODE
solvers, B-130: 2.1–2.4e-4 in M2), so the comparison uses
``step_control="shared"``; pyproteus's default ``"independent"`` is asserted in
the same class.

Cases: *tiny* (4 elements, PPW 3, 5 bubbles, 2 frames, 1 interaction,
``Standard``) and *small* (8 elements, PPW 4, 12 bubbles, 2 frames, 2
interactions, ``Amplitude modulation``: three pulses incl. the odd/even
element masks), each on the hybrid and the full path; full path with the
strict self-sense (MATLAB) and the default impulse-response mode (D-27).
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pyproteus.io.frames import RFFile
from pyproteus.numerics import MatlabRNG
from tests.oracle.main_rf_case import (
    Case,
    bubble_errors,
    build,
    run_matlab_main_rf,
    run_matlab_transmit_only,
    total_errors,
    transmit_only_rf,
    write_local_paths,
)

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))
try:  # conda's shared libraries before matlab.engine (B-1, B-11)
    from tracer_bullet import kwave_tracer as kt

    kt.preload_before_engine()
    HAVE_KWAVE = True
except ImportError:
    HAVE_KWAVE = False

RF_REL = 1e-3  # D-24

TINY = Case(name="tiny")
SMALL = Case(
    name="small", n_el=8, ppw=4.0, n_interactions=2, n_mb=12, element_height=2e-3,
    scheme="Amplitude modulation",
)  # fmt: skip
CASES = {
    "tiny-hybrid": replace(TINY, name="tiny_hybrid", hybrid=True),
    "tiny-full": replace(TINY, name="tiny_full", hybrid=False),
    "small-hybrid": replace(SMALL, name="small_hybrid", hybrid=True),
    "small-full": replace(SMALL, name="small_full", hybrid=False),
}


@pytest.fixture(scope="module")
def keng(matlab_engine: Any, oracle: Any) -> Any:
    if not HAVE_KWAVE:
        pytest.skip("k-wave-python needed")
    if not (kt.KWAVE_MATLAB / "kspaceFirstOrder3DC.m").is_file():
        pytest.skip(f"MATLAB k-Wave not found at {kt.KWAVE_MATLAB} (set KWAVE_MATLAB_PATH)")
    if not kt.binary_available():
        pytest.skip("k-Wave binary does not start: run tools/tracer_bullet/setup_libs.sh")
    kt.prepare_engine(matlab_engine)
    yield matlab_engine
    matlab_engine.rmpath(str(kt.KWAVE_MATLAB), nargout=0)


@pytest.fixture(scope="module")
def runs(keng: Any, oracle: Any, tmp_path_factory: pytest.TempPathFactory) -> dict:
    """MATLAB results per case, computed once: (root, case, frames, transmit-only RF)."""
    root = tmp_path_factory.mktemp("main_rf")
    lp = write_local_paths(root, kt.KWAVE_MATLAB, kt.WRAPPER_DIR, oracle.path)
    cache: dict[str, Any] = {}

    def get(key: str) -> tuple:
        if key not in cache:
            case = CASES[key]
            settings_file, gt = build(keng, oracle.path, root, case)
            ref = run_matlab_main_rf(
                keng, settings_file, gt, root / "results" / case.name, lp, case.seed
            )
            t_ref = run_matlab_transmit_only(keng, settings_file, lp, case.seed)
            t_py = transmit_only_rf(
                settings_file, gt, binary_path=kt.WRAPPER, binary_env=kt.binary_env(),
                seed=case.seed,
            )  # fmt: skip
            cache[key] = (root, case, settings_file, gt, ref, t_ref, t_py)
        return cache[key]

    return get  # type: ignore[return-value]


def run_python(root: Path, case: Case, settings_file: Path, gt: Path, tag: str, **kw) -> list:
    from pyproteus.acoustic import main_RF

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = main_RF(
            settings_file, gt, root / f"py_{case.name}_{tag}", rng=MatlabRNG(case.seed),
            binary_path=kt.WRAPPER, binary_env=kt.binary_env(), verbose=False, **kw,
        )  # fmt: skip
    return list(RFFile(out))


def check(ref: list, py: list, t_ref: list, t_py: list, case: Case) -> tuple[float, float]:
    assert len(ref) == len(py) == case.n_frames
    for r, q in zip(ref, py, strict=True):
        assert r.frame == q.frame
        assert len(r.rf) == len(q.rf) == case.n_seq
        assert q.dt == pytest.approx(r.dt, rel=1e-12)
        for a, b in zip(r.rf, q.rf, strict=True):
            assert np.shape(a) == np.shape(b)
            assert np.asarray(b).dtype == np.float32  # DATA_CAST_RF 'single'
        assert [np.asarray(b["radii"]).size for b in q.bubbles] == [
            np.asarray(b["radii"]).size for b in r.bubbles
        ]
    # the transmit-only references agree at the k-Wave class
    t_err = max(np.abs(a - b).max() / np.abs(a).max() for a, b in zip(t_ref, t_py, strict=True))
    assert t_err <= 1e-5
    tot = float(total_errors(ref, py).max())
    bub = float(bubble_errors(ref, py, t_py, t_ref).max())
    assert tot <= RF_REL
    assert bub <= RF_REL
    return tot, bub


@pytest.mark.slow
@pytest.mark.parametrize("key", ["tiny-hybrid", "small-hybrid"])
@pytest.mark.parametrize("step_control", ["shared", "independent"])
def test_main_rf_hybrid(runs: Any, key: str, step_control: str) -> None:
    root, case, sfile, gt, ref, t_ref, t_py = runs(key)
    py = run_python(root, case, sfile, gt, step_control, step_control=step_control)
    tot, bub = check(ref, py, t_ref, t_py, case)
    print(f"{key} {step_control}: total {tot:.2e}, bubble part {bub:.2e}")


@pytest.mark.slow
@pytest.mark.parametrize("key", ["tiny-full", "small-full"])
@pytest.mark.parametrize("self_sense", ["strict", "impulse_response"])
def test_main_rf_full(runs: Any, key: str, self_sense: str) -> None:
    root, case, sfile, gt, ref, t_ref, t_py = runs(key)
    py = run_python(
        root,
        case,
        sfile,
        gt,
        self_sense,
        step_control="shared",
        self_sense=self_sense,
        self_sense_jobs=4,
    )
    tot, bub = check(ref, py, t_ref, t_py, case)
    print(f"{key} {self_sense}: total {tot:.2e}, bubble part {bub:.2e}")
