"""RF impact of the opt-in Green's-function self-sense (D-27) against the default.

Full path, pyproteus only (its full path matches MATLAB main_RF, see
tests/oracle/test_main_rf.py). For each medium (Inhomogeneity 0.02, the GUI
default, and 0) runs main_RF with self_sense = "strict" (MATLAB), the default
"impulse_response" and "green"; reports, per mode, the bubble part of the RF
(RF minus the transmit-only RF) relative to strict, and the wall time.

Usage: PYTHONPATH=src:.:tools python tools/m3/green_self_sense.py /tmp/claude/m3/green
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
from tests.oracle.main_rf_case import Case, build, transmit_only_rf

from pyproteus.acoustic import main_RF
from pyproteus.io.frames import RFFile
from pyproteus.numerics import MatlabRNG

BASE = Case(
    name="green",
    hybrid=False,
    n_el=8,
    ppw=6.0,
    n_interactions=2,
    n_mb=8,
    n_frames=1,
    element_height=1.5e-3,
)


def main(root: Path) -> dict:
    from m3.run_case import ORACLE, engine
    from tracer_bullet import kwave_tracer as kt

    root.mkdir(parents=True, exist_ok=True)
    eng = engine()
    out = {}
    for inh in (0.02, 0.0):
        case = replace(BASE, name=f"green_inh{inh:g}", inhomogeneity=inh)
        sfile, gt = build(eng, ORACLE, root, case)
        T = transmit_only_rf(
            sfile, gt, binary_path=kt.WRAPPER, binary_env=kt.binary_env(), seed=case.seed
        )
        rf, times = {}, {}
        for mode in ("strict", "impulse_response", "green"):
            t0 = time.time()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                f = main_RF(
                    sfile,
                    gt,
                    root / f"{case.name}_{mode}",
                    rng=MatlabRNG(case.seed),
                    binary_path=kt.WRAPPER,
                    binary_env=kt.binary_env(),
                    verbose=False,
                    self_sense=mode,
                )
            times[mode] = time.time() - t0
            rf[mode] = np.asarray(next(iter(RFFile(f))).rf[0], dtype=float) - T[0]
        ref = rf["strict"]
        res = {m: float(np.abs(rf[m] - ref).max() / np.abs(ref).max()) for m in rf}
        out[f"inhomogeneity {inh:g}"] = {"rel_diff_vs_strict": res, "seconds": times}
        print(inh, res, {k: f"{v:.1f}" for k, v in times.items()})
    (root / "green_self_sense.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main(Path(sys.argv[1]))
