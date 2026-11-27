"""Run one tiny main_RF case in MATLAB (shared engine) and pyproteus; print RF errors.

Usage: PYTHONPATH=src:. python tools/m3/run_case.py <root> [hybrid|full] [key=value ...]
Connects to the shared MATLAB session named by $PYPROTEUS_MATLAB_SESSION (default 'pyp').
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracer_bullet import kwave_tracer as kt

kt.preload_before_engine()
import os  # noqa: E402

import matlab.engine  # noqa: E402
import numpy as np  # noqa: E402
from tests.oracle.main_rf_case import (  # noqa: E402
    Case,
    bubble_errors,
    build,
    run_matlab_main_rf,
    run_matlab_transmit_only,
    total_errors,
    transmit_only_rf,
    write_local_paths,
)

from pyproteus.acoustic import main_RF  # noqa: E402
from pyproteus.io.frames import RFFile  # noqa: E402
from pyproteus.numerics import MatlabRNG  # noqa: E402

ORACLE = Path("/Users/theodoremichel/Repos/TWENTE/Ultrasound/PROTEUS-cleaning/PROTEUS-oracle")
MODS = (
    "GUIfunctions",
    "acoustic-module",
    "microbubble-simulator",
    "streamline-module",
    "delay-and-sum",
    "mesh-voxelisation",
    "time-estimation",
)


def engine():
    eng = matlab.engine.connect_matlab(os.environ.get("PYPROTEUS_MATLAB_SESSION", "pyp"))
    for m in MODS:
        eng.addpath(eng.genpath(str(ORACLE / m)), nargout=0)
    eng.addpath(str(Path(__file__).resolve().parents[2] / "tests" / "oracle" / "shims"), nargout=0)
    kt.prepare_engine(eng)
    return eng


def run(root: Path, case: Case, *, matlab=True, **py_kw):
    eng = engine() if matlab else None
    root.mkdir(parents=True, exist_ok=True)
    if matlab:
        settings_file, gt = build(eng, ORACLE, root, case)
        lp = write_local_paths(root, kt.KWAVE_MATLAB, kt.WRAPPER_DIR, ORACLE)
        t = time.time()
        ref = run_matlab_main_rf(
            eng, settings_file, gt, root / "results" / f"mat_{case.name}", lp, case.seed
        )
        print(f"MATLAB {time.time() - t:.1f}s")
    else:
        settings_file, gt, ref = root / f"{case.name}.mat", root / "gt" / case.name, None
    t = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = main_RF(
            settings_file,
            gt,
            root / f"py_{case.name}",
            rng=MatlabRNG(case.seed),
            binary_path=kt.WRAPPER,
            binary_env=kt.binary_env(),
            verbose=False,
            **py_kw,
        )
    print(f"pyproteus {time.time() - t:.1f}s")
    py = list(RFFile(out))
    return ref, py, out


if __name__ == "__main__":
    root = Path(sys.argv[1])
    kw = dict(a.split("=") for a in sys.argv[3:])
    step = kw.pop("step_control", "shared")
    ss = kw.pop("self_sense", "strict")
    use_matlab = kw.pop("matlab", "1") == "1"
    conv = {
        "n_el": int,
        "n_interactions": int,
        "n_frames": int,
        "n_mb": int,
        "ppw": float,
        "inhomogeneity": float,
        "scheme": str,
        "element_height": float,
    }
    case = Case(
        name=sys.argv[2],
        hybrid=sys.argv[2].startswith("hyb"),
        **{k: conv[k](v) for k, v in kw.items()},
    )
    ref, py, out = run(root, case, matlab=use_matlab, step_control=step, self_sense=ss)
    if ref is None:
        from pyproteus.io.frames import read_matlab_rf_frame

        ref = [
            read_matlab_rf_frame(f)
            for f in sorted((root / "results" / f"mat_{case.name}").glob("Frame_*.mat"))
        ]
    T = transmit_only_rf(
        root / f"{case.name}.mat",
        root / "gt" / case.name,
        binary_path=kt.WRAPPER,
        binary_env=kt.binary_env(),
        seed=case.seed,
    )
    Tm = run_matlab_transmit_only(
        engine(), root / f"{case.name}.mat", root / "proteus_local_paths.m", case.seed
    )
    print(
        "transmit-only RF error:",
        max(np.abs(a - b).max() / np.abs(a).max() for a, b in zip(Tm, T, strict=True)),
    )
    print("total errors (frame x pulse):", total_errors(ref, py).ravel())
    print("bubble-part errors (frame x pulse):", bubble_errors(ref, py, T, Tm).ravel())
    print(
        "max|T| / max|bubble part|:",
        max(np.abs(t).max() for t in T)
        / min(np.abs(np.asarray(q.rf[0], float) - T[0]).max() for q in py),
    )
