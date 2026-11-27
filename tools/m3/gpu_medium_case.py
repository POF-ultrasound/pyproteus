"""M3 "Medium case on GPU": main_RF on the CUDA k-Wave binary + CuPy vs the CPU run.

Three steps; only ``run --device gpu`` needs the cluster (D-16).

1. ``prepare`` (local, needs the MATLAB shared session 'pyp' or a MATLAB Engine):
   writes the medium-case settings (``medium.mat``) and a MATLAB ground truth
   (``gt/medium``) into FOLDER. Copy FOLDER to the cluster.
2. ``run --device cpu|gpu [--path hybrid|full]``: pyproteus ``main_RF`` on
   FOLDER, Solver ``3DC`` (OMP binary, NumPy propagation) or ``3DG`` (CUDA
   binary, CuPy propagation on ``--device-number``); output
   ``FOLDER/<device>_<path>/rf.h5`` and the wall time in ``timing.json``.
   ``--gridded off`` (default) runs both devices with exact receive distances
   (the D-30 GPU default), so the comparison sees only CPU/GPU differences;
   ``--gridded on`` adds a gridded run (``<device>_<path>_on``) for the D-30 timing.
3. ``compare``: RF GPU vs CPU per frame and pulse, relative to max|RF| and to
   the bubble part. Pass: ≤ 1e-5 (k-Wave class: FFTW vs cuFFT in single
   precision dominate; propagation and RF alone are at 1e-6, D-13/M2).

Medium case: P4-1 with 32 elements (9.4 mm), PPW 6, bounding box
10 × 6 × 2 mm from 10 mm depth, 200 bubbles, 2 frames, 2 interactions,
Standard pulsing, GUI-default medium (Inhomogeneity 0.02, BonA 6, 0.75 dB/cm/MHz^1.5).

Cluster commands (from the repository root, env with pyproteus[dev] + cupy):

    python tools/m3/gpu_medium_case.py run FOLDER --device gpu --device-number 0
    python tools/m3/gpu_medium_case.py run FOLDER --device cpu  # or copy cpu_hybrid/
    python tools/m3/gpu_medium_case.py compare FOLDER
    python tools/m3/gpu_medium_case.py run FOLDER --device gpu --gridded on   # D-30 timing
    python tools/m3/gpu_medium_case.py run FOLDER --device gpu --path full   # optional, slow
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

MEDIUM = dict(
    name="medium",
    n_el=32,
    element_height=4e-3,
    ppw=6.0,
    start_depth=10e-3,
    diagonal=(10e-3, 6e-3, 2e-3),
    n_interactions=2,
    n_mb=200,
    n_frames=2,
)


def prepare(folder: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from tests.oracle.main_rf_case import Case, matlab_settings, write_ground_truth

    from m3.run_case import ORACLE, engine

    folder.mkdir(parents=True, exist_ok=True)
    eng = engine()
    case = Case(**MEDIUM)
    matlab_settings(eng, ORACLE, case)
    eng.eval(f"save('{folder / 'medium.mat'}', '-struct', 'S');", nargout=0)
    write_ground_truth(eng, folder / "gt" / "medium", case)
    eng.eval("clear S", nargout=0)
    (folder / "case.json").write_text(json.dumps(MEDIUM, indent=1))
    print(f"prepared {folder}")


def run(
    folder: Path, device: str, path: str, device_number: int, binary: str | None, gridded: str
) -> None:
    from pyproteus.acoustic import main_RF
    from pyproteus.acoustic.setup import load_settings
    from pyproteus.numerics import MatlabRNG

    s = load_settings(folder / "medium.mat")
    s.SimulationParameters.Solver = "3DG" if device == "gpu" else "3DC"
    s.SimulationParameters.HybridSimulation = path == "hybrid"
    kw = {}
    if device == "cpu" and sys.platform == "darwin" and binary is None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from tracer_bullet import kwave_tracer as kt  # D-10 wrapper on the Mac

        binary, kw = str(kt.WRAPPER), {"binary_env": kt.binary_env()}
    out = folder / f"{device}_{path}" if gridded == "off" else folder / f"{device}_{path}_{gridded}"
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main_RF(
            s,
            folder / "gt" / "medium",
            out,
            gridded_receive={"auto": None, "on": True, "off": False}[gridded],
            rng=MatlabRNG(7),
            binary_path=binary,
            device_number=device_number if device == "gpu" else None,
            **kw,
        )
    dt = time.time() - t0
    (out / "timing.json").write_text(json.dumps({"seconds": dt, "device": device, "path": path}))
    print(f"{device} {path}: {dt:.1f} s -> {out / 'rf.h5'}")


def compare(folder: Path, path: str) -> int:
    from pyproteus.io.frames import RFFile

    cpu = list(RFFile(folder / f"cpu_{path}" / "rf.h5"))
    gpu = list(RFFile(folder / f"gpu_{path}" / "rf.h5"))
    worst = 0.0
    for a, b in zip(cpu, gpu, strict=True):
        for k, (x, y) in enumerate(zip(a.rf, b.rf, strict=True)):
            x, y = np.asarray(x, float), np.asarray(y, float)
            e = np.abs(x - y).max() / np.abs(x).max()
            worst = max(worst, e)
            print(f"frame {a.frame} pulse {k + 1}: max|GPU-CPU|/max|CPU| = {e:.2e}")
    for d in ("cpu", "gpu"):
        t = folder / f"{d}_{path}" / "timing.json"
        if t.is_file():
            print(d, json.loads(t.read_text())["seconds"], "s")
    ok = worst <= 1e-5
    print("PASS" if ok else "FAIL", f"(worst {worst:.2e}, class 1e-5)")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("step", choices=["prepare", "run", "compare"])
    ap.add_argument("folder", type=Path)
    ap.add_argument("--device", choices=["cpu", "gpu"], default="gpu")
    ap.add_argument("--path", choices=["hybrid", "full"], default="hybrid")
    ap.add_argument("--device-number", type=int, default=0)
    ap.add_argument(
        "--gridded",
        choices=["off", "on", "auto"],
        default="off",
        help="hybrid receive distances: off = exact (D-30 GPU default, both "
        "devices alike for the comparison), on = gridded (PROTEUS), auto = D-30",
    )
    ap.add_argument("--binary", default=None, help="k-Wave binary (default: k-wave-python's)")
    a = ap.parse_args()
    if a.step == "prepare":
        prepare(a.folder)
    elif a.step == "run":
        run(a.folder, a.device, a.path, a.device_number, a.binary, a.gridded)
    else:
        sys.exit(compare(a.folder, a.path))
