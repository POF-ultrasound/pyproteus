"""Q-100 investigation driver: run ``q100_point_scatterer.m`` through the MATLAB Engine.

Cases (as run on 2026-11-24, results in B-101): hybrid and full paths at 5 and
7 mm (``geometry``), and a hybrid sweep across one grid cell at 5 mm
(``sweep``: depths 5 mm + k*dx/6, k = 0..5). Every geometry runs with bubble
radius 0.5 µm (``<name>_r05``) and 1 µm (``<name>_r10``); ``q100_analyse.py``
pairs them by that suffix. Element height 1 mm, no elevation focus,
Transmit/Transducer 250 MHz, ppw 6.

Usage (sandbox off: MATLAB Engine)::

    python tools/investigations/q100_run.py <work_dir> geometry|sweep

``<work_dir>`` gets a copy of the oracle (never run on the oracle itself) and
the outputs (``<work_dir>/out_<set>``). Needs the k-Wave libs from
``tools/tracer_bullet/setup_libs.sh``. Investigation only.
"""

from __future__ import annotations

import os
import shutil
import ssl  # noqa: F401  # B-1: before matlab.engine
import sys
import xml.parsers.expat  # noqa: F401  # B-11
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ORACLE = Path(
    os.environ.get(
        "PYPROTEUS_ORACLE",
        "/Users/theodoremichel/Repos/TWENTE/Ultrasound/PROTEUS-cleaning/PROTEUS-oracle",
    )
)
KWAVE = os.environ.get(
    "KWAVE_MATLAB_PATH",
    "/Users/theodoremichel/Repos/TWENTE/Ultrasound/PROTEUS-cleaning/PROTEUS/external/k-wave/k-Wave",
)
DX = 0.00010266666666666666  # grid size of these settings (ppw 6, 2.5 MHz)


def cases(which: str) -> list[dict]:
    out = []
    if which == "geometry":
        geoms = [
            (f"{m}_d{d[0]}", float(d), h)
            for m, h in (("hyb", True), ("full", False))
            for d in ("5e-3", "7e-3")
        ]
    elif which == "sweep":
        geoms = [(f"sw{k}", 5e-3 + k * DX / 6, True) for k in range(6)]
    else:
        raise SystemExit("set must be 'geometry' or 'sweep'")
    for name, depth, hybrid in geoms:
        for radius, tag in ((0.5e-6, "r05"), (1.0e-6, "r10")):
            case = dict(name=f"{name}_{tag}", depth=depth, radius=radius, hybrid=hybrid)
            case.update(fs_transmit=250e6, ppw=6.0, height=1e-3, efocus=float("inf"))
            out.append(case)
    return out


def main(work: Path, which: str) -> None:
    import matlab.engine

    # Never write into the oracle or the PROTEUS checkout (k-Wave lives in it).
    protected = [ORACLE.resolve(), Path(KWAVE).resolve().parents[2]]
    for root in protected:
        if work == root or work.is_relative_to(root):
            raise SystemExit(f"work_dir {work} is inside {root}: use a temporary folder")

    copy = work / "proteus"
    if not copy.exists():
        shutil.copytree(ORACLE, copy, ignore=shutil.ignore_patterns(".git"))
    out = work / f"out_{which}"
    out.mkdir(parents=True, exist_ok=True)
    eng = matlab.engine.start_matlab("-nodesktop -nosplash")
    try:
        eng.addpath(str(HERE), nargout=0)
        eng.cd(str(work), nargout=0)
        eng.setenv("KWAVE_LIBS", str(REPO / "tools/tracer_bullet/.kwave-libs/lib"), nargout=0)
        eng.workspace["pyp_cases"] = [
            {k: (float(v) if isinstance(v, int | float) and not isinstance(v, bool) else v)
             for k, v in c.items()}
            for c in cases(which)
        ]  # fmt: skip
        eng.eval(
            "cases = [pyp_cases{:}]; "
            f"q100_point_scatterer('{copy}', '{KWAVE}', '{REPO / 'tools/tracer_bullet/bin'}', "
            f"cases, '{out}');",
            nargout=0,
        )
    finally:
        eng.quit()


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), sys.argv[2])
