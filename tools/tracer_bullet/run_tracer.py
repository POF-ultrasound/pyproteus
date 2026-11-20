"""Run the k-Wave tracer bullet and write a reference file plus a short report.

    ~/miniconda3/envs/pyproteus/bin/python tools/tracer_bullet/run_tracer.py

Needs MATLAB R2025a + Engine, MATLAB k-Wave 1.4.x (``KWAVE_MATLAB_PATH``),
k-wave-python 0.6.2 and the binary's runtime libraries (``setup_libs.sh``).
Writes ``tools/tracer_bullet/data/`` (gitignored): k-Wave HDF5 input/output
files of every run and ``tracer_reference.h5`` with the case inputs and the
MATLAB sensor data (rows in find order).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # tools/

from tracer_bullet import cases
from tracer_bullet import kwave_tracer as kt

ORACLE = Path("/Users/theodoremichel/Repos/TWENTE/Ultrasound/PROTEUS-cleaning/PROTEUS-oracle")


def main() -> int:
    kt.preload_before_engine()
    import h5py
    import kwave

    data = kt.HERE / "data"
    data.mkdir(exist_ok=True)
    eng = kt.start_engine()
    kt.check_engine_layout(eng)
    kwave_matlab_version = (
        (kt.KWAVE_MATLAB / "Contents.m").read_text().splitlines()[1].lstrip("% ").strip()
    )
    oracle_commit = subprocess.run(
        ["git", "-C", str(ORACLE), "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()

    lines = []
    with h5py.File(data / "tracer_reference.h5", "w") as ref:
        ref.attrs["kwave_python_version"] = kwave.__version__
        ref.attrs["kwave_binary_version"] = kt.kwave_binary_version()
        ref.attrs["kwave_matlab_version"] = kwave_matlab_version
        ref.attrs["proteus_oracle_commit"] = oracle_commit
        ref.attrs["note"] = (
            "arrays [ix,iy,iz] stored as h5py C-order (x slowest); rows in MATLAB find order"
        )
        for name, make in cases.CASES.items():
            c = make()
            pm = kt.run_matlab(eng, c, data)
            g = ref.create_group(name)
            for k, v in {**c.medium, **c.source}.items():
                g.create_dataset(k, data=np.asarray(v))
            g.create_dataset("sensor_mask", data=c.sensor_mask)
            g.create_dataset("p_matlab", data=pm.astype(np.float32))
            res = {
                "kwp cpp (adapter)": kt.rel_err(kt.run_kwp(c, data, "cpp"), pm),
                "kwp python (adapter)": kt.rel_err(kt.run_kwp(c, None, "python"), pm),
                "kwp cpp, C-order source rows": kt.rel_err(
                    kt.run_kwp(c, data, "cpp", raw_rows="c"), pm
                ),
            }
            try:
                res["kwp legacy kspaceFirstOrder3D"] = kt.rel_err(kt.run_kwp_legacy(c, data), pm)
            except Exception as e:
                res["kwp legacy kspaceFirstOrder3D"] = f"{type(e).__name__}: {e}"
            diff = kt.compare_inputs(
                data / f"matlab_{name}_input.h5", data / f"kwp_cpp_{name}" / "kwave_input.h5"
            )
            lines.append(f"## {name}: {c.info}")
            lines += [
                f"- {k}: " + (f"max rel err {v:.3e}" if isinstance(v, float) else v)
                for k, v in res.items()
            ]
            lines += [f"- input {k}: {v}" for k, v in diff.items() if v != "equal"]
    eng.quit()
    report = "\n".join(lines)
    (data / "report.md").write_text(report + "\n")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
