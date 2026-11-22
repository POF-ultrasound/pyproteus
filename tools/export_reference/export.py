"""Export MATLAB PROTEUS reference data to HDF5 (run through the MATLAB Engine).

    ~/miniconda3/envs/pyproteus/bin/python -m tools.export_reference.export --list
    ~/miniconda3/envs/pyproteus/bin/python -m tools.export_reference.export  # quick, no k-Wave
    ~/miniconda3/envs/pyproteus/bin/python -m tools.export_reference.export full_run.tiny
    ~/miniconda3/envs/pyproteus/bin/python -m tools.export_reference.export --all

Run from the repository root, outside any sandbox that blocks the MATLAB
Engine. Output: ``tools/export_reference/data/`` (gitignored) or ``--out``.
Then ``python -m tools.export_reference.update_registry`` records the hashes.
Configuration (oracle, k-Wave, binary): see ``ExportConfig.from_env``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from pyproteus.io.reference_format import write_reference

from .engine import KWAVE_SOLVER, ExportConfig, ExportSession
from .specs import SPECS, ExportSpec, M

DEFAULT_OUT = Path(__file__).resolve().parent / "data"


def export_spec(session: ExportSession, spec: ExportSpec, out_dir: Path) -> Path:
    """Run one spec in an open session and write ``<out_dir>/<spec.name>.h5``."""
    eng = session.eng
    eng.eval("clear pyp_in pyp_out pyp_extra pyp_tmp__; pyp_in = struct();", nargout=0)
    eng.eval(f"rng({int(spec.seed)}, 'twister');", nargout=0)
    names: list[str] = []
    for name, value in spec.inputs().items():
        if isinstance(value, M):
            eng.eval(f"pyp_in.{name} = {value.expr};", nargout=0)
        else:
            session.put("pyp_tmp__", value)
            eng.eval(f"pyp_in.{name} = pyp_tmp__; clear pyp_tmp__;", nargout=0)
        names.append(name)
    statement = spec.statement(names)
    eng.eval("pyp_out = struct();", nargout=0)
    eng.eval(statement, nargout=0)
    inputs = session.flatten("pyp_in")
    outputs = session.flatten("pyp_out")
    has_extra = bool(eng.eval("exist('pyp_extra','var') == 1", nargout=1))
    extra = session.flatten("pyp_extra") if has_extra else None
    if not isinstance(outputs, dict) or not outputs:
        raise RuntimeError(f"{spec.name}: the call assigned no pyp_out fields")
    if spec.kind == "full_run":
        _check_full_run(spec, inputs)
    if spec.outputs and list(outputs) != list(spec.outputs):
        raise RuntimeError(f"{spec.name}: outputs {list(outputs)} != {list(spec.outputs)}")
    attrs = {
        "name": spec.name,
        "kind": spec.kind,
        "module": spec.module,
        "function": spec.function,
        "description": spec.description,
        "call": statement,
        "seeds": spec.all_seeds(),
        **session.provenance(),
    }
    eng.eval("clear pyp_in pyp_out pyp_extra;", nargout=0)
    return write_reference(out_dir / f"{spec.name}.h5", attrs, inputs, outputs, extra)


def _check_full_run(spec: ExportSpec, inputs: dict) -> None:
    """Provenance must describe what ran: solver from the settings, seeds from the inputs."""
    solver = inputs["settings"]["SimulationParameters"]["Solver"]
    if solver != "3DC":
        raise RuntimeError(
            f"{spec.name}: settings use solver {solver!r}; only '3DC' (the recorded C++ "
            f"binary, {KWAVE_SOLVER}) is supported by the exporter"
        )
    seeds = spec.all_seeds()
    for key, inp in (
        ("matlab_rng_twister_before_main_RF", "rng_seed"),
        ("ground_truth_mt19937ar", "ground_truth_seed"),
    ):
        if float(inputs[inp][0, 0]) != seeds[key]:
            raise RuntimeError(f"{spec.name}: input {inp} != recorded seed {key}")


def select(names: list[str], *, all_: bool) -> list[ExportSpec]:
    if names:
        unknown = [n for n in names if n not in SPECS]
        if unknown:
            raise SystemExit(f"unknown spec(s): {unknown}; see --list")
        return [SPECS[n] for n in names]
    # Default: quick specs that need no k-Wave binary. Full runs: by name or --all.
    return [s for s in SPECS.values() if all_ or not (s.slow or s.needs_kwave)]


def run(specs: list[ExportSpec], out_dir: Path, cfg: ExportConfig) -> list[Path]:
    written: list[Path] = []
    kwave = any(s.needs_kwave for s in specs)
    with ExportSession(cfg, kwave=kwave) as session:
        for spec in specs:
            t0 = time.perf_counter()
            path = export_spec(session, spec, out_dir)
            print(f"{spec.name}: {path} ({time.perf_counter() - t0:.1f} s)")
            written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("specs", nargs="*", help="spec names (default: quick specs without k-Wave)")
    p.add_argument("--all", action="store_true", help="every spec, incl. slow and k-Wave ones")
    p.add_argument("--list", action="store_true", help="list specs and exit")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output folder")
    p.add_argument("--kwave-binary-dir", type=Path, help="k-Wave BinaryPath folder")
    p.add_argument("--kwave-matlab-path", type=Path, help="MATLAB k-Wave 1.4.x folder")
    p.add_argument("--workdir", type=Path, help="keep scratch files here (default: temporary)")
    args = p.parse_args(argv)
    if args.list:
        for s in SPECS.values():
            flags = ", ".join(f for f, on in (("k-Wave", s.needs_kwave), ("slow", s.slow)) if on)
            print(f"{s.name:40s} {s.kind:9s} {flags:14s} {s.description}")
        return 0
    cfg = ExportConfig.from_env()
    if args.kwave_binary_dir:
        cfg.kwave_binary_dir = args.kwave_binary_dir
    if args.kwave_matlab_path:
        cfg.kwave_matlab_path = args.kwave_matlab_path
    if args.workdir:
        cfg.workdir = args.workdir
    out = args.out.resolve()
    if out.is_relative_to(cfg.oracle.resolve()):
        raise SystemExit("never write exports into the oracle checkout")
    run(select(args.specs, all_=args.all), out, cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
