#!/usr/bin/env python
"""Convert the constants shipped with MATLAB PROTEUS into pyproteus package data.

Reads the ``.mat`` files listed in :data:`SOURCES` from the oracle checkout
(PROTEUS commit ``8d37938``, D-7) with :func:`pyproteus.config.matfile.loadmat`
and writes them to ``src/pyproteus/data/``:

* small files → JSON (lossless MATLAB value-tree encoding of
  :mod:`pyproteus.config.json_io`; doubles round-trip bit for bit);
* the complex transfer-function fits (3 × 50 000 values) → compressed ``.npz``.

``MANIFEST.json`` records, per data file, the MATLAB file name PROTEUS loads,
every source path, the oracle commit and the source SHA-256. Run it again only
when the oracle is re-pinned::

    python tools/convert_data.py [--oracle PATH]

``time-estimation/beta_coeff.mat`` is out of parity scope and not converted.
These files come from PROTEUS (MIT licence), so shipping them is fine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from pyproteus.config.json_io import dumps, tree_to_jsonable
from pyproteus.config.matfile import loadmat

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "src" / "pyproteus" / "data"
DEFAULT_ORACLE = Path(
    "/Users/theodoremichel/Repos/TWENTE/Ultrasound/PROTEUS-cleaning/PROTEUS-oracle"
)

#: data file → (name PROTEUS passes to ``load``, source paths in the oracle).
#: Identical copies (same SHA-256) are listed together; the first is read.
SOURCES: dict[str, tuple[str, list[str]]] = {
    "tissue_properties.json": (
        "tissue_properties.mat",
        ["GUIfunctions/tissue_properties.mat"],
    ),
    "fit_SigmaR_evaluated.json": (
        "fit_SigmaR_evaluated.mat",
        [
            "GUIfunctions/fit_SigmaR_evaluated.mat",
            "microbubble-simulator/functions/fit_SigmaR_evaluated.mat",
        ],
    ),
    "fit_SigmaR_evaluated_30PA.json": (
        "fit_SigmaR_evaluated_30PA.mat",
        [
            "microbubble-simulator/functions/fit_SigmaR_evaluated_30PA.mat",
            "example_custom_data/fit_SigmaR_evaluated_30PA.mat",
        ],
    ),
    "fit_SigmaR_evaluated_50PA.json": (
        "fit_SigmaR_evaluated_50PA.mat",
        [
            "microbubble-simulator/functions/fit_SigmaR_evaluated_50PA.mat",
            "example_custom_data/fit_SigmaR_evaluated_50PA.mat",
        ],
    ),
    "fit_SigmaR_04-08-2017.json": (
        "fit_SigmaR_04-08-2017.mat",
        ["microbubble-simulator/functions/fit_SigmaR_04-08-2017.mat"],
    ),
    "TransmitTransferFunctionFit.npz": (
        "TransmitTransferFunctionFit.mat",
        ["microbubble-simulator/functions/TransmitTransferFunctionFit.mat"],
    ),
    "ReceiveTransferFunctionFit.npz": (
        "ReceiveTransferFunctionFit.mat",
        ["microbubble-simulator/functions/ReceiveTransferFunctionFit.mat"],
    ),
    "TwoWayTransferFunctionFit.npz": (
        "TwoWayTransferFunctionFit.mat",
        ["microbubble-simulator/functions/TwoWayTransferFunctionFit.mat"],
    ),
    "geometry_data/renal_tree/GeometryProperties.json": (
        "renal_tree/GeometryProperties.mat",
        ["geometry_data/renal_tree/GeometryProperties.mat"],
    ),
    "geometry_data/mouse_brain/GeometryProperties.json": (
        "mouse_brain/GeometryProperties.mat",
        ["geometry_data/mouse_brain/GeometryProperties.mat"],
    ),
    "Transducer-models/C1-6D.json": (
        "C1-6D.mat",
        ["Transducer-models/C1-6D.mat"],
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_npz(path: Path, variables: dict) -> None:
    for name, v in variables.items():
        if not isinstance(v, np.ndarray) or v.dtype == object:
            raise TypeError(f"{path.name}: variable {name} is not a numeric array")
    np.savez_compressed(path, **variables)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    args = ap.parse_args()
    oracle: Path = args.oracle
    commit = subprocess.run(
        ["git", "-C", str(oracle), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    manifest: dict = {
        "description": "Constants converted from MATLAB PROTEUS by tools/convert_data.py",
        "licence": "MIT (PROTEUS)",
        "oracle_commit": commit,
        "files": {},
    }
    for out_name, (mat_name, sources) in SOURCES.items():
        paths = [oracle / s for s in sources]
        digests = {sha256(p) for p in paths}
        if len(digests) != 1:
            raise SystemExit(f"{out_name}: listed copies differ: {sources}")
        variables = loadmat(paths[0])
        out = DATA / out_name
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix == ".npz":
            write_npz(out, variables)
        else:
            out.write_text(dumps(tree_to_jsonable(variables)) + "\n", encoding="utf-8")
        manifest["files"][out_name] = {
            "matlab_name": mat_name,
            "sources": sources,
            "sha256": digests.pop(),
            "variables": list(variables),
        }
        print(f"{out_name:50s} <- {sources[0]}")
    (DATA / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
