"""Export specs: which oracle function to call, on which inputs.

A later milestone session adds references by appending specs here (or in a
module imported at the bottom), nothing else. One spec = one HDF5 file named
``<name>.h5`` (flat: Zenodo records have no folders).

Inputs are built in call order, each either a Python value (sent through
``tests/oracle/helpers.to_matlab``: wrap 1-D arrays in ``row``/``col``) or a
:class:`M` MATLAB expression evaluated in the engine. They land in the MATLAB
struct ``pyp_in``; the call assigns fields of ``pyp_out`` (and optionally
``pyp_extra``). By default the call is ``[pyp_out.<o1>, ...] = <function>(pyp_in.<i1>, ...)``.
Before the inputs are built, ``rng(seed, 'twister')`` is set.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_FILE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class M:
    """A MATLAB expression evaluated in the engine, e.g. ``M("reset_transducer()")``."""

    expr: str


@dataclass(frozen=True)
class ExportSpec:
    name: str  # file stem, e.g. "GUIfunctions.reset_transmit"
    module: str  # PROTEUS folder of the function
    function: str  # MATLAB function under test
    description: str
    inputs: Callable[[], Mapping[str, Any]] = dict
    outputs: tuple[str, ...] = ()
    call: str | None = None  # custom MATLAB statement (must assign pyp_out)
    kind: str = "module"  # "module" or "full_run"
    seed: int = 0  # rng(seed,'twister') before building the inputs
    seeds: Mapping[str, int] = field(default_factory=dict)  # further seeds to record
    needs_kwave: bool = False
    slow: bool = False  # skipped unless asked for by name or with --all

    def __post_init__(self) -> None:
        if not _FILE.match(self.name) or self.name.endswith(".h5"):
            raise ValueError(f"spec name {self.name!r} must be a flat file stem")
        if not _NAME.match(self.function):
            raise ValueError(f"bad MATLAB function name {self.function!r}")
        if self.call is None and not self.outputs:
            raise ValueError(f"{self.name}: give outputs (or a custom call)")
        for o in self.outputs:
            if not _NAME.match(o):
                raise ValueError(f"{self.name}: bad output name {o!r}")
        if self.kind not in ("module", "full_run"):
            raise ValueError(f"{self.name}: kind must be 'module' or 'full_run'")
        if self.kind == "full_run" and not self.needs_kwave:
            raise ValueError(f"{self.name}: full runs use the k-Wave binary (needs_kwave)")

    def statement(self, input_names: list[str]) -> str:
        if self.call is not None:
            return self.call
        lhs = ", ".join(f"pyp_out.{o}" for o in self.outputs)
        args = ", ".join(f"pyp_in.{n}" for n in input_names)
        return f"[{lhs}] = {self.function}({args});"

    def all_seeds(self) -> dict[str, int]:
        return {"matlab_rng_twister": self.seed, **dict(self.seeds)}


SPECS: dict[str, ExportSpec] = {}


def register(spec: ExportSpec) -> ExportSpec:
    if spec.name in SPECS:
        raise ValueError(f"duplicate export spec {spec.name!r}")
    SPECS[spec.name] = spec
    return spec


# ----------------------------------------------------------------------------- M0 examples
# reset_* defaults: nothing ported yet, so they exercise the export path end to end
# (scalar struct of doubles, chars, logicals, Inf, row vectors from get_*_signal).

register(
    ExportSpec(
        name="GUIfunctions.reset_acquisition",
        module="GUIfunctions",
        function="reset_acquisition",
        description="Default Acquisition settings (no inputs).",
        outputs=("Acquisition",),
    )
)

register(
    ExportSpec(
        name="GUIfunctions.reset_transmit",
        module="GUIfunctions",
        function="reset_transmit",
        description=(
            "Default Transmit settings for the default transducer (reset_transducer()); "
            "includes the voltage and pressure signals."
        ),
        inputs=lambda: {"Transducer": M("reset_transducer()")},
        outputs=("Transmit",),
    )
)

# ----------------------------------------------------------------------------- full runs
# main_RF on the real k-Wave C++ binary (kspaceFirstOrder3DC; D-10: k-wave-python's
# v1.4.1 OMP binary on macOS arm64). Case settings: matlab/pyp_full_run_settings.m.

_FULL_RUN_CALL = (
    "[pyp_out, pyp_extra, pyp_in.settings, pyp_in.ground_truth] = "
    "pyp_export_full_run(pyp_in.case_name, pyp_in.ground_truth_seed, pyp_in.rng_seed);"
)


FULL_RUN_GROUND_TRUTH_SEED = 1  # RandStream('mt19937ar') of write_synthetic_ground_truth
FULL_RUN_RNG_SEED = 0  # rng(seed,'twister') right before main_RF (random medium background)


def _full_run(case: str, description: str, *, slow: bool) -> ExportSpec:
    return ExportSpec(
        name=f"full_run.{case}",
        module="acoustic-module",
        function="main_RF",
        description=description,
        inputs=lambda: {
            "case_name": case,
            "ground_truth_seed": float(FULL_RUN_GROUND_TRUTH_SEED),
            "rng_seed": float(FULL_RUN_RNG_SEED),
        },
        call=_FULL_RUN_CALL,
        kind="full_run",
        seeds={
            "matlab_rng_twister_before_main_RF": FULL_RUN_RNG_SEED,
            "ground_truth_mt19937ar": FULL_RUN_GROUND_TRUTH_SEED,
        },
        needs_kwave=True,
        slow=slow,
    )


register(
    _full_run(
        "tiny",
        "Hybrid main_RF, PROTEUS regression case rf_hybrid_standard (tiny_box, 8 elements, "
        "4 bubbles, 2 frames).",
        slow=False,
    )
)
register(
    _full_run(
        "small",
        "Hybrid main_RF, pulse inversion (tiny_box, 16 elements, 6 ppw, 12 bubbles, 3 frames).",
        slow=True,
    )
)
register(
    _full_run(
        "medium",
        "Hybrid main_RF, default settings on renal_tree, 2 frames (GPU workstation).",
        slow=True,
    )
)
