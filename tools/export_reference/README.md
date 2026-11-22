# Reference export (MATLAB PROTEUS → HDF5)

Exports frozen MATLAB PROTEUS results for CI parity tests (ROADMAP §3, §4).
Format: [`docs/REFERENCE_DATA.md`](../../docs/REFERENCE_DATA.md). Loader and
pooch registry: `src/pyproteus/io/reference.py`.

| File | Purpose |
|---|---|
| `specs.py` | registry of export specs (function, input builder, outputs, seeds); **add new references here** |
| `export.py` | CLI: runs specs through the MATLAB Engine, writes `data/<name>.h5` |
| `engine.py` | engine session (oracle path as `tests/oracle/conftest.py`, k-Wave, `PROTEUS_LOCAL_PATHS`), provenance |
| `records.py` | rebuilds MATLAB values (exact class/size) from `matlab/pyp_flatten.m` records |
| `update_registry.py` | writes SHA-256 hashes into `src/pyproteus/io/reference_registry.txt` |
| `matlab/` | MATLAB helpers (pyproteus code, MIT): `pyp_flatten`, full-run settings and driver, tiny geometry |
| `zenodo_metadata.json` | **draft** metadata for the first Zenodo record (nothing created yet) |
| `data/` | output (gitignored) |

## Run

macOS/Linux only (symlinks, shell wrapper). Needs MATLAB R2025a + Engine (`pip install -e ".[dev,oracle]"`), the oracle
checkout at commit `8d37938`, and for full runs MATLAB k-Wave 1.4.x and a
k-Wave C++ binary. The MATLAB Engine does not start inside the Claude Bash
sandbox: run outside it. From the repository root:

```bash
python -m tools.export_reference.export --list
```

```bash
python -m tools.export_reference.export                  # quick module specs (no k-Wave)
```

```bash
python -m tools.export_reference.export full_run.tiny    # by name; --all = everything
```

```bash
python -m tools.export_reference.update_registry          # hash data/*.h5 into the registry
```

`update_registry` refuses files from a dirty or unpinned oracle or a dirty
pyproteus checkout (`--allow-unclean` overrides, for local experiments only)
and keeps entries without a local file unless `--prune` (use it for a full
re-export for a new Zenodo version).

### Status of the full runs (2026-11-22)

| Case | Status |
|---|---|
| `tiny` | exported on the developer's Mac (≈ 5 s); RF matches PROTEUS' `rf_hybrid_standard` baseline to 1.4e-6 rel |
| `small` | exported on the developer's Mac (≈ 45 s) |
| `medium` | **TODO**: needs the renal tree STL (Zenodo 10573662, `PYPROTEUS_GEOMETRIES_PATH`) and a GPU workstation; the exporter only records the `3DC` binary, so a `3DG` run needs the solver/binary provenance extended first |

Configuration (environment variables, or the CLI flags shown by `--help`):

| Variable | Default |
|---|---|
| `PYPROTEUS_ORACLE`, `PYPROTEUS_ORACLE_COMMIT` | developer's `PROTEUS-oracle`, `8d37938` (as the oracle tests) |
| `KWAVE_MATLAB_PATH` | `PROTEUS-cleaning/PROTEUS/external/k-wave/k-Wave` |
| `PYPROTEUS_KWAVE_BINARY_DIR` | `tools/tracer_bullet/bin` (wrapper running k-wave-python's v1.4.1 OMP binary, D-10); on Linux point it at the official k-Wave `binaries/` folder |
| `PYPROTEUS_KWAVE_BINARY` | executable behind a wrapper (auto-detected for the tracer wrapper); recorded and hashed |
| `PYPROTEUS_KWAVE_LIBS` | runtime libraries for the wrapper (default `tools/tracer_bullet/.kwave-libs/lib`, made by `setup_libs.sh`) |
| `PYPROTEUS_GEOMETRIES_PATH` | oracle `geometry_data/` (no STL files: the `medium` case needs the renal tree STL from Zenodo record 10573662 here) |
| `PYPROTEUS_EXPORT_WORKDIR` | scratch folder (MATLAB cwd, PROTEUS settings/ground truth/RESULTS); temporary if unset |
| `PYPROTEUS_MATLAB_SESSION` | connect to a shared MATLAB session |

The exporter never writes into the oracle: PROTEUS' `path_setup()` is pointed
at scratch folders with a `PROTEUS_LOCAL_PATHS` file, and the oracle's
`git status --ignored` must be unchanged at the end. It refuses an oracle at
another commit or with local changes (unless `PYPROTEUS_ORACLE_COMMIT=any`;
the file then records `oracle_dirty`).

## Adding a reference (later milestones)

Append a spec to `specs.py`:

```python
register(
    ExportSpec(
        name="GUIfunctions.compute_delays.steered",
        module="GUIfunctions",
        function="compute_delays",
        description="Linear array, focus -20 mm, 12 deg.",
        inputs=lambda: {
            "Transmit": M(
                "setfield(setfield(reset_transmit(reset_transducer()),'LateralFocus',-20e-3),'Angle',12)"
            ),
            "Transducer": M("reset_transducer()"),
            "Medium": M("reset_medium()"),
        },
        outputs=("Transmit",),
    )
)
```

- Inputs are built in call order: a `M("<MATLAB expression>")`, or a Python
  value sent through `tests/oracle/helpers.to_matlab` (1-D arrays must be
  wrapped in `row()`/`col()`). Choose inputs that hit the traps (ties at .5,
  equal values, odd shapes, out-of-range queries).
- `outputs` names the output arguments; or give `call="..."` assigning
  `pyp_out.<field>` (and optionally `pyp_extra`) yourself.
- `seed`: `rng(seed,'twister')` before the inputs are built; list any other
  seed in `seeds` so it is recorded.
- `needs_kwave=True` sets up k-Wave and the PROTEUS data folders; `slow=True`
  excludes the spec from the default run.

Then export, run `update_registry`, and load it in a test with
`pyproteus.io.reference.load(name)` (skip on `ReferenceDataUnavailable`).

## Publishing on Zenodo (how-to; not done yet)

Nobody has created a Zenodo record. When the developer decides to publish:

1. Export everything on the reference machine (`--all`, outside the sandbox),
   with a clean oracle at the pinned commit and a clean pyproteus checkout
   (`exporter_commit` must not end in `-dirty`).
2. `python -m tools.export_reference.update_registry`; commit the registry.
3. On zenodo.org (group community, two admins, ROADMAP §13): *New upload*,
   upload every `data/*.h5` (flat, no zip), fill the metadata from
   `zenodo_metadata.json` (creators, license and title still to be confirmed:
   the draft says CC-BY-4.0), reserve the DOI, publish.
4. Set the base URL to the record DOI and commit:
   `python -m tools.export_reference.update_registry --base-url doi:10.5281/zenodo.<record>/`.
   Check from a clean cache: `PYPROTEUS_REFERENCE_CACHE=$(mktemp -d) pytest -m "not oracle"`.
5. Later versions: *New version* on the same concept record, upload the full
   set again, update `--base-url` to the new version's DOI.
