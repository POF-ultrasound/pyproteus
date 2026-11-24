# Milestones

Checklists per milestone. Tick items here as work lands; the main `pyproteus` session mirrors progress to the Obsidian TaskNotes tasks.

## M0: tracer bullet and setup

Plan: pyproteus Roadmap §6. k-Wave, pyproteus Roadmap §4. Parity method

### Prerequisites
- [ ] Revamp PR merged, MATLAB PROTEUS tagged v1.1.0 and declared frozen
- [ ] Reserve `pyproteus` (and `jproteus`) on PyPI
- [x] Create `POF-ultrasound/pyproteus` repo, public (2026-11-20)
- [ ] MIT license, protected main, CODEOWNERS
- [ ] Pointer from PROTEUS-SIM/PROTEUS README to pyproteus (PR)

### k-Wave tracer bullet (do first)
- [x] conda env `pyproteus`: Python 3.12, `matlabengine` 25.1.2, k-wave-python 0.6.2 (2026-11-20)
- [x] Tiny hybrid case run in MATLAB with the real 3DC binary; save source, sensor mask and sensor data (2026-11-20, `tools/tracer_bullet/`)
- [x] Same case through k-wave-python `cpp` backend (2026-11-20; also `python` backend)
- [x] Asymmetric mask test: map source and sensor rows between Fortran (MATLAB) and C order (k-wave-python) (2026-11-20, `tools/tracer_bullet/adapter.py`, B-10)
- [x] Compare sensor data (target ≤ 1e-5 rel); check additive pressure source scaling (2026-11-20: ≤ 7e-7; source inputs bit-identical)
- [x] Velocity source (`ux`/`uy`, curved array) case (2026-11-20)
- [ ] Check university cluster GPU compute capability (CUDA binaries need ≥ 7.5); try CuPy backend if older (open: needs cluster access, see findings note)
- [x] Note findings in PROTEUS-py Project §Notes (2026-11-20, shared note `tracer-bullet-findings.md`; mirror to Obsidian pending)

### Repo setup
- [x] `pyproject.toml` (hatchling, requires-python >=3.11, extras `[cuda] [gui] [oracle] [docs] [dev]`)
- [x] ruff, pytest, pre-commit; CI on Linux/Windows/macOS arm64, Python 3.11-3.13
- [x] CI check banning bare `np.round`, `argsort` without `kind="stable"`, `np.interp`
- [ ] Docs site skeleton (open); [x] `docs/PARITY.md`, `docs/QUIRKS.md`, MATLAB→Python name table (in CONTRIBUTING.md)
- [x] CONTRIBUTING with the MATLAB-to-NumPy semantics checklist

### Parity harness
- [x] `tests/oracle/`: fixture starting the MATLAB Engine with PROTEUS on the path; skipped without MATLAB
- [x] `tools/` reference export to HDF5 (per module, tiny/small/medium full runs) (2026-11-22, `tools/export_reference/`, format `docs/REFERENCE_DATA.md`; tiny and small exported locally, medium needs the renal tree STL from Zenodo)
- [x] pooch registry for reference data; first Zenodo record (draft) (2026-11-22, `src/pyproteus/io/reference.py`; Zenodo metadata drafted in `tools/export_reference/zenodo_metadata.json`, nothing published)

### Numerics helpers
- [x] MATLAB `round` (halves away from zero)
- [x] stable sort / argsort wrappers
- [x] Fortran-order index helpers (`find`, `sub2ind`, `ind2sub`)
- [x] `interp1` (NaN outside), `griddedInterpolant` (linear extrapolation), `pchip` wrappers
- [x] symmetric `ifft` (`irfft`), `hilbert` along axis 0, `convn(...,'same')`
- [x] MT19937 `rand`/`randi` matching MATLAB `rng(seed,'twister')`
- [x] Oracle tests for every helper, with tie-hitting inputs

### Settings
- [x] pydantic models for the 7 settings groups, accepting MATLAB field names (2026-11-22, `src/pyproteus/config/models.py`; MATLAB field order kept)
- [x] Defaults ported from `reset_*` (2026-11-22; signal fields pending M1: `Transducer.*ImpulseResponse`, `Transmit.VoltageSignal/PressureSignal/MechanicalIndex`)
- [x] Readers: MATLAB settings `.mat`, ground-truth frames, transducer models (`Transducer` and Verasonics `Trans`) (2026-11-22, v5 and v7.3)
- [x] Convert shipped constants (shell fits, transfer-function fits, tissue properties, geometry metadata) into package data (2026-11-22, `src/pyproteus/data/`, `tools/convert_data.py`)
- [x] JSON I/O for settings, lossless round trip (2026-11-22)

## M1: deterministic core

Plan: pyproteus Roadmap §10. Workflow and port order (steps 3-5). Each item: port + oracle test + frozen reference + row in `docs/PARITY.md`.

### GUIfunctions leaves
- [x] `get_voltage_signal`, `get_pressure_signal` (2026-11-23, `pyproteus.transmit`)
- [ ] `compute_gaussian_distribution`, `compute_polydisperse_distribution`
- [x] `estimate_impulse_response` (2026-11-23, `pyproteus.transducer.impulse_response`; SciPy SOS, D-17/B-80)
- [x] `load_transducer_model`, `get_transducer_properties` (2026-11-22, `pyproteus.transducer`, settings PR)
- [x] `get_element_geometry` (linear and curved) (2026-11-23, `pyproteus.transducer.geometry`)
- [x] `compute_simulation_domain`, `compute_sound_speed_extrema` (2026-11-23, `pyproteus.transducer.domain`; extrema in the settings PR)
- [x] `compute_delays` (2026-11-23, `pyproteus.transducer.delays`)
- [x] medium/liquid property helpers (`assign_medium_properties`, `assign_liquid_properties`, `get_gas_properties`) (2026-11-22, `pyproteus.medium`, `pyproteus.microbubble`; also `compute_sound_speed_extrema`, `compute_polydisperse_distribution`)

### Grid, medium, transducer
- [x] `define_grid`, `optimize_grid_size` (2026-11-23, `pyproteus.acoustic`/`pyproteus.medium`)
- [x] `random_truncated_normal` (float class ≤ 1e-9, `erfcinv` via SciPy) (2026-11-23, `pyproteus.acoustic`/`pyproteus.medium`)
- [x] `define_medium`, `voxelize_media_points` (2026-11-23, `pyproteus.acoustic`/`pyproteus.medium`; STL voxelisation via `pyproteus.geometry` since 2026-11-24, PR #19)
- [x] `get_transducer_integration_points`, `get_transducer_integration_delays` (2026-11-23, `pyproteus.acoustic`/`pyproteus.medium`; STL voxelisation in M4)
- [x] `compute_travel_times`, `evaluate_delta_function` (2026-11-23, `pyproteus.acoustic`/`pyproteus.medium`)
- [x] `sinc_interpolation`, `resample_signal` (2026-11-23, `pyproteus.acoustic`/`pyproteus.medium`)

### Delay-and-sum
- [ ] `das_transmit_distance`, `initialize_das_parameters`
- [ ] `compute_das_matrix`, `compute_das_matrix_general` (sample indices: exact)
- [ ] `DAS_reconstruction`, `DAS_reconstruction_general`
- [ ] `load_RF_data`

### Done when
- [ ] All of the above within their comparison class on reference settings and tie cases

## M2: bubbles and homogeneous path

Plan: pyproteus Roadmap §5. MATLAB semantics to replicate, port order steps 6-7.

### Clean-room ode45
- [x] Implement Dormand-Prince from Shampine & Reichelt 1997 (no MATLAB source): error norm, step-size rule, initial step, `Refine` Hermite output (from the papers; DOPRI5 dense output for `Refine`; solutions within class D-13)
- [x] Vectorised over a batch of bubbles (stacked system, as MATLAB does; Q-50)
- [ ] Oracle tests on simple ODEs and on `simple_RP` (simple ODEs and an RP-like RHS done; `simple_RP` itself waits for its port)

### Microbubble model
- [ ] `getMaterialProperties`, `getShellProperties` (Segers fit / lookup table), `getEqParam`
- [ ] `calc_surface_tension`, `calc_thermal_damp`, `compress_bubble`, `find_A_m0`
- [ ] `simple_RP` right-hand side (Marmottant shell)
- [ ] `calcBubbleResponse` (pchip driving pressure; timer ported as quirk, off in tests)
- [ ] `compute_bubble_mass_source` (stable radius sort, batching as MATLAB)

### Homogeneous path
- [ ] `calc_scatter_attenuated` (Green's function)
- [ ] `run_simulation_homogeneous`
- [ ] `compute_RF`, `compute_RF_data` (symmetric ifft)

### Done when
- [ ] End-to-end RF on given bubble positions matches MATLAB (RF ≤ 1e-4)
- [ ] CuPy path matches NumPy path

## M3: k-Wave paths

Plan: pyproteus Roadmap §6. k-Wave, port order steps 8-9.

### Adapter
- [x] `backends/kwave.py` over pinned k-wave-python (from the M0 tracer bullet) (2026-11-22)
- [x] Source/sensor row-order mapping covered by tests (2026-11-22)
- [x] `filterTimeSeries`, `getWin` from k-wave-python, checked against MATLAB k-Wave (2026-11-22; B-60 workaround for ZeroPhase)

### Hybrid simulator (default path)
- [ ] `preprocess_transmit`
- [ ] `define_source_transducer` (pressure and velocity sources)
- [ ] `define_sensor_transducer`, `define_sensor_MB`, `define_sensor_MB_all`
- [ ] `update_sensor`, `update_sensor_fast`, `extract_sensor_data`
- [ ] `hybrid_simulator` incl. interaction loop
- [ ] `main_RF` (outputs to HDF5)

### Full simulator
- [ ] `update_source` (source scaling parity)
- [ ] `compute_self_sense_pressure` (try `python` backend in-process)
- [ ] `full_simulator`
- [ ] `main_pressure_field`

### Done when
- [ ] Hybrid and full paths match MATLAB on tiny and small cases (CPU)
- [ ] Medium case on GPU

## M4: flow and geometry

Plan: port order steps 10-11. Streamlines use nearest-voxel velocity: compare statistically in serial mode (D-13).

### Clean-room ode23
- [ ] Bogacki-Shampine with terminal event location (no MATLAB source)
- [ ] Oracle tests incl. events

### Streamlines
- [ ] `load_vessel_data` (VTU via meshio/pyvista), `get_vtu_indices`, `get_velocity`, `get_step_size`
- [ ] `points_to_distribution`, `smooth_distribution` (`convn` same), `get_probability_density`
- [ ] `randi_weighted`, `draw_start_position`, `draw_random_radii`
- [ ] `exitVesselFcn`, `filter_points`, `filter_points_proximity`
- [ ] `generate_streamlines`, `backpropagation` (serial mode parity)
- [ ] Ground-truth frame output; reading MATLAB ground-truth frames

### Geometry
- [ ] Reference geometries (renal tree, mouse brain) from Zenodo via pooch (m1-grid-medium, PR #18)
- [x] Port `VOXELISE` (keep the notice; the licence is BSD-3-Clause) (2026-11-23, moved into M1: `pyproteus.geometry`, also `READ_stl` and the `define_medium` vessel block; 100 % voxel parity)

### Done when
- [ ] Streamlines: paths agree until the first voxel flip, and the distributions match (D-13)
- [ ] Full pipeline from geometry to RF matches MATLAB on a small case

## M5: execution and hardware

Plan: pyproteus Roadmap §9. Hardware and execution

### Job model
- [ ] Simulation → independent jobs (frame × pulse × variant); one HDF5 per job with settings, seeds, versions
- [ ] Merge step
- [ ] Local executor: process pool, one worker per GPU (`CUDA_VISIBLE_DEVICES`) or core group
- [ ] Thread limits per worker (`OMP_NUM_THREADS`, threadpoolctl); spawn-safe on macOS/Windows

### CLI
- [ ] `pyproteus run`
- [ ] `pyproteus estimate` (field + hybrid sensor memory, time)
- [ ] `pyproteus import-matlab`
### Hardware validation
- [ ] macOS arm64 (CPU), locally
- [ ] Windows (CPU, CUDA)
- [ ] Linux (CPU, CUDA)
- [ ] GPU hardware test on the university GPU cluster (single node, interactive or one job; no Slurm tooling)

Cluster tooling (Slurm templates, containers) and `pyproteus sweep` moved out of scope: see code-pyproteus enhancement DL datasets.

## 1.0: parity release

### Release checklist
- [ ] All MATLAB v1.1.0 simulation modes ported (hybrid, full, pressure field, both DAS)
- [ ] Parity report generated from test output; every module within its comparison class
- [ ] `docs/PARITY.md` complete; `docs/QUIRKS.md` lists all ported quirks and planned fixes
- [ ] Notebooks for every MATLAB example script
- [ ] One T-UFFC figure reproduced
- [ ] Reference data published on Zenodo (versioned)
- [ ] GPU and cluster runs on a university machine
- [ ] CITATION.cff with software DOI; CHANGELOG
- [ ] Trusted publishing to PyPI; tag `v1.0.0`
- [ ] Announce (IUS, k-Wave forum, mailing lists); label good first issues
- [ ] MATLAB PROTEUS README points to pyproteus; archive mode announced

## 1.1: GUI

Plan: pyproteus Roadmap §12. GUI. Can start once the settings models are stable (after M1).

- [ ] Confirm GUI technology (NiceGUI recommended)
- [ ] Main window: load/save settings, run, progress, results
- [ ] Settings editors generated from the models: Transducer, Transmit, Medium, Microbubbles, Geometry, Acquisition, Simulation parameters
- [ ] Live previews (pulse, size distribution, transducer and bubble positions in 3D)
- [ ] Simulations in a separate process with progress updates
- [ ] Works locally and on a cluster node through an SSH tunnel
- [ ] `pyproteus gui` command and docs

## Phase B: pyproteus' own CI

Once parity with MATLAB is confirmed, CI tests pyproteus' own physics and code instead of MATLAB behaviour. Plan: pyproteus Roadmap §4. Parity method.

- [ ] Freeze MATLAB oracle tests into an archived, non-gating job
- [ ] Analytical physics tests: free-field Green's function, Minnaert / linear RP resonance, point source k-Wave vs analytical, DAS point-spread function
- [ ] Conservation, symmetry and grid-convergence checks
- [ ] Unit and property tests per module (shapes, dtypes, units, seeds)
- [ ] Settings schema and public API tests
- [ ] CPU vs CUDA agreement tests (run on the university GPU cluster before release)
- [ ] pyproteus' own frozen regression references, updated only with a changelog entry
- [ ] Remove MATLAB-specific CI bans that no longer apply (keep the ones that guard correctness)

## Enhancement: deep-learning datasets

Backlog, not a priority. Builds on the M5 job model.

- [ ] `pyproteus sweep`: parameter grid / random sampling over settings → one job per sample
- [ ] Dataset manifest (settings, seeds, versions per sample)
- [ ] PyTorch `Dataset` reader for the HDF5 output (optional `[torch]` extra)
- [ ] Example notebook: small dataset on a laptop GPU
- [ ] Later: cluster runs (Slurm arrays, containers)
