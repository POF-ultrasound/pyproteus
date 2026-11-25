<!-- Mirror of the Obsidian roadmap note (v4). Edit both or tell the main session. -->
# pyproteus Roadmap

Back to PROTEUS-py Project · Web version: https://claude.ai/artifact/D8QzuMHVtkyasxfMZ1Dr6F

A separate public repository, https://github.com/POF-ultrasound/pyproteus (hosted on the group organisation, where most contributors are; PROTEUS-SIM/PROTEUS will point to it), that converts PROTEUS from MATLAB to Python, matches the MATLAB results at the time of conversion, and runs on laptops and GPU workstations on any OS. After that, development continues in Python only.

Reference: PROTEUS v1.1.0 · team: 1 developer + Claude Code · reviewed twice against the code (13 Nov 2026).

## Summary

1. **Same results, better implementation (D-15).** pyproteus 1.0 reproduces every simulation mode of MATLAB PROTEUS v1.1.0 within the D-13 tolerances; that behaviour is the only reference we have. It is not an exact copy: where PROTEUS's implementation is limiting (toolbox and package dependencies, workflow, data handling, memory use, speed, usability), the Python version may do it differently, as long as results stay within tolerance. Result-changing quirks are still reproduced by default and logged; fixes come in 1.x unless the developer decides otherwise.
2. **MATLAB is the test oracle and stays untouched.** Tests call the original MATLAB functions through the MATLAB Engine for Python and compare with the port on the same inputs. Frozen reference files keep the comparison in CI without MATLAB.
3. **Silent MATLAB-vs-NumPy differences are the main risk** (rounding of halves, sort stability, extrapolation, column-major order). Dedicated helpers on day one, a lint rule, and test inputs that hit ties on purpose.
4. **Same random streams:** the simulation path only uses `rand`/`randi`; seeding NumPy's MT19937 the documented way gives the same draws (checked with seed 42). Results are compared at the float tolerance, not bit for bit.
5. **First task: a k-Wave tracer bullet** (one tiny hybrid case through MATLAB 3DC and k-wave-python, compared).
6. **CPU and GPU first, cluster later:** CPU on every OS, CUDA on Linux/Windows, Apple Silicon on CPU; parallel runs locally (one worker per GPU or core group). Cluster tooling (Slurm templates, containers) is out of scope until after 1.0.
7. **Two test phases:** Phase A uses MATLAB as the oracle until parity is confirmed; Phase B replaces it with pyproteus' own CI testing physics and code behaviour.

## 1. Goals and constraints

- **Goal:** leave MATLAB. Free tooling, larger community, Python ecosystem (k-wave-python, SciPy, meshio, pyvista, PyTorch).
- **Parity:** at creation, same behaviour and results as MATLAB PROTEUS v1.1.0 within medium tolerances (§4, D-12/D-13). Not a bit-for-bit copy of MATLAB: no black-box reverse-engineering of MATLAB internals.
- **Hardware:** CPU on Linux/Windows/macOS, NVIDIA GPU (CUDA), Apple Silicon on CPU; batched and parallel runs on one machine. Clusters later.
- **GUI:** similar to the MATLAB apps, cleaner and easier.
- **Resources:** 1 developer + Claude Code; research software; rely on the community later.
- **License:** MIT (unchanged).

## 2. The MATLAB code today

- ~10.7k lines outside tests, 94 tests, 8 App Designer apps (MainGUI + 7 settings editors).
- Modules: acoustic-module 3.4k, GUIfunctions 1.8k (holds core physics too), mesh-voxelisation 1.2k (third party, BSD-3-Clause with non-endorsement clause), microbubble-simulator 1.2k, streamline-module 1.1k, delay-and-sum 0.7k.
- Engines: k-Wave 1.4.1 (`3DC` default, `3DG` CUDA, MATLAB solver), k-Wave helpers `filterTimeSeries`, `getWin`; `ode45` for bubbles; `ode23` with terminal event for streamlines; `parfor`; `gpuArray`.
- Streamlines use nearest-voxel velocity lookup via `round` (`get_vtu_indices.m`), not interpolation.
- Data: 7 settings structs in `.mat` (all 25 shipped `.mat` are v5); flow data VTU (Zenodo record 10573662); outputs `.mat`, some v7.3.
- Reproducibility hazards (ported as-is, fixed after 1.0):
  - wall-clock timer stops bubble integration after `60 s × N_MB` (`calcBubbleResponse.m:8-12`);
  - bubble ODE batches share adaptive steps after sorting by radius → results depend on batch size;
  - random draws inside `parfor` during streamline tracking → parallel tracking not reproducible even in MATLAB.

## 3. MATLAB side: untouched reference

1. Tag and freeze MATLAB PROTEUS v1.1.0 after the revamp merge. No feature work in MATLAB during the port; critical fixes only, mirrored in Python.
2. No MATLAB code changes for the port: test-only shims on the MATLAB path where needed (like `tests/shims/kspaceFirstOrder3DC.m`).
3. Reference export as a script in pyproteus (`tools/`) via the MATLAB Engine: per-module inputs/outputs and full runs (tiny, small, medium) with the **real** k-Wave C++ binary (current PROTEUS baselines use a shim swapping 3DC for the MATLAB solver). HDF5, versioned Zenodo record.

## 4. Parity method

**Layer 1, live oracle tests** (developer machine with MATLAB)
- MATLAB Engine for Python calls the original; the test calls the port with the same inputs. Engine for R2025a supports Python 3.9-3.12 → oracle env pinned to 3.12.
- Inputs: reference settings + cases built to break things (ties at .5, equal radii, non-square/odd grids, single element, curved arrays, out-of-range queries).
- Hypothesis-randomised inputs for leaf functions only (engine calls are slow).

**Layer 2, frozen reference tests** (CI, everyone): reference HDF5 from Zenodo via pooch.

| Quantity | Target | Why |
|---|---|---|
| Integer-valued after `round` (masks, voxel indices, DAS sample indices) | statistical: ≥ 99.9 % equal, differences reported; downstream effect checked at the float/RF class | float noise at a .5 tie moves an index |
| Geometry, delays, weights, signals, interpolation | ≤ 1e-9 rel (plus a small atol near zero) | same formulas; libm/FFT/BLAS differ |
| Random draws | same stream; compared at the float class | documented MT19937 seeding |
| Medium background (truncated normal) | ≤ 1e-9 rel | `erfcinv` implementations differ |
| Bubble radius, mass source | ≈ 10·RelTol rel + AbsTol abs of the solver settings; plus accuracy vs analytical/high-accuracy references | same ODE method, adaptive steps may differ |
| Streamlines (serial) | statistical: paths agree until the first voxel flip; distributions (positions, velocities, counts) match | piecewise-constant velocity |
| k-Wave fields, RF | ≤ 1e-5 rel (single) | same binary; input preparation |
| End-to-end RF, images | RF ≤ 1e-3 × max (D-24; propagation stage alone ≤ 1e-9); image metrics equal | bubble ODE differences pass into RF |

Tolerance policy (D-12, 2026-11-22): reproduce behaviour and results, not MATLAB's bits. Never tune code to MATLAB ulps (libm rounding, summation order, formula grouping, solver step sequences) by probing MATLAB; use documented semantics, papers and standard library routines.

Parity report generated from test output. Parity covers serial mode; after 1.0, one seeded stream per bubble makes parallel = serial.

**Phase B, own CI (after parity is confirmed):** the MATLAB oracle tests are frozen into an archived job and no longer gate changes. CI then tests pyproteus on its own terms:
- physics: analytical cases (free-field Green's function, Rayleigh-Plesset linear resonance, Minnaert frequency, point-source k-Wave vs analytical, DAS point-spread function), conservation and symmetry checks, convergence with grid spacing;
- code behaviour: unit tests per module, property tests (shapes, dtypes, units, seeds), API and settings-schema tests, CPU vs CUDA agreement;
- regression: pyproteus' own frozen reference outputs (replacing MATLAB references), updated deliberately with a changelog entry.

## 5. MATLAB semantics to replicate

Each gets a helper in `numerics/` before any physics is ported.

| MATLAB | NumPy/SciPy trap | Where |
|---|---|---|
| `round` halves away from zero (documented) | `np.round` halves to even; `floor(x+0.5)` wrong near 0.5 | all index computations, DAS sample indices |
| `sort` stable (documented) | default `argsort` not stable | radius sort before ODE batching |
| `interp1` NaN outside range | `np.interp` clamps, `interp1d` raises | `draw_random_radii`, `getShellProperties`, `compute_surface_tension` |
| `griddedInterpolant` linear extrapolation | SciPy defaults differ | `getShellProperties`, `compute_bubble_mass_source` |
| `pchip` end rule | documented shape-preserving end slopes; SciPy/textbook form at the float class | bubble driving pressure |
| column-major `find`, `sub2ind`, `cumsum`, `reshape` | row-major default | sensor/source row order, weighted sampling |
| `ifft(...,'symmetric')` | = `irfft` of first N/2+1 bins | `compute_RF`, `define_source_transducer` |
| `hilbert` along columns | SciPy: last axis | RF, DAS |
| `convn(...,'same')` even kernels | off by one vs `scipy.signal.convolve` | `smooth_distribution` |
| `unique`/`intersect`/`ismember` order; float equality | different index outputs | `extract_sensor_data`, `generate_streamlines` |
| 1-based indexing, `squeeze`, implicit expansion | off-by-one, shapes | everywhere |

**ODE solvers:** clean-room `ode45` (Dormand-Prince) and `ode23` (Bogacki-Shampine, with event location) from Shampine & Reichelt 1997, checked against the oracle. Do **not** translate MATLAB's readable solver source (MathWorks copyright in an MIT repo). Same for k-Wave MATLAB code (LGPL): use k-wave-python's `filterTimeSeries`, `getWin`, source scaling.

**Random numbers:** `rand` via MT19937 with documented seeding; `randi` as `floor(u·n)+1`; `randn` only when the truncated-normal cutoff is infinite (not default); `randperm` only in plotting/preprocessing (out of scope).

## 6. k-Wave

- Hybrid path (default) runs k-Wave once per pulse type for all frames: time-varying pressure or velocity source on the transducer, binary sensor mask (bubble voxels of all frames + transducer voxels), `PMLInside=false`, `Smooth=false`, `kgrid.dt` set directly. No mass sources, no `kWaveArray`.
- Full simulator: bubble mass sources → additive pressure sources via k-Wave source scaling; self-sense runs one small k-Wave simulation per bubble per iteration.
- **Row order:** PROTEUS uses column-major indices (`find(source.p_mask)`, `intersect(mask_idx, …)`); k-wave-python returns C-order since 0.6.1 → adapter maps source and sensor rows explicitly; first test uses an asymmetric mask.
- Verify source scaling vs k-Wave 1.4.1 (additive pressure, velocity sources).
- k-wave-python 0.6.x pinned exactly, behind a thin adapter. `cpp` backend = parity target; `python` backend (NumPy/CuPy, in-process) for self-sense runs. Fallback: write the HDF5 input file and call the binary.
- Prefetch binaries in containers (offline compute nodes).

## 7. Python package

**Name:** `pyproteus` (PyPI + import; free; `proteus` is taken). `jproteus` reserved for a JAX version. Repo at POF-ultrasound/pyproteus; register the PyPI name early.

```
pyproteus/
  pyproject.toml          uv; requires-python >=3.11; extras [cuda] [gui] [oracle] [docs] [dev]
  src/pyproteus/
    numerics/             MATLAB semantics helpers, clean-room ode45/ode23, MT19937 rand/randi
    config/               settings models (7 groups), defaults (= reset_*), JSON I/O, .mat readers
    data/                 shipped constants converted once from .mat
    transducer/ transmit/ medium/ microbubble/ flow/ geometry/ acoustic/ reconstruction/
    backends/kwave.py     adapter over k-wave-python
    execution/            job model, local executor (process pool, GPUs)
    io/                   HDF5 results, pooch registry
    gui/                  optional
    cli.py                pyproteus run | estimate | gui | import-matlab  (sweep later)
  tests/ oracle/ reference/ unit/
  tools/                  reference export via MATLAB Engine
  docs/  examples/        notebooks mirroring example_scripts/
```

Rules:
- Modules follow MATLAB folders; functions keep MATLAB names in snake_case (`calcBubbleResponse` → `calc_bubble_response`), lookup table in docs. Better names after parity.
- Settings models accept MATLAB field names.
- No global state, no paths in library code.
- CI bans bare `np.round`, `argsort` without `kind="stable"`, `np.interp`.

## 8. Component mapping

| MATLAB | Python | Risk |
|---|---|---|
| k-Wave 3D | k-wave-python (`cpp` / `python`) | high until tracer bullet passes |
| `filterTimeSeries`, `getWin` | k-wave-python equivalents | verify |
| `ode45`, `ode23` + events | clean-room ports | medium |
| interpolation, rounding, sorting | `numerics/` helpers | silent-bug risk |
| vtkToolbox `readVTK` | meshio, pyvista | low |
| `VOXELISE` (BSD-3-Clause) | direct port, keep notice (done: `pyproteus.geometry`, `LICENSES/`) | low |
| `parfor` | process pool | low |
| `gpuArray` | CuPy | medium |
| settings, `.mat` | pydantic; `scipy.io.loadmat` (v5), h5py (v7.3) | low |
| App Designer | see GUI | effort |
| `VideoWriter`, plots | matplotlib, imageio-ffmpeg | low |
| `matlab.unittest` | pytest | low |
| `vtu2matlab` (`alphaShape`), time-estimation (`fit`) | out of parity scope; ship outputs | — |

## 9. Hardware and execution

| Part | CPU | NVIDIA | Apple Silicon |
|---|---|---|---|
| k-Wave | C++ OpenMP binary | CUDA binary (CC ≥ 7.5) or CuPy solver | arm64 OpenMP binary |
| Bubble ODEs | NumPy vectorised | CuPy | NumPy |
| Green's, RF, DAS | NumPy | CuPy | NumPy (Accelerate) |
| Streamlines | NumPy, parallel over bubbles | CPU | CPU |

- Array backend: NumPy + CuPy only; parity on NumPy float64; CuPy checked against NumPy.
- Apple Silicon: CPU only (MPS/jax-metal no float64, MLX not array-API, no Metal k-Wave). Revisit after 1.0.
- Older NVIDIA (V100, P100): CuPy solver or older binary. Check cluster GPUs in the tracer bullet.
- AMD: not a 1.0 target.
- Memory: 10⁷-10⁸ voxels at 6 ppw, 2-5 MHz, few cm → ~2-15 GB CUDA; hybrid sensor (bubble voxels × Nt) can exceed the field. `pyproteus estimate` reports both.
- **Job model:** simulation → independent jobs (frame × pulse × variant), one HDF5 per job (settings, seeds, versions), merge step.
  - Local: process pool, one worker per GPU (`CUDA_VISIBLE_DEVICES`) or core group; set `OMP_NUM_THREADS`/threadpoolctl per worker; spawn on macOS/Windows → picklable jobs.
  - Cluster runs (Slurm arrays, containers): not before 1.0.
  - Enhancement (backlog, not a priority): deep-learning datasets through `pyproteus sweep` and a PyTorch `Dataset` reader for the HDF5 output.
- Testing: GitHub CI Linux/Windows/macOS arm64, CPU, Python 3.11-3.13, tiny cases; CPU runs locally on the developer's Mac; a GPU hardware test on the university GPU cluster before each release.

## 10. Workflow and port order

Per function:
1. Claude Code ports with the semantics checklist (§5) and writes an oracle test with tie-hitting inputs.
2. Iterate until within the comparison class; freeze reference data.
3. Developer reviews every line that indexes, rounds or sorts, plus the physics.
4. Track in `docs/PARITY.md`; quirks in `docs/QUIRKS.md`.

Port order (call graph):
1. Numerics helpers
2. Settings models, `reset_*` defaults, `.mat` readers
3. GUIfunctions leaves (signals, size distributions, impulse response, transducer models, element geometry, simulation domain, sound-speed extrema)
4. Grid, medium, transducer (`define_grid`, truncated normal, `define_medium`, integration points/delays, travel times, delta function, sinc interpolation)
5. Delay-and-sum (both variants)
6. Microbubble model + clean-room `ode45`, bubble response, mass source
7. Homogeneous path (`run_simulation_homogeneous`, `compute_RF`, `compute_RF_data`)
8. Hybrid path (k-Wave adapter, transmit preprocessing, sources/sensors, `hybrid_simulator`, `main_RF`)
9. Full path (self-sense, `full_simulator`, `main_pressure_field`)
10. Streamlines + clean-room `ode23` with events
11. Voxelisation

Bottleneck: validation and hardware testing, not writing code.

## 11. Milestones

- **M0** Tracer bullet and setup → code-pyproteus M0 tracer bullet and setup
- **M1** Deterministic core → code-pyproteus M1 deterministic core
- **M2** Bubbles and homogeneous path → code-pyproteus M2 bubbles and homogeneous path
- **M3** k-Wave paths → code-pyproteus M3 k-Wave paths
- **M4** Flow and geometry → code-pyproteus M4 flow and geometry
- **M5** Execution and hardware → code-pyproteus M5 execution and hardware
- **1.0** Parity release → code-pyproteus 1.0 parity release
- **1.1** GUI → code-pyproteus 1.1 GUI

## 12. GUI

- Same workflow as the MATLAB apps, with live previews and validation.
- Settings editors generated from the settings models.
- **NiceGUI** (browser-based, MIT): one codebase for laptop and cluster (SSH tunnel). Simulations in a separate process with progress. 3D via plotly or NiceGUI scene.
- Alternative: PySide6 + pyvistaqt (use PySide6/LGPL, not PyQt/GPL).

## 13. Community and licensing

- ruff, pytest, pre-commit, CI checks; CONTRIBUTING ("add a transducer / bubble model / simulation type"); templates; protected main.
- Contributors never need MATLAB.
- semver, trusted publishing to PyPI, CITATION.cff with software DOI, Zenodo per release; announce at 1.0 (IUS, k-Wave forum); "good first issue" labels.
- Two admins on GitHub, PyPI, Zenodo.
- Licenses OK with MIT: k-wave-python (LGPL, unmodified), meshio, pyvista, trimesh, NiceGUI (MIT), PySide6 (LGPL). VOXELISE BSD-3-Clause: notice kept in `LICENSES/`; package licence expression "MIT AND BSD-3-Clause" for the bundled port. No translation of MathWorks or k-Wave MATLAB source.

## 14. After parity

- Fix quirks (timer, batch-dependent ODEs, parallel randomness) with documented changes.
- Rename/restructure; rewrite geometry preprocessing and time estimation in Python.
- New simulation types, transducer models, GPU bubble ODEs, batched self-sense, Apple Silicon float32 paths.
- `jproteus` (JAX): realistic for the homogeneous path (bubbles via own model or jbubble, Green's, RF, DAS); heterogeneous 3D propagation stays research (jwave last release 2024).

## 15. Decisions

Tracked as checkboxes in PROTEUS-py Project §Decisions.

## 16. Development setup

- Local env: conda env `pyproteus` (Python 3.12, conda-forge core deps, pip: k-wave-python 0.6.2 = latest on PyPI, matlabengine 25.1.2 for R2025a).
- MATLAB engine ships its own libcrypto: import `ssl` (or k-wave-python) **before** `matlab.engine`, otherwise HTTPS breaks in that process.
- MATLAB oracle source: the PROTEUS checkout at `PROTEUS-cleaning/PROTEUS` (branch `revamp`, v1.1.0 once tagged).
- Logs, bugs and decisions live in the repo under `docs/dev/` and are summarised in PROTEUS-py Project.

## 17. Review log

- **v3 → v4** (20 Nov 2026, group decisions): host on POF-ultrasound (public); Phase B own CI after parity; Slurm templates and cluster runs dropped until after 1.0; focus on CPU (any OS) and CUDA; DL datasets as backlog enhancement; GPU hardware test on the university cluster before each release.

- **v2 → v3** (second review vs code and MATLAB R2025a): clean-room ODEs (license); streamlines are nearest-voxel; semantics table + integer comparison class + CI bans; no MATLAB v1.2 release; tracer bullet first; explicit row-order mapping; binaries prefetched; cut PyTorch backend and generated Slurm executor; thread limits, Python versions, data files, port order.
- **v1 → v2** (group clarification): dropped MATLAB restructure; 1.0 = full parity; oracle method; job model; hardware matrix.
- **v1** reviewed for hybrid-path k-Wave needs, self-sense hotspot, ODE batching, timer, shim-based baselines.
