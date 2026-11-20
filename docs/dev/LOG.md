# Rewrite log

Newest first. One entry per working session or milestone step: what was done, what was found, what is next. Link bugs (B-n) and decisions (D-n).

## 2026-11-20 — M0 k-Wave tracer bullet (kwave-tracer)
- Tiny PROTEUS-style hybrid cases (37×29×23, Nt=300, PML 10/12/9 outside, `Smooth=false`, heterogeneous nonlinear absorbing medium with a "vessel", asymmetric masks): pressure (linear array, monopole) and velocity (curved array, `ux`/`uy` along the normal). `tools/tracer_bullet/`, test `tests/oracle/test_kwave_tracer.py` (three cases incl. `no_dispersion`).
- Binary: MATLAB k-Wave 1.4.1 ships none for macOS arm64; both sides use k-wave-python's `kspaceFirstOrder-OMP` v1.4.1 (C++ build 2.17, "v1.3") through a wrapper (`bin/kspaceFirstOrder-OMP`) that points dyld at an isolated conda prefix with fftw/hdf5/libomp/zlib (`setup_libs.sh`, gitignored `.kwave-libs/`). MATLAB's `kspaceFirstOrder3DC` accepts the binary's HDF5 format unchanged.
- Results vs MATLAB `kspaceFirstOrder3DC`: k-wave-python `kspaceFirstOrder(backend="cpp")` 4.4e-7 / 5.6e-7 rel (pressure / velocity), `backend="python"` 6.8e-7 / 5.4e-7. cpp HDF5 inputs identical to MATLAB's except `rho0_sg*` last plane (B-13). Source scaling (additive p and u) reproduced bit-exactly from the documented formulas.
- Found B-10 (backend conventions: cpp needs find-order rows + external scaling, returns `(Nt, n_sensor)`; python wants C rows), B-11 (engine vs pyexpat), B-12 (legacy API scaling bug + crash), B-13 (staggered density wrap).
- Reviewer pass: added a `no_dispersion` case (found B-14), per-row error check (≤ 8.9e-6 at rows ≥ 1e-3 of the peak; tolerance 2e-5 per row), undo of k-wave-python's time-axis permutation when `Nt == n_sensor`, per-axis velocity scaling, pinned runtime libs.
- Scope not covered (later milestones): parity is against `kspaceFirstOrder3DC` on the OMP binary only (references must be generated with 3DC, not MATLAB-native `kspaceFirstOrder3D`/CUDA `3DG`); dipole source with `ux` only; the merged-mask source of `update_source.m`; source signals ≥ Nt.
- Open: cluster GPU compute capability (needs developer); upstream reports for B-10/B-12/B-13; B-14 decision (QUIRKS entry, python backend for self-sense).
- Next: M3 lifts `adapter.py`/`ordering.py` into `backends/kwave.py`.

## 2026-11-20 — M0 repo setup and oracle fixture (m0-setup)
- `pyproject.toml` (hatchling, src layout, Python >=3.11, k-wave-python==0.6.2 pinned, extras cuda/gui/oracle/docs/dev, pytest markers `oracle`/`slow`), ruff 0.17.6 config, pre-commit, CI (lint job + Linux/Windows/macOS-14 arm64 × Python 3.11-3.13).
- `tools/check_bans.py`: AST checker for `np.round`/`.round(`/builtin `round`, `sort`/`argsort` without `kind="stable"`, `np.interp`, outside `src/pyproteus/numerics/`; `# bans: allow` opt-out; tests in `tests/unit/test_check_bans.py`.
- `tests/oracle/conftest.py`: session engine fixture (imports `ssl` first, B-1), oracle modules + `tests/oracle/shims/` on the MATLAB path, commit check against D-7 (`PYPROTEUS_ORACLE`, `PYPROTEUS_ORACLE_COMMIT`, `PYPROTEUS_MATLAB_SESSION`), skips without matlabengine/MATLAB/oracle. Does not call `path_setup.m` (it creates folders inside the oracle and needs k-Wave).
- `tests/oracle/helpers.py`: NumPy ↔ MATLAB conversion; 1-D arrays must be wrapped in `row()`/`col()`. Engine preserves N-D index order both ways (checked).
- Found: the MATLAB Engine cannot start inside the Claude Bash sandbox (`EngineError: Transport stopped.`); oracle tests must run with the sandbox disabled. Engine start ≈ 3.5 s.
- Next: numerics helpers (second PR).

## 2026-11-20 — setup
- Repo created at POF-ultrasound/pyproteus (public). Roadmap v4 in `docs/ROADMAP.md`, checklists in `docs/dev/MILESTONES.md`.
- conda env `pyproteus`: Python 3.12, k-wave-python 0.6.2 (latest on PyPI), matlabengine 25.1.2 (MATLAB R2025a Update 1).
- Verified in the env: MATLAB `rng(42,'twister'); rand(1,3)` equals NumPy `RandomState(42).random_sample(3)` bit for bit.
- Found B-1 (MATLAB engine vs conda OpenSSL).
- Next: M0 (tracer bullet, repo setup, parity harness, numerics helpers, settings).
