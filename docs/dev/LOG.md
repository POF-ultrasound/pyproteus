# Rewrite log

Newest first. One entry per working session or milestone step: what was done, what was found, what is next. Link bugs (B-n) and decisions (D-n).

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
