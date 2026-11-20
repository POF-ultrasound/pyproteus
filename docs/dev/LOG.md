# Rewrite log

Newest first. One entry per working session or milestone step: what was done, what was found, what is next. Link bugs (B-n) and decisions (D-n).

## 2026-11-20 — M0 numerics helpers (m0-setup)
- Setup PR merged (POF-ultrasound/pyproteus#1). Numerics on branch `m0/numerics`.
- `src/pyproteus/numerics/`: `matlab_round`, `sort`/`argsort` (stable both directions, MATLAB NaN placement, default dim), `find`/`sub2ind`/`ind2sub` (0-based, column-major), `interp1` (linear/nearest/pchip, NaN outside, `'extrap'`, fill), `gridded_interpolant` (1-D, extrapolates with its method), `pchip`, `ifft_symmetric`, `hilbert` (MATLAB axis rule), `conv`/`convn` (full/same/valid, trailing-singleton semantics), `rng`/`MatlabRNG` (`rand`, `randi`).
- MATLAB facts found by black-box probing (R2025a): `rng(0,'twister')` = MT seed 5489; `randi` = `floor(u*n)+lo` with one double per draw, interleaves with `rand`; `round(-0.4) = -0`; descending sort is stable with NaNs first; `conv(...,'same')` starts at `floor(nb/2)` (SciPy: `(nb-1)//2`); `hilbert` of a row vector works along the row; `griddedInterpolant` rejects unsorted points while `interp1` sorts them; `find(x,0)` errors; linear interpolation is `v0*(1-t)+v1*t`, bit-identical to MATLAB; `'nearest'` ties go to the upper point.
- pchip: slopes from Fritsch–Butland/Moler matched bit for bit; coefficient form `(3δ-2d0-d1)/h` did not (≈ 89 % equal, SciPy behaves the same); de Boor's divided-difference form gives bit-identical results (0 ulp).
- Oracle tests (first round): 55 numerics oracle tests + 16 harness (71); Hypothesis sweep at 300 examples per test passed (153 tests incl. unit, ≈ 30 s). FFT-based helpers within 1e-12 rel (FFTW vs pocketfft), the rest bit-identical.
- `tools/check_bans.py` now skips calls on `pyproteus.numerics` aliases (`nm.sort(...)`, `pyproteus.numerics.sort(...)`); a name bound to both numpy and the wrapper is flagged.
- Reviewer round 1 (CHANGES REQUESTED) fixed: linear interpolation was not bit-identical on flat segments (`v0 == v1`, ~8 % of such queries 1 ulp off) and with Inf/NaN neighbours at nodes; the exact rule was deduced by black-box probing (flat → `v0`; NaN from a zero weight at a node → node value; checked over all special-value triples). Also: `sort` along an axis ≥ ndim is now the identity (MATLAB trailing singleton), `ifft_symmetric` defaults to MATLAB's dimension, `interp1(...,'pchip')` drops NaN samples, vector `xq` with matrix `v` gives MATLAB's numel(xq)×m, `rand` size rules (negative → 0, trailing 1s dropped), non-integer `randi`/size arguments raise, empty `convn` raises, engine fixture wrapped so Hypothesis can print failing cases.
- Licensing note: the reviewer said the pchip coefficient grouping resembles MathWorks' `pwch.m` (from memory; nobody opened it). The grouping is the expansion of the Newton divided-difference form (de Boor) and was selected among algebraically equal forms by comparing with MATLAB's `unmkpp(pchip(...))` output; variables were renamed to divided-difference names and the derivation is documented in the code. Raised with the main session for a decision.
- Sweep: 182 tests (unit + oracle) passed twice with `PYPROTEUS_ORACLE_EXAMPLES=300`.
- Reviewer round 2: SIGN-OFF. Opened B-20 (matrix `v` sign of zero, minor). Checker limitation (shadowed alias) documented.
- Not done: `randn` (out of scope per ROADMAP §5), `unique`/`intersect`/`ismember` index order (port case by case).

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
