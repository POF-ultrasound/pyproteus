# Contributing to pyproteus

Until 1.0, pyproteus is a **faithful port** of MATLAB PROTEUS v1.1.0 (pinned
oracle: commit `8d37938`, see `docs/dev/DECISIONS.md` D-7). A ported function
must give the same numbers as the MATLAB original, quirks included. Quirks are
recorded in [docs/QUIRKS.md](docs/QUIRKS.md), not fixed. Parity status lives in
[docs/PARITY.md](docs/PARITY.md).

You never need MATLAB to contribute: oracle tests skip without it, and CI runs
everything else.

## Setup

```bash
conda env create -f environment.yml
conda activate pyproteus
pip install -e ".[dev]"
pre-commit install
```

Checks run in CI and by pre-commit:

```bash
ruff check . && ruff format --check .
python tools/check_bans.py
pytest -m "not oracle"
```

### Oracle tests (MATLAB machines only)

```bash
pip install "matlabengine==25.1.*"   # MATLAB R2025a, Python <= 3.12
HYPOTHESIS_PROFILE=oracle pytest -m oracle
```

- `PYPROTEUS_ORACLE`: path to the read-only MATLAB PROTEUS checkout.
- `PYPROTEUS_ORACLE_COMMIT`: expected commit (default `8d37938`; `any` to skip the check).
- `PYPROTEUS_MATLAB_SESSION`: connect to a shared MATLAB session
  (`matlab.engine.shareEngine('name')` in MATLAB) instead of starting one.
- One engine is shared by the whole test session (`matlab_engine` fixture in
  `tests/oracle/conftest.py`). Engine calls are slow: batch many cases into one call.
- Pass arrays with `tests/oracle/helpers.py`: 1-D arrays must be wrapped in
  `row(...)` or `col(...)` because MATLAB has no 1-D arrays.
- The MATLAB Engine cannot start inside a macOS sandbox (e.g. an agent's
  sandboxed shell): it fails with `EngineError: Transport stopped.` and the
  oracle tests skip.

## Porting a function

1. Port the PROTEUS MATLAB code (MIT, porting is fine). Keep the MATLAB name in
   snake_case (`calcBubbleResponse` → `calc_bubble_response`) and add it to the
   name table below.
2. **Never translate MathWorks toolbox source** (`interp1.m`, `pchip.m`,
   `ode45.m`, `hilbert.m`, ...) or k-Wave MATLAB source (LGPL). Use the
   documented behaviour and published papers, or k-wave-python. Do not probe
   MATLAB to reproduce its internals at ulp level (D-12).
3. Use the `pyproteus.numerics` helpers for every operation in the checklist below.
4. Add an oracle test (same inputs through MATLAB and Python) whose inputs hit
   the traps: `.5` ties (also negative), equal values, odd and non-square
   shapes, single elements, out-of-range queries, NaN. Add pure unit tests so CI
   covers the code without MATLAB.
5. Compare within the comparison class of ROADMAP §4 (D-13: ≤ 1e-9 relative for
   geometry/signals/interpolation, statistical agreement for integers after
   rounding, solver-tolerance-based for ODEs, ≤ 1e-5 for k-Wave fields, ...).
   Add a row to `docs/PARITY.md`.

## MATLAB → NumPy semantics checklist

| MATLAB | Trap in NumPy/SciPy | Use |
|---|---|---|
| `round(x)` | `np.round`, `round()`, `.round()` round halves to even | `numerics.matlab_round` |
| `sort`, `[~,I] = sort(x)` | default `np.sort`/`argsort` is not stable; NaN placement; `'descend'` is not a reversed ascending sort | `numerics.sort`, `numerics.argsort` |
| `interp1(x,v,xq)` | `np.interp` clamps outside the range (MATLAB: NaN) | `numerics.interp1` |
| `griddedInterpolant` | linear extrapolation by default; SciPy differs | `numerics.gridded_interpolant` |
| `pchip` | documented shape-preserving end-slope rule | `numerics.pchip` |
| `find`, `sub2ind`, `ind2sub`, `x(:)`, `reshape`, `cumsum` on N-D | column-major order and 1-based indices | `numerics.find`, `numerics.sub2ind`, `numerics.ind2sub`, `order="F"` |
| `ifft(X,[],dim,'symmetric')` | uses only the first half of the spectrum | `numerics.ifft_symmetric` |
| `hilbert(X)` | MATLAB works along columns (dim 1); SciPy along the last axis | `numerics.hilbert` |
| `conv`, `convn(A,B)`, `convn(A,B,'same')` | centre of even kernels differs | `numerics.convn` |
| `sind`, `cosd` | `np.sin(np.deg2rad(x))` is not exact at multiples of 90 (`cos(pi/2) = 6e-17`, not 0) | `numerics.sind`, `numerics.cosd` |
| `rand`, `randi` with `rng(seed,'twister')` | column-major fill of `rand(m,n)`; `randi` mapping | `numerics.MatlabRNG` |
| `unique`, `intersect`, `ismember` | output order and index outputs | port case by case, test the indices |
| 1-based indexing, `end`, `squeeze`, implicit expansion | off-by-one, dropped axes | review shapes; keep MATLAB index values in variables named `*_m1` only when they are 1-based |

`tools/check_bans.py` (CI and pre-commit) rejects `np.round`/`.round(`/`round(`,
`sort`/`argsort` without `kind="stable"` and `np.interp` outside
`src/pyproteus/numerics/`. A line can opt out with `# bans: allow` and a reason.

## MATLAB → Python names

| MATLAB | pyproteus |
|---|---|
| `round` | `pyproteus.numerics.matlab_round` |
| `sort` | `pyproteus.numerics.sort`, `pyproteus.numerics.argsort` |
| `find` | `pyproteus.numerics.find` |
| `sub2ind`, `ind2sub` | `pyproteus.numerics.sub2ind`, `pyproteus.numerics.ind2sub` |
| `interp1` | `pyproteus.numerics.interp1` |
| `griddedInterpolant` | `pyproteus.numerics.gridded_interpolant` |
| `pchip` | `pyproteus.numerics.pchip` |
| `ifft(...,'symmetric')` | `pyproteus.numerics.ifft_symmetric` |
| `hilbert` | `pyproteus.numerics.hilbert` |
| `conv`, `convn` | `pyproteus.numerics.convn` |
| `rng(seed,'twister')`, `rand`, `randi` | `pyproteus.numerics.MatlabRNG` |

## Workflow

One branch per task, PR to `main`, CI green before merging. Log work in
`docs/dev/LOG.md`, behaviour differences in `docs/dev/BUGS.md`, decisions in
`docs/dev/DECISIONS.md`.
