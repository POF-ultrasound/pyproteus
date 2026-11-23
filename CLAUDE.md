# pyproteus — instructions for Claude Code sessions

Python port of MATLAB PROTEUS. Read `docs/ROADMAP.md` (plan), `docs/dev/MILESTONES.md` (checklists), `docs/dev/DECISIONS.md` before working.

## Phase A rules (until parity is confirmed)
- Reproduce MATLAB PROTEUS results (D-13 tolerances), quirks that change results included: record each in `docs/QUIRKS.md`, don't fix it before parity.
- Don't idealize PROTEUS (D-15). pyproteus is a step towards a better, more efficient, easier-to-use PROTEUS, not an exact copy. Challenge PROTEUS implementation choices (toolbox/package dependencies, workflow, data handling, memory, performance, API) and use a better Python design when results stay within tolerance. Note each such change in `docs/PARITY.md`; propose result-changing improvements to the main session instead of applying them.
- MATLAB oracle: `/Users/theodoremichel/Repos/TWENTE/Ultrasound/PROTEUS-cleaning/PROTEUS-oracle` (commit 8d37938, detached). Read-only: never edit, commit or switch branches there. Test-only shims go in `tests/oracle/shims/`.
- Porting PROTEUS's own MATLAB code is fine (MIT). Never translate MathWorks toolbox source (e.g. `ode45.m`, `interp1.m`) or k-Wave MATLAB source (LGPL): clean-room from papers/docs, or use k-wave-python.
- Every ported function gets an oracle test (same inputs through MATLAB Engine and Python) with inputs that hit ties at .5, equal values, odd/non-square shapes, out-of-range queries. Comparison classes: ROADMAP §4 (medium tolerances, D-13).
- Parity means same behaviour and results within medium tolerance, not a bit-for-bit copy (D-12). Never probe MATLAB to reproduce its internals at ulp level (libm, summation order, formula grouping, solver step sequences): that is reverse-engineering proprietary software. Follow documented semantics and papers; use standard NumPy/SciPy routines when they meet the tolerance.
- Use the `pyproteus.numerics` helpers for rounding, sorting, interpolation, column-major indexing, symmetric ifft, hilbert, convn. Never bare `np.round`, `np.argsort` without `kind="stable"`, or `np.interp`.
- Keep MATLAB function names in snake_case; settings models accept MATLAB field names.

## Environment (this Mac)
- Python: `~/miniconda3/envs/pyproteus/bin/python` (conda env `pyproteus`, Python 3.12). Install the package with `pip install -e ".[dev]"` in that env.
- Import `ssl` before `matlab.engine` (bug B-1).
- MATLAB R2025a at `/Applications/MATLAB_R2025a.app`; engine calls are slow — share one engine per test session (session-scoped fixture).
- The Bash sandbox breaks TLS for `gh`, `pip` and `conda`: run those with the sandbox disabled. Writing to `~/miniconda3` also needs it.

## Workflow
- One branch per task (`m0/numerics`, `m1/transducer`, ...), PR to `main`. Before merging: CI green and a reviewer agent pass (fresh agent, told to challenge parity, semantics traps and test strength). Then self-merge (D-6). No attribution lines in commits or PRs.
- Log as you go: `docs/dev/LOG.md` (dated entry per step), `docs/dev/BUGS.md` (B-n, any behaviour difference), `docs/dev/DECISIONS.md` (D-n). Tick `docs/dev/MILESTONES.md`.
- When a result contradicts the plan or needs a design decision (tolerance missed, MATLAB behaviour unclear, scope question), stop that item, write it up in BUGS.md or as an open question, and report it to the main `pyproteus` session, which asks the developer. Carry on with independent items meanwhile.
