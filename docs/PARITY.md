# Parity with MATLAB PROTEUS

Status of each ported function against the MATLAB oracle (PROTEUS commit
`8d37938`, MATLAB R2025a). Comparison classes from ROADMAP §4:

- **exact**: integer-valued results (indices, masks, rounding) equal; random draws bit-identical
- **1e-12 rel**: geometry, delays, weights, signals
- **ulp**: within a few ulp (e.g. `erfcinv`)
- see ROADMAP §4 for the bubble, k-Wave and end-to-end classes

Status: ✅ oracle test passes, 🟡 partial, ❌ fails / open bug, ⬜ not started.

## Numerics helpers (`pyproteus.numerics`)

| MATLAB | Python | Class | Oracle test | Status |
|---|---|---|---|---|
| test harness (array transfer) | `tests/oracle/helpers.py` | exact | `tests/oracle/test_numerics_harness.py` | ✅ |
| `round(x)` (double, single) | `matlab_round` | exact (incl. sign of zero) | `test_numerics_round_sort_index.py` | ✅ |
| `[S,I] = sort(x, dim, 'ascend'/'descend')` | `sort`, `argsort` (0-based `I`) | exact | `test_numerics_round_sort_index.py` | ✅ |
| `find(x)`, `find(x,n)`, `find(x,n,'last')` | `find` (0-based, column-major) | exact | `test_numerics_round_sort_index.py` | ✅ |
| `sub2ind`, `ind2sub` | `sub2ind`, `ind2sub` (0-based, column-major) | exact | `test_numerics_round_sort_index.py` | ✅ |
| `interp1(x,v,xq)` linear, `'extrap'`, fill value | `interp1` | bit-identical (class 1e-12) | `test_numerics_interp.py` | ✅ |
| `interp1(...,'nearest')` | `interp1(..., "nearest")` | bit-identical | `test_numerics_interp.py` | ✅ |
| `griddedInterpolant(x,v)` (linear, linear extrapolation) | `gridded_interpolant` | bit-identical | `test_numerics_interp.py` | ✅ |
| `pchip`, `interp1(...,'pchip')`, `griddedInterpolant(...,'pchip')` | `pchip`, `interp1`, `gridded_interpolant` | bit-identical (0 ulp) | `test_numerics_interp.py` | ✅ |
| `ifft(X,n,dim,'symmetric')` | `ifft_symmetric` | 1e-12 rel (FFTW vs pocketfft) | `test_numerics_signal.py` | ✅ |
| `hilbert(x)` (vectors along length, matrices along columns) | `hilbert` | 1e-12 rel | `test_numerics_signal.py` | ✅ |
| `conv(u,v,shape)`, `convn(A,B,shape)` | `conv`, `convn` | exact on integers, else 1e-12 rel | `test_numerics_signal.py` | ✅ |
| `rng(seed,'twister')`, `rand`, `randi` | `rng`, `MatlabRNG.rand`, `MatlabRNG.randi` | bit-identical | `test_numerics_random.py` | ✅ |
| `randn` | not provided | — | — | ⬜ (only used with infinite truncation cutoff) |

Notes:

- Every oracle test runs Hypothesis-generated batches as well as fixed tie cases;
  `PYPROTEUS_ORACLE_EXAMPLES=300` was used for the sweep recorded in `docs/dev/LOG.md`.
- `rng(0,'twister')` is MT19937 seed 5489 (MATLAB's default state), not seed 0.
- Linear interpolation (interp1, griddedInterpolant) is `v0*(1-t)+v1*t` plus two
  corrections found by black-box comparison: flat segments return `v0`, and on a
  node a NaN caused only by a zero weight (0·Inf) is replaced by the node value.
  Checked exhaustively over all {1, ±0, ±Inf, NaN, 2} value triples. Matrix-valued
  `v` can differ in the sign of zero on ±0 flat segments (B-20; not used by PROTEUS).
- pchip: SciPy's `PchipInterpolator` has the same slopes but different
  coefficient rounding (≈ 89 % of points bit-equal, up to thousands of ulp near
  zero crossings). `pyproteus.numerics` uses the Newton divided-difference form
  (de Boor), chosen among algebraically equal forms by comparing with MATLAB's
  `unmkpp(pchip(...))` coefficients; it matches bit for bit.
- `interp1(...,'pchip')` drops NaN sample values like MATLAB. Not compared yet:
  `pchip(x,v,xq)` and `griddedInterpolant(...,'pchip')` with NaN in `v`.
- `convn`/`conv` with empty inputs raise `ValueError` (MATLAB's sizes are irregular;
  PROTEUS never does this).
- Out of scope so far: `fft`/plain `ifft` with MATLAB's default dimension (use
  `np.fft` with an explicit `axis`), `unique`/`intersect`/`ismember` index order.

## ODE solvers (`pyproteus.numerics`)

| MATLAB | Python | Class | Oracle test | Status |
|---|---|---|---|---|
| `ode45` step sequence (RHS calls, accepted/rejected steps, `sol.x`, `sol.y`, `sol.stats`) | `ode45` (`t_steps`, `y_steps`, `stats`) | bit-identical | `test_ode45.py` | ✅ |
| `[t,y] = ode45(...)` at steps (`Refine` 1, or `tspan` points that are steps) | `ode45` | bit-identical | `test_ode45.py` | ✅ |
| `[t,y] = ode45(...)` dense output (`Refine` > 1, `tspan` points inside a step) | `ode45` | ≤ 1e-15 rel (bit-identical for 1 equation) | `test_ode45.py` | 🟡 B-50 |
| Rayleigh–Plesset-like batch, PROTEUS options | `ode45` + test RHS | bit-identical steps (fdlibm RHS); ≤ 1.5e-13 rel with NumPy `**` (class 1e-8) | `test_ode45.py::test_rayleigh_plesset_batch` | ✅ |
| `OutputFcn` protocol (flags, arguments, stop) | `ode45(..., OutputFcn=...)` | exact | `test_ode45.py::test_outputfcn_protocol` | ✅ |
| scalar/array `x^0.2` (fdlibm `pow`) | `ode45._pow5` | bit-identical | `test_ode45.py::test_step_rule_power_is_fdlibm` | ✅ |
| `ode23` + events | — | — | — | ⬜ (M4) |

Notes:

- Options covered: `RelTol`, `AbsTol` (scalar, vector), `NormControl`, `MaxStep` (default and set), `InitialStep` (set and automatic), `Refine`, `OutputFcn`; `BDF`/`MaxOrder` accepted and ignored (as MATLAB's ode45 does). Forward and backward integration; 2-point and many-point `tspan`; step-size failure.
- Not supported (raise): `Events`, `Mass`, `NonNegative`, `Vectorized` and other options; `Stats` is accepted and ignored. PROTEUS' bubble code needs none.
- Bit parity of a full simulation also needs the RHS to use MATLAB's arithmetic (B-51, B-52).
