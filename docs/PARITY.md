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

## k-Wave backend (`pyproteus.backends.kwave`, M3)

Against MATLAB k-Wave 1.4.1 (`external/k-wave/k-Wave`, added by the test, not by
the oracle fixture). Both sides run k-wave-python's `kspaceFirstOrder-OMP` v1.4.1
(D-10), so the field rows check input preparation and row order, not the binary.

| MATLAB | Python | Class | Oracle test | Status |
|---|---|---|---|---|
| `kspaceFirstOrder3DC(kgrid, medium, source, sensor, ...)` as in `run_simulation.m` (`PMLInside=false`, `Smooth=false`, `DataCast='single'`, `PMLSize`/`PMLAlpha`): pressure, velocity (`ux`+`uy`, `ux` only), mixed velocity + pressure sources, `alpha_mode='no_dispersion'`, `Nt == n_sensor`; `record` `{'p'}`, `{'p_max','p_min'}`, `{'p','p_max','p_min'}` | `kspace_first_order(..., backend="cpp")` | 1e-5 rel | `test_kwave_backend.py` | ✅ |
| same | `kspace_first_order(..., backend="python")` (NumPy float32) | 1e-5 rel | `test_kwave_backend.py` | ✅ (`alpha_mode` dropped, D-9) |
| binary command line built by `kspaceFirstOrder3DC` | `binary_command` | exact | `test_kwave_backend.py::test_binary_command_matches_matlab` | ✅ (`-g` for 3DG not checked: no CUDA) |
| `kgrid.k_max` | `KGrid.k_max` | 1e-12 rel | `test_kwave_signal.py` | ✅ |
| `filterTimeSeries(Filter, Filter, x, 'ZeroPhase', true, 'TransitionWidth', TW, 'PPW', 2)` (struct kgrid/medium) and option sweep | `filter_time_series` | 1e-12 rel | `test_kwave_signal.py` | ✅ (B-60 workaround) |
| `getWin(N, 'Tukey', 'Param', p)` | `get_win` | 1e-12 rel | `test_kwave_signal.py` | ✅ |

Notes:

- `getWin` is only called in `get_transducer_integration_points.m` inside
  `if apodization_Z == true` with `apodization_Z = false` hard-coded: dead code.
  `get_win` is a thin pass-through with an oracle test; port the call site only
  if something reaches it.
- Not covered: `kspaceFirstOrder3DG` / CUDA (`device="gpu"`, `device_num`), CuPy
  (`backend="python", device="gpu"`), source signals longer than `Nt`, anisotropic
  grids with pressure sources (refused), `record_start_index` (PROTEUS does not set it).
