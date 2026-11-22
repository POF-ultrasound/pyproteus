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

## Settings (`pyproteus.config`, `pyproteus.data`)

Compared as flattened trees: every node's class, size, field names **in order** and values (bit-exact, sign of zero included).

| MATLAB | Python | Class | Oracle test | Status |
|---|---|---|---|---|
| `reset_acquisition`, `reset_simulation_parameters`, `reset_medium` | `config.reset_*` | exact | `tests/oracle/test_config.py` | ✅ |
| `reset_microbubble` | `config.reset_microbubble` | exact; `Distribution.Probabilities` 1e-12 rel (7e-16 seen, B-31/B-32) | `test_config.py` | ✅ |
| `reset_transducer` | `config.reset_transducer` | exact; impulse responses pending M1 (`estimate_impulse_response`) | `test_config.py` | 🟡 |
| `reset_transmit` (all catalogue configurations, missing `Configuration`) | `config.reset_transmit` | exact; `VoltageSignal`, `PressureSignal`, `MechanicalIndex` pending M1 | `test_config.py` | 🟡 |
| `reset_geometry` (renal tree, mouse brain, no `Folder`, existing fields kept) | `config.reset_geometry` | exact | `test_config.py` | ✅ |
| `get_transducer_properties` (5 types, `Custom linear`, unknown) | `config.properties.get_transducer_properties` | exact | `test_config.py` | ✅ |
| `load_transducer_model` (PROTEUS both forms, Verasonics, v7.3, single/int IR, errors) | `config.properties.load_transducer_model` | exact | `test_config.py` | ✅ |
| `get_gas_properties`, `assign_medium_properties` (all tissues, substrings, Q-30), `assign_liquid_properties`, `compute_sound_speed_extrema` (NaN, Inf) | `config.properties.*` | exact | `test_config.py` | ✅ |
| `compute_polydisperse_distribution` | `config.properties.compute_polydisperse_distribution` | `R` exact, `P` 1e-12 rel (B-31/B-32) | `test_config.py` | ✅ |
| `linspace(d1,d2,n)` | `config.properties.linspace` | bit-identical (black-box) | `test_config.py::test_linspace` | ✅ |
| `load` of settings files (`write_to_file`, v6/v7/v7.3) | `config.load_settings_mat` | exact (incl. extra fields, cells, struct arrays) | `test_config.py::test_load_settings_mat` | ✅ |
| `load` of ground-truth frames, `FlowSimulationParameters` (v7, v7.3), frame file names | `config.load_frame`, `load_flow_simulation_parameters`, `frame_file_name` | exact | `test_config.py` | ✅ |
| `load` of every shipped `.mat` (example data, models, constants, `tests/baseline`), also re-saved as v7.3 | `config.loadmat` | exact | `test_config.py::test_loadmat_shipped_files` | ✅ |
| shipped constants (`tissue_properties`, `fit_SigmaR_*`, `*TransferFunctionFit`, `GeometryProperties`, `C1-6D`) | `pyproteus.data.load` | exact (SHA-256 of source checked) | `tests/oracle/test_data.py` | ✅ |

Notes:

- `linspace` is matched black-box (`d1 + k*(d2-d1)/(n-1)`, last element set to `d2`); not a numerics helper yet.
- Settings JSON (`config.save_settings_json`) is lossless for every value tree (round trip tested on real settings files).

