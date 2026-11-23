# Parity with MATLAB PROTEUS

Status of each ported function against the MATLAB oracle (PROTEUS commit
`8d37938`, MATLAB R2025a). Comparison classes from ROADMAP §4:

Classes are D-13 (medium tolerance; parity is behaviour and results, not MATLAB's bits, D-12):

- **float**: ≤ 1e-9 rel (small atol near zero): geometry, delays, weights, signals, interpolation, truncated normal, random draws
- **statistical**: integers after rounding (indices, masks), ≥ 99.9 % equal, differences reported, downstream effect at the float/RF class
- **exact**: only where it falls out naturally (data read from files, documented integer semantics on exact inputs, storage round trips)
- **ODE**: ≈ 10·RelTol rel + AbsTol abs; **k-Wave** ≤ 1e-5; **end-to-end RF** ≤ 1e-4 (ROADMAP §4)

Rows still marked "bit-identical" predate D-12 and are being moved to these classes.

Status: ✅ oracle test passes, 🟡 partial, ❌ fails / open bug, ⬜ not started.

## Numerics helpers (`pyproteus.numerics`)

Classes per D-13. "exact" here means the helper's own output is integers or a
permutation of exact inputs (documented semantics, not float noise).

| MATLAB | Python | Class | Oracle test | Status |
|---|---|---|---|---|
| test harness (array transfer) | `tests/oracle/helpers.py` | exact (storage round trip) | `tests/oracle/test_numerics_harness.py` | ✅ |
| `round(x)` (double, single) | `matlab_round` | exact (sign of zero not compared) | `test_numerics_round_sort_index.py` | ✅ |
| `[S,I] = sort(x, dim, 'ascend'/'descend')` | `sort`, `argsort` (0-based `I`) | exact | `test_numerics_round_sort_index.py` | ✅ |
| `find(x)`, `find(x,n)`, `find(x,n,'last')` | `find` (0-based, column-major) | exact | `test_numerics_round_sort_index.py` | ✅ |
| `sub2ind`, `ind2sub` | `sub2ind`, `ind2sub` (0-based, column-major) | exact | `test_numerics_round_sort_index.py` | ✅ |
| `interp1(x,v,xq)` linear, `'extrap'`, fill value | `interp1` | float (B-70 with Inf/NaN values) | `test_numerics_interp.py` | ✅ |
| `interp1(...,'nearest')` | `interp1(..., "nearest")` | exact (selects sample values) | `test_numerics_interp.py` | ✅ |
| `griddedInterpolant(x,v)` (linear, linear extrapolation) | `gridded_interpolant` | float (B-70) | `test_numerics_interp.py` | ✅ |
| `pchip`, `interp1(...,'pchip')`, `griddedInterpolant(...,'pchip')` | `pchip`, `interp1`, `gridded_interpolant` (textbook Hermite, Fritsch–Butland/Moler slopes) | float (B-71 with NaN/±Inf samples) | `test_numerics_interp.py` | ✅ |
| `ifft(X,n,dim,'symmetric')` | `ifft_symmetric` | float (rel. to signal max) | `test_numerics_signal.py` | ✅ |
| `hilbert(x)` (vectors along length, matrices along columns) | `hilbert` | float (rel. to signal max) | `test_numerics_signal.py` | ✅ |
| `conv(u,v,shape)`, `convn(A,B,shape)` | `conv`, `convn` | exact on small integers, else float | `test_numerics_signal.py` | ✅ |
| `rng(seed,'twister')`, `rand`, `randi` | `rng`, `MatlabRNG.rand`, `MatlabRNG.randi` | float (same MT19937 stream) | `test_numerics_random.py` | ✅ |
| `randn` | not provided | — | — | ⬜ (only used with infinite truncation cutoff) |

Notes:

- Every oracle test runs Hypothesis-generated batches as well as fixed tie cases;
  `PYPROTEUS_ORACLE_EXAMPLES=300` was used for the sweep recorded in `docs/dev/LOG.md`.
- Float class for interpolation: `|a - e| ≤ 1e-9·|e| + 1e-12·max|e|`, NaN and ±Inf
  positions equal. Measured (300 random cases): pchip ≤ 2.2e-14 × max|ref|
  (1.5e-16 on a bubble-pulse case), linear ≤ 2e-15 elementwise rel.
- `rng(0,'twister')` is MT19937 seed 5489 (MATLAB's default state), not seed 0.
- Linear interpolation is the plain blend `v0*(1-t)+v1*t` (D-14). With Inf/NaN
  sample values, a query on a node, on a flat segment or at ±Inf of that interval
  can give NaN where MATLAB gives a number (B-70); the oracle test checks those
  queries are MATLAB's value or NaN. Finite data are unaffected. ±Inf queries with
  `'extrap'` give the limit of the end line (value if flat, else ±Inf), as MATLAB.
- pchip: Fritsch–Butland interior slopes, Moler's shape-preserving end slopes,
  textbook local power form evaluated by Horner (so ±Inf queries on a flat cubic
  give NaN, as MATLAB). Own code rather than SciPy's `PchipInterpolator` because
  SciPy rejects non-finite and complex values. A unit test checks the slope rules
  against an independent Hermite-basis implementation.
- NaN/Inf/complex sample values with pchip (oracle-tested): `interp1`/`pchip` drop
  NaN samples (column by column for matrix `v`); `griddedInterpolant` keeps them
  and a NaN secant makes its node slopes NaN, so it can give NaN next to a NaN
  sample where MATLAB gives a number (B-71; never a number where MATLAB gives
  NaN); ±Inf values propagate by IEEE rules, so any pchip API can give NaN where
  MATLAB gives ±Inf or a number (B-71); complex values are interpolated per
  real/imaginary part, also for linear ±Inf queries.
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
| `kgrid.k_max` | `KGrid.k_max` | float (1e-9 rel) | `test_kwave_signal.py` | ✅ |
| `filterTimeSeries(Filter, Filter, x, 'ZeroPhase', true, 'TransitionWidth', TW, 'PPW', 2)` (struct kgrid/medium) and option sweep | `filter_time_series` | float (1e-9 rel) | `test_kwave_signal.py` | ✅ (B-60 workaround) |
| `getWin(N, 'Tukey', 'Param', p)` | `get_win` | float (1e-9 rel) | `test_kwave_signal.py` | ✅ |

Notes:

- `getWin` is only called in `get_transducer_integration_points.m` inside
  `if apodization_Z == true` with `apodization_Z = false` hard-coded: dead code.
  `get_win` is a thin pass-through with an oracle test; port the call site only
  if something reaches it.
- Not covered: `kspaceFirstOrder3DG` / CUDA (`device="gpu"`, `device_num`), CuPy
  (`backend="python", device="gpu"`), anisotropic
  grids with pressure sources (refused), `record_start_index` (PROTEUS does not set it),
  source signals longer than `Nt` (refused), a `2*pml >= N` grid against MATLAB (B-61;
  only cpp vs python). An empty sensor mask raises; `run_simulation.m` returns
  `zeros(0, Nt)` itself, which its port must reproduce.

## Settings (`pyproteus.config`, `pyproteus.data`)

Compared as flattened trees: every node's class, size, field names **in order** and values (bit-exact, sign of zero included).

| MATLAB | Python | Class | Oracle test | Status |
|---|---|---|---|---|
| `reset_acquisition`, `reset_simulation_parameters`, `reset_medium` | `config.reset_*` | exact | `tests/oracle/test_config.py` | ✅ |
| `reset_microbubble` | `config.reset_microbubble` | exact; `Distribution.Probabilities` 1e-12 rel (7e-16 seen, B-31/B-32) | `test_config.py` | ✅ |
| `reset_transducer` | `config.reset_transducer` | exact; `TransmitImpulseResponse`, `ReceiveImpulseResponse`: default pending M1 (signal leaves: `estimate_impulse_response`) | `test_config.py` | 🟡 |
| `reset_transmit` (all catalogue configurations, missing `Configuration`) | `config.reset_transmit` | exact; `VoltageSignal`, `PressureSignal`, `MechanicalIndex`: default pending M1 (signal leaves: `get_voltage_signal`, `get_pressure_signal`) | `test_config.py` | 🟡 |
| `reset_geometry` (renal tree, mouse brain, no `Folder`, existing fields kept) | `config.reset_geometry` | exact | `test_config.py` | ✅ |
| `get_transducer_properties` (5 types, `Custom linear`, unknown) | `transducer.get_transducer_properties` | exact | `test_config.py` | ✅ |
| `load_transducer_model` (PROTEUS both forms, Verasonics, v7.3, single/int IR, errors) | `transducer.load_transducer_model` | exact | `test_config.py` | ✅ |
| `get_gas_properties`, `assign_medium_properties` (all tissues, substrings, Q-30), `assign_liquid_properties`, `compute_sound_speed_extrema` (NaN, Inf) | `microbubble.get_gas_properties`, `medium.*` | exact | `test_config.py` | ✅ |
| `compute_polydisperse_distribution` | `microbubble.compute_polydisperse_distribution` | 1e-12 rel (`R`, `P`; B-31/B-32) | `test_config.py` | ✅ |
| `linspace(d1,d2,n)` | `microbubble.linspace` | 1e-12 rel | `test_config.py::test_linspace` | ✅ |
| `load` of settings files (`write_to_file`, v6/v7/v7.3) | `config.load_settings_mat` | exact (incl. extra fields, cells, struct arrays) | `test_config.py::test_load_settings_mat` | ✅ |
| `load` of ground-truth frames, `FlowSimulationParameters` (v7, v7.3), frame file names | `config.load_frame`, `load_flow_simulation_parameters`, `frame_file_name` | exact | `test_config.py` | ✅ |
| `load` of every shipped `.mat` (example data, models, constants, `tests/baseline`), also re-saved as v7.3 | `config.loadmat` | exact | `test_config.py::test_loadmat_shipped_files` | ✅ |
| shipped constants (`tissue_properties`, `fit_SigmaR_*`, `*TransferFunctionFit`, `GeometryProperties`, `C1-6D`) | `pyproteus.data.load` | exact (SHA-256 of source checked) | `tests/oracle/test_data.py` | ✅ |

Notes:

- Models store MATLAB doubles and logicals (what PROTEUS writes). A file with other classes in typed fields (e.g. a `single` `Pitch`, an `int32` `SamplingRate`) loads with those fields converted to double; untyped (extra) fields keep their class. The raw reader `config.loadmat` keeps every class.
- Field-less structs (`struct()`, `repmat(struct(),2,1)`, `struct('a',{})`) and v7.3 `canonical empty` values load like MATLAB (`test_load_settings_mat`).
- **Default pending M1 (signal leaves):** the 5 fields above are `None` (absent) in the Python defaults and listed in `config.PENDING_FIELDS`; the oracle test skips exactly these. The M1 session that ports `estimate_impulse_response`, `get_voltage_signal` and `get_pressure_signal` completes `reset_transducer`/`reset_transmit`, empties `PENDING_FIELDS` and turns these rows ✅.
- Looser class, named fields only: `Microbubble.Distribution.Radii` and `Probabilities` (computed with `linspace`, `exp`, `sum`) are compared at 1e-12; every other default field (constants, catalogue values, plain arithmetic) and all loaded data are exact.
- `linspace` is the textbook formula (`d1 + k*(d2-d1)/(n-1)`, last element `d2`), compared at 1e-12 rel. Per the developer's policy (2026-11-22) no code matches MATLAB's ulps or summation order; computed values are compared within tolerance, loaded data and constants exactly.
- Settings JSON (`config.save_settings_json`) is lossless for every value tree (round trip tested on real settings files).

## Transducer geometry and transmit (`pyproteus.transducer`, M1)

| MATLAB | Python | Class | Oracle test | Status |
|---|---|---|---|---|
| `El = get_element_geometry(Transducer)` (linear, curved, `RCA`/other → linear; catalogue incl. C1-6D model file; `reset_transducer`; errors) | `transducer.get_element_geometry` → `ElementGeometry` | float (1e-9 rel; positions atol 1e-14·L, L = max(N·Pitch, R)); measured ≤ 2.2e-16·L on 500 random curved arrays (one-off script) | `tests/oracle/test_transducer_geometry.py` | ✅ |

Notes:

- Divergence (D-15): MATLAB returns a struct `El` with N×1 fields; pyproteus returns a frozen
  `ElementGeometry` dataclass with 1-D arrays and snake_case names (`radius`, `apex`, `center`,
  `theta_max`, plus `curved`, `n_elements`). `ElementGeometry.to_struct()` gives the MATLAB
  struct (field names, N×1 columns); the oracle test compares through it.
- Divergence (input validation): `NumberOfElements` must be a non-negative integer, else
  `ValueError`. MATLAB errors for a fractional N on linear arrays (`zeros(2.5,1)`) but on
  curved arrays silently uses `1:floor(N)` with `(N+1)/2` from the fractional N; a negative N
  gives empty arrays in MATLAB. PROTEUS never passes such values (catalogue/GUI integers).
- Errors are `ValueError` where MATLAB raises (`get_element_geometry:invalidRadius`, missing
  fields); the oracle test checks MATLAB raises in the same cases.
- The Hypothesis test runs `PYPROTEUS_ORACLE_EXAMPLES` batches (default 25) of up to 20 arrays;
  a 300-batch sweep passed.
- `is_curved(transducer)` is the one exact-match test for `'Curved'` (Q-80), shared by the ports
  of `compute_delays`, `compute_simulation_domain` and the acoustic module.
- Positions are differences of `R·cos` terms of the size of the array, so their absolute floor
  scales with `max(N·Pitch, R)`, not with the (possibly tiny) value: for a nearly flat array
  the apex `R(1 - cos θmax)` carries an absolute error of order eps·R on both sides.

## ODE solvers (`pyproteus.numerics`)

Class D-13 (PR #10; replaces the old 1e-8 bubble class): per component `max|y_py - y_matlab| ≤ 10·RelTol·max|y_matlab| + AbsTol`, at `tspan` points (or the final value for a 2-point `tspan`). Step sequences are not compared.

| MATLAB | Python | Class | Oracle test | Status |
|---|---|---|---|---|
| `[t,y] = ode45(f, tspan, y0, opts)` on smooth problems | `ode45` | D-13 (measured worst: 38 % of bound vs MATLAB, 83 % vs DOP853 reference) | `test_ode45.py::test_solution_matches_matlab` | ✅ |
| Stacked Rayleigh–Plesset-like batch, PROTEUS options | `ode45` | D-13 (measured ≤ 4.5e-6 rel at RelTol 1e-6) | `test_ode45.py::test_rayleigh_plesset_batch` | ✅ |
| `ode23` + events | — | — | — | ⬜ (M4) |

Notes: options covered `RelTol`, `AbsTol` (scalar, vector), `NormControl`, `MaxStep`, `InitialStep`, `Refine`, `OutputFcn`; `BDF`/`MaxOrder`/`Stats` accepted and ignored; other options raise. Forward and backward; 2-point and many-point `tspan`.

## Delay-and-sum (`pyproteus.reconstruction`, M1)

Classes per D-13. Sample indices are compared per (pixel, element) entry of the
DAS matrix (MATLAB returns `find(M)`); weights where both sides keep the same
sample at the float class. Tie cases use dyadic coordinates with `c = Fs = 1`,
so `(t_del - t(1))*Fs` hits exact .5 values on both sides (half away from zero).

| MATLAB | Python | Class | Oracle test (`tests/oracle/test_das.py`) | Status |
|---|---|---|---|---|
| `das_transmit_distance` (linear, curved; plane, diverging, focused, focus 0; steered) | `das_transmit_distance` | float | `test_das_transmit_distance` | ✅ |
| `compute_das_matrix` (plane, diverging; even/odd elements; `X ≤ 0`; out-of-range samples) | `compute_das_matrix` | statistical (measured 100 %: 5.28 M entries on a λ/5, 64-element image; all tie cases exact) | `test_compute_das_matrix_*` | ✅ |
| `compute_das_matrix_general` (linear, curved; plane, steered, diverging, focused, curved diverging; cosine/binary; `axis_delay`; defaults) | `compute_das_matrix_general` | statistical (measured 100 % incl. 500 Hypothesis cases); weights float | `test_compute_das_matrix_general*` | ✅ (B-100: `-0` pixel) |
| `load_RF_data` (v7 and v7.3, single and double, all pulsing schemes, frame order) | `load_rf_data`, `iter_rf_frames` | exact | `test_load_rf_data*` | ✅ |
| TGC expression of `DAS_reconstruction` (incl. Q-101) | `time_gain_compensation` | float | `test_time_gain_compensation` | ✅ |
| colon `a:d:b` (pixel axes) | `reconstruction.colon` | float values, exact length on tested cases | `test_colon` | ✅ |
| `DAS_reconstruction` (double RF; v7.3 + pulse inversion + diverging + full-simulation k-Wave correction; single RF) | `das_reconstruction` | float on the linear image `10^(IMG/20)` for double RF (measured 4.4e-16); single RF ≤ 1e-6 abs (measured 5.5e-8) | `test_das_reconstruction` | ✅ |
| `initialize_das_parameters`, `DAS_reconstruction_general` | — | — | — | ⬜ (wait for `get_element_geometry`) |

Design differences with MATLAB (D-15; results within the classes above):

- **DAS matrix built in pixel blocks.** MATLAB forms dense `Npixels`-by-`Nelem` delays, indices, angles and weights for the whole image before `sparse`. pyproteus builds the matrix in blocks of pixels (`block_size`, default 16384), so peak memory is the sparse matrix plus a few `block_size*Nelem` arrays. Output is a `scipy.sparse.csr_array` (`cupyx.scipy.sparse` has the same type for a later GPU path). Each (pixel, element) pair maps to one column, so the duplicate accumulation of MATLAB `sparse` cannot occur (asserted in the oracle test), and zero weights are not stored (MATLAB drops them too).
- **Frames streamed.** `iter_rf_frames` reads each `Frame*.mat` once, only the `RF` variable; `das_reconstruction` beamforms blocks of frames as they are read. MATLAB reads the first file twice and holds all frames plus the complex `Nt*Nelem`-by-`Nframes` matrix. `load_rf_data` still returns the stacked array for callers that want it.
- **No transpose.** The column order of the DAS matrix (`sample + Nt*element`) is the C-order flattening of an `Nelem`-by-`Nt` frame, so RF goes into `M @ ...` without MATLAB's `permute`.
- **Double precision throughout.** MATLAB keeps `single` RF (the PROTEUS default) in `single` through the TGC and `hilbert`; pyproteus computes in double. Effect: single-precision rounding of the input chain (5.5e-8 of the image peak on the test case).
- **Settings object accepted.** `das_reconstruction(folder, settings)` takes a settings `.mat` path (as MATLAB) or a `Settings` object.

Notes:

- `colon` counts `(b-a)/d` within 1e-10 (relative) below an integer as that integer, so a range meant to end on `b` keeps its last element; values are `a + k*d` (MATLAB may differ in the last bits). Used for the pixel axes; a shared `numerics.colon` would serve `define_grid` too (proposed to the main session).
- The randomised general-matrix test excludes `-0.0` coordinates (B-100).
