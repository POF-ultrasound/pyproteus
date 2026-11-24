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
| `a:d:b` (colon operator) | `colon` | float values; length equal except `b` within ~4 eps of a grid point below it (own count rule, see note) | `tests/oracle/test_das.py::test_colon` | ✅ |
| `ifft(X,n,dim,'symmetric')` | `ifft_symmetric` | float (rel. to signal max) | `test_numerics_signal.py` | ✅ |
| `hilbert(x)` (vectors along length, matrices along columns) | `hilbert` | float (rel. to signal max) | `test_numerics_signal.py` | ✅ |
| `conv(u,v,shape)`, `convn(A,B,shape)` | `conv`, `convn` | exact on small integers, else float | `test_numerics_signal.py` | ✅ |
| `rng(seed,'twister')`, `rand`, `randi` | `rng`, `MatlabRNG.rand`, `MatlabRNG.randi` | float (same MT19937 stream) | `test_numerics_random.py` | ✅ |
| `sind(x)`, `cosd(x)` (exact at integer multiples of 90°, documented) | `sind`, `cosd` (exact reduction modulo 360 and to the nearest multiple of 90; exact zeros are +0) | exact at multiples of 90°, else float (1e-9 rel + 1e-15 abs) | `test_numerics_trig.py` | ✅ |
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
| `reset_transducer` | `config.reset_transducer` | exact; `TransmitImpulseResponse`, `ReceiveImpulseResponse` 1e-7 rel. to max (D-17, B-80) | `test_config.py` | ✅ |
| `reset_transmit` (all catalogue configurations, missing `Configuration`) | `config.reset_transmit` | exact (`VoltageSignal` too); `PressureSignal` 1e-7 rel. to max (D-17, B-80), `MechanicalIndex` 1e-12 | `test_config.py` | ✅ |
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
- The signal defaults (impulse responses, voltage and pressure pulses, mechanical index) are filled since M1; `config.PENDING_FIELDS` is empty. Impulse responses and pressure pulses are compared at the D-17 class (1e-7 rel. to max).
- Looser class, named fields only: `Microbubble.Distribution.Radii` and `Probabilities` (computed with `linspace`, `exp`, `sum`) are compared at 1e-12; every other default field (constants, catalogue values, plain arithmetic) and all loaded data are exact.
- `linspace` is the textbook formula (`d1 + k*(d2-d1)/(n-1)`, last element `d2`), compared at 1e-12 rel. Per the developer's policy (2026-11-22) no code matches MATLAB's ulps or summation order; computed values are compared within tolerance, loaded data and constants exactly.
- Settings JSON (`config.save_settings_json`) is lossless for every value tree (round trip tested on real settings files).

## Transducer geometry and transmit (`pyproteus.transducer`, M1)

| MATLAB | Python | Class | Oracle test | Status |
|---|---|---|---|---|
| `El = get_element_geometry(Transducer)` (linear, curved, `RCA`/other → linear; catalogue incl. C1-6D model file; `reset_transducer`; errors) | `transducer.get_element_geometry` → `ElementGeometry` | float (1e-9 rel; positions atol 1e-14·L, L = max(N·Pitch, R)); measured ≤ 2.2e-16·L on 500 random curved arrays (one-off script) | `tests/oracle/test_transducer_geometry.py` | ✅ |
| `Transmit = compute_delays(Transmit, Transducer, Medium)` (linear, curved, RCA; focused, diverging, plane/diverging wave, NaN focus, angles incl. ±90/180; errors) | `transducer.compute_delays` (+ `transmit_delays` on plain numbers) | float (1e-9 rel; atol 1e-14·L/c, L = max(N·Pitch, \|f\|, R)) | `tests/oracle/test_transducer_domain_delays.py` | ✅ |
| `Geometry = compute_simulation_domain(Geometry, Transducer, Transmit)` (auto and manual domain, both shipped geometries, start depth in front of/behind a curved apex, catalogue + odd/single/curved arrays, `Custom`, odd RCA error, warning) | `transducer.compute_simulation_domain` (+ `compute_transducer_vertices`, `compute_transducer_projection`, `compute_end_depth`) | float (1e-9 rel; atol 1e-14·max\|output\|); Domain field order and the warning compared | `test_transducer_domain_delays.py` | ✅ |

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
- `sind`/`cosd` come from `numerics` (see the numerics table).
  This matters in `compute_transducer_projection`, which divides by `LateralFocus*cosd(Angle)`:
  at `Angle = 90` the beam lines are ±Inf/NaN on both sides, and the domain bounds ignore NaN
  like MATLAB's `max`/`min` (`np.fmax`/`np.fmin`).
- Divergence (input validation): `compute_delays` raises for an odd RCA element count (MATLAB
  silently builds `floor(N/2)` delays centred on the wrong element and returns the wrong
  length) and for a non-integer count; `compute_simulation_domain` already raised for odd RCA.
- Divergence (API, D-15): `transmit_delays(transducer, focus, angle, c)` returns the delays
  as a 1-D array without the settings structs; `compute_delays` wraps it and stores the 1×N
  row in a copy of `Transmit`. The domain helpers MATLAB keeps as subfunctions are public.
  `compute_simulation_domain` returns a copy of `Geometry` (MATLAB value semantics).
- `compute_transducer_projection` with `Configuration = 'Custom'` returns before reading
  `Pitch`, `ElementWidth`, `ElevationFocus`, `LateralFocus`, `Angle`; MATLAB reads them first
  and errors if one is missing. No effect on results.
- Positions are differences of `R·cos` terms of the size of the array, so their absolute floor
  scales with `max(N·Pitch, R)`, not with the (possibly tiny) value: for a nearly flat array
  the apex `R(1 - cos θmax)` carries an absolute error of order eps·R on both sides.

## Transmit signals (`pyproteus.transmit`, `pyproteus.transducer`, M1)

| MATLAB | Python | Class | Oracle test | Status |
|---|---|---|---|---|
| `get_voltage_signal` (three-level, cosine envelope rectangular/Hann, voltage/pressure amplitude mode, unknown type/envelope, `round` ties in the length) | `transmit.get_voltage_signal` (+ `tri_level_signal`, `rect_pulse`, `hann_pulse`) | three-level: statistical (equal except at threshold ties, none seen); cosine pulses: float (1e-9 rel + 1e-14·max) | `tests/oracle/test_transmit_signals.py` | ✅ |
| `get_pressure_signal` (convolution with MATLAB's impulse responses: P4-1 estimate, C1-6D model, L22-14v; pressure/voltage/other mode; `Custom (pressure)` row/column/NaN) | `transmit.get_pressure_signal`, `compute_mechanical_index` | float (1e-9 rel + 1e-14·max) | `test_transmit_signals.py` | ✅ |
| `estimate_impulse_response` (catalogue bands, other rates, narrow bands) | `transducer.estimate_impulse_response`, `butterworth_impulse_response` | D-17: vs MATLAB 1e-7 rel. to max on catalogue-like bands (measured ≤ 1.5e-8; narrow bands 1–2 % reported, not gated, B-80/Q-83); vs zpk partial-fraction reference 1e-12 (measured ≤ 5e-14; narrow bands 1e-11, measured 2.7e-12) | `test_transmit_signals.py` | ✅ |

Notes:

- Divergence (D-15, D-17, B-80): the Butterworth bandpass is designed and applied in second-order
  sections (`scipy.signal.butter(..., output='sos')`, `sosfilt`), not in the (b, a) form +
  `impz` of PROTEUS. The (b, a) form of this filter is ill-conditioned; the sections are
  accurate to ~1e-14 of max|IR|, MATLAB's result to 1e-8 (P4-1) and only to percent level for
  narrow bands at high sampling rates. No Signal Processing Toolbox code is translated.
- Divergence (shape): `get_pressure_signal` always returns a 1×N row from the convolution.
  MATLAB `conv` can return a column for column inputs; PROTEUS's signals are rows
  (`load_transducer_model` stores impulse responses as rows). `Custom (pressure)` signals keep
  their shape, as in MATLAB.
- Three-level pulses threshold `sin` samples at `sin(0.33·pi/2)`; a sample within an ulp of the
  threshold may fall on the other side in MATLAB. The test accepts differences only at such ties
  (within 1e-12); none occurred in the fixed cases or a 300-batch Hypothesis sweep.

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
| `compute_das_matrix` (plane, diverging; even/odd elements; `X ≤ 0`, `X = 0`; samples off both ends of the record, asserted present; exact .5 ties inside and at -0.5 / Nt-0.5) | `compute_das_matrix` | statistical (measured 100 %: 5.28 M entries on a λ/5, 64-element image; all tie cases exact) | `test_compute_das_matrix_*` | ✅ |
| `compute_das_matrix_general` (linear, curved; plane, steered, diverging, focused, curved diverging; cosine/binary; `axis_delay`; defaults) | `compute_das_matrix_general` | statistical (measured 100 % incl. 500 Hypothesis cases); weights float | `test_compute_das_matrix_general*` | ✅ (B-100: `-0` pixel) |
| `load_RF_data` (v7 and v7.3, single and double, all pulsing schemes, frame order) | `load_rf_data`, `iter_rf_frames` | exact | `test_load_rf_data*` | ✅ |
| TGC expression of `DAS_reconstruction` | `time_gain_compensation` | float where MATLAB's TGC is valid (`max(t) > \|min(t)\|`); D-18 divergence otherwise | `test_time_gain_compensation`, `test_time_gain_compensation_q101_fixed`, `test_das_reconstruction_q101_regime` | ✅ |
| `DAS_reconstruction` (double RF; v7.3 + pulse inversion + diverging + full-simulation k-Wave correction; single RF) | `das_reconstruction` | float on the linear image `10^(IMG/20)` for double RF (measured 4.4e-16); single RF ≤ 1e-6 abs (measured 5.5e-8; class accepted by the developer, D-18); short record (samples past the end) | `test_das_reconstruction` | ✅ |
| `initialize_das_parameters` (linear even/odd, curved; steered, diverging, focused; lens/k-Wave variants; `Transmit.Apodization`; errors; delay-mismatch warning) | `initialize_das_parameters` (warning: `DasDelayMismatchWarning`) | float (axes, time axis, element geometry); exact length of axes | `test_initialize_das_parameters*` | ✅ |
| `DAS_reconstruction_general` (linear 16/15 elements, curved 17/16; plane, steered diverging, focused; v7/v7.3; single RF) | `das_reconstruction_general` | as `DAS_reconstruction` (measured ≤ 6.7e-16 double, 9.8e-8 single) | `test_das_reconstruction_general` | ✅ |

Result-changing divergence (developer decision D-18):

- **TGC normalisation (was Q-101).** MATLAB divides by `max(sqrt(t))` over the complex roots; when `|min(t)| ≥ max(t)` (time to peak + lens delay longer than the record after it) that maximum is imaginary and the whole image is NaN. pyproteus divides by `sqrt(max(t))`. Identical to MATLAB whenever MATLAB's output is valid (oracle-tested at the float class, including `t(1) = -0.98·t(end)`); in the degenerate regime MATLAB's TGC and image are checked to be degenerate and pyproteus' to be the real-root formula and a finite image.

Design differences with MATLAB (D-15; results within the classes above):

- **DAS matrix built in pixel blocks.** MATLAB forms dense `Npixels`-by-`Nelem` delays, indices, angles and weights for the whole image before `sparse`. pyproteus builds the matrix in blocks of pixels (`block_size`, default 16384), so peak memory is the sparse matrix plus a few `block_size*Nelem` arrays. Output is a `scipy.sparse.csr_array` (`cupyx.scipy.sparse` has the same type for a later GPU path). Each (pixel, element) pair maps to one column, so the duplicate accumulation of MATLAB `sparse` cannot occur (asserted in the oracle test), and zero weights are not stored (MATLAB drops them too).
- **Frames streamed.** `iter_rf_frames` reads each `Frame*.mat` once, only the `RF` variable; `das_reconstruction` beamforms blocks of frames as they are read. MATLAB reads the first file twice and holds all frames plus the complex `Nt*Nelem`-by-`Nframes` matrix. `load_rf_data` still returns the stacked array for callers that want it.
- **No transpose.** The column order of the DAS matrix (`sample + Nt*element`) is the C-order flattening of an `Nelem`-by-`Nt` frame, so RF goes into `M @ ...` without MATLAB's `permute`.
- **Double precision throughout.** MATLAB keeps `single` RF (the PROTEUS default) in `single` through the TGC and `hilbert`; pyproteus computes in double. Effect: single-precision rounding of the input chain (5.5e-8 of the image peak on the test case).
- **Settings object accepted.** `das_reconstruction(folder, settings)` takes a settings `.mat` path (as MATLAB) or a `Settings` object.
- **Frame axis always present.** With one frame, `load_rf_data`, `das_reconstruction` and `das_reconstruction_general` return `(..., 1)` arrays; MATLAB drops the trailing singleton.
- **Missing fields raise.** `transducer.azimuth` (general matrix) and `SimulationParameters.HybridSimulation` are required (`ValueError`), as MATLAB errors on them; a silent default would give NaN weights or the wrong k-Wave correction.
- **Frame file names are matched case-sensitively** (`Frame*.mat`); MATLAB `dir` on a case-insensitive file system (macOS, Windows) would also list `frame_1.MAT`.

Notes:

- `numerics.colon` (pixel axes; shared with `define_grid`): MATLAB documents `a:d:b = [a, a+d, ..., a+m*d]`, `m = fix((b-a)/d)`. The count rule is our own (not probed): `(b-a)/d` below an integer by less than `4·eps·max(|q|, max(|a|,|b|)/|d|, 1)` (its rounding error) counts as that integer, so `0:0.1:0.3` has 4 elements and `0:1:3-1e-14` has 3. A `b` within that window below a grid point may give one more element than MATLAB.
- The randomised general-matrix test excludes `-0.0` coordinates (B-100).

## Grid, medium, transducer integration (`pyproteus.acoustic`, `pyproteus.medium`, M1)

| MATLAB | Python | Class | Oracle test | Status |
|---|---|---|---|---|
| `optimize_grid_size(N, range, maxPrime)`, `max(factor(n))` | `acoustic.optimize_grid_size`, `acoustic.largest_prime_factor` | exact (integer search on exact inputs) | `test_acoustic_grid.py` | ✅ |
| `define_grid(SimulationParameters, Geometry)` (`MaxPrime`/`PMLmin` present or absent) | `acoustic.define_grid` → `Grid` | default `pml_rounding="ceil"`: sizes/PML exact, coordinates float (MATLAB's grid, Q-90 reproduced); `"floor"` (Q-90 fix): full sizes exact, documented core +2 / PML −1 on odd-padding axes | `test_acoustic_grid.py` (both modes; both padding parities asserted to occur) | ✅ |
| `random_truncated_normal(cutoff, sz)` after `rng(seed,'twister')` | `medium.random_truncated_normal(cutoff, sz, rng)` | float (measured ≤ 2.8e-13 rel, ≤ 1.8e-15 abs; same number of uniforms) | `test_medium_random.py` (also PROTEUS' recorded toolbox draws) | ✅ (infinite cutoff: ⬜, see notes) |
| `voxelize_media_points(MP, Grid)` | `medium.voxelize_media_points` (0-based) | statistical (measured 100 % equal); points and exclusion mask exact | `test_medium_voxelize.py` | ✅ |
| `evaluate_delta_function(x, xi, dx, N)`, also `xi` d×M (implicit expansion, `update_sensor_fast.m`) | `acoustic.evaluate_delta_function` | float (atol 1e-12 × max) | `test_acoustic_signals.py` | ✅ |
| `sinc_interpolation(X, V, Xq)` (vector, row, `len(X)×k` incl. k = 1; single `V`) | `acoustic.sinc_interpolation` | float (atol 1e-12 × max); single `V`: single class | `test_acoustic_signals.py` | ✅ |
| `resample_signal(p, fs_old, fs_new, false)` | `acoustic.resample_signal` | float (measured ≤ 1.3e-15 × max) | `test_acoustic_signals.py` | ✅ |
| `get_transducer_integration_delays` (linear, curved, RCA, Custom; finite/infinite/zero focus) | `acoustic.get_transducer_integration_delays` | float | `test_acoustic_transducer.py` | ✅ |
| `compute_travel_times` (linear and vertex branches, steered/focused transmits) | `acoustic.compute_travel_times` | float | `test_acoustic_transducer.py`, `test_acoustic_integration_points.py` | ✅ |
| `get_transducer_integration_points` (linear, curved, RCA incl. odd-N error, Custom incl. orientation warning) | `acoustic.get_transducer_integration_points` | counts exact; points, weights float | `test_acoustic_integration_points.py` | ✅ |
| `define_medium` (background, `alpha_mode`, curved coupling region, vessel mask from MATLAB `VOXELISE`) | `medium.define_medium(..., rng=, vessel=)` | float | `test_medium_define.py` | ✅ (STL voxelisation: M4) |

Notes:

- Everything is compared on settings built by MATLAB itself (`reset_*`, `compute_simulation_domain`, `update_simulation_parameters`, `write_to_file`'s grid size and delays) and read into the Python models from a `.mat` file, plus hand-made tie cases: domain bounds on and half-way between grid nodes, one-point and empty axes, odd/non-cubic grids, `MaxPrime` 2/3/7, single elements, element heights at exact multiples of the elevation spacing, sinc/resample queries outside the sample range and on samples, `floor` ties in the resampled length, voxel ties at `.5`.
- Integer results (`ceil`/`floor`/`round` of IEEE quotients of identical inputs) agree 100 % in every test; the voxel test still asserts the statistical class (≥ 99.9 %) and prints any difference.
- `random_truncated_normal` with an infinite cutoff (MATLAB: untruncated `randn`) raises `NotImplementedError`: MATLAB's `randn` algorithm is not documented, so its stream is not reproduced (ROADMAP §5). Default settings use a finite cutoff.
- `define_medium` with `Geometry.EmbedVessel` (true for both shipped geometries) needs `READ_stl`/`VOXELISE` (M4). Until then the caller passes the vessel mask (`vessel=`); `NotImplementedError` (pointing to M4) is raised only when `EmbedVessel` is true and no mask is given. The oracle test passes MATLAB's `VOXELISE` output for an asymmetric, off-centre box STL and asserts the mask contract M4's voxeliser must meet: `bool`, shape `(Nx, Ny, Nz)`, indexed `[ix, iy, iz]` (checked against the box computed from the grid coordinates, so a transposed or C/Fortran-reshuffled mask fails). `test_define_medium_shipped_geometries` does the same for the renal tree (default settings, 2 points per wavelength) and mouse brain (10 µm grid over its bounding box: its ~15 µm vessels miss every voxel centre of an ultrasound grid) STLs when `PYPROTEUS_GEOMETRIES_PATH` points to them; it checks each file's SHA-256 against `src/pyproteus/io/geometry_registry.txt` and skips without the files (`python -m tools.fetch_geometries` fetches them). The STLs are members of Zenodo 10573662's `*_no_vtu.zip` archives (not in the oracle checkout); both cases pass locally.

Design divergences (D-15; results unchanged unless marked, parity checked as above):

| Where | MATLAB | pyproteus | Why |
|---|---|---|---|
| `define_grid` `pml_rounding` option (D-19, developer) | PML `ceil((M - N)/2)` only (Q-90) | `pml_rounding="ceil"` (default until 1.0, MATLAB's grid) or `"floor"` (Q-90 fix: the odd point stays in the core, `N + 1` points, PML one thinner, same `M`) | the fix is available now without breaking default-case k-Wave/RF parity or MATLAB references; the default flips to `"floor"` in 1.x. Measured: the default settings of both shipped geometries pad every axis oddly (renal tree core 446×298×178 MATLAB/`ceil` vs 448×300×180 `floor`). Code building the k-Wave grid must use `Grid.Nx/Ny/Nz` and `Grid.PML` (never recompute them) |
| `define_grid` | struct `pgrid`, 1×N coordinate rows | frozen `Grid` dataclass (`PML` record), 1-D coordinates, `shape` property | typed, immutable, no 1-D/2-D ambiguity |
| `optimize_grid_size` | `maxPrime < 2` loops forever | raises `ValueError` | fail fast |
| `optimize_grid_size` range | `N + (min:max)` | `numerics.colon` | shared MATLAB colon semantics |
| `random_truncated_normal`, `define_medium` | global `rand` stream | explicit `rng` (`numerics.MatlabRNG`) argument | no global state (ROADMAP §7); seeds per job |
| `sinc_interpolation`, `resample_signal` | full `numel(Xq) × numel(X)` `ndgrid` matrix | same matrix built in row blocks of at most 2^22 elements (32 MB) | bounded memory for long mass-source signals (an FFT-based resampler is a post-parity candidate) |
| `resample_signal(..., disp_fig=true)` | plots | raises `NotImplementedError` | no plotting in the library |
| `voxelize_media_points` | `[MP, MP_nodes, MB_idx, idx_exclude]`, 1-based | `VoxelizedPoints` named tuple, 0-based nodes and column-major linear indices | pyproteus index convention |
| `get_transducer_integration_points` | loop over elements | vectorised over elements | speed for large arrays |
| `get_transducer_integration_points` | `apodization_Z` branch (`getWin` Tukey) behind a hard-coded `false` | not ported | dead code |
| `define_medium` `alpha_coeff` (always) and `BonA` (no vessel voxel) | full `Nx×Ny×Nz` arrays of one value | read-only `np.broadcast_to` views: same values, shape and indexing, no memory | two fewer full 3-D double arrays; writing to them raises instead of silently diverging |
| `define_medium` vessel | reads and voxelises the STL inside `define_medium` | mask passed in (`vessel=`, bool `(Nx, Ny, Nz)`); voxelisation stays outside (M4) | separates geometry from medium; default path testable before M4 (approved by the main session) |
| `define_medium` | builds the STL path first: `Geometry.GeometriesPath`/`Folder` must exist even without a vessel | not needed when no STL is read | no spurious error |
| `sinc_interpolation` with single `V` (k-Wave sensor data) | double × single product: single result, accumulated in single | `float32` result, accumulated in double and rounded once | more accurate; agrees with MATLAB at the single-precision class (oracle: ≤ 1e-5 × max), not 1e-9 |
| `define_grid`, `define_medium`: missing fields | `SensorOnGrid` absent, `EmbedVessel` absent or `Medium.Vessel` absent → error | `sensor_on_grid=None`; `EmbedVessel` absent = false; `Medium.Vessel` only read when the mask has vessel voxels | lenient where MATLAB fails; no effect on results |
| NaN in `max`/`min` (integration delays, Custom orientation check, travel times) | `max`/`min` skip NaN | `np.nanmax`/`np.nanmin` | same semantics (all-NaN → NaN) |
| `get_transducer_integration_*`, `compute_travel_times` | modify and return the struct | return an updated copy (`copy_struct`); `run_param` is a dict | value semantics, as MATLAB |
