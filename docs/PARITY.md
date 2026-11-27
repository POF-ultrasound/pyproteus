# Parity with MATLAB PROTEUS

Status of each ported function against the MATLAB oracle (PROTEUS commit
`8d37938`, MATLAB R2025a). Comparison classes from ROADMAP §4:

Classes are D-13 (medium tolerance; parity is behaviour and results, not MATLAB's bits, D-12):

- **float**: ≤ 1e-9 rel (small atol near zero): geometry, delays, weights, signals, interpolation, truncated normal, random draws
- **statistical**: integers after rounding (indices, masks), ≥ 99.9 % equal, differences reported, downstream effect at the float/RF class
- **exact**: only where it falls out naturally (data read from files, documented integer semantics on exact inputs, storage round trips)
- **ODE**: ≈ 10·RelTol rel + AbsTol abs; **k-Wave** ≤ 1e-5; **end-to-end RF** ≤ 1e-4 (ROADMAP §4)


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
| `[R,P] = compute_gaussian_distribution(Microbubble)` (mean radius on a grid node, at/outside the ends, PDI 0.1–1e4, negative, 0; defaults) | `microbubble.compute_gaussian_distribution` (+ `gaussian_distribution(mean_radius, pdi)`) | float, relative everywhere (1e-9 rel + 1e-300; zero and NaN positions equal; measured ≤ 1.1e-15) | `tests/oracle/test_microbubble_distribution.py` | ✅ |
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

## Microbubble model and mass source (`pyproteus.microbubble`, `pyproteus.acoustic.mass_source`, M2)

Bubble-response class (D-22, absolute term amended 2026-11-25 for B-123): per bubble
`max|Δy| ≤ 1e-3·max|y| + 3e-5` for the solver states `y = x = R/R0 - 1` and `y = xdot = Rdot·T/R0`;
for the mass source the same per bubble with the absolute term scaled by `4πρR0³/T`. The D-13 bound
`10·RelTol·max|y| + AbsTol` is missed by MATLAB against itself (bubble alone vs in a batch: 17–34×),
and both solvers are ~1e-4 of the excursion from a DOP853 rtol 1e-13 reference (MATLAB ≤ 1.2e-4,
pyproteus ≤ 2.3e-4, one quiet bubble in a loud batch 6.8e-4). Accuracy is also tested in CI against
that reference (≤ 1e-3).

| MATLAB | Python | Class | Oracle test | Status |
|---|---|---|---|---|
| `getMaterialProperties` | `microbubble.get_material_properties` | exact | `test_microbubble_model.py` | ✅ |
| `getShellProperties` (Marmottant, SegersTable; warning range) | `get_shell_properties` | float | `test_microbubble_model.py` | ✅ |
| `getShellProperties`, `calc_surface_tension`, `find_domain_boundaries`, `find_A_m0` for the `'Segers'` polynomial | same | B-120 class: roots 1e-3 rel, σ 2e-4 N/m abs, χ-derived parameters 1e-2 rel | `test_microbubble_model.py` | 🟡 (B-120) |
| `calc_surface_tension` (Marmottant, SegersTable; buckled/elastic/ruptured, NaN, exact table ends Q-120) | `calc_surface_tension` | float (1e-9 rel + 1e-15) | `test_microbubble_model.py` | ✅ |
| `calc_thermal_damp` (Hypothesis: R0 0.3–10 µm, σ0 0–0.072, f 0.1–20 MHz) | `calc_thermal_damp` | float | `test_microbubble_model.py` | ✅ |
| `getEqParam` (Prosperetti/Adiabatic/Isothermal × shell models, equal radii) | `get_eq_param` | float | `test_microbubble_model.py` | ✅ |
| `compress_bubble` (over/underpressure, Marmottant, SegersTable) | `compress_bubble` | float | `test_microbubble_model.py` (needs the `boolean` shim, B-122) | ✅ |
| `simple_RP` (random states, off-node and extrapolated times) | `simple_rp` + `PchipBatch` | float (1e-9 rel + 1e-12·max) | `test_microbubble_model.py` | ✅ |
| `calcBubbleResponse` (single bubbles: buckled/elastic/ruptured, 0.5 µm, 6 µm; batches incl. equal radii; SegersTable, Segers) | `calc_bubble_response(..., step_control="shared")` vs MATLAB's batch; default `"independent"` vs MATLAB integrating each bubble alone | bubble class (worst 0.82 independent / 0.56 shared with the amended 3e-5 term; was 1.17 / 0.79 with 1e-5, B-123) | `test_microbubble_model.py` | ✅ |
| `compute_bubble_mass_source` batch membership (stable radius sort, ties) | `get_batch_indices` | exact | `test_mass_source.py` | ✅ |
| `compute_bubble_mass_source` (defaults, batch sizes 1/2/3/15, ties, single bubble, Segers table, `Advanced` viscosity, float32 input) | `compute_bubble_mass_source` (default independent steps vs MATLAB with `BatchSize = 1`; `step_control="shared"` vs MATLAB at the same batch size) | bubble class (independent worst 0.70, shared 0.22) | `test_mass_source.py` | ✅ |

Batch-size sensitivity (default Marmottant, 7 bubbles, batch 1 vs 7, max per-bubble difference relative
to its max): MATLAB 4.6e-5–2.4e-4; pyproteus shared steps 1.0e-4–1.0e-3; pyproteus independent steps
(default): identical results (`test_mass_source.py::test_batch_size_sensitivity`).

Step control and failure detection (D-22), result-changing within the class:
- Default `step_control="independent"`: each bubble has its own DOPRI5 steps, error test and time scale
  `T = sqrt(rho R0²/P0)` (`pyproteus.microbubble.solver.dopri5_batch`, same published method and
  controller as `numerics.ode45`). Batch results equal single-bubble results bit for bit; the reference
  is MATLAB integrating each bubble alone. `step_control="shared"` keeps PROTEUS' stacked solve (Q-50,
  Q-121) for MATLAB comparisons.
- Timing (3-cycle 2.5 MHz pulses 20–300 kPa, radii 1–4 µm, 5 µs at 250 MHz, Apple M-series, best of 3):

  | bubbles | shared (PROTEUS) | independent | loop of single solves |
  |---|---|---|---|
  | 1 | 0.024 s | 0.031 s | 0.031 s |
  | 3 | 0.050 s | 0.077 s | 0.159 s |
  | 15 (PROTEUS default batch) | 0.089 s | 0.092 s | 0.688 s |
  | 50 | 0.130 s | 0.110 s | 2.166 s |

- PROTEUS' wall-clock timer (60 s × N, `calcBubbleResponse.m:8-12`, formerly Q-122) is not ported: a
  bubble fails deterministically on a non-finite state that step reduction cannot recover, radius ≤ 0,
  radius above `max_radius_ratio·R0` (100), step underflow (`h < 16 eps |t|`, `|t|` floored at 1e-6 of
  the span) or a step budget (`max_steps`, 1e6); an optional `wall_clock_limit` is off by default. Failed
  bubbles are listed in one `BubbleIntegrationError` (index, time, reason, settings) after the other
  bubbles (and, in `compute_bubble_mass_source`, all batches) finished; their results are NaN from the
  failure on. In shared mode a failure stops the whole batch, as the stacked solve must.
- Batch membership in `compute_bubble_mass_source` is unchanged (stable radius sort, consecutive batches);
  the radii are sorted once instead of once per batch.

Design differences (D-15), results unchanged within the class:
- A `Shell` holds a batch as arrays (struct-of-arrays) and every function is vectorised over bubbles;
  `simple_rp` evaluates the surface tension and the driving pressures of the whole batch in one array
  operation. The per-bubble pchip interpolants are built once as one coefficient matrix
  (`PchipBatch`) instead of one `griddedInterpolant` per bubble evaluated in a loop per RHS call.
- Responses are `(n_bubbles, n_t)` arrays instead of a struct array.
- No `parfor`: batches are independent, so serial results are what `parfor` gives.
- float32 sensed pressure: integrated in double, returned as float32 (MATLAB stores batch results in
  single and evaluates part of the RHS in single); same class.
- `find_A_m0` raises when no area in the fit domain gives σ0 (MATLAB returns `[]` and fails later);
  shell-radius warnings are issued once per call instead of once per bubble.
- `calc_thermal_damp` uses `beta/tanh(beta)` for `beta*coth(beta)`.
- Accepted where MATLAB errors (unreachable from the pipeline): `compress_bubble` treats a NaN sign
  change as a crossing (MATLAB's `boolean(NaN)` throws); `get_microbubble_shell_properties` reads a
  missing `Microbubble.Advanced` as false.
- The Segers fit and table are loaded once per process (cached), not on every call.

## Geometry (`pyproteus.geometry`, M4 item moved into M1)

| MATLAB | Python | Class | Oracle test | Status |
|---|---|---|---|---|
| `READ_stl(file)` (ASCII, binary, format detection) | `read_stl`, `identify_stl_format` | exact (data read from files) | `test_geometry_voxelise.py::test_read_stl_*` | ✅ |
| `VOXELISE(gx, gy, gz, meshXYZ, raydirection)` | `voxelise` | integer (D-13: ≤ 0.1 % of the voxels inside either mask may differ, all on the surface; the threshold is taken over the inside voxels, not the grid, since a vessel is < 0.1 % of the grid); measured 100 % on every case | `test_geometry_voxelise.py` | ✅ |
| `define_medium.m` vessel block (`READ_stl`, `*STLunit`, `-BoundingBox.Center`, `rotate_stl`, `+Center`, `VOXELISE`) | `place_vessel_mesh`, `vessel_mask` | integer for the mask; downstream `sound_speed`/`density`/`BonA` float (measured 0 rel. error) | `test_geometry_voxelise.py::test_define_medium_downstream` | ✅ |

Cases: renal tree and mouse brain on PROTEUS's default grid (`make_default_settings` + `write_to_file` grid size + `define_grid`; 446×298×178 and 270×298×178) and on odd non-cubic grids resolving the vessels (151×113×207, 101×113×389, unequal spacings, unaligned origin); `sample.stl` (ASCII) on a 37×23×51 grid with 7 ray-direction strings; synthetic boxes with vertices on rays (vertex branch, both outcomes, silhouette vertices, a degenerate NaN-normal facet), face diagonals on rays (edge hits, odd crossings, neighbour corrections), open boxes, a top with a hole (sequential correction order), holes at the grid edge and corner (zero border padding), a tilted box, a mesh larger than the grid (B-110), the B-90 tie. A reviewer's mutation run (reversed or non-sequential corrections, edge padding, vertex grouping cut, vertex branch always/never, empty or shifted mask) is caught by these tests. Shipped STLs come from `PYPROTEUS_GEOMETRIES_PATH` (Zenodo 10573662); those tests skip without it.

Library choice (D-15): **direct port, vectorised, no new dependency.**
- trimesh (MIT) was evaluated as the established library: `Trimesh.contains` (single-ray parity per point, rtree backend; embree is unavailable on Apple Silicon) matched VOXELISE on the watertight `sample.stl` but took 150× longer (1.5 s vs 0.01 s for 43 k points) and did not finish the renal tree on 300 k points (> 4 min, 9.5 GB). `voxelized().fill()` uses its own grid and a surface-voxel fill, not the grid-point semantics PROTEUS uses. The mouse brain mesh is not watertight (73 boundary/non-manifold edges), exactly where VOXELISE's 3-direction vote and neighbour corrections decide the result and a point-in-mesh test does not reproduce it. numpy-stl (BSD-3) would only replace ~40 lines of STL reading.
- Licence: the toolbox licence is BSD-3-Clause (third clause names The Christie NHS Foundation Trust); full text in `LICENSES/mesh-voxelisation-BSD.txt`, shipped in sdist and wheel (`license-files`), `license = "MIT AND BSD-3-Clause"` in `pyproject.toml`. ROADMAP §8/§13 call it BSD-2.
- The port keeps VOXELISE's formulas (edge side tests, plane equation, 1e-12 rounding, vote, correction order) but runs vectorised over (facet, ray) pairs: 0.2 s for the renal tree default grid (MATLAB 0.3 s), 0.35 s for the mouse brain (MATLAB 0.06 s; the arteriole covers no ray at that resolution).

Divergences from MATLAB (D-15):

| MATLAB | pyproteus | Why |
|---|---|---|
| `VOXELISE` errors ("Colon operands must be real scalars") when a mesh bound is equidistant from two grid lines (B-90, logged in BUGS.md by m1-grid-medium, PR #18) | no error: the port does not restrict rays to the nearest-grid-line range (that range never excludes a ray a facet can cross) | robustness; result equals MATLAB's once the tie is broken by 1e-9 (`test_mesh_bound_halfway_between_grid_lines`) |
| a scalar grid argument is a voxel count (auto grid); a one-point `Grid.z` (Nz = 1) is therefore misread (error, or an empty vessel when z = 0) | every grid argument is coordinates; a one-point axis is one plane | PROTEUS always means coordinates; m1-grid-medium supports Nz = 1 |
| grid coordinates in any order | must be non-decreasing (`ValueError`) | binary search per axis; PROTEUS grids are increasing |
| STL read and voxelised inline in `define_medium.m` | `pyproteus.geometry.vessel_mask(...)`, called by `define_medium` (or a precomputed mask via `vessel=`) | geometry separate from medium (agreed with m1-grid-medium); bool `(Nx, Ny, Nz)`, `[ix, iy, iz]` |

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
| `initialize_das_parameters` (compared with `proteus_compat=True`; linear even/odd, curved; steered, diverging, focused; lens/k-Wave variants; `Transmit.Apodization`; errors; delay-mismatch warning) | `initialize_das_parameters` (warning: `DasDelayMismatchWarning`) | float (axes, time axis, element geometry); exact length of axes | `test_initialize_das_parameters*` | ✅ |
| `DAS_reconstruction_general` (compared with `proteus_compat=True`, and with the default on symmetric unapodized cases; linear 16/15 elements, curved 17/16; plane, steered diverging, focused; v7/v7.3; single RF) | `das_reconstruction_general` | as `DAS_reconstruction` (measured ≤ 6.7e-16 double, 9.8e-8 single) | `test_das_reconstruction_general` | ✅ |
| — (not in PROTEUS) | `interpolation="linear"` option of `compute_das_matrix`, `compute_das_matrix_general`, `das_reconstruction`, `das_reconstruction_general` (D-20; default `"nearest"` = MATLAB) | analytical, not MATLAB: one-element interpolation error ≤ `max|f''|·dt²/8` (measured 0.016 vs 0.14 for nearest on a 2.5 MHz pulse at 40 MHz); 32-element PSF within 2 % of DAS on the continuous RF and ≥ 4× closer than nearest; equal to nearest on on-sample delays | `tests/unit/test_das_linear.py` | ✅ |

Result-changing defaults (developer decision D-21 (amended); `proteus_compat=True` restores PROTEUS):

- **Pixel axes span the domain (was Q-103).** PROTEUS images `-width/2:d:width/2` laterally whatever `Ymin`/`Ymax` are, and the reference beamformer starts at `x = 0`. pyproteus uses `Ymin:d:Ymax` laterally; axially `Xmin:d:Xmax` (general beamformer, as PROTEUS) and `max(Xmin, 0):d:Xmax` (reference beamformer: PROTEUS settings have `Xmin` ≈ -2λ, above the array face, so real settings give PROTEUS's axis; a positive `Xmin` is honoured) (`pixel_axes`). Checked against MATLAB with the default: symmetric domains (`test_das_reconstruction_general_default_symmetric`, reference cases) and real-settings `Xmin = -2λ` (`test_das_reconstruction_real_settings_xmin`); asymmetric domains match MATLAB with `proteus_compat=True` (`test_das_reconstruction_asymmetric_domain`, `test_initialize_das_parameters`, `test_das_reconstruction_general`); the default images the scatterer where it is and honours `Xmin > 0` (`tests/unit/test_das_d21.py`).
- **Transmit apodization masks the pixels (was Q-104).** PROTEUS builds `Tx.apod` and never uses it. `compute_das_matrix_general` now drops pixels whose transmit ray comes from a switched-off element (`apod == 0`): binary mask, not an amplitude weight, since the simulated RF already carries the transmit amplitude (weighting again would give `apod²` striping). Ray origin (`transmit_apodization_weight`): across-beam position for a plane wave, direction from the virtual source / focus otherwise, edge element outside the aperture; ties go to the lower element index; the focal point itself is kept if any element fires; a scalar apodization applies to all elements. Any apodization without zeros (uniform, tapered, random) leaves the image unchanged — checked against MATLAB for unapodized transmits and bit-equal to `proteus_compat=True` for tapered and random ones (unit tests); pixels insonified only by switched-off elements get no contribution. The reference beamformer (`das_reconstruction`, plane/diverging only) ignores `Transmit.Apodization`, as in PROTEUS.

Option beyond MATLAB (developer decision D-20):

- **Linear sample interpolation.** `interpolation="linear"` splits each (pixel, element) weight `v` between samples `floor(s)` and `floor(s)+1` (`v·(1-w)`, `v·w`, `s = (t_del - t(1))·Fs`), kept when `0 ≤ s ≤ Nt-1` (a slightly smaller range than nearest's `-0.5 ≤ s < Nt-0.5`). Removes the sensitivity of nearest-sample DAS to .5 ties and improves the point-spread function (Perrot et al. 2021); at most twice the non-zeros. Default stays `"nearest"` (MATLAB, oracle-tested) until 1.x, where it becomes `"linear"`.

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
- **Empty frames warn.** `log_compress` gives MATLAB's all-NaN image for a frame with no signal and also emits `DasEmptyFrameWarning` (MATLAB is silent).
- **DAS matrix cache.** `das_reconstruction` and `das_reconstruction_general` reuse the matrix of an identical previous call (key: hash of every input of the matrix, including interpolation, `proteus_compat` and the transmit apodization; at most 2 kept, thread-safe). The cached matrices stay in memory after the call (a large image can take hundreds of MB): `cache=False` or `clear_das_matrix_cache()` releases them. No result change.
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
| `define_medium` (background, `alpha_mode`, curved coupling region, vessel from the STL) | `medium.define_medium(..., rng=, geometries_path=, vessel=)` | float (maps); integer (mask, via `pyproteus.geometry`) | `test_medium_define.py`, `test_geometry_voxelise.py::test_default_pipeline_end_to_end_renal_tree` | ✅ |

Notes:

- Everything is compared on settings built by MATLAB itself (`reset_*`, `compute_simulation_domain`, `update_simulation_parameters`, `write_to_file`'s grid size and delays) and read into the Python models from a `.mat` file, plus hand-made tie cases: domain bounds on and half-way between grid nodes, one-point and empty axes, odd/non-cubic grids, `MaxPrime` 2/3/7, single elements, element heights at exact multiples of the elevation spacing, sinc/resample queries outside the sample range and on samples, `floor` ties in the resampled length, voxel ties at `.5`.
- Integer results (`ceil`/`floor`/`round` of IEEE quotients of identical inputs) agree 100 % in every test; the voxel test still asserts the statistical class (≥ 99.9 %) and prints any difference.
- `random_truncated_normal` with an infinite cutoff (MATLAB: untruncated `randn`) raises `NotImplementedError`: MATLAB's `randn` algorithm is not documented, so its stream is not reproduced (ROADMAP §5). Default settings use a finite cutoff.
- `define_medium` with `Geometry.EmbedVessel` (true for both shipped geometries) reads `GeometriesPath/Folder/STLfile` and voxelises it with `pyproteus.geometry.vessel_mask` (PR #19); `GeometriesPath` comes from `geometries_path=` or MATLAB's `Geometry.GeometriesPath` field (`ValueError` if neither and no `vessel=`). A precomputed `vessel=` mask is still accepted (no file read). The oracle tests compare the Python-voxelised medium and mask with MATLAB's `define_medium` for an asymmetric, off-centre box STL (orientation checked against the box computed from the grid coordinates) and for the renal tree (default settings, 2 points per wavelength) and mouse brain (10 µm grid over its bounding box: its ~15 µm vessels miss every voxel centre of an ultrasound grid) when `PYPROTEUS_GEOMETRIES_PATH` points to the STLs; each file's SHA-256 is checked against `src/pyproteus/io/geometry_registry.txt` and the tests skip without the files (`python -m tools.fetch_geometries`). The STLs are members of Zenodo 10573662's `*_no_vtu.zip` archives (not in the oracle checkout). `test_default_pipeline_end_to_end_renal_tree` runs the default settings end to end in Python (settings → domain → grid → `define_medium`) and matches MATLAB's vessel (4710 voxels) on the shared grid points (D-19: Python's grid keeps one more point per axis).

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
| `define_medium` vessel | reads and voxelises the STL inside `define_medium` | same by default (`pyproteus.geometry.vessel_mask`); a precomputed mask can be passed (`vessel=`), `GeometriesPath` can be passed (`geometries_path=`) | geometry separate from medium; masks reusable across frames/jobs |
| `define_medium` with an embedded vessel thinner than the grid spacing (e.g. mouse brain on the default grid, B-112) | silently simulates without a vessel | same result, plus `EmptyVesselWarning` naming dx | usability (D-15, main session); result unchanged |
| `define_medium` | builds the STL path first: `Geometry.GeometriesPath`/`Folder` must exist even without a vessel | not needed when no STL is read | no spurious error |
| `sinc_interpolation` with single `V` (k-Wave sensor data) | double × single product: single result, accumulated in single | `float32` result, accumulated in double and rounded once | more accurate; agrees with MATLAB at the single-precision class (oracle: ≤ 1e-5 × max), not 1e-9 |
| `define_grid`, `define_medium`: missing fields | `SensorOnGrid` absent, `EmbedVessel` absent or `Medium.Vessel` absent → error | `sensor_on_grid=None`; `EmbedVessel` absent = false; `Medium.Vessel` only read when the mask has vessel voxels | lenient where MATLAB fails; no effect on results |
| NaN in `max`/`min` (integration delays, Custom orientation check, travel times) | `max`/`min` skip NaN | `np.nanmax`/`np.nanmin` | same semantics (all-NaN → NaN) |
| `get_transducer_integration_*`, `compute_travel_times` | modify and return the struct | return an updated copy (`copy_struct`); `run_param` is a dict | value semantics, as MATLAB |

## Homogeneous propagation and RF (`pyproteus.acoustic`, M2)

| MATLAB | Python | Class | Oracle test | Status |
|---|---|---|---|---|
| `calc_scatter_attenuated(mass_source, medium, kgrid, r)` (1–4 sources; `N` = `Nt` odd/even, `N` < `Nt`, `N` > `Nt`; tissue/water/lossless media, `b` 1.05–2.5, `no_dispersion`; distances below a wavelength and zero; MATLAB bubble responses as sources; plus far sources after the window and beyond `2 Nt`, D-23) | `acoustic.calc_scatter_attenuated` | double: float (measured ≤ 2.0e-15 × max); single: 1e-5 × max (measured ≤ 8.1e-7, MATLAB single vs double 8.0e-7, pyproteus vs MATLAB double 1.1e-7). With far sources (D-23, longer FFT): near rows 1e-4 × max (measured ≤ 4.1e-6), far rows silent (≤ 1e-5 × max; MATLAB wraps them in). `b ≤ 1` with dispersion: MATLAB NaN, pyproteus finite with `reference_frequency` (D-23) | `tests/oracle/test_homogeneous.py` | ✅ |
| k-Wave `powerLawKramersKronig`, `db2neper` (as called by `calc_scatter_attenuated`) | `acoustic.power_law_kramers_kronig`, `acoustic.db2neper` (clean-room from Szabo 1994 / Waters et al. 2005 and the unit definitions; no k-Wave source) | float; finite/non-finite positions equal (`y` 0.5–2.7, `ω0` 0 and 2π·1 MHz) | `test_homogeneous.py::test_power_law_kramers_kronig_and_db2neper` | ✅ |
| `run_simulation_homogeneous` (direct: linear and curved sensor arrays with odd counts, bubbles on/off axis and next to the array, chunked and default `MaxArrayElements`, single/double; gridded with a bubble on a sensor; bubble-to-bubble with MATLAB bubble responses and `no_dispersion`; averaged medium maps) | `acoustic.run_simulation_homogeneous` (+ `propagate_homogeneous` on arrays) | as `calc_scatter_attenuated`; gridded indices (`round`) equal in every case | `test_homogeneous.py` | ✅ |
| `gridded_distance` (subfunction) | `acoustic.gridded_distance` (0-based indices) | indices exact (statistical class, 100 % seen), grid float | `test_homogeneous.py::test_gridded_distance_matches_matlab` | ✅ |
| `compute_RF` / `compute_RF_data` (P4-1 and C1-6D with odd element counts, finite/infinite elevation focus, odd/even `M`, random receive apodization, per-point weights, dense/sparse sensor weights, `DATA_CAST_RF` single/double, `RF_type = 'pressure'`) | `acoustic.compute_RF`, `acoustic.compute_RF_data` | output size exact; double: float (measured ≤ 5.0e-15 × max); single: 1e-5 × max (measured ≤ 8.8e-7) | `tests/oracle/test_rf.py` | ✅ |
| signal length `M + ceil(max(delays)/dt)` → `optimize_grid_size(N, [0 max(10, round(0.05N))], 5)` | `acoustic.rf_signal_length` | exact | `test_rf.py::test_rf_signal_length` | ✅ |

Notes:

- The mass sources include bubble responses computed by MATLAB PROTEUS (`compute_bubble_mass_source`, default microbubble/medium/transmit, as the oracle's `regression_cases.m`), so these tests do not depend on the Python microbubble model. The end-to-end "RF on given bubble positions" check (M2 Done when) follows once both M2 halves are merged.
- The oracle tests need MATLAB k-Wave on the path (`KWAVE_MATLAB_PATH`), for `powerLawKramersKronig`/`db2neper`.
- The RF transducers are built by MATLAB (`get_transducer_properties`, `get_transducer_integration_points`, `get_transducer_integration_delays`); `test_compute_RF_python_transducer_chain` checks that pyproteus's own M1 chain gives the same RF.
- Unit tests check the physics without MATLAB: lossless free-field Green's function `m'(t - r/c)/(4πr)` (≤ 1e-9), attenuation and Kramers–Kronig phase between two distances, `no_dispersion`, wrap-around, batching/chunking independence, gridded self-silence, integer and fractional lens delays, apodization/weight normalisation, impulse-response convolution, the GPU out-of-memory fallback.
- Single-precision tolerances are relative to the largest value in a case (from the nearest sensor), so far sensors (crop/wrap distances) are effectively checked at ~1e-3 of their own amplitude there; the double-precision cases check every distance at 1e-9 of the case maximum.
- CuPy path (`DATA_CAST`/`DATA_CAST_RF = 'gpuArray-*'`): the code runs on an `xp` array module (`array_module`), selects the device (`DEVICE_NUM`), keeps small outputs on the device and accumulates outputs above `MaxGpuElements` on the host, and gathers RF to the host. CuPy cannot run here (no NVIDIA GPU on the development Mac or in CI), so `tests/unit/test_homogeneous_backend.py` runs the GPU code path with a CuPy stand-in: NumPy underneath, device arrays that are not `ndarray`s, can only reach the host through `.get()` (`np.asarray(dev)` and `host[...] = dev` raise, as with CuPy) and refuse host arrays in device operations; injected bugs (a missing gather, a host array in a device product, a NumPy kernel on device data) all fail the tests: results equal the NumPy path bit for bit, single and double, direct and gridded, host accumulation, `compute_RF`. The real CuPy-vs-NumPy run (M2 "Done when") is left for the university cluster (D-3, D-16), after the complex64 kernel (M5).

### End-to-end RF on given bubble positions (M2 "Done when")

`tests/oracle/test_end_to_end_rf.py`: MATLAB-built default settings (P4-1, 96 elements) and a curved C1-6D (33 elements); 6 fixed bubbles 2–26 mm deep, off-axis, with equal-radius ties, driven by the same incident plane wave (the transmit pulse delayed by `x/c`; the k-Wave first iteration is M3); each side then runs its own integration points and delays → `compute_bubble_mass_source` → `run_simulation_homogeneous` (gridded) → `compute_RF_data` (double precision). A third case (P4-1, 5 elements, `AttenuationB = 1.05`) runs the `no_dispersion` branch. The sensors are the integration points themselves with unit weights; PROTEUS's hybrid last stage senses at k-Wave grid points weighted by `define_sensor_transducer` (M3).

| Stage compared (× max) | P4-1 (96 el.) | C1-6D (33 el.) | P4-1 `no_dispersion` (5 el.) |
|---|---|---|---|
| propagation + RF from MATLAB's mass source | 2.2e-13 | 7.0e-14 | 3.6e-14 |
| mass source, `step_control="shared"` | 2.2e-4 | 2.1e-4 | 2.2e-4 |
| RF, `step_control="shared"` | 1.01e-4 | 6.0e-5 | 1.00e-4 |
| mass source, `"independent"` (default, D-22) | 2.1e-4 | 2.4e-4 | 2.1e-4 |
| RF, `"independent"` | 6.0e-5 | 5.3e-5 | 5.9e-5 |

The RF difference is the bubble-model difference (inside the D-22 class, 1e-3) passed on almost one to one; the former D-13 1e-4 target was missed by ≤ 1 % on the P4-1 cases in shared mode (B-130). **D-24 (developer):** end-to-end RF vs MATLAB ≤ 1e-3 × max|RF| (the D-22 bubble class it inherits), mass source in the same class, both step-control modes; the propagation + RF stage, fed MATLAB's mass source, stays at 1e-9 so propagation and RF bugs are still caught.

Design divergences (D-15; results unchanged unless marked, parity checked as above):

| Where | MATLAB | pyproteus | Why |
|---|---|---|---|
| `calc_scatter_attenuated`, `run_simulation_homogeneous` spectra | full FFTs of length `N` and `2 Nt`; complex `Nsensor × 2Nt × Nsrc` transfer function, then a mirror to make it Hermitian | half spectra (`rfft`/`irfft`, `Nt + 1` bins); DC and Nyquist imaginary parts dropped explicitly (same as MATLAB's `real(ifft)`) | half the memory and work; cuFFT-safe |
| `run_simulation_homogeneous` source sum | one inverse FFT per sensor-source pair, summed in time | sources summed in the frequency domain, one inverse FFT per sensor | `Nsrc`× fewer inverse FFTs; linear, same result |
| `run_simulation_homogeneous` memory | sensor chunks × source batches bounded by `MaxArrayElements` (≈ 4 complex arrays of `chunk·2Nt·batch`); output preallocated in `DATA_CAST` | output allocated in the mass-source dtype and written once per chunk (gridded: accumulated in that dtype, as MATLAB); intermediates ≈ 4 complex128 arrays of `MaxArrayElements/4` (≈ `16·MaxArrayElements` bytes, 1 GB at the default; MATLAB double the same, single half); source spectra per batch unless all fit in `MaxArrayElements/4`; gridded field built in blocks | bounded memory; results independent of the block size (unit tests, incl. a tracemalloc bound). Measured, 20 000 sensors × Nt 2000, single, gridded: peak 342 MB for a 160 MB output (MATLAB ≈ 430 MB) |
| single precision (`DATA_CAST = 'single'`, PROTEUS default) | distances cast to single, transfer function and FFTs in single (phase `2π r f/c` with ~1e-7 relative error) | float64 inside, output cast to float32 | more accurate (MATLAB single is 8e-7 from its own double result, pyproteus 1.1e-7, i.e. float32 output rounding). **To do before the GPU test (M2 CuPy item):** a complex64 kernel for `gpuArray-single`, since the D-16 GPU (Turing) runs FP64 at 1/32 rate; check it against the float64 kernel at the single class |
| `compute_RF` delays and element sum | `N_el·N_int × N` complex array: FFT, delay, inverse FFT per integration point, then apodization and weighted sum in time | delay, apodization and weights applied to half spectra and summed per element, in blocks of elements bounded by `run_param.MaxArrayElements` (default 2^26); one inverse FFT per element | `N_int`× fewer inverse FFTs, no full-size complex array; same result (linear) |
| `compute_RF` maximum delay | `max` skips NaN | `np.nanmax` | same semantics |
| `powerLawKramersKronig`, `db2neper` | k-Wave (LGPL) functions | own implementations from the published formulas, although k-wave-python has equivalents | array-module (`xp`) support for CuPy, IEEE behaviour at the `ω0 = 0` singularity without k-wave-python's warnings/logging, no import of k-wave-python on the homogeneous path |
| `compute_RF` single precision | FFT, delays and phase in single | float64 inside; the integration-point pressure and the element pressure are rounded to float32 where MATLAB casts them | as above |
| `compute_RF_data` | catches `parallel:gpu:array:OOM`, prints, switches `DATA_CAST_RF` to CPU | catches CuPy `OutOfMemoryError`, `ResourceWarning`, returns a changed copy of `run_param` | value semantics; warnings instead of prints |
| `calc_scatter_attenuated` with `0 < b ≤ 1` and dispersion (**D-23, result change**, was Q-130) | Kramers–Kronig with reference frequency 0 (`powerLawKramersKronig(2πf, 0, …)`), singular for `b ≤ 1`: phase speed ±0/NaN, whole field NaN (default runs reach it with `AttenuationB ≤ 0.9`; `define_medium` sets `no_dispersion` for `|b-1| < 0.1`) | finite reference frequency `reference_frequency` (the transmit centre frequency; `c(f0) = c0`), required there (`ValueError` otherwise); `f/c(f)` taken as its limit 0 at DC. PROTEUS's formula unchanged for every other `b`, so arrival times equal MATLAB's wherever MATLAB is finite. `run_simulation_homogeneous(..., reference_frequency=)` | a usable field instead of NaN. Tests: oracle (MATLAB NaN, pyproteus finite, error without reference), unit (`c(f0) = c0`, transfer function between two distances = `exp(-Δr (α + 2πi f/c(f)))` at f0, `b > 1` unaffected by the reference). The finite reference for all `b` is a 1.x backlog item (changes arrival times) |
| arrivals later than `2·Nt·dt` (**D-23, result change**, was Q-131) | periodic FFT of length `2·Nt`: a source farther than `Nt·dt·c0` can wrap its arrival into the window (`τ mod 2·Nt·dt`) | FFT length `2·Nt` when every distance is ≤ `Nt·dt·c0` (MATLAB-identical), otherwise the 5-smooth size from `Nt + ceil(1.1·r_max/(c0·dt))`, so nothing wraps; arrivals after the window are cropped | unreachable with PROTEUS's own `Nt` sizing (`compute_travel_times`). With a user-chosen short `Nt`: the window equals the start of a longer simulation (unit test ≤ 1e-9); vs MATLAB the near sources agree within 4.1e-6 × max (MATLAB's periodic band-limited leakage moves with the period) and the far sources are silent where MATLAB wraps them in |
| `run_simulation_homogeneous` progress output, GPU reset (`gpuDevice`) | `disp` per batch; resets the device | no output; selects the CuPy device (`DEVICE_NUM`) without reset | library code; no global GPU reset |
| MATLAB structs | `kgrid`, `medium`, `source`, `sensor`, `run_param` structs | dicts/structs accepted (`_struct.get`); array-level `propagate_homogeneous(mass_source, source_points, sensor_points, dt=, nt=, medium=)` | clearer API for the hybrid simulator |

## Run setup and run data (`pyproteus.acoustic.setup`, `pyproteus.io.frames`, M3)

| MATLAB | Python | Class | Oracle test (`tests/oracle/test_frames_setup.py`) | Status |
|---|---|---|---|---|
| `load_microbubbles` (random rotation, N = 1/7/13, 9/10/100 frames → zero padding, 3 pulses, v7 and v7.3) | `acoustic.setup.load_microbubbles` (`transform_microbubbles`, optional `frame_tree`) | float (points, velocities ≤ 1e-9 rel); radii exact | `test_load_microbubbles` | ✅ |
| `check_ground_truth_data` (match; each mismatch and several at once; 6-digit rounding incl. exact binary .5 ties at the 6th digit; one pulse ignores the time between pulses; distribution values/lengths) | `acoustic.setup.check_ground_truth_data` (`GroundTruthMismatch`, `GroundTruthDistributionWarning`) | exact (error and warning text) | `test_check_ground_truth_data` | ✅ |
| `input_handling` (varargin 1–3: defaults, logical/numeric, clipping of `EndFrame`, invalid values) | `acoustic.setup.input_handling` | exact | `test_input_handling`, `test_input_handling_start_after_end` | ✅ (B-161) |
| `sim_setup` ('3D', '3DC', '3DG'; `path_setup` replaced by a temporary stand-in so nothing is created in the oracle) | `acoustic.setup.sim_setup`, `make_run_config` → `RunConfig` | exact | `test_sim_setup` | ✅ (B-162) |
| `sim_startup` | `acoustic.setup.sim_startup` → `RunPaths` | unit-tested only (MATLAB's version calls the real `path_setup`, which creates folders in the oracle) | `tests/unit/test_frames_setup.py::test_sim_startup` | 🟡 (B-163) |
| `compute_travel_times` | `acoustic.compute_travel_times` (M1) | float | see "Grid, medium, transducer integration" | ✅ |
| ground-truth `Frame_N.mat`, `FlowSimulationParameters.mat` | `io.frames.read_ground_truth_pulse`, `read_flow_simulation_parameters` (on `config.load_frame`, `load_flow_simulation_parameters`, oracle-tested in `test_config.py`) | exact | `test_load_microbubbles`, `test_config.py::test_ground_truth_frames` | ✅ |
| RF `Frame_N.mat` of `main_RF` (`RF` cell of single, `dt`, `Frame` cell of MB structs; v7 and v7.3) | `io.frames.read_matlab_rf_frame`, `iter_rf_source` | exact | `test_matlab_rf_frames` | ✅ |
| `load_RF_data` (all four schemes) on MATLAB frames and on the same frames copied into an RF HDF5 file | `reconstruction.load_rf_data`, `iter_rf_frames` | exact | `test_matlab_rf_frames` | ✅ |
| `medium.mat` (v7.3: `medium`, `vessel_grid`, `Grid`) | `io.frames.load_medium_cache` | exact | `test_matlab_medium_cache` | ✅ |

Design differences with MATLAB (D-15; results unchanged):

| Where | MATLAB | pyproteus | Why |
|---|---|---|---|
| RF output (**D-25**) | one `Frame_<m>.mat` per frame (`RF`, `dt`, `Frame`), `FlowSimulationParameters.mat` copied next to them, settings in a separate file | one HDF5 file per run (`<results>/rf.h5`, format `pyproteus-rf` v1, layout in `io/frames.py`): settings (lossless JSON), flow-simulation parameters, one group per frame with `rf` `(n_pulses, Nelem, Nt)` in the RF class, `dt` and the frame's bubbles per pulse | one self-describing file; a stopped run keeps every finished frame and resumes (`RFWriter(mode="resume")` checks settings, sequence and frame count, drops an incomplete frame) |
| per-frame flush | `save` per frame | the frame group is written under a temporary name, renamed when complete, and the file is closed after each frame | an interrupted frame is never mistaken for a finished one. Residual risk: a process killed inside an HDF5 metadata write can still damage the file (HDF5 has no journal) |
| re-running and resuming | each run overwrites only the `Frame_<m>.mat` files it writes | `setup_run` always opens the RF file in `"resume"` mode (created if absent; nothing truncated before a frame is computed); frames outside `StartFrame..EndFrame` are kept; without `Continue` the range is recomputed and replaced (`replace_frames`), the old version readable until the new one is complete. A resume accepts the stored settings when they equal the run's settings up to the run-control fields (`io.frames.RUN_CONTROL_FIELDS`: `Continue`, `StartFrame`, `EndFrame`, `DeviceNumber`, `BatchSize`, `UseParfor`, `Transducer.OnGrid`/`IntegrationDensity`, `GeometriesPath`, `Medium.Save`); other differences refuse it | same frames on disk as MATLAB as long as the settings are unchanged up to run control (MATLAB overwrites frames computed with other physics settings; pyproteus refuses with `ValueError`); after `lock_timeout` `write_frame` raises and the caller (`main_RF`) must keep or dump the frame; `mode="overwrite"` only on explicit request |
| `Continue` (**B-166**) | reuse the saved medium; recompute every frame of the range | also skip frames already in the RF file (`frames_to_simulate`) | an interrupted run continues where it stopped |
| readers during a run | other programs can read finished `.mat` files | HDF5 locks the file while it is open: a writer waits (retries for `lock_timeout`, default 600 s, keeping the frame in memory) while a reader holds it; one writer per file. Several processes on different frame ranges of one run (MATLAB: one GPU each into one folder) are not supported yet: D-28 (M5) gives each process its own file (`rf_<start>-<end>.h5`), read as a list or folder, with an optional merge tool. With `HDF5_USE_FILE_LOCKING=FALSE` (some cluster file systems) do not read a file that is being written | no lost frames from a short read |
| file size | one file per frame | deleted or replaced frames do not shrink the file; `h5repack` compacts it | HDF5 |
| bubbles in the RF file | `Frame{k}` = whole MB struct | numeric fields of the MB dict (`points`, `radii`, `velocities`, `nodes`, `idx`, …) | no MATLAB objects in HDF5 |
| reading RF | folder of `Frame*.mat` | `load_rf_data`, `iter_rf_frames`, `das_reconstruction(_general)` accept the HDF5 file or a MATLAB folder; with the HDF5 file the pulsing scheme and the settings default to the stored ones | MATLAB results stay usable |
| k-Wave medium cache | `medium.mat` (v7.3), reloaded on `Continue` without any check | `<results>/medium.h5`: medium maps as datasets, scalars/strings as attributes, `vessel_grid` (gzip), the `Grid` dataclass, and a SHA-256 fingerprint of `SimulationParameters`, `Geometry`, `Medium`, `Transducer` without the run-control fields and `Solver`/`NumberOfInteractions`/`HybridSimulation` (+ caller extras such as the seed), so another GPU or solver reuses the cache; `load_medium_cache(fingerprint=...)` refuses a cache built from other settings; written to a temporary file and renamed. MATLAB `medium.mat` is still read | a stale medium silently changes every frame of a continued run in MATLAB |
| `check_ground_truth_data` | always creates `savedir` and saves the parameters there; error names the last failing check | no side effect unless `savedir` is given (then a byte copy); `GroundTruthMismatch.parameters` lists every mismatch, message as MATLAB; distribution vectors compared flattened | library code; the parameters go into the RF file (D-25) |
| `load_microbubbles` | reads `Frame_<m>.mat` once per pulse | optional `frame_tree` (one read per frame) | I/O |
| `sim_setup`, `sim_startup` | `path_setup` + `addpath`; `PATHS` struct | explicit folders (`settings_dir`, `ground_truth_dir`, `results_dir`, `binary_path`, `data_path`, `geometries_path`); binary from `backends.kwave.binary_path`; `RunConfig` dataclass, `run_param()` gives the MATLAB field names plus `backend`/`device` | no global path state; `setup_run` chains the whole preamble of `main_RF` |
| settings file | `.mat` | `.mat` or pyproteus `.json` | JSON settings (M0) |

## k-Wave sources and sensors (`pyproteus.acoustic.sensor`, `pyproteus.acoustic.source`, M3)

`update_sensor`/`update_sensor_fast`, `define_sensor_transducer`, `define_sensor_MB`, `define_sensor_MB_all`, `extract_sensor_data`, `preprocess_transmit`, `define_source_transducer`. Oracle: `tests/oracle/test_sensor.py` (both MATLAB `update_sensor` versions; masks, indices and sparsity patterns exact, weights ≤ 1e-9 rel; odd/even/short grids, odd/even `full_size`, points on nodes / on one or two grid lines / at .5 ties with power-of-two spacing / within the truncation width of the edge / on the domain boundary / coincident / outside, non-empty input mask, `Grid.threshold`; PROTEUS's own `UpdateSensorTest` set-up on the default settings' grid), `tests/oracle/test_source.py` (P4-1 and C1-6D, steered, on/off grid, `SourceType` absent/dipole/monopole, every `SeqPulse` with odd element counts and apodization; signals ≤ 1e-9 × max, masks exact; `preprocess_transmit` with MATLAB k-Wave ≤ 1e-9 × max).

Row orders: sensor masks and source masks are `[ix, iy, iz]` arrays as MATLAB's; source rows (`source.p`, `ux`, `uy`) and sensor weight columns are in MATLAB `find(mask)` order (column-major), which `backends.kwave.kspace_first_order` takes and returns on every backend. Feeding k-wave-python directly needs C order: `rows_f_to_c(rows, mask)` for source rows and recorded data, `weights[:, f_to_c_rows(mask)]` for weight columns (unit test `test_sensor_weights_interpolate_and_c_order`). Linear indices (`points_idx`, `source_mask_idx`, `mb_idx_all`, `MB.idx`) are 0-based.

| Where | MATLAB | pyproteus | Why |
|---|---|---|---|
| `update_sensor` / `update_sensor_fast` | two implementations: a per-point loop and a vectorised version guarded by a 2 GiB memory estimate that falls back to the loop | one vectorised implementation over chunks of points (≤ 2^22 candidate nodes per chunk), `update_sensor_fast` an alias; semantics of `update_sensor.m` where they differ (B-150) | bounded memory without a slow path. Measured on a 200×300×40 grid, off grid: 1 000 points 0.13 s (MATLAB loop 0.76 s, fast 0.05 s); 20 000 points (13.7 M weights) 3.3 s, MATLAB 31 s / 33 s (fast falls back to the loop) |
| sensor weights | `sparse(i,j,B)`, size `max(i) × max(j)`, built from preallocated triplets | SciPy CSR, always `N_points × nnz(mask)` (same as MATLAB whenever the input mask is empty, i.e. every PROTEUS call); explicit zeros dropped as MATLAB | a shape that always matches the recorded data; CSR for `W @ data` |
| `define_sensor_MB` `max_dist` | full `N × N` distance matrix | same formula in row blocks | memory `O(N)` per block instead of `N²` (10⁵ bubbles: 80 GB in MATLAB) |
| `define_sensor_MB_all` | reads every frame itself (`load_microbubbles(folder, frame, pulse, Geometry, Nframes)`) | takes the loaded frames (any iterable, e.g. a generator over `load_microbubbles`; structs with `points` or `N × 3` arrays) | frame reading in one place (`io`), one frame in memory at a time; same mask, indices and `max_mb` (oracle writes real `Frame_XX.mat` files for MATLAB) |
| `define_source_transducer` delay | full FFT of length `N`, symmetric frequency axis (negative Nyquist for even `N`), `ifft(...,'symmetric')` | `rfft`/`irfft` on the first `N//2 + 1` bins of the same axis | half the memory and work; identical by construction (unit test against the full-spectrum form, even and odd `N`) |
| `define_source_transducer` `SeqPulse` | any unknown string transmits unchanged | `ValueError` (B-151); `pulsing_sequence()` gives `main_RF`'s sequence per `PulsingScheme` | catches typos |
| structs | `sensor`, `source`, `Transmit` structs, modified in place | `Sensor`/`Source` dataclasses (fields as MATLAB, absent ones `None`); inputs (settings models, dicts, objects) are copied, never modified | value semantics |
| `preprocess_transmit` | `filterTimeSeries` (k-Wave, LGPL) | `backends.kwave.filter_time_series` (k-wave-python, B-60 zero-phase workaround) + `resample_signal` | no k-Wave code copied |
