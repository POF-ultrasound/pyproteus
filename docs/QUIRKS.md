# MATLAB PROTEUS quirks ported on purpose

Phase A reproduces MATLAB PROTEUS v1.1.0 behaviour exactly, including behaviour
that looks wrong. Each such quirk is listed here with the planned fix after
parity (Phase B). Do not fix a quirk before parity is confirmed.

Format:

```
## Q-n — short title
- MATLAB: file and line (oracle commit 8d37938)
- Behaviour: what MATLAB does
- Ported in: pyproteus module/function
- Test: tests/... that pins the behaviour
- Planned fix: what changes after parity, and its expected effect
```

## Q-1 — `alpha_mode='no_dispersion'` is ignored on the k-Wave C++/CUDA path (B-14)
- MATLAB: `acoustic-module/define_medium.m:80-81` sets `medium.alpha_mode = 'no_dispersion'` when `|alpha_power - 1| < 0.1`; `run_simulation.m` passes the medium to `kspaceFirstOrder3DC`/`3DG` (`sim_setup.m:12,18`).
- Behaviour: the k-Wave C++ HDF5 input format has no `alpha_mode` field, so MATLAB 3DC/3DG run full power-law absorption **with** dispersion; the output is bit-identical with and without `alpha_mode`. Only PROTEUS' MATLAB-native `kspaceFirstOrder3D` mode (local movie) honours it.
- Ported in: `pyproteus.backends.kwave.medium_for_kwp` drops `alpha_mode` on every k-wave-python backend (D-9): `cpp` would raise, `python` would apply it (16 % different on the tracer case).
- Test: `tests/oracle/test_kwave_tracer.py::test_matlab_3dc_ignores_alpha_mode`; `tests/oracle/test_kwave_backend.py::test_sensor_data_matches_matlab[python-pressure_nodisp]` (python backend with `alpha_mode` set matches MATLAB 3DC ≤ 1e-5); `tests/unit/test_kwave.py::test_alpha_mode_dropped_on_every_backend`.
- Planned fix: after parity, honour `no_dispersion` on every backend (or drop the setting from `define_medium`), so all solvers agree; changes absorption-dispersion near y = 1.

## Q-30 — tissue names are matched by substring, first match wins
- MATLAB: `GUIfunctions/assign_medium_properties.m:25-31` (`~isempty(strfind(tissue_names{i}, Medium.Tissue))`).
- Behaviour: `'Muscle'` selects `'Muscle (along the fibers)'`, `'Eye'` the lens, `'ate'` selects `'Water'`; matching is case-sensitive and an empty name matches nothing (`strfind(x,'')` is empty), raising `'Tissue not found'`.
- Ported in: `pyproteus.medium.assign_medium_properties`.
- Test: `tests/oracle/test_config.py::test_assign_medium_properties` (substring cases), `test_assign_medium_properties_not_found`.
- Planned fix: exact name match (or a dropdown-only API) after parity.

## Q-31 — unknown catalogue keys are handled inconsistently
- MATLAB: `get_transducer_properties.m` (no `otherwise`: unknown `Type` returns the struct unchanged), `assign_liquid_properties.m` (unknown liquid: unchanged), `get_gas_properties.m` and `assign_medium_properties.m` (unknown name: error).
- Ported in: `pyproteus.transducer`, `pyproteus.medium`, `pyproteus.microbubble` (same behaviour; errors are `ValueError`).
- Test: `tests/oracle/test_config.py::test_get_transducer_properties[Unknown type]`, `test_assign_liquid_properties[Plasma]`, `test_get_gas_properties_unknown`.
- Planned fix: raise on every unknown key after parity.

## Q-32 — Verasonics impulse responses stored as integers saturate
- MATLAB: `GUIfunctions/load_transducer_model.m:91-92` (`S.Trans.IR1wy*Fs`, then `double(...)` at line 99).
- Behaviour: MATLAB multiplies in the class of the stored `IR1wy`: an integer IR is rounded and saturated before the conversion to double (`int16([1 -2 3])*250e6` → `[32767 -32768 32767]`); a `single` IR stays single.
- Ported in: `pyproteus.transducer.properties._times_double`.
- Test: `tests/oracle/test_config.py::test_load_transducer_model[verasonics_int16]`, `[verasonics_uint8]`; `tests/unit/test_config_models.py::test_verasonics_integer_ir_saturates`.
- Planned fix: convert to double before scaling.

## Q-50 — Bubbles integrated in one batch share ode45's adaptive steps
- MATLAB: `microbubble-simulator/calcBubbleResponse.m:28-52` stacks `[x; xdot]` of all bubbles of a batch into one state vector for one `ode45` call; `acoustic-module/compute_bubble_mass_source.m` (`get_batch_indices`) groups bubbles into batches after sorting by radius.
- Behaviour: one step size for the whole vector (the error test covers all components), so a bubble's response depends on which other bubbles share its batch and on the batch size (at the level of the tolerance).
- Ported in: `pyproteus.numerics.ode45` (one state vector, shared steps); `pyproteus.microbubble.calc_bubble_response` and `pyproteus.acoustic.mass_source` (stable radius sort, batches of `batch_size`, default 15). See also Q-121.
- Test: `tests/oracle/test_ode45.py::test_batch_dependence_similar`, `tests/unit/test_ode45.py::test_stacked_batch_shares_steps`.
- Fixed by default (D-22, 2026-11-25): per-bubble step control (`pyproteus.microbubble.solver.dopri5_batch`, vectorised over the batch); a bubble's result equals its single-bubble solve bit for bit (`tests/unit/test_microbubble_model.py::test_independent_batch_equals_single_bitwise`). The shared-step mode stays available as `step_control="shared"`.

## Q-120 — SegersTable surface tension is 0 exactly at the table's right end
- MATLAB: `microbubble-simulator/functions/calc_surface_tension.m` (`'SegersTable'` branch): `sig = A_m*0`, then `sig(A_m>A_m2) = sig_l` and the interpolant only for `A_m1 < A_m < A_m2`.
- Behaviour: a normalised area exactly equal to `A_m2` gets 0 (buckled value) instead of `sig_l`; exactly `A_m1` gets 0 (correct). Measure zero in practice.
- Ported in: `pyproteus.microbubble.calc_surface_tension`.
- Test: `tests/oracle/test_microbubble_model.py::test_surface_tension_table_ends`.
- Planned fix: `>=`/`<=` at the ends (1.x).

## Q-121 — Bubbles of one batch share the nondimensional time scale
- MATLAB: `microbubble-simulator/calcBubbleResponse.m`: `T = median(sqrt(rho R0.^2/P0))` over the batch; the ODE tolerances act on `xdot = Rdot·T/R0`.
- Behaviour: in addition to the shared steps (Q-50), the meaning of `AbsTol` on the velocity depends on the batch's median radius, so the batch changes results at the tolerance level. Measured with the default batching rule (stable radius sort, `compute_bubble_mass_source`): batch 1 vs 7 changes a bubble's mass source by up to 2.4e-4 (MATLAB) / 1.0e-3 (pyproteus) of its maximum.
- Ported in: `pyproteus.microbubble.calc_bubble_response`, `pyproteus.acoustic.mass_source` (`get_batch_indices`).
- Test: `tests/oracle/test_mass_source.py::test_batch_size_sensitivity`, `tests/oracle/test_microbubble_model.py::test_matlab_batch_dependence`.
- Fixed by default (D-22): `step_control="independent"` gives every bubble its own steps and time scale; `step_control="shared"` reproduces this quirk for MATLAB comparisons.

## Q-122 — Wall-clock timer stops bubble integration after 60 s × N (not ported: replaced, D-22)
- MATLAB: `microbubble-simulator/calcBubbleResponse.m:8-12` (timer sets `timeout_reached`), `functions/odeOutputFcn.m` (returns it, which stops `ode45`).
- Behaviour: results depend on machine speed; the response is truncated and `compute_bubble_mass_source` then fails with a size mismatch.
- Replaced (D-22): deterministic failure detection and an optional wall-clock limit, off by default; see `docs/PARITY.md` (microbubble section).

## Q-100 — DAS time to peak uses the 1-based index (unresolved; decide after the M3 point-scatterer check)
- MATLAB: `delay-and-sum/DAS_reconstruction.m:51-54`, `initialize_das_parameters.m:62-65`.
- Behaviour: `V_ref = conv(Transmit.PressureSignal, Transducer.ReceiveImpulseResponse)`, `[~,I] = max(abs(hilbert(V_ref)))`, `timeToPeak = I/Transmit.SamplingRate`. `V_ref` starts at index 1 = time 0, so the peak time is `(I-1)/Transmit.SamplingRate`: the axis is shifted by one *transmit* sample, `1/Transmit.SamplingRate` (4 ns ≈ 3 µm in depth at the GUI default of 250 MHz).
- Three sampling rates are involved: `Transmit.SamplingRate` and `Transducer.SamplingRate` (both 250 MHz by default, `reset_transmit.m`, `reset_transducer.m`) and `SimulationParameters.SamplingRate` (≈ 20·f0 by default, `update_simulation_parameters.m`). `preprocess_transmit.m` resamples Transmit to the simulation rate inside `main_RF.m`, but the settings file DAS reads keeps the raw rate. A second quirk sits in the same line: `conv` combines the pressure signal (Transmit rate) with the impulse response (Transducer rate) sample by sample, which is only meaningful when the two rates are equal (a loaded transducer model can have its own rate).
- Time axes checked (2026-11-23): the RF time axis is 0-based throughout PROTEUS. Transmit source sample m is applied at `(m-1)*dt` (`define_source_transducer.m`). k-Wave `t_array(n) = (n-1)*dt`. `compute_RF.m` convolves from lag 0 (`convn(p,IR)*dt`), so RF sample k is at `(k-1)*dt`. DAS uses `t = (0:Nt-1)/Fs`. pyproteus reproduces MATLAB's images to 1e-16, so the port adds no shift.
- The k-Wave correction (`+dx/(2c) + dt` hybrid, `+dx/(2c) + 3/2·dt` full, with `dt` the simulation step) is undocumented. Only if all three rates are equal could its `+dt` absorb the extra index; with the default rates the two terms differ (4 ns vs ≈ 20 ns), so a compensation is unlikely and a plain off-by-one is the most probable reading.
- Ported in: `pyproteus.reconstruction.time_to_peak` (reproduced, with both rates as in MATLAB).
- Test: `tests/oracle/test_das.py::test_das_reconstruction`, `test_initialize_das_parameters` (time axis at the float class; cases with 250/250/40 MHz and 250/100/40 MHz rates).
- Point-scatterer run of MATLAB PROTEUS (2026-11-24, `tools/investigations/`, B-101): the end-to-end DAS timing error is depth dependent (≈ −26 ns at 5 mm, ≈ +8 ns at 7 mm, hybrid and full alike; stable within ±3 ns across a grid cell), larger than the 4 ns at stake. Inconclusive for this term.
- Planned fix: decided after the M3 point-scatterer depth check (milestone added by the main session): a point scatterer at a known depth imaged through the simulator, with a linear scatterer (a small density or sound-speed contrast) so a bubble's resonance phase is not mistaken for a timing offset. If it is an off-by-one, use `(I-1)/Fs` and resample the `V_ref` components to a common rate; the k-Wave terms get their own documented derivation.

## Q-102 — Reference DAS apodization keeps pixels at or above the array surface
- MATLAB: `delay-and-sum/compute_das_matrix.m:65-66` (`alpha = atan(abs(y_el-Y)./X)`).
- Behaviour: at `X = 0` the element at the pixel's own lateral position gives 0/0 = NaN, which is not above the threshold, so it is kept (the others are at `atan(Inf) = pi/2` and dropped); at `X < 0` every angle is negative and every element is kept.
- Ported in: `pyproteus.reconstruction.compute_das_matrix`.
- Test: `tests/oracle/test_das.py::test_compute_das_matrix_realistic` (axial grid with `X < 0` and `X = 0`), `tests/unit/test_reconstruction.py::test_compute_das_matrix_matches_brute_force`.
- Planned fix (1.x, developer D-18: keep in 1.0): use `abs(X)` or drop pixels with `X ≤ 0` (the general beamformer uses `atan2` with a threshold on `|alpha|`).

## Q-80 — any `Configuration` other than exactly `'Curved'` gives a linear array
- MATLAB: `GUIfunctions/get_element_geometry.m` (`is_curved`: `strcmp(Transducer.Configuration,'Curved')`).
- Behaviour: case-sensitive exact match; `'curved'`, `'RCA'` (RC6gV) or a missing field give the linear geometry (elements at x = 0, `ElementWidth` used, `Radius` ignored).
- Ported in: `pyproteus.transducer.get_element_geometry`.
- Test: `tests/oracle/test_transducer_geometry.py::test_fixed_cases[other]`, `test_catalogue_transducers[RC6gV]`.
- Note: a curved array ignores `ElementWidth` (element width = pitch, no kerf); that is PROTEUS's documented convention, not a quirk.
- Planned fix (proposal for after parity; changes results for misspelt values): an enumerated, validated `Configuration` and an explicit RCA geometry. One test, `transducer.is_curved`, keeps the rule in one place meanwhile.

## Q-81 — the pressure pulse assumes the voltage and impulse response share a sampling rate
- MATLAB: `GUIfunctions/get_pressure_signal.m` (`conv(V, IR)/Transducer.SamplingRate`).
- Behaviour: `Transmit.VoltageSignal` is sampled at `Transmit.SamplingRate`, the impulse response at `Transducer.SamplingRate`; neither is checked or resampled. With a calibrated model sampled at another rate the pulse has the wrong time scale and amplitude.
- Ported in: `pyproteus.transmit.get_pressure_signal` (same behaviour).
- Test: `tests/oracle/test_transmit_signals.py::test_pressure_signal` (model impulse responses).
- Planned fix (proposal for after parity): resample the impulse response to the transmit rate, or raise.

## Q-82 — an all-zero voltage pulse or a pulse without negative pressure gives NaN/Inf
- MATLAB: `get_voltage_signal.m` (`V*(VoltageAmplitude/max(abs(V)))`), `get_pressure_signal.m` (`p*(AcousticPressure/pnp)`).
- Behaviour: a three-level pulse sampled too coarsely can be all zeros (e.g. 26.09 MHz at 62.5 MHz, one cycle): voltage mode gives NaN everywhere. A pressure pulse with `min(p) = 0` gives Inf/NaN in pressure mode. No warning.
- Ported in: `pyproteus.transmit` (same behaviour, NumPy warnings silenced).
- Test: `tests/unit/test_transmit_signals.py::test_get_voltage_signal_modes`; Hypothesis cases in `tests/oracle/test_transmit_signals.py`.
- Planned fix (proposal for after parity): raise when the pulse has no amplitude.

## Q-83 — narrow-band estimated impulse responses are inaccurate in MATLAB (not reproduced, D-17)
- MATLAB: `GUIfunctions/estimate_impulse_response.m` (`[b,a] = butter(3,Wn,'bandpass'); impz(b,a,N,Fs)`).
- Behaviour: the (b, a) form of the 6th-order bandpass is ill-conditioned when the band is narrow relative to the sampling rate; MATLAB's impulse response is then off by 1–2 % of its maximum (0.5–0.6 MHz at 250 MHz; 1–2 MHz at 1 GHz) by up to 1.0e-8 for the catalogue bands (P4-1) and 1.5e-8 for other wide bands tested (B-80).
- pyproteus deliberately does **not** reproduce this (D-17): `pyproteus.transducer.estimate_impulse_response` uses second-order sections, accurate to ~1e-14 (catalogue bands) and ~3e-12 (narrow bands) of the exact response.
- Test: `tests/oracle/test_transmit_signals.py::test_impulse_response_vs_reference` (gated), `::test_impulse_response_vs_matlab` (catalogue bands gated at 1e-7, narrow bands reported).

## Q-84 — the Gaussian size distribution: 1000 radii, NaN for PDI = 0
- MATLAB: `GUIfunctions/compute_gaussian_distribution.m`.
- Behaviour: 1000 radii from 0.5 to 6 µm, while `compute_polydisperse_distribution` uses 1001 over the same range. `PDI = 0` or `MeanRadius = 0` gives a zero standard deviation and NaN probabilities (0/0) without an error; a mean radius far outside the range underflows every probability to 0, also giving NaN. The sign of `PDI` is irrelevant.
- Ported in: `pyproteus.microbubble.gaussian_distribution` (same behaviour, NumPy warnings silenced).
- Test: `tests/oracle/test_microbubble_distribution.py::test_fixed_cases`, `tests/unit/test_microbubble_distribution.py`.
- Planned fix (proposal for after parity): validate `PDI > 0` and `MeanRadius > 0`, share one radius grid between the distributions, and make the grid (N, Rmin, Rmax) parameters with PROTEUS's values as defaults.

## Q-90 — `define_grid` drops the last grid point when the PML padding is odd
- MATLAB: `acoustic-module/define_grid.m:80-93` (oracle commit 8d37938).
- Behaviour: the full size `M = optimize_grid_size(N, 2*PMLmin, MaxPrime)` searches the single candidate `N + 2*PMLmin` and grows from there; the PML is `ceil((M - N)/2)` per side and the core grid becomes `M - 2*PML`. When `M - N` is odd the core grid has `N - 1` points, so the last grid point inside the domain is lost (e.g. `N = 100` → `M = 135`, PML 18, core 99). A domain one grid point thick gives an **empty** axis (`N = 1` → `M = 32`, PML 16, core 0). The default settings of both shipped geometries pad every axis oddly.
- Ported in: `pyproteus.acoustic.define_grid`, reproduced by default (`pml_rounding="ceil"`, D-19).
- Test: `tests/unit/test_acoustic.py::test_define_grid_q90_default_drops_a_point`; `tests/oracle/test_acoustic_grid.py::test_define_grid_domains`, `test_define_grid_default_settings` (default mode equals MATLAB's grid exactly).
- Fix available now: `define_grid(..., pml_rounding="floor")` keeps the odd point in the core (`N + 1` points, PML one thinner, same full size; tests `test_define_grid_floor_keeps_the_domain` and the oracle `"floor"` checks). The default flips to `"floor"` in 1.x (D-19); every downstream index changes for odd-padding axes.

<!-- Q-130 and Q-131 were fixed under D-23: see docs/PARITY.md (homogeneous propagation divergences). -->

## Q-132 — the mass-source derivative is spectral on the source's own (periodic) length
- MATLAB: `acoustic-module/calc_scatter_attenuated.m:35-41` (`MS_dot = real(ifft(2i*pi*f.*fft(mass_source,[],2),[],2))`).
- Behaviour: the time derivative treats the mass source as one period of a periodic signal of length `N`. A source that does not end where it starts (e.g. a bubble still oscillating at the end of the record, or a non-zero offset) has a jump at the wrap, and its derivative rings (Gibbs) over the whole record. Bubble sources start at rest and usually decay, so the effect is normally small.
- Ported in: `pyproteus.acoustic.homogeneous._source_spectra` (same spectral derivative).
- Test: covered by every `calc_scatter_attenuated` oracle case (the Gaussian test sources are not exactly zero at the ends).
- Planned fix (proposal for after parity): pad the source (e.g. to the `2·Nt` propagation length) or taper it before differentiating, or use a finite-difference derivative; changes results for sources truncated at the end of the record.

## Q-170 — self-sense runs evaluate the band-limited delta with the global grid's period
- MATLAB: `acoustic-module/compute_self_sense_pressure.m:35` (`Grid_local.full_size = Grid.full_size`), used by `update_sensor.m` → `evaluate_delta_function(..., Grid.full_size)` (oracle commit 8d37938).
- Behaviour: each local single-bubble k-Wave grid is `core + 2*10` voxels per axis (periodic over that length), but the bubble's sensor/source weights use the band-limited delta for the *global* grid's full size. The weights differ from the local grid's own delta by ~0.1 % (max, offset 0.45 dx, 72-voxel local vs 200–600-voxel global grid), so the self-sensed pressure (∝ Σ w_i w_j G_ij) changes by ~0.2 %, above the k-Wave class.
- Ported in: `pyproteus.acoustic.full.compute_self_sense_pressure` (reproduced: `full_size` from the global grid).
- Test: `tests/oracle/test_full_pieces.py::test_self_sense_matches_matlab` (global `full_size` 49×44×41, local 32³).
- Planned fix (proposal for after parity): use the local grid's full size (`Nx_local + 20`, ...) in the self-sense weights, so the subtracted self term matches what the local grid actually propagates.
