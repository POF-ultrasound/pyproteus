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
- Ported in: `pyproteus.numerics.ode45` (one state vector, shared steps; batching itself comes with `compute_bubble_mass_source`).
- Test: `tests/oracle/test_ode45.py::test_batch_dependence_similar`, `tests/unit/test_ode45.py::test_stacked_batch_shares_steps`.
- Planned fix: after parity, per-bubble step control (vectorised solver with per-bubble steps, or one solve per bubble), so results no longer depend on batching.

## Q-100 — DAS time to peak uses the 1-based index (likely off-by-one; to be confirmed)
- MATLAB: `delay-and-sum/DAS_reconstruction.m:51-54`, `initialize_das_parameters.m:62-65`.
- Behaviour: `V_ref = conv(Transmit.PressureSignal, Transducer.ReceiveImpulseResponse)`, `[~,I] = max(abs(hilbert(V_ref)))`, `timeToPeak = I/Transmit.SamplingRate`. `V_ref` starts at index 1 = time 0, so the peak time is `(I-1)/Transmit.SamplingRate`: the axis is shifted by one *transmit* sample, `1/Transmit.SamplingRate` (4 ns ≈ 3 µm in depth at the GUI default of 250 MHz).
- Three sampling rates are involved: `Transmit.SamplingRate` and `Transducer.SamplingRate` (both 250 MHz by default, `reset_transmit.m`, `reset_transducer.m`) and `SimulationParameters.SamplingRate` (≈ 20·f0 by default, `update_simulation_parameters.m`). `preprocess_transmit.m` resamples Transmit to the simulation rate inside `main_RF.m`, but the settings file DAS reads keeps the raw rate. A second quirk sits in the same line: `conv` combines the pressure signal (Transmit rate) with the impulse response (Transducer rate) sample by sample, which is only meaningful when the two rates are equal (a loaded transducer model can have its own rate).
- Time axes checked (2026-11-23): the RF time axis is 0-based throughout PROTEUS. Transmit source sample m is applied at `(m-1)*dt` (`define_source_transducer.m`). k-Wave `t_array(n) = (n-1)*dt`. `compute_RF.m` convolves from lag 0 (`convn(p,IR)*dt`), so RF sample k is at `(k-1)*dt`. DAS uses `t = (0:Nt-1)/Fs`. pyproteus reproduces MATLAB's images to 1e-16, so the port adds no shift.
- The k-Wave correction (`+dx/(2c) + dt` hybrid, `+dx/(2c) + 3/2·dt` full, with `dt` the simulation step) is undocumented. Only if all three rates are equal could its `+dt` absorb the extra index; with the default rates the two terms differ (4 ns vs ≈ 20 ns), so a compensation is unlikely and a plain off-by-one is the most probable reading.
- Ported in: `pyproteus.reconstruction.time_to_peak` (reproduced, with both rates as in MATLAB).
- Test: `tests/oracle/test_das.py::test_das_reconstruction`, `test_initialize_das_parameters` (time axis at the float class; cases with 250/250/40 MHz and 250/100/40 MHz rates).
- Planned fix: confirm with a point scatterer at a known depth imaged through MATLAB PROTEUS (a linear scatterer, i.e. a small density or sound-speed contrast, so a bubble's resonance phase is not mistaken for a timing offset). That is physics validation of the original simulator, not probing, and can be done before M3. Then use `(I-1)/Fs` and resample `V_ref` components to a common rate; the k-Wave terms get their own documented derivation.

## Q-102 — Reference DAS apodization keeps pixels at or above the array surface
- MATLAB: `delay-and-sum/compute_das_matrix.m:65-66` (`alpha = atan(abs(y_el-Y)./X)`).
- Behaviour: at `X = 0` the element at the pixel's own lateral position gives 0/0 = NaN, which is not above the threshold, so it is kept (the others are at `atan(Inf) = pi/2` and dropped); at `X < 0` every angle is negative and every element is kept.
- Ported in: `pyproteus.reconstruction.compute_das_matrix`.
- Test: `tests/oracle/test_das.py::test_compute_das_matrix_realistic` (axial grid with `X < 0` and `X = 0`), `tests/unit/test_reconstruction.py::test_compute_das_matrix_matches_brute_force`.
- Planned fix (1.x, developer D-18: keep in 1.0): use `abs(X)` or drop pixels with `X ≤ 0` (the general beamformer uses `atan2` with a threshold on `|alpha|`).

## Q-103 — DAS lateral axis is centred on 0, whatever the domain's lateral bounds
- MATLAB: `delay-and-sum/DAS_reconstruction.m:37-41` (`y = -width/2:pixelSize:width/2`), `initialize_das_parameters.m:31-35`.
- Behaviour: only the domain width `Ymax - Ymin` is used; an asymmetric domain (e.g. `Ymin = -3.1 mm`, `Ymax = 2.9 mm`) is imaged over `[-3.0, 3.0] mm`. The reference beamformer also starts the axial axis at 0 instead of `Xmin`.
- Ported in: `pyproteus.reconstruction.das_reconstruction`, `initialize_das_parameters`.
- Test: `tests/oracle/test_das.py::test_initialize_das_parameters`, `test_das_reconstruction_general` (asymmetric domain).
- Planned fix: span `[Ymin, Ymax]` (and start at `Xmin` in the reference beamformer).

## Q-104 — transmit apodization never reaches the general beamformer
- MATLAB: `delay-and-sum/initialize_das_parameters.m:110-121` builds `Tx.apod`; `compute_das_matrix_general.m` never reads it.
- Behaviour: elements switched off in transmit still define the transmit model (as documented, `t = 0` is the first element over all elements) and no apodization weighting is applied to the image.
- Ported in: `DasTransmit.apod` is filled and ignored, as in MATLAB.
- Test: `tests/oracle/test_das.py::test_initialize_das_parameters` (`Transmit.Apodization` case).
- Planned fix: use `Tx.apod` (e.g. drop pixels outside the insonified region) or remove the field.

## Q-80 — any `Configuration` other than exactly `'Curved'` gives a linear array
- MATLAB: `GUIfunctions/get_element_geometry.m` (`is_curved`: `strcmp(Transducer.Configuration,'Curved')`).
- Behaviour: case-sensitive exact match; `'curved'`, `'RCA'` (RC6gV) or a missing field give the linear geometry (elements at x = 0, `ElementWidth` used, `Radius` ignored).
- Ported in: `pyproteus.transducer.get_element_geometry`.
- Test: `tests/oracle/test_transducer_geometry.py::test_fixed_cases[other]`, `test_catalogue_transducers[RC6gV]`.
- Note: a curved array ignores `ElementWidth` (element width = pitch, no kerf); that is PROTEUS's documented convention, not a quirk.
- Planned fix (proposal for after parity; changes results for misspelt values): an enumerated, validated `Configuration` and an explicit RCA geometry. One test, `transducer.is_curved`, keeps the rule in one place meanwhile.
