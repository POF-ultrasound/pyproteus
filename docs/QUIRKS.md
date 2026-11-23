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

## Q-100 — DAS time to peak is one sample late
- MATLAB: `delay-and-sum/DAS_reconstruction.m:51-54`, `initialize_das_parameters.m:62-65`.
- Behaviour: `timeToPeak = I/Transmit.SamplingRate` with `I` the 1-based index of the envelope maximum of the two-way pulse, so the time axis is shifted by one sample (`(I-1)/Fs` is the peak time). The envelope is computed at `Transducer.SamplingRate` but converted with `Transmit.SamplingRate` (equal in PROTEUS settings).
- Ported in: `pyproteus.reconstruction.time_to_peak`.
- Test: `tests/oracle/test_das.py::test_das_reconstruction` (image at the float class, so a one-sample shift would fail).
- Planned fix: `(I-1)/Fs` after parity; moves the image by `c/(2Fs)` in depth (≈ 19 µm at 40 MHz).

## Q-101 — DAS TGC normalised by an imaginary maximum when the time axis starts far before 0
- MATLAB: `delay-and-sum/DAS_reconstruction.m:82-84`, `DAS_reconstruction_general.m:51-53`.
- Behaviour: `TGC = sqrt(t)/max(sqrt(t)).*...` with negative `t` gives an imaginary `sqrt`; MATLAB's `max` of a complex array picks the largest magnitude, so when `|t(1)| > t(end)` (time to peak + lens correction longer than the recording after it) the normalisation is imaginary, `TGC` is imaginary for `t ≥ 0`, `hilbert` drops the imaginary part of `RF.*TGC`, and the image is all NaN (0/0).
- Ported in: `pyproteus.reconstruction.time_gain_compensation` returns `real(TGC)`.
- Test: `tests/oracle/test_das.py::test_time_gain_compensation` (`t0,t1 = -12e-6,10e-6`), `tests/unit/test_reconstruction.py::test_tgc_matches_formula_and_imaginary_quirk`.
- Planned fix: normalise by `sqrt(max(t))`; only changes the degenerate case.

## Q-102 — Reference DAS apodization keeps pixels at or above the array surface
- MATLAB: `delay-and-sum/compute_das_matrix.m:65-66` (`alpha = atan(abs(y_el-Y)./X)`).
- Behaviour: at `X = 0` the element at the pixel's own lateral position gives 0/0 = NaN, which is not above the threshold, so it is kept (the others are at `atan(Inf) = pi/2` and dropped); at `X < 0` every angle is negative and every element is kept.
- Ported in: `pyproteus.reconstruction.compute_das_matrix`.
- Test: `tests/oracle/test_das.py::test_compute_das_matrix_realistic` (axial grid with `X < 0` and `X = 0`), `tests/unit/test_reconstruction.py::test_compute_das_matrix_matches_brute_force`.
- Planned fix: use `abs(X)` or drop pixels with `X ≤ 0` (the general beamformer uses `atan2` with a threshold on `|alpha|`).
