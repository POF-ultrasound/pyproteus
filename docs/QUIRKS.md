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

## Q-80 — any `Configuration` other than exactly `'Curved'` gives a linear array
- MATLAB: `GUIfunctions/get_element_geometry.m` (`is_curved`: `strcmp(Transducer.Configuration,'Curved')`).
- Behaviour: case-sensitive exact match; `'curved'`, `'RCA'` (RC6gV) or a missing field give the linear geometry (elements at x = 0, `ElementWidth` used, `Radius` ignored).
- Ported in: `pyproteus.transducer.get_element_geometry`.
- Test: `tests/oracle/test_transducer_geometry.py::test_fixed_cases[other]`, `test_catalogue_transducers[RC6gV]`.
- Note: a curved array ignores `ElementWidth` (element width = pitch, no kerf); that is PROTEUS's documented convention, not a quirk.
- Planned fix (proposal for after parity; changes results for misspelt values): an enumerated, validated `Configuration` and an explicit RCA geometry. One test, `transducer.is_curved`, keeps the rule in one place meanwhile.

## Q-90 — `define_grid` drops the last grid point when the PML padding is odd
- MATLAB: `acoustic-module/define_grid.m:80-93` (oracle commit 8d37938).
- Behaviour: the full size `M = optimize_grid_size(N, 2*PMLmin, MaxPrime)` searches the single candidate `N + 2*PMLmin` and grows from there; the PML is `ceil((M - N)/2)` per side and the core grid becomes `M - 2*PML`. When `M - N` is odd the core grid has `N - 1` points, so the last grid point inside the domain is lost (e.g. `N = 100` → `M = 135`, PML 18, core 99). A domain one grid point thick gives an **empty** axis (`N = 1` → `M = 32`, PML 16, core 0).
- Ported in: `pyproteus.acoustic.define_grid`.
- Test: `tests/unit/test_acoustic.py::test_define_grid_q90_odd_padding_drops_a_point`; `tests/oracle/test_acoustic_grid.py::test_define_grid_domains` (single-point z domain).
- Planned fix: k-Wave's PML is symmetric per axis, so keep the odd point in the core instead (`PML = floor((M - N)/2)`, core `N` or `N + 1` points, the extra one just outside the domain), or accept only full sizes with `M - N` even. Either keeps the whole domain on the grid; it changes grid sizes, hence every downstream index, for about half the settings.
