# Bugs and behaviour differences

Anything that behaves differently from what the plan or MATLAB expects. Status: open / workaround / fixed / wontfix.
Format: `## B-n — title (status)`, then where, symptom, cause, fix or workaround, link to test.

## B-1 — MATLAB engine breaks HTTPS in the same process (workaround)
- Where: any process that imports `matlab.engine` before `ssl`.
- Symptom: `URLError: unknown url type: https` (k-wave-python binary download); `_ssl` fails to load with a symbol missing from `/Applications/MATLAB_R2025a.app/bin/maca64/libcrypto.3.dylib`.
- Cause: the engine puts MATLAB's libcrypto first; conda's `_ssl` then binds to it.
- Workaround: import `ssl` (or `kwave`) before `matlab.engine`. The oracle fixture must do this.

## B-120 — PROTEUS' Segers polynomial fit is ill-conditioned (observation; class widened for that model)
- Where: `microbubble-simulator/functions/getShellProperties.m` (`'Segers'`), `find_domain_boundaries.m`, `find_A_m0.m`, `calc_surface_tension.m` (`polyval`); fit `fit_SigmaR_04-08-2017.mat`, degree 10 in the monomial basis, coefficients up to 9e10.
- Symptom: `roots` gives a cluster of three real roots near A_m = 0.91; MATLAB and NumPy (`numpy.roots`, same companion-matrix method) place `A_m1` 6e-4 rel apart, `A_m2` 1e-4, `A_N` 4e-5; `polyval` has evaluation noise ~eps·Σ|c| = 8e-5 N/m, so `chi` from `getEqParam`'s finite difference differs by up to 0.4 %. Near the fit's maximum (σ0 = 0.0719) MATLAB finds no `A_m0` while NumPy finds one. Both are equally valid evaluations of an ill-conditioned problem.
- Handling: no code to match MATLAB (D-12). The `'Segers'` polynomial path is tested at a class from its conditioning (roots 1e-3 rel, σ 2e-4 N/m, χ-derived 1e-2); its bubble responses still meet the bubble class. The pipeline (`compute_bubble_mass_source`) maps `'Segers'` to the lookup table, which is well conditioned (float class).
- Proposal (1.x, D-15): drop the polynomial path, or evaluate the fit in a centred/scaled or Chebyshev basis.

## B-123 — Quiet bubbles crossing the Marmottant buckling kink are integrated less accurately (accepted: D-22 amended)
- Where: `calc_surface_tension` clamps σ at the buckling radius R_b (and at σ_l): a kink in the RHS.
- Symptom: 2 µm, σ0 = 0.01, 2 MHz, 7–20 kPa sweep, fraction of the D-22 class against DOP853 rtol 1e-13: MATLAB up to 0.82 (12 kPa), pyproteus up to 1.16 (10–11 kPa); oracle case `batch_equal_radii` (10 kPa bubble) at 1.17 of the class vs MATLAB. With R_b out of reach (σ0 = 0.04) both ≤ 0.17.
- Cause: a 5th-order step straddling the kink loses order; the error depends on where steps land. Hairer–Wanner step stabilisation (β = 0.04) only moves the unlucky amplitude (1.71 at 9 kPa), so it was not adopted.
- Resolution (developer, D-22 amended): the class's absolute term is 3e-5 (rel 1e-3 unchanged); buckling/rupture events (R = R_b, σ = σ_l) are in the 1.x backlog.

## B-121 — D-13 bubble class not reachable (resolved: D-22 class)
- Where: `calcBubbleResponse` with PROTEUS' options (RelTol = AbsTol = 1e-6).
- Symptom: pyproteus vs MATLAB 6–30× the D-13 bound; MATLAB vs MATLAB (bubble alone vs in a batch) 17–34×. Against DOP853 at rtol 1e-13 both are ~1e-4 of the excursion.
- Cause: global error of an oscillator integrated over ~40 time scales with local tolerance 1e-6; adaptive step sequences differ (D-12).
- Resolution (D-22): class `1e-3·max|y| + 1e-5` per bubble plus a CI accuracy test against the high-accuracy reference (≤ 1e-3 of the excursion).

## B-122 — PROTEUS `compress_bubble` needs a toolbox function `boolean` (observation)
- Where: `microbubble-simulator/functions/compress_bubble.m:54` (`find(boolean(diff(sign(...))))`).
- Symptom: `Undefined function 'boolean'` on a MATLAB installation without the toolbox that provides it (R2025a here). Not called by the simulation pipeline.
- Handling: Python uses a plain logical test; the oracle test puts a test-only shim `tests/oracle/shims/boolean.m` (`logical(x)`) on the path.

## B-20 — interp1/griddedInterpolant with matrix v: sign of zero on flat segments (closed: irrelevant under D-12)
- Was: with matrix-valued `v`, MATLAB R2025a returned +0 where pyproteus returned -0 on some ±0 flat segments (76 of 2,187 special-value comparisons; values equal).
- Closed 2026-11-22: the flat-segment rule that caused it was removed (D-14), and the sign of zero is not part of the comparison classes (D-12/D-13).

## B-10 — k-wave-python 0.6.2 `kspaceFirstOrder()`: backends disagree on source rows, scaling and output shape (workaround)
- Where: `kwave.kspaceFirstOrder.kspaceFirstOrder` (the >= 0.6.1 API), `backend="cpp"` vs `backend="python"`.
- Symptom: with MATLAB-convention inputs (rows in `find` order, unscaled signals) the `cpp` backend goes unstable (NaN); with C-order rows the results are wrong (1.2-1.6 rel); the documented `(n_sensor, Nt)` output is actually `(Nt, n_sensor)` for `cpp`.
- Cause (`kwave/solvers/cpp_simulation.py`): the cpp path writes `source.p`/`ux`/`uy` to the HDF5 file unchanged next to an F-order (`find`) index list and does not apply k-Wave's time-varying source scaling; it reads `p` back as `(Nt, n_sensor)`, and its F→C row reordering only runs when `shape[0] == n_sensor`, so it is skipped (and if `Nt == n_sensor` it would permute the time axis). The `python` backend expects C-order source rows, scales internally and returns `(n_sensor, Nt)` in C order.
- Workaround: `pyproteus.backends.kwave` (lifted from the tracer's `adapter.py`; its `kspace_first_order` runs the binary itself for `cpp` and reads the output file, so the output-shape part no longer applies there): cpp = find-order rows × `additive_source_scale` (p: `2dt/(3 c0 dx)`, u: `2 c0 dt/dx`, per voxel), output transposed (and, if `Nt == n_sensor`, k-wave-python's spurious time-axis permutation undone); python = C-order rows, output reordered C→F. With it both backends match MATLAB k-Wave to < 7e-7 rel and the cpp source datasets agree with MATLAB's within float32 noise. Pin k-wave-python exactly. Test: `tests/oracle/test_kwave_tracer.py`.

## B-11 — MATLAB engine breaks `pyexpat` (matplotlib, k-wave-python legacy API) in the same process (fixed)
- Where: any process that imports `matlab.engine` before `xml.parsers.expat` (pulled in by matplotlib → plistlib, e.g. `kwave.kspaceFirstOrder3D`).
- Symptom: `ImportError: ... pyexpat ... Symbol not found: _XML_SetAllocTrackerActivationThreshold ... Expected in /Applications/MATLAB_R2025a.app/bin/maca64/libexpat.1.9.3.dylib`.
- Cause: same as B-1: the engine loads MATLAB's older libexpat first.
- Fix: `tests/oracle/conftest.py` imports `xml.parsers.expat` next to `ssl`, before any `matlab.engine` import. Scripts outside pytest use `tools/tracer_bullet/kwave_tracer.preload_before_engine()`.

## B-12 — k-wave-python 0.6.2 legacy `kspaceFirstOrder3D`: wrong pressure-source scaling, crash on per-voxel velocity sources (wontfix here, upstream)
- Where: `kwave/kWaveSimulation_helper/scale_source_terms_func.py` (legacy MATLAB-style API).
- Symptom: pressure case 3.8e-3 rel off MATLAB; per-row ratio to MATLAB's `p_source_input` 0.995-1.005. Velocity case: `TypeError: 'int' object is not subscriptable` (`source_val.size[0]`).
- Cause: the scaling uses the sound speed at the next voxel along x (`c0` indexed with 1-based `find` indices as if 0-based; matches to 1e-4 m/s with offset +1). Only shows with heterogeneous `c0`, i.e. always in PROTEUS.
- Workaround: do not use the legacy API; M3 adapter uses `kspaceFirstOrder()` (B-10). Test: `test_legacy_api_matches_matlab` (xfail; XPASS means fixed upstream). Report upstream.

## B-13 — k-wave-python staggered density wraps at the last plane (open, within tolerance)
- Where: `CppSimulation._compute_staggered_density` (`np.roll` average).
- Symptom: `rho0_sgx/y/z` in the HDF5 input differ from MATLAB k-Wave's only on the last grid plane along the staggered axis (inside the outer PML), up to 0.55 % rel; elsewhere within float32 noise (≤ 1e-6 rel).
- Cause: `np.roll` averages the last plane with the first; MATLAB k-Wave keeps `rho0` there. Contributes the ~5e-7 residual in sensor data, well below the 1e-5 target.
- Workaround: none needed for parity; if a case ever misses 1e-5, write the HDF5 input with the adapter's own `rho0_sg*` (fallback route in ROADMAP §6). Test: `test_cpp_input_file_matches_matlab` (asserts it is still the only difference).

## B-14 — PROTEUS `alpha_mode='no_dispersion'` has no effect on the k-Wave C++ path (quirk, reproduced)
- Where: oracle `acoustic-module/define_medium.m` sets `medium.alpha_mode = 'no_dispersion'` when `|alpha_power - 1| < 0.1`; `run_simulation.m` then calls `kspaceFirstOrder3DC`/`3DG`.
- Symptom: MATLAB 3DC output is identical with and without `alpha_mode` within the binary's run-to-run noise (~1e-6, B-63) (the C++ HDF5 format has no field for it), so PROTEUS runs full power-law absorption + dispersion on the C++/CUDA path; the MATLAB-native `kspaceFirstOrder3D` (PROTEUS' local-movie mode) and k-wave-python's `python` backend honour it (16 % rel difference on the tracer case with y = 1.05). k-wave-python's `cpp` backend raises `ValueError`.
- Faithful-port handling (D-9): `pyproteus.backends.kwave.medium_for_kwp` drops `alpha_mode` on every backend, so `cpp` and `python` both match MATLAB 3DC (≤ 1e-5, case `pressure_nodisp` in `tests/oracle/test_kwave_backend.py`). Recorded as `docs/QUIRKS.md` Q-1.
- Test: `test_matlab_3dc_ignores_alpha_mode`, case `pressure_nodisp`.

## B-60 — k-wave-python 0.6.2 `filter_time_series(zerophase=True)` crashes (workaround)
- Where: `kwave.utils.filters.apply_filter`, zero-phase branch.
- Symptom: `IndexError: index L is out of bounds` for any signal; PROTEUS always calls `filterTimeSeries(..., 'ZeroPhase', true, ...)` (`preprocess_transmit.m`, `compute_bubble_mass_source.m`).
- Cause: the reverse pass indexes `filtered_signal[np.arange(L + N, 1, -1)]` (1-based leftover) and then calls `np.fliplr` on a 1-D array.
- Workaround: `pyproteus.backends.kwave.filter_time_series` builds the zero-phase result from k-wave-python's causal filter: design at half the stop-band attenuation, filter forwards, then filter the time-reversed result and reverse it back. Provenance: forward-backward filtering is the standard zero-phase construction (as SciPy's `filtfilt`); the cascade squares the magnitude response, doubling the stop-band attenuation in dB, hence half per pass. k-Wave's `filterTimeSeries` help text (via `help` in MATLAB) only says "zero phase or linear phase". No k-Wave MATLAB source (`filterTimeSeries.m`, `applyFilter.m`) was opened. Matches MATLAB k-Wave `filterTimeSeries` within the float class (test: 1e-9 rel, `tests/oracle/test_kwave_signal.py`). Upstream draft: `docs/dev/upstream/k-wave-python-issues.md` issue 5.

## B-80 — MATLAB `estimate_impulse_response` (butter in b,a form + impz) is inaccurate, by percent for narrow bands (PROTEUS numerical bug; not reproduced, D-17)
- Where: `GUIfunctions/estimate_impulse_response.m` (`[b,a] = butter(3,Wn,'bandpass'); impz(b,a,N,Fs)`), default P4-1 and every estimated transducer.
- Symptom: against an independent reference (partial fractions of SciPy's digital zpk), MATLAB's impulse response is off by 1.0e-8 (P4-1, 1.5–3.5 MHz at 250 MHz), 5.3e-9 (C1-6D bands), 4e-10 (9L-D, RC6gV), and 1–2 % for narrow bands (0.5–0.6 MHz at 250 MHz; 1–2 MHz at 1 GHz), relative to max|IR|. SciPy `sos`/`sosfilt` is within 5e-14 (2.7e-12 for the narrow bands).
- Cause: the transfer-function (b,a) form of a 6th-order bandpass with poles near z = 1 is ill-conditioned (documented numerical caveat of the ba form).
- Impact: the D-13 float class (1e-9) vs MATLAB is not reachable for P4-1/C1-6D by an accurate method.
- Resolution (D-17, developer): pyproteus uses SciPy second-order sections; oracle tests compare with the zpk reference at 1e-12 and with MATLAB at 1e-7 rel. to max on catalogue-like bands; narrow-band MATLAB errors are reported by `tests/oracle/test_transmit_signals.py::test_impulse_response_vs_matlab`, not gated. Quirk Q-83. Notes: side-session-notes/…/m1-leaves-impulse-response.md.

## B-61 — k-wave-python 0.6.2 rejects `2*pml_size >= N` even with the PML outside the grid (workaround)
- Where: `kwave.solvers.validation.validate_pml`, called by `kspaceFirstOrder()` on the *unexpanded* grid.
- Symptom: `ValueError: pml_size[i]=10 is too large for grid size N[i]=17` with `pml_inside=False`. MATLAB `kspaceFirstOrder3DC` with `PMLInside=false` accepts it. PROTEUS can hit it: `compute_self_sense_pressure.m` uses a 10-point PML on local grids cropped at the domain edge.
- Workaround: `kspace_first_order` expands the grid itself with k-wave-python's `_expand_for_pml_outside` (what `pml_inside=False` does internally) and calls `kspaceFirstOrder(pml_inside=True)`; same input file. Test: `tests/unit/test_kwave.py::test_cpp_and_python_backends_agree` (grid 20×17×15, PML 8/10/7); the oracle cases run through the same path.

## B-62 — k-wave-python 0.6.2 `kspaceFirstOrder(backend="cpp")` ignores `sensor.record` (workaround)
- Where: `kwave/kspaceFirstOrder.py` (cpp branch warns "sensor.record is not yet supported for backend='cpp'") and `CppSimulation._execute` (passes no output flags, so the binary records only `p`).
- Symptom: `sensor.record = ['p_max', 'p_min']` (PROTEUS `main_pressure_field.m`) returns `p` only on `cpp`; the `python` backend honours it.
- Workaround: `pyproteus.backends.kwave` writes the input file with `kspaceFirstOrder(save_only=True)` (one function, `_write_cpp_input`), runs the binary with the flags MATLAB `kspaceFirstOrder3DC` passes (captured from MATLAB: `--p_raw`, `--p_max`, `--p_min`, `-t`; `binary_command`) and reads the output file (`read_cpp_output`). Tests: `tests/oracle/test_kwave_backend.py::test_binary_command_matches_matlab`, cases `nt_eq_n_sensor`, `pressure_field`. Upstream draft: issue 4.

## B-63 — k-Wave OpenMP binary is not bit-reproducible run to run (open, within tolerance)
- Where: `kspaceFirstOrder-OMP` v1.4.1 (k-wave-python's macOS arm64 build), from MATLAB 3DC and from `pyproteus.backends.kwave` alike.
- Symptom: two runs of the same input file differ by ~1e-6 relative in the sensor data (seen in `tests/unit/test_kwave.py::test_cpp_data_path_unique_folders_and_delete_data`, about 1 run in 6).
- Cause: likely FFTW planning / OpenMP thread scheduling (summation order); not investigated.
- Impact: within the 1e-5 k-Wave class. Frozen references (D-10) and run-vs-run tests cannot be compared bit for bit; tests compare at 1e-5.

## B-30 — `scipy.io.loadmat(mat_dtype=True)` drops the imaginary part of complex arrays (workaround)
- Where: SciPy 1.15 `loadmat(..., mat_dtype=True)`, e.g. `microbubble-simulator/functions/*TransferFunctionFit.mat` (complex 1×50000).
- Symptom: real-valued result plus a `ComplexWarning`. Without `mat_dtype`, integer-valued doubles come back in their storage class (`10` → `uint8`).
- Workaround: `pyproteus.config.matfile` reads twice (typed + raw) and takes complex arrays from the raw read. Test: `tests/unit/test_config_matfile.py::test_v5_classes_and_shapes`, `tests/oracle/test_data.py`.

## B-31 — MATLAB `exp` (R2025a maca64) is not correctly rounded: 1-ulp differences (observation, within class)
- Where: any port that evaluates transcendental functions; first seen in `compute_polydisperse_distribution` (`reset_microbubble` default `Distribution.Probabilities`).
- Symptom: on 41k random inputs MATLAB `exp` equals the correctly rounded result 90.2 % of the time, NumPy/Apple libm 99.8 %; 87 of 1001 values differ by 1 ulp in the default distribution. `Probabilities` agree to 7e-16 rel.
- Cause: MATLAB ships its own libm. Not reproducible without reverse-engineering it.
- Handling: observation only (developer policy 2026-11-22: no replication of MATLAB's libm). Such fields are compared at the ROADMAP §4 signal class (≤ 1e-12 rel). Test: `tests/oracle/test_config.py::test_compute_polydisperse_distribution`, `test_reset_microbubble`.

## B-32 — MATLAB `sum` of a vector differs from NumPy's pairwise sum in the last bits (observation, within class)
- Where: `P/sum(P)` in `compute_polydisperse_distribution` and wherever PROTEUS sums vectors.
- Symptom: `sum` of the 1001-element distribution differs from `np.sum` by 1.4e-14 abs (1e-16 rel).
- Handling: observation only (developer policy 2026-11-22: no replication of MATLAB's summation order, no `numerics.sum`). Within the 1e-12 class.

## B-33 — SciPy reads MATLAB's 0-byte `[]` matrix element as 1×0; MATLAB loads 0×0 (workaround)
- Where: `GUIfunctions/tissue_properties.mat` (`properties_name{2,5}`), written by MATLAB on Windows in 2022; older MATLAB versions write `[]` inside cells/struct fields as a miMATRIX element of 0 bytes. R2025a writes a full header instead.
- Workaround: `matfile._zero_byte_elements` walks the v5 element tree, finds 0-byte elements and sets them to 0×0 double. Tests: `tests/unit/test_config_matfile.py::test_zero_byte_empty_element`, `tests/oracle/test_config.py::test_loadmat_shipped_files`.

## B-34 — empty char arrays lose their size in the value tree (wontfix)
- Where: `pyproteus.config.matfile`, models, JSON.
- Behaviour: `''` (0×0), `zeros(1,0)` char and `char(zeros(0,3))` all become `""`. MATLAB code tests them with `isempty`/`strcmp`, which treat them alike; PROTEUS never relies on the size of an empty char.
- Also: an empty (0-element) struct array without fields saved in v5/v6 loads as an empty cell (SciPy cannot tell them apart); v7.3 gives an empty `StructArray`. PROTEUS never writes such values.

## B-70 — linear interp1/griddedInterpolant next to Inf/NaN sample values gives NaN where MATLAB gives a number (accepted, D-14)
- Where: `pyproteus.numerics.interpolation._linear` (plain blend `v0*(1-t) + v1*t`).
- Symptom: on an interval with an Inf or NaN end value, a query on a node gives NaN (`0*Inf`, `0*NaN`), e.g. `interp1([0 1 2],[1 Inf 3],0)`: MATLAB 1, pyproteus NaN. Likewise a flat segment of infinities queried on a node or extrapolated, and ±Inf queries on such an interval (`[Inf Inf]` at `Inf`: MATLAB Inf, pyproteus NaN). Finite data are unaffected (±Inf queries on finite data match MATLAB).
- Cause: MATLAB's handling was found by black-box probing; the special cases reproducing it were removed under D-14 (no documented semantics).
- Impact: none known; PROTEUS interpolates finite tables.
- Test: `tests/oracle/test_numerics_interp.py::assert_linear` (such queries must be MATLAB's value or NaN), `tests/unit/test_numerics.py::test_interp1_flat_and_nonfinite_neighbours`.

## B-71 — pchip next to NaN or ±Inf samples: NaN where MATLAB sometimes gives a number or ±Inf (accepted, D-14)
- Where: `pyproteus.numerics.interpolation.pchip_slopes` (NaN secant → NaN slopes at its nodes).
- Symptom: `griddedInterpolant(x, v, 'pchip')` with a NaN in `v` gives NaN on the intervals next to the NaN sample. MATLAB R2025a gives NaN there in some cases and a number in others (e.g. `x = 0:5`, `v = [NaN 5 4 3 2 1]` at 1.5: MATLAB 4.625, pyproteus NaN; `v = [NaN 1 2 3 4 5]` at 1.5: both NaN). With ±Inf samples, all three pchip APIs (`interp1`, `pchip`, `griddedInterpolant`) can give NaN where MATLAB gives ±Inf or a number (Inf - Inf in the slopes/coefficients), e.g. `interp1([0 1],[-3 Inf],[0.5 -1],'pchip')`: MATLAB `[Inf -Inf]`, pyproteus NaN (MATLAB's own `pchip()` also gives NaN there). A reviewer sweep of 3000 random cases found pyproteus never returns a number where MATLAB returns NaN, never a wrong Inf sign, and agrees at the float class wherever both are finite.
- Cause: MATLAB's NaN/Inf slope handling is undocumented; replicating it would need probing (D-14). Plain IEEE propagation is used instead.
- Impact: none known; PROTEUS interpolates finite tables.
- Test: `tests/oracle/test_numerics_interp.py::test_pchip_nonfinite_values` (`assert_value_or_nan`), `tests/unit/test_numerics.py::test_pchip_complex_and_nonfinite_values`.

## B-52 — MATLAB and NumPy elementary functions need not round identically (observation only)
- Where: transcendental functions (`cos`, `sin`, `exp`, `log`, `power`) in MATLAB vs NumPy.
- Observation: their last bit can differ between the two environments, so ported formulas are not expected to be bit-identical to MATLAB.
- Impact: ~1 ulp per operation, far inside the tolerance classes (D-12/D-13/D-14).
- Action: none; no code acts on this.

## B-110 — `VOXELISE` extends the grid along one axis only (quirk, reproduced)
- Where: `VOXELISE.m`, "Check that the output grid is large enough to cover the mesh" (`if ... elseif ... elseif`).
- Behaviour: when the mesh extends beyond the grid, a grid line at the mesh bound is added (and cropped afterwards) only along the first traced axis that needs it; the other axes are left short. The extra line only changes which neighbouring rays the odd-crossing corrections see.
- Handling: reproduced in `pyproteus.geometry.voxelisation.voxelise`; `test_mesh_larger_than_grid` (`xyz`, `yz`, `z`) agrees exactly. Irrelevant for PROTEUS, whose grids contain the vessel.

## B-111 — `VOXELISE` drops facets with a projected area below an absolute 1e-14 (quirk, reproduced)
- Where: `VOXELISEinternal`, `if abs(planecoC) < 1e-14, planecoC = 0`.
- Behaviour: the threshold is in the mesh's length unit squared (m² in PROTEUS, after `STLunit` scaling), so facets with a projected area under ~5e-15 m² give no crossing; the ray becomes odd and is filled from its neighbours. Unit-dependent, but such a facet is almost never crossed by a ray.
- Handling: reproduced (same threshold). Candidate for a relative threshold after parity.

## B-112 — default settings voxelise the mouse brain arteriole to an empty mask (observation)
- Where: `define_medium` with `make_default_settings(PATHS, 'mouse_brain')`: grid spacing c/(f0·6) ≈ 103 µm; the arteriole's bounding box is 0.3 × 0.3 × 1.2 mm with vessels thinner than a voxel.
- Behaviour: MATLAB and pyproteus both give 0 vessel voxels (the renal tree gives 4710 of 23.7 M). The default mouse-brain simulation has no vessel in the medium.
- Impact: PROTEUS behaviour, reproduced. `vessel_mask`/`define_medium` now warn (`EmptyVesselWarning`, naming dx; D-15). Reported to the main session, which flags it to the developer.

## B-100 — General DAS matrix: pixel at axial coordinate −0 (observation, outside the classes)
- Where: `compute_das_matrix_general` with a pixel at `x = -0.0` exactly above an element at `x_el = 0`, `y = y_el`.
- Symptom: found by the randomised oracle test: MATLAB keeps that (pixel, element) entry, pyproteus drops it.
- Cause: `atan2(0, -0) = pi` in IEEE 754 / NumPy, so the angle exceeds the threshold. The sign of zero is not part of the D-13 classes (as B-20); not investigated further in MATLAB (D-12).
- Workaround: none needed; the randomised test normalises `-0.0` to `+0.0` (`tests/oracle/test_das.py::general_case`). A pixel exactly on an element is degenerate anyway.

## B-90 — MATLAB `VOXELISE` errors when a mesh bound is equidistant from two grid lines (observation, for M4)
- Where: `mesh-voxelisation/Mesh_voxelisation/VOXELISE.m` (`VOXELISEinternal`, `meshXminp`...`meshYmaxp = find(abs(gridCO - meshmin) == min(...))`), called by `define_medium.m` with `EmbedVessel`.
- Symptom: a facet bound exactly half-way between two grid coordinates gives two indices, and the ray loop `for loopY = meshYminp:meshYmaxp` fails with "Colon operands must be real scalars". Found with a synthetic box STL in `tests/oracle/test_medium_define.py` (faces now 0.3 dx past a node).
- Impact: real meshes rarely hit an exact tie; the M4 port should pick one index (first) and note the difference.
- Resolved (2026-11-24, PR #19): `pyproteus.geometry.voxelise` does not use this nearest-grid-line range (it never excludes a ray a facet can cross), so ties raise no error; result equals MATLAB's once the tie is broken (`test_mesh_bound_halfway_between_grid_lines`). Divergence in PARITY.md (Geometry).

## B-101 — MATLAB PROTEUS DAS time axis: depth-dependent timing error of a point scatterer (observation, open)
- Where: `delay-and-sum/initialize_das_parameters.m` / `DAS_reconstruction.m` time axis applied to RF from `main_RF` (hybrid and full paths), oracle 8d37938.
- Symptom: a small bubble on the axis is imaged off its true two-way time by ≈ −26 ns at 5 mm and ≈ +8 ns at 7 mm (16 elements, 1-mm element height, no lens, homogeneous lossless medium, 51 MHz simulation, plane wave). Across one grid cell at 5 mm it varies by ±3 ns, with a jump where the nearest grid node changes (interpolation-stencil effect, not nearest-node snapping, which would give ±35 ns).
- Known contributions, each of the size of the 4 ns Q-100 term: the bubble radiates ≈ the second derivative of the asymmetric two-way pulse (envelope peak moved by ≈ −9 ns); bubble resonance group delay (estimated +5 to +7 ns); element-height path excess (≈ +4 to +6 ns, depth dependent). The remaining depth dependence behaves like near-field/diffraction pulse-shape change (aperture near field ≈ 9 mm).
- Cause: not established; the Q-100 one-sample term (4 ns) cannot be resolved against these.
- Next: far field, several depths, a linear (density) scatterer, and the expected waveform (d²V or cross-correlation) as reference, once the hybrid path is ported (M3). Scripts: `tools/investigations/q100_run.py` (driver), `q100_point_scatterer.m`, `q100_analyse.py`.

## B-130 — End-to-end RF vs MATLAB: 1.01e-4 × max on the default case, just above the D-13 1e-4 class (closed: D-24)
- Where: `tests/oracle/test_end_to_end_rf.py` (default P4-1 settings, 6 fixed bubbles, double precision, `step_control="shared"`).
- Measured: RF 1.01e-4 × max (shared), 6.0e-5 (independent); curved C1-6D 6.0e-5 / 5.3e-5; P4-1 `no_dispersion` (5 elements) 1.00e-4 / 5.9e-5. Propagation + RF from MATLAB's own mass source: 2.2e-13, so the difference is entirely the bubble mass source (2.1–2.4e-4 × max vs MATLAB, inside the D-22 bubble class of 1e-3).
- Cause: independent ODE solvers (D-12/D-22); a 1e-4 end-to-end class is not consistent with a 1e-3 bubble class.
- Status: closed by D-24 (developer): the end-to-end RF class is 1e-3 × max|RF| (the D-22 bubble class); the propagation + RF stage stays at 1e-9 from MATLAB's mass source.

## B-170 — `run_simulation` solver `'3D'`: k-wave-python python backend, no movie (divergence)
- Where: `pyproteus.acoustic.simulation.run_simulation` (`run_simulation.m`, `sim_setup.m` `Solver = '3D'`).
- MATLAB: `kspaceFirstOrder3D` (MATLAB-native solver) with `RecordMovie` (sim_setup sets `record_movie = true` for `'3D'`).
- pyproteus: k-wave-python's `python` backend (NumPy, in-process, `alpha_mode` dropped as on every backend, D-9); `record_movie` is ignored with a warning (no movie writer). Parity is against `kspaceFirstOrder3DC` (D-10): the python backend is within the k-Wave class of it (measured 5.1e-7 full source, 4.2e-7 self-sense). MATLAB-native `3D` is not compared (it honours `alpha_mode`, B-14).
- Status: by design (D-15); PARITY.md.

## B-171 — `update_source` with a single-precision mass source: one float32 ulp (divergence)
- Where: `pyproteus.acoustic.full.update_source` (`update_source.m:27`).
- MATLAB divides the single `mass_source` by the voxel volume in single precision, then converts to double; pyproteus divides in float64. Measured differences ≤ 1.04e-7 relative (176 of 49720 values), compared at 2^-23; double mass sources agree at 1e-9.
- Status: by design (D-12, D-15); PARITY.md.

## B-172 — self-sense delta function uses the global grid's full size (quirk Q-170, reproduced)
- Where: `pyproteus.acoustic.full.compute_self_sense_pressure` (`compute_self_sense_pressure.m:35`).
- Effect: ~0.2 % on the self-sensed pressure versus the local grid's own delta function (QUIRKS.md Q-170).
- Status: reproduced (Phase A); fix proposed for after parity.

## B-173 — self-sense impulse-response mode drops the local nonlinearity (by design, D-27)
- Where: `pyproteus.acoustic.full.compute_self_sense_pressure` (default `impulse_response=True`).
- Effect: exact for `BonA = 0`; with `BonA = 6` relative difference to the strict loop 4.5e-6 at a 5.9 kPa self-sense peak, 4.7e-5 at 59 kPa (above the k-Wave class from ~10 kPa).
- Status: developer decision D-27 (default on, `impulse_response=False` for strict-parity runs); PARITY.md.

## B-174 — homogeneous self-sense option changes results (opt-in, D-27)
- Where: `compute_self_sense_pressure(..., homogeneous=True)`.
- Effect: 4.8e-3 (Inhomogeneity 0.02) and 1.2e-2 (0.05) relative to the heterogeneous crop; up to 3.8e-4 PML-reflection residual even in a homogeneous medium (20-voxel core). Off by default; RF impact to be measured in M3 wave 2.
- Cost: one Green run per distinct local medium value set. With PROTEUS's per-voxel Inhomogeneity every bubble is distinct, so the option makes as many runs as the default mode and caches one 17³ x Nt float32 tensor per bubble (~98 MB at Nt = 5000). It only pays off for piecewise-constant media; keying on background (pre-inhomogeneity) or quantized values would fix that and is proposed to the main session (changes results further).
- Status: by design (D-27).

