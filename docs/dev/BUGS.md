# Bugs and behaviour differences

Anything that behaves differently from what the plan or MATLAB expects. Status: open / workaround / fixed / wontfix.
Format: `## B-n — title (status)`, then where, symptom, cause, fix or workaround, link to test.

## B-1 — MATLAB engine breaks HTTPS in the same process (workaround)
- Where: any process that imports `matlab.engine` before `ssl`.
- Symptom: `URLError: unknown url type: https` (k-wave-python binary download); `_ssl` fails to load with a symbol missing from `/Applications/MATLAB_R2025a.app/bin/maca64/libcrypto.3.dylib`.
- Cause: the engine puts MATLAB's libcrypto first; conda's `_ssl` then binds to it.
- Workaround: import `ssl` (or `kwave`) before `matlab.engine`. The oracle fixture must do this.

## B-20 — interp1/griddedInterpolant with matrix v: sign of zero on flat segments (open, minor)
- Where: `pyproteus.numerics.interpolation._linear` (flat-segment rule `v0 == v1 → v0`).
- Symptom: with matrix-valued `v` (one column per series), MATLAB R2025a returns +0 where pyproteus returns -0 on some flat segments made of ±0, e.g. `x=[0 1 3]`, `V(:,2)=[-0 0 1]`, `xq=0` or `0.5`. 76 of 2,187 special-value comparisons differ, all only in the sign of zero. Vector `v` matches MATLAB exactly.
- Cause: MATLAB appears to use a different code path for matrix values (black-box observation).
- Impact: none known; PROTEUS only interpolates vector `v`. Values are equal (`-0 == 0`).
- Fix: not planned unless a ported function interpolates matrix-valued `v` with signed zeros.

## B-10 — k-wave-python 0.6.2 `kspaceFirstOrder()`: backends disagree on source rows, scaling and output shape (workaround)
- Where: `kwave.kspaceFirstOrder.kspaceFirstOrder` (the >= 0.6.1 API), `backend="cpp"` vs `backend="python"`.
- Symptom: with MATLAB-convention inputs (rows in `find` order, unscaled signals) the `cpp` backend goes unstable (NaN); with C-order rows the results are wrong (1.2-1.6 rel); the documented `(n_sensor, Nt)` output is actually `(Nt, n_sensor)` for `cpp`.
- Cause (`kwave/solvers/cpp_simulation.py`): the cpp path writes `source.p`/`ux`/`uy` to the HDF5 file unchanged next to an F-order (`find`) index list and does not apply k-Wave's time-varying source scaling; it reads `p` back as `(Nt, n_sensor)`, and its F→C row reordering only runs when `shape[0] == n_sensor`, so it is skipped (and if `Nt == n_sensor` it would permute the time axis). The `python` backend expects C-order source rows, scales internally and returns `(n_sensor, Nt)` in C order.
- Workaround: `tools/tracer_bullet/adapter.py` (to become `backends/kwave.py`): cpp = find-order rows × `additive_source_scale` (p: `2dt/(3 c0 dx)`, u: `2 c0 dt/dx`, per voxel), output transposed (and, if `Nt == n_sensor`, k-wave-python's spurious time-axis permutation undone); python = C-order rows, output reordered C→F. With it both backends match MATLAB k-Wave to < 7e-7 rel and the cpp source datasets are bit-identical to MATLAB's. Pin k-wave-python exactly. Test: `tests/oracle/test_kwave_tracer.py`.

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
- Symptom: `rho0_sgx/y/z` in the HDF5 input differ from MATLAB k-Wave's only on the last grid plane along the staggered axis (inside the outer PML), up to 0.55 % rel; elsewhere ≤ 2 float32 ulp.
- Cause: `np.roll` averages the last plane with the first; MATLAB k-Wave keeps `rho0` there. Contributes the ~5e-7 residual in sensor data, well below the 1e-5 target.
- Workaround: none needed for parity; if a case ever misses 1e-5, write the HDF5 input with the adapter's own `rho0_sg*` (fallback route in ROADMAP §6). Test: `test_cpp_input_file_matches_matlab` (asserts it is still the only difference).

## B-14 — PROTEUS `alpha_mode='no_dispersion'` has no effect on the k-Wave C++ path (quirk, reproduced)
- Where: oracle `acoustic-module/define_medium.m` sets `medium.alpha_mode = 'no_dispersion'` when `|alpha_power - 1| < 0.1`; `run_simulation.m` then calls `kspaceFirstOrder3DC`/`3DG`.
- Symptom: MATLAB 3DC output is bit-identical with and without `alpha_mode` (the C++ HDF5 format has no field for it), so PROTEUS runs full power-law absorption + dispersion on the C++/CUDA path; the MATLAB-native `kspaceFirstOrder3D` (PROTEUS' local-movie mode) and k-wave-python's `python` backend honour it (16 % rel difference on the tracer case with y = 1.05). k-wave-python's `cpp` backend raises `ValueError`.
- Faithful-port handling: the adapter drops `alpha_mode` for `cpp` (matches MATLAB 3DC to 5.9e-7) and refuses it for `python` (`adapter.medium_for_kwp`). Recorded as `docs/QUIRKS.md` Q-1. Decided (D-9): M3 drops `alpha_mode` on the `python` backend too.
- Test: `test_matlab_3dc_ignores_alpha_mode`, case `pressure_nodisp`.

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


## B-52 — MATLAB and NumPy elementary functions need not round identically (observation only)
- Where: transcendental functions (`cos`, `sin`, `exp`, `log`, `power`) in MATLAB vs NumPy.
- Observation: their last bit can differ between the two environments, so ported formulas are not expected to be bit-identical to MATLAB.
- Impact: ~1 ulp per operation, far inside the tolerance classes (D-12/D-13/D-14).
- Action: none; no code acts on this.
