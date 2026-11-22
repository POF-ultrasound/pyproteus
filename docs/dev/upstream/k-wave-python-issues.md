# Draft upstream issues for k-wave-python (D-11)

**Status: drafts only. The developer approves each one before it is posted** to https://github.com/waltsims/k-wave-python/issues. Found by the M0 tracer bullet (`tools/tracer_bullet/`, `tests/oracle/test_kwave_tracer.py`) against MATLAB k-Wave 1.4.1 `kspaceFirstOrder3DC`, using the same `kspaceFirstOrder-OMP` v1.4.1 binary on both sides. Line numbers refer to k-wave-python 0.6.2 from PyPI.

Before posting: re-check each issue against the latest k-wave-python release and its open issues, and attach a minimal script (a ~30-line reduction of `tools/tracer_bullet/cases.py`).

---

## Issue 1 (B-10): `kspaceFirstOrder(backend="cpp")` does not scale time-varying sources and returns sensor data as `(Nt, n_sensor)` in F order

**k-wave-python version:** 0.6.2 (macOS arm64 and Linux; Python 3.12)

**Summary.** With the new `kspaceFirstOrder()` API, the `cpp` and `python` backends interpret the same inputs differently:

| | source rows expected | additive source scaling | `result["p"]` |
|---|---|---|---|
| `backend="python"` | C order | applied internally | `(n_sensor, Nt)`, rows in C order (as documented) |
| `backend="cpp"` | F order (`find` order) | **not applied** | `(Nt, n_sensor)`, columns in F order |

**Details** (`kwave/solvers/cpp_simulation.py`):
1. `_write_hdf5` writes `source.p` (and `ux`/`uy`/`uz`) unchanged to `p_source_input` (lines ~256-264), next to an F-order mask index. The k-Wave MATLAB front end multiplies additive time-varying sources by `2·dt/(N·c0·dx)` (pressure) or `2·c0·dt/dx` (velocity) before writing that dataset (k-Wave User Manual 1.1, time-varying sources). Without that scaling the simulation diverges (NaN) for typical source amplitudes.
2. The C++ binary output `p` is read back as `(Nt, n_sensor)`. The F→C row reordering (lines ~96-112) only fires when `val.shape[0] == n_sensor`, so it is skipped and the output is neither the documented shape nor in C order. When `Nt == n_sensor`, it fires on the **time** axis instead and permutes time samples.

**Expected.** Both backends accept the same, documented convention (C-order rows, unscaled signals) and return `(n_sensor, Nt)` in C order.

**Reproduction.** A 37×29×23 grid with a heterogeneous medium, an asymmetric 95-voxel `p_mask`, distinct per-voxel signals, and a binary sensor mask. Compare `backend="cpp"` with `backend="python"`, and with MATLAB `kspaceFirstOrder3DC`. With our workaround (scale the rows ourselves, transpose, undo the permutation), both backends match MATLAB to < 7e-7 relative, and the cpp `p_source_input` dataset is bit-identical to MATLAB's.

---

## Issue 2 (B-12): legacy `kspaceFirstOrder3D`: pressure-source scaling uses the sound speed of the wrong voxel; velocity sources crash with per-voxel signals

**k-wave-python version:** 0.6.2

**Summary.** In `kwave/kWaveSimulation_helper/scale_source_terms_func.py`:
1. **Pressure, heterogeneous `c0`:** `scale_pressure_source_uniform_grid` indexes `c0.ravel(order="F")` with `p_source_pos_index`, which holds MATLAB-style 1-based linear indices. Each source row is therefore scaled with the sound speed of the next voxel along x. Against MATLAB k-Wave the per-row scale ratio is 0.995-1.005, and the sensor data are 3.8e-3 relative off. A homogeneous `c0` hides it.
2. **Velocity, one signal per voxel:** `range(source_val.size[0])` raises `TypeError: 'int' object is not subscriptable`, because `ndarray.size` is an int (`.shape[0]` was presumably intended). It is hit by per-voxel velocity signals (line ~323; non-uniform grid ~361). The same `.size[0]` bug sits in the pressure non-uniform-grid path (~165) and the stress path (~233).
3. **Likely, not verified:** the velocity path (~323) also indexes `c0[u_source_pos_index[...]]` with the 1-based indices, so it probably has the same off-by-one as item 1 once the crash is fixed.

**Expected.** Scale with `c0` at the source voxel (0-based index), and support per-voxel velocity signals, as MATLAB k-Wave does.

---

## Issue 3 (B-13): `CppSimulation._compute_staggered_density` wraps around at the last grid plane

**k-wave-python version:** 0.6.2

**Summary.** `_compute_staggered_density` (`kwave/solvers/cpp_simulation.py` ~line 293-304) computes `rho0_sg{x,y,z}` as `0.5 * (rho0 + np.roll(rho0, -1, axis))`. On the last plane along each axis this averages with the **first** plane. MATLAB k-Wave 1.4.1 keeps `rho0` on that plane, so the HDF5 inputs differ there by up to 0.55 % (heterogeneous medium). Everywhere else they agree to ≤ 2 float32 ulp. (The TODO next to it notes that `np.roll` averaging is an approximation of spectral interpolation; it does not mention the boundary wrap.)

**Impact.** Small: the plane sits inside the outer PML when `PMLInside=False`. In our tracer case it adds about 5e-7 relative to the sensor data. It can matter with `PMLInside=True` and strong density contrast at the boundary.

**Expected.** Match MATLAB k-Wave's staggered-grid density on the boundary plane (no wrap-around).

---

## Issue 4 (B-62): `kspaceFirstOrder(backend="cpp")` ignores `sensor.record`

**Status: draft, not posted.**

**k-wave-python version:** 0.6.2

**Summary.** With `backend="cpp"`, `kspaceFirstOrder()` warns that `sensor.record` is not yet supported and runs the binary without output flags, so only `p` is recorded. `sensor.record = ["p_max", "p_min"]` returns `p` instead. The `python` backend honours `sensor.record`.

**Details.** `CppSimulation._execute` (`kwave/solvers/cpp_simulation.py` ~line 353) builds `-i/-o/-t/-g/--verbose` only. MATLAB k-Wave 1.4.1 `kspaceFirstOrder3DC` maps `sensor.record` to binary flags; captured from MATLAB for `{'p'}`, `{'p_max','p_min'}` and `{'p','p_max','p_min'}`: `--p_raw`, `--p_max`, `--p_min` (in that order), plus `-t n` for `NumThreads`.

**Expected.** Translate `sensor.record` into the binary's output flags (and `record_start_index` into `-s`), and return those fields with the same shapes as the `python` backend.

---

## Issue 5 (B-60): `filter_time_series(..., zerophase=True)` raises `IndexError`

**Status: draft, not posted.**

**k-wave-python version:** 0.6.2

**Summary.** `kwave.utils.filters.apply_filter` with `zero_phase=True` (reached from `filter_time_series(zerophase=True)`) fails for every input with `IndexError: index L+N is out of bounds`.

**Details.** The reverse pass (~line 541) indexes `filtered_signal[np.arange(L + N, 1, -1)]`, a 1-based range carried over from MATLAB (valid 0-based indices are `L + N - 1` down to `0`), then calls `np.fliplr` on a 1-D array. Expected: filter the time-reversed signal and reverse the result.

**Check.** Built from the causal path instead (filter at half the stop-band attenuation, forwards, then backwards on the reversed result), the output matches MATLAB k-Wave `filterTimeSeries(..., 'ZeroPhase', true)` to ≤ 1e-12 relative on several signals and grids, so a fix of the indexing should give the same.
