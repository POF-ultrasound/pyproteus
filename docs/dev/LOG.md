# Rewrite log

Newest first. One entry per working session or milestone step: what was done, what was found, what is next. Link bugs (B-n) and decisions (D-n).
Add new entries at the top; don't read the whole file. M0–M2 entries: `docs/dev/log/2026-11-M0-M2.md`.

## 2026-11-27 — M3 wave 1A: k-Wave sources and sensors (m3-sources-sensors)
- `pyproteus.acoustic.sensor`: `update_sensor` (= `update_sensor_fast`, one vectorised chunked implementation), `define_sensor_transducer`, `define_sensor_MB`, `define_sensor_MB_all` (takes loaded frames: session B's `load_microbubbles` plugs in as a generator), `extract_sensor_data`, `Sensor`. `pyproteus.acoustic.source`: `preprocess_transmit`, `define_source_transducer` (pressure/velocity, absent/dipole/monopole `SourceType`, curved `ux`/`uy` with mask merge, every `SeqPulse`), `pulsing_sequence`, `Source`.
- Conventions: 0-based column-major linear indices; masks `[ix, iy, iz]`; source rows and weight columns in `find` order (what `backends.kwave.kspace_first_order` takes); weights SciPy CSR `N_points × nnz(mask)`.
- Oracle: 29 sensor tests (both MATLAB update_sensor versions, ties at .5, edge truncation, on-grid lines, threshold, existing mask, default-settings grid, real `Frame_XX.mat` files for `define_sensor_MB_all`), 31 source tests (P4-1/C1-6D, on/off grid, all source types and pulses; `preprocess_transmit` with MATLAB k-Wave). Masks/indices exact, weights and signals ≤ 1e-9. 12 unit tests (loop reference, chunk invariance, C-order mapping, full-FFT delay reference).
- Found B-150 (`update_sensor_fast.m` weight columns wrong on a pre-filled mask; unreachable in PROTEUS; pyproteus follows `update_sensor.m`), B-151 (unknown `SeqPulse`/`PulsingScheme` raise), Q-150 (no staggered-grid shift when `SourceType` is absent, which is every `main_RF` run).
- Performance: 20 000 off-grid points in 3.3 s vs 31 s in MATLAB (fast version falls back to its loop above its 2 GiB estimate).

## 2026-11-27 — M3 plan and wave 1 (main)
- Recorded D-25 (HDF5 RF output, one file per run, resumable) and D-26 (self-sense: strict port first, then a speedup proposal).
- Started wave 1, three parallel sessions on disjoint files:
  - A, sources and sensors (B-150..159);
  - B, ground-truth frames, run setup and HDF5 I/O (B-160..169; frame reading moved forward from M4);
  - C, `run_simulation`, `update_source` and self-sense (B-170..179).
- Wave 2 starts once A–C have merged: hybrid and full simulators, `main_RF`, `main_pressure_field`, MATLAB parity, and the Q-100 far-field check (B-180..189).
- Real-GPU CuPy test merged (PR #32); the cluster run passed and M2 is done. D-27 (self-sense speedups) recorded after the M3-C measurements.

## 2026-11-27 — real-GPU CuPy parity test (main)
- `tests/gpu/test_cupy_parity.py` (marker `gpu`): NumPy vs CuPy on propagation (`run_simulation_homogeneous`, gridded and exact) + element RF (`compute_RF_data`), 300 bubbles → 96 × 5 integration points, single and double; host accumulation above `MaxGpuElements`; block-size independence on the GPU; `DEVICE_NUM` with two devices; slow benchmark printing CPU/GPU times. Classes: double ≤ 1e-9 × max (D-13), single ≤ 1e-6 × max (D-18).
- Skipped without CuPy or a CUDA device (Mac, CI). Dry-run against the NumPy-backed CuPy stand-in: 7 passed, 1 skipped (two-device test). The real run is for the developer on the cluster (D-16); closes the M2 CuPy item once it passes.
- Cluster run (developer, 2026-11-27): Quadro RTX 6000, CC 7.5, CuPy 14.2.0, 3 devices. All 8 tests pass, including `DEVICE_NUM`.
  - Max relative difference: p ≤ 9.1e-16 and RF ≤ 7.4e-16 in double; p ≤ 1.9e-13 and RF ≤ 2.4e-12 in single.
  - Propagation speedup: ×4–4.4 gridded and ×20–41 exact on the test case; ×8.7 gridded and ×64 exact on the benchmark (1000 bubbles, 864 points, single).
  - On the GPU the exact path is faster than the gridded one (gridded loops per source in small blocks): candidate for wave 2 / 1.x (e.g. exact by default on GPU, or a batched gridded kernel). M2 is done.
