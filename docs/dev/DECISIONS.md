# Decisions

Format: `## D-n — title (date)`, then decision, why, who. Questions that need the developer go to the main `pyproteus` session first.

## D-1 — Hosting and visibility (2026-11-20)
POF-ultrasound/pyproteus, public from day one. PROTEUS-SIM/PROTEUS README links to it.

## D-2 — Two test phases (2026-11-20)
Phase A: MATLAB PROTEUS v1.1.0 is the oracle (live engine tests, then frozen references) until parity is confirmed. Phase B: CI tests pyproteus' own physics and code behaviour; oracle tests archived and non-gating.

## D-3 — Hardware scope for 1.0 (2026-11-20)
CPU on Linux/Windows/macOS, CUDA on Linux/Windows, Apple Silicon on CPU. No Slurm templates or containers before 1.0. GPU hardware test on the university GPU cluster before each release; CPU runs locally on the developer's Mac.

## D-4 — DL datasets (2026-11-20)
`pyproteus sweep` and a PyTorch `Dataset` reader are a backlog enhancement, not a 1.0 item.

## D-5 — Earlier roadmap decisions (2026-11-13)
Faithful port first, quirks included; clean-room `ode45`/`ode23` (no MathWorks or k-Wave MATLAB source translated); k-Wave tracer bullet first; package name `pyproteus`; GUI (NiceGUI) in 1.1; MIT license.

## D-6 — Self-merge (2026-11-20)
Milestone sessions merge their own PRs into `main` once CI is green and a reviewer agent has signed off. The developer reviews merged PRs afterwards and may revert.

## D-7 — Oracle reference (2026-11-20)
Until PROTEUS v1.1.0 is tagged, the oracle is PROTEUS commit `8d37938` (revamp branch), checked out read-only at `PROTEUS-cleaning/PROTEUS-oracle` (detached git worktree). Every reference file records this commit. Re-pin to the `v1.1.0` tag when it exists.

## D-8 — pchip coefficient form is clean-room (2026-11-20) — superseded by D-14
Keep the de Boor divided-difference form for the pchip power-form coefficients in `pyproteus.numerics` (`_hermite` in `src/pyproteus/numerics/interpolation.py`, see the "Newton form of the cubic with double nodes" comment). The form is derived from the Newton divided-difference representation of the cubic Hermite interpolant in de Boor, *A Practical Guide to Splines*. Among algebraically equal forms it was chosen by black-box comparison with MATLAB output (`unmkpp(pchip(...))` coefficients and evaluated values), because it is bit-identical to MATLAB where the textbook `(3δ - 2d_k - d_{k+1})/h` form is not. No MathWorks source was read. Raised by the PR #2 reviewer (resemblance to `pwch.m`, from memory, unverified); option (a) chosen by the developer over falling back to the ≤ 1e-12 textbook form. Who: developer, via the main session.

## D-9 — `alpha_mode` in self-sense runs on the k-wave-python `python` backend (2026-11-20)
Drop `alpha_mode` on every k-wave-python backend, so the `python` backend (M3 self-sense) also runs full power-law absorption with dispersion, exactly like MATLAB `kspaceFirstOrder3DC`/`3DG` (B-14, `docs/QUIRKS.md` Q-1). Faithful port; honour `no_dispersion` on all solvers in 1.x. The M3 adapter replaces the current "refuse on `python`" behaviour of `tools/tracer_bullet/adapter.py::medium_for_kwp`. Who: developer (round 1, via the main session).

## D-10 — Platform and binary for frozen k-Wave references (2026-11-20)
Generate the frozen k-Wave reference data on the developer's Mac (arm64): MATLAB `kspaceFirstOrder3DC` runs k-wave-python's prebuilt `kspaceFirstOrder-OMP` v1.4.1 through `tools/tracer_bullet/bin/kspaceFirstOrder-OMP`, since MATLAB k-Wave 1.4.1 ships no macOS arm64 binary. Parity therefore checks input preparation (scaling, row order, medium, PML), not the binary itself. (Implementation note from the main session, not part of the decision: each reference file records the binary path and version.) Optional, later: one spot check on Linux with the official k-Wave binary. Who: developer (round 1, via the main session).

## D-11 — Upstream k-wave-python reports (2026-11-20)
Draft issues for B-10, B-12 and B-13 in `docs/dev/upstream/k-wave-python-issues.md`. Nothing is posted upstream until the developer approves each text. Who: developer (round 1, via the main session).

## D-12 — Parity means medium tolerance, not a bit-for-bit copy of MATLAB (2026-11-22)
pyproteus reproduces MATLAB PROTEUS v1.1.0 behaviour and results within medium tolerances; it is not a bit-for-bit copy of MATLAB. Do not probe MATLAB (black-box) to reproduce its internals at ulp level: libm rounding, summation order, floating-point formula grouping, solver step sequences or step-size heuristics. Reverse-engineering proprietary software with that software is a legal problem, and MATLAB and Python differ fundamentally. Build a self-sufficient Python version from documented MATLAB semantics (round half away from zero, stable sort, interp1 NaN outside the range, column-major order, documented rng seeding), published papers and standard NumPy/SciPy routines, with workarounds where needed. Reading MATLAB data files exactly is fine. Who: developer.

## D-13 — Comparison classes (2026-11-22)
Replaces the ROADMAP §4 table targets:
- deterministic floats (geometry, delays, weights, signals, interpolation, truncated-normal medium): ≤ 1e-9 rel, with a small atol near zero;
- integers after rounding (masks, voxel and DAS sample indices): statistical, ≥ 99.9 % equal with the differences reported, and the downstream effect checked at the float/RF class (the 99.9 % threshold is the main session's default; adjust per case with a note in PARITY.md);
- bubble ODEs and mass source: about 10·RelTol rel + AbsTol abs of the solver settings in use (adaptive steps may differ), plus accuracy checks against analytical or high-accuracy references;
- random draws: same documented MT19937 stream, compared at the float class;
- streamlines: paths agree until the first voxel flip, and the distributions match;
- unchanged: k-Wave fields/RF ≤ 1e-5 rel; end-to-end RF ≤ 1e-4, image metrics equal.
Exact comparison stays where it falls out naturally (loaded data, integer settings). Who: developer (float, integer, ODE classes); main session (thresholds marked as defaults).

## D-14 — Replace code tuned by probing MATLAB (2026-11-22)
Supersedes D-8. The pchip coefficient grouping chosen to match MATLAB bit for bit, and the interp1 rules for flat segments and Inf/NaN neighbours found by probing, are replaced by standard textbook or SciPy forms, keeping only documented MATLAB semantics (NaN outside the range, shape-preserving end slopes, ...). Those oracle tests move to the D-13 float class. ulp-level observations (B-31, B-32) stay as notes, with no code to replicate them. Who: developer.

## D-15 — pyproteus is a better PROTEUS, not an exact copy (2026-11-23)
Maximise behavioural similarity to MATLAB PROTEUS v1.1.0 (the only reference so far), within the D-13 tolerances, but do not idealize PROTEUS or copy its implementation where a different Python design is better. Sessions and reviewers should challenge PROTEUS choices — toolbox and package dependencies, workflow, data handling and file formats, memory use, performance, API and usability — and use the better design when results stay within tolerance. Record each such divergence in `docs/PARITY.md` (what differs, why, how parity is still checked). Changes that alter results (including quirk fixes) are proposed to the main session for a developer decision and otherwise land in 1.x. Who: developer.

## D-16 — University cluster GPU (2026-11-23)
Quadro RTX 6000 (compute capability 7.5, driver 620.70.06, 24 GB). Meets the k-Wave CUDA binary's CC ≥ 7.5 requirement, so the release GPU test uses the C++/CUDA binary; the CuPy fallback is not needed for this cluster. 24 GB bounds the medium case (`pyproteus estimate`, M5). Who: developer (measurement).

## D-17 — Impulse response: accurate SOS instead of PROTEUS's (b,a) path (2026-11-23)
PROTEUS's `estimate_impulse_response` (`butter(3,Wn,'bandpass')` in (b,a) form, then `impz`) is ill-conditioned for poles near z = 1. Against an exact zpk partial-fraction reference, MATLAB is off by about 1e-8 for the default P4-1 and by 1–2 % for narrow bands. pyproteus uses SciPy second-order sections (≤ 5e-14 from the reference). Tests: vs the reference at 1e-12; vs MATLAB at 1e-7 relative to max|IR| for catalogue-like bands. Narrow-band MATLAB errors are logged as a PROTEUS bug (B-80) and not reproduced (QUIRKS). `PressureSignal` and the IR/PressureSignal defaults inherit this class. Who: developer (first result-changing improvement under D-15).

## D-18 — Delay-and-sum quirks (2026-11-23)
- Q-101 (TGC divides by `max(sqrt(t))` over complex values, giving an all-NaN image when |t(1)| > t(end)): fixed now with the real/magnitude maximum; results unchanged wherever MATLAB's output is valid.
- Q-102 (reference apodization keeps pixels at X ≤ 0): reproduced; fix in 1.x.
- Q-100 (image one sample deep from the 1-based `I/Fs` time-to-peak): first check that it is a genuine off-by-one inside MATLAB PROTEUS and not an indexing artefact of the port; if genuine, reproduce and fix in 1.x.
- Single-precision RF (PROTEUS default) images: compared at ≤ 1e-6 abs on the linear image, since single precision cannot meet 1e-9.
Who: developer (Q-100..Q-102); main session (single-precision class).

## D-19 — Grid PML split and shipped geometries (2026-11-23)
- Q-90: `define_grid(..., pml_rounding="ceil"|"floor")`. PROTEUS's `ceil` drops the odd point from the core grid (and empties a one-point axis); `floor` keeps it. Revised 2026-11-24: both shipped geometries have odd padding on all three axes, so `floor` would change every default k-Wave grid (renal tree core 446×298×178 → 448×300×180). Default is therefore `"ceil"` (MATLAB-identical) until 1.0; `"floor"` is available now and becomes the default in 1.x.
- The shipped STL geometries (Zenodo record 10573662) are downloaded into a gitignored folder and registered with pooch, so the default-geometry tests run.
Who: developer.

## D-20 — Optional linear interpolation in delay-and-sum (2026-11-24)
Both beamformers get `interpolation="linear"` (between RF samples; removes the .5-tie sensitivity of nearest-sample lookup and improves the point-spread function, Perrot et al. 2021). Default stays `"nearest"` for parity; switch the default in 1.x. Linear mode is tested against analytical/high-resolution references, not MATLAB. Who: developer.

## D-21 — Delay-and-sum improvements P5–P8 (2026-11-24)
From the PR #17 review, done now:
- P5: pixel axes span the domain (lateral `Ymin:…:Ymax`, reference beamformer starts at `Xmin`), fixing Q-103.
- P6: `compute_das_matrix_general` applies `Tx.apod`, fixing Q-104.
- P7: `log_compress` warns on all-zero frames instead of returning silent NaN.
- P8: the DAS matrix is cached per settings and RF size.
P5 and P6 are the default; a `proteus_compat=True` keyword restores PROTEUS behaviour for parity tests. Results are unchanged for symmetric domains and unapodized transmits. Who: developer (P5, P6); main session under D-15 (P7, P8, no result change).

## D-21 amendment (2026-11-24)
After the PR #26 review: (1) the reference beamformer's depth axis starts at `max(Xmin, 0)`, because real settings have `Xmin ≈ -2λ` (margin above the array) and starting at `Xmin` would add rows above the array face; results are then identical to MATLAB for real settings. (2) `Tx.apod` is used as a binary mask (`apod != 0`), not an amplitude weight, since the source is already scaled by the apodization (amplitude weighting would give apod²). Who: developer.

## D-22 — Bubble dynamics: comparison class, per-bubble steps, failure detection (2026-11-25)
- Class: the D-13 ODE bound (10·RelTol + AbsTol) is unreachable for `calcBubbleResponse`: MATLAB against itself (same bubble, different batch) misses it 17–34×. Bubble radius and velocity are compared per bubble at max|Δx| ≤ 1e-3·max|x| + 3e-5 vs MATLAB (abs term widened from 1e-5 on 2026-11-25: quiet bubbles crossing the Marmottant buckling kink, B-123), and pyproteus must stay within 1e-3 of the radial excursion against a high-accuracy reference (DOP853, rtol 1e-13).
- Per-bubble adaptive step control is the default, so results no longer depend on batch size (fixes Q-50). Batching must stay vectorised and faster than per-bubble loops; the shared-step stacked mode stays available as an option for MATLAB-comparison diagnostics.
- PROTEUS's wall-clock timer (60 s × N_MB) is replaced by deterministic failure detection (NaN/Inf, non-physical radius, step-size underflow, step/evaluation limits), which raises an error naming the failing bubbles, time and reason while the rest of the batch completes. An optional wall-clock limit stays available, off by default.
- Solver tolerances stay at PROTEUS's values.
Who: developer.

## D-23 — Homogeneous path: Kramers–Kronig reference frequency and FFT wrap (2026-11-25)
- Q-130: `calc_scatter_attenuated` uses reference frequency 0 in the power-law Kramers–Kronig dispersion, which is singular for `alpha_power` ≤ 1 (bubble echoes all NaN for AttenuationB ≤ 0.9). pyproteus uses a finite reference frequency (transmit centre frequency, c(ω0) = c0) only in that singular case and keeps PROTEUS's formula elsewhere; applying it for all b is in the 1.x backlog.
- Q-131: arrivals later than 2·Nt·dt no longer wrap into the RF window (FFT padding / far sources zeroed); nothing reachable with PROTEUS's own Nt sizing changes.
Who: developer.

## D-24 — End-to-end RF comparison class (2026-11-26)
End-to-end RF from fixed bubble positions is compared with MATLAB at ≤ 1e-3 × max|RF| (replacing D-13's 1e-4), consistent with the D-22 bubble class: the bubble ODE difference (mass source ~2e-4) passes into the RF almost one to one (measured RF 1.01e-4 for P4-1 with shared steps). The propagation + RF stage, fed MATLAB's mass source, stays at ≤ 1e-9 (measured ~1e-13), so propagation errors are still caught. Who: developer.

## D-25 — RF output format (2026-11-27)
`main_RF` writes one HDF5 file per run with one group per frame (RF per pulse in the sequence, `dt`, that frame's bubbles), plus the run settings and ground truth. Each frame is flushed when it is written, so a run can resume (`Acquisition.Continue` / `StartFrame`). PROTEUS writes one `Frame_N.mat` per frame instead. MATLAB `Frame_N.mat` files (ground truth and RF) stay readable, and the DAS loader accepts both formats. Who: developer.

## D-26 — Self-sense pressure in the full simulator (2026-11-27)
`compute_self_sense_pressure` (one local k-Wave run per bubble per interaction) is first ported strictly, as the parity reference. The M3 session then measures its cost and proposes batching or caching, for example one run per unique local medium and sub-voxel offset. Parts that keep results within the k-Wave class can go in directly. Anything that changes results comes back to the developer. Who: developer.

## D-27 — Self-sense speedups (2026-11-27)
Follows the M3-C measurements under D-26. The strict port matches MATLAB 3DC at ≤ 2.6e-7 and costs about 60 s per bubble per iteration at realistic sizes.
- Each bubble's self-sense impulse response is cached across interaction iterations, **on by default**, with a flag to turn it off for strict-parity runs. This is exact for BonA = 0. With BonA = 6 the error is 4.5e-6 to 4.7e-5 (self-sense peaks of 5.9 to 59 kPa): above the k-Wave class but far below the end-to-end RF class (D-24).
- A homogeneous local medium with a Green's-function tensor (no per-bubble k-Wave runs) is an **opt-in** option, off by default. Its self-sense error is 4.8e-3 at 2 % inhomogeneity and 1.2e-2 at 5 %. Wave 2 measures the effect on RF.
- Bubble-parallel runs (`n_jobs`) and the solver override stay in; results are within class.
- Q-170 (the self-sense delta function uses the global `full_size`, about 0.2 %) is reproduced, with the fix after parity.
Who: developer.

## D-28 — Multi-process RF runs (2026-11-27)
Extends D-25. Several processes running different frame ranges of one run (as with MATLAB `main_RF` on several GPUs) each write their own HDF5 file (`rf_<start>-<end>.h5`). Readers and DAS accept a list of files or a folder, and an optional tool merges them into one run file. This avoids HDF5 file locking, which is often disabled on cluster file systems. Built in M5. Until then one process writes one file. Who: developer.

## D-29 — Q-100 fix plan (2026-11-27)
Extends D-18, after the M3 far-field study (B-185: off-by-one at PPW 4/6/8). In 1.x: `timeToPeak = (I−1)/Fs`; the components of `V_ref` (transmit pressure signal, receive impulse response) resampled to one sampling rate before the convolution; the k-Wave staggered-grid correction derived or calibrated for each grid instead of the fixed `dx/(2c) + dt` (`+ 3/2 dt` full). 1.0 keeps reproducing Q-100 exactly as PROTEUS. Who: developer.

## D-30 — Exact receive propagation on the GPU (2026-11-27)
From B-187 (M2 cluster run: exact ×64 vs gridded ×8.7). When `DATA_CAST` is `gpuArray-*`, the hybrid receive stage uses exact distances by default (`gridded=False`); the CPU stays gridded (PROTEUS). Changes results by the distance rounding only. Measured with `tools/m3/gpu_medium_case.py` on the cluster by the developer before release. Who: developer.
