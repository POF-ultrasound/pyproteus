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
- Q-90: `define_grid` uses `PML = floor((M-N)/2)` instead of PROTEUS's `ceil`, so with odd M-N the extra point stays in the core grid and a one-point axis is not emptied. Grids then differ from MATLAB by one core point in those cases; oracle tests document that difference and compare everything else.
- The shipped STL geometries (Zenodo record 10573662) are downloaded into a gitignored folder and registered with pooch, so the default-geometry tests run.
Who: developer.
