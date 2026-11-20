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

## D-8 — pchip coefficient form is clean-room (2026-11-20)
Keep the de Boor divided-difference form for the pchip power-form coefficients in `pyproteus.numerics` (`_hermite` in `src/pyproteus/numerics/interpolation.py`, see the "Newton form of the cubic with double nodes" comment). The form is derived from the Newton divided-difference representation of the cubic Hermite interpolant in de Boor, *A Practical Guide to Splines*. Among algebraically equal forms it was chosen by black-box comparison with MATLAB output (`unmkpp(pchip(...))` coefficients and evaluated values), because it is bit-identical to MATLAB where the textbook `(3δ - 2d_k - d_{k+1})/h` form is not. No MathWorks source was read. Raised by the PR #2 reviewer (resemblance to `pwch.m`, from memory, unverified); option (a) chosen by the developer over falling back to the ≤ 1e-12 textbook form. Who: developer, via the main session.
