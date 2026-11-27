# pyproteus: session context

Read this file instead of the full docs (`CLAUDE.md`, always loaded, has the binding rules and environment). It holds every decision in one line, the comparison classes and the current state. Open other docs only for the section you need: `grep -n` for it, then read that range.

## Reading rules (token budget)
- Never read `docs/dev/LOG.md`, `docs/dev/BUGS.md`, `docs/PARITY.md` or `docs/QUIRKS.md` in full. To add an entry, read the first ~15 lines (format) and insert or append. To check an item, `grep -n "B-123\|Q-100"` and read only that entry.
- `docs/ROADMAP.md`, `docs/dev/MILESTONES.md`, `docs/dev/DECISIONS.md`: grep for the section you need (your milestone, one D-n). The one-liners below are usually enough.
- Old log entries (M0–M2) are in `docs/dev/log/`; don't read them unless a brief points you there.
- Read only the shared notes your brief names, not the whole notes folder.
- Reviewer agents: give them the PR diff, this file and the brief's acceptance criteria, not the whole doc set.

## Comparison classes (ROADMAP §4)
| Quantity | Class |
|---|---|
| Deterministic floats | ≤ 1e-9 rel |
| Integers after rounding | ≥ 99.9 % equal, downstream effect at the float class |
| Bubble radius, mass source | 1e-3·max\|x\| + 3e-5 (D-22) |
| k-Wave fields | ≤ 1e-5 rel |
| End-to-end RF | ≤ 1e-3 × max; propagation stage ≤ 1e-9 (D-24) |
| Single-precision DAS images | ≤ 1e-6 abs (D-18) |
| CPU vs GPU | 1e-9 double, 1e-6 single |

## Decisions (one line each; full text: `grep -n "^## D-N " docs/dev/DECISIONS.md`)
- D-1 Public repo POF-ultrasound/pyproteus. D-2 Phase A MATLAB oracle, then Phase B own CI. D-3 1.0 hardware: CPU on all OSes, CUDA on Linux/Windows. D-4 DL datasets in the backlog. D-5 MIT; clean-room ode45/ode23; GUI in 1.1.
- D-6 Sessions self-merge after green CI + reviewer. D-7 Oracle = PROTEUS 8d37938. D-8 superseded by D-14. D-9 `alpha_mode` dropped on all k-wave-python backends. D-10 k-wave-python OMP binary also used by MATLAB. D-11 Upstream issue drafts need developer approval.
- D-12 Medium tolerance, no probing. D-13 Comparison classes. D-14 Probed code replaced by standard forms. D-15 Better PROTEUS, not a copy. D-16 Cluster GPU: Quadro RTX 6000, CC 7.5, 24 GB.
- D-17 Impulse response via SciPy SOS. D-18 DAS: Q-101 fixed, Q-102 reproduced, Q-100 checked in M3, single-precision class 1e-6. D-19 `pml_rounding` default "ceil" until 1.0; STLs via pooch. D-20 Optional linear DAS interpolation (default nearest).
- D-21 DAS axes span the domain (depth from max(Xmin,0)), `Tx.apod` as a binary mask, zero-frame warning, matrix cache; `proteus_compat=True` restores PROTEUS behaviour.
- D-22 Bubble class; per-bubble steps by default (`step_control="shared"` as an option); failure detection instead of the timer.
- D-23 Kramers–Kronig finite reference frequency for b ≤ 1 only; no FFT wrap. D-24 End-to-end RF 1e-3 × max.
- D-25 RF output: one HDF5 per run, a group per frame, resumable; MATLAB Frame_N.mat stays readable. D-26 Self-sense: strict port first. D-27 Self-sense IR cache on by default (flag off for strict parity); Green's-function self-sense opt-in. D-28 Multi-process runs: one HDF5 per process, readers take lists (M5).

## Current state (update when a milestone moves)
- M0, M1, M2 done (M2 CuPy check passed on the cluster, PR #32).
- M3 done except "Medium case on GPU" (developer's cluster run: `tools/m3/gpu_medium_case.py`). Wave 1: sources/sensors (#34), run setup/HDF5 (#35), `run_simulation`/self-sense (#36). Wave 2: `acoustic/simulators.py` (`hybrid_simulator`, `full_simulator`), `acoustic/main.py` (`main_RF` → `rf.h5`, `main_pressure_field`, `prepare_simulation`); MATLAB `main_RF` parity on tiny/small cases, both paths (B-184). Q-100: off-by-one (PPW 4/6/8), k-Wave correction short by ≈ 0.05–0.1 dt (B-185, fix in 1.x). Open proposals: exact propagation by default on GPU (B-187); Green's-function self-sense stays opt-in (B-186).
- Next: M4 (flow and geometry), M5 (execution, CLI, hardware).
- Package map: `numerics/`, `config/`, `data/`, `io/`, `transducer/`, `transmit/`, `medium/`, `microbubble/`, `acoustic/` (grid, integration, homogeneous, rf, mass_source, sensor, source, setup, simulation, full, simulators, main), `geometry/`, `reconstruction/` (DAS), `backends/kwave.py`.
- Oracle `main_RF` cases: `tests/oracle/main_rf_case.py` (MATLAB settings + ground truth, `PROTEUS_LOCAL_PATHS` into a temp folder); `tools/m3/run_case.py` runs one case against a shared MATLAB session.
