# Rewrite log

Newest first. One entry per working session or milestone step: what was done, what was found, what is next. Link bugs (B-n) and decisions (D-n).

## 2026-11-20 — setup
- Repo created at POF-ultrasound/pyproteus (public). Roadmap v4 in `docs/ROADMAP.md`, checklists in `docs/dev/MILESTONES.md`.
- conda env `pyproteus`: Python 3.12, k-wave-python 0.6.2 (latest on PyPI), matlabengine 25.1.2 (MATLAB R2025a Update 1).
- Verified in the env: MATLAB `rng(42,'twister'); rand(1,3)` equals NumPy `RandomState(42).random_sample(3)` bit for bit.
- Found B-1 (MATLAB engine vs conda OpenSSL).
- Next: M0 (tracer bullet, repo setup, parity harness, numerics helpers, settings).
