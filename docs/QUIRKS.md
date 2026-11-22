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
- Ported in: `tools/tracer_bullet/adapter.py` `medium_for_kwp` (drops `alpha_mode` for the k-wave-python `cpp` backend; refuses it for the `python` backend, which would apply it: 16 % different on the tracer case). Moves to `pyproteus.backends.kwave` in M3.
- Test: `tests/oracle/test_kwave_tracer.py::test_matlab_3dc_ignores_alpha_mode`, case `pressure_nodisp`.
- Decided (D-9): drop `alpha_mode` on the `python` backend too (M3 self-sense), so every backend matches MATLAB 3DC.
- Planned fix: after parity, honour `no_dispersion` on every backend (or drop the setting from `define_medium`), so all solvers agree; changes absorption-dispersion near y = 1.

## Q-2 — Bubbles integrated in one batch share ode45's adaptive steps
- MATLAB: `microbubble-simulator/calcBubbleResponse.m:28-52` stacks `[x; xdot]` of all bubbles of a batch into one state vector for one `ode45` call; `acoustic-module/compute_bubble_mass_source.m` (`get_batch_indices`) groups bubbles into batches after sorting by radius.
- Behaviour: ode45 takes one step size for the whole vector (error norm = max over all components), so a bubble's response depends on which other bubbles are in its batch and on the batch size. Rayleigh–Plesset-like probe: bubble 1 alone vs in a batch of 3 differs by 1.3e-6 rel in radius (MATLAB and pyproteus alike).
- Ported in: `pyproteus.numerics.ode45` (one state vector, shared steps; batching itself comes with `compute_bubble_mass_source`).
- Test: `tests/oracle/test_ode45.py::test_batch_dependence_quirk`, `tests/unit/test_ode45.py::test_vectorised_batch_shares_steps`.
- Planned fix: after parity, integrate each bubble with its own step control (vectorised solver with per-bubble steps, or one solve per bubble), so results no longer depend on batching; changes bubble responses at the ~1e-6 level.
