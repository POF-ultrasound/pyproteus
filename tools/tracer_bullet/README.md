# k-Wave tracer bullet (M0)

Checks that k-wave-python 0.6.2 reproduces MATLAB k-Wave 1.4.1 on tiny
PROTEUS-style hybrid cases. The row-order and source-scaling mapping it found
now lives in the package (`pyproteus.backends.kwave`, M3); this folder imports
it from there and keeps the cases, the MATLAB runner and the binary wrapper,
which `tests/oracle/test_kwave_backend.py` reuses. Findings: BUGS.md
B-10 to B-14 and `docs/dev/LOG.md` (2026-11-20).

| File | Purpose |
|---|---|
| `cases.py` | the cases (37×29×23, asymmetric masks): pressure (linear array), velocity `ux`+`uy` (curved array), velocity `ux` only, mixed velocity + pressure sources, pressure with `alpha_mode='no_dispersion'`, `Nt == n_sensor` recording `p`/`p_max`/`p_min` |
| `kwave_tracer.py` | runners: MATLAB `kspaceFirstOrder3DC` via the engine, k-wave-python `kspaceFirstOrder()` `cpp`/`python` and legacy API (mapping from `pyproteus.backends.kwave`); HDF5 input diff |
| `run_kwave_matlab.m` | MATLAB side, options as PROTEUS `run_simulation.m`; returns every recorded field |
| `run_tracer.py` | runs everything, writes `data/` (gitignored): k-Wave HDF5 files, `tracer_reference.h5`, `report.md` |
| `bin/kspaceFirstOrder-OMP` | wrapper so MATLAB and Python run the same binary with its runtime libraries |
| `setup_libs.sh` | creates `.kwave-libs/` (gitignored conda prefix: fftw, hdf5, libomp, zlib) |

## Setup (macOS arm64)

MATLAB k-Wave 1.4.1 ships no macOS arm64 binary. Both sides use the binary
k-wave-python downloads (`kspaceFirstOrder-OMP` v1.4.1), which links Homebrew
paths. Either `brew install fftw hdf5 zlib libomp`, or (no system change):

```bash
tools/tracer_bullet/setup_libs.sh
```

Then (MATLAB needs to run outside the Claude Bash sandbox):

```bash
~/miniconda3/envs/pyproteus/bin/python -m pytest tests/oracle/test_kwave_tracer.py
```

```bash
~/miniconda3/envs/pyproteus/bin/python tools/tracer_bullet/run_tracer.py
```

`KWAVE_MATLAB_PATH` overrides the MATLAB k-Wave folder (default: the
developer's `PROTEUS-cleaning/PROTEUS/external/k-wave/k-Wave`).

## Licensing

k-Wave MATLAB (LGPL) and k-wave-python (LGPL) are called, never copied or
translated. The additive source scaling factors come from the k-Wave User
Manual 1.1 (time-varying sources) and k-wave-python's documented python-solver
scaling (citing Cox et al., IEEE IUS 2018); they match MATLAB's input files bit
for bit.
