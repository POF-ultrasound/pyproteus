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
