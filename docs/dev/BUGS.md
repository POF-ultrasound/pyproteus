# Bugs and behaviour differences

Anything that behaves differently from what the plan or MATLAB expects. Status: open / workaround / fixed / wontfix.
Format: `## B-n — title (status)`, then where, symptom, cause, fix or workaround, link to test.

## B-1 — MATLAB engine breaks HTTPS in the same process (workaround)
- Where: any process that imports `matlab.engine` before `ssl`.
- Symptom: `URLError: unknown url type: https` (k-wave-python binary download); `_ssl` fails to load with a symbol missing from `/Applications/MATLAB_R2025a.app/bin/maca64/libcrypto.3.dylib`.
- Cause: the engine puts MATLAB's libcrypto first; conda's `_ssl` then binds to it.
- Workaround: import `ssl` (or `kwave`) before `matlab.engine`. The oracle fixture must do this.
