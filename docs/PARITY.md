# Parity with MATLAB PROTEUS

Status of each ported function against the MATLAB oracle (PROTEUS commit
`8d37938`, MATLAB R2025a). Comparison classes from ROADMAP §4:

- **exact**: integer-valued results (indices, masks, rounding) equal; random draws bit-identical
- **1e-12 rel**: geometry, delays, weights, signals
- **ulp**: within a few ulp (e.g. `erfcinv`)
- see ROADMAP §4 for the bubble, k-Wave and end-to-end classes

Status: ✅ oracle test passes, 🟡 partial, ❌ fails / open bug, ⬜ not started.

## Numerics helpers (`pyproteus.numerics`)

| MATLAB | Python | Class | Oracle test | Status |
|---|---|---|---|---|
| test harness (array transfer) | `tests/oracle/helpers.py` | exact | `tests/oracle/test_numerics_harness.py` | ✅ |
