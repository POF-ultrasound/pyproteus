"""k-wave-python 0.6.2: kspaceFirstOrder(backend="cpp") vs backend="python".

Shows (1) the cpp backend does not scale additive pressure sources (the result
is off by exactly 1 / (2 dt / (N c0 dx))), and (2) cpp returns p as (Nt, n_sensor)
with sensor columns in Fortran (MATLAB find) order instead of (n_sensor, Nt) in C order.
Set KWAVE_BINARY_PATH to use a custom OMP binary.
"""

import os

import numpy as np
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder import kspaceFirstOrder

N, dx, c0 = (32, 28, 24), 1e-4, 1500.0
kgrid = kWaveGrid(N, (dx, dx, dx))
kgrid.makeTime(c0, cfl=0.3, t_end=4e-6)
nt, dt = int(kgrid.Nt), float(kgrid.dt)
medium = kWaveMedium(sound_speed=c0, density=1000.0)

source = kSource()
source.p_mask = np.zeros(N, dtype=bool)
source.p_mask[16, 14, 12] = True
t = np.arange(nt) * dt
source.p = (1e3 * np.sin(2 * np.pi * 2e6 * t) * (t < 1.5e-6))[None, :]

sensor = kSensor(np.zeros(N, dtype=bool))
for idx in [(20, 10, 12), (10, 14, 18), (16, 20, 6)]:  # F order differs from C order
    sensor.mask[idx] = True

common = dict(
    pml_size=8, pml_alpha=2.0, pml_inside=False, smooth_p0=False, device="cpu", quiet=True
)
py = np.asarray(kspaceFirstOrder(kgrid, medium, source, sensor, backend="python", **common)["p"])
cpp_kw = (
    {"binary_path": os.environ["KWAVE_BINARY_PATH"]} if "KWAVE_BINARY_PATH" in os.environ else {}
)
cpp = np.asarray(
    kspaceFirstOrder(kgrid, medium, source, sensor, backend="cpp", **common, **cpp_kw)["p"]
)

print("python p shape:", py.shape, " cpp p shape:", cpp.shape)  # (3, Nt) vs (Nt, 3)
# map cpp columns (F order) to C order explicitly
lin_f_of_c = np.ravel_multi_index(np.nonzero(sensor.mask), N, order="F")
col_of_c_row = np.argsort(
    np.argsort(lin_f_of_c)
)  # position of each C-order sensor among F-order columns
cpp_rows = cpp.T[col_of_c_row]

expected_scale = 2 * dt / (len(N) * c0 * dx)
ratio = np.max(np.abs(cpp_rows)) / np.max(np.abs(py))
print(f"max|cpp| / max|python| = {ratio:.6e};  1 / (2 dt / (N c0 dx)) = {1 / expected_scale:.6e}")
fixed = cpp_rows * expected_scale
print("rel. diff after scaling + reordering:", np.max(np.abs(fixed - py)) / np.max(np.abs(py)))
naive = cpp.T * expected_scale  # scaled, but columns taken in the order returned
print(
    "rel. diff with scaling only (no reordering):", np.max(np.abs(naive - py)) / np.max(np.abs(py))
)
