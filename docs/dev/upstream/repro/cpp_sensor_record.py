"""k-wave-python 0.6.2: kspaceFirstOrder(backend="cpp") ignores sensor.record.

The python backend returns p_max; the cpp backend warns and returns only p.
MATLAB k-Wave 1.4.1 kspaceFirstOrder3DC passes record = {'p','p_max','p_min'}
to the binary as --p_raw --p_max --p_min. Set KWAVE_BINARY_PATH for a custom binary.
"""

import os

import numpy as np
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder import kspaceFirstOrder

N = (24, 20, 16)
kgrid = kWaveGrid(N, (1e-4,) * 3)
kgrid.makeTime(1500.0, t_end=2e-6)
medium = kWaveMedium(sound_speed=1500.0, density=1000.0)
source = kSource()
source.p0 = np.zeros(N)
source.p0[12, 10, 8] = 1.0
sensor = kSensor(np.zeros(N, dtype=bool))
sensor.mask[16, 10, 8] = sensor.mask[8, 12, 6] = True
sensor.record = ["p_max"]

common = dict(pml_size=6, pml_alpha=2.0, pml_inside=False, smooth_p0=False, device="cpu", quiet=True)
py = kspaceFirstOrder(kgrid, medium, source, sensor, backend="python", **common)
cpp_kw = {"binary_path": os.environ["KWAVE_BINARY_PATH"]} if "KWAVE_BINARY_PATH" in os.environ else {}
cpp = kspaceFirstOrder(kgrid, medium, source, sensor, backend="cpp", **common, **cpp_kw)
print("python fields:", sorted(py))  # includes 'p_max'
print("cpp fields:   ", sorted(cpp))  # only 'p' (plus metadata)
