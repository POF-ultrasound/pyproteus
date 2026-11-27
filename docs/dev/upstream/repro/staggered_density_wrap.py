"""k-wave-python 0.6.2: rho0_sg* in the cpp input wrap around at the last grid plane.

CppSimulation._compute_staggered_density uses 0.5 * (rho0 + np.roll(rho0, -1, axis)),
so the last plane averages with the first one. MATLAB k-Wave 1.4.1 keeps rho0 there.
"""

import h5py
import numpy as np
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.solvers.cpp_simulation import CppSimulation

N = (16, 12, 10)
kgrid = kWaveGrid(N, (1e-4,) * 3)
kgrid.makeTime(1500.0)
rho0 = np.full(N, 1000.0)
rho0[0, :, :] = 2000.0  # dense first plane along x
medium = kWaveMedium(sound_speed=1500.0, density=rho0)
source = kSource()
source.p0 = np.zeros(N)
source.p0[8, 6, 5] = 1.0
sensor = kSensor(np.ones(N, dtype=bool))

sim = CppSimulation(kgrid, medium, source, sensor, pml_size=(4, 4, 4), pml_alpha=(2.0, 2.0, 2.0))
input_file, _ = sim.prepare()
with h5py.File(input_file) as f:
    rho_sgx = f["rho0_sgx"][()]  # stored transposed (z, y, x)
print("rho0_sgx on the last x plane:", np.unique(rho_sgx[:, :, -1]))  # 1500 (wrapped), MATLAB: 1000
