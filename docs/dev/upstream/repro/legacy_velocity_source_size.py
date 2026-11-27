"""k-wave-python 0.6.2: legacy kspaceFirstOrder3D crashes for per-voxel velocity signals.

scale_source_terms_func.scale_velocity_source uses range(source_val.size[0]); ndarray.size
is an int, so this raises TypeError whenever c0 is heterogeneous and source.ux has one row
per source voxel. Same pattern at the pressure non-uniform, stress and velocity non-uniform paths.
"""

import numpy as np
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksource import kSource
from kwave.ksensor import kSensor
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions

N = (24, 20, 16)
kgrid = kWaveGrid(N, (1e-4,) * 3)
kgrid.makeTime(1500.0, t_end=2e-6)
c0 = np.full(N, 1500.0)
c0[:, :, 8:] = 1550.0  # heterogeneous sound speed
medium = kWaveMedium(sound_speed=c0, density=1000.0)
source = kSource()
source.u_mask = np.zeros(N, dtype=bool)
source.u_mask[12, 8:12, 8] = True  # 4 voxels
t = np.arange(int(kgrid.Nt)) * float(kgrid.dt)
source.ux = np.vstack([np.sin(2 * np.pi * 2e6 * t + k) for k in range(4)]) * 1e-3  # one row per voxel
sensor = kSensor(np.zeros(N, dtype=bool))
sensor.mask[16, 10, 8] = True

kspaceFirstOrder3D(
    kgrid=kgrid, medium=medium, source=source, sensor=sensor,
    simulation_options=SimulationOptions(pml_inside=False, pml_size=6, save_to_disk=True),
    execution_options=SimulationExecutionOptions(is_gpu_simulation=False),
)  # TypeError: 'int' object is not subscriptable
