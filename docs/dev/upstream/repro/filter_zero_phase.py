"""k-wave-python 0.6.2: filter_time_series(..., zerophase=True) raises IndexError for any input."""

import numpy as np
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.utils.filters import filter_time_series

kgrid = kWaveGrid((64,), (1e-4,))
kgrid.makeTime(1500.0)
medium = kWaveMedium(sound_speed=1500.0)
t = np.arange(int(kgrid.Nt)) * float(kgrid.dt)
signal = np.sin(2 * np.pi * 1e6 * t)[None, :]

filter_time_series(kgrid, medium, signal, zerophase=False)  # works
filter_time_series(kgrid, medium, signal, zerophase=True)  # IndexError in apply_filter
