"""Acoustic module (ports of ``acoustic-module``): grid, transducer integration, signals,
homogeneous propagation, RF synthesis, k-Wave sources and sensors."""

from .grid import PML, Grid, define_grid, largest_prime_factor, optimize_grid_size
from .homogeneous import (
    average_medium,
    calc_scatter_attenuated,
    db2neper,
    gridded_distance,
    power_law_kramers_kronig,
    propagate_homogeneous,
    run_simulation_homogeneous,
)
from .rf import compute_RF, compute_RF_data, rf_signal_length
from .sensor import (
    Sensor,
    define_sensor_MB,
    define_sensor_MB_all,
    define_sensor_transducer,
    extract_sensor_data,
    update_sensor,
    update_sensor_fast,
)
from .signals import evaluate_delta_function, resample_signal, sinc_interpolation, sinc_matrix
from .source import (
    SEQ_PULSES,
    Source,
    define_source_transducer,
    preprocess_transmit,
    pulsing_sequence,
)
from .transducer_integration import (
    compute_travel_times,
    get_transducer_integration_delays,
    get_transducer_integration_points,
)

__all__ = [
    "PML",
    "SEQ_PULSES",
    "Grid",
    "Sensor",
    "Source",
    "average_medium",
    "calc_scatter_attenuated",
    "compute_RF",
    "compute_RF_data",
    "compute_travel_times",
    "db2neper",
    "define_grid",
    "define_sensor_MB",
    "define_sensor_MB_all",
    "define_sensor_transducer",
    "define_source_transducer",
    "evaluate_delta_function",
    "extract_sensor_data",
    "get_transducer_integration_delays",
    "get_transducer_integration_points",
    "gridded_distance",
    "largest_prime_factor",
    "optimize_grid_size",
    "power_law_kramers_kronig",
    "preprocess_transmit",
    "propagate_homogeneous",
    "pulsing_sequence",
    "resample_signal",
    "rf_signal_length",
    "run_simulation_homogeneous",
    "sinc_interpolation",
    "sinc_matrix",
    "update_sensor",
    "update_sensor_fast",
]
