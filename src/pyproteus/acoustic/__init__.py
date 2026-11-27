"""Acoustic module (ports of ``acoustic-module``): grid, transducer integration, signals,
homogeneous propagation, RF synthesis, k-Wave sources and sensors, run setup, the
hybrid and full simulators and the ``main_RF`` / ``main_pressure_field`` entry points."""

from .full import SelfSenseCache, compute_self_sense_pressure, update_source
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
from .main import PressureMaps, main_pressure_field, main_RF, prepare_simulation
from .mass_source import compute_bubble_mass_source
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
from .setup import (
    RunConfig,
    RunSetup,
    check_ground_truth_data,
    input_handling,
    load_microbubbles,
    setup_run,
    sim_setup,
    sim_startup,
)
from .signals import evaluate_delta_function, resample_signal, sinc_interpolation, sinc_matrix
from .simulation import run_simulation
from .simulators import SELF_SENSE_MODES, full_simulator, hybrid_simulator
from .source import (
    PULSE_SEQUENCES,
    SEQ_PULSES,
    Source,
    define_source_transducer,
    preprocess_transmit,
    pulse_sequence,
)
from .transducer_integration import (
    compute_travel_times,
    get_transducer_integration_delays,
    get_transducer_integration_points,
)

__all__ = [
    "PML",
    "PULSE_SEQUENCES",
    "SELF_SENSE_MODES",
    "SEQ_PULSES",
    "Grid",
    "PressureMaps",
    "RunConfig",
    "RunSetup",
    "SelfSenseCache",
    "Sensor",
    "Source",
    "average_medium",
    "calc_scatter_attenuated",
    "check_ground_truth_data",
    "compute_RF",
    "compute_RF_data",
    "compute_bubble_mass_source",
    "compute_self_sense_pressure",
    "compute_travel_times",
    "db2neper",
    "define_grid",
    "define_sensor_MB",
    "define_sensor_MB_all",
    "define_sensor_transducer",
    "define_source_transducer",
    "evaluate_delta_function",
    "extract_sensor_data",
    "full_simulator",
    "get_transducer_integration_delays",
    "get_transducer_integration_points",
    "gridded_distance",
    "hybrid_simulator",
    "input_handling",
    "largest_prime_factor",
    "load_microbubbles",
    "main_RF",
    "main_pressure_field",
    "optimize_grid_size",
    "power_law_kramers_kronig",
    "prepare_simulation",
    "preprocess_transmit",
    "propagate_homogeneous",
    "pulse_sequence",
    "resample_signal",
    "rf_signal_length",
    "run_simulation",
    "run_simulation_homogeneous",
    "setup_run",
    "sim_setup",
    "sim_startup",
    "sinc_interpolation",
    "sinc_matrix",
    "update_sensor",
    "update_sensor_fast",
    "update_source",
]
