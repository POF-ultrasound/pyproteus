"""Acoustic module (ports of ``acoustic-module``): grid, transducer integration, signals."""

from .grid import PML, Grid, define_grid, largest_prime_factor, optimize_grid_size
from .signals import evaluate_delta_function, resample_signal, sinc_interpolation, sinc_matrix
from .transducer_integration import (
    compute_travel_times,
    get_transducer_integration_delays,
    get_transducer_integration_points,
)

__all__ = [
    "PML",
    "Grid",
    "compute_travel_times",
    "define_grid",
    "evaluate_delta_function",
    "get_transducer_integration_delays",
    "get_transducer_integration_points",
    "largest_prime_factor",
    "optimize_grid_size",
    "resample_signal",
    "sinc_interpolation",
    "sinc_matrix",
]
