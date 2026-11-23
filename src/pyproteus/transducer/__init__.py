"""Transducer: catalogue, model files and element geometry (M1 adds integration points)."""

from .delays import compute_delays, transmit_delays
from .domain import (
    GeometryBehindTransducerWarning,
    compute_end_depth,
    compute_simulation_domain,
    compute_transducer_projection,
    compute_transducer_vertices,
)
from .geometry import ElementGeometry, get_element_geometry, is_curved
from .properties import get_transducer_properties, load_transducer_model

__all__ = [
    "ElementGeometry",
    "GeometryBehindTransducerWarning",
    "compute_delays",
    "compute_end_depth",
    "compute_simulation_domain",
    "compute_transducer_projection",
    "compute_transducer_vertices",
    "get_element_geometry",
    "get_transducer_properties",
    "is_curved",
    "load_transducer_model",
    "transmit_delays",
]
