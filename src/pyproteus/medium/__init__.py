"""Medium: tissue and liquid properties (M1 adds define_medium and friends)."""

from .properties import (
    assign_liquid_properties,
    assign_medium_properties,
    compute_sound_speed_extrema,
)

__all__ = ["assign_liquid_properties", "assign_medium_properties", "compute_sound_speed_extrema"]
