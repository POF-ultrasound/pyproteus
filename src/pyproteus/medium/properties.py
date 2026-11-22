"""Medium and liquid properties (ports of PROTEUS ``GUIfunctions``).

* ``assign_medium_properties`` (tissue table ``tissue_properties.mat``)
* ``assign_liquid_properties``
* ``compute_sound_speed_extrema``

Functions return the updated struct; the input is not modified.
"""

from __future__ import annotations

import numpy as np

from pyproteus.config.models import Medium, Vessel

__all__ = ["assign_liquid_properties", "assign_medium_properties", "compute_sound_speed_extrema"]

# ---------------------------------------------------------------------- medium


def _strfind_nonempty(text: str, pattern: str) -> bool:
    """MATLAB ``~isempty(strfind(text, pattern))``: an empty pattern never matches."""
    return len(pattern) > 0 and pattern in text


def assign_medium_properties(medium: Medium | Vessel) -> Medium | Vessel:
    """Port of ``assign_medium_properties.m``.

    ``'General tissue'`` has fixed values; ``'Custom'`` is left unchanged; any
    other name takes the **first** row of ``tissue_properties.mat`` whose name
    *contains* ``Tissue`` (MATLAB ``strfind``; quirk Q-30: ``'Muscle'`` selects
    ``'Muscle (along the fibers)'``). Unknown names raise ``ValueError('Tissue
    not found')``.
    """
    M = medium.copy_struct()
    if M.Tissue == "General tissue":
        M.SpeedOfSound = 1540.0
        M.Density = 1000.0
        M.BonA = 6.0
        M.AttenuationA = 0.75
        M.AttenuationB = 1.5
    elif M.Tissue != "Custom":
        from pyproteus import data

        tissues = data.load("tissue_properties.mat")
        names = tissues["tissue_names"]
        props = tissues["tissue_properties"]
        flat = names.reshape(-1, order="F")
        matches = [k for k, name in enumerate(flat) if _strfind_nonempty(name, M.Tissue)]
        if not matches:
            raise ValueError("Tissue not found")
        i = matches[0]
        M.Density = props[i, 0] * 1000
        M.SpeedOfSound = props[i, 1]
        M.AttenuationA = props[i, 3]
        M.AttenuationB = props[i, 4]
        M.BonA = props[i, 5]
    return M


_LIQUIDS: dict[str, dict[str, float]] = {
    "Blood": {
        "ThermalConductivity": 0.52,
        "SpecificHeat": 3770.0,
        "DynamicViscosity": 4.5e-3,
        "SurfaceTension": 0.053,
        "Temperature": 310.0,
        "Pressure": 1.013e5,
    },
    "Water": {
        "ThermalConductivity": 0.6,
        "SpecificHeat": 4184.0,
        "DynamicViscosity": 1e-3,
        "SurfaceTension": 0.072,
        "Temperature": 293.0,
        "Pressure": 1.013e5,
    },
}


def assign_liquid_properties(liquid: Vessel, liquid_type: str) -> Vessel:
    """Port of ``assign_liquid_properties.m``; other types leave ``liquid`` unchanged."""
    L = liquid.copy_struct()
    for k, v in _LIQUIDS.get(liquid_type, {}).items():
        setattr(L, k, v)
    return L


def compute_sound_speed_extrema(medium: Medium) -> Medium:
    """Port of ``compute_sound_speed_extrema.m`` (MATLAB ``min``/``max`` omit NaN)."""
    M = medium.copy_struct()
    cutoff = M.InhomogeneityCutoff
    inhom = M.Inhomogeneity
    cmin = M.SpeedOfSound * (1 - inhom * cutoff)
    cmax = M.SpeedOfSound * (1 + inhom * cutoff)
    M.SpeedOfSoundMinimum = float(np.fmin(cmin, M.Vessel.SpeedOfSound))
    M.SpeedOfSoundMaximum = float(np.fmax(cmax, M.Vessel.SpeedOfSound))
    return M
