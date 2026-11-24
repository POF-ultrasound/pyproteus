"""Microbubble gas properties and size distribution (ports of PROTEUS ``GUIfunctions``).

* ``get_gas_properties``
* ``compute_polydisperse_distribution`` (with a ``linspace`` helper)
* ``compute_gaussian_distribution``
"""

from __future__ import annotations

from typing import Any

import numpy as np

from pyproteus.config.models import Gas

__all__ = [
    "compute_gaussian_distribution",
    "compute_polydisperse_distribution",
    "gaussian_distribution",
    "get_gas_properties",
    "linspace",
]

# ------------------------------------------------------------------------- gas

_GASES: dict[str, dict[str, float]] = {
    "Sulfur hexafluoride": {
        "ThermalConductivity": 0.0139,
        "Density": 5.80,
        "MolarMass": 0.146,
        "SpecificHeat": 687.0,
        "HeatCapacityRatio": 1.09,
    },
    "Perfluoropropane": {
        "ThermalConductivity": 0.0132,
        "Density": 7.53,
        "MolarMass": 0.188,
        "SpecificHeat": 814.0,
        "HeatCapacityRatio": 1.06,
    },
    "Perfluorobutane": {
        "ThermalConductivity": 0.0136,
        "Density": 9.64,
        "MolarMass": 0.238,
        "SpecificHeat": 883.0,
        "HeatCapacityRatio": 1.05,
    },
}


def get_gas_properties(gas: Gas) -> Gas:
    """Port of ``get_gas_properties.m`` (NIST values at 1 atm, 37 °C)."""
    if gas.Type not in _GASES:
        raise ValueError("Gas not found")
    G = gas.copy_struct()
    for k, v in _GASES[gas.Type].items():
        setattr(G, k, v)
    return G


# ----------------------------------------------------------------- microbubble


def linspace(d1: float, d2: float, n: int) -> np.ndarray:
    """``linspace(d1, d2, n)`` as a 1×n row: ``d1 + k*(d2-d1)/(n-1)``, last element ``d2``."""
    n = int(np.floor(n))
    if n < 1:
        return np.zeros((1, 0))
    if n == 1:
        return np.array([[float(d2)]])
    n1 = n - 1
    k = np.arange(n, dtype=np.float64)
    y = d1 + k * (d2 - d1) / n1
    y[-1] = d2
    return y.reshape(1, -1)


def compute_polydisperse_distribution() -> tuple[np.ndarray, np.ndarray]:
    """Port of ``compute_polydisperse_distribution.m`` (BR-14 fit, 1001 radii in m).

    Returns ``(R, P)`` as 1×1001 rows.
    """
    N = 1e3
    Rmin = 0.5
    Rmax = 6
    R = linspace(Rmin, Rmax, int(N) + 1)
    a = 2.19
    P = R**2 * np.exp(-a * R)
    P = P / np.sum(P)
    R = R * 1e-6
    return R, P


def gaussian_distribution(mean_radius: float, pdi: float) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian size distribution on 1000 radii from 0.5 to 6 um (1×1000 rows, in m).

    Standard deviation ``mean_radius*pdi/100``; probabilities normalised to sum 1.
    ``pdi = 0`` gives NaN probabilities, as in MATLAB (Q-84).
    """
    N = 1e3  # number of points (the polydisperse fit uses N + 1, Q-84)
    Rmin = 0.5e-6
    Rmax = 6e-6
    R = linspace(Rmin, Rmax, int(N))
    sR = mean_radius * pdi / 100  # standard deviation (m)
    with np.errstate(divide="ignore", invalid="ignore"):
        P = np.exp(-((R - mean_radius) ** 2) / (2 * sR**2))
        P = P / np.sum(P)
    return R, P


def compute_gaussian_distribution(microbubble: Any) -> tuple[np.ndarray, np.ndarray]:
    """Port of ``compute_gaussian_distribution.m``: ``(R, P)`` from
    ``Microbubble.Distribution.MeanRadius`` [m] and ``PDI`` [%] (see
    :func:`gaussian_distribution`)."""
    dist = getattr(microbubble, "Distribution", None)
    if dist is None:
        raise ValueError("Microbubble has no field Distribution")
    values = []
    for name in ("MeanRadius", "PDI"):
        v = getattr(dist, name, None)
        if v is None:
            raise ValueError(f"Microbubble.Distribution has no field {name}")
        values.append(float(np.asarray(v, dtype=np.float64).reshape(())))
    return gaussian_distribution(*values)
