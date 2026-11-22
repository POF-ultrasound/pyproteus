"""Microbubble gas properties and size distribution (ports of PROTEUS ``GUIfunctions``).

* ``get_gas_properties``
* ``compute_polydisperse_distribution`` (with a black-box matched MATLAB ``linspace``)
"""

from __future__ import annotations

import numpy as np

from pyproteus.config.models import Gas

__all__ = ["compute_polydisperse_distribution", "get_gas_properties", "linspace"]

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
    """MATLAB ``linspace(d1, d2, n)`` as a 1×n row (black-box matched, see PARITY.md)."""
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
