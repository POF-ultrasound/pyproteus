"""Catalogue and lookup functions from PROTEUS ``GUIfunctions`` used by the defaults.

Ports (PROTEUS commit 8d37938):

* ``get_transducer_properties`` (transducer catalogue)
* ``load_transducer_model`` (calibrated model files, PROTEUS and Verasonics formats)
* ``get_gas_properties``
* ``assign_medium_properties`` (tissue table ``tissue_properties.mat``)
* ``assign_liquid_properties``
* ``compute_sound_speed_extrema``
* ``compute_polydisperse_distribution``

Like the MATLAB originals, each function returns the updated struct; the
input model is not modified (MATLAB value semantics).
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from .matfile import loadmat, whosmat
from .models import Gas, Medium, Transducer, Vessel

__all__ = [
    "assign_liquid_properties",
    "assign_medium_properties",
    "compute_polydisperse_distribution",
    "compute_sound_speed_extrema",
    "get_gas_properties",
    "get_transducer_properties",
    "linspace",
    "load_transducer_model",
]

# ------------------------------------------------------------------- transducer

_CATALOGUE: dict[str, dict[str, Any]] = {
    "P4-1": {
        "Configuration": "Linear",
        "NumberOfElements": 96.0,
        "Pitch": 0.295e-3,
        "ElementWidth": 0.245e-3,
        "ElementHeight": 16e-3,
        "ElevationFocus": 110e-3,
        "BandwidthLow": 1.5e6,
        "BandwidthHigh": 3.5e6,
        "CenterFrequency": 2.5e6,
        "Radius": 0.0,
    },
    "C1-6D": {
        "Configuration": "Curved",
        "NumberOfElements": 192.0,
        "Pitch": 0.350e-3,
        "ElementWidth": 0.340e-3,
        "ElementHeight": 11.4e-3,
        "ElevationFocus": 72.7e-3,
        "BandwidthLow": 1.5e6,
        "BandwidthHigh": 5.5e6,
        "CenterFrequency": 3.4e6,
        "Radius": 56.8e-3,
    },
    "L22-14v": {
        "Configuration": "Linear",
        "NumberOfElements": 128.0,
        "Pitch": 100e-6,
        "ElementWidth": 80e-6,
        "ElementHeight": 1.6e-3,
        "ElevationFocus": 8e-3,
        "BandwidthLow": 14e6,
        "BandwidthHigh": 22e6,
        "CenterFrequency": 18e6,
        "Radius": 0.0,
    },
    "9L-D": {
        "Configuration": "Linear",
        "NumberOfElements": 192.0,
        "Pitch": 0.23e-3,
        "ElementWidth": 0.21e-3,
        "ElementHeight": 6e-3,
        "ElevationFocus": 28e-3,
        "BandwidthLow": 3.3e6,
        "BandwidthHigh": 7.3e6,
        "CenterFrequency": 5.3e6,
        "Radius": 0.0,
    },
    "RC6gV": {
        "Configuration": "RCA",
        "NumberOfElements": 256.0,
        "Pitch": 0.2e-3,
        "ElementWidth": 0.175e-3,
        "ElementHeight": 0.025575,
        "ElevationFocus": float("inf"),
        "BandwidthLow": 3e6,
        "BandwidthHigh": 9e6,
        "CenterFrequency": 6e6,
        "Radius": 0.0,
    },
}


def get_transducer_properties(
    transducer: Transducer, models_folder: str | PathLike[str] | None = None
) -> Transducer:
    """Port of ``get_transducer_properties.m``: catalogue values for ``Transducer.Type``.

    Unknown types are left unchanged (MATLAB ``switch`` without ``otherwise``).
    The calibrated model file, if any, is applied last (:func:`load_transducer_model`).
    ``models_folder`` is passed on to :func:`load_transducer_model` (default:
    the models shipped in :mod:`pyproteus.data`, = PROTEUS ``Transducer-models``).
    """
    T = transducer.copy_struct()
    if T.Type in _CATALOGUE:
        T.calibrated = 0.0
        T.ImpulseResponseType = "Estimate"
        for k, v in _CATALOGUE[T.Type].items():
            setattr(T, k, v)
    elif T.Type == "Custom linear":
        T.Configuration = "Linear"
        T.Radius = 0.0
        T.calibrated = 0.0
        if T.ImpulseResponseType is not None and T.ImpulseResponseType == "Calibrated":
            T.ImpulseResponseType = "Load file"
    return load_transducer_model(T, models_folder)


_MODEL_PROPERTIES = (
    "NumberOfElements",
    "Pitch",
    "ElementWidth",
    "ElementHeight",
    "ElevationFocus",
    "Radius",
    "Configuration",
    "BandwidthLow",
    "BandwidthHigh",
    "CenterFrequency",
)


def _read_model_file(transducer_type: str, folder: str | PathLike[str] | None) -> Any:
    """Variables of ``<folder>/<Type>.mat`` (or the shipped model), or None if absent.

    Returns ``(names in the file, loader)`` where ``loader(name)`` reads one variable.
    """
    if folder is None:
        from pyproteus import data

        if transducer_type not in data.transducer_models():
            return None
        variables = data.load(f"{transducer_type}.mat")
        return list(variables), lambda name: variables[name]
    file = Path(folder) / f"{transducer_type}.mat"
    if not file.is_file():
        return None
    return whosmat(file), lambda name: loadmat(file, [name])[name]


def _as_row(ir: Any) -> np.ndarray:
    """MATLAB ``reshape(double(IR), 1, [])`` (column-major)."""
    arr = np.asarray(ir)
    if arr.dtype == object or arr.dtype.kind not in "biuf":
        raise TypeError(f"impulse response must be a real numeric array, got {arr.dtype}")
    return np.asarray(arr, dtype=np.float64).reshape(1, -1, order="F")


def _isfield(s: dict[str, Any], *names: str) -> bool:
    return isinstance(s, dict) and all(n in s for n in names)


def load_transducer_model(
    transducer: Transducer, folder: str | PathLike[str] | None = None
) -> Transducer:
    """Port of ``load_transducer_model.m``: apply ``<folder>/<Transducer.Type>.mat``.

    ``folder=None`` uses the models shipped in :mod:`pyproteus.data` (the
    PROTEUS ``Transducer-models`` folder). Without a model file the transducer
    is returned unchanged. Accepts a ``Transducer`` variable (PROTEUS format:
    ``TransmitImpulseResponse``/``ReceiveImpulseResponse``/``SamplingRate`` or
    ``IR_trans``/``IR_rec``/``Fs_trans``/``Fs_rec``) or a ``Trans`` variable
    (Verasonics: ``IR1wy`` at 250 MHz, multiplied by the sampling rate).
    Raises ``ValueError`` where MATLAB raises an error.
    """
    T = transducer.copy_struct()
    found = _read_model_file(T.Type, folder)
    if found is None:
        return T
    names, read = found
    where = f"{T.Type}.mat" if folder is None else str(Path(folder) / f"{T.Type}.mat")

    if "Transducer" in names:
        model = read("Transducer")
        if not isinstance(model, dict):
            raise ValueError(f"{where}: variable Transducer is not a 1×1 struct")
        for name in _MODEL_PROPERTIES:
            if name in model:
                setattr(T, name, model[name])
        if _isfield(model, "TransmitImpulseResponse", "ReceiveImpulseResponse", "SamplingRate"):
            ir_trans = model["TransmitImpulseResponse"]
            ir_rec = model["ReceiveImpulseResponse"]
            fs = model["SamplingRate"]
        elif _isfield(model, "IR_trans", "IR_rec", "Fs_trans", "Fs_rec"):
            # MATLAB ``if Fs_trans ~= Fs_rec``: true when non-empty and all elements differ.
            differ = np.asarray(model["Fs_trans"]) != np.asarray(model["Fs_rec"])
            if differ.size > 0 and bool(np.all(differ)):
                raise ValueError(
                    f"load_transducer_model:samplingRate: {where}: the transmit and "
                    "receive impulse responses must have the same sampling rate."
                )
            ir_trans = model["IR_trans"]
            ir_rec = model["IR_rec"]
            fs = model["Fs_trans"]
        else:
            raise ValueError(
                f"load_transducer_model:noImpulseResponse: {where} does not contain "
                "impulse responses."
            )
    elif "Trans" in names:
        trans = read("Trans")
        if not _isfield(trans, "IR1wy"):
            raise ValueError(
                f"load_transducer_model:noImpulseResponse: {where} does not contain Trans.IR1wy."
            )
        fs = 250e6
        ir = np.asarray(trans["IR1wy"])
        ir_trans = ir * fs
        ir_rec = ir * fs
    else:
        raise ValueError(
            f"load_transducer_model:unknownFormat: {where} contains neither a Transducer "
            "nor a Trans variable."
        )

    T.TransmitImpulseResponse = _as_row(ir_trans)
    T.ReceiveImpulseResponse = _as_row(ir_rec)
    T.SamplingRate = fs
    T.ImpulseResponseType = "Calibrated"
    T.calibrated = 1.0
    return T


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
