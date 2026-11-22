"""Transducer catalogue and calibrated model files (ports of PROTEUS ``GUIfunctions``).

* ``get_transducer_properties``
* ``load_transducer_model`` (PROTEUS ``Transducer`` and Verasonics ``Trans`` formats)

Functions return the updated struct; the input is not modified (MATLAB value semantics).
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from pyproteus.config.matfile import loadmat, whosmat
from pyproteus.config.models import Transducer

__all__ = ["get_transducer_properties", "load_transducer_model"]

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
    if T.Type is None:
        raise ValueError("Transducer has no field Type")  # MATLAB: reference error
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


def _times_double(a: np.ndarray, x: float) -> np.ndarray:
    """MATLAB ``a*x`` for a double scalar ``x``: the result keeps ``a``'s class.

    Integer classes round to nearest (halves away from zero) and saturate
    (Q-32); single stays single.
    """
    if a.dtype.kind in "iu":
        from pyproteus.numerics import matlab_round

        info = np.iinfo(a.dtype)
        y = matlab_round(a.astype(np.float64) * x)
        return np.clip(y, info.min, info.max).astype(a.dtype)
    if a.dtype == np.bool_:
        return a.astype(np.float64) * x
    return (a * np.asarray(x, dtype=a.dtype)).astype(a.dtype)


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
    if T.Type is None:
        raise ValueError("Transducer has no field Type")  # MATLAB: reference error
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
        ir_trans = _times_double(np.asarray(trans["IR1wy"]), fs)
        ir_rec = ir_trans.copy()
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
