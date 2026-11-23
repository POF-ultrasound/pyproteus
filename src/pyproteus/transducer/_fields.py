"""Field access on MATLAB-style structs (settings models or mappings)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def field(s: Any, name: str) -> Any:
    """``s.(name)``, or None if the field is absent."""
    if isinstance(s, Mapping):
        return s.get(name)
    return getattr(s, name, None)


def required(s: Any, name: str, owner: str) -> Any:
    value = field(s, name)
    if value is None:
        raise ValueError(f"{owner} has no field {name}")  # MATLAB: reference error
    return value


def scalar(s: Any, name: str, owner: str = "Transducer") -> float:
    """A required real scalar field as a Python float."""
    arr = np.asarray(required(s, name, owner), dtype=np.float64)
    if arr.size != 1:
        raise ValueError(f"{owner}.{name} must be a scalar, got size {arr.shape}")
    return float(arr.reshape(()))


def configuration_is(s: Any, name: str) -> bool:
    """MATLAB ``isfield(s,'Configuration') && strcmp(s.Configuration, name)``.

    Exact, case-sensitive match (Q-80). PROTEUS always stores a char row here;
    a non-string value never matches.
    """
    value = field(s, "Configuration")
    return isinstance(value, str) and value == name


def element_count(s: Any) -> int:
    """``Transducer.NumberOfElements`` as a non-negative int (ValueError otherwise)."""
    n = scalar(s, "NumberOfElements")
    if not (np.isfinite(n) and n >= 0 and n == np.floor(n)):
        raise ValueError(f"Transducer.NumberOfElements must be a non-negative integer, got {n!r}")
    return int(n)
