"""Field access on MATLAB-style structs (settings models, dicts or plain objects)."""

from __future__ import annotations

from typing import Any

import numpy as np


def get(struct: Any, name: str, default: Any = None) -> Any:
    """``S.(name)`` if ``isfield(S, name)``, else ``default``."""
    v = struct.get(name, default) if isinstance(struct, dict) else getattr(struct, name, default)
    return default if v is None else v


def isfield(struct: Any, name: str) -> bool:
    """MATLAB ``isfield(S, name)``."""
    if hasattr(struct, "isfield"):
        return bool(struct.isfield(name))
    return get(struct, name) is not None


def config_is(struct: Any, name: str) -> bool:
    """MATLAB ``isfield(S,'Configuration') && strcmp(S.Configuration, name)``."""
    value = get(struct, "Configuration")
    return isinstance(value, str) and value == name


def copy(struct: Any) -> Any:
    """Value-semantics copy of a struct (MATLAB structs are copied on assignment)."""
    if hasattr(struct, "copy_struct"):
        return struct.copy_struct()
    if isinstance(struct, dict):
        return dict(struct)
    raise TypeError(f"cannot copy {type(struct)!r}")


def set_field(struct: Any, name: str, value: Any) -> None:
    if isinstance(struct, dict):
        struct[name] = value
    else:
        setattr(struct, name, value)


def scalar(v: Any) -> float:
    """A real scalar (any 1-element array) as a Python float."""
    a = np.asarray(v, dtype=float)
    if a.size != 1:
        raise ValueError(f"expected a scalar, got shape {a.shape}")
    return float(a.reshape(-1)[0])
