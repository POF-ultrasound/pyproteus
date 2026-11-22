"""Constants shipped with MATLAB PROTEUS, converted once by ``tools/convert_data.py``.

:func:`load` is the counterpart of MATLAB ``load('<name>.mat')`` for these
files: it returns the variables as a MATLAB value tree (see
:mod:`pyproteus.config.matfile`), equal bit for bit to what MATLAB loads.
``MANIFEST.json`` records each file's MATLAB source, the oracle commit and the
source SHA-256.

Available names (:func:`names`): ``tissue_properties.mat``,
``fit_SigmaR_evaluated.mat`` (+ ``_30PA``, ``_50PA``), ``fit_SigmaR_04-08-2017.mat``,
``TransmitTransferFunctionFit.mat``, ``ReceiveTransferFunctionFit.mat``,
``TwoWayTransferFunctionFit.mat``, ``<geometry>/GeometryProperties.mat``
(``renal_tree``, ``mouse_brain``) and the transducer model ``C1-6D.mat``.
"""

from __future__ import annotations

import io
import json
from functools import cache
from importlib import resources
from typing import Any

import numpy as np

__all__ = ["geometry_folders", "load", "manifest", "names", "transducer_models"]


def _files() -> Any:
    return resources.files(__name__)


@cache
def manifest() -> dict[str, Any]:
    """Contents of ``MANIFEST.json``."""
    return json.loads((_files() / "MANIFEST.json").read_text(encoding="utf-8"))


@cache
def _by_matlab_name() -> dict[str, str]:
    return {info["matlab_name"]: f for f, info in manifest()["files"].items()}


def names() -> list[str]:
    """MATLAB file names accepted by :func:`load`."""
    return sorted(_by_matlab_name())


def geometry_folders() -> list[str]:
    """Geometry folders with shipped ``GeometryProperties`` (``Geometry.Folder`` values)."""
    return sorted(
        n.split("/")[0] for n in _by_matlab_name() if n.endswith("/GeometryProperties.mat")
    )


def transducer_models() -> list[str]:
    """Transducer types with a shipped model file (``Transducer-models/<Type>.mat``)."""
    return sorted(
        info["matlab_name"][: -len(".mat")]
        for f, info in manifest()["files"].items()
        if f.startswith("Transducer-models/")
    )


def load(name: str, variables: list[str] | None = None) -> dict[str, Any]:
    """MATLAB ``load(name)`` for a shipped constant file; a fresh copy every call.

    ``name`` is the file name PROTEUS loads (``'tissue_properties.mat'``,
    ``'renal_tree/GeometryProperties.mat'``, ...). ``variables`` restricts the
    result like ``load(name, 'var1', ...)``.
    """
    from pyproteus.config.json_io import tree_from_jsonable

    try:
        rel = _by_matlab_name()[name]
    except KeyError:
        raise FileNotFoundError(f"no shipped data file {name!r}; known: {names()}") from None
    res = _files().joinpath(*rel.split("/"))
    if rel.endswith(".npz"):
        with np.load(io.BytesIO(res.read_bytes()), allow_pickle=False) as npz:
            out = {k: np.array(npz[k]) for k in npz.files}
    else:
        out = tree_from_jsonable(json.loads(res.read_text(encoding="utf-8")))
    if variables is not None:
        out = {k: v for k, v in out.items() if k in variables}
    return out
