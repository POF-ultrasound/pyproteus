"""Port of ``run_simulation.m``: one k-Wave run with PROTEUS' options.

``run_param`` is the MATLAB struct (dict, settings model or attribute object)
with MATLAB field names: ``solver`` (``'kspaceFirstOrder3D'``, ``'...3DC'``,
``'...3DG'``, or the ``SimulationParameters.Solver`` short forms ``'3D'``,
``'3DC'``, ``'3DG'``), ``PML`` (``Grid.PML``: ``X_SIZE``/``Y_SIZE``/``Z_SIZE``,
``Alpha``), ``DATA_CAST``, and optional ``DATA_PATH``, ``DEVICE_NUM``,
``BINARY_PATH``, ``record_movie``.

Solver -> :func:`pyproteus.backends.kwave.kspace_first_order` backend:

==========================  ===========================================
``kspaceFirstOrder3D``      ``backend="python"`` (NumPy, in-process)
``kspaceFirstOrder3DC``     ``backend="cpp"``, OpenMP binary (parity target)
``kspaceFirstOrder3DG``     ``backend="cpp"``, CUDA binary
==========================  ===========================================

Divergences (docs/PARITY.md, BUGS.md B-170, B-171): ``'3D'`` runs
k-wave-python's solver instead of MATLAB-native ``kspaceFirstOrder3D``
(within the k-Wave class of 3DC, which is what PROTEUS' 3DC/3DG runs give);
``record_movie`` (only set for ``'3D'``) is ignored with a warning: no movie.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np

from pyproteus.backends.kwave import KGrid, kspace_first_order

from ._struct import get as _get

__all__ = ["SOLVERS", "run_simulation", "solver_backend"]

#: MATLAB solver name -> (backend, device)
SOLVERS = {
    "kspaceFirstOrder3D": ("python", "cpu"),
    "kspaceFirstOrder3DC": ("cpp", "cpu"),
    "kspaceFirstOrder3DG": ("cpp", "gpu"),
}


def solver_backend(solver: str) -> tuple[str, str]:
    """``run_param.solver`` (or ``Solver`` short form) -> ``(backend, device)``."""
    name = str(solver)
    if not name.startswith("kspaceFirstOrder"):
        name = "kspaceFirstOrder" + name
    try:
        return SOLVERS[name]
    except KeyError:
        raise ValueError(
            f"unknown solver {solver!r}; expected one of {tuple(SOLVERS)} or '3D'/'3DC'/'3DG'"
        ) from None


def _pml(pml: Any) -> tuple[tuple[int, int, int], float]:
    if pml is None:
        raise ValueError("run_param.PML is required (run_param.PML = Grid.PML)")
    size = tuple(int(_get(pml, k)) for k in ("X_SIZE", "Y_SIZE", "Z_SIZE"))
    return size, float(_get(pml, "Alpha"))  # type: ignore[return-value]


def _sensor_mask(sensor: Any) -> np.ndarray:
    mask = sensor.get("mask") if isinstance(sensor, Mapping) else getattr(sensor, "mask", None)
    if mask is None:
        raise ValueError("sensor.mask is required")
    return np.asarray(mask)


def run_simulation(
    run_param: Any,
    kgrid: KGrid,
    medium: Any,
    source: Any,
    sensor: Any,
    *,
    binary_env: Mapping[str, str] | None = None,
    num_threads: int | None = None,
) -> dict[str, np.ndarray]:
    """Port of ``run_simulation.m``.

    Options as MATLAB: ``PMLInside=false``, ``PMLAlpha=PML.Alpha``,
    ``PMLSize=[PML.X_SIZE, PML.Y_SIZE, PML.Z_SIZE]``, ``DataCast=DATA_CAST``,
    ``Smooth=false``, plus ``DataPath``/``DeviceNum``/``BinaryPath`` when the
    fields are present. ``binary_env`` and ``num_threads`` are forwarded to
    :func:`~pyproteus.backends.kwave.kspace_first_order` (not in MATLAB:
    environment of the binary, e.g. the macOS wrapper's ``KWAVE_LIBS``, and
    ``NumThreads``).

    Returns ``sensor_data`` as a dict: ``{"p": (n_sensor, Nt) float32, ...}``,
    rows in ``find(sensor.mask)`` order. An all-zero sensor mask returns
    ``{"p": zeros((0, Nt))}`` with a warning, without running k-Wave (as MATLAB).
    """
    backend, device = solver_backend(_get(run_param, "solver"))
    pml_size, pml_alpha = _pml(_get(run_param, "PML"))
    if bool(_get(run_param, "record_movie", False)):
        warnings.warn(
            "run_param.record_movie: movie recording is not supported, running without it",
            stacklevel=2,
        )
    if not np.any(_sensor_mask(sensor)):
        warnings.warn(
            "No nonzero elements in sensor mask, returning empty sensor data.", stacklevel=2
        )
        return {"p": np.zeros((0, kgrid.Nt))}
    data_path = _get(run_param, "DATA_PATH")
    device_num = _get(run_param, "DEVICE_NUM")
    binary = _get(run_param, "BINARY_PATH") if backend == "cpp" else None
    return kspace_first_order(
        kgrid,
        medium,
        source,
        sensor,
        pml_size=pml_size,
        pml_alpha=pml_alpha,
        pml_inside=False,
        smooth=False,
        data_cast=str(_get(run_param, "DATA_CAST", "single")),
        backend=backend,
        device=device,
        device_num=None if device_num is None else int(device_num),
        num_threads=num_threads,
        binary=os.fspath(binary) if binary else None,
        binary_env=binary_env,
        data_path=data_path or None,
    )
