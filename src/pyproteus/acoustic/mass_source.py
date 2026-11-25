"""Port of ``acoustic-module/compute_bubble_mass_source.m``: bubble responses as mass sources.

Steps, as PROTEUS:

1. resample the sensed pressures (one row per bubble, k-Wave time step
   ``kgrid.dt``) to the microbubble sampling rate with ``sinc_interpolation``;
2. sort the bubbles by radius (stable sort, ties keep their input order) and
   integrate them in consecutive batches of ``batch_size`` with
   :func:`pyproteus.microbubble.calc_bubble_response` (shared steps and time
   scale inside a batch: results depend on the batching, Q-50/Q-121);
3. mass source ``4 pi rho R^2 Rdot`` per bubble, zero-phase filtered with
   k-Wave's ``filterTimeSeries`` (through k-wave-python, B-60) and resampled
   back to the k-Wave time axis.

Differences from PROTEUS (``docs/PARITY.md``): ``parfor`` is not used (the
batches are independent, so serial results are the same); the timer is a
keyword (``timeout``, Q-122) and a timeout raises an error (MATLAB fails with a
size mismatch when it assigns the truncated response); float32 sensed pressures
are integrated in double precision and the mass source is returned as float32
(MATLAB stores the batch results in single and mixes single into the RHS).
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

from pyproteus.acoustic.signals import sinc_interpolation
from pyproteus.microbubble.dynamics import PROTEUS_TIMEOUT, Pulse, calc_bubble_response
from pyproteus.microbubble.model import KS_C1, KS_C2, GasProps, Liquid, Shell
from pyproteus.numerics import argsort, gridded_interpolant, interp1

__all__ = [
    "compute_bubble_mass_source",
    "get_batch_indices",
    "get_microbubble_material_properties",
    "get_microbubble_shell_properties",
]

#: PROTEUS ``sim_setup.m``: ``run_param.MicrobubblesBatchSize = 15``.
DEFAULT_BATCH_SIZE = 15


def _get(s: Any, name: str) -> Any:
    v = getattr(s, name, None)
    if v is None and isinstance(s, dict):
        v = s.get(name)
    if v is None:
        raise ValueError(f"missing field {name}")
    return v


def _f(v: Any) -> float:
    return float(np.asarray(v, dtype=np.float64).reshape(()))


def get_batch_indices(batch_index: int, n_mb: int, batch_size: int, radii: ArrayLike) -> np.ndarray:
    """0-based bubble indices of batch ``batch_index`` (0-based): bubbles sorted by radius
    with MATLAB's stable ``sort``, cut into consecutive groups of ``batch_size``."""
    lin = batch_index * batch_size + np.arange(batch_size)
    lin = lin[lin < n_mb]
    return argsort(np.asarray(radii, dtype=np.float64).reshape(-1))[lin]


def get_microbubble_material_properties(medium: Any, microbubble: Any) -> tuple[Liquid, GasProps]:
    """``get_microbubble_material_properties``: liquid from ``Medium.Vessel``, gas from
    ``Microbubble.Gas``, thermal model from ``Microbubble.ThermalModel``."""
    L = _get(medium, "Vessel")
    G = _get(microbubble, "Gas")
    liquid = Liquid(
        k=_f(_get(L, "ThermalConductivity")),
        rho=_f(_get(L, "Density")),
        cp=_f(_get(L, "SpecificHeat")),
        nu=_f(_get(L, "DynamicViscosity")),
        c=_f(_get(L, "SpeedOfSound")),
        T0=_f(_get(L, "Temperature")),
        P0=_f(_get(L, "Pressure")),
        ThermalModel=str(_get(microbubble, "ThermalModel")),
    )
    gas = GasProps(
        k=_f(_get(G, "ThermalConductivity")),
        rho=_f(_get(G, "Density")),
        Mg=_f(_get(G, "MolarMass")),
        gam=_f(_get(G, "HeatCapacityRatio")),
        cp=_f(_get(G, "SpecificHeat")),
    )
    return liquid, gas


def get_microbubble_shell_properties(R0: ArrayLike, medium: Any, microbubble: Any) -> Shell:
    """``get_microbubble_shell_properties`` for a batch of radii ``R0`` (one :class:`Shell`).

    ``'Segers'`` and ``'Custom'`` use the lookup table
    ``Microbubble.Shell.NormalizedArea``/``SurfaceTension`` (model
    ``'SegersTable'``). MATLAB warns once per bubble outside 0.5–6 µm; this
    port warns once per batch.
    """
    R0 = np.asarray(R0, dtype=np.float64).reshape(-1)
    S = _get(microbubble, "Shell")
    sig_0 = _f(_get(S, "InitialSurfaceTension"))
    model_in = str(_get(S, "Model"))
    model = "SegersTable" if model_in in ("Segers", "Custom") else model_in
    if np.any((R0 < 0.5e-6) | (R0 > 6e-6)):
        warnings.warn("Microbubble dynamics uncertain for given microbubble radius.", stacklevel=2)
    if model == "Marmottant":
        chi = _f(_get(S, "Elasticity"))
        shell = Shell(
            model=model,
            sig_0=sig_0,
            chi=chi,
            Rb=R0 / np.sqrt(1 + sig_0 / chi),
            sig_l=_f(_get(_get(medium, "Vessel"), "SurfaceTension")),
        )
    elif model == "SegersTable":
        sig = np.asarray(_get(S, "SurfaceTension"), dtype=np.float64).reshape(-1)
        A_m_list = np.asarray(_get(S, "NormalizedArea"), dtype=np.float64).reshape(-1)
        A_0 = 4 * np.pi * R0**2
        A_m0 = float(interp1(sig, A_m_list, sig_0))
        shell = Shell(
            model=model,
            sig_0=sig_0,
            A_m1=A_m_list.min(),
            A_m2=A_m_list.max(),
            A_N=A_0 / A_m0,
            sig=gridded_interpolant(A_m_list, sig),
            sig_l=sig.max(),
        )
    else:
        raise ValueError("Unknown shell model")
    advanced = bool(np.asarray(getattr(microbubble, "Advanced", False) or False).reshape(()))
    shell.Ks = (
        np.full(R0.shape, _f(_get(S, "Viscosity"))) if advanced else KS_C1 * np.exp(KS_C2 * R0)
    )
    return shell


def compute_bubble_mass_source(
    sensed_p: ArrayLike,
    radii: ArrayLike,
    kgrid: Any,
    medium: Any,
    microbubble: Any,
    transmit: Any,
    *,
    batch_size: int | None = None,
    timeout: float | Literal["proteus"] | None = PROTEUS_TIMEOUT,
) -> np.ndarray:
    """Port of ``compute_bubble_mass_source.m``.

    Parameters
    ----------
    sensed_p
        Pressure at the bubbles, ``(n_bubbles, n_t)`` on the k-Wave time axis
        ``(0:n_t-1) * kgrid.dt``.
    radii
        Bubble radii (m), one per bubble.
    kgrid
        Object with ``dt`` (s) and ``k_max`` (rad/m).
    medium, microbubble, transmit
        PROTEUS settings (``Medium.Vessel``, ``Medium.SpeedOfSoundMinimum``;
        ``Microbubble.SamplingRate``, ``Gas``, ``Shell``, ``ThermalModel``,
        ``Advanced``; ``Transmit.CenterFrequency``).
    batch_size
        Bubbles per ODE batch; default ``Microbubble.BatchSize`` if present, else
        15 (PROTEUS ``sim_setup.m``).
    timeout
        Per-batch wall-clock limit (see :func:`calc_bubble_response`); ``None`` = off.

    Returns
    -------
    Mass source ``(n_bubbles, n_t)`` (kg/s), same dtype as ``sensed_p`` (float32 or float64).
    """
    from pyproteus.backends.kwave import filter_time_series

    sp = np.asarray(sensed_p)
    if sp.ndim != 2:
        raise ValueError("sensed_p must be n_bubbles x n_t")
    out_dtype = np.float32 if sp.dtype == np.float32 else np.float64
    n_mb, n = sp.shape
    radii = np.asarray(radii, dtype=np.float64).reshape(-1)
    if radii.size != n_mb:
        raise ValueError(f"{radii.size} radii for {n_mb} bubbles")
    if batch_size is None:
        bs = getattr(microbubble, "BatchSize", None)
        batch_size = DEFAULT_BATCH_SIZE if bs is None else int(_f(bs))
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    n_batch = -(-n_mb // batch_size)

    fs_mb = _f(_get(microbubble, "SamplingRate"))
    dt = _f(kgrid.dt)
    t_kwave = np.arange(n, dtype=np.float64) * dt
    m = int(np.floor(t_kwave[-1] * fs_mb)) + 1
    t_mb = np.arange(m, dtype=np.float64) / fs_mb

    # (n_bubbles, m); float32 in -> float32, as MATLAB
    p_mb = np.atleast_2d(sinc_interpolation(t_kwave, sp.T, t_mb))

    f0 = _f(_get(transmit, "CenterFrequency"))
    w = f0 * 2 * np.pi
    liquid, gas = get_microbubble_material_properties(medium, microbubble)

    mass = np.zeros((n_mb, m), dtype=out_dtype)
    for k in range(n_batch):
        idx = get_batch_indices(k, n_mb, batch_size, radii)
        shell = get_microbubble_shell_properties(radii[idx], medium, microbubble)
        pulse = Pulse(t=t_mb, p=p_mb[idx].astype(np.float64), tq=t_mb, w=w, f=f0, fs=fs_mb)
        resp, _ = calc_bubble_response(liquid, gas, shell, radii[idx], pulse, timeout=timeout)
        if resp.stopped:
            raise RuntimeError(
                f"bubble batch {k + 1}/{n_batch} stopped early (timeout); "
                "MATLAB PROTEUS fails here with a size mismatch (Q-122)"
            )
        mass[idx] = 4 * np.pi * liquid.rho * resp.R**2 * resp.Rdot

    tw = 0.1 / (fs_mb * dt)
    c_min = _get(medium, "SpeedOfSoundMinimum")
    k_max = _f(kgrid.k_max)
    filtered = np.empty((n_mb, m), dtype=np.float64)
    for i in range(n_mb):
        filtered[i] = filter_time_series(
            1 / fs_mb,
            k_max,
            c_min,
            mass[i].astype(np.float64),
            zero_phase=True,
            transition_width=tw,
            ppw=2,
        )
    mass = filtered.astype(out_dtype)
    return np.atleast_2d(sinc_interpolation(t_mb, mass.T, t_kwave)).astype(out_dtype)
