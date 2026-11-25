"""Port of ``acoustic-module/compute_bubble_mass_source.m``: bubble responses as mass sources.

Steps, as PROTEUS:

1. resample the sensed pressures (one row per bubble, k-Wave time step
   ``kgrid.dt``) to the microbubble sampling rate with ``sinc_interpolation``;
2. sort the bubbles by radius (stable sort, ties keep their input order) and
   integrate them in consecutive batches of ``batch_size`` with
   :func:`pyproteus.microbubble.calc_bubble_response`;
3. mass source ``4 pi rho R^2 Rdot`` per bubble, zero-phase filtered with
   k-Wave's ``filterTimeSeries`` (through k-wave-python, B-60) and resampled
   back to the k-Wave time axis.

Differences from PROTEUS (``docs/PARITY.md``, D-22): per-bubble step control
by default, so results do not depend on the batching (``step_control="shared"``
gives PROTEUS' shared steps, Q-50/Q-121); no wall-clock timer: failed bubbles
are reported in one :class:`BubbleIntegrationError` after all batches ran;
``parfor`` is not used (batches are independent); float32 sensed pressures
are integrated in double precision and the mass source is returned as float32
(MATLAB stores the batch results in single and mixes single into the RHS).
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

from pyproteus.acoustic.signals import sinc_interpolation
from pyproteus.microbubble.dynamics import (
    BubbleIntegrationError,
    Pulse,
    calc_bubble_response,
)
from pyproteus.microbubble.model import KS_C1, KS_C2, GasProps, Liquid, Shell
from pyproteus.microbubble.solver import BatchFailure
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
    step_control: Literal["independent", "shared"] = "independent",
    wall_clock_limit: float | None = None,
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
    step_control
        ``"independent"`` (default, D-22): results do not depend on the batching;
        ``"shared"``: PROTEUS' shared steps per batch (Q-50).
    wall_clock_limit
        Per-batch wall-clock limit in seconds; ``None`` (default) = off.

    Raises
    ------
    BubbleIntegrationError
        If bubbles fail (all batches are still computed); ``failures`` use
        indices into ``radii`` and ``mass_source`` holds the result with NaN rows
        for failed bubbles.

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
    order = argsort(radii)  # stable: ties keep their input order, as MATLAB sort
    failures: list[BatchFailure] = []
    for k in range(n_batch):
        idx = order[k * batch_size : (k + 1) * batch_size]
        shell = get_microbubble_shell_properties(radii[idx], medium, microbubble)
        pulse = Pulse(t=t_mb, p=p_mb[idx].astype(np.float64), tq=t_mb, w=w, f=f0, fs=fs_mb)
        try:
            resp, _ = calc_bubble_response(
                liquid, gas, shell, radii[idx], pulse, step_control=step_control,
                wall_clock_limit=wall_clock_limit,
            )  # fmt: skip
        except BubbleIntegrationError as exc:
            resp = exc.response
            failures += [BatchFailure(int(idx[f.index]), f.t, f.reason) for f in exc.failures]
        mass[idx] = 4 * np.pi * liquid.rho * resp.R**2 * resp.Rdot

    tw = 0.1 / (fs_mb * dt)
    c_min = _get(medium, "SpeedOfSoundMinimum")
    k_max = _f(kgrid.k_max)
    filtered = np.full((n_mb, m), np.nan)
    failed = sorted({f.index for f in failures})
    for i in sorted(set(range(n_mb)) - set(failed)):
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
    out = np.atleast_2d(sinc_interpolation(t_mb, mass.T, t_kwave)).astype(out_dtype)
    if failures:
        failures.sort(key=lambda f: f.index)
        exc = BubbleIntegrationError(
            failures, None, f"compute_bubble_mass_source, step_control={step_control!r}, "
            f"batch_size={batch_size}, radii of failed bubbles {radii[failed]} m",
        )  # fmt: skip
        exc.mass_source = out
        raise exc
    return out
