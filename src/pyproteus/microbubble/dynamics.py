"""Microbubble dynamics: ports of ``simple_RP.m``, ``odeOutputFcn.m`` and ``calcBubbleResponse.m``.

The Marmottant-shell Rayleigh–Plesset equation is integrated with the
clean-room :func:`pyproteus.numerics.ode45` for a batch of bubbles stacked in
one state vector ``[x1, xdot1, x2, xdot2, ...]`` (nondimensional radius
excursion ``x = R/R0 - 1`` and its derivative in units of the batch time scale
``T``), as PROTEUS does. All bubbles of a batch therefore share the adaptive
steps and the time scale ``T`` (median over the batch), so results depend on
the batch at the level of the solver tolerance (Q-50, Q-121).

Design differences from PROTEUS (D-15, ``docs/PARITY.md``), results unchanged
within the ODE class:

- the right-hand side is vectorised over bubbles: surface tensions and driving
  pressures of the whole batch in one array operation. The pchip interpolants
  of the driving pressures are built once (:class:`PchipBatch`, coefficients of
  all bubbles in one matrix) instead of one ``griddedInterpolant`` per bubble
  evaluated in a loop;
- PROTEUS' wall-clock timer (60 s × number of bubbles, ``calcBubbleResponse.m``)
  is a keyword (``timeout``); ``timeout=None`` turns it off (Q-122);
- outputs are arrays ``(n_bubbles, n_t)`` instead of a struct array.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

from pyproteus.numerics import ode45, odeset
from pyproteus.numerics.interpolation import pchip_slopes

from .model import (
    EqParam,
    GasProps,
    Liquid,
    Shell,
    calc_surface_tension,
    get_eq_param,
    stack_shells,
)

__all__ = [
    "PROTEUS_TIMEOUT",
    "BubbleResponse",
    "PchipBatch",
    "Pulse",
    "calc_bubble_response",
    "simple_rp",
]

#: ``timeout`` value meaning PROTEUS' timer: 60 s per bubble of the batch.
PROTEUS_TIMEOUT = "proteus"
#: ODE options of ``calcBubbleResponse.m`` (nondimensional time and state).
REL_TOL = 1e-6
ABS_TOL = 1e-6
INITIAL_STEP = 1e-12


@dataclass
class Pulse:
    """PROTEUS ``pulse`` struct for :func:`calc_bubble_response`.

    ``p`` has one driving pressure (Pa) per bubble as rows, sampled at ``t``
    (s); ``tq`` are the output times; ``w`` the angular centre frequency
    (rad/s, used by the thermal model).
    """

    t: np.ndarray
    p: np.ndarray
    tq: np.ndarray
    w: float
    f: float | None = None
    fs: float | None = None
    dispProgress: bool = False


@dataclass
class BubbleResponse:
    """Radius ``R`` (m) and wall velocity ``Rdot`` (m/s), shape ``(n_bubbles, n_t)``, at ``t`` (s).

    ``stopped`` is true when the timeout ended the integration early; ``t`` is
    then shorter than ``pulse.tq``.
    """

    R: np.ndarray
    Rdot: np.ndarray
    t: np.ndarray
    stopped: bool = False


class PchipBatch:
    """MATLAB ``griddedInterpolant(t, p(i,:), 'pchip')`` for every row ``i`` at once.

    Shape-preserving cubic Hermite interpolation (node slopes from
    :func:`pyproteus.numerics.interpolation.pchip_slopes`; Fritsch–Butland
    interior, Moler's end rule), with pchip extrapolation outside ``t`` as
    ``griddedInterpolant``. The local power-form coefficients of all rows are
    computed once; a call at a scalar time costs one interval search.
    """

    def __init__(self, t: ArrayLike, p: ArrayLike) -> None:
        t = np.asarray(t, dtype=np.float64).reshape(-1)
        p = np.atleast_2d(np.asarray(p, dtype=np.float64))
        if p.shape[1] != t.size:
            raise ValueError(f"p has {p.shape[1]} samples per row, t has {t.size}")
        if t.size < 2 or np.any(np.diff(t) <= 0):
            raise ValueError("t must be strictly increasing with at least two points")
        v = p.T  # (n_t, n_rows)
        d = pchip_slopes(t, v)
        h = np.diff(t)[:, None]
        delta = np.diff(v, axis=0) / h
        self.t = t
        # rows: intervals, columns: bubbles; c0 + s (c1 + s (c2 + s c3)), s = tq - t_k
        self.c0 = v[:-1]
        self.c1 = d[:-1]
        self.c2 = (3 * delta - 2 * d[:-1] - d[1:]) / h
        self.c3 = (d[:-1] - 2 * delta + d[1:]) / (h * h)

    def __call__(self, tq: float) -> np.ndarray:
        k = int(np.searchsorted(self.t, tq, side="right")) - 1
        k = min(max(k, 0), self.t.size - 2)
        s = tq - self.t[k]
        return self.c0[k] + s * (self.c1[k] + s * (self.c2[k] + s * self.c3[k]))


def simple_rp(
    tau: float,
    vec: np.ndarray,
    liquid: Liquid,
    shell: Shell,
    eqparam: EqParam,
    R0: np.ndarray,
    P_acc: PchipBatch,
    T: float,
) -> np.ndarray:
    """Port of ``simple_RP.m``: nondimensional Rayleigh–Plesset equation (Marmottant et al.,
    JASA 118, 2005, Eq. 3) for a stacked batch, vectorised over bubbles.

    ``vec = [x1, xdot1, x2, xdot2, ...]``; returns its time derivative.
    """
    rhol, P0, c = liquid.rho, liquid.P0, liquid.c
    sig_0, Ks = shell.sig_0, shell.Ks
    kappa, nu = eqparam.kappa, eqparam.nu
    t = tau * T
    v = vec.reshape(-1, 2)
    x = v[:, 0]
    xdot = v[:, 1]
    R = R0 * (1 + x)
    sig = calc_surface_tension(R, shell)
    P = P_acc(t)
    xdotdot = (
        1
        / (1 + x)
        * (
            -3 / 2 * xdot**2
            + P0
            / rhol
            / R0**2
            * T**2
            * (
                (1 + 2 * sig_0 / (R0 * P0))
                * (1 + x) ** (-3 * kappa)
                * (1 - 3 * kappa / c / T * R0 * xdot)
                - 1
                - 2 * sig / (R0 * P0) * (1 + x) ** (-1)
                - 4 * nu / P0 / T * xdot / (1 + x)
                - 4 * Ks / P0 / R0 / T * xdot / (1 + x) ** 2
                - P / P0
            )
        )
    )
    out = np.empty_like(v)
    out[:, 0] = xdot
    out[:, 1] = xdotdot
    return out.reshape(-1)


class _OutputFcn:
    """``odeOutputFcn.m``: optional progress display and the wall-clock timeout."""

    def __init__(self, T: float, disp_progress: bool, timeout: float | None) -> None:
        self.T = T
        self.disp_progress = disp_progress
        self.timeout = timeout
        self.tplo = 0.0  # time to display progress (s)
        self.dtplo = 5e-7
        self.deadline = np.inf
        self.timeout_reached = False

    def start(self) -> None:
        if self.timeout is not None:
            self.deadline = time.monotonic() + self.timeout

    def __call__(self, tau: np.ndarray, y: np.ndarray, flag: str) -> bool:
        if self.disp_progress and flag == "" and np.size(tau):
            t = float(np.asarray(tau).reshape(-1)[-1]) * self.T
            if t >= self.tplo:
                print(f"t = {self.tplo:.2e} s")
                self.tplo += self.dtplo
        if not self.timeout_reached and time.monotonic() >= self.deadline:
            self.timeout_reached = True
            print("timeout reached")
        return self.timeout_reached


def calc_bubble_response(
    liquid: Liquid,
    gas: GasProps,
    shell: Shell | Sequence[Shell],
    R0: ArrayLike,
    pulse: Pulse,
    *,
    timeout: float | Literal["proteus"] | None = PROTEUS_TIMEOUT,
) -> tuple[BubbleResponse, EqParam]:
    """Port of ``calcBubbleResponse.m``: radial response of a batch of microbubbles.

    Parameters
    ----------
    liquid, gas
        Material properties (:class:`Liquid`, :class:`GasProps`).
    shell
        Shell properties of the batch (:class:`Shell` with per-bubble arrays,
        or one per bubble, stacked with :func:`stack_shells`).
    R0
        Equilibrium radii (m), one per bubble.
    pulse
        Driving pressures (rows of ``pulse.p``), sample times, output times and
        angular frequency.
    timeout
        Wall-clock limit of the integration in seconds. ``"proteus"`` (default)
        is PROTEUS' ``60 s × n_bubbles``; ``None`` disables it (Q-122). When
        reached, integration stops with a warning and the response covers only
        the output times reached so far.

    Returns
    -------
    (BubbleResponse, EqParam)
    """
    R0 = np.asarray(R0, dtype=np.float64).reshape(-1)
    n_mb = R0.size
    shell = stack_shells(shell)
    if shell.size not in (1, n_mb):
        raise ValueError(f"shell describes {shell.size} bubbles, R0 has {n_mb}")
    p = np.atleast_2d(np.asarray(pulse.p, dtype=np.float64))
    if p.shape[0] != n_mb:
        raise ValueError(f"pulse.p has {p.shape[0]} rows, expected {n_mb}")
    if timeout == PROTEUS_TIMEOUT:
        timeout = 60.0 * n_mb
    elif timeout is not None:
        timeout = float(timeout)

    eqparam = get_eq_param(liquid, gas, shell, R0, pulse.w)

    x0v = np.zeros(2 * n_mb)
    T = float(np.median(np.sqrt(liquid.rho * R0**2 / liquid.P0)))  # characteristic time scale
    tau = np.asarray(pulse.tq, dtype=np.float64).reshape(-1) / T
    P_acc = PchipBatch(pulse.t, p)

    out_fcn = _OutputFcn(T, bool(pulse.dispProgress), timeout)
    options = odeset(
        BDF="on", AbsTol=ABS_TOL, RelTol=REL_TOL, InitialStep=INITIAL_STEP, OutputFcn=out_fcn
    )

    def rhs(tau_: float, vec: np.ndarray) -> np.ndarray:
        return simple_rp(tau_, vec, liquid, shell, eqparam, R0, P_acc, T)

    out_fcn.start()
    sol = ode45(rhs, tau, x0v, options)
    if out_fcn.timeout_reached and sol.stopped:
        warnings.warn(
            f"bubble integration stopped by the wall-clock timeout ({timeout:g} s) at "
            f"t = {sol.t[-1] * T:.3e} s; the response is truncated (Q-122)",
            RuntimeWarning,
            stacklevel=2,
        )
    X = sol.y.reshape(-1, n_mb, 2)  # (n_t, bubble, [x, xdot])
    response = BubbleResponse(
        R=R0[:, None] * (1 + X[:, :, 0].T),
        Rdot=R0[:, None] * X[:, :, 1].T / T,
        t=sol.t * T,
        stopped=bool(sol.stopped),
    )
    return response, eqparam
