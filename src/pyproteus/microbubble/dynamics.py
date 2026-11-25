"""Microbubble dynamics: ports of ``simple_RP.m`` and ``calcBubbleResponse.m``.

The Marmottant-shell Rayleigh–Plesset equation (nondimensional radius
excursion ``x = R/R0 - 1`` and its derivative ``xdot`` in units of a time
scale ``T = sqrt(rho R0^2 / P0)``) is integrated with Dormand–Prince 5(4) and
PROTEUS' options (RelTol = AbsTol = 1e-6, first step 1e-12 ``T``).

Step control (D-22):

- ``step_control="independent"`` (default): every bubble has its own steps,
  error test and time scale ``T`` (its own radius), as MATLAB PROTEUS gives
  for a bubble integrated alone, so a bubble's response does not depend on the
  other bubbles of a batch (fixes Q-50/Q-121). The batch is still advanced as
  arrays (:func:`pyproteus.microbubble.solver.dopri5_batch`), one RHS
  evaluation per stage for all active bubbles.
- ``step_control="shared"``: PROTEUS' stacked solve (one state vector
  ``[x1, xdot1, x2, ...]`` through :func:`pyproteus.numerics.ode45`, shared
  steps, ``T`` = median over the batch), for MATLAB-comparison diagnostics.

PROTEUS' wall-clock timer (60 s × number of bubbles) is replaced by
deterministic failure detection (D-22): non-finite state, radius ≤ 0, radius
above ``max_radius_ratio·R0``, step size underflow and a step budget; failed
bubbles are reported in :class:`BubbleIntegrationError` while the others
finish. An optional wall-clock limit is available (off by default).

Other design differences (D-15, ``docs/PARITY.md``): the RHS is vectorised over
bubbles; the driving pressures' pchip interpolants are one coefficient matrix
(:class:`PchipBatch`); outputs are ``(n_bubbles, n_t)`` arrays.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
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
from .solver import BatchFailure, dopri5_batch

__all__ = [
    "BubbleIntegrationError",
    "BubbleResponse",
    "PchipBatch",
    "Pulse",
    "calc_bubble_response",
    "simple_rp",
]

#: ODE options of ``calcBubbleResponse.m`` (nondimensional time and state).
REL_TOL = 1e-6
ABS_TOL = 1e-6
INITIAL_STEP = 1e-12
#: Default failure thresholds (D-22).
MAX_RADIUS_RATIO = 100.0
MAX_STEPS = 1_000_000


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

    ``T`` holds the time scale(s) used: one per bubble (independent steps) or
    the batch median (shared steps). After a failure a bubble's entries from
    the failure on are NaN.
    """

    R: np.ndarray
    Rdot: np.ndarray
    t: np.ndarray
    T: np.ndarray
    failures: list[BatchFailure] = field(default_factory=list)


class BubbleIntegrationError(RuntimeError):
    """Some bubbles could not be integrated to the end (D-22).

    ``failures`` lists ``(index, t [s], reason)`` per failed bubble (indices into
    the batch passed in); ``response`` holds the results of all bubbles, NaN
    after each failure (``None`` when raised by ``compute_bubble_mass_source``,
    which attaches ``mass_source`` instead).
    """

    def __init__(
        self, failures: list[BatchFailure], response: BubbleResponse | None, settings: str
    ):
        self.failures = failures
        self.response = response
        lines = [f"  bubble {f.index}: t = {f.t:.4e} s: {f.reason}" for f in failures]
        super().__init__(
            f"{len(failures)} bubble(s) could not be integrated ({settings}):\n" + "\n".join(lines)
        )


class PchipBatch:
    """MATLAB ``griddedInterpolant(t, p(i,:), 'pchip')`` for every row ``i`` at once.

    Shape-preserving cubic Hermite interpolation (node slopes from
    :func:`pyproteus.numerics.interpolation.pchip_slopes`; Fritsch–Butland
    interior, Moler's end rule), with pchip extrapolation outside ``t`` as
    ``griddedInterpolant``. Coefficients of all rows are computed once.
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
        """All rows at one time."""
        k = int(np.searchsorted(self.t, tq, side="right")) - 1
        k = min(max(k, 0), self.t.size - 2)
        s = tq - self.t[k]
        return self.c0[k] + s * (self.c1[k] + s * (self.c2[k] + s * self.c3[k]))

    def at(self, tq: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """Row ``rows[j]`` at time ``tq[j]``."""
        k = np.clip(np.searchsorted(self.t, tq, side="right") - 1, 0, self.t.size - 2)
        s = tq - self.t[k]
        return self.c0[k, rows] + s * (
            self.c1[k, rows] + s * (self.c2[k, rows] + s * self.c3[k, rows])
        )


def _rp(x, xdot, R0, T, sig_0, Ks, kappa, nu, sig, P, liquid: Liquid) -> np.ndarray:
    """``xdotdot`` of ``simple_RP.m`` (Marmottant et al., JASA 118, 2005, Eq. 3)."""
    rhol, P0, c = liquid.rho, liquid.P0, liquid.c
    return (
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
    """Port of ``simple_RP.m`` for a stacked batch (shared time scale ``T``).

    ``vec = [x1, xdot1, x2, xdot2, ...]``; returns its derivative in ``tau = t/T``.
    """
    v = vec.reshape(-1, 2)
    x, xdot = v[:, 0], v[:, 1]
    sig = calc_surface_tension(R0 * (1 + x), shell)
    P = P_acc(tau * T)
    out = np.empty_like(v)
    out[:, 0] = xdot
    out[:, 1] = _rp(
        x, xdot, R0, T, shell.sig_0, shell.Ks, eqparam.kappa, eqparam.nu, sig, P, liquid
    )
    return out.reshape(-1)


def _broadcast_shell(shell: Shell, n: int) -> Shell:
    out = Shell(model=shell.model, sig_0=0.0, coeff=shell.coeff, sig=shell.sig)
    for f in ("sig_0", "sig_l", "Ks", "chi", "Rb", "A_m1", "A_m2", "A_N"):
        v = getattr(shell, f)
        setattr(out, f, None if v is None else np.broadcast_to(v, (n,)).astype(np.float64))
    return out


def calc_bubble_response(
    liquid: Liquid,
    gas: GasProps,
    shell: Shell | Sequence[Shell],
    R0: ArrayLike,
    pulse: Pulse,
    *,
    step_control: Literal["independent", "shared"] = "independent",
    wall_clock_limit: float | None = None,
    max_steps: int = MAX_STEPS,
    max_radius_ratio: float = MAX_RADIUS_RATIO,
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
    step_control
        ``"independent"`` (default, D-22): per-bubble steps and time scale;
        ``"shared"``: PROTEUS' stacked solve (results depend on the batch, Q-50).
    wall_clock_limit
        Seconds for the whole batch; ``None`` (default) = no limit.
    max_steps
        Independent: attempted steps per bubble (accepted and rejected).
        Shared: accepted steps of the stacked solve (counted by its output function).
    max_radius_ratio
        A radius above ``max_radius_ratio * R0`` counts as a blow-up.

    Raises
    ------
    BubbleIntegrationError
        If any bubble fails; the others are integrated to the end and their
        results are in the exception's ``response``.
    """
    R0 = np.asarray(R0, dtype=np.float64).reshape(-1)
    n_mb = R0.size
    shell = stack_shells(shell)
    if shell.size not in (1, n_mb):
        raise ValueError(f"shell describes {shell.size} bubbles, R0 has {n_mb}")
    p = np.atleast_2d(np.asarray(pulse.p, dtype=np.float64))
    if p.shape[0] != n_mb:
        raise ValueError(f"pulse.p has {p.shape[0]} rows, expected {n_mb}")
    if step_control not in ("independent", "shared"):
        raise ValueError(f"unknown step_control {step_control!r}")

    eqparam = get_eq_param(liquid, gas, shell, R0, pulse.w)
    tq = np.asarray(pulse.tq, dtype=np.float64).reshape(-1)
    P_acc = PchipBatch(pulse.t, p)
    T_each = np.sqrt(liquid.rho * R0**2 / liquid.P0)
    settings = (
        f"step_control={step_control!r}, RelTol={REL_TOL:g}, AbsTol={ABS_TOL:g}, "
        f"max_steps={max_steps}, max_radius_ratio={max_radius_ratio:g}, "
        f"wall_clock_limit={wall_clock_limit}, R0 = {np.array2string(R0, precision=4)} m"
    )
    if step_control == "shared":
        T = float(np.median(T_each))
        x, xdot, failures, t_out = _shared(
            liquid, shell, eqparam, R0, P_acc, T, tq, wall_clock_limit, max_steps,
            max_radius_ratio,
        )  # fmt: skip
        Ts = np.full(n_mb, T)
    else:
        x, xdot, failures = _independent(
            liquid, shell, eqparam, R0, P_acc, T_each, tq, wall_clock_limit, max_steps,
            max_radius_ratio,
        )  # fmt: skip
        t_out, Ts = tq, T_each
    response = BubbleResponse(
        R=R0[:, None] * (1 + x),
        Rdot=R0[:, None] * xdot / Ts[:, None],
        t=t_out,
        T=Ts if step_control == "independent" else Ts[:1],
        failures=failures,
    )
    if failures:
        raise BubbleIntegrationError(failures, response, settings)
    return response, eqparam


_RADIUS_LE_ZERO, _BLOW_UP = 1, 2


def _independent(liquid, shell, eqparam, R0, P_acc, T, tq, wall, max_steps, ratio):
    n = R0.size
    sh = _broadcast_shell(shell, n)
    sig_0, Ks = sh.sig_0, sh.Ks
    kappa, nu = eqparam.kappa, eqparam.nu
    all_idx = np.arange(n)

    def fun(tau, Y, idx):
        full = idx.size == n
        s = sh if full else sh.take(idx)
        x, xdot = Y[:, 0], Y[:, 1]
        Ti = T[idx]
        sig = calc_surface_tension(R0[idx] * (1 + x), s)
        P = P_acc.at(tau * Ti, all_idx if full else idx)
        out = np.empty_like(Y)
        out[:, 0] = xdot
        out[:, 1] = _rp(x, xdot, R0[idx], Ti, sig_0[idx], Ks[idx], kappa[idx], nu[idx], sig, P,
                        liquid)  # fmt: skip
        return out

    def check(Y, idx):
        codes = np.zeros(idx.size, np.int64)
        codes[Y[:, 0] + 1 > ratio] = _BLOW_UP
        codes[Y[:, 0] <= -1] = _RADIUS_LE_ZERO
        return codes

    res = dopri5_batch(
        fun, tq[None, :] / T[:, None], np.zeros((n, 2)), rtol=REL_TOL, atol=ABS_TOL,
        initial_step=INITIAL_STEP, max_steps=max_steps, check=check,
        check_reasons={_RADIUS_LE_ZERO: "radius <= 0",
                       _BLOW_UP: f"radius above {ratio:g} R0 (blow-up)"},
        wall_clock_limit=wall,
    )  # fmt: skip
    for f in res.failures:
        f.t = f.t * float(T[f.index])  # nondimensional -> seconds
    return res.y[:, :, 0], res.y[:, :, 1], res.failures


def _shared(liquid, shell, eqparam, R0, P_acc, T, tq, wall, max_steps, ratio):
    n = R0.size
    counter = {"steps": 0}
    deadline = None if wall is None else time.monotonic() + wall
    reason: list[str] = []

    def out_fcn(t, y, flag):
        if flag != "":
            return False
        counter["steps"] += 1
        Y = np.asarray(y).reshape(n, 2, -1)
        if not np.all(np.isfinite(Y)):
            reason.append("non-finite state")
        elif np.any(Y[:, 0] <= -1) or np.any(Y[:, 0] + 1 > ratio):
            reason.append("radius <= 0 or blow-up")
        elif counter["steps"] > max_steps:
            reason.append(f"more than {max_steps} steps")
        elif deadline is not None and time.monotonic() >= deadline:
            reason.append(f"wall-clock limit of {wall:g} s reached")
        return bool(reason)

    options = odeset(
        BDF="on", AbsTol=ABS_TOL, RelTol=REL_TOL, InitialStep=INITIAL_STEP, OutputFcn=out_fcn
    )
    rhs = lambda tau, vec: simple_rp(tau, vec, liquid, shell, eqparam, R0, P_acc, T)  # noqa: E731
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sol = ode45(rhs, tq / T, np.zeros(2 * n), options)
    m = sol.t.size
    X = np.full((tq.size, n, 2), np.nan)
    X[:m] = sol.y.reshape(m, n, 2)
    failures: list[BatchFailure] = []
    if sol.stopped or reason:  # a flag on the final step does not set ``stopped``
        why = reason[0] if reason else "step size underflow"
        t_fail = float(sol.t[-1] * T)
        failures = [BatchFailure(i, t_fail, f"{why} (shared steps: whole batch stopped)")
                    for i in range(n)]  # fmt: skip
    return X[:, :, 0].T, X[:, :, 1].T, failures, tq
