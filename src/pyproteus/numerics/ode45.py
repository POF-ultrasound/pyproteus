"""``ode45``: explicit Runge–Kutta 5(4) (Dormand–Prince) with MATLAB's interface.

Written from published sources only:

- J. R. Dormand, P. J. Prince, "A family of embedded Runge-Kutta formulae",
  J. Comput. Appl. Math. 6 (1980) 19-26: the 7-stage FSAL pair RK5(4)7M,
  advanced with the 5th-order solution (local extrapolation).
- E. Hairer, S. P. Nørsett, G. Wanner, *Solving Ordinary Differential
  Equations I* (2nd ed., Springer 1993), §II.4 (step-size control, starting
  step size) and §II.6 (dense output of DOPRI5, coefficients ``_D``).
- L. F. Shampine, M. W. Reichelt, "The MATLAB ODE Suite", SIAM J. Sci. Comput.
  18 (1997) 1-22: ode45 is this pair with a continuous extension used for
  ``Refine`` output; the step is reduced after a failure and not increased
  right after one.
- MATLAB's public documentation of ``ode45``/``odeset`` for the interface:
  ``tspan`` with 2 entries (output at every step, ``Refine`` points per step,
  default 4) versus more entries (output only there); the error test
  ``|e(i)| <= max(RelTol*|y(i)|, AbsTol(i))`` (``NormControl``: on norms);
  ``MaxStep`` default ``|tspan(end) - tspan(1)| / 10``; ``InitialStep``;
  ``OutputFcn(t, y, flag)`` with flags ``'init'``, ``''``, ``'done'``.

Choices the sources leave open (standard choices, documented here, not tuned
to reproduce any other solver step for step):

- error measure ``err = max_i |h (f @ E)_i| / max(|y_i|, |ynew_i|, AbsTol_i/RelTol)``;
  the step is accepted when ``err <= RelTol`` (the documented test, using the
  larger of the old and new ``|y|``);
- after a success the step grows by ``min(5, 0.8 (RelTol/err)^(1/5))``, but not
  right after a failure; after a first failure it shrinks by
  ``max(0.1, 0.8 (RelTol/err)^(1/5))``, then halves on repeated failures;
  smallest step ``16 eps(t)``;
- starting step from Hairer–Nørsett–Wanner §II.4 (one extra RHS call), capped
  by ``MaxStep``; ``InitialStep`` overrides it;
- the step is stretched to land on ``tfinal`` when ``1.1 h`` would reach it;
- ``Refine`` points at ``t + h k/Refine`` (k = 1 .. Refine-1); ``OutputFcn`` is
  called after each accepted step that produced output.

The state is one vector: PROTEUS stacks a batch of bubbles into it, so all
bubbles of a batch share the adaptive steps (``docs/QUIRKS.md``, Q-50).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["ODEResult", "ODEStats", "ode45", "odeset"]

# Dormand & Prince (1980), RK5(4)7M. _A[i, j]: coefficient of slope f_j in stage i.
_C = np.array([0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0])
_A = np.zeros((7, 7))
_A[1, :1] = [1 / 5]
_A[2, :2] = [3 / 40, 9 / 40]
_A[3, :3] = [44 / 45, -56 / 15, 32 / 9]
_A[4, :4] = [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729]
_A[5, :5] = [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656]
_A[6, :6] = [35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]
#: 5th-order weights (equal to the last stage row: first same as last).
_B = _A[6].copy()
#: Error weights b5 - b4 (b4: embedded 4th-order weights).
_E = np.array([71 / 57600, 0.0, -71 / 16695, 71 / 1920, -17253 / 339200, 22 / 525, -1 / 40])
#: Dense output of DOPRI5 (Hairer, Nørsett & Wanner, §II.6).
_D = np.array(
    [
        -12715105075 / 11282082432,
        0.0,
        87487479700 / 32700410799,
        -10690763975 / 1880347072,
        701980252875 / 199316789632,
        -1453857185 / 822651844,
        69997945 / 29380423,
    ]
)

#: ``odeset`` names understood by :func:`ode45`. ``BDF``/``MaxOrder`` are
#: ``ode15s`` options; they do not apply to ode45 and are ignored (PROTEUS
#: passes ``'BDF','on'``). ``Stats`` is accepted and ignored.
_KNOWN = {
    "RelTol",
    "AbsTol",
    "NormControl",
    "MaxStep",
    "InitialStep",
    "Refine",
    "OutputFcn",
    "Stats",
    "BDF",
    "MaxOrder",
}

#: ``OutputFcn(t, y, flag) -> bool``; ``flag`` is ``"init"``, ``""`` or ``"done"``.
OutputFcn = Callable[[np.ndarray, np.ndarray, str], Any]


def odeset(**options: Any) -> dict[str, Any]:
    """MATLAB ``odeset`` restricted to the options ``ode45`` uses (case-sensitive names)."""
    unknown = set(options) - _KNOWN
    if unknown:
        raise ValueError(f"unsupported ode45 options: {sorted(unknown)}")
    return dict(options)


@dataclass
class ODEStats:
    """Counters like MATLAB's ``sol.stats``."""

    nsteps: int = 0
    nfailed: int = 0
    nfevals: int = 0


@dataclass
class ODEResult:
    """Result of :func:`ode45`; unpacks as ``t, y = ode45(...)``.

    ``t`` has shape ``(n,)``, ``y`` shape ``(n, neq)`` (MATLAB's ``[t, y]``).
    ``t_steps``/``y_steps`` hold every accepted step.
    """

    t: np.ndarray
    y: np.ndarray
    t_steps: np.ndarray
    y_steps: np.ndarray
    stats: ODEStats = field(default_factory=ODEStats)
    #: True if integration ended before ``tfinal`` (``OutputFcn`` returned
    #: true, or the step size fell below the minimum).
    stopped: bool = False
    message: str = ""

    def __iter__(self):
        yield self.t
        yield self.y


def _dense(y, ynew, f, h, theta):
    """DOPRI5 continuous extension at ``t + theta*h`` (Hairer et al. §II.6)."""
    r2 = ynew - y
    r3 = h * f[:, 0] - r2
    r4 = r2 - h * f[:, 6] - r3
    r5 = h * (f @ _D)
    return y + theta * (r2 + (1 - theta) * (r3 + theta * (r4 + (1 - theta) * r5)))


def _scaled_rms(v: np.ndarray, sc: np.ndarray) -> float:
    return float(np.sqrt(np.mean((v / sc) ** 2))) if v.size else 0.0


def ode45(
    fun: Callable[[float, np.ndarray], ArrayLike],
    tspan: ArrayLike,
    y0: ArrayLike,
    options: Mapping[str, Any] | None = None,
) -> ODEResult:
    """MATLAB-style ``[t, y] = ode45(odefun, tspan, y0, options)``.

    Parameters
    ----------
    fun
        ``fun(t, y) -> dy`` with ``y`` a 1-D array of length ``neq``.
    tspan
        ``[t0, tf]`` (output at every step, refined by ``Refine``) or more
        points, strictly monotone (output at exactly those points). ``tf < t0``
        integrates backwards.
    y0
        Initial state (flattened in column-major order like MATLAB's ``y0(:)``).
    options
        :func:`odeset` dictionary: ``RelTol`` (1e-3), ``AbsTol`` (1e-6, scalar
        or per component), ``NormControl`` (``"off"``), ``MaxStep``
        (``|tf - t0|/10``), ``InitialStep`` (automatic), ``Refine`` (4),
        ``OutputFcn``. An ``OutputFcn`` returning true stops the integration
        after the current step (hook for PROTEUS' wall-clock timeout).
    """
    opts = dict(options or {})
    unknown = set(opts) - _KNOWN
    if unknown:
        raise ValueError(f"unsupported ode45 options: {sorted(unknown)}")

    tspan = np.asarray(tspan, dtype=np.float64).reshape(-1)
    if tspan.size < 2:
        raise ValueError("tspan must have at least 2 elements")
    t0, tfinal = float(tspan[0]), float(tspan[-1])
    if t0 == tfinal:
        raise ValueError("the last entry of tspan must differ from the first")
    tdir = 1.0 if tfinal > t0 else -1.0
    if np.any(tdir * np.diff(tspan) <= 0):
        raise ValueError("tspan must be strictly increasing or strictly decreasing")
    y0 = np.asarray(y0, dtype=np.float64).reshape(-1, order="F").copy()
    neq = y0.size

    rtol = float(opts.get("RelTol", 1e-3))
    atol = np.asarray(opts.get("AbsTol", 1e-6), dtype=np.float64).reshape(-1)
    if atol.size not in (1, neq):
        raise ValueError("AbsTol must be a scalar or have one entry per equation")
    if rtol <= 0 or np.any(atol <= 0):
        raise ValueError("RelTol and AbsTol must be positive")
    rtol = max(rtol, 100 * np.finfo(float).eps)
    atol_vec = np.broadcast_to(atol, (neq,)).copy()
    norm_control = str(opts.get("NormControl", "off")).lower() == "on"
    if norm_control and atol.size != 1:
        raise ValueError("NormControl 'on' needs a scalar AbsTol")
    span = abs(tfinal - t0)
    hmax = float(opts.get("MaxStep", 0.1 * span))
    if hmax <= 0:
        raise ValueError("MaxStep must be positive")
    refine = int(opts.get("Refine", 4))
    if refine < 1:
        raise ValueError("Refine must be a positive integer")
    output_fcn: OutputFcn | None = opts.get("OutputFcn")
    threshold = atol_vec / rtol

    stats = ODEStats()

    def call(t: float, y: np.ndarray) -> np.ndarray:
        stats.nfevals += 1
        out = np.asarray(fun(t, y), dtype=np.float64).reshape(-1, order="F")
        if out.size != neq:
            raise ValueError(f"odefun returned {out.size} values, expected {neq}")
        return out

    def error(y: np.ndarray, ynew: np.ndarray, est: np.ndarray) -> float:
        if norm_control:
            scale = max(np.linalg.norm(y), np.linalg.norm(ynew), float(threshold[0]))
            return float(np.linalg.norm(est)) / scale
        scale = np.maximum(np.maximum(np.abs(y), np.abs(ynew)), threshold)
        return float(np.max(np.abs(est) / scale)) if neq else 0.0

    t = t0
    y = y0
    f = np.empty((neq, 7))
    f[:, 0] = call(t, y)
    hmin = 16 * np.spacing(abs(t))

    # Starting step (Hairer, Nørsett & Wanner, §II.4, with the order-4 estimate).
    if opts.get("InitialStep") is not None:
        absh = float(opts["InitialStep"])
        if absh <= 0:
            raise ValueError("InitialStep must be positive")
    else:
        sc = atol_vec + np.abs(y) * rtol
        d0 = _scaled_rms(y, sc)
        d1 = _scaled_rms(f[:, 0], sc)
        h0 = 1e-6 if (d0 < 1e-5 or d1 < 1e-5) else 0.01 * d0 / d1
        h0 = min(h0, hmax, span)
        f1 = call(t + tdir * h0, y + tdir * h0 * f[:, 0])
        d2 = _scaled_rms(f1 - f[:, 0], sc) / h0
        dm = max(d1, d2)
        h1 = max(1e-6, h0 * 1e-3) if dm <= 1e-15 else (0.01 / dm) ** (1 / 5)
        absh = min(100 * h0, h1)
    absh = min(max(absh, hmin), hmax, span)

    many = tspan.size > 2
    tout: list[float] = [t0]
    yout: list[np.ndarray] = [y0.copy()]
    t_steps: list[float] = [t0]
    y_steps: list[np.ndarray] = [y0.copy()]
    next_idx = 1  # next tspan entry to output (many-point mode)
    stopped = False
    message = ""
    if output_fcn is not None:
        output_fcn(tspan.copy(), y0.copy(), "init")

    done = False
    while not done:
        hmin = 16 * np.spacing(abs(t))
        absh = min(hmax, max(hmin, absh))
        if 1.1 * absh >= abs(tfinal - t):  # stretch the step to land on tfinal
            absh = abs(tfinal - t)
            done = True

        nofailed = True
        while True:
            h = tdir * absh
            for i in range(1, 6):
                f[:, i] = call(t + _C[i] * h, y + h * (f[:, :i] @ _A[i, :i]))
            tnew = tfinal if done else t + h
            h = tnew - t  # the step actually taken
            ynew = y + h * (f[:, :6] @ _B[:6])
            f[:, 6] = call(tnew, ynew)
            err = error(y, ynew, abs(h) * (f @ _E))
            if err <= rtol:
                break
            stats.nfailed += 1
            if absh <= hmin:
                message = (
                    f"Failure at t={t:e}: unable to meet integration tolerances "
                    f"without reducing the step size below the smallest value "
                    f"allowed ({hmin:e})."
                )
                warnings.warn(message, RuntimeWarning, stacklevel=2)
                break
            if nofailed:
                nofailed = False
                shrink = max(0.1, 0.8 * (rtol / err) ** 0.2) if np.isfinite(err) else 0.1
                absh = max(hmin, absh * shrink)
            else:
                absh = max(hmin, 0.5 * absh)
            done = False
        if message:
            stopped = True
            break
        stats.nsteps += 1

        new_t: list[float] = []
        new_y: list[np.ndarray] = []
        if many:
            while next_idx < tspan.size and tdir * (tnew - tspan[next_idx]) >= 0:
                tq = float(tspan[next_idx])
                new_t.append(tq)
                if tq == tnew:
                    new_y.append(ynew.copy())
                else:
                    new_y.append(_dense(y, ynew, f, h, (tq - t) / h))
                next_idx += 1
        else:
            for k in range(1, refine):
                theta = k / refine
                new_t.append(t + h * theta)
                new_y.append(_dense(y, ynew, f, h, theta))
            new_t.append(tnew)
            new_y.append(ynew.copy())
        tout.extend(new_t)
        yout.extend(new_y)
        t_steps.append(tnew)
        y_steps.append(ynew.copy())

        stop = (
            output_fcn is not None and new_t and output_fcn(np.array(new_t), np.array(new_y).T, "")
        )
        if stop:
            stopped = not done
            done = True

        if done:
            break
        if nofailed:
            absh *= 5.0 if err == 0 else min(5.0, 0.8 * (rtol / err) ** 0.2)
        t, y = tnew, ynew
        f[:, 0] = f[:, 6]

    if output_fcn is not None:
        output_fcn(np.empty(0), np.empty((neq, 0)), "done")

    return ODEResult(
        t=np.array(tout),
        y=np.array(yout).reshape(len(tout), neq),
        t_steps=np.array(t_steps),
        y_steps=np.array(y_steps).reshape(len(t_steps), neq),
        stats=stats,
        stopped=stopped,
        message=message,
    )
