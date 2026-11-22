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

Choices the sources leave open (documented here, not tuned to reproduce any
other solver step for step):

- error measure ``err = max_i |h (f @ E)_i| / max(|y_i|, |ynew_i|, AbsTol_i/RelTol)``
  (scale ``max(|y|, |ynew|)`` as in Hairer et al. eq. (4.10)); the step is
  accepted when ``err <= RelTol``, i.e. the documented test;
- new step ``h * min(facmax, max(facmin, 0.9 (RelTol/err)^(1/5)))``: Hairer et
  al. eq. (4.13) with the safety factor and bounds of their DOPRI5 code
  (0.9, ``facmin = 0.2``, ``facmax = 10``), and ``facmax = 1`` for the step
  that follows a rejection (their recommendation; the step may still shrink);
  after a rejection the retry uses ``max(facmin, 0.9 (RelTol/err)^(1/5))``;
- own choice: no step-size stabilisation (DOPRI5's default ``beta = 0.04``
  is not used; plain eq. (4.13) is the textbook controller and PROTEUS'
  bubble problems are non-stiff, where the gain is small);
- from DOPRI5: every attempt lands on ``tfinal`` when ``1.01 h`` reaches it
  (here only if the remaining distance is within ``MaxStep``, which is never
  exceeded); ``RelTol`` must exceed ``10 eps``; integration stops with a
  warning when the error test fails at ``|h| <= 10 eps |t|`` (own addition:
  ``|t|`` is floored at ``|tfinal - t0|`` so the test also works near
  ``t = 0``);
- starting step from Hairer et al. §II.4 (one extra RHS call), capped by
  ``MaxStep``; ``InitialStep`` overrides it; with a non-finite RHS at ``t0``
  the start is ``MaxStep`` and error control shrinks it;
- ``Refine`` points at ``t + h k/Refine`` (k = 1 .. Refine-1); ``OutputFcn`` is
  called after each accepted step that produced output (with a sparse
  ``tspan`` that can be much rarer than every step, which limits how quickly an
  ``OutputFcn``-based timeout reacts).

Near a singularity relative error control can accept steps on a very large
``y``; the solver then stops shortly after (not exactly at) the singularity.

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

#: Step-size controller constants (Hairer, Nørsett & Wanner, DOPRI5 defaults).
_SAFETY = 0.9
_FACMIN = 0.2
_FACMAX = 10.0
_EPS = float(np.finfo(float).eps)

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
    if not np.all(np.isfinite(tspan)):
        raise ValueError("tspan must be finite")
    if tspan.size < 2:
        raise ValueError("tspan must have at least 2 elements")
    t_start, t_end = float(tspan[0]), float(tspan[-1])
    if t_start == t_end:
        raise ValueError("the last entry of tspan must differ from the first")
    sign = 1.0 if t_end > t_start else -1.0
    if np.any(sign * np.diff(tspan) <= 0):
        raise ValueError("tspan must be strictly increasing or strictly decreasing")
    x0 = np.asarray(y0, dtype=np.float64).reshape(-1, order="F").copy()
    n = x0.size

    rtol = float(opts.get("RelTol", 1e-3))
    atol = np.asarray(opts.get("AbsTol", 1e-6), dtype=np.float64).reshape(-1)
    if atol.size not in (1, n):
        raise ValueError("AbsTol must be a scalar or have one entry per equation")
    if rtol <= 0 or np.any(atol <= 0):
        raise ValueError("RelTol and AbsTol must be positive")
    if rtol <= 10 * _EPS:
        raise ValueError("RelTol must exceed 10*eps")
    atol = np.broadcast_to(atol, (n,)).copy()
    norm_control = str(opts.get("NormControl", "off")).lower() == "on"
    if norm_control and np.any(atol != atol[0]):
        raise ValueError("NormControl 'on' needs a scalar AbsTol")
    span = abs(t_end - t_start)
    max_step = float(opts.get("MaxStep", 0.1 * span))
    if max_step <= 0:
        raise ValueError("MaxStep must be positive")
    refine = int(opts.get("Refine", 4))
    if refine < 1:
        raise ValueError("Refine must be a positive integer")
    output_fcn: OutputFcn | None = opts.get("OutputFcn")
    floor = (
        atol / rtol
    )  # |e_i| <= max(RelTol |y_i|, AbsTol_i)  <=>  |e_i| / max(|y_i|, floor_i) <= RelTol

    stats = ODEStats()

    def rhs(t: float, x: np.ndarray) -> np.ndarray:
        stats.nfevals += 1
        out = np.asarray(fun(t, x), dtype=np.float64).reshape(-1, order="F")
        if out.size != n:
            raise ValueError(f"odefun returned {out.size} values, expected {n}")
        return out

    def scaled_error(x: np.ndarray, x_next: np.ndarray, est: np.ndarray) -> float:
        if norm_control:
            denom = max(np.linalg.norm(x), np.linalg.norm(x_next), float(floor[0]))
            return float(np.linalg.norm(est)) / denom
        denom = np.maximum(np.maximum(np.abs(x), np.abs(x_next)), floor)
        return float(np.max(np.abs(est) / denom)) if n else 0.0

    def min_step(t: float) -> float:
        return 10 * _EPS * max(abs(t), span)

    t = t_start
    x = x0
    k = np.empty((n, 7))
    k[:, 0] = rhs(t, x)

    # Starting step: Hairer, Nørsett & Wanner, §II.4 (order-4 estimate, exponent 1/5).
    if opts.get("InitialStep") is not None:
        step = float(opts["InitialStep"])
        if step <= 0:
            raise ValueError("InitialStep must be positive")
    elif not np.all(np.isfinite(k[:, 0])):
        step = max_step  # no usable estimate; error control shrinks it
    else:
        sc = atol + np.abs(x) * rtol
        d0 = _scaled_rms(x, sc)
        d1 = _scaled_rms(k[:, 0], sc)
        h0 = 1e-6 if (d0 < 1e-5 or d1 < 1e-5) else 0.01 * d0 / d1
        h0 = min(h0, max_step, span)
        k1 = rhs(t + sign * h0, x + sign * h0 * k[:, 0])
        d2 = _scaled_rms(k1 - k[:, 0], sc) / h0
        if not (np.isfinite(d1) and np.isfinite(d2)):
            step = h0
        else:
            dm = max(d1, d2)
            h1 = max(1e-6, h0 * 1e-3) if dm <= 1e-15 else (0.01 / dm) ** (1 / 5)
            step = min(100 * h0, h1)
    step = min(max(step, min_step(t)), max_step, span)

    many = tspan.size > 2
    tout: list[float] = [t_start]
    yout: list[np.ndarray] = [x0.copy()]
    t_steps: list[float] = [t_start]
    y_steps: list[np.ndarray] = [x0.copy()]
    next_out = 1  # next tspan entry to output (many-point mode)
    stopped = False
    message = ""
    if output_fcn is not None:
        output_fcn(tspan.copy(), x0.copy(), "init")

    finished = False
    while not finished:
        hmin = min_step(t)
        step = min(max_step, max(hmin, step))
        rejected = False
        while True:
            remaining = abs(t_end - t)
            # Land on t_end when 1.01*step reaches it (HNW DOPRI5), never beyond MaxStep.
            last = 1.01 * step >= remaining and remaining <= max_step
            if last:
                step = remaining
            h = sign * step
            for i in range(1, 6):
                k[:, i] = rhs(t + _C[i] * h, x + h * (k[:, :i] @ _A[i, :i]))
            t_next = t_end if last else t + h
            h = t_next - t  # the step actually taken
            x_next = x + h * (k[:, :6] @ _B[:6])
            k[:, 6] = rhs(t_next, x_next)
            err = scaled_error(x, x_next, abs(h) * (k @ _E))
            if err <= rtol:
                break
            stats.nfailed += 1
            if step <= hmin:
                message = (
                    f"ode45 stopped at t = {t:.6e}: the error test fails even at the "
                    f"smallest allowed step ({hmin:.3e})."
                )
                warnings.warn(message, RuntimeWarning, stacklevel=2)
                break
            rejected = True
            factor = max(_FACMIN, _SAFETY * (rtol / err) ** 0.2) if np.isfinite(err) else _FACMIN
            step = max(hmin, step * min(1.0, factor))
        if message:
            stopped = True
            break
        stats.nsteps += 1

        new_t: list[float] = []
        new_y: list[np.ndarray] = []
        if many:
            while next_out < tspan.size and sign * (t_next - tspan[next_out]) >= 0:
                tq = float(tspan[next_out])
                new_t.append(tq)
                if tq == t_next:
                    new_y.append(x_next.copy())
                else:
                    new_y.append(_dense(x, x_next, k, h, (tq - t) / h))
                next_out += 1
        else:
            for j in range(1, refine):
                theta = j / refine
                new_t.append(t + h * theta)
                new_y.append(_dense(x, x_next, k, h, theta))
            new_t.append(t_next)
            new_y.append(x_next.copy())
        tout.extend(new_t)
        yout.extend(new_y)
        t_steps.append(t_next)
        y_steps.append(x_next.copy())

        finished = last
        stop = (
            output_fcn is not None and new_t and output_fcn(np.array(new_t), np.array(new_y).T, "")
        )
        if stop:
            stopped = not last
            finished = True
        if finished:
            break
        # Next step, HNW eq. (4.13); facmax = 1 right after a rejection.
        factor = _FACMAX if err == 0 else _SAFETY * (rtol / err) ** 0.2
        step *= min(1.0 if rejected else _FACMAX, max(_FACMIN, factor))
        t, x = t_next, x_next
        k[:, 0] = k[:, 6]

    if output_fcn is not None:
        output_fcn(np.empty(0), np.empty((n, 0)), "done")

    return ODEResult(
        t=np.array(tout),
        y=np.array(yout).reshape(len(tout), n),
        t_steps=np.array(t_steps),
        y_steps=np.array(y_steps).reshape(len(t_steps), n),
        stats=stats,
        stopped=stopped,
        message=message,
    )
