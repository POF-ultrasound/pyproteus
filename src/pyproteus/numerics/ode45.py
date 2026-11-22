"""Clean-room MATLAB ``ode45`` (explicit Runge–Kutta 5(4), Dormand–Prince pair).

No MathWorks source code was read or translated. Sources:

- J. R. Dormand, P. J. Prince, "A family of embedded Runge-Kutta formulae",
  J. Comput. Appl. Math. 6 (1980) 19-26: the 7-stage FSAL tableau (RK5(4)7M).
- L. F. Shampine, M. W. Reichelt, "The MATLAB ODE Suite", SIAM J. Sci. Comput.
  18 (1997) 1-22, and L. F. Shampine, I. Gladwell, S. Thompson, *Solving ODEs
  with MATLAB* (Cambridge 2003): local extrapolation, mixed relative/absolute
  error test, step-size selection (initial step from the scaled slope,
  ``0.8 (rtol/err)^(1/5)`` after a first failure, halving after repeated
  failures, growth limited to 5x and not right after a failure, last step
  stretched by up to 10 %), ``Refine`` output through a 4th-order continuous
  extension, ``MaxStep`` default ``|tf - t0| / 10``.
- MATLAB documentation of ``ode45`` and ``odeset`` (option meanings, output
  rules for a 2-element versus a longer ``tspan``, ``OutputFcn`` protocol).

Details the sources leave open were fixed by black-box probing of MATLAB
R2025a (every right-hand-side call ``(t, y)`` logged and compared bit for bit;
``docs/dev/LOG.md``, M2 ode45 entry; pinned by ``tests/oracle/test_ode45.py``):

- stage ``j`` is ``y + h * (f_1 a_j1 + f_2 a_j2 + ...)`` summed left to right;
  the last stage uses ``h = tnew - t`` (the representable step);
- the error is ``absh * max(|f @ e| / max(|y|, |ynew|, AbsTol/RelTol))`` with
  ``absh`` the step *before* that adjustment;
- the step-size rule's ``x^(1/5)`` is fdlibm's ``pow`` (see ``_fdlibm_pow``);
- without ``InitialStep`` the first step is also capped by the first
  ``tspan`` interval;
- the continuous extension's coefficients (``_BI``) were identified by a
  least-squares fit to MATLAB's ``Refine`` output and recognised as rationals;
  output times are ``t + h*k/Refine`` and the extension is evaluated at
  ``s = (tq - t)/h``;
- ``OutputFcn`` gets ``[t0, tfinal]`` at ``"init"`` (even for a longer
  ``tspan``), is called only for steps that produce output, and ``"done"``
  passes empty arrays.

With these, the RHS calls, accepted/rejected steps and the solution at the
steps are bit-identical to MATLAB on every probe. Dense output (``Refine``
points, ``tspan`` points inside a step) agrees within 1e-15 of the solution
scale; for systems of several equations MATLAB's evaluation order is not
pinned down (``docs/dev/BUGS.md`` B-50).

The state is one vector; PROTEUS stacks a batch of bubbles into it, so all
bubbles of a batch share the adaptive steps (``docs/QUIRKS.md``, Q-2).
"""

from __future__ import annotations

import math
import struct
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["ODEResult", "ODEStats", "ode45", "odeset"]

# Dormand & Prince (1980), RK5(4)7M. Column j of _A holds the coefficients
# a_{j+1, i} that build stage j+1 (0-based) from the slopes f_1 .. f_j.
_C = np.array([0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0])
_A = np.zeros((7, 7))
_A[0, 1] = 1 / 5
_A[0:2, 2] = [3 / 40, 9 / 40]
_A[0:3, 3] = [44 / 45, -56 / 15, 32 / 9]
_A[0:4, 4] = [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729]
_A[0:5, 5] = [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656]
_A[0:6, 6] = [35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84]
#: 5th-order weights (= the last stage: first same as last).
_B = _A[:, 6].copy()
#: Error weights b5 - b4 (b4 is the embedded 4th-order solution).
_E = np.array([71 / 57600, 0.0, -71 / 16695, 71 / 1920, -17253 / 339200, 22 / 525, -1 / 40])
#: 4th-order continuous extension in power form: the solution at ``t + s*h``
#: is ``y + h * f @ (_BI @ [s, s^2, s^3, s^4])``; at ``s = 1`` the weights are the
#: 5th-order weights ``_B``. Identified by black-box fit (module docstring).
_BI = np.array(
    [
        [1.0, -183 / 64, 37 / 12, -145 / 128],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 1500 / 371, -1000 / 159, 1000 / 371],
        [0.0, -125 / 32, 125 / 12, -375 / 64],
        [0.0, 9477 / 3392, -729 / 106, 25515 / 6784],
        [0.0, -11 / 7, 11 / 3, -55 / 28],
        [0.0, 3 / 2, -4.0, 5 / 2],
    ]
)
#: Exponent of the step-size rule (1 / (order + 1) for the 4th-order error estimate).
_POW = 0.2


# --- fdlibm pow --------------------------------------------------------------
# MATLAB's scalar ``x^0.2`` is not correctly rounded (NumPy/libm ``x**0.2``
# differs by 1 ulp for ~10 % of inputs, which changes ode45's step sizes by an
# ulp and makes the step sequences drift apart). Black-box probing showed it is
# bit-identical to Java ``StrictMath.pow``, i.e. fdlibm 5.3 ``e_pow.c``
# (200 000 random inputs in [1e-8, 1e1], all equal; see docs/dev/LOG.md).
# The function below is a port of the fdlibm algorithm for finite x > 0 (all
# ode45 needs). fdlibm is not MathWorks code; its notice, preserved as its
# licence asks:
#
#   Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
#   Developed at SunSoft, a Sun Microsystems, Inc. business.
#   Permission to use, copy, modify, and distribute this software is freely
#   granted, provided that this notice is preserved.


def _hi(x: float) -> int:
    return struct.unpack("<q", struct.pack("<d", x))[0] >> 32


def _lo(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0] & 0xFFFFFFFF


def _mk(hi: int, lo: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", ((hi & 0xFFFFFFFF) << 32) | (lo & 0xFFFFFFFF)))[0]


def _clear_lo(x: float) -> float:
    return _mk(_hi(x), 0)


_BP = (1.0, 1.5)
_DP_H = (0.0, _mk(0x3FE2B803, 0x40000000))
_DP_L = (0.0, _mk(0x3E4CFDEB, 0x43CFD006))
_TWO53 = _mk(0x43400000, 0)
_LC = tuple(
    _mk(*w)
    for w in (
        (0x3FE33333, 0x33333303),
        (0x3FDB6DB6, 0xDB6FABFF),
        (0x3FD55555, 0x518F264D),
        (0x3FD17460, 0xA91D4101),
        (0x3FCD864A, 0x93C9DB65),
        (0x3FCA7E28, 0x4A454EEF),
    )
)
_PC = tuple(
    _mk(*w)
    for w in (
        (0x3FC55555, 0x5555553E),
        (0xBF66C16C, 0x16BEBD93),
        (0x3F11566A, 0xAF25DE2C),
        (0xBEBBBD41, 0xC5D26BF1),
        (0x3E663769, 0x72BEA4D0),
    )
)
_LG2 = _mk(0x3FE62E42, 0xFEFA39EF)
_LG2_H = _mk(0x3FE62E43, 0x00000000)
_LG2_L = _mk(0xBE205C61, 0x0CA86C39)
_CP = _mk(0x3FEEC709, 0xDC3A03FD)
_CP_H = _mk(0x3FEEC709, 0xE0000000)
_CP_L = _mk(0xBE3E2FE0, 0x145B01F5)


def _fdlibm_pow(x: float, y: float) -> float:
    """fdlibm ``pow(x, y)`` for finite ``x > 0``, finite ``y`` and a result in the normal range."""
    l1, l2, l3, l4, l5, l6 = _LC
    p1, p2, p3, p4, p5 = _PC
    # log2(x) = n + log2(ax), ax in [sqrt(2)/2, sqrt(2)) roughly, as t1 + t2.
    ax = x
    ix = _hi(ax) & 0x7FFFFFFF
    n = 0
    if ix < 0x00100000:  # subnormal
        ax *= _TWO53
        n -= 53
        ix = _hi(ax)
    n += (ix >> 20) - 0x3FF
    j = ix & 0x000FFFFF
    ix = j | 0x3FF00000
    if j <= 0x3988E:
        k = 0
    elif j < 0xBB67A:
        k = 1
    else:
        k = 0
        n += 1
        ix -= 0x00100000
    ax = _mk(ix, _lo(ax))
    u = ax - _BP[k]
    v = 1.0 / (ax + _BP[k])
    ss = u * v
    s_h = _clear_lo(ss)
    t_h = _mk(((ix >> 1) | 0x20000000) + 0x00080000 + (k << 18), 0)
    t_l = ax - (t_h - _BP[k])
    s_l = v * ((u - s_h * t_h) - s_h * t_l)
    s2 = ss * ss
    r = s2 * s2 * (l1 + s2 * (l2 + s2 * (l3 + s2 * (l4 + s2 * (l5 + s2 * l6)))))
    r += s_l * (s_h + ss)
    s2 = s_h * s_h
    t_h = _clear_lo(3.0 + s2 + r)
    t_l = r - ((t_h - 3.0) - s2)
    u = s_h * t_h
    v = s_l * t_h + t_l * ss
    p_h = _clear_lo(u + v)
    p_l = v - (p_h - u)
    z_h = _CP_H * p_h
    z_l = _CP_L * p_h + p_l * _CP + _DP_L[k]
    t = float(n)
    t1 = _clear_lo(((z_h + z_l) + _DP_H[k]) + t)
    t2 = z_l - (((t1 - t) - _DP_H[k]) - z_h)
    # y * log2(x) = p_h + p_l, then 2**(p_h + p_l).
    y1 = _clear_lo(y)
    p_l = (y - y1) * t1 + y * t2
    p_h = y1 * t1
    z = p_l + p_h
    j = _hi(z)
    i = j & 0x7FFFFFFF
    k = (i >> 20) - 0x3FF
    n = 0
    if i > 0x3FE00000:  # |z| > 0.5: n = nearest integer to z
        n = j + (0x00100000 >> (k + 1))
        k = ((n & 0x7FFFFFFF) >> 20) - 0x3FF
        t = _mk(n & ~(0x000FFFFF >> k), 0)
        n = ((n & 0x000FFFFF) | 0x00100000) >> (20 - k)
        if j < 0:
            n = -n
        p_h -= t
    t = _clear_lo(p_l + p_h)
    u = t * _LG2_H
    v = (p_l - (t - p_h)) * _LG2 + t * _LG2_L
    z = u + v
    w = v - (z - u)
    t = z * z
    t1 = z - t * (p1 + t * (p2 + t * (p3 + t * (p4 + t * p5))))
    r = (z * t1) / (t1 - 2.0) - (w + z * w)
    z = 1.0 - (r - z)
    if ((_hi(z) + (n << 20)) >> 20) <= 0:
        return math.ldexp(z, n)
    return _mk(_hi(z) + (n << 20), _lo(z))


def _pow5(x: float) -> float:
    """MATLAB scalar ``x^0.2`` (fdlibm), with IEEE special cases."""
    x = float(x)
    if x > 0 and math.isfinite(x):
        return _fdlibm_pow(x, _POW)
    if x == 0:
        return 0.0
    return x if x > 0 or math.isnan(x) else math.nan  # +Inf, NaN; negative -> NaN


#: ``odeset`` names understood by :func:`ode45`. ``BDF`` and ``MaxOrder``
#: belong to ``ode15s``; MATLAB's ``ode45`` accepts and ignores them, and so
#: does this port (PROTEUS sets ``'BDF','on'``).
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
    """MATLAB ``odeset`` with the field names ``ode45`` uses (case-sensitive)."""
    unknown = set(options) - _KNOWN
    if unknown:
        raise ValueError(f"unsupported ode45 options: {sorted(unknown)}")
    return dict(options)


@dataclass
class ODEStats:
    """MATLAB ``sol.stats`` counters."""

    nsteps: int = 0
    nfailed: int = 0
    nfevals: int = 0


@dataclass
class ODEResult:
    """Outputs of :func:`ode45`; unpacks as ``t, y = ode45(...)``.

    ``t`` has shape ``(n,)`` and ``y`` shape ``(n, neq)`` (MATLAB's ``[t, y]``
    with ``t`` a column). ``t_steps``/``y_steps`` hold every accepted step
    (MATLAB ``sol.x``/``sol.y.'``), whatever ``tspan`` asks for.
    """

    t: np.ndarray
    y: np.ndarray
    t_steps: np.ndarray
    y_steps: np.ndarray
    stats: ODEStats = field(default_factory=ODEStats)
    #: True if integration ended early (``OutputFcn`` returned true, or the
    #: step size fell below the minimum).
    stopped: bool = False
    message: str = ""

    def __iter__(self):
        yield self.t
        yield self.y


def _as_vector(a: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim > 1 and sum(n > 1 for n in arr.shape) > 1:
        raise ValueError(f"{name} must be a vector, got shape {arr.shape}")
    return arr.reshape(-1)


def _lincomb(f: np.ndarray, coef: np.ndarray, n: int) -> np.ndarray:
    """``f[:, :n] @ coef[:n]`` summed left to right (fixed order, no FMA)."""
    acc = f[:, 0] * coef[0]
    for i in range(1, n):
        acc = acc + f[:, i] * coef[i]
    return acc


def _interp(y: np.ndarray, f: np.ndarray, h: float, s: float) -> np.ndarray:
    """Continuous extension at ``t + s*h`` (weights in Horner form)."""
    w = s * (_BI[:, 0] + s * (_BI[:, 1] + s * (_BI[:, 2] + s * _BI[:, 3])))
    return y + h * _lincomb(f, w, 7)


def _eps(x: float) -> float:
    """MATLAB ``eps(x)``: spacing of doubles at ``|x|`` (``eps(0)`` is the smallest subnormal)."""
    return float(np.spacing(abs(x)))


def ode45(
    fun: Callable[[float, np.ndarray], ArrayLike],
    tspan: ArrayLike,
    y0: ArrayLike,
    options: Mapping[str, Any] | None = None,
    *,
    record_calls: list[tuple[float, np.ndarray]] | None = None,
) -> ODEResult:
    """MATLAB ``[t, y] = ode45(odefun, tspan, y0, options)``.

    Parameters
    ----------
    fun
        ``fun(t, y) -> dy`` with ``y`` a 1-D array of length ``neq`` (MATLAB's
        column ``y0``); the result is flattened (MATLAB: column vector).
    tspan
        ``[t0, tf]`` (output at every step, refined by ``Refine``) or more
        points, strictly monotone (output at exactly those points). ``tf < t0``
        integrates backwards.
    y0
        Initial state (any shape; flattened in column-major order like MATLAB's ``y0(:)``).
    options
        :func:`odeset` dictionary: ``RelTol`` (1e-3), ``AbsTol`` (1e-6, scalar
        or per component), ``NormControl`` (``"off"``), ``MaxStep``
        (``|tf - t0|/10``), ``InitialStep`` (automatic), ``Refine`` (4),
        ``OutputFcn``. An ``OutputFcn`` returning true stops the integration
        after the current step (hook for PROTEUS' wall-clock timeout).
    record_calls
        Testing aid: if a list is given, every right-hand-side call appends ``(t, y.copy())``.
    """
    opts = dict(options or {})
    unknown = set(opts) - _KNOWN
    if unknown:
        raise ValueError(f"unsupported ode45 options: {sorted(unknown)}")

    tspan = _as_vector(tspan, "tspan")
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
    # MATLAB raises RelTol below 100*eps with a warning; same here, silently.
    rtol = max(rtol, 100 * np.finfo(float).eps)
    norm_control = str(opts.get("NormControl", "off")).lower() == "on"
    span = abs(tfinal - t0)
    hmax = float(opts.get("MaxStep", 0.1 * span))
    if hmax <= 0:
        raise ValueError("MaxStep must be positive")
    hmax = min(hmax, span)
    htry = opts.get("InitialStep")
    refine = int(opts.get("Refine", 4))
    output_fcn: OutputFcn | None = opts.get("OutputFcn")
    threshold = atol / rtol
    scalar_atol = atol.size == 1
    if scalar_atol:
        threshold = float(threshold[0])
        atol_s = float(atol[0])

    stats = ODEStats()

    def call(t: float, y: np.ndarray) -> np.ndarray:
        stats.nfevals += 1
        if record_calls is not None:
            record_calls.append((t, y.copy()))
        out = np.asarray(fun(t, y), dtype=np.float64).reshape(-1, order="F")
        if out.size != neq:
            raise ValueError(f"odefun returned {out.size} values, expected {neq}")
        return out

    def norm(v: np.ndarray) -> float:
        return float(np.max(np.abs(v))) if v.size else 0.0

    t = t0
    y = y0
    f0 = call(t, y)

    # Initial step (Shampine & Reichelt 1997; Shampine, Gladwell & Thompson §1.3).
    hmin = 16 * _eps(t)
    if htry is None:
        absh = min(hmax, abs(float(tspan[1]) - t0))  # first output interval (probed)
        if norm_control:
            rh = (np.linalg.norm(f0) / max(np.linalg.norm(y), atol_s / rtol)) / (0.8 * _pow5(rtol))
        else:
            rh = norm(f0 / np.maximum(np.abs(y), threshold)) / (0.8 * _pow5(rtol))
        if absh * rh > 1:
            absh = 1 / rh
        absh = max(absh, hmin)
    else:
        htry = float(htry)
        if htry <= 0:
            raise ValueError("InitialStep must be positive")
        absh = min(hmax, max(hmin, htry))

    many = tspan.size > 2
    # Output buffers.
    tout: list[float] = [t0]
    yout: list[np.ndarray] = [y0.copy()]
    t_steps: list[float] = [t0]
    y_steps: list[np.ndarray] = [y0.copy()]
    next_idx = 1  # next tspan index to output (many-point mode)

    stopped = False
    message = ""
    if output_fcn is not None:
        # MATLAB passes [t0, tfinal] here even for a longer tspan (probed).
        output_fcn(np.array([t0, tfinal]), y0.copy(), "init")

    f = np.empty((neq, 7))
    f[:, 0] = f0
    done = False
    while not done:
        hmin = 16 * _eps(t)
        absh = min(hmax, max(hmin, absh))
        h = tdir * absh
        if 1.1 * absh >= abs(tfinal - t):
            h = tfinal - t
            absh = abs(h)
            done = True

        nofailed = True
        while True:
            # Stages 2..6 at t + c_j h; stage 7 (the 5th-order solution, FSAL)
            # at tnew, with h replaced by the representable step tnew - t.
            for j in range(1, 6):
                ytmp = y + h * _lincomb(f, _A[:, j], j)
                f[:, j] = call(t + _C[j] * h, ytmp)
            tnew = tfinal if done else t + h
            h = tnew - t
            ynew = y + h * _lincomb(f, _B, 6)
            f[:, 6] = call(tnew, ynew)

            # Error estimate, max norm of the weighted local error.
            if norm_control:
                errwt = max(max(np.linalg.norm(y), np.linalg.norm(ynew)), threshold)
                err = absh * (np.linalg.norm(_lincomb(f, _E, 7)) / errwt)
            else:
                w = np.maximum(np.maximum(np.abs(y), np.abs(ynew)), threshold)
                err = absh * norm(_lincomb(f, _E, 7) / w)

            if err > rtol:
                stats.nfailed += 1
                if absh <= hmin:
                    # MATLAB warns (MATLAB:ode45:IntegrationTolNotMet) and
                    # returns the solution so far; same text here.
                    message = (
                        f"Failure at t={t:e}.  Unable to meet integration tolerances "
                        f"without reducing the step size below the smallest value "
                        f"allowed ({hmin:e}) at time t."
                    )
                    warnings.warn(message, RuntimeWarning, stacklevel=2)
                    stopped = True
                    break
                if nofailed:
                    nofailed = False
                    absh = max(hmin, absh * max(0.1, 0.8 * _pow5(rtol / err)))
                else:
                    absh = max(hmin, 0.5 * absh)
                h = tdir * absh
                done = False
            else:
                break
        if message:
            break
        stats.nsteps += 1

        # Output for this step.
        new_t: list[float] = []
        new_y: list[np.ndarray] = []
        if many:
            while next_idx < tspan.size and tdir * (tnew - tspan[next_idx]) >= 0:
                tq = float(tspan[next_idx])
                if tq == tnew:
                    yq = ynew.copy()
                else:
                    s = (tq - t) / h
                    yq = _interp(y, f, h, s)
                new_t.append(tq)
                new_y.append(yq)
                next_idx += 1
        else:
            for k in range(1, refine):
                tq = t + h * (k / refine)
                yq = _interp(y, f, h, (tq - t) / h)
                new_t.append(tq)
                new_y.append(yq)
            new_t.append(tnew)
            new_y.append(ynew.copy())
        tout.extend(new_t)
        yout.extend(new_y)
        t_steps.append(tnew)
        y_steps.append(ynew.copy())

        if output_fcn is not None and new_t:
            status = output_fcn(np.array(new_t), np.array(new_y).T, "")
            if status:
                stopped = True
                done = True

        if done:
            break

        # Step-size update after an accepted step.
        if nofailed:
            temp = 1.25 * _pow5(err / rtol)
            absh = absh / temp if temp > 0.2 else 5.0 * absh

        t = tnew
        y = ynew
        f[:, 0] = f[:, 6]

    if output_fcn is not None:
        output_fcn(np.empty(0), np.empty((0, 0)), "done")

    return ODEResult(
        t=np.array(tout),
        y=np.array(yout).reshape(len(tout), neq),
        t_steps=np.array(t_steps),
        y_steps=np.array(y_steps).reshape(len(t_steps), neq),
        stats=stats,
        stopped=stopped,
        message=message,
    )
