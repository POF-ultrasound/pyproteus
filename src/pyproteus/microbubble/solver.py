"""Dormand–Prince 5(4) with independent step control per system, vectorised over a batch (D-22).

``N`` independent ODE systems of ``m`` equations each (one microbubble = one
system with ``m = 2``) are advanced together: every system has its own time,
step size, error estimate and accept/reject decision, and the right-hand side
is evaluated once per stage for all still-active systems. A system's result
does not depend on which other systems share the batch: the stage sums run in
a fixed order, elementwise, with no matmul/BLAS reduction, so batch results
equal single-system results bit for bit as long as NumPy's elementwise ufuncs
(pow, exp) do not depend on array length or alignment (tested, not guaranteed:
``tests/unit/test_microbubble_model.py``).

Method, written from the same published sources as
:func:`pyproteus.numerics.ode45` (Dormand & Prince 1980; Hairer, Nørsett &
Wanner, *Solving ODEs I*, §II.4 and §II.6) and with the same choices: error
``max_i |h (K E)_i| / max(|y_i|, |ynew_i|, AbsTol/RelTol)`` accepted when
``<= RelTol``; step factor ``0.9 (RelTol/err)^(1/5)`` bounded to [0.2, 10],
at most 1 right after a rejection; landing on the last output time when
``1.01 h`` reaches it within ``MaxStep``; output at the requested times by
the DOPRI5 continuous extension.

Failure detection (D-22, replaces PROTEUS' wall-clock timer): a system stops
and is reported when its state becomes non-finite and cannot be recovered by
shrinking the step, when a user check flags it (e.g. radius ≤ 0 or blow-up),
when its step falls below ``16 eps |t|`` (``|t|`` floored at ``1e-6`` of its
time span), or when it exceeds ``max_steps`` attempted steps. An optional
wall-clock limit stops all active systems. Other systems carry on.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from pyproteus.numerics.ode45 import _A, _B, _C, _D, _E

__all__ = ["BatchFailure", "BatchODEResult", "dopri5_batch"]

_SAFETY, _FACMIN, _FACMAX = 0.9, 0.2, 10.0
_EPS = float(np.finfo(float).eps)


@dataclass
class BatchFailure:
    """One system that did not reach the end."""

    index: int
    t: float
    reason: str


@dataclass
class BatchODEResult:
    """``y`` has shape ``(N, M, m)`` (NaN after a failure); ``n_out[i]`` output times reached."""

    y: np.ndarray
    n_out: np.ndarray
    failures: list[BatchFailure] = field(default_factory=list)
    nsteps: np.ndarray | None = None
    nfailed: np.ndarray | None = None
    nfevals: int = 0  # RHS calls (each for the whole active batch)


def _stage_sum(K: list[np.ndarray], coeffs: np.ndarray) -> np.ndarray:
    """``sum_j coeffs[j] K[j]`` in a fixed order (batch-size independent rounding)."""
    acc = None
    for c, k in zip(coeffs, K, strict=False):
        if c == 0.0:
            continue
        acc = c * k if acc is None else acc + c * k
    return acc if acc is not None else np.zeros_like(K[0])


def dopri5_batch(
    fun: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    tgrid: np.ndarray,
    y0: np.ndarray,
    *,
    rtol: float,
    atol: float,
    initial_step: float | np.ndarray,
    max_step: np.ndarray | None = None,
    max_steps: int = 1_000_000,
    check: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    check_reasons: dict[int, str] | None = None,
    wall_clock_limit: float | None = None,
) -> BatchODEResult:
    """Integrate ``N`` independent systems ``dy/dt = fun(t, y, idx)``.

    Parameters
    ----------
    fun
        ``fun(t, y, idx) -> dy``: ``t`` shape ``(k,)``, ``y`` shape ``(k, m)``
        for the active systems ``idx`` (indices into the batch, shape ``(k,)``).
    tgrid
        Output times per system, shape ``(N, M)``, strictly increasing along
        axis 1, ``M >= 2``; integration runs from ``tgrid[:, 0]`` to ``tgrid[:, -1]``.
    y0
        Initial states, shape ``(N, m)``.
    rtol, atol
        Scalar tolerances (as ``ode45``'s ``RelTol``/``AbsTol``).
    initial_step
        First step per system (scalar or ``(N,)``).
    max_step
        Per system; default a tenth of its time span (``ode45`` default).
    max_steps
        Attempted steps per system before it is reported as failed.
    check
        ``check(ynew, idx) -> int codes``, 0 where the accepted new state is
        valid; a nonzero code stops that system with reason ``check_reasons[code]``.
    wall_clock_limit
        Seconds; ``None`` (default) = no limit.
    """
    tgrid = np.asarray(tgrid, dtype=np.float64)
    y0 = np.asarray(y0, dtype=np.float64)
    N, M = tgrid.shape
    m = y0.shape[1]
    if M < 2 or np.any(np.diff(tgrid, axis=1) <= 0):
        raise ValueError("tgrid must be strictly increasing with at least two times per system")
    t0, tend = tgrid[:, 0].copy(), tgrid[:, -1].copy()
    span = tend - t0
    hmax = 0.1 * span if max_step is None else np.broadcast_to(max_step, (N,)).astype(float)
    tfloor = 1e-6 * span
    floor = atol / rtol

    y_out = np.full((N, M, m), np.nan)
    y_out[:, 0] = y0
    n_out = np.ones(N, dtype=np.int64)
    t = t0.copy()
    y = y0.copy()
    h = np.minimum(np.broadcast_to(initial_step, (N,)).astype(float), np.minimum(hmax, span))
    rejected = np.zeros(N, dtype=bool)  # last attempt of the current step was rejected
    nonfinite = np.zeros(N, dtype=bool)
    nsteps = np.zeros(N, dtype=np.int64)
    nfailed = np.zeros(N, dtype=np.int64)
    attempts = np.zeros(N, dtype=np.int64)
    active = np.arange(N)
    failures: list[BatchFailure] = []
    failed = np.zeros(N, dtype=bool)
    n_reported = 0
    nfev = 0

    k1 = np.asarray(fun(t, y, active), dtype=np.float64).reshape(N, m)
    nfev += 1
    deadline = None if wall_clock_limit is None else time.monotonic() + wall_clock_limit

    def fail(sel: np.ndarray, reason: str) -> None:
        for i in sel:
            failures.append(BatchFailure(int(i), float(t[i]), reason))

    while active.size:
        if deadline is not None and time.monotonic() >= deadline:
            fail(active, f"wall-clock limit of {wall_clock_limit:g} s reached")
            break
        idx = active
        ti, yi, ki1 = t[idx], y[idx], k1[idx]
        hi = np.minimum(h[idx], hmax[idx])  # MaxStep is never exceeded (as ode45)
        remaining = tend[idx] - ti
        last = (1.01 * hi >= remaining) & (remaining <= hmax[idx])
        hi = np.where(last, remaining, hi)
        hreq = hi.copy()  # requested step (``tn - ti`` below may round it up)
        K = [ki1]
        for s in range(1, 6):
            ys = yi + hi[:, None] * _stage_sum(K, _A[s, :s])
            K.append(np.asarray(fun(ti + _C[s] * hi, ys, idx), dtype=np.float64).reshape(-1, m))
            nfev += 1
        tn = np.where(last, tend[idx], ti + hi)
        hi = tn - ti
        yn = yi + hi[:, None] * _stage_sum(K, _B[:6])
        K.append(np.asarray(fun(tn, yn, idx), dtype=np.float64).reshape(-1, m))
        nfev += 1
        est = np.abs(hi)[:, None] * _stage_sum(K, _E)
        with np.errstate(invalid="ignore", over="ignore"):
            den = np.maximum(np.maximum(np.abs(yi), np.abs(yn)), floor)
            err = np.max(np.abs(est) / den, axis=1)
        finite = np.isfinite(err) & np.all(np.isfinite(yn), axis=1)
        err = np.where(finite, err, np.inf)
        ok = err <= rtol
        attempts[idx] += 1

        # ---- rejected attempts: shrink, or fail on step underflow
        bad = ~ok
        if bad.any():
            b = idx[bad]
            nfailed[b] += 1
            nonfinite[b] = ~finite[bad]
            hmin = 16 * _EPS * np.maximum(np.abs(t[b]), tfloor[b])
            under = hreq[bad] <= hmin
            if under.any():
                u = b[under]
                for i, nf in zip(u, nonfinite[u], strict=True):
                    why = "non-finite state" if nf else "error test fails"
                    failures.append(
                        BatchFailure(int(i), float(t[i]), f"step size underflow ({why})")
                    )
            fac = np.where(
                np.isfinite(err[bad]),
                np.maximum(_FACMIN, _SAFETY * (rtol / err[bad]) ** 0.2),
                _FACMIN,
            )
            h[b] = np.maximum(hmin, hreq[bad] * np.minimum(1.0, fac))
            rejected[b] = True

        # ---- accepted steps
        if ok.any():
            a = idx[ok]
            ya, yna, ha, ta, tna = yi[ok], yn[ok], hi[ok], ti[ok], tn[ok]
            Ka = [k[ok] for k in K]
            codes = (
                np.asarray(check(yna, a), dtype=np.int64)
                if check is not None
                else np.zeros(a.size, np.int64)
            )
            invalid = codes != 0
            # outputs crossed by this step (invalid new states write none)
            pos = n_out[a].copy()
            live = ~invalid
            look = np.arange(16)
            while True:
                cand = pos[:, None] + look[None, :]  # next 16 outputs of each system
                tq = tgrid[a[:, None], np.minimum(cand, M - 1)]
                cross = live[:, None] & (cand < M) & (tq <= tna[:, None])
                if not cross.any():
                    break
                r, c = np.nonzero(cross)
                theta = (tq[r, c] - ta[r]) / ha[r]
                vals = _dense(ya[r], yna[r], [k[r] for k in Ka], ha[r], theta)
                exact = tq[r, c] == tna[r]
                vals[exact] = yna[r][exact]
                y_out[a[r], cand[r, c]] = vals
                pos += cross.sum(axis=1)
            n_out[a] = pos
            nsteps[a] += 1
            with np.errstate(divide="ignore"):
                fac = np.where(err[ok] == 0, _FACMAX, _SAFETY * (rtol / err[ok]) ** 0.2)
            fac = np.minimum(np.where(rejected[a], 1.0, _FACMAX), np.maximum(_FACMIN, fac))
            h[a] = ha * fac
            rejected[a] = False
            t[a], y[a], k1[a] = tna, yna, Ka[6]
            for i, c in zip(a[invalid], codes[invalid], strict=True):
                reasons = check_reasons or {}
                failures.append(
                    BatchFailure(int(i), float(t[i]), reasons.get(int(c), f"check {c}"))
                )

        for f in failures[n_reported:]:
            failed[f.index] = True
        n_reported = len(failures)
        finished = t[idx] >= tend[idx]
        over = (attempts[idx] >= max_steps) & ~finished & ~failed[idx]
        for i in idx[over]:
            failures.append(
                BatchFailure(int(i), float(t[i]), f"{max_steps} steps attempted without finishing")
            )
            failed[i] = True
        n_reported = len(failures)
        active = idx[~finished & ~failed[idx]]

    failures.sort(key=lambda f: f.index)
    return BatchODEResult(
        y=y_out, n_out=n_out, failures=failures, nsteps=nsteps, nfailed=nfailed, nfevals=nfev
    )


def _dense(y, yn, K, h, theta):
    """DOPRI5 continuous extension at ``theta`` (one value per row) inside each row's step."""
    r2 = yn - y
    r3 = h[:, None] * K[0] - r2
    r4 = r2 - h[:, None] * K[6] - r3
    r5 = h[:, None] * _stage_sum(K, _D)
    th = theta[:, None]
    return y + th * (r2 + (1 - th) * (r3 + th * (r4 + (1 - th) * r5)))
