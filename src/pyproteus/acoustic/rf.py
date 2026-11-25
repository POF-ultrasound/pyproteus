"""Element RF data from the pressure sensed on the transducer.

Ports of ``compute_RF.m`` and ``compute_RF_data.m``.

Per element ``e`` with integration points ``k`` (lens delay ``tau_ek``, receive
apodization ``A_ek``, normalised quadrature weight ``w_ek``), the element
pressure is ``p_e(t) = sum_k w_ek A_ek p_ek(t - tau_ek)``, where ``p_ek`` is
the sensor data interpolated to the integration point (``sensor_weights``).
The delays are applied in the frequency domain on a signal of length
``N >= M + ceil(max(tau)/dt)`` with prime factors ≤ 5. The RF signal is
``p_e`` convolved with the receive impulse response (resampled to ``1/dt``)
times ``dt``.

Design (D-15, ``docs/PARITY.md``): the delay, apodization and weighted sum
are done in the frequency domain on half spectra, element by element in
blocks, so there is one inverse FFT per element instead of one per
integration point and no ``N_el*N_int × N`` complex array; the work is done
in float64 for both ``DATA_CAST_RF`` precisions (the output takes the
requested dtype).
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np

from pyproteus.numerics import convn, matlab_round

from ._struct import get as _get
from ._struct import scalar as _scalar
from .grid import optimize_grid_size
from .homogeneous import _irfft, array_module, to_numpy
from .signals import resample_signal

__all__ = ["compute_RF", "compute_RF_data", "rf_signal_length"]

_CASTS = {
    "single": (np.float32, False),
    "double": (np.float64, False),
    "gpuArray-single": (np.float32, True),
    "gpuArray-double": (np.float64, True),
}
#: Default bound on the elements of the largest intermediate array (as PROTEUS's
#: ``MaxArrayElements`` for the homogeneous simulation).
_MAX_ARRAY_ELEMENTS = 1 << 26


def rf_signal_length(m: int, max_delay: float, dt: float, dtype: Any = np.float64) -> int:
    """Signal length used to apply the lens delays (``compute_RF.m``).

    ``N = M + ceil(max(delays)/dt)`` (evaluated in ``dtype``, as MATLAB divides
    a ``single`` maximum delay in single precision), then the size in
    ``N + (0:max(10, round(0.05 N)))`` with the smallest largest prime factor
    (at most 5, :func:`optimize_grid_size`).
    """
    ratio = dtype(max_delay) / dtype(dt)
    n = int(m) + int(np.ceil(ratio))
    max_expansion = max(10, int(matlab_round(n * 0.05)))
    return optimize_grid_size(n, [0, max_expansion], 5)


def _sensor_pressure(sensor_data: Any) -> Any:
    p = sensor_data.get("p") if isinstance(sensor_data, dict) else getattr(sensor_data, "p", None)
    return sensor_data if p is None else p


def compute_RF(
    Transducer: Any, sensor_data: Any, sensor_weights: Any, Grid: Any, run_param: Any
) -> np.ndarray:
    """Port of ``compute_RF.m``: element RF data (``N_el × L``).

    - ``Transducer``: ``integration_points`` (``N_el × N_int × 3``, for the
      sizes), ``integration_receive_apodization`` and ``integration_receive_delays``
      (``N_el × N_int``), ``integration_weights`` (scalar or ``N_el × N_int``;
      normalised to sum 1 per element), ``ReceiveImpulseResponse`` and
      ``SamplingRate``.
    - ``sensor_data``: ``{'p': Nsensor × M}`` (or the array itself).
    - ``sensor_weights``: ``N_el*N_int × Nsensor`` (dense or SciPy sparse), rows
      in MATLAB column-major order of ``(element, point)``.
    - ``Grid``: ``dt``.
    - ``run_param``: ``DATA_CAST_RF`` (``'single'``, ``'double'``,
      ``'gpuArray-single'``, ``'gpuArray-double'``), optional ``RF_type``
      (``'pressure'``: return the element pressure ``N_el × N``, before the
      impulse response) and ``MaxArrayElements`` (default 2^26: bound on the
      complex elements of the per-block intermediates, about four live at once;
      results do not depend on it).

    Returns the RF data, ``N_el × (N + numel(IR) - 1)``, in the requested dtype.
    """
    cast = _get(run_param, "DATA_CAST_RF")
    if cast not in _CASTS:
        raise ValueError(f"DATA_CAST_RF must be one of {sorted(_CASTS)}, got {cast!r}")
    dtype, use_gpu = _CASTS[cast]
    xp = array_module(cast) if use_gpu else np
    dt = _scalar(_get(Grid, "dt"))

    n_el, n_int = np.shape(_get(Transducer, "integration_points"))[:2]
    apod = np.asarray(_get(Transducer, "integration_receive_apodization"), dtype=float)
    apod = np.broadcast_to(apod, (n_el, n_int))
    delays = np.asarray(_get(Transducer, "integration_receive_delays"), dtype=float)
    delays = np.broadcast_to(delays, (n_el, n_int))
    weights = np.asarray(_get(Transducer, "integration_weights"), dtype=float)
    if weights.size == 1:
        weights = np.full((n_el, n_int), 1.0 / n_int)
    else:
        weights = np.broadcast_to(weights / np.sum(weights, axis=-1, keepdims=True), (n_el, n_int))
    # MATLAB casts the delays to DATA_CAST_RF before using them
    delays = delays.astype(dtype).astype(float)

    p_sensor = _sensor_pressure(sensor_data)
    m = int(np.shape(p_sensor)[1])
    n = rf_signal_length(m, float(np.nanmax(delays)), dt, dtype)  # MATLAB max skips NaN

    # sensor pressure → integration points (MATLAB: sensor_weights*double(p)),
    # half spectrum on N points, rows in column-major (element, point) order
    p_sensor = to_numpy(p_sensor).astype(float, copy=False)
    # errstate: Apple Accelerate's matmul raises spurious FP flags on finite data.
    with np.errstate(all="ignore"):
        p_int = np.asarray(sensor_weights @ p_sensor)
    if dtype == np.float32:  # MATLAB: p = cast(p, dataType) before the FFT
        p_int = p_int.astype(np.float32).astype(float)
    p_int = p_int.reshape(n_int, n_el, m).transpose(1, 0, 2)  # el × point × M

    f = np.arange(n // 2 + 1, dtype=float) / (n * dt)
    coef = (weights * apod)[:, :, None]
    p = np.empty((n_el, n))
    max_el = int(_get(run_param, "MaxArrayElements", _MAX_ARRAY_ELEMENTS))
    per_el = max(1, max_el // (4 * max(1, n_int * f.size)))
    for i in range(0, n_el, per_el):
        blk = slice(i, min(i + per_el, n_el))
        P = xp.fft.rfft(xp.asarray(p_int[blk]), n=n, axis=-1)
        shift = xp.asarray(coef[blk]) * xp.exp(
            -2j * math.pi * xp.asarray(delays[blk])[:, :, None] * xp.asarray(f)
        )
        P *= shift
        p[blk] = to_numpy(_irfft(xp.sum(P, axis=1), n, xp, overwrite=True))

    if _get(run_param, "RF_type") == "pressure":
        return p.astype(dtype)

    ir = resample_signal(
        _get(Transducer, "ReceiveImpulseResponse"),
        _scalar(_get(Transducer, "SamplingRate")),
        1 / dt,
    )
    rf = convn(p.astype(dtype).astype(float), ir.reshape(1, -1)) * dt
    return rf.astype(dtype)


def compute_RF_data(
    Transducer: Any, sensor_data: Any, sensor_weights: Any, Grid: Any, run_param: Any
) -> tuple[np.ndarray, Any]:
    """Port of ``compute_RF_data.m``: :func:`compute_RF`, retried on the CPU after a GPU OOM.

    On a CuPy out-of-memory error, warns (``ResourceWarning``), switches
    ``DATA_CAST_RF`` from ``'gpuArray-single'``/``'gpuArray-double'`` to
    ``'single'``/``'double'`` and retries. Returns ``(RF, run_param)``;
    ``run_param`` is a copy when it was changed (MATLAB value semantics).
    Other errors propagate.
    """
    try:
        return compute_RF(Transducer, sensor_data, sensor_weights, Grid, run_param), run_param
    except Exception as exc:
        if not _is_gpu_oom(exc):
            raise
    warnings.warn(
        "Out of memory on GPU device: unable to allocate sufficient resources on the GPU "
        "to complete the RF computation. Switching to CPU.",
        ResourceWarning,
        stacklevel=2,
    )
    run_param = dict(run_param) if isinstance(run_param, dict) else _copy(run_param)
    cast = _get(run_param, "DATA_CAST_RF")
    fallback = {"gpuArray-double": "double", "gpuArray-single": "single"}.get(cast, cast)
    if isinstance(run_param, dict):
        run_param["DATA_CAST_RF"] = fallback
    else:
        run_param.DATA_CAST_RF = fallback
    return compute_RF(Transducer, sensor_data, sensor_weights, Grid, run_param), run_param


def _copy(struct: Any) -> Any:
    from ._struct import copy

    return copy(struct)


def _is_gpu_oom(exc: BaseException) -> bool:
    """CuPy's ``OutOfMemoryError`` (MATLAB ``parallel:gpu:array:OOM``)."""
    try:
        from cupy.cuda.memory import OutOfMemoryError
    except ImportError:
        return False
    return isinstance(exc, OutOfMemoryError)
