"""Linear propagation of microbubble mass sources in a homogeneous medium.

Ports of ``calc_scatter_attenuated.m`` and ``run_simulation_homogeneous.m``.

Physics (as PROTEUS): a monopole with mass source ``m(t)`` [kg/s] radiates
``p(r, t) = (dm/dt * g_r)(t) / (4 pi r)`` where ``g_r`` is the material
transfer function of a power-law medium (Szabo 2004, ch. 4; Kelly et al.
2008, JASA 124(5)):

``W(r, f) = exp(-r alpha(f) - 2 pi i r f / c(f))``,
``alpha(f) = a' |f_MHz|^b`` [Np/m], ``a' = a ln(10) 100 / 20``,

with the dispersive phase speed ``c(f)`` from the Kramers–Kronig relation for
power-law attenuation (:func:`power_law_kramers_kronig`, reference frequency
0) or ``c(f) = c0`` with ``alpha_mode = 'no_dispersion'``. The time
derivative is spectral, on the source's own length; the propagation uses an
FFT of length ``2*Nt`` (domain extension against wrap-around) and keeps the
first ``Nt`` samples.

Divergences decided by the developer (D-23, ``docs/PARITY.md``):

- ``0 < b <= 1`` with dispersion, where PROTEUS's reference frequency 0 is
  singular (NaN field): the reference frequency is the medium's
  ``reference_frequency`` (the transmit centre frequency, ``c(f0) = c0``).
  PROTEUS's formula is kept for every other ``b``.
- arrivals can never wrap around: when the farthest source–sensor distance
  exceeds ``Nt*dt*c0`` (impossible with PROTEUS's own ``Nt``), the FFT is
  lengthened to at least ``Nt + 1.1*r_max/(c0*dt)`` samples instead of ``2*Nt``.

Design (D-15, ``docs/PARITY.md``): both stages use half spectra
(``rfft``/``irfft``, ``Nt + 1`` bins instead of ``2*Nt``); the sources are
summed in the frequency domain, so there is one inverse FFT per sensor instead
of one per sensor-source pair; the work is done in float64 whatever the input
dtype (the output takes the dtype of the mass source); memory is bounded by
blocks of sensors × sources. Array functions take an array module ``xp``
(NumPy, or CuPy for ``DATA_CAST = 'gpuArray-*'``).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from pyproteus.numerics import matlab_round

from ._struct import get as _get
from ._struct import isfield as _isfield
from ._struct import scalar as _scalar
from .grid import optimize_grid_size

__all__ = [
    "average_medium",
    "calc_scatter_attenuated",
    "db2neper",
    "gridded_distance",
    "power_law_kramers_kronig",
    "propagate_homogeneous",
    "run_simulation_homogeneous",
]

#: Default bound on the elements of the largest intermediate array (as PROTEUS).
MAX_ARRAY_ELEMENTS = 2**26
#: Default bound on the output elements kept on the GPU (as PROTEUS).
MAX_GPU_ELEMENTS = 2**30


# ---------------------------------------------------------------------------
# Array backend
# ---------------------------------------------------------------------------


def array_module(data_cast: str | None) -> Any:
    """NumPy, or CuPy when ``data_cast`` starts with ``'gpuArray'`` (MATLAB ``DATA_CAST``)."""
    if isinstance(data_cast, str) and data_cast.startswith("gpuArray"):
        try:
            import cupy
        except ImportError as exc:  # MATLAB errors without a GPU as well
            raise RuntimeError(
                f"DATA_CAST {data_cast!r} needs CuPy and a CUDA GPU; use 'single' or 'double'"
            ) from exc
        return cupy
    return np


def to_numpy(x: Any) -> np.ndarray:
    """MATLAB ``gather``: a NumPy array from a NumPy or CuPy array."""
    get = getattr(x, "get", None)
    return get() if callable(get) and not isinstance(x, np.ndarray) else np.asarray(x)


def _irfft(X: Any, n: int, xp: Any, overwrite: bool = False) -> Any:
    """``irfft`` along the last axis, using only the real part of DC and Nyquist.

    That is MATLAB's ``real(ifft(...))`` of a conjugate-symmetric extension
    (and ``ifft(..., 'symmetric')``); NumPy does this implicitly, cuFFT does not
    promise it, so the imaginary parts are dropped explicitly (in place on ``X``
    with ``overwrite=True``).
    """
    if not overwrite:
        X = X.copy()
    X[..., 0] = X[..., 0].real
    if n % 2 == 0:
        X[..., n // 2] = X[..., n // 2].real
    return xp.fft.irfft(X, n=n, axis=-1)


# ---------------------------------------------------------------------------
# Attenuation and dispersion
# ---------------------------------------------------------------------------


def db2neper(alpha: ArrayLike, y: float = 1.0) -> np.ndarray:
    """Power-law prefactor in dB/(MHz^y cm) → Np/((rad/s)^y m).

    ``1 dB = 1/(20 log10 e) Np``, ``1/cm = 100/m`` and ``f_MHz = omega 1e-6/(2 pi)``,
    so the factor is ``100 (1e-6/(2 pi))^y / (20 log10 e)``.
    """
    return (
        100.0
        * np.asarray(alpha, dtype=float)
        * (1e-6 / (2 * math.pi)) ** y
        / (20.0 * math.log10(math.e))
    )


def power_law_kramers_kronig(
    omega: ArrayLike, omega0: float, c0: float, a0: float, y: float, xp: Any = np
) -> Any:
    """Phase speed ``c(omega)`` of a medium with attenuation ``a0 omega^y`` [Np/m].

    Kramers–Kronig relation for power-law attenuation (Szabo 1994, JASA 96(1)
    491; Waters et al. 2005, JASA 118(5)):

    - ``0 < y < 3``, ``y != 1``:
      ``1/c = 1/c0 + a0 tan(pi y/2) (omega^(y-1) - omega0^(y-1))``,
    - ``y == 1``: ``1/c = 1/c0 - (2/pi) a0 ln(omega/omega0)``,
    - otherwise the relation does not hold and ``c = c0``.

    ``c(omega0) = c0``. ``a0`` is in Np/((rad/s)^y m) (:func:`db2neper`).
    Evaluated with IEEE arithmetic: PROTEUS uses ``omega0 = 0``, which is
    singular for ``y <= 1`` (``omega0^(y-1)`` or ``ln(omega/0)`` infinite, speed
    ±0, field NaN); the propagation uses a finite reference there (D-23).
    """
    w = xp.asarray(omega, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if not 0 < y < 3:
            return c0 * xp.ones_like(w)
        if y == 1:
            return 1.0 / (1.0 / c0 - 2.0 * a0 * xp.log(w / omega0) / math.pi)
        w0_term = np.float64(omega0) ** np.float64(y - 1)
        return 1.0 / (1.0 / c0 + a0 * math.tan(math.pi * y / 2) * (w ** (y - 1) - w0_term))


class _Medium:
    """Averaged medium: ``c`` [m/s], ``a`` [dB/(MHz^b cm)], ``b``, dispersion, reference."""

    def __init__(self, medium: Any) -> None:
        self.c0 = _scalar(_get(medium, "c"))
        self.a = _scalar(_get(medium, "a"))
        self.b = _scalar(_get(medium, "b"))
        mode = _get(medium, "alpha_mode") if _isfield(medium, "alpha_mode") else None
        self.no_dispersion = isinstance(mode, str) and mode == "no_dispersion"
        f_ref = _get(medium, "reference_frequency")
        self.f_ref = None if f_ref is None else _scalar(f_ref)

    def reference_omega(self) -> float:
        """Kramers–Kronig reference [rad/s]: 0 as PROTEUS, finite for ``b <= 1`` (D-23)."""
        if not 0 < self.b <= 1:
            return 0.0
        if self.f_ref is None or not self.f_ref > 0:
            raise ValueError(
                f"alpha_power = {self.b:g} <= 1 with dispersion needs a finite Kramers-Kronig "
                "reference frequency (PROTEUS's 0 is singular, D-23): pass the transmit centre "
                "frequency as reference_frequency, or use alpha_mode='no_dispersion'"
            )
        return 2 * math.pi * self.f_ref

    def wavenumber(self, f: Any, xp: Any) -> Any:
        """Complex ``K(f) = alpha(f) + 2 pi i f / c(f)``, so that ``W = exp(-r K)``.

        ``f/c(f)`` is taken as 0 at ``f = 0`` (its limit; with a finite reference
        frequency and ``b <= 1`` the speed itself tends to 0 there).
        """
        a_np = self.a * math.log(10) * 100 / 20  # dB/(MHz^b cm) -> Np/(MHz^b m)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            alpha = a_np * xp.abs(f / 1e6) ** self.b
            if self.no_dispersion:
                phase = f / self.c0
            else:
                c = power_law_kramers_kronig(
                    2 * math.pi * f,
                    self.reference_omega(),
                    self.c0,
                    float(db2neper(self.a, self.b)),
                    self.b,
                    xp,
                )
                phase = xp.where(f == 0, 0.0, f / c)
            return alpha + 2j * math.pi * phase


def _fft_length(nt: int, r_max: float, c0: float, dt: float) -> int:
    """Propagation FFT length: ``2*Nt`` as PROTEUS, longer if an arrival could wrap (D-23).

    A direct arrival at ``r/c0`` lasting ``Nt`` samples wraps into the window when
    ``r/c0 > Nt*dt``. Then the length is the 5-smooth size from
    ``Nt + ceil(1.1*r_max/(c0*dt))``. The 10 % is an empirical margin, not a
    bound: components slower than ``c0`` (``2 < b < 3``, and low frequencies
    with ``b <= 1`` and a finite reference) can arrive later, but they are
    strongly attenuated or carry little energy (the source spectrum is ∝ f);
    measured in-window leakage ≤ 2e-11 × peak for ``b`` 0.5–2.5 at 2.3 and 10
    × ``Nt*dt*c0`` (unit test ``test_far_arrivals_never_wrap``).
    """
    if not r_max > nt * dt * c0:
        return 2 * nt
    n = nt + math.ceil(1.1 * r_max / (c0 * dt))
    return optimize_grid_size(n, [0, max(10, int(matlab_round(0.05 * n)))], 5)


# ---------------------------------------------------------------------------
# Spectral kernel
# ---------------------------------------------------------------------------


def _source_spectra(mass_source: Any, fs: float, nt: int, L: int, xp: Any) -> Any:
    """Half spectrum (FFT length ``L``, normally ``2*Nt``) of the derivative of each source row.

    MATLAB: ``MS_dot = real(ifft(2i*pi*f.*fft(m,[],2),[],2))`` on the source's
    own length ``N`` (Nyquist bin of an even ``N`` contributes nothing), then
    zero-padded or cropped to ``Nt``, then ``fft(MS_dot, 2*Nt, 2)``.
    """
    m = xp.asarray(mass_source, dtype=xp.float64)
    n = m.shape[1]
    f = xp.arange(n // 2 + 1, dtype=xp.float64) / n * fs
    ms_dot = _irfft(2j * math.pi * f * xp.fft.rfft(m, axis=1), n, xp)
    if n >= nt:
        ms_dot = ms_dot[:, :nt]
    else:
        ms_dot = xp.concatenate([ms_dot, xp.zeros((m.shape[0], nt - n))], axis=1)
    return xp.fft.rfft(ms_dot, n=L, axis=1)


def _frequencies(fs: float, L: int, xp: Any) -> Any:
    """Non-negative frequencies of the length-``L`` propagation FFT."""
    return xp.arange(L // 2 + 1, dtype=xp.float64) / L * fs


def _green(r: Any, K: Any, xp: Any) -> Any:
    """``exp(-r K) / (4 pi r)`` for distances ``r`` (any shape) → shape ``r.shape + (F,)``.

    ``r == 0`` gives 0: a source cannot sense itself.
    """
    r = xp.asarray(r, dtype=xp.float64)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):  # NaN K stays NaN
        G = (-r)[..., None] * K
        xp.exp(G, out=G)
        g = xp.where(r == 0, 0.0, 1.0 / (4 * math.pi * r))
        G *= g[..., None]
    return G


def calc_scatter_attenuated(
    mass_source: ArrayLike, medium: Any, kgrid: Any, r: ArrayLike
) -> np.ndarray:
    """Port of ``calc_scatter_attenuated.m``: pressure of each source at each sensor.

    - ``mass_source``: ``Nsrc × N`` (one source per row; a 1-D array is one source).
    - ``medium``: averaged medium with ``c`` [m/s], ``a`` [dB/(MHz^b cm)], ``b``,
      optional ``alpha_mode`` (``'no_dispersion'``: ``c(f) = c0``) and
      ``reference_frequency`` [Hz] (needed only for ``0 < b <= 1`` with
      dispersion, D-23; PROTEUS gives NaN there).
    - ``kgrid``: ``dt`` [s] and ``Nt`` (output length).
    - ``r``: ``Nsensor × Nsrc`` distances [m] (a 1-D array: ``Nsensor × 1``).

    Returns ``Nsensor × Nt × Nsrc`` (MATLAB's shape), dtype of ``mass_source``
    (float32 or float64). Zero distance gives zero pressure. Arrivals later
    than ``Nt*dt`` are cropped and never wrap around (D-23; PROTEUS wraps
    arrivals later than ``2*Nt*dt``).
    """
    ms = np.asarray(mass_source)
    if ms.ndim == 1:
        ms = ms.reshape(1, -1)
    out_dtype = np.float32 if ms.dtype == np.float32 else np.float64
    rr = np.asarray(r, dtype=float)
    if rr.ndim == 1:
        rr = rr.reshape(-1, 1)
    if rr.shape[1] != ms.shape[0]:
        raise ValueError(f"r has {rr.shape[1]} columns for {ms.shape[0]} sources")
    fs = 1.0 / _scalar(_get(kgrid, "dt"))
    nt = int(_scalar(_get(kgrid, "Nt")))
    med = _Medium(medium)
    L = _fft_length(nt, float(np.max(rr, initial=0.0)), med.c0, 1.0 / fs)
    S = _source_spectra(ms, fs, nt, L, np)  # Nsrc × F
    K = med.wavenumber(_frequencies(fs, L, np), np)
    G = _green(rr, K, np)  # Nsensor × Nsrc × F
    G *= S[None, :, :]
    p = _irfft(G, L, np, overwrite=True)[:, :, :nt]  # Nsensor × Nsrc × Nt
    return np.ascontiguousarray(p.transpose(0, 2, 1)).astype(out_dtype, copy=False)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def average_medium(medium: Any) -> dict[str, Any]:
    """``average_medium`` of ``run_simulation_homogeneous.m``: mean of each property.

    Reads k-Wave medium fields ``density``, ``sound_speed``, ``alpha_coeff``,
    ``alpha_power`` (arrays or scalars) and keeps ``alpha_mode``. Returns
    ``{'rho', 'c', 'a', 'b'[, 'alpha_mode']}``.
    """
    out: dict[str, Any] = {
        "rho": float(np.mean(np.asarray(_get(medium, "density"), dtype=float))),
        "c": float(np.mean(np.asarray(_get(medium, "sound_speed"), dtype=float))),
        "a": float(np.mean(np.asarray(_get(medium, "alpha_coeff"), dtype=float))),
        "b": float(np.mean(np.asarray(_get(medium, "alpha_power"), dtype=float))),
    }
    if _isfield(medium, "alpha_mode"):
        out["alpha_mode"] = _get(medium, "alpha_mode")
    return out


def gridded_distance(d: ArrayLike, dx: float) -> tuple[np.ndarray, np.ndarray]:
    """``gridded_distance`` of ``run_simulation_homogeneous.m``: snap ``d`` to a grid.

    The grid has step ``dx`` and passes through ``mean(d)``; returns the grid
    ``d_grid`` (1-D, MATLAB column) and **0-based** indices ``i_grid`` with
    ``d_grid[i_grid[n]]`` the grid point nearest to ``d[n]`` (MATLAB ``round``:
    halves away from zero).
    """
    d = np.asarray(d, dtype=float).reshape(-1)
    mean = float(np.mean(d))
    i = matlab_round((d - mean) / dx).astype(np.int64)
    lo = int(i.min())
    d_grid = np.arange(lo, int(i.max()) + 1, dtype=float) * dx + mean
    return d_grid, i - lo


def _blocks(n: int, size: int) -> list[slice]:
    return [slice(i, min(i + size, n)) for i in range(0, n, size)]


def propagate_homogeneous(
    mass_source: ArrayLike,
    source_points: ArrayLike,
    sensor_points: ArrayLike,
    *,
    dt: float,
    nt: int,
    medium: Any,
    gridded: bool = False,
    max_array_elements: int = MAX_ARRAY_ELEMENTS,
    xp: Any = np,
    out: Any = None,
    dtype: Any = np.float64,
) -> Any:
    """Total pressure at each sensor from all sources (``Nsensor × Nt``).

    The array-level core of :func:`run_simulation_homogeneous`.

    - ``mass_source``: ``Nsrc × N``; ``source_points`` ``Nsrc × 3``;
      ``sensor_points`` ``Nsensor × 3`` [m].
    - ``medium``: averaged medium (``c``, ``a``, ``b``, ``alpha_mode``,
      ``reference_frequency``; see :func:`calc_scatter_attenuated`).
    - ``gridded``: per source, snap the source–sensor distances to a grid of
      step ``dt*c`` through their mean (:func:`gridded_distance`), compute the
      field on that grid and copy it to the sensors (PROTEUS's fast mode for many
      sensors); sensors at the source position stay silent.
    - ``max_array_elements``: bound on the complex elements of the largest
      intermediate array (``sensors × sources × (Nt+1)`` per block); results do
      not depend on it beyond rounding.
    - ``out``: optional ``Nsensor × Nt`` array to accumulate into (e.g. a NumPy
      array while computing with CuPy); otherwise a new ``xp`` array of ``dtype``.
    - ``dtype``: output dtype (float32 for PROTEUS's single precision). The
      kernel works in float64; results are rounded to ``dtype`` when written
      (the gridded path accumulates the sources in ``dtype``, as MATLAB).

    Sources are summed in the frequency domain (one inverse FFT per sensor);
    their spectra are computed per batch, so memory beyond ``out`` and the
    inputs is bounded by ``max_array_elements``.
    """
    src = np.asarray(source_points, dtype=float).reshape(-1, 3)
    sen = np.asarray(sensor_points, dtype=float).reshape(-1, 3)
    n_src, n_sen = src.shape[0], sen.shape[0]
    fs = 1.0 / float(dt)
    nt = int(nt)
    med = _Medium(medium)
    c0 = med.c0
    r_max = max(
        (float(np.max(np.linalg.norm(sen - src[m], axis=1))) for m in range(n_src)),
        default=0.0,
    )
    # gridded distances can exceed r_max by dt*c0/2: covered by the margin (or
    # identical to MATLAB at 2*Nt)
    L = _fft_length(nt, r_max, c0, float(dt))
    F = L // 2 + 1
    K = med.wavenumber(_frequencies(fs, L, xp), xp)
    ms = xp.asarray(mass_source).reshape(n_src, -1)
    if out is None:
        out = xp.zeros((n_sen, nt), dtype=dtype)
    on_host = isinstance(out, np.ndarray) and xp is not np

    def add(rows: slice | np.ndarray, p: Any) -> None:
        out[rows] += (to_numpy(p) if on_host else p).astype(out.dtype, copy=False)

    if gridded:
        rows_per_chunk = max(1, max_array_elements // (2 * nt))
        for m in range(n_src):
            S = _source_spectra(ms[m : m + 1], fs, nt, L, xp)[0]
            d0 = np.linalg.norm(sen - src[m], axis=1)
            d_grid, i_grid = gridded_distance(d0, dt * c0)
            p = xp.empty((d_grid.size, nt), dtype=out.dtype)
            for gblk in _blocks(d_grid.size, max(1, max_array_elements // (4 * F))):
                G = _green(xp.asarray(d_grid[gblk]), K, xp)
                G *= S
                p[gblk] = _irfft(G, L, xp, overwrite=True)[:, :nt]
            silent = d0 == 0
            for blk in _blocks(n_sen, rows_per_chunk):
                pc = p[xp.asarray(i_grid[blk])]
                if silent[blk].any():
                    pc[xp.asarray(silent[blk])] = 0
                add(blk, pc)
        return out

    budget = max(1, max_array_elements // 4)
    n_chunk = max(1, min(n_sen, budget // F))
    n_batch = max(1, min(n_src, budget // (F * n_chunk)))
    sblocks, mblocks = _blocks(n_sen, n_chunk), _blocks(n_src, n_batch)
    # spectra of all sources once if they fit the budget, else per batch
    S_all = _source_spectra(ms, fs, nt, L, xp) if n_src * F <= budget else None
    for sblk in sblocks:
        P = xp.zeros((sblk.stop - sblk.start, F), dtype=xp.complex128)
        for mblk in mblocks:
            S = S_all[mblk] if S_all is not None else _source_spectra(ms[mblk], fs, nt, L, xp)
            r = np.linalg.norm(sen[sblk, None, :] - src[None, mblk, :], axis=2)
            G = _green(xp.asarray(r), K, xp)  # chunk × batch × F
            P += xp.einsum("smf,mf->sf", G, S)
        add(sblk, _irfft(P, L, xp, overwrite=True)[:, :nt])
    return out


def _param(run_param: Any, name: str, default: Any) -> Any:
    return _get(run_param, name, default) if run_param is not None else default


def run_simulation_homogeneous(
    run_param: Any,
    kgrid: Any,
    medium: Any,
    source: Any,
    sensor: Any,
    *,
    reference_frequency: float | None = None,
) -> dict[str, Any]:
    """Port of ``run_simulation_homogeneous.m``: sensor data of bubbles in a homogeneous medium.

    - ``run_param``: ``gridded`` (bool), ``DATA_CAST`` (``'gpuArray-*'`` uses
      CuPy on device ``DEVICE_NUM``), optional ``MaxArrayElements`` (default
      2^26) and ``MaxGpuElements`` (2^30: larger outputs are accumulated on the
      host).
    - ``kgrid``: ``dt``, ``Nt``. ``medium``: k-Wave medium (arrays are averaged,
      :func:`average_medium`).
    - ``source``: ``points`` ``Nsrc × 3``, ``mass_source`` ``Nsrc × N``.
    - ``sensor``: ``points`` ``Nsensor × 3``.
    - ``reference_frequency`` [Hz]: Kramers–Kronig reference frequency for
      ``0 < alpha_power <= 1`` with dispersion (the transmit centre frequency,
      D-23); required there, unused otherwise (PROTEUS: NaN field).

    Returns ``{'p': Nsensor × Nt}`` in the dtype of ``source.mass_source``
    (NumPy, or CuPy when on the GPU and small enough).
    """
    avg = average_medium(medium)
    if reference_frequency is not None:
        avg["reference_frequency"] = float(reference_frequency)
    ms = _get(source, "mass_source")
    dtype = np.float32 if getattr(ms, "dtype", None) == np.float32 else np.float64
    nt = int(_scalar(_get(kgrid, "Nt")))
    dt = _scalar(_get(kgrid, "dt"))
    sensor_points = np.asarray(_get(sensor, "points"), dtype=float).reshape(-1, 3)
    xp = array_module(_param(run_param, "DATA_CAST", None))
    out = None
    if xp is not np:
        xp.cuda.Device(int(_param(run_param, "DEVICE_NUM", 0))).use()
        if sensor_points.shape[0] * nt > int(_param(run_param, "MaxGpuElements", MAX_GPU_ELEMENTS)):
            out = np.zeros((sensor_points.shape[0], nt), dtype=dtype)
    p = propagate_homogeneous(
        ms,
        _get(source, "points"),
        sensor_points,
        dt=dt,
        nt=nt,
        medium=avg,
        gridded=bool(_param(run_param, "gridded", False)),
        max_array_elements=int(_param(run_param, "MaxArrayElements", MAX_ARRAY_ELEMENTS)),
        xp=xp,
        out=out,
        dtype=dtype,
    )
    return {"p": p}
