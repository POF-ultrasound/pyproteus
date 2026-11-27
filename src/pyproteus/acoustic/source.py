"""k-Wave transducer source of PROTEUS.

Ports of ``preprocess_transmit.m`` and ``define_source_transducer.m``.

Source rows (``source.p``, ``source.ux``, ``source.uy``) are in MATLAB
``find(mask)`` order (column-major) and unscaled, as MATLAB passes them to
``kspaceFirstOrder3DC``; :func:`pyproteus.backends.kwave.kspace_first_order`
takes them as they are (it applies k-Wave's additive source scaling for the
``cpp`` backend and the F -> C row mapping for the ``python`` backend).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike

from pyproteus.backends.kwave import filter_time_series
from pyproteus.numerics import ifft_symmetric
from pyproteus.transducer.geometry import get_element_geometry

from . import _struct
from .sensor import _find, define_sensor_transducer
from .signals import resample_signal, sinc_interpolation

__all__ = [
    "SEQ_PULSES",
    "Source",
    "define_source_transducer",
    "preprocess_transmit",
    "pulsing_sequence",
]

#: ``Transmit.SeqPulse`` values: ``'even'`` silences the odd-numbered (1-based)
#: elements, ``'odd'`` the even-numbered ones, ``'minus'`` inverts the pulse;
#: the others (``'all'``, ``'plus'``, ``'pulse'``, and ``'full'`` of
#: ``main_pressure_field.m``) transmit on every element unchanged.
SEQ_PULSES = ("odd", "even", "all", "plus", "minus", "pulse", "full")

_SEQUENCES = {
    "Amplitude modulation": ("odd", "even", "all"),
    "Pulse inversion": ("plus", "minus"),
    "Standard": ("pulse",),
    "Amplitude modulation with pulse inversion": ("odd", "even", "minus"),
}


@dataclasses.dataclass
class Source:
    """MATLAB k-Wave ``source`` struct as PROTEUS builds it (additive sources).

    Either ``p_mask`` + ``p`` (pressure, monopole) or ``u_mask`` + ``ux``
    (+ ``uy`` for a curved array) (velocity, dipole). Signals are
    ``(nnz(mask), M)``, rows in ``find(mask)`` order.
    """

    p_mask: np.ndarray | None = None
    p: np.ndarray | None = None
    u_mask: np.ndarray | None = None
    ux: np.ndarray | None = None
    uy: np.ndarray | None = None


def pulsing_sequence(pulsing_scheme: str) -> tuple[str, ...]:
    """``Transmit.SeqPulse`` values of an ``Acquisition.PulsingScheme`` (``main_RF.m``).

    Raises ``ValueError`` for an unknown scheme (MATLAB leaves ``sequence``
    undefined and fails later).
    """
    try:
        return _SEQUENCES[pulsing_scheme]
    except KeyError:
        raise ValueError(
            f"unknown Acquisition.PulsingScheme {pulsing_scheme!r}; expected one of "
            f"{tuple(_SEQUENCES)}"
        ) from None


def preprocess_transmit(transmit: Any, medium: Any, kgrid: Any) -> Any:
    """Port of ``preprocess_transmit.m``: filter and resample the transmit signals.

    ``Transmit.PressureSignal`` and ``Transmit.VoltageSignal`` (sampled at
    ``Transmit.SamplingRate``) are low-pass filtered with k-Wave
    ``filterTimeSeries`` (zero phase, ``PPW = 2``, transition width
    ``0.1/(fs*kgrid.dt)``, maximum frequency from ``kgrid.k_max`` and
    ``Medium.SpeedOfSoundMinimum``), then sinc-resampled to ``1/kgrid.dt``
    (:func:`~pyproteus.acoustic.resample_signal`). Returns a copy of
    ``transmit`` with both signals (``1 × N`` rows, as MATLAB) and ``SamplingRate = 1/kgrid.dt``.

    ``kgrid`` needs ``dt`` and ``k_max``
    (:class:`pyproteus.backends.kwave.KGrid`).
    """
    fs = _struct.scalar(_struct.get(transmit, "SamplingRate"))
    dt = float(_struct.get(kgrid, "dt"))
    k_max = float(_struct.get(kgrid, "k_max"))
    c_min = _struct.scalar(_struct.get(medium, "SpeedOfSoundMinimum"))
    tw = 0.1 / (fs * dt)
    out = _struct.copy(transmit)
    for name in ("PressureSignal", "VoltageSignal"):
        x = np.asarray(_struct.get(transmit, name), dtype=float).reshape(-1)
        filt = filter_time_series(
            1 / fs, k_max, c_min, x, zero_phase=True, transition_width=tw, ppw=2
        )
        _struct.set_field(out, name, resample_signal(filt, fs, 1 / dt, False).reshape(1, -1))
    _struct.set_field(out, "SamplingRate", 1 / dt)
    return out


def _source_weights_from_sensor(transducer: Any, grid: Any) -> tuple[sp.csr_matrix, np.ndarray]:
    """``recompute_source_weights``: transposed sensor weights and 0-based mask indices."""
    sensor, weights = define_sensor_transducer(transducer, grid)
    return weights.T.tocsr(), _find(sensor.mask)


def _shifted(transducer: Any, axis: int, shift: float) -> Any:
    try:
        T = _struct.copy(transducer)
    except TypeError:
        import copy

        T = copy.copy(transducer)
    ip = np.array(_struct.get(transducer, "integration_points"), dtype=float)
    ip[..., axis] = ip[..., axis] - shift
    _struct.set_field(T, "integration_points", ip)
    return T


def _merge_masks(
    wx: sp.spmatrix, idx_x: np.ndarray, wy: sp.spmatrix, idx_y: np.ndarray
) -> tuple[sp.csr_matrix, sp.csr_matrix, np.ndarray]:
    """``merge_masks``: two ``Nmask × Npoints`` weight matrices on the union of their masks."""
    idx = np.union1d(idx_x, idx_y)

    def onto(w: sp.spmatrix, rows_idx: np.ndarray) -> sp.csr_matrix:
        coo = w.tocoo()
        rows = np.searchsorted(idx, rows_idx)[coo.row]
        return sp.csr_matrix((coo.data, (rows, coo.col)), shape=(idx.size, w.shape[1]))

    return onto(wx, idx_x), onto(wy, idx_y), idx


def _surface_normals(transducer: Any) -> tuple[np.ndarray, np.ndarray]:
    """Axial and lateral components of the outward normal of a curved array, per point."""
    el = get_element_geometry(transducer)
    ip = np.asarray(_struct.get(transducer, "integration_points"), dtype=float)
    x = ip[..., 0].reshape(-1, order="F")
    y = ip[..., 1].reshape(-1, order="F")
    phi = np.arctan2(y, x - float(np.asarray(el.center).reshape(-1)[0]))
    return np.cos(phi), np.sin(phi)


def _flat(a: ArrayLike) -> np.ndarray:
    return np.asarray(a, dtype=float).reshape(-1, order="F")


def define_source_transducer(
    transducer: Any,
    transmit: Any,
    medium: Any,
    grid: Any,
    source_weights: sp.spmatrix | np.ndarray,
    source_mask_idx: ArrayLike,
) -> Source:
    """Port of ``define_source_transducer.m``: the transducer as a k-Wave source.

    ``source_weights`` (``nnz(mask) × N_points``, the transpose of
    :func:`~pyproteus.acoustic.define_sensor_transducer`'s weights) spreads the
    ``N_points = N_el*N_int`` integration-point signals over the source mask
    ``source_mask_idx`` (0-based column-major, ascending). Each point emits
    ``Transmit.PressureSignal`` delayed by its lens delay plus the element's
    ``Transmit.Delays`` (exact spectral shift over ``M + ceil(max(delay)/dt)``
    samples), times the element apodization (``Transmit.Apodization``, the
    point apodization and ``Transmit.SeqPulse``, see :data:`SEQ_PULSES`) and
    the integration weight.

    ``Transducer.SourceType``:

    * ``'monopole'``: pressure source ``p_mask``/``p``; the pulse is sinc
      interpolated half a time step later (k-Wave staggered time grid).
    * ``'dipole'``: velocity source ``u_mask``/``ux`` = pressure / (rho c);
      the weights are recomputed for points shifted by ``-dx/2`` (staggered
      velocity grid), and for a curved array ``uy`` with points shifted by
      ``-dy/2``, both on the union of the two masks; a curved array's velocity
      follows its surface normal.
    * absent: velocity source with the given weights, **no** staggered-grid
      shift (QUIRKS.md Q-150).

    Divergences (D-15, docs/PARITY.md): the delay is applied on half spectra
    (``rfft``/``irfft``, same result); an unknown ``SeqPulse`` raises
    ``ValueError`` (MATLAB transmits unchanged).
    """
    p = np.asarray(_struct.get(transmit, "PressureSignal"), dtype=float).reshape(-1)
    M = p.size
    dt = float(grid.dt)
    curved = _struct.config_is(transducer, "Curved")
    source_type = _struct.get(transducer, "SourceType")
    W = sp.csr_matrix(source_weights)
    mask_idx = np.asarray(source_mask_idx, dtype=np.intp).reshape(-1)
    if curved:
        nx, ny = _surface_normals(transducer)

    Wy = W
    if source_type == "dipole":
        W, mask_idx = _source_weights_from_sensor(_shifted(transducer, 0, grid.dx / 2), grid)
        if curved:
            Wy, idx_y = _source_weights_from_sensor(_shifted(transducer, 1, grid.dy / 2), grid)
            W, Wy, mask_idx = _merge_masks(W, mask_idx, Wy, idx_y)

    if source_type == "monopole":
        t = np.arange(M) * dt
        p = sinc_interpolation(t, p, t + dt / 2)

    delays = np.asarray(_struct.get(transducer, "integration_transmit_delays"), dtype=float)
    apod = np.asarray(_struct.get(transducer, "integration_transmit_apodization"), dtype=float)
    weights = _flat(_struct.get(transducer, "integration_weights"))
    n_el = delays.shape[0]
    delays = delays.reshape(n_el, -1) + _flat(_struct.get(transmit, "Delays"))[:, None]
    apod = apod.reshape(n_el, -1) * _flat(_struct.get(transmit, "Apodization"))[:, None]

    seq = _struct.get(transmit, "SeqPulse")
    if seq not in SEQ_PULSES:
        raise ValueError(f"Transmit.SeqPulse must be one of {SEQ_PULSES}, got {seq!r}")
    if seq == "even":
        apod[0::2, :] = 0
    elif seq == "odd":
        apod[1::2, :] = 0
    elif seq == "minus":
        apod = -apod
    delays = _flat(delays)
    apod = _flat(apod)

    Z = _struct.scalar(_struct.get(medium, "SpeedOfSound")) * _struct.scalar(
        _struct.get(medium, "Density")
    )
    N = M + int(np.ceil(np.max(delays) / dt))

    # Exact delay on the half spectrum. MATLAB builds the full symmetric
    # frequency axis (Nyquist bin negative for even N) and takes
    # ifft(...,'symmetric'), which reads the first N//2 + 1 bins only.
    # (The sign of the Nyquist frequency does not matter: its bin is real
    # after the 'symmetric' inverse, which ignores its imaginary part.)
    f = np.arange(N // 2 + 1) / (N * dt)
    if N % 2 == 0:
        f[-1] = -f[-1]
    spectrum = np.fft.rfft(p[None, :] * apod[:, None], n=N, axis=1)
    spectrum *= np.exp(-2j * np.pi * delays[:, None] * f[None, :])
    # zero-padded to N: ifft_symmetric reads only the first N//2 + 1 bins
    pressure_source = ifft_symmetric(spectrum, n=N, axis=1)

    shape = (grid.Nx, grid.Ny, grid.Nz)
    mask = np.zeros(shape, dtype=bool)
    if mask_idx.size:
        mask[np.unravel_index(mask_idx, shape, order="F")] = True
    src = pressure_source * weights[:, None]
    if source_type is None or source_type == "dipole":
        if curved:
            return Source(
                u_mask=mask,
                ux=np.asarray(W @ (src * nx[:, None])) / Z,
                uy=np.asarray(Wy @ (src * ny[:, None])) / Z,
            )
        return Source(u_mask=mask, ux=np.asarray(W @ src) / Z)
    return Source(p_mask=mask, p=np.asarray(W @ src))
