"""Delay-and-sum matrices (MATLAB ``delay-and-sum/``).

Ports of ``das_transmit_distance.m``, ``compute_das_matrix.m`` and
``compute_das_matrix_general.m`` (PROTEUS v1.1.0, MIT).

Conventions (PROTEUS): ``x`` axial (depth), ``y`` lateral. The DAS matrix maps
the element RF data, stacked as a column-major ``Nt``-by-``Nelem`` matrix
(column index ``sample + Nt*element``, 0-based), to the pixels. This is also
the C-order flattening of an ``Nelem``-by-``Nt`` array, which is how
:func:`pyproteus.reconstruction.load_rf_data` returns each frame, so no
transpose is needed in Python.

Pixel order (as MATLAB): :func:`compute_das_matrix` orders pixels lateral
first (row ``iy + Ny*ix``), :func:`compute_das_matrix_general` axial first
(row ``ix + Nx*iy``).

Design (D-15, ``docs/PARITY.md``): MATLAB builds dense ``Npixels``-by-``Nelem``
intermediates (delays, indices, angles, weights) for the whole image before
calling ``sparse``. Here the matrix is built in blocks of pixels, so memory is
bounded by ``block_size*Nelem`` intermediates plus the sparse matrix itself,
and returned as a ``scipy.sparse.csr_array`` (fast ``M @ RF`` for many frames;
``cupyx.scipy.sparse`` offers the same type for a later GPU path). The
values are the same: each (pixel, element) pair maps to one column, so the
duplicate-index accumulation of MATLAB ``sparse`` never applies, and zero
weights are not stored (MATLAB ``sparse`` drops them as well).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp
from numpy.typing import ArrayLike

from pyproteus.numerics import matlab_round

__all__ = [
    "DasRecon",
    "DasTransducer",
    "DasTransmit",
    "compute_das_matrix",
    "compute_das_matrix_general",
    "das_transmit_distance",
    "transmit_apodization_weight",
]

#: Pixels per block when building a DAS matrix (bounds the dense intermediates
#: to about ``block_size*Nelem`` values each).
DEFAULT_BLOCK_SIZE = 16384


def _field(obj: Any, name: str, default: Any = None) -> Any:
    """``obj.name`` for dataclasses / settings models, ``obj[name]`` for mappings."""
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    value = getattr(obj, name, default)
    return default if value is None else value


def _required(obj: Any, name: str, owner: str) -> Any:
    """A field MATLAB reads unconditionally: ``ValueError`` if absent (MATLAB errors)."""
    value = _field(obj, name)
    if value is None:
        raise ValueError(f"{owner} has no field {name}")
    return value


def _row(a: ArrayLike) -> np.ndarray:
    """MATLAB ``a(:)'`` as a 1-D float array."""
    return np.asarray(a, dtype=np.float64).reshape(-1, order="F")


@dataclass
class DasRecon:
    """``recon`` struct of ``compute_das_matrix_general`` (MATLAB field names)."""

    x: np.ndarray
    y: np.ndarray
    c: float
    t: np.ndarray
    Fs: float
    alpha_th: float
    apodization: str = "cosine"


@dataclass
class DasTransducer:
    """``transducer`` struct of ``compute_das_matrix_general`` / ``das_transmit_distance``."""

    x_el: np.ndarray
    y_el: np.ndarray
    azimuth: np.ndarray | None = None
    center: float = -np.inf
    radius: float = np.inf
    type: str | None = None
    configuration: str | None = None


@dataclass
class DasTransmit:
    """``transmit`` struct of ``compute_das_matrix_general`` / ``das_transmit_distance``."""

    focus: float = np.inf
    steer: float = 0.0
    apod: np.ndarray | None = None
    axis_delay: float = 0.0


def das_transmit_distance(X: ArrayLike, Y: ArrayLike, transducer: Any, transmit: Any) -> np.ndarray:
    """Transmit path length of the delay-and-sum model (``das_transmit_distance.m``).

    For each point ``(X, Y)`` (axial, lateral) [m], the distance travelled by
    the transmitted wave from the moment the first element fires (``t = 0``,
    smallest transmit delay over all elements, firing or not) until the wave
    reaches the point. ``X`` and ``Y`` are broadcast; the result has their
    broadcast shape.

    ``transducer``: ``x_el``, ``y_el`` [m], ``center`` (axial centre of
    curvature, ``-Inf`` for a linear array). ``transmit``: ``focus`` [m]
    (> 0 focused, < 0 diverging, ``Inf`` plane wave, or diverging from the
    centre of curvature for a curved array) and ``steer`` [rad]. Either may be
    a dataclass from this module or a mapping with MATLAB field names.
    """
    x_el = _row(_field(transducer, "x_el"))
    y_el = _row(_field(transducer, "y_el"))
    center = float(_field(transducer, "center", -np.inf))
    F = float(_field(transmit, "focus"))
    steer = float(_field(transmit, "steer"))
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)

    if not np.isfinite(F) and np.isfinite(center):
        F = center  # curved array: diverging wave from the centre of curvature

    if np.isfinite(F):
        Fx = F * np.cos(steer)
        Fy = F * np.sin(steer)
        D_el = np.sqrt((y_el - Fy) ** 2 + (x_el - Fx) ** 2)
        D = np.sqrt((Y - Fy) ** 2 + (X - Fx) ** 2)
        if F < 0:
            return D - np.min(D_el)
        s = np.sign((Fy - Y) * np.sin(steer) + (Fx - X) * np.cos(steer))
        return np.max(D_el) - s * D
    return (
        Y * np.sin(steer) + X * np.cos(steer) - np.min(y_el * np.sin(steer) + x_el * np.cos(steer))
    )


def transmit_apodization_weight(
    X: ArrayLike, Y: ArrayLike, transducer: Any, transmit: Any
) -> np.ndarray | None:
    """Transmit apodization seen by each pixel (D-21; PROTEUS ignores it, see PARITY).

    Each pixel takes the apodization of the element its transmit ray comes
    from: the element nearest to the pixel's position across the beam for a
    plane wave (``-x sin(steer) + y cos(steer)``), or the element in the
    pixel's direction seen from the virtual source / focus (for a focused
    wave, beyond the focus the ray comes from the opposite side). Pixels
    outside the aperture take the apodization of the edge element, so a
    uniform apodization gives weight 1 everywhere (results unchanged), and
    pixels insonified only by switched-off elements get 0. Ties (a pixel
    exactly between two elements) go to the lower element index; a pixel at
    the focus itself (all rays meet) gets ``max(apod)``. A scalar ``apod``
    applies to every element, as in PROTEUS's source definition.

    Returns an array with the broadcast shape of ``X`` and ``Y``, or ``None``
    when ``transmit`` has no ``apod``.
    """
    apod = _field(transmit, "apod")
    if apod is None:
        return None
    apod = _row(apod)
    x_el = _row(_field(transducer, "x_el"))
    y_el = _row(_field(transducer, "y_el"))
    if apod.size == 1:
        apod = np.full(x_el.size, apod[0])
    if apod.size != x_el.size:
        raise ValueError(f"transmit apod has {apod.size} values for {x_el.size} elements")
    center = float(_field(transducer, "center", -np.inf))
    F = float(_field(transmit, "focus", np.inf))
    steer = float(_field(transmit, "steer", 0.0))
    X = np.asarray(X, dtype=np.float64)[..., None]
    Y = np.asarray(Y, dtype=np.float64)[..., None]
    if not np.isfinite(F) and np.isfinite(center):
        F = center
    if np.isfinite(F):
        Fx, Fy = F * np.cos(steer), F * np.sin(steer)
        phi_el = np.arctan2(y_el - Fy, x_el - Fx)
        phi_px = np.arctan2(Y - Fy, X - Fx)
        if F > 0:  # beyond the focus the ray comes from the opposite side
            beyond = ((Fy - Y) * np.sin(steer) + (Fx - X) * np.cos(steer)) < 0
            phi_px = phi_px + np.pi * beyond
        dist = np.abs(np.angle(np.exp(1j * (phi_px - phi_el))))
        at_focus = np.hypot(X[..., 0] - Fx, Y[..., 0] - Fy) <= 1e-9 * abs(F)
        if F > 0 and np.any(at_focus):
            w = apod[np.argmin(dist, axis=-1)]
            return np.where(at_focus, np.max(apod), w)
    else:
        u_el = -x_el * np.sin(steer) + y_el * np.cos(steer)
        u_px = -X * np.sin(steer) + Y * np.cos(steer)
        dist = np.abs(u_px - u_el)
    return apod[np.argmin(dist, axis=-1)]


def _assemble(rows: list, cols: list, vals: list, shape: tuple[int, int]) -> sp.csr_array:
    if rows:
        r = np.concatenate(rows)
        c = np.concatenate(cols)
        v = np.concatenate(vals)
    else:
        r = c = np.zeros(0, dtype=np.int64)
        v = np.zeros(0)
    # Each (row, column) appears once, so the COO -> CSR conversion sums
    # nothing (MATLAB ``sparse`` accumulation never applies).
    M = sp.coo_array((v, (r, c)), shape=shape).tocsr()
    M.sort_indices()
    return M


INTERPOLATIONS = ("nearest", "linear")


def _check_interpolation(interpolation: str) -> None:
    if interpolation not in INTERPOLATIONS:
        raise ValueError(f"interpolation must be one of {INTERPOLATIONS}, not {interpolation!r}")


def _block_entries(
    p: np.ndarray, s: np.ndarray, v: np.ndarray, Nt: int, interpolation: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sparse entries of one pixel block.

    ``s`` is the fractional 0-based sample position ``(t_del - t(1))*Fs`` and
    ``v`` the weight of each (pixel, element) pair (0 where apodized away).

    * ``'nearest'`` (MATLAB): one entry at ``round(s)`` (half away from zero),
      kept if ``0 <= round(s) < Nt``.
    * ``'linear'`` (D-20): weights ``v*(1-w)`` at ``floor(s)`` and ``v*w`` at
      ``floor(s)+1``, ``w = s - floor(s)``, kept if ``0 <= s <= Nt-1`` (both
      samples inside the record); zero weights are not stored.
    """
    if interpolation == "nearest":
        j = matlab_round(s)
        keep = (j >= 0) & (j < Nt) & (v != 0)
        pix, el = np.nonzero(keep)  # row-major: pixel order, then element order
        return p[pix], j[pix, el].astype(np.int64) + el * Nt, v[pix, el]
    with np.errstate(invalid="ignore"):
        keep = (s >= 0) & (s <= Nt - 1) & (v != 0)
    pix, el = np.nonzero(keep)
    sk = s[pix, el]
    i0 = np.floor(sk)
    w = sk - i0
    vk = v[pix, el]
    i0 = i0.astype(np.int64)
    rows = np.concatenate([p[pix], p[pix]])
    cols = np.concatenate([i0 + el * Nt, i0 + 1 + el * Nt])
    vals = np.concatenate([vk * (1 - w), vk * w])
    nz = vals != 0  # w = 0 (on a sample, incl. s = Nt-1) leaves one entry
    return rows[nz], cols[nz], vals[nz]


def _blocks(n: int, block_size: int):
    block_size = max(1, int(block_size))
    for start in range(0, n, block_size):
        yield start, min(n, start + block_size)


def compute_das_matrix(
    t: ArrayLike,
    y: ArrayLike,
    x: ArrayLike,
    y_el: ArrayLike,
    c: float,
    Fs: float,
    focus: float,
    *,
    interpolation: str = "nearest",
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> sp.csr_array:
    """Reference delay-and-sum matrix for a linear array (``compute_das_matrix.m``).

    Note the argument order (lateral ``y`` before axial ``x``), as in MATLAB.
    Returns the ``(Ny*Nx)``-by-``(Nt*Nelem)`` sparse matrix, pixels lateral
    first (row ``iy + Ny*ix``). Only negative and infinite ``focus`` are
    supported (``ValueError`` otherwise, as MATLAB's ``error``).

    Binary apodization: elements seen from the pixel at more than ``pi/3``
    (``atan(|y_el - Y| / X)``) are discarded. As in MATLAB, a pixel at
    ``X = 0`` keeps the element at its own lateral position (0/0 → NaN, which
    is not above the threshold) and pixels at ``X < 0`` keep every element
    (negative angle) — see ``docs/QUIRKS.md``.

    ``interpolation``: ``'nearest'`` (default, MATLAB) or ``'linear'``
    (D-20: weights split between the two samples around the delay; the
    default becomes ``'linear'`` in 1.x).
    """
    t = _row(t)
    y = _row(y)
    x = _row(x)
    y_el = _row(y_el)
    c = float(c)
    Fs = float(Fs)
    focus = float(focus)
    alpha_th = np.pi / 3
    _check_interpolation(interpolation)

    if np.isfinite(focus) and focus >= 0:
        raise ValueError("Only negative focus and infinite focus supported.")

    Nx, Ny, Nt, Nelem = x.size, y.size, t.size, y_el.size
    Npixels = Nx * Ny
    F = abs(focus)

    rows, cols, vals = [], [], []
    for p0, p1 in _blocks(Npixels, block_size):
        p = np.arange(p0, p1)
        # ndgrid(y, x) flattened column-major: lateral index fastest.
        Y = y[p % Ny][:, None]
        X = x[p // Ny][:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            if np.isfinite(F):
                t_del = (
                    np.sqrt((Y - y_el) ** 2 + X**2) / c + np.sqrt((X + F) ** 2 + Y**2) / c - F / c
                )
            else:
                t_del = np.sqrt((Y - y_el) ** 2 + X**2) / c + X / c
            s = (t_del - t[0]) * Fs  # fractional 0-based sample position
            alpha = np.arctan(np.abs(y_el - Y) / X)
        v = (~(alpha > alpha_th)).astype(np.float64)
        r, cidx, val = _block_entries(p, s, v, Nt, interpolation)
        rows.append(r)
        cols.append(cidx)
        vals.append(val)
    return _assemble(rows, cols, vals, (Npixels, Nt * Nelem))


def compute_das_matrix_general(
    recon: Any,
    transducer: Any,
    transmit: Any,
    *,
    interpolation: str = "nearest",
    proteus_compat: bool = False,
    block_size: int = DEFAULT_BLOCK_SIZE,
) -> sp.csr_array:
    """General delay-and-sum matrix (``compute_das_matrix_general.m``).

    Linear and curved arrays, plane, diverging and focused (steered) transmits.
    Returns the ``(Nx*Ny)``-by-``(Nt*Nelem)`` sparse matrix, pixels axial first
    (the image is ``(M @ rf).reshape(Ny, Nx).T``).

    ``recon``: ``x``, ``y``, ``c``, ``t``, ``Fs``, ``alpha_th``, ``apodization``
    (``'cosine'`` default, or ``'binary'``). ``transducer``: ``x_el``, ``y_el``,
    ``azimuth``, ``center``/``radius`` (default ``-Inf``/``Inf``: linear).
    ``transmit``: ``focus`` (default ``Inf``), ``steer`` (default 0),
    ``axis_delay`` (default 0). Dataclasses from this module or mappings with
    MATLAB field names. ``interpolation``: ``'nearest'`` (default, MATLAB) or
    ``'linear'`` (D-20), as in :func:`compute_das_matrix`.

    The transmit apodization ``transmit.apod`` (if present) weights each pixel
    (:func:`transmit_apodization_weight`, D-21). ``proteus_compat=True``
    ignores it, as PROTEUS does (``docs/PARITY.md``, D-21).
    """
    x = _row(_field(recon, "x"))
    y = _row(_field(recon, "y"))
    c = float(_field(recon, "c"))
    t = _row(_field(recon, "t"))
    Fs = float(_field(recon, "Fs"))
    alpha_th = float(_field(recon, "alpha_th"))
    _check_interpolation(interpolation)
    apodization = _field(recon, "apodization", "cosine")
    if apodization not in ("cosine", "binary"):
        raise ValueError(f"Unknown apodization '{apodization}'.")

    # MATLAB defaults: center/radius only if 'center' is absent.
    center = _field(transducer, "center")
    if center is None:
        center, radius = -np.inf, np.inf
    else:
        radius = _field(transducer, "radius")
    center = float(center)
    radius = float(radius)
    tr = {"x_el": _field(transducer, "x_el"), "y_el": _field(transducer, "y_el"), "center": center}
    x_el = _row(tr["x_el"])
    y_el = _row(tr["y_el"])
    azimuth = _row(_required(transducer, "azimuth", "transducer"))
    tx = {
        "focus": float(_field(transmit, "focus", np.inf)),
        "steer": float(_field(transmit, "steer", 0.0)),
    }
    axis_delay = float(_field(transmit, "axis_delay", 0.0))
    tx_apod = None if proteus_compat else _field(transmit, "apod")
    tr_apod = dict(tr, x_el=x_el, y_el=y_el)

    Nx, Ny, Nt, Nelem = x.size, y.size, t.size, x_el.size
    Npixels = Nx * Ny

    rows, cols, vals = [], [], []
    for p0, p1 in _blocks(Npixels, block_size):
        p = np.arange(p0, p1)
        # ndgrid(x, y) flattened column-major: axial index fastest.
        X = x[p % Nx][:, None]
        Y = y[p // Nx][:, None]
        d_in = das_transmit_distance(X, Y, tr, tx)
        d_back = np.sqrt((Y - y_el) ** 2 + (X - x_el) ** 2)
        t_del = d_back / c + d_in / c + axis_delay
        s = (t_del - t[0]) * Fs  # fractional 0-based sample position

        alpha = np.arctan2(Y - y_el, X - x_el) - azimuth
        if apodization == "cosine":
            v = np.cos(np.pi / 2 * alpha / alpha_th)
        else:
            v = np.ones((p.size, Nelem))
        v[np.abs(alpha) > alpha_th] = 0
        if np.isfinite(radius):
            inside = (Y**2 + (X - center) ** 2 < radius**2).reshape(-1)
            v[inside, :] = 0
        if tx_apod is not None:
            w_tx = transmit_apodization_weight(X[:, 0], Y[:, 0], tr_apod, {**tx, "apod": tx_apod})
            v = v * w_tx[:, None]

        r, cidx, val = _block_entries(p, s, v, Nt, interpolation)
        rows.append(r)
        cols.append(cidx)
        vals.append(val)
    return _assemble(rows, cols, vals, (Npixels, Nt * Nelem))
