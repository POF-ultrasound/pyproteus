"""Oracle tests: ``pyproteus.reconstruction`` vs MATLAB PROTEUS ``delay-and-sum/``.

Comparison classes (D-13, ``docs/PARITY.md``):

* transmit distances, apodization weights, time axes, TGC: float class,
  ``|a - e| <= 1e-9*|e| + atol`` with a small ``atol`` relative to the data;
* DAS sample indices (after ``round``): statistical, >= 99.9 % of the
  (pixel, element) entries equal, differences reported; the inputs include
  exact .5 ties (dyadic coordinates, ``c = Fs = 1``) where both sides must
  round half away from zero;
* RF files: exact (data read from files);
* images: linear magnitude ``10^(IMG/20)`` (1 at the peak) at the float class
  for double RF data; single RF data (the PROTEUS default) at the
  single-precision class below, because MATLAB keeps single precision
  through the TGC and ``hilbert`` while pyproteus computes in double.

MATLAB sparse matrices cannot cross the engine: they come back as
``[i, j, v] = find(M)`` and ``size(M)``.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pyproteus.config.models import Medium
from pyproteus.reconstruction import (
    DasDelayMismatchWarning,
    colon,
    compute_das_matrix,
    compute_das_matrix_general,
    das_reconstruction,
    das_reconstruction_general,
    das_transmit_distance,
    initialize_das_parameters,
    load_rf_data,
    time_gain_compensation,
)
from tests.oracle.helpers import ORACLE_SETTINGS, from_matlab, put, row, to_matlab, vec

FLOAT_REL = 1e-9
INDEX_EQUAL = 0.999
#: Image class for single-precision RF input: |a - e| <= 1e-6 (linear, peak 1);
#: measured 5.5e-8 (single rounding of the TGC/hilbert input chain).
SINGLE_IMAGE_ATOL = 1e-6


def assert_float(actual: Any, expected: Any, atol: float = 0.0, err_msg: str = "") -> None:
    a = np.asarray(actual, dtype=np.float64)
    e = np.asarray(expected, dtype=np.float64)
    assert a.shape == e.shape, f"shape {a.shape} != {e.shape} {err_msg}"
    np.testing.assert_array_equal(np.isnan(a), np.isnan(e), err_msg=err_msg)
    np.testing.assert_allclose(a, e, rtol=FLOAT_REL, atol=atol, equal_nan=True, err_msg=err_msg)


# --------------------------------------------------------------------- helpers
def matlab_sparse(eng: Any, expr: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple]:
    """``[i, j, v] = find(expr)`` (0-based i, j) and ``size(expr)``."""
    eng.eval(f"pyp_M__ = {expr}; [pyp_i__, pyp_j__, pyp_v__] = find(pyp_M__);", nargout=0)
    i = vec(eng.workspace["pyp_i__"]).astype(np.int64) - 1
    j = vec(eng.workspace["pyp_j__"]).astype(np.int64) - 1
    v = vec(eng.workspace["pyp_v__"]).astype(np.float64)
    shape = tuple(int(s) for s in vec(eng.eval("size(pyp_M__)", nargout=1)))
    eng.eval("clear pyp_M__ pyp_i__ pyp_j__ pyp_v__", nargout=0)
    return i, j, v, shape


def entry_maps(i, j, v, npix: int, nt: int, nelem: int) -> tuple[np.ndarray, np.ndarray]:
    """(pixel, element) → sample index (-1 if absent) and weight (0 if absent)."""
    el = j // nt
    assert np.all(el < nelem)
    S = np.full((npix, nelem), -1, dtype=np.int64)
    V = np.zeros((npix, nelem))
    # Each (pixel, element) pair appears at most once (no duplicates).
    assert np.unique(i * nelem + el).size == i.size
    S[i, el] = j - el * nt
    V[i, el] = v
    return S, V


def compare_das(M_py, ml, nt: int, nelem: int, label: str, require_all: bool = False) -> dict:
    i, j, v, shape = ml
    assert M_py.shape == shape, label
    coo = M_py.tocoo()
    Sp, Vp = entry_maps(coo.row, coo.col, coo.data, shape[0], nt, nelem)
    Sm, Vm = entry_maps(i, j, v, shape[0], nt, nelem)
    present = (Sp >= 0) | (Sm >= 0)
    same = (Sp == Sm) & present
    n = int(present.sum())
    frac = same.sum() / n if n else 1.0
    stats = {"entries": n, "equal": int(same.sum()), "fraction": frac}
    if n and frac < 1:
        diff = np.argwhere(present & ~same)[:10]
        stats["examples"] = [(int(p), int(e), int(Sp[p, e]), int(Sm[p, e])) for p, e in diff]
    assert frac >= (1.0 if require_all else INDEX_EQUAL), f"{label}: {stats}"
    # Weights where both sides keep the entry with the same sample:
    assert_float(Vp[same], Vm[same], atol=1e-15, err_msg=label)
    return stats


def linear_case(nelem: int, pitch: float = 0.3e-3):
    y_el = -pitch / 2 * (nelem - 1) + np.arange(nelem) * pitch
    return np.zeros(nelem), y_el, np.zeros(nelem)


def curved_case(nelem: int, radius: float, dtheta: float):
    theta = (np.arange(nelem) - (nelem - 1) / 2) * dtheta
    center = -radius
    return center + radius * np.cos(theta), radius * np.sin(theta), theta, center


# --------------------------------------------------------- das_transmit_distance
TRANSMITS = [(np.inf, 0.0), (np.inf, 0.2), (-20e-3, 0.0), (-15e-3, -0.15), (30e-3, 0.1), (0.0, 0.0)]


@pytest.mark.parametrize("focus,steer", TRANSMITS)
@pytest.mark.parametrize("geometry", ["linear9", "linear8", "curved"])
def test_das_transmit_distance(matlab_engine, geometry, focus, steer) -> None:
    eng = matlab_engine
    rng = np.random.default_rng(1)
    if geometry == "curved":
        x_el, y_el, _, center = curved_case(17, 40e-3, 0.01)
    else:
        x_el, y_el, _ = linear_case(int(geometry[-1]))
        center = -np.inf
    X = rng.uniform(-2e-3, 50e-3, size=(7, 5))
    Y = rng.uniform(-20e-3, 20e-3, size=(7, 5))
    # Points on the focal axis and at the focus (sign(0) = 0):
    if np.isfinite(focus):
        X[0, 0], Y[0, 0] = focus * np.cos(steer), focus * np.sin(steer)
    else:
        X[0, 0], Y[0, 0] = 0.0, 0.0
    put(eng, X=X, Y=Y, xel=row(x_el), yel=row(y_el))
    eng.eval(
        f"d = das_transmit_distance(X, Y, struct('x_el', xel, 'y_el', yel, 'center', {center}), "
        f"struct('focus', {focus}, 'steer', {steer}));",
        nargout=0,
    )
    expected = from_matlab(eng.workspace["d"])
    tr = {"x_el": x_el, "y_el": y_el, "center": center}
    actual = das_transmit_distance(X, Y, tr, {"focus": focus, "steer": steer})
    assert_float(actual, expected, atol=1e-18)


# ------------------------------------------------------------ compute_das_matrix
def run_compute_das_matrix(eng, t, y, x, y_el, c, Fs, focus):
    put(eng, t=row(t), y=row(y), x=row(x), yel=row(y_el))
    return matlab_sparse(eng, f"compute_das_matrix(t, y, x, yel, {c}, {Fs}, {focus})")


@pytest.mark.parametrize("focus", [np.inf, -10e-3, -3e-3])
@pytest.mark.parametrize("nelem", [8, 9])
def test_compute_das_matrix_realistic(matlab_engine, nelem, focus) -> None:
    c, Fs, Nt = 1540.0, 40e6, 600
    _, y_el, _ = linear_case(nelem)
    t = np.arange(Nt) / Fs - 0.4e-6  # starts before 0: early samples out of range
    x = np.linspace(-0.5e-3, 9e-3, 37)  # includes X < 0 and points near 0
    x[2] = 0.0
    y = np.linspace(-2.1e-3, 2.1e-3, 23)
    ml = run_compute_das_matrix(matlab_engine, t, y, x, y_el, c, Fs, focus)
    M = compute_das_matrix(t, y, x, y_el, c, Fs, focus, block_size=97)
    stats = compare_das(M, ml, Nt, nelem, f"nelem={nelem} focus={focus}")
    print(stats)


@pytest.mark.parametrize("focus", [np.inf, -2.0])
def test_compute_das_matrix_ties(matlab_engine, focus) -> None:
    """Dyadic coordinates with c = Fs = 1: many delays are exact .5 ties."""
    c, Fs, Nt = 1.0, 1.0, 9
    t = np.arange(Nt) - 1.5  # t(1) = -1.5: (t_del - t(1)) hits k + .5 exactly
    y_el = np.array([-0.75, 0.0, 0.25, 1.0])
    x = np.arange(-2, 15) * 0.125
    y = np.arange(-9, 10) * 0.125
    ml = run_compute_das_matrix(matlab_engine, t, y, x, y_el, c, Fs, focus)
    M = compute_das_matrix(t, y, x, y_el, c, Fs, focus)
    stats = compare_das(M, ml, Nt, y_el.size, f"ties focus={focus}", require_all=True)
    # The case must actually contain .5 ties and out-of-range samples:
    Y, X = np.meshgrid(y, x)  # rows: x, cols: y  -> flatten C = lateral fastest
    d = np.sqrt((Y.reshape(-1, 1) - y_el) ** 2 + X.reshape(-1, 1) ** 2)
    F = abs(focus)
    if np.isfinite(F):
        tdel = d + np.sqrt((X.reshape(-1, 1) + F) ** 2 + Y.reshape(-1, 1) ** 2) - F
    else:
        tdel = d + X.reshape(-1, 1)
    frac = (tdel - t[0]) - np.floor(tdel - t[0])
    # (plane wave: dozens; diverging: both square roots must be exact, fewer)
    assert np.sum(frac == 0.5) >= (20 if not np.isfinite(F) else 5)
    assert stats["entries"] > 0


def test_compute_das_matrix_rejects_positive_focus(matlab_engine) -> None:
    with pytest.raises(Exception, match="Only negative focus"):
        matlab_engine.eval("compute_das_matrix(0:9, 0, 1, 0, 1, 1, 5)", nargout=0)
    with pytest.raises(ValueError, match="Only negative focus"):
        compute_das_matrix(np.arange(10), [0.0], [1.0], [0.0], 1.0, 1.0, 5.0)


# ---------------------------------------------------- compute_das_matrix_general
GENERAL_CASES = [
    # (geometry, focus, steer, apodization, axis_delay)
    ("linear9", np.inf, 0.0, "binary", 0.0),
    ("linear8", np.inf, 0.15, "cosine", 0.0),
    ("linear8", -10e-3, -0.1, "cosine", 0.0),
    ("linear9", 6e-3, 0.05, "cosine", 0.0),
    ("linear9", 6e-3, 0.0, "binary", 0.1e-6),
    ("curved", np.inf, 0.0, "cosine", 0.0),
    ("curved", np.inf, 0.12, "cosine", 0.0),
    ("curved", -20e-3, 0.05, "binary", 0.0),
    ("curved", 8e-3, -0.08, "cosine", 0.0),
]


def general_inputs(geometry: str, focus: float, steer: float, apodization: str, axis_delay):
    c, Fs, Nt = 1540.0, 40e6, 700
    if geometry == "curved":
        x_el, y_el, az, center = curved_case(17, 12e-3, 0.06)
        radius, alpha_th = 12e-3, np.pi / 6
        x = np.linspace(-3e-3, 10e-3, 41)
        y = np.linspace(-7e-3, 7e-3, 29)
    else:
        x_el, y_el, az = linear_case(int(geometry[-1]))
        center, radius, alpha_th = -np.inf, np.inf, np.pi / 3
        x = np.linspace(-0.3e-3, 10e-3, 39)
        y = np.linspace(-2.5e-3, 2.5e-3, 27)
    t = np.arange(Nt) / Fs - 0.3e-6
    recon = dict(x=x, y=y, c=c, t=t, Fs=Fs, alpha_th=alpha_th, apodization=apodization)
    tr = dict(x_el=x_el, y_el=y_el, azimuth=az, center=center, radius=radius)
    tx = dict(focus=focus, steer=steer, axis_delay=axis_delay)
    return recon, tr, tx


def run_general(eng, recon, tr, tx):
    put(eng, rx=row(recon["x"]), ry=row(recon["y"]), rt=row(recon["t"]))
    put(eng, xel=row(tr["x_el"]), yel=row(tr["y_el"]), az=row(tr["azimuth"]))
    eng.eval(
        f"recon = struct('x', rx, 'y', ry, 'c', {recon['c']}, 't', rt, 'Fs', {recon['Fs']}, "
        f"'alpha_th', {recon['alpha_th']!r}, 'apodization', '{recon['apodization']}');"
        f"tr = struct('x_el', xel, 'y_el', yel, 'azimuth', az, 'center', {tr['center']}, "
        f"'radius', {tr['radius']});"
        f"tx = struct('focus', {tx['focus']}, 'steer', {tx['steer']!r}, "
        f"'axis_delay', {tx['axis_delay']!r});",
        nargout=0,
    )
    return matlab_sparse(eng, "compute_das_matrix_general(recon, tr, tx)")


@pytest.mark.parametrize("case", GENERAL_CASES, ids=lambda c: "-".join(map(str, c)))
def test_compute_das_matrix_general(matlab_engine, case) -> None:
    recon, tr, tx = general_inputs(*case)
    ml = run_general(matlab_engine, recon, tr, tx)
    M = compute_das_matrix_general(recon, tr, tx, block_size=101)
    stats = compare_das(M, ml, recon["t"].size, tr["x_el"].size, str(case))
    print(stats)


def test_compute_das_matrix_general_ties(matlab_engine) -> None:
    """Dyadic linear case, plane wave, c = Fs = 1: exact .5 ties, binary weights."""
    t = np.arange(10) - 1.5
    y_el = np.array([-0.75, 0.0, 0.25, 1.0])
    recon = dict(
        x=np.arange(-2, 15) * 0.125,
        y=np.arange(-9, 10) * 0.125,
        c=1.0,
        t=t,
        Fs=1.0,
        alpha_th=np.pi / 3,
        apodization="binary",
    )
    tr = dict(x_el=np.zeros(4), y_el=y_el, azimuth=np.zeros(4), center=-np.inf, radius=np.inf)
    tx = dict(focus=np.inf, steer=0.0, axis_delay=0.0)
    ml = run_general(matlab_engine, recon, tr, tx)
    M = compute_das_matrix_general(recon, tr, tx)
    compare_das(M, ml, t.size, 4, "general ties", require_all=True)


def test_compute_das_matrix_general_defaults(matlab_engine) -> None:
    """Missing transmit fields and missing center/radius take MATLAB's defaults."""
    recon, tr, _ = general_inputs("linear8", np.inf, 0.0, "cosine", 0.0)
    eng = matlab_engine
    run_general(eng, recon, tr, dict(focus=np.inf, steer=0.0, axis_delay=0.0))
    eng.eval(
        "tr = rmfield(tr, {'center', 'radius'}); recon = rmfield(recon, 'apodization');", nargout=0
    )
    ml = matlab_sparse(eng, "compute_das_matrix_general(recon, tr, struct())")
    recon.pop("apodization")
    tr2 = {k: tr[k] for k in ("x_el", "y_el", "azimuth")}
    M = compute_das_matrix_general(recon, tr2, {})
    compare_das(M, ml, recon["t"].size, 8, "defaults")


# ------------------------------------------------------------------ load_RF_data
SCHEMES = [
    "Standard",
    "Pulse inversion",
    "Amplitude modulation",
    "Amplitude modulation with pulse inversion",
]


def write_frames(eng, folder: Path, frames: dict[str, list[np.ndarray]], version: str) -> None:
    for name, pulses in frames.items():
        eng.workspace["pyp_rf__"] = [to_matlab(p) for p in pulses]
        flag = ", '-v7.3'" if version == "v7.3" else ""
        eng.eval(
            f"RF = reshape(pyp_rf__, 1, []); dt = 1/40e6; Frame = {{}}; "
            f"save('{folder / name}', 'RF', 'dt', 'Frame'{flag}); clear RF dt Frame pyp_rf__",
            nargout=0,
        )


@pytest.mark.parametrize("version", ["v7", "v7.3"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_load_rf_data(matlab_engine, tmp_path, version, dtype) -> None:
    rng = np.random.default_rng(3)
    # Frame numbers out of name order (Frame_10 < Frame_9 by name), 3 pulses each.
    names = ["Frame_10.mat", "Frame_2.mat", "Frame_9.mat", "Frame_1.mat"]
    frames = {n: [rng.standard_normal((5, 13)).astype(dtype) for _ in range(3)] for n in names}
    write_frames(matlab_engine, tmp_path, frames, version)
    for scheme in SCHEMES:
        matlab_engine.eval(f"pyp_out__ = load_RF_data('{tmp_path}', '{scheme}');", nargout=0)
        expected = from_matlab(matlab_engine.workspace["pyp_out__"])
        actual = load_rf_data(tmp_path, scheme)
        assert actual.dtype == expected.dtype, scheme
        assert actual.shape == expected.shape, scheme
        np.testing.assert_array_equal(actual, expected, err_msg=scheme)


def test_load_rf_data_single_frame(matlab_engine, tmp_path) -> None:
    """One frame, one pulse: MATLAB returns Nelem-by-Nt (trailing 1 dropped)."""
    write_frames(matlab_engine, tmp_path, {"Frame_1.mat": [np.ones((3, 4))]}, "v7")
    matlab_engine.eval(f"pyp_out__ = load_RF_data('{tmp_path}', 'Standard');", nargout=0)
    expected = from_matlab(matlab_engine.workspace["pyp_out__"])
    actual = load_rf_data(tmp_path, "Standard")
    np.testing.assert_array_equal(actual[:, :, 0], expected)


# ------------------------------------------------------------- time gain (TGC)
@pytest.mark.parametrize("t0,t1", [(-0.5e-6, 10e-6), (-12e-6, 10e-6), (0.0, 5e-6), (-5e-6, 5e-6)])
def test_time_gain_compensation(matlab_engine, t0, t1) -> None:
    """Including |t(1)| > t(end), where MATLAB's max picks an imaginary sqrt."""
    t = np.linspace(t0, t1, 301)
    eng = matlab_engine
    put(eng, t=row(t))
    eng.eval(
        "c = 1540; f0 = 2.5e6; att = 0.5*(f0*1e-6)^1.1;"
        "TGC = sqrt(t)/max(sqrt(t)).*10.^(att.*t.*c.*1e2./20./2); TGC(t<0) = 0;"
        "TGCr = real(TGC);",
        nargout=0,
    )
    expected = vec(eng.workspace["TGCr"])
    med = Medium(AttenuationA=0.5, AttenuationB=1.1)
    actual = time_gain_compensation(t, med, 2.5e6, 1540.0)
    assert_float(actual, expected, atol=1e-15 * np.max(np.abs(expected)))


# ------------------------------------------------------------------ colon / axes
@pytest.mark.parametrize(
    "a,d,b",
    [(0, 0.1, 0.3), (0, 1540 / 2.5e6 / 5, 8e-3), (-2.4e-3, 1540 / 2.5e6 / 5, 2.4e-3), (0, 0.25, 1)],
)
def test_colon(matlab_engine, a, d, b) -> None:
    expected = vec(matlab_engine.eval(f"{a!r}:{d!r}:{b!r}", nargout=1))
    assert_float(colon(a, d, b), expected, atol=1e-15 * max(abs(a), abs(b)))


# ------------------------------------------------------------ DAS_reconstruction
SETTINGS_M = """
Acquisition.PulsingScheme = '{scheme}';
SimulationParameters.SamplingRate = 40e6;
SimulationParameters.GridSize = 1e-4;
SimulationParameters.HybridSimulation = {hybrid};
Transmit.LateralFocus = {focus};
Transmit.CenterFrequency = 2.5e6;
Transmit.SamplingRate = 40e6;
tt = (0:79)/40e6;
Transmit.PressureSignal = sin(2*pi*2.5e6*tt).*exp(-((tt-1e-6)/0.4e-6).^2);
Transducer.Pitch = 0.3e-3;
Transducer.SamplingRate = 40e6;
Transducer.ReceiveImpulseResponse = exp(-((0:39)/40e6 - 0.4e-6).^2/(0.2e-6)^2);
Transducer.ElementHeight = 5e-3;
Transducer.ElevationFocus = {efocus};
Geometry.Domain.Xmin = 0; Geometry.Domain.Xmax = 8e-3;
Geometry.Domain.Ymin = -2.4e-3; Geometry.Domain.Ymax = 2.4e-3;
Medium.SpeedOfSound = 1540; Medium.AttenuationA = 0.5; Medium.AttenuationB = 1.1;
save('{path}', 'Acquisition', 'SimulationParameters', 'Transmit', 'Transducer', ...
    'Geometry', 'Medium');
clear Acquisition SimulationParameters Transmit Transducer Geometry Medium tt
"""


def scatterer_rf(nelem: int, nt: int, scatterers, rng) -> np.ndarray:
    """Point-scatterer echoes (plane wave) plus a little noise, Nelem-by-Nt."""
    Fs, c, pitch = 40e6, 1540.0, 0.3e-3
    y_el = -pitch / 2 * (nelem - 1) + np.arange(nelem) * pitch
    t = np.arange(nt) / Fs
    rf = 1e-3 * rng.standard_normal((nelem, nt))
    for xs, ys in scatterers:
        tau = (xs + np.hypot(xs, ys - y_el)) / c + 1.2e-6
        rf += np.sin(2 * np.pi * 2.5e6 * (t - tau[:, None])) * np.exp(
            -(((t - tau[:, None]) / 0.4e-6) ** 2)
        )
    return rf


DAS_CASES = [
    # (dtype, version, scheme, hybrid, focus, elevation focus, frames)
    (np.float64, "v7", "Standard", "true", "Inf", "20e-3", 2),
    (np.float64, "v7.3", "Pulse inversion", "false", "-6e-3", "Inf", 2),
    (np.float32, "v7", "Standard", "true", "Inf", "20e-3", 3),
]


@pytest.mark.parametrize("case", DAS_CASES, ids=lambda c: f"{np.dtype(c[0]).name}-{c[1]}-{c[2]}")
def test_das_reconstruction(matlab_engine, tmp_path, case) -> None:
    dtype, version, scheme, hybrid, focus, efocus, nframes = case
    eng = matlab_engine
    rng = np.random.default_rng(7)
    data = tmp_path / "rf"
    data.mkdir()
    settings = tmp_path / "settings.mat"
    eng.eval(
        SETTINGS_M.format(scheme=scheme, hybrid=hybrid, focus=focus, efocus=efocus, path=settings),
        nargout=0,
    )
    nelem, nt = 16, 520
    frames = {}
    for k in range(1, nframes + 1):
        sc = [(3e-3 + 1e-3 * k, -0.6e-3 * k), (6.1e-3, 1.1e-3)]
        pulses = [scatterer_rf(nelem, nt, sc, rng).astype(dtype) for _ in range(2)]
        frames[f"Frame_{k}.mat"] = pulses
    write_frames(eng, data, frames, version)

    eng.eval(
        f"[pyp_I__, pyp_x__, pyp_y__] = DAS_reconstruction('{data}', '{settings}');", nargout=0
    )
    IMG_m = from_matlab(eng.workspace["pyp_I__"])
    x_m = vec(eng.workspace["pyp_x__"])
    y_m = vec(eng.workspace["pyp_y__"])
    eng.eval("clear pyp_I__ pyp_x__ pyp_y__", nargout=0)
    if IMG_m.ndim == 2:
        IMG_m = IMG_m[:, :, None]

    IMG, x, y = das_reconstruction(data, settings)
    assert_float(x, x_m, atol=1e-15)
    assert_float(y, y_m, atol=1e-15)
    assert IMG.shape == IMG_m.shape
    lin, lin_m = 10 ** (IMG / 20), 10 ** (IMG_m / 20)
    err = np.max(np.abs(lin - lin_m))
    print(f"{case}: max |linear image difference| = {err:.3g}")
    if dtype == np.float64:
        np.testing.assert_allclose(lin, lin_m, rtol=FLOAT_REL, atol=1e-12)
    else:
        assert err <= SINGLE_IMAGE_ATOL
    # Peaks at the same pixels:
    for f in range(IMG.shape[2]):
        assert np.argmax(IMG[:, :, f]) == np.argmax(IMG_m[:, :, f])


# ------------------------------------------------------- randomised (Hypothesis)
@st.composite
def general_case(draw):
    curved = draw(st.booleans())
    nelem = draw(st.integers(1, 12))
    c = draw(st.floats(1400, 1600))
    Fs = draw(st.sampled_from([20e6, 31.25e6, 40e6, 62.5e6]))
    nx, ny = draw(st.integers(1, 13)), draw(st.integers(1, 11))
    t0 = draw(st.floats(-2e-6, 1e-6))
    steer = draw(st.floats(-0.3, 0.3))
    focus = draw(st.sampled_from([np.inf, -25e-3, -5e-3, 4e-3, 15e-3]))
    apod = draw(st.sampled_from(["cosine", "binary"]))
    if curved:
        radius = draw(st.floats(8e-3, 60e-3))
        x_el, y_el, az, center = curved_case(nelem, radius, draw(st.floats(0.005, 0.05)))
        alpha_th = np.pi / 6
    else:
        x_el, y_el, az = linear_case(nelem, draw(st.floats(0.1e-3, 0.5e-3)))
        center, radius, alpha_th = -np.inf, np.inf, np.pi / 3
    x = np.sort(
        np.array(draw(st.lists(st.floats(-3e-3, 15e-3), min_size=nx, max_size=nx))), kind="stable"
    )
    y = np.sort(
        np.array(draw(st.lists(st.floats(-8e-3, 8e-3), min_size=ny, max_size=ny))), kind="stable"
    )
    # -0.0 coordinates are excluded: the sign of zero is outside the D-13
    # classes (atan2(0, -0) = pi in IEEE; see BUGS.md B-100).
    x, y = x + 0.0, y + 0.0
    recon = dict(
        x=x, y=y, c=c, t=t0 + np.arange(900) / Fs, Fs=Fs, alpha_th=alpha_th, apodization=apod
    )
    tr = dict(x_el=x_el, y_el=y_el, azimuth=az, center=center, radius=radius)
    tx = dict(focus=focus, steer=steer, axis_delay=draw(st.sampled_from([0.0, 0.05e-6])))
    return recon, tr, tx


@ORACLE_SETTINGS
@given(case=general_case())
def test_compute_das_matrix_general_random(matlab_engine, case) -> None:
    recon, tr, tx = case
    ml = run_general(matlab_engine, recon, tr, tx)
    M = compute_das_matrix_general(recon, tr, tx, block_size=7)
    compare_das(M, ml, recon["t"].size, tr["x_el"].size, "random")


def test_compute_das_matrix_large(matlab_engine) -> None:
    """Reference-size image (lambda/5 pixels, 64 elements): index statistics."""
    c, Fs, f0 = 1540.0, 62.5e6, 5e6
    pix = c / f0 / 5
    x = colon(0.0, pix, 20e-3)
    y = colon(-9.6e-3, pix, 9.6e-3)
    _, y_el, _ = linear_case(64)
    t = np.arange(2200) / Fs - 0.7e-6
    stats = {}
    for focus in (np.inf, -12e-3):
        ml = run_compute_das_matrix(matlab_engine, t, y, x, y_el, c, Fs, focus)
        M = compute_das_matrix(t, y, x, y_el, c, Fs, focus)
        stats[focus] = compare_das(M, ml, t.size, 64, f"large focus={focus}")
    print(stats)


# ------------------------------------- initialize_das_parameters / general DAS
GENERAL_SETTINGS_M = """
clear Acquisition SimulationParameters Transmit Transducer Geometry Medium
Acquisition.PulsingScheme = 'Standard';
SimulationParameters.SamplingRate = 40e6;
SimulationParameters.GridSize = 1e-4;
SimulationParameters.HybridSimulation = {hybrid};
Transmit.LateralFocus = {focus};
Transmit.Angle = {angle};
Transmit.CenterFrequency = 2.5e6;
Transmit.SamplingRate = 40e6;
tt = (0:79)/40e6;
Transmit.PressureSignal = sin(2*pi*2.5e6*tt).*exp(-((tt-1e-6)/0.4e-6).^2);
Transducer.Type = 'test';
Transducer.Configuration = '{config}';
Transducer.NumberOfElements = {nelem};
Transducer.Pitch = 0.3e-3;
Transducer.ElementWidth = 0.27e-3;
Transducer.Radius = {radius};
Transducer.SamplingRate = 40e6;
Transducer.ReceiveImpulseResponse = exp(-((0:39)/40e6 - 0.4e-6).^2/(0.2e-6)^2);
Transducer.ElementHeight = 5e-3;
Transducer.ElevationFocus = {efocus};
Geometry.Domain.Xmin = {xmin}; Geometry.Domain.Xmax = 9e-3;
Geometry.Domain.Ymin = -3.1e-3; Geometry.Domain.Ymax = 2.9e-3;
Medium.SpeedOfSound = 1540; Medium.AttenuationA = 0.5; Medium.AttenuationB = 1.1;
{extra}
"""

INIT_CASES = [
    # (config, nelem, radius, focus, angle, hybrid, efocus, xmin, extra)
    ("Linear", 16, "Inf", "Inf", "0", "true", "20e-3", "0", ""),
    ("Linear", 15, "Inf", "-8e-3", "7", "false", "Inf", "0.5e-3", ""),
    ("Curved", 17, "15e-3", "Inf", "0", "true", "20e-3", "-1e-3", ""),
    ("Curved", 16, "12e-3", "6e-3", "-5", "true", "Inf", "0", "Transmit.Apodization = rand(1,16);"),
]


def make_general_settings(eng, case) -> None:
    config, nelem, radius, focus, angle, hybrid, efocus, xmin, extra = case
    eng.eval(
        GENERAL_SETTINGS_M.format(
            config=config,
            nelem=nelem,
            radius=radius,
            focus=focus,
            angle=angle,
            hybrid=hybrid,
            efocus=efocus,
            xmin=xmin,
            extra=extra,
        ),
        nargout=0,
    )


def python_settings(eng) -> Any:
    """Settings model from the MATLAB workspace structs (via a v7 file round trip)."""
    from pyproteus.config.readers import load_settings_mat

    path = Path(eng.eval("[tempname '.mat']", nargout=1))
    eng.eval(
        f"save('{path}', 'Acquisition', 'SimulationParameters', 'Transmit', 'Transducer', "
        "'Geometry', 'Medium');",
        nargout=0,
    )
    try:
        return load_settings_mat(path)
    finally:
        path.unlink()


@pytest.mark.parametrize("case", INIT_CASES, ids=lambda c: f"{c[0]}{c[1]}-{c[3]}-{c[4]}")
def test_initialize_das_parameters(matlab_engine, case) -> None:
    eng = matlab_engine
    eng.eval("rng(0, 'twister');", nargout=0)
    make_general_settings(eng, case)
    nelem, Nt = case[1], 431
    eng.eval(
        "[R, T, X] = initialize_das_parameters(Medium, SimulationParameters, Transducer, "
        f"Transmit, Geometry, {nelem}, {Nt});",
        nargout=0,
    )
    S = python_settings(eng)
    recon, trans, tx = initialize_das_parameters(
        S.Medium, S.SimulationParameters, S.Transducer, S.Transmit, S.Geometry, nelem, Nt
    )
    for name in ("x", "y", "t"):
        expected = vec(eng.eval(f"R.{name}", nargout=1))
        assert_float(getattr(recon, name), expected, atol=1e-15 * np.max(np.abs(expected)))
    for name in ("c", "Fs", "alpha_th"):
        assert_float(getattr(recon, name), eng.eval(f"R.{name}", nargout=1))
    assert recon.apodization == eng.eval("R.apodization", nargout=1)
    for name in ("x_el", "y_el", "azimuth"):
        assert_float(getattr(trans, name), vec(eng.eval(f"T.{name}", nargout=1)), atol=1e-18)
    assert_float(trans.center, eng.eval("T.center", nargout=1), atol=1e-18)
    assert_float(trans.radius, eng.eval("T.radius", nargout=1))
    assert trans.type == eng.eval("T.type", nargout=1)
    assert trans.configuration == eng.eval("T.configuration", nargout=1)
    for name in ("focus", "steer", "axis_delay"):
        assert_float(getattr(tx, name), eng.eval(f"X.{name}", nargout=1), atol=1e-18)
    assert_float(tx.apod, vec(eng.eval("X.apod", nargout=1)))
    eng.eval("clear R T X", nargout=0)


def test_initialize_das_parameters_errors_and_warning(matlab_engine) -> None:
    eng = matlab_engine
    make_general_settings(eng, INIT_CASES[0])
    S = python_settings(eng)
    args = "Medium, SimulationParameters, Transducer, Transmit, Geometry"
    # Element count mismatch:
    with pytest.raises(Exception, match="15 elements, the transducer 16"):
        eng.eval(f"initialize_das_parameters({args}, 15, 100);", nargout=0)
    with pytest.raises(ValueError, match="15 elements, the transducer 16"):
        initialize_das_parameters(
            S.Medium, S.SimulationParameters, S.Transducer, S.Transmit, S.Geometry, 15, 100
        )
    # Unsupported configuration:
    eng.eval("Transducer.Configuration = 'RCA';", nargout=0)
    with pytest.raises(Exception, match="not supported"):
        eng.eval(f"initialize_das_parameters({args}, 16, 100);", nargout=0)
    S.Transducer.Configuration = "RCA"
    with pytest.raises(ValueError, match="not supported"):
        initialize_das_parameters(
            S.Medium, S.SimulationParameters, S.Transducer, S.Transmit, S.Geometry, 16, 100
        )
    # Delays that do not match the model (plane wave declared, 1 us ramp stored):
    for ramp, warns in ((1e-6, True), (1e-9, False)):
        eng.eval(
            f"Transducer.Configuration = 'Linear'; Transmit.Delays = linspace(0, {ramp}, 16);"
            f"lastwarn(''); initialize_das_parameters({args}, 16, 100); [~, id] = lastwarn;",
            nargout=0,
        )
        assert (eng.eval("id", nargout=1) == "initialize_das_parameters:delays") == warns
        S = python_settings(eng)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            initialize_das_parameters(
                S.Medium, S.SimulationParameters, S.Transducer, S.Transmit, S.Geometry, 16, 100
            )
        assert any(issubclass(w.category, DasDelayMismatchWarning) for w in rec) == warns


GENERAL_DAS_CASES = [
    (np.float64, "v7", INIT_CASES[0]),
    (np.float64, "v7.3", INIT_CASES[1]),
    (np.float64, "v7", INIT_CASES[2]),
    (np.float32, "v7", INIT_CASES[3]),
]


@pytest.mark.parametrize(
    "dtype,version,case",
    GENERAL_DAS_CASES,
    ids=lambda c: c[0] + str(c[1]) if isinstance(c, tuple) else str(c),
)
def test_das_reconstruction_general(matlab_engine, tmp_path, dtype, version, case) -> None:
    eng = matlab_engine
    eng.eval("rng(0, 'twister');", nargout=0)
    make_general_settings(eng, case)
    settings = tmp_path / "settings.mat"
    eng.eval(
        f"save('{settings}', 'Acquisition', 'SimulationParameters', 'Transmit', "
        "'Transducer', 'Geometry', 'Medium');",
        nargout=0,
    )
    rng = np.random.default_rng(11)
    nelem, nt = case[1], 560
    data = tmp_path / "rf"
    data.mkdir()
    frames = {}
    for k in (1, 2):
        rf = 1e-3 * rng.standard_normal((nelem, nt))
        tau = 1.5e-6 + (np.arange(nelem) - nelem / 2) ** 2 * 2e-9 + 3e-6 * k
        tt = np.arange(nt) / 40e6
        rf += np.sin(2 * np.pi * 2.5e6 * (tt - tau[:, None])) * np.exp(
            -(((tt - tau[:, None]) / 0.4e-6) ** 2)
        )
        frames[f"Frame_{k}.mat"] = [rf.astype(dtype)]
    write_frames(eng, data, frames, version)
    eng.eval(
        f"[pyp_I__, pyp_x__, pyp_y__] = DAS_reconstruction_general('{data}', '{settings}');",
        nargout=0,
    )
    IMG_m = from_matlab(eng.workspace["pyp_I__"])
    x_m, y_m = vec(eng.workspace["pyp_x__"]), vec(eng.workspace["pyp_y__"])
    eng.eval("clear pyp_I__ pyp_x__ pyp_y__", nargout=0)

    IMG, x, y = das_reconstruction_general(data, settings)
    assert_float(x, x_m, atol=1e-15)
    assert_float(y, y_m, atol=1e-15)
    assert IMG.shape == IMG_m.shape
    lin, lin_m = 10 ** (IMG / 20), 10 ** (IMG_m / 20)
    err = np.max(np.abs(lin - lin_m))
    print(f"{case[:2]} {np.dtype(dtype).name}: max |linear image difference| = {err:.3g}")
    if dtype == np.float64:
        np.testing.assert_allclose(lin, lin_m, rtol=FLOAT_REL, atol=1e-12)
    else:
        assert err <= SINGLE_IMAGE_ATOL
    for f in range(IMG.shape[2]):
        assert np.argmax(IMG[:, :, f]) == np.argmax(IMG_m[:, :, f])
