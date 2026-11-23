"""Unit tests for ``pyproteus.reconstruction`` (no MATLAB needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.io

from pyproteus.config.models import Medium
from pyproteus.numerics import matlab_round
from pyproteus.reconstruction import (
    beamform_frames,
    compute_das_matrix,
    compute_das_matrix_general,
    das_reconstruction,
    das_reconstruction_general,
    das_transmit_distance,
    frame_files,
    initialize_das_parameters,
    load_rf_data,
    log_compress,
    time_gain_compensation,
)

C, FS = 1540.0, 40e6


def linear(nelem, pitch=0.3e-3):
    return -pitch / 2 * (nelem - 1) + np.arange(nelem) * pitch


# ---------------------------------------------------- das_transmit_distance
def test_transmit_distance_at_elements_is_c_times_delay() -> None:
    """At each element, d_in equals c times the textbook transmit delay."""
    y_el = linear(9)
    x_el = np.zeros(9)
    tr = {"x_el": x_el, "y_el": y_el, "center": -np.inf}
    # Plane wave, steered: delay = (y sin + x cos - min) / c.
    st = 0.2
    d = das_transmit_distance(x_el, y_el, tr, {"focus": np.inf, "steer": st})
    np.testing.assert_allclose(d, y_el * np.sin(st) - np.min(y_el * np.sin(st)), atol=1e-18)
    assert np.min(d) == 0
    # Focused: the element farthest from the focus fires first (d = 0 there).
    F = 20e-3
    d = das_transmit_distance(x_el, y_el, tr, {"focus": F, "steer": 0.0})
    dist = np.hypot(y_el, F)
    np.testing.assert_allclose(d, dist.max() - dist, atol=1e-18)
    # Diverging: the element closest to the virtual source fires first.
    d = das_transmit_distance(x_el, y_el, tr, {"focus": -F, "steer": 0.0})
    np.testing.assert_allclose(d, dist - dist.min(), atol=1e-18)


def test_transmit_distance_curved_plane_wave_is_diverging_from_center() -> None:
    R = 40e-3
    th = np.linspace(-0.3, 0.3, 11)
    tr = {"x_el": -R + R * np.cos(th), "y_el": R * np.sin(th), "center": -R}
    d_inf = das_transmit_distance(0.01, 0.002, tr, {"focus": np.inf, "steer": 0.0})
    d_c = das_transmit_distance(0.01, 0.002, tr, {"focus": -R, "steer": 0.0})
    assert d_inf == d_c


def test_transmit_distance_broadcasts() -> None:
    tr = {"x_el": np.zeros(3), "y_el": linear(3), "center": -np.inf}
    d = das_transmit_distance(np.ones((4, 1)), np.zeros((1, 5)), tr, {"focus": -5e-3, "steer": 0})
    assert d.shape == (4, 5)


# ------------------------------------------------------ compute_das_matrix
def brute_force_reference(t, y, x, y_el, c, Fs, focus) -> np.ndarray:
    """Independent loop implementation of compute_das_matrix (dense)."""
    Nt, Nelem = len(t), len(y_el)
    M = np.zeros((len(x) * len(y), Nt * Nelem))
    F = abs(focus)
    for ix, X in enumerate(x):
        for iy, Y in enumerate(y):
            p = iy + len(y) * ix
            for e, ye in enumerate(y_el):
                d_back = np.hypot(Y - ye, X)
                d_in = X if not np.isfinite(F) else np.hypot(X + F, Y) - F
                j = int(matlab_round((d_back / c + d_in / c - t[0]) * Fs))
                with np.errstate(divide="ignore", invalid="ignore"):
                    alpha = np.arctan(abs(ye - Y) / X)
                if 0 <= j < Nt and not alpha > np.pi / 3:
                    M[p, j + e * Nt] = 1
    return M


@pytest.mark.parametrize("focus", [np.inf, -4e-3])
def test_compute_das_matrix_matches_brute_force(focus) -> None:
    t = np.arange(300) / FS - 0.2e-6
    x = np.array([-0.2e-3, 0.0, 0.7e-3, 2e-3, 3.1e-3])
    y = np.linspace(-1e-3, 1.2e-3, 7)
    y_el = linear(5)
    M = compute_das_matrix(t, y, x, y_el, C, FS, focus, block_size=4)
    np.testing.assert_array_equal(M.toarray(), brute_force_reference(t, y, x, y_el, C, FS, focus))


def test_compute_das_matrix_half_away_from_zero() -> None:
    """(t_del - t(1))*Fs = 0.5 and 1.5 exactly: round up (MATLAB), not to even."""
    # c = Fs = 1, element under the pixel: t_del = 2X.
    M = compute_das_matrix([0.0, 1, 2, 3], [0.0], [0.25, 0.75], [0.0], 1.0, 1.0, np.inf)
    np.testing.assert_array_equal(M.toarray(), [[0, 1, 0, 0], [0, 0, 1, 0]])


def test_compute_das_matrix_block_size_invariant() -> None:
    t = np.arange(400) / FS
    x, y = np.linspace(0, 5e-3, 13), np.linspace(-2e-3, 2e-3, 11)
    A = compute_das_matrix(t, y, x, linear(8), C, FS, -6e-3)
    B = compute_das_matrix(t, y, x, linear(8), C, FS, -6e-3, block_size=1)
    assert (A != B).nnz == 0


def test_compute_das_matrix_positive_focus_rejected() -> None:
    with pytest.raises(ValueError, match="Only negative focus"):
        compute_das_matrix(np.arange(5), [0.0], [1.0], [0.0], C, FS, 0.0)


def test_compute_das_matrix_empty_when_out_of_range() -> None:
    M = compute_das_matrix(np.arange(5) / FS + 1.0, [0.0], [1e-3], linear(4), C, FS, np.inf)
    assert M.shape == (1, 20) and M.nnz == 0


# ---------------------------------------------- compute_das_matrix_general
def general_linear(nelem=8, apod="binary", t=None):
    t = np.arange(600) / FS if t is None else t
    x, y = np.linspace(0.2e-3, 6e-3, 15), np.linspace(-2e-3, 2e-3, 9)
    recon = dict(x=x, y=y, c=C, t=t, Fs=FS, alpha_th=np.pi / 3, apodization=apod)
    tr = dict(x_el=np.zeros(nelem), y_el=linear(nelem), azimuth=np.zeros(nelem))
    return recon, tr


@pytest.mark.parametrize("nelem", [8, 9])
def test_general_plane_wave_equals_reference(nelem) -> None:
    """As in PROTEUS' DelayAndSumTest: same matrix up to the pixel order."""
    recon, tr = general_linear(nelem)
    M0 = compute_das_matrix(recon["t"], recon["y"], recon["x"], tr["y_el"], C, FS, np.inf)
    M1 = compute_das_matrix_general(recon, tr, {"focus": np.inf, "steer": 0.0})
    Nx, Ny = recon["x"].size, recon["y"].size
    r = np.arange(Nx * Ny)  # reference row r = iy + Ny*ix -> general row ix + Nx*iy
    perm = r // Ny + Nx * (r % Ny)
    assert (M1[perm] != M0).nnz == 0


def test_general_diverging_odd_equals_reference() -> None:
    """Odd element count: an element sits at the centre, so both models coincide."""
    recon, tr = general_linear(9)
    M0 = compute_das_matrix(recon["t"], recon["y"], recon["x"], tr["y_el"], C, FS, -10e-3)
    M1 = compute_das_matrix_general(recon, tr, {"focus": -10e-3, "steer": 0.0})
    Nx, Ny = recon["x"].size, recon["y"].size
    r = np.arange(Nx * Ny)
    assert (M1[r // Ny + Nx * (r % Ny)] != M0).nnz == 0


def test_general_cosine_taper() -> None:
    recon, tr = general_linear(1, apod="cosine", t=np.arange(5000) / FS)
    recon["x"] = np.array([5e-3])
    recon["y"] = np.linspace(-10e-3, 10e-3, 201)
    M = compute_das_matrix_general(recon, tr, {})
    alpha = np.arctan2(recon["y"], recon["x"][0])
    expected = np.cos(np.pi / 2 * alpha / recon["alpha_th"])
    expected[np.abs(alpha) > recon["alpha_th"]] = 0
    np.testing.assert_allclose(M.max(axis=1).toarray().ravel(), expected, atol=1e-15)


def test_general_curved_ignores_pixels_inside_circle() -> None:
    R = 10e-3
    th = np.linspace(-0.5, 0.5, 15)
    tr = dict(x_el=-R + R * np.cos(th), y_el=R * np.sin(th), azimuth=th, center=-R, radius=R)
    recon = dict(
        x=np.linspace(-5e-3, 15e-3, 21),
        y=np.linspace(-10e-3, 10e-3, 17),
        c=C,
        t=np.arange(1500) / FS,
        Fs=FS,
        alpha_th=np.pi / 6,
    )
    M = compute_das_matrix_general(recon, tr, {})
    X, Y = np.meshgrid(recon["x"], recon["y"], indexing="ij")
    inside = (Y.ravel(order="F") ** 2 + (X.ravel(order="F") + R) ** 2) < R**2
    used = np.diff(M.indptr) > 0
    assert not used[inside].any() and used[~inside].any()


def test_general_samples_stay_within_their_element() -> None:
    recon, tr = general_linear(8, t=(np.arange(61) + 199) / FS)
    M = compute_das_matrix_general(recon, tr, {"focus": np.inf, "steer": 0.17}).tocoo()
    el = M.col // 61
    s = M.col - 61 * el
    Nx = recon["x"].size
    X, Y = recon["x"][M.row % Nx], recon["y"][M.row // Nx]
    d_in = das_transmit_distance(X, Y, dict(tr, center=-np.inf), {"focus": np.inf, "steer": 0.17})
    t_del = (np.hypot(Y - tr["y_el"][el], X) + d_in) / C
    np.testing.assert_array_equal(matlab_round((t_del - recon["t"][0]) * FS), s)


def test_general_unknown_apodization() -> None:
    recon, tr = general_linear()
    recon["apodization"] = "hann"
    with pytest.raises(ValueError, match="Unknown apodization"):
        compute_das_matrix_general(recon, tr, {})


def test_general_point_scatterer_imaged_at_position() -> None:
    """Synthetic echoes from a scatterer at (axial 12 mm, lateral 1 mm), steered plane wave."""
    Fs, steer = 100e6, 0.1
    y_el = linear(32)
    tr = dict(x_el=np.zeros(32), y_el=y_el, azimuth=np.zeros(32))
    S = np.array([12e-3, 1e-3])
    t_tx = (S[1] * np.sin(steer) + S[0] * np.cos(steer) - np.min(y_el * np.sin(steer))) / C
    t_echo = t_tx + np.hypot(S[0], S[1] - y_el) / C
    t = np.arange(int(2 * t_echo.max() * Fs)) / Fs
    rf = np.exp(-(((t[None, :] - t_echo[:, None]) / 30e-9) ** 2))  # Nelem x Nt
    pix = 0.05e-3
    recon = dict(
        x=S[0] + np.arange(-20, 21) * pix,
        y=S[1] + np.arange(-20, 21) * pix,
        c=C,
        t=t,
        Fs=Fs,
        alpha_th=np.pi / 3,
    )
    M = compute_das_matrix_general(recon, tr, {"focus": np.inf, "steer": steer})
    img = (M @ rf.reshape(-1)).reshape(41, 41).T  # axial-by-lateral
    ix, iy = np.unravel_index(np.argmax(img), img.shape)
    assert abs(recon["x"][ix] - S[0]) <= 1.01 * pix and abs(recon["y"][iy] - S[1]) <= 1.01 * pix


# ---------------------------------------------------------------- RF files
def save_frame(path: Path, pulses) -> None:
    cell = np.empty((1, len(pulses)), dtype=object)
    for k, p in enumerate(pulses):
        cell[0, k] = p
    scipy.io.savemat(path, {"RF": cell, "dt": 1 / FS})


def test_frame_files_sorted_by_number(tmp_path) -> None:
    for name in ["Frame_10.mat", "Frame_9.mat", "Frame_x.mat", "Frame_01.mat", "other.mat"]:
        (tmp_path / name).write_bytes(b"")
    names = [p.name for p in frame_files(tmp_path)]
    assert names == ["Frame_01.mat", "Frame_9.mat", "Frame_10.mat", "Frame_x.mat"]


@pytest.mark.parametrize(
    "scheme,expected",
    [
        ("Standard", lambda a, b, c: a),
        ("Pulse inversion", lambda a, b, c: a + b),
        ("Amplitude modulation", lambda a, b, c: c - a - b),
        ("Amplitude modulation with pulse inversion", lambda a, b, c: c + a + b),
    ],
)
def test_load_rf_data_schemes(tmp_path, scheme, expected) -> None:
    rng = np.random.default_rng(0)
    frames = []
    for k in (2, 1):
        pulses = [rng.standard_normal((4, 6)).astype(np.float32) for _ in range(3)]
        save_frame(tmp_path / f"Frame_{k}.mat", pulses)
        frames.append((k, pulses))
    rf = load_rf_data(tmp_path, scheme)
    assert rf.shape == (4, 6, 2) and rf.dtype == np.float32
    for k, pulses in frames:
        np.testing.assert_array_equal(rf[:, :, k - 1], expected(*pulses))


def test_load_rf_data_errors(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_rf_data(tmp_path, "Standard")
    save_frame(tmp_path / "Frame_1.mat", [np.ones((3, 4))])
    with pytest.raises(ValueError, match="pulsing scheme"):
        load_rf_data(tmp_path, "Harmonic")
    save_frame(tmp_path / "Frame_2.mat", [np.ones((3, 5))])
    with pytest.raises(ValueError, match="expected 3-by-4"):
        load_rf_data(tmp_path, "Standard")


def test_load_rf_data_class_of_first_frame(tmp_path) -> None:
    save_frame(tmp_path / "Frame_1.mat", [np.ones((2, 2), dtype=np.float32)])
    save_frame(tmp_path / "Frame_2.mat", [np.full((2, 2), 1 / 3)])
    rf = load_rf_data(tmp_path, "Standard")
    assert rf.dtype == np.float32
    assert rf[0, 0, 1] == np.float32(1 / 3)


# ---------------------------------------------------------- TGC / beamform
def test_tgc_matches_formula_and_q101_fix() -> None:
    med = Medium(AttenuationA=0.5, AttenuationB=1.1)
    att = 0.5 * 2.5**1.1
    for t in (np.linspace(-1e-6, 10e-6, 23), np.linspace(-10e-6, 5e-6, 23)):
        tgc = time_gain_compensation(t, med, 2.5e6, C)
        expected = np.sqrt(np.maximum(t, 0)) / np.sqrt(t.max()) * 10 ** (att * t * C * 1e2 / 40)
        expected[t < 0] = 0
        # The second axis has |t(1)| > t(end): MATLAB would normalise by an
        # imaginary number (Q-101); pyproteus normalises by sqrt(max(t)) (D-18).
        np.testing.assert_allclose(tgc, expected, rtol=1e-13)
        assert tgc.max() > 0
    np.testing.assert_array_equal(time_gain_compensation(np.linspace(-2, 0, 5), med, 2.5e6, C), 0)


def test_beamform_frames_streaming_equals_batch() -> None:
    rng = np.random.default_rng(1)
    frames = [rng.standard_normal((4, 50)) for _ in range(5)]
    recon, tr = general_linear(4, t=np.arange(50) / FS)
    M = compute_das_matrix_general(recon, tr, {})
    tgc = np.linspace(0, 1, 50)
    a = beamform_frames(iter(frames), tgc, M, frames_per_block=2)
    b = beamform_frames(frames, tgc, M, frames_per_block=10)
    np.testing.assert_allclose(a, b, rtol=1e-14, atol=1e-14)
    assert a.shape == (M.shape[0], 5)


def test_log_compress_peak_is_zero_db() -> None:
    img = np.random.default_rng(2).standard_normal((5, 6, 3)) + 0j
    out = log_compress(img)
    np.testing.assert_allclose(out.max(axis=(0, 1)), 0, atol=1e-12)


# ------------------------------------------------------- das_reconstruction
def write_case(tmp_path: Path, nframes: int = 2) -> tuple[Path, Path, tuple]:
    tt = np.arange(80) / FS
    settings = {
        "Acquisition": {"PulsingScheme": "Standard"},
        "SimulationParameters": {"SamplingRate": FS, "GridSize": 1e-4, "HybridSimulation": True},
        "Transmit": {
            "LateralFocus": np.inf,
            "CenterFrequency": 2.5e6,
            "SamplingRate": FS,
            "PressureSignal": (
                np.sin(2 * np.pi * 2.5e6 * tt) * np.exp(-(((tt - 1e-6) / 0.4e-6) ** 2))
            ).reshape(1, -1),
        },
        "Transducer": {
            "Pitch": 0.3e-3,
            "SamplingRate": FS,
            "ReceiveImpulseResponse": np.exp(
                -((np.arange(40) / FS - 0.4e-6) ** 2) / (0.2e-6) ** 2
            ).reshape(1, -1),
            "ElementHeight": 5e-3,
            "ElevationFocus": 20e-3,
        },
        "Geometry": {"Domain": {"Xmin": 0.0, "Xmax": 8e-3, "Ymin": -2.4e-3, "Ymax": 2.4e-3}},
        "Medium": {"SpeedOfSound": C, "AttenuationA": 0.5, "AttenuationB": 1.1},
    }
    path = tmp_path / "settings.mat"
    scipy.io.savemat(path, settings)
    data = tmp_path / "rf"
    data.mkdir()
    y_el = linear(16)
    t = np.arange(520) / FS
    S = (5e-3, 0.9e-3)
    tau = (S[0] + np.hypot(S[0], S[1] - y_el)) / C + 1.2e-6
    rf = np.sin(2 * np.pi * 2.5e6 * (t - tau[:, None])) * np.exp(
        -(((t - tau[:, None]) / 0.4e-6) ** 2)
    )
    for k in range(1, nframes + 1):
        save_frame(data / f"Frame_{k}.mat", [rf.astype(np.float32)])
    return data, path, S


def test_das_reconstruction_images_scatterer(tmp_path) -> None:
    data, settings, S = write_case(tmp_path)
    IMG, x, y = das_reconstruction(data, settings)
    assert IMG.shape == (x.size, y.size, 2)
    pix = C / 2.5e6 / 5
    assert x[0] == 0 and np.isclose(x[1], pix)
    for f in range(2):
        ix, iy = np.unravel_index(np.argmax(IMG[:, :, f]), IMG.shape[:2])
        assert IMG[ix, iy, f] == 0
        # Within about a wavelength (the time-to-peak quirk shifts by one sample):
        assert abs(x[ix] - S[0]) < 4 * pix and abs(y[iy] - S[1]) < 2 * pix


# ------------------------------------------ initialize_das_parameters / general
def general_settings(tmp_path: Path, config="Curved", nelem=16, **transmit):
    data, path, _ = write_case(tmp_path)
    from pyproteus.config.readers import load_settings_mat

    S = load_settings_mat(path)
    S.Transducer.Type = "test"
    S.Transducer.Configuration = config
    S.Transducer.NumberOfElements = nelem
    S.Transducer.Radius = 15e-3
    S.Transducer.ElementWidth = 0.27e-3
    S.Transmit.Angle = transmit.get("Angle", 0.0)
    if "Delays" in transmit:
        S.Transmit.Delays = transmit["Delays"]
    return data, S


def test_initialize_das_parameters_curved(tmp_path) -> None:
    from pyproteus.transducer import get_element_geometry

    _, S = general_settings(tmp_path, Angle=10.0)
    recon, tr, tx = initialize_das_parameters(
        S.Medium, S.SimulationParameters, S.Transducer, S.Transmit, S.Geometry, 16, 300
    )
    El = get_element_geometry(S.Transducer)
    np.testing.assert_array_equal(tr.x_el, El.x)
    assert tr.center == El.center and tr.radius == 15e-3
    assert recon.alpha_th == np.pi / 6 and recon.apodization == "cosine"
    assert np.isclose(recon.x[1] - recon.x[0], C / 2.5e6 / 4)
    assert recon.t.size == 300
    np.testing.assert_allclose(tx.steer, np.deg2rad(10.0), rtol=1e-15)
    np.testing.assert_array_equal(tx.apod, np.ones(16))


def test_initialize_das_parameters_errors(tmp_path) -> None:
    _, S = general_settings(tmp_path, config="Linear")
    args = (S.Medium, S.SimulationParameters, S.Transducer, S.Transmit, S.Geometry)
    with pytest.raises(ValueError, match="15 elements, the transducer 16"):
        initialize_das_parameters(*args, 15, 100)
    S.Transducer.Configuration = "RCA"
    with pytest.raises(ValueError, match="not supported"):
        initialize_das_parameters(*args, 16, 100)


def test_initialize_das_parameters_delay_warning(tmp_path) -> None:
    import warnings

    from pyproteus.reconstruction import DasDelayMismatchWarning

    ramp = np.linspace(0, 1e-6, 16).reshape(1, -1)
    _, S = general_settings(tmp_path, config="Linear", Delays=ramp)
    args = (S.Medium, S.SimulationParameters, S.Transducer, S.Transmit, S.Geometry)
    with pytest.warns(DasDelayMismatchWarning):
        initialize_das_parameters(*args, 16, 100)
    S.Transmit.Delays = np.zeros((1, 16))  # plane wave, no steering: matches
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        initialize_das_parameters(*args, 16, 100)


def test_das_reconstruction_general_linear_images_scatterer(tmp_path) -> None:
    data, S = general_settings(tmp_path, config="Linear")
    IMG, x, y = das_reconstruction_general(data, S)
    assert IMG.shape == (x.size, y.size, 2)
    ix, iy = np.unravel_index(np.argmax(IMG[:, :, 0]), IMG.shape[:2])
    pix = C / 2.5e6 / 4
    assert abs(x[ix] - 5e-3) < 3 * pix and abs(y[iy] - 0.9e-3) < 2 * pix
