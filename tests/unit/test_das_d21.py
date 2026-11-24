"""D-21 defaults (domain-spanning axes, transmit apodization) and D-15 extras
(empty-frame warning, DAS matrix cache). No MATLAB needed."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pyproteus.reconstruction import (
    DasEmptyFrameWarning,
    clear_das_matrix_cache,
    compute_das_matrix_general,
    das_reconstruction,
    das_reconstruction_general,
    log_compress,
    pixel_axes,
    transmit_apodization_weight,
)
from pyproteus.reconstruction import beamform as bf

C, FS = 1540.0, 40e6


def linear(n, pitch=0.3e-3):
    return np.zeros(n), (np.arange(n) - (n - 1) / 2) * pitch


# ------------------------------------------------------------------ P5 axes
def test_pixel_axes_default_and_compat() -> None:
    dom = {"Xmin": 0.5e-3, "Xmax": 5e-3, "Ymin": -3.1e-3, "Ymax": 2.0e-3}
    x, y = pixel_axes(dom, 0.2e-3, x_start_compat=0.0, proteus_compat=False)
    assert x[0] == 0.5e-3 and y[0] == -3.1e-3 and y[-1] <= 2.0e-3
    xc, yc = pixel_axes(dom, 0.2e-3, x_start_compat=0.0, proteus_compat=True)
    assert xc[0] == 0.0 and yc[0] == -2.55e-3
    # Symmetric domain, Xmin = 0: identical.
    dom = {"Xmin": 0.0, "Xmax": 5e-3, "Ymin": -2.4e-3, "Ymax": 2.4e-3}
    a = pixel_axes(dom, 0.123e-3, x_start_compat=0.0, proteus_compat=False)
    b = pixel_axes(dom, 0.123e-3, x_start_compat=0.0, proteus_compat=True)
    for u, v in zip(a, b, strict=True):
        np.testing.assert_array_equal(u, v)


# ------------------------------------------------------- P6 transmit weight
def test_transmit_weight_uniform_is_one() -> None:
    x_el, y_el = linear(8)
    X, Y = np.meshgrid(np.linspace(0, 10e-3, 7), np.linspace(-6e-3, 6e-3, 9), indexing="ij")
    for focus, steer in [(np.inf, 0.0), (np.inf, 0.3), (-10e-3, 0.1), (8e-3, -0.2)]:
        w = transmit_apodization_weight(
            X, Y, {"x_el": x_el, "y_el": y_el}, {"focus": focus, "steer": steer, "apod": np.ones(8)}
        )
        np.testing.assert_array_equal(w, 1.0)
    assert (
        transmit_apodization_weight(X, Y, {"x_el": x_el, "y_el": y_el}, {"focus": np.inf}) is None
    )


def test_transmit_weight_geometry() -> None:
    x_el, y_el = linear(8)  # y_el = -1.05 ... 1.05 mm
    apod = (y_el > 0).astype(float)  # right half fires
    tr = {"x_el": x_el, "y_el": y_el}
    X = np.full(4, 5e-3)
    Y = np.array([-1e-3, -0.2e-3, 0.2e-3, 1e-3])
    # Plane wave, no steering: straight down from the elements.
    w = transmit_apodization_weight(X, Y, tr, {"focus": np.inf, "steer": 0.0, "apod": apod})
    np.testing.assert_array_equal(w, [0, 0, 1, 1])
    # Steered plane wave: rays come from y0 = Y - X*tan(steer).
    st = np.arctan(0.2)  # shift of 1 mm at 5 mm depth
    w = transmit_apodization_weight(X, Y, tr, {"focus": np.inf, "steer": st, "apod": apod})
    # origins y0 = Y - 1 mm = -2, -1.2, -0.8, 0.1 mm:
    w = transmit_apodization_weight(
        X, Y + np.array([0, 0, 0, 0.1e-3]), tr, {"focus": np.inf, "steer": st, "apod": apod}
    )
    np.testing.assert_array_equal(w, [0, 0, 0, 1])
    # origins -0.5, 0.3, 0.7, 1.5 mm:
    w = transmit_apodization_weight(X, Y + 1.5e-3, tr, {"focus": np.inf, "steer": st, "apod": apod})
    np.testing.assert_array_equal(w, [0, 1, 1, 1])
    # Focused at 5 mm on axis: beyond the focus the sides swap.
    w = transmit_apodization_weight(
        np.full(2, 8e-3), np.array([-1e-3, 1e-3]), tr, {"focus": 5e-3, "steer": 0.0, "apod": apod}
    )
    np.testing.assert_array_equal(w, [1, 0])
    w = transmit_apodization_weight(
        np.full(2, 2e-3),
        np.array([-0.3e-3, 0.3e-3]),
        tr,
        {"focus": 5e-3, "steer": 0.0, "apod": apod},
    )
    np.testing.assert_array_equal(w, [0, 1])
    # Diverging from a virtual source behind the array: same side.
    w = transmit_apodization_weight(
        np.full(2, 8e-3), np.array([-3e-3, 3e-3]), tr, {"focus": -5e-3, "steer": 0.0, "apod": apod}
    )
    np.testing.assert_array_equal(w, [0, 1])


def test_general_matrix_applies_and_compat_ignores_apod() -> None:
    x_el, y_el = linear(16)
    recon = dict(x=np.linspace(1e-3, 8e-3, 15), y=np.linspace(-3e-3, 3e-3, 13), c=C,
                 t=np.arange(600) / FS, Fs=FS, alpha_th=np.pi / 3)  # fmt: skip
    tr = dict(x_el=x_el, y_el=y_el, azimuth=np.zeros(16))
    apod = (y_el > 0).astype(float)
    on = compute_das_matrix_general(recon, tr, {"apod": apod})
    off = compute_das_matrix_general(recon, tr, {"apod": apod}, proteus_compat=True)
    none = compute_das_matrix_general(recon, tr, {})
    assert (off != none).nnz == 0
    assert (compute_das_matrix_general(recon, tr, {"apod": np.ones(16)}) != none).nnz == 0
    Nx = recon["x"].size
    used = np.diff(on.indptr) > 0
    Ypix = recon["y"][np.arange(on.shape[0]) // Nx]
    assert not used[Ypix < 0].any()  # left half not insonified
    assert used[Ypix > 0.3e-3].all() == np.diff(none.indptr)[Ypix > 0.3e-3].astype(bool).all()


# --------------------------------------------- end-to-end defaults (no MATLAB)
def test_general_default_images_scatterer_in_asymmetric_domain(tmp_path) -> None:
    from tests.unit.test_reconstruction import general_settings

    data, S = general_settings(tmp_path, config="Linear")
    S.Geometry.Domain.Ymin = -1.0e-3  # asymmetric; scatterer at y = 0.9 mm
    S.Geometry.Domain.Ymax = 2.6e-3
    IMG, x, y = das_reconstruction_general(data, S)
    assert y[0] == -1.0e-3 and y[-1] <= 2.6e-3
    ix, iy = np.unravel_index(np.argmax(IMG[:, :, 0]), IMG.shape[:2])
    pix = C / 2.5e6 / 4
    assert abs(x[ix] - 5e-3) < 3 * pix and abs(y[iy] - 0.9e-3) < 2 * pix
    _, _, yc = das_reconstruction_general(data, S, proteus_compat=True)
    assert yc[0] == -1.8e-3


def test_general_default_masks_switched_off_half(tmp_path) -> None:
    from tests.unit.test_reconstruction import general_settings

    data, S = general_settings(tmp_path, config="Linear")
    S.Transmit.Apodization = (np.arange(16) >= 8).astype(float).reshape(1, -1)  # right half
    IMG, _, y = das_reconstruction_general(data, S)
    IMGc, _, _ = das_reconstruction_general(data, S, proteus_compat=True)
    left = y < -0.6e-3
    assert np.isneginf(IMG[:, left, 0]).all()  # masked: no contribution
    assert np.isfinite(IMGc[:, left, 0]).any()


# ------------------------------------------------------- P7 / P8 (D-15)
def test_log_compress_warns_on_empty_frame() -> None:
    img = np.ones((3, 4, 2), dtype=complex)
    img[:, :, 1] = 0
    with pytest.warns(DasEmptyFrameWarning, match=r"\[1\]"):
        out = log_compress(img)
    assert np.isnan(out[:, :, 1]).all() and (out[:, :, 0] == 0).all()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        log_compress(np.ones((2, 2, 1)))


def test_das_matrix_cache(tmp_path, monkeypatch) -> None:
    from tests.unit.test_reconstruction import write_case

    data, settings, _ = write_case(tmp_path)
    clear_das_matrix_cache()
    calls = []
    real = bf.compute_das_matrix
    monkeypatch.setattr(bf, "compute_das_matrix", lambda *a, **k: calls.append(1) or real(*a, **k))
    a = das_reconstruction(data, settings)[0]
    b = das_reconstruction(data, settings)[0]
    assert len(calls) == 1
    np.testing.assert_array_equal(a, b)
    das_reconstruction(data, settings, interpolation="linear")
    das_reconstruction(data, settings, cache=False)
    assert len(calls) == 3
    clear_das_matrix_cache()
    das_reconstruction(data, settings)
    assert len(calls) == 4


@pytest.fixture(autouse=True)
def _fresh_cache():
    clear_das_matrix_cache()
    yield
    clear_das_matrix_cache()


def test_transmit_weight_curved_plane_wave_uses_center() -> None:
    R = 15e-3
    th = np.linspace(-0.4, 0.4, 9)
    tr = {"x_el": -R + R * np.cos(th), "y_el": R * np.sin(th), "center": -R}
    apod = np.zeros(9)
    apod[0], apod[-1] = 1, 2  # only the edges fire
    # Rays from the centre of curvature: far lateral pixels see the edge elements.
    w = transmit_apodization_weight(
        np.full(3, 10e-3), np.array([-12e-3, 0.0, 12e-3]), tr,
        {"focus": np.inf, "steer": 0.0, "apod": apod},
    )  # fmt: skip
    np.testing.assert_array_equal(w, [1, 0, 2])


def test_transmit_weight_scalar_apod_and_focus_pixel() -> None:
    x_el, y_el = linear(8)
    tr = {"x_el": x_el, "y_el": y_el}
    w = transmit_apodization_weight(np.ones(3), np.zeros(3), tr, {"focus": 5e-3, "apod": 0.5})
    np.testing.assert_array_equal(w, 0.5)
    apod = (y_el > 0).astype(float)
    w = transmit_apodization_weight(
        np.array([5e-3]), np.array([0.0]), tr, {"focus": 5e-3, "steer": 0.0, "apod": apod}
    )
    np.testing.assert_array_equal(w, [1.0])  # all rays meet at the focus


def test_general_cache_key_covers_apod_and_domain(tmp_path, monkeypatch) -> None:
    from pyproteus.reconstruction import general as gen
    from tests.unit.test_reconstruction import general_settings

    data, S = general_settings(tmp_path, config="Linear")
    calls = []
    real = gen.compute_das_matrix_general
    monkeypatch.setattr(
        gen, "compute_das_matrix_general", lambda *a, **k: calls.append(1) or real(*a, **k)
    )
    a = das_reconstruction_general(data, S)[0]
    das_reconstruction_general(data, S)
    assert len(calls) == 1
    S.Transmit.Apodization = (np.arange(16) >= 8).astype(float).reshape(1, -1)
    b = das_reconstruction_general(data, S)[0]
    assert len(calls) == 2 and not np.array_equal(a, b, equal_nan=True)
    S.Geometry.Domain.Ymax = 2.0e-3
    das_reconstruction_general(data, S)
    assert len(calls) == 3
