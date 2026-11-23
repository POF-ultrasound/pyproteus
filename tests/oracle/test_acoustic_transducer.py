"""Oracle tests: ``get_transducer_integration_delays`` and ``compute_travel_times``.

Integration points are built by hand here (random, plus the regular layout of
a linear array) so these tests do not depend on
``get_transducer_integration_points``; that function gets its own tests.
D-13 float class (≤ 1e-9 rel; delays near zero with an absolute floor).
"""

from __future__ import annotations

import zlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pyproteus import acoustic
from tests.oracle.helpers import put
from tests.oracle.test_acoustic_grid import (
    assert_float,
    build_settings,
    load_settings,
    matlab_value,
)


@pytest.fixture(scope="module")
def default_settings(matlab_engine: Any, oracle: Any, tmp_path_factory: pytest.TempPathFactory):
    tmp = tmp_path_factory.mktemp("transducer_settings")
    build_settings(matlab_engine, oracle.path, var="pyp_S0__")
    settings = load_settings(matlab_engine, tmp, var="pyp_S0__")
    return settings


def _points(n_el: int, n_int: int, seed: int, layout: str) -> np.ndarray:
    rs = np.random.RandomState(seed)
    if layout == "random":
        p = rs.uniform(-5e-3, 5e-3, size=(n_el, n_int, 3))
        p[:, :, 0] = rs.uniform(-1e-3, 1e-3, size=(n_el, n_int))
        return p
    # linear-array-like: x = 0, y per element, z symmetric (contains z = 0 and ±z ties)
    ny = max(1, n_int // 3)
    nz = n_int // ny
    y = (np.arange(n_el)[:, None] * 3e-4 + np.arange(ny)[None, :] * 1e-4) - n_el * 1.5e-4
    z = (np.arange(nz) - (nz - 1) / 2) * 2e-4
    pts = np.zeros((n_el, ny * nz, 3))
    pts[:, :, 1] = np.repeat(y, nz, axis=1)
    pts[:, :, 2] = np.tile(z, ny)[None, :]
    return pts


CONFIGS = [None, "Linear", "Curved", "RCA", "Custom", "Custom-with-delays"]
FOCI = [20e-3, -15e-3, np.inf, -np.inf, 0.0]


def _setup(eng: Any, settings: Any, config: str | None, focus: float, pts: np.ndarray):
    """Same transducer in MATLAB (``pyp_T__``) and Python."""
    T = settings.Transducer.copy_struct()
    T.ElevationFocus = float(focus)
    T.integration_points = pts
    put(eng, pyp_pts__=pts, pyp_f__=float(focus))
    eng.eval(
        "pyp_T__ = pyp_S0__.Transducer; pyp_T__.integration_points = pyp_pts__;"
        "pyp_T__.ElevationFocus = pyp_f__;",
        nargout=0,
    )
    if config is None:
        if T.isfield("Configuration"):
            T.rmfield("Configuration")
        eng.eval(
            "if isfield(pyp_T__,'Configuration'), pyp_T__ = rmfield(pyp_T__,'Configuration'); end",
            nargout=0,
        )
    else:
        name = "Custom" if config.startswith("Custom") else config
        T.Configuration = name
        eng.eval(f"pyp_T__.Configuration = '{name}';", nargout=0)
    apod = np.ones(pts.shape[:2])
    if config in ("RCA", "Custom", "Custom-with-delays"):
        T.integration_transmit_apodization = apod
        put(eng, pyp_apod__=apod)
        eng.eval("pyp_T__.integration_transmit_apodization = pyp_apod__;", nargout=0)
    if config == "Custom-with-delays":
        d = np.random.RandomState(1).uniform(0, 1e-6, pts.shape[:2])
        T.integration_transmit_delays = d
        T.integration_receive_delays = 2 * d
        put(eng, pyp_d__=d)
        eng.eval(
            "pyp_T__.integration_transmit_delays = pyp_d__;"
            "pyp_T__.integration_receive_delays = 2*pyp_d__;",
            nargout=0,
        )
    return T


@pytest.mark.parametrize("config", CONFIGS)
@pytest.mark.parametrize("focus", FOCI)
@pytest.mark.parametrize("layout", ["random", "linear"])
def test_integration_delays(
    matlab_engine: Any, default_settings: Any, tmp_path: Path, config, focus, layout
) -> None:
    eng = matlab_engine
    n_el = int(default_settings.Transducer.NumberOfElements)
    pts = _points(n_el, 9, seed=zlib.crc32(f"{config}-{layout}".encode()), layout=layout)
    T = _setup(eng, default_settings, config, focus, pts)
    ref = matlab_value(eng, "get_transducer_integration_delays(pyp_T__, pyp_S0__.Medium)", tmp_path)
    mine = acoustic.get_transducer_integration_delays(T, default_settings.Medium)
    for k in ("integration_transmit_delays", "integration_receive_delays"):
        e = ref[k]
        a = np.asarray(getattr(mine, k))
        scale = float(np.max(np.abs(e), initial=0.0))
        assert_float(a, e, atol=1e-12 * max(scale, 1e-9), err_msg=f"{k} {config} {focus}")
    # the input is not modified
    assert not (config is None and T.isfield("integration_transmit_delays"))


@pytest.mark.parametrize("config", [None, "Linear", "Curved", "RCA", "Custom"])
@pytest.mark.parametrize("focus", [20e-3, np.inf])
@pytest.mark.parametrize("n_int", [1, 12])
def test_compute_travel_times(
    matlab_engine: Any, default_settings: Any, tmp_path: Path, config, focus, n_int
) -> None:
    eng = matlab_engine
    n_el = int(default_settings.Transducer.NumberOfElements)
    pts = _points(n_el, n_int, seed=5, layout="linear" if config != "Custom" else "random")
    T = _setup(eng, default_settings, config, focus, pts)
    T = acoustic.get_transducer_integration_delays(T, default_settings.Medium)
    eng.eval(
        "pyp_T__ = get_transducer_integration_delays(pyp_T__, pyp_S0__.Medium);"
        "pyp_rp__ = struct('existing', 7);",
        nargout=0,
    )
    ref = matlab_value(
        eng,
        "compute_travel_times(pyp_rp__, pyp_S0__.Geometry, pyp_S0__.Medium, pyp_T__,"
        " pyp_S0__.Transmit)",
        tmp_path,
    )
    s = default_settings
    out = acoustic.compute_travel_times({"existing": 7}, s.Geometry, s.Medium, T, s.Transmit)
    assert out["existing"] == 7
    assert ref["tr"].shape == (1, 3)
    assert_float(out["tr"], ref["tr"].reshape(-1), err_msg=f"tr {config}")
    assert_float(out["pulse_length"], ref["pulse_length"].item())
    assert_float(out["max_trans_dist"], ref["max_trans_dist"].item())


def test_travel_times_steered_transmit(matlab_engine: Any, oracle: Any, tmp_path: Path) -> None:
    """Nonzero transmit delays (steered plane wave, focused) in both branches."""
    eng = matlab_engine
    from tests.oracle.test_acoustic_grid import update_settings

    for focus, angle in ((np.inf, 10.0), (-20e-3, 0.0), (30e-3, -5.0)):
        build_settings(eng, oracle.path, var="pyp_S1__")
        eng.eval(
            f"pyp_S1__.Transmit.LateralFocus = {focus if np.isfinite(focus) else 'Inf'};"
            f"pyp_S1__.Transmit.Angle = {angle};",
            nargout=0,
        )
        update_settings(eng, "pyp_S1__")
        s = load_settings(eng, tmp_path, var="pyp_S1__")
        assert np.ptp(s.Transmit.Delays) > 0
        n_el = int(s.Transducer.NumberOfElements)
        for config in ("Linear", "Curved"):
            pts = _points(n_el, 6, seed=2, layout="linear")
            put(eng, pyp_pts__=pts)
            eng.eval(
                f"pyp_T__ = pyp_S1__.Transducer; pyp_T__.Configuration = '{config}';"
                "pyp_T__.integration_points = pyp_pts__;"
                "pyp_T__ = get_transducer_integration_delays(pyp_T__, pyp_S1__.Medium);",
                nargout=0,
            )
            ref = matlab_value(
                eng,
                "compute_travel_times(struct(), pyp_S1__.Geometry, pyp_S1__.Medium, pyp_T__,"
                " pyp_S1__.Transmit)",
                tmp_path,
            )
            T = s.Transducer.copy_struct()
            T.Configuration = config
            T.integration_points = pts
            T = acoustic.get_transducer_integration_delays(T, s.Medium)
            out = acoustic.compute_travel_times(None, s.Geometry, s.Medium, T, s.Transmit)
            assert_float(out["tr"], ref["tr"].reshape(-1), err_msg=f"{focus} {angle} {config}")
            assert_float(out["max_trans_dist"], ref["max_trans_dist"].item())
