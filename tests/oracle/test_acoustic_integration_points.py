"""Oracle tests: ``get_transducer_integration_points`` (linear, curved, RCA, custom).

Integration point counts (``ceil``/``floor`` of exact-input ratios) are compared
exactly; coordinates, weights and the downstream delays/travel times at the
D-13 float class (≤ 1e-9 rel, absolute floor 1e-12 × the aperture size).
"""

from __future__ import annotations

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
    update_settings,
)

pytest.importorskip("pyproteus.transducer.geometry", reason="needs get_element_geometry")


class _Grid:
    def __init__(self, dy: float, dz: float) -> None:
        self.dy, self.dz = dy, dz


def _compare(ref: dict[str, Any], out: Any, label: str) -> None:
    pts_ref = ref["integration_points"]
    pts = np.asarray(out.integration_points)
    assert pts.shape == pts_ref.shape, f"{label}: {pts.shape} vs {pts_ref.shape}"
    scale = float(np.max(np.abs(pts_ref), initial=0.0))
    assert_float(pts, pts_ref, atol=1e-12 * scale, err_msg=f"points {label}")
    assert_float(
        out.integration_weights,
        ref["integration_weights"].reshape(np.shape(out.integration_weights)),
        err_msg=f"weights {label}",
    )
    for k in ("integration_transmit_apodization", "integration_receive_apodization"):
        np.testing.assert_array_equal(np.asarray(getattr(out, k)), ref[k], err_msg=f"{k} {label}")
    assert float(out.NumberOfElements) == float(ref["NumberOfElements"].item())


@pytest.fixture(scope="module")
def settings(matlab_engine: Any, oracle: Any, tmp_path_factory: pytest.TempPathFactory):
    build_settings(matlab_engine, oracle.path, var="pyp_S2__")
    return load_settings(matlab_engine, tmp_path_factory.mktemp("ip"), var="pyp_S2__")


LINEAR_CASES = [
    # (IntegrationDensity, ElementHeight factor of dz, NumberOfElements, dy, dz)
    (1.0, None, None, None, None),  # default transducer, default grid
    (2.0, None, None, None, None),
    (3.7, None, None, None, None),
    (1.0, 10.0, 8.0, 1e-4, 1e-4),  # H/2 = 5 dz exactly: Z1/Z2 on integers
    (2.0, 7.0, 3.0, 1e-4, 1e-4),  # H/2 = 3.5 dz: half-spacing tie at density 2 → 7 dz
    (1.0, 0.5, 1.0, 1.3e-4, 0.9e-4),  # single element, H below one spacing
    (0.5, 13.0, 5.0, 2e-4, 1.5e-4),  # density < 1
]


@pytest.mark.parametrize("case", LINEAR_CASES)
@pytest.mark.parametrize("config", ["Linear", "Curved", "RCA"])
def test_integration_points(
    matlab_engine: Any, settings: Any, tmp_path: Path, case: tuple, config: str
) -> None:
    eng = matlab_engine
    density, hfac, n_el, dy, dz = case
    eng.eval("pyp_T__ = pyp_S2__.Transducer;", nargout=0)
    T = settings.Transducer.copy_struct()
    if dy is None:
        grid = acoustic.define_grid(settings.SimulationParameters, settings.Geometry)
        eng.eval(
            "pyp_Grid__ = define_grid(pyp_S2__.SimulationParameters, pyp_S2__.Geometry);", nargout=0
        )
    else:
        grid = _Grid(dy, dz)
        put(eng, pyp_dy__=dy, pyp_dz__=dz)
        eng.eval("pyp_Grid__ = struct('dy', pyp_dy__, 'dz', pyp_dz__);", nargout=0)
    updates: dict[str, Any] = {"IntegrationDensity": density, "Configuration": config}
    if hfac is not None:
        updates["ElementHeight"] = hfac * grid.dz
        updates["NumberOfElements"] = n_el * (2 if config == "RCA" else 1)
    if config == "Curved":
        updates["Radius"] = 30e-3 if dy is None else 3e-3
    for k, v in updates.items():
        setattr(T, k, v)
        if isinstance(v, str):
            eng.eval(f"pyp_T__.{k} = '{v}';", nargout=0)
        else:
            put(eng, pyp_v__=float(v))
            eng.eval(f"pyp_T__.{k} = pyp_v__;", nargout=0)
    ref = matlab_value(eng, "get_transducer_integration_points(pyp_T__, pyp_Grid__)", tmp_path)
    out = acoustic.get_transducer_integration_points(T, grid)
    _compare(ref, out, f"{config} {case}")

    # downstream: lens delays on these points
    eng.eval(
        "pyp_T2__ = get_transducer_integration_delays("
        "get_transducer_integration_points(pyp_T__, pyp_Grid__), pyp_S2__.Medium);",
        nargout=0,
    )
    ref_d = matlab_value(eng, "pyp_T2__.integration_transmit_delays", tmp_path)
    d = acoustic.get_transducer_integration_delays(out, settings.Medium).integration_transmit_delays
    assert_float(d, ref_d.reshape(d.shape), atol=1e-12 * max(float(np.max(ref_d)), 1e-9))
    eng.eval("clear pyp_T__ pyp_T2__ pyp_Grid__ pyp_v__ pyp_dy__ pyp_dz__", nargout=0)


def test_rca_odd_raises(matlab_engine: Any, settings: Any) -> None:
    import matlab.engine

    eng = matlab_engine
    eng.eval(
        "pyp_T__ = pyp_S2__.Transducer; pyp_T__.Configuration = 'RCA';"
        "pyp_T__.NumberOfElements = 7; pyp_T__.IntegrationDensity = 1;",
        nargout=0,
    )
    with pytest.raises(matlab.engine.MatlabExecutionError, match="even"):
        eng.eval(
            "get_transducer_integration_points(pyp_T__, struct('dy',1e-4,'dz',1e-4));", nargout=0
        )
    T = settings.Transducer.copy_struct()
    T.Configuration, T.NumberOfElements, T.IntegrationDensity = "RCA", 7.0, 1.0
    with pytest.raises(ValueError, match="even"):
        acoustic.get_transducer_integration_points(T, _Grid(1e-4, 1e-4))


@pytest.mark.parametrize("apod", [False, True])
@pytest.mark.parametrize("along_x", [False, True])
def test_custom(
    matlab_engine: Any, settings: Any, tmp_path: Path, apod: bool, along_x: bool
) -> None:
    eng = matlab_engine
    rs = np.random.RandomState(4)
    pts = rs.uniform(-1e-3, 1e-3, size=(3, 4, 3))
    if along_x:
        pts[:, :, 0] *= 20
    area = rs.uniform(1e-9, 1e-8, size=(3, 4))
    T = settings.Transducer.copy_struct()
    T.Configuration = "Custom"
    T.integration_points = pts
    T.integration_area = area
    put(eng, pyp_pts__=pts, pyp_area__=area)
    eng.eval(
        "pyp_T__ = pyp_S2__.Transducer; pyp_T__.Configuration = 'Custom';"
        "pyp_T__.integration_points = pyp_pts__; pyp_T__.integration_area = pyp_area__;",
        nargout=0,
    )
    if apod:
        a = rs.uniform(0, 1, size=(3, 4))
        T.integration_transmit_apodization = a
        put(eng, pyp_a__=a)
        eng.eval("pyp_T__.integration_transmit_apodization = pyp_a__;", nargout=0)
    eng.eval("lastwarn('');", nargout=0)
    ref = matlab_value(
        eng, "get_transducer_integration_points(pyp_T__, struct('dy',1e-4,'dz',2e-4))", tmp_path
    )
    _, warn_id = eng.eval("lastwarn", nargout=2)
    assert (warn_id == "get_transducer_integration_points:orientation") == along_x
    if along_x:
        with pytest.warns(UserWarning, match="orientation"):
            out = acoustic.get_transducer_integration_points(T, _Grid(1e-4, 2e-4))
    else:
        out = acoustic.get_transducer_integration_points(T, _Grid(1e-4, 2e-4))
    _compare(ref, out, f"custom apod={apod}")


def test_curved_default_pipeline(matlab_engine: Any, oracle: Any, tmp_path: Path) -> None:
    """Curved array settings end to end: grid → points → delays → travel times."""
    eng = matlab_engine
    build_settings(eng, oracle.path, var="pyp_S3__")
    eng.eval(
        "pyp_S3__.Transducer.Configuration = 'Curved'; pyp_S3__.Transducer.Radius = 40e-3;"
        "pyp_S3__.Transmit = reset_transmit(pyp_S3__.Transducer);",
        nargout=0,
    )
    update_settings(eng, "pyp_S3__")
    s = load_settings(eng, tmp_path, var="pyp_S3__")
    eng.eval(
        "pyp_G__ = define_grid(pyp_S3__.SimulationParameters, pyp_S3__.Geometry);"
        "pyp_T__ = pyp_S3__.Transducer; pyp_T__.IntegrationDensity = 1;"
        "pyp_T__ = get_transducer_integration_points(pyp_T__, pyp_G__);"
        "pyp_T__ = get_transducer_integration_delays(pyp_T__, pyp_S3__.Medium);",
        nargout=0,
    )
    ref_T = matlab_value(eng, "pyp_T__", tmp_path)
    ref_rp = matlab_value(
        eng,
        "compute_travel_times(struct(), pyp_S3__.Geometry, pyp_S3__.Medium, pyp_T__,"
        " pyp_S3__.Transmit)",
        tmp_path,
    )
    grid = acoustic.define_grid(s.SimulationParameters, s.Geometry)
    T = s.Transducer.copy_struct()
    T.IntegrationDensity = 1.0
    T = acoustic.get_transducer_integration_points(T, grid)
    T = acoustic.get_transducer_integration_delays(T, s.Medium)
    _compare(ref_T, T, "curved default")
    rp = acoustic.compute_travel_times(None, s.Geometry, s.Medium, T, s.Transmit)
    assert_float(rp["tr"], ref_rp["tr"].reshape(-1))
    assert_float(rp["max_trans_dist"], ref_rp["max_trans_dist"].item())
