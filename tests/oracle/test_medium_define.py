"""Oracle tests: ``define_medium`` (non-STL path, curved coupling region, vessel mask).

Background maps are seeded truncated-normal draws → D-13 float class
(≤ 1e-9 rel). The vessel mask comes from MATLAB's own ``VOXELISE`` on a small
synthetic STL (the STL reader/voxeliser are M4) and is passed to the Python
port, so the vessel property assignment is checked against MATLAB too.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pyproteus import acoustic, medium
from pyproteus import numerics as nm
from tests.oracle.test_acoustic_grid import (
    assert_float,
    build_settings,
    load_settings,
    matlab_value,
    update_settings,
)

pytest.importorskip("pyproteus.transducer.geometry", reason="needs get_element_geometry")

KEYS = ("sound_speed", "density", "BonA", "alpha_coeff")


def _compare(ref: dict[str, Any], kw: dict[str, Any], vessel: np.ndarray, ref_vessel, label: str):
    for k in KEYS:
        e = ref[k]
        a = np.asarray(kw[k])
        if e.ndim == 2 and a.ndim == 3 and a.shape[2] == 1:  # MATLAB drops a trailing singleton
            e = e.reshape(a.shape)
        assert_float(a, e, err_msg=f"{k} {label}")
    assert kw["alpha_power"] == ref["alpha_power"].item()
    assert ("alpha_mode" in kw) == ("alpha_mode" in ref)
    if "alpha_mode" in ref:
        assert kw["alpha_mode"] == ref["alpha_mode"]
    np.testing.assert_array_equal(vessel, np.asarray(ref_vessel, dtype=bool).reshape(vessel.shape))


def _small_settings(eng: Any, oracle_path: Path, var: str, ppw: float = 2.0) -> None:
    """Default settings on a coarse grid (few voxels), 16 elements."""
    build_settings(eng, oracle_path, var=var)
    eng.eval(
        f"{var}.Transducer.NumberOfElements = 16; {var}.Transducer.ElementHeight = 2e-3;"
        f"{var}.Transmit = reset_transmit({var}.Transducer);"
        f"{var}.SimulationParameters.PointsPerWavelength = {ppw};"
        f"{var}.Geometry.EmbedVessel = false;",
        nargout=0,
    )
    update_settings(eng, var)


@pytest.mark.parametrize("seed", [0, 1, 17])
@pytest.mark.parametrize("attenuation_b", [1.5, 1.05, 0.9])
@pytest.mark.parametrize("cutoff", [3.0, 0.7])
def test_define_medium_background(
    matlab_engine: Any, oracle: Any, tmp_path: Path, seed: int, attenuation_b: float, cutoff: float
) -> None:
    eng = matlab_engine
    _small_settings(eng, oracle.path, "pyp_S4__")
    eng.eval(
        f"pyp_S4__.Medium.AttenuationB = {attenuation_b};"
        f"pyp_S4__.Medium.InhomogeneityCutoff = {cutoff};"
        "pyp_G__ = define_grid(pyp_S4__.SimulationParameters, pyp_S4__.Geometry);"
        f"rng({seed},'twister');"
        # Geometry.GeometriesPath/Folder are needed by MATLAB even without an STL
        "pyp_S4__.Geometry.GeometriesPath = tempdir;"
        "[pyp_m__, pyp_v__] = define_medium(pyp_G__, pyp_S4__.Medium, pyp_S4__.Geometry);",
        nargout=0,
    )
    ref = matlab_value(eng, "pyp_m__", tmp_path)
    ref_v = matlab_value(eng, "pyp_v__", tmp_path)
    s = load_settings(eng, tmp_path, var="pyp_S4__")
    eng.eval("clear pyp_m__ pyp_v__ pyp_G__; rng(0,'twister');", nargout=0)
    grid = acoustic.define_grid(s.SimulationParameters, s.Geometry)
    assert min(grid.shape) > 1
    kw, vessel = medium.define_medium(grid, s.Medium, s.Geometry, rng=nm.rng(seed))
    _compare(ref, kw, vessel, ref_v, f"seed {seed}")


@pytest.mark.parametrize("radius", [3e-3, 10e-3])
def test_define_medium_curved_coupling(
    matlab_engine: Any, oracle: Any, tmp_path: Path, radius: float
) -> None:
    eng = matlab_engine
    _small_settings(eng, oracle.path, "pyp_S5__")
    eng.eval(
        "pyp_S5__.Transducer.Configuration = 'Curved';"
        f"pyp_S5__.Transducer.Radius = {radius};"
        "pyp_S5__.Transmit = reset_transmit(pyp_S5__.Transducer);",
        nargout=0,
    )
    update_settings(eng, "pyp_S5__")
    eng.eval(
        "pyp_S5__.Geometry.GeometriesPath = tempdir;"
        "pyp_G__ = define_grid(pyp_S5__.SimulationParameters, pyp_S5__.Geometry);"
        "rng(3,'twister');"
        "[pyp_m__, pyp_v__] = define_medium(pyp_G__, pyp_S5__.Medium, pyp_S5__.Geometry,"
        " pyp_S5__.Transducer);",
        nargout=0,
    )
    ref = matlab_value(eng, "pyp_m__", tmp_path)
    ref_v = matlab_value(eng, "pyp_v__", tmp_path)
    s = load_settings(eng, tmp_path, var="pyp_S5__")
    eng.eval("clear pyp_m__ pyp_v__ pyp_G__; rng(0,'twister');", nargout=0)
    grid = acoustic.define_grid(s.SimulationParameters, s.Geometry)
    kw, vessel = medium.define_medium(grid, s.Medium, s.Geometry, s.Transducer, rng=nm.rng(3))
    n_coupling = int(np.count_nonzero(kw["sound_speed"] == s.Medium.SpeedOfSound))
    assert n_coupling > 0, "test must hit the coupling region"
    _compare(ref, kw, vessel, ref_v, f"curved R={radius}")


def _write_box_stl(path: Path, lo: np.ndarray, hi: np.ndarray) -> None:
    """ASCII STL of an axis-aligned box (12 triangles), coordinates in micrometres."""
    v = np.array(
        [
            [
                lo[0] if i & 1 == 0 else hi[0],
                lo[1] if i & 2 == 0 else hi[1],
                lo[2] if i & 4 == 0 else hi[2],
            ]
            for i in range(8)
        ]
    )
    faces = [
        (0, 2, 1),
        (1, 2, 3),
        (4, 5, 6),
        (5, 7, 6),
        (0, 1, 4),
        (1, 5, 4),
        (2, 6, 3),
        (3, 6, 7),
        (0, 4, 2),
        (2, 4, 6),
        (1, 3, 5),
        (3, 7, 5),
    ]
    lines = ["solid box"]
    for f in faces:
        lines += ["facet normal 0 0 0", "outer loop"]
        lines += [f"vertex {v[i][0]:.9g} {v[i][1]:.9g} {v[i][2]:.9g}" for i in f]
        lines += ["endloop", "endfacet"]
    lines.append("endsolid box")
    path.write_text("\n".join(lines) + "\n")


def test_define_medium_vessel_from_matlab_voxelise(
    matlab_engine: Any, oracle: Any, tmp_path: Path
) -> None:
    eng = matlab_engine
    _small_settings(eng, oracle.path, "pyp_S6__")
    folder = tmp_path / "geo" / "box"
    folder.mkdir(parents=True)
    _write_box_stl(
        folder / "box.stl", np.array([-1500.0, -1000.0, -800.0]), np.array([1500.0, 1200.0, 900.0])
    )
    eng.eval(
        f"pyp_S6__.Geometry.GeometriesPath = '{tmp_path / 'geo'}';"
        "pyp_S6__.Geometry.Folder = 'box'; pyp_S6__.Geometry.STLfile = 'box.stl';"
        "pyp_S6__.Geometry.STLunit = 1e-6; pyp_S6__.Geometry.EmbedVessel = true;"
        "pyp_S6__.Geometry.BoundingBox.Center = [0; 0; 0]; pyp_S6__.Geometry.Rotation = eye(3);"
        "pyp_G__ = define_grid(pyp_S6__.SimulationParameters, pyp_S6__.Geometry);"
        "rng(5,'twister');"
        "[pyp_m__, pyp_v__] = define_medium(pyp_G__, pyp_S6__.Medium, pyp_S6__.Geometry);",
        nargout=0,
    )
    ref = matlab_value(eng, "pyp_m__", tmp_path)
    ref_v = np.asarray(matlab_value(eng, "pyp_v__", tmp_path), dtype=bool)
    s = load_settings(eng, tmp_path, var="pyp_S6__")
    eng.eval("clear pyp_m__ pyp_v__ pyp_G__; rng(0,'twister');", nargout=0)
    grid = acoustic.define_grid(s.SimulationParameters, s.Geometry)
    ref_v = ref_v.reshape(grid.shape)
    assert ref_v.any() and not ref_v.all(), "the box must cover part of the grid"
    with pytest.raises(NotImplementedError, match="M4"):
        medium.define_medium(grid, s.Medium, s.Geometry, rng=nm.rng(5))
    kw, vessel = medium.define_medium(grid, s.Medium, s.Geometry, rng=nm.rng(5), vessel=ref_v)
    _compare(ref, kw, vessel, ref_v, "vessel")
