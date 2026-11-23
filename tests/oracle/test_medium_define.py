"""Oracle tests: ``define_medium`` (non-STL path, curved coupling region, vessel mask).

Background maps are seeded truncated-normal draws → D-13 float class
(≤ 1e-9 rel). The vessel mask comes from MATLAB's own ``VOXELISE`` on a small
synthetic STL (the STL reader/voxeliser are M4) and is passed to the Python
port, so the vessel property assignment is checked against MATLAB too.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pyproteus import medium
from pyproteus import numerics as nm
from tests.oracle.test_acoustic_grid import (
    assert_float,
    build_settings,
    grid_from_matlab,
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
    grid = grid_from_matlab(matlab_value(eng, "pyp_G__", tmp_path))  # same grid both sides
    eng.eval("clear pyp_m__ pyp_v__ pyp_G__; rng(0,'twister');", nargout=0)
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
    grid = grid_from_matlab(matlab_value(eng, "pyp_G__", tmp_path))  # same grid both sides
    eng.eval("clear pyp_m__ pyp_v__ pyp_G__; rng(0,'twister');", nargout=0)
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


def _voxelise_in_matlab(eng: Any, var: str, tmp: Path, seed: int):
    """Run MATLAB ``define_medium`` with ``EmbedVessel`` (``READ_stl`` + ``VOXELISE``)."""
    eng.eval(
        f"pyp_G__ = define_grid({var}.SimulationParameters, {var}.Geometry);"
        f"rng({seed},'twister');"
        f"[pyp_m__, pyp_v__] = define_medium(pyp_G__, {var}.Medium, {var}.Geometry);",
        nargout=0,
    )
    ref = matlab_value(eng, "pyp_m__", tmp)
    ref_v = matlab_value(eng, "pyp_v__", tmp)
    s = load_settings(eng, tmp, var=var)
    grid = grid_from_matlab(matlab_value(eng, "pyp_G__", tmp))  # same grid both sides
    eng.eval("clear pyp_m__ pyp_v__ pyp_G__; rng(0,'twister');", nargout=0)
    # The mask contract M4's voxeliser must meet: bool, (Nx, Ny, Nz), index [ix, iy, iz]
    # (MATLAB's own subscripts; the .mat reader keeps them, no C/Fortran reshuffle).
    assert ref_v.dtype == bool, ref_v.dtype
    assert ref_v.shape == grid.shape, (ref_v.shape, grid.shape)
    return ref, ref_v, s, grid


def test_define_medium_vessel_from_matlab_voxelise(
    matlab_engine: Any, oracle: Any, tmp_path: Path
) -> None:
    """Asymmetric box STL: MATLAB's mask, its orientation, and the Python vessel path."""
    eng = matlab_engine
    _small_settings(eng, oracle.path, "pyp_S6__")
    eng.eval("pyp_G0__ = define_grid(pyp_S6__.SimulationParameters, pyp_S6__.Geometry);", nargout=0)
    grid0 = grid_from_matlab(matlab_value(eng, "pyp_G0__", tmp_path))
    center = np.asarray(eng.eval("pyp_S6__.Geometry.Center")).reshape(-1)
    # Box faces 0.3 dx past a grid node (not half-way: VOXELISE errors when a face is
    # equidistant from two grid lines), a different extent along each axis,
    # placed off-centre so a transposed or mirrored mask cannot match.
    axes = (grid0.x, grid0.y, grid0.z)
    lo_idx, hi_idx = (2, 1, 1), (len(grid0.x) // 2, len(grid0.y) - 3, len(grid0.z) // 3 + 1)
    lo = np.array([a[i] + 0.3 * grid0.dx for a, i in zip(axes, lo_idx, strict=True)])
    hi = np.array([a[i] + 0.3 * grid0.dx for a, i in zip(axes, hi_idx, strict=True)])
    folder = tmp_path / "geo" / "box"
    folder.mkdir(parents=True)
    _write_box_stl(folder / "box.stl", (lo - center) / 1e-6, (hi - center) / 1e-6)
    eng.eval(
        f"pyp_S6__.Geometry.GeometriesPath = '{tmp_path / 'geo'}';"
        "pyp_S6__.Geometry.Folder = 'box'; pyp_S6__.Geometry.STLfile = 'box.stl';"
        "pyp_S6__.Geometry.STLunit = 1e-6; pyp_S6__.Geometry.EmbedVessel = true;"
        "pyp_S6__.Geometry.BoundingBox.Center = [0; 0; 0]; pyp_S6__.Geometry.Rotation = eye(3);",
        nargout=0,
    )
    ref, ref_v, s, grid = _voxelise_in_matlab(eng, "pyp_S6__", tmp_path, seed=5)
    expected = (
        ((grid.x > lo[0]) & (grid.x < hi[0]))[:, None, None]
        & ((grid.y > lo[1]) & (grid.y < hi[1]))[None, :, None]
        & ((grid.z > lo[2]) & (grid.z < hi[2]))[None, None, :]
    )
    assert len(set(expected.shape)) == 3 and expected.any() and not expected.all()
    np.testing.assert_array_equal(ref_v, expected)  # orientation: [ix, iy, iz]
    with pytest.raises(NotImplementedError, match="M4"):
        medium.define_medium(grid, s.Medium, s.Geometry, rng=nm.rng(5))
    kw, vessel = medium.define_medium(grid, s.Medium, s.Geometry, rng=nm.rng(5), vessel=ref_v)
    _compare(ref, kw, vessel, ref_v, "box vessel")


SHIPPED = {"renal_tree": "maa.stl", "mouse_brain": "lpa_mouse_t10.stl"}


@pytest.mark.parametrize("folder", sorted(SHIPPED))
def test_define_medium_shipped_geometries(
    matlab_engine: Any, oracle: Any, tmp_path: Path, folder: str
) -> None:
    """Default settings of both shipped geometries with MATLAB's VOXELISE mask.

    The STL files are not in the oracle checkout (Zenodo 10573662); set
    ``PYPROTEUS_GEOMETRIES_PATH`` to a folder with ``renal_tree/maa.stl`` and
    ``mouse_brain/lpa_mouse_t10.stl``.
    """
    root = os.environ.get("PYPROTEUS_GEOMETRIES_PATH")
    if not root or not (Path(root) / folder / SHIPPED[folder]).is_file():
        pytest.skip(f"{folder}/{SHIPPED[folder]} not found (set PYPROTEUS_GEOMETRIES_PATH)")
    eng = matlab_engine
    build_settings(eng, oracle.path, var="pyp_S7__", folder=folder)
    eng.eval(
        "pyp_S7__.SimulationParameters.PointsPerWavelength = 2;"
        f"pyp_S7__.Geometry.GeometriesPath = '{root}';",
        nargout=0,
    )
    update_settings(eng, "pyp_S7__")
    ref, ref_v, s, grid = _voxelise_in_matlab(eng, "pyp_S7__", tmp_path, seed=2)
    assert ref_v.any(), "the vessel must cover part of the grid"
    with pytest.raises(NotImplementedError, match="M4"):
        medium.define_medium(grid, s.Medium, s.Geometry, s.Transducer, rng=nm.rng(2))
    kw, vessel = medium.define_medium(grid, s.Medium, s.Geometry, rng=nm.rng(2), vessel=ref_v)
    _compare(ref, kw, vessel, ref_v, folder)
