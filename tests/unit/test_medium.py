"""Unit tests for the M1 medium functions (no MATLAB)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.stats import truncnorm

from pyproteus import acoustic, medium
from pyproteus import numerics as nm
from pyproteus.config.models import (
    BoundingBox,
    Domain,
    Geometry,
    Medium,
    SimulationParameters,
    Transducer,
    Vessel,
)

# ------------------------------------------------------ truncated normal


@pytest.mark.parametrize("cutoff", [0.5, 1.0, 3.0, 8.0])
def test_truncated_normal_inverse_transform(cutoff: float) -> None:
    """Same uniforms through SciPy's truncnorm.ppf (independent formula) give the same draws."""
    x = medium.random_truncated_normal(cutoff, [40, 30], nm.rng(5))
    u = nm.rng(5).rand(40, 30)
    ref = truncnorm.ppf(u, -cutoff, cutoff)
    np.testing.assert_allclose(x, ref, rtol=1e-9, atol=1e-12)
    assert np.abs(x).max() <= cutoff


def test_truncated_normal_stream_and_shape() -> None:
    g = nm.rng(9)
    x = medium.random_truncated_normal(2.0, [3, 4, 5], g)
    assert x.shape == (3, 4, 5)
    h = nm.rng(9)
    h.rand(1, 60)
    assert g.rand().item() == h.rand().item()  # one uniform per draw
    assert medium.random_truncated_normal(2.0, 3, nm.rng(1)).shape == (3, 3)
    big = medium.random_truncated_normal(3.0, [200, 200], nm.rng(2))
    assert abs(big.mean()) < 0.02 and 0.9 < big.std() < 1.0  # truncated at 3 sd: var 0.973


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), [1.0, 2.0], "a"])
def test_truncated_normal_invalid(bad) -> None:
    with pytest.raises(ValueError, match="positive scalar"):
        medium.random_truncated_normal(bad, [2, 2], nm.rng(0))


def test_truncated_normal_infinite_cutoff_not_supported() -> None:
    with pytest.raises(NotImplementedError):
        medium.random_truncated_normal(float("inf"), [2, 2], nm.rng(0))


# ----------------------------------------------------------- voxelize


def _grid(dx=1e-4):
    return acoustic.define_grid(
        SimulationParameters(GridSize=dx, SamplingRate=1e7, SensorOnGrid=False),
        Geometry(
            Domain=Domain(
                Xmin=0.0, Xmax=40 * dx, Ymin=-10 * dx, Ymax=15 * dx, Zmin=0.0, Zmax=8 * dx
            )
        ),
    )


def test_voxelize_ties_and_exclusion() -> None:
    g = _grid()
    lo = np.array([g.x[0], g.y[0], g.z[0]])
    hi = np.array([g.x[-1], g.y[-1], g.z[-1]])
    pts = np.array(
        [
            lo + np.array([2.5, 0.5, 1.5]) * g.dx,  # ties → away from zero
            lo,
            hi,
            lo - [1e-9, 0, 0],  # outside
            hi + np.array([0, 0, 1e-9]),  # outside
            lo + np.array([3.4, 4.6, 0.49]) * g.dx,
        ]
    )
    out = medium.voxelize_media_points(pts, g)
    np.testing.assert_array_equal(out.exclude, [False, False, False, True, True, False])
    np.testing.assert_array_equal(out.points, pts[[0, 1, 2, 5]])
    np.testing.assert_array_equal(
        out.nodes, [[3, 1, 2], [0, 0, 0], [g.Nx - 1, g.Ny - 1, g.Nz - 1], [3, 5, 0]]
    )
    # column-major linear index
    i, j, k = out.nodes.T
    np.testing.assert_array_equal(out.index, i + g.Nx * (j + g.Ny * k))
    assert out.index.dtype == np.intp


def test_voxelize_empty_and_nan() -> None:
    g = _grid()
    out = medium.voxelize_media_points(np.zeros((0, 3)), g)
    assert out.points.shape == (0, 3) and out.nodes.shape == (0, 3) and out.index.shape == (0,)
    with pytest.raises(ValueError):
        medium.voxelize_media_points(np.array([[np.nan, 0, 0]]), g)


# ------------------------------------------------------- define_medium


def _medium(**kw) -> Medium:
    base = dict(
        SpeedOfSound=1540.0,
        Density=1000.0,
        BonA=6.0,
        AttenuationA=0.75,
        AttenuationB=1.5,
        Inhomogeneity=0.01,
        InhomogeneityCutoff=3.0,
        Vessel=Vessel(SpeedOfSound=1570.0, Density=1060.0, BonA=9.0),
    )
    base.update(kw)
    return Medium(**base)


def test_define_medium_background() -> None:
    g = _grid()
    m = _medium()
    kw, vessel = medium.define_medium(g, m, Geometry(EmbedVessel=False), rng=nm.rng(3))
    bg = 1 + 0.01 * medium.random_truncated_normal(3.0, list(g.shape), nm.rng(3))
    np.testing.assert_array_equal(kw["sound_speed"], 1540.0 * bg)
    np.testing.assert_array_equal(kw["density"], 1000.0 * bg)
    assert kw["BonA"].shape == g.shape and (kw["BonA"] == 6.0).all()
    assert kw["alpha_coeff"].shape == g.shape and (kw["alpha_coeff"] == 0.75).all()
    assert not kw["alpha_coeff"].flags.writeable  # broadcast view (D-15)
    assert kw["alpha_power"] == 1.5 and "alpha_mode" not in kw
    assert vessel.dtype == bool and vessel.shape == g.shape and not vessel.any()
    kw2, _ = medium.define_medium(
        g, _medium(AttenuationB=1.05), Geometry(EmbedVessel=False), rng=nm.rng(3)
    )
    assert kw2["alpha_mode"] == "no_dispersion"


def test_define_medium_vessel_mask() -> None:
    g = _grid()
    with pytest.raises(ValueError, match="geometries_path"):
        medium.define_medium(g, _medium(), Geometry(EmbedVessel=True), rng=nm.rng(0))
    for empty in ("", np.zeros((0, 0))):  # MATLAB '' or [] GeometriesPath: missing
        with pytest.raises(ValueError, match="geometries_path"):
            medium.define_medium(
                g, _medium(), Geometry(EmbedVessel=True, GeometriesPath=empty), rng=nm.rng(0)
            )
    mask = np.zeros(g.shape, dtype=bool)
    mask[3:6, 2:4, 1] = True
    kw, out_mask = medium.define_medium(
        g, _medium(), Geometry(EmbedVessel=True), rng=nm.rng(0), vessel=mask
    )
    assert (kw["sound_speed"][mask] == 1570.0).all() and (kw["density"][mask] == 1060.0).all()
    assert (kw["BonA"][mask] == 9.0).all() and (kw["BonA"][~mask] == 6.0).all()
    assert kw["sound_speed"][~mask].std() > 0
    assert np.array_equal(out_mask, mask)
    with pytest.raises(ValueError):
        medium.define_medium(
            g, _medium(), Geometry(EmbedVessel=True), rng=nm.rng(0), vessel=mask[:-1]
        )
    with pytest.raises(ValueError):
        medium.define_medium(g, _medium(), Geometry(EmbedVessel=False), rng=nm.rng(0), vessel=mask)


def test_define_medium_curved_coupling() -> None:
    pytest.importorskip("pyproteus.transducer.geometry", reason="get_element_geometry")
    from pyproteus.transducer import get_element_geometry

    T = Transducer(
        Configuration="Curved", NumberOfElements=16.0, Pitch=3e-4, Radius=2e-3, ElementWidth=2.5e-4
    )
    g = acoustic.define_grid(
        SimulationParameters(GridSize=1e-4, SamplingRate=1e7, SensorOnGrid=False),
        Geometry(Domain=Domain(Xmin=-2e-3, Xmax=3e-3, Ymin=-2e-3, Ymax=2e-3, Zmin=0.0, Zmax=3e-4)),
    )
    kw, _ = medium.define_medium(g, _medium(), Geometry(EmbedVessel=False), T, rng=nm.rng(4))
    el = get_element_geometry(T)
    inside = (g.x[:, None] - el.center) ** 2 + g.y[None, :] ** 2 <= el.radius**2
    assert inside.any() and (~inside).any()
    assert (kw["sound_speed"][inside] == 1540.0).all()
    assert (kw["sound_speed"][~inside] != 1540.0).all()
    # a linear transducer has no effect
    lin, _ = medium.define_medium(
        g, _medium(), Geometry(EmbedVessel=False), Transducer(Configuration="Linear"), rng=nm.rng(4)
    )
    ref, _ = medium.define_medium(g, _medium(), Geometry(EmbedVessel=False), rng=nm.rng(4))
    np.testing.assert_array_equal(lin["sound_speed"], ref["sound_speed"])


def test_define_medium_broadcast_views_work_with_kwave_adapter() -> None:
    """Read-only broadcast ``alpha_coeff``/``BonA`` give the same k-Wave result as dense arrays."""
    from pyproteus.backends.kwave import KGrid, kspace_first_order

    g = acoustic.define_grid(
        SimulationParameters(GridSize=1e-4, SamplingRate=1.5e7, SensorOnGrid=False),
        Geometry(
            Domain=Domain(Xmin=0.0, Xmax=1.5e-3, Ymin=-5e-4, Ymax=5e-4, Zmin=-3e-4, Zmax=3e-4)
        ),
    )
    kw, _ = medium.define_medium(g, _medium(), Geometry(EmbedVessel=False), rng=nm.rng(1))
    dense = {k: np.array(v) if isinstance(v, np.ndarray) else v for k, v in kw.items()}
    kg = KGrid(g.shape, (g.dx, g.dy, g.dz), g.dt, 30)
    src_mask = np.zeros(g.shape, bool)
    src_mask[2, g.Ny // 2, g.Nz // 2] = True
    sen_mask = np.zeros(g.shape, bool)
    sen_mask[-3, :, g.Nz // 2] = True
    source = {"p_mask": src_mask, "p": np.sin(np.arange(10) / 2.0)[None, :]}
    pml = (g.PML.X_SIZE, g.PML.Y_SIZE, g.PML.Z_SIZE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = kspace_first_order(kg, kw, source, {"mask": sen_mask}, pml_size=pml, backend="python")
        b = kspace_first_order(
            kg, dense, source, {"mask": sen_mask}, pml_size=pml, backend="python"
        )
    np.testing.assert_array_equal(a["p"], b["p"])


def test_geometry_registry_lists_the_shipped_stls() -> None:
    from tools.fetch_geometries import SOURCES, read_registry

    entries = read_registry()
    assert set(entries) == set(SOURCES)
    assert set(entries) == {"renal_tree/maa.stl", "mouse_brain/lpa_mouse_t10.stl"}
    assert all(v.startswith("sha256:") and len(v) == 7 + 64 for v in entries.values())


def test_define_medium_single_plane_keeps_three_dims() -> None:
    """Nz = 1: MatlabRNG.rand drops the trailing singleton; the maps stay 3-D."""
    g = acoustic.define_grid(
        SimulationParameters(GridSize=1e-4, SamplingRate=1e7, SensorOnGrid=False, PMLmin=12.0),
        Geometry(Domain=Domain(Xmin=0.0, Xmax=1e-3, Ymin=0.0, Ymax=5e-4, Zmin=0.0, Zmax=0.0)),
    )
    assert g.Nz == 1
    mask = np.zeros(g.shape, dtype=bool)
    mask[2, 1, 0] = True
    kw, vessel = medium.define_medium(
        g, _medium(), Geometry(EmbedVessel=True), rng=nm.rng(1), vessel=mask
    )
    for k in ("sound_speed", "density", "BonA", "alpha_coeff"):
        assert kw[k].shape == g.shape, k
    assert kw["sound_speed"][2, 1, 0] == 1570.0 and vessel.shape == g.shape


# ----------------------------------------------------------- define_medium with STL


def _box_stl(path, lo, hi) -> None:
    """Binary STL of an axis-aligned box (outward facets), coordinates in µm."""
    c = np.array(
        [
            [lo[0], lo[1], lo[2]],
            [hi[0], lo[1], lo[2]],
            [hi[0], hi[1], lo[2]],
            [lo[0], hi[1], lo[2]],
            [lo[0], lo[1], hi[2]],
            [hi[0], lo[1], hi[2]],
            [hi[0], hi[1], hi[2]],
            [lo[0], hi[1], hi[2]],
        ]
    )
    quads = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4), (2, 3, 7, 6), (1, 2, 6, 5), (0, 4, 7, 3)]
    tris = [c[list(t)] for a, b, cc, d in quads for t in ((a, b, cc), (a, cc, d))]
    rec = np.zeros(len(tris), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["v"] = np.stack(tris)
    path.write_bytes(b"box".ljust(80, b" ") + np.int32(len(tris)).tobytes() + rec.tobytes())


def _stl_geometry() -> Geometry:
    return Geometry(
        EmbedVessel=True,
        Folder="box",
        STLfile="box.stl",
        STLunit=1e-6,
        Rotation=np.eye(3),
        BoundingBox=BoundingBox(Center=np.zeros((3, 1))),
        Center=np.zeros((3, 1)),
    )


def test_define_medium_reads_and_voxelises_stl(tmp_path) -> None:
    """EmbedVessel without vessel=: the STL is voxelised (pyproteus.geometry), from
    geometries_path or Geometry.GeometriesPath (MATLAB's field)."""
    from pyproteus.geometry import place_vessel_mesh, read_stl, voxelise

    g = _grid()
    (tmp_path / "box").mkdir()
    lo, hi = (1030.0, -310.0, 120.0), (2270.0, 640.0, 470.0)  # µm, inside the grid
    _box_stl(tmp_path / "box" / "box.stl", lo, hi)
    geo = _stl_geometry()
    want = voxelise(g.x, g.y, g.z, place_vessel_mesh(read_stl(tmp_path / "box" / "box.stl"), geo))
    assert want.any() and not want.all()
    kw, v = medium.define_medium(g, _medium(), geo, rng=nm.rng(0), geometries_path=tmp_path)
    np.testing.assert_array_equal(v, want)
    assert np.all(kw["sound_speed"][v] == 1570.0) and np.all(kw["BonA"][v] == 9.0)
    assert np.all(kw["sound_speed"][~v] != 1570.0)
    geo.GeometriesPath = str(tmp_path)  # MATLAB field, extra on the model
    _, v2 = medium.define_medium(g, _medium(), geo, rng=nm.rng(0))
    np.testing.assert_array_equal(v2, want)


def test_define_medium_warns_on_empty_vessel(tmp_path) -> None:
    """A vessel thinner than the grid spacing covers no grid point: warn (B-112, D-15)."""
    from pyproteus.geometry import EmptyVesselWarning

    g = _grid()  # dx = 100 µm
    (tmp_path / "box").mkdir()
    _box_stl(tmp_path / "box" / "box.stl", (1010.0, 10.0, 110.0), (1090.0, 90.0, 190.0))
    with pytest.warns(EmptyVesselWarning, match=r"dx = 100\.0 µm"):
        _, v = medium.define_medium(
            g, _medium(), _stl_geometry(), rng=nm.rng(0), geometries_path=tmp_path
        )
    assert not v.any()


def test_define_medium_warns_when_vessel_outside_grid(tmp_path) -> None:
    """A misplaced vessel (wrong Center/STLunit) is reported as outside the grid, and
    the warning points at the caller, not at pyproteus."""
    from pyproteus.geometry import EmptyVesselWarning

    g = _grid()
    (tmp_path / "box").mkdir()
    _box_stl(tmp_path / "box" / "box.stl", (9e4, 9e4, 9e4), (9.5e4, 9.5e4, 9.5e4))
    with pytest.warns(EmptyVesselWarning, match="outside the grid") as rec:
        medium.define_medium(g, _medium(), _stl_geometry(), rng=nm.rng(0), geometries_path=tmp_path)
    assert rec[0].filename == __file__
