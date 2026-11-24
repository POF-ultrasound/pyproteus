"""Unit tests for ``pyproteus.acoustic`` (no MATLAB): grid, signals, transducer integration."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pyproteus import acoustic
from pyproteus.config.models import (
    BoundingBox,
    Domain,
    Geometry,
    Medium,
    SimulationParameters,
    Transducer,
    Transmit,
)

# ------------------------------------------------------------------- grid


def _brute_largest_prime(n: int) -> int:
    if n < 2:
        return n
    best, m, p = 1, n, 2
    while m > 1:
        if m % p == 0:
            best, m = p, m // p
        else:
            p += 1
    return best


def test_largest_prime_factor() -> None:
    for n in range(0, 3000):
        assert acoustic.largest_prime_factor(n) == _brute_largest_prime(n), n
    assert acoustic.largest_prime_factor(2**31 - 1) == 2**31 - 1
    with pytest.raises(ValueError):
        acoustic.largest_prime_factor(-1)
    with pytest.raises(ValueError):
        acoustic.largest_prime_factor(2.5)


def test_optimize_grid_size_rules() -> None:
    # scalar range: single candidate N + 30 = 120 = 2^3*3*5
    assert acoustic.optimize_grid_size(90, 30, 5) == 120
    # 100 + 30 = 130 = 2*5*13 → grows to 135 = 3^3*5
    assert acoustic.optimize_grid_size(100, 30, 5) == 135
    # first of the minima: 24 (largest factor 3) wins over 27; 32 has 2 → 32 wins
    assert acoustic.optimize_grid_size(24, [0, 8], 3) == 32
    assert acoustic.optimize_grid_size(24, [0, 7], 3) == 24  # 24 = 2^3*3 before 27 = 3^3
    # range order does not matter
    assert acoustic.optimize_grid_size(97, [10, 0], 5) == acoustic.optimize_grid_size(
        97, [0, 10], 5
    )
    # growth starts from N + max(range)
    assert acoustic.optimize_grid_size(31, [0, 1], 2) == 32
    assert acoustic.optimize_grid_size(33, [0, 1], 2) == 64
    with pytest.raises(ValueError):
        acoustic.optimize_grid_size(33, [0, 1], 1)


def _settings(dx=1e-4, **extra):
    return SimulationParameters(GridSize=dx, SamplingRate=40e6, SensorOnGrid=True, **extra)


def _geometry(**dom):
    return Geometry(Domain=Domain(**dom))


def test_define_grid_basic() -> None:
    dx = 1e-4
    g = acoustic.define_grid(
        _settings(dx),
        _geometry(Xmin=0.0, Xmax=90 * dx, Ymin=-20 * dx, Ymax=20 * dx, Zmin=0.0, Zmax=0.0),
    )
    # x: N = 91 → 91 + 30 = 121 = 11^2 → 125; PML = ceil(34/2) = 17; core 125 - 34 = 91
    assert g.full_size[0] == 125 and g.PML.X_SIZE == 17 and g.Nx == 91
    np.testing.assert_allclose(g.x, np.arange(91) * dx)
    assert g.dt == 1 / 40e6 and g.PML.Alpha == 2.0 and g.sensor_on_grid is True
    assert g.shape == (g.Nx, g.Ny, g.Nz)
    for M in g.full_size:
        assert acoustic.largest_prime_factor(M) <= 5
    for M, N, pml in zip(
        g.full_size, g.shape, (g.PML.X_SIZE, g.PML.Y_SIZE, g.PML.Z_SIZE), strict=True
    ):
        assert N + 2 * pml == M and pml >= 15


def test_define_grid_q90_default_drops_a_point() -> None:
    """Q-90, reproduced by default (D-19): odd padding drops the last domain point."""
    dx = 1e-4
    geo = _geometry(Xmin=0.0, Xmax=99 * dx, Ymin=0.0, Ymax=89 * dx, Zmin=0.0, Zmax=0.0)
    g = acoustic.define_grid(_settings(dx), geo)
    # x: N = 100 → M = 135, M - N = 35 → PML 18, core 99 (domain has 100 points)
    assert (g.full_size[0], g.PML.X_SIZE, g.Nx) == (135, 18, 99)
    np.testing.assert_allclose(g.x, np.arange(99) * dx)
    assert (g.full_size[1], g.PML.Y_SIZE, g.Ny) == (120, 15, 90)  # even padding: all kept
    # z: a one-point domain gives an empty core (N = 1 → M = 32, PML 16, core 0)
    assert (g.full_size[2], g.PML.Z_SIZE, g.Nz) == (32, 16, 0) and g.z.size == 0
    with pytest.raises(ValueError, match="pml_rounding"):
        acoustic.define_grid(_settings(dx), geo, pml_rounding="round")


def test_define_grid_floor_keeps_the_domain() -> None:
    """D-19 fix: PML = floor((M - N)/2), so an odd padding point stays in the core grid."""
    dx = 1e-4
    g = acoustic.define_grid(
        _settings(dx),
        _geometry(Xmin=0.0, Xmax=99 * dx, Ymin=0.0, Ymax=89 * dx, Zmin=0.0, Zmax=0.0),
        pml_rounding="floor",
    )
    # x: N = 100 → M = 135, M - N = 35 → PML 17, core 101 (whole domain + one point beyond)
    assert (g.full_size[0], g.PML.X_SIZE, g.Nx) == (135, 17, 101)
    np.testing.assert_allclose(g.x, np.arange(101) * dx)
    assert (g.full_size[1], g.PML.Y_SIZE, g.Ny) == (120, 15, 90)  # even padding: unchanged
    # z: a one-point domain keeps its point (N = 1 → M = 32, PML 15, core 2)
    assert (g.full_size[2], g.PML.Z_SIZE, g.Nz) == (32, 15, 2)
    np.testing.assert_allclose(g.z, [0.0, dx])


def test_define_grid_overrides() -> None:
    dx = 1e-4
    geo = _geometry(Xmin=0.0, Xmax=50 * dx, Ymin=0.0, Ymax=50 * dx, Zmin=0.0, Zmax=50 * dx)
    g = acoustic.define_grid(_settings(dx, MaxPrime=7.0, PMLmin=10.0), geo)
    # N = 51 → 51 + 20 = 71 (prime) → 72 = 2^3*3^2; odd padding 21 → PML 11, core 50 (Q-90)
    assert g.full_size == (72, 72, 72) and g.PML.X_SIZE == 11 and g.Nx == 50
    f = acoustic.define_grid(_settings(dx, MaxPrime=7.0, PMLmin=10.0), geo, pml_rounding="floor")
    assert f.full_size == (72, 72, 72) and f.PML.X_SIZE == 10 and f.Nx == 52  # D-19 fix
    g2 = acoustic.define_grid(_settings(dx, MaxPrime=2.0), geo)
    assert g2.full_size == (128, 128, 128)


# ---------------------------------------------------------------- signals


@pytest.mark.parametrize("N", [5, 6, 7, 8, 33])
def test_delta_function_sums_to_one_and_is_one_at_centre(N: int) -> None:
    x = np.arange(N, dtype=float)[None, :] * 2e-4
    b = acoustic.evaluate_delta_function(x, [0.3 * 2e-4], [2e-4], [N])
    assert b.shape == (N,)
    assert abs(b.sum() - 1) < 1e-12  # k = 0 Fourier coefficient of the band-limited delta
    on = acoustic.evaluate_delta_function(x, [2 * 2e-4], [2e-4], [N])
    np.testing.assert_allclose(on, np.eye(N)[2], atol=1e-14)  # Kronecker delta on nodes


def test_delta_function_product_over_dimensions() -> None:
    rs = np.random.RandomState(0)
    x = rs.uniform(-1, 1, size=(3, 10)) * 1e-3
    xi, dx, N = [1e-4, -2e-4, 0.0], [1e-4, 1.5e-4, 2e-4], [10, 11, 12]
    b = acoustic.evaluate_delta_function(x, xi, dx, N)
    parts = [
        acoustic.evaluate_delta_function(x[k : k + 1], [xi[k]], [dx[k]], [N[k]]) for k in range(3)
    ]
    np.testing.assert_allclose(b, parts[0] * parts[1] * parts[2], rtol=1e-14)
    # odd uses sin, even uses tan in the denominator
    r = x[0] - xi[0]
    even = np.sin(np.pi * r / dx[0]) / (N[0] * np.tan(np.pi * r / (N[0] * dx[0])))
    np.testing.assert_allclose(parts[0], even, rtol=1e-14)
    with pytest.raises(ValueError):
        acoustic.evaluate_delta_function(x, [0, 0], dx, N)


def test_sinc_interpolation_reproduces_samples_and_shapes() -> None:
    X = np.arange(16) * 0.1
    V = np.cos(2 * np.pi * 0.7 * X)
    np.testing.assert_allclose(acoustic.sinc_interpolation(X, V, X), V, atol=1e-14)
    M = np.column_stack([V, 2 * V, -V])
    out = acoustic.sinc_interpolation(X, M, X[:5] + 0.05)
    assert out.shape == (3, 5)
    np.testing.assert_allclose(out[1], 2 * out[0])
    np.testing.assert_allclose(
        acoustic.sinc_interpolation(X, V.reshape(1, -1), X[:3]), V[:3], atol=1e-14
    )
    # chunked evaluation gives the same result as one block
    Xq = np.linspace(-0.3, 1.8, 9001)
    with np.errstate(all="ignore"):  # spurious Accelerate matmul flags
        full = acoustic.sinc_matrix(Xq, X, spacing=0.1) @ V
    np.testing.assert_allclose(acoustic.sinc_interpolation(X, V, Xq), full, rtol=1e-13, atol=1e-15)
    with pytest.raises(ValueError):
        acoustic.sinc_interpolation(X, np.zeros((16, 2, 2)), X)


def test_resample_signal() -> None:
    rs = np.random.RandomState(1)
    p = rs.standard_normal(30)
    np.testing.assert_allclose(acoustic.resample_signal(p, 10e6, 10e6), p, atol=1e-13)
    out = acoustic.resample_signal(p, 10.0, 25.0)  # t_old(end)*fs_new = 72.5 → 73 samples
    assert out.shape == (73,)
    np.testing.assert_allclose(out[::5], p[::2], atol=1e-13)  # common sample times
    assert acoustic.resample_signal([3.0], 1.0, 7.0).tolist() == [3.0]
    with pytest.raises(NotImplementedError):
        acoustic.resample_signal(p, 1.0, 2.0, True)


# ------------------------------------------------- transducer integration


def _linear_transducer(**kw) -> Transducer:
    base = dict(
        Configuration="Linear",
        NumberOfElements=4.0,
        Pitch=3e-4,
        ElementWidth=2.5e-4,
        ElementHeight=1e-3,
        ElevationFocus=20e-3,
    )
    base.update(kw)
    return Transducer(**base)


def test_integration_delays_focus() -> None:
    T = _linear_transducer()
    z = np.array([[-1e-3, 0.0, 1e-3]] * 2)
    pts = np.zeros((2, 3, 3))
    pts[:, :, 2] = z
    T.integration_points = pts
    c = 1500.0
    out = acoustic.get_transducer_integration_delays(T, Medium(SpeedOfSound=c))
    d = -np.sqrt(z**2 + 0.02**2) / c
    np.testing.assert_allclose(out.integration_transmit_delays, d - d.min(), rtol=1e-12)
    np.testing.assert_array_equal(out.integration_receive_delays, out.integration_transmit_delays)
    assert not T.isfield("integration_transmit_delays")  # input untouched
    # negative focus: delays not negated → edges largest... centre smallest
    T.ElevationFocus = -0.02
    neg = acoustic.get_transducer_integration_delays(T, Medium(SpeedOfSound=c))
    assert (
        neg.integration_transmit_delays[0, 1] == 0.0 and neg.integration_transmit_delays[0, 0] > 0
    )
    T.ElevationFocus = float("inf")
    flat = acoustic.get_transducer_integration_delays(T, Medium(SpeedOfSound=c))
    np.testing.assert_array_equal(flat.integration_transmit_delays, np.zeros((2, 3)))


def test_integration_delays_rca_custom() -> None:
    T = _linear_transducer(Configuration="RCA")
    T.integration_transmit_apodization = np.ones((4, 5))
    out = acoustic.get_transducer_integration_delays(T, Medium(SpeedOfSound=1540.0))
    assert (
        out.integration_transmit_delays.shape == (4, 5) and not out.integration_receive_delays.any()
    )
    T.Configuration = "Custom"
    T.integration_receive_delays = np.full((4, 5), 7.0)
    out = acoustic.get_transducer_integration_delays(T, Medium(SpeedOfSound=1540.0))
    assert not out.integration_transmit_delays.any()
    assert (out.integration_receive_delays == 7.0).all()


def test_compute_travel_times_linear_by_hand() -> None:
    T = _linear_transducer()
    pts = np.zeros((2, 2, 3))
    pts[0, :, 1] = [-1e-3, -0.5e-3]
    pts[1, :, 1] = [0.5e-3, 1e-3]
    T.integration_points = pts
    geo = Geometry(
        Domain=Domain(Xmax=0.02, Ymin=-0.01, Ymax=0.01),
        BoundingBox=BoundingBox(Diagonal=np.array([[3e-3], [4e-3], [12e-3]])),
    )
    tx = Transmit(
        SamplingRate=100e6, Delays=np.array([[0.0, 1e-7]]), PressureSignal=np.zeros((1, 11))
    )
    c = 1500.0
    out = acoustic.compute_travel_times(None, geo, Medium(SpeedOfSound=c), T, tx)
    y = np.array([-0.75e-3, 0.75e-3])
    dist = np.maximum(np.hypot(y + 0.01, 0.02), np.hypot(y - 0.01, 0.02))
    t_tx = np.max(dist / c + [0.0, 1e-7])
    t_rx = np.hypot(1e-3 + 0.01, 0.02) / c
    pd = 10 / 100e6
    np.testing.assert_allclose(
        out["tr"], [2 * pd + t_tx, 2 * pd + t_tx + 13e-3 / c, 2 * pd + t_tx + t_rx]
    )
    np.testing.assert_allclose(out["pulse_length"], c * pd)
    np.testing.assert_allclose(out["max_trans_dist"], t_tx * c)


def test_compute_travel_times_vertices_branch() -> None:
    T = _linear_transducer(Configuration="Curved")
    pts = np.zeros((2, 1, 3))
    pts[:, 0, 1] = [-1e-3, 1e-3]
    T.integration_points = pts
    T.integration_transmit_delays = np.array([[0.0], [2e-7]])
    T.integration_receive_delays = np.array([[3e-7], [0.0]])
    verts = np.array([[0.0, -0.01, 0], [0.02, 0.01, 0], [0.02, -0.01, 0.005]])
    geo = Geometry(
        Domain=Domain(Vertices=verts), BoundingBox=BoundingBox(Diagonal=np.zeros((3, 1)))
    )
    tx = Transmit(SamplingRate=1e6, Delays=np.array([[1e-7, 0.0]]), PressureSignal=np.zeros((1, 1)))
    out = acoustic.compute_travel_times({}, geo, Medium(SpeedOfSound=1000.0), T, tx)
    d = np.max(np.linalg.norm(pts[:, 0, None, :] - verts[None], axis=2), axis=1)
    t_tx = np.max(d / 1000 + np.array([0.0, 2e-7]) + np.array([1e-7, 0.0]))
    t_rx = np.max(d / 1000 + np.array([3e-7, 0.0]))
    np.testing.assert_allclose(out["tr"], [t_tx, t_tx, t_tx + t_rx])


@pytest.fixture
def element_geometry():
    """Integration points need ``get_element_geometry`` (lands with m1-leaves)."""
    return pytest.importorskip("pyproteus.transducer.geometry", reason="get_element_geometry")


class _G:
    dy = dz = 1e-4


def test_integration_points_linear_layout(element_geometry) -> None:
    T = _linear_transducer(IntegrationDensity=2.0)
    out = acoustic.get_transducer_integration_points(T, _G())
    # N_int_y = ceil(2*2.5e-4/1e-4) = 5, spacing 5e-5; z spacing 5e-5, |z| <= 5e-4 → 21
    assert out.integration_points.shape == (4, 105, 3)
    p = out.integration_points
    assert (p[..., 0] == 0).all()
    np.testing.assert_allclose(p[0, :5, 1], -(3 * 3e-4 + 2.5e-4) / 2 + (np.arange(5) + 0.5) * 5e-5)
    np.testing.assert_allclose(p[0, ::5, 2], np.arange(-10, 11) * 5e-5)  # lateral index fastest
    np.testing.assert_allclose(p[1, :, 1] - p[0, :, 1], 3e-4)
    assert out.integration_weights == pytest.approx(0.25)
    assert out.integration_transmit_apodization.shape == (4, 105)


def test_integration_points_rca_and_custom(element_geometry) -> None:
    T = _linear_transducer(Configuration="RCA", IntegrationDensity=1.0)
    out = acoustic.get_transducer_integration_points(T, _G())
    n = out.integration_points.shape[0]
    assert n == 4 and out.NumberOfElements == 4
    np.testing.assert_array_equal(
        out.integration_points[2:], out.integration_points[:2][:, :, [0, 2, 1]]
    )
    assert out.integration_receive_apodization.shape[0] == 4
    with pytest.raises(ValueError, match="even"):
        acoustic.get_transducer_integration_points(
            _linear_transducer(Configuration="RCA", NumberOfElements=3.0), _G()
        )
    C = _linear_transducer(Configuration="Custom")
    C.integration_points = np.zeros((2, 3, 3))
    C.integration_points[:, :, 0] = [[0, 5e-3, 1e-2]] * 2
    C.integration_area = 2e-8
    with pytest.warns(UserWarning, match="orientation"):
        out = acoustic.get_transducer_integration_points(C, _G())
    assert out.integration_weights == pytest.approx(2.0)
    C.integration_points[:, :, 0] = 0
    C.integration_points[:, :, 1] = [[0, 5e-3, 1e-2]] * 2
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        acoustic.get_transducer_integration_points(C, _G())


def test_sinc_complex_and_delta_row_centres() -> None:
    X = np.arange(8) * 0.5
    V = np.exp(1j * X)
    out = acoustic.sinc_interpolation(X, V, X)
    assert np.iscomplexobj(out)
    np.testing.assert_allclose(out, V, atol=1e-14)
    x = np.array([[0.0, 0.3, 0.6]])
    b = acoustic.evaluate_delta_function(x, np.array([0.0, 0.3, 0.1]), [0.1], [16])
    np.testing.assert_allclose(b[:2], [1.0, 1.0])
