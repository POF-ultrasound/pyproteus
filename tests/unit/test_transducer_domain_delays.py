"""Unit tests (no MATLAB): degree trigonometry, transmit delays, simulation domain."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyproteus import config
from pyproteus import transducer as ptr
from pyproteus.config.models import Geometry, Medium, Transducer, Transmit


def lin(n: int = 5, conf: str = "Linear") -> Transducer:
    return Transducer(
        Configuration=conf,
        NumberOfElements=n,
        Pitch=1e-3,
        ElementWidth=8e-4,
        ElementHeight=5e-3,
        ElevationFocus=30e-3,
        CenterFrequency=5e6,
    )


C = Medium(SpeedOfSound=1500.0)


def delays(t: Transducer, f: float, a: float) -> np.ndarray:
    d = ptr.compute_delays(Transmit(LateralFocus=f, Angle=a), t, C).Delays
    assert d.ndim == 2 and d.shape[0] == 1 and d.min(initial=0) == 0
    return d.ravel()


def test_linear_delays() -> None:
    np.testing.assert_array_equal(delays(lin(), math.inf, 0.0), np.zeros(5))
    np.testing.assert_allclose(delays(lin(), math.inf, 30.0), np.arange(5) * 0.5e-3 / 1500)
    focused = delays(lin(), 20e-3, 0.0)  # converging: centre element fires last
    assert np.argmax(focused) == 2 and focused[0] == focused[-1] == 0
    diverging = delays(lin(), -20e-3, 0.0)  # virtual source: centre fires first
    assert diverging[2] == 0 and diverging[0] == pytest.approx(diverging[-1])
    np.testing.assert_allclose(diverging[0], (math.hypot(2e-3, 20e-3) - 20e-3) / 1500)
    assert delays(lin(0), math.inf, 0.0).shape == (0,)


def test_rca_and_curved_delays() -> None:
    d = delays(lin(8, "RCA"), math.inf, 10.0)
    assert d.size == 8 and np.array_equal(d[:4], d[4:])
    curved = Transducer(Configuration="Curved", NumberOfElements=9, Pitch=1e-3, Radius=0.02)
    # infinite focus: diverging wave from the centre of curvature, all elements equidistant
    np.testing.assert_allclose(delays(curved, math.inf, 0.0), 0.0, atol=1e-20)
    assert delays(curved, math.nan, 0.0).max() < 1e-18
    steered = delays(curved, math.inf, 15.0)
    # the virtual source (centre of curvature, f < 0) rotated by +15 deg lies at y < 0:
    # element 1 fires first
    assert np.argmin(steered) == 0 and np.all(np.diff(steered) > 0)


@pytest.mark.parametrize(
    ("t", "match"),
    [
        (lin(5, "Custom"), "not supported"),
        (lin(5, "linear"), "not supported"),
        (lin(7, "RCA"), "even"),
        (Transducer(NumberOfElements=4.0, Pitch=1e-3), "no field Configuration"),
    ],
)
def test_delay_errors(t: Transducer, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ptr.compute_delays(Transmit(LateralFocus=math.inf, Angle=0.0), t, C)


def geometry(manual: bool = False) -> Geometry:
    g = config.reset_geometry()
    g.Domain.Manual = manual
    g.Domain.Margin = 1e-3
    return g


def test_domain_flat_array() -> None:
    g = geometry()
    out = ptr.compute_simulation_domain(g, lin(), Transmit(LateralFocus=math.inf, Angle=0.0))
    d = out.Domain
    assert d.TransducerSurface.shape == d.TransducerProjection.shape == (4, 3)
    assert d.Vertices.shape == (8, 3) and out.Center.shape == (3, 1)
    np.testing.assert_allclose(d.TransducerSurface[:, 1], [2.4e-3, -2.4e-3, -2.4e-3, 2.4e-3])
    assert d.Xmin == -1e-3 and d.Ymax == pytest.approx(2.4e-3 + 1e-3)
    end = ptr.compute_end_depth(g)
    assert d.Xmax == pytest.approx(end + 1e-3)
    assert out.Center[0, 0] == pytest.approx((g.startDepth + end) / 2)
    assert g.Domain.Xmax is None  # input unchanged
    assert d.present_fields()[:2] == ["Manual", "Margin"]


def test_domain_manual_box_grows() -> None:
    g = geometry(manual=True)
    d = g.Domain
    d.Xmin, d.Xmax, d.Ymin, d.Ymax, d.Zmin, d.Zmax = 0.0, 0.05, -1e-4, 1e-4, 0.0, 0.0
    out = ptr.compute_simulation_domain(g, lin(), Transmit(LateralFocus=math.inf, Angle=0.0)).Domain
    assert (out.Xmin, out.Xmax) == (-1e-3, 0.05)
    assert out.Ymax == pytest.approx(3.4e-3) and out.Zmin == pytest.approx(-3.5e-3)
    assert np.all(out.TransducerProjection[:, 0] == 0.05)


def test_domain_rca_uses_columns() -> None:
    tx = Transmit(LateralFocus=math.inf, Angle=0.0)
    rca = ptr.compute_simulation_domain(geometry(), lin(10, "RCA"), tx).Domain
    flat = ptr.compute_simulation_domain(geometry(), lin(5), tx).Domain
    np.testing.assert_array_equal(rca.TransducerSurface, flat.TransducerSurface)
    with pytest.raises(ValueError, match="even"):
        ptr.compute_simulation_domain(geometry(), lin(9, "RCA"), tx)


def test_domain_curved_array_warns_behind_apex() -> None:
    t = ptr.get_transducer_properties(Transducer(Type="C1-6D"))
    g = geometry()
    g.startDepth = 5e-3
    with pytest.warns(ptr.GeometryBehindTransducerWarning, match="apex"):
        out = ptr.compute_simulation_domain(g, t, Transmit(LateralFocus=math.inf, Angle=0.0))
    assert out.Domain.TransducerSurface.shape == (4 * 192, 3)
    el = ptr.get_element_geometry(t)
    assert out.Domain.Xmax > el.apex
    np.testing.assert_allclose(out.Domain.TransducerSurface[[1, -4], 0], 0.0, atol=1e-15)


def test_domain_custom_configuration() -> None:
    t = lin(5, "Custom")
    t.integration_points = np.array([[1e-3, -2e-3, -1e-3], [2e-3, 3e-3, 4e-3]])
    out = ptr.compute_simulation_domain(geometry(), t, Transmit(LateralFocus=10e-3, Angle=0.0))
    np.testing.assert_array_equal(out.Domain.TransducerSurface[:, 0], 1e-3)
    assert out.Domain.TransducerProjection[0, 1] == 3e-3
