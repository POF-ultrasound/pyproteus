"""Unit tests for ``get_element_geometry`` (no MATLAB): geometric invariants."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pyproteus import config
from pyproteus import transducer as ptr
from pyproteus.config.models import Transducer


def test_linear_array() -> None:
    el = ptr.get_element_geometry(
        Transducer(Configuration="Linear", NumberOfElements=4, Pitch=1e-3, ElementWidth=8e-4)
    )
    np.testing.assert_array_equal(el.y, [-1.5e-3, -0.5e-3, 0.5e-3, 1.5e-3])
    np.testing.assert_array_equal(el.x, np.zeros(4))
    np.testing.assert_array_equal(el.theta, np.zeros(4))
    assert (el.width, el.radius, el.apex, el.center, el.theta_max) == (
        8e-4,
        math.inf,
        0.0,
        -math.inf,
        0.0,
    )
    assert not el.curved and el.n_elements == 4


@pytest.mark.parametrize("n", [1, 2, 7, 192])
@pytest.mark.parametrize("radius", [0.01, 56.8e-3, 1.0])
def test_curved_array_invariants(n: int, radius: float) -> None:
    p = 0.35e-3 if n * 0.35e-3 < math.pi * radius else 1e-4
    el = ptr.get_element_geometry(
        Transducer(
            Configuration="Curved", NumberOfElements=n, Pitch=p, ElementWidth=1.0, Radius=radius
        )
    )
    assert el.curved and el.width == p  # ElementWidth is not used
    assert el.theta_max == pytest.approx(n * p / (2 * radius))
    # centres on the circle about (center, 0)
    np.testing.assert_allclose(np.hypot(el.x - el.center, el.y), radius, rtol=1e-14)
    # arc length between neighbouring centres is the pitch; symmetric about y = 0
    np.testing.assert_allclose(np.diff(el.theta) * radius, p, rtol=1e-12)
    np.testing.assert_allclose(el.theta, -el.theta[::-1], atol=1e-15)
    assert np.all(np.diff(el.y) > 0)  # element 1 most negative lateral
    # outer element edges lie at x = 0, apex at the arc centre
    edge = el.center + radius * math.cos(el.theta_max)
    assert edge == pytest.approx(0.0, abs=1e-15 * radius)
    assert el.apex == pytest.approx(el.center + radius)
    assert np.all(el.x <= el.apex + 1e-15 * radius)


def test_c1_6d_and_default() -> None:
    t = ptr.get_transducer_properties(Transducer(Type="C1-6D"))  # calibrated model file
    el = ptr.get_element_geometry(t)
    assert el.curved and el.n_elements == 192 and el.radius == t.Radius
    assert el.theta_max == 192 * t.Pitch / (2 * t.Radius) and el.width == t.Pitch
    d = ptr.get_element_geometry(config.reset_transducer())  # P4-1, linear
    assert not d.curved and d.n_elements == 96 and d.width == 0.245e-3


def test_configuration_must_be_exactly_curved() -> None:
    for conf in ("curved", "RCA", None):
        el = ptr.get_element_geometry(
            {
                "Configuration": conf,
                "NumberOfElements": 3.0,
                "Pitch": 1e-3,
                "ElementWidth": 1e-3,
                "Radius": 0.05,
            }
        )
        assert not el.curved


def test_is_curved_and_read_only_arrays() -> None:
    assert ptr.is_curved({"Configuration": "Curved"})
    assert not ptr.is_curved({"Configuration": "curved"}) and not ptr.is_curved({})
    assert not ptr.is_curved({"Configuration": ["Curved"]})
    el = ptr.get_element_geometry({"NumberOfElements": 3.0, "Pitch": 1e-3, "ElementWidth": 1e-3})
    with pytest.raises(ValueError, match="read-only"):
        el.x[0] = 1.0


def test_to_struct_shapes() -> None:
    s = ptr.get_element_geometry(
        Transducer(Configuration="Curved", NumberOfElements=5, Pitch=1e-3, Radius=0.05)
    ).to_struct()
    assert list(s) == ["theta", "x", "y", "width", "Radius", "Apex", "Center", "ThetaMax"]
    assert s["x"].shape == s["y"].shape == s["theta"].shape == (5, 1)


def test_empty_array() -> None:
    for conf in ("Linear", "Curved"):
        el = ptr.get_element_geometry(
            Transducer(
                Configuration=conf, NumberOfElements=0, Pitch=1e-3, ElementWidth=1e-3, Radius=0.05
            )
        )
        assert el.x.shape == el.y.shape == el.theta.shape == (0,)


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        ({"Configuration": "Curved", "Radius": 0.0}, "positive finite"),
        ({"Configuration": "Curved", "Radius": -1.0}, "positive finite"),
        ({"Configuration": "Curved", "Radius": math.inf}, "positive finite"),
        ({"Configuration": "Curved", "Radius": math.nan}, "positive finite"),
        ({"Configuration": "Curved", "Radius": np.array([0.05, 0.06])}, "positive finite"),
        ({"Configuration": "Curved", "Radius": 1e-3}, "shorter than pi"),
        ({"Configuration": "Curved"}, "no field Radius"),
        ({"Configuration": "Linear"}, "no field ElementWidth"),
        (
            {"Configuration": "Linear", "ElementWidth": 1e-3, "NumberOfElements": 2.5},
            "non-negative integer",
        ),
        (
            {"Configuration": "Linear", "ElementWidth": 1e-3, "NumberOfElements": -1.0},
            "non-negative integer",
        ),
        (
            {"Configuration": "Linear", "ElementWidth": 1e-3, "NumberOfElements": None},
            "no field NumberOfElements",
        ),
    ],
)
def test_errors(fields: dict, match: str) -> None:
    t = {"NumberOfElements": 8.0, "Pitch": 1e-3, **fields}
    with pytest.raises(ValueError, match=match):
        ptr.get_element_geometry(t)
