"""Unit tests for numerics.sind/cosd (degrees, exact at multiples of 90)."""

from __future__ import annotations

import numpy as np

from pyproteus.numerics import cosd, sind


def test_exact_at_multiples_of_90() -> None:
    k = np.arange(-12, 13)
    x = 90.0 * k
    s = np.array([0.0, 1.0, 0.0, -1.0])[np.mod(k, 4)]
    c = np.array([1.0, 0.0, -1.0, 0.0])[np.mod(k, 4)]
    np.testing.assert_array_equal(sind(x), s)
    np.testing.assert_array_equal(cosd(x), c)
    assert not np.signbit(sind(x)[s == 0]).any() and not np.signbit(cosd(x)[c == 0]).any()
    assert sind(90.0 * 1e12) == 0.0 and cosd(-0.0) == 1.0


def test_generic_angles() -> None:
    x = np.linspace(-1000, 1000, 20001) + 0.37
    np.testing.assert_allclose(sind(x), np.sin(np.deg2rad(x)), rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(cosd(x), np.cos(np.deg2rad(x)), rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(sind(30.0), 0.5, rtol=1e-15)


def test_nonfinite_and_shapes() -> None:
    assert np.isnan(sind(np.inf)) and np.isnan(cosd(-np.inf)) and np.isnan(sind(np.nan))
    assert sind(np.zeros((2, 3))).shape == (2, 3) and np.ndim(cosd(1.0)) == 0
