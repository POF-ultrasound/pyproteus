"""Unit tests for ``pyproteus.numerics.colon`` (MATLAB ``a:d:b``)."""

from __future__ import annotations

import numpy as np
import pytest

from pyproteus.numerics import colon


@pytest.mark.parametrize(
    "a,d,b,n",
    [
        (0, 0.1, 0.3, 4),  # (b-a)/d is 1 ulp below 3: snapped
        (0, 1, 3, 4),
        (0, 1, 3 - 1e-14, 3),  # genuinely short
        (0, 0.25, 0.9, 4),
        (1, 1, 0, 0),
        (0, 0.1, 0.0, 1),
        (1, -0.25, 0, 5),
        (0, -1, 1, 0),
        (0, 0, 1, 0),
        (-2.4e-3, 1540 / 2.5e6 / 5, 2.4e-3, 39),
    ],
)
def test_colon_length_and_values(a, d, b, n) -> None:
    out = colon(a, d, b)
    assert out.size == n
    np.testing.assert_array_equal(out, a + np.arange(n) * d)


def test_colon_never_far_past_b() -> None:
    rng = np.random.default_rng(0)
    for _ in range(2000):
        a, b = sorted(rng.uniform(-1, 1, 2))
        d = 10 ** rng.uniform(-4, 0)
        out = colon(a, d, b)
        assert out[-1] <= b + 8 * np.finfo(float).eps * max(abs(a), abs(b), d)


def test_colon_rejects_non_finite() -> None:
    with pytest.raises(ValueError):
        colon(0, 1, np.inf)
    with pytest.raises(ValueError):
        colon(np.nan, 1, 2)
