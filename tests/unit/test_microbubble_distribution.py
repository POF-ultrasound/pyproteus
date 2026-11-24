"""Unit tests (no MATLAB): Gaussian size distribution."""

from __future__ import annotations

import numpy as np
import pytest

from pyproteus import microbubble as pmb
from pyproteus.config.models import Distribution, Microbubble


def test_gaussian_distribution() -> None:
    r, p = pmb.gaussian_distribution(2.5e-6, 10.0)
    assert r.shape == p.shape == (1, 1000)
    assert r[0, 0] == 0.5e-6 and r[0, -1] == 6e-6
    assert p.sum() == pytest.approx(1.0, rel=1e-14)
    mean = float((r * p).sum())
    std = float(np.sqrt(((r - mean) ** 2 * p).sum()))
    assert mean == pytest.approx(2.5e-6, rel=1e-6) and std == pytest.approx(0.25e-6, rel=1e-3)
    _, q = pmb.gaussian_distribution(2.5e-6, -10.0)  # sign of PDI does not matter
    np.testing.assert_array_equal(p, q)


def test_gaussian_distribution_quirks_and_errors() -> None:
    _, p = pmb.gaussian_distribution(2.5e-6, 0.0)  # Q-84: NaN
    assert np.all(np.isnan(p))
    mb = Microbubble(Distribution=Distribution(MeanRadius=2e-6, PDI=20.0))
    np.testing.assert_array_equal(
        pmb.compute_gaussian_distribution(mb)[1], pmb.gaussian_distribution(2e-6, 20.0)[1]
    )
    with pytest.raises(ValueError, match="PDI"):
        pmb.compute_gaussian_distribution(Microbubble(Distribution=Distribution(MeanRadius=2e-6)))
    with pytest.raises(ValueError, match="Distribution"):
        pmb.compute_gaussian_distribution(Microbubble())
