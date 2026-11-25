"""Unit tests for ``pyproteus.microbubble.solver.dopri5_batch`` (per-system DOPRI5, D-22)."""

from __future__ import annotations

import numpy as np
import pytest

from pyproteus.microbubble.solver import dopri5_batch
from pyproteus.numerics import ode45, odeset


def decay(lam):
    def f(t, y, idx):
        return -lam[idx, None] * y

    return f


def test_single_system_matches_ode45_class() -> None:
    """Same published method and controller as ``ode45``: solutions within the D-13 bound."""
    lam = np.array([3.0])
    tg = np.linspace(0, 2, 41)
    r = dopri5_batch(decay(lam), tg[None], np.ones((1, 1)), rtol=1e-6, atol=1e-9, initial_step=1e-3)
    sol = ode45(
        lambda t, y: -3.0 * y, tg, [1.0], odeset(RelTol=1e-6, AbsTol=1e-9, InitialStep=1e-3)
    )
    np.testing.assert_allclose(r.y[0, :, 0], sol.y[:, 0], rtol=1e-5, atol=1e-9)
    np.testing.assert_allclose(r.y[0, :, 0], np.exp(-3 * tg), rtol=1e-5, atol=1e-9)
    assert r.n_out[0] == tg.size and not r.failures


@pytest.mark.parametrize("window", [(5.0, 6.5), (7.0, 8.0), (3.0, 3.6)])
def test_max_step_does_not_skip_a_silent_stretch(window) -> None:
    """y' = 1 on a window inside [0, 10]: MaxStep (span/10) stops it being stepped over."""
    a, b = window

    def f(t, y, idx):
        return ((t >= a) & (t <= b)).astype(float)[:, None]

    tg = np.linspace(0, 10, 11)
    r = dopri5_batch(f, tg[None], np.zeros((1, 1)), rtol=1e-6, atol=1e-6, initial_step=1e-3)
    np.testing.assert_allclose(r.y[0, -1, 0], b - a, rtol=1e-3)
    assert r.nsteps[0] >= 10


def test_never_steps_past_the_end() -> None:
    seen = []

    def f(t, y, idx):
        seen.append(t.max())
        return np.ones_like(y)

    tg = np.array([[0.0, 0.3, 1.0]])
    r = dopri5_batch(f, tg, np.zeros((1, 1)), rtol=1e-6, atol=1e-6, initial_step=0.5)
    assert max(seen) <= 1.0
    np.testing.assert_allclose(r.y[0, :, 0], tg[0], rtol=1e-12)


def test_sparse_and_unequal_grids_and_batch_invariance() -> None:
    """> 16 outputs per step, different grids per system; batch == single bit for bit."""
    lam = np.array([0.5, 2.0, 0.1])
    tg = np.vstack([np.linspace(0, 1, 401), np.linspace(0, 3, 401), np.linspace(0, 0.2, 401)])
    y0 = np.array([[1.0], [2.0], [-1.0]])
    r = dopri5_batch(decay(lam), tg, y0, rtol=1e-8, atol=1e-10, initial_step=0.1)
    for i in range(3):
        np.testing.assert_allclose(r.y[i, :, 0], y0[i, 0] * np.exp(-lam[i] * tg[i]), rtol=1e-7)
        one = dopri5_batch(decay(lam[i : i + 1]), tg[i : i + 1], y0[i : i + 1], rtol=1e-8,
                           atol=1e-10, initial_step=0.1)  # fmt: skip
        np.testing.assert_array_equal(one.y[0], r.y[i])
    assert r.nsteps.max() < 400  # many outputs per step on these smooth problems


def test_max_steps_counts_only_unfinished() -> None:
    lam = np.array([1.0])
    tg = np.linspace(0, 1, 3)[None]
    ref = dopri5_batch(decay(lam), tg, np.ones((1, 1)), rtol=1e-6, atol=1e-6, initial_step=0.01)
    n = int(ref.nsteps[0] + ref.nfailed[0])
    ok = dopri5_batch(decay(lam), tg, np.ones((1, 1)), rtol=1e-6, atol=1e-6, initial_step=0.01,
                      max_steps=n)  # fmt: skip
    assert not ok.failures
    bad = dopri5_batch(decay(lam), tg, np.ones((1, 1)), rtol=1e-6, atol=1e-6, initial_step=0.01,
                       max_steps=n - 1)  # fmt: skip
    assert [f.index for f in bad.failures] == [0] and np.isnan(bad.y[0, -1, 0])


def test_nan_in_one_system_does_not_touch_others() -> None:
    def f(t, y, idx):
        out = -y.copy()
        out[(idx == 1) & (t > 0.5)] = np.nan
        return out

    tg = np.tile(np.linspace(0, 1, 11), (3, 1))
    r = dopri5_batch(f, tg, np.ones((3, 1)), rtol=1e-6, atol=1e-6, initial_step=1e-3)
    assert [x.index for x in r.failures] == [1]
    assert "underflow" in r.failures[0].reason and 0.4 < r.failures[0].t <= 0.5 + 1e-9
    clean = dopri5_batch(decay(np.ones(3)), tg, np.ones((3, 1)), rtol=1e-6, atol=1e-6,
                         initial_step=1e-3)  # fmt: skip
    np.testing.assert_array_equal(r.y[[0, 2]], clean.y[[0, 2]])
