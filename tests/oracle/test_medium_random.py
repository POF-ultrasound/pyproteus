"""Oracle tests: ``random_truncated_normal`` (D-13 float class, same MT19937 stream).

The draws use the same documented ``rand`` stream; ``erfc``/``erfcinv`` are
SciPy's, so values agree at ≤ 1e-9 rel (absolute floor 1e-12 for draws near 0).
The number of uniforms consumed must match (the next ``rand`` is equal).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pyproteus import config, medium
from pyproteus import numerics as nm
from tests.oracle.helpers import ORACLE_SETTINGS, get, put, row
from tests.oracle.test_acoustic_grid import assert_float

ATOL = 1e-12


def _matlab_draw(eng: Any, seed: int, cutoff: float, size: list[int]) -> tuple[np.ndarray, float]:
    put(eng, pyp_c__=float(cutoff), pyp_sz__=row(np.asarray(size, dtype=float)))
    eng.eval(
        f"rng({seed},'twister'); pyp_x__ = random_truncated_normal(pyp_c__, pyp_sz__);"
        "pyp_next__ = rand;",
        nargout=0,
    )
    x, nxt = get(eng, "pyp_x__", "pyp_next__")
    eng.eval("clear pyp_c__ pyp_sz__ pyp_x__ pyp_next__; rng(0,'twister');", nargout=0)
    return x, float(nxt.item())


def _check(eng: Any, seed: int, cutoff: float, size: list[int]) -> None:
    ref, ref_next = _matlab_draw(eng, seed, cutoff, size)
    g = nm.rng(seed)
    mine = medium.random_truncated_normal(cutoff, size, g)
    assert_float(mine, ref, atol=ATOL, err_msg=f"seed {seed} cutoff {cutoff} size {size}")
    assert g.rand().item() == ref_next  # same number of uniforms consumed
    assert np.all(np.abs(mine) <= cutoff)


CUTOFFS = [1e-3, 0.5, 1, 2, 2.5, 3, 5, 8.2, 20, 37.5]
SIZES = [[2, 3, 4], [1, 5], [7, 1], [3, 0], [1, 1], [4, 5, 1], [2, 2, 2, 2]]


@pytest.mark.parametrize("cutoff", CUTOFFS)
def test_fixed_cases(matlab_engine: Any, cutoff: float) -> None:
    for seed in (0, 1, 42):
        for size in SIZES:
            _check(matlab_engine, seed, cutoff, size)


def test_scalar_size_is_square(matlab_engine: Any) -> None:
    """``sz = n`` gives ``n × n`` as MATLAB ``rand(n)``."""
    put(matlab_engine, pyp_c__=3.0)
    matlab_engine.eval(
        "rng(3,'twister'); pyp_x__ = random_truncated_normal(pyp_c__, 4);", nargout=0
    )
    ref = get(matlab_engine, "pyp_x__")
    matlab_engine.eval("clear pyp_*; rng(0,'twister');", nargout=0)
    assert_float(medium.random_truncated_normal(3.0, 4, nm.rng(3)), ref, atol=ATOL)


def test_large_background(matlab_engine: Any) -> None:
    """A define_medium-sized draw (odd, non-cubic grid)."""
    _check(matlab_engine, 7, 3.0, [61, 47, 33])


@ORACLE_SETTINGS
@given(
    st.integers(0, 2**32 - 1),
    st.floats(1e-6, 40.0),
    st.lists(st.integers(0, 6), min_size=2, max_size=3),
)
def test_random_cases(matlab_engine: Any, seed: int, cutoff: float, size: list[int]) -> None:
    _check(matlab_engine, seed, cutoff, size)


@pytest.mark.parametrize("cutoff", [0.0, -1.0, float("nan"), [1.0, 2.0]])
def test_invalid_cutoff(matlab_engine: Any, cutoff: Any) -> None:
    import matlab.engine

    arg = row(np.asarray(cutoff, dtype=float)) if isinstance(cutoff, list) else cutoff
    put(matlab_engine, pyp_c__=arg)
    with pytest.raises(matlab.engine.MatlabExecutionError, match="positive scalar"):
        matlab_engine.eval("random_truncated_normal(pyp_c__, [2 2]);", nargout=0)
    matlab_engine.eval("clear pyp_c__", nargout=0)
    with pytest.raises(ValueError, match="positive scalar"):
        medium.random_truncated_normal(cutoff, [2, 2], nm.rng(0))


def test_matches_toolbox_baseline(oracle: Any) -> None:
    """PROTEUS' recorded Statistics Toolbox draws (``tests/baseline/truncated_normal.mat``).

    Data read from a file; the comparison is the float class.
    """
    path = Path(oracle.path) / "tests" / "baseline" / "truncated_normal.mat"
    cases = config.loadmat(path)["cases"]
    elements = cases.elements.reshape(-1, order="F")
    assert elements.size > 0
    for c in elements:
        seed = int(c["seed"].item())
        cutoff = float(c["cutoff"].item())
        size = [int(v) for v in c["size"].reshape(-1)]
        g = nm.rng(seed)
        mine = medium.random_truncated_normal(cutoff, size, g)
        assert_float(mine, c["x"], atol=ATOL, err_msg=f"seed {seed} cutoff {cutoff} {size}")
        assert g.rand().item() == float(c["next"].item())
