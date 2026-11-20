"""Oracle tests: rng(seed,'twister') + rand/randi (comparison class: bit-identical)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pyproteus import numerics as nm
from tests.oracle.helpers import ORACLE_SETTINGS, assert_same_float, get

SEEDS = [0, 1, 5, 42, 5489, 12345, 2**31 - 1, 2**32 - 1]


@pytest.mark.parametrize("seed", SEEDS)
def test_rand_shapes_and_sequence(matlab_engine, seed) -> None:
    matlab_engine.eval(
        f"rng({seed},'twister'); a = rand(3,4); b = rand(2,3,4); c = rand(5); d = rand;"
        "e = rand(1,7); f = rand([2 5]);",
        nargout=0,
    )
    g = nm.rng(seed)
    for name, mine in [
        ("a", g.rand(3, 4)),
        ("b", g.rand(2, 3, 4)),
        ("c", g.rand(5)),
        ("d", g.rand()),
        ("e", g.rand(1, 7)),
        ("f", g.rand([2, 5])),
    ]:
        ref = np.asarray(get(matlab_engine, name))
        assert_same_float(mine, ref, err_msg=f"seed {seed} {name}")


@pytest.mark.parametrize("seed", SEEDS)
def test_randi_interleaved(matlab_engine, seed) -> None:
    matlab_engine.eval(
        f"rng({seed},'twister'); a = randi(10, 4, 3); b = rand(2,2); c = randi([3 7], 1, 9);"
        "d = randi(2^31, 3, 1); e = randi([-5 5], 2, 2, 2); f = randi(1, 2, 3);",
        nargout=0,
    )
    g = nm.rng(seed)
    for name, mine in [
        ("a", g.randi(10, 4, 3)),
        ("b", g.rand(2, 2)),
        ("c", g.randi([3, 7], 1, 9)),
        ("d", g.randi(2**31, 3, 1)),
        ("e", g.randi([-5, 5], 2, 2, 2)),
        ("f", g.randi(1, 2, 3)),
    ]:
        ref = np.asarray(get(matlab_engine, name))
        assert_same_float(mine, ref, err_msg=f"seed {seed} {name}")


@ORACLE_SETTINGS
@given(
    st.integers(0, 2**32 - 1),
    st.lists(
        st.tuples(
            st.sampled_from(["rand", "randi"]),
            st.integers(1, 1000),
            st.lists(st.integers(0, 6), min_size=2, max_size=3),
        ),
        min_size=1,
        max_size=6,
    ),
)
def test_random_call_sequences(matlab_engine, seed, calls) -> None:
    script = [f"rng({seed},'twister'); pyp_out = {{}};"]
    g = nm.rng(seed)
    mine = []
    for kind, imax, size in calls:
        dims = ",".join(str(s) for s in size)
        if kind == "rand":
            script.append(f"pyp_out{{end+1}} = rand({dims});")
            mine.append(g.rand(*size))
        else:
            script.append(f"pyp_out{{end+1}} = randi({imax},{dims});")
            mine.append(g.randi(imax, *size))
    matlab_engine.eval(" ".join(script), nargout=0)
    refs = matlab_engine.workspace["pyp_out"]
    for m, r, call in zip(mine, refs, calls, strict=True):
        ref = np.asarray(r).reshape(m.shape) if m.size else np.asarray(r)
        assert m.size == ref.size, call
        if m.size:
            assert_same_float(m, ref, err_msg=str(call))
