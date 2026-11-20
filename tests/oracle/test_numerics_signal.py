"""Oracle tests: ifft(...,'symmetric'), hilbert, conv/convn.

FFT-based results use pocketfft (NumPy) vs FFTW (MATLAB): class ≤ 1e-12
relative to the signal maximum. Convolutions are direct sums: exact for
integer-valued data, ≤ 1e-12 relative otherwise.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from pyproteus import numerics as nm
from tests.oracle.helpers import (
    ORACLE_SETTINGS,
    assert_rel,
    assert_same_float,
    batch,
    col,
    get,
    put,
    row,
)

ORACLE = ORACLE_SETTINGS
reals = st.floats(-1e3, 1e3, allow_nan=False)


@pytest.mark.parametrize("n", [1, 2, 3, 8, 9, 64, 101])
def test_ifft_symmetric_nonsymmetric_input(matlab_engine, n) -> None:
    # Deliberately NOT conjugate-symmetric: MATLAB must ignore the upper half.
    rs = np.random.RandomState(n)
    X = rs.standard_normal((3, n)) + 1j * rs.standard_normal((3, n))
    put(matlab_engine, X=X)
    matlab_engine.eval("pyp_r = ifft(X, [], 2, 'symmetric');", nargout=0)
    ref = get(matlab_engine, "pyp_r")
    assert_rel(nm.ifft_symmetric(X, axis=1), ref)
    matlab_engine.eval("pyp_r = ifft(X.', [], 1, 'symmetric');", nargout=0)
    assert_rel(nm.ifft_symmetric(X.T, axis=0), get(matlab_engine, "pyp_r"))


def test_ifft_symmetric_proteus_pattern(matlab_engine) -> None:
    # compute_RF / define_source_transducer: fft(p, N, 2) ... ifft(p, [], 2, 'symmetric')
    rs = np.random.RandomState(0)
    p = rs.standard_normal((4, 37))
    N = 64
    put(matlab_engine, p=p, N=float(N))
    matlab_engine.eval(
        "P = fft(p, N, 2); P = P .* exp(-1i*(0:N-1)/7); pyp_r = ifft(P, [], 2, 'symmetric');",
        nargout=0,
    )
    P = np.fft.fft(p, N, axis=1) * np.exp(-1j * np.arange(N) / 7)
    assert_rel(nm.ifft_symmetric(P, axis=1), get(matlab_engine, "pyp_r"))


@pytest.mark.parametrize("n_arg", [5, 8, 12])
def test_ifft_symmetric_n(matlab_engine, n_arg) -> None:
    rs = np.random.RandomState(n_arg)
    X = rs.standard_normal((2, 8)) + 1j * rs.standard_normal((2, 8))
    put(matlab_engine, X=X, n=float(n_arg))
    matlab_engine.eval("pyp_r = ifft(X, n, 2, 'symmetric');", nargout=0)
    assert_rel(nm.ifft_symmetric(X, n_arg, axis=1), get(matlab_engine, "pyp_r"))


@ORACLE
@given(
    st.lists(
        hnp.arrays(
            np.float64, hnp.array_shapes(min_dims=2, max_dims=2, max_side=40), elements=reals
        ),
        min_size=1,
        max_size=8,
    )
)
def test_hilbert_matrices(matlab_engine, arrays) -> None:
    (H,) = batch(matlab_engine, "@(x) hilbert(x)", arrays)
    for a, h in zip(arrays, H, strict=True):
        assert_rel(nm.hilbert(a), h.reshape(a.shape), err_msg=str(a.shape))


@pytest.mark.parametrize("n", [1, 2, 7, 8, 129])
def test_hilbert_vectors(matlab_engine, n) -> None:
    v = np.random.RandomState(n).standard_normal(n)
    (H,) = batch(matlab_engine, "@(x) hilbert(x)", [row(v), col(v)])
    assert_rel(nm.hilbert(v.reshape(1, -1)), H[0].reshape(1, -1))
    assert_rel(nm.hilbert(v.reshape(-1, 1)), H[1].reshape(-1, 1))
    assert_rel(nm.hilbert(v), H[0].reshape(-1))


# --- conv / convn ------------------------------------------------------------------


@ORACLE
@given(
    st.lists(st.integers(-9, 9).map(float), min_size=1, max_size=30),
    st.lists(st.integers(-9, 9).map(float), min_size=1, max_size=8),
    st.sampled_from(["full", "same", "valid"]),
)
def test_conv_integer_exact(matlab_engine, u, v, shape) -> None:
    put(matlab_engine, u=row(u), v=row(v))
    matlab_engine.eval(f"pyp_r = conv(u, v, '{shape}');", nargout=0)
    ref = np.asarray(get(matlab_engine, "pyp_r")).reshape(-1)
    assert_same_float(nm.conv(u, v, shape) + 0.0, ref + 0.0)  # +0.0: -0 vs 0 irrelevant


@ORACLE
@given(
    hnp.arrays(
        np.float64, hnp.array_shapes(min_dims=1, max_dims=3, min_side=1, max_side=7), elements=reals
    ),
    hnp.arrays(
        np.float64, hnp.array_shapes(min_dims=1, max_dims=3, min_side=1, max_side=4), elements=reals
    ),
    st.sampled_from(["full", "same", "valid"]),
)
def test_convn_random(matlab_engine, a, b, shape) -> None:
    # MATLAB has no 1-D arrays: give 1-D inputs an explicit trailing singleton.
    a2 = a.reshape(-1, 1) if a.ndim == 1 else a
    b2 = b.reshape(-1, 1) if b.ndim == 1 else b
    put(matlab_engine, a=a2, b=b2)
    matlab_engine.eval(f"pyp_r = convn(a, b, '{shape}');", nargout=0)
    ref = np.asarray(get(matlab_engine, "pyp_r"))
    mine = nm.convn(a2, b2, shape)
    # MATLAB drops trailing singleton dimensions of the result.
    nd = max(mine.ndim, ref.ndim)
    mine = mine.reshape(mine.shape + (1,) * (nd - mine.ndim))
    ref = ref.reshape(ref.shape + (1,) * (nd - ref.ndim))
    if ref.size == 0:
        assert mine.size == 0
        return
    assert_rel(mine, ref)


def test_convn_smooth_distribution_pattern(matlab_engine) -> None:
    # smooth_distribution: convn(P, Iconv, 'same') with P having one singleton axis
    rs = np.random.RandomState(3)
    P = rs.random_sample((1, 21, 16))
    K = rs.random_sample((1, 6, 5))
    put(matlab_engine, P=P, K=K)
    matlab_engine.eval("pyp_r = convn(P, K, 'same');", nargout=0)
    assert_rel(nm.convn(P, K, "same"), get(matlab_engine, "pyp_r"))
    # compute_RF: convn(p, IR) with a row kernel acting along axis 1
    p = rs.standard_normal((5, 40))
    ir = rs.standard_normal((1, 9))
    put(matlab_engine, p=p, ir=ir)
    matlab_engine.eval("pyp_r = convn(p, ir);", nargout=0)
    assert_rel(nm.convn(p, ir), get(matlab_engine, "pyp_r"))
