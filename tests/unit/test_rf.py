"""``compute_RF`` on its own terms.

Delays, apodization, quadrature weights, impulse response, GPU fallback.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from pyproteus.acoustic import compute_RF, compute_RF_data, largest_prime_factor, rf_signal_length
from pyproteus.acoustic import rf as rfmod

DT = 1 / 40e6


def transducer(
    n_el: int, n_int: int, delays=None, apod=None, weights=1.0, ir=None, fs=None
) -> dict:
    return {
        "integration_points": np.zeros((n_el, n_int, 3)),
        "integration_receive_apodization": np.ones((n_el, n_int)) if apod is None else apod,
        "integration_receive_delays": np.zeros((n_el, n_int)) if delays is None else delays,
        "integration_weights": weights,
        "ReceiveImpulseResponse": np.array([1.0]) if ir is None else ir,
        "SamplingRate": 1 / DT if fs is None else fs,
    }


def echo(m: int, t0: float) -> np.ndarray:
    t = np.arange(m) * DT
    return np.sin(2 * np.pi * 3e6 * (t - t0)) * np.exp(-(((t - t0) / 3e-7) ** 2))


def test_rf_signal_length() -> None:
    # smallest largest prime factor within N + (0:max(10, round(0.05 N))), first on ties
    assert rf_signal_length(100, 0.0, DT) == 108  # 108 = 2^2 3^3 beats 100 = 2^2 5^2
    assert rf_signal_length(97, 0.0, DT) == 100  # 97..107: 100 is the only 5-smooth size
    assert rf_signal_length(100, 2.5 * DT, DT) == 108  # N = 103
    assert rf_signal_length(334, 0.0, DT) == 360  # no 5-smooth size in 334..351: grow past it
    for m in range(1, 3000, 37):
        n = rf_signal_length(m, 0.0, DT)
        assert n >= m and largest_prime_factor(n) <= 5


def test_integer_sample_delays_shift_the_signal() -> None:
    """A delay of k samples shifts the element pressure by k samples (circular on N)."""
    m = 200
    s = echo(m, 1.5e-6)
    k = np.array([[0, 3], [7, 12]], dtype=float)  # 2 elements × 2 points
    T = transducer(2, 2, delays=k * DT)
    W = np.eye(4)  # sensor j = row j (column-major (element, point))
    p = s[None, :] * np.array([[1.0], [2.0], [3.0], [4.0]])
    out = compute_RF(T, {"p": p}, W, {"dt": DT}, {"DATA_CAST_RF": "double", "RF_type": "pressure"})
    n = rf_signal_length(m, 12 * DT, DT)
    assert out.shape == (2, n)
    padded = np.zeros(n)
    padded[:m] = s
    # rows of W (column-major): (el0,pt0), (el1,pt0), (el0,pt1), (el1,pt1)
    exp0 = 0.5 * (1 * np.roll(padded, 0) + 3 * np.roll(padded, 3))
    exp1 = 0.5 * (2 * np.roll(padded, 7) + 4 * np.roll(padded, 12))
    np.testing.assert_allclose(out[0], exp0, atol=1e-12)
    np.testing.assert_allclose(out[1], exp1, atol=1e-12)


def test_fractional_delay_is_band_limited_shift() -> None:
    m = 300
    T = transducer(1, 1, delays=np.array([[2.37 * DT]]))
    out = compute_RF(
        T,
        {"p": echo(m, 2e-6)[None]},
        np.eye(1),
        {"dt": DT},
        {"DATA_CAST_RF": "double", "RF_type": "pressure"},
    )
    exact = echo(out.shape[1], 2e-6 + 2.37 * DT)
    np.testing.assert_allclose(out[0], exact, atol=1e-9)


def test_apodization_and_weights_normalisation() -> None:
    m = 64
    s = echo(m, 0.6e-6)
    apod = np.array([[1.0, 0.0, 0.5]])
    w = np.array([[1.0, 2.0, 1.0]])
    T = transducer(1, 3, apod=apod, weights=w)
    p = np.vstack([s, 10 * s, 2 * s])
    out = compute_RF(
        T, {"p": p}, np.eye(3), {"dt": DT}, {"DATA_CAST_RF": "double", "RF_type": "pressure"}
    )
    # weights / sum = [0.25, 0.5, 0.25]; point 2 switched off
    np.testing.assert_allclose(out[0, :m], 0.25 * s + 0.25 * 0.5 * 2 * s, atol=1e-12)
    # scalar weight: plain mean over the points
    T = transducer(1, 3, weights=7.0)
    out = compute_RF(
        T, {"p": p}, np.eye(3), {"dt": DT}, {"DATA_CAST_RF": "double", "RF_type": "pressure"}
    )
    np.testing.assert_allclose(out[0, :m], (1 + 10 + 2) / 3 * s, atol=1e-12)


def test_impulse_response_convolution() -> None:
    m = 50
    s = echo(m, 0.4e-6)
    ir = np.array([1.0, -0.5, 0.25])
    T = transducer(1, 1, ir=ir)
    rf = compute_RF(T, {"p": s[None]}, np.eye(1), {"dt": DT}, {"DATA_CAST_RF": "double"})
    n = rf_signal_length(m, 0.0, DT)
    padded = np.zeros(n)
    padded[:m] = s
    np.testing.assert_allclose(rf[0], np.convolve(padded, ir) * DT, atol=1e-15)
    assert rf.shape == (1, n + ir.size - 1)


def test_sparse_and_dense_weights_and_dtypes() -> None:
    rng = np.random.default_rng(0)
    W = rng.random((6, 5)) * (rng.random((6, 5)) > 0.5)
    p = rng.standard_normal((5, 77))
    T = transducer(
        3, 2, delays=rng.uniform(0, 5 * DT, (3, 2)), ir=np.array([0.5, 1.0, 0.5]), fs=2 / DT
    )
    dense = compute_RF(T, {"p": p}, W, {"dt": DT}, {"DATA_CAST_RF": "double"})
    sparse = compute_RF(T, p, sp.csr_matrix(W), {"dt": DT}, {"DATA_CAST_RF": "double"})
    np.testing.assert_allclose(sparse, dense, rtol=0, atol=1e-14 * np.max(np.abs(dense)))
    single = compute_RF(T, {"p": p.astype(np.float32)}, W, {"dt": DT}, {"DATA_CAST_RF": "single"})
    assert single.dtype == np.float32 and dense.dtype == np.float64
    assert np.max(np.abs(single - dense)) <= 1e-6 * np.max(np.abs(dense))


def test_block_size_does_not_change_results(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(1)
    T = transducer(5, 3, delays=rng.uniform(0, 4 * DT, (5, 3)), apod=rng.random((5, 3)))
    W = rng.random((15, 8))
    p = rng.standard_normal((8, 60))
    ref = compute_RF(T, {"p": p}, W, {"dt": DT}, {"DATA_CAST_RF": "double"})
    monkeypatch.setattr(rfmod, "_MAX_ARRAY_ELEMENTS", 1)  # one element per block
    np.testing.assert_allclose(
        compute_RF(T, {"p": p}, W, {"dt": DT}, {"DATA_CAST_RF": "double"}),
        ref,
        rtol=0,
        atol=1e-15 * np.max(np.abs(ref)),
    )


def test_bad_cast() -> None:
    with pytest.raises(ValueError, match="DATA_CAST_RF"):
        compute_RF(
            transducer(1, 1), np.zeros((1, 4)), np.eye(1), {"dt": DT}, {"DATA_CAST_RF": "half"}
        )


def test_compute_RF_data_falls_back_to_cpu_on_gpu_oom(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    class FakeOOM(Exception):
        pass

    def fake_compute(T, sd, W, G, rp):
        calls.append(rp["DATA_CAST_RF"])
        if rp["DATA_CAST_RF"].startswith("gpuArray"):
            raise FakeOOM
        return np.ones((1, 1))

    monkeypatch.setattr(rfmod, "compute_RF", fake_compute)
    monkeypatch.setattr(rfmod, "_is_gpu_oom", lambda exc: isinstance(exc, FakeOOM))
    rp = {"DATA_CAST_RF": "gpuArray-single", "other": 1}
    with pytest.warns(ResourceWarning, match="Switching to CPU"):
        _, rp2 = compute_RF_data(None, None, None, None, rp)
    assert calls == ["gpuArray-single", "single"]
    assert rp2 == {"DATA_CAST_RF": "single", "other": 1}
    assert rp["DATA_CAST_RF"] == "gpuArray-single"  # caller's struct unchanged (value semantics)


def test_compute_RF_data_reraises_other_errors() -> None:
    with pytest.raises(ValueError):
        compute_RF_data(
            transducer(1, 1), np.zeros((1, 4)), np.eye(1), {"dt": DT}, {"DATA_CAST_RF": "bad"}
        )
