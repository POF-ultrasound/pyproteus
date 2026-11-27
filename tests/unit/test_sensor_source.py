"""Unit tests: k-Wave sensors and transducer source (no MATLAB; oracle tests in
``tests/oracle/test_sensor.py`` and ``tests/oracle/test_source.py``)."""

from __future__ import annotations

import numpy as np
import pytest

from pyproteus import acoustic
from pyproteus.acoustic import grid as grid_mod
from pyproteus.acoustic import sensor as sensor_mod
from pyproteus.backends import kwave as kw
from pyproteus.medium import voxelize_media_points
from pyproteus.numerics import ifft_symmetric, ind2sub, sub2ind


def make_grid(on_grid: bool = False, n=(13, 12, 7), d=2.0**-13, full=(45, 42, 37)):
    x, y, z = (
        np.arange(s, s + k, dtype=float) * d
        for s, k in zip((-2, -n[1] // 2, -n[2] // 2), n, strict=True)
    )
    return grid_mod.Grid(
        x=x, y=y, z=z, Nx=n[0], Ny=n[1], Nz=n[2], dx=d, dy=d, dz=d, dt=1e-8,
        full_size=full, PML=grid_mod.PML(1, 1, 1), sensor_on_grid=on_grid,
    )  # fmt: skip


def points(g, rs, n=20):
    lo = np.array([g.x[0], g.y[0], g.z[0]])
    hi = np.array([g.x[-1], g.y[-1], g.z[-1]])
    P = lo + rs.random_sample((n, 3)) * (hi - lo)
    P[1] = P[0]
    P[2] = [g.x[3], g.y[2], g.z[1]]
    P[3] = [g.x[3], g.y[2] + 0.5 * g.dy, g.z[0] + 0.25 * g.dz]
    P[4] = hi
    return voxelize_media_points(P, g)


def reference_loop(mask, P, idx, g, th=4):
    """Literal loop over points as ``update_sensor.m`` (independent of the vectorised code)."""
    shape = mask.shape
    axes = (g.x, g.y, g.z)
    mask = mask.copy()
    boxes = []
    for p, k in zip(P, idx, strict=True):
        sub = ind2sub(shape, k)
        ranges = []
        for dim in range(3):
            s = int(sub[dim])
            r = [s] if p[dim] == axes[dim][s] else list(range(s - th, s + th + 1))
            ranges.append([v for v in r if 0 <= v < shape[dim]])
        ii, jj, kk = np.meshgrid(*ranges, indexing="ij")
        lin = sub2ind(shape, ii.ravel(order="F"), jj.ravel(order="F"), kk.ravel(order="F"))
        boxes.append((lin, np.vstack([g.x[ii.ravel(order="F")], g.y[jj.ravel(order="F")],
                                      g.z[kk.ravel(order="F")]])))  # fmt: skip
        mask[np.unravel_index(lin, shape, order="F")] = 1
    mask_idx = np.flatnonzero(mask.ravel(order="F"))
    W = np.zeros((len(P), mask_idx.size))
    for n, (lin, xyz) in enumerate(boxes):
        b = acoustic.evaluate_delta_function(xyz, P[n], [g.dx, g.dy, g.dz], list(g.full_size))
        W[n, np.searchsorted(mask_idx, lin)] = b
    return mask, W


@pytest.mark.parametrize("existing", [False, True])
def test_update_sensor_matches_loop(existing: bool) -> None:
    g = make_grid()
    rs = np.random.RandomState(0)
    vox = points(g, rs)
    m0 = (rs.random_sample(g.shape) < 0.03) if existing else np.zeros(g.shape, bool)
    s, w = acoustic.update_sensor(acoustic.Sensor(m0), vox.points, vox.index, g)
    mask_ref, W_ref = reference_loop(m0, vox.points, vox.index, g)
    np.testing.assert_array_equal(s.mask, mask_ref)
    np.testing.assert_array_equal(w.toarray(), W_ref)
    assert isinstance(w, type(w.tocsr())) and w.format == "csr"
    np.testing.assert_array_equal(m0, m0 & s.mask)  # input mask kept, not modified


def test_update_sensor_chunking_invariant(monkeypatch: pytest.MonkeyPatch) -> None:
    g = make_grid()
    vox = points(g, np.random.RandomState(1), n=30)
    s, w = acoustic.update_sensor(acoustic.Sensor(np.zeros(g.shape)), vox.points, vox.index, g)
    monkeypatch.setattr(sensor_mod, "_CHUNK_ELEMENTS", 1000)  # 1 point per chunk
    s2, w2 = acoustic.update_sensor(acoustic.Sensor(np.zeros(g.shape)), vox.points, vox.index, g)
    np.testing.assert_array_equal(s.mask, s2.mask)
    np.testing.assert_array_equal(w.toarray(), w2.toarray())
    assert (
        acoustic.update_sensor_fast(acoustic.Sensor(np.zeros(g.shape)), vox.points, vox.index, g)[
            1
        ].nnz
        == w.nnz
    )


def test_update_sensor_on_grid_and_point_on_node() -> None:
    g = make_grid(on_grid=True)
    vox = points(g, np.random.RandomState(2))
    s, w = acoustic.update_sensor(acoustic.Sensor(np.zeros(g.shape, bool)), vox.points,
                                  vox.index, g)  # fmt: skip
    assert s.mask.dtype == bool
    assert s.mask.sum() == np.unique(vox.index).size
    np.testing.assert_array_equal(w.sum(axis=1).A.ravel(), 1.0)
    cols = np.flatnonzero(s.mask.ravel(order="F"))[w.indices]
    np.testing.assert_array_equal(cols, vox.index)
    # off-grid: a point exactly on a node is one sensor with weight 1
    g2 = make_grid(on_grid=False)
    P = np.array([[g2.x[4], g2.y[5], g2.z[3]]])
    v = voxelize_media_points(P, g2)
    s2, w2 = acoustic.update_sensor(acoustic.Sensor(np.zeros(g2.shape)), v.points, v.index, g2)
    assert s2.mask.sum() == 1 and w2.toarray().tolist() == [[1.0]]


def test_update_sensor_mask_only_and_empty() -> None:
    g = make_grid()
    s, w = acoustic.update_sensor(
        {"mask": np.zeros(g.shape), "record": ("p",)}, np.zeros((0, 3)), [], g
    )
    assert w.shape == (0, 0) and s.record == ("p",) and not s.mask.any()
    vox = points(g, np.random.RandomState(3))
    s, w = acoustic.update_sensor(acoustic.Sensor(np.zeros(g.shape)), vox.points, vox.index, g,
                                  True)  # fmt: skip
    assert w is None and s.mask.any()
    with pytest.raises(ValueError, match="sensor_on_grid"):
        acoustic.update_sensor(acoustic.Sensor(np.zeros(g.shape)), vox.points, vox.index,
                               make_grid(on_grid=None))  # fmt: skip


def test_sensor_weights_interpolate_and_c_order() -> None:
    """``W @ data`` interpolates; C-order columns via ``f_to_c_rows`` give the same result."""
    g = make_grid()
    vox = points(g, np.random.RandomState(4))
    s, w = acoustic.update_sensor(acoustic.Sensor(np.zeros(g.shape)), vox.points, vox.index, g)
    field = np.random.RandomState(5).normal(size=g.shape)
    data_f = field.ravel(order="F")[np.flatnonzero(s.mask.ravel(order="F"))]
    data_c = kw.rows_f_to_c(data_f, s.mask)
    np.testing.assert_array_equal(data_c, field[s.mask.astype(bool)])  # C order = boolean indexing
    perm = kw.f_to_c_rows(s.mask)
    w_c = w.tocsc()[:, perm]
    np.testing.assert_allclose(w_c @ data_c, w @ data_f, rtol=1e-13, atol=1e-13)


def test_define_sensor_mb_and_all() -> None:
    g = make_grid()
    rs = np.random.RandomState(6)
    P = np.vstack([points(g, rs).points, [[g.x[0] - 1e-3, 0, 0]]])
    mb = {
        "points": P,
        "radii": np.arange(len(P))[:, None] * 1e-6,
        "velocities": np.ones((len(P), 3)),
    }
    s, _, mb2, md = acoustic.define_sensor_MB(g, mb)
    assert len(mb2["points"]) == len(P) - 1 and mb2["radii"][-1, 0] == (len(P) - 2) * 1e-6
    d = np.sqrt(((mb2["points"][:, None] - mb2["points"][None]) ** 2).sum(-1))
    assert md == d.max()
    s_all, idx, max_mb = acoustic.define_sensor_MB_all(g, [mb, mb2["points"][:3]])
    np.testing.assert_array_equal(s_all.mask.astype(bool), s.mask.astype(bool))
    np.testing.assert_array_equal(idx, np.flatnonzero(s.mask.ravel(order="F")))
    assert max_mb == len(P) - 1


def test_extract_sensor_data() -> None:
    p = np.arange(5 * 8, dtype=float).reshape(5, 8)
    mask_idx = np.array([2, 5, 9, 11, 40])
    a, b = acoustic.extract_sensor_data({"p": p}, mask_idx, [40, 5], [11, 2, 99], {"tr": [3.5e-8]},
                                        {"dt": 1e-8})  # fmt: skip
    np.testing.assert_array_equal(a["p"], p[[0, 3], :4])
    np.testing.assert_array_equal(b["p"], p[[1, 4], :])


# ------------------------------------------------------------------ source


def transducer(n_el=5, n_int=3, curved=False, source_type=None):
    rs = np.random.RandomState(7)
    ip = np.zeros((n_el, n_int, 3))
    ip[:, :, 0] = 2 * 2.0**-13
    ip[:, :, 1] = (np.arange(n_el)[:, None] - n_el // 2) * 2 * 2.0**-13 + 0.3 * 2.0**-13
    ip[:, :, 2] = (np.arange(n_int)[None, :] - 1) * 2.0**-13 * 0.7
    T = {
        "integration_points": ip,
        "integration_weights": rs.uniform(0.5, 1, (n_el, n_int)),
        "integration_transmit_delays": rs.uniform(0, 5e-8, (n_el, n_int)),
        "integration_transmit_apodization": np.ones((n_el, n_int)),
        "NumberOfElements": float(n_el),
        "OnGrid": False,
        "Configuration": "Curved" if curved else "Linear",
    }
    if source_type:
        T["SourceType"] = source_type
    return T


def transmit(n_el, seq="pulse", m=40):
    t = np.arange(m)
    return {
        "PressureSignal": (np.sin(2 * np.pi * t / 9) * np.hanning(m)).reshape(1, -1),
        "Delays": np.linspace(0, 3e-8, n_el).reshape(1, -1),
        "Apodization": np.ones((1, n_el)),
        "SeqPulse": seq,
    }


MEDIUM = {"SpeedOfSound": 1540.0, "Density": 1000.0}


@pytest.mark.parametrize("m", [40, 41])
def test_source_delay_equals_full_fft(m: int) -> None:
    """Half-spectrum delay = MATLAB's full symmetric axis + ``ifft(...,'symmetric')``."""
    g = make_grid()
    T = transducer()
    s, w = acoustic.define_sensor_transducer(T, g)
    idx = np.flatnonzero(s.mask.ravel(order="F"))
    T["SourceType"] = "monopole"
    Tx = transmit(5, m=m)
    src = acoustic.define_source_transducer(T, Tx, MEDIUM, g, w.T, idx)
    # independent full-spectrum version (as define_source_transducer.m)
    p = acoustic.sinc_interpolation(np.arange(m) * g.dt, Tx["PressureSignal"],
                                    np.arange(m) * g.dt + g.dt / 2)  # fmt: skip
    delays = (T["integration_transmit_delays"] + Tx["Delays"].T).ravel(order="F")
    weights = T["integration_weights"].ravel(order="F")
    N = m + int(np.ceil(delays.max() / g.dt))
    f = np.arange(N) / (N * g.dt)
    f[int(np.ceil(N / 2 + 1)) - 1 :] = -f[int(np.floor(1 + N / 2)) - 1 : 0 : -1]
    X = np.fft.fft(np.tile(p, (delays.size, 1)), N, axis=1) * np.exp(
        -2j * np.pi * delays[:, None] * f
    )
    ps = ifft_symmetric(X, axis=1)
    np.testing.assert_allclose(
        src.p, w.T @ (ps * weights[:, None]), atol=1e-12 * np.abs(src.p).max()
    )
    assert src.p.shape == (idx.size, N)


def test_source_types_and_masks() -> None:
    g = make_grid(n=(15, 14, 9), full=(46, 44, 39))
    for curved in (False, True):
        T = transducer(curved=curved)
        T["Radius"], T["Pitch"] = 0.02, 2 * 2.0**-13
        s, w = acoustic.define_sensor_transducer(T, g)
        idx = np.flatnonzero(s.mask.ravel(order="F"))
        src = acoustic.define_source_transducer(T, transmit(5), MEDIUM, g, w.T, idx)
        assert src.p is None and src.u_mask is not None
        np.testing.assert_array_equal(src.u_mask, s.mask)  # absent SourceType: no shift (Q-150)
        assert (src.uy is not None) == curved
        T["SourceType"] = "dipole"
        dip = acoustic.define_source_transducer(T, transmit(5), MEDIUM, g, w.T, idx)
        assert dip.ux.shape[0] == dip.u_mask.sum()
        assert not np.array_equal(dip.u_mask, s.mask)  # half-cell shift changes the mask
        if curved:
            assert dip.uy.shape == dip.ux.shape


def test_preprocess_transmit_shapes() -> None:
    pytest.importorskip("kwave")
    m = 200
    Tx = {
        "PressureSignal": np.sin(np.arange(m) / 5).reshape(1, -1),
        "VoltageSignal": np.cos(np.arange(m) / 5).reshape(1, -1),
        "SamplingRate": 250e6,
    }
    out = acoustic.preprocess_transmit(Tx, {"SpeedOfSoundMinimum": 1480.0},
                                       kw.KGrid((32, 32, 32), (1e-4,) * 3, 2e-8, 10))  # fmt: skip
    n = int(np.floor((m - 1) / 250e6 / 2e-8)) + 1
    assert out["PressureSignal"].shape == (1, n) and out["VoltageSignal"].shape == (1, n)
    assert out["SamplingRate"] == 1 / 2e-8 and Tx["SamplingRate"] == 250e6
