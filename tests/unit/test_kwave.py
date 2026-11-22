"""Unit tests for ``pyproteus.backends.kwave`` (no MATLAB).

Row-order mapping, source scaling, input validation and the binary call.
Tests that run the k-Wave C++ binary skip when it does not start (CI without
its runtime libraries); MATLAB parity is in ``tests/oracle/test_kwave_*.py``.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pyproteus.backends import kwave as kw
from pyproteus.numerics import find

WRAPPER = Path(__file__).resolve().parents[2] / "tools" / "tracer_bullet" / "bin" / kw.OMP_NAME

# --- row order

SHAPES = [
    (37, 29, 23),
    (5, 1, 7),
    (1, 4, 3),
    (6, 6, 6),
    (4, 3, 2),
    (7, 5),
    (1, 9),
    (6, 1),
    (8, 8),
]


def _rows_f_and_c(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rows tagged with their voxel subscripts, built without the module under test."""
    shape = mask.shape
    sub = np.indices(shape).reshape(len(shape), -1)  # C-order enumeration of all voxels
    lin_f = np.ravel_multi_index(sub, shape, order="F")
    rows_c = sub.T[mask.ravel()]  # == np.argwhere(mask): C order
    rows_f = sub.T[np.argsort(lin_f, kind="stable")][mask.ravel(order="F")]  # find order
    return rows_f, rows_c


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("fill", [0.0, 0.05, 0.3, 1.0])
def test_row_order_maps_find_order_to_c_order(shape, fill) -> None:
    rng = np.random.default_rng(abs(hash((shape, fill))) % 2**32)
    mask = rng.random(shape) < fill
    rows_f, rows_c = _rows_f_and_c(mask)
    assert np.array_equal(rows_c, np.argwhere(mask))
    assert np.array_equal(kw.rows_f_to_c(rows_f, mask), rows_c)
    assert np.array_equal(kw.rows_c_to_f(rows_c, mask), rows_f)
    perm = kw.f_to_c_rows(mask)
    assert np.array_equal(perm[kw.c_to_f_rows(mask)], np.arange(perm.size))
    # matlab_find: 1-based column-major linear indices of the rows in find order
    expected = np.ravel_multi_index(tuple(rows_f.T), shape, order="F") + 1
    assert np.array_equal(kw.matlab_find(mask), expected)
    assert np.array_equal(kw.matlab_find(mask), find(mask) + 1)


def test_row_order_asymmetric_mask_is_not_identity() -> None:
    """A hand-checked 2x3 case: F order (0,0),(1,0),(0,2) vs C order (0,0),(0,2),(1,0)."""
    mask = np.array([[1, 0, 1], [1, 0, 0]], bool)
    rows_f = np.array([[10], [20], [30]])  # voxels (0,0), (1,0), (0,2)
    assert np.array_equal(kw.rows_f_to_c(rows_f, mask), [[10], [30], [20]])
    assert np.array_equal(kw.matlab_find(mask), [1, 2, 5])


def test_row_order_keeps_trailing_axes() -> None:
    mask = np.zeros((3, 4, 5), bool)
    mask[2, 0, 1] = mask[0, 3, 0] = mask[1, 1, 4] = True
    rows_f, rows_c = _rows_f_and_c(mask)
    sig = np.arange(3 * 7).reshape(3, 7)  # (n, Nt) rows in find order
    out = kw.rows_f_to_c(sig, mask)
    order = [int(np.flatnonzero((rows_f == rc).all(1))[0]) for rc in rows_c]
    assert np.array_equal(out, sig[order])


def test_row_order_rejects_wrong_row_count() -> None:
    mask = np.zeros((3, 4, 5), bool)
    mask[1, 2, 3] = mask[2, 0, 1] = True
    with pytest.raises(ValueError, match="rows"):
        kw.rows_f_to_c(np.zeros((3, 4)), mask)
    with pytest.raises(ValueError, match="rows"):
        kw.rows_c_to_f(np.zeros((1, 4)), mask)


# --- kspaceFirstOrder conventions


def test_cpp_sensor_rows_shape_guards() -> None:
    mask = np.zeros((3, 4, 5), bool)
    mask[0, 1, 2] = mask[2, 3, 4] = True
    p = np.arange(10.0).reshape(5, 2)  # (Nt, n_sensor) as k-wave-python 0.6.2 cpp returns it
    assert np.array_equal(kw.sensor_rows_from_kwp(p, mask, 5, "cpp"), p.T)
    with pytest.raises(ValueError, match="shape"):
        kw.sensor_rows_from_kwp(p.T, mask, 5, "cpp")
    with pytest.raises(ValueError, match="backend"):
        kw.sensor_rows_from_kwp(p, mask, 5, "matlab")


def test_python_sensor_rows_c_to_f() -> None:
    mask = np.zeros((3, 4, 5), bool)
    mask[2, 0, 0] = mask[0, 3, 0] = mask[0, 0, 4] = True  # C order: (0,0,4),(0,3,0),(2,0,0)
    p_c = np.array([[3.0, 3], [2, 2], [1, 1]])  # tagged with their F rank
    assert np.array_equal(kw.sensor_rows_from_kwp(p_c, mask, 2, "python")[:, 0], [1, 2, 3])
    with pytest.raises(ValueError, match="shape"):
        kw.sensor_rows_from_kwp(p_c.T, mask, 2, "python")


@pytest.mark.parametrize("n", [3, 7])
def test_undoes_kwp_time_permutation_when_nt_equals_n_sensor(n) -> None:
    """With Nt == n_sensor, k-wave-python 0.6.2 permutes the time axis (B-10); undo it.

    Runs k-wave-python's own ``CppSimulation._fix_output_order`` on a fake output.
    """
    kwave_cpp = pytest.importorskip("kwave.solvers.cpp_simulation")
    shape = (4, 3, 5)
    rng = np.random.default_rng(n)
    while True:  # a mask whose F->C permutation is not the identity
        mask = np.zeros(shape, bool)
        mask.ravel()[rng.choice(mask.size, n, replace=False)] = True
        if not np.array_equal(kw.f_to_c_rows(mask), np.arange(n)):
            break
    truth = rng.standard_normal((n, n))  # (n_sensor, Nt), rows in find order
    fake = SimpleNamespace(
        ndim=3,
        kgrid=SimpleNamespace(N=shape),
        sensor=SimpleNamespace(mask=mask),
        _FULL_GRID_SUFFIXES=kwave_cpp.CppSimulation._FULL_GRID_SUFFIXES,
    )
    out = kwave_cpp.CppSimulation._fix_output_order(fake, {"p": truth.T.copy()})["p"]
    assert not np.array_equal(out, truth.T)  # k-wave-python scrambled it
    assert np.array_equal(kw.sensor_rows_from_kwp(out, mask, n, "cpp"), truth)


def test_source_scale_axes_and_refusals() -> None:
    mask = np.zeros((3, 4, 5), bool)
    mask[1, 2, 3] = mask[0, 0, 0] = True
    c = np.full(mask.shape, 1500.0)
    c[0, 0, 0] = 1600.0  # rows in find order: (0,0,0) first
    spacing = (1e-4, 2e-4, 4e-4)
    for kind, d in zip(("ux", "uy", "uz"), spacing, strict=True):
        s = kw.additive_source_scale(kind, mask, c, 1e-8, spacing)
        assert np.array_equal(s, [2 * 1600 * 1e-8 / d, 2 * 1500 * 1e-8 / d])
    s = kw.additive_source_scale("p", mask, c, 1e-8, (1e-4,) * 3)
    assert np.array_equal(s, [2 * 1e-8 / (3 * 1600 * 1e-4), 2 * 1e-8 / (3 * 1500 * 1e-4)])
    with pytest.raises(ValueError, match="anisotropic"):
        kw.additive_source_scale("p", mask, c, 1e-8, spacing)
    with pytest.raises(ValueError, match="spacing"):
        kw.additive_source_scale("ux", mask, c, 1e-8, (1e-4, 1e-4))
    with pytest.raises(ValueError, match="unknown"):
        kw.additive_source_scale("uz", mask[..., 0], c[..., 0], 1e-8, (1e-4, 1e-4))


def test_alpha_mode_dropped_on_every_backend() -> None:
    """D-9 / QUIRKS.md Q-1: MATLAB 3DC ignores alpha_mode, so the adapter drops it."""
    medium = {"sound_speed": 1500.0, "alpha_power": 1.05, "alpha_mode": "no_dispersion"}
    out = kw.medium_for_kwp(medium)
    assert "alpha_mode" not in out
    assert out["alpha_power"] == 1.05
    assert "alpha_mode" in medium  # input untouched


def test_source_rows_to_kwp_per_backend() -> None:
    mask = np.zeros((3, 4, 5), bool)
    mask[2, 0, 0] = mask[0, 3, 0] = True  # F order: (2,0,0), (0,3,0); C order reversed
    c = np.full(mask.shape, 1500.0)
    rows = np.array([[1.0, 2.0], [3.0, 4.0]])
    kwargs = dict(kind="p", sound_speed=c, dt=1e-8, spacing=(1e-4,) * 3)
    scale = 2 * 1e-8 / (3 * 1500 * 1e-4)
    assert np.allclose(kw.source_rows_to_kwp(rows, mask, "cpp", **kwargs), rows * scale, rtol=0)
    assert np.array_equal(kw.source_rows_to_kwp(rows, mask, "python", **kwargs), rows[::-1])
    with pytest.raises(ValueError, match="rows"):
        kw.source_rows_to_kwp(rows[:1], mask, "cpp", **kwargs)
    with pytest.raises(ValueError, match="backend"):
        kw.source_rows_to_kwp(rows, mask, "gpu", **kwargs)


# --- grid, signal tools


@pytest.mark.parametrize(
    ("shape", "spacing"),
    [
        ((37, 29, 23), (1e-4,) * 3),
        ((16, 9, 10), (1e-4, 1e-4, 1e-4)),
        ((8, 8, 8), (1e-4, 2e-4, 5e-5)),
    ],
)
def test_kgrid_k_max(shape, spacing) -> None:
    pytest.importorskip("kwave")
    g = kw.KGrid(shape, spacing, 2e-8, 100)
    # largest |k| along each axis of an N-point FFT grid, then the smallest over axes
    per_axis = [
        np.pi / d * (1.0 if n % 2 == 0 else (n - 1) / n)
        for n, d in zip(shape, spacing, strict=True)
    ]
    assert g.k_max == pytest.approx(min(per_axis), rel=1e-14)


def test_kgrid_validation() -> None:
    with pytest.raises(ValueError):
        kw.KGrid((4, 4), (1e-4,) * 3, 1e-8, 10)
    g = kw.KGrid([4.0, 5, 6], [1e-4] * 3, 1e-8, 10.0)
    assert g.shape == (4, 5, 6) and g.Nt == 10 and g.ndim == 3


@pytest.mark.parametrize("shape", [(64,), (1, 64), (64, 1)])
def test_filter_time_series_keeps_vector_shape(shape) -> None:
    pytest.importorskip("kwave")
    x = np.random.default_rng(0).standard_normal(64).reshape(shape)
    y = kw.filter_time_series(1e-8, np.pi / 1e-4, 1480.0, x, zero_phase=True, ppw=2)
    assert y.shape == shape
    assert not np.allclose(y, x)
    with pytest.raises(ValueError, match="vector"):
        kw.filter_time_series(1e-8, np.pi / 1e-4, 1480.0, np.zeros((2, 8)))


def test_get_win_column() -> None:
    pytest.importorskip("kwave")
    win, cg = kw.get_win(9, "Tukey", param=0.5)
    assert win.shape == (9, 1)
    assert win[0, 0] == 0 and win[4, 0] == 1
    assert cg == pytest.approx(win.mean())


# --- kspace_first_order: inputs

G = kw.KGrid((12, 10, 9), (1e-4,) * 3, 2e-8, 40)


def _tiny():
    mask = np.zeros(G.shape, bool)
    mask[2, 3:6, 4] = True
    sensor = np.zeros(G.shape, bool)
    sensor[8, 2:8, 3] = True
    medium = {"sound_speed": np.full(G.shape, 1500.0), "density": np.full(G.shape, 1000.0)}
    source = {"p_mask": mask, "p": np.ones((3, 20))}
    return medium, source, {"mask": sensor}


@pytest.mark.parametrize(
    ("change", "exc", "match"),
    [
        (dict(pml_inside=True), NotImplementedError, "PMLInside"),
        (dict(smooth=True), NotImplementedError, "Smooth"),
        (dict(data_cast="double"), NotImplementedError, "DataCast"),
        (dict(backend="matlab"), ValueError, "backend"),
        (dict(device="tpu"), ValueError, "device"),
        (dict(pml_size=(10, 10)), ValueError, "pml_size"),
    ],
)
def test_kspace_first_order_rejects_options(change, exc, match) -> None:
    medium, source, sensor = _tiny()
    kwargs = dict(pml_size=10, pml_alpha=2.0) | change
    with pytest.raises(exc, match=match):
        kw.kspace_first_order(G, medium, source, sensor, **kwargs)


@pytest.mark.parametrize(
    ("edit", "exc", "match"),
    [
        (lambda m, s, n: n.update(record=["p", "ux"]), NotImplementedError, "record"),
        (lambda m, s, n: n.update(record=["p", "p"]), ValueError, "duplicate"),
        (lambda m, s, n: n.update(mask=np.zeros(G.shape, bool)), ValueError, "no active"),
        (lambda m, s, n: n.update(mask=np.zeros((3, 3, 3), bool)), ValueError, "sensor.mask"),
        (lambda m, s, n: n.pop("mask"), ValueError, "sensor.mask"),
        (lambda m, s, n: s.update(p=np.ones((2, 20))), ValueError, "one row per voxel"),
        (lambda m, s, n: s.update(p_mode="dirichlet"), NotImplementedError, "additive"),
        (lambda m, s, n: s.update(p0=np.ones(G.shape)), NotImplementedError, "p0"),
        (lambda m, s, n: s.update(p_mask=np.zeros(G.shape)), ValueError, "no active"),
        (lambda m, s, n: s.update(p_mask=2 * s["p_mask"]), ValueError, "binary"),
        (lambda m, s, n: s.pop("p"), ValueError, "without"),
        (lambda m, s, n: s.update(ux=np.ones((3, 20))), ValueError, "no u_mask"),
        (lambda m, s, n: m.pop("sound_speed"), ValueError, "sound_speed"),
    ],
)
def test_kspace_first_order_rejects_inputs(edit, exc, match) -> None:
    pytest.importorskip("kwave")
    medium, source, sensor = _tiny()
    edit(medium, source, sensor)
    with pytest.raises(exc, match=match):
        kw.kspace_first_order(G, medium, source, sensor, pml_size=10, backend="python")


def test_kspace_first_order_rejects_2d() -> None:
    g = kw.KGrid((12, 10), (1e-4,) * 2, 2e-8, 40)
    with pytest.raises(NotImplementedError, match="3-D"):
        kw.kspace_first_order(g, {}, {}, {"mask": np.ones((12, 10))}, pml_size=10)


def test_binary_path_resolution(tmp_path, monkeypatch) -> None:
    exe = tmp_path / "kspaceFirstOrder-OMP"
    exe.write_text("")
    assert kw.binary_path("cpu", exe) == exe
    monkeypatch.setenv(kw.BINARY_ENV, str(exe))
    assert kw.binary_path() == exe
    with pytest.raises(FileNotFoundError):
        kw.binary_path("cpu", tmp_path / "missing")
    with pytest.raises(ValueError, match="device"):
        kw.binary_path("npu")


def test_cpp_binary_command_line(tmp_path, monkeypatch) -> None:
    """Record flags, threads and device number reach the binary as MATLAB passes them."""
    pytest.importorskip("kwave")
    exe = tmp_path / "fake-binary"
    exe.write_text("")
    calls = []

    def fake_run(cmd, *, env, **kw_):
        calls.append((cmd, env))
        return subprocess.CompletedProcess(cmd, 1, "", "fake failure")

    monkeypatch.setattr(kw.subprocess, "run", fake_run)
    medium, source, sensor = _tiny()
    sensor["record"] = ["p_min", "p_max"]  # flags follow MATLAB's order, not this one
    with pytest.raises(RuntimeError, match="fake failure"):
        kw.kspace_first_order(
            G, medium, source, sensor, pml_size=10, binary=exe, num_threads=3, device_num=1,
            binary_env={"KWAVE_LIBS": "/x"}, data_path=tmp_path / "data",
        )  # fmt: skip
    cmd, env = calls[0]
    assert cmd[0] == str(exe)
    assert cmd[cmd.index("-i") + 1] == str((tmp_path / "data" / "kwave_input.h5").resolve())
    assert cmd[-2:] == ["--p_max", "--p_min"] and "--p_raw" not in cmd
    assert cmd[cmd.index("-t") + 1] == "3" and cmd[cmd.index("-g") + 1] == "1"
    assert env["KWAVE_LIBS"] == "/x"
    assert (tmp_path / "data" / "kwave_input.h5").is_file()


# --- binary runs (skip without it)


def _working_binary() -> Path | None:
    candidates = [os.environ.get(kw.BINARY_ENV), str(WRAPPER)]
    with contextlib.suppress(FileNotFoundError, ImportError):
        candidates.append(str(kw.binary_path()))
    for c in candidates:
        if not c or not Path(c).is_file():
            continue
        try:
            r = subprocess.run([c, "--version"], capture_output=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            continue
        if r.returncode == 0 and b"kspaceFirstOrder" in r.stdout:
            return Path(c)
    return None


@pytest.fixture(scope="module")
def binary() -> Path:
    pytest.importorskip("kwave")
    exe = _working_binary()
    if exe is None:
        pytest.skip("k-Wave C++ binary does not start (see tools/tracer_bullet/README.md)")
    return exe


def _small_case(n_sensor: int | None = None):
    """Heterogeneous, absorbing, asymmetric masks; optionally exactly ``n_sensor`` sensors."""
    rng = np.random.default_rng(7)
    g = kw.KGrid((20, 17, 15), (1e-4,) * 3, 2e-8, 90)
    c = 1500.0 + 40.0 * rng.random(g.shape)
    medium = {
        "sound_speed": c,
        "density": 1000.0 + 30.0 * rng.random(g.shape),
        "BonA": np.full(g.shape, 6.0),
        "alpha_coeff": np.full(g.shape, 0.5),
        "alpha_power": 1.5,
    }
    p_mask = np.zeros(g.shape, bool)
    p_mask[2, 3:12, 5:9] = True
    p_mask[3, 14, 2] = True
    n = int(p_mask.sum())
    t = np.arange(60) * g.dt
    delay = rng.uniform(0, 3e-7, n)[:, None]
    p = 1e5 * np.sin(2 * np.pi * 3e6 * (t - delay)) * np.exp(-(((t - delay - 4e-7) / 2e-7) ** 2))
    u_mask = np.zeros(g.shape, bool)
    u_mask[4, 2:6, 10:13] = True
    m = int(u_mask.sum())
    ux = rng.standard_normal((m, 1)) * p[:m] / 1.5e6
    sensor = np.zeros(g.shape, bool)
    sensor[8:, :, :][rng.random((12, 17, 15)) < 0.08] = True
    sensor[2, 3:12, 5:9] = True
    if n_sensor is not None:
        idx = np.flatnonzero(sensor.ravel())
        if idx.size >= n_sensor:
            sensor.ravel()[idx[n_sensor:]] = False
        else:
            free = np.flatnonzero(~sensor.ravel())
            sensor.ravel()[rng.choice(free, n_sensor - idx.size, replace=False)] = True
        assert sensor.sum() == n_sensor
    source = {"p_mask": p_mask, "p": p, "u_mask": u_mask, "ux": ux}
    return g, medium, source, sensor


def _rel(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


@pytest.mark.slow
@pytest.mark.parametrize("n_sensor", [None, 90])  # 90 == Nt
def test_cpp_and_python_backends_agree(binary, n_sensor, tmp_path) -> None:
    g, medium, source, sensor = _small_case(n_sensor)
    record = ["p", "p_max", "p_min"]
    kwargs = dict(pml_size=(8, 10, 7), pml_alpha=2.0)
    cpp = kw.kspace_first_order(
        g, medium, source, {"mask": sensor, "record": record}, binary=binary, **kwargs
    )
    py = kw.kspace_first_order(
        g, medium, source, {"mask": sensor, "record": record}, backend="python", **kwargs
    )
    n = int(sensor.sum())
    assert cpp["p"].shape == py["p"].shape == (n, g.Nt)
    assert cpp["p"].dtype == np.float32 and py["p"].dtype == np.float32
    assert np.abs(cpp["p"]).max() > 1e2  # the wave reached the sensors
    assert _rel(py["p"], cpp["p"]) < 1e-5
    for key, fn in (("p_max", np.max), ("p_min", np.min)):
        assert cpp[key].shape == (n,)
        assert np.array_equal(cpp[key], fn(cpp["p"], axis=1))
        assert np.array_equal(py[key], fn(py["p"], axis=1))
    # cpp rows are in find order: the source-plane sensors see the strongest signal
    # first in time where the source fires first (a reorder would break this link)
    assert _rel(kw.rows_c_to_f(cpp["p"], sensor), cpp["p"]) > 0.1


@pytest.mark.slow
def test_cpp_data_path_keeps_files_and_is_reusable(binary, tmp_path) -> None:
    g, medium, source, sensor = _small_case()
    kwargs = dict(pml_size=8, binary=binary, data_path=tmp_path)
    a = kw.kspace_first_order(g, medium, source, {"mask": sensor}, **kwargs)
    b = kw.kspace_first_order(g, medium, source, {"mask": sensor}, **kwargs)  # files overwritten
    assert (tmp_path / "kwave_input.h5").is_file() and (tmp_path / "kwave_output.h5").is_file()
    assert np.array_equal(a["p"], b["p"])
