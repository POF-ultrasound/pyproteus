"""Array-module switch (``DATA_CAST``/``DATA_CAST_RF = 'gpuArray-*'``) tested without a GPU.

CuPy needs an NVIDIA GPU, which the development Mac and CI do not have; the
real CuPy-vs-NumPy run is left for the university cluster (D-3, D-16). Here a
fake array module stands in for CuPy: it delegates to NumPy, but its arrays are
a separate type that must be moved to the host with ``.get()`` (as CuPy's), and
it records which device was selected. This checks that the GPU code path is
taken, threads the array module everywhere, gathers results to the host where
PROTEUS does, and gives the NumPy results.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from pyproteus.acoustic import homogeneous as hom
from pyproteus.acoustic import rf as rfmod


def _is_host(x: object) -> bool:
    return isinstance(x, np.ndarray) and x.ndim > 0


def _unwrap(x: object) -> object:
    """Device → NumPy for the stand-in's own use; host arrays are refused (as CuPy)."""
    if isinstance(x, DeviceArray):
        return x._a
    if _is_host(x):
        raise TypeError("host (NumPy) array passed to a device operation")
    if isinstance(x, list | tuple):
        return type(x)(_unwrap(v) for v in x)
    return x


def _wrap(x: object) -> object:
    return DeviceArray(x) if isinstance(x, np.ndarray) else x


class DeviceArray(np.lib.mixins.NDArrayOperatorsMixin):
    """Stand-in for a CuPy array (not an ``ndarray``).

    As CuPy: ``.get()`` is the only way to the host (``np.asarray(dev)`` and
    ``host[...] = dev`` raise TypeError), and device operations refuse host arrays.
    """

    def __init__(self, a: np.ndarray) -> None:
        self._a = a

    def get(self) -> np.ndarray:
        return self._a.copy()

    def __array__(self, *args, **kwargs):
        raise TypeError("implicit conversion of a device array to NumPy is not allowed")

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if "out" in kwargs:
            kwargs["out"] = tuple(_unwrap(o) for o in kwargs["out"])
        out = getattr(ufunc, method)(*(_unwrap(x) for x in inputs), **kwargs)
        if "out" in kwargs and len(kwargs["out"]) == 1 and isinstance(inputs[0], DeviceArray):
            return next(x for x in (*inputs,) if x is not None) if out is None else _wrap(out)
        return _wrap(out)

    shape = property(lambda self: self._a.shape)
    dtype = property(lambda self: self._a.dtype)
    ndim = property(lambda self: self._a.ndim)
    size = property(lambda self: self._a.size)
    real = property(lambda self: DeviceArray(self._a.real))
    imag = property(lambda self: DeviceArray(self._a.imag))

    def __len__(self) -> int:
        return len(self._a)

    def __getitem__(self, key):
        return _wrap(self._a[_unwrap(key)])

    def __setitem__(self, key, value) -> None:
        self._a[_unwrap(key)] = _unwrap(value)

    def copy(self):
        return DeviceArray(self._a.copy())

    def astype(self, dtype, copy=True):
        return DeviceArray(self._a.astype(dtype, copy=copy))

    def reshape(self, *shape):
        return DeviceArray(self._a.reshape(*shape))

    def transpose(self, *axes):
        return DeviceArray(self._a.transpose(*axes))

    def sum(self, *args, **kwargs):
        return _wrap(self._a.sum(*args, **kwargs))

    def any(self):
        return bool(self._a.any())


class FakeCuPy(types.ModuleType):
    """NumPy behind a CuPy-like module returning :class:`DeviceArray`; records calls and devices."""

    _TO_DEVICE = frozenset({"asarray", "array"})  # the only host → device transfers

    def __init__(self) -> None:
        super().__init__("fake_cupy")
        self.calls: dict[str, int] = {}
        self.devices: list[int] = []
        fake = self

        class _Device:
            def __init__(self, n: int) -> None:
                self.n = n

            def use(self) -> None:
                fake.devices.append(self.n)

        self.cuda = types.SimpleNamespace(Device=_Device)
        self.fft = types.SimpleNamespace(
            rfft=self._fn(np.fft.rfft, "rfft"), irfft=self._fn(np.fft.irfft, "irfft")
        )

    def _fn(self, fn, name):
        def inner(*args, **kwargs):
            self.calls[name] = self.calls.get(name, 0) + 1
            if name in self._TO_DEVICE:
                x = args[0]
                a = x._a if isinstance(x, DeviceArray) else np.asarray(x)
                return DeviceArray(
                    np.array(a, dtype=kwargs.get("dtype", args[1] if len(args) > 1 else None))
                )
            out = fn(*(_unwrap(a) for a in args), **{k: _unwrap(v) for k, v in kwargs.items()})
            return _wrap(out)

        return inner

    def __getattr__(self, name: str):
        attr = getattr(np, name)
        return self._fn(attr, name) if callable(attr) and not isinstance(attr, type) else attr


@pytest.fixture
def fake_cupy(monkeypatch: pytest.MonkeyPatch) -> FakeCuPy:
    fake = FakeCuPy()
    real = hom.array_module

    def array_module(data_cast):
        return (
            fake
            if isinstance(data_cast, str) and data_cast.startswith("gpuArray")
            else real(data_cast)
        )

    monkeypatch.setattr(hom, "array_module", array_module)
    monkeypatch.setattr(rfmod, "array_module", array_module)
    return fake


DT = 1 / 40e6


def problem(seed: int = 0):
    rng = np.random.default_rng(seed)
    nt = 160
    t = np.arange(nt) * DT
    ms = (np.sin(2 * np.pi * 3e6 * t) * np.exp(-(((t - 1e-6) / 3e-7) ** 2)))[None] * rng.uniform(
        0.5, 2, (4, 1)
    )
    medium = {"density": 1000.0, "sound_speed": 1540.0, "alpha_coeff": 0.75, "alpha_power": 1.5}
    source = {"points": 1e-3 * rng.random((4, 3)), "mass_source": ms}
    sensor = {"points": 1e-3 * (rng.random((25, 3)) + np.array([2, 0, 0]))}
    return nt, medium, source, sensor


@pytest.mark.parametrize("gridded", [False, True])
@pytest.mark.parametrize("cast", ["single", "double"])
def test_run_simulation_homogeneous_gpu_path(fake_cupy: FakeCuPy, gridded: bool, cast: str) -> None:
    nt, medium, source, sensor = problem()
    dtype = np.float32 if cast == "single" else np.float64
    source = {**source, "mass_source": source["mass_source"].astype(dtype)}
    kg = {"dt": DT, "Nt": nt}
    cpu = hom.run_simulation_homogeneous(
        {"DATA_CAST": cast, "gridded": gridded, "MaxArrayElements": 4000},
        kg,
        medium,
        source,
        sensor,
    )["p"]
    gpu = hom.run_simulation_homogeneous(
        {
            "DATA_CAST": f"gpuArray-{cast}",
            "DEVICE_NUM": 1,
            "gridded": gridded,
            "MaxArrayElements": 4000,
        },
        kg,
        medium,
        source,
        sensor,
    )["p"]
    assert fake_cupy.devices == [1] and fake_cupy.calls.get("irfft", 0) > 0
    assert isinstance(gpu, DeviceArray)  # small output stays on the device (MaxGpuElements)
    np.testing.assert_array_equal(gpu.get(), cpu)
    assert gpu.dtype == dtype


def test_large_gpu_output_is_accumulated_on_the_host(fake_cupy: FakeCuPy) -> None:
    nt, medium, source, sensor = problem(1)
    kg = {"dt": DT, "Nt": nt}
    rp = {"DATA_CAST": "gpuArray-double", "gridded": False, "MaxGpuElements": 10}
    out = hom.run_simulation_homogeneous(rp, kg, medium, source, sensor)["p"]
    assert type(out) is np.ndarray  # host array, as PROTEUS above MaxGpuElements
    ref = hom.run_simulation_homogeneous({**rp, "DATA_CAST": "double"}, kg, medium, source, sensor)[
        "p"
    ]
    np.testing.assert_array_equal(out, ref)


@pytest.mark.parametrize("cast", ["single", "double"])
def test_compute_rf_gpu_path(fake_cupy: FakeCuPy, cast: str) -> None:
    rng = np.random.default_rng(2)
    n_el, n_int, n_sen, m = 4, 3, 9, 120
    T = {
        "integration_points": np.zeros((n_el, n_int, 3)),
        "integration_receive_apodization": rng.random((n_el, n_int)),
        "integration_receive_delays": rng.uniform(0, 5 * DT, (n_el, n_int)),
        "integration_weights": 1.0,
        "ReceiveImpulseResponse": np.array([0.2, 1.0, 0.2]),
        "SamplingRate": 1 / DT,
    }
    W = rng.random((n_el * n_int, n_sen))
    p = rng.standard_normal((n_sen, m))
    cpu = rfmod.compute_RF(T, {"p": p}, W, {"dt": DT}, {"DATA_CAST_RF": cast})
    gpu = rfmod.compute_RF(T, {"p": p}, W, {"dt": DT}, {"DATA_CAST_RF": f"gpuArray-{cast}"})
    assert fake_cupy.calls.get("rfft", 0) > 0
    assert type(gpu) is np.ndarray  # gathered to the host, as MATLAB
    np.testing.assert_array_equal(gpu, cpu)
