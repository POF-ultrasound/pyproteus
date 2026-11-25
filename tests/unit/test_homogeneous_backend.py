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


class DeviceArray(np.ndarray):
    """Stand-in for a CuPy array: ``.get()`` returns a plain NumPy array.

    Like CuPy, arithmetic with a plain (non-scalar) NumPy array raises, so a
    host array leaking into the device computation is caught.
    """

    def get(self) -> np.ndarray:
        return np.asarray(self).view(np.ndarray).copy()

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        args = []
        for x in inputs + tuple(kwargs.get("out", ())):
            if type(x) is np.ndarray and x.ndim > 0:
                raise TypeError(f"host array mixed into a device computation ({ufunc.__name__})")
        for x in inputs:
            args.append(x.view(np.ndarray) if isinstance(x, DeviceArray) else x)
        if "out" in kwargs:
            kwargs["out"] = tuple(o.view(np.ndarray) for o in kwargs["out"])
        out = getattr(ufunc, method)(*args, **kwargs)
        if method == "at":
            return None
        if isinstance(out, tuple):
            return tuple(o.view(DeviceArray) if isinstance(o, np.ndarray) else o for o in out)
        return out.view(DeviceArray) if isinstance(out, np.ndarray) else out

    def __setitem__(self, key, value):
        if type(value) is np.ndarray and value.ndim > 0:
            raise TypeError("host array assigned into a device array")
        super().__setitem__(key, value)


class FakeCuPy(types.ModuleType):
    """NumPy behind a CuPy-like module; array constructors return :class:`DeviceArray`."""

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
            rfft=self._wrap(np.fft.rfft, "rfft"), irfft=self._wrap(np.fft.irfft, "irfft")
        )

    def _wrap(self, fn, name):
        def inner(*args, **kwargs):
            self.calls[name] = self.calls.get(name, 0) + 1
            out = fn(*args, **kwargs)
            return out.view(DeviceArray) if isinstance(out, np.ndarray) else out

        return inner

    def __getattr__(self, name: str):
        attr = getattr(np, name)
        return self._wrap(attr, name) if callable(attr) and not isinstance(attr, type) else attr


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
