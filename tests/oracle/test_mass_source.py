"""Oracle tests: ``compute_bubble_mass_source`` (``acoustic-module/``), M2.

- batch membership (stable radius sort, consecutive batches): exact;
- shell properties of ``get_microbubble_shell_properties``: float (1e-9 rel);
- mass source: bubble-response class (provisional ``CLASS_REL``/``CLASS_ABS`` of
  ``test_microbubble_model``) applied per bubble to the mass source, with the
  absolute term scaled by ``4 pi rho R0^3 / T``.

Needs MATLAB k-Wave on the path (``filterTimeSeries``), like ``test_kwave_signal``.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from pyproteus import config, data
from pyproteus.acoustic import mass_source as ms
from tests.oracle.helpers import from_matlab, put, row, vec
from tests.oracle.test_microbubble_model import CLASS_ABS, CLASS_REL, g17

pytest.importorskip("kwave")

KWAVE_MATLAB = Path(
    os.environ.get(
        "KWAVE_MATLAB_PATH",
        "/Users/theodoremichel/Repos/TWENTE/Ultrasound/PROTEUS-cleaning/PROTEUS/external/k-wave/k-Wave",
    )
)


@pytest.fixture(scope="module")
def keng(matlab_engine):
    if not (KWAVE_MATLAB / "filterTimeSeries.m").is_file():
        pytest.skip(f"MATLAB k-Wave not found at {KWAVE_MATLAB} (set KWAVE_MATLAB_PATH)")
    matlab_engine.addpath(str(KWAVE_MATLAB), nargout=0)
    yield matlab_engine
    matlab_engine.rmpath(str(KWAVE_MATLAB), nargout=0)


# local functions of compute_bubble_mass_source.m are not callable; the batch
# rule is re-expressed with MATLAB's sort and compared exactly.
@pytest.mark.parametrize(
    "radii",
    [
        [2e-6, 1.5e-6, 2e-6, 3e-6, 1.5e-6, 2e-6, 2.5e-6],
        [2e-6] * 9,
        [3e-6, 2e-6, 1e-6],
        list(np.random.default_rng(1).choice([1e-6, 2e-6, 3e-6], 40)),
    ],
)
@pytest.mark.parametrize("batch_size", [1, 2, 3, 15])
def test_batch_indices(matlab_engine, radii: list[float], batch_size: int) -> None:
    eng = matlab_engine
    r = np.array(radii)
    put(eng, pyp_r=row(r))
    n_batch = -(-r.size // batch_size)
    eng.eval(
        f"[~, pyp_I] = sort(pyp_r); pyp_idx = cell(1, {n_batch});"
        f"for k = 1:{n_batch}, l = (k-1)*{batch_size} + (1:{batch_size});"
        " l(l>numel(pyp_r)) = []; pyp_idx{k} = pyp_I(l); end",
        nargout=0,
    )
    for k in range(n_batch):
        ref = vec(eng.eval(f"pyp_idx{{{k + 1}}}")).astype(int) - 1
        np.testing.assert_array_equal(ms.get_batch_indices(k, r.size, batch_size, r), ref)


def settings(eng: Any, tmp: Path, shell_model: str, advanced: bool) -> tuple[Any, Any]:
    """Default Medium/Microbubble (``reset_*``) in MATLAB, loaded into the Python models."""
    eng.eval(
        "S = struct(); S.Medium = reset_medium(); S.Microbubble = reset_microbubble();"
        f"S.Microbubble.Shell.Model = '{shell_model}';"
        f"S.Microbubble.Advanced = {'true' if advanced else 'false'};",
        nargout=0,
    )
    if shell_model == "Segers":
        fit = data.load("fit_SigmaR_evaluated.mat")
        put(eng, pyp_A=np.asarray(fit["A_m"], float), pyp_s=np.asarray(fit["sig"], float))
        eng.eval(
            "S.Microbubble.Shell.NormalizedArea = pyp_A; S.Microbubble.Shell.SurfaceTension"
            " = pyp_s;",
            nargout=0,
        )
    # Python side: the ported reset_* (exact vs MATLAB, tests/oracle/test_config.py)
    medium, microbubble = config.reset_medium(), config.reset_microbubble()
    microbubble.Shell.Model = shell_model
    microbubble.Advanced = advanced
    if shell_model == "Segers":
        microbubble.Shell.NormalizedArea = np.asarray(fit["A_m"], float)
        microbubble.Shell.SurfaceTension = np.asarray(fit["sig"], float)
    return medium, microbubble


def sensed(dt: float, f0: float, amps: np.ndarray, dur: float = 4e-6) -> np.ndarray:
    t = np.arange(0, dur, dt)
    nc = 3 / f0
    env = np.where(t < nc, np.sin(np.pi * t / nc) ** 2, 0.0)
    return amps[:, None] * (env * np.sin(2 * np.pi * f0 * t))[None, :]


def run_both(eng, sp, radii, medium, microbubble, f0, dt, k_max, batch_size):
    put(eng, pyp_sp=sp, pyp_radii=radii.reshape(-1, 1))
    eng.eval(
        f"kg = struct('dt', {g17(dt)}, 'k_max', {g17(k_max)});"
        f"S.Microbubble.BatchSize = {batch_size}; S.Microbubble.UseParfor = 'off';"
        f"Tx = struct('CenterFrequency', {g17(f0)}); warning('off','all');"
        "evalc('pyp_m = compute_bubble_mass_source(pyp_sp, pyp_radii, kg, S.Medium,"
        " S.Microbubble, Tx);'); warning('on','all');"
        "stop(timerfindall); delete(timerfindall);",
        nargout=0,
    )
    ref = from_matlab(eng.eval("pyp_m"))
    kgrid = SimpleNamespace(dt=dt, k_max=k_max)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = ms.compute_bubble_mass_source(
            sp,
            radii,
            kgrid,
            medium,
            microbubble,
            SimpleNamespace(CenterFrequency=f0),
            batch_size=batch_size,
            timeout=None,
        )
    return got, ref


def mass_errors(got, ref, radii, rho=1060.0, P0=1.013e5) -> np.ndarray:
    T = np.median(np.sqrt(rho * radii**2 / P0))
    scale = 4 * np.pi * rho * radii**3 / T
    bound = CLASS_REL * np.max(np.abs(ref), axis=1) + CLASS_ABS * scale
    return np.max(np.abs(got - ref), axis=1) / bound


CASES = {
    # name: (shell model, advanced, radii, amplitudes (Pa), batch size)
    "default_marmottant": (
        "Marmottant",
        False,
        [2e-6, 1.5e-6, 2e-6, 3e-6, 1.5e-6, 2e-6, 2.5e-6],
        [50e3, 100e3, 20e3, 150e3, 80e3, 200e3, 60e3],
        15,
    ),
    "batches_of_3_ties": (
        "Marmottant",
        False,
        [2e-6, 1.5e-6, 2e-6, 3e-6, 1.5e-6, 2e-6, 2.5e-6],
        [50e3, 100e3, 20e3, 150e3, 80e3, 200e3, 60e3],
        3,
    ),
    "single_bubble": ("Marmottant", False, [2.2e-6], [120e3], 15),
    "segers_table": ("Segers", False, [1.2e-6, 2e-6, 4e-6], [60e3, 120e3, 250e3], 2),
    "advanced_viscosity": ("Marmottant", True, [0.6e-6, 5.5e-6], [80e3, 80e3], 1),
}


@pytest.mark.parametrize("name", list(CASES))
def test_compute_bubble_mass_source(keng, tmp_path, name: str) -> None:
    eng = keng
    model, advanced, radii, amps, bs = CASES[name]
    medium, microbubble = settings(eng, tmp_path, model, advanced)
    radii = np.array(radii)
    f0, dt = 2.5e6, 1 / 40e6
    k_max = np.pi / (1540 / (f0 * 6))
    sp = sensed(dt, f0, np.array(amps, float))
    got, ref = run_both(eng, sp, radii, medium, microbubble, f0, dt, k_max, bs)
    assert got.shape == ref.shape and got.dtype == np.float64
    err = mass_errors(got, ref, radii)
    print(f"{name}: worst fraction of the class {err.max():.3f}")
    assert np.all(err <= 1), err


def test_mass_source_single_precision(keng, tmp_path) -> None:
    """float32 sensed pressure (k-Wave output): float32 result; MATLAB integrates
    partly in single, pyproteus in double; same class."""
    eng = keng
    medium, microbubble = settings(eng, tmp_path, "Marmottant", False)
    radii = np.array([1.8e-6, 2.4e-6])
    f0, dt = 2.5e6, 1 / 40e6
    k_max = np.pi / (1540 / (f0 * 6))
    sp = sensed(dt, f0, np.array([90e3, 40e3])).astype(np.float32)
    got, ref = run_both(eng, sp, radii, medium, microbubble, f0, dt, k_max, 15)
    assert got.dtype == np.float32 and ref.dtype == np.float32
    err = mass_errors(got.astype(float), ref.astype(float), radii)
    print(f"single: worst fraction of the class {err.max():.3f}")
    assert np.all(err <= 1), err


def test_batch_size_sensitivity(keng, tmp_path) -> None:
    """Measured and reported (Q-50/Q-121): batch size 1 vs 7, in MATLAB and in pyproteus."""
    eng = keng
    medium, microbubble = settings(eng, tmp_path, "Marmottant", False)
    _, _, radii, amps, _ = CASES["default_marmottant"]
    radii = np.array(radii)
    f0, dt = 2.5e6, 1 / 40e6
    k_max = np.pi / (1540 / (f0 * 6))
    sp = sensed(dt, f0, np.array(amps, float))
    g1, m1 = run_both(eng, sp, radii, medium, microbubble, f0, dt, k_max, 1)
    g7, m7 = run_both(eng, sp, radii, medium, microbubble, f0, dt, k_max, 7)
    rel = lambda a, b: np.max(np.abs(a - b), axis=1) / np.max(np.abs(b), axis=1)  # noqa: E731
    print(f"MATLAB 1 vs 7: {rel(m7, m1)}")
    print(f"pyproteus 1 vs 7: {rel(g7, g1)}")
    print(f"pyproteus vs MATLAB, batch 1: {rel(g1, m1)}; batch 7: {rel(g7, m7)}")
    assert np.all(mass_errors(m7, m1, radii) <= 1)
    assert np.all(mass_errors(g7, g1, radii) <= 1)
