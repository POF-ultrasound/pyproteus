"""Oracle tests: ``optimize_grid_size`` and ``define_grid`` (M1, grid/medium/transducer).

Classes (D-13): grid sizes, PML sizes and integer bounds are integers computed
from exact inputs with IEEE division, ``ceil``/``floor`` and integer search, so
they are compared exactly (statistical class not needed: no float noise is
involved beyond the documented IEEE operations). Coordinates are compared at
the float class (≤ 1e-9 rel).

Shared helpers for the other ``test_acoustic_*`` / ``test_medium_*`` modules
live here too (no test functions are imported from this module).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pyproteus import acoustic, config
from pyproteus.config.models import Domain
from tests.oracle.helpers import ORACLE_SETTINGS, batch, from_matlab, put, row, vec

RTOL = 1e-9


# ------------------------------------------------------------------ helpers


def assert_float(actual: Any, expected: Any, atol: float = 0.0, err_msg: str = "") -> None:
    """D-13 float class: ``|a - e| <= 1e-9 |e| + atol``; same shape; NaN/Inf positions equal."""
    a, e = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    assert a.shape == e.shape, f"shape {a.shape} != {e.shape} {err_msg}"
    np.testing.assert_allclose(a, e, rtol=RTOL, atol=atol, equal_nan=True, err_msg=err_msg)


def build_settings(eng: Any, oracle_path: Path, var: str = "S", folder: str = "renal_tree") -> None:
    """MainGUI start-up + updateParameters (as ``tests/helpers/make_default_settings.m``)."""
    paths = f"struct('GeometriesPath', '{oracle_path / 'geometry_data'}')"
    eng.eval(
        f"{var}.Microbubble = reset_microbubble(); {var}.Medium = reset_medium();"
        f"{var}.Acquisition = reset_acquisition(); {var}.Transducer = reset_transducer();"
        f"{var}.Transmit = reset_transmit({var}.Transducer);"
        f"{var}.SimulationParameters = reset_simulation_parameters();"
        f"{var}.Geometry = reset_geometry(struct('Folder','{folder}'), {paths});",
        nargout=0,
    )
    update_settings(eng, var)


def update_settings(eng: Any, var: str = "S") -> None:
    """``update_derived_settings`` + the fields ``write_to_file`` derives (grid size, delays)."""
    eng.eval(
        f"{var}.Geometry.Domain.Margin = 2*{var}.Medium.SpeedOfSound/"
        f"{var}.Transmit.CenterFrequency;"
        f"{var}.Geometry = compute_simulation_domain({var}.Geometry, {var}.Transducer,"
        f" {var}.Transmit);"
        f"{var}.SimulationParameters = update_simulation_parameters("
        f"{var}.SimulationParameters, {var}.Medium, {var}.Transmit.CenterFrequency);"
        # GridSize as write_to_file.m sets it: c/(f0*PointsPerWavelength).
        f"{var}.SimulationParameters.GridSize = {var}.Medium.SpeedOfSound/("
        f"{var}.Transmit.CenterFrequency*{var}.SimulationParameters.PointsPerWavelength);"
        # and the delays/apodization write_to_file.m adds:
        f"if ~isfield({var}.Transducer,'Configuration'), {var}.Transducer.Configuration"
        f" = 'Linear'; end;"
        f"if strcmp({var}.Transmit.DelayType,'Compute delays'), {var}.Transmit = "
        f"compute_delays({var}.Transmit, {var}.Transducer, {var}.Medium); end;"
        f"if strcmp({var}.Transmit.DelayType,'No delays'), {var}.Transmit.Delays = "
        f"zeros(1, {var}.Transducer.NumberOfElements); end;"
        f"if strcmp({var}.Transmit.ApodizationType,'Uniform apodization'), "
        f"{var}.Transmit.Apodization = ones(1, {var}.Transducer.NumberOfElements); end;",
        nargout=0,
    )


def load_settings(eng: Any, tmp: Path, var: str = "S") -> config.Settings:
    """Save MATLAB settings struct ``var`` and read it into the Python models."""
    file = tmp / f"{var}_settings.mat"
    eng.eval(f"save('{file}', '-struct', '{var}');", nargout=0)
    return config.load_settings_mat(file)


def matlab_value(eng: Any, expr: str, tmp: Path, name: str = "pyp_val__") -> Any:
    """Evaluate ``expr`` in MATLAB and return it as a MATLAB value tree (via a .mat file)."""
    file = tmp / f"{name}.mat"
    eng.eval(f"{name} = {expr}; save('{file}', '{name}'); clear {name}", nargout=0)
    return config.loadmat(file)[name]


def assert_grid(grid: acoustic.Grid, ref: dict[str, Any], err_msg: str = "") -> list[str]:
    """``define_grid(..., pml_rounding="floor")`` vs MATLAB ``pgrid`` (Q-90 fix, D-19).

    Per axis, with ``M - N`` even everything matches. With ``M - N`` odd MATLAB's
    PML is ``ceil`` (core ``N - 1``) and pyproteus' ``floor`` (core ``N + 1``):
    the core has exactly 2 more points, the PML 1 fewer, the full size is the
    same, and the coordinates continue MATLAB's (first ``N - 1`` equal, then
    ``+dx`` steps). Returns the padding parity per axis (``"even"``/``"odd"``).
    """
    parities = []
    for k in ("dx", "dy", "dz", "dt"):
        assert_float(getattr(grid, k), ref[k].item(), err_msg=f"{k} {err_msg}")
    assert grid.full_size == tuple(int(v) for v in ref["full_size"].reshape(-1)), err_msg
    assert ref["full_size"].shape == (3, 1)
    pml = ref["PML"]
    for ax, pk, M in zip("xyz", ("X_SIZE", "Y_SIZE", "Z_SIZE"), grid.full_size, strict=True):
        n_ref, n_py = int(ref["N" + ax].item()), getattr(grid, "N" + ax)
        pml_ref, pml_py = int(pml[pk].item()), getattr(grid.PML, pk)
        assert M == n_py + 2 * pml_py == n_ref + 2 * pml_ref, f"{ax} sizes {err_msg}"
        e = ref[ax]
        assert e.shape == (1, n_ref), f"{ax} shape {err_msg}"
        coords = getattr(grid, ax)
        assert coords.shape == (n_py,)
        # MATLAB's core is always M - 2*ceil(...), so its parity cannot tell the cases
        # apart; the domain count N (MATLAB: n_ref + 1 when odd) decides via pyproteus.
        if n_py == n_ref:  # even padding: identical
            parities.append("even")
            assert pml_py == pml_ref, f"{ax} {err_msg}"
        else:  # odd padding (D-19): MATLAB ceil vs pyproteus floor
            parities.append("odd")
            assert (n_py, pml_py) == (n_ref + 2, pml_ref - 1), f"{ax} D-19 {err_msg}"
            assert (M - (n_ref + 1)) % 2 == 1, f"{ax}: not an odd-padding axis {err_msg}"
        d = grid.dx
        assert_float(coords[:n_ref], e.reshape(-1), atol=1e-12 * d, err_msg=f"{ax} {err_msg}")
        if n_py > n_ref:
            start = coords[0] if n_ref == 0 else e.reshape(-1)[-1]
            k0 = 0 if n_ref == 0 else 1
            expect = start + np.arange(k0, k0 + n_py - n_ref) * d
            assert_float(coords[n_ref:], expect, atol=1e-12 * d, err_msg=f"{ax} tail {err_msg}")
    assert grid.PML.Alpha == pml["Alpha"].item()
    assert bool(grid.sensor_on_grid) == bool(ref["sensor_on_grid"].item())
    return parities


def assert_grid_exact(grid: acoustic.Grid, ref: dict[str, Any], err_msg: str = "") -> None:
    """Default ``define_grid`` (``pml_rounding="ceil"``) is MATLAB's grid: sizes, PML, coords."""
    m = grid_from_matlab(ref)
    assert (grid.Nx, grid.Ny, grid.Nz) == (m.Nx, m.Ny, m.Nz), err_msg
    assert grid.PML == m.PML and grid.full_size == m.full_size, err_msg
    for ax in "xyz":
        assert_float(
            getattr(grid, ax), getattr(m, ax), atol=1e-12 * m.dx, err_msg=f"{ax} {err_msg}"
        )


def grid_from_matlab(ref: dict[str, Any]) -> acoustic.Grid:
    """``acoustic.Grid`` holding MATLAB's ``pgrid`` exactly.

    Downstream oracle tests (voxelisation, medium) use it so both sides see the
    same grid despite the D-19 core-size divergence of ``define_grid``.
    """
    pml = ref["PML"]
    return acoustic.Grid(
        x=ref["x"].reshape(-1).astype(float),
        y=ref["y"].reshape(-1).astype(float),
        z=ref["z"].reshape(-1).astype(float),
        Nx=int(ref["Nx"].item()),
        Ny=int(ref["Ny"].item()),
        Nz=int(ref["Nz"].item()),
        dx=float(ref["dx"].item()),
        dy=float(ref["dy"].item()),
        dz=float(ref["dz"].item()),
        dt=float(ref["dt"].item()),
        full_size=tuple(int(v) for v in ref["full_size"].reshape(-1)),
        PML=acoustic.PML(
            X_SIZE=int(pml["X_SIZE"].item()),
            Y_SIZE=int(pml["Y_SIZE"].item()),
            Z_SIZE=int(pml["Z_SIZE"].item()),
            Alpha=float(pml["Alpha"].item()),
        ),
        sensor_on_grid=bool(ref["sensor_on_grid"].item()),
    )


# ------------------------------------------------------- optimize_grid_size

FIXED_CASES = [
    # (N, expand_range, maxPrime)
    (100, 30, 5),  # define_grid: single candidate N + 30, then grows
    (90, 30, 5),  # 120 = 2^3*3*5 accepted at once
    (1, 30, 5),
    (1, 0, 5),  # factor(1) = 1
    (2, [0, 0], 2),
    (97, [0, 10], 5),
    (97, [10, 0], 5),  # min/max of the range, order irrelevant
    (101, [0, 4], 3),
    (127, [0, 1], 7),
    (5, [0, 3], 5),  # ties: several candidates with largest factor 5 → first
    (24, [0, 8], 3),  # 24 and 27 and 32: first minimum (24, factor 3; 32 has 2)
    (31, [0, 1], 2),  # grows to 32
    (1000, 30, 7),
    (511, [-3, 3], 5),  # negative expansion allowed by the colon
    (251, 2, 11),
    (17, 30, 13),
]


def test_optimize_grid_size_fixed(matlab_engine: Any) -> None:
    ns = [float(c[0]) for c in FIXED_CASES]
    ranges = [row(np.atleast_1d(np.asarray(c[1], dtype=float))) for c in FIXED_CASES]
    primes = [float(c[2]) for c in FIXED_CASES]
    (out,) = batch(matlab_engine, "@optimize_grid_size", ns, ranges, primes)
    for case, ref in zip(FIXED_CASES, out, strict=True):
        assert acoustic.optimize_grid_size(*case) == int(ref.item()), case


@ORACLE_SETTINGS
@given(
    st.lists(
        st.tuples(
            st.integers(1, 3000),
            st.integers(-5, 40),
            st.integers(0, 40),
            st.sampled_from([2, 3, 5, 7, 11, 13]),
        ),
        min_size=1,
        max_size=40,
    )
)
def test_optimize_grid_size_random(matlab_engine: Any, cases: list) -> None:
    cases = [(n, sorted([a, b]), p) for n, a, b, p in cases if n + min(a, b) >= 1]
    if not cases:
        return
    (out,) = batch(
        matlab_engine,
        "@optimize_grid_size",
        [float(c[0]) for c in cases],
        [row(np.asarray(c[1], dtype=float)) for c in cases],
        [float(c[2]) for c in cases],
    )
    for case, ref in zip(cases, out, strict=True):
        assert acoustic.optimize_grid_size(*case) == int(ref.item()), case


def test_largest_prime_factor(matlab_engine: Any) -> None:
    ns = np.arange(0, 2500, dtype=float)
    put(matlab_engine, pyp_n__=row(ns))
    ref = vec(matlab_engine.eval("arrayfun(@(n) max(factor(n)), pyp_n__)"))
    matlab_engine.eval("clear pyp_n__", nargout=0)
    mine = [acoustic.largest_prime_factor(int(n)) for n in ns]
    np.testing.assert_array_equal(mine, ref)


# ---------------------------------------------------------------- define_grid


def _domain_cases() -> list[dict[str, Any]]:
    dx = 1.1e-4
    cases = [
        # bounds exactly on multiples of dx (k*dx: ceil/floor of an exact-ish ratio)
        dict(dx=dx, X=(0.0, 100 * dx), Y=(-20 * dx, 20 * dx), Z=(-7 * dx, 9 * dx)),
        # half-integer ratios, odd / non-cubic
        dict(dx=dx, X=(0.5 * dx, 60.5 * dx), Y=(-10.5 * dx, 33.5 * dx), Z=(-0.5 * dx, 0.5 * dx)),
        # single grid point along z (Zmin == Zmax == 0)
        dict(dx=dx, X=(1e-3, 3e-3), Y=(-1e-3, 1e-3), Z=(0.0, 0.0)),
        # bounds just inside / outside a node
        dict(dx=dx, X=(dx * (1 - 1e-12), dx * (40 + 1e-12)), Y=(-3e-4, 2.9e-4), Z=(-1e-4, 1e-4)),
        # renal-tree-like domain, coarse spacing
        dict(dx=1.925e-4, X=(1.4e-3, 1.48e-2), Y=(-1.3e-2, 1.3e-2), Z=(-2e-3, 2.1e-2)),
        # MaxPrime / PMLmin overrides
        dict(dx=dx, X=(0.0, 5e-3), Y=(-2e-3, 2e-3), Z=(-1e-3, 1e-3), MaxPrime=7, PMLmin=10),
        dict(dx=dx, X=(0.0, 5e-3), Y=(-2e-3, 2e-3), Z=(-1e-3, 1e-3), MaxPrime=3, PMLmin=20),
        dict(dx=dx, X=(0.0, 5e-3), Y=(-2e-3, 2e-3), Z=(-1e-3, 1e-3), MaxPrime=2),
    ]
    return cases


def test_define_grid_domains(matlab_engine: Any, tmp_path: Path) -> None:
    """All domain cases; together they must hit both padding parities (D-19)."""
    eng = matlab_engine
    seen: list[str] = []
    for case in _domain_cases():
        (x0, x1), (y0, y1), (z0, z1) = case["X"], case["Y"], case["Z"]
        sp: dict[str, Any] = dict(GridSize=case["dx"], SamplingRate=33e6, SensorOnGrid=True)
        for opt in ("MaxPrime", "PMLmin"):
            if opt in case:
                sp[opt] = float(case[opt])
        domain = dict(Xmin=x0, Xmax=x1, Ymin=y0, Ymax=y1, Zmin=z0, Zmax=z1)
        values = {**sp, **domain}
        put(eng, **{f"pyp_{k}__": float(v) for k, v in values.items() if k != "SensorOnGrid"})
        fields = ",".join(f"'{k}',pyp_{k}__" for k in sp if k != "SensorOnGrid")
        dfields = ",".join(f"'{k}',pyp_{k}__" for k in domain)
        eng.eval(
            f"pyp_SP__ = struct({fields}, 'SensorOnGrid', true);"
            f"pyp_G__ = struct('Domain', struct({dfields}));",
            nargout=0,
        )
        ref = matlab_value(eng, "define_grid(pyp_SP__, pyp_G__)", tmp_path)
        eng.eval("clear pyp_*", nargout=0)
        args = (config.SimulationParameters(**sp), config.Geometry(Domain=Domain(**domain)))
        assert_grid_exact(acoustic.define_grid(*args), ref, str(case))  # default: MATLAB's grid
        floor = acoustic.define_grid(*args, pml_rounding="floor")  # Q-90 fix (D-19)
        seen += assert_grid(floor, ref, err_msg=str(case))
    assert "even" in seen and "odd" in seen, seen


@pytest.mark.parametrize("folder", ["renal_tree", "mouse_brain"])
@pytest.mark.parametrize("ppw", [None, 3, 4.5])
def test_define_grid_default_settings(
    matlab_engine: Any, oracle: Any, tmp_path: Path, folder: str, ppw: float | None
) -> None:
    eng = matlab_engine
    build_settings(eng, oracle.path, folder=folder)
    if ppw is not None:
        eng.eval(f"S.SimulationParameters.PointsPerWavelength = {ppw};", nargout=0)
        update_settings(eng)
    settings = load_settings(eng, tmp_path)
    ref = matlab_value(eng, "define_grid(S.SimulationParameters, S.Geometry)", tmp_path)
    eng.eval("clear S", nargout=0)
    grid = acoustic.define_grid(settings.SimulationParameters, settings.Geometry)
    assert_grid_exact(grid, ref, f"{folder} ppw={ppw}")  # default "ceil" = MATLAB (D-19)
    floor = acoustic.define_grid(
        settings.SimulationParameters, settings.Geometry, pml_rounding="floor"
    )
    parities = assert_grid(floor, ref, err_msg=f"{folder} ppw={ppw}")
    if ppw is None:  # measured: default settings pad every axis oddly (Q-90 applies)
        assert parities == ["odd", "odd", "odd"], parities
    assert grid.shape == (grid.Nx, grid.Ny, grid.Nz)
    # every full size has prime factors <= 5 (default MaxPrime)
    assert all(acoustic.largest_prime_factor(m) <= 5 for m in grid.full_size)


def test_matlab_value_roundtrip(matlab_engine: Any, tmp_path: Path) -> None:
    """The .mat exchange keeps sizes (guards the helper used above)."""
    v = matlab_value(matlab_engine, "struct('a', 1:3, 'b', zeros(2,0), 'c', true)", tmp_path)
    assert v["a"].shape == (1, 3) and v["b"].shape == (2, 0) and v["c"].dtype == bool
    assert from_matlab(1.0).shape == (1, 1)
