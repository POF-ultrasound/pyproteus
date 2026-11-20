"""k-Wave tracer bullet: MATLAB k-Wave 1.4.1 vs k-wave-python 0.6.2 (M0).

Tiny PROTEUS-style hybrid cases (``tools/tracer_bullet/cases.py``: 37x29x23
grid, heterogeneous nonlinear absorbing medium, asymmetric source and sensor
masks, PML outside, ``Smooth=false``) run through

* MATLAB ``kspaceFirstOrder3DC`` (k-Wave's MATLAB front end + the C++ binary),
* k-wave-python ``kspaceFirstOrder(backend="cpp")`` on the *same* binary, and
* k-wave-python ``kspaceFirstOrder(backend="python")`` (NumPy solver),

with the row-order / scaling mapping of ``tools/tracer_bullet/adapter.py``.
Comparison class: k-Wave fields <= 1e-5 relative (ROADMAP §4).

Needs MATLAB (shared ``matlab_engine`` fixture), MATLAB k-Wave 1.4.x
(``KWAVE_MATLAB_PATH``), k-wave-python 0.6.2 and the binary's runtime
libraries (``tools/tracer_bullet/setup_libs.sh`` or Homebrew); skips otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))

from tracer_bullet import adapter, ordering  # noqa: E402  (numpy only)

try:  # load conda's shared libraries before matlab.engine (B-1, B-11)
    from tracer_bullet import cases
    from tracer_bullet import kwave_tracer as kt

    kt.preload_before_engine()
    HAVE_KWAVE = True
except ImportError:  # k-wave-python / scipy / h5py missing
    HAVE_KWAVE = False

TOL = 1e-5  # k-Wave fields, single precision (ROADMAP §4)
CASE_NAMES = ("pressure", "velocity", "pressure_nodisp")
ROW_TOL = 2e-5  # per sensor row (rows >= 1e-3 of the peak row norm): float32 noise floor


# ------------------------------------------------------------------------ row order (no MATLAB)


@pytest.mark.parametrize("shape", [(37, 29, 23), (5, 1, 7), (1, 4, 3), (6, 6, 6)])
@pytest.mark.parametrize("fill", [0.0, 0.3, 1.0])
def test_row_order_maps_find_order_to_c_order(shape, fill) -> None:
    rng = np.random.default_rng(hash((shape, fill)) % 2**32)
    mask = rng.random(shape) < fill
    if fill == 1.0:
        assert mask.all()
    # rows tagged with their voxel subscripts, built independently of ordering.py
    ijk = np.indices(shape).reshape(3, -1)  # C-order enumeration of all voxels
    lin_f = np.ravel_multi_index(ijk, shape, order="F")
    rows_c = ijk.T[mask.ravel()]  # == np.argwhere(mask): C order
    rows_f = ijk.T[np.argsort(lin_f, kind="stable")][mask.ravel(order="F")]  # MATLAB find order
    assert np.array_equal(rows_c, np.argwhere(mask))
    assert np.array_equal(ordering.rows_f_to_c(rows_f, mask), rows_c)
    assert np.array_equal(ordering.rows_c_to_f(rows_c, mask), rows_f)
    perm = ordering.f_to_c_rows(mask)
    assert np.array_equal(perm[ordering.c_to_f_rows(mask)], np.arange(perm.size))
    # matlab_find: 1-based column-major linear indices of the rows in find order
    expected = np.ravel_multi_index(tuple(rows_f.T), shape, order="F") + 1
    assert np.array_equal(ordering.matlab_find(mask), expected)


def test_row_order_rejects_wrong_row_count() -> None:
    mask = np.zeros((3, 4, 5), bool)
    mask[1, 2, 3] = mask[2, 0, 1] = True
    with pytest.raises(ValueError, match="rows"):
        ordering.rows_f_to_c(np.zeros((3, 4)), mask)


def test_adapter_cpp_sensor_shape_guards() -> None:
    mask = np.zeros((3, 4, 5), bool)
    mask[0, 1, 2] = mask[2, 3, 4] = True
    p = np.arange(10.0).reshape(5, 2)  # (Nt, n_sensor) as k-wave-python 0.6.2 cpp returns it
    assert np.array_equal(adapter.sensor_rows_from_kwp(p, mask, 5, "cpp"), p.T)
    with pytest.raises(ValueError, match="shape"):
        adapter.sensor_rows_from_kwp(p.T, mask, 5, "cpp")


@pytest.mark.parametrize("n", [3, 7])
def test_adapter_undoes_kwp_time_permutation_when_nt_equals_n_sensor(n) -> None:
    """With Nt == n_sensor, k-wave-python 0.6.2 permutes the time axis (B-10); undo it.

    Runs k-wave-python's own ``CppSimulation._fix_output_order`` on a fake output.
    """
    kwave_cpp = pytest.importorskip("kwave.solvers.cpp_simulation")
    from types import SimpleNamespace

    shape = (4, 3, 5)
    rng = np.random.default_rng(n)
    while True:  # a mask whose F->C permutation is not the identity
        mask = np.zeros(shape, bool)
        mask.ravel()[rng.choice(mask.size, n, replace=False)] = True
        if not np.array_equal(ordering.f_to_c_rows(mask), np.arange(n)):
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
    assert np.array_equal(adapter.sensor_rows_from_kwp(out, mask, n, "cpp"), truth)


def test_adapter_source_scale_axes_and_alpha_mode() -> None:
    mask = np.zeros((3, 4, 5), bool)
    mask[1, 2, 3] = True
    c = np.full(mask.shape, 1500.0)
    spacing = (1e-4, 2e-4, 4e-4)
    for kind, d in zip(("ux", "uy", "uz"), spacing, strict=True):
        assert adapter.additive_source_scale(kind, mask, c, 1e-8, spacing)[0] == 2 * 1500 * 1e-8 / d
    with pytest.raises(ValueError, match="anisotropic"):
        adapter.additive_source_scale("p", mask, c, 1e-8, spacing)
    with pytest.raises(ValueError, match="spacing"):
        adapter.additive_source_scale("ux", mask, c, 1e-8, (1e-4, 1e-4))
    medium = {"sound_speed": c, "alpha_mode": "no_dispersion"}
    assert "alpha_mode" not in adapter.medium_for_kwp(medium, "cpp")
    with pytest.raises(ValueError, match="B-14"):
        adapter.medium_for_kwp(medium, "python")


# ------------------------------------------------------------------------ MATLAB vs k-wave-python


@pytest.fixture(scope="module")
def kwave_engine(matlab_engine):
    """Shared engine with MATLAB k-Wave and the tracer helpers on the path."""
    if not HAVE_KWAVE:
        pytest.skip("k-wave-python (and scipy, h5py) needed")
    if not (kt.KWAVE_MATLAB / "kspaceFirstOrder3DC.m").is_file():
        pytest.skip(f"MATLAB k-Wave not found at {kt.KWAVE_MATLAB} (set KWAVE_MATLAB_PATH)")
    if not kt.binary_available():
        pytest.skip("k-Wave binary does not start: run tools/tracer_bullet/setup_libs.sh")
    kt.prepare_engine(matlab_engine)
    kt.check_engine_layout(matlab_engine)
    yield matlab_engine
    matlab_engine.rmpath(str(kt.HERE), nargout=0)
    matlab_engine.rmpath(str(kt.KWAVE_MATLAB), nargout=0)


@pytest.fixture(scope="module", params=CASE_NAMES)
def run(request, kwave_engine, tmp_path_factory):
    """One case through MATLAB and both k-wave-python backends (rows in find order)."""
    case = cases.CASES[request.param]()
    data = tmp_path_factory.mktemp(f"kwave_{case.name}")
    return {
        "case": case,
        "data": data,
        "matlab": kt.run_matlab(kwave_engine, case, data),
        "cpp": kt.run_kwp(case, data, "cpp"),
        "python": None if "alpha_mode" in case.medium else kt.run_kwp(case, None, "python"),
    }


def test_masks_are_asymmetric(run) -> None:
    """Flipping any axis changes the masks, so an order mix-up cannot cancel out.

    (A transpose is impossible: the grid is not cubic.)"""
    c = run["case"]
    src = c.source["p_mask" if "p_mask" in c.source else "u_mask"]
    for m in (src, c.sensor_mask):
        for axis in range(3):
            assert not np.array_equal(m, np.flip(m, axis))
    # source rows differ from each other (distinct delays and amplitudes)
    sig = c.source.get("p", c.source.get("ux"))
    assert len(np.unique(sig, axis=0)) == sig.shape[0]


def test_matlab_find_matches_engine(run, kwave_engine) -> None:
    import matlab

    c = run["case"]
    for mask in (c.sensor_mask, *[v for k, v in c.source.items() if k.endswith("mask")]):
        idx = np.asarray(kwave_engine.find(matlab.logical(mask)), dtype=np.int64).ravel()
        assert np.array_equal(idx, ordering.matlab_find(mask))


def test_cpp_sensor_data_matches_matlab(run) -> None:
    pm = run["matlab"]
    assert pm.shape == (int(run["case"].sensor_mask.sum()), run["case"].Nt)
    assert np.abs(pm).max() > 1e3  # the wave reached the sensors
    assert kt.rel_err(run["cpp"], pm) <= TOL
    assert kt.rel_err_rows(run["cpp"], pm) <= ROW_TOL  # weak "bubble" rows too


def test_python_backend_matches_matlab(run) -> None:
    if run["python"] is None:
        pytest.skip("alpha_mode: MATLAB 3DC ignores it, the python backend would apply it (B-14)")
    assert kt.rel_err(run["python"], run["matlab"]) <= TOL
    assert kt.rel_err_rows(run["python"], run["matlab"]) <= ROW_TOL


def test_matlab_3dc_ignores_alpha_mode(run, kwave_engine) -> None:
    """PROTEUS' alpha_mode='no_dispersion' has no effect on the 3DC path (B-14)."""
    c = run["case"]
    if "alpha_mode" not in c.medium:
        pytest.skip("case without alpha_mode")
    import copy

    c2 = copy.deepcopy(c)
    c2.name += "_nomode"
    del c2.medium["alpha_mode"]
    assert np.array_equal(kt.run_matlab(kwave_engine, c2, run["data"]), run["matlab"])


def test_cpp_input_file_matches_matlab(run) -> None:
    """Same HDF5 input as MATLAB: source (scaling + row order) bit-identical.

    Only difference: the staggered densities ``rho0_sg*``. k-wave-python
    averages with ``np.roll`` (the last plane wraps to the first, inside the
    PML); MATLAB k-Wave keeps ``rho0`` there (B-13). Elsewhere <= 2 ulp.
    """
    c, data = run["case"], run["data"]
    mat = kt.read_h5(data / f"matlab_{c.name}_input.h5")
    kwp = kt.read_h5(data / f"kwp_cpp_{c.name}" / "kwave_input.h5")
    assert set(mat) == set(kwp)
    for key in sorted(mat):
        if key.startswith("rho0_sg"):
            continue
        assert np.array_equal(mat[key], kwp[key]), key
    for axis, key in enumerate(("rho0_sgx", "rho0_sgy", "rho0_sgz")):
        a = mat[key].transpose(2, 1, 0).astype(np.float64)  # h5py (z,y,x) -> [x,y,z]
        b = kwp[key].transpose(2, 1, 0).astype(np.float64)
        rel = np.abs(a - b) / a
        assert np.delete(rel, -1, axis=axis).max() <= 2.4e-7, key
        assert np.take(rel, -1, axis=axis).max() > 0, key  # B-13 still present (update if fixed)


def test_wrong_source_row_order_is_detected(run) -> None:
    """Feeding C-order source rows to the cpp backend breaks the comparison."""
    p_wrong = kt.run_kwp(run["case"], run["data"], "cpp", raw_rows="c")
    assert kt.rel_err(p_wrong, run["matlab"]) > 0.1


def test_wrong_sensor_row_order_is_detected(run) -> None:
    c = run["case"]
    p_wrong = ordering.rows_c_to_f(run["cpp"], c.sensor_mask)  # treat F rows as C rows
    assert kt.rel_err(p_wrong, run["matlab"]) > 0.1


def test_unscaled_cpp_source_is_wrong(run) -> None:
    """k-wave-python 0.6.2's cpp backend does not scale additive sources (B-10)."""
    with np.errstate(invalid="ignore"):
        p_raw = kt.run_kwp(run["case"], run["data"], "cpp", raw_rows="f", unscaled=True)
        err = kt.rel_err(p_raw, run["matlab"])
    assert not err <= 0.1  # NaN (unstable) or far off


@pytest.mark.xfail(reason="B-12: legacy kspaceFirstOrder3D scales p by c0 of the next voxel")
def test_legacy_api_matches_matlab(run) -> None:
    """Documents why the M3 adapter uses kspaceFirstOrder(), not the legacy API (XPASS = fixed)."""
    try:
        p = kt.run_kwp_legacy(run["case"], run["data"])
    except TypeError:  # velocity case: per-voxel ux crashes (B-12)
        pytest.xfail("B-12: legacy API crashes on per-voxel velocity sources")
    assert kt.rel_err(p, run["matlab"]) <= TOL
