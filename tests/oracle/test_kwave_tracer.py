"""k-Wave tracer bullet: MATLAB k-Wave 1.4.1 vs k-wave-python 0.6.2 (M0).

Tiny PROTEUS-style hybrid cases (``tools/tracer_bullet/cases.py``: 37x29x23
grid, heterogeneous nonlinear absorbing medium, asymmetric source and sensor
masks, PML outside, ``Smooth=false``) run through

* MATLAB ``kspaceFirstOrder3DC`` (k-Wave's MATLAB front end + the C++ binary),
* k-wave-python ``kspaceFirstOrder(backend="cpp")`` on the *same* binary, and
* k-wave-python ``kspaceFirstOrder(backend="python")`` (NumPy solver),

with the row-order / scaling mapping of ``pyproteus.backends.kwave``
(``tests/oracle/test_kwave_backend.py`` tests the package's own runner).
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

from pyproteus.backends import kwave as kw  # noqa: E402

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


# Row-order / scaling unit tests (no MATLAB): tests/unit/test_kwave.py


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
        "python": kt.run_kwp(case, None, "python"),  # alpha_mode dropped there too (D-9)
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
        assert np.array_equal(idx, kw.matlab_find(mask))


def test_cpp_sensor_data_matches_matlab(run) -> None:
    pm = run["matlab"]
    assert pm.shape == (int(run["case"].sensor_mask.sum()), run["case"].Nt)
    assert np.abs(pm).max() > 1e3  # the wave reached the sensors
    assert kt.rel_err(run["cpp"], pm) <= TOL
    assert kt.rel_err_rows(run["cpp"], pm) <= ROW_TOL  # weak "bubble" rows too


def test_python_backend_matches_matlab(run) -> None:
    """Incl. ``pressure_nodisp``: alpha_mode dropped, so python matches 3DC (D-9)."""
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
    p_wrong = kw.rows_c_to_f(run["cpp"], c.sensor_mask)  # treat F rows as C rows
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
