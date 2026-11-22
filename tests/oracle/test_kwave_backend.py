"""``pyproteus.backends.kwave.kspace_first_order`` vs MATLAB k-Wave 1.4.1 ``kspaceFirstOrder3DC``.

Reuses the tracer-bullet cases (``tools/tracer_bullet/cases.py``: 37x29x23
grid, heterogeneous nonlinear absorbing medium, asymmetric masks, PML outside,
``Smooth=false``, ``DataCast='single'``) and the MATLAB runner of
``tools/tracer_bullet/kwave_tracer.py`` (options as PROTEUS ``run_simulation.m``).
Both sides run the same C++ binary (D-10), so this checks the adapter's input
preparation and row order. Cases:

* ``pressure``, ``velocity`` (``ux``+``uy``), ``velocity_ux`` (``ux`` only):
  the transducer sources of ``define_source_transducer.m``;
* ``mixed``: dipole transducer (``ux``+``uz``) plus pressure point sources, as
  ``update_source.m`` builds them in ``full_simulator.m``;
* ``pressure_nodisp``: ``alpha_mode='no_dispersion'`` (dropped on every backend, D-9);
* ``nt_eq_n_sensor``: ``Nt == n_sensor``, recording ``p``, ``p_max``, ``p_min``;
* ``pressure_field``: same mask, only ``{'p_max','p_min'}`` (``main_pressure_field.m``).

``test_binary_command_matches_matlab`` captures the command line MATLAB 3DC
gives the binary (a logging shim in front of the wrapper) and checks the
adapter builds the same one for every ``sensor.record`` PROTEUS uses.

Comparison class: k-Wave fields <= 1e-5 relative (ROADMAP §4).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from pyproteus.backends import kwave as kw

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))

try:  # load conda's shared libraries before matlab.engine (B-1, B-11)
    from tracer_bullet import cases
    from tracer_bullet import kwave_tracer as kt

    kt.preload_before_engine()
    HAVE_KWAVE = True
except ImportError:  # k-wave-python / scipy / h5py missing
    HAVE_KWAVE = False

TOL = 1e-5  # k-Wave fields, single precision (ROADMAP §4)
ROW_TOL = 2e-5  # per sensor row (rows >= 1e-3 of the peak row norm)
CASE_NAMES = (
    "pressure",
    "velocity",
    "velocity_ux",
    "mixed",
    "pressure_nodisp",
    "nt_eq_n_sensor",
    "pressure_field",
)


@pytest.fixture(scope="module")
def kwave_engine(matlab_engine):
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


def _run(case, backend: str, data_path=None) -> dict[str, np.ndarray]:
    kgrid = kw.KGrid(case.shape, (case.dx,) * 3, case.dt, case.Nt)
    return kw.kspace_first_order(
        kgrid,
        case.medium,
        case.source,
        {"mask": case.sensor_mask, "record": list(case.record)},
        pml_size=case.pml_size,
        pml_alpha=case.pml_alpha,
        pml_inside=False,
        smooth=False,
        data_cast="single",
        backend=backend,
        binary=kt.WRAPPER if backend == "cpp" else None,
        binary_env=kt.binary_env() if backend == "cpp" else None,
        data_path=data_path,
        delete_data=data_path is None,
    )


@pytest.fixture(scope="module", params=CASE_NAMES)
def run(request, kwave_engine, tmp_path_factory):
    case = cases.CASES[request.param]()
    data = tmp_path_factory.mktemp(f"kwave_backend_{case.name}")
    return {
        "case": case,
        "data": data,
        "matlab": kt.run_matlab_fields(kwave_engine, case, data),
        "cpp": _run(case, "cpp", data / "pyproteus_cpp"),
        "python": _run(case, "python"),
    }


def test_case_setup(run) -> None:
    """Masks asymmetric, source rows distinct, MATLAB recorded what was asked."""
    c, pm = run["case"], run["matlab"]
    masks = [v for k, v in c.source.items() if k.endswith("mask")] + [c.sensor_mask]
    for m in masks:
        for axis in range(3):
            assert not np.array_equal(m, np.flip(m, axis))
    # per source mask, the rows of all its signals together are distinct
    # (a single component may repeat, e.g. uy = 0 where the normal is along x)
    for mask_key, kinds in (("p_mask", ("p",)), ("u_mask", ("ux", "uy", "uz"))):
        sig = [c.source[k] for k in kinds if k in c.source]
        if sig:
            rows = np.hstack(sig)
            assert len(np.unique(rows, axis=0)) == rows.shape[0], mask_key
    n = int(c.sensor_mask.sum())
    assert set(pm) == set(c.record)
    if "p" in pm:
        assert pm["p"].shape == (n, c.Nt)
        assert np.abs(pm["p"]).max() > 1e3  # the wave reached the sensors
    else:
        assert np.abs(pm["p_max"]).max() > 1e3
    if c.name == "nt_eq_n_sensor":
        assert n == c.Nt


@pytest.mark.parametrize("backend", ["cpp", "python"])
def test_sensor_data_matches_matlab(run, backend) -> None:
    c, pm, out = run["case"], run["matlab"], run[backend]
    assert set(out) == set(c.record)
    assert all(v.dtype == np.float32 for v in out.values())
    if "p" in c.record:
        assert out["p"].shape == pm["p"].shape
        assert kt.rel_err(out["p"], pm["p"]) <= TOL
        assert kt.rel_err_rows(out["p"], pm["p"]) <= ROW_TOL  # weak "bubble" rows too
    for key in set(c.record) - {"p"}:
        assert out[key].shape == pm[key].shape == (c.sensor_mask.sum(),), key
        assert kt.rel_err(out[key], pm[key]) <= TOL, key


def test_cpp_input_file_matches_matlab(run) -> None:
    """The adapter's cpp input file carries MATLAB's data (``rho0_sg*`` aside, B-13)."""
    c, data = run["case"], run["data"]
    mat = kt.read_h5(data / f"matlab_{c.name}_input.h5")
    (run_dir,) = (data / "pyproteus_cpp").iterdir()
    ours = kt.read_h5(run_dir / "kwave_input.h5")
    kt.assert_inputs_close(mat, ours)


def test_wrong_sensor_row_order_is_detected(run) -> None:
    """The tolerance is tight enough to catch a C/F row mix-up (and a transposed p)."""
    c = run["case"]
    key = "p" if "p" in c.record else "p_max"
    wrong = kw.rows_c_to_f(run["cpp"][key], c.sensor_mask)
    assert kt.rel_err(wrong, run["matlab"][key]) > 0.1
    if key == "p" and c.Nt == c.sensor_mask.sum():  # square: a transpose would fit the shape
        assert kt.rel_err(run["cpp"]["p"].T, run["matlab"]["p"]) > 0.1


SHIM = """#!/bin/sh
printf '%s\\n' "$@" > "$ARGV_LOG"
exec "{wrapper}" "$@"
"""


def _argv_without_files(path: Path) -> list[str]:
    argv = path.read_text().split("\n")[:-1]
    for flag in ("-i", "-o"):
        i = argv.index(flag)
        del argv[i : i + 2]
    return argv


@pytest.mark.parametrize("record", [("p",), ("p_max", "p_min"), ("p", "p_max", "p_min")])
def test_binary_command_matches_matlab(kwave_engine, tmp_path, record) -> None:
    """Same binary flags as MATLAB kspaceFirstOrder3DC (record modes PROTEUS uses, NumThreads)."""
    shim_dir = tmp_path / "shim"
    shim_dir.mkdir()
    shim = shim_dir / kw.OMP_NAME
    shim.write_text(SHIM.format(wrapper=kt.WRAPPER))
    shim.chmod(0o755)
    case = cases.CASES["pressure"]()
    case.record = record
    for threads in (None, 2):
        extra = {} if threads is None else {"num_threads": float(threads)}
        mat_log, our_log = tmp_path / f"matlab_{threads}.txt", tmp_path / f"ours_{threads}.txt"
        kwave_engine.setenv("ARGV_LOG", str(mat_log), nargout=0)
        kt.run_matlab_fields(kwave_engine, case, tmp_path / "m", binary_dir=shim_dir, **extra)
        kgrid = kw.KGrid(case.shape, (case.dx,) * 3, case.dt, case.Nt)
        kw.kspace_first_order(
            kgrid, case.medium, case.source, {"mask": case.sensor_mask, "record": list(record)},
            pml_size=case.pml_size, pml_alpha=case.pml_alpha, binary=shim, num_threads=threads,
            binary_env={**kt.binary_env(), "ARGV_LOG": str(our_log)},
        )  # fmt: skip
        mat = _argv_without_files(mat_log)
        assert mat == _argv_without_files(our_log)
        assert ("--p_raw" in mat) == ("p" in record)  # MATLAB's spelling of -p
