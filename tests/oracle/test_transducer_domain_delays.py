"""Oracle tests: ``compute_delays`` and ``compute_simulation_domain`` against MATLAB.

Class D-13 float: ``|a - e| <= 1e-9*|e| + atol``. Delays are differences of
path lengths (``sqrt(...) - min``) and domain coordinates differences of
positions, so the absolute floor is relative to the problem's scale:
``atol = 1e-14*max|e|`` over the compared output (domain) or
``1e-14*L/c`` with ``L`` the largest finite length involved (delays).
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pyproteus import config
from pyproteus import transducer as ptr
from pyproteus.config.models import Geometry, Medium, Transducer, Transmit
from tests.oracle.helpers import ORACLE_SETTINGS, from_matlab, to_matlab

REL = 1e-9
ATOL_FRACTION = 1e-14


# ------------------------------------------------------------------ helpers


def to_engine(v: Any) -> Any:
    """Settings value tree → engine value (dict → struct; arrays keep MATLAB size)."""
    if isinstance(v, dict):
        return {k: to_engine(x) for k, x in v.items()}
    if isinstance(v, np.ndarray):
        return to_matlab(v if v.ndim >= 2 else v.reshape(1, -1))
    if isinstance(v, bool | float | str):
        return v
    if isinstance(v, int):
        return float(v)
    raise TypeError(f"not a tree value: {type(v)!r}")


def from_engine(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: from_engine(x) for k, x in v.items()}
    return v if isinstance(v, str) else from_matlab(v)


def tree(model: Any) -> Any:
    return to_engine(model.to_tree() if hasattr(model, "to_tree") else model)


def assert_float(actual: Any, expected: Any, atol: float, err_msg: str) -> None:
    a, e = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    assert a.shape == e.shape, f"shape {a.shape} != {e.shape} {err_msg}"
    assert np.array_equal(np.isnan(a), np.isnan(e)), f"NaN positions differ {err_msg}"
    inf = np.isinf(e)
    assert np.array_equal(np.isinf(a), inf), f"Inf positions differ {err_msg}"
    np.testing.assert_array_equal(a[inf], e[inf], err_msg=err_msg)
    fin = np.isfinite(e)
    err = np.abs(a[fin] - e[fin])
    bound = REL * np.abs(e[fin]) + atol
    assert np.all(err <= bound), f"max excess {np.max(err - bound, initial=0):.3g} {err_msg}"


def transducer(ttype: str, **changes: Any) -> Transducer:
    t = ptr.get_transducer_properties(Transducer(Type=ttype))
    for k, v in changes.items():
        setattr(t, k, v)
    return t


TRANSDUCERS = {
    "P4-1": transducer("P4-1"),
    "C1-6D": transducer("C1-6D"),
    "L22-14v": transducer("L22-14v"),
    "9L-D": transducer("9L-D"),
    "RC6gV": transducer("RC6gV"),
    "linear-1": transducer(
        "Custom linear",
        NumberOfElements=1,
        Pitch=3e-4,
        ElementWidth=2e-4,
        ElementHeight=5e-3,
        ElevationFocus=40e-3,
    ),
    "linear-7": transducer(
        "Custom linear",
        NumberOfElements=7,
        Pitch=1e-3 / 3,
        ElementWidth=0.3e-3,
        ElementHeight=4e-3,
        ElevationFocus=20e-3,
    ),
    "curved-1": transducer("C1-6D", NumberOfElements=1),
    "curved-7-strong": transducer("C1-6D", NumberOfElements=7, Pitch=1e-3, Radius=5e-3),
    "curved-128-flat": transducer("C1-6D", NumberOfElements=128, Pitch=0.3e-3, Radius=2.0),
}


# ------------------------------------------------------------------ delays


def matlab_delays(eng: Any, cases: list[tuple[Transmit, Transducer, Medium]]) -> list[np.ndarray]:
    eng.workspace["pyp_tx__"] = [tree(c[0]) for c in cases]
    eng.workspace["pyp_tr__"] = [tree(c[1]) for c in cases]
    eng.workspace["pyp_m__"] = [tree(c[2]) for c in cases]
    eng.eval(
        "pyp_d__ = cellfun(@(a,b,c) getfield(compute_delays(a,b,c),'Delays'),"
        " pyp_tx__, pyp_tr__, pyp_m__, 'UniformOutput', false);",
        nargout=0,
    )
    out = [from_matlab(d) for d in eng.workspace["pyp_d__"]]
    eng.eval("clear pyp_tx__ pyp_tr__ pyp_m__ pyp_d__", nargout=0)
    return out


def delay_scale(tx: Transmit, tr: Transducer, c: float) -> float:
    lengths = [abs(tr.Pitch) * tr.NumberOfElements]
    if math.isfinite(tx.LateralFocus):
        lengths.append(abs(tx.LateralFocus))
    if tr.Configuration == "Curved":
        lengths.append(tr.Radius)
    return max(lengths) / c


def check_delays(eng: Any, cases: list[tuple[Transmit, Transducer, Medium]]) -> None:
    for (tx, tr, m), expected in zip(cases, matlab_delays(eng, cases), strict=True):
        actual = ptr.compute_delays(tx, tr, m).Delays
        if expected.size == 0:
            expected = expected.reshape(1, 0)
        atol = ATOL_FRACTION * delay_scale(tx, tr, m.SpeedOfSound)
        assert_float(
            actual,
            expected,
            atol,
            f"{tr.Type} N={tr.NumberOfElements} f={tx.LateralFocus} a={tx.Angle}",
        )


FOCI = [math.inf, -math.inf, math.nan, 30e-3, -20e-3, 1e-6, 0.0]
ANGLES = [0.0, 12.5, -20.0, 45.0, 90.0, -90.0, 180.0]


@pytest.mark.parametrize("name", list(TRANSDUCERS))
def test_delays_fixed(matlab_engine: Any, name: str) -> None:
    tr = TRANSDUCERS[name]
    cases = [
        (Transmit(LateralFocus=f, Angle=a), tr, Medium(SpeedOfSound=c))
        for f in FOCI
        for a in ANGLES
        for c in (1540.0, 1480.33)
    ]
    check_delays(matlab_engine, cases)


def test_delays_default_settings(matlab_engine: Any) -> None:
    eng = matlab_engine
    eng.eval(
        "pyp_T__ = reset_transducer(); pyp_d__ = compute_delays(reset_transmit(pyp_T__),"
        " pyp_T__, reset_medium()); pyp_d__ = pyp_d__.Delays;",
        nargout=0,
    )
    expected = from_matlab(eng.workspace["pyp_d__"])
    eng.eval("clear pyp_T__ pyp_d__", nargout=0)
    t = config.reset_transducer()
    actual = ptr.compute_delays(config.reset_transmit(t), t, config.reset_medium()).Delays
    assert_float(actual, expected, 1e-14 * 1e-1 / 1540, "defaults")


def test_delays_errors(matlab_engine: Any) -> None:
    eng = matlab_engine
    bad = {
        "custom": transducer("P4-1", Configuration="Custom"),
        "lowercase": transducer("P4-1", Configuration="linear"),
    }
    for name, tr in bad.items():
        eng.workspace["pyp_tr__"] = tree(tr)
        eng.eval(
            "try, compute_delays(struct('LateralFocus',Inf,'Angle',0), pyp_tr__,"
            " struct('SpeedOfSound',1540)); pyp_r__ = 'none'; catch e, pyp_r__ = e.message;"
            " end",
            nargout=0,
        )
        assert "not supported" in eng.workspace["pyp_r__"], name
        with pytest.raises(ValueError, match="not supported"):
            ptr.compute_delays(
                Transmit(LateralFocus=math.inf, Angle=0.0), tr, Medium(SpeedOfSound=1540.0)
            )
    eng.eval("clear pyp_tr__ pyp_r__", nargout=0)
    # Odd RCA count: MATLAB silently returns a vector of the wrong length; pyproteus raises
    with pytest.raises(ValueError, match="even"):
        ptr.compute_delays(
            Transmit(LateralFocus=math.inf, Angle=0.0),
            transducer("RC6gV", NumberOfElements=255),
            Medium(SpeedOfSound=1540.0),
        )


@st.composite
def delay_cases(draw: st.DrawFn) -> tuple[Transmit, Transducer, Medium]:
    n = draw(st.integers(1, 256))
    p = draw(st.floats(5e-5, 1e-3))
    conf = draw(st.sampled_from(["Linear", "Curved", "RCA"]))
    if conf == "RCA":
        n = 2 * n
    tr = Transducer(Type="x", Configuration=conf, NumberOfElements=n, Pitch=p, ElementWidth=p)
    if conf == "Curved":
        tr.Radius = n * p / (2 * draw(st.floats(1e-3, 1.5)))
    f = draw(st.one_of(st.just(math.inf), st.floats(-0.2, 0.2)))
    a = draw(st.one_of(st.floats(-60, 60), st.sampled_from([0.0, 30.0, -30.0])))
    return Transmit(LateralFocus=f, Angle=a), tr, Medium(SpeedOfSound=draw(st.floats(1300, 1700)))


@ORACLE_SETTINGS
@given(st.lists(delay_cases(), min_size=1, max_size=20))
def test_delays_random(matlab_engine: Any, cases: list) -> None:
    check_delays(matlab_engine, cases)


# ------------------------------------------------------------------ domain

DOMAIN_FIELDS = (
    "TransducerSurface",
    "TransducerProjection",
    "Xmax",
    "Ymax",
    "Zmax",
    "Xmin",
    "Ymin",
    "Zmin",
    "Vertices",
)


def geometry(
    folder: str, start_depth: float, manual: bool, margin: float, box: tuple[float, ...] = ()
) -> Geometry:
    g = config.reset_geometry(Geometry(Folder=folder))
    g.startDepth = start_depth
    g.Domain.Manual = manual
    g.Domain.Margin = margin
    if box:
        d = g.Domain
        d.Xmin, d.Xmax, d.Ymin, d.Ymax, d.Zmin, d.Zmax = box
    return g


def matlab_domain(eng: Any, g: Geometry, tr: Transducer, tx: Transmit) -> tuple[dict, str]:
    eng.workspace["pyp_g__"] = tree(g)
    eng.workspace["pyp_tr__"] = tree(tr)
    eng.workspace["pyp_tx__"] = tree(tx)
    eng.eval(
        "lastwarn(''); pyp_w__ = warning('off',"
        " 'compute_simulation_domain:geometryBehindTransducer');"
        " pyp_out__ = compute_simulation_domain(pyp_g__, pyp_tr__, pyp_tx__);"
        " [~, pyp_wid__] = lastwarn(); warning(pyp_w__);"
        " pyp_fields__ = strjoin(fieldnames(pyp_out__.Domain), ',');",
        nargout=0,
    )
    out = from_engine(eng.workspace["pyp_out__"])
    out["_domain_fields"] = eng.workspace["pyp_fields__"]
    wid = eng.workspace["pyp_wid__"]
    eng.eval("clear pyp_g__ pyp_tr__ pyp_tx__ pyp_out__ pyp_w__ pyp_wid__ pyp_fields__", nargout=0)
    return out, wid if isinstance(wid, str) else ""


def check_domain(eng: Any, g: Geometry, tr: Transducer, tx: Transmit, label: str) -> None:
    expected, wid = matlab_domain(eng, g, tr, tx)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        actual = ptr.compute_simulation_domain(g, tr, tx)
    py_warned = any(issubclass(w.category, ptr.GeometryBehindTransducerWarning) for w in caught)
    assert py_warned == (wid == "compute_simulation_domain:geometryBehindTransducer"), label
    assert ",".join(actual.Domain.present_fields()) == expected["_domain_fields"], label
    ed = expected["Domain"]
    finite = np.concatenate(
        [np.asarray(ed[k], dtype=float).ravel() for k in DOMAIN_FIELDS]
        + [np.asarray(expected["Center"], dtype=float).ravel()]
    )
    finite = finite[np.isfinite(finite)]
    atol = ATOL_FRACTION * np.max(np.abs(finite))
    for k in DOMAIN_FIELDS:
        a = np.asarray(getattr(actual.Domain, k), dtype=float)
        e = ed[k]
        assert_float(a.reshape(e.shape) if a.size == e.size == 1 else a, e, atol, f"{k} {label}")
    assert_float(actual.Center, expected["Center"], atol, f"Center {label}")
    assert actual.Domain.Manual == g.Domain.Manual


TRANSMITS = [
    Transmit(LateralFocus=math.inf, Angle=0.0),
    Transmit(LateralFocus=math.inf, Angle=-12.5),
    Transmit(LateralFocus=40e-3, Angle=0.0),
    Transmit(LateralFocus=40e-3, Angle=10.0),
    Transmit(LateralFocus=-15e-3, Angle=20.0),
    Transmit(LateralFocus=math.nan, Angle=5.0),
    # axial focus coordinate 0 (signed zero decides ±Inf/NaN lines):
    Transmit(LateralFocus=30e-3, Angle=90.0),
    Transmit(LateralFocus=30e-3, Angle=-90.0),
    Transmit(LateralFocus=-30e-3, Angle=90.0),
    Transmit(LateralFocus=30e-3, Angle=270.0),
]


@pytest.mark.parametrize("name", list(TRANSDUCERS))
@pytest.mark.parametrize("folder", ["renal_tree", "mouse_brain"])
def test_domain_auto(matlab_engine: Any, name: str, folder: str) -> None:
    tr = TRANSDUCERS[name]
    margin = 2 * 1540 / (tr.CenterFrequency or 5e6)
    for start in (25e-3, 5e-3, 1e-3):  # 1-5 mm: behind the apex of strongly curved arrays
        g = geometry(folder, start, False, margin)
        for i, tx in enumerate(TRANSMITS):
            check_domain(matlab_engine, g, tr, tx, f"{name} {folder} start={start} tx={i}")


@pytest.mark.parametrize("name", ["P4-1", "C1-6D", "RC6gV", "linear-7"])
def test_domain_manual(matlab_engine: Any, name: str) -> None:
    tr = TRANSDUCERS[name]
    boxes = [
        (0.0, 60e-3, -20e-3, 20e-3, -5e-3, 5e-3),  # larger than the transducer
        (5e-3, 6e-3, -1e-4, 1e-4, 0.0, 0.0),  # smaller: grown to hold it
    ]
    for box in boxes:
        for tx in TRANSMITS[:5]:
            g = geometry("renal_tree", 25e-3, True, 1e-3, box)
            check_domain(matlab_engine, g, tr, tx, f"{name} manual {box}")


def test_domain_custom_configuration(matlab_engine: Any) -> None:
    rng = np.random.default_rng(3)
    tr = transducer("P4-1", Configuration="Custom")
    pts = rng.uniform(-5e-3, 5e-3, size=(17, 3))
    # N×3, 3×N and a 3N row: MATLAB reshape(points, [], 3) is column-major
    for layout in (pts, pts.T.copy(), pts.reshape(1, -1)):
        tr.integration_points = layout  # extra field
        for manual in (False, True):
            g = geometry("renal_tree", 25e-3, manual, 1e-3, (0.0, 4e-2, -1e-2, 1e-2, -1e-2, 1e-2))
            check_domain(matlab_engine, g, tr, TRANSMITS[2], f"custom {layout.shape} {manual}")


def test_domain_default_settings(matlab_engine: Any, oracle: Any) -> None:
    eng = matlab_engine
    paths = f"struct('GeometriesPath', '{oracle.path / 'geometry_data'}')"
    eng.eval(
        f"pyp_T__ = reset_transducer(); pyp_G__ = reset_geometry(struct(), {paths});"
        " pyp_G__.Domain.Margin = 2*1540/pyp_T__.CenterFrequency;"
        " pyp_G__ = compute_simulation_domain(pyp_G__, pyp_T__, reset_transmit(pyp_T__));"
        " pyp_V__ = pyp_G__.Domain.Vertices; pyp_C__ = pyp_G__.Center;",
        nargout=0,
    )
    v, c = from_matlab(eng.workspace["pyp_V__"]), from_matlab(eng.workspace["pyp_C__"])
    eng.eval("clear pyp_T__ pyp_G__ pyp_V__ pyp_C__", nargout=0)
    t = config.reset_transducer()
    g = config.reset_geometry()
    g.Domain.Margin = 2 * 1540 / t.CenterFrequency
    out = ptr.compute_simulation_domain(g, t, config.reset_transmit(t))
    atol = ATOL_FRACTION * np.max(np.abs(v))
    assert_float(out.Domain.Vertices, v, atol, "Vertices")
    assert_float(out.Center, c, atol, "Center")


def test_domain_odd_rca(matlab_engine: Any) -> None:
    eng = matlab_engine
    tr = transducer("RC6gV", NumberOfElements=255)
    g = geometry("renal_tree", 25e-3, False, 1e-3)
    eng.workspace["pyp_g__"], eng.workspace["pyp_tr__"] = tree(g), tree(tr)
    eng.workspace["pyp_tx__"] = tree(TRANSMITS[0])
    eng.eval(
        "try, compute_simulation_domain(pyp_g__, pyp_tr__, pyp_tx__); pyp_r__ = 'none';"
        " catch e, pyp_r__ = e.message; end",
        nargout=0,
    )
    assert "must be even" in eng.workspace["pyp_r__"]
    eng.eval("clear pyp_g__ pyp_tr__ pyp_tx__ pyp_r__", nargout=0)
    with pytest.raises(ValueError, match="must be even"):
        ptr.compute_simulation_domain(g, tr, TRANSMITS[0])
