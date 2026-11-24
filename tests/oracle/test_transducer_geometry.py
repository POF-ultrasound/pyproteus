"""Oracle tests: ``get_element_geometry`` (linear and curved arrays) against MATLAB.

Class D-13 float: ``|a - e| <= 1e-9*|e| + atol``. Positions (``x``, ``y``,
``Apex``, ``Center``) are differences of quantities of the size of the array
(``R*cos`` terms), so their absolute floor is relative to the array's length
scale ``L = max(N*Pitch, R)``: ``atol = 1e-14*L``
(measured: <= 2.2e-16*L). Angles use ``1e-14*max|e|``.
±Inf (linear ``Radius``/``Center``) must match exactly.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from pyproteus import config
from pyproteus import transducer as ptr
from pyproteus.config.models import Transducer
from tests.oracle.helpers import ORACLE_SETTINGS, from_matlab

REL = 1e-9  # D-13 float class
ATOL_FRACTION = 1e-14  # measured error <= 2.2e-16*L

FIELDS = ("theta", "x", "y", "width", "Radius", "Apex", "Center", "ThetaMax")
LENGTHS = {"x", "y", "width", "Radius", "Apex", "Center"}


def assert_float(actual: Any, expected: Any, atol: float, err_msg: str) -> None:
    a, e = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    assert a.shape == e.shape, f"shape {a.shape} != {e.shape} {err_msg}"
    inf = np.isinf(e)
    assert np.array_equal(np.isinf(a), inf), f"Inf positions differ {err_msg}"
    np.testing.assert_array_equal(a[inf], e[inf], err_msg=err_msg)
    assert not np.isnan(e).any() and not np.isnan(a).any(), err_msg
    fin = ~inf
    err = np.abs(a[fin] - e[fin])
    bound = REL * np.abs(e[fin]) + atol
    assert np.all(err <= bound), f"max excess {np.max(err - bound):.3g} {err_msg}"


def matlab_geometry(eng: Any, structs: list[dict[str, Any]]) -> list[dict[str, np.ndarray]]:
    """``get_element_geometry`` on each struct, in one engine call."""
    eng.workspace["pyp_T__"] = structs
    eng.eval(
        "pyp_El__ = cellfun(@get_element_geometry, pyp_T__, 'UniformOutput', false);", nargout=0
    )
    out = eng.workspace["pyp_El__"]
    eng.eval("clear pyp_T__ pyp_El__", nargout=0)
    return [{k: from_matlab(v) for k, v in el.items()} for el in out]


def compare(py: ptr.ElementGeometry, ml: dict[str, np.ndarray], label: str) -> None:
    assert set(ml) == set(FIELDS), f"MATLAB fields {sorted(ml)} {label}"
    s = py.to_struct()
    n = py.n_elements
    radius = float(ml["Radius"].reshape(()))
    scale = max(n * abs(float(s["width"])), radius if math.isfinite(radius) else 0.0, 1e-300)
    for k in FIELDS:
        e = ml[k]
        a = np.asarray(s[k], dtype=float)
        if e.shape == (1, 1) and a.ndim == 0:
            a = a.reshape(1, 1)
        if k in ("theta", "x", "y") and n == 0:
            e = e.reshape(0, 1)  # the engine returns MATLAB's 0x1 as an empty array
        fin = e[np.isfinite(e)]
        if k in LENGTHS:
            atol = ATOL_FRACTION * scale
        else:
            atol = ATOL_FRACTION * (np.max(np.abs(fin)) if fin.size else 0.0)
        assert_float(a, e, atol, f"{k} ({label})")


def struct(**fields: Any) -> dict[str, Any]:
    return {k: (float(v) if isinstance(v, int) else v) for k, v in fields.items()}


def check(eng: Any, cases: list[dict[str, Any]]) -> None:
    for case, ml in zip(cases, matlab_geometry(eng, cases), strict=True):
        compare(ptr.get_element_geometry(Transducer(**case)), ml, repr(case))


# --------------------------------------------------------------- fixed cases

LINEAR = [
    struct(Configuration="Linear", NumberOfElements=n, Pitch=p, ElementWidth=0.8 * p)
    for n in (1, 2, 3, 7, 96, 127, 128, 255)
    for p in (0.3e-3, 1e-3 / 3, 0.1e-3)
]

CURVED = [
    struct(Configuration="Curved", NumberOfElements=n, Pitch=p, ElementWidth=0.9 * p, Radius=r)
    for n in (1, 2, 3, 7, 64, 191, 192)
    for p in (0.35e-3, 1e-3 / 3)
    for r in (56.8e-3, 0.2, 1.0, 10.0, 0.75 * n * p / math.pi * 2)
    # ^ last radius: theta_max = 2/3*pi/2 (strongly curved)
]

NEAR_LIMIT = [
    # theta_max just below pi/2: half-circle arrays
    struct(Configuration="Curved", NumberOfElements=n, Pitch=p, ElementWidth=p, Radius=r)
    for n, p in ((3, 1e-3), (128, 0.2e-3), (255, 0.1e-3))
    for r in ((1 + 1e-9) * n * p / math.pi, (1 + 1e-6) * n * p / math.pi)
]

OTHER = [
    # any Configuration other than exactly 'Curved' is linear (case-sensitive strcmp)
    struct(Configuration="curved", NumberOfElements=5, Pitch=2e-4, ElementWidth=1e-4, Radius=0.05),
    struct(Configuration="RCA", NumberOfElements=256, Pitch=0.2e-3, ElementWidth=0.175e-3),
    struct(NumberOfElements=9, Pitch=0.3e-3, ElementWidth=0.25e-3),
    struct(Configuration="Linear", NumberOfElements=4, Pitch=3e-4, ElementWidth=2e-4, Radius=-1),
    # empty arrays
    struct(Configuration="Linear", NumberOfElements=0, Pitch=3e-4, ElementWidth=2e-4),
    struct(Configuration="Curved", NumberOfElements=0, Pitch=3e-4, Radius=0.05),
    # negative pitch
    struct(Configuration="Linear", NumberOfElements=5, Pitch=-3e-4, ElementWidth=2e-4),
    struct(Configuration="Curved", NumberOfElements=5, Pitch=-3e-4, Radius=0.05),
]


@pytest.mark.parametrize(
    "cases", [LINEAR, CURVED, NEAR_LIMIT, OTHER], ids=["linear", "curved", "near-limit", "other"]
)
def test_fixed_cases(matlab_engine: Any, cases: list[dict[str, Any]]) -> None:
    check(matlab_engine, cases)


@pytest.mark.parametrize("ttype", ["P4-1", "C1-6D", "L22-14v", "9L-D", "RC6gV"])
def test_catalogue_transducers(matlab_engine: Any, ttype: str) -> None:
    """Shipped catalogue models (C1-6D: curved, with its calibrated model file)."""
    eng = matlab_engine
    eng.eval(
        f"pyp_El__ = get_element_geometry(get_transducer_properties(struct('Type','{ttype}')));",
        nargout=0,
    )
    ml = {k: from_matlab(v) for k, v in eng.workspace["pyp_El__"].items()}
    eng.eval("clear pyp_El__", nargout=0)
    T = ptr.get_transducer_properties(Transducer(Type=ttype))
    compare(ptr.get_element_geometry(T), ml, ttype)


def test_default_settings(matlab_engine: Any) -> None:
    eng = matlab_engine
    eng.eval("pyp_El__ = get_element_geometry(reset_transducer());", nargout=0)
    ml = {k: from_matlab(v) for k, v in eng.workspace["pyp_El__"].items()}
    eng.eval("clear pyp_El__", nargout=0)
    compare(ptr.get_element_geometry(config.reset_transducer()), ml, "reset_transducer")


# ------------------------------------------------------------------ errors

ERRORS = {
    "radius-zero": struct(Configuration="Curved", NumberOfElements=8, Pitch=3e-4, Radius=0),
    "radius-negative": struct(Configuration="Curved", NumberOfElements=8, Pitch=3e-4, Radius=-0.05),
    "radius-inf": struct(Configuration="Curved", NumberOfElements=8, Pitch=3e-4, Radius=math.inf),
    "radius-nan": struct(Configuration="Curved", NumberOfElements=8, Pitch=3e-4, Radius=math.nan),
    # theta_max = 2*(pi/2)/(2*1) = pi/2 exactly: not shorter than pi*R
    "arc-half-circle": struct(
        Configuration="Curved", NumberOfElements=2, Pitch=math.pi / 2, Radius=1
    ),
    "arc-too-long": struct(Configuration="Curved", NumberOfElements=192, Pitch=1e-3, Radius=0.05),
}


@pytest.mark.parametrize("name", list(ERRORS))
def test_errors(matlab_engine: Any, name: str) -> None:
    eng = matlab_engine
    eng.workspace["pyp_T__"] = ERRORS[name]
    eng.eval(
        "try, get_element_geometry(pyp_T__); pyp_id__ = 'none';"
        " catch e, pyp_id__ = e.identifier; end",
        nargout=0,
    )
    ident = eng.workspace["pyp_id__"]
    eng.eval("clear pyp_T__ pyp_id__", nargout=0)
    assert ident == "get_element_geometry:invalidRadius"
    with pytest.raises(ValueError, match=r"Radius|arc"):
        ptr.get_element_geometry(Transducer(**ERRORS[name]))


def test_empty_radius_and_missing_fields(matlab_engine: Any) -> None:
    eng = matlab_engine
    eng.eval(
        "pyp_ids__ = {};"
        "try, get_element_geometry(struct('Configuration','Curved','NumberOfElements',4,"
        "'Pitch',1e-3,'Radius',[])); pyp_ids__{end+1} = 'none'; catch e,"
        " pyp_ids__{end+1} = e.identifier; end;"
        "try, get_element_geometry(struct('Configuration','Curved','NumberOfElements',4,"
        "'Pitch',1e-3)); pyp_ids__{end+1} = 'none'; catch e, pyp_ids__{end+1} = 'error'; end;"
        "try, get_element_geometry(struct('Configuration','Linear','NumberOfElements',4,"
        "'Pitch',1e-3)); pyp_ids__{end+1} = 'none'; catch e, pyp_ids__{end+1} = 'error'; end;",
        nargout=0,
    )
    ids = list(eng.workspace["pyp_ids__"])
    eng.eval("clear pyp_ids__", nargout=0)
    assert ids == ["get_element_geometry:invalidRadius", "error", "error"]
    with pytest.raises(ValueError, match="Radius"):
        ptr.get_element_geometry(
            {
                "Configuration": "Curved",
                "NumberOfElements": 4.0,
                "Pitch": 1e-3,
                "Radius": np.zeros((0, 0)),
            }
        )
    with pytest.raises(ValueError, match="no field Radius"):
        ptr.get_element_geometry(Transducer(Configuration="Curved", NumberOfElements=4, Pitch=1e-3))
    with pytest.raises(ValueError, match="no field ElementWidth"):
        ptr.get_element_geometry(Transducer(Configuration="Linear", NumberOfElements=4, Pitch=1e-3))


def test_non_integer_element_count(matlab_engine: Any) -> None:
    """MATLAB: the linear branch errors (``zeros(2.5,1)``); the curved branch silently
    uses ``1:floor(N)`` but ``(N+1)/2`` with the fractional N. pyproteus raises for
    both (docs/PARITY.md)."""
    eng = matlab_engine
    eng.eval(
        "try, get_element_geometry(struct('Configuration','Linear','NumberOfElements',2.5,"
        "'Pitch',1e-3,'ElementWidth',1e-3)); pyp_r__ = 'none'; catch e, pyp_r__ = 'error'; end",
        nargout=0,
    )
    assert eng.workspace["pyp_r__"] == "error"
    eng.eval("clear pyp_r__", nargout=0)
    for conf in ("Linear", "Curved"):
        with pytest.raises(ValueError, match="non-negative integer"):
            ptr.get_element_geometry(
                Transducer(
                    Configuration=conf,
                    NumberOfElements=2.5,
                    Pitch=1e-3,
                    ElementWidth=1e-3,
                    Radius=0.05,
                )
            )


# --------------------------------------------------------------- random cases


@st.composite
def transducers(draw: st.DrawFn) -> dict[str, Any]:
    n = draw(st.integers(1, 300))
    p = draw(st.floats(5e-6, 2e-3))
    if draw(st.booleans()):
        return struct(
            Configuration="Linear",
            NumberOfElements=n,
            Pitch=p,
            ElementWidth=draw(st.floats(1e-6, 2e-3)),
        )
    # theta_max = n*p/(2R) in (1e-5, 0.999*pi/2): apex >= 5e-11*R, above the atol
    theta_max = draw(st.floats(1e-5, 0.999 * math.pi / 2))
    return struct(
        Configuration="Curved", NumberOfElements=n, Pitch=p, Radius=n * p / (2 * theta_max)
    )


@ORACLE_SETTINGS
@given(st.lists(transducers(), min_size=1, max_size=20))
def test_random(matlab_engine: Any, cases: list[dict[str, Any]]) -> None:
    check(matlab_engine, cases)
