"""MATLAB oracle fixtures (Phase A parity tests, ROADMAP §4).

Every test under ``tests/oracle/`` is marked ``oracle``. The session-scoped
:func:`matlab_engine` fixture starts one MATLAB Engine for the whole run, puts
the oracle PROTEUS checkout and ``tests/oracle/shims/`` on the MATLAB path and
skips cleanly when ``matlab.engine`` or MATLAB is unavailable (CI, contributors).

Environment variables:

``PYPROTEUS_ORACLE``
    Path to the read-only MATLAB PROTEUS checkout (default: the developer's
    ``PROTEUS-cleaning/PROTEUS-oracle``). The fixture never writes there.
``PYPROTEUS_ORACLE_COMMIT``
    Expected oracle commit prefix (default ``8d37938``, D-7). A mismatch is an
    error: parity claims are tied to one MATLAB commit. Set to ``any`` to accept
    whatever is checked out (exploration only).
``PYPROTEUS_MATLAB_SESSION``
    Name of a shared MATLAB session (``matlab.engine.shareEngine('name')``)
    to connect to instead of starting a new engine; faster while iterating.
"""

from __future__ import annotations

import os
import ssl  # noqa: F401  # B-1: must be imported before matlab.engine
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

DEFAULT_ORACLE = Path(
    "/Users/theodoremichel/Repos/TWENTE/Ultrasound/PROTEUS-cleaning/PROTEUS-oracle"
)
DEFAULT_COMMIT = "8d37938"
SHIMS = Path(__file__).resolve().parent / "shims"

# Oracle folders put on the MATLAB path (recursively; genpath skips private/,
# @class and +package folders). The repository's own tests/ and example
# scripts are left out so their helpers cannot shadow PROTEUS functions.
ORACLE_MODULES = (
    "GUIfunctions",
    "acoustic-module",
    "microbubble-simulator",
    "streamline-module",
    "delay-and-sum",
    "mesh-voxelisation",
    "time-estimation",
)


@dataclass
class Oracle:
    """What the oracle tests ran against; recorded in the pytest header."""

    path: Path
    commit: str
    matlab_version: str


def oracle_path() -> Path:
    return Path(os.environ.get("PYPROTEUS_ORACLE", DEFAULT_ORACLE)).expanduser()


def oracle_commit(path: Path) -> str:
    """Full commit hash of the oracle checkout (read-only git call)."""
    out = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout.strip()


def oracle_status(path: Path) -> str:
    """``git status --porcelain`` of the oracle (no lock files written)."""
    out = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(path), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    return out.stdout


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    here = Path(__file__).resolve().parent
    for item in items:
        if here in Path(str(item.path)).resolve().parents:
            item.add_marker(pytest.mark.oracle)


def _import_engine() -> Any:
    try:
        import matlab.engine
    except (ImportError, OSError) as exc:  # not installed, or MATLAB/Python unsupported
        pytest.skip(f"matlab.engine not importable ({exc}); oracle tests need MATLAB")
    return matlab.engine


@pytest.fixture(scope="session")
def oracle() -> Oracle:
    """Oracle checkout and commit, without starting MATLAB."""
    path = oracle_path()
    if not path.is_dir():
        pytest.skip(f"oracle PROTEUS checkout not found at {path} (set PYPROTEUS_ORACLE)")
    commit = oracle_commit(path)
    expected = os.environ.get("PYPROTEUS_ORACLE_COMMIT", DEFAULT_COMMIT).strip()
    if not expected:
        pytest.fail("PYPROTEUS_ORACLE_COMMIT is empty; use a commit prefix or 'any'")
    dirty = oracle_status(path)
    if dirty and expected != "any":
        pytest.fail(f"oracle checkout at {path} has local changes:\n{dirty}")
    if expected != "any" and not commit.startswith(expected):
        pytest.fail(
            f"oracle at {path} is at commit {commit[:12]}, expected {expected} (D-7). "
            "Check out the pinned commit or set PYPROTEUS_ORACLE_COMMIT."
        )
    return Oracle(path=path, commit=commit, matlab_version="")


@pytest.fixture(scope="session")
def matlab_engine(oracle: Oracle, tmp_path_factory: pytest.TempPathFactory) -> Any:
    """One MATLAB Engine per test session, with the oracle PROTEUS on the path.

    MATLAB's working folder is a fresh temporary directory, so relative writes
    by PROTEUS functions land there and no stray ``.m`` file shadows PROTEUS.
    At the end of the session the oracle checkout must be unchanged.

    Tests must not leave state behind that other tests depend on: set ``rng``
    explicitly and clear workspace variables they create if it matters.
    Under pytest-xdist every worker starts its own MATLAB.
    """
    engine_mod = _import_engine()
    session = os.environ.get("PYPROTEUS_MATLAB_SESSION")
    try:
        if session:
            eng = engine_mod.connect_matlab(session)
        else:
            eng = engine_mod.start_matlab("-nodesktop -nosplash")
    except engine_mod.EngineError as exc:
        pytest.skip(f"could not start MATLAB ({exc})")

    status_before = oracle_status(oracle.path)
    original_path = eng.path(nargout=1)
    original_cwd = eng.pwd(nargout=1)
    eng.cd(str(tmp_path_factory.mktemp("matlab_cwd")), nargout=0)
    eng.addpath(str(oracle.path), nargout=0)  # root functions, not recursive
    for module in ORACLE_MODULES:
        folder = oracle.path / module
        if not folder.is_dir():
            pytest.fail(f"oracle module folder missing: {folder}")
        eng.addpath(eng.genpath(str(folder)), nargout=0)
    if SHIMS.is_dir():
        eng.addpath(str(SHIMS), nargout=0)  # added last = searched first
    eng.eval("format long g;", nargout=0)
    version = eng.version(nargout=1)
    oracle.matlab_version = version

    yield eng

    if session:
        eng.path(original_path, nargout=0)
        eng.cd(original_cwd, nargout=0)
    else:
        eng.quit()
    status_after = oracle_status(oracle.path)
    assert status_after == status_before, (
        f"oracle checkout {oracle.path} was modified during the test session:\n{status_after}"
    )


@pytest.fixture
def mrng(matlab_engine: Any):
    """Reset MATLAB's global stream to rng(0,'twister') before and after a test."""
    matlab_engine.eval("rng(0,'twister');", nargout=0)
    yield matlab_engine
    matlab_engine.eval("rng(0,'twister');", nargout=0)
