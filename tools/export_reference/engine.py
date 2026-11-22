"""MATLAB Engine session for the reference export, and provenance gathering.

The path setup mirrors the oracle test fixture (``tests/oracle/conftest.py``,
whose constants and git helpers are reused): oracle root + module folders on
the path, MATLAB working folder in a scratch directory, oracle checkout
checked clean before and after. In addition, the export can put k-Wave on
the path and point PROTEUS' ``path_setup()`` at scratch folders through the
``PROTEUS_LOCAL_PATHS`` file mechanism, so nothing is ever written into the
oracle checkout.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import os
import platform
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from tests.oracle.conftest import (
    DEFAULT_COMMIT,
    DEFAULT_ORACLE,
    ORACLE_MODULES,
    oracle_commit,
    oracle_status,
)
from tests.oracle.helpers import to_matlab

from .records import records_to_value

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MATLAB_DIR = HERE / "matlab"
TRACER_BIN = REPO / "tools" / "tracer_bullet" / "bin"
DEFAULT_KWAVE_MATLAB = Path(
    "/Users/theodoremichel/Repos/TWENTE/Ultrasound/PROTEUS-cleaning/PROTEUS/external/k-wave/k-Wave"
)
KWAVE_SOLVER = "kspaceFirstOrder3DC"
KWAVE_BINARY_NAME = "kspaceFirstOrder-OMP"  # name kspaceFirstOrder3DC runs on Linux/macOS


@dataclass
class ExportConfig:
    """Where things are. Every field has an environment variable (see ``from_env``)."""

    oracle: Path = DEFAULT_ORACLE
    oracle_commit: str = DEFAULT_COMMIT
    #: MATLAB k-Wave 1.4.x folder (holds kWaveGrid.m).
    kwave_matlab_path: Path = DEFAULT_KWAVE_MATLAB
    #: Folder passed to k-Wave as ``BinaryPath`` (PROTEUS ``PATHS.BinaryPath``);
    #: holds ``kspaceFirstOrder-OMP`` or a wrapper with that name (D-10: the
    #: tracer-bullet wrapper around k-wave-python's v1.4.1 binary on macOS arm64).
    kwave_binary_dir: Path = TRACER_BIN
    #: Executable actually run, when ``kwave_binary_dir`` holds a wrapper
    #: (recorded and hashed; exported to MATLAB as ``KWAVE_BINARY``).
    kwave_binary: Path | None = None
    #: Runtime library folder for the tracer wrapper (``KWAVE_LIBS``).
    kwave_libs: Path | None = None
    #: Scratch root (MATLAB working folder, PROTEUS data folders). Temporary if None.
    workdir: Path | None = None
    #: Shared MATLAB session name to connect to instead of starting MATLAB.
    matlab_session: str | None = None
    extra_env: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> ExportConfig:
        e = os.environ.get
        cfg = cls()
        cfg.oracle = Path(e("PYPROTEUS_ORACLE", str(cfg.oracle))).expanduser()
        cfg.oracle_commit = e("PYPROTEUS_ORACLE_COMMIT", cfg.oracle_commit).strip()
        cfg.kwave_matlab_path = Path(e("KWAVE_MATLAB_PATH", str(cfg.kwave_matlab_path)))
        cfg.kwave_binary_dir = Path(e("PYPROTEUS_KWAVE_BINARY_DIR", str(cfg.kwave_binary_dir)))
        if e("PYPROTEUS_KWAVE_BINARY"):
            cfg.kwave_binary = Path(e("PYPROTEUS_KWAVE_BINARY"))
        if e("PYPROTEUS_KWAVE_LIBS"):
            cfg.kwave_libs = Path(e("PYPROTEUS_KWAVE_LIBS"))
        if e("PYPROTEUS_EXPORT_WORKDIR"):
            cfg.workdir = Path(e("PYPROTEUS_EXPORT_WORKDIR"))
        cfg.matlab_session = e("PYPROTEUS_MATLAB_SESSION") or None
        return cfg


def _preload() -> None:
    import ssl  # noqa: F401  # B-1
    import xml.parsers.expat  # noqa: F401  # B-11


def kwave_matlab_version(path: Path) -> str:
    """Version from k-Wave's ``Contents.m`` (same rule as PROTEUS ``path_setup``)."""
    contents = path / "Contents.m"
    if not contents.is_file():
        raise FileNotFoundError(f"no k-Wave Contents.m in {path}")
    m = re.search(r"%\s*Version\s+(\d+\.\d+(?:\.\d+)?)", contents.read_text(errors="replace"))
    if not m:
        raise ValueError(f"cannot read the k-Wave version from {contents}")
    return m.group(1)


def binary_version_text(stdout: str) -> str:
    """Compact version string from ``kspaceFirstOrder-* --version`` (box-drawing table)."""
    fields = []
    for raw in stdout.splitlines():
        ln = raw.strip().strip("│|").strip()
        if not ln or set(ln) <= set("┌┐└┘├┤─-+"):
            continue
        if ln.startswith("kspaceFirstOrder"):
            fields.append(ln)
        elif ln.split(":")[0].strip() in ("Build number", "Build date", "Build time"):
            fields.append(" ".join(ln.split()))
    if not fields:
        raise ValueError(f"cannot parse the k-Wave binary version from:\n{stdout}")
    return "; ".join(fields)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def repo_commit() -> str:
    """pyproteus commit of the exporter, ``-dirty`` if tracked files changed."""
    try:
        sha = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return sha + ("-dirty" if dirty else "")


def _mstr(x: object) -> str:
    """MATLAB char literal content (single quotes doubled)."""
    return str(x).replace("'", "''")


class ExportSession:
    """One MATLAB Engine with the oracle on the path (``with ExportSession(cfg) as s:``)."""

    def __init__(self, cfg: ExportConfig, *, kwave: bool = False, engine: Any = None) -> None:
        """``engine``: reuse a running MATLAB Engine (path and folder restored on exit)."""
        self.cfg = cfg
        self.kwave = kwave
        self._given = engine
        self._restore: tuple[Any, Any, str] | None = None
        self.eng: Any = None
        self._tmp: tempfile.TemporaryDirectory | None = None
        self.workdir: Path | None = None
        self.paths_extra: list[str] = []
        self.oracle_commit = ""
        self._status_before = ""
        self._kwave_info: dict[str, str] = {}

    # -- lifecycle -------------------------------------------------------------------------

    def __enter__(self) -> ExportSession:
        cfg = self.cfg
        if not cfg.oracle.is_dir():
            raise FileNotFoundError(f"oracle checkout not found: {cfg.oracle}")
        self.oracle_commit = oracle_commit(cfg.oracle)
        self._status_before = oracle_status(cfg.oracle)
        if cfg.oracle_commit != "any":
            if self._status_before:
                raise RuntimeError(f"oracle {cfg.oracle} has local changes")
            if not self.oracle_commit.startswith(cfg.oracle_commit):
                raise RuntimeError(
                    f"oracle at {self.oracle_commit[:12]}, expected {cfg.oracle_commit} (D-7)"
                )
        if self.kwave:
            self._kwave_info = self._check_kwave()
        if cfg.workdir is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="pyp-export-")
            self.workdir = Path(self._tmp.name)
        else:
            self.workdir = Path(cfg.workdir).resolve()
            self.workdir.mkdir(parents=True, exist_ok=True)
        if self.workdir.resolve().is_relative_to(cfg.oracle.resolve()):
            raise ValueError("the export work folder must be outside the oracle checkout")

        _preload()
        import matlab.engine

        if self._given is not None:
            self.eng = self._given
        elif cfg.matlab_session:
            self.eng = matlab.engine.connect_matlab(cfg.matlab_session)
        else:
            self.eng = matlab.engine.start_matlab("-nodesktop -nosplash")
        if self._given is not None or cfg.matlab_session:
            self._restore = (
                self.eng.path(nargout=1),
                self.eng.pwd(nargout=1),
                self.eng.getenv("PROTEUS_LOCAL_PATHS", nargout=1),
            )
        self._setup_path()
        return self

    def __exit__(self, *exc: object) -> None:
        try:
            if self.eng is not None:
                if self._restore is not None:
                    self.eng.eval("clear pyp_*;", nargout=0)
                    path, cwd, local_paths = self._restore
                    self.eng.cd(cwd, nargout=0)
                    self.eng.path(path, nargout=0)
                    self.eng.setenv("PROTEUS_LOCAL_PATHS", local_paths, nargout=0)
                else:
                    self.eng.quit()
        finally:
            after = oracle_status(self.cfg.oracle)
            if self._tmp is not None:
                self._tmp.cleanup()
            if after != self._status_before:
                raise RuntimeError(f"oracle checkout modified during the export:\n{after}")

    # -- setup -----------------------------------------------------------------------------

    def _check_kwave(self) -> dict[str, str]:
        cfg = self.cfg
        version = kwave_matlab_version(cfg.kwave_matlab_path)
        wrapper = cfg.kwave_binary_dir / KWAVE_BINARY_NAME
        if not wrapper.is_file():
            raise FileNotFoundError(
                f"no {KWAVE_BINARY_NAME} in {cfg.kwave_binary_dir} (set PYPROTEUS_KWAVE_BINARY_DIR)"
            )
        binary = cfg.kwave_binary
        if binary is None and wrapper.resolve() == (TRACER_BIN / KWAVE_BINARY_NAME).resolve():
            import kwave  # the tracer wrapper runs k-wave-python's binary (D-10)

            binary = Path(kwave.BINARY_PATH) / KWAVE_BINARY_NAME
        binary = Path(binary) if binary is not None else wrapper
        env = self._binary_env(binary, wrapper)
        out = subprocess.run(
            [str(wrapper), "--version"],
            env={**os.environ, **env},
            capture_output=True,
            text=True,
            timeout=120,
        )
        if out.returncode != 0 or "kspaceFirstOrder" not in out.stdout:
            raise RuntimeError(
                f"k-Wave binary does not start ({wrapper}): {out.stdout}{out.stderr}\n"
                "On macOS run tools/tracer_bullet/setup_libs.sh or set PYPROTEUS_KWAVE_LIBS."
            )
        info = {
            "kwave_matlab_version": version,
            "kwave_matlab_path": str(cfg.kwave_matlab_path),
            "kwave_solver": KWAVE_SOLVER,
            "kwave_binary_path": str(binary),
            "kwave_binary_wrapper": str(wrapper) if binary != wrapper else "",
            "kwave_binary_version": binary_version_text(out.stdout),
            "kwave_binary_sha256": sha256_file(binary),
        }
        try:  # D-10: binary shipped by k-wave-python; record which release
            import kwave

            if Path(binary).resolve().is_relative_to(Path(kwave.BINARY_PATH).resolve()):
                info["kwave_python_version"] = str(kwave.__version__)
                info["kwave_python_binary_release"] = str(kwave.BINARY_VERSION)
        except ImportError:  # pragma: no cover
            pass
        return info

    def _binary_env(self, binary: Path, wrapper: Path) -> dict[str, str]:
        env = dict(self.cfg.extra_env)
        if binary != wrapper:
            env["KWAVE_BINARY"] = str(binary)
        if self.cfg.kwave_libs is not None:
            env["KWAVE_LIBS"] = str(self.cfg.kwave_libs)
        return env

    def _setup_path(self) -> None:
        eng, cfg = self.eng, self.cfg
        assert self.workdir is not None
        cwd = self.workdir / "matlab_cwd"
        cwd.mkdir(exist_ok=True)
        eng.cd(str(cwd), nargout=0)
        eng.addpath(str(cfg.oracle), nargout=0)
        for module in ORACLE_MODULES:
            folder = cfg.oracle / module
            if not folder.is_dir():
                raise FileNotFoundError(f"oracle module folder missing: {folder}")
            eng.addpath(eng.genpath(str(folder)), nargout=0)
        eng.addpath(str(MATLAB_DIR), nargout=0)
        self.paths_extra = [str(MATLAB_DIR.relative_to(REPO))]
        eng.eval("format long g;", nargout=0)
        if self.kwave:
            self._setup_kwave()

    def _setup_kwave(self) -> None:
        """k-Wave + oracle test helpers on the path, PROTEUS data folders in the work folder."""
        eng, cfg = self.eng, self.cfg
        assert self.workdir is not None
        info = self._kwave_info
        binary, wrapper = Path(info["kwave_binary_path"]), cfg.kwave_binary_dir / KWAVE_BINARY_NAME
        for k, v in self._binary_env(binary, wrapper).items():
            eng.setenv(k, v, nargout=0)
        eng.addpath(str(cfg.kwave_matlab_path), nargout=0)
        # PROTEUS' own headless helpers (make_tiny_settings, write_settings, ...).
        # Only tests/helpers: tests/shims would replace the C++ solver by MATLAB's.
        eng.addpath(str(cfg.oracle / "tests" / "helpers"), nargout=0)
        self.paths_extra += ["<kwave_matlab_path>", "<oracle>/tests/helpers"]
        folders = {
            "GeometriesPath": "geometry_data",
            "GroundTruthPath": "ground_truth_frames",
            "ResultsPath": "RESULTS",
            "DataPath": "RESULTS",
            "SettingsPath": "simulation-settings",
        }
        lines = [
            f"PATHS_LOCAL.kWavePath = '{_mstr(cfg.kwave_matlab_path)}';",
            f"PATHS_LOCAL.BinaryPath = '{_mstr(str(cfg.kwave_binary_dir).rstrip('/'))}/';",
        ]
        for key, sub in folders.items():
            (self.workdir / sub).mkdir(exist_ok=True)
            lines.append(f"PATHS_LOCAL.{key} = '{_mstr(self.workdir / sub)}';")
        paths_file = self.workdir / "proteus_local_paths.m"
        paths_file.write_text("\n".join(lines) + "\n")
        eng.setenv("PROTEUS_LOCAL_PATHS", str(paths_file), nargout=0)
        # Shipped flow geometries are tiny folders: link them next to the synthetic ones.
        for g in (cfg.oracle / "geometry_data").iterdir():
            dest = self.workdir / "geometry_data" / g.name
            if g.is_dir() and not dest.exists():
                dest.symlink_to(g, target_is_directory=True)
        eng.pyp_write_tiny_geometry(str(self.workdir / "geometry_data"), nargout=0)
        eng.eval("PATHS = path_setup();", nargout=0)  # fails early on bad paths
        eng.eval("clear PATHS;", nargout=0)

    # -- use -------------------------------------------------------------------------------

    def put(self, name: str, value: Any) -> None:
        self.eng.workspace[name] = to_matlab(value)

    def flatten(self, expr: str) -> Any:
        """Python representation of a MATLAB expression's value (via ``pyp_flatten``)."""
        self.eng.eval(f"pyp_records__ = pyp_flatten({expr});", nargout=0)
        records = self.eng.workspace["pyp_records__"]
        self.eng.eval("clear pyp_records__;", nargout=0)
        return records_to_value(list(records))

    def provenance(self) -> dict[str, Any]:
        eng = self.eng
        info = dict.fromkeys(
            (
                "kwave_matlab_version",
                "kwave_matlab_path",
                "kwave_solver",
                "kwave_binary_path",
                "kwave_binary_version",
                "kwave_binary_sha256",
            ),
            "",
        )
        info.update(self._kwave_info)
        if not info.get("kwave_binary_wrapper"):
            info.pop("kwave_binary_wrapper", None)
        from pyproteus import __version__

        return {
            "matlab_version": str(eng.version(nargout=1)),
            "proteus_version": str(eng.proteus_version(nargout=1)),
            "oracle_commit": self.oracle_commit,
            "oracle_dirty": bool(self._status_before),
            **info,
            "matlab_path_extra": list(self.paths_extra),
            "created_utc": _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat(),
            "exporter_version": __version__,
            "exporter_commit": repo_commit(),
            "platform": f"{platform.system()}-{platform.release()}-{platform.machine()}",
            "numpy_version": np.__version__,
        }
