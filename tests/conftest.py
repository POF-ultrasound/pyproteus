"""Shared pytest configuration."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from hypothesis import HealthCheck, settings

# Hypothesis profiles: "dev" (default) is quick; "ci" explores more; "oracle"
# keeps example counts low because each example batch costs an engine call.
settings.register_profile("dev", max_examples=100, deadline=None)
settings.register_profile("ci", max_examples=300, deadline=None, derandomize=True)
settings.register_profile(
    "oracle",
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture, HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))


def pytest_report_header(config) -> list[str]:
    from tests.oracle.conftest import DEFAULT_COMMIT, oracle_path

    path = oracle_path()
    if not path.is_dir():
        return [f"oracle: not found at {path} (oracle tests will skip)"]
    try:
        commit = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short=12", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    expected = os.environ.get("PYPROTEUS_ORACLE_COMMIT", DEFAULT_COMMIT)
    return [f"oracle: {Path(path)} @ {commit} (expected {expected})"]
