"""Oracle tests for ``pyproteus.data``: shipped constants equal MATLAB's ``load``, bit for bit."""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from pyproteus import data
from tests.oracle.test_config import (  # noqa: F401  (meng is a fixture)
    assert_same_tree,
    flatten_tree,
    matlab_flatten,
    meng,
)

FILES = sorted(data.manifest()["files"].items())


@pytest.mark.parametrize(("out_name", "info"), FILES, ids=[f for f, _ in FILES])
def test_data_matches_matlab_load(
    meng: Any,  # noqa: F811
    oracle: Any,
    out_name: str,
    info: dict,
) -> None:
    for source in info["sources"]:
        src = oracle.path / source
        assert hashlib.sha256(src.read_bytes()).hexdigest() == info["sha256"], source
        expected = matlab_flatten(meng, f"load('{src}')")
        assert_same_tree(flatten_tree(data.load(info["matlab_name"])), expected)


def test_manifest_commit(oracle: Any) -> None:
    assert oracle.commit.startswith(data.manifest()["oracle_commit"][:7])
    assert data.manifest()["oracle_commit"] == oracle.commit


def test_all_shipped_constants_converted(oracle: Any) -> None:
    """Every constant .mat PROTEUS loads by name is in the package (beta_coeff excluded)."""
    sources = {s for _, info in FILES for s in info["sources"]}
    wanted = {
        "GUIfunctions/tissue_properties.mat",
        "GUIfunctions/fit_SigmaR_evaluated.mat",
        "Transducer-models/C1-6D.mat",
        *(
            str(p.relative_to(oracle.path))
            for p in (oracle.path / "microbubble-simulator" / "functions").glob("*.mat")
        ),
        *(str(p.relative_to(oracle.path)) for p in (oracle.path / "geometry_data").glob("*/*.mat")),
    }
    assert wanted <= sources, sorted(wanted - sources)
