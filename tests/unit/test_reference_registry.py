"""pooch registry, local override and loader for reference data (no MATLAB)."""

from __future__ import annotations

import functools
import http.server
import threading

import numpy as np
import pytest
from tools.export_reference import update_registry

from pyproteus.io import reference as ref
from pyproteus.io.reference_format import write_reference
from tests.unit.test_reference_format import attrs

H = "sha256:" + "0" * 64


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for v in (ref.ENV_DIR, ref.ENV_URL, ref.ENV_VERIFY):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv(ref.ENV_CACHE, str(tmp_path / "cache"))


def make_file(folder, name="mod.fn"):
    return write_reference(
        folder / f"{name}.h5", attrs(name=name), {"x": np.ones((2, 3))}, {"y": np.eye(3)}
    )


def write_registry(path, entries, base=""):
    text = (
        "# header\n# base_url: " + base + "\n" + "".join(f"{k} {v}\n" for k, v in entries.items())
    )
    path.write_text(text)
    return path


def test_shipped_registry_parses():
    entries, _base = ref.parse_registry(ref._registry_text())
    assert all(v.startswith("sha256:") for v in entries.values())


@pytest.mark.parametrize(
    "line",
    [
        "a.h5",  # no hash
        "a.h5 md5:abc",
        "a.h5 sha256:123",
        "sub/a.h5 " + H,
        "a.mat " + H,
        "a.h5 " + H + " extra",
    ],
)
def test_parse_rejects(line):
    with pytest.raises(ValueError):
        ref.parse_registry(line + "\n")


def test_parse_duplicate_and_base():
    with pytest.raises(ValueError, match="duplicate"):
        ref.parse_registry(f"a.h5 {H}\na.h5 {H}\n")
    entries, base = ref.parse_registry(f"# base_url: doi:10.5281/zenodo.1/\n\na.h5 {H}\n")
    assert entries == {"a.h5": H} and base == "doi:10.5281/zenodo.1/"
    assert ref.parse_registry("# base_url:\n")[1] is None


def test_base_url_env_override(tmp_path, monkeypatch):
    reg = write_registry(tmp_path / "r.txt", {}, "https://a/")
    assert ref.base_url(reg) == "https://a/"
    monkeypatch.setenv(ref.ENV_URL, "https://b/")
    assert ref.base_url(reg) == "https://b/"


def test_file_name():
    assert ref.file_name("GUIfunctions.reset_medium") == "GUIfunctions.reset_medium.h5"
    assert ref.file_name("x.h5") == "x.h5"
    with pytest.raises(ValueError):
        ref.file_name("a/b")


def test_local_dir_with_hash_check(tmp_path, monkeypatch):
    data = tmp_path / "data"
    p = make_file(data)
    reg = write_registry(tmp_path / "r.txt", {"mod.fn.h5": ref.sha256(p)})
    monkeypatch.setenv(ref.ENV_DIR, str(data))
    assert ref.fetch("mod.fn", registry_path=reg) == p
    r = ref.load("mod.fn", registry_path=reg)
    np.testing.assert_array_equal(r.outputs["y"], np.eye(3))

    write_registry(reg, {"mod.fn.h5": H})
    with pytest.raises(ref.ReferenceHashMismatch):
        ref.fetch("mod.fn", registry_path=reg)
    monkeypatch.setenv(ref.ENV_VERIFY, "0")
    assert ref.fetch("mod.fn", registry_path=reg) == p


def test_local_dir_unregistered_file_ok_missing_file_unavailable(tmp_path, monkeypatch):
    data = tmp_path / "data"
    make_file(data, "new.case")
    reg = write_registry(tmp_path / "r.txt", {})
    monkeypatch.setenv(ref.ENV_DIR, str(data))
    assert ref.fetch("new.case", registry_path=reg).name == "new.case.h5"
    with pytest.raises(ref.ReferenceDataUnavailable):
        ref.fetch("absent", registry_path=reg)


def test_unavailable_without_url_or_registration(tmp_path):
    reg = write_registry(tmp_path / "r.txt", {"a.h5": H})
    with pytest.raises(ref.ReferenceDataUnavailable, match="not in the reference registry"):
        ref.fetch("b", registry_path=reg)
    with pytest.raises(ref.ReferenceDataUnavailable, match="no download URL"):
        ref.fetch("a", registry_path=reg)


@pytest.fixture
def http_folder(tmp_path):
    served = tmp_path / "served"
    served.mkdir()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(served))
    handler.log_message = lambda *a, **k: None  # type: ignore[attr-defined]
    try:
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    except OSError as exc:  # e.g. a sandbox that forbids binding
        pytest.skip(f"cannot start a local HTTP server: {exc}")
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield served, f"http://127.0.0.1:{srv.server_address[1]}/"
    srv.shutdown()


def test_download_through_pooch(tmp_path, monkeypatch, http_folder):
    served, url = http_folder
    p = make_file(served)
    reg = write_registry(tmp_path / "r.txt", {"mod.fn.h5": ref.sha256(p)}, url)
    got = ref.fetch("mod.fn", registry_path=reg)
    assert got.parent == tmp_path / "cache"
    assert ref.sha256(got) == ref.sha256(p)
    assert ref.load("mod.fn", registry_path=reg).name == "mod.fn"

    # Corrupted download: hash mismatch is reported, not a silent pass.
    write_registry(reg, {"other.h5": H}, url)
    (served / "other.h5").write_bytes(b"not hdf5")
    with pytest.raises(ref.ReferenceHashMismatch):
        ref.fetch("other", registry_path=reg)
    # Missing on the server.
    write_registry(reg, {"gone.h5": H}, url)
    with pytest.raises(ref.ReferenceDataUnavailable):
        ref.fetch("gone", registry_path=reg)


def test_update_registry_round_trip(tmp_path):
    data = tmp_path / "data"
    p1 = make_file(data, "a.one")
    p2 = make_file(data, "b.two")
    reg = tmp_path / "r.txt"
    reg.write_text("# keep me\n# base_url: https://old/\nz.kept.h5 " + H + "\n", encoding="utf-8")
    argv = ["--data", str(data), "--registry", str(reg)]
    assert update_registry.main([*argv, "--check"]) == 1
    assert update_registry.main(argv) == 0
    text = reg.read_text()
    entries, base = ref.parse_registry(text)
    assert entries == {"a.one.h5": ref.sha256(p1), "b.two.h5": ref.sha256(p2), "z.kept.h5": H}
    assert base == "https://old/" and text.startswith("# keep me\n")
    assert update_registry.main([*argv, "--check"]) == 0
    assert update_registry.main([*argv, "--base-url", "doi:10.5281/zenodo.9/"]) == 0
    assert ref.parse_registry(reg.read_text())[1] == "doi:10.5281/zenodo.9/"


def test_update_registry_rejects_misnamed(tmp_path):
    data = tmp_path / "data"
    p = make_file(data, "a.one")
    p.rename(data / "renamed.h5")
    reg = write_registry(tmp_path / "r.txt", {})
    with pytest.raises(SystemExit, match="does not match"):
        update_registry.main(["--data", str(data), "--registry", str(reg)])
