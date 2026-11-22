"""Frozen MATLAB reference data: pooch registry, download cache and loader.

Reference files (format: ``docs/REFERENCE_DATA.md``) are exported from MATLAB
PROTEUS by ``tools/export_reference/`` and published on Zenodo. Only their
names and SHA-256 hashes live in git, in ``reference_registry.txt`` next to
this module (pooch registry format, ``# base_url:`` header line).

Environment variables:

``PYPROTEUS_REFERENCE_DIR``
    Local folder holding reference files (e.g. a fresh export). When set, files
    are read from there and nothing is downloaded. Files listed in the registry
    must still match its hash, unless ``PYPROTEUS_REFERENCE_VERIFY=0``.
``PYPROTEUS_REFERENCE_URL``
    Base URL overriding the registry's ``# base_url:`` (``https://...`` or
    ``doi:10.5281/zenodo.<record>/``).
``PYPROTEUS_REFERENCE_CACHE``
    Download cache folder (default: the OS cache folder, ``pyproteus/reference``).
``PYPROTEUS_REFERENCE_VERIFY``
    ``0`` skips the hash check of files in ``PYPROTEUS_REFERENCE_DIR``.
"""

from __future__ import annotations

import os
from importlib import resources
from pathlib import Path

from .reference_format import Reference, read_reference

ENV_DIR = "PYPROTEUS_REFERENCE_DIR"
ENV_URL = "PYPROTEUS_REFERENCE_URL"
ENV_CACHE = "PYPROTEUS_REFERENCE_CACHE"
ENV_VERIFY = "PYPROTEUS_REFERENCE_VERIFY"
REGISTRY_FILE = "reference_registry.txt"
SUFFIX = ".h5"


class ReferenceDataUnavailable(LookupError):
    """A reference file is not available (not registered, no URL, not downloaded)."""


class ReferenceHashMismatch(ValueError):
    """A local reference file does not match the registry hash."""


def _registry_text(path: str | os.PathLike | None = None) -> str:
    if path is not None:
        return Path(path).read_text(encoding="utf-8")
    return resources.files(__package__).joinpath(REGISTRY_FILE).read_text(encoding="utf-8")


def parse_registry(text: str) -> tuple[dict[str, str], str | None]:
    """Parse registry text → ({file name: "sha256:<hex>"}, base URL from the header or None)."""
    entries: dict[str, str] = {}
    base: str | None = None
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            key, _, value = line[1:].strip().partition(":")
            if key.strip() == "base_url" and value.strip():
                base = value.strip()
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"registry line {lineno}: expected '<file> <hash>', got {raw!r}")
        name, digest = parts
        if not digest.startswith("sha256:") or len(digest) != len("sha256:") + 64:
            raise ValueError(f"registry line {lineno}: hash must be sha256:<64 hex>")
        if "/" in name or "\\" in name or not name.endswith(SUFFIX):
            raise ValueError(f"registry line {lineno}: {name!r} must be a flat *.h5 file name")
        if name in entries:
            raise ValueError(f"registry line {lineno}: duplicate {name!r}")
        entries[name] = digest
    return entries, base


def registry(path: str | os.PathLike | None = None) -> dict[str, str]:
    """Registered reference files and their hashes."""
    return parse_registry(_registry_text(path))[0]


def base_url(path: str | os.PathLike | None = None) -> str | None:
    """Download base URL: ``PYPROTEUS_REFERENCE_URL`` or the registry header (None if neither)."""
    env = os.environ.get(ENV_URL, "").strip()
    if env:
        return env
    return parse_registry(_registry_text(path))[1]


def file_name(name: str) -> str:
    """``"GUIfunctions.reset_medium"`` or ``"...h5"`` → registry file name."""
    if "/" in name or "\\" in name:
        raise ValueError(f"reference names are flat file names, got {name!r}")
    return name if name.endswith(SUFFIX) else name + SUFFIX


def sha256(path: str | os.PathLike) -> str:
    """``sha256:<hex>`` of a file (registry hash format)."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def cache_dir() -> Path:
    env = os.environ.get(ENV_CACHE, "").strip()
    if env:
        return Path(env).expanduser()
    import pooch

    return Path(pooch.os_cache("pyproteus")) / "reference"


def fetch(name: str, *, registry_path: str | os.PathLike | None = None) -> Path:
    """Local path of a reference file: from ``PYPROTEUS_REFERENCE_DIR`` or downloaded (cached).

    Raises :class:`ReferenceDataUnavailable` when the file cannot be provided
    (tests skip on it) and :class:`ReferenceHashMismatch` when a local file
    does not match the registry.
    """
    fname = file_name(name)
    entries = registry(registry_path)
    local = os.environ.get(ENV_DIR, "").strip()
    if local:
        path = Path(local).expanduser() / fname
        if not path.is_file():
            raise ReferenceDataUnavailable(f"{fname} not found in {ENV_DIR}={local}")
        verify = os.environ.get(ENV_VERIFY, "1").strip() != "0"
        if verify and fname in entries:
            actual = sha256(path)
            if actual != entries[fname]:
                raise ReferenceHashMismatch(
                    f"{path}: {actual} != registry {entries[fname]} "
                    f"(re-run the registry update or set {ENV_VERIFY}=0)"
                )
        return path
    if fname not in entries:
        raise ReferenceDataUnavailable(f"{fname} is not in the reference registry")
    url = base_url(registry_path)
    if not url:
        raise ReferenceDataUnavailable(
            f"no download URL for reference data: set {ENV_URL} or {ENV_DIR} "
            "(no Zenodo record is registered yet)"
        )
    import pooch

    # Hashes are checked here, not by pooch: pooch raises ValueError for many
    # unrelated failures (file missing from a Zenodo record, DOI resolution...).
    fetcher = pooch.create(
        path=cache_dir(),
        base_url=url if url.endswith("/") else url + "/",
        registry=dict.fromkeys(entries),
        retry_if_failed=2,
    )
    try:
        path = Path(fetcher.fetch(fname))
    except (OSError, ValueError) as exc:  # network, HTTP 404, missing in record, bad DOI
        raise ReferenceDataUnavailable(f"could not download {fname}: {exc}") from exc
    actual = sha256(path)
    if actual != entries[fname]:
        path.unlink(missing_ok=True)  # never leave a corrupt file in the cache
        raise ReferenceHashMismatch(f"downloaded {fname}: {actual} != registry {entries[fname]}")
    return path


def load(name: str, *, registry_path: str | os.PathLike | None = None) -> Reference:
    """Fetch and read a reference file (:class:`pyproteus.io.reference_format.Reference`)."""
    return read_reference(fetch(name, registry_path=registry_path))
