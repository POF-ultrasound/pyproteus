"""Fetch the shipped PROTEUS vessel geometries (STL) from Zenodo record 10573662.

    python -m tools.fetch_geometries [--out data/geometries] [--check]

The STLs are members of the record's ``*_no_vtu.zip`` archives (0.57 GB and
2.3 GB). Only the two members are read, with HTTP range requests (about 20 MB;
the zip CRC-32 is checked by :mod:`zipfile`), and written to
``<out>/<folder>/<file>``. Their SHA-256 must match
``src/pyproteus/io/geometry_registry.txt``. Point ``PYPROTEUS_GEOMETRIES_PATH``
at ``<out>`` to run ``tests/oracle/test_medium_define.py::test_define_medium_shipped_geometries``.
``--check`` only verifies files already present.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

RECORD = "https://zenodo.org/api/records/10573662/files"
REGISTRY = (
    Path(__file__).resolve().parents[1] / "src" / "pyproteus" / "io" / "geometry_registry.txt"
)
#: registry path → (zip on the record, member inside the zip)
SOURCES = {
    "renal_tree/maa.stl": ("renal_tree_no_vtu.zip", "renal_tree_no_vtu/maa.stl"),
    "mouse_brain/lpa_mouse_t10.stl": (
        "mouse_brain_no_vtu.zip",
        "mouse_brain_no_vtu/lpa_mouse_t10.stl",
    ),
}


def read_registry(path: Path = REGISTRY) -> dict[str, str]:
    """``{relative path: "sha256:<hex>"}`` (pooch registry format, ``#`` comments)."""
    entries: dict[str, str] = {}
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2 or not parts[1].startswith("sha256:") or len(parts[1]) != 71:
            raise ValueError(f"{path.name} line {lineno}: expected '<file> sha256:<hex>'")
        entries[parts[0]] = parts[1]
    return entries


def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


class _HttpRangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file through HTTP range requests."""

    def __init__(self, url: str) -> None:
        self.url, self.pos, self.fetched = url, 0, 0
        self.size = int(self._get(0, 0)[1].split("/")[-1])

    def _get(self, a: int, b: int) -> tuple[bytes, str]:
        for _ in range(10):
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={a}-{b}"})
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    if r.status != 206:
                        raise RuntimeError(f"server ignored the range request ({r.status})")
                    return r.read(), r.headers["Content-Range"]
            except urllib.error.HTTPError as e:
                if e.code != 429:
                    raise
                time.sleep(int(e.headers.get("retry-after", 60)) + 1)
        raise RuntimeError("rate limited by zenodo.org")

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def readinto(self, b) -> int:  # type: ignore[no-untyped-def]
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        data = self._get(self.pos, self.pos + n - 1)[0]
        b[: len(data)] = data
        self.pos += len(data)
        self.fetched += len(data)
        return len(data)


def fetch(out: Path) -> None:
    registry = read_registry()
    for name, (zip_name, member) in SOURCES.items():
        target = out / name
        if target.is_file() and sha256(target) == registry[name]:
            print(f"{name}: present, hash ok")
            continue
        remote = _HttpRangeFile(f"{RECORD}/{zip_name}/content")
        with zipfile.ZipFile(io.BufferedReader(remote, buffer_size=1 << 20)) as z:
            data = z.read(member)  # CRC-32 checked
        digest = "sha256:" + hashlib.sha256(data).hexdigest()
        if digest != registry[name]:
            raise RuntimeError(f"{name}: {digest} does not match the registry {registry[name]}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        print(f"{name}: {len(data)} bytes ({remote.fetched} read from {zip_name})")


def check(out: Path) -> int:
    bad = 0
    for name, digest in read_registry().items():
        path = out / name
        ok = path.is_file() and sha256(path) == digest
        print(f"{name}: {'ok' if ok else 'missing or different'}")
        bad += not ok
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data/geometries"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.check:
        return check(args.out)
    fetch(args.out)
    return check(args.out)


if __name__ == "__main__":
    sys.exit(main())
