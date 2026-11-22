"""Record the SHA-256 of exported reference files in the pooch registry.

    python -m tools.export_reference.update_registry [--data DIR] [--base-url URL] [--check]

Reads every ``*.h5`` in the data folder (default ``tools/export_reference/data``),
checks it is a valid reference file, and rewrites
``src/pyproteus/io/reference_registry.txt`` keeping the header comments.
Entries for files not present locally are kept unless ``--prune``. Files
from a dirty or unpinned oracle, or a dirty exporter checkout, are refused
unless ``--allow-unclean``. ``--check`` only reports differences (exit 1).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from pyproteus.io import reference as ref
from pyproteus.io.reference_format import read_reference

HERE = Path(__file__).resolve().parent
REGISTRY = HERE.parents[1] / "src" / "pyproteus" / "io" / ref.REGISTRY_FILE


def render(text: str, entries: dict[str, str], base_url: str | None) -> str:
    """Registry text: header comments of ``text`` (base_url replaced), then sorted entries."""
    header = []
    for line in text.splitlines():
        s = line.strip()
        if not s.startswith("#"):
            continue
        key = s[1:].strip().partition(":")[0].strip()
        if key == "base_url":
            continue
        header.append(line)
    header.append(f"# base_url: {base_url or ''}".rstrip())
    body = [f"{k} {entries[k]}" for k in sorted(entries)]
    return "\n".join(header + body) + "\n"


PINNED_ORACLE = "8d37938"  # D-7; re-pin with the v1.1.0 tag


def provenance_problems(attrs: dict) -> list[str]:
    """Reasons a file must not be frozen (exploratory exports)."""
    problems = []
    if attrs["oracle_dirty"]:
        problems.append("oracle checkout had local changes")
    if not attrs["oracle_commit"].startswith(PINNED_ORACLE):
        problems.append(f"oracle commit {attrs['oracle_commit'][:12]} != {PINNED_ORACLE}")
    if attrs["exporter_commit"].endswith("-dirty") or attrs["exporter_commit"] == "unknown":
        problems.append(f"exporter commit {attrs['exporter_commit']!r} is not clean")
    return problems


def collect(data: Path, *, allow_unclean: bool = False) -> dict[str, str]:
    found = {}
    for path in sorted(data.glob("*" + ref.SUFFIX)):
        r = read_reference(path)  # validates format and provenance
        if r.name + ref.SUFFIX != path.name:
            raise SystemExit(f"{path}: file name does not match its 'name' attribute {r.name!r}")
        problems = provenance_problems(r.attrs)
        if problems and not allow_unclean:
            raise SystemExit(f"{path}: {'; '.join(problems)} (--allow-unclean to register anyway)")
        found[path.name] = ref.sha256(path)
    return found


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=HERE / "data")
    p.add_argument("--registry", type=Path, default=REGISTRY)
    p.add_argument("--base-url", help="set the registry base URL (e.g. doi:10.5281/zenodo.N/)")
    p.add_argument("--check", action="store_true", help="report differences, write nothing")
    p.add_argument(
        "--prune",
        action="store_true",
        help="drop registry entries with no local file (full re-export for a new Zenodo version)",
    )
    p.add_argument(
        "--allow-unclean",
        action="store_true",
        help="register files from a dirty/unpinned oracle or a dirty exporter checkout",
    )
    args = p.parse_args(argv)
    text = args.registry.read_text(encoding="utf-8")
    entries, base = ref.parse_registry(text)
    found = collect(args.data, allow_unclean=args.allow_unclean)
    changed = {k: v for k, v in found.items() if entries.get(k) != v}
    missing = sorted(set(entries) - set(found))
    if args.check:
        for k, v in changed.items():
            print(f"{k}: registry {entries.get(k, '(missing)')} != local {v}")
        for k in missing:
            print(f"{k}: registered, no local file")
        return 1 if changed or (args.prune and missing) else 0
    entries.update(found)
    if args.prune:
        for k in missing:
            del entries[k]
    new = render(text, entries, args.base_url if args.base_url is not None else base)
    ref.parse_registry(new)  # round-trip sanity
    tmp = args.registry.with_name(args.registry.name + ".partial")
    tmp.write_text(new, encoding="utf-8")
    os.replace(tmp, args.registry)
    pruned = len(missing) if args.prune else 0
    print(
        f"{len(found)} file(s) hashed, {len(changed)} changed, {pruned} pruned -> {args.registry}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
