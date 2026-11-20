#!/usr/bin/env python
"""Ban NumPy calls whose semantics differ silently from MATLAB.

MATLAB and NumPy disagree on rounding halves, sort stability and interpolation
outside the data range (ROADMAP §5). Code outside ``src/pyproteus/numerics/``
must use the ``pyproteus.numerics`` wrappers instead. Flagged:

* ``np.round`` / ``np.around`` / ``np.rint``, any ``.round(...)`` method call
  and the builtin ``round(...)``: all round halves to even (MATLAB: away from
  zero). Use ``pyproteus.numerics.matlab_round``.
* ``sort`` / ``argsort`` calls (function or method) without ``kind="stable"``:
  NumPy's default quicksort is not stable (MATLAB ``sort`` is).
* ``np.interp``: clamps outside the range (MATLAB ``interp1`` returns NaN).
* ``from numpy import`` of any of the above names.

A line can opt out with a trailing ``# bans: allow`` comment and a reason.

Usage: ``python tools/check_bans.py [paths...]`` (default: ``src tests tools``).
Exit status 1 when anything is flagged.
"""

from __future__ import annotations

import argparse
import ast
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

ALLOW_MARK = "# bans: allow"
EXEMPT_DIRS = (Path("src") / "pyproteus" / "numerics",)
NUMPY_MODULES = {"numpy", "cupy"}
ROUND_FUNCS = {"round", "around", "rint", "round_"}
SORT_FUNCS = {"sort", "argsort"}
INTERP_FUNCS = {"interp"}
BANNED_IMPORTS = ROUND_FUNCS | SORT_FUNCS | INTERP_FUNCS


@dataclass(frozen=True)
class Violation:
    path: Path
    line: int
    col: int
    message: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}:{self.col + 1}: {self.message}"


def _is_stable_kind(call: ast.Call) -> bool:
    for kw in call.keywords:
        if kw.arg == "kind":
            return isinstance(kw.value, ast.Constant) and kw.value.value == "stable"
    return False


class _Visitor(ast.NodeVisitor):
    def __init__(self, path: Path, lines: list[str]) -> None:
        self.path = path
        self.lines = lines
        self.numpy_aliases: set[str] = set()
        self.violations: list[Violation] = []

    def _flag(self, node: ast.AST, message: str) -> None:
        lineno = getattr(node, "lineno", 1)
        end = getattr(node, "end_lineno", lineno) or lineno
        if any(ALLOW_MARK in self.lines[i - 1] for i in range(lineno, end + 1)):
            return
        self.violations.append(
            Violation(self.path, lineno, getattr(node, "col_offset", 0), message)
        )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module in NUMPY_MODULES:
            for alias in node.names:
                if alias.name in BANNED_IMPORTS:
                    self._flag(
                        node,
                        f"`from {node.module} import {alias.name}` is banned "
                        "(MATLAB semantics differ); use pyproteus.numerics",
                    )
        self.generic_visit(node)

    def _is_numpy(self, expr: ast.expr) -> bool:
        return isinstance(expr, ast.Name) and expr.id in self.numpy_aliases

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Name) and func.id == "round":
            self._flag(node, "builtin `round` rounds halves to even; use numerics.matlab_round")
        elif isinstance(func, ast.Attribute):
            name = func.attr
            if name == "round" or (name in ROUND_FUNCS and self._is_numpy(func.value)):
                self._flag(node, f"`.{name}(` rounds halves to even; use numerics.matlab_round")
            elif name in SORT_FUNCS and not _is_stable_kind(node):
                self._flag(
                    node,
                    f'`.{name}(` without kind="stable"; use numerics.sort/argsort '
                    'or pass kind="stable"',
                )
            elif name in INTERP_FUNCS and self._is_numpy(func.value):
                self._flag(node, "`np.interp` clamps outside the range; use numerics.interp1")
        self.generic_visit(node)


def check_source(source: str, path: Path) -> list[Violation]:
    """Return the violations in one Python source string."""
    tree = ast.parse(source, filename=str(path))
    visitor = _Visitor(path, source.splitlines())
    # Collect numpy aliases first so that order of imports does not matter.
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in NUMPY_MODULES:
                    visitor.numpy_aliases.add(alias.asname or alias.name)
    visitor.visit(tree)
    return visitor.violations


def _is_exempt(path: Path, root: Path) -> bool:
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return any(rel.is_relative_to(d) for d in EXEMPT_DIRS)


def iter_python_files(paths: Iterable[Path]) -> Iterator[Path]:
    for p in paths:
        if p.is_dir():
            yield from sorted(p.rglob("*.py"))
        elif p.suffix == ".py":
            yield p


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="repository root used to locate the exempt numerics package",
    )
    args = parser.parse_args(argv)
    paths = args.paths or [args.root / d for d in ("src", "tests", "tools")]

    violations: list[Violation] = []
    for path in iter_python_files(paths):
        if _is_exempt(path, args.root):
            continue
        violations.extend(check_source(path.read_text(encoding="utf-8"), path))

    for v in violations:
        print(v)
    if violations:
        print(
            f"\n{len(violations)} banned call(s). See tools/check_bans.py and CONTRIBUTING.md.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
