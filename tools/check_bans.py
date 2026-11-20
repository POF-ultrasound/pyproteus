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
* ``from numpy import`` of any of the above names, ``from numpy import *``,
  and references such as ``r = np.round``.
* ``floor(x + 0.5)`` / ``int(x + 0.5)``: wrong for negative halves and for
  0.49999999999999994.

``list.sort(key=...)``/``reverse=`` calls are allowed (Python's sort is stable).

A line can opt out with a trailing ``# bans: allow`` comment and a reason.

Limitation: calls on a ``pyproteus.numerics`` import alias (``nm.sort``) are
trusted by name, so a local variable or parameter that shadows the alias is
trusted too. Don't reuse the alias name for anything else.

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
WRAPPER_MODULE = "pyproteus.numerics"  # calls such as nm.sort(...) are the fix, not the bug
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
    """kind="stable" given as keyword or as any positional string constant."""
    for kw in call.keywords:
        if kw.arg == "kind":
            return isinstance(kw.value, ast.Constant) and kw.value.value == "stable"
    return any(isinstance(a, ast.Constant) and a.value == "stable" for a in call.args)


def _is_python_list_sort(call: ast.Call) -> bool:
    """list.sort / sorted-style calls (key= or reverse=): always stable, reject kind=."""
    return any(kw.arg in ("key", "reverse") for kw in call.keywords)


def _is_half(expr: ast.expr) -> bool:
    return isinstance(expr, ast.Constant) and expr.value == 0.5


def _adds_half(expr: ast.expr) -> bool:
    return (
        isinstance(expr, ast.BinOp)
        and isinstance(expr.op, ast.Add | ast.Sub)
        and (_is_half(expr.left) or _is_half(expr.right))
    )


class _Visitor(ast.NodeVisitor):
    def __init__(self, path: Path, lines: list[str]) -> None:
        self.path = path
        self.lines = lines
        self.numpy_aliases: set[str] = set()
        self.wrapper_aliases: set[str] = set()
        self.violations: list[Violation] = []
        self._calls_seen: set[int] = set()

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
                if alias.name == "*":
                    self._flag(node, f"`from {node.module} import *` hides banned names")
                elif alias.name in BANNED_IMPORTS:
                    self._flag(
                        node,
                        f"`from {node.module} import {alias.name}` is banned "
                        "(MATLAB semantics differ); use pyproteus.numerics",
                    )
        self.generic_visit(node)

    def _is_wrapper(self, expr: ast.expr) -> bool:
        """``nm`` (an import alias of pyproteus.numerics) or the dotted path itself."""
        if isinstance(expr, ast.Name):
            return expr.id in self.wrapper_aliases
        return ast.unparse(expr) == WRAPPER_MODULE

    def _is_numpy(self, expr: ast.expr) -> bool:
        return isinstance(expr, ast.Name) and expr.id in self.numpy_aliases

    def visit_Attribute(self, node: ast.Attribute) -> None:
        # np.round / np.interp / np.argsort used as a value (r = np.round; r(x)).
        if (
            id(node) not in self._calls_seen
            and self._is_numpy(node.value)
            and node.attr in ROUND_FUNCS | INTERP_FUNCS | SORT_FUNCS
        ):
            self._flag(
                node,
                f"reference to `{node.value.id}.{node.attr}` is banned "
                "(MATLAB semantics differ); use pyproteus.numerics",
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func = node.func
        if isinstance(func, ast.Attribute):
            self._calls_seen.add(id(func))
        if isinstance(func, ast.Name) and func.id == "round":
            self._flag(node, "builtin `round` rounds halves to even; use numerics.matlab_round")
        elif (
            isinstance(func, ast.Name | ast.Attribute)
            and (func.id if isinstance(func, ast.Name) else func.attr) in ("floor", "int", "fix")
            and node.args
            and _adds_half(node.args[0])
        ):
            self._flag(
                node,
                "`floor(x + 0.5)` is not MATLAB round (wrong for negatives and "
                "0.49999999999999994); use numerics.matlab_round",
            )
        elif isinstance(func, ast.Attribute) and not self._is_wrapper(func.value):
            name = func.attr
            if name == "round" or (name in ROUND_FUNCS and self._is_numpy(func.value)):
                self._flag(node, f"`.{name}(` rounds halves to even; use numerics.matlab_round")
            elif (
                name in SORT_FUNCS and not _is_stable_kind(node) and not _is_python_list_sort(node)
            ):
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
                elif alias.name == WRAPPER_MODULE and alias.asname:
                    visitor.wrapper_aliases.add(alias.asname)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            for alias in node.names:
                if f"{node.module}.{alias.name}" == WRAPPER_MODULE:
                    visitor.wrapper_aliases.add(alias.asname or alias.name)
    # A name bound to both numpy and pyproteus.numerics is never trusted.
    conflicts = visitor.numpy_aliases & visitor.wrapper_aliases
    visitor.wrapper_aliases -= conflicts
    visitor.visit(tree)
    for name in sorted(conflicts):
        visitor._flag(
            tree.body[0] if tree.body else tree,
            f"`{name}` is bound to both numpy and pyproteus.numerics",
        )
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
