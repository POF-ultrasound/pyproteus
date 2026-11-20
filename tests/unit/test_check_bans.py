"""Tests for tools/check_bans.py."""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("check_bans", ROOT / "tools" / "check_bans.py")
check_bans = importlib.util.module_from_spec(_spec)
sys.modules["check_bans"] = check_bans
_spec.loader.exec_module(check_bans)


def lines_flagged(src: str) -> list[int]:
    return [v.line for v in check_bans.check_source(textwrap.dedent(src), Path("x.py"))]


@pytest.mark.parametrize(
    "src",
    [
        "import numpy as np\nnp.round(x)",
        "import numpy\nnumpy.round(x)",
        "import numpy as xp\nxp.around(x)",
        "import numpy as np\nnp.rint(x)",
        "x.round()",
        "x.round(2)",
        "round(1.5)",
        "import numpy as np\nnp.sort(x)",
        "import numpy as np\nnp.argsort(x)",
        "x.argsort()",
        "x.sort()",
        "import numpy as np\nnp.argsort(x, kind='quicksort')",
        "import numpy as np\nnp.argsort(x, kind='mergesort')",
        "import numpy as np\nnp.argsort(x, kind=k)",
        "import numpy as np\nnp.interp(q, x, y)",
        "from numpy import round",
        "from numpy import interp as i",
        "from numpy import argsort",
        "import numpy as np\ny = f(np.round(x) + 1)",
        "import numpy as np\nr = np.round\nr(x)",
        "import numpy as np\nf = np.interp",
        "import numpy as np\nkey = np.argsort",
        "from numpy import *",
        "import numpy as np\nnp.floor(x + 0.5)",
        "math.floor(0.5 + x)",
        "int(x + 0.5)",
    ],
)
def test_flags(src: str) -> None:
    assert lines_flagged(src), src


@pytest.mark.parametrize(
    "src",
    [
        "import numpy as np\nnp.sort(x, kind='stable')",
        'import numpy as np\nnp.argsort(x, axis=0, kind="stable")',
        "x.argsort(kind='stable')",
        "sorted(x)",
        "import numpy as np\nnp.floor(x)",
        "import numpy as np\nnp.lexsort(x)",
        "import scipy\nscipy.interp(x)",  # not numpy: interp check is numpy-only
        "from pyproteus.numerics import matlab_round\nmatlab_round(x)",
        "s = 'np.round(x)'",
        "import numpy as np\nnp.round(x)  # bans: allow (display only)",
        "import numpy as np\nnp.round(\n    x,\n)  # bans: allow",
        "from numpy import floor",
        "xs.sort(key=len)",
        "xs.sort(reverse=True)",
        "import numpy as np\nnp.sort(x, 0, 'stable')",
        "x.sort(0, 'stable')",
        "import numpy as np\nnp.floor(x + 1.5)",
        "import numpy as np\nnp.floor(x)",
    ],
)
def test_allows(src: str) -> None:
    assert lines_flagged(src) == [], src


def test_alias_declared_after_use() -> None:
    assert lines_flagged("def f(x):\n    return np.interp(x, 1, 2)\nimport numpy as np\n") == [2]


def test_reports_line_numbers() -> None:
    src = """
    import numpy as np
    a = 1
    b = np.round(a)
    c = a.argsort()
    """
    assert lines_flagged(src) == [4, 5]


def test_main_exempts_numerics(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    num = tmp_path / "src" / "pyproteus" / "numerics"
    num.mkdir(parents=True)
    (num / "rounding.py").write_text("import numpy as np\nnp.round(1)\n")
    other = tmp_path / "src" / "pyproteus" / "grid.py"
    other.write_text("import numpy as np\nnp.round(1)\n")
    assert check_bans.main(["--root", str(tmp_path), str(tmp_path / "src")]) == 1
    out = capsys.readouterr().out
    assert "grid.py:2" in out
    assert "rounding.py" not in out
    other.unlink()
    assert check_bans.main(["--root", str(tmp_path), str(tmp_path / "src")]) == 0


def test_repository_is_clean() -> None:
    assert check_bans.main(["--root", str(ROOT)]) == 0
