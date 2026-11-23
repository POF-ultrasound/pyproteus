"""STL reading: port of ``READ_stl.m`` (mesh voxelisation toolbox).

Ported from the MATLAB "Mesh voxelisation" toolbox by Adam H. Aitkenhead
(The Christie NHS Foundation Trust), shipped with PROTEUS in
``mesh-voxelisation/``. Copyright (c) 2013, Adam H. Aitkenhead. All rights
reserved. Redistributed under the BSD-3-Clause licence (non-endorsement
clause naming The Christie NHS Foundation Trust); full text and disclaimer
in ``LICENSES/mesh-voxelisation-BSD.txt``.

The STL format itself is public: binary files are an 80-byte header, an int32
facet count and 50-byte records (normal and three vertices as float32, two
padding bytes); ASCII files are ``solid`` / ``facet normal`` / ``vertex`` text.
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path

import numpy as np

__all__ = ["identify_stl_format", "read_stl"]

_BINARY_RECORD = np.dtype(
    [("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")]
)


def identify_stl_format(path: str | PathLike[str]) -> str:
    """``'ascii'`` or ``'binary'``, by the rules of ``IDENTIFY_stl_format``.

    A file whose size is not ``84 + 50 n`` is ASCII. Otherwise it is ASCII
    only if it starts with ``solid`` (after stripping whitespace) and its last
    80 bytes contain ``endsolid``.
    """
    raw = Path(path).read_bytes()
    # MATLAB rem() keeps the sign: files shorter than 84 bytes go on to the
    # 'solid' check like MATLAB; strtrim also strips NUL characters.
    if len(raw) >= 84 and (len(raw) - 84) % 50 != 0:
        return "ascii"
    if raw[:80].strip(b" \t\n\r\x0b\x0c\x00")[:5] != b"solid":
        return "binary"
    return "ascii" if b"endsolid" in raw[-80:] else "binary"


def read_stl(
    path: str | PathLike[str], stl_format: str = "auto", *, return_normals: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Read a triangle mesh from an STL file (port of ``READ_stl.m``).

    Returns ``mesh_xyz``, an ``(N, 3, 3)`` float64 array with one row per
    facet, the x, y, z coordinates along axis 1 and the three vertices along
    axis 2 (``meshXYZ(facet, coord, vertex)`` in MATLAB). Binary float32
    coordinates are widened to float64 exactly, as MATLAB's ``fread``.
    With ``return_normals=True`` also returns the ``(N, 3)`` facet normals
    stored in the file.

    ``stl_format`` is ``'ascii'``, ``'binary'`` or ``'auto'`` (default,
    :func:`identify_stl_format`).
    """
    fmt = stl_format.lower()
    if fmt not in ("ascii", "binary"):
        fmt = identify_stl_format(path)
    if fmt == "binary":
        vertices, normals = _read_binary(path)
    else:
        vertices, normals = _read_ascii(path)
    return (vertices, normals) if return_normals else vertices


def _read_binary(path: str | PathLike[str]) -> tuple[np.ndarray, np.ndarray]:
    raw = Path(path).read_bytes()
    count = int(np.frombuffer(raw, "<i4", count=1, offset=80)[0])
    records = np.frombuffer(raw, _BINARY_RECORD, count=count, offset=84)
    # record vertices are (vertex, coord); MATLAB layout is (facet, coord, vertex)
    vertices = records["vertices"].astype(np.float64).transpose(0, 2, 1)
    return np.ascontiguousarray(vertices), records["normal"].astype(np.float64)


def _read_ascii(path: str | PathLike[str]) -> tuple[np.ndarray, np.ndarray]:
    lines = [ln.strip() for ln in Path(path).read_text(encoding="latin-1").splitlines()]
    lines = [ln for ln in lines if ln]
    normals = [ln[12:].split() for ln in lines if ln.startswith("facet normal")]
    verts = [ln[6:].split() for ln in lines if ln.startswith("vertex")]
    n = sum(ln == "endfacet" for ln in lines)
    if len(verts) != 3 * n:
        raise ValueError(f"{path}: {len(verts)} vertex lines for {n} facets (triangles only)")
    vertices = np.asarray(verts, dtype=np.float64).reshape(n, 3, 3).transpose(0, 2, 1)
    return np.ascontiguousarray(vertices), np.asarray(normals, dtype=np.float64).reshape(-1, 3)
