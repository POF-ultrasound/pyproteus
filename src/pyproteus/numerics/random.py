"""MATLAB ``rng(seed, 'twister')`` streams: ``rand`` and ``randi``.

MATLAB's ``'twister'`` generator is the Mersenne Twister MT19937 (Matsumoto &
Nishimura 1998) with the reference ``init_genrand(seed)`` seeding and 53-bit
doubles built from two 32-bit outputs (the reference ``genrand_res53``), which
is what NumPy's legacy ``RandomState`` implements. The same seed therefore
gives the same stream. Oracle tests compare draws at the D-13 float class:

- ``rng(s,'twister'); rand(...)`` equals ``RandomState(s).random_sample(...)``,
  except that **seed 0 means MT seed 5489** (MATLAB's default state).
- ``rand(m, n, ...)`` fills in column-major order.
- ``randi(imax, ...)`` is ``floor(u*imax) + 1`` and ``randi([lo hi], ...)``
  is ``floor(u*(hi-lo+1)) + lo``, one double ``u`` per draw from the same
  stream, so ``rand`` and ``randi`` calls interleave as in MATLAB.

``randn`` is not provided (MATLAB's normal generator is not MT-based in the
same way); PROTEUS only uses it when the truncated-normal cutoff is infinite.
"""

from __future__ import annotations

import numpy as np

MATLAB_DEFAULT_SEED = 5489


def _as_int(d) -> int:
    if int(d) != d:
        raise ValueError(f"size arguments must be integers, got {d!r}")
    return max(int(d), 0)  # MATLAB treats negative sizes as 0


def _size(args: tuple) -> tuple[int, ...]:
    """MATLAB size arguments: rand(n) → n×n, rand(m,n,...), rand([m n ...]).

    Negative sizes count as 0 and trailing singleton dimensions beyond the
    second are dropped, as in MATLAB.
    """
    if len(args) == 0:
        dims: tuple[int, ...] = (1, 1)
    elif len(args) == 1:
        a = args[0]
        if isinstance(a, tuple | list | np.ndarray):
            dims = tuple(_as_int(d) for d in a)
            if len(dims) == 1:
                dims = (dims[0], dims[0])
        else:
            dims = (_as_int(a), _as_int(a))
    else:
        dims = tuple(_as_int(d) for d in args)
    while len(dims) > 2 and dims[-1] == 1:
        dims = dims[:-1]
    return dims


class MatlabRNG:
    """One MATLAB random stream (``rng(seed, 'twister')``).

    Size arguments follow MATLAB, including the trap that ``rand(n)`` is
    **n×n**: use ``rand(1, n)`` or ``rand(n, 1)`` for vectors. Results are
    2-D (or N-D) arrays with MATLAB's size; flatten them explicitly.
    """

    def __init__(self, seed: int = 0) -> None:
        if not 0 <= int(seed) < 2**32:
            raise ValueError("seed must be in [0, 2^32)")
        self.seed = int(seed)
        self._rs = np.random.RandomState(MATLAB_DEFAULT_SEED if seed == 0 else int(seed))

    def _uniform(self, size: tuple[int, ...]) -> np.ndarray:
        n = int(np.prod(size))
        return self._rs.random_sample(n).reshape(size, order="F")

    def rand(self, *size) -> np.ndarray:
        """MATLAB ``rand(...)``: uniform doubles in (0, 1), column-major fill."""
        return self._uniform(_size(size))

    def randi(self, imax, *size) -> np.ndarray:
        """MATLAB ``randi(imax, ...)`` or ``randi([lo, hi], ...)``; returns doubles."""
        bounds = list(imax) if isinstance(imax, tuple | list | np.ndarray) else [1, imax]
        if len(bounds) != 2 or any(int(b) != b for b in bounds):
            raise ValueError("randi: imax must be an integer or [lo, hi] integers")
        lo, hi = (int(b) for b in bounds)
        if hi < lo:
            raise ValueError("randi: upper bound must be >= lower bound")
        u = self._uniform(_size(size))
        return np.floor(u * (hi - lo + 1)) + lo

    def get_state(self):
        return self._rs.get_state()

    def set_state(self, state) -> None:
        self._rs.set_state(state)


def rng(seed: int = 0) -> MatlabRNG:
    """MATLAB ``rng(seed, 'twister')``: a new stream (no global state)."""
    return MatlabRNG(seed)
