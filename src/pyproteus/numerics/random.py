"""MATLAB ``rng(seed, 'twister')`` streams: ``rand`` and ``randi``, bit for bit.

MATLAB's ``'twister'`` generator is MT19937 seeded with ``init_genrand(seed)``
and draws doubles with 53-bit resolution (two 32-bit outputs per double), the
same as NumPy's legacy ``RandomState``. Verified against MATLAB R2025a:

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


def _size(args: tuple) -> tuple[int, ...]:
    """MATLAB size arguments: rand(n) → n×n, rand(m,n,...), rand([m n ...])."""
    if len(args) == 0:
        return (1, 1)
    if len(args) == 1:
        a = args[0]
        if isinstance(a, tuple | list | np.ndarray):
            dims = tuple(int(d) for d in a)
            if len(dims) == 1:
                return (dims[0], dims[0])
            return dims
        return (int(a), int(a))
    return tuple(int(d) for d in args)


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
        n = int(np.prod(size)) if size else 1
        return self._rs.random_sample(n).reshape(size, order="F")

    def rand(self, *size) -> np.ndarray:
        """MATLAB ``rand(...)``: uniform doubles in (0, 1), column-major fill."""
        return self._uniform(_size(size))

    def randi(self, imax, *size) -> np.ndarray:
        """MATLAB ``randi(imax, ...)`` or ``randi([lo, hi], ...)``; returns doubles."""
        if isinstance(imax, tuple | list | np.ndarray):
            lo, hi = (int(v) for v in imax)
        else:
            lo, hi = 1, int(imax)
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
