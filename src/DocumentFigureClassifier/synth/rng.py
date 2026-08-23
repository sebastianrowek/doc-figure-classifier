"""
One seeded random stream per sample.

Everything about a sample -- its size, style, data and degradation -- is drawn
from a single ``Rng``. That is what makes ``--regenerate`` work: given
``(label, sub_type, seed)`` the image comes back byte-identical.

Consequence worth knowing: adding a draw in the middle of a renderer shifts
every draw after it, so previously generated ids no longer reproduce the same
image. That is fine during development, but do not expect ids to be stable
across generator versions.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import TypeVar

import numpy as np

T = TypeVar("T")


class Rng:
    """Thin, readable wrapper around numpy's Generator."""

    __slots__ = ("seed", "_g")

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self._g = np.random.default_rng(self.seed)

    # -- scalars ----------------------------------------------------------

    def chance(self, p: float) -> bool:
        """True with probability p."""
        return bool(self._g.random() < p)

    def uniform(self, lo: float, hi: float) -> float:
        return float(self._g.uniform(lo, hi))

    def randint(self, lo: int, hi: int) -> int:
        """Inclusive on both ends -- randint(3, 5) yields 3, 4 or 5."""
        return int(self._g.integers(lo, hi + 1))

    def normal(self, mu: float = 0.0, sigma: float = 1.0) -> float:
        return float(self._g.normal(mu, sigma))

    def beta(self, a: float, b: float) -> float:
        return float(self._g.beta(a, b))

    # -- sequences --------------------------------------------------------

    def pick(self, seq: Sequence[T]) -> T:
        return seq[int(self._g.integers(0, len(seq)))]

    def sample(self, seq: Sequence[T], k: int) -> list[T]:
        """k distinct elements, order preserved from the draw."""
        idx = self._g.choice(len(seq), size=k, replace=False)
        return [seq[int(i)] for i in idx]

    def shuffled(self, seq: Iterable[T]) -> list[T]:
        out = list(seq)
        self._g.shuffle(out)  # type: ignore[arg-type]
        return out

    def weighted(self, options: Mapping[T, float]) -> T:
        """Pick a key with probability proportional to its weight."""
        keys = list(options)
        w = np.asarray([options[k] for k in keys], dtype=float)
        return keys[int(self._g.choice(len(keys), p=w / w.sum()))]

    def floats(self, n: int, lo: float, hi: float) -> list[float]:
        return [float(x) for x in self._g.uniform(lo, hi, size=n)]

    def noise(self, shape: tuple[int, ...], sigma: float) -> np.ndarray:
        return self._g.normal(0.0, sigma, size=shape)
