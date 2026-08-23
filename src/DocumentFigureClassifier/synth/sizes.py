r"""
Target image sizes drawn from the real crop distribution.

Guessing a size range is the easiest way to build a synthetic set that is
subtly wrong everywhere. Instead we sample from the width/height pairs the
extraction pipeline actually produced and jitter them slightly.

Measured over data/parsed/manifest.jsonl (n=408, one annual report):

    width   p05=222  p25=407  med=623  p75=760  p95=1256  max=2381
    height  p05=174  p25=254  med=360  p75=504  p95=938   max=1683
    aspect  p05=0.78 p25=1.11 med=1.48 p75=2.28 p95=3.06  max=5.27

Re-run ``python -m ...synth.sizes <manifest>`` after extracting more PDFs to
see whether the distribution has moved; the fallback table below is only used
when no manifest is available.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from ..taxonomy import MAX_ASPECT, MIN_AREA_PX, MIN_SIDE_PX
from .rng import Rng

# Fallback: quantiles of the measured distribution, expanded into a small
# synthetic cloud. Only used when no real manifest can be found.
_FALLBACK: tuple[tuple[int, int], ...] = (
    (222, 174), (260, 200), (300, 240), (340, 268), (380, 300),
    (407, 254), (440, 300), (470, 380), (500, 340), (540, 420),
    (580, 300), (623, 360), (660, 440), (700, 300), (740, 500),
    (760, 504), (800, 360), (860, 560), (900, 420), (960, 640),
    (1020, 480), (1100, 700), (1180, 540), (1256, 938), (1400, 620),
    (1600, 900), (1900, 1100), (2100, 1300),
)


class SizeSampler:
    """Samples (width, height) in pixels from an empirical cloud."""

    def __init__(self, pairs: list[tuple[int, int]], source: str) -> None:
        if not pairs:
            raise ValueError("no size pairs")
        self.pairs = pairs
        self.source = source

    # -- construction -----------------------------------------------------

    @classmethod
    def from_manifest(cls, path: Path | None) -> SizeSampler:
        """Load real crop sizes; fall back to the built-in table."""
        if path is None or not path.is_file():
            return cls([*_FALLBACK], source="fallback_table")

        pairs: list[tuple[int, int]] = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    w, h = int(rec["width"]), int(rec["height"])
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
                # Only crops the pipeline would have kept -- the manifest also
                # lists nothing else, but be explicit rather than assume.
                if _passes_junk_filter(w, h):
                    pairs.append((w, h))

        if len(pairs) < 30:
            return cls([*_FALLBACK], source="fallback_table")
        return cls(pairs, source=str(path))

    # -- sampling ---------------------------------------------------------

    def draw(self, rng: Rng) -> tuple[int, int]:
        """
        Pick a real pair, then jitter it so we do not reproduce only the 408
        exact sizes that happen to be in the manifest.

        Scale and aspect are jittered separately: a shared scale factor keeps
        the aspect-ratio distribution intact, a small independent aspect factor
        fills in the gaps between observed points.
        """
        w0, h0 = rng.pick(self.pairs)
        scale = rng.uniform(0.85, 1.18)
        aspect = math.sqrt(rng.uniform(0.93, 1.08))

        w = w0 * scale * aspect
        h = h0 * scale / aspect

        # Respect the same junk filter the extraction pipeline applies, so the
        # synthetic set contains no size a real crop could never have had.
        for _ in range(8):
            if _passes_junk_filter(w, h):
                break
            w, h = w * 1.12, h * 1.12

        return max(MIN_SIDE_PX, round(w)), max(MIN_SIDE_PX, round(h))


def _passes_junk_filter(w: float, h: float) -> bool:
    if w < MIN_SIDE_PX or h < MIN_SIDE_PX:
        return False
    if w * h < MIN_AREA_PX:
        return False
    return max(w / h, h / w) <= MAX_ASPECT


def _report(path: Path) -> None:
    """Print the quantile table for a manifest -- keeps the docstring honest."""
    s = SizeSampler.from_manifest(path)
    ws = sorted(w for w, _ in s.pairs)
    hs = sorted(h for _, h in s.pairs)
    ars = sorted(w / h for w, h in s.pairs)

    def q(v: list[float], p: float) -> float:
        return v[int(p * (len(v) - 1))]

    print(f"source: {s.source}  n={len(s.pairs)}")
    for name, v in (("width", ws), ("height", hs), ("aspect", ars)):
        print(
            f"{name:7s} p05={q(v, .05):7.2f} p25={q(v, .25):7.2f} "
            f"med={q(v, .5):7.2f} p75={q(v, .75):7.2f} "
            f"p95={q(v, .95):7.2f} max={max(v):7.2f}"
        )


if __name__ == "__main__":
    _report(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/parsed/manifest.jsonl"))
