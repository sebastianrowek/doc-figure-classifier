"""
Colour schemes.

Corporate design in annual reports is not random: it is one or two brand
colours, a neutral grey, and occasionally a green/red pair for direction. The
palettes below are hand-picked to sit in that space rather than sampled from
RGB, because a model trained on arbitrary colours learns nothing useful about
what a real chart looks like.
"""

from __future__ import annotations

from dataclasses import dataclass

from .rng import Rng

# --------------------------------------------------------------------------
# Colour helpers
# --------------------------------------------------------------------------


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    r, g, b = (max(0, min(255, round(c))) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def mix(a: str, b: str, t: float) -> str:
    """Linear blend; t=0 -> a, t=1 -> b."""
    ra, ga, ba = hex_to_rgb(a)
    rb, gb, bb = hex_to_rgb(b)
    return rgb_to_hex((ra + (rb - ra) * t, ga + (gb - ga) * t, ba + (bb - ba) * t))


def lighten(c: str, t: float) -> str:
    return mix(c, "#ffffff", t)


def darken(c: str, t: float) -> str:
    return mix(c, "#000000", t)


def luminance(c: str) -> float:
    r, g, b = hex_to_rgb(c)
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0


def readable_on(bg: str) -> str:
    """Pick near-black or near-white text for a background."""
    return "#f2f4f7" if luminance(bg) < 0.45 else "#1b1f24"


# --------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Palette:
    name: str
    series: tuple[str, ...]  # data series colours, in order
    background: str
    text: str  # titles, data labels
    muted: str  # axis lines, tick labels, source note
    grid: str
    accent: str  # single highlighted bar / reference line
    positive: str  # waterfall increase, later phases
    negative: str  # waterfall decrease
    dark: bool

    def color(self, i: int) -> str:
        return self.series[i % len(self.series)]


# Brand-like seed sets. Roughly: German industrials, financials, chemicals,
# utilities, healthcare, and the muted consulting-deck look.
_SEEDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("navy_steel", ("#123a63", "#2e6da4", "#7aa6c8", "#b9cfe0", "#e2ebf2")),
    ("deep_blue", ("#0b2d5b", "#1f5f9e", "#4e91cd", "#93bde0", "#cfe1f0")),
    ("teal_slate", ("#0f4c5c", "#1b7f95", "#4bb0c0", "#93d2dc", "#d3ecef")),
    ("forest", ("#1d4d2b", "#2f7d44", "#63ad74", "#a3d3ab", "#dcecdf")),
    ("olive_sand", ("#4a5320", "#7c8b32", "#a9b45f", "#cdd398", "#ecefd6")),
    ("corporate_red", ("#8c1c22", "#c02a30", "#dc6b6f", "#eeaaac", "#f8dcdd")),
    ("burgundy", ("#5b1a33", "#8f2b52", "#bf5f80", "#dda3b6", "#f2dce4")),
    ("graphite_orange", ("#2b2f36", "#5a6270", "#9aa3b0", "#e08a2e", "#f3c98c")),
    ("petrol_lime", ("#124f57", "#1f8a8c", "#6bbf59", "#b3dd9d", "#e2f1d6")),
    ("indigo_gold", ("#25306b", "#4c5aa8", "#8b95cf", "#c9a227", "#e7d791")),
    ("chemical_green", ("#00594f", "#008573", "#4fb3a1", "#9ad6c9", "#d8efe9")),
    ("utility_cyan", ("#00456b", "#00789e", "#3ba9c8", "#8bd0e0", "#d0edf4")),
    ("pharma_violet", ("#3d2a63", "#63479c", "#9280c4", "#c2b6e0", "#e6e0f2")),
    ("mint_ink", ("#16303a", "#2f6b6f", "#54a09a", "#94c9c1", "#d4e9e5")),
    ("bank_blue_grey", ("#1b3a57", "#456c8c", "#7e9db4", "#b4c7d5", "#e0e8ee")),
    ("earth_brown", ("#4a3222", "#7d5738", "#ac8560", "#d0b799", "#eee1d2")),
    ("industrial_grey_blue", ("#37424d", "#5c6b7a", "#8896a5", "#b6c1cb", "#e1e6ea")),
    ("aqua_navy", ("#0d2c4a", "#146d8e", "#2fa8b8", "#87cfd6", "#d2ecef")),
    ("plum_grey", ("#412a3f", "#6f4a68", "#9c7995", "#c6adc1", "#e9dde5")),
    ("sun_teal", ("#e08c1a", "#f0b04c", "#1c6e78", "#57a3aa", "#a9cfd3")),
    ("nordic", ("#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51")),
    ("moss_stone", ("#3c4a3e", "#65795f", "#93a58a", "#c1cbb6", "#e6ebe0")),
    ("ocean_deep", ("#03254c", "#1167b1", "#187bcd", "#2a9df4", "#d0efff")),
    ("clay_navy", ("#1f2f47", "#3f5a7d", "#c1663f", "#e2a077", "#f3d9c5")),
)

# Neutral backgrounds a report page actually uses.
_LIGHT_BACKGROUNDS = ("#ffffff", "#ffffff", "#ffffff", "#fbfbf9", "#f7f8fa", "#f4f2ee", "#eef1f4")
_DARK_BACKGROUNDS = ("#1c2027", "#182029", "#232830")

# Direction pairs for waterfalls / signed values (phase 2, defined here so the
# whole colour vocabulary lives in one place).
_DIRECTION_PAIRS = (
    ("#2e7d32", "#c62828"),  # classic green / red
    ("#3f8f5c", "#b0453a"),  # muted
    ("#1b6ca8", "#d98634"),  # blue / orange, colour-blind friendlier
    ("#4a4a4a", "#9e9e9e"),  # dark / light monochrome
    ("#0f7b6c", "#a63d40"),
)


def _finish(
    name: str,
    series: tuple[str, ...],
    background: str,
    accent: str,
    rng: Rng,
    dark: bool = False,
) -> Palette:
    text = readable_on(background)
    muted = mix(text, background, 0.45)
    grid = mix(text, background, 0.86 if not dark else 0.80)
    pos, neg = rng.pick(_DIRECTION_PAIRS)
    if dark:
        pos, neg = lighten(pos, 0.25), lighten(neg, 0.25)
    return Palette(
        name=name,
        series=series,
        background=background,
        text=text,
        muted=muted,
        grid=grid,
        accent=accent,
        positive=pos,
        negative=neg,
        dark=dark,
    )


def _corporate(rng: Rng) -> Palette:
    name, seed = rng.pick(_SEEDS)
    series = seed
    # Real charts rarely use the whole ramp; often just the first two or three.
    if rng.chance(0.45):
        series = seed[: rng.randint(2, 3)]
    if rng.chance(0.15):
        series = tuple(reversed(seed))
    bg = rng.pick(_LIGHT_BACKGROUNDS)
    accent = rng.pick((seed[0], seed[-2] if len(seed) > 2 else seed[-1], "#e08a2e", "#c02a30"))
    return _finish(f"corp_{name}", series, bg, accent, rng)


def _mono_ramp(rng: Rng) -> Palette:
    name, seed = rng.pick(_SEEDS)
    base = seed[0]
    n = rng.randint(3, 6)
    # Light-to-dark ramp of one brand colour -- extremely common for stacked
    # segments and multi-year columns.
    series = tuple(lighten(base, t) for t in [0.72 - 0.72 * i / max(1, n - 1) for i in range(n)])
    if rng.chance(0.5):
        series = tuple(reversed(series))
    bg = rng.pick(_LIGHT_BACKGROUNDS)
    return _finish(f"mono_{name}", series, bg, darken(base, 0.15), rng)


def _accent_on_grey(rng: Rng) -> Palette:
    _, seed = rng.pick(_SEEDS)
    accent = seed[rng.randint(0, 1)]
    greys = ("#b9bec6", "#cbd0d6", "#a6acb4", "#d6dade")
    n = rng.randint(3, 5)
    series = tuple(rng.pick(greys) for _ in range(n))
    bg = rng.pick(_LIGHT_BACKGROUNDS)
    return _finish("accent_on_grey", series, bg, accent, rng)


def _greyscale(rng: Rng) -> Palette:
    n = rng.randint(3, 5)
    lo, hi = rng.uniform(0.15, 0.30), rng.uniform(0.72, 0.88)
    series = tuple(
        rgb_to_hex((v * 255,) * 3)
        for v in [lo + (hi - lo) * i / max(1, n - 1) for i in range(n)]
    )
    if rng.chance(0.5):
        series = tuple(reversed(series))
    bg = rng.pick(("#ffffff", "#ffffff", "#f7f7f7"))
    return _finish("greyscale", series, bg, "#111111", rng)


def _dark(rng: Rng) -> Palette:
    name, seed = rng.pick(_SEEDS)
    # Most seeds are dark by design, so a flat lighten leaves navy bars on a
    # near-black panel -- invisible after JPEG. Lift each colour until it
    # actually separates from the background.
    series = tuple(_min_luminance(c, 0.42) for c in seed[: rng.randint(2, 4)])
    bg = rng.pick(_DARK_BACKGROUNDS)
    return _finish(f"dark_{name}", series, bg, _min_luminance(seed[0], 0.62), rng, dark=True)


def _min_luminance(c: str, floor: float) -> str:
    """Lighten a colour until it reaches a minimum luminance."""
    for _ in range(12):
        if luminance(c) >= floor:
            break
        c = lighten(c, 0.16)
    return c


def contrast_on(c: str, background: str, min_delta: float = 0.26) -> str:
    """
    Push a colour away from the background until it is clearly visible.

    A pale tint that reads fine as a large bar fill disappears entirely as a
    1.5 pt line, and after downsampling and JPEG it is gone. Fills can stay
    light; strokes cannot.
    """
    bg = luminance(background)
    for _ in range(14):
        if abs(luminance(c) - bg) >= min_delta:
            break
        c = darken(c, 0.14) if bg > 0.5 else lighten(c, 0.14)
    return c


_BUILDERS = {
    _corporate: 0.50,
    _mono_ramp: 0.18,
    _accent_on_grey: 0.16,
    _greyscale: 0.08,
    _dark: 0.08,
}


def sample_palette(rng: Rng) -> Palette:
    return rng.weighted(_BUILDERS)(rng)


def ramp(pal: Palette, n: int) -> list[str]:
    """
    n visually distinct colours from a palette.

    ``Palette.color()`` cycles, which is fine for two or three bar series but
    wrong for pie slices and stack segments: with a three-colour palette and six
    slices, slice 1 and slice 4 come out identical and the boundary between
    adjacent wedges disappears. Interpolating along the palette instead keeps
    every segment distinguishable and happens to reproduce the monochrome ramp
    that corporate design uses for exactly this case.
    """
    base = list(pal.series)
    if n <= len(base):
        return base[:n]
    if len(base) < 3:
        # Too short to interpolate usefully; widen the ends first.
        base = [lighten(base[0], 0.45), *base, darken(base[-1], 0.28)]

    out = []
    for i in range(n):
        t = i * (len(base) - 1) / (n - 1)
        lo = int(t)
        hi = min(lo + 1, len(base) - 1)
        out.append(mix(base[lo], base[hi], t - lo))
    return out


# Style presets need to force a particular family (a monochrome print look must
# not draw a five-colour brand palette), so the builders are public too.
corporate = _corporate
mono_ramp = _mono_ramp
accent_on_grey = _accent_on_grey
greyscale = _greyscale
dark = _dark
