"""
Logos and pictograms -- brand marks, seals, awards, icon sets, SDG-style tiles.

This class is entirely procedural: real logos cannot be used (trademark), and
the model does not need real ones -- it needs to learn the *shape* of the class,
a single centred mark on empty ground with no axes and no data series. That is
also the invariant (base._check_logo): no bar, line or pie series present.

Everything is drawn as matplotlib patches on an equal-aspect, axis-off canvas.
The SDG tiles are an original design, deliberately not the UN colour set or
numbering -- visually a numbered coloured-tile grid is all the model needs.

Sub-types
---------
wordmark    invented company name + a geometric signet
seal        concentric rings with text on a circular path, a certification look
badge       award badge -- laurel wreath / ribbon / stars
icon_grid   a 2x3 (etc.) grid of simple pictograms, the sustainability-chapter look
sdg_tiles   numbered coloured tiles, original design
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle, Wedge

from ..palettes import Palette, darken, lighten, mix, readable_on
from ..rng import Rng
from ..style import StyleSheet
from .base import FigureSpec, RenderResult, Structure

SUBTYPES = {
    "wordmark": 0.34,
    "icon_grid": 0.22,
    "seal": 0.16,
    "badge": 0.14,
    "sdg_tiles": 0.14,
}

_NAME_PARTS = (
    "Nova", "AValon", "Meri", "Tecto", "Vireo", "Kando", "Arca", "Delune",
    "Orbis", "Silva", "Verta", "Cobalt", "Aureus", "Lumen", "Pyra", "Kestra",
    "Nord", "Baltra", "Cortex", "Vantis", "Solvay", "Merida", "Elyon", "Zephyr",
)
_NAME_SUFFIX = ("Group", "AG", "Industries", "Systems", "Global", "Partners",
                "Holding", "Technologies", "Werke", "SE", "& Co.", "Solutions")
_SEAL_WORDS_DE = ("ZERTIFIZIERT", "GEPRÜFTE QUALITÄT", "SEIT 1924", "ISO 9001",
                  "NACHHALTIG", "MADE IN GERMANY", "AUSGEZEICHNET")
_SEAL_WORDS_EN = ("CERTIFIED", "QUALITY ASSURED", "SINCE 1924", "ISO 9001",
                  "SUSTAINABLE", "AWARD WINNER", "TRUSTED PARTNER")


@dataclass
class _LogoLayout:
    pass


def build_spec(sub_type: str, style: StyleSheet, rng: Rng) -> FigureSpec:
    name = rng.pick(_NAME_PARTS)
    if rng.chance(0.5):
        name = name + rng.pick(("tec", "ex", "on", "va", "co", "is", "ia"))
    suffix = rng.pick(_NAME_SUFFIX) if rng.chance(0.6) else ""
    return FigureSpec(
        label="logo_icon", sub_type=sub_type,
        title=None, subtitle=None, source=None, unit="",
        categories=[], series=[], series_names=[],
        decimals=0, percent=False,
        extra={"name": name, "suffix": suffix},
    )


# --------------------------------------------------------------------------
# Render
# --------------------------------------------------------------------------


def render_logo(spec: FigureSpec, style: StyleSheet, rng: Rng, oversample: float) -> RenderResult:
    from ..engines import mpl

    pal = style.palette
    fig, ax = mpl.new_figure(style, oversample)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.axis("off")

    # Logos often sit on white regardless of the sampled background.
    bg = pal.background if rng.chance(0.5) else ("#ffffff" if not pal.dark else pal.background)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    st = spec.sub_type
    if st == "wordmark":
        _wordmark(ax, spec, style, rng, pal, bg)
    elif st == "seal":
        _seal(ax, spec, style, rng, pal, bg)
    elif st == "badge":
        _badge(ax, spec, style, rng, pal, bg)
    elif st == "icon_grid":
        _icon_grid(ax, spec, style, rng, pal, bg)
    else:
        _sdg_tiles(ax, spec, style, rng, pal, bg)

    image = mpl.to_pil(fig)
    fig.clear()
    return RenderResult(image=image, structure=Structure(), meta={"legend": "none", "yaxis": False})


def _fk(style, size, bold=False):
    return {"family": style.font_family, "fontsize": size, "fontweight": "bold" if bold else "normal"}


def _signet(ax, cx, cy, r, color, rng, edge=None):
    """A small abstract geometric mark."""
    kind = rng.pick(("circle", "ring", "diamond", "chevron", "wave", "triangle", "squares", "leaf"))
    ec = edge or "none"
    if kind == "circle":
        ax.add_patch(Circle((cx, cy), r, facecolor=color, edgecolor=ec, linewidth=1.2))
    elif kind == "ring":
        ax.add_patch(Circle((cx, cy), r, facecolor="none", edgecolor=color, linewidth=r * 0.5))
    elif kind == "diamond":
        ax.add_patch(Polygon([(cx, cy + r), (cx + r, cy), (cx, cy - r), (cx - r, cy)],
                     facecolor=color, edgecolor=ec))
    elif kind == "triangle":
        ax.add_patch(Polygon([(cx, cy + r), (cx + r * 0.9, cy - r * 0.7), (cx - r * 0.9, cy - r * 0.7)],
                     facecolor=color, edgecolor=ec))
    elif kind == "chevron":
        for k, sh in enumerate((0, 0.45)):
            c = color if k == 0 else lighten(color, 0.35)
            ax.add_patch(Polygon([(cx - r, cy - r + sh * r), (cx, cy + sh * r),
                                  (cx + r, cy - r + sh * r), (cx, cy - r * 0.4 + sh * r)],
                         facecolor=c, edgecolor="none"))
    elif kind == "wave":
        th = [i / 30 * 2 * math.pi for i in range(31)]
        xs = [cx - r + (i / 30) * 2 * r for i in range(31)]
        ys = [cy + math.sin(t) * r * 0.5 for t in th]
        ax.plot(xs, ys, color=color, linewidth=r * 0.45, solid_capstyle="round")
    elif kind == "squares":
        for dx, dy, c in ((-0.5, 0.5, color), (0.5, 0.5, lighten(color, 0.3)),
                          (-0.5, -0.5, lighten(color, 0.15)), (0.5, -0.5, darken(color, 0.15))):
            ax.add_patch(Rectangle((cx + dx * r - r * 0.42, cy + dy * r - r * 0.42), r * 0.84, r * 0.84,
                         facecolor=c, edgecolor="none"))
    else:  # leaf
        ax.add_patch(Wedge((cx, cy), r, 20, 160, facecolor=color, edgecolor="none"))
        ax.add_patch(Wedge((cx, cy), r, 200, 340, facecolor=lighten(color, 0.3), edgecolor="none"))


def _wordmark(ax, spec, style, rng, pal, bg):
    name = spec.extra["name"]
    suffix = spec.extra["suffix"]
    color = pal.color(0)
    color = color if abs(_lum(color) - _lum(bg)) > 0.25 else pal.accent
    txt_col = color if rng.chance(0.6) else readable_on(bg)

    layout = rng.pick(("left_mark", "top_mark", "text_only", "text_only"))
    if layout == "text_only":
        ax.text(50, 52, name, ha="center", va="center", color=txt_col, **_fk(style, 30, True))
        if suffix:
            ax.text(50, 34, suffix.upper(), ha="center", va="center", color=mix(txt_col, bg, 0.4),
                    **_fk(style, 10))
    elif layout == "top_mark":
        _signet(ax, 50, 66, 15, color, rng)
        ax.text(50, 34, name, ha="center", va="center", color=txt_col, **_fk(style, 24, True))
        if suffix:
            ax.text(50, 20, suffix, ha="center", va="center", color=mix(txt_col, bg, 0.4), **_fk(style, 9))
    else:
        _signet(ax, 22, 50, 14, color, rng)
        ax.text(40, 50, name, ha="left", va="center", color=txt_col, **_fk(style, 24, True))
        if suffix:
            ax.text(40, 36, suffix, ha="left", va="center", color=mix(txt_col, bg, 0.4), **_fk(style, 9))


def _seal(ax, spec, style, rng, pal, bg):
    color = pal.color(0) if abs(_lum(pal.color(0)) - _lum(bg)) > 0.25 else pal.accent
    R = 40
    ax.add_patch(Circle((50, 50), R, facecolor="none", edgecolor=color, linewidth=2.4))
    ax.add_patch(Circle((50, 50), R - 5, facecolor="none", edgecolor=color, linewidth=1.0))
    if rng.chance(0.5):
        # star wreath
        for k in range(12):
            a = k / 12 * 2 * math.pi
            ax.plot([50 + (R - 2.5) * math.cos(a)], [50 + (R - 2.5) * math.sin(a)],
                    marker="*", markersize=3.5, color=color)
    # circular top text
    word = rng.pick(_SEAL_WORDS_DE if style.language == "de" else _SEAL_WORDS_EN)
    _circular_text(ax, word, 50, 50, R - 10, style, color, top=True)
    year = str(rng.randint(1890, 2010))
    # Centre: a year or a two-letter monogram. Star/check glyphs are missing in
    # too many of the sampled fonts and render as a tofu box.
    monogram = (spec.extra["name"][:2]).upper()
    ax.text(50, 50, rng.pick((year, monogram, monogram)), ha="center", va="center",
            color=color, **_fk(style, 16, True))
    _circular_text(ax, rng.pick(("QUALITY", "GEPRÜFT", "CERTIFIED", "PREMIUM")),
                   50, 50, R - 10, style, color, top=False)


def _badge(ax, spec, style, rng, pal, bg):
    color = pal.color(0) if abs(_lum(pal.color(0)) - _lum(bg)) > 0.25 else pal.accent
    # laurel wreath: two arcs of small leaves
    for side in (-1, 1):
        for k in range(9):
            a = math.radians(250 + side * (k * 9 + 5)) if side < 0 else math.radians(290 - (k * 9 + 5))
            lx, ly = 50 + 30 * math.cos(a), 50 + 30 * math.sin(a)
            ax.add_patch(Wedge((lx, ly), 4, math.degrees(a) + 60, math.degrees(a) + 160,
                         facecolor=lighten(color, 0.15), edgecolor="none"))
    ax.add_patch(Circle((50, 55), 16, facecolor=color, edgecolor="none"))
    ax.text(50, 55, rng.pick(("#1", "TOP", "A+", "Nr.1", "01")), ha="center", va="center",
            color=readable_on(color), **_fk(style, 14, True))
    # ribbon
    for dx in (-6, 6):
        ax.add_patch(Polygon([(50 + dx * 1.2, 40), (50 + dx * 2.2, 18), (50 + dx * 0.7, 24), (50, 36)],
                     facecolor=darken(color, 0.1), edgecolor="none"))
    year = str(rng.randint(2018, 2025))
    ax.text(50, 55 - 22, year, ha="center", va="center", color=readable_on(color) if False else color,
            **_fk(style, 8, True))


def _icon_grid(ax, spec, style, rng, pal, bg):
    cols = rng.randint(2, 3)
    rows = rng.randint(2, 3)
    colors = [pal.color(i) for i in range(max(cols * rows, 1))]
    single = rng.chance(0.4)
    base = pal.color(0) if abs(_lum(pal.color(0)) - _lum(bg)) > 0.25 else pal.accent
    mx, my = 100 / (cols + 0.5), 100 / (rows + 0.5)
    for r in range(rows):
        for c in range(cols):
            cx = (c + 0.75) * mx + mx * 0.1
            cy = 100 - (r + 0.75) * my - my * 0.1
            col = base if single else (colors[(r * cols + c) % len(colors)]
                                       if abs(_lum(colors[(r * cols + c) % len(colors)]) - _lum(bg)) > 0.2 else base)
            _pictogram(ax, cx, cy, min(mx, my) * 0.32, col, rng)


def _pictogram(ax, cx, cy, r, color, rng):
    kind = rng.pick(("person", "leaf", "gear", "globe", "bolt", "drop", "shield", "chart", "handshake", "recycle"))
    lw = r * 0.28
    if kind == "person":
        ax.add_patch(Circle((cx, cy + r * 0.5), r * 0.35, facecolor=color, edgecolor="none"))
        ax.add_patch(Wedge((cx, cy - r * 0.35), r * 0.7, 20, 160, facecolor=color, edgecolor="none"))
    elif kind == "leaf":
        ax.add_patch(Wedge((cx, cy), r, 25, 200, facecolor=color, edgecolor="none"))
    elif kind == "gear":
        ax.add_patch(Circle((cx, cy), r * 0.6, facecolor="none", edgecolor=color, linewidth=lw))
        for k in range(8):
            a = k / 8 * 2 * math.pi
            ax.plot([cx + r * 0.6 * math.cos(a), cx + r * math.cos(a)],
                    [cy + r * 0.6 * math.sin(a), cy + r * math.sin(a)], color=color, linewidth=lw)
    elif kind == "globe":
        ax.add_patch(Circle((cx, cy), r, facecolor="none", edgecolor=color, linewidth=lw))
        ax.plot([cx - r, cx + r], [cy, cy], color=color, linewidth=lw * 0.7)
        ax.add_patch(Wedge((cx, cy), r, 90, 270, facecolor="none", edgecolor=color, linewidth=lw * 0.7, width=0.01))
    elif kind == "bolt":
        ax.add_patch(Polygon([(cx - r * 0.3, cy + r), (cx + r * 0.4, cy + r * 0.1),
                              (cx, cy + r * 0.1), (cx + r * 0.3, cy - r),
                              (cx - r * 0.4, cy - r * 0.1), (cx, cy - r * 0.1)],
                     facecolor=color, edgecolor="none"))
    elif kind == "drop":
        ax.add_patch(Polygon([(cx, cy + r)] + [(cx + r * math.cos(a), cy - r * 0.3 + r * 0.7 * math.sin(a))
                     for a in [math.radians(d) for d in range(-40, 221, 20)]], facecolor=color, edgecolor="none"))
    elif kind == "shield":
        ax.add_patch(Polygon([(cx, cy + r), (cx + r * 0.8, cy + r * 0.4), (cx + r * 0.7, cy - r * 0.6),
                              (cx, cy - r), (cx - r * 0.7, cy - r * 0.6), (cx - r * 0.8, cy + r * 0.4)],
                     facecolor="none", edgecolor=color, linewidth=lw))
    elif kind == "chart":
        for k, h in enumerate((0.5, 0.9, 0.7)):
            ax.add_patch(Rectangle((cx - r * 0.7 + k * r * 0.55, cy - r), r * 0.4, r * (0.5 + h),
                         facecolor=color, edgecolor="none"))
    elif kind == "handshake":
        ax.add_patch(Wedge((cx, cy), r, 200, 340, facecolor=color, edgecolor="none"))
        ax.add_patch(Wedge((cx, cy), r * 0.9, 20, 160, facecolor=lighten(color, 0.3), edgecolor="none"))
    else:  # recycle
        for k in range(3):
            a = math.radians(90 + k * 120)
            ax.add_patch(Wedge((cx, cy), r, math.degrees(a) - 30, math.degrees(a) + 30,
                         facecolor=color, edgecolor="none", width=r * 0.4))


def _sdg_tiles(ax, spec, style, rng, pal, bg):
    # Original numbered coloured tiles -- NOT the UN asset set.
    cols = rng.randint(2, 4)
    rows = rng.randint(1, 3)
    tile_colors = ["#3d6b3f", "#c25a3c", "#2f7d9e", "#c99a2e", "#8f3f6b", "#4a5aa8",
                   "#5a8a3c", "#b03a4a", "#2f9e8a", "#8a6a3c", "#6a4aa0", "#3c8a6b"]
    tile_colors = rng.shuffled(tile_colors)
    gap = 2.0
    tw = (100 - gap * (cols + 1)) / cols
    th = (100 - gap * (rows + 1)) / rows
    icons_kind = ("leaf", "drop", "bolt", "gear", "globe", "person", "recycle", "shield")
    n = 0
    for r in range(rows):
        for c in range(cols):
            x0 = gap + c * (tw + gap)
            y0 = 100 - gap - (r + 1) * th - r * gap
            col = tile_colors[(r * cols + c) % len(tile_colors)]
            ax.add_patch(Rectangle((x0, y0), tw, th, facecolor=col, edgecolor="none"))
            n += 1
            ax.text(x0 + tw * 0.14, y0 + th * 0.78, str(n), ha="left", va="center",
                    color="#ffffff", **_fk(style, min(tw, th) * 0.34, True))
            _pictogram(ax, x0 + tw * 0.5, y0 + th * 0.36, min(tw, th) * 0.20, "#ffffff", rng)


def _circular_text(ax, text, cx, cy, r, style, color, top):
    n = len(text)
    span = math.radians(150)
    start = math.radians(90) + span / 2 if top else math.radians(270) - span / 2
    for i, ch in enumerate(text):
        a = start - (span * i / max(n - 1, 1)) if top else start + (span * i / max(n - 1, 1))
        rot = math.degrees(a) - 90 if top else math.degrees(a) + 90
        ax.text(cx + r * math.cos(a), cy + r * math.sin(a), ch, ha="center", va="center",
                rotation=rot, color=color, **_fk(style, 7, True))


def _lum(c: str) -> float:
    from ..palettes import luminance
    return luminance(c)
