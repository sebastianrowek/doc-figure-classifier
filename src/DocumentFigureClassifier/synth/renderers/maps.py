"""
Maps -- a geographic representation as the dominant element.

Honest constraint: this runs with no network, so there is no GeoJSON to load and
hand-transcribing real country outlines from memory would be worse than useless
(a wrong "Germany" shape teaches the model a lie). The design doc's GeoJSON plan
is therefore replaced by procedural geography that reproduces the *visual
signature* a coarse figure classifier actually keys on:

  * a graticule -- the lat/lon grid that says "map" more than any coastline
  * several irregular, jagged landmasses (radial-noise polygons) rather than the
    clean blobs of a bubble chart
  * choropleth fills, location pins, proportional bubbles, region borders

Corporate maps are heavily stylised anyway -- Germany as its states, a world
outline with office dots, a choropleth by country -- so an abstract-but-jagged
map sits inside that distribution. When a real multi-region GeoJSON becomes
available this module can swap `_landmasses()` for it without touching anything
else.

The invariant (base._check_map) requires the land to cover >= 45 % of the frame;
coverage is computed by shoelace and generation targets ~0.6, comfortably above.

Sub-types
---------
choropleth   countries filled from a sequential ramp by value
pins         light land + location pins, optionally with leader-line labels
bubble       light land + proportional circles by value
outline      borders only / faint fill, the minimalist office-map look
kpi_callout  a map with a few KPI boxes tied to points by leader lines
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Polygon

from .. import content
from ..engines import mpl
from ..palettes import darken, lighten, mix, readable_on
from ..rng import Rng
from ..style import StyleSheet
from .base import FigureSpec, RenderResult, Structure

SUBTYPES = {
    "choropleth": 0.34,
    "pins": 0.24,
    "bubble": 0.18,
    "outline": 0.12,
    "kpi_callout": 0.12,
}

# A wide pool on purpose. A classifier keys on the map imagery, but heading text
# recurs as a *visual* pattern, so a handful of titles all sharing "Region" /
# "Land" would hand the model a shortcut that then fails on real maps without
# those words. Roughly half of these carry no geographic word at all, and the
# plain KPI framings ("Umsatz", "Mitarbeiter") deliberately overlap with the
# bar/line/pie topic pool so no single heading word can mark a crop as a map.
_MAP_TITLES_DE = ("Standorte weltweit", "Umsatz nach Ländern", "Vertriebsregionen",
                  "Produktionsnetzwerk", "Marktpräsenz", "Umsatz nach Region",
                  "Mitarbeiter nach Ländern", "Globale Präsenz", "Internationale Präsenz",
                  "Konzernstandorte", "Absatz nach Kontinenten", "Wachstumsmärkte",
                  "Unsere Werke", "Produktionsstandorte", "Forschungsstandorte",
                  "Vertriebsnetz", "Umsatz nach Absatzmärkten", "Kernmärkte",
                  "Weltweite Präsenz", "Absatzmärkte", "Umsatz", "Mitarbeiter")
_MAP_TITLES_EN = ("Locations worldwide", "Revenue by country", "Sales regions",
                  "Production network", "Market presence", "Revenue by region",
                  "Employees by country", "Global footprint", "International presence",
                  "Group locations", "Sales by continent", "Growth markets",
                  "Our sites", "Production sites", "R&D sites",
                  "Distribution network", "Revenue by market", "Core markets",
                  "Worldwide presence", "Key markets", "Revenue", "Employees")


@dataclass
class _MapLayout:
    title_lines: list
    title_pt: float
    subtitle: str | None
    subtitle_pt: float
    source: str | None
    graticule: bool
    legend: bool


def build_spec(sub_type: str, style: StyleSheet, rng: Rng) -> FigureSpec:
    lang = style.language
    title = rng.pick(_MAP_TITLES_DE if lang == "de" else _MAP_TITLES_EN)
    return FigureSpec(
        label="map", sub_type=sub_type,
        title=title if style.title_mode != "none" else None,
        subtitle=("Geschäftsjahr 2024" if lang == "de" else "financial year 2024")
        if style.title_mode == "title_subtitle" else None,
        source=content.source_note(rng, lang) if style.source_note else None,
        unit="%", categories=[], series=[], series_names=[],
        decimals=0, percent=False, extra={},
    )


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def _blob(cx, cy, r, rng, jag=0.28, n=52):
    """A jagged closed polygon -- a stylised landmass, not a smooth blob."""
    ks = (rng.randint(2, 3), rng.randint(3, 5), rng.randint(5, 8))
    amps = (rng.uniform(0.06, jag), rng.uniform(0.04, jag * 0.7), rng.uniform(0.02, jag * 0.5))
    phs = (rng.uniform(0, 2 * math.pi), rng.uniform(0, 2 * math.pi), rng.uniform(0, 2 * math.pi))
    pts = []
    for i in range(n):
        th = 2 * math.pi * i / n
        rr = r * (1.0 + sum(a * math.sin(k * th + p) for a, k, p in zip(amps, ks, phs)))
        pts.append((cx + rr * math.cos(th), cy + rr * math.sin(th)))
    return pts


def _area(pts):
    s = 0.0
    for i in range(len(pts)):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % len(pts)]
        s += x0 * y1 - x1 * y0
    return abs(s) / 2.0


def _landmasses(rng, x0, y0, x1, y1, big=False):
    """
    Pack jagged countries into the region [x0,x1]x[y0,y1].

    A rough jittered grid keeps them from overlapping too much; small gaps read
    as borders. Returns (polygon_points, centre) pairs.
    """
    w, h = x1 - x0, y1 - y0
    if big:  # one dominant country subdivided-looking cluster
        cols = rng.randint(2, 3)
        rows = rng.randint(2, 3)
    else:
        cols = rng.randint(3, 5)
        rows = rng.randint(2, 4)
    cw, ch = w / cols, h / rows
    out = []
    for r in range(rows):
        for c in range(cols):
            if not big and rng.chance(0.15):
                continue  # a little ocean
            cx = x0 + (c + 0.5) * cw + rng.uniform(-0.12, 0.12) * cw
            cy = y0 + (r + 0.5) * ch + rng.uniform(-0.12, 0.12) * ch
            rad = min(cw, ch) * rng.uniform(0.52, 0.66)
            out.append((_blob(cx, cy, rad, rng), (cx, cy)))
    return out


# --------------------------------------------------------------------------
# Render
# --------------------------------------------------------------------------


def render_map(spec: FigureSpec, style: StyleSheet, rng: Rng, oversample: float) -> RenderResult:
    pal = style.palette
    fig, ax = mpl.new_figure(style, oversample)
    lay = _fit(spec, style, rng)

    top = 5.0 + (lay.title_pt * 1.35 * len(lay.title_lines) + 6.0 if lay.title_lines else 0.0)
    top += lay.subtitle_pt * 1.4 if lay.subtitle else 0.0
    bottom = 6.0 + (style.base_pt * 1.9 if lay.source else 0.0)
    mpl.apply_margins(fig, style, mpl.Margins(left=6.0, right=6.0, top=top, bottom=bottom))

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("auto")
    ax.axis("off")
    fig.patch.set_facecolor(pal.background)
    # A faint "ocean" panel makes the land read as land.
    ocean = mix(pal.color(0), pal.background, 0.90 if not pal.dark else 0.7)
    ax.add_patch(Polygon([(0, 0), (100, 0), (100, 100), (0, 100)], facecolor=ocean, edgecolor="none", zorder=0))
    ax.set_facecolor(ocean)

    if lay.graticule:
        _graticule(ax, pal, rng)

    st = spec.sub_type
    lands = _landmasses(rng, 4, 4, 96, 96, big=(st == "choropleth" and rng.chance(0.4)))
    total_area = sum(_area(p) for p, _ in lands)
    area_frac = min(0.92, total_area / 10000.0)

    if st == "choropleth":
        _choropleth(ax, lands, spec, style, rng, lay)
    elif st == "pins":
        _pins(ax, lands, spec, style, rng)
    elif st == "bubble":
        _bubbles(ax, lands, spec, style, rng)
    elif st == "kpi_callout":
        _callout(ax, lands, spec, style, rng)
    else:
        _outline(ax, lands, spec, style, rng)

    _draw_headings(fig, spec, style, lay)

    image = mpl.to_pil(fig)
    fig.clear()

    structure = Structure(geometry_area_frac=area_frac)
    meta = {"sub_kind": st, "n_regions": len(lands), "area_frac": round(area_frac, 3),
            "graticule": lay.graticule, "legend": lay.legend, "yaxis": False,
            "title_wrapped": len(lay.title_lines) > 1}
    return RenderResult(image=image, structure=structure, meta=meta)


def _graticule(ax, pal, rng):
    col = mix(pal.muted, pal.background, 0.55)
    curved = rng.chance(0.4)
    for gx in range(10, 100, rng.pick((12, 16, 20))):
        if curved:
            ys = np.linspace(2, 98, 20)
            xs = gx + 6 * np.sin((ys - 50) / 50 * math.pi / 2)
            ax.plot(xs, ys, color=col, linewidth=0.4, zorder=1)
        else:
            ax.plot([gx, gx], [2, 98], color=col, linewidth=0.4, zorder=1)
    for gy in range(10, 100, rng.pick((12, 16, 20))):
        ax.plot([2, 98], [gy, gy], color=col, linewidth=0.4, zorder=1)


def _choropleth(ax, lands, spec, style, rng, lay):
    pal = style.palette
    base = pal.color(0) if abs(_lum(pal.color(0)) - _lum(pal.background)) > 0.2 else pal.accent
    vals = [rng.uniform(0, 1) for _ in lands]
    lo, hi = min(vals), max(vals)
    for (pts, ctr), v in zip(lands, vals):
        norm = (v - lo) / (hi - lo + 1e-9)
        fc = lighten(base, 0.75 * (1 - norm))  # high value -> dark
        ax.add_patch(Polygon(pts, facecolor=fc, edgecolor=pal.background, linewidth=1.0, zorder=2))
    # a handful of value labels
    if rng.chance(0.55):
        for (pts, ctr), v in zip(lands, vals):
            if rng.chance(0.5):
                ax.text(ctr[0], ctr[1], content.format_number(round(v * 100), 0, style.number_locale),
                        ha="center", va="center", color=pal.text, **_fk(style, style.tick_pt * 0.8, True))
    if lay.legend:
        _legend_gradient(ax, base, style, lo, hi)


def _pins(ax, lands, spec, style, rng):
    pal = style.palette
    fc = mix(pal.color(0), pal.background, 0.55)
    for pts, ctr in lands:
        ax.add_patch(Polygon(pts, facecolor=fc, edgecolor=mix(pal.muted, pal.background, 0.3),
                     linewidth=0.7, zorder=2))
    pin_col = pal.accent if abs(_lum(pal.accent) - _lum(fc)) > 0.2 else darken(pal.color(0), 0.2)
    labelled = rng.chance(0.4)
    names = content.categories(rng, "region", min(len(lands), 6), style.language)
    for i, (pts, ctr) in enumerate(lands):
        if rng.chance(0.75):
            x, y = ctr[0] + rng.uniform(-6, 6), ctr[1] + rng.uniform(-4, 4)
            # teardrop pin: circle head + small tail
            ax.add_patch(Circle((x, y + 1.5), 2.2, facecolor=pin_col, edgecolor="white", linewidth=0.6, zorder=5))
            ax.add_patch(Polygon([(x - 1.3, y + 1), (x + 1.3, y + 1), (x, y - 2.5)],
                         facecolor=pin_col, edgecolor="none", zorder=4))
            if labelled and i < len(names):
                ax.text(x + 3, y + 2, names[i], ha="left", va="center", color=pal.text,
                        **_fk(style, style.tick_pt * 0.75))


def _bubbles(ax, lands, spec, style, rng):
    pal = style.palette
    fc = mix(pal.color(0), pal.background, 0.6)
    for pts, ctr in lands:
        ax.add_patch(Polygon(pts, facecolor=fc, edgecolor=mix(pal.muted, pal.background, 0.3),
                     linewidth=0.7, zorder=2))
    bcol = pal.accent if abs(_lum(pal.accent) - _lum(fc)) > 0.15 else darken(pal.color(0), 0.15)
    for pts, ctr in lands:
        if rng.chance(0.8):
            rad = rng.uniform(2.5, 8.0)
            ax.add_patch(Circle(ctr, rad, facecolor=bcol, edgecolor="white", linewidth=0.5, alpha=0.65, zorder=5))
            if rng.chance(0.4):
                ax.text(ctr[0], ctr[1], content.format_number(round(rad * rng.uniform(20, 60)), 0, style.number_locale),
                        ha="center", va="center", color="white", **_fk(style, style.tick_pt * 0.7, True))


def _outline(ax, lands, spec, style, rng):
    pal = style.palette
    fc = mix(pal.color(0), pal.background, 0.78) if rng.chance(0.6) else "none"
    ec = pal.color(0) if abs(_lum(pal.color(0)) - _lum(pal.background)) > 0.2 else pal.muted
    for pts, ctr in lands:
        ax.add_patch(Polygon(pts, facecolor=fc, edgecolor=ec, linewidth=1.1, zorder=2))
    if rng.chance(0.4):
        for pts, ctr in lands:
            if rng.chance(0.5):
                ax.add_patch(Circle(ctr, 1.4, facecolor=pal.accent, edgecolor="none", zorder=5))


def _callout(ax, lands, spec, style, rng):
    pal = style.palette
    _pins(ax, lands, spec, style, rng)
    labels_pool = content.categories(rng, "region", 3, style.language)
    n = min(rng.randint(2, 3), len(lands))
    picks = rng.sample(lands, n) if len(lands) >= n else lands
    corners = [(8, 82), (8, 20), (72, 82), (72, 20)]
    corners = rng.shuffled(corners)[:n]
    for (pts, ctr), (bx, by), name in zip(picks, corners, labels_pool):
        ax.plot([ctr[0], bx + 10], [ctr[1], by + 4], color=pal.muted, linewidth=0.6, zorder=4)
        fc = mix(pal.color(0), pal.background, 0.2 if not pal.dark else 0.0)
        ax.add_patch(FancyBboxPatch((bx, by), 20, 9, boxstyle="round,pad=0,rounding_size=1.5",
                     facecolor=fc, edgecolor="none", zorder=6))
        val = content.format_number(rng.randint(120, 3200), 0, style.number_locale)
        ax.text(bx + 10, by + 6, name, ha="center", va="center", color=readable_on(fc),
                **_fk(style, style.tick_pt * 0.72, True))
        ax.text(bx + 10, by + 2.5, val + (" Mio." if style.language == "de" else "m"),
                ha="center", va="center", color=mix(readable_on(fc), fc, 0.3), **_fk(style, style.tick_pt * 0.66))


def _legend_gradient(ax, base, style, lo, hi):
    pal = style.palette
    x, y, w, h = 6, 8, 22, 4
    steps = 24
    for i in range(steps):
        t = i / (steps - 1)
        ax.add_patch(Polygon([(x + w * t, y), (x + w * (t + 1 / steps), y),
                              (x + w * (t + 1 / steps), y + h), (x + w * t, y + h)],
                     facecolor=lighten(base, 0.75 * (1 - t)), edgecolor="none", zorder=6))
    ax.text(x, y + h + 1.5, content.format_number(round(lo * 100), 0, style.number_locale),
            ha="left", va="bottom", color=pal.muted, **_fk(style, style.tick_pt * 0.7))
    ax.text(x + w, y + h + 1.5, content.format_number(round(hi * 100), 0, style.number_locale) + " %",
            ha="right", va="bottom", color=pal.muted, **_fk(style, style.tick_pt * 0.7))


# --------------------------------------------------------------------------


def _fit(spec, style, rng):
    title_lines = [spec.title] if spec.title else []
    return _MapLayout(
        title_lines=title_lines, title_pt=style.title_pt,
        subtitle=spec.subtitle, subtitle_pt=style.base_pt * 0.95,
        source=spec.source, graticule=rng.chance(0.6), legend=rng.chance(0.5),
    )


def _draw_headings(fig, spec, style, lay: _MapLayout):
    pal = style.palette
    x = 0.035 if style.title_align == "left" else 0.5
    ha = "left" if style.title_align == "left" else "center"
    y = 1.0 - mpl.frac_h(5.0, style)
    for line in lay.title_lines:
        fig.text(x, y, line, ha=ha, va="top", color=pal.text,
                 **_fk(style, lay.title_pt, style.bold_title))
        y -= mpl.frac_h(lay.title_pt * 1.3, style)
    if lay.subtitle:
        fig.text(x, y, lay.subtitle, ha=ha, va="top", color=pal.muted, **_fk(style, lay.subtitle_pt))
    if lay.source:
        fig.text(0.035, mpl.frac_h(4.0, style), lay.source, ha="left", va="bottom",
                 color=pal.muted, **_fk(style, style.base_pt * 0.85))


def _fk(style, size, bold=False):
    return {"family": style.font_family, "fontsize": size, "fontweight": "bold" if bold else "normal"}


def _lum(c: str) -> float:
    from ..palettes import luminance
    return luminance(c)
