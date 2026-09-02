"""
Scatter plots: the ``scatter`` tier-1 class.

Promoted out of ``other`` in taxonomy v1.2. A scatter plot is points in an x/y
system with no line connecting them; the class exists to keep them from leaking
into ``line`` (guide, decision-tree step 12), which is why the hard sub-type is
``trend_line`` -- a scatter carrying a dashed regression line, the case that
looks most like a line chart and must not be labelled one. Bubble charts
(scatter with a size encoding) live here too, per the guide.

Real corporate scatters are sparser and heavier than a matplotlib default cloud:
few, thick markers, varied symbols, thicker axes, and (in positioning /
materiality charts) a category word printed next to every point. The ``labeled``
sub-type reproduces that; it is deliberately capped at <=10 points so the
captions -- which are placed without a measured collision pass -- stay apart.
"""

from __future__ import annotations

import numpy as np

from .. import content
from ..engines import mpl
from ..palettes import contrast_on, ramp
from ..rng import Rng
from ..style import StyleSheet
from .base import FigureSpec, RenderResult, Structure

SUBTYPES = {
    "single": 0.28,      # one cloud of points
    "multi": 0.22,       # two or three colour-coded series
    "labeled": 0.20,     # few points, each captioned (positioning / materiality)
    "trend_line": 0.12,  # a cloud plus a dashed regression line (the line look-alike)
    "bubble": 0.18,      # points with a size encoding
}

# Corporate scatters rarely use plain dots; vary the mark.
_MARKERS = ("o", "s", "D", "^", "v", "P", "X", "*")


def build_spec(sub_type: str, style: StyleSheet, rng: Rng) -> FigureSpec:
    # Points are generated at render time (like the other diagram renderers);
    # the spec only needs to carry the sub-type.
    return FigureSpec(
        label="scatter", sub_type=sub_type,
        title=None, subtitle=None, source=None, unit="",
        categories=[], series=[], series_names=[], decimals=0, percent=False, extra={},
    )


def _plain_axes(fig, ax, style, pal) -> None:
    fig.subplots_adjust(left=0.12, right=0.95, top=0.9, bottom=0.12)
    ax.set_facecolor(pal.background)
    fig.patch.set_facecolor(pal.background)
    # Thicker spines than the matplotlib default -- corporate scatters draw a
    # firm frame, and thin lines vanish once the crop is downsampled and JPEGed.
    for s in ax.spines.values():
        s.set_color(pal.muted)
        s.set_linewidth(1.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=pal.muted, labelsize=style.tick_pt * 0.85, width=1.1, length=4)
    for lb in ax.get_xticklabels() + ax.get_yticklabels():
        lb.set_fontfamily(style.font_family)


def render(spec: FigureSpec, style: StyleSheet, rng: Rng, oversample: float) -> RenderResult:
    pal = style.palette
    fig, ax = mpl.new_figure(style, oversample)
    _plain_axes(fig, ax, style, pal)
    st = spec.sub_type

    if st == "bubble":
        n = rng.randint(5, 12)
        xs = [rng.uniform(0.06, 0.94) for _ in range(n)]
        ys = [rng.uniform(0.06, 0.94) for _ in range(n)]
        sz = [rng.uniform(120, 900) for _ in range(n)]
        cols = ramp(pal, min(n, 5))
        ax.scatter(xs, ys, s=sz, c=[cols[i % len(cols)] for i in range(n)],
                   alpha=0.6, edgecolors=pal.background, linewidths=0.8)
        points = n

    elif st == "labeled":
        # Few points, every one captioned with a category word -- the corporate
        # positioning / materiality look. Points are pulled off the frame edge so
        # a caption never runs outside the axes.
        names = content.categories(
            rng, "segment" if rng.chance(0.5) else "region", rng.randint(5, 10), style.language
        )
        n = len(names)
        marker = rng.pick(_MARKERS)
        col = contrast_on(pal.color(0), pal.background)
        gen = np.random.default_rng(rng.randint(0, 10**6))
        xs = np.clip(gen.uniform(0.14, 0.86, n), 0.0, 1.0)
        ys = np.clip(gen.uniform(0.16, 0.84, n), 0.0, 1.0)
        ax.scatter(xs, ys, s=rng.uniform(90, 170), marker=marker, color=col,
                   alpha=0.85, edgecolors=pal.background, linewidths=0.8, zorder=3)
        for x, y, name in zip(xs, ys, names):
            right = x > 0.72
            ax.annotate(
                name, (float(x), float(y)), textcoords="offset points",
                xytext=(-5 if right else 5, 5), ha="right" if right else "left",
                va="bottom", color=pal.text, **mpl.font_kwargs(style, style.tick_pt * 0.82),
            )
        points = n

    else:  # single / multi / trend_line
        n_series = rng.randint(2, 3) if st == "multi" else 1
        markers = rng.sample(_MARKERS, n_series)
        points = 0
        for si in range(n_series):
            n = rng.randint(6, 12) if st == "multi" else rng.randint(7, 15)
            cx, cy = rng.uniform(0.25, 0.75), rng.uniform(0.25, 0.75)
            gen = np.random.default_rng(rng.randint(0, 10**6))
            xs = np.clip(gen.normal(cx, 0.16, n), 0.02, 0.98)
            ys = np.clip(gen.normal(cy, 0.16, n), 0.02, 0.98)
            ax.scatter(xs, ys, s=rng.uniform(60, 140), marker=markers[si],
                       color=pal.color(si), alpha=0.8, edgecolors=pal.background,
                       linewidths=0.8, zorder=3)
            points += n
        if st == "trend_line":
            # A dashed regression line, NOT a data series through the points --
            # so it is not reported as line_series and stays a scatter. This is
            # the deliberate look-alike that teaches scatter != line.
            ax.plot([0, 1], [rng.uniform(0, 0.4), rng.uniform(0.6, 1)],
                    color=pal.muted, linestyle="--", linewidth=1.4)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    image = mpl.to_pil(fig)
    fig.clear()

    structure = Structure(scatter_points=points)
    meta = {"sub_kind": st, "n_points": points, "legend": "none", "yaxis": True}
    return RenderResult(image=image, structure=structure, meta=meta)
