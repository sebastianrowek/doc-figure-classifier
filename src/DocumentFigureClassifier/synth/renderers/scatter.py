"""
Scatter plots: the ``scatter`` tier-1 class.

Promoted out of ``other`` in taxonomy v1.2. A scatter plot is points in an x/y
system with no line connecting them; the class exists to keep them from leaking
into ``line`` (guide, decision-tree step 12), which is why the hard sub-type is
``trend_line`` -- a scatter carrying a dashed regression line, the case that
looks most like a line chart and must not be labelled one. Bubble charts
(scatter with a size encoding) live here too, per the guide.

The drawing code was lifted from the old other._scatter / other._bubble when the
class was promoted, so the visuals that used to ship as ``other`` now ship as
``scatter``.
"""

from __future__ import annotations

import numpy as np

from ..engines import mpl
from ..palettes import ramp
from ..rng import Rng
from ..style import StyleSheet
from .base import FigureSpec, RenderResult, Structure

SUBTYPES = {
    "single": 0.35,   # one cloud of points
    "multi": 0.30,    # two or three colour-coded series
    "trend_line": 0.15,  # a cloud plus a dashed regression line (the line look-alike)
    "bubble": 0.20,   # points with a size encoding
}


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
    for s in ax.spines.values():
        s.set_color(pal.muted)
        s.set_linewidth(0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=pal.muted, labelsize=style.tick_pt * 0.7)
    for lb in ax.get_xticklabels() + ax.get_yticklabels():
        lb.set_fontfamily(style.font_family)


def render(spec: FigureSpec, style: StyleSheet, rng: Rng, oversample: float) -> RenderResult:
    pal = style.palette
    fig, ax = mpl.new_figure(style, oversample)
    _plain_axes(fig, ax, style, pal)
    st = spec.sub_type

    if st == "bubble":
        n = rng.randint(6, 18)
        xs = [rng.uniform(0, 1) for _ in range(n)]
        ys = [rng.uniform(0, 1) for _ in range(n)]
        sz = [rng.uniform(30, 700) for _ in range(n)]
        cols = ramp(pal, min(n, 5))
        ax.scatter(xs, ys, s=sz, c=[cols[i % len(cols)] for i in range(n)], alpha=0.6, edgecolors="none")
        points = n
    else:
        n_series = rng.randint(2, 3) if st == "multi" else 1
        points = 0
        for si in range(n_series):
            n = rng.randint(12, 40)
            cx, cy = rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)
            gen = np.random.default_rng(rng.randint(0, 10**6))
            xs = np.clip(gen.normal(cx, 0.15, n), 0, 1)
            ys = np.clip(gen.normal(cy, 0.15, n), 0, 1)
            ax.scatter(xs, ys, s=rng.uniform(8, 30), color=pal.color(si), alpha=0.7, edgecolors="none")
            points += n
        if st == "trend_line":
            # A dashed regression line, NOT a data series through the points --
            # so it is not reported as line_series and stays a scatter. This is
            # the deliberate look-alike that teaches scatter != line.
            ax.plot([0, 1], [rng.uniform(0, 0.4), rng.uniform(0.6, 1)],
                    color=pal.muted, linestyle="--", linewidth=1)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    image = mpl.to_pil(fig)
    fig.clear()

    structure = Structure(scatter_points=points)
    meta = {"sub_kind": st, "n_points": points, "legend": "none", "yaxis": True}
    return RenderResult(image=image, structure=structure, meta=meta)
