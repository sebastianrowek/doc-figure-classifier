"""
Flow charts: the ``flow`` tier-1 class -- process/workflow diagrams and org
charts, boxes or nodes joined by arrows.

Promoted out of ``other`` in taxonomy v1.2 (was other/org_chart +
other/process_flow). The base DocumentFigureClassifier model already knows flow
charts, so this is a future parsing target; see the labeling guide.

The drawing code was lifted from the old other._org_chart / other._process_flow
when the class was promoted. Each renderer reports the node/edge counts it drew
so base._check_flow can assert the image really is a connected diagram.
"""

from __future__ import annotations

import math

import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon

from ..engines import mpl
from ..palettes import lighten, mix, ramp, readable_on
from ..rng import Rng
from ..style import StyleSheet
from .base import FigureSpec, RenderResult, Structure

SUBTYPES = {
    "org_chart": 0.45,       # hierarchy: a root box over child boxes
    "process_flow": 0.40,    # chevrons left to right
    "process_cycle": 0.15,   # circular arrangement with arrows
}

# Short department labels for org-chart boxes -- the boxes are narrow, so longer
# names would overrun them.
_DEPT_DE = ("CEO", "CFO", "COO", "Vertrieb", "F&E", "HR", "IT", "Finanz",
            "Einkauf", "Mktg", "Werk A", "Werk B", "Technik", "Recht")
_DEPT_EN = ("CEO", "CFO", "COO", "Sales", "R&D", "HR", "IT", "Finance",
            "Buying", "Mktg", "Plant A", "Plant B", "Legal", "Ops")

# Short single-word step labels; a chevron is too narrow for long names.
_STEPS_DE = ("Analyse", "Konzept", "Planung", "Aufbau", "Betrieb", "Start",
             "Ziel", "Test", "Ausbau", "Review")
_STEPS_EN = ("Analyze", "Concept", "Design", "Build", "Operate", "Start",
             "Goal", "Test", "Scale", "Review")


def build_spec(sub_type: str, style: StyleSheet, rng: Rng) -> FigureSpec:
    return FigureSpec(
        label="flow", sub_type=sub_type,
        title=None, subtitle=None, source=None, unit="",
        categories=[], series=[], series_names=[], decimals=0, percent=False, extra={},
    )


def _fk(style, size, bold=False):
    return {"family": style.font_family, "fontsize": size, "fontweight": "bold" if bold else "normal"}


def _canvas(fig, ax, pal, bg=None):
    bg = bg or pal.background
    fig.subplots_adjust(left=0.03, right=0.97, top=0.95, bottom=0.05)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("auto")
    ax.axis("off")
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    return ax


def _box(ax, x, y, w, h, fc, ec, style, text=None, tc=None, fs=8, bold=False, rounded=True):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=%.1f" % (min(w, h) * 0.12) if rounded else "square,pad=0",
        facecolor=fc, edgecolor=ec, linewidth=1.0, mutation_aspect=1.0,
    )
    ax.add_patch(patch)
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, **_fk(style, fs, bold))


def _org_chart(fig, ax, style, rng) -> tuple[int, int]:
    pal = style.palette
    _canvas(fig, ax, pal)
    root_word = "Vorstand" if style.language == "de" else "Board"
    words = rng.shuffled(list(_DEPT_DE if style.language == "de" else _DEPT_EN))
    words.append(root_word)  # popped first as the root
    fc = mix(pal.color(0), pal.background, 0.25 if not pal.dark else 0.0)
    ec = pal.color(0)
    tc = readable_on(fc)
    n_child = rng.randint(2, 4)
    # Box width has to leave a gap between children: bw*n_child must fit in ~84.
    bw = min(22, 84.0 / n_child - 2.0)
    bh = 11
    fs = 7.5 if bw > 18 else 6.3
    _box(ax, 50 - bw / 2, 80, bw, bh, fc, ec, style, words.pop(), tc, 8, True)
    nodes, edges = 1, 0
    xs = np.linspace(8, 92 - bw, n_child)
    ymid = 55
    for x in xs:
        ax.plot([50, x + bw / 2], [80, ymid + bh], color=ec, linewidth=0.8, zorder=1)
        _box(ax, x, ymid, bw, bh, lighten(fc, 0.1), ec, style, words.pop() if words else "—", tc, fs)
        nodes += 1
        edges += 1
        if rng.chance(0.5) and words:
            _box(ax, x, ymid - 20, bw, bh, lighten(fc, 0.25), ec, style, words.pop(), tc, fs - 0.5)
            ax.plot([x + bw / 2, x + bw / 2], [ymid, ymid - 20 + bh], color=ec, linewidth=0.8, zorder=1)
            nodes += 1
            edges += 1
    return nodes, edges


def _process(fig, ax, style, rng, cycle: bool) -> tuple[int, int]:
    pal = style.palette
    _canvas(fig, ax, pal)
    steps = rng.randint(3, 4)
    labels = rng.sample(list(_STEPS_DE if style.language == "de" else _STEPS_EN), steps)
    cols = ramp(pal, steps)
    if cycle:
        R = 30
        for i in range(steps):
            a = math.radians(90 - i * 360 / steps)
            cx, cy = 50 + R * math.cos(a), 50 + R * math.sin(a)
            ax.add_patch(Circle((cx, cy), 12, facecolor=cols[i], edgecolor="none"))
            ax.text(cx, cy, labels[i], ha="center", va="center", color=readable_on(cols[i]), **_fk(style, 7, True))
            ax.add_patch(FancyArrowPatch(
                (50 + (R + 2) * math.cos(math.radians(90 - i * 360 / steps - 12)),
                 50 + (R + 2) * math.sin(math.radians(90 - i * 360 / steps - 12))),
                (50 + (R + 2) * math.cos(math.radians(90 - (i + 1) * 360 / steps + 12)),
                 50 + (R + 2) * math.sin(math.radians(90 - (i + 1) * 360 / steps + 12))),
                connectionstyle="arc3,rad=0.3", arrowstyle="-|>", mutation_scale=10,
                color=pal.muted, linewidth=1.2))
        return steps, steps  # a closed cycle: one arrow per node
    # chevrons left to right
    w = 92 / steps
    y = 46
    h = 13
    # A number goes inside the chevron; the word goes on its own line below,
    # where it has the full step width to itself instead of the chevron body.
    for i in range(steps):
        x = 4 + i * w
        pts = [(x, y), (x + w * 0.82, y), (x + w * 0.96, y + h / 2), (x + w * 0.82, y + h),
               (x, y + h), (x + w * 0.14, y + h / 2)]
        ax.add_patch(Polygon(pts, facecolor=cols[i], edgecolor="none"))
        ax.text(x + w * 0.42, y + h / 2, str(i + 1), ha="center", va="center",
                color=readable_on(cols[i]), **_fk(style, 9, True))
        ax.text(x + w * 0.45, y - 5, labels[i], ha="center", va="top",
                color=pal.text, **_fk(style, min(6.5, w * 0.30)))
    return steps, steps - 1  # chevrons chain one into the next


def render(spec: FigureSpec, style: StyleSheet, rng: Rng, oversample: float) -> RenderResult:
    pal = style.palette
    fig, ax = mpl.new_figure(style, oversample)
    st = spec.sub_type

    if st == "org_chart":
        nodes, edges = _org_chart(fig, ax, style, rng)
    else:
        nodes, edges = _process(fig, ax, style, rng, cycle=(st == "process_cycle"))

    image = mpl.to_pil(fig)
    fig.clear()

    structure = Structure(flow_nodes=nodes, flow_edges=edges)
    meta = {"sub_kind": st, "nodes": nodes, "edges": edges, "legend": "none", "yaxis": False}
    return RenderResult(image=image, structure=structure, meta=meta)
