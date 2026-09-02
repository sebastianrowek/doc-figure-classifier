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
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

from ..engines import mpl
from ..palettes import lighten, mix, ramp, readable_on
from ..rng import Rng
from ..style import StyleSheet
from .base import FigureSpec, RenderResult, Structure

SUBTYPES = {
    "org_chart": 0.28,       # hierarchy: a root box over child boxes
    "process_flow": 0.24,    # chevrons left to right
    "process_cycle": 0.10,   # circular arrangement with arrows
    "value_chain": 0.16,     # a segmented right-pointing arrow (Porter style)
    "swimlane": 0.12,        # boxes hopping between labelled horizontal lanes
    "funnel": 0.10,          # stacked, narrowing stages
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

# Value-chain activities (Porter): primary stages, left to right.
_VALUE_DE = ("Beschaffung", "Produktion", "Logistik", "Marketing", "Vertrieb", "Service")
_VALUE_EN = ("Inbound", "Operations", "Outbound", "Marketing", "Sales", "Service")

# Funnel stages, widest first.
_FUNNEL_DE = ("Besucher", "Leads", "Angebote", "Abschlüsse", "Kunden")
_FUNNEL_EN = ("Visitors", "Leads", "Offers", "Deals", "Customers")


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


def _value_chain(fig, ax, style, rng) -> tuple[int, int]:
    """A segmented right-pointing arrow -- the Porter value-chain look."""
    pal = style.palette
    _canvas(fig, ax, pal)
    labels = rng.sample(list(_VALUE_DE if style.language == "de" else _VALUE_EN), rng.randint(4, 6))
    n = len(labels)
    cols = ramp(pal, n)
    y, h = 40, 22
    x0, x1, tip = 5, 95, 8.0
    seg_w = (x1 - tip - x0) / n
    for i, lb in enumerate(labels):
        xa = x0 + i * seg_w
        xb = xa + seg_w
        if i < n - 1:
            pts = [(xa, y), (xb, y), (xb, y + h), (xa, y + h)]
        else:  # last segment carries the arrow tip
            pts = [(xa, y), (xb, y), (x1, y + h / 2), (xb, y + h), (xa, y + h)]
        ax.add_patch(Polygon(pts, facecolor=cols[i], edgecolor=pal.background, linewidth=1.2))
        ax.text((xa + xb) / 2, y + h / 2, lb, ha="center", va="center",
                color=readable_on(cols[i]), **_fk(style, min(8.0, seg_w * 0.44), True))
    heading = "Wertschöpfungskette" if style.language == "de" else "Value chain"
    ax.text(50, 74, heading, ha="center", va="center", color=pal.text, **_fk(style, 11, True))
    return n, n - 1


def _swimlane(fig, ax, style, rng) -> tuple[int, int]:
    """Boxes stepping between labelled horizontal lanes, joined by arrows."""
    pal = style.palette
    _canvas(fig, ax, pal)
    lanes = rng.sample(list(_DEPT_DE if style.language == "de" else _DEPT_EN), rng.randint(2, 3))
    nl = len(lanes)
    top, bot = 88, 12
    lh = (top - bot) / nl
    tab = mix(pal.color(0), pal.background, 0.30 if not pal.dark else 0.0)
    for r, lane in enumerate(lanes):
        ly = bot + (nl - 1 - r) * lh  # first lane on top
        band = mix(pal.color(0), pal.background, 0.10 if r % 2 == 0 else 0.03)
        ax.add_patch(Rectangle((14, ly), 82, lh, facecolor=band, edgecolor=pal.background, linewidth=1.0))
        ax.add_patch(Rectangle((4, ly), 10, lh, facecolor=tab, edgecolor="none"))
        ax.text(9, ly + lh / 2, lane, ha="center", va="center", rotation=90,
                color=readable_on(tab), **_fk(style, 6.5, True))
    steps = rng.sample(list(_STEPS_DE if style.language == "de" else _STEPS_EN), rng.randint(3, 5))
    ns = len(steps)
    bw, bh = 15, min(lh * 0.5, 10)
    xs = np.linspace(20, 90 - bw, ns)
    lane_of = [rng.randint(0, nl - 1) for _ in range(ns)]
    centers = []
    for step, li, x in zip(steps, lane_of, xs):
        cy = bot + (nl - 1 - li) * lh + lh / 2
        fc = mix(pal.color(0), pal.background, 0.25 if not pal.dark else 0.0)
        _box(ax, x, cy - bh / 2, bw, bh, fc, pal.color(0), style, step, readable_on(fc), 6.5)
        centers.append((x, x + bw, cy))  # (left, right, cy)
    for i in range(ns - 1):
        ax.add_patch(FancyArrowPatch(
            (centers[i][1], centers[i][2]), (centers[i + 1][0], centers[i + 1][2]),
            arrowstyle="-|>", mutation_scale=8, color=pal.muted, linewidth=1.1))
    return ns, ns - 1


def _funnel(fig, ax, style, rng) -> tuple[int, int]:
    """Stacked, narrowing stages -- a conversion / sales funnel."""
    pal = style.palette
    _canvas(fig, ax, pal)
    labels = rng.sample(list(_FUNNEL_DE if style.language == "de" else _FUNNEL_EN), rng.randint(3, 5))
    n = len(labels)
    cols = ramp(pal, n)
    top, bot = 86, 14
    sh = (top - bot) / n
    wmax, wmin, cx = 76, 24, 50
    for i, lb in enumerate(labels):
        yt = top - i * sh
        yb = yt - sh * 0.84
        wt = wmax - (wmax - wmin) * i / n
        wb = wmax - (wmax - wmin) * (i + 1) / n
        pts = [(cx - wt / 2, yt), (cx + wt / 2, yt), (cx + wb / 2, yb), (cx - wb / 2, yb)]
        ax.add_patch(Polygon(pts, facecolor=cols[i], edgecolor=pal.background, linewidth=1.4))
        ax.text(cx, (yt + yb) / 2, lb, ha="center", va="center",
                color=readable_on(cols[i]), **_fk(style, 8.0, True))
    return n, n - 1


def render(spec: FigureSpec, style: StyleSheet, rng: Rng, oversample: float) -> RenderResult:
    pal = style.palette
    fig, ax = mpl.new_figure(style, oversample)
    st = spec.sub_type

    if st == "org_chart":
        nodes, edges = _org_chart(fig, ax, style, rng)
    elif st == "value_chain":
        nodes, edges = _value_chain(fig, ax, style, rng)
    elif st == "swimlane":
        nodes, edges = _swimlane(fig, ax, style, rng)
    elif st == "funnel":
        nodes, edges = _funnel(fig, ax, style, rng)
    else:
        nodes, edges = _process(fig, ax, style, rng, cycle=(st == "process_cycle"))

    image = mpl.to_pil(fig)
    fig.clear()

    structure = Structure(flow_nodes=nodes, flow_edges=edges)
    meta = {"sub_kind": st, "nodes": nodes, "edges": edges, "legend": "none", "yaxis": False}
    return RenderResult(image=image, structure=structure, meta=meta)
