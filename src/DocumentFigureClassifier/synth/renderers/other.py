"""
The catch-all class -- everything that is not one of the eleven chart types.

`other` is internally the most heterogeneous class and gets the most
sub-generators. Its invariant (base._check_other) is deliberately open: the
whole point is that anything lands here. What matters is coverage of the shapes
the extraction pipeline actually produces, especially the two that get forgotten
and then hurt the model in production:

* blank_artifact -- near-empty crops, half a letter, an edge strip. The pipeline
  emits these constantly; without them the model labels extraction junk as a
  confident chart.
* infographic_frame / multi_chart -- rules R2 and R1. A chart taking less than
  half the frame, or two charts that cannot be cut apart, are `other`.

The chart-shaped sub-types (scatter, radar, bubble, tornado, boxplot) are real
plots that are simply not tier-1 classes; the sankey/org/process/timeline/matrix
group are diagrams; kpi_tile and gauge_progress are the dashboard furniture the
guide lists.
"""

from __future__ import annotations

import math

import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch, Polygon, Rectangle, Wedge
from matplotlib.path import Path as MplPath

from .. import content, series
from ..engines import mpl
from ..palettes import darken, lighten, mix, ramp, readable_on
from ..rng import Rng
from ..style import StyleSheet
from .base import FigureSpec, RenderResult, Structure

SUBTYPES = {
    "org_chart": 0.09,
    "process_flow": 0.09,
    "timeline": 0.08,
    "matrix": 0.07,
    "kpi_tile": 0.09,
    "gauge_progress": 0.08,
    "sankey": 0.05,
    "radar": 0.05,
    "bubble": 0.05,
    "tornado": 0.05,
    "scatter": 0.05,
    "boxplot": 0.05,
    "decorative": 0.06,
    "blank_artifact": 0.09,
    "multi_chart": 0.03,
    "infographic_frame": 0.03,
    # Named hard variants from the guide / design doc §5.
    "table_with_bars": 0.05,
    "donut_progress": 0.04,
}


def build_spec(sub_type: str, style: StyleSheet, rng: Rng) -> FigureSpec:
    return FigureSpec(
        label="other", sub_type=sub_type,
        title=None, subtitle=None, source=None, unit="",
        categories=[], series=[], series_names=[], decimals=0, percent=False, extra={},
    )


def _fk(style, size, bold=False):
    return {"family": style.font_family, "fontsize": size, "fontweight": "bold" if bold else "normal"}


def render_other(spec: FigureSpec, style: StyleSheet, rng: Rng, oversample: float) -> RenderResult:
    pal = style.palette
    fig, ax = mpl.new_figure(style, oversample)
    st = spec.sub_type

    fn = _DISPATCH[st]
    meta = fn(fig, ax, style, rng) or {}

    image = mpl.to_pil(fig)
    fig.clear()
    meta.setdefault("legend", "none")
    meta.setdefault("yaxis", False)
    return RenderResult(image=image, structure=Structure(), meta={"sub_kind": st, **meta})


# --------------------------------------------------------------------------
# Canvas helper for diagram-style sub-types
# --------------------------------------------------------------------------


def _canvas(fig, ax, pal, rng, bg=None):
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
    patch = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0,rounding_size=%.1f" % (min(w, h) * 0.12) if rounded else "square,pad=0",
                           facecolor=fc, edgecolor=ec, linewidth=1.0, mutation_aspect=1.0)
    ax.add_patch(patch)
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                color=tc, **_fk(style, fs, bold))


# --------------------------------------------------------------------------
# Diagrams
# --------------------------------------------------------------------------

_ORG_DE = ("Vorstand", "CEO", "CFO", "COO", "Vertrieb", "Produktion", "F&E",
           "Personal", "Finanzen", "IT", "Einkauf", "Marketing", "Region Nord",
           "Region Süd", "Werk A", "Werk B", "Tochter GmbH")
_ORG_EN = ("Board", "CEO", "CFO", "COO", "Sales", "Production", "R&D", "HR",
           "Finance", "IT", "Procurement", "Marketing", "Region North",
           "Region South", "Plant A", "Plant B", "Subsidiary Ltd")

# Short single-word step labels; a chevron is too narrow for "Tochter GmbH".
_STEPS_DE = ("Analyse", "Konzept", "Planung", "Aufbau", "Betrieb", "Start",
             "Ziel", "Test", "Ausbau", "Review")
_STEPS_EN = ("Analyze", "Concept", "Design", "Build", "Operate", "Start",
             "Goal", "Test", "Scale", "Review")

# Short department labels for org-chart boxes -- the boxes are narrow, so the
# long entries in _ORG_* ("Subsidiary Ltd", "Region North") would overrun them.
_DEPT_DE = ("CEO", "CFO", "COO", "Vertrieb", "F&E", "HR", "IT", "Finanz",
            "Einkauf", "Mktg", "Werk A", "Werk B", "Technik", "Recht")
_DEPT_EN = ("CEO", "CFO", "COO", "Sales", "R&D", "HR", "IT", "Finance",
            "Buying", "Mktg", "Plant A", "Plant B", "Legal", "Ops")


def _org_chart(fig, ax, style, rng):
    pal = style.palette
    _canvas(fig, ax, pal, rng)
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
    xs = np.linspace(8, 92 - bw, n_child)
    ymid = 55
    for i, x in enumerate(xs):
        ax.plot([50, x + bw / 2], [80, ymid + bh], color=ec, linewidth=0.8, zorder=1)
        _box(ax, x, ymid, bw, bh, lighten(fc, 0.1), ec, style, words.pop() if words else "—", tc, fs)
        if rng.chance(0.5) and words:
            _box(ax, x, ymid - 20, bw, bh, lighten(fc, 0.25), ec, style, words.pop(), tc, fs - 0.5)
            ax.plot([x + bw / 2, x + bw / 2], [ymid, ymid - 20 + bh], color=ec, linewidth=0.8, zorder=1)
    return {"sub_kind": "org_chart"}


def _process_flow(fig, ax, style, rng):
    pal = style.palette
    _canvas(fig, ax, pal, rng)
    steps = rng.randint(3, 4)
    labels = rng.sample(list(_STEPS_DE if style.language == "de" else _STEPS_EN), steps)
    cycle = rng.chance(0.3)
    cols = ramp(pal, steps)
    if cycle:
        R = 30
        for i in range(steps):
            a = math.radians(90 - i * 360 / steps)
            cx, cy = 50 + R * math.cos(a), 50 + R * math.sin(a)
            ax.add_patch(Circle((cx, cy), 12, facecolor=cols[i], edgecolor="none"))
            ax.text(cx, cy, labels[i], ha="center", va="center", color=readable_on(cols[i]), **_fk(style, 7, True))
            a2 = math.radians(90 - (i + 0.5) * 360 / steps)
            ax.add_patch(FancyArrowPatch((50 + (R + 2) * math.cos(math.radians(90 - i * 360 / steps - 12)),
                                          50 + (R + 2) * math.sin(math.radians(90 - i * 360 / steps - 12))),
                                         (50 + (R + 2) * math.cos(math.radians(90 - (i + 1) * 360 / steps + 12)),
                                          50 + (R + 2) * math.sin(math.radians(90 - (i + 1) * 360 / steps + 12))),
                                         connectionstyle="arc3,rad=0.3", arrowstyle="-|>", mutation_scale=10,
                                         color=pal.muted, linewidth=1.2))
    else:
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
    return {"sub_kind": "process_flow"}


def _timeline(fig, ax, style, rng):
    pal = style.palette
    _canvas(fig, ax, pal, rng)
    n = rng.randint(3, 6)
    y0 = rng.randint(2015, 2020)
    vertical = rng.chance(0.3)
    col = pal.color(0) if abs(_lum(pal.color(0)) - _lum(pal.background)) > 0.25 else pal.accent
    labels = rng.sample(list(_ORG_DE if style.language == "de" else _ORG_EN), n)
    if vertical:
        ax.plot([25, 25], [8, 92], color=col, linewidth=2)
        ys = np.linspace(85, 15, n)
        for i, y in enumerate(ys):
            ax.add_patch(Circle((25, y), 2.4, facecolor=col, edgecolor="none"))
            ax.text(20, y, str(y0 + i), ha="right", va="center", color=col, **_fk(style, 8, True))
            ax.text(30, y, labels[i], ha="left", va="center", color=pal.text, **_fk(style, 7.5))
    else:
        ax.plot([6, 94], [50, 50], color=col, linewidth=2)
        xs = np.linspace(12, 88, n)
        for i, x in enumerate(xs):
            up = i % 2 == 0
            ax.add_patch(Circle((x, 50), 2.4, facecolor=col, edgecolor="none"))
            yy = 62 if up else 38
            ax.plot([x, x], [50, yy], color=pal.muted, linewidth=0.6)
            ax.text(x, yy + (4 if up else -4), str(y0 + i), ha="center",
                    va="bottom" if up else "top", color=col, **_fk(style, 8, True))
            ax.text(x, yy + (10 if up else -10), labels[i], ha="center",
                    va="bottom" if up else "top", color=pal.text, **_fk(style, 6.5))
    return {"sub_kind": "timeline"}


def _matrix(fig, ax, style, rng):
    pal = style.palette
    _canvas(fig, ax, pal, rng)
    n = rng.pick((2, 2, 3))
    lo, hi = 12, 88
    step = (hi - lo) / n
    # quadrant fills
    for r in range(n):
        for c in range(n):
            t = (r + c) / (2 * (n - 1)) if n > 1 else 0.5
            fc = mix(lighten(pal.color(0), 0.6), pal.negative if pal.negative else pal.color(0), t * 0.5)
            ax.add_patch(Rectangle((lo + c * step, lo + r * step), step, step,
                         facecolor=mix(fc, pal.background, 0.4), edgecolor=pal.background, linewidth=1))
    ax.add_patch(Rectangle((lo, lo), hi - lo, hi - lo, facecolor="none", edgecolor=pal.muted, linewidth=1))
    # scattered dots (materiality style)
    if rng.chance(0.6):
        for _ in range(rng.randint(4, 9)):
            ax.add_patch(Circle((rng.uniform(lo + 3, hi - 3), rng.uniform(lo + 3, hi - 3)),
                         rng.uniform(1.5, 3.5), facecolor=pal.accent, edgecolor="none", alpha=0.85))
    ax_lbl = ("gering", "hoch") if style.language == "de" else ("low", "high")
    ax.annotate("", (hi + 2, lo), (lo, lo), arrowprops=dict(arrowstyle="->", color=pal.muted))
    ax.annotate("", (lo, hi + 2), (lo, lo), arrowprops=dict(arrowstyle="->", color=pal.muted))
    ax.text(hi, lo - 4, ax_lbl[1], ha="right", va="top", color=pal.muted, **_fk(style, 7))
    ax.text(lo - 3, hi, ax_lbl[1], ha="right", va="top", rotation=90, color=pal.muted, **_fk(style, 7))
    return {"sub_kind": "matrix"}


def _kpi_tile(fig, ax, style, rng):
    pal = style.palette
    bg = pal.background
    _canvas(fig, ax, pal, rng, bg)
    n = rng.randint(1, 3)
    topics = rng.sample(range(len(_KPI_LABELS_DE)), n)
    labels_pool = _KPI_LABELS_DE if style.language == "de" else _KPI_LABELS_EN
    gap = 3
    tw = (100 - gap * (n + 1)) / n
    for i, ti in enumerate(topics):
        x0 = gap + i * (tw + gap)
        filled = rng.chance(0.5)
        fc = mix(pal.color(i % len(pal.series)), bg, 0.15 if not pal.dark else 0.0) if filled else bg
        tc = readable_on(fc)
        if filled or rng.chance(0.5):
            ax.add_patch(FancyBboxPatch((x0, 20), tw, 60, boxstyle="round,pad=0,rounding_size=3",
                         facecolor=fc, edgecolor=pal.muted if not filled else "none", linewidth=0.8))
        label, mag, unit, dec = labels_pool[ti]
        val = mag * rng.uniform(0.7, 1.3)
        num = content.format_number(round(val, dec), dec, style.language) + (unit or "")
        up = rng.chance(0.65)
        arrow = "▲" if up else "▼"
        acol = pal.positive if up else pal.negative
        ax.text(x0 + tw / 2, 58, num, ha="center", va="center", color=tc, **_fk(style, min(tw * 0.32, 22), True))
        ax.text(x0 + tw / 2, 40, label, ha="center", va="center", color=mix(tc, bg, 0.35), **_fk(style, 8))
        chg = content.format_number(round(rng.uniform(0.5, 18), 1), 1, style.language)
        ax.text(x0 + tw / 2, 30, f"{arrow} {chg}%", ha="center", va="center", color=acol, **_fk(style, 8, True))
    return {"sub_kind": "kpi_tile"}


_KPI_LABELS_DE = (
    ("Umsatz", 4200, " Mio.", 0), ("EBIT-Marge", 11.5, "%", 1), ("Mitarbeiter", 8600, "", 0),
    ("ROCE", 13.6, "%", 1), ("Dividende", 1.15, " €", 2), ("CO2", 420, " kt", 0),
    ("Auftragseingang", 1150, " Mio.", 0), ("Eigenkapitalquote", 42.0, "%", 1),
)
_KPI_LABELS_EN = (
    ("Revenue", 4200, "m", 0), ("EBIT margin", 11.5, "%", 1), ("Employees", 8600, "", 0),
    ("ROCE", 13.6, "%", 1), ("Dividend", 1.15, " €", 2), ("CO2", 420, " kt", 0),
    ("Order intake", 1150, "m", 0), ("Equity ratio", 42.0, "%", 1),
)


def _gauge_progress(fig, ax, style, rng):
    pal = style.palette
    _canvas(fig, ax, pal, rng)
    kind = rng.pick(("gauge", "ring", "bars", "traffic"))
    col = pal.color(0) if abs(_lum(pal.color(0)) - _lum(pal.background)) > 0.25 else pal.accent
    val = rng.uniform(0.25, 0.92)
    if kind == "gauge":
        cx, cy, R = 50, 38, 34
        ax.add_patch(Wedge((cx, cy), R, 0, 180, facecolor=mix(pal.muted, pal.background, 0.6),
                     edgecolor="none", width=R * 0.32))
        ax.add_patch(Wedge((cx, cy), R, 180 - val * 180, 180, facecolor=col, edgecolor="none", width=R * 0.32))
        a = math.radians(180 - val * 180)
        ax.plot([cx, cx + R * 0.7 * math.cos(a)], [cy, cy + R * 0.7 * math.sin(a)], color=pal.text, linewidth=2)
        ax.add_patch(Circle((cx, cy), 2.5, facecolor=pal.text, edgecolor="none"))
        ax.text(cx, cy - 14, f"{round(val*100)} %", ha="center", va="center", color=pal.text, **_fk(style, 14, True))
    elif kind == "ring":
        cx, cy, R = 50, 50, 32
        ax.add_patch(Wedge((cx, cy), R, 0, 360, facecolor=mix(pal.muted, pal.background, 0.6),
                     edgecolor="none", width=R * 0.28))
        ax.add_patch(Wedge((cx, cy), R, 90, 90 - val * 360, facecolor=col, edgecolor="none", width=R * 0.28))
        ax.text(cx, cy, f"{round(val*100)} %", ha="center", va="center", color=pal.text, **_fk(style, 16, True))
    elif kind == "bars":
        labels = rng.sample(list(_ORG_DE if style.language == "de" else _ORG_EN), rng.randint(3, 5))
        for i, lb in enumerate(labels):
            y = 82 - i * (70 / len(labels))
            v = rng.uniform(0.3, 0.98)
            ax.add_patch(Rectangle((30, y), 60, 7, facecolor=mix(pal.muted, pal.background, 0.7), edgecolor="none"))
            ax.add_patch(Rectangle((30, y), 60 * v, 7, facecolor=ramp(pal, len(labels))[i], edgecolor="none"))
            ax.text(28, y + 3.5, lb, ha="right", va="center", color=pal.text, **_fk(style, 7))
            ax.text(91, y + 3.5, f"{round(v*100)}%", ha="left", va="center", color=pal.muted, **_fk(style, 7))
    else:  # traffic
        for i, c in enumerate(("#c62828", "#f6a821", "#2e7d32")):
            on = i == rng.randint(0, 2)
            ax.add_patch(Circle((50, 74 - i * 24), 10, facecolor=c if on else mix(c, pal.background, 0.7),
                         edgecolor=pal.muted, linewidth=1))
    return {"sub_kind": "gauge_progress"}


def _sankey(fig, ax, style, rng):
    pal = style.palette
    _canvas(fig, ax, pal, rng)
    n_l = rng.randint(2, 3)
    n_r = rng.randint(2, 4)
    left_y = np.linspace(75, 20, n_l)
    right_y = np.linspace(80, 15, n_r)
    cols = ramp(pal, n_l)
    lh = 60 / n_l * 0.7
    rh = 65 / n_r * 0.7
    for i, ly in enumerate(left_y):
        ax.add_patch(Rectangle((10, ly - lh / 2), 5, lh, facecolor=cols[i], edgecolor="none"))
    for j, ry in enumerate(right_y):
        ax.add_patch(Rectangle((85, ry - rh / 2), 5, rh, facecolor=mix(pal.muted, pal.background, 0.3), edgecolor="none"))
    for i, ly in enumerate(left_y):
        for j, ry in enumerate(right_y):
            if rng.chance(0.6):
                w = rng.uniform(1.5, 5)
                verts = [(15, ly), (48, ly), (52, ry), (85, ry)]
                codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
                ax.add_patch(PathPatch(MplPath(verts, codes), facecolor="none",
                             edgecolor=cols[i], linewidth=w, alpha=0.4))
    return {"sub_kind": "sankey"}


# --------------------------------------------------------------------------
# Chart-shaped (real axes, but not tier-1 classes)
# --------------------------------------------------------------------------


def _plain_axes(fig, ax, style, pal):
    fig.subplots_adjust(left=0.12, right=0.95, top=0.9, bottom=0.12)
    ax.set_facecolor(pal.background)
    fig.patch.set_facecolor(pal.background)
    for s in ax.spines.values():
        s.set_color(pal.muted)
        s.set_linewidth(0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=pal.muted, labelsize=style.tick_pt * 0.8)
    for lb in ax.get_xticklabels() + ax.get_yticklabels():
        lb.set_fontfamily(style.font_family)


def _radar(fig, ax, style, rng):
    pal = style.palette
    fig.delaxes(ax)
    ax = fig.add_subplot(111, projection="polar")
    fig.subplots_adjust(left=0.1, right=0.9, top=0.88, bottom=0.1)
    fig.patch.set_facecolor(pal.background)
    ax.set_facecolor(pal.background)
    k = rng.randint(4, 7)
    labels = rng.sample(list(_ORG_DE if style.language == "de" else _ORG_EN), k)
    angles = np.linspace(0, 2 * np.pi, k, endpoint=False).tolist()
    angles += angles[:1]
    n_series = rng.randint(1, 2)
    for si in range(n_series):
        vals = [rng.uniform(0.3, 1.0) for _ in range(k)]
        vals += vals[:1]
        c = pal.color(si)
        ax.plot(angles, vals, color=c, linewidth=1.6)
        ax.fill(angles, vals, color=c, alpha=0.22)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=style.tick_pt * 0.75, fontfamily=style.font_family, color=pal.muted)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1.05)
    ax.grid(color=pal.grid, linewidth=0.6)
    ax.spines["polar"].set_color(pal.muted)
    return {"sub_kind": "radar"}


def _bubble(fig, ax, style, rng):
    pal = style.palette
    _plain_axes(fig, ax, style, pal)
    n = rng.randint(6, 18)
    xs = [rng.uniform(0, 1) for _ in range(n)]
    ys = [rng.uniform(0, 1) for _ in range(n)]
    sz = [rng.uniform(30, 700) for _ in range(n)]
    cols = ramp(pal, min(n, 5))
    ax.scatter(xs, ys, s=sz, c=[cols[i % len(cols)] for i in range(n)], alpha=0.6, edgecolors="none")
    ax.set_xticks([]); ax.set_yticks([])
    return {"sub_kind": "bubble"}


def _tornado(fig, ax, style, rng):
    pal = style.palette
    _plain_axes(fig, ax, style, pal)
    k = rng.randint(4, 8)
    labels = rng.sample(list(_ORG_DE if style.language == "de" else _ORG_EN), k)
    lows = [-rng.uniform(0.2, 1.0) for _ in range(k)]
    highs = [rng.uniform(0.2, 1.0) for _ in range(k)]
    order = sorted(range(k), key=lambda i: highs[i] - lows[i])
    y = range(k)
    for rank, i in enumerate(order):
        ax.barh(rank, lows[i], color=pal.negative, height=0.7)
        ax.barh(rank, highs[i], color=pal.positive, height=0.7)
    ax.axvline(0, color=pal.muted, linewidth=1)
    ax.set_yticks(list(y))
    ax.set_yticklabels([labels[i] for i in order], fontsize=style.tick_pt * 0.75, fontfamily=style.font_family)
    ax.set_xticks([])
    return {"sub_kind": "tornado"}


def _scatter(fig, ax, style, rng):
    pal = style.palette
    _plain_axes(fig, ax, style, pal)
    n_series = rng.randint(1, 3)
    for si in range(n_series):
        n = rng.randint(12, 40)
        cx, cy = rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8)
        xs = np.clip(np.random.default_rng(rng.randint(0, 10**6)).normal(cx, 0.15, n), 0, 1)
        ys = np.clip(np.random.default_rng(rng.randint(0, 10**6)).normal(cy, 0.15, n), 0, 1)
        ax.scatter(xs, ys, s=rng.uniform(8, 30), color=pal.color(si), alpha=0.7, edgecolors="none")
    if rng.chance(0.4):
        ax.plot([0, 1], [rng.uniform(0, 0.4), rng.uniform(0.6, 1)], color=pal.muted, linestyle="--", linewidth=1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.tick_params(labelsize=style.tick_pt * 0.7)
    return {"sub_kind": "scatter"}


def _boxplot(fig, ax, style, rng):
    pal = style.palette
    _plain_axes(fig, ax, style, pal)
    k = rng.randint(3, 6)
    data = [np.random.default_rng(rng.randint(0, 10**6)).normal(rng.uniform(0.3, 0.7), rng.uniform(0.08, 0.2), 40)
            for _ in range(k)]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6)
    cols = ramp(pal, k)
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c)
        patch.set_edgecolor(pal.muted)
    for w in bp["whiskers"] + bp["caps"] + bp["medians"]:
        w.set_color(pal.muted)
    ax.set_xticklabels([str(2020 + i) for i in range(k)], fontsize=style.tick_pt * 0.75, fontfamily=style.font_family)
    ax.set_yticks([])
    return {"sub_kind": "boxplot"}


# --------------------------------------------------------------------------
# Non-charts
# --------------------------------------------------------------------------


def _decorative(fig, ax, style, rng):
    pal = style.palette
    _canvas(fig, ax, pal, rng)
    kind = rng.pick(("gradient", "stripes", "blocks", "ornament", "rule"))
    if kind == "gradient":
        grad = np.linspace(0, 1, 256).reshape(1, -1)
        c0, c1 = pal.color(0), lighten(pal.color(0), 0.7)
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("g", [c0, c1])
        ax.imshow(grad, extent=[0, 100, 0, 100], aspect="auto", cmap=cmap)
    elif kind == "stripes":
        cols = ramp(pal, 5)
        for i in range(rng.randint(4, 10)):
            ax.add_patch(Rectangle((i * 10, 0), 10, 100, facecolor=cols[i % len(cols)], edgecolor="none"))
    elif kind == "blocks":
        for _ in range(rng.randint(3, 8)):
            x, y = rng.uniform(0, 80), rng.uniform(0, 80)
            ax.add_patch(Rectangle((x, y), rng.uniform(10, 30), rng.uniform(10, 30),
                         facecolor=pal.color(rng.randint(0, len(pal.series) - 1)), edgecolor="none", alpha=0.8))
    elif kind == "ornament":
        for i in range(rng.randint(3, 6)):
            ax.add_patch(Circle((rng.uniform(20, 80), rng.uniform(20, 80)), rng.uniform(6, 20),
                         facecolor="none", edgecolor=pal.color(0), linewidth=1.5, alpha=0.7))
    else:  # rule
        for _ in range(rng.randint(1, 3)):
            y = rng.uniform(20, 80)
            ax.plot([10, 90], [y, y], color=pal.color(0), linewidth=rng.uniform(1, 4))
    return {"sub_kind": "decorative"}


def _blank_artifact(fig, ax, style, rng):
    """
    Near-empty extraction junk. The single most-forgotten `other` sub-type and
    the one that stops the model calling extraction noise a confident chart.
    """
    pal = style.palette
    bg = rng.pick((pal.background, "#ffffff", "#fbfbf9", "#f4f2ee"))
    _canvas(fig, ax, pal, rng, bg)
    kind = rng.pick(("almost_empty", "half_text", "edge_strip", "thin_lines", "corner_blob", "faint"))
    if kind == "almost_empty":
        if rng.chance(0.5):
            ax.plot([rng.uniform(0, 40), rng.uniform(60, 100)], [rng.uniform(40, 60)] * 2,
                    color=pal.muted, linewidth=0.6)
    elif kind == "half_text":
        # a strip of text clipped at the top or bottom edge
        y = rng.pick((97, 3))
        words = rng.sample(list(_ORG_DE if style.language == "de" else _ORG_EN), 3)
        ax.text(rng.uniform(5, 30), y, " ".join(words), ha="left",
                va="top" if y > 50 else "bottom", color=pal.text, **_fk(style, 13))
    elif kind == "edge_strip":
        if rng.chance(0.5):
            ax.add_patch(Rectangle((0, rng.pick((0, 90))), 100, 10, facecolor=pal.color(0), edgecolor="none"))
        else:
            ax.add_patch(Rectangle((rng.pick((0, 92)), 0), 8, 100, facecolor=pal.color(0), edgecolor="none"))
    elif kind == "thin_lines":
        for _ in range(rng.randint(1, 3)):
            if rng.chance(0.5):
                x = rng.uniform(10, 90)
                ax.plot([x, x], [10, 90], color=pal.muted, linewidth=0.5)
            else:
                y = rng.uniform(10, 90)
                ax.plot([10, 90], [y, y], color=pal.muted, linewidth=0.5)
    elif kind == "corner_blob":
        ax.add_patch(Circle((rng.pick((5, 95)), rng.pick((5, 95))), rng.uniform(8, 20),
                     facecolor=pal.color(0), edgecolor="none", alpha=0.5))
    else:  # faint watermark-ish
        ax.text(50, 50, rng.pick(("§", "—", "×", "+", "·")), ha="center", va="center",
                color=mix(pal.muted, bg, 0.6), **_fk(style, 40))
    return {"sub_kind": "blank_artifact"}


# The chart classes an R1/R2 frame may embed. Deliberately excludes other/map/
# table/logo/photo so a frame never recurses into itself or wraps a non-chart.
_EMBEDDABLE = ("bar_vertical", "bar_horizontal", "bar_stacked", "line",
               "pie_donut", "waterfall", "combo_bar_line")


def _embed_chart(style, rng, target_px, allow=_EMBEDDABLE):
    """
    Render a *real* chart to a PIL image, to be composited into a frame.

    R1 and R2 only mean something if the thing inside the frame genuinely looks
    like a chart -- a stand-in mini-plot would teach the model that framed
    mini-plots are `other` while framed real charts are not. Importing the
    registry here (not at module load) breaks the config -> renderers.other
    import cycle.
    """
    from ..config import REGISTRY
    from ..style import sample_style

    label = rng.pick(allow)
    plan = REGISTRY[label]
    w, h = max(80, target_px[0]), max(80, target_px[1])
    sub_style = sample_style(rng, w, h)
    spec = plan.build_spec(rng.pick(list(plan.subtypes)), sub_style, rng)
    result = plan.render(spec, sub_style, rng, 1.4)
    return result.image, label


def _place_image(fig, rect, pil_img):
    """imshow a PIL image into a figure-fraction rectangle, no ticks."""
    a = fig.add_axes(rect)
    a.imshow(np.asarray(pil_img), aspect="auto", interpolation="antialiased")
    a.axis("off")
    return a


def _multi_chart(fig, ax, style, rng):
    """Two independent real charts side by side -- R1, not cuttable apart."""
    pal = style.palette
    fig.delaxes(ax)
    fig.patch.set_facecolor(pal.background)
    fw, fh = fig.get_size_inches()
    dpi = fig.dpi
    # A shared title band on top makes the two panels read as one uncuttable
    # figure rather than two crops that happen to be adjacent.
    band = fig.add_axes([0, 0.9, 1, 0.1]); band.axis("off")
    band.text(0.03, 0.5, rng.pick(list(_ORG_DE if style.language == "de" else _ORG_EN)),
              ha="left", va="center", color=pal.text, transform=band.transAxes, **_fk(style, 11, True))
    labels = []
    for x0 in (0.04, 0.52):
        px = (int(fw * 0.44 * dpi), int(fh * 0.78 * dpi))
        img, lb = _embed_chart(style, rng, px)
        _place_image(fig, [x0, 0.06, 0.44, 0.8], img)
        labels.append(lb)
    return {"sub_kind": "multi_chart", "embedded": labels}


def _table_with_bars(fig, ax, style, rng):
    """
    A table with a bar embedded in one column -- the guide is explicit that this
    is `other`, not `table` (table, boundary note). It shares a KPI table's shape
    but the value column is a horizontal bar, which is exactly the
    `table_has_graphics` case the `table` invariant rejects.
    """
    pal = style.palette
    _canvas(fig, ax, pal, rng, pal.background)
    labels = rng.sample(list(_ORG_DE if style.language == "de" else _ORG_EN), rng.randint(4, 7))
    vals = [rng.uniform(0.25, 1.0) for _ in labels]
    bar_col = pal.color(0) if abs(_lum(pal.color(0)) - _lum(pal.background)) > 0.2 else pal.accent
    rule = mix(pal.text, pal.background, 0.6)
    top, bottom = 88, 10
    rh = (top - bottom) / len(labels)
    ax.plot([4, 96], [top + 3, top + 3], color=rule, linewidth=1.0)
    for i, (lb, v) in enumerate(zip(labels, vals)):
        y = top - (i + 0.5) * rh
        ax.text(5, y, lb, ha="left", va="center", color=pal.text, **_fk(style, style.tick_pt * 0.85))
        ax.add_patch(Rectangle((52, y - rh * 0.28), 40 * v, rh * 0.56, facecolor=bar_col, edgecolor="none"))
        ax.text(93, y, content.format_number(round(v * 100), 0, style.number_locale),
                ha="right", va="center", color=pal.muted, **_fk(style, style.tick_pt * 0.8))
        if rng.chance(0.5):
            ax.plot([4, 96], [y - rh * 0.5, y - rh * 0.5], color=mix(rule, pal.background, 0.5), linewidth=0.4)
    ax.plot([4, 96], [bottom - 1, bottom - 1], color=rule, linewidth=1.0)
    ax.plot([48, 48], [bottom - 1, top + 3], color=mix(rule, pal.background, 0.4), linewidth=0.5)
    return {"sub_kind": "table_with_bars"}


def _donut_progress(fig, ax, style, rng):
    """
    A ring as a *progress* indicator, not a part-to-whole breakdown -- `other`,
    and the deliberate borderline case against pie_donut (design doc §5). Two
    segments (done / remaining), a big percentage in the hole, a goal label.
    """
    pal = style.palette
    _canvas(fig, ax, pal, rng)
    val = rng.uniform(0.35, 0.92)
    done = pal.color(0) if abs(_lum(pal.color(0)) - _lum(pal.background)) > 0.2 else pal.accent
    rest = mix(pal.muted, pal.background, 0.6)
    cx, cy, R = 50, 52, 30
    width = R * rng.uniform(0.22, 0.4)  # thin ring -> reads as a donut
    ax.add_patch(Wedge((cx, cy), R, 90 - val * 360, 90, facecolor=done, edgecolor="none", width=width))
    ax.add_patch(Wedge((cx, cy), R, 90, 90 - val * 360, facecolor=rest, edgecolor="none", width=width))
    ax.text(cx, cy, f"{round(val * 100)} %", ha="center", va="center", color=pal.text, **_fk(style, 18, True))
    goal = (rng.pick(("Zielerreichung", "Fortschritt", "Umsetzungsgrad", "Auslastung"))
            if style.language == "de" else rng.pick(("Progress", "Completion", "Utilisation")))
    ax.text(cx, cy - R - 8, goal, ha="center", va="center", color=pal.muted, **_fk(style, style.tick_pt))
    return {"sub_kind": "donut_progress"}


def _mini(a, kind, style, rng, pal):
    n = rng.randint(3, 5)
    if kind == "bars":
        a.bar(range(n), [rng.uniform(1, 10) for _ in range(n)], color=pal.color(0))
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
        a.spines["bottom"].set_visible(True); a.spines["bottom"].set_color(pal.muted)
    elif kind == "line":
        a.plot(range(n), [rng.uniform(1, 10) for _ in range(n)], color=pal.color(0), linewidth=1.5, marker="o", markersize=3)
        a.set_xticks([]); a.set_yticks([])
        for s in a.spines.values():
            s.set_visible(False)
    else:
        vals = [rng.uniform(1, 5) for _ in range(n)]
        a.pie(vals, colors=ramp(pal, n), radius=1.1, wedgeprops={"width": rng.pick((1.0, 0.45))})
    a.set_title(rng.pick(list(_ORG_DE if style.language == "de" else _ORG_EN)),
                fontsize=style.tick_pt * 0.8, fontfamily=style.font_family, color=pal.text)


def _infographic_frame(fig, ax, style, rng):
    """
    A chart embedded in a designed layout, taking well under half the area -- R2,
    which makes the whole crop `other` regardless of the chart type inside it.
    """
    pal = style.palette
    fig.delaxes(ax)
    fig.patch.set_facecolor(pal.background)
    # heading band
    band = fig.add_axes([0, 0.82, 1, 0.18]); band.axis("off")
    band.add_patch(Rectangle((0, 0), 1, 1, facecolor=mix(pal.color(0), pal.background, 0.2), edgecolor="none",
                   transform=band.transAxes))
    band.text(0.05, 0.5, rng.pick(("Nachhaltigkeit", "Strategie 2030", "Unser Weg", "Highlights"))
              if style.language == "de" else rng.pick(("Sustainability", "Strategy 2030", "Highlights")),
              ha="left", va="center", color=readable_on(mix(pal.color(0), pal.background, 0.2)),
              transform=band.transAxes, **_fk(style, 15, True))
    # body text blocks (grey bars standing in for paragraphs)
    txt = fig.add_axes([0.05, 0.1, 0.5, 0.66]); txt.axis("off")
    for i in range(rng.randint(4, 7)):
        y = 0.9 - i * 0.14
        txt.add_patch(Rectangle((0, y), rng.uniform(0.6, 1.0), 0.05,
                      facecolor=mix(pal.muted, pal.background, 0.4), edgecolor="none", transform=txt.transAxes))
    # A real chart in the corner, deliberately small. Rule R2: the chart takes
    # well under half the frame, so the whole crop is `other` however genuine
    # the chart inside it looks -- which is the point of embedding a real one.
    rect = [0.60, 0.14, 0.34, 0.46]
    fw, fh = fig.get_size_inches()
    px = (int(fw * rect[2] * fig.dpi), int(fh * rect[3] * fig.dpi))
    img, emb = _embed_chart(style, rng, px)
    _place_image(fig, rect, img)
    return {"sub_kind": "infographic_frame", "chart_area_frac": round(rect[2] * rect[3], 3),
            "embedded": emb}


def _lum(c: str) -> float:
    from ..palettes import luminance
    return luminance(c)


_DISPATCH = {
    "org_chart": _org_chart, "process_flow": _process_flow, "timeline": _timeline,
    "matrix": _matrix, "kpi_tile": _kpi_tile, "gauge_progress": _gauge_progress,
    "sankey": _sankey, "radar": _radar, "bubble": _bubble, "tornado": _tornado,
    "scatter": _scatter, "boxplot": _boxplot, "decorative": _decorative,
    "blank_artifact": _blank_artifact, "multi_chart": _multi_chart,
    "infographic_frame": _infographic_frame, "table_with_bars": _table_with_bars,
    "donut_progress": _donut_progress,
}
