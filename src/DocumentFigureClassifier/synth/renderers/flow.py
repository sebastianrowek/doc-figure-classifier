"""
Flow charts: the ``flow`` tier-1 class -- process/workflow diagrams and org
charts, nodes joined by drawn connectors into a path or a hierarchy.

Promoted out of ``other`` in taxonomy v1.2 (was other/org_chart +
other/process_flow). The base DocumentFigureClassifier model already knows flow
charts, so this is a future parsing target; see the labeling guide.

Taxonomy v1.5 tightened the class: ``flow`` is now **never circular**. The old
``process_cycle`` sub-type was removed and belongs to ``other`` -- keeping the
rule exception-free gives the classifier "round => not flow" as a clean
discriminative feature. Chevrons and funnel stages stay, because the shape
itself carries the direction even though no separate connector is drawn.

The sub-types deliberately cover the *dense* end of the real distribution
(multi-level trees, branching, phases, icon steps) -- an audit of the held-out
test set showed the model had learned "flow == sparse box-and-arrow" from
synth that was far simpler than real annual-report diagrams.

Each renderer reports the node/edge counts it drew so base._check_flow can
assert the image really is a connected diagram.
"""

from __future__ import annotations

import math
import textwrap

import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

from ..engines import mpl
from ..palettes import lighten, mix, ramp, readable_on
from ..rng import Rng
from ..style import StyleSheet
from .base import FigureSpec, RenderResult, Structure

SUBTYPES = {
    "org_chart": 0.20,        # multi-level hierarchy with drawn reporting lines
    "process_flow": 0.18,     # chevrons or a box chain, horizontal or vertical
    "decision_tree": 0.14,    # diamonds with yes/no branches
    "swimlane": 0.10,         # boxes hopping between labelled horizontal lanes
    "value_chain": 0.10,      # a segmented right-pointing arrow (Porter style)
    "nested_phase": 0.09,     # phase containers holding steps, arrows between
    "branch_merge": 0.08,     # one path splits into parallel tracks and rejoins
    "icon_step_chain": 0.06,  # icon + caption per step, joined by arrows
    "funnel": 0.05,           # stacked, narrowing stages
}

# Short department labels for org-chart boxes -- the boxes are narrow, so longer
# names would overrun them.
_DEPT_DE = ("CEO", "CFO", "COO", "Vertrieb", "F&E", "HR", "IT", "Finanz",
            "Einkauf", "Mktg", "Werk A", "Werk B", "Technik", "Recht")
_DEPT_EN = ("CEO", "CFO", "COO", "Sales", "R&D", "HR", "IT", "Finance",
            "Buying", "Mktg", "Plant A", "Plant B", "Legal", "Ops")

# Governance bodies -- longer, wrapped over two lines. Real reports are full of
# these, and they are what makes a governance chart look dense.
_GOV_DE = ("Aufsichtsrat", "Vorstand", "Prüfungsausschuss", "Risikoausschuss",
           "Nominierungsausschuss", "Vergütungsausschuss", "Interne Revision",
           "Compliance Committee", "Nachhaltigkeitsrat", "Konzernrevision",
           "Bereichsleitung", "Fachabteilungen", "Konzernsicherheit")
_GOV_EN = ("Supervisory Board", "Management Board", "Audit Committee",
           "Risk Committee", "Nomination Committee", "Remuneration Committee",
           "Internal Audit", "Compliance Committee", "Sustainability Council",
           "Group Audit", "Divisional Management", "Operating Units",
           "Group Security")

# Short single-word step labels; a chevron is too narrow for long names.
_STEPS_DE = ("Analyse", "Konzept", "Planung", "Aufbau", "Betrieb", "Start",
             "Ziel", "Test", "Ausbau", "Review")
_STEPS_EN = ("Analyze", "Concept", "Design", "Build", "Operate", "Start",
             "Goal", "Test", "Scale", "Review")

# Multi-word step names for the box-chain and phase variants, where the node is
# wide enough to wrap two or three lines.
_PHRASE_DE = ("Risiken identifizieren", "Maßnahmen ableiten", "Umsetzung steuern",
              "Wirksamkeit prüfen", "Bericht an den Vorstand", "Daten erheben",
              "Kennzahlen validieren", "Freigabe einholen", "Ergebnisse dokumentieren",
              "Lieferanten bewerten", "Schulung durchführen")
_PHRASE_EN = ("Identify risks", "Define measures", "Steer implementation",
              "Review effectiveness", "Report to the Board", "Collect data",
              "Validate key figures", "Obtain approval", "Document results",
              "Assess suppliers", "Deliver training")

# Decision questions for the diamond nodes.
_DECIDE_DE = ("Wesentlich?", "Schwellenwert\nüberschritten?", "Freigabe\nerteilt?",
              "Risiko\nakzeptabel?", "Kriterien\nerfüllt?")
_DECIDE_EN = ("Material?", "Threshold\nexceeded?", "Approval\ngranted?",
              "Risk\nacceptable?", "Criteria\nmet?")

# Value-chain activities (Porter): primary stages, left to right.
_VALUE_DE = ("Beschaffung", "Produktion", "Logistik", "Marketing", "Vertrieb", "Service")
_VALUE_EN = ("Inbound", "Operations", "Outbound", "Marketing", "Sales", "Service")

# Funnel stages, widest first.
_FUNNEL_DE = ("Besucher", "Leads", "Angebote", "Abschlüsse", "Kunden")
_FUNNEL_EN = ("Visitors", "Leads", "Offers", "Deals", "Customers")

# Phase / lane headers.
_PHASE_DE = ("Analyse", "Konzeption", "Umsetzung", "Kontrolle", "Berichterstattung")
_PHASE_EN = ("Analysis", "Design", "Implementation", "Control", "Reporting")

# Chrome: titles, footnotes and the "1st/2nd/3rd line of defence" lane captions
# that real governance charts carry down their left edge.
_TITLE_DE = ("Konzernstruktur", "Governance-Struktur", "Prozessablauf",
             "Risikomanagement-Prozess", "Ablauf der Berichterstattung",
             "Organisation des Konzerns", "Steuerungsmodell")
_TITLE_EN = ("Group structure", "Governance structure", "Process flow",
             "Risk management process", "Reporting process",
             "Group organisation", "Operating model")
_NOTE_DE = ("1) Stand: 31.12.", "Quelle: Interne Darstellung",
            "Vereinfachte Darstellung")
_NOTE_EN = ("1) As at 31 Dec", "Source: internal", "Simplified presentation")
_LANE_DE = ("1. Verteidigungslinie", "2. Verteidigungslinie", "3. Verteidigungslinie")
_LANE_EN = ("First line of defence", "Second line of defence", "Third line of defence")

_YES = {"de": "Ja", "en": "Yes"}
_NO = {"de": "Nein", "en": "No"}

_ICONS = ("doc", "gear", "person", "chart", "check", "globe")


def build_spec(sub_type: str, style: StyleSheet, rng: Rng) -> FigureSpec:
    return FigureSpec(
        label="flow", sub_type=sub_type,
        title=None, subtitle=None, source=None, unit="",
        categories=[], series=[], series_names=[], decimals=0, percent=False, extra={},
    )


def _words(style, kind):
    """Pick the language-matched vocabulary tuple."""
    de = style.language == "de"
    return {
        "dept": _DEPT_DE if de else _DEPT_EN,
        "gov": _GOV_DE if de else _GOV_EN,
        "step": _STEPS_DE if de else _STEPS_EN,
        "phrase": _PHRASE_DE if de else _PHRASE_EN,
        "decide": _DECIDE_DE if de else _DECIDE_EN,
        "value": _VALUE_DE if de else _VALUE_EN,
        "funnel": _FUNNEL_DE if de else _FUNNEL_EN,
        "phase": _PHASE_DE if de else _PHASE_EN,
        "title": _TITLE_DE if de else _TITLE_EN,
        "note": _NOTE_DE if de else _NOTE_EN,
        "lane": _LANE_DE if de else _LANE_EN,
    }[kind]


def _pool(rng, vocab, need):
    """
    A shuffled label pool guaranteed to survive `need` pops.

    The vocabularies are deliberately short so the labels stay plausible; a
    dense chart can want more nodes than there are distinct phrases, and an
    exhausted pool renders as empty boxes.
    """
    out: list[str] = []
    while len(out) < need:
        out.extend(rng.shuffled(list(vocab)))
    return out


def _fs(style, rel=1.0):
    """
    Font size for text drawn on the 0-100 canvas, relative to the figure width.

    Point sizes are absolute but the canvas is always 100 units wide, so a fixed
    pt size covers far more of a narrow crop than of a wide one -- which is how
    the first version of these renderers produced overlapping, oversized text on
    small figures. ~2pt per inch of figure width keeps a 12-character label
    inside a 20-unit box at any physical crop size, so the *proportions* stay
    constant, which is what actually reads as a real diagram.

    Height bounds it too: a wide, short crop has plenty of horizontal room but
    the rows are thin, so width alone would still overflow them vertically.
    """
    return max(3.2, min(2.05 * style.fig_w_in, 3.4 * style.fig_h_in, 12.0)) * rel


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


def _wrap(text, width):
    """
    Wrap without splitting words. German compounds ("Nachhaltigkeitsrat") are
    longer than a narrow box, and a mid-word break is a tell that no real
    diagram has -- the typesetter shrinks the type instead, which is what
    _box does with the width this returns.
    """
    return "\n".join(textwrap.wrap(text, max(6, int(width)),
                                   break_long_words=False)) or text


def _box(ax, x, y, w, h, fc, ec, style, text=None, tc=None, fs=8, bold=False,
         rounded=True, wrap_at=None):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=%.1f" % (min(w, h) * 0.12) if rounded else "square,pad=0",
        facecolor=fc, edgecolor=ec, linewidth=1.0, mutation_aspect=1.0,
    )
    ax.add_patch(patch)
    if text:
        if wrap_at:
            text = _wrap(text, wrap_at)
            # A word that still overruns the box shrinks the type rather than
            # spilling past the edge (or being cut mid-word).
            longest = max(len(line) for line in text.split("\n"))
            if longest > wrap_at:
                fs *= max(0.60, wrap_at / longest)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc,
                linespacing=1.15, **_fk(style, fs, bold))


def _diamond(ax, cx, cy, w, h, fc, ec, style, text=None, tc=None, fs=6.5):
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    ax.add_patch(Polygon(pts, facecolor=fc, edgecolor=ec, linewidth=1.0))
    if text:
        ax.text(cx, cy, text, ha="center", va="center", color=tc,
                linespacing=1.1, **_fk(style, fs, False))


def _arrow(ax, p0, p1, color, lw=1.0, scale=8, dashed=False):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=scale, color=color, linewidth=lw,
        linestyle=(0, (2.5, 2)) if dashed else "solid", shrinkA=0, shrinkB=0, zorder=1))


def _elbow(ax, p0, p1, color, lw=0.9, dashed=False, arrow=False, mid=None):
    """Right-angled connector (down, across, down) -- how real org charts route."""
    x0, y0 = p0
    x1, y1 = p1
    my = mid if mid is not None else (y0 + y1) / 2
    ls = (0, (2.5, 2)) if dashed else "solid"
    ax.plot([x0, x0], [y0, my], color=color, lw=lw, ls=ls, zorder=1, solid_capstyle="butt")
    ax.plot([x0, x1], [my, my], color=color, lw=lw, ls=ls, zorder=1, solid_capstyle="butt")
    if arrow:
        _arrow(ax, (x1, my), (x1, y1), color, lw, 7, dashed)
    else:
        ax.plot([x1, x1], [my, y1], color=color, lw=lw, ls=ls, zorder=1, solid_capstyle="butt")


def _elbow_h(ax, p0, p1, color, lw=0.9, dashed=False, arrow=False, mid=None):
    """
    Right-angled connector routed horizontally first (across, up/down, across).

    The mirror of _elbow, for left-to-right structures: splitting into parallel
    tracks, or hopping between swimlanes. Using the vertical-first variant there
    routes the leg through the wrong axis.
    """
    x0, y0 = p0
    x1, y1 = p1
    mx = mid if mid is not None else (x0 + x1) / 2
    ls = (0, (2.5, 2)) if dashed else "solid"
    ax.plot([x0, mx], [y0, y0], color=color, lw=lw, ls=ls, zorder=1, solid_capstyle="butt")
    ax.plot([mx, mx], [y0, y1], color=color, lw=lw, ls=ls, zorder=1, solid_capstyle="butt")
    if arrow:
        _arrow(ax, (mx, y1), (x1, y1), color, lw, 7, dashed)
    else:
        ax.plot([mx, x1], [y1, y1], color=color, lw=lw, ls=ls, zorder=1, solid_capstyle="butt")


def _icon(ax, cx, cy, r, kind, color):
    """A small abstract pictogram -- enough to read as an icon, not as a chart."""
    if kind == "doc":
        ax.add_patch(Rectangle((cx - r * 0.5, cy - r * 0.65), r, r * 1.3,
                               facecolor="none", edgecolor=color, linewidth=1.2))
        for k in range(3):
            y = cy + r * 0.35 - k * r * 0.35
            ax.plot([cx - r * 0.28, cx + r * 0.28], [y, y], color=color, lw=0.9)
    elif kind == "gear":
        pts = [(cx + r * (1.0 if i % 2 == 0 else 0.62) * math.cos(math.radians(i * 30)),
                cy + r * (1.0 if i % 2 == 0 else 0.62) * math.sin(math.radians(i * 30)))
               for i in range(12)]
        ax.add_patch(Polygon(pts, facecolor="none", edgecolor=color, linewidth=1.2))
    elif kind == "person":
        ax.add_patch(Polygon([(cx - r * 0.6, cy - r * 0.7), (cx + r * 0.6, cy - r * 0.7),
                              (cx + r * 0.45, cy + r * 0.1), (cx - r * 0.45, cy + r * 0.1)],
                             facecolor="none", edgecolor=color, linewidth=1.2))
        ax.add_patch(Polygon(_circle_pts(cx, cy + r * 0.55, r * 0.38),
                             facecolor="none", edgecolor=color, linewidth=1.2))
    elif kind == "chart":
        for k, hh in enumerate((0.5, 0.95, 0.7)):
            ax.add_patch(Rectangle((cx - r * 0.6 + k * r * 0.45, cy - r * 0.7),
                                   r * 0.3, r * 1.3 * hh,
                                   facecolor=color, edgecolor="none"))
    elif kind == "check":
        ax.add_patch(Polygon(_circle_pts(cx, cy, r * 0.95),
                             facecolor="none", edgecolor=color, linewidth=1.2))
        ax.plot([cx - r * 0.42, cx - r * 0.08, cx + r * 0.45],
                [cy + r * 0.02, cy - r * 0.35, cy + r * 0.42], color=color, lw=1.4)
    else:  # globe
        ax.add_patch(Polygon(_circle_pts(cx, cy, r * 0.95),
                             facecolor="none", edgecolor=color, linewidth=1.2))
        ax.plot([cx - r * 0.95, cx + r * 0.95], [cy, cy], color=color, lw=0.9)
        for sx in (0.45, -0.45):
            ys = np.linspace(-r * 0.95, r * 0.95, 24)
            ax.plot(cx + sx * np.sqrt(np.maximum(0, r ** 2 - ys ** 2)) / r * 1.0,
                    cy + ys, color=color, lw=0.9)


def _circle_pts(cx, cy, r, n=40):
    return [(cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n))
            for i in range(n)]


def _chrome(ax, style, rng):
    """
    Optional title / subtitle / footnote. Returns the (bottom, top) vertical band
    left for the diagram, so every sub-type lays out inside whatever is free.

    Real report diagrams nearly always carry chrome; the first synth generation
    had none, which is part of why the model's flow prototype was so sparse.
    """
    pal = style.palette
    top, bot = 96.0, 4.0
    if rng.chance(0.60):
        ax.text(4, top, rng.pick(_words(style, "title")), ha="left", va="top",
                color=pal.text, **_fk(style, _fs(style, 1.50), True))
        top -= 8.5
        if rng.chance(0.30):
            ax.text(4, top + 1.0, rng.pick(_words(style, "phrase")), ha="left", va="top",
                    color=pal.muted, **_fk(style, _fs(style, 0.97)))
            top -= 5.5
    if rng.chance(0.28):
        ax.text(4, bot - 1.0, rng.pick(_words(style, "note")), ha="left", va="bottom",
                color=pal.muted, **_fk(style, _fs(style, 0.80)))
        bot += 4.5
    return bot, top


def _legend(ax, style, rng, cols, labels, y):
    """A small swatch strip -- real governance charts explain their colour coding."""
    pal = style.palette
    x = 5.0
    for c, lb in zip(cols, labels):
        ax.add_patch(Rectangle((x, y), 3.2, 2.2, facecolor=c, edgecolor="none"))
        ax.text(x + 4.2, y + 1.1, lb, ha="left", va="center", color=pal.muted,
                **_fk(style, _fs(style, 0.78)))
        x += 6.0 + len(lb) * 1.5


def _node_colors(pal, rng, n):
    """Saturated fills with readable text -- real diagrams are not all pastel."""
    if rng.chance(0.5):
        return ramp(pal, n)
    base = pal.color(rng.randint(0, 3))
    return [mix(base, pal.background, 0.0 if rng.chance(0.6) else 0.25) for _ in range(n)]


# --------------------------------------------------------------------------- #
# sub-types
# --------------------------------------------------------------------------- #

def _org_chart(fig, ax, style, rng) -> tuple[int, int]:
    """A 3-4 level hierarchy with drawn (often right-angled) reporting lines."""
    pal = style.palette
    _canvas(fig, ax, pal)
    bot, top = _chrome(ax, style, rng)

    long_names = rng.chance(0.55)
    vocab = list(_words(style, "gov" if long_names else "dept"))
    words = rng.shuffled(vocab)

    ec = pal.color(0)
    base = mix(pal.color(0), pal.background, 0.20 if not pal.dark else 0.0)

    # Build a small tree: root -> level1 -> (optional) level2. Every level-1 node
    # claims at least one leaf slot, so the slot widths must be summed over
    # max(k, 1) -- summing only the real children overflows the canvas.
    n1 = rng.randint(2, 5)
    kids = [rng.randint(0, 3) if rng.chance(0.65) else 0 for _ in range(n1)]
    slots = [max(k, 1) for k in kids]
    n_leaf = sum(slots)

    lanes = rng.chance(0.30) and n1 >= 3
    left = 16.0 if lanes else 5.0
    right = 96.0
    span = right - left

    # Level geometry inside the band the chrome left us.
    levels = 3 if any(kids) else 2
    lh = (top - bot) / levels
    bh = min(9.0, lh * 0.45)
    y_root = top - bh
    y1 = y_root - lh
    y2 = y1 - lh

    # Leaf slots drive the x layout; parents centre over their children.
    slot_w = span / max(n_leaf, 1)
    bw1 = min(20.0, span / n1 - 2.0)
    bw2 = min(16.0, slot_w - 1.5)
    fs1 = _fs(style, 0.92 if long_names else 1.03)
    fs2 = _fs(style, 0.80 if long_names else 0.92)

    if lanes:
        lane_names = list(_words(style, "lane"))[:levels]
        for li, lname in enumerate(lane_names):
            yy = [y_root, y1, y2][li]
            ax.text(3.0, yy + bh / 2, _wrap(lname, 12), ha="left", va="center",
                    color=pal.muted, **_fk(style, _fs(style, 0.75)))

    _box(ax, 50 - bw1 / 2, y_root, bw1, bh, base, ec, style, words.pop(),
         readable_on(base), fs1, True, wrap_at=bw1 * 0.85)
    nodes, edges = 1, 0

    leaf_cursor = 0.0
    centers1 = []
    for i in range(n1):
        k = kids[i]
        width = slots[i] * slot_w
        cx = left + leaf_cursor + width / 2
        leaf_cursor += width
        fc = lighten(base, 0.10)
        _box(ax, cx - bw1 / 2, y1, bw1, bh, fc, ec, style,
             words.pop() if words else "—", readable_on(fc), fs1, wrap_at=bw1 * 0.85)
        _elbow(ax, (50, y_root), (cx, y1 + bh), pal.muted, 0.9,
               mid=(y_root + y1 + bh) / 2)
        nodes += 1
        edges += 1
        centers1.append(cx)

        for j in range(k):
            lx = left + (leaf_cursor - width) + j * slot_w + slot_w / 2
            fc2 = lighten(base, 0.26)
            _box(ax, lx - bw2 / 2, y2, bw2, bh * 0.88, fc2, ec, style,
                 words.pop() if words else "—", readable_on(fc2), fs2,
                 wrap_at=bw2 * 0.85)
            _elbow(ax, (cx, y1), (lx, y2 + bh * 0.88), pal.muted, 0.8,
                   mid=(y1 + y2 + bh) / 2)
            nodes += 1
            edges += 1

    # A dotted staff line to a side body (internal audit reporting straight to
    # the supervisory board) -- very common, and visually distinctive.
    if rng.chance(0.35) and words:
        sw = 17.0
        sx = right - sw
        fc = mix(pal.background, pal.text, 0.06)
        _box(ax, sx, y_root, sw, bh, fc, ec, style, words.pop(), pal.text, fs2,
             wrap_at=sw * 0.85)
        ax.plot([50 + bw1 / 2, sx], [y_root + bh / 2, y_root + bh / 2],
                color=pal.muted, lw=0.8, ls=(0, (2.5, 2)), zorder=1)
        nodes += 1
        edges += 1

    return nodes, edges


def _process(fig, ax, style, rng) -> tuple[int, int]:
    """A linear chain: chevrons, boxes joined by arrows, or a vertical stack."""
    pal = style.palette
    _canvas(fig, ax, pal)
    bot, top = _chrome(ax, style, rng)
    mode = rng.weighted({"chevron": 0.38, "boxes": 0.42, "vertical": 0.20})

    if mode == "vertical":
        steps = rng.randint(3, 6)
        labels = rng.sample(list(_words(style, "phrase")), steps)
        cols = _node_colors(pal, rng, steps)
        bh = min(11.0, (top - bot) / steps - 3.0)
        gap = ((top - bot) - steps * bh) / max(steps - 1, 1)
        bw = rng.uniform(46, 70)
        for i, lb in enumerate(labels):
            y = top - bh - i * (bh + gap)
            _box(ax, 50 - bw / 2, y, bw, bh, cols[i], "none", style, lb,
                 readable_on(cols[i]), _fs(style, 0.97), wrap_at=bw * 0.42)
            if i:
                _arrow(ax, (50, y + bh + gap), (50, y + bh), pal.muted, 1.1, 9)
        return steps, steps - 1

    steps = rng.randint(4, 7)
    labels = rng.sample(list(_words(style, "step" if mode == "chevron" else "phrase")), steps)
    cols = _node_colors(pal, rng, steps)
    mid = (top + bot) / 2

    if mode == "chevron":
        w = 92 / steps
        h = min(14.0, (top - bot) * 0.34)
        y = mid - h / 2
        # A number goes inside the chevron; the word goes on its own line below,
        # where it has the full step width to itself instead of the chevron body.
        for i in range(steps):
            x = 4 + i * w
            pts = [(x, y), (x + w * 0.82, y), (x + w * 0.96, y + h / 2), (x + w * 0.82, y + h),
                   (x, y + h), (x + w * 0.14, y + h / 2)]
            ax.add_patch(Polygon(pts, facecolor=cols[i], edgecolor="none"))
            ax.text(x + w * 0.42, y + h / 2, str(i + 1), ha="center", va="center",
                    color=readable_on(cols[i]), **_fk(style, _fs(style, 1.25), True))
            ax.text(x + w * 0.45, y - 5, labels[i], ha="center", va="top",
                    color=pal.text, **_fk(style, _fs(style, min(0.90, w * 0.042))))
        return steps, steps - 1

    # boxes joined by arrows, with a little vertical jitter so the chain is not
    # a perfect ruler line
    gapw = 92.0 / steps
    bw = gapw * 0.78
    bh = min(13.0, (top - bot) * 0.30)
    for i in range(steps):
        x = 4 + i * gapw
        y = mid - bh / 2 + (rng.uniform(-3, 3) if rng.chance(0.4) else 0)
        _box(ax, x, y, bw, bh, cols[i], "none", style, labels[i],
             readable_on(cols[i]), _fs(style, min(0.92, bw * 0.058)), wrap_at=bw * 0.55)
        if i:
            _arrow(ax, (x - (gapw - bw), mid), (x, mid), pal.muted, 1.1, 9)
    return steps, steps - 1


def _decision_tree(fig, ax, style, rng) -> tuple[int, int]:
    """Diamonds with labelled yes/no branches -- a real decision flow."""
    pal = style.palette
    _canvas(fig, ax, pal)
    bot, top = _chrome(ax, style, rng)
    yes, no = _YES[style.language], _NO[style.language]

    ec = pal.color(0)
    fc = mix(pal.color(0), pal.background, 0.22 if not pal.dark else 0.0)
    dc = mix(pal.accent, pal.background, 0.25 if not pal.dark else 0.0)
    phrases = _pool(rng, _words(style, "phrase"), 12)
    decides = _pool(rng, _words(style, "decide"), 4)

    n_dec = rng.randint(1, 2)
    rows = 2 + n_dec * 2
    rh = (top - bot) / rows
    bh = min(9.0, rh * 0.62)
    bw, dw, dh = 30.0, 26.0, min(13.0, rh * 0.9)
    cx = 42.0

    y = top - bh
    _box(ax, cx - bw / 2, y, bw, bh, fc, ec, style, phrases.pop(),
         readable_on(fc), _fs(style, 0.92), wrap_at=bw * 0.55)
    nodes, edges = 1, 0
    prev_bottom = y

    for d in range(n_dec):
        ycd = prev_bottom - rh * 0.55 - dh / 2
        _arrow(ax, (cx, prev_bottom), (cx, ycd + dh / 2), pal.muted, 1.0, 8)
        _diamond(ax, cx, ycd, dw, dh, dc, ec, style, decides.pop(), readable_on(dc), _fs(style, 0.80))
        nodes += 1
        edges += 1

        # "no" branches off to the right into its own terminal box
        nx = cx + 34.0
        _arrow(ax, (cx + dw / 2, ycd), (nx - bw * 0.32, ycd), pal.muted, 1.0, 8)
        ax.text((cx + dw / 2 + nx - bw * 0.32) / 2, ycd + 1.6, no, ha="center",
                va="bottom", color=pal.muted, **_fk(style, _fs(style, 0.78)))
        nfc = mix(pal.background, pal.text, 0.07)
        _box(ax, nx - bw * 0.32, ycd - bh / 2, bw * 0.64, bh, nfc, ec, style,
             phrases.pop(), pal.text, _fs(style, 0.80), wrap_at=bw * 0.40)
        nodes += 1
        edges += 1

        # "yes" continues downward
        y = ycd - dh / 2 - rh * 0.55 - bh
        _arrow(ax, (cx, ycd - dh / 2), (cx, y + bh), pal.muted, 1.0, 8)
        ax.text(cx + 1.6, (ycd - dh / 2 + y + bh) / 2, yes, ha="left", va="center",
                color=pal.muted, **_fk(style, _fs(style, 0.78)))
        last = d == n_dec - 1
        lfc = mix(pal.positive, pal.background, 0.25) if last else fc
        _box(ax, cx - bw / 2, y, bw, bh, lfc, ec, style, phrases.pop(),
             readable_on(lfc), _fs(style, 0.89), wrap_at=bw * 0.55)
        nodes += 1
        edges += 1
        prev_bottom = y

    return nodes, edges


def _branch_merge(fig, ax, style, rng) -> tuple[int, int]:
    """One path splits into parallel tracks and rejoins -- an operating model."""
    pal = style.palette
    _canvas(fig, ax, pal)
    bot, top = _chrome(ax, style, rng)

    ec = pal.color(0)
    phrases = _pool(rng, _words(style, "phrase"), 8)
    n = rng.randint(2, 4)
    cols = _node_colors(pal, rng, n)

    mid = (top + bot) / 2
    bw, bh = 20.0, min(11.0, (top - bot) * 0.22)
    x0, x1 = 4.0, 76.0

    fc = mix(pal.color(0), pal.background, 0.22 if not pal.dark else 0.0)
    _box(ax, x0, mid - bh / 2, bw, bh, fc, ec, style, phrases.pop(),
         readable_on(fc), _fs(style, 0.86), wrap_at=bw * 0.55)
    _box(ax, x1, mid - bh / 2, bw, bh, fc, ec, style, phrases.pop(),
         readable_on(fc), _fs(style, 0.86), wrap_at=bw * 0.55)
    nodes, edges = 2, 0

    track_h = (top - bot) * 0.78
    ys = np.linspace(mid + track_h / 2 - bh / 2, mid - track_h / 2 + bh / 2, n)
    mx = 40.0
    for i, y in enumerate(ys):
        _box(ax, mx - bw / 2, y - bh / 2, bw, bh, cols[i], "none", style,
             phrases.pop(), readable_on(cols[i]), _fs(style, 0.83),
             wrap_at=bw * 0.55)
        _elbow_h(ax, (x0 + bw, mid), (mx - bw / 2, y), pal.muted, 0.9, arrow=True)
        _elbow_h(ax, (mx + bw / 2, y), (x1, mid), pal.muted, 0.9, arrow=True)
        nodes += 1
        edges += 2

    return nodes, edges


def _nested_phase(fig, ax, style, rng) -> tuple[int, int]:
    """Phase containers, each holding sub-steps, with arrows between phases."""
    pal = style.palette
    _canvas(fig, ax, pal)
    bot, top = _chrome(ax, style, rng)

    n = rng.randint(2, 4)
    names = rng.sample(list(_words(style, "phase")), n)
    phrases = _pool(rng, _words(style, "phrase"), 16)
    cols = _node_colors(pal, rng, n)

    gap = 3.0
    cw = (92.0 - gap * (n - 1)) / n
    ch = min(top - bot, (top - bot) * 0.92)
    cy = bot + ((top - bot) - ch) / 2
    hdr = min(8.0, ch * 0.20)

    nodes, edges = 0, 0
    for i in range(n):
        x = 4 + i * (cw + gap)
        body = mix(cols[i], pal.background, 0.80)
        ax.add_patch(FancyBboxPatch(
            (x, cy), cw, ch, boxstyle="round,pad=0,rounding_size=1.2",
            facecolor=body, edgecolor=cols[i], linewidth=1.0))
        ax.add_patch(Rectangle((x, cy + ch - hdr), cw, hdr, facecolor=cols[i],
                               edgecolor="none"))
        ax.text(x + cw / 2, cy + ch - hdr / 2, names[i], ha="center", va="center",
                color=readable_on(cols[i]), **_fk(style, _fs(style, 1.00), True))
        nodes += 1

        k = rng.randint(2, 3)
        sh = min(9.0, (ch - hdr - 4.0) / k - 2.0)
        for j in range(k):
            sy = cy + ch - hdr - 3.0 - sh - j * (sh + 2.0)
            if sy < cy + 1.5:
                break
            _box(ax, x + 2.5, sy, cw - 5.0, sh, pal.background, cols[i], style,
                 phrases.pop(), pal.text,
                 _fs(style, min(0.80, cw * 0.042)), wrap_at=cw * 0.52)
            nodes += 1
        if i:
            _arrow(ax, (x - gap - 0.5, cy + ch / 2), (x - 0.5, cy + ch / 2),
                   pal.muted, 1.3, 10)
            edges += 1

    return nodes, edges


def _icon_step_chain(fig, ax, style, rng) -> tuple[int, int]:
    """Icon + caption per step, joined by drawn arrows."""
    pal = style.palette
    _canvas(fig, ax, pal)
    bot, top = _chrome(ax, style, rng)

    n = rng.randint(3, 5)
    labels = rng.sample(list(_words(style, "phrase")), n)
    kinds = rng.sample(list(_ICONS), n) if n <= len(_ICONS) else [rng.pick(_ICONS) for _ in range(n)]
    cols = _node_colors(pal, rng, n)

    mid = (top + bot) / 2
    step = 92.0 / n
    r = min(7.5, step * 0.26)
    filled = rng.chance(0.55)

    for i in range(n):
        cx = 4 + step * (i + 0.5)
        cy = mid + r * 0.6
        if filled:
            ax.add_patch(Polygon(_circle_pts(cx, cy, r * 1.35), facecolor=cols[i],
                                 edgecolor="none"))
            _icon(ax, cx, cy, r * 0.72, kinds[i], readable_on(cols[i]))
        else:
            ax.add_patch(Polygon(_circle_pts(cx, cy, r * 1.35), facecolor="none",
                                 edgecolor=cols[i], linewidth=1.3))
            _icon(ax, cx, cy, r * 0.72, kinds[i], cols[i])
        ax.text(cx, cy - r * 1.35 - 3.0, _wrap(labels[i], max(10, int(step * 0.55))),
                ha="center", va="top", color=pal.text, linespacing=1.2,
                **_fk(style, _fs(style, min(0.89, step * 0.042))))
        if i:
            px = 4 + step * (i - 0.5)
            _arrow(ax, (px + r * 1.35 + 1.0, cy), (cx - r * 1.35 - 1.0, cy),
                   pal.muted, 1.1, 9)
    return n, n - 1


def _value_chain(fig, ax, style, rng) -> tuple[int, int]:
    """A segmented right-pointing arrow -- the Porter value-chain look."""
    pal = style.palette
    _canvas(fig, ax, pal)
    bot, top = _chrome(ax, style, rng)
    labels = rng.sample(list(_words(style, "value")), rng.randint(4, 6))
    n = len(labels)
    cols = ramp(pal, n)
    h = min(22.0, (top - bot) * 0.45)
    y = (top + bot) / 2 - h / 2
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
        ax.text((xa + xb) / 2, y + h / 2, _wrap(lb, max(8, int(seg_w * 0.6))),
                ha="center", va="center", color=readable_on(cols[i]),
                linespacing=1.1, **_fk(style, _fs(style, min(1.11, seg_w * 0.061)), True))
    return n, n - 1


def _swimlane(fig, ax, style, rng) -> tuple[int, int]:
    """Boxes stepping between labelled horizontal lanes, joined by arrows."""
    pal = style.palette
    _canvas(fig, ax, pal)
    bot, top = _chrome(ax, style, rng)
    lanes = rng.sample(list(_words(style, "dept")), rng.randint(3, 5))
    nl = len(lanes)
    lh = (top - bot) / nl
    tab = mix(pal.color(0), pal.background, 0.30 if not pal.dark else 0.0)
    for r, lane in enumerate(lanes):
        ly = bot + (nl - 1 - r) * lh  # first lane on top
        band = mix(pal.color(0), pal.background, 0.10 if r % 2 == 0 else 0.03)
        ax.add_patch(Rectangle((14, ly), 82, lh, facecolor=band, edgecolor=pal.background, linewidth=1.0))
        ax.add_patch(Rectangle((4, ly), 10, lh, facecolor=tab, edgecolor="none"))
        ax.text(9, ly + lh / 2, lane, ha="center", va="center", rotation=90,
                color=readable_on(tab), **_fk(style, _fs(style, 0.90), True))
    steps = rng.sample(list(_words(style, "step")), rng.randint(4, 7))
    ns = len(steps)
    bw, bh = min(15.0, 74.0 / ns), min(lh * 0.5, 10)
    xs = np.linspace(20, 92 - bw, ns)
    lane_of = [rng.randint(0, nl - 1) for _ in range(ns)]
    centers = []
    for step, li, x in zip(steps, lane_of, xs):
        cy = bot + (nl - 1 - li) * lh + lh / 2
        fc = mix(pal.color(0), pal.background, 0.25 if not pal.dark else 0.0)
        _box(ax, x, cy - bh / 2, bw, bh, fc, pal.color(0), style, step,
             readable_on(fc), _fs(style, min(0.90, bw * 0.062)))
        centers.append((x, x + bw, cy))  # (left, right, cy)
    for i in range(ns - 1):
        # step down/up between lanes with a right-angled hop, like real swimlanes
        if abs(centers[i][2] - centers[i + 1][2]) > 1.0:
            _elbow_h(ax, (centers[i][1], centers[i][2]), (centers[i + 1][0], centers[i + 1][2]),
                     pal.muted, 1.0, arrow=True)
        else:
            _arrow(ax, (centers[i][1], centers[i][2]), (centers[i + 1][0], centers[i + 1][2]),
                   pal.muted, 1.1, 8)
    return ns, ns - 1


def _funnel(fig, ax, style, rng) -> tuple[int, int]:
    """Stacked, narrowing stages -- a conversion / sales funnel."""
    pal = style.palette
    _canvas(fig, ax, pal)
    bot, top = _chrome(ax, style, rng)
    labels = rng.sample(list(_words(style, "funnel")), rng.randint(3, 5))
    n = len(labels)
    cols = ramp(pal, n)
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
                color=readable_on(cols[i]), **_fk(style, _fs(style, 1.11), True))
    return n, n - 1


_DISPATCH = {
    "org_chart": _org_chart,
    "process_flow": _process,
    "decision_tree": _decision_tree,
    "swimlane": _swimlane,
    "value_chain": _value_chain,
    "nested_phase": _nested_phase,
    "branch_merge": _branch_merge,
    "icon_step_chain": _icon_step_chain,
    "funnel": _funnel,
}


def render(spec: FigureSpec, style: StyleSheet, rng: Rng, oversample: float) -> RenderResult:
    fig, ax = mpl.new_figure(style, oversample)
    st = spec.sub_type
    nodes, edges = _DISPATCH[st](fig, ax, style, rng)

    image = mpl.to_pil(fig)
    fig.clear()

    structure = Structure(flow_nodes=nodes, flow_edges=edges)
    meta = {"sub_kind": st, "nodes": nodes, "edges": edges, "legend": "none", "yaxis": False}
    return RenderResult(image=image, structure=structure, meta=meta)
