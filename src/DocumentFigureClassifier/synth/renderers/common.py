"""
Layout machinery shared by every chart renderer.

Extracted from bars.py once the second renderer needed it. The important part
is the fitting discipline established in phase 0: no decision that can produce
overlapping text is made from a character-count estimate. Fonts are sampled
across 30 families whose widths differ by nearly a factor of two, so every such
decision measures the actual string.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .. import content
from ..engines import mpl
from ..rng import Rng
from ..style import StyleSheet
from .base import FigureSpec


@dataclass
class Layout:
    """Everything the fitting pass decided, in points."""

    title_lines: list[str] = field(default_factory=list)
    title_pt: float = 10.0
    subtitle: str | None = None
    subtitle_pt: float = 8.0
    tick_rotation: float = 0.0
    tick_pt: float = 8.0
    data_labels: list[str] | None = None  # None -> do not draw
    label_pt: float = 8.0
    value_axis: bool = True  # the numeric axis (y for columns, x for bars)
    legend: str = "none"
    legend_ncol: int = 1
    legend_rows: int = 0
    legend_w_pt: float = 0.0
    ref_text: bool = False


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------


def fit_title(fig, spec: FigureSpec, style: StyleSheet) -> tuple[list[str], float]:
    """Shrink the title to fit the figure width, then wrap it if that is not enough."""
    if not spec.title:
        return [], style.title_pt

    budget = style.fig_w_in * 72.0 * 0.94
    pt = style.title_pt
    w = mpl.text_width_pt(fig, spec.title, style, pt, style.bold_title)
    if w > budget:
        pt = max(style.base_pt * 0.95, pt * budget / w)
        w = mpl.text_width_pt(fig, spec.title, style, pt, style.bold_title)
    if w <= budget:
        return [spec.title], pt
    return wrap(fig, spec.title, style, pt, budget, style.bold_title), pt


def fit_subtitle(fig, spec: FigureSpec, style: StyleSheet) -> tuple[str | None, float]:
    """Shrink; drop it entirely rather than wrap -- a two-line subtitle is rarer."""
    if not spec.subtitle:
        return None, style.base_pt * 0.95

    budget = style.fig_w_in * 72.0 * 0.94
    pt = style.base_pt * 0.95
    w = mpl.text_width_pt(fig, spec.subtitle, style, pt)
    if w > budget:
        pt = pt * budget / w
        if pt < style.base_pt * 0.68:
            return None, pt
    return spec.subtitle, pt


def wrap(fig, text: str, style: StyleSheet, pt: float, budget: float, bold: bool) -> list[str]:
    """Break a string into at most two lines at the most balanced word boundary."""
    words = text.split()
    if len(words) < 2:
        return [text]
    best, best_cost = None, float("inf")
    for i in range(1, len(words)):
        a, b = " ".join(words[:i]), " ".join(words[i:])
        wa = mpl.text_width_pt(fig, a, style, pt, bold)
        wb = mpl.text_width_pt(fig, b, style, pt, bold)
        cost = max(wa, wb) + abs(wa - wb) * 0.25
        if cost < best_cost:
            best, best_cost = [a, b], cost
    return best or [text]


def fit_category_ticks(
    fig, categories: list[str], style: StyleSheet, slot_pt: float
) -> tuple[float, float]:
    """
    Return (tick_pt, rotation) for category labels sharing ``slot_pt`` each.

    Shrink first, rotate only if that is not enough -- which is the order a
    designer would use, and keeps rotated labels from dominating the dataset.
    """
    tick_pt, rotation = style.tick_pt, style.tick_rotation
    widest = max(mpl.text_width_pt(fig, c, style, tick_pt) for c in categories)
    if rotation == 0.0 and widest > slot_pt * 0.92:
        tick_pt = max(style.tick_pt * 0.75, style.tick_pt * slot_pt * 0.92 / widest)
        widest = max(mpl.text_width_pt(fig, c, style, tick_pt) for c in categories)
        if widest > slot_pt * 0.92:
            rotation = 45.0 if widest * 0.71 <= slot_pt * 0.92 else 90.0
    return tick_pt, rotation


def fit_value_labels(
    fig, labels: list[str], style: StyleSheet, slot_pt: float
) -> tuple[list[str] | None, float]:
    """Return (labels, size) or (None, size) if they cannot be made to fit."""
    pt = style.label_pt
    widest = max(mpl.text_width_pt(fig, t, style, pt, style.bold_labels) for t in labels)
    if widest > slot_pt * 0.9:
        pt = max(style.label_pt * 0.78, style.label_pt * slot_pt * 0.9 / widest)
        widest = max(mpl.text_width_pt(fig, t, style, pt, style.bold_labels) for t in labels)
    return (None if widest > slot_pt * 0.9 else labels), pt


def value_label_texts(
    values: list[float], spec: FigureSpec, style: StyleSheet, signed: bool = False
) -> list[str]:
    out = []
    for v in values:
        txt = content.format_number(v, spec.decimals, style.number_locale)
        if signed and v > 0:
            txt = f"+{txt}"
        if style.data_labels == "value_unit":
            txt = f"{txt} %" if spec.percent else f"{txt} {spec.unit}"
        elif spec.percent:
            txt = f"{txt} %"
        out.append(txt)
    return out


def fit_legend(
    fig, names: list[str], style: StyleSheet, lay: Layout, avail_w_pt: float | None = None
) -> None:
    """
    Decide how many columns the legend gets, and how much room it needs.

    A fixed ``ncol=4`` is what produced legends running off the right edge and
    legends silently wrapping onto a second row the top margin had not
    reserved. Both are measured here instead.

    ``avail_w_pt`` must be the **axes** width, not the figure width: a top or
    bottom legend is anchored to the axes, so sizing it against the figure
    overflows by exactly the horizontal margins. Callers pass a rough figure
    estimate on the first pass and the real axes width on the second.
    """
    if lay.legend == "none" or not names:
        lay.legend_ncol, lay.legend_rows, lay.legend_w_pt = 1, 0, 0.0
        return

    # Handle plus gap; matplotlib's defaults at handlelength=1.4.
    widest = max(
        mpl.text_width_pt(fig, n, style, lay.label_pt) + lay.label_pt * 2.8 for n in names
    )

    if lay.legend == "right":
        lay.legend_ncol, lay.legend_rows = 1, len(names)
        lay.legend_w_pt = widest
        return

    avail = avail_w_pt if avail_w_pt is not None else style.fig_w_in * 72.0 * 0.8
    lay.legend_ncol = max(1, min(len(names), int(avail / widest)))
    lay.legend_rows = -(-len(names) // lay.legend_ncol)  # ceil
    lay.legend_w_pt = 0.0


def cap_title_block(style: StyleSheet, lay: Layout, max_frac: float = 0.34) -> None:
    """
    Keep the heading block from eating the plot.

    ``apply_margins`` clamps margins that would leave no axes, and the clamp is
    silent -- the visible result is a two-line title drawn straight over the
    bars on a short crop. Shrinking the heading here instead means the clamp
    never fires. Dropping the title entirely is an acceptable last resort:
    untitled charts are already a fifth of the distribution.
    """
    limit = style.fig_h_in * 72.0 * max_frac
    for _ in range(14):
        if top_margin(style, lay) <= limit or not lay.title_lines:
            return
        if lay.subtitle:
            lay.subtitle = None
            continue
        if lay.title_pt > style.base_pt * 0.8:
            lay.title_pt *= 0.9
            continue
        lay.title_lines = []
        return


def legend_block_pt(lay: Layout) -> float:
    """Vertical room a top/bottom legend needs, in points."""
    if lay.legend not in ("top", "bottom") or not lay.legend_rows:
        return 0.0
    return lay.label_pt * (1.55 * lay.legend_rows + 0.8)


# --------------------------------------------------------------------------
# Margins
# --------------------------------------------------------------------------


def top_margin(style: StyleSheet, lay: Layout) -> float:
    top = 6.0
    if lay.title_lines:
        top += lay.title_pt * 1.35 * len(lay.title_lines)
    if lay.subtitle:
        top += lay.subtitle_pt * 1.4
    if lay.title_lines or lay.subtitle:
        top += 5.0
    if lay.legend == "top":
        top += legend_block_pt(lay)
    return top


def bottom_extra(style: StyleSheet, lay: Layout, spec: FigureSpec) -> float:
    """Everything below the plot that is not the category axis."""
    extra = 0.0
    if spec.source:
        extra += style.base_pt * 1.9
    if lay.legend == "bottom":
        extra += legend_block_pt(lay) + lay.label_pt * 0.8
    return extra


def category_axis_height(lay: Layout, categories: list[str]) -> float:
    """Vertical room needed by category labels below a plot, in points."""
    rot_factor = {0.0: 1.7, 30.0: 2.9, 45.0: 3.4, 90.0: 4.6}.get(lay.tick_rotation, 2.5)
    longest = max((len(c) for c in categories), default=4)
    h = lay.tick_pt * rot_factor
    # The caps used to be 12/14 characters, which is fine for years and short
    # segment names but truncates the reason-for-change labels on a bridge --
    # "Organisches Wachstum" is 20.
    if lay.tick_rotation in (30.0, 45.0):
        h += lay.tick_pt * 0.32 * min(longest, 16)
    elif lay.tick_rotation == 90.0:
        h += lay.tick_pt * 0.52 * min(longest, 20)
    return h


def right_margin(style: StyleSheet, lay: Layout, spec: FigureSpec) -> float:
    m = 8.0
    if lay.legend == "right":
        # Measured in fit_legend; the estimate is only a fallback for callers
        # that compute margins before fitting.
        m += lay.legend_w_pt or lay.label_pt * (
            0.6 * min(max((len(n) for n in spec.series_names), default=6), 14) + 2.0
        )
    return m


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------


def apply_spines_grid(ax, style: StyleSheet, value_axis: str) -> None:
    """
    Spines and gridlines. Grid runs perpendicular to the value axis, which is
    what makes it readable -- horizontal rules for columns, vertical for bars.
    """
    pal = style.palette
    visible = {
        "all": ("left", "bottom", "top", "right"),
        "left_bottom": ("left", "bottom"),
        "bottom": ("bottom",),
        "none": (),
    }[style.spines]
    for name, spine in ax.spines.items():
        spine.set_visible(name in visible)
        spine.set_color(pal.muted)
        spine.set_linewidth(0.8)

    if style.grid != "none":
        ax.grid(
            axis="both" if style.grid == "full" else value_axis,
            color=pal.grid,
            linewidth=0.6,
            linestyle="--" if style.grid == "h_dashed" else "-",
            zorder=0,
        )
        ax.set_axisbelow(True)


def apply_value_axis(ax, style: StyleSheet, lay: Layout, spec: FigureSpec, axis: str) -> None:
    """Locale-formatted numeric ticks, or none at all (rule R4)."""
    pal = style.palette
    target = ax.yaxis if axis == "y" else ax.xaxis

    if not lay.value_axis:
        # Rule R4: corporate design deletes the numeric axis and writes the
        # values on the bars instead. This must be common in training.
        target.set_ticks([])
        return

    first = spec.series[0][0] if spec.series and spec.series[0] else 0.0
    dec = 0 if abs(first) >= 100 else spec.decimals
    loc = style.number_locale
    target.set_major_formatter(
        mpl.FuncFormatter(lambda v, _pos: content.format_number(v, dec, loc))
    )
    ax.tick_params(axis=axis, colors=pal.muted, labelsize=lay.tick_pt, length=3)
    labels = ax.get_yticklabels() if axis == "y" else ax.get_xticklabels()
    for lbl in labels:
        lbl.set_fontfamily(style.font_family)


def apply_category_axis(ax, style: StyleSheet, lay: Layout, axis: str) -> None:
    ax.tick_params(
        axis=axis,
        colors=style.palette.muted,
        labelsize=lay.tick_pt,
        length=0 if style.spines == "none" else 3,
    )


def draw_legend(ax, style: StyleSheet, lay: Layout, reverse: bool = False) -> None:
    pal = style.palette
    loc, anchor = {
        "top": ("lower left", (0.0, 1.01)),
        "bottom": ("upper left", (0.0, -0.14)),
        "right": ("center left", (1.02, 0.5)),
    }[lay.legend]
    ncol = max(1, lay.legend_ncol)

    handles, labels = ax.get_legend_handles_labels()
    if reverse:  # stacked charts read top-down; the legend should match
        handles, labels = handles[::-1], labels[::-1]
    leg = ax.legend(
        handles,
        labels,
        loc=loc,
        bbox_to_anchor=anchor,
        ncol=ncol,
        frameon=False,
        handlelength=1.4,
        columnspacing=1.2,
        prop={"family": style.font_family, "size": lay.label_pt},
    )
    for text in leg.get_texts():
        text.set_color(pal.muted)


def draw_headings(fig, spec: FigureSpec, style: StyleSheet, lay: Layout) -> None:
    """Title, subtitle and source in figure coordinates, top-down."""
    pal = style.palette
    # 3.5 % rather than 2 %: the degradation crop takes a couple of percent off
    # each side, and a title starting at 2 % loses its first letter on most
    # samples.
    x = 0.035 if style.title_align == "left" else 0.5
    ha = "left" if style.title_align == "left" else "center"
    y = 1.0 - mpl.frac_h(5.0, style)

    for line in lay.title_lines:
        fig.text(
            x, y, line, ha=ha, va="top", color=pal.text,
            **mpl.font_kwargs(style, lay.title_pt, style.bold_title),
        )
        y -= mpl.frac_h(lay.title_pt * 1.3, style)

    if lay.subtitle:
        fig.text(
            x, y, lay.subtitle, ha=ha, va="top", color=pal.muted,
            **mpl.font_kwargs(style, lay.subtitle_pt),
        )

    if spec.source:
        fig.text(
            0.035, mpl.frac_h(4.0, style), spec.source,
            ha="left", va="bottom", color=pal.muted,
            **mpl.font_kwargs(style, style.base_pt * 0.85),
        )


def maybe_force_legend(style: StyleSheet, rng: Rng, n_series: int, p: float = 0.7) -> str:
    """Multi-series charts are unreadable without a legend often enough to force one."""
    if n_series > 1 and style.legend == "none" and rng.chance(p):
        return rng.pick(("top", "bottom", "right"))
    return style.legend
