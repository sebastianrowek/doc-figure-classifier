"""
Bar charts: ``bar_vertical``, ``bar_horizontal`` and ``bar_stacked``.

Three tier-1 classes share this module because they share their geometry. The
labeling guide draws the lines between them precisely:

* orientation separates vertical from horizontal, semantics do not
* orientation is *not* distinguished for stacked -- "Orientierung wird hier
  nicht unterschieden"
* grouping is a property of the data, not a chart type, so grouped columns stay
  `bar_vertical`
* a reference line does not make a chart a combo, however it is captioned
  (guide v1.1)

Sub-types
---------
bar_vertical / bar_horizontal
    plain, grouped, highlight, target_line, average_line

bar_stacked
    plain, horizontal, pct100, red_green_signed, single_dominant

The last two are hard variants: a stack coloured green/red with signed labels
is the closest a stacked chart ever comes to looking like a bridge, and a stack
whose second segment is tiny is the closest it comes to looking like a plain
bar. Both are labelled by construction, which is the whole point of generating
them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .. import content, palettes, series
from ..engines import mpl
from ..palettes import readable_on
from ..rng import Rng
from ..style import StyleSheet
from . import common
from .base import FigureSpec, RenderResult, Structure
from .common import Layout


@dataclass
class _StackLayout(Layout):
    """Stacked charts need per-segment label metrics; see _draw_stacked_labels."""

    label_widths: list[float] = field(default_factory=list)
    axes_w_pt: float = 1.0
    axes_h_pt: float = 1.0

SUBTYPES_VERTICAL = {
    "plain": 0.45,
    "grouped": 0.20,
    "highlight": 0.12,
    "target_line": 0.13,
    "average_line": 0.10,
}

SUBTYPES_HORIZONTAL = {
    "plain": 0.50,
    "grouped": 0.18,
    "highlight": 0.14,
    "target_line": 0.10,
    "average_line": 0.08,
}

SUBTYPES_STACKED = {
    "plain": 0.34,
    "horizontal": 0.20,
    "pct100": 0.20,
    "red_green_signed": 0.14,
    "single_dominant": 0.12,
}


# --------------------------------------------------------------------------
# Specs
# --------------------------------------------------------------------------


def _n_categories(style: StyleSheet, rng: Rng, horizontal: bool) -> int:
    """
    Category count is bounded by the physical size of the crop.

    For columns the constraint is width; for horizontal bars it is height,
    because each bar needs a row of its own.
    """
    if horizontal:
        rows = style.fig_h_in / 0.24  # roughly one bar per quarter inch
        return max(3, min(rng.randint(4, 10), int(rows)))
    if style.fig_w_in < 2.2:
        return rng.randint(3, 5)
    if style.fig_w_in < 3.5:
        return rng.randint(4, 7)
    return rng.randint(4, 9)


def build_spec(sub_type: str, style: StyleSheet, rng: Rng) -> FigureSpec:
    """Content for a plain (vertical) bar chart."""
    return _build_plain_spec(sub_type, style, rng, "bar_vertical", horizontal=False)


def build_spec_horizontal(sub_type: str, style: StyleSheet, rng: Rng) -> FigureSpec:
    return _build_plain_spec(sub_type, style, rng, "bar_horizontal", horizontal=True)


def _build_plain_spec(
    sub_type: str, style: StyleSheet, rng: Rng, label: str, horizontal: bool
) -> FigureSpec:
    topic = content.pick_topic(rng, style.language)
    n_cat = _n_categories(style, rng, horizontal)

    n_series = 1
    if sub_type == "grouped":
        n_cat = max(3, n_cat - 2)
        n_series = rng.randint(2, 3)

    # Writing values on the bars costs room per bar. Budget for it here rather
    # than discovering the collision at render time -- dropping labels later
    # would quietly under-represent the axis-less look of rule R4.
    if style.data_labels != "none" and not horizontal:
        digits = len(content.format_number(topic.magnitude, topic.decimals, style.number_locale))
        chars = digits + (len(topic.unit) + 1 if style.data_labels == "value_unit" else 0)
        needed_pt = chars * style.label_pt * 0.58 + 5.0
        n_cat = max(3, min(n_cat, int(style.fig_w_in * 72.0 * 0.82 / (needed_pt * n_series))))

    cats = content.categories(rng, topic.category_kind, n_cat, style.language)
    n_cat = len(cats)

    # Net income and similar KPIs do go negative; the bars still start at the
    # baseline, so this stays a plain bar chart.
    allow_neg = rng.chance(0.08) and not topic.percent

    if n_series > 1:
        values = series.grouped_series(rng, n_cat, n_series, topic.magnitude)
        names = content.categories(rng, "year", n_series, style.language)
    elif topic.category_kind in ("year", "quarter"):
        values = [series.kpi_series(rng, n_cat, topic.magnitude, allow_neg)]
        names = [topic.title]
    else:
        values = [series.category_series(rng, n_cat, topic.magnitude)]
        names = [topic.title]

    # Horizontal bar charts are usually rankings; sorting them is the norm.
    if horizontal and n_series == 1 and rng.chance(0.65):
        order = sorted(range(n_cat), key=lambda i: values[0][i], reverse=True)
        cats = [cats[i] for i in order]
        values = [[values[0][i] for i in order]]

    dec = content.decimals_for(max(abs(v) for row in values for v in row), topic.decimals)
    values = [series.round_nicely(row, dec) for row in values]

    extra: dict = {}
    if sub_type == "highlight":
        extra["highlight"] = rng.randint(0, n_cat - 1) if rng.chance(0.4) else n_cat - 1
    elif sub_type in ("target_line", "average_line"):
        flat = [v for row in values for v in row]
        kind = "target" if sub_type == "target_line" else "average"
        extra["ref_value"] = (
            max(flat) * rng.uniform(1.02, 1.14) if kind == "target" else sum(flat) / len(flat)
        )
        extra["ref_kind"] = kind
        extra["ref_label"] = content.reference_label(rng, style.language, kind)

    return FigureSpec(
        label=label,
        sub_type=sub_type,
        title=topic.title if style.title_mode != "none" else None,
        subtitle=topic.subtitle if style.title_mode == "title_subtitle" else None,
        source=content.source_note(rng, style.language) if style.source_note else None,
        unit=topic.unit,
        categories=cats,
        series=values,
        series_names=names,
        decimals=dec,
        percent=topic.percent,
        extra=extra,
    )


def build_spec_stacked(sub_type: str, style: StyleSheet, rng: Rng) -> FigureSpec:
    """
    Content for a stacked chart.

    ``series`` holds one list per *segment*, each with one value per category --
    the transpose of what the plain builder produces, because that is the order
    a stack is drawn in.
    """
    horizontal = sub_type == "horizontal" or (sub_type != "plain" and rng.chance(0.25))
    topic = content.pick_topic(rng, style.language)

    n_cat = max(3, _n_categories(style, rng, horizontal) - 1)
    n_seg = 2 if sub_type in ("red_green_signed", "single_dominant") else rng.randint(2, 5)
    if sub_type == "pct100":
        n_seg = rng.randint(2, 4)

    # A stack's categories are the axis; its segments are the legend.
    cat_kind = topic.category_kind if topic.category_kind in ("year", "quarter") else "year"
    cats = content.categories(rng, cat_kind, n_cat, style.language)
    n_cat = len(cats)
    seg_names = content.categories(
        rng, "segment" if rng.chance(0.6) else "region", n_seg, style.language
    )
    n_seg = len(seg_names)

    # Segment values: correlated, so the composition stays plausible year to year.
    segs: list[list[float]] = []
    for si in range(n_seg):
        share = rng.uniform(0.5, 1.0) ** rng.uniform(1.0, 2.0)
        segs.append([v * share for v in series.kpi_series(rng, n_cat, topic.magnitude / n_seg)])

    if sub_type == "single_dominant":
        # One segment reduced to a sliver -- the case that reads as a plain bar.
        segs[1] = [v * rng.uniform(0.03, 0.09) for v in segs[0]]

    if sub_type == "pct100":
        totals = [sum(segs[si][ci] for si in range(n_seg)) for ci in range(n_cat)]
        segs = [[100.0 * segs[si][ci] / totals[ci] for ci in range(n_cat)] for si in range(n_seg)]

    dec = 1 if sub_type == "pct100" else content.decimals_for(
        max(abs(v) for row in segs for v in row), topic.decimals
    )
    segs = [series.round_nicely(row, dec) for row in segs]

    extra = {"horizontal": horizontal}
    if sub_type == "red_green_signed":
        # Green/red plus signed labels: everything a bridge looks like, on a
        # chart that is unambiguously a stack. See guide rule R3.
        extra["direction_colors"] = True
        extra["signed_labels"] = True
        seg_names = (
            ["Zunahme", "Abnahme"] if style.language == "de" else ["Increase", "Decrease"]
        )

    return FigureSpec(
        label="bar_stacked",
        sub_type=sub_type,
        title=topic.title if style.title_mode != "none" else None,
        subtitle=topic.subtitle if style.title_mode == "title_subtitle" else None,
        source=content.source_note(rng, style.language) if style.source_note else None,
        unit="%" if sub_type == "pct100" else topic.unit,
        categories=cats,
        series=segs,
        series_names=seg_names,
        decimals=dec,
        percent=sub_type == "pct100" or topic.percent,
        extra=extra,
    )


# --------------------------------------------------------------------------
# bar_vertical / bar_horizontal
# --------------------------------------------------------------------------


def render_bar_vertical(
    spec: FigureSpec, style: StyleSheet, rng: Rng, oversample: float
) -> RenderResult:
    return _render_plain(spec, style, rng, oversample, horizontal=False)


def render_bar_horizontal(
    spec: FigureSpec, style: StyleSheet, rng: Rng, oversample: float
) -> RenderResult:
    return _render_plain(spec, style, rng, oversample, horizontal=True)


def _render_plain(
    spec: FigureSpec, style: StyleSheet, rng: Rng, oversample: float, horizontal: bool
) -> RenderResult:
    pal = style.palette
    fig, ax = mpl.new_figure(style, oversample)

    n_cat = len(spec.categories)
    n_series = len(spec.series)
    pos = list(range(n_cat))
    value_axis = "x" if horizontal else "y"

    lay = _fit_plain(fig, spec, style, rng, n_series, horizontal)
    mpl.apply_margins(fig, style, _margins_plain(spec, style, lay, horizontal))

    # -- bars -------------------------------------------------------------
    each_w = style.bar_width / n_series
    containers = []
    for si, row in enumerate(spec.series):
        offset = (si - (n_series - 1) / 2) * each_w
        colors = [pal.color(si)] * n_cat
        if spec.sub_type == "highlight":
            colors[spec.extra["highlight"]] = pal.accent
        kwargs = dict(
            color=colors,
            edgecolor=pal.text if style.bar_edge else "none",
            linewidth=0.7 if style.bar_edge else 0.0,
            hatch=style.hatch,
            label=spec.series_names[si] if n_series > 1 else None,
            zorder=3,
        )
        offsets = [p + offset for p in pos]
        width = each_w * (0.98 if n_series > 1 else 1.0)
        containers.append(
            ax.barh(offsets, row, height=width, **kwargs)
            if horizontal
            else ax.bar(offsets, row, width=width, **kwargs)
        )

    # -- reference line (NOT a data series -- guide v1.1) ------------------
    ref_in_legend = False
    if "ref_value" in spec.extra:
        ref_in_legend = lay.legend != "none" and rng.chance(0.5)
        line_fn = ax.axvline if horizontal else ax.axhline
        line_fn(
            spec.extra["ref_value"],
            color=pal.accent if spec.extra["ref_kind"] == "target" else pal.muted,
            linestyle=rng.pick(("--", ":", "-.")),
            linewidth=rng.uniform(0.9, 1.6),
            zorder=4,
            label=spec.extra["ref_label"] if ref_in_legend else None,
        )

    # -- scale ------------------------------------------------------------
    flat = [v for row in spec.series for v in row]
    vmax, vmin = _vmax(spec), min(flat)
    headroom = 1.26 if lay.data_labels and style.label_pos == "outside" else 1.08
    lo, hi = min(0.0, vmin * 1.25), (vmax * headroom if vmax > 0 else vmax * 0.9)
    pad = 0.1 if n_series > 1 else 0.0

    if horizontal:
        ax.set_xlim(lo, hi)
        ax.set_ylim(-0.5 - pad, n_cat - 0.5 + pad)
        ax.invert_yaxis()  # rankings read top-down
        ax.set_yticks(pos)
        ax.set_yticklabels(
            spec.categories, color=pal.muted, **mpl.font_kwargs(style, lay.tick_pt)
        )
    else:
        ax.set_ylim(lo, hi)
        ax.set_xlim(-0.5 - pad, n_cat - 0.5 + pad)
        ax.set_xticks(pos)
        ax.set_xticklabels(
            spec.categories,
            rotation=lay.tick_rotation,
            ha="right" if lay.tick_rotation not in (0.0, 90.0) else "center",
            color=pal.muted,
            **mpl.font_kwargs(style, lay.tick_pt),
        )

    # The reference caption is placed after the scale is known: a target line
    # sits near the top by construction, and captioning it beyond the line runs
    # it into the title or off the edge.
    if "ref_value" in spec.extra and not ref_in_legend and lay.ref_text:
        ref = spec.extra["ref_value"]
        near_end = (ref - lo) / (hi - lo) > 0.86
        if horizontal:
            ax.text(
                ref, -0.45, spec.extra["ref_label"],
                ha="right" if near_end else "left", va="bottom", color=pal.muted,
                **mpl.font_kwargs(style, style.label_pt * 0.95),
            )
        else:
            ax.text(
                n_cat - 0.45, ref, spec.extra["ref_label"],
                ha="right", va="top" if near_end else "bottom", color=pal.muted,
                **mpl.font_kwargs(style, style.label_pt * 0.95),
            )

    common.apply_spines_grid(ax, style, value_axis)
    common.apply_value_axis(ax, style, lay, spec, value_axis)
    common.apply_category_axis(ax, style, lay, "y" if horizontal else "x")

    if lay.data_labels:
        _draw_bar_labels(ax, containers, spec, style, lay, n_cat)

    if lay.legend != "none" and (n_series > 1 or ref_in_legend):
        common.draw_legend(ax, style, lay)

    common.draw_headings(fig, spec, style, lay)

    image = mpl.to_pil(fig)
    fig.clear()

    structure = Structure(
        bar_series=n_series,
        bars_total=n_cat * n_series,
        segments_per_bar=1,
        orientation="horizontal" if horizontal else "vertical",
        bars_on_baseline=n_cat * n_series,
        bars_floating=0,
        line_series=1 if "ref_value" in spec.extra else 0,
        line_is_reference=True,
        line_in_legend=ref_in_legend,
    )
    return RenderResult(image=image, structure=structure, meta=_meta(lay, style, n_cat, n_series))


def _fit_plain(
    fig, spec: FigureSpec, style: StyleSheet, rng: Rng, n_series: int, horizontal: bool
) -> Layout:
    n_cat = len(spec.categories)
    lay = Layout(legend=common.maybe_force_legend(style, rng, n_series))
    lay.title_lines, lay.title_pt = common.fit_title(fig, spec, style)
    lay.subtitle, lay.subtitle_pt = common.fit_subtitle(fig, spec, style)
    common.cap_title_block(style, lay)
    lay.tick_pt, lay.tick_rotation = style.tick_pt, style.tick_rotation
    lay.label_pt, lay.value_axis = style.label_pt, style.yaxis

    entries = list(spec.series_names[:n_series]) if n_series > 1 else []
    if "ref_label" in spec.extra:
        entries = entries + [spec.extra["ref_label"]]
    common.fit_legend(fig, entries, style, lay)
    axes_w = mpl.axes_width_pt(style, _margins_plain(spec, style, lay, horizontal))
    # Second pass: a top/bottom legend is anchored to the axes, so it has to be
    # sized against the axes width, which is only known once margins exist.
    common.fit_legend(fig, entries, style, lay, axes_w)
    axes_w = mpl.axes_width_pt(style, _margins_plain(spec, style, lay, horizontal))

    if horizontal:
        # Category labels sit to the left; the constraint is the margin width,
        # which _margins_plain already sized from a measurement.
        lay.tick_pt = style.tick_pt
    else:
        lay.tick_pt, lay.tick_rotation = common.fit_category_ticks(
            fig, spec.categories, style, axes_w / max(1, n_cat)
        )

    if style.data_labels != "none" and n_cat * n_series <= 12:
        texts = common.value_label_texts(
            [v for row in spec.series for v in row], spec, style
        )
        # A horizontal bar's label sits beyond its end, so horizontally it
        # competes with the plot width once, not once per category.
        slot = axes_w * 0.22 if horizontal else axes_w / (n_cat * n_series)
        lay.data_labels, lay.label_pt = common.fit_value_labels(fig, texts, style, slot)

        # Horizontally the binding constraint is vertical: with three grouped
        # series the bars are a third of a row high, and the labels stack on
        # top of each other however much width is available.
        if horizontal and lay.data_labels:
            margins = _margins_plain(spec, style, lay, horizontal)
            axes_h = max(1.0, style.fig_h_in * 72.0 - margins.top - margins.bottom)
            per_bar = axes_h / (n_cat * n_series)
            if per_bar < lay.label_pt * 1.25:
                lay.data_labels = None

        if lay.data_labels is None:
            # Neither a numeric axis nor value labels is unreadable; the
            # corporate fallback is to put the axis back.
            lay.value_axis = True

    if "ref_value" in spec.extra:
        w = mpl.text_width_pt(fig, spec.extra["ref_label"], style, style.label_pt * 0.95)
        lay.ref_text = w < axes_w * 0.4
    return lay


def _margins_plain(
    spec: FigureSpec, style: StyleSheet, lay: Layout, horizontal: bool
) -> mpl.Margins:
    """
    Margins in points, computed from what is actually being drawn.

    tight_layout is not usable here: it ignores bar_label annotations, which on
    an axis-less corporate chart are most of the ink.
    """
    m = mpl.Margins()
    m.top = common.top_margin(style, lay)
    m.right = common.right_margin(style, lay, spec)
    bottom = 6.0 + common.bottom_extra(style, lay, spec)

    if horizontal:
        # Category labels go in the left margin; size it to the longest one.
        longest = max((len(c) for c in spec.categories), default=6)
        m.left = 8.0 + lay.tick_pt * 0.56 * min(longest, 20)
        bottom += lay.tick_pt * 1.7 if lay.value_axis else 2.0
        # Value labels sit beyond the bar ends, so they need room on the right.
        if lay.data_labels:
            m.right += lay.label_pt * 3.0
    else:
        bottom += common.category_axis_height(lay, spec.categories)
        if lay.value_axis:
            width = len(content.format_number(_vmax(spec), spec.decimals, style.number_locale))
            m.left = 8.0 + lay.tick_pt * 0.62 * width
        else:
            m.left = 7.0

    m.bottom = bottom
    return m


def _draw_bar_labels(ax, containers, spec, style, lay: Layout, n_cat: int) -> None:
    pal = style.palette
    inside = style.label_pos == "inside"
    for si, container in enumerate(containers):
        ax.bar_label(
            container,
            labels=lay.data_labels[si * n_cat : (si + 1) * n_cat],
            label_type="center" if inside else "edge",
            padding=0 if inside else 2,
            color=readable_on(pal.color(si)) if inside else pal.text,
            fontfamily=style.font_family,
            fontsize=lay.label_pt,
            fontweight="bold" if style.bold_labels else "normal",
        )


# --------------------------------------------------------------------------
# bar_stacked
# --------------------------------------------------------------------------


def render_bar_stacked(
    spec: FigureSpec, style: StyleSheet, rng: Rng, oversample: float
) -> RenderResult:
    pal = style.palette
    horizontal = bool(spec.extra.get("horizontal"))
    fig, ax = mpl.new_figure(style, oversample)

    n_cat = len(spec.categories)
    n_seg = len(spec.series)
    pos = list(range(n_cat))
    value_axis = "x" if horizontal else "y"

    lay = _fit_stacked(fig, spec, style, rng, horizontal)
    mpl.apply_margins(fig, style, _margins_stacked(spec, style, lay, horizontal))

    if spec.extra.get("direction_colors"):
        seg_colors = [pal.positive, pal.negative]
    else:
        # ramp(), not color(): a cycling palette makes segment 1 and segment 4
        # identical and the boundary between them vanish.
        seg_colors = palettes.ramp(pal, n_seg)

    # -- segments ---------------------------------------------------------
    base = [0.0] * n_cat
    containers = []
    for si, row in enumerate(spec.series):
        kwargs = dict(
            color=seg_colors[si % len(seg_colors)],
            edgecolor=pal.background if style.bar_edge else "none",
            linewidth=0.6 if style.bar_edge else 0.0,
            label=spec.series_names[si],
            zorder=3,
        )
        containers.append(
            ax.barh(pos, row, left=base, height=style.bar_width, **kwargs)
            if horizontal
            else ax.bar(pos, row, bottom=base, width=style.bar_width, **kwargs)
        )
        base = [b + v for b, v in zip(base, row)]

    totals = base
    hi = max(totals) * (1.10 if lay.data_labels else 1.05)

    if horizontal:
        ax.set_xlim(0, hi)
        ax.set_ylim(-0.6, n_cat - 0.4)
        ax.invert_yaxis()
        ax.set_yticks(pos)
        ax.set_yticklabels(
            spec.categories, color=pal.muted, **mpl.font_kwargs(style, lay.tick_pt)
        )
    else:
        ax.set_ylim(0, hi)
        ax.set_xlim(-0.6, n_cat - 0.4)
        ax.set_xticks(pos)
        ax.set_xticklabels(
            spec.categories,
            rotation=lay.tick_rotation,
            ha="right" if lay.tick_rotation not in (0.0, 90.0) else "center",
            color=pal.muted,
            **mpl.font_kwargs(style, lay.tick_pt),
        )

    common.apply_spines_grid(ax, style, value_axis)
    common.apply_value_axis(ax, style, lay, spec, value_axis)
    common.apply_category_axis(ax, style, lay, "y" if horizontal else "x")

    # -- segment labels ---------------------------------------------------
    if lay.data_labels:
        _draw_stacked_labels(ax, containers, spec, style, lay, seg_colors, totals)

    if lay.legend != "none":
        common.draw_legend(ax, style, lay, reverse=not horizontal)

    common.draw_headings(fig, spec, style, lay)

    image = mpl.to_pil(fig)
    fig.clear()

    structure = Structure(
        bar_series=n_seg,
        bars_total=n_cat,
        segments_per_bar=n_seg,
        orientation="horizontal" if horizontal else "vertical",
        bars_on_baseline=n_cat,
        bars_floating=0,
        connectors=False,  # a stack never has them (guide, R3)
    )
    meta = _meta(lay, style, n_cat, n_seg)
    meta["stack_orientation"] = "horizontal" if horizontal else "vertical"
    return RenderResult(image=image, structure=structure, meta=meta)


def _fit_stacked(fig, spec: FigureSpec, style: StyleSheet, rng: Rng, horizontal: bool) -> Layout:
    n_cat = len(spec.categories)
    # A stack is unreadable without a legend; force one far more often than
    # for a grouped chart.
    lay = _StackLayout(legend=common.maybe_force_legend(style, rng, 2, p=0.9))
    lay.title_lines, lay.title_pt = common.fit_title(fig, spec, style)
    lay.subtitle, lay.subtitle_pt = common.fit_subtitle(fig, spec, style)
    common.cap_title_block(style, lay)
    lay.tick_pt, lay.tick_rotation = style.tick_pt, style.tick_rotation
    lay.label_pt, lay.value_axis = style.label_pt, style.yaxis
    common.fit_legend(fig, list(spec.series_names), style, lay)
    margins = _margins_stacked(spec, style, lay, horizontal)
    common.fit_legend(fig, list(spec.series_names), style, lay, mpl.axes_width_pt(style, margins))
    margins = _margins_stacked(spec, style, lay, horizontal)
    axes_w = mpl.axes_width_pt(style, margins)
    if not horizontal:
        lay.tick_pt, lay.tick_rotation = common.fit_category_ticks(
            fig, spec.categories, style, axes_w / max(1, n_cat)
        )

    # Labels go inside the segments, so the constraint is the bar width, and
    # every segment has to be big enough to hold a line of text.
    if style.data_labels != "none" and n_cat <= 8:
        texts = common.value_label_texts(
            [v for row in spec.series for v in row],
            spec,
            style,
            signed=bool(spec.extra.get("signed_labels")),
        )
        slot = (axes_w * 0.3) if horizontal else (axes_w / n_cat) * style.bar_width
        lay.data_labels, lay.label_pt = common.fit_value_labels(fig, texts, style, slot)
        if lay.data_labels is None:
            lay.value_axis = True
        else:
            # Which segments can actually hold their label is decided per
            # segment at draw time, and the test differs by orientation: a
            # vertical segment needs to be tall enough for a line of text, a
            # horizontal one wide enough for the whole string.
            lay.label_widths = [
                mpl.text_width_pt(fig, t, style, lay.label_pt, style.bold_labels)
                for t in lay.data_labels
            ]
            lay.axes_w_pt = axes_w
            lay.axes_h_pt = max(
                1.0, style.fig_h_in * 72.0 - margins.top - margins.bottom
            )
    return lay


def _margins_stacked(
    spec: FigureSpec, style: StyleSheet, lay: Layout, horizontal: bool
) -> mpl.Margins:
    m = mpl.Margins()
    m.top = common.top_margin(style, lay)
    m.right = common.right_margin(style, lay, spec)
    bottom = 6.0 + common.bottom_extra(style, lay, spec)

    if horizontal:
        longest = max((len(c) for c in spec.categories), default=6)
        m.left = 8.0 + lay.tick_pt * 0.56 * min(longest, 20)
        bottom += lay.tick_pt * 1.7 if lay.value_axis else 2.0
    else:
        bottom += common.category_axis_height(lay, spec.categories)
        if lay.value_axis:
            total = max(sum(col) for col in zip(*spec.series))
            width = len(content.format_number(total, spec.decimals, style.number_locale))
            m.left = 8.0 + lay.tick_pt * 0.62 * width
        else:
            m.left = 7.0

    m.bottom = bottom
    return m


def _draw_stacked_labels(ax, containers, spec, style, lay: _StackLayout, seg_colors, totals) -> None:
    """
    Label a segment only if it is big enough to hold the text.

    Small segments are the norm in a stack, and matplotlib will happily write a
    label straight across the boundary into the neighbouring colour. The test
    differs by orientation: a vertical segment must be tall enough for a line
    of text, a horizontal one wide enough for the whole string -- which is
    several times more demanding, and was what made horizontal stacks collide.
    """
    n_cat = len(spec.categories)
    horizontal = bool(spec.extra.get("horizontal"))
    span = max(totals) if totals else 1.0
    extent = lay.axes_w_pt if horizontal else lay.axes_h_pt

    for si, container in enumerate(containers):
        labels = []
        for ci, v in enumerate(spec.series[si]):
            i = si * n_cat + ci
            needed_pt = (lay.label_widths[i] + 5.0) if horizontal else (lay.label_pt * 1.9)
            min_value = needed_pt / max(1.0, extent) * span
            labels.append(lay.data_labels[i] if abs(v) >= min_value else "")
        ax.bar_label(
            container,
            labels=labels,
            label_type="center",
            color=readable_on(seg_colors[si % len(seg_colors)]),
            fontfamily=style.font_family,
            fontsize=lay.label_pt,
            fontweight="bold" if style.bold_labels else "normal",
        )


# --------------------------------------------------------------------------
# Shared
# --------------------------------------------------------------------------


def _vmax(spec: FigureSpec) -> float:
    vmax = max(v for row in spec.series for v in row)
    return max(vmax, spec.extra.get("ref_value", vmax))


def _meta(lay: Layout, style: StyleSheet, n_cat: int, n_series: int) -> dict:
    return {
        "n_categories": n_cat,
        "n_series": n_series,
        "yaxis": lay.value_axis,
        "data_labels": style.data_labels if lay.data_labels else "none",
        "tick_rotation": lay.tick_rotation,
        "legend": lay.legend,
        "title_wrapped": len(lay.title_lines) > 1,
    }
