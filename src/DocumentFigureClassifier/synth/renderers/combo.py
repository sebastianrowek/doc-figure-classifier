"""
Combo charts: bars and a line as substantive data series in one plot.

Decision-tree step 5 puts this class *before* waterfall, stacked and plain bars,
so anything with both marks lands here regardless of what the bars look like.
That is why ``stacked_bars`` is a sub-type of combo and not of bar_stacked.

The boundary against a reference line is the one the guide sharpened in v1.1:
the test is whether the line **varies across the categories**, not whether it
has a legend entry. So:

    plain           revenue columns + margin line on a secondary axis
    flat_line       a margin that moves by a couple of tenths -- visually almost
                    a reference line, but it has real per-category values, its
                    own axis and its own legend entry. The hard variant.
    stacked_bars    stacked columns + line; step 5 beats step 8
    same_axis       the line shares the primary scale (no secondary axis), so
                    the legend is what identifies it as a series
    markers_line    the line carries point markers

The mirror image of ``flat_line`` lives in bars.py as ``target_line`` and
``average_line``: genuinely constant lines over bars, labelled ``bar``.
Between them the two classes cover both sides of the confusion.
"""

from __future__ import annotations

from dataclasses import dataclass

from .. import content, palettes, series
from ..engines import mpl
from ..rng import Rng
from ..style import StyleSheet
from . import common
from .base import FigureSpec, RenderResult, Structure
from .common import Layout

SUBTYPES = {
    "plain": 0.36,
    "flat_line": 0.20,
    "stacked_bars": 0.18,
    "same_axis": 0.14,
    "markers_line": 0.12,
}

# (bar title, bar unit, bar magnitude, line name, line unit, line magnitude, line decimals)
_PAIRS_DE = (
    ("Umsatz", "Mio. €", 1800, "EBIT-Marge", "%", 11.5, 1),
    ("Konzernumsatz", "Mio. €", 4200, "Umsatzrendite", "%", 8.4, 1),
    ("Absatzmenge", "Tsd. Stück", 640, "Durchschnittspreis", "€", 128.0, 0),
    ("Investitionen", "Mio. €", 180, "Investitionsquote", "%", 6.2, 1),
    ("EBITDA", "Mio. €", 540, "EBITDA-Marge", "%", 17.2, 1),
    ("F&E-Aufwendungen", "Mio. €", 145, "F&E-Quote", "%", 4.8, 1),
    ("Mitarbeiter", "Mitarbeiter", 8600, "Fluktuationsrate", "%", 7.4, 1),
    ("Auftragseingang", "Mio. €", 1150, "Book-to-Bill", "x", 1.05, 2),
    ("CO2-Emissionen", "kt CO2e", 420, "Emissionsintensität", "t/Mio. €", 0.23, 2),
    ("Nettofinanzschulden", "Mio. €", 640, "Verschuldungsgrad", "x", 1.6, 1),
)
_PAIRS_EN = (
    ("Revenue", "€ m", 1800, "EBIT margin", "%", 11.5, 1),
    ("Group revenue", "€ m", 4200, "Return on sales", "%", 8.4, 1),
    ("Sales volume", "k units", 640, "Average price", "€", 128.0, 0),
    ("Capital expenditure", "€ m", 180, "Capex ratio", "%", 6.2, 1),
    ("EBITDA", "€ m", 540, "EBITDA margin", "%", 17.2, 1),
    ("Order intake", "€ m", 1150, "Book-to-bill", "x", 1.05, 2),
    ("Net financial debt", "€ m", 640, "Leverage", "x", 1.6, 1),
)

_MARKERS = ("o", "s", "D", "^")


@dataclass
class _ComboLayout(Layout):
    secondary_axis: bool = True
    line_labels: list[str] | None = None


# --------------------------------------------------------------------------
# Spec
# --------------------------------------------------------------------------


def build_spec(sub_type: str, style: StyleSheet, rng: Rng) -> FigureSpec:
    lang = style.language
    bar_name, bar_unit, bar_mag, line_name, line_unit, line_mag, line_dec = rng.pick(
        _PAIRS_DE if lang == "de" else _PAIRS_EN
    )

    if style.fig_w_in < 2.4:
        n = rng.randint(3, 5)
    elif style.fig_w_in < 3.6:
        n = rng.randint(4, 6)
    else:
        n = rng.randint(5, 9)

    cats = content.categories(rng, "year" if rng.chance(0.75) else "quarter", n, lang)
    n = len(cats)

    n_seg = rng.randint(2, 4) if sub_type == "stacked_bars" else 1
    if n_seg > 1:
        segs = []
        for _ in range(n_seg):
            share = rng.uniform(0.5, 1.0) ** rng.uniform(1.0, 2.0)
            segs.append([v * share for v in series.kpi_series(rng, n, bar_mag / n_seg)])
        bar_series = segs
        seg_names = content.categories(rng, "segment", n_seg, lang)
    else:
        bar_series = [series.kpi_series(rng, n, bar_mag)]
        seg_names = [bar_name]

    bar_dec = content.decimals_for(max(v for row in bar_series for v in row), 0)
    bar_series = [series.round_nicely(row, bar_dec) for row in bar_series]

    # The line. It always varies -- a constant line would be a reference line
    # and, per guide v1.1, would make this a bar chart rather than a combo.
    if sub_type == "same_axis":
        # Same scale as the bars: a sub-total that tracks the columns, e.g.
        # "of which international". Its own axis is therefore unavailable, and
        # the legend has to carry the identification.
        totals = [sum(col) for col in zip(*bar_series)]
        line = [round(t * rng.uniform(0.35, 0.72), bar_dec) for t in totals]
        line_name = ("davon Ausland" if lang == "de" else "of which international")
        line_unit, line_dec = bar_unit, bar_dec
    elif sub_type == "flat_line":
        # Visually almost flat -- a couple of tenths of movement -- but every
        # point is a real value. This is the case the guide's old wording got
        # wrong, and the reason the test is "does it vary", not "is it in the
        # legend".
        base = line_mag * rng.uniform(0.9, 1.1)
        line = [round(base * (1.0 + rng.normal(0.0, 0.012)), line_dec) for _ in range(n)]
        if len(set(line)) == 1:  # a genuinely constant line would be mislabeled
            line[-1] = round(line[-1] + 10.0 ** -line_dec, line_dec)
    else:
        line = series.round_nicely(series.kpi_series(rng, n, line_mag), line_dec)

    return FigureSpec(
        label="combo_bar_line",
        sub_type=sub_type,
        title=bar_name if style.title_mode != "none" else None,
        subtitle=(
            f"{bar_unit} / {line_name} in {line_unit}"
            if style.title_mode == "title_subtitle"
            else None
        ),
        source=content.source_note(rng, lang) if style.source_note else None,
        unit=bar_unit,
        categories=cats,
        series=bar_series,
        series_names=seg_names,
        decimals=bar_dec,
        percent=False,
        extra={
            "line": line,
            "line_name": line_name,
            "line_unit": line_unit,
            "line_decimals": line_dec,
            "line_percent": line_unit == "%",
            "marker": rng.pick(_MARKERS) if sub_type == "markers_line" else None,
        },
    )


# --------------------------------------------------------------------------
# Render
# --------------------------------------------------------------------------


def render_combo(
    spec: FigureSpec, style: StyleSheet, rng: Rng, oversample: float
) -> RenderResult:
    pal = style.palette
    fig, ax = mpl.new_figure(style, oversample)

    n = len(spec.categories)
    n_seg = len(spec.series)
    x = list(range(n))
    line = spec.extra["line"]

    lay = _fit(fig, spec, style, rng, n_seg)
    mpl.apply_margins(fig, style, _margins(spec, style, lay))

    # -- bars -------------------------------------------------------------
    seg_colors = palettes.ramp(pal, n_seg) if n_seg > 1 else [pal.color(0)]
    base = [0.0] * n
    for si, row in enumerate(spec.series):
        ax.bar(
            x, row, bottom=base, width=style.bar_width,
            color=seg_colors[si],
            edgecolor=pal.background if n_seg > 1 and style.bar_edge else "none",
            linewidth=0.6 if n_seg > 1 and style.bar_edge else 0.0,
            label=spec.series_names[si],
            zorder=3,
        )
        base = [b + v for b, v in zip(base, row)]

    totals = base
    ax.set_ylim(0, max(totals) * (1.18 if lay.data_labels else 1.08))
    ax.set_xlim(-0.6, n - 0.4)

    # -- the line ---------------------------------------------------------
    # A colour that has to read as a separate series on top of the bars, so it
    # must not be another tint of the same brand colour.
    line_color = palettes.contrast_on(pal.accent, pal.background, 0.30)
    target = ax.twinx() if lay.secondary_axis else ax
    target.plot(
        x, line,
        color=line_color,
        linewidth=rng.uniform(1.4, 2.4),
        marker=spec.extra.get("marker"),
        markersize=style.base_pt * 0.72 if spec.extra.get("marker") else 0,
        markerfacecolor=line_color,
        markeredgecolor=pal.background,
        markeredgewidth=0.6,
        label=spec.extra["line_name"],
        zorder=5,
    )

    lo, hi = min(line), max(line)
    pad = max((hi - lo) * 0.35, abs(hi) * 0.06, 1e-6)
    if lay.secondary_axis:
        target.set_ylim(max(0.0, lo - pad) if lo >= 0 else lo - pad, hi + pad)
        _style_secondary(target, spec, style, lay)
    else:
        # Sharing the primary scale, the line has to stay inside it.
        ax.set_ylim(0, max(max(totals), hi) * (1.18 if lay.data_labels else 1.08))

    common.apply_spines_grid(ax, style, "y")
    common.apply_value_axis(ax, style, lay, spec, "y")
    common.apply_category_axis(ax, style, lay, "x")

    ax.set_xticks(x)
    ax.set_xticklabels(
        spec.categories,
        rotation=lay.tick_rotation,
        ha="right" if lay.tick_rotation not in (0.0, 90.0) else "center",
        color=pal.muted,
        **mpl.font_kwargs(style, lay.tick_pt),
    )

    # -- labels -----------------------------------------------------------
    if lay.line_labels:
        for xi, v, text in zip(x, line, lay.line_labels):
            target.annotate(
                text, (xi, v), textcoords="offset points", xytext=(0, 5),
                ha="center", va="bottom", color=line_color,
                **mpl.font_kwargs(style, lay.label_pt),
            )

    if lay.legend != "none":
        _draw_combined_legend(ax, target if lay.secondary_axis else None, style, lay)

    common.draw_headings(fig, spec, style, lay)

    image = mpl.to_pil(fig)
    fig.clear()

    structure = Structure(
        bar_series=n_seg,
        bars_total=n * n_seg,
        segments_per_bar=n_seg,
        orientation="vertical",
        bars_on_baseline=n * n_seg,
        line_series=1,
        line_is_reference=False,  # it varies across categories, by construction
        line_own_axis=lay.secondary_axis,
        line_in_legend=lay.legend != "none",
    )
    meta = {
        "n_categories": n,
        "n_series": n_seg,
        "yaxis": lay.value_axis,
        "data_labels": "value" if lay.line_labels else "none",
        "tick_rotation": lay.tick_rotation,
        "legend": lay.legend,
        "secondary_axis": lay.secondary_axis,
        "title_wrapped": len(lay.title_lines) > 1,
    }
    return RenderResult(image=image, structure=structure, meta=meta)


def _fit(fig, spec: FigureSpec, style: StyleSheet, rng: Rng, n_seg: int) -> _ComboLayout:
    n = len(spec.categories)
    lay = _ComboLayout()
    lay.title_lines, lay.title_pt = common.fit_title(fig, spec, style)
    lay.subtitle, lay.subtitle_pt = common.fit_subtitle(fig, spec, style)
    common.cap_title_block(style, lay)
    lay.tick_pt, lay.tick_rotation = style.tick_pt, style.tick_rotation
    lay.label_pt, lay.value_axis = style.label_pt, style.yaxis

    lay.secondary_axis = spec.sub_type != "same_axis" and rng.chance(0.82)
    lay.legend = style.legend if style.legend != "none" else rng.weighted(
        {"top": 0.4, "bottom": 0.35, "none": 0.25}
    )

    # The invariant needs the line to be identifiable as a series: it must have
    # its own axis or its own legend entry. If the secondary axis was dropped,
    # the legend is no longer optional.
    if not lay.secondary_axis and lay.legend == "none":
        lay.legend = rng.pick(("top", "bottom"))

    names = list(spec.series_names[:n_seg]) + [spec.extra["line_name"]]
    common.fit_legend(fig, names, style, lay)
    axes_w = mpl.axes_width_pt(style, _margins(spec, style, lay))
    common.fit_legend(fig, names, style, lay, axes_w)
    axes_w = mpl.axes_width_pt(style, _margins(spec, style, lay))

    lay.tick_pt, lay.tick_rotation = common.fit_category_ticks(
        fig, spec.categories, style, axes_w / max(1, n)
    )

    # Only the line gets value labels: labelling the bars as well is more ink
    # than a combo chart ever carries.
    if style.data_labels != "none" and n <= 8 and rng.chance(0.55):
        dec = spec.extra["line_decimals"]
        texts = [
            content.format_number(v, dec, style.number_locale)
            + (" %" if spec.extra["line_percent"] else "")
            for v in spec.extra["line"]
        ]
        lay.line_labels, lay.label_pt = common.fit_value_labels(fig, texts, style, axes_w / n)
    return lay


def _style_secondary(ax2, spec: FigureSpec, style: StyleSheet, lay: _ComboLayout) -> None:
    pal = style.palette
    dec = spec.extra["line_decimals"]
    loc = style.number_locale
    suffix = " %" if spec.extra["line_percent"] else ""
    ax2.yaxis.set_major_formatter(
        mpl.FuncFormatter(lambda v, _p: content.format_number(v, dec, loc) + suffix)
    )
    ax2.tick_params(axis="y", colors=pal.muted, labelsize=lay.tick_pt, length=3)
    for lbl in ax2.get_yticklabels():
        lbl.set_fontfamily(style.font_family)
    for name, spine in ax2.spines.items():
        spine.set_visible(name == "right" and style.spines in ("all", "left_bottom"))
        spine.set_color(pal.muted)
        spine.set_linewidth(0.8)
    ax2.grid(False)  # two grids over one plot is unreadable


def _draw_combined_legend(ax, ax2, style: StyleSheet, lay: Layout) -> None:
    """The line lives on the other axes, so its handle has to be merged in."""
    pal = style.palette
    handles, labels = ax.get_legend_handles_labels()
    if ax2 is not None:
        h2, l2 = ax2.get_legend_handles_labels()
        handles, labels = handles + h2, labels + l2

    loc, anchor = {
        "top": ("lower left", (0.0, 1.01)),
        "bottom": ("upper left", (0.0, -0.14)),
        "right": ("center left", (1.06, 0.5)),
    }[lay.legend]
    leg = ax.legend(
        handles, labels,
        loc=loc, bbox_to_anchor=anchor, ncol=max(1, lay.legend_ncol),
        frameon=False, handlelength=1.4, columnspacing=1.2,
        prop={"family": style.font_family, "size": lay.label_pt},
    )
    for t in leg.get_texts():
        t.set_color(pal.muted)


def _margins(spec: FigureSpec, style: StyleSheet, lay: _ComboLayout) -> mpl.Margins:
    m = mpl.Margins()
    m.top = common.top_margin(style, lay)
    m.bottom = (
        6.0
        + common.bottom_extra(style, lay, spec)
        + common.category_axis_height(lay, spec.categories)
    )

    if lay.value_axis:
        total = max(sum(col) for col in zip(*spec.series))
        width = len(content.format_number(total, spec.decimals, style.number_locale))
        m.left = 8.0 + lay.tick_pt * 0.62 * width
    else:
        m.left = 7.0

    m.right = 8.0
    if lay.secondary_axis:
        dec = spec.extra["line_decimals"]
        widest = max(
            len(content.format_number(v, dec, style.number_locale)) for v in spec.extra["line"]
        )
        m.right += lay.tick_pt * 0.62 * (widest + (2 if spec.extra["line_percent"] else 0))
    return m
