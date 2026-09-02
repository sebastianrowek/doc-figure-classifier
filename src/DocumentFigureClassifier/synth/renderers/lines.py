"""
Line and area charts -- both are ``line``.

The guide folds area charts in deliberately: "Flächendiagramme (Linie mit
eingefärbter Fläche darunter), auch gestapelt. Vorerst zusammengefasst." So
``area`` and ``area_stacked`` are sub-types here, not classes, and the manifest
records which is which so the split into an `area` class stays cheap if the
volume ever justifies it.

Sub-types
---------
single            one series over years or quarters
multi             two to four series
markers           a single or multi line with point markers
area              one series with the area below it filled
area_stacked      several filled series stacked on each other
index_benchmark   a dense share-price curve against an index -- the one line
                  chart shape that looks nothing like the others
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
    "single": 0.26,
    "multi": 0.22,
    "markers": 0.18,
    "area": 0.14,
    "area_stacked": 0.10,
    "index_benchmark": 0.10,
}

_BENCHMARKS = ("DAX", "MDAX", "SDAX", "STOXX Europe 600", "TecDAX")
_MARKERS = ("o", "s", "D", "^", "v")

# index_benchmark used to carry one fixed title per language, which is exactly the
# kind of always-present caption a classifier can shortcut on. A small pool breaks
# that up; the dense two-line share/index shape is the real signature anyway.
_BENCH_TITLES_DE = ("Aktienkursentwicklung", "Kursentwicklung der Aktie",
                    "Entwicklung des Aktienkurses", "Aktie im Vergleich",
                    "Wertentwicklung der Aktie", "Aktienperformance")
_BENCH_TITLES_EN = ("Share price development", "Share price performance",
                    "Share price vs. benchmark", "Total shareholder return",
                    "Share performance", "Stock price development")


def build_spec(sub_type: str, style: StyleSheet, rng: Rng) -> FigureSpec:
    topic = content.pick_topic(rng, style.language)

    if sub_type == "index_benchmark":
        return _benchmark_spec(style, rng)

    n_series = {
        "single": 1,
        "markers": rng.randint(1, 2),
        "area": 1,
        "multi": rng.randint(2, 4),
        "area_stacked": rng.randint(2, 4),
    }[sub_type]

    # Lines carry more points than bars before they get crowded, but the x
    # labels still have to fit.
    n_pts = rng.randint(5, 8) if style.fig_w_in < 2.6 else rng.randint(6, 12)

    kind = topic.category_kind if topic.category_kind in ("year", "quarter") else "year"
    cats = content.categories(rng, kind, n_pts, style.language)
    n_pts = len(cats)

    if n_series > 1:
        values = series.grouped_series(rng, n_pts, n_series, topic.magnitude)
        names = content.categories(
            rng, "segment" if rng.chance(0.6) else "region", n_series, style.language
        )
    else:
        values = [series.kpi_series(rng, n_pts, topic.magnitude)]
        names = [topic.title]

    dec = content.decimals_for(max(abs(v) for row in values for v in row), topic.decimals)
    values = [series.round_nicely(row, dec) for row in values]

    return FigureSpec(
        label="line",
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
        extra={"marker": rng.pick(_MARKERS) if sub_type == "markers" else None},
    )


def _benchmark_spec(style: StyleSheet, rng: Rng) -> FigureSpec:
    """
    An indexed share-price chart: many points, two correlated series, no markers.

    Worth its own builder because the shape is genuinely different -- a dense,
    noisy curve rather than six annual points -- and it is one of the most
    common line charts in an annual report.
    """
    # Two to three years, not five: a share-price chart in an annual report
    # covers the reporting period, and an ISO date is not how either language
    # prints it on an axis.
    n_pts = rng.randint(24, 38)
    start_year = rng.randint(2020, 2023)
    cats = [f"{i % 12 + 1:02d}/{(start_year + i // 12) % 100:02d}" for i in range(n_pts)]

    def walk(drift: float, vol: float) -> list[float]:
        v, out = 100.0, []
        for _ in range(n_pts):
            v *= 1.0 + drift + rng.normal(0.0, vol)
            out.append(round(v, 1))
        return out

    market = walk(rng.uniform(-0.002, 0.006), 0.022)
    # The share tracks the index loosely; independent walks look wrong.
    own = [
        round(m * rng.uniform(0.97, 1.03) * (1.0 + 0.004 * i * rng.uniform(-1, 1.6)), 1)
        for i, m in enumerate(market)
    ]

    company = rng.pick(("Aktie", "Share")) if style.language == "de" else "Share"
    return FigureSpec(
        label="line",
        sub_type="index_benchmark",
        title=(rng.pick(_BENCH_TITLES_DE if style.language == "de" else _BENCH_TITLES_EN))
        if style.title_mode != "none"
        else None,
        subtitle=("indexiert, 01.01. = 100" if style.language == "de" else "indexed, 1 Jan = 100")
        if style.title_mode == "title_subtitle"
        else None,
        source=content.source_note(rng, style.language) if style.source_note else None,
        unit="Index",
        categories=cats,
        series=[own, market],
        series_names=[company, rng.pick(_BENCHMARKS)],
        decimals=0,
        percent=False,
        extra={"marker": None, "dense": True},
    )


# --------------------------------------------------------------------------
# Render
# --------------------------------------------------------------------------


def render_line(
    spec: FigureSpec, style: StyleSheet, rng: Rng, oversample: float
) -> RenderResult:
    pal = style.palette
    fig, ax = mpl.new_figure(style, oversample)

    n_pts = len(spec.categories)
    n_series = len(spec.series)
    x = list(range(n_pts))
    stacked = spec.sub_type == "area_stacked"
    filled = spec.sub_type == "area"

    lay = _fit(fig, spec, style, rng, n_series)
    mpl.apply_margins(fig, style, _margins(spec, style, lay))

    # Fills may stay pale; strokes must not, or a light palette renders a line
    # chart as an empty frame once it has been downsampled and JPEGed.
    stroke = [palettes.contrast_on(pal.color(i), pal.background) for i in range(n_series)]

    if stacked:
        ax.stackplot(
            x,
            *spec.series,
            colors=[pal.color(i) for i in range(n_series)],
            labels=spec.series_names,
            edgecolor=pal.background,
            linewidth=0.6,
            zorder=3,
        )
        top = [sum(col) for col in zip(*spec.series)]
        vmax, vmin = max(top), 0.0
    else:
        for si, row in enumerate(spec.series):
            ax.plot(
                x,
                row,
                color=stroke[si],
                linewidth=rng.uniform(1.1, 2.2),
                linestyle="-" if si == 0 or n_series == 2 else rng.pick(("-", "--")),
                marker=spec.extra.get("marker"),
                markersize=style.base_pt * 0.72 if spec.extra.get("marker") else 0,
                markerfacecolor=stroke[si],
                markeredgecolor=pal.background,
                markeredgewidth=0.6,
                label=spec.series_names[si],
                zorder=3 + si,
            )
            if filled:
                ax.fill_between(x, row, color=stroke[si], alpha=rng.uniform(0.16, 0.35), zorder=2)
        flat = [v for row in spec.series for v in row]
        vmax, vmin = max(flat), min(flat)

    # Line charts rarely start at zero -- the whole point is the shape of the
    # curve, and a zero baseline flattens it. Bar charts are the opposite.
    span = max(vmax - vmin, 1e-6)
    if stacked or filled or rng.chance(0.35):
        lo = 0.0 if vmin >= 0 else vmin - 0.1 * span
    else:
        lo = vmin - rng.uniform(0.12, 0.45) * span
    ax.set_ylim(lo, vmax + rng.uniform(0.08, 0.22) * span)
    ax.set_xlim(-0.02 * n_pts, (n_pts - 1) * 1.02)

    # -- x ticks: a dense series only gets a handful ----------------------
    idx = _tick_indices(n_pts, lay.max_ticks)
    ax.set_xticks([x[i] for i in idx])
    ax.set_xticklabels(
        [spec.categories[i] for i in idx],
        rotation=lay.tick_rotation,
        ha="right" if lay.tick_rotation not in (0.0, 90.0) else "center",
        color=pal.muted,
        **mpl.font_kwargs(style, lay.tick_pt),
    )

    common.apply_spines_grid(ax, style, "y")
    common.apply_value_axis(ax, style, lay, spec, "y")
    common.apply_category_axis(ax, style, lay, "x")

    # -- end labels: the corporate alternative to a legend -----------------
    if lay.end_labels:
        for si, row in enumerate(spec.series):
            ax.text(
                n_pts - 1 + 0.15,
                row[-1],
                spec.series_names[si],
                ha="left",
                va="center",
                color=stroke[si],
                **mpl.font_kwargs(style, lay.label_pt),
            )

    if lay.legend != "none":
        common.draw_legend(ax, style, lay, reverse=stacked)

    common.draw_headings(fig, spec, style, lay)

    image = mpl.to_pil(fig)
    fig.clear()

    structure = Structure(
        bar_series=0,
        line_series=n_series,
        line_is_reference=False,
        line_in_legend=lay.legend != "none",
    )
    meta = {
        "n_categories": n_pts,
        "n_series": n_series,
        "yaxis": lay.value_axis,
        "data_labels": "none",
        "tick_rotation": lay.tick_rotation,
        "legend": lay.legend,
        "title_wrapped": len(lay.title_lines) > 1,
        "filled": stacked or filled,
    }
    return RenderResult(image=image, structure=structure, meta=meta)


@dataclass
class _LineLayout(Layout):
    max_ticks: int = 12
    end_labels: bool = False


def _fit(fig, spec: FigureSpec, style: StyleSheet, rng: Rng, n_series: int) -> _LineLayout:
    lay = _LineLayout()
    lay.legend = common.maybe_force_legend(style, rng, n_series, p=0.85)
    lay.title_lines, lay.title_pt = common.fit_title(fig, spec, style)
    lay.subtitle, lay.subtitle_pt = common.fit_subtitle(fig, spec, style)
    common.cap_title_block(style, lay)
    lay.label_pt = style.label_pt
    lay.tick_pt, lay.tick_rotation = style.tick_pt, style.tick_rotation
    # A line chart with no numeric axis and no data labels is unreadable, so
    # rule R4's axis-less look applies here far less often than to bars.
    lay.value_axis = style.yaxis or rng.chance(0.75)
    lay.end_labels = False
    lay.max_ticks = 12
    names = list(spec.series_names[:n_series])
    # A one-series legend just repeats the title; real charts omit it.
    if n_series == 1 and rng.chance(0.8):
        lay.legend = "none"
    common.fit_legend(fig, names, style, lay)
    axes_w = mpl.axes_width_pt(style, _margins(spec, style, lay))
    common.fit_legend(fig, names, style, lay, axes_w)
    axes_w = mpl.axes_width_pt(style, _margins(spec, style, lay))

    # Direct end-of-line labelling instead of a legend -- common in corporate
    # design, and it changes the visual signature enough to be worth sampling.
    if n_series > 1 and lay.legend in ("none", "right") and rng.chance(0.3):
        longest = max(spec.series_names, key=len)
        if mpl.text_width_pt(fig, longest, style, lay.label_pt) < axes_w * 0.3:
            lay.end_labels, lay.legend = True, "none"
            common.fit_legend(fig, [], style, lay)

    n_pts = len(spec.categories)
    widest = max(mpl.text_width_pt(fig, c, style, lay.tick_pt) for c in spec.categories)
    # Fit as many ticks as will not collide, down to three.
    lay.max_ticks = max(3, min(n_pts, int(axes_w / max(widest * 1.35, 1.0))))
    if lay.max_ticks < n_pts and lay.tick_rotation == 0.0 and lay.max_ticks < 4:
        lay.tick_rotation = 45.0
        lay.max_ticks = max(3, min(n_pts, int(axes_w / max(widest * 0.8, 1.0))))
    return lay


def _margins(spec: FigureSpec, style: StyleSheet, lay: Layout) -> mpl.Margins:
    m = mpl.Margins()
    m.top = common.top_margin(style, lay)
    m.right = common.right_margin(style, lay, spec)
    if getattr(lay, "end_labels", False):
        longest = max((len(n) for n in spec.series_names), default=6)
        m.right += lay.label_pt * (0.6 * min(longest, 16) + 1.5)

    m.bottom = (
        6.0
        + common.bottom_extra(style, lay, spec)
        + common.category_axis_height(lay, spec.categories)
    )

    if lay.value_axis:
        vmax = max(v for row in spec.series for v in row)
        width = len(content.format_number(vmax, spec.decimals, style.number_locale))
        m.left = 8.0 + lay.tick_pt * 0.62 * width
    else:
        m.left = 7.0
    return m


def _tick_indices(n: int, max_ticks: int) -> list[int]:
    """Evenly spaced tick positions, always including the first and last."""
    if n <= max_ticks:
        return list(range(n))
    step = (n - 1) / (max_ticks - 1)
    return sorted({round(i * step) for i in range(max_ticks)})
