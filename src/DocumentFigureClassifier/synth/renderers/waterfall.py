"""
Waterfall / bridge charts.

The guide calls this the most expensive class to get wrong, and lists five
identifying features of which "at least two should apply":

1. first and last bar on the baseline, the ones between floating
2. connector lines between bar ends
3. two-colour coding for increases and decreases
4. signed labels on the intermediate bars
5. axis labels naming reasons for change rather than categories or years

Feature 1 is structural and always true here -- it is what makes the chart a
bridge, and it is what the invariant checks. Features 2 to 5 are cosmetic, and
a generator that always emits all four would teach the model to look for
green/red plus connectors rather than for floating bars. So the sub-types
switch them off one at a time:

    plain               all five features
    no_connectors       feature 2 removed
    categorical_colors  feature 3 removed -- coloured per category instead
    stacked_lookalike   bars packed edge to edge, which is what a stack looks
                        like; the closest a bridge comes to bar_stacked
    subtotals           extra bars back on the baseline mid-chart, so the
                        "only first and last touch the baseline" reading fails

All five are ``waterfall`` by construction. See guide rule R3.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .. import content, palettes
from ..engines import mpl
from ..palettes import readable_on
from ..rng import Rng
from ..style import StyleSheet
from . import common
from .base import FigureSpec, RenderResult, Structure
from .common import Layout

SUBTYPES = {
    "plain": 0.32,
    "no_connectors": 0.18,
    "categorical_colors": 0.16,
    "stacked_lookalike": 0.14,
    "subtotals": 0.20,
}

# (title, unit, magnitude, decimals, start label prefix)
_BRIDGES_DE = (
    ("EBIT-Brücke", "Mio. €", 320, 0, "EBIT"),
    ("Umsatzbrücke", "Mio. €", 1800, 0, "Umsatz"),
    ("Überleitung Nettofinanzschulden", "Mio. €", 640, 0, "Nettoschulden"),
    ("Cashflow-Überleitung", "Mio. €", 430, 0, "Cashflow"),
    ("Veränderung Konzernergebnis", "Mio. €", 210, 0, "Ergebnis"),
    ("Entwicklung Eigenkapital", "Mio. €", 2400, 0, "Eigenkapital"),
    ("Überleitung EBITDA", "Mio. €", 540, 0, "EBITDA"),
    ("Entwicklung Mitarbeiterzahl", "Mitarbeiter", 8600, 0, "Belegschaft"),
)
_BRIDGES_EN = (
    ("EBIT bridge", "€ m", 320, 0, "EBIT"),
    ("Revenue bridge", "€ m", 1800, 0, "Revenue"),
    ("Net debt reconciliation", "€ m", 640, 0, "Net debt"),
    ("Cash flow reconciliation", "€ m", 430, 0, "Cash flow"),
    ("Change in net income", "€ m", 210, 0, "Net income"),
    ("Headcount development", "employees", 8600, 0, "Headcount"),
)

# Reasons for change -- feature 5. Never years, never categories.
_REASONS_DE = (
    "Volumen", "Preis", "Währung", "M&A", "Portfolio", "Kosten", "Mix",
    "Rohstoffe", "Personal", "Einmaleffekte", "Sonstiges", "Abschreibungen",
    "Energie", "Investitionen", "Organisches Wachstum", "Effizienz",
)
_REASONS_EN = (
    "Volume", "Price", "Currency", "M&A", "Portfolio", "Cost", "Mix",
    "Raw materials", "Personnel", "One-offs", "Other", "Depreciation",
    "Energy", "Capex", "Organic growth", "Efficiency",
)
_SUBTOTALS_DE = ("Operativ", "Zwischensumme", "vor Sondereffekten", "bereinigt")
_SUBTOTALS_EN = ("Operating", "Subtotal", "before specials", "adjusted")


@dataclass
class _WfLayout(Layout):
    connectors: bool = True
    label_widths: list[float] = field(default_factory=list)


# --------------------------------------------------------------------------
# Spec
# --------------------------------------------------------------------------


def build_spec(sub_type: str, style: StyleSheet, rng: Rng) -> FigureSpec:
    lang = style.language
    title, unit, magnitude, dec, stem = rng.pick(_BRIDGES_DE if lang == "de" else _BRIDGES_EN)

    # A bridge needs a start, at least two changes and an end. Width caps it.
    if style.fig_w_in < 2.4:
        n_delta = rng.randint(2, 3)
    elif style.fig_w_in < 3.6:
        n_delta = rng.randint(2, 4)
    else:
        n_delta = rng.randint(3, 6)

    y0 = rng.randint(2019, 2024)
    reasons = rng.sample(_REASONS_DE if lang == "de" else _REASONS_EN, n_delta)

    start = magnitude * rng.uniform(0.8, 1.2)
    # Deltas are a fraction of the starting level; a bridge whose steps dwarf
    # the opening balance is not a reconciliation of anything.
    deltas = []
    for _ in range(n_delta):
        size = start * rng.uniform(0.03, 0.22)
        deltas.append(size if rng.chance(0.62) else -size)

    items: list[tuple[str, str, float]] = [("start", f"{stem} {y0}", round(start, dec))]
    for name, d in zip(reasons, deltas):
        items.append(("delta", name, round(d, dec)))

    if sub_type == "subtotals" and n_delta >= 3:
        # Insert one subtotal bar back on the baseline, part-way through.
        pos = rng.randint(2, n_delta - 1)
        label = rng.pick(_SUBTOTALS_DE if lang == "de" else _SUBTOTALS_EN)
        items.insert(pos + 1, ("subtotal", label, 0.0))

    items.append(("end", f"{stem} {y0 + 1}", 0.0))

    return FigureSpec(
        label="waterfall",
        sub_type=sub_type,
        title=title if style.title_mode != "none" else None,
        subtitle=(f"in {unit}" if style.title_mode == "title_subtitle" else None),
        source=content.source_note(rng, lang) if style.source_note else None,
        unit=unit,
        categories=[name for _kind, name, _v in items],
        series=[[v for _kind, _name, v in items]],
        series_names=[title],
        decimals=dec,
        percent=False,
        extra={"kinds": [kind for kind, _n, _v in items]},
    )


def _resolve(spec: FigureSpec) -> tuple[list[float], list[float], list[float]]:
    """
    Turn the item list into (bottom, height, level-after) per bar.

    ``level-after`` is where the connector to the next bar has to sit.
    """
    kinds = spec.extra["kinds"]
    values = spec.series[0]
    bottoms, heights, levels = [], [], []
    cum = 0.0

    for kind, v in zip(kinds, values):
        if kind == "start":
            cum = v
            bottoms.append(0.0)
            heights.append(v)
        elif kind == "delta":
            bottoms.append(cum if v >= 0 else cum + v)
            heights.append(abs(v))
            cum += v
        else:  # subtotal | end -- back on the baseline, value is the running total
            bottoms.append(0.0)
            heights.append(cum)
        levels.append(cum)

    return bottoms, heights, levels


# --------------------------------------------------------------------------
# Render
# --------------------------------------------------------------------------


def render_waterfall(
    spec: FigureSpec, style: StyleSheet, rng: Rng, oversample: float
) -> RenderResult:
    pal = style.palette
    fig, ax = mpl.new_figure(style, oversample)

    kinds = spec.extra["kinds"]
    values = spec.series[0]
    n = len(kinds)
    x = list(range(n))
    bottoms, heights, levels = _resolve(spec)

    lay = _fit(fig, spec, style, rng, bottoms, heights)
    mpl.apply_margins(fig, style, _margins(spec, style, lay, bottoms, heights))

    # -- colours ----------------------------------------------------------
    if spec.sub_type == "categorical_colors":
        # Feature 3 removed: coloured per category, exactly like a bar chart.
        colors = palettes.ramp(pal, n)
    else:
        anchor = palettes.contrast_on(pal.color(0), pal.background, 0.22)
        colors = [
            anchor if k != "delta" else (pal.positive if v >= 0 else pal.negative)
            for k, v in zip(kinds, values)
        ]

    width = 0.98 if spec.sub_type == "stacked_lookalike" else style.bar_width
    ax.bar(
        x, heights, bottom=bottoms, width=width,
        color=colors,
        edgecolor=pal.background if spec.sub_type == "stacked_lookalike" else (
            pal.text if style.bar_edge else "none"
        ),
        linewidth=0.6 if (style.bar_edge or spec.sub_type == "stacked_lookalike") else 0.0,
        zorder=3,
    )

    # -- connectors -------------------------------------------------------
    if lay.connectors:
        half = width / 2.0
        for i in range(n - 1):
            ax.plot(
                [x[i] + half, x[i + 1] - half],
                [levels[i], levels[i]],
                color=pal.muted,
                linewidth=rng.uniform(0.6, 1.0),
                linestyle=rng.pick(("-", "--", (0, (2, 2)))),
                zorder=2,
            )

    # -- scale ------------------------------------------------------------
    tops = [b + h for b, h in zip(bottoms, heights)]
    vmax = max(tops)
    vmin = min(0.0, min(bottoms))
    span = max(vmax - vmin, 1e-6)
    ax.set_ylim(vmin - 0.04 * span, vmax + (0.20 if lay.data_labels else 0.08) * span)
    ax.set_xlim(-0.5 - (0.02 if width > 0.9 else 0.0), n - 0.5 + (0.02 if width > 0.9 else 0.0))

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

    # -- signed labels (feature 4) ----------------------------------------
    if lay.data_labels:
        _draw_labels(ax, spec, style, lay, bottoms, heights, colors, span)

    common.draw_headings(fig, spec, style, lay)

    image = mpl.to_pil(fig)
    fig.clear()

    on_baseline = sum(1 for k in kinds if k != "delta")
    structure = Structure(
        bar_series=1,
        bars_total=n,
        segments_per_bar=1,
        orientation="vertical",
        bars_on_baseline=on_baseline,
        bars_floating=n - on_baseline,
        connectors=lay.connectors,
    )
    meta = {
        "n_categories": n,
        "n_series": 1,
        "yaxis": lay.value_axis,
        "data_labels": style.data_labels if lay.data_labels else "none",
        "tick_rotation": lay.tick_rotation,
        "legend": "none",
        "connectors": lay.connectors,
        "n_floating": n - on_baseline,
        "title_wrapped": len(lay.title_lines) > 1,
    }
    return RenderResult(image=image, structure=structure, meta=meta)


def _fit(
    fig, spec: FigureSpec, style: StyleSheet, rng: Rng, bottoms, heights
) -> _WfLayout:
    n = len(spec.categories)
    lay = _WfLayout(legend="none")
    lay.title_lines, lay.title_pt = common.fit_title(fig, spec, style)
    lay.subtitle, lay.subtitle_pt = common.fit_subtitle(fig, spec, style)
    common.cap_title_block(style, lay)
    lay.tick_pt, lay.tick_rotation = style.tick_pt, style.tick_rotation
    lay.label_pt, lay.value_axis = style.label_pt, style.yaxis

    # Feature 2 is on for every sub-type except the one named after removing it,
    # and even then a bridge without connectors is common enough in print.
    lay.connectors = spec.sub_type != "no_connectors" and rng.chance(0.88)

    axes_w = mpl.axes_width_pt(style, _margins(spec, style, lay, bottoms, heights))
    lay.tick_pt, lay.tick_rotation = common.fit_category_ticks(
        fig, spec.categories, style, axes_w / max(1, n)
    )

    # Reasons for change are long words; a bridge axis rotates far more often
    # than a year axis does.
    if style.data_labels != "none" and n <= 10:
        texts = _label_texts(spec, style)
        lay.data_labels, lay.label_pt = common.fit_value_labels(fig, texts, style, axes_w / n)
        if lay.data_labels is None:
            lay.value_axis = True
        else:
            lay.label_widths = [
                mpl.text_width_pt(fig, t, style, lay.label_pt, style.bold_labels)
                for t in lay.data_labels
            ]
    return lay


def _label_texts(spec: FigureSpec, style: StyleSheet) -> list[str]:
    """Signed on the changes, plain on the anchors -- guide feature 4."""
    kinds = spec.extra["kinds"]
    _b, _h, levels = _resolve(spec)
    out = []
    for i, (kind, v) in enumerate(zip(kinds, spec.series[0])):
        if kind == "delta":
            sign = "+" if v >= 0 else "-"
            out.append(sign + content.format_number(abs(v), spec.decimals, style.number_locale))
        else:
            out.append(content.format_number(levels[i], spec.decimals, style.number_locale))
    return out


def _draw_labels(ax, spec, style, lay: _WfLayout, bottoms, heights, colors, span) -> None:
    """
    Above the bar for a rise, below for a fall -- and inside when the bar is
    tall enough to hold the text, which is how anchor bars are usually labelled.
    """
    pal = style.palette
    kinds = spec.extra["kinds"]
    pad = span * 0.015

    for i, (kind, b, h) in enumerate(zip(kinds, bottoms, heights)):
        text = lay.data_labels[i]
        inside = kind != "delta" and h > span * 0.22 and style.label_pos == "inside"
        if inside:
            y, va, color = b + h / 2.0, "center", readable_on(colors[i])
        elif spec.series[0][i] < 0 and kind == "delta":
            y, va, color = b - pad, "top", pal.text
        else:
            y, va, color = b + h + pad, "bottom", pal.text
        ax.text(
            i, y, text, ha="center", va=va, color=color,
            **mpl.font_kwargs(style, lay.label_pt, style.bold_labels),
        )


def _margins(spec: FigureSpec, style: StyleSheet, lay: Layout, bottoms, heights) -> mpl.Margins:
    m = mpl.Margins()
    m.top = common.top_margin(style, lay)
    m.right = 8.0
    m.bottom = (
        6.0
        + common.bottom_extra(style, lay, spec)
        + common.category_axis_height(lay, spec.categories)
    )
    if lay.value_axis:
        vmax = max(b + h for b, h in zip(bottoms, heights))
        width = len(content.format_number(vmax, spec.decimals, style.number_locale))
        m.left = 8.0 + lay.tick_pt * 0.62 * width
    else:
        m.left = 7.0
    return m
