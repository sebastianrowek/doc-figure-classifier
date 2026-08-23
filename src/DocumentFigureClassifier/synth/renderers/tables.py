"""
Tables -- a pure row/column grid with no graphical encoding of values.

The design doc pencilled in a separate PIL engine for this. In practice the
matplotlib engine already gives everything a table needs -- crisp rules from
plotted lines, measured text placement, the same font and degradation path as
every other class -- so tables are drawn here on the existing engine rather than
introducing a second one. The one thing that mattered from that note is honoured:
nothing graphical is drawn, because a table with an embedded bar or sparkline is
`other`, not `table` (guide, table). That variant is generated in other.py.

Coordinates are points throughout: the axes spans (axes_width_pt, table_height)
with y inverted, so a "row height of 14 pt" means 14 pt regardless of the crop's
aspect ratio. That is what keeps a wide table from being drawn with stretched
rows.

Sub-types
---------
multi_year      row labels + several year columns, the standard KPI table
comparison      two years plus a change column (absolute or %)
segment_matrix  segments down the side, metrics across the top
key_figures     a compact key-figure list
"""

from __future__ import annotations

from dataclasses import dataclass, field

from matplotlib.patches import Rectangle

from .. import content, series
from ..engines import mpl
from ..palettes import mix
from ..rng import Rng
from ..style import StyleSheet
from .base import FigureSpec, RenderResult, Structure

SUBTYPES = {
    "multi_year": 0.40,
    "comparison": 0.24,
    "segment_matrix": 0.20,
    "key_figures": 0.16,
}

# P&L / balance-sheet / KPI rows. (label, magnitude, decimals, percent, subtotal)
_PL_DE = (
    ("Umsatzerlöse", 4200, 0, False, False),
    ("Materialaufwand", 1450, 0, False, False),
    ("Personalaufwand", 760, 0, False, False),
    ("Sonstige Aufwendungen", 280, 0, False, False),
    ("EBITDA", 540, 0, False, True),
    ("Abschreibungen", 220, 0, False, False),
    ("EBIT", 320, 0, False, True),
    ("Finanzergebnis", 35, 0, False, False),
    ("Ergebnis vor Steuern", 285, 0, False, True),
    ("Steuern", 75, 0, False, False),
    ("Konzernergebnis", 210, 0, False, True),
)
_PL_EN = (
    ("Revenue", 4200, 0, False, False),
    ("Cost of materials", 1450, 0, False, False),
    ("Personnel expenses", 760, 0, False, False),
    ("Other expenses", 280, 0, False, False),
    ("EBITDA", 540, 0, False, True),
    ("Depreciation", 220, 0, False, False),
    ("EBIT", 320, 0, False, True),
    ("Financial result", 35, 0, False, False),
    ("Profit before tax", 285, 0, False, True),
    ("Taxes", 75, 0, False, False),
    ("Net income", 210, 0, False, True),
)
_BALANCE_DE = (
    ("Immaterielle Vermögenswerte", 620, 0, False, False),
    ("Sachanlagen", 1840, 0, False, False),
    ("Vorräte", 720, 0, False, False),
    ("Forderungen", 540, 0, False, False),
    ("Zahlungsmittel", 410, 0, False, False),
    ("Bilanzsumme", 5400, 0, False, True),
    ("Eigenkapital", 2280, 0, False, False),
    ("Rückstellungen", 860, 0, False, False),
    ("Finanzverbindlichkeiten", 1240, 0, False, False),
    ("Verbindlichkeiten aus L+L", 470, 0, False, False),
)
_BALANCE_EN = (
    ("Intangible assets", 620, 0, False, False),
    ("Property, plant & equipment", 1840, 0, False, False),
    ("Inventories", 720, 0, False, False),
    ("Receivables", 540, 0, False, False),
    ("Cash and equivalents", 410, 0, False, False),
    ("Total assets", 5400, 0, False, True),
    ("Equity", 2280, 0, False, False),
    ("Provisions", 860, 0, False, False),
    ("Financial liabilities", 1240, 0, False, False),
    ("Trade payables", 470, 0, False, False),
)
_KPI_DE = (
    ("Umsatz (Mio. €)", 4200, 0, False, False),
    ("EBIT (Mio. €)", 320, 0, False, False),
    ("EBIT-Marge (%)", 11.5, 1, True, False),
    ("Konzernergebnis (Mio. €)", 210, 0, False, False),
    ("Ergebnis je Aktie (€)", 3.4, 2, False, False),
    ("Dividende je Aktie (€)", 1.15, 2, False, False),
    ("Eigenkapitalquote (%)", 42.0, 1, True, False),
    ("ROCE (%)", 13.6, 1, True, False),
    ("Investitionen (Mio. €)", 180, 0, False, False),
    ("Mitarbeiter", 8600, 0, False, False),
)
_KPI_EN = (
    ("Revenue (€ m)", 4200, 0, False, False),
    ("EBIT (€ m)", 320, 0, False, False),
    ("EBIT margin (%)", 11.5, 1, True, False),
    ("Net income (€ m)", 210, 0, False, False),
    ("Earnings per share (€)", 3.4, 2, False, False),
    ("Dividend per share (€)", 1.15, 2, False, False),
    ("Equity ratio (%)", 42.0, 1, True, False),
    ("ROCE (%)", 13.6, 1, True, False),
    ("Capital expenditure (€ m)", 180, 0, False, False),
    ("Employees", 8600, 0, False, False),
)

_TITLES = {
    "_PL_DE": "Gewinn- und Verlustrechnung", "_PL_EN": "Income statement",
    "_BALANCE_DE": "Bilanz", "_BALANCE_EN": "Balance sheet",
    "_KPI_DE": "Kennzahlen", "_KPI_EN": "Key figures",
}


@dataclass
class _Row:
    label: str
    cells: list[str]
    bold: bool = False
    rule_above: bool = False
    indent: int = 0


@dataclass
class _TableLayout:
    title_lines: list[str] = field(default_factory=list)
    title_pt: float = 11.0
    subtitle: str | None = None
    subtitle_pt: float = 9.0
    font_pt: float = 9.0
    header_pt: float = 9.0
    row_h: float = 16.0
    col_x: list[float] = field(default_factory=list)
    label_w: float = 120.0
    header_fill: bool = False
    zebra: bool = False
    rules: str = "horizontal"  # full | horizontal | header_only | minimal


# --------------------------------------------------------------------------
# Spec
# --------------------------------------------------------------------------


def build_spec(sub_type: str, style: StyleSheet, rng: Rng) -> FigureSpec:
    lang = style.language
    max_rows = max(4, int(style.fig_h_in / 0.26))

    if sub_type == "segment_matrix":
        return _segment_spec(style, rng, max_rows)

    n_year = 2 if sub_type == "comparison" else rng.randint(2, 5)

    if sub_type == "key_figures":
        pool = _KPI_DE if lang == "de" else _KPI_EN
        pool_name = "_KPI_DE" if lang == "de" else "_KPI_EN"
    else:
        pools = ((_PL_DE, "_PL_DE"), (_BALANCE_DE, "_BALANCE_DE"), (_KPI_DE, "_KPI_DE")) if lang == "de" \
            else ((_PL_EN, "_PL_EN"), (_BALANCE_EN, "_BALANCE_EN"), (_KPI_EN, "_KPI_EN"))
        pool, pool_name = rng.weighted({pools[0]: 0.55, pools[1]: 0.25, pools[2]: 0.20})

    cap = max(3, min(len(pool), max_rows))
    n_row = rng.randint(min(4, cap), cap)
    picked = pool[:n_row]
    title = _TITLES.get(pool_name, "Kennzahlen" if lang == "de" else "Key figures")

    end_year = rng.randint(2022, 2025)
    years = [str(end_year - n_year + 1 + i) for i in range(n_year)]

    rows: list[list] = []
    for label, mag, dec, pct, sub in picked:
        vals = series.kpi_series(rng, n_year, abs(mag))
        cells = []
        for v in vals:
            s = content.format_number(round(v, dec), dec, lang)
            if pct:
                s += " %"
            cells.append(s)
        rows.append([label, cells, sub])

    return FigureSpec(
        label="table",
        sub_type=sub_type,
        title=title if style.title_mode != "none" else None,
        subtitle=(rng.pick(("in Mio. €", "Angaben in Mio. €")) if lang == "de" else "in € million")
        if style.title_mode == "title_subtitle" else None,
        source=content.source_note(rng, lang) if style.source_note else None,
        unit="",
        categories=years,
        series=[[0.0]],
        series_names=[title],
        decimals=0,
        percent=False,
        extra={"rows": rows, "col_headers": years,
               "change_col": ("%" if rng.chance(0.5) else "abs") if sub_type == "comparison" else None},
    )


def _segment_spec(style: StyleSheet, rng: Rng, max_rows: int) -> FigureSpec:
    lang = style.language
    cap = max(3, min(6, max_rows))
    n_seg = rng.randint(min(4, cap), cap)
    segs = content.categories(rng, "segment", n_seg, lang)
    metrics = list(("Umsatz", "EBIT", "Marge", "Mitarbeiter") if lang == "de"
                   else ("Revenue", "EBIT", "Margin", "Employees"))[: rng.randint(3, 4)]
    rows = []
    for seg in segs:
        cells = []
        for m in metrics:
            if m in ("Marge", "Margin"):
                cells.append(content.format_number(round(rng.uniform(4, 20), 1), 1, lang) + " %")
            elif m in ("Mitarbeiter", "Employees"):
                cells.append(content.format_number(rng.randint(300, 4000), 0, lang))
            else:
                cells.append(content.format_number(rng.randint(120, 2200), 0, lang))
        rows.append([seg, cells, False])
    return FigureSpec(
        label="table", sub_type="segment_matrix",
        title=("Segmentbericht" if lang == "de" else "Segment report") if style.title_mode != "none" else None,
        subtitle=None,
        source=content.source_note(rng, lang) if style.source_note else None,
        unit="", categories=metrics, series=[[0.0]], series_names=metrics,
        decimals=0, percent=False,
        extra={"rows": rows, "col_headers": metrics, "change_col": None},
    )


# --------------------------------------------------------------------------
# Render
# --------------------------------------------------------------------------


def render_table(spec: FigureSpec, style: StyleSheet, rng: Rng, oversample: float) -> RenderResult:
    pal = style.palette
    fig, ax = mpl.new_figure(style, oversample)

    rows_data = [_Row(r[0], list(r[1]), bold=r[2]) for r in spec.extra["rows"]]
    headers = list(spec.extra["col_headers"])

    if spec.extra.get("change_col"):
        headers.append("Δ" if rng.chance(0.5) else ("Veränd." if style.language == "de" else "Change"))
        for r in rows_data:
            a, b = _parse(r.cells[-1]), _parse(r.cells[0])
            if spec.extra["change_col"] == "%":
                chg = (100.0 * (a / b - 1.0)) if b else 0.0
                r.cells.append(("+" if chg >= 0 else "") + content.format_number(round(chg, 1), 1, style.language) + " %")
            else:
                chg = a - b
                r.cells.append(("+" if chg >= 0 else "") + content.format_number(round(chg, 0), 0, style.language))

    n_data_cols = len(headers)
    lay = _fit(fig, spec, style, rng, rows_data, headers)

    header_h = lay.row_h * 1.15
    table_h = header_h + lay.row_h * len(rows_data)

    top_pt = 4.0
    if lay.title_lines:
        top_pt += lay.title_pt * 1.35 * len(lay.title_lines) + 6.0
    if lay.subtitle:
        top_pt += lay.subtitle_pt * 1.5
    bottom_pt = 6.0 + (style.base_pt * 1.9 if spec.source else 0.0)
    margins = mpl.Margins(left=6.0, right=6.0, top=top_pt, bottom=bottom_pt)
    mpl.apply_margins(fig, style, margins)

    axes_w = mpl.axes_width_pt(style, margins)
    axes_h = max(table_h, style.fig_h_in * 72.0 - top_pt - bottom_pt)

    ax.set_xlim(0, axes_w)
    ax.set_ylim(0, axes_h)
    ax.invert_yaxis()
    ax.axis("off")

    text_col = pal.text
    rule_col = mix(pal.text, pal.background, 0.5)

    if lay.header_fill:
        ax.add_patch(Rectangle((0, 0), axes_w, header_h,
                     facecolor=mix(pal.color(0), pal.background, 0.30 if not pal.dark else 0.0),
                     edgecolor="none", zorder=1))

    for ci, htext in enumerate(headers):
        ax.text(lay.col_x[ci + 1], header_h * 0.5, htext, ha="right", va="center",
                color=text_col, **mpl.font_kwargs(style, lay.header_pt, True))

    if lay.rules in ("full", "horizontal", "header_only"):
        ax.plot([0, axes_w], [header_h, header_h], color=rule_col, linewidth=1.0, zorder=5)
        if lay.rules != "header_only":
            ax.plot([0, axes_w], [0, 0], color=rule_col, linewidth=1.0, zorder=5)

    y = header_h
    for ri, r in enumerate(rows_data):
        if lay.zebra and ri % 2 == 1:
            ax.add_patch(Rectangle((0, y), axes_w, lay.row_h,
                         facecolor=mix(pal.text, pal.background, 0.94), edgecolor="none", zorder=0))
        if r.bold and lay.rules != "minimal" and ri > 0:
            ax.plot([0, axes_w], [y, y], color=rule_col, linewidth=0.7, zorder=4)
        cy = y + lay.row_h * 0.5
        ax.text(lay.col_x[0], cy, r.label, ha="left", va="center",
                color=text_col, **mpl.font_kwargs(style, lay.font_pt, r.bold))
        for ci, cell in enumerate(r.cells):
            ax.text(lay.col_x[ci + 1], cy, cell, ha="right", va="center",
                    color=text_col, **mpl.font_kwargs(style, lay.font_pt, r.bold))
        y += lay.row_h

    if lay.rules in ("full", "horizontal"):
        ax.plot([0, axes_w], [y, y], color=rule_col, linewidth=1.0, zorder=5)
    if lay.rules == "full":
        sep_x = lay.col_x[0] + lay.label_w
        ax.plot([sep_x, sep_x], [0, y], color=mix(rule_col, pal.background, 0.4), linewidth=0.6, zorder=3)

    _draw_headings(fig, spec, style, lay)

    image = mpl.to_pil(fig)
    fig.clear()

    structure = Structure(table_rows=len(rows_data), table_cols=n_data_cols + 1, table_has_graphics=False)
    meta = {
        "n_rows": len(rows_data), "n_cols": n_data_cols + 1,
        "rules": lay.rules, "zebra": lay.zebra, "header_fill": lay.header_fill,
        "yaxis": False, "legend": "none", "title_wrapped": len(lay.title_lines) > 1,
    }
    return RenderResult(image=image, structure=structure, meta=meta)


def _fit(fig, spec, style, rng, rows_data, headers) -> _TableLayout:
    lay = _TableLayout()
    lay.title_lines = [spec.title] if spec.title else []
    lay.title_pt = style.title_pt
    if spec.title:
        budget = style.fig_w_in * 72.0 * 0.94
        w = mpl.text_width_pt(fig, spec.title, style, lay.title_pt, style.bold_title)
        if w > budget:
            lay.title_pt = max(style.base_pt, lay.title_pt * budget / w)
    lay.subtitle = spec.subtitle
    lay.subtitle_pt = style.base_pt * 0.95

    lay.font_pt = style.tick_pt
    lay.header_pt = style.tick_pt
    lay.row_h = max(lay.font_pt * 1.7, 11.0)
    lay.header_fill = rng.chance(0.4)
    lay.zebra = rng.chance(0.3) and not lay.header_fill
    lay.rules = rng.weighted({"horizontal": 0.4, "minimal": 0.25, "full": 0.2, "header_only": 0.15})

    axes_w = style.fig_w_in * 72.0 - 12.0
    n_cols = len(headers)

    def widths(fpt):
        lw = max(mpl.text_width_pt(fig, r.label, style, fpt, True) for r in rows_data) + 8.0
        cw = 0.0
        for r in rows_data:
            for c in r.cells:
                cw = max(cw, mpl.text_width_pt(fig, c, style, fpt, True))
        for h in headers:
            cw = max(cw, mpl.text_width_pt(fig, h, style, fpt, True))
        return lw, cw + 10.0

    label_w, cell_w = widths(lay.font_pt)
    total = label_w + cell_w * n_cols
    # Shrink the font until the whole block fits. The +8/+10 padding constants do
    # not scale with the font, so two passes converge; the floor is low enough
    # that clamping the label column (which causes labels to overrun the first
    # number) is a genuine last resort rather than the common path.
    for _ in range(2):
        if total <= axes_w:
            break
        lay.font_pt = max(style.tick_pt * 0.5, lay.font_pt * axes_w / total * 0.99)
        lay.header_pt = lay.font_pt
        lay.row_h = max(lay.font_pt * 1.7, 9.0)
        label_w, cell_w = widths(lay.font_pt)
        total = label_w + cell_w * n_cols
    if total > axes_w:
        label_w = max(20.0, axes_w - cell_w * n_cols)

    lay.label_w = label_w
    lay.col_x = [2.0]
    x = 2.0 + label_w
    for _ in range(n_cols):
        x += cell_w
        lay.col_x.append(x - 4.0)
    return lay


def _draw_headings(fig, spec, style, lay: _TableLayout) -> None:
    pal = style.palette
    x = 0.035 if style.title_align == "left" else 0.5
    ha = "left" if style.title_align == "left" else "center"
    y = 1.0 - mpl.frac_h(5.0, style)
    for line in lay.title_lines:
        fig.text(x, y, line, ha=ha, va="top", color=pal.text,
                 **mpl.font_kwargs(style, lay.title_pt, style.bold_title))
        y -= mpl.frac_h(lay.title_pt * 1.3, style)
    if lay.subtitle:
        fig.text(x, y, lay.subtitle, ha=ha, va="top", color=pal.muted,
                 **mpl.font_kwargs(style, lay.subtitle_pt))
    if spec.source:
        fig.text(0.035, mpl.frac_h(4.0, style), spec.source, ha="left", va="bottom",
                 color=pal.muted, **mpl.font_kwargs(style, style.base_pt * 0.85))


def _parse(s: str) -> float:
    s = s.replace("%", "").replace("+", "").strip()
    if "," in s:
        s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return 0.0
