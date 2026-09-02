"""
Pie and donut charts -- both are ``pie_donut``.

The guide is explicit that full and ring form are not distinguished: "identische
Semantik, und das Corporate Design wechselt willkürlich zwischen beiden." Semi-
circles and gauge-style displays are in the same class, which is where this
class touches ``other``: a ring used as a *progress* indicator is not a
part-to-whole display and belongs in `other`. That pair is generated
deliberately in phase 3; here the semicircle is always a share breakdown.

Sub-types
---------
pie             full circle
donut           ring
donut_center    ring with a figure in the middle
exploded        one segment pulled out
semicircle      half circle, shares only (guide: "Halbkreis- und
                Tachodarstellungen")
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .. import content, palettes
from ..engines import mpl
from ..palettes import readable_on
from ..rng import Rng
from ..style import StyleSheet
from . import common
from .base import FigureSpec, RenderResult, Structure
from .common import Layout

SUBTYPES = {
    "pie": 0.30,
    "donut": 0.28,
    "donut_center": 0.16,
    "exploded": 0.14,
    "semicircle": 0.12,
}

# The title routes the category kind (see build_spec): "aktionär"/"shareholder"/
# "float" -> shareholder names, "region" -> regions, otherwise segments. Kept
# wide so no single caption word marks a crop as a pie -- the circular shape has
# to do that. Several share phrasing with the bar/line topics ("Umsatz nach ...").
_TOPICS_DE = (
    ("Aktionärsstruktur", "Anteile am Grundkapital"),
    ("Umsatz nach Regionen", "Anteile in Prozent"),
    ("Umsatz nach Segmenten", "Geschäftsjahr 2024"),
    ("Belegschaft nach Regionen", "zum 31.12.2024"),
    ("Streubesitz", "Aktionärsstruktur"),
    ("Umsatz nach Kundengruppen", "Anteile in Prozent"),
    ("Energiemix", "Anteile in Prozent"),
    ("Investitionen nach Bereichen", "Geschäftsjahr 2024"),
    ("Umsatz nach Produktgruppen", "Anteile in Prozent"),
    ("Umsatz nach Geschäftsfeldern", "Geschäftsjahr 2024"),
    ("Umsatz nach Branchen", "Anteile am Konzernumsatz"),
    ("Ergebnis nach Segmenten", "Anteile in Prozent"),
    ("Mitarbeiter nach Segmenten", "zum 31.12.2024"),
    ("Umsatz nach Vertriebskanälen", "Anteile in Prozent"),
    ("Auftragseingang nach Regionen", "Anteile in Prozent"),
    ("Vermögensstruktur", "Anteile an der Bilanzsumme"),
    ("Kapitalstruktur", "Anteile an der Bilanzsumme"),
    ("Kostenstruktur", "Anteile in Prozent"),
    ("Rohstoffeinsatz", "Anteile in Prozent"),
    ("Aktionärsstruktur nach Investorentyp", "Anteile am Grundkapital"),
)
_TOPICS_EN = (
    ("Shareholder structure", "share of subscribed capital"),
    ("Revenue by region", "share in percent"),
    ("Revenue by segment", "financial year 2024"),
    ("Employees by region", "as of 31 Dec 2024"),
    ("Free float", "shareholder structure"),
    ("Energy mix", "share in percent"),
    ("Revenue by product group", "share in percent"),
    ("Revenue by business unit", "financial year 2024"),
    ("Revenue by industry", "share of group revenue"),
    ("Earnings by segment", "share in percent"),
    ("Employees by segment", "as of 31 Dec 2024"),
    ("Order intake by region", "share in percent"),
    ("Asset structure", "share of total assets"),
    ("Capital structure", "share of total assets"),
    ("Cost structure", "share in percent"),
    ("Revenue by sales channel", "share in percent"),
)
_HOLDERS_DE = ("Streubesitz", "Familienbesitz", "Institutionelle Investoren",
               "Eigene Aktien", "Gründerfamilie", "Ankeraktionär")
_HOLDERS_EN = ("Free float", "Family holding", "Institutional investors",
               "Treasury shares", "Founding family", "Anchor shareholder")


def build_spec(sub_type: str, style: StyleSheet, rng: Rng) -> FigureSpec:
    lang = style.language
    title, subtitle = rng.pick(_TOPICS_DE if lang == "de" else _TOPICS_EN)

    # More than about seven slices stops being readable and stops being drawn.
    n = rng.randint(3, 5) if style.fig_w_in < 2.6 else rng.randint(3, 7)

    if "aktionär" in title.lower() or "shareholder" in title.lower() or "float" in title.lower():
        pool = _HOLDERS_DE if lang == "de" else _HOLDERS_EN
        cats = rng.sample(pool, min(n, len(pool)))
    else:
        kind = "region" if "region" in title.lower() else "segment"
        cats = content.categories(rng, kind, n, lang)
    n = len(cats)

    # One dominant share plus a tail is what these charts almost always show.
    weights = [rng.uniform(0.12, 1.0) ** rng.uniform(1.1, 2.2) for _ in range(n)]
    total = sum(weights)
    shares = [100.0 * w / total for w in weights]
    if rng.chance(0.7):
        shares.sort(reverse=True)
    dec = 1 if rng.chance(0.45) else 0
    shares = [round(s, dec) for s in shares]

    return FigureSpec(
        label="pie_donut",
        sub_type=sub_type,
        title=title if style.title_mode != "none" else None,
        subtitle=subtitle if style.title_mode == "title_subtitle" else None,
        source=content.source_note(rng, lang) if style.source_note else None,
        unit="%",
        categories=cats,
        series=[shares],
        series_names=cats,
        decimals=dec,
        percent=True,
        extra={},
    )


# --------------------------------------------------------------------------
# Render
# --------------------------------------------------------------------------


@dataclass
class _PieLayout(Layout):
    label_mode: str = "legend"  # legend | outside | inside_pct | outside_pct
    donut_width: float = 0.0
    center_text: tuple[str, str] | None = None
    center_pt: float = 12.0
    min_pct_share: float = 7.5


# Outside labels are placed at each wedge's mid-angle with no collision
# handling, so two adjacent small slices put their captions on top of each
# other. Rather than build a leader-line solver, the mode is only chosen when
# the geometry makes collisions unlikely: few slices, none of them thin.
_OUTSIDE_MIN_SHARE = 8.0
_OUTSIDE_MAX_SLICES = 5


def render_pie(spec: FigureSpec, style: StyleSheet, rng: Rng, oversample: float) -> RenderResult:
    pal = style.palette
    fig, ax = mpl.new_figure(style, oversample)

    values = spec.series[0]
    n = len(values)
    semi = spec.sub_type == "semicircle"

    lay = _fit(fig, spec, style, rng, n)
    mpl.apply_margins(fig, style, _margins(spec, style, lay))

    # ramp(), not color(): a cycling palette gives two slices the same fill and
    # the boundary between them disappears.
    colors = palettes.ramp(pal, n)
    draw_values = list(values)
    draw_colors = list(colors)
    if semi:
        # A half circle is a full pie with an invisible wedge of equal total.
        draw_values = values + [sum(values)]
        draw_colors = colors + ["none"]

    explode = None
    if spec.sub_type == "exploded":
        k = rng.randint(0, n - 1)
        explode = [0.0] * len(draw_values)
        explode[k] = rng.uniform(0.06, 0.16)

    wedgeprops = {"linewidth": 0.8, "edgecolor": pal.background}
    if lay.donut_width:
        wedgeprops["width"] = lay.donut_width

    show_pct = lay.label_mode in ("inside_pct", "outside_pct")
    wedges, *_rest = ax.pie(
        draw_values,
        colors=draw_colors,
        startangle=180 if semi else rng.randint(0, 3) * 90 + rng.randint(-25, 25),
        counterclock=False if semi else rng.chance(0.35),
        explode=explode,
        wedgeprops=wedgeprops,
        labels=None,
        autopct=_pct_formatter(spec, style, semi, lay) if show_pct else None,
        # On a donut the percentage belongs on the ring, not in the hole. The
        # ring's mid-radius is 1 - width/2; the previous fixed 0.65 put every
        # label inside the hole, on top of the centre figure.
        pctdistance=(
            (1.0 - lay.donut_width / 2.0 if lay.donut_width else 0.68)
            if lay.label_mode == "inside_pct"
            else 1.16
        ),
        textprops={"family": style.font_family, "fontsize": lay.label_pt},
        radius=1.0,
    )
    ax.set_aspect("equal")

    # Percentage text colour has to follow the wedge it sits on.
    if lay.label_mode == "inside_pct":
        texts = [t for t in ax.texts if t.get_text().strip()]
        for t, c in zip(texts, colors):
            t.set_color(readable_on(c))
    elif lay.label_mode == "outside_pct":
        for t in ax.texts:
            t.set_color(pal.muted)

    # -- category labels around the circle --------------------------------
    if lay.label_mode in ("outside", "outside_pct"):
        _draw_outside_labels(ax, wedges[:n], spec, style, lay, with_pct=lay.label_mode == "outside")

    if semi:
        # Crop the invisible half so the drawing fills the crop.
        ax.set_ylim(-0.06, 1.22)
        ax.set_xlim(-1.15, 1.15)

    if lay.center_text:
        big, small = lay.center_text
        # Shrink the headline figure until it fits inside the hole. The hole
        # radius is (1 - donut_width) in data units, and the axes spans 2.6
        # units across its short side.
        hole_pt = 2.0 * (1.0 - lay.donut_width) * _short_extent_pt(style, lay) / 2.6
        pt = lay.center_pt
        while pt > style.base_pt * 0.8 and (
            mpl.text_width_pt(fig, big, style, pt, True) > hole_pt * 0.86
        ):
            pt *= 0.9
        ax.text(0, 0.07 if small else 0, big, ha="center", va="center", color=pal.text,
                **mpl.font_kwargs(style, pt, True))
        if small:
            small_pt = pt * 0.42
            if mpl.text_width_pt(fig, small, style, small_pt) < hole_pt * 0.92:
                ax.text(0, -0.15, small, ha="center", va="center", color=pal.muted,
                        **mpl.font_kwargs(style, small_pt))

    if lay.legend != "none":
        for w, name in zip(wedges[:n], spec.categories):
            w.set_label(name)
        common.draw_legend(ax, style, lay)

    common.draw_headings(fig, spec, style, lay)

    image = mpl.to_pil(fig)
    fig.clear()

    structure = Structure(pie_segments=n)
    meta = {
        "n_categories": n,
        "n_series": 1,
        "yaxis": False,
        "data_labels": "value" if show_pct else "none",
        "legend": lay.legend,
        "label_mode": lay.label_mode,
        "title_wrapped": len(lay.title_lines) > 1,
    }
    return RenderResult(image=image, structure=structure, meta=meta)


def _fit(fig, spec: FigureSpec, style: StyleSheet, rng: Rng, n: int) -> _PieLayout:
    lay = _PieLayout()
    lay.title_lines, lay.title_pt = common.fit_title(fig, spec, style)
    lay.subtitle, lay.subtitle_pt = common.fit_subtitle(fig, spec, style)
    # A pie is sized by the space left over, so an oversized heading does not
    # overlap it -- it shrinks it to nothing. Cap it harder than for bars.
    common.cap_title_block(style, lay, max_frac=0.30)
    lay.label_pt = style.label_pt
    lay.value_axis = False

    if spec.sub_type in ("donut", "donut_center"):
        lay.donut_width = rng.uniform(0.30, 0.52)

    aspect = style.fig_w_in / max(style.fig_h_in, 1e-6)
    longest = max(spec.categories, key=len)
    label_w = mpl.text_width_pt(fig, longest, style, lay.label_pt)
    outside_ok = n <= _OUTSIDE_MAX_SLICES and min(spec.series[0]) >= _OUTSIDE_MIN_SHARE

    # A pie is square; on a wide crop the space next to it is where the legend
    # goes, and on a narrow one there is no room for labels around the circle.
    if aspect > 1.45 and label_w < style.fig_w_in * 72.0 * 0.34:
        lay.label_mode = rng.weighted({"legend": 0.6, "inside_pct": 0.4})
        lay.legend = "right" if lay.label_mode == "legend" else "none"
    elif style.fig_w_in < 2.4 or label_w > style.fig_w_in * 72.0 * 0.3:
        lay.label_mode = "inside_pct"
        lay.legend = rng.weighted({"bottom": 0.5, "none": 0.3, "right": 0.2})
    elif outside_ok:
        lay.label_mode = rng.weighted(
            {"outside": 0.3, "inside_pct": 0.28, "legend": 0.25, "outside_pct": 0.17}
        )
        lay.legend = "bottom" if lay.label_mode == "inside_pct" and rng.chance(0.5) else "none"
        if lay.label_mode == "legend":
            lay.legend = rng.pick(("right", "bottom"))
    else:
        lay.label_mode = rng.weighted({"inside_pct": 0.55, "legend": 0.45})
        lay.legend = rng.pick(("right", "bottom")) if lay.label_mode == "legend" else (
            "bottom" if rng.chance(0.5) else "none"
        )

    if spec.sub_type == "donut_center":
        lay.center_text = _center_text(spec, style, rng)
        # A big number needs a big hole.
        lay.donut_width = rng.uniform(0.22, 0.38)
        lay.center_pt = style.title_pt * 1.15
        # Percentages on a narrow ring plus a headline figure in the hole is
        # more text than the shape can carry; real ones use a legend.
        if lay.label_mode == "inside_pct":
            lay.label_mode = "legend"
            lay.legend = rng.pick(("right", "bottom"))

    # A pie small enough that captions would not fit gets none: the crop is
    # then just a coloured ring, which is exactly how small ones look in print.
    if _short_extent_pt(style, lay) < 78.0 and lay.label_mode in ("outside", "outside_pct"):
        lay.label_mode = "inside_pct"
        lay.legend = "none"

    lay.min_pct_share = _min_pct_share(fig, style, lay)

    common.fit_legend(fig, list(spec.categories), style, lay)
    common.fit_legend(
        fig, list(spec.categories), style, lay,
        mpl.axes_width_pt(style, _margins(spec, style, lay)),
    )
    return lay


def _min_pct_share(fig, style: StyleSheet, lay: _PieLayout) -> float:
    """
    Smallest share that still gets a printed percentage.

    A fixed threshold is wrong because the same 8 % slice has plenty of room on
    a 4-inch pie and none at all on a 1-inch one. The label needs about its own
    width of arc at the radius it sits on, so derive the threshold from the
    circle's actual circumference in points.
    """
    r_units = (1.0 - lay.donut_width / 2.0) if lay.donut_width else 0.68
    if lay.label_mode == "outside_pct":
        r_units = 1.16
    r_pt = r_units * _short_extent_pt(style, lay) / 2.6
    circumference = 2.0 * math.pi * max(r_pt, 1.0)

    # "88,8 %" is the widest a share label gets.
    widest = mpl.text_width_pt(fig, "88,8 %", style, lay.label_pt)
    floor = 7.5 if lay.label_mode == "inside_pct" else 4.0
    return max(floor, 100.0 * (widest + 3.0) / circumference)


def _center_text(spec: FigureSpec, style: StyleSheet, rng: Rng) -> tuple[str, str]:
    """
    The headline figure in a donut's hole.

    It must not be the sum of the shares: that is 100 % by construction and
    reads as a rounding bug when it comes out as 99 % or 101 %. Real donuts put
    an absolute total or the leading share there.
    """
    if rng.chance(0.55):
        magnitude, unit = rng.pick(
            (
                (rng.uniform(800, 9000), "Mio. €" if style.language == "de" else "€ m"),
                (rng.uniform(3000, 40000), "Mitarbeiter" if style.language == "de" else "employees"),
                (rng.uniform(1.5, 9.5), "Mrd. €" if style.language == "de" else "€ bn"),
            )
        )
        dec = 1 if magnitude < 100 else 0
        return content.format_number(round(magnitude, dec), dec, style.number_locale), unit

    top = max(spec.series[0])
    idx = spec.series[0].index(top)
    return (
        content.format_number(top, spec.decimals, style.number_locale) + " %",
        spec.categories[idx],
    )


def _short_extent_pt(style: StyleSheet, lay: Layout) -> float:
    """The dimension that sizes the circle: aspect is equal, so it is the smaller one."""
    m = mpl.Margins()
    m.top = common.top_margin(style, lay)
    h = style.fig_h_in * 72.0 - m.top - 10.0
    w = style.fig_w_in * 72.0 - 20.0
    return max(20.0, min(h, w))


def _margins(spec: FigureSpec, style: StyleSheet, lay: Layout) -> mpl.Margins:
    m = mpl.Margins()
    m.top = common.top_margin(style, lay)
    m.bottom = 6.0 + common.bottom_extra(style, lay, spec)
    m.left = 7.0
    m.right = common.right_margin(style, lay, spec)
    if getattr(lay, "label_mode", "") in ("outside", "outside_pct"):
        # Labels sit beyond the circle on both sides.
        longest = max((len(c) for c in spec.categories), default=8)
        pad = lay.label_pt * 0.55 * min(longest, 18)
        m.left += pad
        m.right += pad
        m.top += lay.label_pt * 1.2
        m.bottom += lay.label_pt * 1.2
    return m


def _pct_formatter(spec: FigureSpec, style: StyleSheet, semi: bool, lay: _PieLayout):
    """
    matplotlib hands autopct a percentage of the drawn total.

    For a semicircle that total includes the phantom wedge, so the number would
    be halved -- undo that rather than let every gauge report 50 % too little.

    The hide threshold is higher for labels sitting *inside* the wedges: two
    adjacent thin slices put their captions almost on the same spot, and there
    is no room to nudge them apart.
    """
    scale = 2.0 if semi else 1.0
    hide_below = lay.min_pct_share

    def fmt(pct: float) -> str:
        value = pct * scale
        if value < hide_below:
            return ""
        return content.format_number(value, spec.decimals, style.number_locale) + " %"

    return fmt


def _draw_outside_labels(ax, wedges, spec, style, lay, with_pct: bool) -> None:
    """
    Category names on the outside of the circle, each joined to its wedge by a
    short leader line -- the callout look real pie charts use. The label sits
    exactly where it did before, so the leader is added without enlarging the
    reserved margin or risking a caption running off the frame.
    """
    pal = style.palette
    values = spec.series[0]
    for w, name, v in zip(wedges, spec.categories, values):
        angle = math.radians((w.theta1 + w.theta2) / 2.0)
        ca, sa = math.cos(angle), math.sin(angle)
        r = 1.08 + (0.06 if lay.donut_width else 0.0)
        xx, yy = r * ca, r * sa
        # leader from the wedge rim (radius 1.0) out to just short of the text
        ax.plot(
            [ca, (r - 0.03) * ca], [sa, (r - 0.03) * sa],
            color=pal.muted, linewidth=0.7, zorder=1,
        )
        text = name
        if with_pct:
            text = f"{name}  {content.format_number(v, spec.decimals, style.number_locale)} %"
        ax.text(
            xx, yy, text,
            ha="left" if xx >= 0 else "right",
            va="center",
            color=pal.muted,
            **mpl.font_kwargs(style, lay.label_pt),
        )
