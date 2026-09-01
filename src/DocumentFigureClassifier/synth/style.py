"""
Style presets and the randomisation axes.

The point of the presets is stated in the design doc: matplotlib's default look
is not the annual-report look. Randomising *around* the defaults gives you
15,000 variations of the same wrong thing. Instead each sample picks one of a
handful of house styles and randomises inside it.

Two axes deserve special attention, both from rule R4 of the labeling guide:
corporate design frequently deletes the y-axis entirely and writes values on
the bars. If that case is rare in training, the model quietly learns "axis
cross => chart" and falls over on exactly the charts this project cares about.
Hence the deliberate over-representation below.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

from . import palettes
from .fonts import available as available_fonts
from .palettes import Palette
from .rng import Rng


@dataclass(frozen=True)
class StyleSheet:
    preset: str
    palette: Palette
    language: str  # "de" | "en"
    number_locale: str  # usually == language

    # typography
    font_family: str
    font_kind: str  # sans | serif | slab
    base_pt: float
    title_pt: float
    label_pt: float
    tick_pt: float
    bold_title: bool
    bold_labels: bool

    # headings
    title_mode: str  # none | title | title_subtitle
    title_align: str  # left | center
    source_note: bool

    # axes and grid
    spines: str  # all | left_bottom | bottom | none
    grid: str  # none | h_light | h_dashed | full
    yaxis: bool  # y tick labels visible
    xticks: bool
    tick_rotation: float

    # series decoration
    data_labels: str  # none | value | value_unit
    label_pos: str  # outside | inside
    legend: str  # none | top | bottom | right
    bar_width: float
    bar_edge: bool
    hatch: str | None

    # geometry
    render_dpi: float  # the PDF page render scale this crop pretends to come from
    fig_w_in: float
    fig_h_in: float

    def digest(self) -> str:
        """Stable hash of the style, for post-training error analysis."""
        parts = [
            self.preset, self.palette.name, self.language, self.font_family,
            f"{self.base_pt:.2f}", self.title_mode, self.title_align,
            str(self.source_note), self.spines, self.grid, str(self.yaxis),
            str(self.xticks), f"{self.tick_rotation:.0f}", self.data_labels,
            self.label_pos, self.legend, f"{self.bar_width:.2f}",
            str(self.bar_edge), str(self.hatch), f"{self.render_dpi:.0f}",
        ]
        return hashlib.sha1("|".join(parts).encode()).hexdigest()[:10]


# --------------------------------------------------------------------------
# Presets
# --------------------------------------------------------------------------

PRESET_WEIGHTS = {
    "corporate_flat": 0.28,
    "classic_axes": 0.24,
    "consulting_thinkcell": 0.18,
    "mono_print": 0.12,
    "web_dashboard": 0.12,
    "dark_panel": 0.06,
}


def sample_style(rng: Rng, target_w: int, target_h: int) -> StyleSheet:
    """
    Draw a complete style for one sample.

    ``target_w/h`` are the final pixel dimensions (from the real crop size
    distribution). They determine the physical figure size, which in turn caps
    the font size -- a 1.5 inch wide chart with 11pt labels looks like nothing
    that was ever printed.
    """
    preset = rng.weighted(PRESET_WEIGHTS)

    # The render scale the crop pretends to come from. Docling's default is
    # --scale 2.0, i.e. 144 dpi; the spread covers other pipelines and settings.
    render_dpi = rng.uniform(108.0, 192.0)
    fig_w_in = target_w / render_dpi
    fig_h_in = target_h / render_dpi

    language = "de" if rng.chance(0.75) else "en"
    fonts = available_fonts()

    # Font size is capped by figure width, not just drawn freely.
    base_pt = min(rng.uniform(6.5, 10.5), 3.2 + 2.6 * fig_w_in)
    base_pt = max(base_pt, 4.6)

    sheet = StyleSheet(
        preset=preset,
        palette=palettes.sample_palette(rng),
        language=language,
        number_locale=language,
        font_family=fonts.pick(rng, "sans"),
        font_kind="sans",
        base_pt=base_pt,
        title_pt=base_pt * rng.uniform(1.15, 1.5),
        label_pt=base_pt * rng.uniform(0.82, 1.0),
        tick_pt=base_pt * rng.uniform(0.82, 1.0),
        bold_title=rng.chance(0.65),
        bold_labels=rng.chance(0.3),
        title_mode=rng.weighted({"title": 0.5, "title_subtitle": 0.3, "none": 0.2}),
        title_align=rng.weighted({"left": 0.7, "center": 0.3}),
        source_note=rng.chance(0.3),
        spines="left_bottom",
        grid="h_light",
        yaxis=True,
        xticks=True,
        tick_rotation=0.0,
        data_labels="none",
        label_pos="outside",
        legend="none",
        bar_width=rng.uniform(0.55, 0.8),
        bar_edge=rng.chance(0.18),
        hatch=None,
        render_dpi=render_dpi,
        fig_w_in=fig_w_in,
        fig_h_in=fig_h_in,
    )

    sheet = _apply_preset(sheet, rng)

    # Cross-cutting draws that every preset shares.
    if rng.chance(0.14):
        sheet = replace(sheet, tick_rotation=rng.pick((30.0, 45.0, 90.0)))
    if rng.chance(0.05):
        sheet = replace(sheet, hatch=rng.pick(("//", "\\\\", "..", "xx")))
    # Serif and slab house styles exist, in every preset. Sans stays dominant;
    # serif is the common alternative and slab (Rockwell / Bodoni) a rarer
    # display look. Every text-placement decision is measured against the real
    # font (engines.mpl.text_width_pt), so switching family never breaks layout.
    # face == "sans" keeps whatever the preset already chose -- which is a serif
    # for mono_print, so that preset stays serif (or, rarely, slab).
    face = rng.weighted({"sans": 0.84, "serif": 0.12, "slab": 0.04})
    if face != "sans":
        sheet = replace(
            sheet,
            font_family=available_fonts().pick(rng, face),
            font_kind=face,
        )
    return sheet


def _apply_preset(s: StyleSheet, rng: Rng) -> StyleSheet:
    p = s.preset

    if p == "corporate_flat":
        # Flat, axis-light, values written on the bars. The dominant look in
        # German annual reports.
        return replace(
            s,
            spines=rng.weighted({"none": 0.55, "bottom": 0.35, "left_bottom": 0.10}),
            grid=rng.weighted({"none": 0.6, "h_light": 0.4}),
            yaxis=rng.chance(0.25),
            data_labels=rng.weighted({"value": 0.75, "value_unit": 0.15, "none": 0.10}),
            legend=rng.weighted({"none": 0.7, "top": 0.15, "bottom": 0.15}),
            bar_width=rng.uniform(0.55, 0.78),
        )

    if p == "classic_axes":
        # The textbook chart: full axes, gridlines, a y-scale, no data labels.
        return replace(
            s,
            spines=rng.weighted({"left_bottom": 0.6, "all": 0.4}),
            grid=rng.weighted({"h_light": 0.55, "full": 0.25, "h_dashed": 0.20}),
            yaxis=True,
            data_labels=rng.weighted({"none": 0.75, "value": 0.25}),
            legend=rng.weighted({"none": 0.45, "top": 0.15, "bottom": 0.2, "right": 0.2}),
        )

    if p == "consulting_thinkcell":
        # Slide-deck look: no axes at all, every bar labelled, one accent colour
        # on grey. Bars sit close together.
        return replace(
            s,
            palette=palettes.accent_on_grey(rng) if rng.chance(0.6) else s.palette,
            spines="none",
            grid="none",
            yaxis=False,
            data_labels=rng.weighted({"value": 0.85, "value_unit": 0.15}),
            bold_labels=rng.chance(0.7),
            bold_title=True,
            title_align="left",
            legend=rng.weighted({"none": 0.75, "top": 0.25}),
            bar_width=rng.uniform(0.7, 0.92),
        )

    if p == "mono_print":
        # Monochrome, serif, thin -- the financial-section look.
        return replace(
            s,
            palette=palettes.greyscale(rng) if rng.chance(0.55) else palettes.mono_ramp(rng),
            font_family=available_fonts().pick(rng, "serif"),
            font_kind="serif",
            spines=rng.weighted({"left_bottom": 0.55, "bottom": 0.25, "all": 0.20}),
            grid=rng.weighted({"h_light": 0.5, "none": 0.3, "h_dashed": 0.2}),
            yaxis=rng.chance(0.7),
            data_labels=rng.weighted({"none": 0.55, "value": 0.45}),
            bar_edge=rng.chance(0.5),
        )

    if p == "web_dashboard":
        return replace(
            s,
            spines=rng.weighted({"none": 0.45, "bottom": 0.4, "left_bottom": 0.15}),
            grid=rng.weighted({"h_light": 0.65, "none": 0.2, "h_dashed": 0.15}),
            yaxis=rng.chance(0.6),
            data_labels=rng.weighted({"none": 0.5, "value": 0.5}),
            legend=rng.weighted({"none": 0.5, "top": 0.25, "right": 0.25}),
            bar_width=rng.uniform(0.5, 0.72),
        )

    if p == "dark_panel":
        return replace(
            s,
            palette=palettes.dark(rng),
            spines=rng.weighted({"none": 0.5, "bottom": 0.3, "left_bottom": 0.2}),
            grid=rng.weighted({"h_light": 0.6, "none": 0.4}),
            yaxis=rng.chance(0.55),
            data_labels=rng.weighted({"value": 0.6, "none": 0.4}),
        )

    raise ValueError(f"unknown preset: {p}")
