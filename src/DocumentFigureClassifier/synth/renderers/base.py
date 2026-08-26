"""
Renderer protocol and label invariants.

This is the part of the generator that keeps the dataset honest. Every renderer
reports what it structurally drew; ``check()`` verifies that this is still
consistent with the label the sample was generated under. A waterfall whose
randomisation happened to put every bar on the baseline is a ``bar``, and
without this check it would ship as a mislabeled waterfall.

The rules below are the labeling guide, section 2 and 3, expressed as code.
One place they are *tighter* than the prose: the guide says a line makes a
chart a combo when it has "its own axis or its own legend entry", but also that
a target line is not a data series -- and target lines are routinely in the
legend. The structural discriminator is whether the line varies across the
categories, which is knowable at generation time. Hence ``line_is_reference``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from PIL import Image

from ..rng import Rng
from ..style import StyleSheet


@dataclass
class Structure:
    """What a renderer actually drew, in terms the labeling guide cares about."""

    # bars
    bar_series: int = 0
    bars_total: int = 0
    segments_per_bar: int = 1
    orientation: str | None = None  # "vertical" | "horizontal"
    bars_on_baseline: int = 0
    bars_floating: int = 0
    connectors: bool = False

    # lines
    line_series: int = 0
    line_is_reference: bool = False  # constant target/average line, not a series
    line_own_axis: bool = False
    line_in_legend: bool = False

    # circular
    pie_segments: int = 0

    # scatter / flow -- both promoted out of `other` in taxonomy v1.2
    scatter_points: int = 0
    flow_nodes: int = 0
    flow_edges: int = 0

    # tables / maps / misc, used from phase 3 on
    table_rows: int = 0
    table_cols: int = 0
    table_has_graphics: bool = False
    geometry_area_frac: float = 0.0
    chart_area_frac: float = 1.0  # set by layout.compose() for rule R2


@dataclass
class FigureSpec:
    """Everything content-related a renderer needs. Filled by the class plan."""

    label: str
    sub_type: str
    title: str | None = None
    subtitle: str | None = None
    source: str | None = None
    unit: str = ""
    categories: list[str] = field(default_factory=list)
    series: list[list[float]] = field(default_factory=list)
    series_names: list[str] = field(default_factory=list)
    decimals: int = 0
    percent: bool = False
    extra: dict = field(default_factory=dict)


@dataclass
class RenderResult:
    image: Image.Image
    structure: Structure
    meta: dict = field(default_factory=dict)


class Renderer(Protocol):
    def __call__(
        self, spec: FigureSpec, style: StyleSheet, rng: Rng, oversample: float
    ) -> RenderResult: ...


class InvariantViolation(RuntimeError):
    """The drawn image no longer matches the label it was generated under."""


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def _check_bar(s: Structure) -> str | None:
    # Orientation no longer separates classes (taxonomy v1.2): `bar` is any
    # plain, single-series bar chart, vertical or horizontal. What must hold is
    # that it is a single series -- several side by side is `bar_grouped`.
    if s.bars_total < 3:
        return f"only {s.bars_total} bars"
    if s.bar_series != 1:
        return f"{s.bar_series} bar series -- several side by side is bar_grouped"
    if s.segments_per_bar != 1:
        return f"{s.segments_per_bar} segments per bar -- that is bar_stacked"
    if s.bars_on_baseline != s.bars_total:
        return f"{s.bars_total - s.bars_on_baseline} bars off the baseline -- that is waterfall"
    if s.line_series and not s.line_is_reference:
        return "a non-reference line series -- that is combo_bar_line"
    if s.connectors:
        return "connector lines -- that is waterfall"
    return None


def _check_bar_grouped(s: Structure) -> str | None:
    # Same geometry as `bar`, but two or more series drawn side by side. Split
    # off because grouped bars are harder to parse (guide, taxonomy v1.2).
    if s.bar_series < 2:
        return f"{s.bar_series} bar series -- a grouped chart needs at least 2"
    if s.segments_per_bar != 1:
        return f"{s.segments_per_bar} segments per bar -- that is bar_stacked"
    if s.bars_on_baseline != s.bars_total:
        return f"{s.bars_total - s.bars_on_baseline} bars off the baseline -- that is waterfall"
    if s.line_series and not s.line_is_reference:
        return "a non-reference line series -- that is combo_bar_line"
    if s.connectors:
        return "connector lines -- that is waterfall"
    return None


def _check_bar_stacked(s: Structure) -> str | None:
    if s.bars_total < 2:
        return f"only {s.bars_total} bars"
    if s.segments_per_bar < 2:
        return "fewer than 2 segments per bar -- that is a plain bar chart"
    if s.bars_on_baseline != s.bars_total:
        return "not every bar starts at the baseline -- that is waterfall"
    if s.connectors:
        return "connector lines -- stacks never have them"
    return None


def _check_waterfall(s: Structure) -> str | None:
    if s.bars_total < 4:
        return f"only {s.bars_total} bars"
    if s.bars_floating < 2:
        return f"{s.bars_floating} floating bars -- a bridge needs at least 2"
    if s.bars_on_baseline < 2:
        return "start and end bar must sit on the baseline"
    if s.segments_per_bar != 1:
        return "more than one segment per bar -- that is bar_stacked"
    return None


def _check_line(s: Structure) -> str | None:
    if s.line_series < 1:
        return "no line series"
    if s.bar_series:
        return "bars present -- that is combo_bar_line"
    return None


def _check_combo(s: Structure) -> str | None:
    if s.bar_series < 1:
        return "no bar series"
    if s.line_series < 1:
        return "no line series"
    if s.line_is_reference:
        return "the line is a reference line, not a data series -- that is bar_*"
    if not (s.line_own_axis or s.line_in_legend):
        return "line has neither its own axis nor a legend entry"
    return None


def _check_pie(s: Structure) -> str | None:
    if s.pie_segments < 2:
        return f"{s.pie_segments} segments"
    return None


def _check_scatter(s: Structure) -> str | None:
    if s.scatter_points < 5:
        return f"only {s.scatter_points} points"
    if s.bar_series or s.pie_segments:
        return "bars or pie segments -- not a scatter"
    if s.line_series and not s.line_is_reference:
        return "a data line through the points -- that is line"
    return None


def _check_flow(s: Structure) -> str | None:
    if s.flow_nodes < 2:
        return f"{s.flow_nodes} node(s) -- a flow needs at least 2"
    return None


def _check_map(s: Structure) -> str | None:
    if s.geometry_area_frac < 0.45:
        return f"map covers only {s.geometry_area_frac:.0%} of the image"
    return None


def _check_table(s: Structure) -> str | None:
    if s.table_rows < 2 or s.table_cols < 2:
        return f"{s.table_rows}x{s.table_cols} is not a grid"
    if s.table_has_graphics:
        return "embedded bars/sparklines -- the guide puts those in other"
    return None


def _check_logo(s: Structure) -> str | None:
    if s.bar_series or s.line_series or s.pie_segments:
        return "a data series in a logo"
    return None


def _check_other(s: Structure) -> str | None:
    # `other` is the catch-all: anything goes, by definition. The one thing
    # worth asserting is that it is not accidentally a clean chart of a class
    # we do have -- but sub-generators like other/table_with_bars are exactly
    # that on purpose, so this stays open.
    return None


_CHECKS = {
    "bar": _check_bar,
    "bar_grouped": _check_bar_grouped,
    "bar_stacked": _check_bar_stacked,
    "waterfall": _check_waterfall,
    "line": _check_line,
    "combo_bar_line": _check_combo,
    "pie_donut": _check_pie,
    "scatter": _check_scatter,
    "flow": _check_flow,
    "map": _check_map,
    "table": _check_table,
    "photo": lambda s: None,
    "logo_icon": _check_logo,
    "other": _check_other,
}


def check(label: str, s: Structure) -> None:
    """
    Raise if the structure contradicts the label.

    Unknown labels raise ``KeyError`` on purpose: adding a renderer without
    adding its invariant should be a hard error, not a silent pass.
    """
    violation = _CHECKS[label](s)

    # Rule R2: once a layout frame is composed around the chart, a chart taking
    # less than half the image is `other` regardless of what was drawn. The
    # generator flips the label rather than failing -- see cli.build_sample.
    if violation:
        raise InvariantViolation(f"{label}: {violation}")
