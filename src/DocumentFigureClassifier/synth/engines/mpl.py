"""
matplotlib back-end.

Two deliberate departures from how matplotlib is normally used:

* No ``pyplot``. The global figure registry is a memory leak waiting to happen
  across 18,000 samples and is not safe in worker processes. ``Figure`` plus
  ``FigureCanvasAgg`` does the same job with no hidden state.

* No ``tight_layout``. It does not account for ``bar_label`` annotations, which
  is exactly what an axis-less corporate chart consists of, so it either clips
  the labels or leaves random whitespace. Margins are computed explicitly in
  points instead -- more code, but predictable.
"""

from __future__ import annotations

import logging as _logging
import warnings as _warnings

# Runs in every process that imports this module -- including the render workers,
# which never see the CLI's logging setup. We randomise across ~30 system font
# families, most of which lack a separate bold/normal weight file, so matplotlib
# logs "findfont: Failed to find font weight ..." on nearly every draw and, more
# rarely, warns that a glyph is missing from the chosen face. Both are cosmetic
# fallbacks (nearest weight / .notdef), not defects, so quiet the two channels
# here rather than making callers pipe the output through grep.
_logging.getLogger("matplotlib.font_manager").setLevel(_logging.ERROR)
_warnings.filterwarnings("ignore", message=r".*missing from font.*", category=UserWarning)

from dataclasses import dataclass

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FuncFormatter  # noqa: F401 - re-exported for renderers
from PIL import Image

from ..style import StyleSheet


@dataclass
class Margins:
    """Figure margins in points, converted to fractions when applied."""

    left: float = 8.0
    right: float = 8.0
    top: float = 8.0
    bottom: float = 8.0


def new_figure(style: StyleSheet, oversample: float) -> tuple[Figure, "object"]:
    """
    Create a figure at the physical size the crop pretends to have.

    The pixel size comes out as ``target_px * oversample``; degradation
    downsamples it to the real target. Rendering at the final size directly
    would leave no headroom for rotation and resampling artefacts, which is
    where a lot of the realism lives.
    """
    fig = Figure(
        figsize=(style.fig_w_in, style.fig_h_in),
        dpi=style.render_dpi * oversample,
        facecolor=style.palette.background,
    )
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.set_facecolor(style.palette.background)
    return fig, ax


def frac_h(pt: float, style: StyleSheet) -> float:
    """Points -> fraction of figure height."""
    return pt / 72.0 / style.fig_h_in


def frac_w(pt: float, style: StyleSheet) -> float:
    """Points -> fraction of figure width."""
    return pt / 72.0 / style.fig_w_in


def apply_margins(fig: Figure, style: StyleSheet, m: Margins, min_extent: float = 0.22) -> None:
    """
    Set the axes rectangle from margins given in points.

    When the requested margins would leave too little axes -- a small crop with
    a two-line title, a rotated category axis and a source note -- both margins
    on that dimension are scaled down *proportionally* rather than reset to a
    fixed fallback. Resetting discards the reservation, and the visible result
    is the title drawn straight over the bars. Scaling keeps the title above the
    plot; it just gets less room, which is fine because ``cap_title_block`` has
    already shrunk the text by then.
    """
    left, right = _fit_axis(frac_w(m.left, style), frac_w(m.right, style), min_extent)
    bottom, top = _fit_axis(frac_h(m.bottom, style), frac_h(m.top, style), min_extent)

    fig.subplots_adjust(left=left, right=1.0 - right, top=1.0 - top, bottom=bottom)


def _fit_axis(lo: float, hi: float, min_extent: float) -> tuple[float, float]:
    total = lo + hi
    budget = 1.0 - min_extent
    if total > budget and total > 0:
        k = budget / total
        return lo * k, hi * k
    return lo, hi


def font_kwargs(style: StyleSheet, size: float, bold: bool = False) -> dict:
    return {
        "family": style.font_family,
        "fontsize": size,
        "fontweight": "bold" if bold else "normal",
    }


def text_width_pt(fig: Figure, text: str, style: StyleSheet, size: float, bold: bool = False) -> float:
    """
    Width of a string in points, measured with the actual font.

    Character-count estimates are off by a factor of two between Bahnschrift
    Condensed and Bookman Old Style, which is exactly the range being sampled.
    Real measurement is a few microseconds and removes a whole class of
    overlapping-label bugs.
    """
    fp = FontProperties(
        family=style.font_family, size=size, weight="bold" if bold else "normal"
    )
    renderer = fig.canvas.get_renderer()
    w, _h, _d = renderer.get_text_width_height_descent(text, fp, ismath=False)
    return w * 72.0 / fig.dpi


def axes_width_pt(style: StyleSheet, m: Margins) -> float:
    return max(1.0, style.fig_w_in * 72.0 - m.left - m.right)


def to_pil(fig: Figure) -> Image.Image:
    """
    Render to RGB without touching the filesystem.

    Dimensions are taken from the buffer rather than from
    ``canvas.get_width_height()`` -- that returns logical units, which stop
    matching the buffer as soon as the DPI is scaled, and ours always is.
    """
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())  # (h, w, 4)
    return Image.fromarray(buf, mode="RGBA").convert("RGB")
