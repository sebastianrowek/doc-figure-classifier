"""
Tier-1 taxonomy from the labeling guide -- the single source of truth.

Both the extraction pipeline and the synthetic data generator import from here
so the label list cannot drift apart in two places.

See docs/labeling_guide_en.md (German original: docs/labeling_guide_de.md).
"""

from __future__ import annotations

TIER1_LABELS = [
    "bar_vertical",
    "bar_horizontal",
    "bar_stacked",
    "waterfall",
    "line",
    "combo_bar_line",
    "pie_donut",
    "map",
    "table",
    "photo",
    "logo_icon",
    "other",
]

# Review-only folders. Not classes, never part of training.
EXTRA_FOLDERS = ["_review", "_unsure", "_broken"]

# Filters the extraction pipeline applies to real crops. The synthetic set
# honours the same limits -- generating images the real pipeline would have
# thrown away as junk only teaches the model about sizes it will never see.
MIN_SIDE_PX = 120
MIN_AREA_PX = 30_000
MAX_ASPECT = 8.0
