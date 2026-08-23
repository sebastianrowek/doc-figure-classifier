"""
Run configuration and the class registry.

``RunConfig`` holds primitives only: it is pickled to worker processes, and
Windows uses spawn, so anything clever in here becomes a debugging session.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .renderers import bars, combo, lines, logos, maps, other, pie, tables, waterfall


@dataclass(frozen=True)
class RunConfig:
    out: str
    n_per_class: int
    base_seed: int = 20260823
    size_manifest: str = "data/parsed/manifest.jsonl"
    degrade: bool = True
    oversample_lo: float = 1.5
    oversample_hi: float = 2.5
    max_invariant_retries: int = 5


@dataclass(frozen=True)
class ClassPlan:
    """How to produce one tier-1 class: its sub-types and how to draw them."""

    label: str
    subtypes: dict[str, float]
    build_spec: Callable
    render: Callable


# Adding a class means one entry here plus its invariant in renderers/base.py;
# nothing else in the pipeline needs to change. An unregistered label raises in
# base.check(), so a renderer cannot ship without its check.
REGISTRY: dict[str, ClassPlan] = {
    "bar_vertical": ClassPlan(
        label="bar_vertical",
        subtypes=bars.SUBTYPES_VERTICAL,
        build_spec=bars.build_spec,
        render=bars.render_bar_vertical,
    ),
    "bar_horizontal": ClassPlan(
        label="bar_horizontal",
        subtypes=bars.SUBTYPES_HORIZONTAL,
        build_spec=bars.build_spec_horizontal,
        render=bars.render_bar_horizontal,
    ),
    "bar_stacked": ClassPlan(
        label="bar_stacked",
        subtypes=bars.SUBTYPES_STACKED,
        build_spec=bars.build_spec_stacked,
        render=bars.render_bar_stacked,
    ),
    "waterfall": ClassPlan(
        label="waterfall",
        subtypes=waterfall.SUBTYPES,
        build_spec=waterfall.build_spec,
        render=waterfall.render_waterfall,
    ),
    "combo_bar_line": ClassPlan(
        label="combo_bar_line",
        subtypes=combo.SUBTYPES,
        build_spec=combo.build_spec,
        render=combo.render_combo,
    ),
    "line": ClassPlan(
        label="line",
        subtypes=lines.SUBTYPES,
        build_spec=lines.build_spec,
        render=lines.render_line,
    ),
    "pie_donut": ClassPlan(
        label="pie_donut",
        subtypes=pie.SUBTYPES,
        build_spec=pie.build_spec,
        render=pie.render_pie,
    ),
    "table": ClassPlan(
        label="table",
        subtypes=tables.SUBTYPES,
        build_spec=tables.build_spec,
        render=tables.render_table,
    ),
    "logo_icon": ClassPlan(
        label="logo_icon",
        subtypes=logos.SUBTYPES,
        build_spec=logos.build_spec,
        render=logos.render_logo,
    ),
    "map": ClassPlan(
        label="map",
        subtypes=maps.SUBTYPES,
        build_spec=maps.build_spec,
        render=maps.render_map,
    ),
    "other": ClassPlan(
        label="other",
        subtypes=other.SUBTYPES,
        build_spec=other.build_spec,
        render=other.render_other,
    ),
}


def allocate(subtypes: dict[str, float], n: int) -> list[str]:
    """
    Deterministic sub-type allocation for n samples.

    Drawing sub-types at random would leave the rare ones underfilled by pure
    luck; largest-remainder allocation guarantees the intended mix.
    """
    keys = list(subtypes)
    total = sum(subtypes.values())
    exact = [n * subtypes[k] / total for k in keys]
    counts = [int(e) for e in exact]

    remainder = n - sum(counts)
    order = sorted(range(len(keys)), key=lambda i: exact[i] - counts[i], reverse=True)
    for i in range(remainder):
        counts[order[i % len(order)]] += 1

    out: list[str] = []
    for k, c in zip(keys, counts):
        out.extend([k] * c)
    return out
