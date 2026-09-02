"""
Run configuration and the class registry.

``RunConfig`` holds primitives only: it is pickled to worker processes, and
Windows uses spawn, so anything clever in here becomes a debugging session.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..taxonomy import TIER1_LABELS
from .renderers import bars, combo, flow, lines, logos, maps, other, pie, scatter, tables, waterfall


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
    "bar": ClassPlan(
        label="bar",
        subtypes=bars.SUBTYPES_BAR,
        build_spec=bars.build_spec,
        render=bars.render_bar,
    ),
    "bar_grouped": ClassPlan(
        label="bar_grouped",
        subtypes=bars.SUBTYPES_GROUPED,
        build_spec=bars.build_spec_grouped,
        render=bars.render_bar,
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
    "scatter": ClassPlan(
        label="scatter",
        subtypes=scatter.SUBTYPES,
        build_spec=scatter.build_spec,
        render=scatter.render,
    ),
    "flow": ClassPlan(
        label="flow",
        subtypes=flow.SUBTYPES,
        build_spec=flow.build_spec,
        render=flow.render,
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


def load_class_counts(path: str | Path) -> dict[str, int]:
    """
    Read a YAML plan of per-class sample counts and validate it against the
    current taxonomy.

    The file is a flat mapping of tier-1 class name to a non-negative integer::

        bar: 1500
        waterfall: 1800
        other: 2100

    Every key must be a current tier-1 label (``taxonomy.TIER1_LABELS``) -- a
    stale or misspelt name such as ``bar_vertical`` is a hard error, not a
    silently skipped class, which is the whole point of validating here rather
    than discovering an empty class folder after a long run. Values must be
    non-negative integers; ``0`` means "skip this class". Classes absent from
    the file are simply not returned; the caller decides what to do with them.

    Raises ``ValueError`` on any structural or taxonomy problem and ``OSError``
    if the file cannot be read.
    """
    import yaml  # deferred: only needed when --counts-file is used

    text = Path(path).read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: not valid YAML -- {exc}") from exc

    if data is None:
        raise ValueError(f"{path}: file is empty")
    if not isinstance(data, dict):
        raise ValueError(
            f"{path}: expected a mapping of 'class: count', got a {type(data).__name__}"
        )

    valid = set(TIER1_LABELS)
    counts: dict[str, int] = {}
    unknown: list[str] = []
    bad_value: list[str] = []
    for key, value in data.items():
        name = str(key)
        if name not in valid:
            unknown.append(name)
            continue
        # bool is a subclass of int -- reject `bar: true`, which is almost
        # certainly a mistake, rather than counting it as 1.
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            bad_value.append(f"{name}: {value!r}")
            continue
        counts[name] = value

    if unknown:
        raise ValueError(
            f"{path}: not current tier-1 classes: {', '.join(sorted(unknown))}. "
            f"Valid classes are: {', '.join(TIER1_LABELS)}"
        )
    if bad_value:
        raise ValueError(
            f"{path}: counts must be non-negative integers; got {', '.join(bad_value)}"
        )
    if not counts:
        raise ValueError(f"{path}: no class counts found")
    return counts
