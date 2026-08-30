"""LLM-judge evaluation toolkit.

Reads a raw per-call results file (see :mod:`loader` for the contract) produced
by the classification runner and reports three layered levels of evaluation:

    1. overall      -- pooled per-call accuracy (micro + macro), cost, CIs
    2. per_class    -- confusion matrix, precision / recall / F1
    3. per_file     -- per-image vote distribution, agreement, buckets

plus threshold-free ``calibration`` (cross-entropy / ECE / reliability) and
``consistency`` (repeat-call stability) blocks, and ``plots``.

Each block is an independent module; ``report`` wires them via editable
registries so any block or figure can be deleted without touching the others.
"""
from __future__ import annotations

from . import loader  # noqa: F401

__all__ = ["loader", "overall", "per_class", "per_file",
           "calibration", "consistency", "plots", "report"]
