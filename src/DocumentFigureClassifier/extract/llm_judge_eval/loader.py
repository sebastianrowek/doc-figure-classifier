"""Raw-results contract + loader for the LLM-judge evaluation.

The classification runner (to be written separately, in ``extract/``) is expected
to append **one JSON object per model call** to a JSONL file. Each record:

    {
      "file":       "bar/bar__web04__quanthub-com.png",  # path relative to data/llm_judge
      "true":       "bar",           # ground-truth Tier-1 label (folder name)
      "pred":       "bar_grouped",   # model label, or null if the call failed / off-schema
      "confidence": 0.72,            # model self-reported confidence in [0,1], or null
      "call_idx":   2,               # 0-based repeat index for this image
      "cost":       0.00013,         # USD billed for this call, or null
      "error":      null             # error string if the call failed, else null
    }

The evaluator never calls the API; it only reads this file. Keeping the two
steps decoupled means the (paid) N-repeat run happens once and every metric /
plot below can be recomputed for free.

This module is the ONLY place that knows the on-disk schema — every metric block
consumes the tidy DataFrame returned by :func:`load_runs`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from DocumentFigureClassifier.taxonomy import TIER1_LABELS

# Canonical column order of the tidy per-call frame handed to every block.
COLUMNS = ["file", "true", "pred", "confidence", "call_idx", "cost", "error", "ok"]


def _true_from_path(file: str) -> str:
    """Ground-truth label is the first path segment (the class folder)."""
    return Path(file).parts[0]


def load_runs(path: str | Path, labels: Iterable[str] = TIER1_LABELS) -> pd.DataFrame:
    """Read a raw-results JSONL into a tidy one-row-per-call DataFrame.

    Adds a boolean ``ok`` column (``pred == true``). Failed calls (``pred`` is
    null) are kept with ``ok = False`` so they weigh on accuracy exactly as a
    wrong answer would -- drop them upstream if you would rather exclude them.
    """
    labels = list(labels)
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        true = r.get("true") or _true_from_path(r["file"])
        pred = r.get("pred")
        rows.append({
            "file": r["file"],
            "true": true,
            "pred": pred,
            "confidence": r.get("confidence"),
            "call_idx": r.get("call_idx", 0),
            "cost": r.get("cost"),
            "error": r.get("error"),
            "ok": (pred is not None) and (pred == true),
        })
    df = pd.DataFrame(rows, columns=COLUMNS)

    # Make the label columns ordered categoricals so every downstream groupby /
    # confusion matrix lists all classes in taxonomy order, even unobserved ones.
    cat = pd.CategoricalDtype(categories=labels, ordered=True)
    df["true"] = df["true"].astype(cat)
    df["pred"] = df["pred"].astype(cat)  # unknown/None -> NaN category
    return df


def validate(df: pd.DataFrame, labels: Iterable[str] = TIER1_LABELS) -> list[str]:
    """Cheap sanity checks; returns a list of human-readable warnings."""
    labels = set(labels)
    warns: list[str] = []
    unknown_true = set(df["true"].dropna().astype(str)) - labels
    if unknown_true:
        warns.append(f"true labels outside taxonomy: {sorted(unknown_true)}")
    n_fail = int(df["pred"].isna().sum())
    if n_fail:
        warns.append(f"{n_fail} call(s) with no valid prediction (counted as wrong)")
    per_img = df.groupby("file", observed=True)["call_idx"].nunique()
    if per_img.nunique() > 1:
        warns.append(f"uneven repeat counts per image: {sorted(per_img.unique())}")
    return warns
