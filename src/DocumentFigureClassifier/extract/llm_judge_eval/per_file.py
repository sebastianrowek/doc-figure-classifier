"""Level 3 -- per-file breakdown.

One row per calibration image summarising its N calls: the vote distribution,
modal answer, agreement (self-consistency), answer entropy, mean confidence and
per-call accuracy. The vote distribution is *descriptive only* -- scoring stays
per-call per your choice; this just shows how the repeats landed.

Auto-buckets the images so the interesting ones surface first:
  * always_wrong -- every call wrong -> a bad crop or a mislabel in OUR set (QC signal)
  * flippy       -- >1 distinct label across calls -> the model is genuinely unsure
  * always_right -- every call correct

Delete this file and its ``report.BLOCKS`` entry to drop the level.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd


LABEL = "Per-file"


def _entropy(counts: list[int]) -> float:
    """Shannon entropy (bits) of a vote distribution; 0 == fully consistent."""
    total = sum(counts)
    if total <= 0:
        return 0.0
    p = np.array([c / total for c in counts if c > 0])
    return float(-(p * np.log2(p)).sum())


def _summ(group: pd.DataFrame) -> pd.Series:
    preds = group["pred"].astype("object").where(group["pred"].notna(), "__none__")
    votes = Counter(preds)
    modal_label, modal_n = votes.most_common(1)[0]
    n = len(group)
    true = group["true"].iloc[0]
    vote_str = ", ".join(f"{k}x{v}" for k, v in votes.most_common())
    return pd.Series({
        "true": true,
        "n_calls": n,
        "votes": vote_str,
        "modal_pred": modal_label,
        "agreement": modal_n / n,                 # self-consistency in [1/n, 1]
        "distinct": len(votes),
        "entropy_bits": _entropy(list(votes.values())),
        "accuracy": float(group["ok"].mean()),    # per-call accuracy for this image
        "mean_conf": float(group["confidence"].mean()),
        "conf_std": float(group["confidence"].std(ddof=0)),
    })


def compute(df: pd.DataFrame, **_) -> dict:
    per_file = (df.groupby("file", observed=True).apply(_summ, include_groups=False)
                .reset_index())
    # worst first: lowest accuracy, then most indecisive
    per_file = per_file.sort_values(["accuracy", "entropy_bits"],
                                    ascending=[True, False]).reset_index(drop=True)

    always_wrong = per_file[per_file["accuracy"] == 0.0]
    always_right = per_file[per_file["accuracy"] == 1.0]
    flippy = per_file[per_file["distinct"] > 1]

    return {
        "per_file": per_file,
        "always_wrong": always_wrong,
        "always_right": always_right,
        "flippy": flippy,
        "n_always_wrong": len(always_wrong),
        "n_always_right": len(always_right),
        "n_flippy": len(flippy),
    }


def table(result: dict, which: str = "per_file") -> pd.DataFrame:
    t = result[which].copy()
    for col in ("agreement", "entropy_bits", "accuracy", "mean_conf", "conf_std"):
        if col in t:
            t[col] = t[col].map(lambda x: f"{x:.2f}")
    return t
