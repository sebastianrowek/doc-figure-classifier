"""Repeat-call consistency block -- the metrics that only exist because you run
N calls per image. Answers "how stable is the judge, run to run?".

  * agreement    -- mean modal-vote fraction (1.0 == always the same answer)
  * flip_rate    -- share of images that produced >1 distinct label
  * mean entropy -- average indecision (bits) across images
  * conf_std     -- average within-image spread of confidence

Reported overall and per class. Delete this file and its ``report.BLOCKS`` entry
to drop it. (Shares the per-image summary shape with :mod:`per_file`; that module
can be deleted independently.)
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

LABEL = "Consistency (repeat calls)"


def _per_image(df: pd.DataFrame) -> pd.DataFrame:
    recs = []
    for f, g in df.groupby("file", observed=True):
        votes = Counter(g["pred"].astype("object").where(g["pred"].notna(), "__none__"))
        n = len(g)
        modal_n = votes.most_common(1)[0][1]
        p = np.array([c / n for c in votes.values()])
        recs.append({
            "file": f, "true": g["true"].iloc[0],
            "agreement": modal_n / n,
            "distinct": len(votes),
            "entropy_bits": float(-(p * np.log2(p)).sum()),
            "conf_std": float(g["confidence"].std(ddof=0)),
        })
    return pd.DataFrame(recs)


def compute(df: pd.DataFrame, **_) -> dict:
    pi = _per_image(df)
    by_class = (pi.groupby("true", observed=True)
                  .agg(agreement=("agreement", "mean"),
                       flip_rate=("distinct", lambda s: float((s > 1).mean())),
                       mean_entropy=("entropy_bits", "mean"))
                  .reset_index())
    return {
        "agreement_mean": float(pi["agreement"].mean()),
        "flip_rate": float((pi["distinct"] > 1).mean()),
        "mean_entropy_bits": float(pi["entropy_bits"].mean()),
        "mean_conf_std": float(pi["conf_std"].mean()),
        "per_class": by_class,
    }


def table(result: dict) -> pd.DataFrame:
    rows = [
        ("Mean agreement (self-consistency)", f"{result['agreement_mean']:.3f}"),
        ("Flip rate (images with >1 label)", f"{result['flip_rate']:.1%}"),
        ("Mean answer entropy (bits)", f"{result['mean_entropy_bits']:.3f}"),
        ("Mean within-image confidence std", f"{result['mean_conf_std']:.3f}"),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])
