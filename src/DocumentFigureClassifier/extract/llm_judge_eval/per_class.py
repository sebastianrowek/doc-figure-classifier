"""Level 2 -- per-category metrics.

Confusion matrix + precision / recall / F1 per class, computed over the pooled
per-call rows via scikit-learn. Recall here == the class's per-call accuracy;
precision exposes classes the model *over*-predicts (e.g. 'other' used as a
catch-all). Also carries per-class mean confidence and support.

Delete this file and its ``report.BLOCKS`` entry to drop the level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from DocumentFigureClassifier.taxonomy import TIER1_LABELS

LABEL = "Per-class"


def _scored(df: pd.DataFrame) -> pd.DataFrame:
    """Rows with a usable prediction; failed calls map to a sentinel so they
    still count against the true class in recall."""
    d = df.copy()
    d["pred_str"] = d["pred"].astype("object").where(d["pred"].notna(), "__none__")
    d["true_str"] = d["true"].astype("object")
    return d


def confusion(df: pd.DataFrame, labels=TIER1_LABELS, normalize: bool = True) -> pd.DataFrame:
    """Row-normalized (true-conditional) confusion matrix as a labelled frame."""
    d = _scored(df)
    cm = confusion_matrix(d["true_str"], d["pred_str"], labels=list(labels))
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    if normalize:
        with np.errstate(invalid="ignore"):
            cm_df = cm_df.div(cm_df.sum(axis=1).replace(0, np.nan), axis=0)
    return cm_df


def compute(df: pd.DataFrame, labels=TIER1_LABELS, **_) -> dict:
    d = _scored(df)
    labels = list(labels)
    p, r, f1, support = precision_recall_fscore_support(
        d["true_str"], d["pred_str"], labels=labels, zero_division=0)

    conf_by_class = df.groupby("true", observed=True)["confidence"].mean()
    imgs_by_class = df.groupby("true", observed=True)["file"].nunique()

    per_class = pd.DataFrame({
        "class": labels,
        "images": [int(imgs_by_class.get(c, 0)) for c in labels],
        "calls": [int(support[i]) for i in range(len(labels))],
        "precision": p,
        "recall": r,          # == per-call accuracy for the class
        "f1": f1,
        "mean_conf": [float(conf_by_class.get(c, np.nan)) for c in labels],
    })
    return {
        "per_class": per_class,
        "confusion_norm": confusion(df, labels, normalize=True),
        "confusion_counts": confusion(df, labels, normalize=False),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": float(np.average(f1, weights=support)) if support.sum() else 0.0,
    }


def table(result: dict) -> pd.DataFrame:
    t = result["per_class"].copy()
    for col in ("precision", "recall", "f1", "mean_conf"):
        t[col] = t[col].map(lambda x: f"{x:.3f}")
    return t.sort_values("class")
