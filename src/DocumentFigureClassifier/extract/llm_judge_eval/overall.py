"""Level 1 -- overall metrics (per-call scoring).

Every one of the N x n_images calls is scored independently (no majority-vote
collapsing). Because repeated calls on the same image are correlated, the
confidence interval uses a *cluster* bootstrap that resamples whole images, not
individual calls -- otherwise the CI would be dishonestly tight.

Pure functions: they take the tidy frame from :mod:`loader` and return plain
dicts/DataFrames. Delete this file and its entry in ``report.BLOCKS`` to drop
the whole level.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LABEL = "Overall (per-call)"


def _macro_accuracy(df: pd.DataFrame) -> float:
    """Mean of per-class recall -- each class weighted equally regardless of size."""
    per_class = df.groupby("true", observed=True)["ok"].mean()
    return float(per_class.mean())


def bootstrap_ci(df: pd.DataFrame, stat, n_boot: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Cluster-bootstrap CI for a scalar statistic, resampling images (files)."""
    rng = np.random.default_rng(seed)
    groups = [g for _, g in df.groupby("file", observed=True)]
    n = len(groups)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, n, n)
        sample = pd.concat([groups[i] for i in pick], ignore_index=True)
        vals[b] = stat(sample)
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def compute(df: pd.DataFrame, **_) -> dict:
    n_calls = len(df)
    n_images = df["file"].nunique()
    micro = float(df["ok"].mean())
    macro = _macro_accuracy(df)
    micro_lo, micro_hi = bootstrap_ci(df, lambda d: d["ok"].mean())
    macro_lo, macro_hi = bootstrap_ci(df, _macro_accuracy)

    costs = df["cost"].dropna()
    n_fail = int(df["pred"].isna().sum())

    return {
        "n_images": int(n_images),
        "n_calls": int(n_calls),
        "calls_per_image": round(n_calls / n_images, 2) if n_images else 0,
        "n_classes": int(df["true"].cat.categories.size),
        "accuracy_micro": micro,
        "accuracy_micro_ci95": [micro_lo, micro_hi],
        "accuracy_macro": macro,
        "accuracy_macro_ci95": [macro_lo, macro_hi],
        "failed_calls": n_fail,
        "failed_call_rate": round(n_fail / n_calls, 4) if n_calls else 0.0,
        "cost_total_usd": round(float(costs.sum()), 4) if len(costs) else None,
        "cost_per_call_usd": round(float(costs.mean()), 6) if len(costs) else None,
    }


def table(result: dict) -> pd.DataFrame:
    """One-column summary, ready for ``to_markdown``."""
    def ci(key):
        lo, hi = result[key]
        return f"[{lo:.3f}, {hi:.3f}]"
    rows = [
        ("Images", result["n_images"]),
        ("Calls (scored)", result["n_calls"]),
        ("Calls / image", result["calls_per_image"]),
        ("Accuracy — micro (per-call)", f"{result['accuracy_micro']:.3f}  {ci('accuracy_micro_ci95')}"),
        ("Accuracy — macro (per-class mean)", f"{result['accuracy_macro']:.3f}  {ci('accuracy_macro_ci95')}"),
        ("Failed calls", f"{result['failed_calls']} ({result['failed_call_rate']:.1%})"),
        ("Total cost (USD)", result["cost_total_usd"]),
        ("Cost / call (USD)", result["cost_per_call_usd"]),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])
