"""Visualizations (seaborn / matplotlib). Each function is standalone: it takes
the result dict of one metric block plus an output path and writes a PNG. The
``PLOTS`` registry in :mod:`report` decides which run -- delete a function here
and its registry entry to drop just that figure.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")            # file-only backend, no display needed
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")


def _save(fig, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def confusion_heatmap(per_class_result: dict, out: Path) -> Path:
    """Row-normalized confusion matrix -- the headline diagnostic."""
    cm = per_class_result["confusion_norm"]
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="rocket_r", vmin=0, vmax=1,
                cbar_kws={"label": "P(pred | true)"}, linewidths=.5,
                linecolor="white", ax=ax)
    ax.set_xlabel("predicted"); ax.set_ylabel("true (ground truth)")
    ax.set_title("Confusion matrix (row-normalized)")
    return _save(fig, out)


def reliability_diagram(calibration_result: dict, out: Path) -> Path:
    """Confidence vs empirical accuracy, with the calibration gap shaded."""
    rel = calibration_result["reliability"].dropna(subset=["accuracy"])
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect calibration")
    ax.plot(rel["mean_conf"], rel["accuracy"], "o-", color="#c44", label="model")
    for _, r in rel.iterrows():
        ax.vlines(r["mean_conf"], r["accuracy"], r["mean_conf"],
                  color="#c44", alpha=0.3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("mean confidence"); ax.set_ylabel("empirical accuracy")
    ax.set_title(f"Reliability  (ECE={calibration_result['ece']:.3f})")
    ax.legend(loc="upper left")
    return _save(fig, out)


def per_class_accuracy_bars(per_class_result: dict, out: Path) -> Path:
    """Per-class recall (== per-call accuracy), worst -> best."""
    t = per_class_result["per_class"].sort_values("recall")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(data=t, y="class", x="recall", hue="class",
                palette="viridis", legend=False, ax=ax)
    ax.set_xlim(0, 1); ax.set_xlabel("recall / per-call accuracy"); ax.set_ylabel("")
    ax.set_title("Per-class accuracy")
    for i, v in enumerate(t["recall"]):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center")
    return _save(fig, out)


def per_class_prf_bars(per_class_result: dict, out: Path) -> Path:
    """Precision / recall / F1 grouped by class."""
    t = per_class_result["per_class"].melt(
        id_vars="class", value_vars=["precision", "recall", "f1"],
        var_name="metric", value_name="score")
    fig, ax = plt.subplots(figsize=(13, 7))
    sns.barplot(data=t, x="class", y="score", hue="metric", ax=ax)
    ax.set_ylim(0, 1); ax.set_xlabel(""); ax.set_title("Precision / recall / F1 by class")
    ax.tick_params(axis="x", rotation=45)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    return _save(fig, out)


def confidence_hist(df, out: Path) -> Path:
    """Confidence distribution split by correct vs wrong -- a good judge is more
    confident when it is right."""
    d = df.dropna(subset=["confidence"]).copy()
    d["outcome"] = np.where(d["ok"], "correct", "wrong")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data=d, x="confidence", hue="outcome", bins=20, stat="density",
                 common_norm=False, element="step", palette={"correct": "#2a9d8f",
                 "wrong": "#e76f51"}, ax=ax)
    ax.set_title("Confidence by outcome"); ax.set_xlabel("confidence")
    return _save(fig, out)


def agreement_hist(consistency_or_perfile: dict, out: Path) -> Path:
    """Distribution of per-image agreement (self-consistency)."""
    key = "per_file" if "per_file" in consistency_or_perfile else "per_class"
    src = consistency_or_perfile.get("per_file")
    vals = src["agreement"] if src is not None else consistency_or_perfile["per_class"]["agreement"]
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.histplot(vals, bins=np.linspace(0, 1, 11), color="#4c72b0", ax=ax)
    ax.set_xlim(0, 1); ax.set_xlabel("modal-vote fraction per image")
    ax.set_title("Self-consistency across repeated calls")
    return _save(fig, out)
