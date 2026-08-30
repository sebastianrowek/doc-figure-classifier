"""Calibration & probabilistic-loss block (threshold-free).

Per the standing eval note, judge quality is measured as model quality +
cross-entropy, with NO downstream routing threshold anywhere in here. Two loss
views, because the model emits a single (label, confidence) rather than a full
distribution:

  (a) confidence log-loss / ECE -- treats ``confidence`` as the probability of
      the *predicted* label and spreads the rest uniformly. Measures whether the
      model's self-reported confidence is honest (calibration of one call).
  (b) empirical log-loss -- builds each image's class distribution from its N
      repeated votes and scores that against the truth. Measures the quality of
      the distribution the model implicitly samples from.

Delete this file and its ``report.BLOCKS`` entry to drop all calibration output.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss

from DocumentFigureClassifier.taxonomy import TIER1_LABELS

LABEL = "Calibration & loss"
_EPS = 1e-12


def _confidence_matrix(df: pd.DataFrame, labels: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-call (P over classes, y_true idx, keep-mask) from self-reported conf.

    p[pred] = confidence; remaining (1-confidence) split evenly over the other
    K-1 classes. Rows with no prediction/confidence are dropped (mask).
    """
    idx = {c: i for i, c in enumerate(labels)}
    K = len(labels)
    keep = df["pred"].notna() & df["confidence"].notna()
    d = df[keep]
    P = np.full((len(d), K), 0.0)
    y = np.empty(len(d), dtype=int)
    for row, (_, r) in enumerate(d.iterrows()):
        c = float(r["confidence"])
        pj = idx[str(r["pred"])]
        P[row, :] = (1.0 - c) / (K - 1)
        P[row, pj] = c
        y[row] = idx[str(r["true"])]
    P = np.clip(P, _EPS, 1.0)
    P /= P.sum(axis=1, keepdims=True)
    return P, y, keep.to_numpy()


def _empirical_matrix(df: pd.DataFrame, labels: list[str], smooth: float = 0.5):
    """Per-image (P over classes, y_true idx) from the N repeated votes.

    Laplace-smoothed so a unanimous-but-wrong image gets a large-but-finite loss
    instead of infinity.
    """
    idx = {c: i for i, c in enumerate(labels)}
    K = len(labels)
    files, P, y = [], [], []
    for f, g in df.groupby("file", observed=True):
        counts = np.full(K, smooth)
        for pr in g["pred"].dropna().astype(str):
            counts[idx[pr]] += 1.0
        P.append(counts / counts.sum())
        y.append(idx[str(g["true"].iloc[0])])
        files.append(f)
    return np.array(P), np.array(y), files


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10):
    """ECE, MCE and the per-bin reliability curve (equal-width confidence bins)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = mce = 0.0
    rows = []
    N = len(conf)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if not m.any():
            rows.append({"bin_lo": lo, "bin_hi": hi, "count": 0,
                         "mean_conf": np.nan, "accuracy": np.nan, "gap": np.nan})
            continue
        acc = correct[m].mean()
        cf = conf[m].mean()
        gap = abs(acc - cf)
        w = m.sum() / N
        ece += w * gap
        mce = max(mce, gap)
        rows.append({"bin_lo": lo, "bin_hi": hi, "count": int(m.sum()),
                     "mean_conf": float(cf), "accuracy": float(acc), "gap": float(gap)})
    return float(ece), float(mce), pd.DataFrame(rows)


def compute(df: pd.DataFrame, labels=TIER1_LABELS, n_bins: int = 10, **_) -> dict:
    labels = list(labels)

    # (a) confidence-based
    Pc, yc, keep = _confidence_matrix(df, labels)
    conf_logloss = float(log_loss(yc, Pc, labels=list(range(len(labels))))) if len(yc) else None
    d_keep = df[keep]
    conf = d_keep["confidence"].to_numpy(dtype=float)
    correct = d_keep["ok"].to_numpy(dtype=bool)
    ece, mce, reliability = expected_calibration_error(conf, correct, n_bins)
    # top-label Brier: (confidence - correct)^2
    brier_top = float(np.mean((conf - correct.astype(float)) ** 2)) if len(conf) else None

    # (b) empirical-vote based
    Pe, ye, _ = _empirical_matrix(df, labels)
    emp_logloss = float(log_loss(ye, Pe, labels=list(range(len(labels))))) if len(ye) else None

    conf_correct = float(df.loc[df["ok"], "confidence"].mean())
    conf_wrong = float(df.loc[~df["ok"], "confidence"].mean())

    return {
        "confidence_logloss": conf_logloss,
        "empirical_logloss": emp_logloss,
        "ece": ece,
        "mce": mce,
        "brier_top_label": brier_top,
        "mean_conf_correct": conf_correct,
        "mean_conf_wrong": conf_wrong,
        "reliability": reliability,
    }


def table(result: dict) -> pd.DataFrame:
    rows = [
        ("Cross-entropy — confidence-based (per call)", result["confidence_logloss"]),
        ("Cross-entropy — empirical votes (per image)", result["empirical_logloss"]),
        ("ECE (expected calibration error)", result["ece"]),
        ("MCE (max calibration error)", result["mce"]),
        ("Brier (top label)", result["brier_top_label"]),
        ("Mean confidence when correct", result["mean_conf_correct"]),
        ("Mean confidence when wrong", result["mean_conf_wrong"]),
    ]
    fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else v
    return pd.DataFrame([(k, fmt(v)) for k, v in rows], columns=["metric", "value"])
