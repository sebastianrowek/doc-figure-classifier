"""
Benchmark a fine-tuned 14-class checkpoint against a held-out split.

Reports model quality -- accuracy, per-class precision / recall / F1, macro-F1,
and the confusion matrix -- plus the cross-entropy loss (mean NLL), the single
number to watch across runs: it rewards being right and confident and punishes
being confidently wrong, so it reflects both label correctness and confidence.
Scope is model quality only; any downstream confidence threshold is decided
separately and is deliberately not measured here.

macro-F1 here uses the same definition train.py selects checkpoints on
(macro_f1), so the reported number matches the selected one.

Uses the full 14-way softmax (predict_proba), so nothing is thrown away.

    PYTHONPATH=src .venv/Scripts/python.exe -m DocumentFigureClassifier.train.evaluate \
        --model models/tier1-smoke --split-index data/splits/index.jsonl --split test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from DocumentFigureClassifier.model import FigureClassifier
from DocumentFigureClassifier.taxonomy import TIER1_LABELS
from DocumentFigureClassifier.train.dataset import load_split

N = len(TIER1_LABELS)


def collect_probs(clf: FigureClassifier, items, chunk: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the model over the split, returning (probs [M,N], targets [M])."""
    all_probs: list[torch.Tensor] = []
    targets: list[int] = []
    for i in range(0, len(items), chunk):
        batch = items[i : i + chunk]
        imgs = [Image.open(p).convert("RGB") for p, _ in batch]
        all_probs.append(clf.predict_proba(imgs))
        targets.extend(label_id for _, label_id in batch)
    probs = torch.cat(all_probs) if all_probs else torch.empty(0, N)
    return probs, torch.tensor(targets, dtype=torch.long)


def confusion(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """C[true, pred] counts, N x N."""
    cm = torch.zeros(N, N, dtype=torch.long)
    for t, p in zip(targets.tolist(), preds.tolist()):
        cm[t, p] += 1
    return cm


def per_class_prf(cm: torch.Tensor) -> list[tuple[float, float, float, int, int]]:
    """(precision, recall, f1, support, predicted) per class over all N classes.

    Standard sklearn convention with zero_division=0: an undefined precision
    (nothing predicted for the class) or recall counts as 0, so f1 is 0 rather
    than dropped. Callers decide which classes to display / average."""
    support, predicted, tp = cm.sum(1), cm.sum(0), cm.diagonal()
    rows: list[tuple[float, float, float, int, int]] = []
    for c in range(N):
        sup, pred = int(support[c]), int(predicted[c])
        prec = (tp[c] / predicted[c]).item() if pred > 0 else 0.0
        rec = (tp[c] / support[c]).item() if sup > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        rows.append((prec, rec, f1, sup, pred))
    return rows


def macro_f1(cm: torch.Tensor) -> float:
    """Macro-F1 averaged over the classes with support in this split (sklearn
    'macro', zero_division=0). Zero-recall classes count as 0, not dropped, so
    this is the single honest number to select checkpoints on and to report.
    NaN only if no class has any support."""
    present = [f1 for (_, _, f1, sup, _) in per_class_prf(cm) if sup > 0]
    return sum(present) / len(present) if present else float("nan")


def plot_confusion(cm: torch.Tensor, out: Path, split: str) -> None:
    """Save a row-normalised confusion-matrix heatmap (rows=true, cols=pred).

    Row-normalised = each row shows where that true class's instances went, so
    recall is the diagonal; robust to the class imbalance that would otherwise
    make raw counts unreadable. Classes with no support are shown blank."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(skipped confusion plot: {e})")
        return

    support = cm.sum(1, keepdim=True).clamp(min=1)
    norm = (cm.float() / support).numpy()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(N)); ax.set_yticks(range(N))
    ax.set_xticklabels(TIER1_LABELS, rotation=90, fontsize=7)
    ax.set_yticklabels(TIER1_LABELS, fontsize=7)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"confusion (row-normalised) -- {split}")
    for i in range(N):
        for j in range(N):
            if cm[i, j] > 0:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=6,
                        color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote confusion plot to {out}")


def print_report(probs: torch.Tensor, targets: torch.Tensor) -> None:
    if len(targets) == 0:
        print("split is empty -- nothing to evaluate")
        return

    preds = probs.argmax(1)
    acc = (preds == targets).float().mean().item()

    # cross-entropy / mean NLL, probs clamped so a confident miss isn't infinite
    p_true = probs[torch.arange(len(targets)), targets].clamp_min(1e-12)
    nll = -p_true.log().mean().item()

    cm = confusion(preds, targets)
    support = cm.sum(1)          # true instances per class
    predicted = cm.sum(0)        # predicted instances per class
    rows = per_class_prf(cm)

    print(f"\nimages evaluated: {len(targets)}")
    print(f"accuracy:         {acc:.3f}")
    print(f"cross-entropy:    {nll:.4f}  (mean NLL)")

    print(f"\n{'label':16s} {'support':>7s} {'prec':>6s} {'rec':>6s} {'f1':>6s}")
    n_present = 0
    for c, lbl in enumerate(TIER1_LABELS):
        prec, rec, f1, sup, pred = rows[c]
        if sup == 0 and pred == 0:
            print(f"{lbl:16s} {sup:7d} {'  N/A':>6s} {'  N/A':>6s} {'  N/A':>6s}")
            continue
        # Precision is undefined with nothing predicted; recall/F1 undefined with
        # no support. Only support>0 classes enter macro-F1 (as 0 if zero-recall).
        prec_s = f"{prec:6.3f}" if pred > 0 else "   N/A"
        rec_s = f"{rec:6.3f}" if sup > 0 else "   N/A"
        f1_s = f"{f1:6.3f}" if sup > 0 else "   N/A"
        if sup > 0:
            n_present += 1
        print(f"{lbl:16s} {sup:7d} {prec_s} {rec_s} {f1_s}")

    macro = macro_f1(cm)
    print(f"\nmacro-F1 (over {n_present} classes present in this split): {macro:.3f}")

    # Confusion matrix over classes that actually appear (as true or predicted).
    active = [c for c in range(N) if support[c] > 0 or predicted[c] > 0]
    print("\nconfusion matrix (rows = true, cols = predicted):")
    print("  index: " + ", ".join(f"{c}={TIER1_LABELS[c]}" for c in active))
    header = "      " + "".join(f"{c:>5d}" for c in active)
    print(header)
    for r in active:
        row = "".join(f"{int(cm[r, c]):>5d}" for c in active)
        print(f"  {r:>2d} {TIER1_LABELS[r][:12]:12s}{row}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True, help="fine-tuned checkpoint dir")
    ap.add_argument("--split-index", type=Path, default=Path("data/splits/index.jsonl"))
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--plot", type=Path, default=None, help="save a confusion-matrix heatmap PNG here")
    args = ap.parse_args()

    items = load_split(args.split_index, args.split)
    print(f"loaded {len(items)} images from split '{args.split}'")
    clf = FigureClassifier(args.model, device=args.device)
    probs, targets = collect_probs(clf, items)
    print_report(probs, targets)
    if args.plot is not None and len(targets):
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        plot_confusion(confusion(probs.argmax(1), targets), args.plot, args.split)


if __name__ == "__main__":
    main()
