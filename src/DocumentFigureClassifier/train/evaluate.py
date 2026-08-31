"""
Benchmark a fine-tuned 12-class checkpoint against a held-out split.

Reports model quality -- accuracy, per-class precision / recall / F1, macro-F1,
and the confusion matrix -- plus the cross-entropy loss (mean NLL), the single
number to watch across runs: it rewards being right and confident and punishes
being confidently wrong, so it reflects both label correctness and confidence.
Scope is model quality only; any downstream confidence threshold is decided
separately and is deliberately not measured here.

Uses the full 12-way softmax (predict_proba), so nothing is thrown away.

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
    """Run the model over the split, returning (probs [M,12], targets [M])."""
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
    tp = cm.diagonal()

    print(f"\nimages evaluated: {len(targets)}")
    print(f"accuracy:         {acc:.3f}")
    print(f"cross-entropy:    {nll:.4f}  (mean NLL)")

    print(f"\n{'label':16s} {'support':>7s} {'prec':>6s} {'rec':>6s} {'f1':>6s}")
    f1s_present: list[float] = []
    for c, lbl in enumerate(TIER1_LABELS):
        sup = int(support[c])
        if sup == 0 and int(predicted[c]) == 0:
            print(f"{lbl:16s} {sup:7d} {'  N/A':>6s} {'  N/A':>6s} {'  N/A':>6s}")
            continue
        prec = (tp[c] / predicted[c]).item() if predicted[c] > 0 else float("nan")
        rec = (tp[c] / support[c]).item() if support[c] > 0 else float("nan")
        if prec == prec and rec == rec and (prec + rec) > 0:  # not nan
            f1 = 2 * prec * rec / (prec + rec)
        else:
            f1 = float("nan")
        if sup > 0 and f1 == f1:
            f1s_present.append(f1)

        def fmt(x: float) -> str:
            return f"{x:6.3f}" if x == x else "   N/A"

        print(f"{lbl:16s} {sup:7d} {fmt(prec)} {fmt(rec)} {fmt(f1)}")

    macro = sum(f1s_present) / len(f1s_present) if f1s_present else float("nan")
    print(f"\nmacro-F1 (over {len(f1s_present)} classes present in this split): {macro:.3f}")

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
    args = ap.parse_args()

    items = load_split(args.split_index, args.split)
    print(f"loaded {len(items)} images from split '{args.split}'")
    clf = FigureClassifier(args.model, device=args.device)
    probs, targets = collect_probs(clf, items)
    print_report(probs, targets)


if __name__ == "__main__":
    main()
