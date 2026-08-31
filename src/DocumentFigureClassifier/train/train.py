"""
Fine-tuning loop for the 12-class figure classifier.

Reads the split index built by split.py (the single source of truth for which
image is in which split), fine-tunes the EfficientNet backbone with a fresh
12-way head (model.build_model), and writes the result with `save_pretrained`
so `model.FigureClassifier` can load it straight back.

Training loss is class-weighted cross-entropy (categorical cross-entropy /
negative log-likelihood); the weights counter the class imbalance. Validation
loss is reported *unweighted*, so it is an honest average NLL comparable across
runs, alongside plain top-1 accuracy.

    PYTHONPATH=src .venv/Scripts/python.exe -m DocumentFigureClassifier.train.train \
        --split-index data/splits/index.jsonl --out models/tier1-smoke \
        --epochs 3 --batch-size 8
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from DocumentFigureClassifier.model import build_model, make_transform
from DocumentFigureClassifier.taxonomy import TIER1_LABELS
from DocumentFigureClassifier.train.dataset import FigureDataset, Item, load_split


def class_weights(items: list[Item], device: str) -> torch.Tensor:
    """Inverse-frequency weights for the training loss. Absent classes are
    clamped to a count of 1 so their weight stays finite (they never appear as a
    target, so the exact value is irrelevant)."""
    counts = torch.zeros(len(TIER1_LABELS))
    for _, label_id in items:
        counts[label_id] += 1
    counts = counts.clamp(min=1.0)
    return (counts.sum() / (len(TIER1_LABELS) * counts)).to(device)


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: str) -> tuple[float, float]:
    """Unweighted mean cross-entropy and top-1 accuracy over a loader."""
    criterion = torch.nn.CrossEntropyLoss()  # unweighted: honest average NLL
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x).logits
        total_loss += criterion(logits, y).item() * len(y)
        correct += int((logits.argmax(1) == y).sum().item())
        n += len(y)
    if n == 0:
        return float("nan"), float("nan")
    return total_loss / n, correct / n


def summarize(name: str, items: list[Item]) -> None:
    by_label = Counter(TIER1_LABELS[i] for _, i in items)
    present = sum(1 for lbl in TIER1_LABELS if by_label[lbl])
    print(f"{name}: {len(items)} images across {present}/{len(TIER1_LABELS)} classes")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-index", type=Path, default=Path("data/splits/index.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("models/tier1-smoke"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if not args.split_index.exists():
        raise SystemExit(
            f"{args.split_index} not found -- run DocumentFigureClassifier.train.split first"
        )

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    train_items = load_split(args.split_index, "train")
    val_items = load_split(args.split_index, "val")
    summarize("train", train_items)
    summarize("val", val_items)
    if not train_items:
        raise SystemExit("train split is empty")

    train_loader = DataLoader(
        FigureDataset(train_items, make_transform(train=True)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        FigureDataset(val_items, make_transform(train=False)),
        batch_size=args.batch_size,
    )

    device = args.device
    model = build_model().to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights(train_items, device))
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            loss = criterion(model(x).logits, y)
            loss.backward()
            optim.step()
            run_loss += loss.item() * len(y)
            n += len(y)
        train_loss = run_loss / max(n, 1)
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(
            f"epoch {epoch}: train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}"
        )

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    print(f"saved fine-tuned model to {args.out}")


if __name__ == "__main__":
    main()
