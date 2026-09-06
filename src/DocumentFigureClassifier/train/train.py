"""
Stage-1 fine-tuning loop for the 14-class figure classifier.

Reads the split index built by split.py (the single source of truth for which
image is in which split) and adapts docling's EfficientNet backbone to our 14
Tier-1 labels with a fresh head (model.build_model). The result is written with
`save_pretrained` so `model.FigureClassifier` can load it straight back.

Stage 1 = adaptation on synth + (table-capped) real. Two mechanisms protect the
good pretrained features while the random head learns:

  * Freeze then unfreeze. For the first --freeze-epochs the backbone is frozen
    and only the head trains, so the head's large early gradients can't wreck the
    backbone. Then the backbone unfreezes.
  * Differential learning rate. Once unfrozen, the backbone trains at a low LR
    (--backbone-lr) and the head at a higher one (--head-lr): standard
    discriminative fine-tuning.

Training loss is class-weighted cross-entropy (the weights counter residual
imbalance). Validation is 100% real; the best checkpoint is chosen on real-val
macro-F1 (tie-break: lower val loss), which is the metric we actually care about
under a table-heavy distribution -- not accuracy. Val loss is reported
*unweighted*, an honest average NLL comparable across runs.

Stage 2 (real-only close-out) is deliberately not built here yet.

    PYTHONPATH=src .venv/Scripts/python.exe -m DocumentFigureClassifier.train.train \
        --split-index data/splits/index.jsonl --out models/tier1-stage1 \
        --epochs 6 --freeze-epochs 1 --batch-size 16 --device auto
"""

from __future__ import annotations

import argparse
import math
import random
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from DocumentFigureClassifier.model import (
    build_model,
    make_transform,
    param_groups,
    set_backbone_trainable,
)
from DocumentFigureClassifier.taxonomy import TIER1_LABELS
from DocumentFigureClassifier.train.dataset import FigureDataset, Item, load_split
from DocumentFigureClassifier.train.evaluate import macro_f1

N = len(TIER1_LABELS)


def resolve_device(choice: str) -> str:
    """'auto' -> cuda if available else cpu; anything else is passed through."""
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return choice


def class_weights(items: list[Item], device: str) -> torch.Tensor:
    """Inverse-frequency weights for the training loss. Absent classes are
    clamped to a count of 1 so their weight stays finite (they never appear as a
    target, so the exact value is irrelevant)."""
    counts = torch.zeros(N)
    for _, label_id in items:
        counts[label_id] += 1
    counts = counts.clamp(min=1.0)
    return (counts.sum() / (N * counts)).to(device)


@torch.no_grad()
def evaluate(model, loader: DataLoader, device: str) -> tuple[float, float, float]:
    """Unweighted mean cross-entropy, top-1 accuracy, and macro-F1 over a loader.

    macro-F1 averages per-class F1 over the classes actually present in this
    split (support > 0), matching evaluate.py; it is NaN if the split is empty."""
    criterion = torch.nn.CrossEntropyLoss()  # unweighted: honest average NLL
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    cm = torch.zeros(N, N, dtype=torch.long)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x).logits
        total_loss += criterion(logits, y).item() * len(y)
        preds = logits.argmax(1)
        correct += int((preds == y).sum().item())
        n += len(y)
        for t, p in zip(y.tolist(), preds.tolist()):
            cm[t, p] += 1
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    # Same macro-F1 definition evaluate.py reports, so selected == reported.
    return total_loss / n, correct / n, macro_f1(cm)


def summarize(name: str, items: list[Item]) -> None:
    by_label = Counter(TIER1_LABELS[i] for _, i in items)
    present = sum(1 for lbl in TIER1_LABELS if by_label[lbl])
    print(f"{name}: {len(items)} images across {present}/{N} classes")


def write_history(history: list[dict], out: Path) -> None:
    """Write history.csv and loss_curve.png (train vs val loss per epoch, with
    val macro-F1 on a secondary axis). Test is intentionally absent -- it is the
    held-out final estimate and is never evaluated per epoch."""
    import csv

    with (out / "history.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["epoch", "train_loss", "val_loss", "val_acc", "val_macro_f1"])
        w.writeheader()
        w.writerows(history)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # plotting is a convenience, never fail the run for it
        print(f"(skipped loss_curve.png: {e})")
        return

    ep = [h["epoch"] for h in history]
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(ep, [h["train_loss"] for h in history], "o-", color="tab:blue", label="train loss")
    ax1.plot(ep, [h["val_loss"] for h in history], "s-", color="tab:orange", label="val loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("cross-entropy loss")
    ax1.set_xticks(ep)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(ep, [h["val_macro_f1"] for h in history], "^--", color="tab:green", label="val macro-F1")
    ax2.set_ylabel("val macro-F1")
    ax2.set_ylim(0, 1)

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="best", fontsize=8)
    fig.suptitle("Stage-1 training")
    fig.tight_layout()
    fig.savefig(out / "loss_curve.png", dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split-index", type=Path, default=Path("data/splits/index.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("models/tier1-stage1"))
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--freeze-epochs", type=int, default=1, help="epochs to keep the backbone frozen (head-only)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--backbone-lr", type=float, default=1e-5, help="LR for the pretrained backbone once unfrozen")
    ap.add_argument("--head-lr", type=float, default=1e-3, help="LR for the fresh 14-way head")
    ap.add_argument("--patience", type=int, default=0, help="early-stop after N epochs with no macro-F1 gain (0=off)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda|...")
    args = ap.parse_args()

    if not args.split_index.exists():
        raise SystemExit(
            f"{args.split_index} not found -- run DocumentFigureClassifier.train.split first"
        )

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = resolve_device(args.device)
    print(f"device: {device}")

    train_items = load_split(args.split_index, "train")
    val_items = load_split(args.split_index, "val")
    summarize("train", train_items)
    summarize("val", val_items)
    if not train_items:
        raise SystemExit("train split is empty")
    if not val_items:
        print("warning: val split is empty -- best-checkpoint selection falls back to the last epoch")

    train_loader = DataLoader(
        FigureDataset(train_items, make_transform(train=True)),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        FigureDataset(val_items, make_transform(train=False)),
        batch_size=args.batch_size,
    )

    model = build_model().to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights(train_items, device))
    optim = torch.optim.AdamW(param_groups(model, args.backbone_lr, args.head_lr))

    # Start frozen unless --freeze-epochs 0. Both param groups stay in the
    # optimizer throughout; frozen params just receive no gradient.
    frozen = args.freeze_epochs > 0
    set_backbone_trainable(model, not frozen)
    print(
        f"backbone {'frozen (head-only)' if frozen else 'trainable'} for epoch 1; "
        f"backbone_lr={args.backbone_lr:g} head_lr={args.head_lr:g}"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    best_macro, best_loss, best_epoch = -1.0, math.inf, 0
    saved_any = False
    epochs_since_best = 0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        if frozen and epoch == args.freeze_epochs + 1:
            set_backbone_trainable(model, True)
            frozen = False
            print("  unfroze backbone")

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
        val_loss, val_acc, val_macro = evaluate(model, val_loader, device)
        print(
            f"epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.3f}  val_macroF1={val_macro:.3f}"
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "val_acc": round(val_acc, 6),
                "val_macro_f1": round(val_macro, 6),
            }
        )

        # Best checkpoint on real-val macro-F1 (tie-break: lower val loss). If val
        # is empty, val_macro is NaN and we save every epoch so --out is the last.
        improved = math.isnan(val_macro) or (
            val_macro > best_macro + 1e-6
            or (abs(val_macro - best_macro) <= 1e-6 and val_loss < best_loss)
        )
        if improved:
            if not math.isnan(val_macro):
                best_macro, best_loss, best_epoch = val_macro, val_loss, epoch
            model.save_pretrained(args.out)
            saved_any = True
            epochs_since_best = 0
            print(f"  ^ saved (best so far) to {args.out}")
        else:
            epochs_since_best += 1
            if args.patience and epochs_since_best >= args.patience:
                print(f"  early stop: no macro-F1 gain for {args.patience} epochs")
                break

    if not saved_any:  # e.g. macro-F1 never defined and never NaN-saved
        model.save_pretrained(args.out)
        print(f"saved final model to {args.out}")
    elif best_epoch:
        print(f"best: epoch {best_epoch}  val_macroF1={best_macro:.3f} -> {args.out}")

    write_history(history, args.out)
    print(f"wrote {args.out / 'history.csv'} and {args.out / 'loss_curve.png'}")


if __name__ == "__main__":
    main()
