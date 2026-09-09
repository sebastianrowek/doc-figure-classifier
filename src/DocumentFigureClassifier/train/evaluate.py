"""
Benchmark a fine-tuned Tier-1 checkpoint against a held-out split.

Reports model quality -- accuracy, per-class precision / recall / F1, macro-F1,
and the confusion matrix -- plus the cross-entropy loss (mean NLL), the single
number to watch across runs: it rewards being right and confident and punishes
being confidently wrong, so it reflects both label correctness and confidence.
Scope is model quality only; any downstream confidence threshold is decided
separately and is deliberately not measured here.

macro-F1 here uses the same definition train.py selects checkpoints on
(macro_f1), so the reported number matches the selected one.

Uses the full softmax (predict_proba), so nothing is thrown away.

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
    """(precision, recall, f1, support, predicted) per class over the classes in
    `cm` (its size, not the global N -- so a collapsed/merged matrix works too).

    Standard sklearn convention with zero_division=0: an undefined precision
    (nothing predicted for the class) or recall counts as 0, so f1 is 0 rather
    than dropped. Callers decide which classes to display / average."""
    support, predicted, tp = cm.sum(1), cm.sum(0), cm.diagonal()
    rows: list[tuple[float, float, float, int, int]] = []
    for c in range(cm.shape[0]):
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


def resolve_merges(groups: list[list[str]]) -> tuple[list[str], list[int]]:
    """Turn label groups to merge into a remapping of the Tier-1 classes.

    Returns (merged_labels, id_map): id_map[c] is the collapsed-class index for
    each original class c (0..N-1), and merged_labels names the collapsed classes
    in original order. A class in no group keeps its own slot. A group's members
    all fold onto the slot of their lowest original index, named 'a+b'. This lets
    us report metrics as if two labels the model needn't separate were one class
    (e.g. --merge bar_grouped,bar_stacked), so their mutual confusion no longer counts."""
    key_of = list(range(N))                       # canonical slot per original id
    name_of = {c: TIER1_LABELS[c] for c in range(N)}
    for g in groups:
        for name in g:
            if name not in TIER1_LABELS:
                raise SystemExit(f"--merge: unknown label {name!r}; valid: {', '.join(TIER1_LABELS)}")
        ids = sorted(TIER1_LABELS.index(name) for name in g)
        head = ids[0]
        for i in ids:
            key_of[i] = head
        name_of[head] = "+".join(TIER1_LABELS[i] for i in ids)
    order: list[int] = []                          # merged slots, first-appearance order
    for c in range(N):
        if key_of[c] not in order:
            order.append(key_of[c])
    merged_index = {k: idx for idx, k in enumerate(order)}
    id_map = [merged_index[key_of[c]] for c in range(N)]
    return [name_of[k] for k in order], id_map


def collapse_cm(cm: torch.Tensor, id_map: list[int], k: int) -> torch.Tensor:
    """Sum the N x N confusion matrix down to k x k using id_map (merged preds
    are the original preds relabelled, so this stays confusion-consistent)."""
    m = torch.zeros(k, k, dtype=cm.dtype)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            m[id_map[i], id_map[j]] += cm[i, j]
    return m


def collapse_probs(probs: torch.Tensor, id_map: list[int], k: int) -> torch.Tensor:
    """Merged-class probability = sum of its members' probabilities, so the merged
    cross-entropy reflects the model's confidence in the correct merged class."""
    m = torch.zeros(probs.shape[0], k)
    for c in range(probs.shape[1]):
        m[:, id_map[c]] += probs[:, c]
    return m


def plot_confusion(cm: torch.Tensor, out: Path, split: str, labels: list[str] = TIER1_LABELS) -> None:
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

    n = cm.shape[0]
    support = cm.sum(1, keepdim=True).clamp(min=1)
    norm = (cm.float() / support).numpy()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"confusion (row-normalised) -- {split}")
    for i in range(n):
        for j in range(n):
            if cm[i, j] > 0:
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=6,
                        color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote confusion plot to {out}")


def print_metrics(cm: torch.Tensor, labels: list[str], nll: float, title: str | None = None) -> None:
    """Print accuracy, per-class PRF, macro-F1 and the confusion matrix for a
    confusion matrix over `labels`. Accuracy is derived from `cm` (trace/total),
    so it stays consistent with the matrix shown; `nll` is passed in because it
    needs the probabilities. Works for the full Tier-1 matrix and a merged one."""
    if title:
        print(title)
    total = int(cm.sum())
    if total == 0:
        print("split is empty -- nothing to evaluate")
        return

    support = cm.sum(1)          # true instances per class
    predicted = cm.sum(0)        # predicted instances per class
    acc = cm.diagonal().sum().item() / total
    rows = per_class_prf(cm)

    print(f"\nimages evaluated: {total}")
    print(f"accuracy:         {acc:.3f}")
    print(f"cross-entropy:    {nll:.4f}  (mean NLL)")

    print(f"\n{'label':16s} {'support':>7s} {'prec':>6s} {'rec':>6s} {'f1':>6s}")
    n_present = 0
    for c, lbl in enumerate(labels):
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
    active = [c for c in range(cm.shape[0]) if support[c] > 0 or predicted[c] > 0]
    print("\nconfusion matrix (rows = true, cols = predicted):")
    print("  index: " + ", ".join(f"{c}={labels[c]}" for c in active))
    header = "      " + "".join(f"{c:>5d}" for c in active)
    print(header)
    for r in active:
        row = "".join(f"{int(cm[r, c]):>5d}" for c in active)
        print(f"  {r:>2d} {labels[r][:12]:12s}{row}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True, help="fine-tuned checkpoint dir")
    ap.add_argument("--split-index", type=Path, default=Path("data/splits/index.jsonl"))
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--plot", type=Path, default=None, help="save a confusion-matrix heatmap PNG here")
    ap.add_argument(
        "--merge",
        action="append",
        metavar="a,b[,c]",
        help="also report metrics with these labels collapsed into one class "
        "(repeatable), e.g. --merge bar_grouped,bar_stacked. The full Tier-1 report is always shown too.",
    )
    args = ap.parse_args()

    items = load_split(args.split_index, args.split)
    print(f"loaded {len(items)} images from split '{args.split}'")
    clf = FigureClassifier(args.model, device=args.device)
    probs, targets = collect_probs(clf, items)

    if len(targets) == 0:
        print("split is empty -- nothing to evaluate")
        return

    preds = probs.argmax(1)
    # cross-entropy / mean NLL, probs clamped so a confident miss isn't infinite
    p_true = probs[torch.arange(len(targets)), targets].clamp_min(1e-12)
    nll = -p_true.log().mean().item()
    cm = confusion(preds, targets)
    print_metrics(cm, list(TIER1_LABELS), nll)

    if args.merge:
        groups = [[s.strip() for s in spec.split(",") if s.strip()] for spec in args.merge]
        merged_labels, id_map = resolve_merges(groups)
        k = len(merged_labels)
        mcm = collapse_cm(cm, id_map, k)
        mprobs = collapse_probs(probs, id_map, k)
        mtargets = torch.tensor([id_map[int(t)] for t in targets.tolist()], dtype=torch.long)
        mp_true = mprobs[torch.arange(len(mtargets)), mtargets].clamp_min(1e-12)
        mnll = -mp_true.log().mean().item()
        print_metrics(mcm, merged_labels, mnll, title=f"\n===== after merging {', '.join('+'.join(g) for g in groups)} =====")

    if args.plot is not None:
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        plot_confusion(cm, args.plot, args.split)


if __name__ == "__main__":
    main()
