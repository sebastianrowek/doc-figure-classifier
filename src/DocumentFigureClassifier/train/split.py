"""
Build the train / val / test split index for the figure classifier.

Three sources, different rules:
  * data/synth/train/<label>/      -- synthetic, ALWAYS train. Sized per class to
    lift the rare classes (waterfall, scatter, ...); absent until a synth run.
  * data/parsed/review/<label>/    -- real, hand-labeled. Split TWO ways
    (train/val) at the DOCUMENT level: every crop from one source PDF lands in
    one split, because two crops from the same PDF can be near-duplicates and
    would leak an answer from train into val. The document is the filename
    substring before the first "__" (adtran_ann_rep_2022__p006__002__bar__1.00.png
    -> adtran_ann_rep_2022).
  * data/parsed_test/review/<label>/ -- real, held-out TEST. Used only as test,
    never mixed into train/val. Kept whole (no cap).

So: train = capped real-train + all synth; val = real only; test = the separate
held-out folder only.

The abundant real classes (table, photo, other, ...) would otherwise dominate
every batch and the epoch clock, so real *train* crops are capped per class
(--train-cap, default 2000, document-aware and seeded). Classes already under the
cap are untouched; val and test are never capped. Synth counts are left alone --
they are the deliberate class-balance knob.

Output (default under data/splits/):
  * index.jsonl          -- one row per image: path, label, split, source, doc.
  * doc_assignments.json -- {doc_stem: split} for train/val real docs; the
    append-stable record so growing the corpus never reshuffles existing docs.

    PYTHONPATH=src .venv/Scripts/python.exe -m DocumentFigureClassifier.train.split \
        --data data/parsed/review --test-data data/parsed_test/review \
        --synth data/synth/train --out data/splits
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from DocumentFigureClassifier.taxonomy import TIER1_LABELS
from DocumentFigureClassifier.train.dataset import IMG_EXTS

SPLITS = ("train", "val", "test")       # reporting columns
ASSIGN_SPLITS = ("train", "val")        # what the document assigner distributes


def doc_of(path: Path) -> str:
    """Source-document stem = filename up to the first '__'."""
    return path.name.split("__", 1)[0]


# --------------------------------------------------------------------------
# Scanning
# --------------------------------------------------------------------------


def scan_real(review_root: Path) -> dict[str, dict[str, list[Path]]]:
    """{doc_stem: {label: [paths]}} over <review_root>/<label>/.

    Only the Tier-1 label folders are read; _review/_unsure/_broken and anything
    else are ignored, so a review tree can be pointed at directly."""
    docs: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for label in TIER1_LABELS:
        d = review_root / label
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in IMG_EXTS:
                docs[doc_of(p)][label].append(p)
    return docs


def scan_synth(synth_root: Path) -> list[tuple[Path, str]]:
    """[(path, label)] over data/synth/train/<label>/. Empty if absent."""
    out: list[tuple[Path, str]] = []
    for label in TIER1_LABELS:
        d = synth_root / label
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in IMG_EXTS:
                out.append((p, label))
    return out


# --------------------------------------------------------------------------
# Document-level assignment (train / val only)
# --------------------------------------------------------------------------


def assign_docs(
    docs: dict[str, dict[str, list[Path]]],
    ratios: tuple[float, float],
    existing: dict[str, str],
    seed: int,
    reassign: bool,
) -> dict[str, str]:
    """Assign each real document to train or val.

    Append-stable: documents already in `existing` keep their split (unless
    --reassign), so only new documents are placed. Placement is a coverage-aware
    greedy: large documents first, each sent to the split that is most under its
    target image count, with a bonus for val when it still misses a class the
    document carries (class coverage decides whether a per-class val metric is
    even defined)."""
    rng = random.Random(seed)
    doc_total = {d: sum(len(v) for v in h.values()) for d, h in docs.items()}
    total_images = sum(doc_total.values())
    target = {s: r * total_images for s, r in zip(ASSIGN_SPLITS, ratios)}

    assign: dict[str, str] = (
        {}
        if reassign
        else {d: s for d, s in existing.items() if d in docs and s in ASSIGN_SPLITS}
    )
    cur_images = {s: 0 for s in ASSIGN_SPLITS}
    present: dict[str, set[str]] = {s: set() for s in ASSIGN_SPLITS}
    for d, s in assign.items():
        cur_images[s] += doc_total[d]
        present[s].update(k for k, v in docs[d].items() if v)

    remaining = sorted(
        (d for d in docs if d not in assign),
        key=lambda d: (-doc_total[d], d),
    )
    for d in remaining:
        classes = {k for k, v in docs[d].items() if v}
        order = list(ASSIGN_SPLITS)
        rng.shuffle(order)  # seeded tie-break
        best, best_score = order[0], None
        for s in order:
            deficit = (target[s] - cur_images[s]) / max(target[s], 1.0)
            gain = len(classes - present[s]) / len(TIER1_LABELS)
            bonus = (0.5 if s == "val" else 0.1) * gain
            score = deficit + bonus
            if best_score is None or score > best_score:
                best, best_score = s, score
        assign[d] = best
        cur_images[best] += doc_total[d]
        present[best].update(classes)

    _ensure_val_nonempty(assign, doc_total, cur_images)
    return assign


def _ensure_val_nonempty(
    assign: dict[str, str], doc_total: dict[str, int], cur_images: dict[str, int]
) -> None:
    """If val ended up with no document (rounding on a tiny corpus), donate the
    smallest training document."""
    if any(v == "val" for v in assign.values()):
        return
    train_docs = [d for d, x in assign.items() if x == "train"]
    if len(train_docs) <= 1:
        return
    d = min(train_docs, key=lambda d: doc_total[d])
    assign[d] = "val"
    cur_images["train"] -= doc_total[d]
    cur_images["val"] += doc_total[d]


# --------------------------------------------------------------------------
# Per-class train downsampling
# --------------------------------------------------------------------------


def cap_train_real(
    docs: dict[str, dict[str, list[Path]]],
    doc_split: dict[str, str],
    cap: int,
    seed: int,
) -> set[Path]:
    """Return real *train* image paths to DROP so no class exceeds ``cap`` train
    crops.

    Applies per class to the abundant real classes (table, photo, other, ...);
    classes already under the cap are untouched. val keeps every real crop and
    test is a separate held-out folder, so neither is capped here. Synth is not
    touched -- it is the deliberate class-balance knob. ``cap <= 0`` disables.

    Document-aware: for each capped class the kept crops are taken round-robin
    across the training documents that contain it, so a cap never collapses onto
    a handful of reports and loses house-style diversity. Deterministic given
    seed (and independent per class)."""
    if cap <= 0:
        return set()
    drop: set[Path] = set()
    for label in TIER1_LABELS:
        by_doc = {
            doc: list(hist[label])
            for doc, hist in docs.items()
            if doc_split.get(doc) == "train" and hist.get(label)
        }
        if sum(len(v) for v in by_doc.values()) <= cap:
            continue
        rng = random.Random(f"{seed}:{label}")  # distinct, deterministic per class
        pools = [by_doc[d] for d in sorted(by_doc)]
        for pool in pools:
            rng.shuffle(pool)

        keep: set[Path] = set()
        depth = 0
        while len(keep) < cap:
            progressed = False
            for pool in pools:
                if depth < len(pool):
                    keep.add(pool[depth])
                    progressed = True
                    if len(keep) >= cap:
                        break
            if not progressed:
                break
            depth += 1

        drop.update(p for pool in pools for p in pool if p not in keep)
    return drop


# --------------------------------------------------------------------------
# Output + report
# --------------------------------------------------------------------------


def _row(path: Path, label: str, split: str, source: str, doc: str | None) -> str:
    return (
        json.dumps(
            {"path": path.as_posix(), "label": label, "split": split, "source": source, "doc": doc},
            ensure_ascii=False,
        )
        + "\n"
    )


def write_index(
    index_path: Path,
    docs: dict[str, dict[str, list[Path]]],
    doc_split: dict[str, str],
    test_docs: dict[str, dict[str, list[Path]]],
    synth: list[tuple[Path, str]],
    drop: set[Path],
) -> None:
    with index_path.open("w", encoding="utf-8") as fh:
        # train / val -- real, capped
        for doc, hist in docs.items():
            split = doc_split[doc]
            for label, paths in hist.items():
                for p in paths:
                    if p not in drop:
                        fh.write(_row(p, label, split, "real", doc))
        # test -- real, held-out folder, whole
        for doc, hist in test_docs.items():
            for label, paths in hist.items():
                for p in paths:
                    fh.write(_row(p, label, "test", "real", doc))
        # synth -- always train
        for p, label in synth:
            fh.write(_row(p, label, "train", "synth", None))


def report(
    docs: dict[str, dict[str, list[Path]]],
    doc_split: dict[str, str],
    test_docs: dict[str, dict[str, list[Path]]],
    synth: list[tuple[Path, str]],
    drop: set[Path],
) -> None:
    # real counts by split kept in the index (dropped train crops excluded).
    real = {lbl: {s: 0 for s in SPLITS} for lbl in TIER1_LABELS}
    docs_per_label: dict[str, set[str]] = defaultdict(set)
    dropped_by_label: dict[str, int] = defaultdict(int)
    for doc, hist in docs.items():
        s = doc_split[doc]
        for label, paths in hist.items():
            real[label][s] += sum(1 for p in paths if p not in drop)
            dropped_by_label[label] += sum(1 for p in paths if p in drop)
            docs_per_label[label].add(doc)
    for doc, hist in test_docs.items():
        for label, paths in hist.items():
            real[label]["test"] += len(paths)
    synth_count: dict[str, int] = defaultdict(int)
    for _, label in synth:
        synth_count[label] += 1

    print("\nper-class counts (real by split | synth->train):")
    print(f"  {'label':16s} {'train':>6s} {'val':>6s} {'test':>6s} {'synth':>7s} {'tr+syn':>7s}")
    for lbl in TIER1_LABELS:
        r = real[lbl]
        print(
            f"  {lbl:16s} {r['train']:6d} {r['val']:6d} {r['test']:6d} "
            f"{synth_count[lbl]:7d} {r['train'] + synth_count[lbl]:7d}"
        )
    tot = {s: sum(real[l][s] for l in TIER1_LABELS) for s in SPLITS}
    print(
        f"  {'TOTAL':16s} {tot['train']:6d} {tot['val']:6d} {tot['test']:6d} "
        f"{len(synth):7d} {tot['train'] + len(synth):7d}"
    )
    if drop:
        capped = ", ".join(f"{l}(-{dropped_by_label[l]})" for l in TIER1_LABELS if dropped_by_label[l])
        print(f"  (train capped: {len(drop)} real crops dropped -> {capped})")

    print("\ndocument assignment (train/val real):")
    for s in ASSIGN_SPLITS:
        ds = sorted(d for d, x in doc_split.items() if x == s)
        print(f"  {s:5s} ({len(ds)} docs): {', '.join(ds) or '-'}")
    print(f"  test  ({len(test_docs)} docs, held-out folder): {', '.join(sorted(test_docs)) or '-'}")

    # Warnings -- data-collection gaps, not code errors.
    warns: list[str] = []
    overlap = set(docs) & set(test_docs)
    if overlap:
        warns.append(
            f"{len(overlap)} document(s) appear in BOTH train/val and test -> "
            f"leakage: {', '.join(sorted(overlap))}"
        )
    for lbl in TIER1_LABELS:
        r = real[lbl]
        if r["train"] + r["val"] + r["test"] == 0:
            warns.append(f"{lbl}: no real images (train can only see it via synth)")
            continue
        if r["val"] == 0:
            warns.append(f"{lbl}: absent from val -- per-class val metric undefined")
        if r["test"] == 0:
            warns.append(f"{lbl}: absent from test -- per-class test metric undefined")
    if not synth:
        warns.append("no synth images found -- train = real-train docs only")
    if warns:
        print("\nwarnings:")
        for w in warns:
            print(f"  ! {w}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--data", type=Path, default=Path("data/parsed/review"), help="real train/val source")
    ap.add_argument("--test-data", type=Path, default=Path("data/parsed_test/review"), help="held-out real test source")
    ap.add_argument("--synth", type=Path, default=Path("data/synth/train"))
    ap.add_argument("--out", type=Path, default=Path("data/splits"))
    ap.add_argument("--ratios", default="0.85,0.15", help="train,val fractions (test is the separate folder)")
    ap.add_argument(
        "--train-cap",
        type=int,
        default=2000,
        help="max real crops per class in the train split (document-aware, seeded); "
        "val/test never capped, synth untouched. 0 disables.",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reassign", action="store_true", help="recompute all doc assignments from scratch")
    args = ap.parse_args()

    ratios = tuple(float(x) for x in args.ratios.split(","))
    if len(ratios) != 2:
        raise SystemExit("--ratios needs exactly two values: train,val")

    docs = scan_real(args.data)
    if not docs:
        raise SystemExit(f"no real images found under {args.data}")
    test_docs = scan_real(args.test_data)
    if not test_docs:
        print(f"warning: no test images found under {args.test_data} -- test split will be empty")
    synth = scan_synth(args.synth)

    assignments_path = args.out / "doc_assignments.json"
    existing: dict[str, str] = {}
    if assignments_path.exists() and not args.reassign:
        existing = json.loads(assignments_path.read_text(encoding="utf-8"))

    doc_split = assign_docs(docs, ratios, existing, args.seed, args.reassign)  # type: ignore[arg-type]
    drop = cap_train_real(docs, doc_split, args.train_cap, args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    assignments_path.write_text(
        json.dumps(doc_split, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_index(args.out / "index.jsonl", docs, doc_split, test_docs, synth, drop)
    report(docs, doc_split, test_docs, synth, drop)
    print(f"\nwrote {args.out / 'index.jsonl'} and {assignments_path}")


if __name__ == "__main__":
    main()
