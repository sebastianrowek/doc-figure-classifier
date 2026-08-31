"""
Build the train / val / test split index for the figure classifier.

Two sources, different rules:
  * data/synth/train/<label>/  -- synthetic, ALWAYS train (absent until a synth
    run has been done).
  * data/parsed/review/<label>/ -- real, hand-labeled. Split three ways at the
    DOCUMENT level: every crop from one source PDF lands in one split, because
    two crops from the same PDF can be near-duplicates and would leak an answer
    from train into val/test. The document is the filename substring before the
    first "__" (adtran_ann_rep_2022__p006__002__bar__1.00.png ->
    adtran_ann_rep_2022).

Output (default under data/splits/):
  * index.jsonl          -- one row per image: path, label, split, source, doc.
  * doc_assignments.json -- {doc_stem: split} for real docs; the append-stable
    record so growing the corpus never reshuffles existing documents.

    PYTHONPATH=src .venv/Scripts/python.exe -m DocumentFigureClassifier.train.split \
        --data data/parsed/review --synth data/synth/train --out data/splits
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from DocumentFigureClassifier.taxonomy import TIER1_LABELS
from DocumentFigureClassifier.train.dataset import IMG_EXTS

SPLITS = ("train", "val", "test")


def doc_of(path: Path) -> str:
    """Source-document stem = filename up to the first '__'."""
    return path.name.split("__", 1)[0]


# --------------------------------------------------------------------------
# Scanning
# --------------------------------------------------------------------------


def scan_real(review_root: Path) -> dict[str, dict[str, list[Path]]]:
    """{doc_stem: {label: [paths]}} over data/parsed/review/<label>/.

    Only the Tier-1 label folders are read; _review/_unsure/_broken and anything
    else are ignored, so the review tree can be pointed at directly."""
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
# Document-level assignment
# --------------------------------------------------------------------------


def assign_docs(
    docs: dict[str, dict[str, list[Path]]],
    ratios: tuple[float, float, float],
    existing: dict[str, str],
    seed: int,
    reassign: bool,
) -> dict[str, str]:
    """Assign each real document to one split.

    Append-stable: documents already in `existing` keep their split (unless
    --reassign), so only new documents are placed. Placement is a coverage-aware
    greedy: large documents first, each sent to the split that is most under its
    target image count, with a bonus for splits still missing a class the
    document carries (weighted toward val/test, where class coverage decides
    whether a per-class metric is even defined)."""
    rng = random.Random(seed)
    doc_total = {d: sum(len(v) for v in h.values()) for d, h in docs.items()}
    total_images = sum(doc_total.values())
    target = {s: r * total_images for s, r in zip(SPLITS, ratios)}

    assign: dict[str, str] = (
        {} if reassign else {d: s for d, s in existing.items() if d in docs}
    )
    cur_images = {s: 0 for s in SPLITS}
    present: dict[str, set[str]] = {s: set() for s in SPLITS}
    for d, s in assign.items():
        cur_images[s] += doc_total[d]
        present[s].update(k for k, v in docs[d].items() if v)

    remaining = sorted(
        (d for d in docs if d not in assign),
        key=lambda d: (-doc_total[d], d),
    )
    for d in remaining:
        classes = {k for k, v in docs[d].items() if v}
        order = list(SPLITS)
        rng.shuffle(order)  # seeded tie-break
        best, best_score = order[0], None
        for s in order:
            deficit = (target[s] - cur_images[s]) / max(target[s], 1.0)
            gain = len(classes - present[s]) / len(TIER1_LABELS)
            bonus = (0.5 if s in ("val", "test") else 0.1) * gain
            score = deficit + bonus
            if best_score is None or score > best_score:
                best, best_score = s, score
        assign[d] = best
        cur_images[best] += doc_total[d]
        present[best].update(classes)

    _ensure_nonempty(assign, doc_total, cur_images)
    return assign


def _ensure_nonempty(
    assign: dict[str, str], doc_total: dict[str, int], cur_images: dict[str, int]
) -> None:
    """If val or test ended up with no document (rounding on a tiny corpus),
    donate the smallest document from the split that has the most to spare."""
    for s in ("val", "test"):
        if any(v == s for v in assign.values()):
            continue
        by_split = {sp: [d for d, x in assign.items() if x == sp] for sp in SPLITS}
        donors = [sp for sp in SPLITS if len(by_split[sp]) > 1]
        if not donors:
            continue
        donor = max(donors, key=lambda sp: cur_images[sp])
        d = min(by_split[donor], key=lambda d: doc_total[d])
        assign[d] = s
        cur_images[donor] -= doc_total[d]
        cur_images[s] += doc_total[d]


# --------------------------------------------------------------------------
# Output + report
# --------------------------------------------------------------------------


def write_index(
    index_path: Path,
    docs: dict[str, dict[str, list[Path]]],
    doc_split: dict[str, str],
    synth: list[tuple[Path, str]],
) -> None:
    with index_path.open("w", encoding="utf-8") as fh:
        for doc, hist in docs.items():
            split = doc_split[doc]
            for label, paths in hist.items():
                for p in paths:
                    fh.write(
                        json.dumps(
                            {
                                "path": p.as_posix(),
                                "label": label,
                                "split": split,
                                "source": "real",
                                "doc": doc,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        for p, label in synth:
            fh.write(
                json.dumps(
                    {
                        "path": p.as_posix(),
                        "label": label,
                        "split": "train",
                        "source": "synth",
                        "doc": None,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def report(
    docs: dict[str, dict[str, list[Path]]],
    doc_split: dict[str, str],
    synth: list[tuple[Path, str]],
) -> None:
    # counts[label][split] over real images, plus a synthetic train column.
    real = {lbl: {s: 0 for s in SPLITS} for lbl in TIER1_LABELS}
    docs_per_label: dict[str, set[str]] = defaultdict(set)
    for doc, hist in docs.items():
        s = doc_split[doc]
        for label, paths in hist.items():
            real[label][s] += len(paths)
            docs_per_label[label].add(doc)
    synth_count: dict[str, int] = defaultdict(int)
    for _, label in synth:
        synth_count[label] += 1

    print("\nper-class counts (real by split | synth->train):")
    print(f"  {'label':16s} {'train':>6s} {'val':>6s} {'test':>6s} {'synth':>7s}")
    for lbl in TIER1_LABELS:
        r = real[lbl]
        print(f"  {lbl:16s} {r['train']:6d} {r['val']:6d} {r['test']:6d} {synth_count[lbl]:7d}")
    tot = {s: sum(real[l][s] for l in TIER1_LABELS) for s in SPLITS}
    print(f"  {'TOTAL':16s} {tot['train']:6d} {tot['val']:6d} {tot['test']:6d} {len(synth):7d}")

    print("\ndocument assignment:")
    for s in SPLITS:
        ds = sorted(d for d, x in doc_split.items() if x == s)
        print(f"  {s:5s} ({len(ds)} docs): {', '.join(ds) or '-'}")

    # Warnings -- data-collection gaps, not code errors.
    warns: list[str] = []
    for lbl in TIER1_LABELS:
        r = real[lbl]
        total = r["train"] + r["val"] + r["test"]
        if total == 0:
            warns.append(f"{lbl}: no real images (train can only see it via synth)")
            continue
        if r["val"] == 0:
            warns.append(f"{lbl}: absent from val -- per-class val metric undefined")
        if r["test"] == 0:
            warns.append(f"{lbl}: absent from test -- per-class test metric undefined")
        if len(docs_per_label[lbl]) == 1:
            warns.append(f"{lbl}: real images come from a single document -> only one split can hold it")
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
    ap.add_argument("--data", type=Path, default=Path("data/parsed/review"))
    ap.add_argument("--synth", type=Path, default=Path("data/synth/train"))
    ap.add_argument("--out", type=Path, default=Path("data/splits"))
    ap.add_argument("--ratios", default="0.7,0.15,0.15", help="train,val,test fractions")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reassign", action="store_true", help="recompute all doc assignments from scratch")
    args = ap.parse_args()

    ratios = tuple(float(x) for x in args.ratios.split(","))
    if len(ratios) != 3:
        raise SystemExit("--ratios needs exactly three values: train,val,test")

    docs = scan_real(args.data)
    if not docs:
        raise SystemExit(f"no real images found under {args.data}")
    synth = scan_synth(args.synth)

    assignments_path = args.out / "doc_assignments.json"
    existing: dict[str, str] = {}
    if assignments_path.exists() and not args.reassign:
        existing = json.loads(assignments_path.read_text(encoding="utf-8"))

    doc_split = assign_docs(docs, ratios, existing, args.seed, args.reassign)  # type: ignore[arg-type]

    args.out.mkdir(parents=True, exist_ok=True)
    assignments_path.write_text(
        json.dumps(doc_split, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_index(args.out / "index.jsonl", docs, doc_split, synth)
    report(docs, doc_split, synth)
    print(f"\nwrote {args.out / 'index.jsonl'} and {assignments_path}")


if __name__ == "__main__":
    main()
