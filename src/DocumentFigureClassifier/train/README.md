# `train` module

Fine-tune docling's `DocumentFigureClassifier` on our 13 Tier-1 figure classes
(see `taxonomy.py`) and benchmark the result.

The module is a chain of steps that hand off to each other through files, not
through imports:

| File | What it does |
|---|---|
| `split.py` | Builds the train / val / test **split index** from the reviewed real crops and the synthetic set. Start here — everything downstream reads this index. |
| `dataset.py` | Shared plumbing. `load_split()` reads the index, `FigureDataset` loads the images. `train.py` and `evaluate.py` both go through it, so they cannot drift apart. |
| `train.py` | Stage-1 / Stage-2 fine-tuning loop. Freeze-then-unfreeze backbone, differential learning rates, class-weighted loss; picks the best checkpoint on **real-val macro-F1**. |
| `evaluate.py` | Benchmarks a checkpoint against one split: accuracy, per-class P/R/F1, macro-F1, confusion matrix, and cross-entropy loss. |

---

# Data splitting — `split.py`

## The problem it solves

Two crops from the *same* annual report are often near-duplicates: same house
style, same colour palette, same chart template. If one lands in train and its
twin lands in val, the model gets to memorise instead of generalise and the
validation score becomes a lie.

So the splitter never splits at the image level. **The source document is the
unit**, and every crop from one document lands in exactly one split.

## Three sources, three different rules

| Source (default path) | Goes to | Rule |
|---|---|---|
| `data/parsed/review/<label>/` | **train + val** | Real, hand-labelled crops. Split at the document level. Train is capped per class. |
| `data/parsed_test/review/<label>/` | **test** | Real, held-out. A physically separate folder so it can never leak into train/val. Taken whole — never capped, never touched. |
| `data/synth/train/<label>/` | **train** | Synthetic. Always train, never val, never test. |

Net result:

```
train = capped real-train crops  +  all synth
val   = real only
test  = the held-out folder only
```

Val and test stay **100 % real** on purpose — that is the only way the numbers
say anything about performance on actual annual reports.

Only the 13 Tier-1 label folders are read from each source. `_review/`,
`_unsure/`, `_broken/` and anything else are ignored, so a live review tree can
be pointed at directly without cleaning it up first.

## The document rule

A crop's source document is its filename up to the first `__`:

```
adtran_ann_rep_2022__p006__002__bar__1.00.png
└──── document ────┘
```

All crops sharing that stem move together. That is the anti-leakage rule, and
it is the whole reason the splitter exists rather than a two-line
`train_test_split`.

## How documents are assigned to train / val

Documents are sorted **biggest first** and handed out one at a time. For each
document the assigner scores both splits and takes the winner:

```
score(split) = deficit + bonus

deficit = how far this split is below its target image count (as a fraction)
bonus   = fraction of Tier-1 classes this document would newly bring to the
          split, weighted 0.5 for val and 0.1 for train
```

- The **deficit** term keeps the split sizes near `--ratios`.
- The **coverage bonus** lets val poach class-diverse documents even when it is
  slightly over quota. This matters: a per-class val metric is *undefined* if
  val holds zero examples of that class, so coverage beats an exact ratio.
- Exact ties are broken by a seeded coin flip, so runs are reproducible.

Biggest-first is what makes the greedy pass work: the large, indivisible chunks
get placed while there is still slack to absorb them, and the small documents
fill the remaining gaps.

**Safety net:** if val ends up with no document at all (rounding on a tiny
corpus), the smallest training document is donated to val.

## Append stability

`doc_assignments.json` records where every document went. On the next run,
documents already listed there **keep their split** — only genuinely new
documents are placed. Growing the corpus therefore never reshuffles the existing
split and never silently invalidates comparisons against earlier runs.

Targets are still computed over the *whole* corpus, so new documents flow toward
whichever split has fallen behind. Documents that have disappeared from the
source folder are dropped from the file.

`--reassign` discards this history and places every document from scratch.

## The per-class train cap

`table`, `photo` and `other` are far more abundant than `waterfall` or
`scatter`. Left alone they dominate every batch and stretch the epoch clock for
no gain. So real **train** crops are capped per class at `--train-cap`
(default `2000`). Classes already under the cap are untouched.

The cap is **document-aware**: for each over-cap class the kept crops are taken
round-robin across the training documents that contain it — one from each
document, then a second from each, and so on until the cap is reached. A naive
random cap could collapse onto a handful of reports and quietly throw away
house-style diversity; this cannot. Each class gets its own seeded RNG, so the
selection is deterministic and independent per class.

What the cap does **not** touch:

- **val and test** — never capped, they keep every real crop.
- **synth** — left exactly as generated. Synth counts are the deliberate
  class-balance knob, so capping them would fight the thing they are there to do.

> **Gotcha:** the cap is recomputed on every run. A document's *split* is stable
> across runs, but *which of its crops survive the cap* can shift when new
> documents join that class.

## Output

Written under `--out` (default `data/splits/`):

| File | Contents |
|---|---|
| `index.jsonl` | One row per image: `{path, label, split, source, doc}`. `source` is `real` or `synth`; synth rows have `doc: null`. This is the **single source of truth** — `train.py` and `evaluate.py` both read it via `dataset.load_split`. |
| `doc_assignments.json` | `{doc_stem: split}` for the real train/val documents. The append-stable record described above. |

## The report

Every run prints a per-class table (real counts by split, synth counts, and the
combined `train + synth` figure the model actually sees), how many crops the cap
dropped and from which classes, and which documents went to which split.

It then warns about **data-collection gaps** — these are not code errors:

- a document appearing in **both** train/val and test → leakage
- a class **absent from val** or **absent from test** → that per-class metric is undefined
- a class with **no real images at all** → train can only see it via synth
- **no synth found** → train is real-train documents only

---

# CLI

## Basic run

All defaults; this is the normal invocation:

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m DocumentFigureClassifier.train.split
```

Fully spelled out, equivalent to the above:

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m DocumentFigureClassifier.train.split --data data/parsed/review --test-data data/parsed_test/review --synth data/synth/train --out data/splits
```

> On PowerShell, set the variable first instead of prefixing it:
> `$env:PYTHONPATH = "src"; .venv/Scripts/python.exe -m DocumentFigureClassifier.train.split`

## Options

| Flag | Default | What it does |
|---|---|---|
| `--data PATH` | `data/parsed/review` | Real train/val source. Reads `<PATH>/<label>/` for the 13 Tier-1 labels only. **Aborts** if it finds no images. |
| `--test-data PATH` | `data/parsed_test/review` | Held-out real test source, same folder layout. Warns (does not abort) if empty — the test split is then simply empty. |
| `--synth PATH` | `data/synth/train` | Synthetic source, same folder layout. Everything found goes to train. A missing folder is not an error; the report warns instead. |
| `--out PATH` | `data/splits` | Output folder, created if missing. Receives `index.jsonl` and `doc_assignments.json`. |
| `--ratios TRAIN,VAL` | `0.85,0.15` | Target image-count fractions for the real corpus. **Exactly two values** — test is not a ratio, it is the separate folder. Treated as targets, not hard quotas: the coverage bonus may pull val slightly off. |
| `--train-cap N` | `2000` | Max real crops per class in the **train** split (document-aware, seeded). `0` disables capping entirely. val/test are never capped, synth is never touched. |
| `--seed N` | `0` | Seeds both the assignment tie-break and the cap's round-robin. Same inputs + same `doc_assignments.json` + same seed → identical split. |
| `--reassign` | off | Ignore `doc_assignments.json` and recompute every document assignment from scratch. |

## Recipes

**Added new reports — extend the split without disturbing it.** Just re-run.
Existing documents keep their split; only the new ones are placed:

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m DocumentFigureClassifier.train.split
```

**Rebalance everything from scratch:**

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m DocumentFigureClassifier.train.split --reassign
```

Use sparingly: documents move between train and val, so metrics from checkpoints
trained on the previous split are no longer comparable.

**Keep every real crop (no cap)** — slower epochs, `table`-heavy batches:

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m DocumentFigureClassifier.train.split --train-cap 0
```

**Tighter cap for a fast smoke run:**

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m DocumentFigureClassifier.train.split --train-cap 300 --out data/splits_smoke
```

**Real only, no synth** — point `--synth` at a folder that does not exist:

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m DocumentFigureClassifier.train.split --synth data/synth/none --out data/splits_realonly
```

For an already-built index, `train.py --real-only` filters synth out at load
time via `dataset.load_split(source="real")` — usually the better route, since
it reuses the same split.

**More validation data (80/20):**

```bash
PYTHONPATH=src .venv/Scripts/python.exe -m DocumentFigureClassifier.train.split --ratios 0.8,0.2 --reassign
```

Without `--reassign` the new ratio only steers *newly added* documents; the
existing assignment stays put.
