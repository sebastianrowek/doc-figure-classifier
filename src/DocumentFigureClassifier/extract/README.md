# `extract` module

Prepare the training data from real annual reports and corporate documents for the document figure classifier model. 
This is done with a 2-step pipeline:
1. Use existing model 
- Extract and crop figures from corporate PDFs 
- Classify them with the existing model
- Map its 26 classes to the defined classes the new classifier should use (Tier1-Labels, see `taxonomy.py`)
2. Use an LLM-Classifier
- To enable the training data preparation at scale, a multimodel LLM can used to provide an additional classification
- In 3 different cases the LLM will we called for a classification:
  - Classes the model was not trained to recognize (e.g. Combo-Bar-Line chart)
  - Classes that are split into more fine-grained classes (bar, stacked bar, grouped bar)
  - When the probability for the main class is beneath a threshold (e.g. 0.7)

## Files

| File/Folder | What it does |
|---|---|
| `extract_and_classify.py` | Main pipeline. Detects figures in PDFs (Docling), filters junk crops, classifies each with the local `DocumentFigureClassifier-v2.5` model, and sorts them into `review/<label>/` folders. |
| `llm_classify.py` | Second-opinion classifier that sends an image to an LLM (Gemini via OpenRouter) and returns `{label, confidence}`. Has a sync and an async function and a `LlmClassifyConfig` for model/decoding settings. |
| `llm_judge_eval` | Submodule for calibrating the LLM-Judge for figure classification. Calibration compares the LLMs figure class predictions against a set of human labeled classifications across all Tier1-Label classes. For details, see the submodules README. |
| `labeling_tool` | Submodule that provides a custom, LLM-built labeling tool because the predictions of the extract_and_classify pipeline need revision. For details, see the submodules README. |


## Report sources

Input PDFs are German annual reports (`Geschäftsbericht`) from listed companies,
named `{company}_ann_rep_{year}.pdf`. Training reports live in `data/reports/`;
held-out **test** reports (companies not used for training) live in
`data/reports/test/`. Download with a browser `User-Agent` on curl — several IR
sites 403 plain requests.

**Candidates for more reports** (not yet collected, publish a German
`Geschäftsbericht`): Gerresheimer, Jungheinrich, Ströer, Fuchs SE, Fielmann,
Freenet, K+S, HOCHTIEF, BayWa, KWS Saat, SMA Solar, PVA TePla, Talanx (full
report is behind a gated URL — needs a real browser).

Skip: Airbus, QIAGEN, Zalando — English-only, no German `Geschäftsbericht`.

## Setup

Both scripts read the API key from a `.env` file in the repo root:

```
OPENROUTER_API_KEY=sk-or-...
```

## Main workflow: extract & classify

Point the script at a PDF (or a folder of PDFs) and an output folder:

```bash
python -m DocumentFigureClassifier.extract.extract_and_classify ./reports ./out --limit 3
```

- `--limit 3` — only the first 3 PDFs. Always smoke-test before a full run.
- `--threshold 0.75` — crops below this confidence go to `_review/`.
- `--include-tables` — also crop detected tables into `table/`. Cheap:
  docling's layout model locates the tables and each region is rendered
  straight from the PDF with PyMuPDF (no TableFormer, no whole-page rasters
  held in memory). Table crops skip the figure classifier and the LLM layer
  and go straight to `table/`, because docling's layout label is more reliable
  than the CNN at the table class.

Output layout under `./out`:

```
out/
  bar/  line/  pie_donut/  ...   # one folder per predicted label
  table/                         # detected tables (only with --include-tables)
  _review/                       # low-confidence crops to check by hand
  manifest.jsonl                 # every crop + its metadata
```

Correcting a label = move the file into the right folder.

---

## `llm_judge_eval` submodule

The **LLM judge** checks how well `llm_classify.py` labels charts. It runs the
classifier over a fixed, hand-labeled image set and scores the results. The
judge code lives in the submodule `llm_judge_eval/` (inside this `extract/`
module); it reuses `llm_classify.py` from here.

### The calibration set: `data/llm_judge/`

```
data/llm_judge/
  bar/  line/  scatter/  ... other/   # 14 class folders = ground truth (folder = correct label)
  _removed/                           # images pulled out of the set (kept for the record)
  _runs/                              # classification runs (created by the runner)
  _evals/                             # evaluation reports (created by the report)
  manifest.csv / manifest.md          # every image, its source, and status (used / removed)
```

The folder an image sits in **is** its ground-truth label.

### Step 1 — run the classification (this is the paid step)

```bash
python -m DocumentFigureClassifier.extract.llm_judge_eval.runner --max-concurrency 5
```

Classifies every image in the class folders and writes a new run directory:

```
data/llm_judge/_runs/run_<timestamp>/
  run.jsonl        # one line per model call (prediction, confidence, cost, error)
  run.meta.json    # model, temperature, reasoning effort, run settings, totals
  failures/        # raw model reply for any call that failed after retries
```

Useful flags: `--repeats N` (call each image N times), `--limit N` (cheap
dry-run on N images), `--retries N`, `--out-dir DIR`.

### Step 2 — build the evaluation report (free, re-runnable)

Pass the run directory; it finds `run.jsonl` itself:

```bash
python -m DocumentFigureClassifier.extract.llm_judge_eval.report data/llm_judge/_runs/run_<timestamp>
```

Writes an eval folder with the **same timestamp** as the run:

```
data/llm_judge/_evals/eval_<timestamp>/
  report.md        # overall / per-class / per-file accuracy, confusion matrix
  metrics.json     # same numbers, machine-readable
  per_file.csv     # per-image results
  plots/           # confusion matrix, reliability, accuracy bars, ...
```

Step 1 costs money and runs once; step 2 is free and can be re-run on the same
run directory as often as you like.
