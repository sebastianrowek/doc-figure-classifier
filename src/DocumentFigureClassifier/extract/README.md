# `extract` module

Prepare the training data from real annual reports and corporate documents 
for the document figure classifier model.

This is done with a multi-step pipeline:
1. Use doclings `DocumentConverter` class with its built-in layout detection model to 
extract and crop figures and tables from corporate PDFs

2. Classify them with doclings `DocumentFigureClassifier` model that we want to fine-tune
- This is a baseline since the model is already capable of detecting certain figure 
classes that we can use in the next step

3. Map its 26 classes to the **new** 13 figure classes the fine-tuned classifier should
output (Tier1-Labels, see `taxonomy.py`)
- Part of the existing classes can be mapped 1-to-1 and may only get a rename (`pie_chart` -> `pie_donut`)
- Some classes are consolidated because the detail is not needed (`logo`, `icon`, `stamp`, `signature` -> `other`; logos/pictograms were merged into `other`)
- Other classes will be expanded to XX for more detailed classification (`bar_chart` -> `bar`, `bar_stacked`, `bar_grouped`). For these classes, only the base label is provided (`bar` in the example) and further
processing is needed to assign the fine-grained class labels

4. Review assigned labels
In any of the 3 cases above, the assigned class needs to be checked by a human to make 
sure the training data is valid. 
- This can be done with the labeling tool in the `labeling_tool` submodule. See the README 
of the submodule for further instructions.
- Re-labeling means moving images between folders that represent the classes, so manual 
re-labeling is in principle also possible

5. OPTIONAL: LLM classification
To enable the training data preparation at scale, a multimodel LLM can be used in an 
additional classification step to provide a further label.

If enabled in the extraction pipeline, the LLM will we called for a classification in 
3 cases:
- For classes the docling model was not trained to recognize (e.g. Combo-Bar-Line chart)
- For classes that are split into more fine-grained classes (like bar charts)
- When the probability of the docling model for the main class is beneath 
a threshold (e.g. 0.75)

In any case, before using the LLM-as-a-judge, the calibration needs to be run with the
`llm_judge_eval` submodule. See its README for more details.



## Files

| File/Folder | What it does |
|---|---|
| `extract_and_classify.py` | Main pipeline. Detects figures in PDFs (Docling), filters junk crops, classifies each with the local `DocumentFigureClassifier-v2.5` model, and sorts them into `review/<label>/` folders. |
| `llm_classify.py` | Second-opinion classifier that sends an image to an LLM (Gemini via OpenRouter) and returns `{label, confidence}`. Has a sync and an async function and a `LlmClassifyConfig` for model/decoding settings. |
| `llm_judge_eval` | Submodule for calibrating the LLM-Judge for figure classification. Calibration compares the LLMs figure class predictions against a set of human labeled classifications across all Tier1-Label classes. For details, see the submodules README. |
| `labeling_tool` | Submodule that provides a custom, LLM-built labeling tool because to review the predictions of the extract_and_classify pipeline. For details, see the submodules README. |


## Report sources

Input PDFs are German annual reports (`Geschäftsbericht`) from listed companies,
named `{company}_ann_rep_{year}.pdf`. Training reports live in `data/reports/`;
held-out **test** reports (one year per company) live in `data/reports/test/`;
extra held-out companies with **4 mixed years each** (2017–2025) live in
`data/reports/candidates/`. Download with a browser `User-Agent` on curl —
several IR sites 403 plain requests; for reports pulled from a company's site
after it was reorganised, the Wayback Machine (`web.archive.org/web/<ts>id_/…`)
still serves the original PDF (note: its replay caps some files at 5 MiB — pick a
snapshot whose CDX length is under that).

**Collected into `candidates/`** (4 years each unless noted): Gerresheimer,
Jungheinrich, Fuchs, Fielmann, Freenet, K+S (`ks_*`), HOCHTIEF, BayWa, KWS,
SMA Solar (`sma_*`), PVA TePla (`pvatepla_*`, via Wayback), Ströer (only 2 —
2022 & 2024; it publishes an online report, not standalone `Geschäftsbericht`
PDFs, so other years offer only thin financial-statement PDFs).

**Still untapped candidates** (German `Geschäftsbericht`, not yet collected):
Talanx (full report behind a gated `_pw` URL — needs a real browser), Evotec,
Aixtron, Verbio, Stabilus, adesso, SAF-Holland, Suss MicroTec, Wacker Neuson.

Skip: Airbus, QIAGEN, Zalando, and PVA TePla's *current* site — English-only.

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
