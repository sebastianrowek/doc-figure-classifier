# `llm_judge_eval` submodule

Runs the LLM classifier over the hand-labeled calibration set (`data/llm_judge/`)
and scores the results. See the [`extract/` README](../README.md) for the full
workflow and CLI calls.

## Files

| File | What it does |
|---|---|
| `runner.py` | CLI. Step 1: Classifies every calibration image (optionally N times) via an LLM and writes `run.jsonl`, `run.meta.json`, and a `failures/` folder into a new run directory. |
| `report.py` | CLI. Step 2: Reads a run directory, runs the metric blocks below, and writes `report.md`, `metrics.json`, `per_file.csv`, and `plots/`. Two registries (`BLOCKS`, `PLOTS`) decide what runs. |
| `loader.py` | Defines the `run.jsonl` record schema and loads it into a tidy one-row-per-call DataFrame. The single place that knows the on-disk format. |
| `overall.py` | Level 1 — overall accuracy (micro + macro), cost, confidence intervals, pooled over all calls. |
| `per_class.py` | Level 2 — confusion matrix and precision / recall / F1 per class. |
| `per_file.py` | Level 3 — one row per image: vote distribution, agreement, and buckets (always-wrong, flippy). |
| `calibration.py` | Threshold-free quality: cross-entropy / Brier / ECE and the reliability curve. |
| `consistency.py` | Run-to-run stability across repeated calls (only meaningful with `--repeats > 1`). |
| `plots.py` | Standalone plotting functions (confusion matrix, reliability, accuracy bars, ...); each takes a block result and writes a PNG. |
