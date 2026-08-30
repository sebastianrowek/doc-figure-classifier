"""Orchestrator: read a raw-results JSONL, run the enabled analysis blocks, and
write ``metrics.json`` + ``report.md`` + ``plots/*.png``.

LAYERING -- everything is opt-in through two registries below:

    BLOCKS : metric modules to run, in order. Each is imported lazily; if you
             delete a module file, also delete its name here (or leave it -- a
             missing module is skipped with a warning, not an error).
    PLOTS  : figures to render. Each names a function in ``plots`` and the block
             result (or "df") it consumes. Delete an entry to drop one figure.

So "delete the parts I don't need" == delete a module file and/or a registry
line. No other file needs editing.

    python -m DocumentFigureClassifier.extract.llm_judge_eval.report _runs/run_20260828_065324
"""
from __future__ import annotations

import argparse
import datetime as _dt
import importlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from DocumentFigureClassifier.taxonomy import TIER1_LABELS
from DocumentFigureClassifier.constants import DEFAULT_LLM_JUDGE_DATA_DIR
from . import loader

PKG = "DocumentFigureClassifier.extract.llm_judge_eval"

# ---- registries (edit these to add/remove analyses) -------------------------
BLOCKS = ["overall", "consistency", "calibration", "per_class", "per_file"]

PLOTS = [
    # (output filename,           plots.<fn>,               source block or "df")
    ("confusion_matrix.png",      "confusion_heatmap",      "per_class"),
    ("reliability.png",           "reliability_diagram",    "calibration"),
    ("per_class_accuracy.png",    "per_class_accuracy_bars", "per_class"),
    ("per_class_prf.png",         "per_class_prf_bars",     "per_class"),
    ("confidence_by_outcome.png", "confidence_hist",        "df"),
    ("agreement.png",             "agreement_hist",         "per_file"),
]
# -----------------------------------------------------------------------------


def _jsonable(obj):
    """Recursively convert numpy / pandas objects to JSON-serializable values."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return obj.where(pd.notna(obj), None).to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return _jsonable(obj.to_dict())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def run_blocks(df: pd.DataFrame, names=BLOCKS, **cfg) -> dict:
    """Import + run each enabled block; missing modules are skipped, not fatal."""
    results = {}
    for name in names:
        try:
            mod = importlib.import_module(f"{PKG}.{name}")
        except ModuleNotFoundError:
            print(f"  [skip] block '{name}' -- module not found")
            continue
        results[name] = {"module": mod, "result": mod.compute(df, labels=TIER1_LABELS, **cfg)}
    return results


def render_plots(df, results, out_dir: Path, specs=PLOTS) -> list[Path]:
    try:
        plots = importlib.import_module(f"{PKG}.plots")
    except ModuleNotFoundError:
        print("  [skip] plots -- module not found")
        return []
    made = []
    for fname, fn_name, source in specs:
        fn = getattr(plots, fn_name, None)
        if fn is None:
            print(f"  [skip] plot '{fn_name}' -- not in plots.py")
            continue
        if source == "df":
            arg = df
        elif source in results:
            arg = results[source]["result"]
        else:
            print(f"  [skip] plot '{fn_name}' -- needs block '{source}' (disabled)")
            continue
        try:
            made.append(fn(arg, out_dir / "plots" / fname))
        except Exception as e:                       # a broken plot must not kill the run
            print(f"  [warn] plot '{fn_name}' failed: {e}")
    return made


def write_markdown(df, results, warns, plots_made, out_dir: Path, src: str):
    L = ["# LLM-Judge Evaluation Report", "",
         f"Source: `{src}`  ·  generated {_dt.datetime.now():%Y-%m-%d %H:%M}", ""]
    if warns:
        L += ["> **Warnings:** " + "; ".join(warns), ""]

    # metric tables, in registry order
    for name in BLOCKS:
        if name not in results:
            continue
        mod = results[name]["module"]; res = results[name]["result"]
        L += [f"## {getattr(mod, 'LABEL', name)}", ""]
        try:
            L += [mod.table(res).to_markdown(index=False), ""]
        except Exception as e:
            L += [f"_(table unavailable: {e})_", ""]
        if name == "per_file":
            for key, title in [("always_wrong", "Always wrong (bad crop or mislabel in our set?)"),
                               ("flippy", "Flippy (model unsure — >1 distinct label)")]:
                sub = res.get(key)
                if sub is not None and len(sub):
                    L += [f"### {title} — {len(sub)}", "",
                          mod.table({key: sub}, key).to_markdown(index=False), ""]

    if plots_made:
        L += ["## Figures", ""]
        for p in plots_made:
            rel = p.relative_to(out_dir).as_posix()
            L += [f"### {p.stem.replace('_', ' ').title()}", "",
                  f"![{p.stem}]({rel})", ""]
    (out_dir / "report.md").write_text("\n".join(L), encoding="utf-8")


def _resolve_run_jsonl(run_dir: Path) -> Path:
    """Return run.jsonl inside a run directory (as produced by the runner)."""
    if not run_dir.is_dir():
        raise SystemExit(f"{run_dir} is not a directory -- pass the run_<timestamp> folder")
    raw = run_dir / "run.jsonl"
    if not raw.exists():
        raise SystemExit(f"no run.jsonl found in {run_dir}")
    return raw


def main():
    ap = argparse.ArgumentParser(description="Evaluate LLM-judge raw results.")
    ap.add_argument("run_dir", type=Path,
                    help="run directory holding run.jsonl (as produced by the runner)")
    ap.add_argument("--out", default=None,
                    help="eval output directory (default: <data-dir>/_evals/eval_<run-timestamp>)")
    ap.add_argument("--bins", type=int, default=10, help="calibration bins")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    run_dir = args.run_dir
    raw = _resolve_run_jsonl(run_dir)
    # mirror the run folder's timestamp (run_YYYYmmdd_HHMMSS) onto the eval folder
    m = re.search(r"(\d{8}_\d{6})", run_dir.name)
    stamp = m.group(1) if m else _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else DEFAULT_LLM_JUDGE_DATA_DIR / "_evals" / f"eval_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = loader.load_runs(raw)
    warns = loader.validate(df)
    for w in warns:
        print(f"  [warn] {w}")

    results = run_blocks(df, n_bins=args.bins)
    plots_made = [] if args.no_plots else render_plots(df, results, out_dir)

    # dump machine-readable metrics (DataFrames -> records)
    metrics = {name: _jsonable(r["result"]) for name, r in results.items()}
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    # full per-file table as its own CSV (97 rows is too long for the md body)
    if "per_file" in results:
        results["per_file"]["result"]["per_file"].to_csv(
            out_dir / "per_file.csv", index=False)

    write_markdown(df, results, warns, plots_made, out_dir, str(raw))
    print(f"\nWrote {out_dir/'report.md'}, metrics.json, {len(plots_made)} plot(s) -> {out_dir}")


if __name__ == "__main__":
    main()
