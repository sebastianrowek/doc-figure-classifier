"""Bulk classification runner for the LLM-judge calibration set.

Walks the class folders under ``data/llm_judge`` (folders named after the Tier-1
taxonomy), classifies every image ``--repeats`` times through the judge model,
and appends one JSON line per call to a JSONL file. That file is the raw-results
contract consumed by ``llm_judge_eval.loader`` -- see its docstring for the schema.

This is the ONLY paid step. Run it once with N repeats; every metric and plot in
``llm_judge_eval`` is then recomputed for free from the JSONL.

Every completed call is written -- successes *and* failures (``pred = null``,
``error`` set) -- so accuracy denominators and per-image repeat counts stay even.

Example:
    python -m DocumentFigureClassifier.extract.llm_judge_eval.runner --repeats 5 --max-concurrency 5
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from DocumentFigureClassifier.extract.llm_classify import DEFAULT_CONFIG, async_llm_classify_image
from DocumentFigureClassifier.schemas import LlmJudgeCallResult
from DocumentFigureClassifier.taxonomy import TIER1_LABELS
from DocumentFigureClassifier.constants import DEFAULT_LLM_JUDGE_DATA_DIR

load_dotenv()
log = logging.getLogger("judge_runner")

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".gif"}

def discover_images(data_dir: Path, labels: list[str]) -> list[tuple[str, Path]]:
    """Return ``(class, path)`` for every image inside a taxonomy-named folder.

    Only the class folders are walked, so scratch dirs (``_removed``, ``_runs``)
    and manifests are skipped automatically.
    """
    found: list[tuple[str, Path]] = []
    for cls in labels:
        d = data_dir / cls
        if not d.is_dir():
            log.warning("missing class folder: %s", d)
            continue
        imgs = [p for p in sorted(d.iterdir()) if p.is_file() and p.suffix.lower() in IMG_EXT]
        if not imgs:
            log.warning("no images in class folder: %s", d)
        found.extend((cls, p) for p in imgs)
    return found


async def _classify_once(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    lock: asyncio.Lock,
    fh,
    data_dir: Path,
    cls: str,
    path: Path,
    call_idx: int,
    retries: int,
    fail_dir: Path,
) -> LlmJudgeCallResult:
    """One classification (with up to ``retries`` extra attempts on failure);
    append the final result line under the lock. If every attempt fails, dump the
    last raw model response to ``fail_dir`` for inspection."""
    total_cost: float | None = None
    attempt = 0
    while True:
        async with sem:
            prediction, cost, error, raw = await async_llm_classify_image(client, path, logger=log)
        if cost is not None:
            total_cost = (total_cost or 0.0) + cost
        if prediction is not None or attempt >= retries:
            break
        attempt += 1
        log.warning("retry %d/%d for %s (%s)", attempt, retries, path.name, error)
        await asyncio.sleep(0.5 * attempt)

    rel = path.relative_to(data_dir).as_posix()
    if prediction is None:
        _dump_failure(fail_dir, rel, call_idx, attempt, error, raw)

    rec: LlmJudgeCallResult = {
        "file": rel,
        "true": cls,
        "pred": prediction["label"] if prediction else None,
        "confidence": prediction["confidence"] if prediction else None,
        "call_idx": call_idx,
        "cost": total_cost,
        "error": error,
    }
    async with lock:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
    return rec


def _dump_failure(
    fail_dir: Path,
    rel: str,
    call_idx: int,
    attempts: int,
    error: str | None,
    raw: str | None,
) -> None:
    """Write the raw model response (plus context) for a call that failed after
    all retries. One uniquely-named file per failing call, so no lock is needed."""
    fail_dir.mkdir(parents=True, exist_ok=True)
    stem = rel.replace("/", "__").rsplit(".", 1)[0]
    fp = fail_dir / f"{stem}__c{call_idx}.txt"
    header = (
        f"file:     {rel}\n"
        f"call_idx: {call_idx}\n"
        f"attempts: {attempts + 1}\n"
        f"error:    {error}\n"
        f"{'-' * 60}\n"
    )
    body = raw if raw is not None else "<no response body — request failed before a reply>"
    fp.write_text(header + body, encoding="utf-8")
    log.error("saved raw failing response -> %s", fp)


def _write_meta(path: Path, meta: dict) -> None:
    """Write light run metadata (model/decoding config + run params + results)
    next to the JSONL, so a result file is always traceable to how it was made."""
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


async def main() -> None:
    ap = argparse.ArgumentParser(description="Run the LLM-judge classification over the calibration set.")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_LLM_JUDGE_DATA_DIR,
                    help="root holding the Tier-1 class folders (default: data/llm_judge)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="run output directory (default: <data-dir>/_runs/run_<timestamp>); "
                         "holds run.jsonl, run.meta.json and failures/")
    ap.add_argument("--repeats", type=int, default=1,
                    help="calls per image, to sample the judge's run-to-run variance")
    ap.add_argument("--max-concurrency", type=int, default=5,
                    help="max in-flight requests to OpenRouter")
    ap.add_argument("--limit", type=int, default=None,
                    help="classify only the first N images (cheap dry-run)")
    ap.add_argument("--retries", type=int, default=2,
                help="extra attempts for a failed/off-schema call, e.g. JSON-decode errors (0 = no retry)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out_dir = args.out_dir or args.data_dir / "_runs" / f"run_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "run.jsonl"
    # raw responses of calls that fail after all retries land here for inspection
    fail_dir = out_dir / "failures"

    images = discover_images(args.data_dir, TIER1_LABELS)
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        log.error("no images found under %s", args.data_dir)
        return

    n_calls = len(images) * args.repeats
    log.info("%d images x %d repeats = %d calls -> %s", len(images), args.repeats, n_calls, out)

    # record run metadata up-front (survives a crash); results filled in at the end
    meta_path = out_dir / "run.meta.json"
    meta = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "out_dir": out_dir.name,
        "config": dataclasses.asdict(DEFAULT_CONFIG),
        "run": {
            "n_images": len(images),
            "repeats": args.repeats,
            "n_calls": n_calls,
            "max_concurrency": args.max_concurrency,
            "retries": args.retries,
            "limit": args.limit,
        },
    }
    _write_meta(meta_path, meta)

    sem = asyncio.Semaphore(args.max_concurrency)
    lock = asyncio.Lock()
    total_cost = 0.0
    n_fail = 0

    with open(out, "w", encoding="utf-8") as fh:
        async with AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
            timeout=60.0,
            max_retries=2,
        ) as client:
            tasks = [
                _classify_once(client, sem, lock, fh, args.data_dir, cls, path, k, args.retries, fail_dir)
                for cls, path in images
                for k in range(args.repeats)
            ]
            done = 0
            for coro in asyncio.as_completed(tasks):
                rec = await coro
                done += 1
                total_cost += rec["cost"] or 0.0
                n_fail += rec["pred"] is None
                if done % 25 == 0 or done == n_calls:
                    log.info("progress %d/%d (%d failed, ~$%.4f)", done, n_calls, n_fail, total_cost)

    meta["finished_at"] = datetime.now().isoformat(timespec="seconds")
    meta["results"] = {"n_calls": n_calls, "n_failed": n_fail, "total_cost_usd": round(total_cost, 6)}
    _write_meta(meta_path, meta)

    log.info("done: %d calls, %d failed, ~$%.4f -> %s", n_calls, n_fail, total_cost, out_dir)


if __name__ == "__main__":
    asyncio.run(main())
