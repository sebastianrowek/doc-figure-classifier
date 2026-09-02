r"""
Synthetic figure generator.

Quick start
-----------
    # phase 0 smoke run: 200 bar charts plus a contact sheet
    python -m DocumentFigureClassifier.synth.cli --n 200 --contact-sheet

    # look at what the degradation is doing to one sample
    python -m DocumentFigureClassifier.synth.cli \
        --regenerate bar__plain__00123456 --dump-stages tmp/stages

Everything about a sample follows from ``(label, sub_type, seed)``, so any
image in the manifest can be reproduced exactly by passing its ``crop_id`` to
``--regenerate``. That is worth the small amount of plumbing: without it,
"why does this one image look wrong" is unanswerable.

The generated manifest uses the same keys as ``data/parsed/manifest.jsonl``
where they overlap (``crop_id``, ``width``, ``height``, ``filename``), so the
real and synthetic sets concatenate without an adapter.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image

from ..taxonomy import TIER1_LABELS
from . import degrade, photos
from .config import REGISTRY, RunConfig, allocate, load_class_counts
from .qa import contact_sheet, stats
from .renderers.base import InvariantViolation, check
from .rng import Rng
from .sizes import SizeSampler
from .style import sample_style
from .writer import Writer

log = logging.getLogger("synth")

# Per-process cache. Built lazily so worker processes each get their own and
# nothing unpicklable has to cross the process boundary.
_SIZES: dict[str, SizeSampler] = {}


def _sizes(cfg: RunConfig) -> SizeSampler:
    key = cfg.size_manifest
    if key not in _SIZES:
        path = Path(key)
        _SIZES[key] = SizeSampler.from_manifest(path if path.is_file() else None)
    return _SIZES[key]


def seed_for(base: int, label: str, sub_type: str, index: int) -> int:
    """
    Stable seed for one sample.

    Python's ``hash()`` of a string is salted per process, so it cannot be used
    here -- the same task would get different seeds in a worker than in the
    parent.
    """
    h = hashlib.sha1(f"{base}|{label}|{sub_type}|{index}".encode()).digest()
    return int.from_bytes(h[:4], "big")


# --------------------------------------------------------------------------
# One sample
# --------------------------------------------------------------------------


def build_sample(
    label: str, sub_type: str, seed: int, cfg: RunConfig, dump_stages: Path | None = None
) -> tuple[dict, bytes]:
    """
    Draw one sample. Retries with a fresh seed if the invariant check fails.

    A violation is not an error in the pipeline -- it means the randomisation
    produced an image whose structure no longer matches the label, and the only
    correct response is to throw it away. If it happens often for a sub-type,
    that sub-type's parameter ranges are wrong.
    """
    plan = REGISTRY[label]
    last: Exception | None = None

    for attempt in range(cfg.max_invariant_retries):
        eff_seed = (seed + attempt * 1_000_003) % (2**32)
        try:
            return _build_once(plan, sub_type, eff_seed, cfg, attempt, dump_stages)
        except InvariantViolation as exc:
            last = exc
            log.debug("invariant retry %d for %s/%s: %s", attempt, label, sub_type, exc)

    raise RuntimeError(
        f"{label}/{sub_type}: {cfg.max_invariant_retries} attempts all violated "
        f"invariants; last was {last}"
    )


def _build_once(
    plan, sub_type: str, eff_seed: int, cfg: RunConfig, attempt: int, dump_stages: Path | None
) -> tuple[dict, bytes]:
    rng = Rng(eff_seed)

    target_w, target_h = _sizes(cfg).draw(rng)
    style = sample_style(rng, target_w, target_h)
    spec = plan.build_spec(sub_type, style, rng)
    oversample = rng.uniform(cfg.oversample_lo, cfg.oversample_hi)

    result = plan.render(spec, style, rng, oversample)
    check(plan.label, result.structure)  # raises InvariantViolation

    if dump_stages is not None:
        dump_stages.mkdir(parents=True, exist_ok=True)
        result.image.save(dump_stages / f"{plan.label}__{sub_type}__{eff_seed:08d}__1_render.png")

    if cfg.degrade:
        severity = degrade.draw_severity(rng)
        image, dparams = degrade.apply(
            result.image, (target_w, target_h), severity, rng, style.palette.background
        )
        degrade_meta = dparams.as_dict()
    else:
        image = result.image.resize((target_w, target_h), Image.Resampling.LANCZOS)
        severity, degrade_meta = 0.0, None

    if dump_stages is not None:
        image.save(dump_stages / f"{plan.label}__{sub_type}__{eff_seed:08d}__2_final.png")

    crop_id = f"{plan.label}__{sub_type}__{eff_seed:08d}"
    record = {
        "crop_id": crop_id,
        "synthetic": True,
        "label": plan.label,
        "sub_type": sub_type,
        "generator": plan.render.__module__,
        "engine": "mpl",
        "seed": eff_seed,
        "retry": attempt,
        "style_preset": style.preset,
        "style_digest": style.digest(),
        "palette": style.palette.name,
        "language": style.language,
        "spines": style.spines,
        "yaxis": style.yaxis,
        "data_labels": style.data_labels,
        "render_dpi": round(style.render_dpi, 1),
        "severity": round(severity, 4),
        "width": image.width,
        "height": image.height,
        "degrade": degrade_meta,
        **result.meta,
    }

    buf = io.BytesIO()
    image.save(buf, format="PNG", optimize=False)
    return record, buf.getvalue()


def _worker(task: tuple[str, str, int, RunConfig]) -> tuple[dict, bytes]:
    label, sub_type, seed, cfg = task
    return build_sample(label, sub_type, seed, cfg)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, default=Path("data/synth"), help="output root")
    ap.add_argument("--n", type=int, default=200, help="samples per class (uniform)")
    ap.add_argument(
        "--classes", default="", help="comma-separated subset; default is everything registered"
    )
    ap.add_argument(
        "--counts-file",
        type=Path,
        default=None,
        help="YAML plan of per-class counts (e.g. `bar: 1500`); overrides --n and cannot be "
        "combined with --classes. Keys must be current tier-1 labels; see configs/synth_counts.yaml",
    )
    ap.add_argument("--seed", type=int, default=20260823)
    ap.add_argument("--workers", type=int, default=1, help="processes; 1 keeps tracebacks readable")
    ap.add_argument(
        "--size-manifest",
        type=Path,
        default=Path("data/parsed/manifest.jsonl"),
        help="real crop manifest whose size distribution is sampled",
    )
    ap.add_argument("--no-degrade", action="store_true", help="skip degradation (for debugging)")
    ap.add_argument("--append", action="store_true", help="append to an existing manifest")
    ap.add_argument("--contact-sheet", action="store_true", help="write QA contact sheets")
    ap.add_argument("--sheet-cols", type=int, default=8)
    ap.add_argument("--regenerate", default="", help="rebuild one crop_id and exit")
    ap.add_argument("--dump-stages", type=Path, default=None, help="with --regenerate: save stages")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--photos-src", type=Path, default=None,
                    help="directory of real photo crops to ingest as the photo class")
    ap.add_argument("--photos-per-source", type=int, default=3,
                    help="augmented copies written per source photo")
    ap.add_argument("--photos-exclude", default="",
                    help="comma-separated report stems reserved for validation (skipped)")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s"
    )
    os.environ.setdefault("MPLBACKEND", "Agg")

    cfg = RunConfig(
        out=str(args.out),
        n_per_class=args.n,
        base_seed=args.seed,
        size_manifest=str(args.size_manifest),
        degrade=not args.no_degrade,
    )

    if args.regenerate:
        return _regenerate(args.regenerate, cfg, args.dump_stages)

    if args.counts_file:
        if args.classes:
            log.error("--counts-file and --classes cannot be combined; the file names the classes")
            return 1
        try:
            counts = load_class_counts(args.counts_file)
        except (OSError, ValueError) as exc:
            log.error("counts file: %s", exc)
            return 1
        # Process in taxonomy order for stable, readable output; skip zeros.
        labels = [c for c in TIER1_LABELS if counts.get(c, 0) > 0]
        omitted = [c for c in TIER1_LABELS if c not in counts]
        zeroed = sorted(c for c, n in counts.items() if n == 0)
        if omitted:
            log.info("counts file omits (not generated): %s", ", ".join(omitted))
        if zeroed:
            log.info("counts file sets 0 (skipped): %s", ", ".join(zeroed))
        if counts.get("photo", 0) > 0:
            log.info(
                "photo count is advisory: photos are collected from the corpus (--photos-src) "
                "at --photos-per-source each, not rendered to a target count"
            )
    else:
        labels = [s for s in args.classes.split(",") if s] or list(REGISTRY)
        unknown = [lb for lb in labels if lb not in REGISTRY and lb != "photo"]
        if unknown:
            log.error(
                "not registered yet: %s (available: %s)", ", ".join(unknown), ", ".join(REGISTRY)
            )
            return 1
        counts = {lb: args.n for lb in labels}

    tasks: list[tuple[str, str, int, RunConfig]] = []
    for label in labels:
        if label == "photo":
            continue  # collected from a corpus, not rendered -- see photos.py
        for i, sub_type in enumerate(allocate(REGISTRY[label].subtypes, counts[label])):
            tasks.append((label, sub_type, seed_for(args.seed, label, sub_type, i), cfg))

    log.info("generating %d samples across %d class(es)", len(tasks), len(labels))
    sampler = _sizes(cfg)
    log.info("crop sizes sampled from %s (n=%d)", sampler.source, len(sampler.pairs))

    t0 = time.time()
    written: dict[str, list[tuple[Path, str]]] = {}
    failures = 0

    with Writer(args.out, split="train", append=args.append) as writer:
        for record, png in _run(tasks, args.workers):
            if record is None:
                failures += 1
                continue
            path = writer.save_png_bytes(png, record)
            written.setdefault(record["label"], []).append((path, record["sub_type"]))

        dt = time.time() - t0
        log.info("done in %.1fs (%.1f img/s)", dt, len(tasks) / max(dt, 1e-6))
        if failures:
            log.warning("%d sample(s) failed and were skipped", failures)
        if args.photos_src or "photo" in labels:
            src = args.photos_src or Path("data/parsed/review/photo")
            exclude = {s for s in args.photos_exclude.split(",") if s}
            photos.collect_photos(src, writer, sampler, args.seed,
                                  per_source=args.photos_per_source, exclude=exclude,
                                  degrade_photos=not args.no_degrade)
        print(writer.summary())
        total_written = sum(writer.counts.values())

    if total_written == 0:
        log.warning("nothing was written -- no stats or contact sheets to build")
        return 0

    qa_dir = args.out / "qa"
    stats.build(args.out / "manifest.jsonl", args.size_manifest, qa_dir / "stats.md")
    log.info("stats: %s", qa_dir / "stats.md")

    if args.contact_sheet:
        for label, entries in written.items():
            entries.sort(key=lambda e: e[1])
            out = contact_sheet.build(
                [p for p, _ in entries][:64],
                qa_dir / f"contact_{label}.png",
                captions=[s for _, s in entries][:64],
                cols=args.sheet_cols,
                title=f"{label} -- {len(entries)} samples, first 64 by sub-type",
            )
            log.info("contact sheet: %s", out)
        photo_dir = args.out / "train" / "photo"
        photo_files = sorted(photo_dir.glob("*.png"))[:64] if photo_dir.is_dir() else []
        if photo_files:
            out = contact_sheet.build(photo_files, qa_dir / "contact_photo.png",
                                      cols=args.sheet_cols,
                                      title=f"photo -- {len(photo_files)} shown (real, augmented)")
            log.info("contact sheet: %s", out)

    return 0


def _run(tasks, workers: int):
    """Yield (record, png) pairs; (None, None) for a sample that failed."""
    if workers <= 1:
        for t in tasks:
            try:
                yield _worker(t)
            except Exception as exc:  # noqa: BLE001 - one bad sample must not kill the run
                log.error("%s/%s failed: %s", t[0], t[1], exc)
                yield None, b""
        return

    with ProcessPoolExecutor(max_workers=workers) as pool:
        for t, fut in [(t, pool.submit(_worker, t)) for t in tasks]:
            try:
                yield fut.result()
            except Exception as exc:  # noqa: BLE001
                log.error("%s/%s failed: %s", t[0], t[1], exc)
                yield None, b""


def _regenerate(crop_id: str, cfg: RunConfig, dump_stages: Path | None) -> int:
    try:
        label, sub_type, seed_str = crop_id.rsplit("__", 2)
        seed = int(seed_str)
    except ValueError:
        log.error("crop_id must look like <label>__<sub_type>__<seed>, got %r", crop_id)
        return 1
    if label not in REGISTRY:
        log.error("not registered: %s", label)
        return 1

    out = dump_stages or Path("tmp/regenerate")
    record, png = build_sample(label, sub_type, seed, cfg, dump_stages=out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{crop_id}.png").write_bytes(png)
    log.info("wrote %s", out / f"{crop_id}.png")
    for k, v in record.items():
        log.info("  %-14s %s", k, v)
    return 0


if __name__ == "__main__":
    sys.exit(main())
