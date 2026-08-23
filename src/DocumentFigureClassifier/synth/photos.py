"""
Photo ingestion -- the one class that cannot be synthesised.

The design doc (§6.5) is explicit: photographs are not generated, they are
*collected*. Annual reports are visually full of photography and the model has
to learn to reject it, so the training photos have to be real photographs. The
recommended primary source is the extraction pipeline's own output -- a single
report already yields 100+ photo crops -- optionally supplemented by an external
corpus (COCO/Open Images) run through the same pipeline.

This module is that collection step. It reads real photo crops from a directory,
augments each one through the shared degradation pipeline (so a modest corpus
turns into a larger, more varied training set), and writes them into
``train/photo/`` with manifest lines marked ``synthetic: false``.

Two rules from the guide are enforced here rather than trusted to be remembered:

* **Document-level split.** Val photos must come from *different reports* than
  training photos, or photo validation is measuring memorisation. ``exclude``
  takes the report stems reserved for validation; any crop whose source report
  is excluded is skipped. With a single report in the corpus no clean split is
  possible yet -- the function still runs, but logs that the output must not be
  mixed with a val set drawn from the same report.
* **Augmentation is not synthesis.** The augmented copies are still the same
  photograph; they inflate the training set, they do not create new ground
  truth, and they never go to val/test (the Writer already refuses non-train
  splits).
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

from . import degrade
from .rng import Rng
from .sizes import SizeSampler

log = logging.getLogger("synth.photos")

_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def _report_stem(path: Path) -> str:
    """
    The source report a crop came from.

    The extraction pipeline names crops ``<ReportStem>__pNNN__NNN__...png``, so
    everything before the first ``__`` identifies the report. Falls back to the
    whole stem for files that do not follow the convention.
    """
    name = path.stem
    return name.split("__", 1)[0] if "__" in name else name


def collect_photos(
    src: Path,
    writer,
    sizes: SizeSampler,
    base_seed: int,
    per_source: int = 3,
    exclude: set[str] | None = None,
    degrade_photos: bool = True,
) -> int:
    """
    Ingest real photo crops from ``src`` into the training set.

    Each source image is written ``per_source`` times: once near-clean and the
    rest augmented, so a corpus of P photos yields ~P*per_source training
    samples. Returns the number of samples written.
    """
    exclude = exclude or set()
    if not src.is_dir():
        log.warning(
            "photo source %s not found -- the photo class will be empty. Point "
            "--photos-src at a directory of extracted photo crops (e.g. the "
            "extraction pipeline's review/photo/ folder).", src,
        )
        return 0

    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in _EXTS)
    if not files:
        log.warning("no image files under %s -- photo class will be empty", src)
        return 0

    reports = {_report_stem(f) for f in files}
    kept = [f for f in files if _report_stem(f) not in exclude]
    skipped = len(files) - len(kept)
    log.info("photo source: %d files across %d report(s); %d excluded for val split",
             len(files), len(reports), skipped)
    if len(reports - exclude) <= 1:
        log.warning(
            "photos come from a single report (%s); a clean document-level "
            "train/val split is not possible yet. Do NOT build a validation set "
            "from the same report, or photo accuracy will be measured on "
            "memorised images. Add more reports before relying on this class.",
            ", ".join(sorted(reports - exclude)) or "none",
        )

    n = 0
    for fi, f in enumerate(kept):
        try:
            img = Image.open(f).convert("RGB")
        except Exception as exc:  # noqa: BLE001 - one unreadable file must not stop the run
            log.warning("skip unreadable %s: %s", f.name, exc)
            continue
        report = _report_stem(f)
        for aug in range(per_source):
            rng = Rng((base_seed + fi * 131 + aug * 7919) % (2**32))
            tw, th = sizes.draw(rng)
            if degrade_photos:
                # First copy stays light; later copies get the full range, so the
                # class spans crisp magazine-quality shots and rough re-scans.
                sev = degrade.draw_severity(rng) * (0.4 if aug == 0 else 1.0)
                out, dparams = degrade.apply(img, (tw, th), sev, rng, background="#ffffff")
                dmeta = dparams.as_dict()
            else:
                out = img.resize((tw, th), Image.Resampling.LANCZOS)
                sev, dmeta = 0.0, None

            crop_id = f"photo__{report}__{f.stem[-24:]}__a{aug}"
            record = {
                "crop_id": crop_id,
                "synthetic": False,
                "label": "photo",
                "sub_type": "real",
                "generator": "photos.collect_photos",
                "engine": "corpus",
                "source_pdf": report,
                "source_file": f.name,
                "aug_index": aug,
                "severity": round(sev, 4),
                "width": out.width,
                "height": out.height,
                "degrade": dmeta,
            }
            writer.save(out, record)
            n += 1

    log.info("photos: wrote %d training samples from %d source photo(s)", n, len(kept))
    return n
