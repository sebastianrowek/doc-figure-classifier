r"""
Extract figures from annual-report PDFs and pre-classify them into a
review folder structure that matches the labeling guide.

Pipeline
--------
1. Docling detects picture regions on each page and renders them as crops.
   (Layout detection, not raster extraction -- this is what makes vector
   charts work; see the note on PyMuPDF at the bottom of this file.)
2. Junk crops are filtered out by size / area / aspect ratio.
3. Each crop is classified by DocumentFigureClassifier-v2.5.
4. Its 26 classes are mapped onto the tier-1 taxonomy from the guide.
5. Crops are written to review/<proposed_label>/, low-confidence ones to
   review/_review/, plus a manifest.jsonl with all metadata.

Correcting a label = moving the file to the right folder.

Install (CPU-only machine with internet)
----------------------------------------
    python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install docling transformers pillow

The CPU index URL is worth the extra line: the default wheels bundle CUDA
and pull down roughly 2.5 GB you will never use.

Quick start
-----------
Nothing to pre-download and nothing to transfer. Both models are fetched
from the Hugging Face Hub on first run and cached in ~/.cache/huggingface
(Windows: %USERPROFILE%\.cache\huggingface); later runs are offline anyway.
Expect a few hundred MB and a couple of minutes on that first call.

    # smoke test on three PDFs first -- always do this before a full corpus
    python extract_and_classify.py ./reports ./out --limit 3

    # for this repo
    python extract_and_classify.py ../../../data/reports ../../../data/parsed --limit 1

    # full run
    python extract_and_classify.py ./reports ./out

Then open ./out/review/ and start moving files between folders.

Useful flags on CPU:

    --threads 8       CPU threads; set to your physical core count
    --scale 3.0       higher render resolution; use if waterfall connector
                      lines look faint in the crops
    --include-tables  also export detected tables (turns on TableFormer,
                      which is expensive -- leave off unless you need it)
    --limit N         process at most N PDFs

Performance
-----------
Runtime is dominated by docling's layout model, not by the classifier
(4M params, ~2% of the total). Budget roughly 1-3 s per page on a modern
laptop CPU, so a 150-page annual report takes a few minutes.

    --device cuda     use an NVIDIA GPU; roughly 4-6x end to end
    --device mps      Apple Silicon
    --device auto     detect automatically (default)

The remaining CPU-bound stage is PDF parsing and page rendering, which
caps the achievable speedup regardless of GPU.

Many cores but no GPU? Parallelising across documents beats throwing
threads at one model, since the layout model stops scaling past ~8:

    ls reports/*.pdf | xargs -P 4 -I{} \
        python extract_and_classify.py {} ./out --threads 2

More RAM does not help. Nothing here is memory-bound above ~8 GB.

Air-gapped machine
------------------
Only relevant if the machine has no internet. Pre-fetch on an online box:

    docling-tools models download -o ./docling-models
    python -c "from huggingface_hub import snapshot_download; \
        snapshot_download('docling-project/DocumentFigureClassifier-v2.5', \
        local_dir='./DocumentFigureClassifier-v2.5')"

copy both folders over, then point the script at them:

    python extract_and_classify.py ./reports ./out \
        --docling-artifacts ./docling-models \
        --model-dir ./DocumentFigureClassifier-v2.5
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from typing import cast
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image
from transformers import EfficientNetForImageClassification

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.items.picture.picture import PictureItem 
from docling_core.types.doc.items.table.table import TableItem

# Moved between modules across docling 2.x releases; older builds have neither.
try:
    from docling.datamodel.accelerator_options import AcceleratorOptions
except ImportError:  # pragma: no cover
    try:
        from docling.datamodel.pipeline_options import AcceleratorOptions  # type: ignore
    except ImportError:
        AcceleratorOptions = None  # type: ignore

log = logging.getLogger("extract")

# --------------------------------------------------------------------------
# Taxonomy
# --------------------------------------------------------------------------

# Tier-1 labels live in taxonomy.py so the synthetic generator and this script
# cannot drift apart. Folders are created for all of them even when the
# pre-classifier can never propose them -- the labeler needs somewhere to move
# corrections to.

from DocumentFigureClassifier.taxonomy import EXTRA_FOLDERS, TIER1_LABELS
from DocumentFigureClassifier.extract.llm_classify import llm_classify_image

# The pretrained model knows nothing about waterfall, stacked/grouped bars,
# combo charts or donuts. Everything it cannot express collapses into a coarse
# proposal that the labeler splits by hand: bar_chart -> bar covers plain,
# grouped and stacked alike, so the labeler carves bar_grouped and bar_stacked
# out of the bar/ folder. It DOES know flow charts and scatter plots, which map
# straight to the new flow / scatter classes.
DOCLING_TO_TIER1 = {
    "bar_chart": "bar",
    "line_chart": "line",
    "pie_chart": "pie_donut",
    "geographical_map": "map",
    "topographical_map": "map",
    "table": "table",
    "photograph": "photo",
    "logo": "logo_icon",
    "icon": "logo_icon",
    "stamp": "logo_icon",
    "signature": "logo_icon",
    "flow_chart": "flow",
    "scatter_plot": "scatter",
    # everything else the model knows is not a tier-1 chart type
    "box_plot": "other",
    "engineering_drawing": "other",
    "chemistry_structure": "other",
    "screenshot_from_computer": "other",
    "screenshot_from_manual": "other",
    "page_thumbnail": "other",
    "full_page_image": "other",
    "music": "other",
    "calendar": "other",
    "qr_code": "other",
    "bar_code": "other",
    "crossword_puzzle": "other",
    "other": "other",
}

# Classes the model reliably confuses with charts in corporate documents.
# Route these to _review regardless of confidence.
ALWAYS_REVIEW = {"other", "full_page_image", "page_thumbnail", "screenshot_from_computer"}

# Normalization from the model card. The std values are unusual (ImageNet is
# [0.229, 0.224, 0.225]) but they match how the model was trained -- do not
# "fix" them.
PREPROCESS = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.47853944, 0.4732864, 0.47434163],
        ),
    ]
)


@dataclass
class Crop:
    """One extracted figure plus everything needed to trace it back."""

    crop_id: str
    source_pdf: str
    page: int
    kind: str  # "picture" | "table"
    width: int
    height: int
    bbox: list[float] | None
    raw_label: str | None = None
    raw_confidence: float | None = None
    proposed_label: str | None = None
    llm_label: str | None = None
    llm_confidence: float | None = None
    llm_cost: float | None = None
    routed_to: str | None = None
    decision_layer: str | None = None
    filename: str | None = None


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


def build_converter(
    artifacts_path: Path | None,
    scale: float,
    ocr: bool,
    device: str,
    threads: int | None,
    do_tables: bool,
) -> DocumentConverter:
    opts = PdfPipelineOptions()
    opts.images_scale = scale  # 1.0 == 72 dpi; 2.0 ~ 144 dpi
    opts.generate_picture_images = True
    opts.generate_page_images = False
    opts.do_ocr = ocr  # annual reports are usually digital -> off is much faster
    # TableFormer is expensive and its output is only used with --include-tables.
    # Leaving it on otherwise burns roughly a third of the runtime for nothing.
    opts.do_table_structure = do_tables
    if artifacts_path is not None:
        opts.artifacts_path = str(artifacts_path)

    if AcceleratorOptions is not None:
        kwargs = {"device": device}
        if threads:
            kwargs["num_threads"] = threads # type: ignore
        opts.accelerator_options = AcceleratorOptions(**kwargs) # type: ignore
    elif device != "auto":
        log.warning("this docling version has no accelerator options; --device ignored")

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )


def is_junk(img: Image.Image, min_side: int, min_area: int, max_aspect: float) -> str | None:
    """Return a reason string if the crop should be discarded, else None."""
    w, h = img.size
    if w < min_side or h < min_side:
        return f"too_small_{w}x{h}"
    if w * h < min_area:
        return f"area_too_small_{w * h}"
    aspect = max(w / h, h / w)
    if aspect > max_aspect:
        return f"extreme_aspect_{aspect:.1f}"
    # near-uniform crops are separator bars, colour fields, blank regions
    # near-uniform crops are separator bars, colour fields, blank regions
    lo, hi = cast(tuple[float, float], img.convert("L").getextrema())
    if hi - lo < 12:
        return "near_uniform"
    return None


def extract_crops(
    pdf_path: Path,
    converter: DocumentConverter,
    include_tables: bool,
) -> list[tuple[Crop, Image.Image]]:
    """Run docling on one PDF and yield (metadata, PIL image) pairs."""
    result = converter.convert(str(pdf_path))
    doc = result.document
    out: list[tuple[Crop, Image.Image]] = []
    counter = 0

    for element, _level in doc.iterate_items():
        if isinstance(element, PictureItem):
            kind = "picture"
        elif include_tables and isinstance(element, TableItem):
            kind = "table"
        else:
            continue

        try:
            img = element.get_image(doc)
        except Exception as exc:  # noqa: BLE001 - one bad figure must not kill the run
            log.warning("%s: could not render %s: %s", pdf_path.name, kind, exc)
            continue
        if img is None:
            continue

        page, bbox = None, None
        if element.prov:
            prov = element.prov[0]
            page = prov.page_no
            b = prov.bbox
            bbox = [b.l, b.t, b.r, b.b]

        counter += 1
        crop = Crop(
            crop_id=f"{pdf_path.stem}__p{page or 0:03d}__{counter:03d}",
            source_pdf=pdf_path.name,
            page=page or 0,
            kind=kind,
            width=img.width,
            height=img.height,
            bbox=bbox,
        )
        out.append((crop, img.convert("RGB")))

    return out


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------


DEFAULT_MODEL = "docling-project/DocumentFigureClassifier-v2.5"


class FigureClassifier:
    def __init__(self, model_dir: str, batch_size: int = 16, threads: int | None = None):
        if threads:
            torch.set_num_threads(threads)
        # A local folder is loaded strictly offline; a bare repo id is fetched
        # from the Hub (and cached) on first use.
        is_local = Path(model_dir).is_dir()
        if is_local:
            log.info("loading classifier from %s", model_dir)
        else:
            log.info("loading classifier %s (downloads on first run)", model_dir)
        self.model = EfficientNetForImageClassification.from_pretrained(
            model_dir, local_files_only=is_local
        )
        self.model.eval()
        self.id2label = cast(dict[int, str], self.model.config.id2label)
        self.batch_size = batch_size

    @torch.no_grad()
    def predict(self, images: list[Image.Image]) -> list[tuple[str, float]]:
        preds: list[tuple[str, float]] = []
        for i in range(0, len(images), self.batch_size):
            chunk = images[i : i + self.batch_size]
            batch = torch.stack([cast(torch.Tensor, PREPROCESS(im)) for im in chunk])
            probs = self.model(batch).logits.softmax(dim=1)
            conf, idx = probs.max(dim=1)
            preds.extend(
                (self.id2label[int(j)], float(c)) for j, c in zip(idx.tolist(), conf.tolist())
            )
        return preds


def route(raw_label: str, confidence: float, threshold: float) -> tuple[str, str]:
    """Map a raw prediction to (proposed_label, destination_folder)."""
    proposed = DOCLING_TO_TIER1.get(raw_label, "other")
    if confidence < threshold or raw_label in ALWAYS_REVIEW:
        return proposed, "_review"
    return proposed, proposed


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path, help="PDF file or directory of PDFs")
    ap.add_argument("output", type=Path, help="output directory")
    ap.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL,
        help=f"local classifier folder, or a HF repo id (default: {DEFAULT_MODEL})",
    )
    ap.add_argument(
        "--docling-artifacts",
        type=Path,
        default=None,
        help="local docling model folder; omit to download on first run",
    )
    ap.add_argument("--scale", type=float, default=2.0, help="render scale; 2.0 ~ 144 dpi")
    ap.add_argument("--threshold", type=float, default=0.75, help="confidence below this -> _review")
    ap.add_argument("--min-side", type=int, default=120, help="discard crops narrower/shorter than this")
    ap.add_argument("--min-area", type=int, default=30_000, help="discard crops below this pixel area")
    ap.add_argument("--max-aspect", type=float, default=8.0, help="discard extreme aspect ratios")
    ap.add_argument("--include-tables", action="store_true", help="also export detected tables")
    ap.add_argument("--ocr", action="store_true", help="enable OCR (slow; only for scanned PDFs)")
    ap.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps", "xpu"],
        help="inference device for docling's layout model (default: auto)",
    )
    ap.add_argument("--threads", type=int, default=None, help="CPU threads; set to physical core count")
    ap.add_argument("--limit", type=int, default=None, help="process at most N PDFs (for testing)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("docling").setLevel(logging.WARNING)

    pdfs = (
        [args.input]
        if args.input.is_file()
        else sorted(p for p in args.input.rglob("*.pdf"))
    )
    if args.limit:
        pdfs = pdfs[: args.limit]
    if not pdfs:
        log.error("no PDFs found under %s", args.input)
        return 1
    log.info("found %d PDF(s)", len(pdfs))

    review_root = args.output / "review"
    for name in TIER1_LABELS + EXTRA_FOLDERS:
        (review_root / name).mkdir(parents=True, exist_ok=True)
    discarded_dir = args.output / "discarded"
    discarded_dir.mkdir(parents=True, exist_ok=True)

    converter = build_converter(
        args.docling_artifacts,
        args.scale,
        args.ocr,
        args.device,
        args.threads,
        do_tables=args.include_tables,
    )
    classifier = FigureClassifier(args.model_dir, threads=args.threads)

    manifest_path = args.output / "manifest.jsonl"
    n_kept = n_junk = n_review = 0
    t0 = time.time()

    total_llm_cost : float = 0.0

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for n, pdf in enumerate(pdfs, 1):
            log.info("[%d/%d] %s", n, len(pdfs), pdf.name)
            try:
                crops = extract_crops(pdf, converter, args.include_tables)
            except Exception as exc:  # noqa: BLE001
                log.error("  failed: %s", exc)
                continue

            keep: list[tuple[Crop, Image.Image]] = []
            for crop, img in crops:
                reason = is_junk(img, args.min_side, args.min_area, args.max_aspect)
                if reason:
                    n_junk += 1
                    img.save(discarded_dir / f"{crop.crop_id}__{reason}.png")
                    continue
                keep.append((crop, img))

            if not keep:
                log.info("  0 usable figures")
                continue

            preds = classifier.predict([img for _, img in keep])

            for (crop, img), (raw_label, conf) in zip(keep, preds):
                proposed, dest = route(raw_label, conf, args.threshold)

                if dest == "_review":
                    llm_pred, llm_cost = llm_classify_image(img, log)

                    if llm_pred and llm_cost:
                        llm_label = llm_pred.get("label")
                        llm_confidence = llm_pred.get("confidence")

                        if llm_confidence and llm_confidence > args.threshold:
                            crop.llm_label = llm_label
                            dest = llm_label
                            crop.llm_confidence = llm_confidence
                            crop.decision_layer = "llm"
                            crop.llm_cost = llm_cost
                            total_llm_cost += llm_cost
                        else:
                            crop.decision_layer = "docling"

                crop.raw_label = raw_label
                crop.raw_confidence = round(conf, 4)
                crop.proposed_label = proposed
                crop.routed_to = dest
                # encode the proposal in the filename so it survives a move
                crop.filename = f"{crop.crop_id}__{proposed}__{conf:.2f}.png"
                img.save(review_root / dest / crop.filename)
                manifest.write(json.dumps(asdict(crop), ensure_ascii=False) + "\n")
                n_kept += 1
                if dest == "_review":
                    n_review += 1

            log.info("  %d figures kept", len(keep))

    dt = time.time() - t0
    log.info(
        "done in %.1fs -- %d figures kept (%d flagged for review), %d discarded",
        dt, n_kept, n_review, n_junk,
    )
    log.info("review here: %s", review_root)
    log.info("manifest:    %s", manifest_path)
    log.info("total LLM cost: %.4f$ct", total_llm_cost)
    return 0


# --------------------------------------------------------------------------
# Companion utility: rebuild a clean training set after hand-labeling
# --------------------------------------------------------------------------


def collect_labeled(review_root: Path, dest: Path) -> None:
    """
    After labeling, copy review/<label>/ into an imagefolder-style dataset,
    skipping _review / _unsure / _broken. Call manually:

        python -c "from extract_and_classify import collect_labeled; \
            from pathlib import Path; \
            collect_labeled(Path('out/review'), Path('dataset'))"
    """
    counts: dict[str, int] = {}
    for label in TIER1_LABELS:
        src = review_root / label
        if not src.is_dir():
            continue
        (dest / label).mkdir(parents=True, exist_ok=True)
        files = sorted(src.glob("*.png"))
        for f in files:
            shutil.copy2(f, dest / label / f.name)
        counts[label] = len(files)
    total = sum(counts.values())
    for label, c in sorted(counts.items(), key=lambda kv: -kv[1]):
        share = 100 * c / total if total else 0
        print(f"{label:16s} {c:6d}  {share:5.1f}%")
    print(f"{'TOTAL':16s} {total:6d}")


if __name__ == "__main__":
    sys.exit(main())


# --------------------------------------------------------------------------
# Why docling and not PyMuPDF
# --------------------------------------------------------------------------
# The obvious approach is page.get_images() in PyMuPDF, or `pdfimages`. Both
# extract only raster image *objects* from the PDF. Charts produced by Excel,
# Think-cell, matplotlib or a design tool are vector drawings -- sequences of
# path operators in the page content stream, not image objects. They are
# invisible to that approach, so on a typical annual report you get the
# photographs and the logos and none of the charts.
#
# Docling instead runs a layout model over a rendered page and crops the
# regions it identifies as pictures, which works regardless of how the chart
# was drawn. The cost is a much heavier dependency and slower processing.
#
# A lighter fallback, if docling is not an option: render each page with
# PyMuPDF at ~150 dpi, then find figure regions with page.get_drawings()
# clustered into bounding boxes. Workable, but you will spend real time
# tuning the clustering, and captions and legends tend to get cut off.