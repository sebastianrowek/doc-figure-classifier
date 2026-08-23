# Synthetic Training Set (12 Tier-1 Classes)

Status: implemented. 11 of the 12 classes render procedurally; `photo` is an
ingestion pipeline that needs a real photo corpus. A full run produces the whole
labelled set in a few minutes.

Goal: reproducible generation of labelled images that look like Docling crops
from annual-report PDFs, for training a lightweight figure classifier.

Reference: [labeling_guide_en.md](labeling_guide_en.md) (authoritative German
version: [labeling_guide_de.md](labeling_guide_de.md)).

This document describes what exists. Where a design decision was later changed by
what the environment or the contact sheets forced, the section says so rather
than keeping the original prediction.

---

## 0. Guiding decisions (the part that matters)

Five decisions determine whether this dataset is worth anything. Everything else
is legwork.

**(1) The label comes from the generator, not from the image.**
Every renderer knows its label up front. The risk is that aggressive
randomization tips an image into a *different* class — a waterfall whose bars all
happen to land on the baseline is a `bar_vertical`. Each renderer therefore
reports the structural facts of what it drew, and a **label invariant** is
checked afterwards; on violation the sample is discarded and redrawn with a
fresh seed. These invariants are the labeling guide expressed as code — see §4.

**(2) Matplotlib's default look is not the annual-report look.**
A model trained only on matplotlib defaults learns DejaVu Sans, axis crosses and
standard gridlines. Real annual reports are flat, axis-less, with data labels
written directly on the bars (rule R4). The generator therefore works from
explicit **style presets** (`corporate_flat`, `consulting_thinkcell`,
`classic_axes`, `mono_print`, `web_dashboard`, `dark_panel`), not from noise
around the defaults.

**(3) Degradation is fitted to the real crops, not guessed.**
The size sampler draws each target size from the empirical width/height pairs the
extraction pipeline actually produced. The reference distribution measured from
the first extracted report (n=408) was:

| Dimension | p05 | p25 | median | p75 | p95 | max |
|---|---|---|---|---|---|---|
| Width | 222 | 407 | 623 | 760 | 1256 | 2381 |
| Height | 174 | 254 | 360 | 504 | 938 | 1683 |
| Aspect ratio | 0.78 | 1.11 | **1.48** | 2.28 | 3.06 | 5.27 |

`sizes.SizeSampler` loads `data/parsed/manifest.jsonl` when present and jitters
real pairs; when it is absent it falls back to a built-in 28-size table distilled
from the quantiles above. **Currently the real manifest is not present in this
environment, so runs use the fallback** — re-extracting the corpus and passing
`--size-manifest data/parsed/manifest.jsonl` restores real-size sampling.

**(4) Synthetic means training only.**
Per the guide, val/test come exclusively from hand-labeled real images.
`writer.Writer` refuses any split other than `train`. Photos are the one class
where real images are used, and they carry a document-level split guard (§6.5) so
training photos never leak into a validation set built from the same report.

**(5) Confusion pairs get generated deliberately, not avoided.**
The guide names `waterfall` vs. `bar_stacked` as the most expensive error source.
A chunk of each class therefore comes from a **hard-variant pool** — waterfalls
without connector lines, stacked bars in green/red with signed labels, bar charts
with a target line (which per the guide is *not* a combo chart). See §5.

---

## 1. Module layout

The package is `DocumentFigureClassifier` (under `src/`). The synth code imports
nothing from the extraction pipeline except the shared taxonomy.

```
src/DocumentFigureClassifier/
├── taxonomy.py              # TIER1_LABELS + folder names + junk filter, single source
├── extract_and_classify.py  # extraction pipeline; imports taxonomy.py
└── synth/
    ├── cli.py               # entry point; sample loop, workers, --regenerate, photo step
    ├── config.py            # RunConfig, ClassPlan, REGISTRY, allocate()
    ├── rng.py               # Rng: one seeded numpy stream per sample
    ├── sizes.py             # SizeSampler: target sizes from the real crop distribution
    ├── style.py             # StyleSheet + presets, every randomization axis
    ├── palettes.py          # 24 corporate seed palettes, ramps, contrast helpers
    ├── fonts.py             # system-font discovery against an allowlist
    ├── content.py           # DE/EN vocabulary: KPIs, regions, segments, years, units, numbers
    ├── series.py            # plausible number series (trends, seasonality, breaks, signs)
    ├── degrade.py           # PDF / screenshot degradation
    ├── writer.py            # ImageFolder output + manifest.jsonl + train-only split guard
    ├── photos.py            # photo ingestion + augmentation (not a renderer)
    ├── engines/
    │   └── mpl.py           # Figure/FigureCanvasAgg -> PIL, no pyplot; text measurement
    ├── renderers/
    │   ├── base.py          # FigureSpec, Structure, RenderResult; the invariant checks
    │   ├── common.py        # shared layout machinery (fitting, legends, margins, headings)
    │   ├── bars.py          # bar_vertical, bar_horizontal, bar_stacked
    │   ├── waterfall.py
    │   ├── lines.py         # line + area (guide: area counts as line)
    │   ├── combo.py
    │   ├── pie.py           # pie, donut, centre-figure, exploded, semicircle
    │   ├── maps.py          # procedural choropleth / pins / bubble / outline / callout
    │   ├── tables.py
    │   ├── logos.py
    │   └── other.py         # 18 sub-generators incl. R1/R2 compose and the junk classes
    └── qa/
        ├── contact_sheet.py # montage grids per class, first 64 by sub-type
        └── stats.py         # balance, size distribution vs. real crops, style coverage
```

What the original plan pencilled in but the implementation did **not** need:
a separate `pil_engine` (matplotlib draws tables and logos fine — §6.2), a
`plotly_engine` (deferred — §6.1), a standalone `layout.py` compose module (R1/R2
live as `other` sub-types — §6.6), an `other/` package (one `other.py` with a
dispatch table is cleaner), and a `qa/smoke_probe.py` (not built — §11).

---

## 2. Core pipeline

A sample is fully determined by `(label, sub_type, seed)`.

```
seed ─► Rng ─► SizeSampler.draw ─► sample_style ─► build_spec ─► render
                                                                    │
                                              RenderResult(image, Structure, meta)
                                                                    │
                                     base.check(label, Structure)  ─┤ violated?
                                                                    │  discard, reseed,
                                                                    │  redraw (≤5 attempts)
                                                                    ▼
                                     degrade.apply  ─►  writer.save  ─►
                                     train/<label>/<crop_id>.png + manifest line
```

`crop_id = f"{label}__{sub_type}__{seed:08d}"`. Any image in the manifest can be
rebuilt exactly:

```
python -m DocumentFigureClassifier.synth.cli --regenerate waterfall__no_connectors__00041732 --dump-stages tmp/stages
```

which also dumps the pre-degradation render, so "why does this one look wrong" is
answerable. On an invariant violation the retry uses `seed + attempt*1_000_003`
(a large stride so the redraw is genuinely different), up to
`RunConfig.max_invariant_retries` (5); exhausting all five raises, which only
happens if a renderer's parameters have drifted into producing off-label images.

R1/R2 composition is **not** a separate post-step: the two cases are `other`
sub-types that compose real charts internally (§6.6), so they are labelled
`other` by construction and need no after-the-fact relabel.

---

## 3. Randomization axes (`style.py`)

A `StyleSheet` is a frozen dataclass; the chosen preset sets the probabilities it
is drawn from. `style_digest` (a hash of the whole sheet) goes into the manifest
so post-training error analysis can ask which style axes produce mistakes.

| Axis | Values |
|---|---|
| Palette | 24 corporate seed sets (2–6 colours), monochrome ramps, single-accent-on-grey, greyscale, dark |
| Background | White, off-white, light grey panel, tinted, rarely dark |
| Typeface | Sampled from installed system fonts filtered against an allowlist — 18 sans / 12 serif / 3 slab on this machine; a serif house style appears in every preset ~12 % of the time |
| Spines | all / left+bottom / bottom only / **none** |
| Gridlines | none / light horizontal / dashed / full |
| Numeric axis | present / **replaced by data labels (R4)** |
| Data labels | none / value / value+unit; inside/outside |
| Legend | none / top / bottom / right / direct series labelling |
| Title | none / title / title+subtitle; left/centred; optional source line |
| Geometry | figure size and aspect from the real crop distribution; render DPI ≈108–192 |
| Bars | width, gap, edge line, occasional hatch |
| Number format | DE `1.234,5` vs. EN `1,234.5`; units `Mio. €`, `€ m`, `%`, `Mitarbeiter`, … |
| Language | 75 % DE, 25 % EN |

Two traps handled explicitly:

- **Axis-lessness is over-represented** (guide R4). The `corporate_flat` and
  `consulting_thinkcell` presets drop spines and the numeric axis frequently, so
  the model does not learn axis geometry as a proxy for "chart" and then fall
  over on real corporate charts. `qa/stats.md` reports the actual coverage.
- **Font-manager noise.** Randomising across ~30 families means many lack a
  separate bold/normal weight file, so matplotlib substitutes the nearest weight
  and logs `findfont: Failed to find font weight …`. The substitution is cosmetic;
  `engines/mpl.py` raises the `matplotlib.font_manager` logger to ERROR (and
  filters the "glyph missing" warning) at import, so it is silenced in the render
  workers too — no `grep`/`-W ignore` needed.

### Nothing is laid out from an estimate

Randomised typography plus guessed text metrics produces a dataset full of
collided labels — a synthetic artefact the model would happily learn, and the
single largest source of defects found while building the class renderers. Every
decision that can overlap text is made against a real measurement
(`engines/mpl.text_width_pt`), because the sampled families differ in width by
nearly a factor of two:

- how many categories fit before the value labels collide (budgeted in `build_spec`)
- whether the numeric axis can be dropped, or has to come back because the labels did not fit
- whether category labels need shrinking, then rotating
- how many columns a legend gets, and how many rows that costs the margin (two-pass, sized against the **axes** width, not the figure width)
- whether a title needs shrinking, wrapping, or dropping (`common.cap_title_block`)
- which stack segments are big enough to hold their own label — a test that differs by orientation: one line's height for a vertical segment, the full string's width for a horizontal one
- the smallest pie share that still gets a printed percentage, derived from the circle's circumference in points rather than a fixed cut-off

---

## 4. Label invariants (the most important part)

Every renderer returns a `Structure` describing what it drew.
`renderers/base.check(label, structure)` raises `InvariantViolation` if that
contradicts the label; the sample loop catches it and redraws. Adding a renderer
without adding its check is a hard `KeyError`, not a silent pass.

| Class | Must hold | Must not hold |
|---|---|---|
| `bar_vertical` | ≥3 bars, all based at 0, vertical, 1 segment per bar | a non-reference line series; connector lines |
| `bar_horizontal` | as above, horizontal | same |
| `bar_stacked` | ≥2 segments per bar, all based at 0 | connector lines |
| `waterfall` | ≥4 bars, ≥2 floating, ≥2 on the baseline, 1 segment per bar | every bar on the baseline |
| `line` | ≥1 line series, **0 bar series** | bars |
| `combo_bar_line` | ≥1 bar series **and** ≥1 non-reference line series; that line has its own axis **or** its own legend entry | the only line is a reference line (constant across categories) |
| `pie_donut` | ≥2 segments | — |
| `map` | land geometry ≥ 45 % of the frame (shoelace) | — |
| `table` | ≥2 rows × ≥2 columns, **no embedded graphics** | bars/sparklines in cells (those are `other`) |
| `logo_icon` | no bar/line/pie data series | a data series |
| `other` | open by design | — |

One place the code is deliberately tighter than the guide's prose. The guide's
rule of thumb makes a chart a combo when the line has "its own axis or its own
legend entry" — but it also says a target line is not a data series, and target
lines routinely appear in the legend. The two statements conflict at the edge.
The structural discriminator is whether the line *varies across the categories*,
which is knowable at generation time; hence the `line_is_reference` flag. A
constant line is a reference line no matter how it is captioned. This resolution
was folded back into the labeling guide as **v1.1**.

The `map` invariant earns its keep in every run: a handful of maps draw under
45 % land on the first try and are redrawn — those are essentially the only
retries a healthy run produces.

---

## 5. Hard-variant pool (confusion budget)

Every row is a confusion the guide names explicitly. All are shipped except the
two noted.

| Sub-type | Label | Why |
|---|---|---|
| `waterfall/no_connectors` | `waterfall` | guide requires only 2 of 5 features |
| `waterfall/categorical_colors` | `waterfall` | colour by category, not direction |
| `waterfall/stacked_lookalike` | `waterfall` | intermediates packed tight, reads as a stack |
| `waterfall/subtotals` | `waterfall` | subtotals sitting back on the baseline |
| `bar_stacked/red_green_signed` | `bar_stacked` | green/red + signed labels, looks like a bridge |
| `bar_stacked/single_dominant` | `bar_stacked` | one sliver segment → reads as a plain bar |
| `bar_vertical/target_line` | `bar_vertical` | a target line is **not** a data series |
| `bar_vertical/average_line` | `bar_vertical` | same |
| `bar_vertical/grouped` | `bar_vertical` | grouping is not its own type |
| `combo_bar_line/flat_line` | `combo_bar_line` | near-constant margin, but real values + own axis + legend |
| `combo_bar_line/stacked_bars` | `combo_bar_line` | stack + line → decision-tree step 5 precedes step 8 |
| `pie_donut/semicircle` | `pie_donut` | half-circle / gauge as a part-to-whole display |
| `other/donut_progress` | `other` | a ring as a *progress* indicator, not part-to-whole |
| `other/table_with_bars` | `other` | a table whose value column is a bar (guide is explicit) |
| `line/area_stacked` | `line` | stacked area stays `line` |
| `other/multi_chart` | `other` | R1: two real charts composed, not separable |
| `other/infographic_frame` | `other` | R2: a real chart under 50 % of the frame |

Not implemented: `map/with_bars` (a map with overlaid bars) — a minor variant,
left for later. The old plan's `pie_donut/gauge_halfcircle` shipped as
`pie_donut/semicircle`.

### How the waterfall sub-types are chosen

The guide lists five identifying features and requires "at least two". Only the
first — first and last bar on the baseline, the ones between floating — is
structural; it is what makes the chart a bridge and what the invariant checks.
The other four are cosmetic. A generator that always emitted all five would teach
the model to look for green/red plus connector lines rather than for floating
bars, and would then fail on every corporate bridge that omits them. So each
sub-type switches one feature off (§5 rows above). All stay `waterfall` by
construction.

The mirror image is in place: `bar_vertical/target_line` and `average_line` draw
genuinely constant lines over bars and stay `bar_vertical`, while
`combo_bar_line/flat_line` draws a line that is visually almost flat but has real
per-category values, its own axis and its own legend entry (with a guard that
nudges the last point if rounding ever produced a truly constant line). Between
them the two classes cover both sides of the reference-line confusion.

---

## 6. Classes in detail

### 6.1 The seven chart classes — matplotlib

`bar_vertical`, `bar_horizontal`, `bar_stacked`, `waterfall`, `line`,
`combo_bar_line`, `pie_donut`. All drawn straight from `Figure` +
`FigureCanvasAgg`, no `pyplot` (its global registry leaks across a large run and
is unsafe in workers). Matplotlib has no built-in waterfall — it is built from
`bar(..., bottom=cumsum)` plus optional connector segments, which gives exactly
the control the invariant needs. `common.py` holds the layout machinery all seven
share (title/subtitle fitting, category-tick fitting, legend sizing, margins,
headings), extracted once the second renderer needed it.

**Plotly is deferred, not skipped.** The plan called for a Plotly engine on ~15 %
of chart samples for a recognizably different look and a native `go.Waterfall`.
Current `kaleido` (1.x) no longer bundles a browser — it drives a system Chrome
via `choreographer`, which this offline environment cannot download or run
headless reliably. The payoff is marginal stylistic variety the matplotlib
presets already cover, so the effort went to the guide-driven items instead. The
renderer interface (`spec, style, rng, oversample -> RenderResult`) is
engine-agnostic, so re-enabling it later means adding `engines/plotly.py` and
choosing which engine a sample draws through — no renderer changes.

### 6.2 `table`

Drawn on the matplotlib engine, **not** a separate PIL engine. Coordinates are in
points throughout (the axes spans `(axes_width_pt, table_height)` with y
inverted), so rules stay crisp and rows stay correctly proportioned regardless of
the crop's aspect ratio, and the font, measurement and degradation path are the
ones every other class already uses. A second engine bought nothing.

Sub-types: `multi_year` (row labels × year columns), `comparison` (two years plus
a change column), `segment_matrix` (segments × metrics), `key_figures`. Content is
real P&L / balance-sheet / KPI vocabulary from `content.py`, with subtotal rows
bold and ruled, optional header fill, zebra striping, and a rules style
(full/horizontal/header-only/minimal). Tables **with** an embedded bar are the
`other/table_with_bars` sub-type — the `table` invariant's `table_has_graphics`
flag rejects them, so they can only be generated under `other`.

### 6.3 `map`

Procedural, not GeoJSON. This environment has no network to fetch map data, and
hand-transcribing real country outlines from memory would be worse than none (a
wrong shape teaches a lie). So maps are a graticule plus jagged radial-noise
landmasses, filled as a choropleth or carrying pins / proportional bubbles /
borders / KPI callout boxes. Corporate maps are heavily stylised anyway, so an
abstract-but-jagged map with a lat/lon grid sits inside their distribution and
the classifier keys on exactly those cues. `maps._landmasses` is the single seam
to swap for real polygons once a simplified multi-region GeoJSON is available —
nothing else changes. The invariant (land ≥ 45 % of the frame) is computed by
shoelace; generation targets ~0.6. Sub-types: `choropleth`, `pins`, `bubble`,
`outline`, `kpi_callout`.

### 6.4 `logo_icon`

Entirely procedural matplotlib patch work, which is sufficient for this class:
the model needs the *shape* of the class (a single centred mark on empty ground,
no axes, no data series — which is the invariant), not real logos, which cannot
be used for trademark reasons anyway. Sub-types: `wordmark` (invented name +
geometric signet), `seal` (concentric rings, text on a circular path), `badge`
(laurel wreath / ribbon / stars), `icon_grid` (2×3 pictograms), `sdg_tiles`
(numbered coloured tiles — an **original** design, not the UN asset set).

### 6.5 `photo`

The one class that cannot be synthesised. It is implemented as an *ingestion*
step (`synth/photos.py`, `collect_photos`), not a renderer, because photographs
are collected, not generated. It reads real photo crops from a directory, writes
each one `--photos-per-source` times through the shared degradation pipeline (one
near-clean, the rest augmented — so a modest corpus becomes a larger, more varied
set), and records them with `synthetic: false` and `engine: "corpus"`.

Two guide rules are enforced, not trusted:

- **Document-level split.** `--photos-exclude` names the report stems reserved
  for validation (the part of a crop's filename before the first `__`); any crop
  from those reports is skipped, so training photos never leak into a val set
  drawn from the same report. With a single report present it runs but warns that
  no clean split is possible yet.
- **Augmentation is not synthesis.** The augmented copies are the same
  photograph; they inflate training only, and the Writer already refuses non-train
  splits.

It is wired into the CLI (`--photos-src`, default `data/parsed/review/photo/`) and
degrades gracefully to an empty class when no source exists — **the current state
here, since the extracted corpus is not present.** The recommended primary source
is the extraction pipeline's own output (a single report yields 100+ photo crops);
an external corpus (COCO/Open Images subset, run through the same pipeline) is a
supplement. The class stays empty until real crops are supplied; nothing fakes a
photograph.

### 6.6 `other` — 18 sub-generators

The most heterogeneous class, so the most sub-generators, dispatched from a table
in `other.py`. Its invariant is open by design — anything lands here — so what
matters is coverage of the shapes the extraction pipeline actually produces.

Diagrams: `org_chart`, `process_flow` (chevrons / cycles), `timeline`
(horizontal / vertical), `matrix` (2×2 / 3×3 risk / materiality point cloud),
`kpi_tile` (big number + label + arrow, single or a row), `gauge_progress`
(gauge / ring / progress bars / traffic light), `sankey`.

Chart-shaped but not tier-1 classes: `radar`, `bubble`, `tornado`, `scatter`,
`boxplot`.

The ones that are easy to forget and expensive to omit:

- `blank_artifact` — near-empty crops, half a letter, an edge strip, thin lines.
  The pipeline emits these constantly; without them the model labels extraction
  junk as a confident chart.
- `decorative` — gradients, stripes, ornaments, rules.
- `multi_chart` (R1) and `infographic_frame` (R2) — these compose **real**
  rendered charts, not stand-ins: `other._embed_chart` calls back into the class
  `REGISTRY` to render an actual bar/line/pie/etc. chart and `imshow`s it into the
  layout. So R1 tiles two genuine charts that cannot be cut apart, and R2 embeds a
  genuine chart under half the frame — which is precisely what teaches the
  "< 50 % → other" boundary.
- `table_with_bars`, `donut_progress` — the two §5 hard variants that must be
  `other` rather than `table` / `pie_donut`.

---

## 7. Degradation (`degrade.py`)

The stage order is not arbitrary — each operates on the previous one's output, or
you get a look no PDF ever produced. The render is done at 1.5–2.5× the target
size first (in the sample loop), giving headroom for rotation and resampling.

1. **Geometry** (`_geometry`): rotation ±0.6°, occasional slight perspective (PIL
   `QUAD`), per-side crop jitter — and, rarely, a hard clip of one side that lops
   off a title or half the legend, exactly like a faulty Docling bbox. Base jitter
   is kept small: a few percent off every side would shave the first letter off
   most titles on every image, which is itself a synthetic tell.
2. **Downsample** (`_resize`) to the target size, mixing Lanczos / bilinear / bicubic.
3. **Blur** (`_blur`): Gaussian σ 0–0.85, occasionally a slight unsharp halo.
4. **JPEG round trip** (`_jpeg`): quality ~35–95, ~10 % of the time applied twice
   (a forwarded screenshot).
5. **Noise** (`_noise`): Gaussian.
6. **Tone** (`_tone`): brightness / contrast / saturation / gamma jitter, rarely
   greyscale, rarely a warm scan tint.
7. **Save as PNG** — real crops are PNGs; the JPEG artefacts live *inside* the
   pixels, not in the container.

A per-sample `severity ∈ [0,1]` drawn from a beta distribution weights everything
toward light degradation; ~10 % of images stay near-clean, because vector charts
rendered from a PDF are often razor sharp and a model that only sees mush loses
exactly those. All degradation parameters are recorded in the manifest.

---

## 8. Volume and how to steer it

There is **no per-class-count flag.** `--n N` is applied uniformly to every
rendered class; `--classes a,b,c` restricts which classes run. For a non-uniform
plan, run once per class (or group) into the same output with `--append` (start
from a clean `data/synth`). Sub-type mix *within* a class is set by the
`SUBTYPES` weight dict at the top of each renderer module; `config.allocate` turns
those weights into exact counts by largest remainder, so rare sub-types are never
starved by chance.

A reasonable target distribution (heavier on the structurally hard classes,
lighter on the easy ones):

| Class | Count | Note |
|---|---|---|
| `bar_vertical` / `bar_horizontal` / `bar_stacked` | ~1,500 each | |
| `waterfall` | ~1,800 | hardest class |
| `combo_bar_line` | ~1,800 | second problem class |
| `line` | ~1,500 | incl. area, stacked area |
| `pie_donut` | ~1,200 | low shape variance |
| `map` | ~900 | |
| `table` | ~1,200 | |
| `logo_icon` | ~1,200 | |
| `other` | ~2,400 | 18 sub-types |
| `photo` | ~1,500 | real, once a corpus exists |
| **Total** | **~18,000** | |

This is a recommendation, not something the code enforces. Class imbalance can
also be handled downstream via class weights or calibration against the real
validation set.

---

## 9. Manifest and output

```
data/synth/
├── train/
│   ├── bar_vertical/ …
│   ├── other/ …
│   └── photo/            # populated only when --photos-src has real crops
├── manifest.jsonl
└── qa/
    ├── contact_bar_vertical.png …
    └── stats.md
```

One JSON line per sample. Keys overlap with `data/parsed/manifest.jsonl`
(`crop_id`, `width`, `height`, `filename`) so the real and synthetic sets
concatenate without an adapter. Rendered sample:

```json
{"crop_id": "waterfall__no_connectors__00041732", "synthetic": true,
 "label": "waterfall", "sub_type": "no_connectors",
 "generator": "DocumentFigureClassifier.synth.renderers.waterfall", "engine": "mpl",
 "seed": 41732, "retry": 0, "style_preset": "corporate_flat", "style_digest": "a3f1…",
 "palette": "corp_navy_steel", "language": "de", "spines": "none", "yaxis": false,
 "data_labels": "value", "render_dpi": 148.0, "severity": 0.42,
 "width": 612, "height": 388, "degrade": {…}, "split": "train",
 "filename": "train/waterfall/…png"}
```

Plus per-renderer `meta` keys (`n_categories`, `sub_kind`, `area_frac`, …).
Photos carry `synthetic: false`, `engine: "corpus"`, `source_pdf`, `aug_index`.
`style_digest` lets you ask, after the first training run, which style axes
produce errors — the feedback loop that turns the generator into a tool rather
than a one-way street.

---

## 10. Performance

`concurrent.futures.ProcessPoolExecutor`, one process per worker, `MPLBACKEND=Agg`,
rendering into an in-memory PNG buffer. Each worker gets a deterministic seed via
`cli.seed_for` (a SHA1 of `base_seed|label|sub_type|index`, salt-free so it is
stable across processes — Python's built-in `hash()` is not). Figures are cleared
after every sample; a leaked-figure memory blow-up across a large run is real.

Observed throughput ≈ 6–10 images/s at 6 workers — the diagram-heavy classes
(`map`, `other`) with many patches are several times slower than a plain bar
chart, so the mean is well below matplotlib's ceiling. A full ~1,650-image
smoke run finishes in a few minutes; an 18,000-image run in well under an hour.

---

## 11. Quality assurance

Two of the three planned stages are built:

1. **Contact sheets** (`--contact-sheet`). An 8×8 montage per class, first 64
   samples sorted and captioned by sub-type, letterboxed so the aspect
   distribution stays visible. Ten minutes of scanning catches most generator
   bugs — every fix made while building the renderers came from these.
2. **`stats.md`** (always written). Class/sub-type balance, the synthetic size
   distribution against the real crops, style-axis coverage (including rule-R4
   axis-lessness), degradation levels, and the invariant retry rate — a rising
   retry rate is the early-warning that a renderer has drifted off-label.

Not built: a **smoke probe** running the pretrained Docling classifier over the
synthetic set (a cheap plausibility check — does it recognize synthetic pie
charts as `pie_chart`), and the decisive step, **the domain-gap measurement**:
train on synthetic only, evaluate on a hand-labeled real validation set, confusion
matrix per class *and* per sub-type. That number is what decides which class needs
generator work next — and it requires the validation set (50–100 real images per
class), which is the outstanding dependency, not generator code.

---

## 12. Implementation status

| Phase | Content | Status |
|---|---|---|
| 0 | scaffolding + `bar_vertical`, degradation dialled in | ✅ |
| 1 | `bar_horizontal`, `bar_stacked`, `line`, `pie_donut` | ✅ |
| 2 | `waterfall`, `combo_bar_line` + hard-variant pool | ✅ |
| 3 | `table`, `logo_icon`, `other` (18 sub-generators) | ✅ |
| 4 | `map` (procedural), `photo` (ingestion pipeline) | ✅ (photo needs a corpus) |
| 5 | R1/R2 compose with real charts, `table_with_bars` + `donut_progress` | ✅ (Plotly deferred) |
| 6 | domain-gap measurement against a real validation set | ⭕ needs the val set |

---

## 13. Running it

Install the generation dependencies (kept out of the inference environment):

```
uv sync --group synth
```

Uniform count across all rendered classes, with real sizes and photos:

```
PYTHONPATH=src .venv/Scripts/python.exe -m DocumentFigureClassifier.synth.cli \
    --n 1500 --out data/synth --workers 8 --contact-sheet \
    --size-manifest data/parsed/manifest.jsonl \
    --photos-src data/parsed/review/photo --photos-exclude <val_report_stems>
```

Before a *complete* run: re-extract the report corpus so `data/parsed/` exists
again — that restores real-size sampling (`--size-manifest`) and provides the
photo crops (`--photos-src`). Two or more reports are needed for a clean photo
train/val split.

Key flags: `--n` (per class), `--classes a,b,c` (subset), `--append` (accumulate,
for per-class counts), `--workers`, `--seed`, `--no-degrade` (debug),
`--regenerate <crop_id> --dump-stages <dir>` (reproduce one sample with stages),
`--photos-src / --photos-per-source / --photos-exclude`, `--contact-sheet /
--sheet-cols`.

### Dependencies

```toml
[dependency-groups]
synth = ["matplotlib>=3.9", "numpy>=2.0"]   # pillow already present
```

Deliberately **not** used: geopandas/GDAL (§6.3), OpenCV (PIL covers the
perspective jitter), any HTML/browser renderer. Plotly + kaleido would be added
only if the deferred second engine (§6.1) is revived.
