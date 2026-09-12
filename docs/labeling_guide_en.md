# Labeling Guide: Chart Types in Annual Reports

**Purpose:** Consistent assignment of class labels to image crops (figures)
extracted automatically from annual-report PDFs. The labels serve as training
and validation data for an image classifier.

**Ground rule:** Each image gets **exactly one** label. When in doubt, the
decision tree in section 2 decides, not your gut.

> English translation of the German original in
> [labeling_guide_de.md](labeling_guide_de.md). The German version is the
> authoritative document for the labeling team; if the two diverge, the German
> one wins.

---

## 1. Class overview

| Label | German | Short definition |
|---|---|---|
| `bar` | Balken-/Säulendiagramm | Plain single-series bars on a shared baseline, vertical or horizontal |
| `bar_grouped` | Gruppierte Säulen/Balken | Several bars per category side by side, one per series |
| `bar_stacked` | Gestapelte Säulen/Balken | Segments stacked on top of each other within one bar |
| `waterfall` | Wasserfall / Brücke | Floating bars between a start and an end value |
| `line` | Liniendiagramm | Values as a connected line |
| `combo_bar_line` | Kombidiagramm | Bars **and** a line in one plot |
| `scatter` | Streudiagramm | Points in an x/y coordinate system, no connecting line |
| `pie_donut` | Kreis-/Ringdiagramm | Circular part-to-whole display |
| `flow` | Flussdiagramm | Nodes joined by **drawn** connectors into a progression or hierarchy |
| `map` | Karte | Geographic representation |
| `table` | Tabelle | Row/column grid without graphical encoding |
| `photo` | Foto | Photographic image |
| `other` | Sonstiges | Logos, seals, icons/pictograms, and anything that fits none of the above |

---

## 2. Decision tree

Work top to bottom, stop at the first match.

```
1.  Is it a photograph (people, buildings, products)?          → photo
2.  Is it a logo, seal, award or pictogram?                    → other
3.  Does the image show charts of MULTIPLE DIFFERENT types?    → Rule R1 (section 4)
4.  Is a geographic map the dominant element?                  → map
5.  Nodes joined by DRAWN connectors into a path/hierarchy?    → flow
    (radial/hub-and-spoke, rings, connector-less stacks        → other)
6.  Are bars AND a line present as data series?                → combo_bar_line
7.  Circular part-to-whole display (full or ring form)?        → pie_donut
8.  Do bars float between a start and an end value (bridge)?   → waterfall
9.  Are the bars divided into stacked segments?                → bar_stacked
10. Multiple bars per category, side by side (grouped)?        → bar_grouped
11. Plain single-series bars (vertical or horizontal)?         → bar
12. Points in an x/y system with no connecting line?           → scatter
13. Is it a line plot (including a filled area below)?         → line
14. A pure row/column grid with no graphics?                   → table
15. Otherwise                                                  → other
```

The order is deliberate: `combo_bar_line` (step 6) comes before
waterfall/stacked/grouped/bar/line, because otherwise a combo chart would get
labeled as a bar or a line chart depending on the viewer's perspective.
`bar_grouped` (step 10) is checked before plain `bar`, and `scatter` (step 12)
before `line`, because in each pair the first otherwise easily passes as the
second.

---

## 3. Classes in detail

### `bar` — bar / column chart

Plain bars of a **single data series**, all starting from the same baseline.
Orientation — vertical (columns) or horizontal (bars) — is **not** distinguished:
it is a pure layout choice with no downstream consequence, and both forms are
parsed the same way.

**Typical:** Revenue by fiscal year, EBIT development, headcount, R&D expense,
capital expenditure; revenue by region, top-10 rankings, workforce age
structure, survey results.

**Also belongs here:**
- Vertical (columns) **and** horizontal (bars) orientation.
- Bars with data labels instead of an axis.
- A single highlighted bar (color accent).

**Does not belong here:**
- Several series shown as adjacent bars per category → `bar_grouped`
- Bars divided into segments → `bar_stacked`
- Bars that do not start at the baseline → `waterfall`
- An additional line as a data series → `combo_bar_line`

### `bar_grouped` — grouped bar / column chart

Several bars per category, placed side by side — one bar per data series, not
stacked. Like `bar`, orientation (vertical or horizontal) is not distinguished.

**Typical:** Prior year vs. reporting year side by side, actual vs. budget,
region comparisons across two or three periods, multi-series KPI comparisons.

**Also belongs here:**
- Two or more series shown as adjacent bars within each category group.
- Vertical (grouped columns) and horizontal (grouped bars) alike.

**Does not belong here:**
- A single series of plain bars → `bar`
- Series stacked within one bar → `bar_stacked`
- A line as an additional series → `combo_bar_line`

**Why separate from `bar`?** Grouped bars are harder to parse: each category holds
several bars that must be assigned to the right series via the legend and colors,
and value labels crowd. Pulling them out gives that harder case its own route —
the same reasoning that keeps `bar_stacked` separate.

### `bar_stacked` — stacked display

Bars are divided into colored segments that build on each other.

**Typical:** Segment shares across several years, revenue composition, workforce
by region, 100 % breakdowns.

**Also belongs here:**
- 100 %-stacked variants (scale ends at 100 %).
- Both vertical and horizontal stacks. Orientation is **not** distinguished
  here.

**Key identifying feature:** All segments of a bar share a common baseline and
sit on top of each other without gaps.

**Why separate from `bar`?** Unlike vertical/horizontal, the stack is kept as its
own class on purpose: it is markedly harder to parse downstream (overlapping
segments, values readable only as a sum) and gets its own processing route.

### `waterfall` — waterfall / bridge chart

Bars "float" between a start and an end value and show a reconciliation.

**Typical:** EBIT bridge, revenue bridge (volume / price / currency / M&A), net
financial debt reconciliation, cash flow reconciliation, year-over-year earnings
change.

**Identifying features — at least two should apply:**
1. The first and last bar sit on the baseline, the ones in between do not.
2. Connector lines between the bar ends.
3. Two-color encoding for increases and decreases (often green/red or
   light/dark).
4. Labels with signs (`+12`, `−8`) on the intermediate bars.
5. Axis labels naming reasons for change instead of categories or years.

**Most common mistake:** Confusion with `bar_stacked`. See rule R3.

### `line` — line chart

Data points connected by a line.

**Typical:** Share price development against DAX/MDAX, multi-year trends,
interest rate development, sales trajectory.

**Also belongs here:**
- Multiple lines in one plot.
- Area charts (line with a filled area below), including stacked ones. Merged
  in for now — if the volume warrants it, split off later as `area`.
- Lines with point markers.

**Note — count data series, not strokes.** Gridlines, axis lines, target /
reference lines, read-off guides and callout leaders are chart furniture and do
not change the type. A chart is `line` if it has ≥1 data series drawn as a
connected line in an x/y system, however many guide strokes surround it (e.g. a
remuneration payout curve with an Ist-Wert marker → `line`).

**Does not belong here:** A line together with bars → `combo_bar_line`.

### `combo_bar_line` — combo chart

Bars and a line as **substantive data series** in one plot, usually with a
secondary axis.

**Typical:** Revenue (columns) plus EBIT margin in % (line); volume plus average
price; capex plus capex ratio.

**Does not belong here:**
- A target or average line over a bar chart. A reference line is not a data
  series → stays `bar_vertical` / `bar_horizontal`.
- Connector lines in a bridge → `waterfall`.

**Rule of thumb:** Does the line **vary across the categories**? Then it is a
data series → `combo_bar_line`. A line at a constant value is a reference line
and does not change the class — **even when it has its own legend entry**.

The legend is not the test. Target and average lines routinely appear in the
legend ("Ziel 2030", "Ø"), which is why the earlier wording ("own axis or own
legend entry") contradicted the rule that a target line is not a data series.
Where the two signals disagree, *does it vary* wins.

### `scatter` — scatter plot

Data points in an x/y coordinate system, **without** a line connecting the
points.

**Typical:** Risk-return scatters, correlation displays, positioning and
portfolio charts with two value axes, bubble charts (points with an added size
encoding).

**Does not belong here:**
- Points connected by a line → `line`.
- Points on a map instead of an axis system → `map`.

**Note:** Scatter is not parsed for content for now. The class exists to
reliably separate scatter plots from `line` — the two look alike (axes, point
markers), and a mix-up would send a line chart off to be parsed by mistake.

### `pie_donut` — pie and donut chart

Shares as circular segments.

**Typical:** Shareholder structure, revenue by region or division, workforce
structure, free float.

**Also belongs here:**
- Donut charts, including those with a number in the center.
- Semicircle and gauge-style displays.
- Circles with an exploded segment.

Full and ring form are **not** distinguished: identical semantics, and corporate
design switches between them arbitrarily.

### `map` — map

Geographic representation as the dominant element.

**Typical:** Site overview, production network, sales regions, revenue by country
as a choropleth.

**Also here:** Maps with overlaid numbers, pins or small bars — as long as the
map dominates the visual impression.

### `flow` — flow chart

Discrete nodes joined by **drawn** connectors into a directed progression or a
hierarchy.

**Two conditions must both hold:**

1. **Explicit, drawn structure.** Either arrows/lines are visible *between* the
   elements, or the element shapes themselves encode the progression —
   interlocking chevrons, a segmented arrow, narrowing funnel stages. Plain
   rectangles that merely sit next to or above one another do **not** qualify.
2. **Direction or hierarchy.** The structure describes a progression (start →
   end) or a parent → child tree. You can follow a path through the diagram.

3. **The layout is linear or branching — never circular.** See below.

**Typical:** Process and workflow diagrams, decision trees, org and governance
charts *with drawn reporting lines*, value chains, swimlane diagrams, funnels.

**Also belongs here:**
- Horizontal as well as vertical flow direction.
- Nodes with symbols or icons inside them — as long as nodes and connectors
  form the structure.

**Does not belong here:**
- **Anything circular → `other`. `flow` is never round.** This is an absolute
  rule, applied before every other test, so that "circular" is a reliable
  signal for *not* `flow`. It covers:
  - **Radial / hub-and-spoke diagrams** — a central element with items arranged
    around it, even when spokes are drawn. There is no progression; the relation
    is "belongs to the centre", not "precedes" or "reports to".
  - **Ring, wheel and segmented-circle graphics** where the circle itself is the
    graphic. Compare with `pie_donut` if it encodes proportions.
  - **Process cycles** — boxes arranged in a loop with arrows (A → B → C → A),
    even though they do show a progression. They are not a parsing target, and
    excluding them keeps the rule free of exceptions.
- **Connector-less stacked or layered boxes → `other`.** Org, governance and
  compliance charts drawn as stacked colour bands or abutting boxes with no
  drawn lines between them. Also layered "temple"/framework diagrams and
  pyramids. If the only cue for hierarchy is that one box sits above or spans
  another, it is **not** `flow`.
- A set of equal peer boxes with no hierarchy or sequence — a styled list or
  icon grid where the only relation is "same category" → `other` (or `table`
  if it is a real grid).
- Sankey diagrams (flows with proportional width) → `other` for now.
- Timelines/roadmaps without connecting flow logic → `other`. Test: remove the
  time axis — if it collapses to "events in date order" it is a timeline
  (`other`); if a relational structure survives (a phase gates/enables the next,
  branches) it is `flow`.

**Quick test:** is it round? → `other`. Otherwise cover the connectors with your
hand: if you can no longer tell what leads to what, it is `flow`; if the layout
alone still carries the whole message, it is `other`.

**Note:** The base model (DocumentFigureClassifier) already knows flow charts, so
fewer real training examples are needed than for a brand-new class. The class is
intended as a future parsing target.

### `table` — table

A pure row/column grid without graphical encoding of values.

**Typical:** Key-figure overview, multi-year comparison, or a qualitative
grid (ESG/target table, governance principles), present as an image rather
than as text in the PDF. Cells may hold **numbers or text** — content does
not decide the class, structure does.

**Note — merged cells and messy structure still count.** The test is
"does a reader see a row/column grid?", not "would this serialize cleanly to
a grid?". Merged/row-spanning cells, empty cells, subtotal or total rows, and
narrative cells spanning several rows are all normal table constructs → still
`table`. Clean parseability is the downstream extractor's job and is
deliberately not encoded in the label.

**Does not belong here:** Tables with embedded bars, traffic lights or
sparklines — label these as `other` and note it in the notes column. Same for
colour-coded grids like risk matrices / heatmaps (cells convey the value by
colour, not text) → `other`.

### `photo` — photograph

**Typical:** Board portraits, plant shots, product images, employee photos,
facilities, mood shots.

This class matters more than it looks: annual reports consist visually to a
substantial degree of photography, and the model must learn to reject it
reliably.

### `other` — miscellaneous (incl. logos and pictograms)

Catch-all class. Not optional: without it the model produces confident false
statements for everything it does not know.

**Typical:**
- Group and brand logos, certification seals (ISO, FSC), awards, rating symbols,
  icon sets in sustainability chapters, SDG tiles, pictograms
- Timelines, milestones, roadmaps
- Materiality and risk matrices
- KPI tiles (large number + label + arrow)
- Progress bars, gauges, traffic lights
- Sankey, radar, tornado
- **Radial and hub-and-spoke diagrams** — a central element with items arranged
  around it, with or without drawn spokes
- **Ring, wheel and segmented-circle graphics** where the circle itself is the
  graphic (strategy wheels, cycle rings with abutting segments)
- **Process cycles** — boxes in a loop joined by arrows (`flow` is never round)
- **Connector-less stacked or layered box diagrams** — governance/compliance
  charts drawn as stacked colour bands, layered "temple"/framework diagrams,
  pyramids
- Decorative graphics, separators, extraction artifacts, empty crops

**Boundary:** An icon or logo **inside** a chart does not make the image
`other` — what counts is the image content as a whole.

Bubble charts go to `scatter` — no longer here. (Logos and pictograms,
previously their own `logo_icon` class, now belong here.) Flow diagrams go to
`flow` **only when drawn connectors form a path or hierarchy** — see the `flow`
section; radial, ring and connector-less box diagrams stay here.

If one of these subcategories occurs frequently, note it in the notes field.
Past a certain volume it is worth its own class (see section 5).

---

## 4. Special rules

### R1 — Multiple charts of different types in one image

An extracted crop contains several charts of **different types** side by side
(common in key-figure chapters) — e.g. a donut next to a bar chart. No single
label is correct, and the charts route to different downstream parsers.

**Rule:** Send the image to post-processing and cut it into individual charts.
Each crop gets its own label.

**If cutting is not possible** (charts overlap, shared legend, shared axis):
label as `other` and note `multi` in the notes field.

**Not counted as multiple charts — label as the single chart class, keep the
crop whole:**
- Several charts of the **same type** (e.g. two donuts, three bar charts). The
  label is unambiguous (all donuts → `pie_donut`), so keep the crop whole and
  label it by that type. Splitting them into separate instances is the parser's
  job, not the classifier's — and the crop reaches the classifier whole at
  inference (no splitter runs before it), so it must be trained and validated
  whole, or you build in a train/inference mismatch.
- One plot with several data series.
- A chart with a separate legend.

**Principle:** Split by routing target, not by chart instance. All charts route
to the same class → one label, keep the crop whole. They route to different
classes → split (or `other` + `multi` if not cleanly cuttable).

### R2 — Infographic frames

A chart is embedded in a designed layout: headline, body text, icons, color
areas around it.

**Rule:** If the chart occupies **more than roughly half** the image area →
chart class. Otherwise → `other`.

This rule is deliberately coarse. What matters is not exactly where the boundary
sits, but that everyone applies the same one.

### R3 — `waterfall` versus `bar_stacked`

The hardest distinction; this is where most errors come from.

| Feature | `waterfall` | `bar_stacked` |
|---|---|---|
| Baseline | Only first and last bar | All bars |
| Segments | One segment per bar, floating | Several segments per bar, stacked |
| Colors | By direction (increase/decrease) | By category |
| X-axis | Reasons for change | Years or categories |
| Connector lines | Often present | Never |
| Signs in labels | Frequent | Never |

**Mnemonic:** A bridge tells you *how* you get from A to B. A stack shows you
*what* a value is made of.

If genuinely uncertain: put the image in `_unsure` and discuss it in review. Do
not guess — wrong labels in exactly this class pair do the most damage.

### R4 — Charts without axes

Corporate design frequently drops the y-axis entirely and writes values directly
on the bars.

**Rule:** Such images are labeled by chart type as normal. Missing axes are
**not** a reason for `other`.

These cases must be well represented in the training set, otherwise the model
learns axis geometry as a feature — which is often absent in real annual
reports.

### R5 — Uncertainty

There is an `_unsure` folder. Anyone who has not reached a decision after 15
seconds puts the image there and moves on. These images are decided collectively
in review.

`_unsure` is not a class and does not go into training. A growing `_unsure`
folder is a signal that this guide needs a rule.

### R6 — Bad crops

Truncated, blurry, empty or misaligned extractions go to `_broken`. Not for
training, but keep them: they reveal errors in the extraction pipeline.

---

## 5. When a new class is warranted

A class is justified when **both** conditions hold:

1. **Downstream benefit:** Some processing step treats it differently from the
   existing classes. This is exactly why `bar_vertical` and `bar_horizontal` were
   merged into `bar` — they are processed identically downstream, so the split
   was unnecessary. `bar_grouped` and `bar_stacked`, by contrast, stay separate
   because both are harder to parse than a plain single-series bar.
2. **Sufficient volume:** At least roughly 100 real examples are findable. Below
   that the model does not learn the class but does dilute its neighbors.
   Exception: for classes the base model already knows (`scatter`, `flow`),
   fewer real examples suffice, because the model only sharpens an existing
   representation rather than learning one from scratch — a trustworthy
   validation set is still mandatory.

Candidates from the `other` notes column, once volume suffices: `timeline`,
`matrix`, `kpi_tile`, `area`, `gauge_progress`.

---

## 6. Working method

**Folder structure.** The pre-classification script already places the crops into
folders with proposed labels. Correcting a label means moving the file to the
right folder. Not moving it means the proposal is confirmed.

```
review/
├── bar/
├── bar_grouped/
├── bar_stacked/
├── waterfall/
├── line/
├── combo_bar_line/
├── scatter/
├── pie_donut/
├── flow/
├── map/
├── table/
├── photo/
├── other/
├── _unsure/      → not for training
└── _broken/      → not for training
```

**Order of work.** Work through `_review` (low model confidence) first, then
cross-check the remaining folders. That is where the largest information gain
per minute sits.

**Beware confirmation drift.** Proposed labels are convenient and tempt you to
wave them through. `waterfall` in particular is systematically proposed as
`bar_chart` by the base model — that folder needs the most thorough review.

**Validation set.** At least 50 to 100 **real** images per class, hand-labeled,
never synthetic, never from the training set. Without that, any measured
accuracy is worthless.

**Quality assurance.** Have roughly 10 % of the images labeled independently by a
second person. Agreement below roughly 90 % means this guide is unclear
somewhere — sharpen it, do not admonish the labelers.

---

## 7. Change history

| Version | Date | Change |
|---|---|---|
| 1.0 | — | First edition: 12 tier-1 classes, rules R1–R6 |
| 1.1 | 2026-08-23 | `combo_bar_line` sharpened: the test is whether the line varies across categories, not whether it has a legend entry. Resolves a contradiction with the target-line rule. |
| 1.2 | 2026-08-26 | Merged `bar_vertical` + `bar_horizontal` into `bar` (parsed identically downstream). Split grouped bars into their own class `bar_grouped`, and kept `bar_stacked` separate — both harder to parse. New classes `scatter` (to separate it from `line`) and `flow` (process/workflow/org charts, future parsing target). Now 14 tier-1 classes. |
| 1.3 | 2026-08-31 | R1 scoped to charts of **different** types. Several charts of the **same** type (e.g. two donuts) are now labeled as that single class and kept whole — the label is unambiguous and the crop reaches the classifier whole at inference; instance-splitting is the parser's job. Decision-tree step 3 reworded accordingly. |
| 1.4 | 2026-09-09 | `logo_icon` merged into `other`. Now 13 tier-1 classes. |
| 1.5 | 2026-09-12 | **`flow` tightened — breaking change, requires relabeling.** `flow` now requires explicit drawn structure (connectors between elements, or directional shapes such as chevrons and funnel stages) *and* a direction/hierarchy, *and* a non-circular layout. Moved to `other`: radial and hub-and-spoke diagrams (even with drawn spokes), ring/wheel/segmented-circle graphics, **process cycles**, and connector-less stacked or layered box diagrams (governance colour bands, temple/framework layers, pyramids). **This reverses the previous rule** that hierarchy expressed purely through layout counted as a connection. "`flow` is never round" is now absolute and exception-free, which also gives the classifier a clean discriminative feature. Motivation: on the held-out test set `flow` recall was 42.5% with 50.9% of flow leaking to `other`, while `other` leaked back at only 1.6% — and the same visual families (governance wheels, org colour bands) were found labeled on both sides, i.e. the old definition was not consistently applicable. |
