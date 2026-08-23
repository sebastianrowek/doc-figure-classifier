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
| `bar_vertical` | Säulendiagramm | Vertical bars on a shared baseline |
| `bar_horizontal` | Balkendiagramm | Horizontal bars on a shared baseline |
| `bar_stacked` | Gestapelte Säulen/Balken | Segments stacked on top of each other within one bar |
| `waterfall` | Wasserfall / Brücke | Floating bars between a start and an end value |
| `line` | Liniendiagramm | Values as a connected line |
| `combo_bar_line` | Kombidiagramm | Bars **and** a line in one plot |
| `pie_donut` | Kreis-/Ringdiagramm | Circular part-to-whole display |
| `map` | Karte | Geographic representation |
| `table` | Tabelle | Row/column grid without graphical encoding |
| `photo` | Foto | Photographic image |
| `logo_icon` | Logo / Piktogramm | Brand, seal, icon, award |
| `other` | Sonstiges | Anything that fits none of the above |

---

## 2. Decision tree

Work top to bottom, stop at the first match.

```
1.  Is it a photograph (people, buildings, products)?          → photo
2.  Is it a logo, seal, award or pictogram?                    → logo_icon
3.  Does the image show MULTIPLE independent charts?           → Rule R1 (section 4)
4.  Is a geographic map the dominant element?                  → map
5.  Are bars AND a line present as data series?                → combo_bar_line
6.  Circular part-to-whole display (full or ring form)?        → pie_donut
7.  Do bars float between a start and an end value (bridge)?   → waterfall
8.  Are the bars divided into segments (stacked)?              → bar_stacked
9.  Are they plain bars?
       vertical → bar_vertical       horizontal → bar_horizontal
10. Is it a line plot (including a filled area below)?         → line
11. A pure row/column grid with no graphics?                   → table
12. Otherwise                                                  → other
```

The order is deliberate: step 5 comes before 7/8/9, because otherwise a combo
chart would get labeled as a bar or a line chart depending on the viewer's
perspective.

---

## 3. Classes in detail

### `bar_vertical` — column chart

Vertical bars, all starting from the same baseline.

**Typical:** Revenue by fiscal year, EBIT development, headcount, R&D expense,
capital expenditure.

**Also belongs here:**
- Grouped columns (several series side by side, e.g. prior year / reporting
  year). Grouping is a property of the data, not a separate chart type.
- Columns with data labels instead of a y-axis.
- A single highlighted column (color accent).

**Does not belong here:**
- Bars divided into segments → `bar_stacked`
- Bars that do not start at the baseline → `waterfall`
- An additional line as a data series → `combo_bar_line`

### `bar_horizontal` — bar chart

Same as above, but horizontal.

**Typical:** Revenue by region, top-10 rankings, workforce age structure, survey
results.

**Boundary note:** Orientation decides, not semantics. A revenue-by-region chart
can fall into `bar_vertical` or `bar_horizontal` depending on layout — both are
correct.

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

### `table` — table

A pure row/column grid without graphical encoding of values.

**Typical:** Key-figure overview, multi-year comparison, present as an image
rather than as text in the PDF.

**Does not belong here:** Tables with embedded bars, traffic lights or
sparklines — label these as `other` and note it in the notes column.

### `photo` — photograph

**Typical:** Board portraits, plant shots, product images, employee photos,
facilities, mood shots.

This class matters more than it looks: annual reports consist visually to a
substantial degree of photography, and the model must learn to reject it
reliably.

### `logo_icon` — logo and pictogram

**Typical:** Group and brand logos, certification seals (ISO, FSC), awards,
rating symbols, icon sets in sustainability chapters, SDG tiles.

**Boundary:** An icon **inside** a chart does not make the image `logo_icon` —
what counts is the image content as a whole.

### `other` — miscellaneous

Catch-all class. Not optional: without it the model produces confident false
statements for everything it does not know.

**Typical:**
- Org charts, process and flow diagrams, value chains
- Timelines, milestones, roadmaps
- Materiality and risk matrices
- KPI tiles (large number + label + arrow)
- Progress bars, gauges, traffic lights
- Sankey, radar, bubble, tornado
- Decorative graphics, separators, extraction artifacts, empty crops

If one of these subcategories occurs frequently, note it in the notes field.
Past a certain volume it is worth its own class (see section 5).

---

## 4. Special rules

### R1 — Multiple charts in one image

An extracted crop contains several independent charts side by side (common in
key-figure chapters).

**Rule:** Send the image to post-processing and cut it into individual charts.
Each crop gets its own label.

**If cutting is not possible** (charts overlap, shared legend, shared axis):
label as `other` and note `multi` in the notes field.

**Not counted as multiple charts:** one plot with several data series; a chart
with a separate legend.

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
   existing classes. If `bar_vertical` and `bar_horizontal` are processed
   identically downstream, the split was unnecessary.
2. **Sufficient volume:** At least roughly 100 real examples are findable. Below
   that the model does not learn the class but does dilute its neighbors.

Candidates from the `other` notes column, once volume suffices: `org_chart`,
`process_flow`, `timeline`, `matrix`, `kpi_tile`, `area`, `gauge_progress`.

---

## 6. Working method

**Folder structure.** The pre-classification script already places the crops into
folders with proposed labels. Correcting a label means moving the file to the
right folder. Not moving it means the proposal is confirmed.

```
review/
├── bar_vertical/
├── bar_horizontal/
├── bar_stacked/
├── waterfall/
├── line/
├── combo_bar_line/
├── pie_donut/
├── map/
├── table/
├── photo/
├── logo_icon/
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
