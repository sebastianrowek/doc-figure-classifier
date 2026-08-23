"""
Dataset statistics.

Two questions this answers that a contact sheet cannot:

* Does the synthetic size distribution actually match the real crops? That is
  the whole premise of sizes.py, and it is worth verifying rather than
  assuming.
* Did any style axis collapse? Randomisation that silently produces 95 %
  ``classic_axes`` is worse than no randomisation, because it looks fine on a
  contact sheet.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


def _load(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _q(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    v = sorted(values)
    return v[int(p * (len(v) - 1))]


def _dist_table(name: str, synth: list[float], real: list[float] | None) -> list[str]:
    rows = [f"| {name} | p05 | p25 | median | p75 | p95 |", "|---|---|---|---|---|---|"]
    rows.append(
        f"| synthetic | {_q(synth, .05):.0f} | {_q(synth, .25):.0f} | {_q(synth, .5):.0f} "
        f"| {_q(synth, .75):.0f} | {_q(synth, .95):.0f} |"
    )
    if real:
        rows.append(
            f"| real | {_q(real, .05):.0f} | {_q(real, .25):.0f} | {_q(real, .5):.0f} "
            f"| {_q(real, .75):.0f} | {_q(real, .95):.0f} |"
        )
    return rows


def _counter_block(title: str, counts: Counter, total: int) -> list[str]:
    rows = [f"**{title}**", "", "| value | n | share |", "|---|---|---|"]
    for key, n in counts.most_common():
        rows.append(f"| `{key}` | {n} | {100 * n / total:.1f} % |")
    rows.append("")
    return rows


def build(synth_manifest: Path, real_manifest: Path | None, out: Path | None = None) -> str:
    recs = _load(synth_manifest)
    if not recs:
        raise ValueError(f"{synth_manifest} is empty")
    n = len(recs)

    real = _load(real_manifest) if real_manifest and real_manifest.is_file() else []

    lines = [f"# Synthetic dataset statistics", "", f"{n} samples from `{synth_manifest}`", ""]

    lines += ["## Class and sub-type", ""]
    lines += _counter_block("label", Counter(r["label"] for r in recs), n)
    lines += _counter_block(
        "sub_type", Counter(f'{r["label"]}/{r["sub_type"]}' for r in recs), n
    )

    lines += ["## Geometry vs. real crops", ""]
    lines += _dist_table("width (px)", [r["width"] for r in recs], [r["width"] for r in real])
    lines += [""]
    lines += _dist_table("height (px)", [r["height"] for r in recs], [r["height"] for r in real])
    lines += [""]
    lines += _dist_table(
        "aspect x100",
        [100 * r["width"] / r["height"] for r in recs],
        [100 * r["width"] / r["height"] for r in real],
    )
    lines += ["", "Aspect is shown x100 to keep the table integer-formatted.", ""]

    lines += ["## Style axes", ""]
    for field in ("style_preset", "palette", "language", "spines", "data_labels", "legend"):
        values = [r.get(field) for r in recs if field in r]
        if values:
            key = field
            if field == "palette":
                # Palette names are per-seed; the family is the interesting axis.
                values = [str(v).split("_")[0] for v in values]
                key = "palette family"
            lines += _counter_block(key, Counter(values), len(values))

    lines += _counter_block("yaxis visible", Counter(str(r.get("yaxis")) for r in recs), n)
    lines += _counter_block(
        "tick rotation", Counter(str(r.get("tick_rotation", 0.0)) for r in recs), n
    )

    axisless = sum(1 for r in recs if r.get("spines") == "none")
    no_yaxis = sum(1 for r in recs if r.get("yaxis") is False)
    lines += [
        "**Rule R4 coverage**",
        "",
        f"- no spines: {axisless} ({100 * axisless / n:.1f} %)",
        f"- no y-axis: {no_yaxis} ({100 * no_yaxis / n:.1f} %)",
        "",
        "Both should stay well represented; the guide warns that missing axes are "
        "common in real reports and a model that never sees them learns axis "
        "geometry as a feature.",
        "",
    ]

    lines += ["## Degradation", ""]
    sev = [r["severity"] for r in recs]
    lines += _dist_table("severity x100", [100 * s for s in sev], None)
    clean = sum(1 for s in sev if s < 0.10)
    deg = [r["degrade"] for r in recs if r.get("degrade")]
    if deg:
        clipped = sum(1 for d in deg if d.get("hard_clip_side"))
        twopass = sum(1 for d in deg if d.get("jpeg_passes", 0) >= 2)
        grey = sum(1 for d in deg if d.get("greyscale"))
        persp = sum(1 for d in deg if d.get("perspective"))
        lines += [
            "",
            f"- near-clean (severity < 0.10): {clean} ({100 * clean / n:.1f} %)",
            f"- hard-clipped on one side: {clipped} ({100 * clipped / len(deg):.1f} %)",
            f"- double JPEG pass: {twopass} ({100 * twopass / len(deg):.1f} %)",
            f"- greyscale: {grey} ({100 * grey / len(deg):.1f} %)",
            f"- perspective warp: {persp} ({100 * persp / len(deg):.1f} %)",
            "",
            "| jpeg quality | p05 | p25 | median | p75 | p95 |",
            "|---|---|---|---|---|---|",
        ]
        q = [d["jpeg_quality"] for d in deg if d.get("jpeg_quality")]
        lines.append(
            f"| all | {_q(q, .05):.0f} | {_q(q, .25):.0f} | {_q(q, .5):.0f} "
            f"| {_q(q, .75):.0f} | {_q(q, .95):.0f} |"
        )
        lines.append("")

    retries = sum(1 for r in recs if r.get("retry", 0) > 0)
    lines += [
        "## Invariants",
        "",
        f"- samples that needed a redraw: {retries} ({100 * retries / n:.1f} %)",
        "",
        "A rising retry rate means a renderer's parameter ranges have drifted "
        "into producing images that no longer match their label.",
        "",
    ]

    text = "\n".join(lines)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    import sys

    synth = Path(sys.argv[1] if len(sys.argv) > 1 else "data/synth/manifest.jsonl")
    realm = Path(sys.argv[2] if len(sys.argv) > 2 else "data/parsed/manifest.jsonl")
    print(build(synth, realm, synth.parent / "qa" / "stats.md"))
