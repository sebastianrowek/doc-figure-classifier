"""
Output in ImageFolder layout plus a manifest.

The split guard is not decoration. The labeling guide is explicit that
validation needs real, hand-labeled images and that synthetic data never
belongs there -- and it is exactly the kind of rule that gets violated at 2am
by someone passing ``--out data/val`` to save a step. Make it impossible
instead of documenting it.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

from PIL import Image

from ..taxonomy import TIER1_LABELS

ALLOWED_SPLITS = ("train",)


class SplitError(ValueError):
    pass


class Writer:
    """Writes ``<root>/<split>/<label>/<sample_id>.png`` and one manifest line each."""

    def __init__(self, root: Path, split: str = "train", append: bool = False) -> None:
        if split not in ALLOWED_SPLITS:
            raise SplitError(
                f"refusing to write split {split!r}. Synthetic data is training-only; "
                "validation and test must be real hand-labeled crops "
                "(labeling guide, section 6)."
            )
        self.root = Path(root)
        self.split = split
        self.split_dir = self.root / split
        self.manifest_path = self.root / "manifest.jsonl"
        self._fh = None
        self._append = append
        self.counts: dict[str, int] = {}

    def __enter__(self) -> Writer:
        for label in TIER1_LABELS:
            (self.split_dir / label).mkdir(parents=True, exist_ok=True)
        self._fh = self.manifest_path.open("a" if self._append else "w", encoding="utf-8")
        return self

    def __exit__(self, *exc) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def save(self, image: Image.Image, record: dict) -> Path:
        buf = io.BytesIO()
        image.save(buf, format="PNG", optimize=False)
        return self.save_png_bytes(buf.getvalue(), record)

    def save_png_bytes(self, data: bytes, record: dict) -> Path:
        """
        Workers hand back encoded PNGs rather than PIL images: it avoids a
        pickle round trip of the pixel buffer and a pointless re-encode here.
        """
        label = record["label"]
        if label not in TIER1_LABELS:
            raise ValueError(f"unknown label {label!r}")

        path = self.split_dir / label / f"{record['crop_id']}.png"
        path.write_bytes(data)

        record = {**record, "split": self.split, "filename": str(path.relative_to(self.root))}
        assert self._fh is not None, "use Writer as a context manager"
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.counts[label] = self.counts.get(label, 0) + 1
        return path

    def summary(self) -> str:
        total = sum(self.counts.values())
        lines = [f"{lbl:16s} {n:6d}  {100 * n / total:5.1f}%"
                 for lbl, n in sorted(self.counts.items(), key=lambda kv: -kv[1])]
        lines.append(f"{'TOTAL':16s} {total:6d}")
        return "\n".join(lines)
