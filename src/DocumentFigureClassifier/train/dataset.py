"""
Shared dataset plumbing for training and evaluation.

The split index (produced by split.py) is the single source of truth for which
image belongs to which split; train.py and evaluate.py both load from it via
`load_split`, so they cannot drift apart.
"""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from DocumentFigureClassifier.model import LABEL2ID

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# (image path, label id in TIER1_LABELS order)
Item = tuple[Path, int]


class FigureDataset(Dataset):
    def __init__(self, items: list[Item], transform):
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        path, label_id = self.items[i]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label_id


def load_split(index_path: Path, split: str) -> list[Item]:
    """Read split.py's index.jsonl and return the (path, label_id) items whose
    `split` field matches. Rows with an unknown label are skipped loudly."""
    items: list[Item] = []
    with Path(index_path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("split") != split:
                continue
            label = row["label"]
            if label not in LABEL2ID:
                print(f"warning: skipping row with unknown label {label!r}: {row.get('path')}")
                continue
            items.append((Path(row["path"]), LABEL2ID[label]))
    return items
