"""
The figure classifier: docling's EfficientNet backbone re-fitted from its
original 26-class head to our 14 Tier-1 labels (taxonomy.TIER1_LABELS).

Two entry points:
  build_model()     -> a model ready to fine-tune. The pretrained backbone is
                       loaded and its classifier head is thrown away and
                       reinitialised for 14 classes.
  FigureClassifier  -> load a fine-tuned checkpoint and predict on PIL images.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
import torchvision.transforms as T
from PIL import Image
from transformers import EfficientNetForImageClassification

from DocumentFigureClassifier.taxonomy import TIER1_LABELS

# The checkpoint we inherit the backbone from -- the same one the extraction
# pipeline uses for zero-shot. We keep its 26-class head only long enough to
# discard it (see build_model).
PRETRAINED_BACKBONE = "docling-project/DocumentFigureClassifier-v2.5"

# Normalisation from the model card. The std values are deliberately not the
# ImageNet ones -- they match how the backbone was trained, so we keep them for
# fine-tuning and inference. (Same constants as extract_and_classify.PREPROCESS.)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.47853944, 0.4732864, 0.47434163]
IMAGE_SIZE = 224

# Fixed, taxonomy-ordered label maps so the head index of every class is stable
# regardless of which classes happen to be present in a given training set.
ID2LABEL = {i: label for i, label in enumerate(TIER1_LABELS)}
LABEL2ID = {label: i for i, label in enumerate(TIER1_LABELS)}


def make_transform(train: bool = False) -> T.Compose:
    """Preprocessing pipeline.

    Eval is resize + normalise, matching the pretrained model card. Train adds
    only mild, chart-safe augmentation: no flips or random crops, which could
    turn one chart type into another (a flipped horizontal bar, a cropped-away
    legend). A couple of degrees of rotation and light colour jitter mirror what
    Docling's bounding boxes and PDF rendering already do to real crops.
    """
    steps: list = [T.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if train:
        steps += [
            T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
            T.RandomRotation(2),
        ]
    steps += [
        T.ToTensor(),
        T.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ]
    return T.Compose(steps)


def build_model(pretrained: str = PRETRAINED_BACKBONE) -> EfficientNetForImageClassification:
    """Load the pretrained backbone and swap in a fresh 14-way classifier head.

    `ignore_mismatched_sizes=True` is what does the replacement: every backbone
    weight is loaded, and only the classifier layer -- whose shape no longer
    matches (26 -> 14) -- is dropped and reinitialised. The new head starts from
    random weights and is what training fits.
    """
    return EfficientNetForImageClassification.from_pretrained(
        pretrained,
        num_labels=len(TIER1_LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )


def set_backbone_trainable(model: EfficientNetForImageClassification, trainable: bool) -> None:
    """Freeze or unfreeze the pretrained backbone (``model.efficientnet``).

    The classifier head (``model.classifier``) is never touched here -- it always
    trains. Freezing the backbone for the first epoch lets the freshly random
    head settle before its large early gradients are allowed to flow back and
    perturb the good pretrained features (catastrophic forgetting)."""
    for p in model.efficientnet.parameters():
        p.requires_grad = trainable


def param_groups(
    model: EfficientNetForImageClassification, backbone_lr: float, head_lr: float
) -> list[dict]:
    """Two AdamW parameter groups: a low LR for the pretrained backbone and a
    higher LR for the freshly initialised head. Both groups are present for the
    whole run; while the backbone is frozen its params simply receive no gradient
    and are skipped by the optimizer, so no optimizer rebuild is needed when it
    unfreezes."""
    return [
        {"params": list(model.efficientnet.parameters()), "lr": backbone_lr},
        {"params": list(model.classifier.parameters()), "lr": head_lr},
    ]


class FigureClassifier:
    """Inference wrapper around a fine-tuned 14-class checkpoint."""

    def __init__(self, model_dir: str | Path, device: str = "cpu", batch_size: int = 16):
        model_dir = Path(model_dir)
        self.model = EfficientNetForImageClassification.from_pretrained(
            str(model_dir), local_files_only=model_dir.is_dir()
        )
        self.model.eval().to(device)
        # Trust the checkpoint's own label map (written by save_pretrained).
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.device = device
        self.batch_size = batch_size
        self.transform = make_transform(train=False)

    @torch.no_grad()
    def predict_proba(self, images: Sequence[Image.Image]) -> torch.Tensor:
        """Full softmax distribution over the 14 classes, shape (N, 14).

        Use this for calibration / loss / confusion analysis -- it keeps the
        whole distribution, not just the winning class.
        """
        out: list[torch.Tensor] = []
        for i in range(0, len(images), self.batch_size):
            chunk = [im.convert("RGB") for im in images[i : i + self.batch_size]]
            batch = torch.stack([self.transform(im) for im in chunk]).to(self.device)
            out.append(self.model(batch).logits.softmax(dim=1).cpu())
        return torch.cat(out) if out else torch.empty(0, len(self.id2label))

    def predict(self, images: Sequence[Image.Image]) -> list[tuple[str, float]]:
        """(label, confidence) for the top class of each image."""
        probs = self.predict_proba(images)
        conf, idx = probs.max(dim=1)
        return [(self.id2label[int(j)], float(c)) for j, c in zip(idx.tolist(), conf.tolist())]
