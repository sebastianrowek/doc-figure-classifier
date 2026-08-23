"""
PDF / screenshot degradation.

A crisp matplotlib render and a Docling crop out of a real PDF do not look
alike, and the difference is not subtle at 224x224. This module closes that
gap.

The stage order is not arbitrary -- each step has to operate on the output of
the previous one. Blurring before downsampling is not the same as blurring
after, and adding noise before JPEG gets quantised away rather than surviving
as noise. Two decisions worth calling out:

* The result is saved as PNG, because real crops are PNGs. The JPEG artefacts
  belong *inside* the pixels, not in the container.

* Roughly 10 % of samples come out nearly clean. Vector charts rendered from a
  PDF at 144 dpi are often razor sharp; a model that only ever saw mush loses
  precisely those.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from .palettes import hex_to_rgb
from .rng import Rng


@dataclass
class DegradeParams:
    """What was applied, for the manifest and for error analysis later."""

    severity: float
    rotation_deg: float = 0.0
    perspective: bool = False
    crop_inset: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    hard_clip_side: str | None = None
    resample: str = "lanczos"
    blur_sigma: float = 0.0
    unsharp: bool = False
    jpeg_quality: int | None = None
    jpeg_passes: int = 0
    noise_sigma: float = 0.0
    brightness: float = 1.0
    contrast: float = 1.0
    saturation: float = 1.0
    gamma: float = 1.0
    greyscale: bool = False
    scan_tint: bool = False

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["crop_inset"] = [round(v, 4) for v in self.crop_inset]
        for k in ("severity", "rotation_deg", "blur_sigma", "noise_sigma",
                  "brightness", "contrast", "saturation", "gamma"):
            d[k] = round(d[k], 4)
        return d


def draw_severity(rng: Rng) -> float:
    """
    Beta(2, 4.2) -- weighted towards light degradation, with a tail.

    Mean is about 0.32. Roughly 10 % of samples land below 0.10 and come out
    essentially clean, which is deliberate (see the module docstring).
    """
    return rng.beta(2.0, 4.2)


def apply(
    img: Image.Image,
    target_wh: tuple[int, int],
    severity: float,
    rng: Rng,
    background: str = "#ffffff",
) -> tuple[Image.Image, DegradeParams]:
    """
    Degrade an oversampled render down to its final target size.

    ``img`` is expected at roughly ``target * oversample``; ``background`` is
    the chart's own background colour, used to fill the corners that rotation
    exposes before the crop removes them.
    """
    p = DegradeParams(severity=severity)
    s = severity
    bg = hex_to_rgb(background)

    img = _geometry(img, p, s, rng, bg)
    img = _resize(img, target_wh, p, rng)
    img = _blur(img, p, s, rng)
    img = _jpeg(img, p, s, rng)
    img = _noise(img, p, s, rng)
    img = _tone(img, p, s, rng)
    return img, p


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------


def _geometry(img: Image.Image, p: DegradeParams, s: float, rng: Rng, bg) -> Image.Image:
    """Rotation, slight perspective, and the crop jitter that mimics bad bboxes."""
    w, h = img.size

    angle = float(np.clip(rng.normal(0.0, 0.30 * s), -0.6, 0.6))
    if abs(angle) > 0.02:
        img = img.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor=bg)
        p.rotation_deg = angle

    if rng.chance(0.15 * s):
        # A photographed or re-scanned page is never perfectly rectangular.
        d = 0.008 * w
        quad = (
            rng.uniform(0, d), rng.uniform(0, d),  # nw
            rng.uniform(0, d), h - rng.uniform(0, d),  # sw
            w - rng.uniform(0, d), h - rng.uniform(0, d),  # se
            w - rng.uniform(0, d), rng.uniform(0, d),  # ne
        )
        img = img.transform((w, h), Image.Transform.QUAD, quad, resample=Image.Resampling.BICUBIC)
        p.perspective = True

    # Crop insets per side. The base inset also removes the background wedges
    # that rotation just introduced at the corners.
    #
    # Kept small on purpose: a few percent off every side sounds harmless, but
    # applied to every image it shaves the first letter off most titles, and a
    # systematically clipped title is a synthetic tell. Genuinely bad crops are
    # modelled by the hard clip below, which is rare and much more severe --
    # which is also how they occur in the real extraction output.
    base = 0.004 + 0.028 * s
    insets = [rng.uniform(0.002, base) for _ in range(4)]  # l, t, r, b

    # Every so often the layout model's bbox is simply wrong and clips a title
    # or half the legend. Those crops exist in the real data, so they exist here.
    if rng.chance(0.14 * s + 0.02):
        side = rng.randint(0, 3)
        insets[side] += rng.uniform(0.04, 0.14)
        p.hard_clip_side = ("left", "top", "right", "bottom")[side]

    l, t, r, b = insets
    p.crop_inset = (l, t, r, b)
    box = (round(w * l), round(h * t), round(w * (1 - r)), round(h * (1 - b)))
    return img.crop(box)


def _resize(img: Image.Image, target_wh: tuple[int, int], p: DegradeParams, rng: Rng) -> Image.Image:
    method, name = rng.weighted({
        (Image.Resampling.LANCZOS, "lanczos"): 0.55,
        (Image.Resampling.BILINEAR, "bilinear"): 0.30,
        (Image.Resampling.BICUBIC, "bicubic"): 0.15,
    })
    p.resample = name
    return img.resize(target_wh, method)


def _blur(img: Image.Image, p: DegradeParams, s: float, rng: Rng) -> Image.Image:
    if rng.chance(0.15):
        # Some viewers sharpen on render; the halo is a real artefact.
        img = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=int(40 + 60 * s), threshold=2))
        p.unsharp = True
        return img
    sigma = rng.uniform(0.0, 0.85 * s)
    if sigma > 0.05:
        img = img.filter(ImageFilter.GaussianBlur(sigma))
        p.blur_sigma = sigma
    return img


def _jpeg(img: Image.Image, p: DegradeParams, s: float, rng: Rng) -> Image.Image:
    q = int(round(92 - 57 * s * rng.uniform(0.6, 1.0)))
    q = int(np.clip(q, 35, 95))
    img = _jpeg_roundtrip(img, q)
    p.jpeg_quality, p.jpeg_passes = q, 1

    if rng.chance(0.10 * s + 0.02):
        # Screenshot of a screenshot, forwarded through a chat client.
        q2 = int(np.clip(q - rng.randint(8, 20), 30, 92))
        img = _jpeg_roundtrip(img, q2)
        p.jpeg_quality, p.jpeg_passes = q2, 2
    return img


def _jpeg_roundtrip(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, subsampling=2)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _noise(img: Image.Image, p: DegradeParams, s: float, rng: Rng) -> Image.Image:
    sigma = 0.6 + 5.5 * s * rng.uniform(0.0, 1.0)
    if sigma < 0.8:
        return img
    arr = np.asarray(img, dtype=np.float32)
    arr += rng.noise(arr.shape, sigma)
    p.noise_sigma = sigma
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


def _tone(img: Image.Image, p: DegradeParams, s: float, rng: Rng) -> Image.Image:
    p.brightness = rng.uniform(1.0 - 0.07 * s, 1.0 + 0.07 * s)
    p.contrast = rng.uniform(1.0 - 0.10 * s, 1.0 + 0.10 * s)
    p.saturation = rng.uniform(1.0 - 0.14 * s, 1.0 + 0.05 * s)
    img = ImageEnhance.Brightness(img).enhance(p.brightness)
    img = ImageEnhance.Contrast(img).enhance(p.contrast)
    img = ImageEnhance.Color(img).enhance(p.saturation)

    p.gamma = rng.uniform(0.92, 1.10)
    if abs(p.gamma - 1.0) > 0.01:
        lut = [int(np.clip(((i / 255.0) ** p.gamma) * 255.0, 0, 255)) for i in range(256)]
        img = img.point(lut * 3)

    if rng.chance(0.05):
        img = img.convert("L").convert("RGB")
        p.greyscale = True
    elif rng.chance(0.06):
        # Warm cast of a flatbed scan or a photographed page.
        arr = np.asarray(img, dtype=np.float32)
        arr[..., 0] *= rng.uniform(1.01, 1.05)
        arr[..., 2] *= rng.uniform(0.94, 0.99)
        img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")
        p.scan_tint = True

    return img
