"""
Contact sheets.

Ten minutes of looking at an 8x8 grid catches most generator bugs -- clipped
labels, colours that vanish into the background, a sub-type that silently
renders identically to another. Cheap, and there is no substitute.

Images are letterboxed rather than stretched: the aspect ratio distribution is
one of the things being reviewed, so distorting it would defeat the purpose.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

_BG = (245, 245, 247)
_CELL_BG = (255, 255, 255)
_BORDER = (205, 208, 213)
_CAPTION = (70, 74, 80)


def build(
    images: list[Path],
    out_path: Path,
    captions: list[str] | None = None,
    cols: int = 8,
    cell: int = 240,
    caption_h: int = 16,
    title: str | None = None,
) -> Path:
    if not images:
        raise ValueError("no images")

    rows = (len(images) + cols - 1) // cols
    pad = 6
    cw, ch = cell, cell + caption_h
    title_h = 26 if title else 0

    sheet = Image.new(
        "RGB",
        (cols * (cw + pad) + pad, rows * (ch + pad) + pad + title_h),
        _BG,
    )
    draw = ImageDraw.Draw(sheet)
    if title:
        draw.text((pad + 2, 7), title, fill=(30, 33, 38))

    for i, path in enumerate(images):
        r, c = divmod(i, cols)
        x0 = pad + c * (cw + pad)
        y0 = pad + title_h + r * (ch + pad)

        draw.rectangle([x0, y0, x0 + cw, y0 + cell], fill=_CELL_BG, outline=_BORDER)
        try:
            im = Image.open(path).convert("RGB")
        except OSError:
            draw.text((x0 + 6, y0 + 6), "unreadable", fill=(180, 60, 60))
            continue

        im.thumbnail((cw - 8, cell - 8), Image.Resampling.LANCZOS)
        sheet.paste(im, (x0 + (cw - im.width) // 2, y0 + (cell - im.height) // 2))

        if captions:
            text = captions[i]
            if len(text) > 40:
                text = text[:38] + "…"
            draw.text((x0 + 3, y0 + cell + 3), text, fill=_CAPTION)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    return out_path
