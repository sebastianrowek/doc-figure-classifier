r"""
Typeface discovery.

matplotlib's DejaVu Sans is instantly recognisable and appears in no annual
report ever printed. Randomising the typeface is one of the cheapest ways to
stop the model latching onto letterforms.

We do not take whatever is installed: C:\Windows\Fonts also holds Wingdings,
Marlett and a pile of symbol fonts that would render charts as garbage. An
allowlist of families known to be used in business documents is intersected
with what matplotlib can actually find.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from matplotlib import font_manager

from .rng import Rng

# Grouped so a style can pick a coherent look rather than mixing a slab serif
# into a flat corporate deck.
_SANS = (
    "Arial", "Calibri", "Segoe UI", "Verdana", "Tahoma", "Trebuchet MS",
    "Candara", "Corbel", "Franklin Gothic Book", "Century Gothic",
    "Bahnschrift", "Lucida Sans", "Gill Sans MT", "Microsoft Sans Serif",
    "Malgun Gothic", "Dubai", "Ebrima", "Selawik", "Open Sans", "Roboto",
    "Lato", "Source Sans Pro", "PT Sans", "IBM Plex Sans", "Noto Sans",
)

_SERIF = (
    "Georgia", "Times New Roman", "Cambria", "Constantia", "Palatino Linotype",
    "Book Antiqua", "Garamond", "Bookman Old Style", "Perpetua", "Sitka Text",
    "Sylfaen", "Merriweather", "PT Serif", "Source Serif Pro", "Noto Serif",
)

_SLAB = ("Rockwell", "Bodoni MT", "Cambria", "Sitka Banner")

_FALLBACK = "DejaVu Sans"


@dataclass(frozen=True)
class FontSet:
    sans: tuple[str, ...]
    serif: tuple[str, ...]
    slab: tuple[str, ...]

    def pick(self, rng: Rng, kind: str) -> str:
        pool = {"sans": self.sans, "serif": self.serif, "slab": self.slab}[kind]
        return rng.pick(pool) if pool else _FALLBACK


@lru_cache(maxsize=1)
def available() -> FontSet:
    """
    Families matplotlib can resolve, intersected with the allowlist.

    Cached because scanning the font manager is slow and every worker process
    would otherwise redo it. lru_cache is per process, which is what we want:
    the result is not picklable in any useful way.
    """
    try:
        installed = set(font_manager.get_font_names())
    except Exception:  # pragma: no cover - very old matplotlib
        installed = {f.name for f in font_manager.fontManager.ttflist}

    def keep(names: tuple[str, ...]) -> tuple[str, ...]:
        found = tuple(n for n in names if n in installed)
        return found or (_FALLBACK,)

    return FontSet(sans=keep(_SANS), serif=keep(_SERIF), slab=keep(_SLAB))


def summary() -> str:
    f = available()
    return (
        f"sans({len(f.sans)}): {', '.join(f.sans)}\n"
        f"serif({len(f.serif)}): {', '.join(f.serif)}\n"
        f"slab({len(f.slab)}): {', '.join(f.slab)}"
    )


if __name__ == "__main__":
    print(summary())
