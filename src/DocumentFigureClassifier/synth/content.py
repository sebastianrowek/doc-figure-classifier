"""
Business vocabulary and number formatting.

The text on a chart is not decoration -- it is a large part of what the crop
looks like at 224x224. Placeholder strings would give the model a systematic
cue that has nothing to do with chart type, so titles, categories, units and
number formats come from real annual-report vocabulary.

German dominates (75/25) because the corpus is German annual reports. This is
the one place where the project deliberately stays German: it is data inside
the product, not a document anyone reads.
"""

from __future__ import annotations

from dataclasses import dataclass

from .rng import Rng

# --------------------------------------------------------------------------
# Topics: a title, the unit it is measured in, and what its categories are
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Topic:
    title: str
    subtitle: str
    unit: str
    category_kind: str  # "year" | "quarter" | "region" | "segment" | "other"
    magnitude: float  # rough order of magnitude of the values
    decimals: int
    percent: bool = False


# (title, subtitle, unit, category_kind, magnitude, decimals, percent)
_TOPICS_DE: tuple[tuple[str, str, str, str, float, int, bool], ...] = (
    ("Umsatzentwicklung", "Konzernumsatz", "Mio. €", "year", 1800, 0, False),
    ("Konzernumsatz", "in Mio. €", "Mio. €", "year", 4200, 0, False),
    ("Umsatz nach Regionen", "Geschäftsjahr 2024", "Mio. €", "region", 900, 0, False),
    ("Umsatz nach Segmenten", "Anteile am Konzernumsatz", "Mio. €", "segment", 700, 0, False),
    ("EBIT", "Ergebnis vor Zinsen und Steuern", "Mio. €", "year", 320, 0, False),
    ("EBITDA", "bereinigt", "Mio. €", "year", 540, 0, False),
    ("EBIT-Marge", "in Prozent vom Umsatz", "%", "year", 11.5, 1, True),
    ("Konzernergebnis", "nach Steuern", "Mio. €", "year", 210, 0, False),
    ("Ergebnis je Aktie", "unverwässert", "€", "year", 3.4, 2, False),
    ("Dividende je Aktie", "Vorschlag an die Hauptversammlung", "€", "year", 1.15, 2, False),
    ("Free Cashflow", "in Mio. €", "Mio. €", "year", 260, 0, False),
    ("Cashflow aus laufender Geschäftstätigkeit", "", "Mio. €", "year", 430, 0, False),
    ("Investitionen", "Sachanlagen und immaterielle Vermögenswerte", "Mio. €", "year", 180, 0, False),
    ("Investitionsquote", "in Prozent vom Umsatz", "%", "year", 6.2, 1, True),
    ("F&E-Aufwendungen", "Forschung und Entwicklung", "Mio. €", "year", 145, 0, False),
    ("Auftragseingang", "in Mio. €", "Mio. €", "quarter", 1150, 0, False),
    ("Auftragsbestand", "zum Bilanzstichtag", "Mrd. €", "year", 3.8, 1, False),
    ("Eigenkapitalquote", "in Prozent der Bilanzsumme", "%", "year", 42.0, 1, True),
    ("Nettofinanzschulden", "in Mio. €", "Mio. €", "year", 640, 0, False),
    ("Bilanzsumme", "in Mio. €", "Mio. €", "year", 5400, 0, False),
    ("Mitarbeiter", "im Jahresdurchschnitt", "Mitarbeiter", "year", 8600, 0, False),
    ("Mitarbeiter nach Regionen", "zum 31.12.2024", "Mitarbeiter", "region", 2400, 0, False),
    ("Belegschaft nach Segmenten", "Vollzeitäquivalente", "FTE", "segment", 1900, 0, False),
    ("Frauenanteil in Führungspositionen", "in Prozent", "%", "year", 27.5, 1, True),
    ("Ausbildungsquote", "in Prozent der Belegschaft", "%", "year", 5.8, 1, True),
    ("Unfallhäufigkeit", "je 1 Mio. Arbeitsstunden", "LTIF", "year", 3.2, 1, False),
    ("CO2-Emissionen", "Scope 1 und 2", "kt CO2e", "year", 420, 0, False),
    ("Energieverbrauch", "in GWh", "GWh", "year", 780, 0, False),
    ("Wasserentnahme", "in Tsd. m³", "Tsd. m³", "year", 1250, 0, False),
    ("Abfallaufkommen", "in Tsd. t", "Tsd. t", "year", 96, 1, False),
    ("Recyclingquote", "in Prozent", "%", "year", 68.0, 1, True),
    ("Aktienkursentwicklung", "indexiert", "Index", "quarter", 118, 0, False),
    ("Marktanteil", "in Prozent", "%", "region", 18.5, 1, True),
    ("Absatzmenge", "in Tsd. Stück", "Tsd. Stück", "year", 640, 0, False),
    ("Produktionsvolumen", "in Tsd. t", "Tsd. t", "quarter", 310, 0, False),
    ("Personalaufwand", "in Mio. €", "Mio. €", "year", 760, 0, False),
    ("Materialaufwand", "in Mio. €", "Mio. €", "year", 1450, 0, False),
    ("Umsatzrendite", "in Prozent", "%", "year", 8.4, 1, True),
    ("Return on Capital Employed", "ROCE", "%", "year", 13.6, 1, True),
    ("Working Capital", "in Mio. €", "Mio. €", "quarter", 520, 0, False),
)

_TOPICS_EN: tuple[tuple[str, str, str, str, float, int, bool], ...] = (
    ("Revenue development", "Group revenue", "€ m", "year", 1800, 0, False),
    ("Group revenue", "in € million", "€ m", "year", 4200, 0, False),
    ("Revenue by region", "financial year 2024", "€ m", "region", 900, 0, False),
    ("Revenue by segment", "share of group revenue", "€ m", "segment", 700, 0, False),
    ("EBIT", "earnings before interest and taxes", "€ m", "year", 320, 0, False),
    ("EBITDA", "adjusted", "€ m", "year", 540, 0, False),
    ("EBIT margin", "as a percentage of revenue", "%", "year", 11.5, 1, True),
    ("Net income", "after taxes", "€ m", "year", 210, 0, False),
    ("Earnings per share", "basic", "€", "year", 3.4, 2, False),
    ("Dividend per share", "proposal to the AGM", "€", "year", 1.15, 2, False),
    ("Free cash flow", "in € million", "€ m", "year", 260, 0, False),
    ("Capital expenditure", "property, plant and equipment", "€ m", "year", 180, 0, False),
    ("R&D expenses", "research and development", "€ m", "year", 145, 0, False),
    ("Order intake", "in € million", "€ m", "quarter", 1150, 0, False),
    ("Equity ratio", "as a percentage of total assets", "%", "year", 42.0, 1, True),
    ("Net financial debt", "in € million", "€ m", "year", 640, 0, False),
    ("Employees", "annual average", "employees", "year", 8600, 0, False),
    ("Employees by region", "as of 31 Dec 2024", "employees", "region", 2400, 0, False),
    ("Women in management positions", "in percent", "%", "year", 27.5, 1, True),
    ("CO2 emissions", "scope 1 and 2", "kt CO2e", "year", 420, 0, False),
    ("Energy consumption", "in GWh", "GWh", "year", 780, 0, False),
    ("Recycling rate", "in percent", "%", "year", 68.0, 1, True),
    ("Market share", "in percent", "%", "region", 18.5, 1, True),
    ("Sales volume", "in thousand units", "k units", "year", 640, 0, False),
    ("Return on capital employed", "ROCE", "%", "year", 13.6, 1, True),
)

# --------------------------------------------------------------------------
# Categories
# --------------------------------------------------------------------------

_REGIONS_DE = (
    "Deutschland", "Europa", "Übriges Europa", "Nordamerika", "Südamerika",
    "Asien/Pazifik", "China", "Naher Osten", "Afrika", "Übrige Welt",
    "EMEA", "Amerika", "Osteuropa", "Frankreich", "USA", "Indien",
)
_REGIONS_EN = (
    "Germany", "Europe", "Rest of Europe", "North America", "South America",
    "Asia/Pacific", "China", "Middle East", "Africa", "Rest of world",
    "EMEA", "Americas", "Eastern Europe", "France", "USA", "India",
)
_SEGMENTS_DE = (
    "Automotive", "Industrie", "Chemie", "Spezialprodukte", "Konsumgüter",
    "Bau", "Energie", "Gesundheit", "Logistik", "Dienstleistungen",
    "Verpackung", "Elektronik", "Agrar", "Sonstige", "Holding",
)
_SEGMENTS_EN = (
    "Automotive", "Industrial", "Chemicals", "Specialty products", "Consumer",
    "Construction", "Energy", "Health", "Logistics", "Services",
    "Packaging", "Electronics", "Agriculture", "Other", "Holding",
)

_SOURCES_DE = (
    "Quelle: Geschäftsbericht 2024",
    "Quelle: Konzernabschluss",
    "Angaben in Mio. €",
    "ungeprüft",
    "Vorjahreswerte angepasst",
    "Quelle: eigene Berechnungen",
)
_SOURCES_EN = (
    "Source: Annual Report 2024",
    "Source: consolidated financial statements",
    "unaudited",
    "prior-year figures restated",
    "Source: own calculations",
)


def pick_topic(rng: Rng, language: str) -> Topic:
    row = rng.pick(_TOPICS_DE if language == "de" else _TOPICS_EN)
    return Topic(*row)  # type: ignore[arg-type]


def categories(rng: Rng, kind: str, n: int, language: str) -> list[str]:
    """n category labels of the requested kind."""
    if kind == "year":
        end = rng.randint(2019, 2025)
        years = [end - n + 1 + i for i in range(n)]
        if rng.chance(0.25):  # short form, very common on narrow charts
            return [f"'{y % 100:02d}" for y in years]
        return [str(y) for y in years]

    if kind == "quarter":
        q0 = rng.randint(1, 4)
        y0 = rng.randint(2021, 2024)
        out = []
        for i in range(n):
            q = (q0 - 1 + i) % 4 + 1
            y = y0 + (q0 - 1 + i) // 4
            out.append(f"Q{q}/{y % 100:02d}" if rng.chance(0.6) else f"Q{q} {y}")
        return out

    pool = {
        ("region", "de"): _REGIONS_DE,
        ("region", "en"): _REGIONS_EN,
        ("segment", "de"): _SEGMENTS_DE,
        ("segment", "en"): _SEGMENTS_EN,
    }.get((kind, language), _SEGMENTS_DE if language == "de" else _SEGMENTS_EN)
    return rng.sample(pool, min(n, len(pool)))


def source_note(rng: Rng, language: str) -> str:
    return rng.pick(_SOURCES_DE if language == "de" else _SOURCES_EN)


def reference_label(rng: Rng, language: str, kind: str) -> str:
    """Caption for a target / average reference line."""
    if kind == "target":
        return rng.pick(("Ziel", "Ziel 2030", "Zielwert", "Planwert")) if language == "de" \
            else rng.pick(("Target", "Target 2030", "Plan"))
    return rng.pick(("Ø", "Durchschnitt", "Mittelwert")) if language == "de" \
        else rng.pick(("Avg.", "Average", "Mean"))


# --------------------------------------------------------------------------
# Number formatting
# --------------------------------------------------------------------------


def format_number(value: float, decimals: int, locale: str) -> str:
    """
    German charts write 1.234,5 -- English ones 1,234.5.

    Getting this wrong is a giveaway that the image is synthetic, and it is one
    character of code, so there is no excuse.
    """
    s = f"{value:,.{decimals}f}"
    if locale == "de":
        s = s.replace(",", "\x00").replace(".", ",").replace("\x00", ".")
    return s


def decimals_for(value: float, topic_decimals: int) -> int:
    """Real charts drop decimals as numbers grow; follow that."""
    a = abs(value)
    if a >= 1000:
        return 0
    if a >= 100:
        return min(topic_decimals, 1)
    return topic_decimals
