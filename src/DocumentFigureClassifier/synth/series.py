"""
Plausible number series.

Uniform random values produce charts no company ever printed: bars that jump
by a factor of five between adjacent years. Real KPI series have a trend, mild
noise, and occasionally one sharp break (a pandemic year, an acquisition).
Shape carries information the model can use, so it should be realistic.
"""

from __future__ import annotations

from .rng import Rng


def kpi_series(rng: Rng, n: int, magnitude: float, allow_negative: bool = False) -> list[float]:
    """
    A single KPI over n periods around a given order of magnitude.

    Shapes: steady growth, decline, flat, recovery (dip then rise), and one
    outlier year. Weights roughly follow what a decade of annual reports looks
    like -- mostly growth, because companies that shrink for ten years stop
    publishing charts about it.
    """
    shape = rng.weighted({"growth": 0.38, "flat": 0.22, "decline": 0.14,
                          "recovery": 0.16, "spike": 0.10})

    base = magnitude * rng.uniform(0.75, 1.25)
    drift = {
        "growth": rng.uniform(0.03, 0.14),
        "flat": rng.uniform(-0.015, 0.015),
        "decline": rng.uniform(-0.12, -0.02),
        "recovery": rng.uniform(0.02, 0.09),
        "spike": rng.uniform(0.0, 0.05),
    }[shape]
    noise = rng.uniform(0.015, 0.07)

    values: list[float] = []
    v = base
    for i in range(n):
        v = v * (1.0 + drift + rng.normal(0.0, noise))
        values.append(v)

    if shape == "recovery" and n >= 4:
        dip = rng.randint(1, max(1, n - 3))
        for j in (dip, dip + 1):
            if j < n:
                values[j] *= rng.uniform(0.55, 0.80)
    elif shape == "spike" and n >= 3:
        k = rng.randint(0, n - 1)
        values[k] *= rng.uniform(1.35, 1.9) if rng.chance(0.6) else rng.uniform(0.45, 0.7)

    if not allow_negative:
        values = [max(v, magnitude * 0.05) for v in values]
    return values


def category_series(rng: Rng, n: int, magnitude: float) -> list[float]:
    """
    Values across unordered categories (regions, segments).

    Not a trend -- a distribution. Usually one dominant category and a long
    tail, which is why revenue-by-region charts nearly always slope.
    """
    weights = [rng.uniform(0.12, 1.0) ** rng.uniform(1.2, 2.4) for _ in range(n)]
    total = sum(weights)
    scale = magnitude * n * rng.uniform(0.7, 1.3)
    values = [w / total * scale for w in weights]
    if rng.chance(0.7):  # ranked charts are the norm
        values.sort(reverse=rng.chance(0.8))
    return values


def grouped_series(
    rng: Rng, n_groups: int, n_series: int, magnitude: float
) -> list[list[float]]:
    """
    Several series over the same categories, e.g. prior year vs. reporting year.

    The series are correlated: a company whose revenue grows usually grows in
    every segment. Independent draws look obviously fake.
    """
    base = kpi_series(rng, n_groups, magnitude)
    out = [base]
    for _ in range(n_series - 1):
        factor = rng.uniform(0.55, 1.45)
        out.append([v * factor * (1.0 + rng.normal(0.0, 0.06)) for v in base])
    return rng.shuffled(out) if rng.chance(0.4) else out


def round_nicely(values: list[float], decimals: int) -> list[float]:
    """Charts show rounded numbers; the bar heights should match the labels."""
    return [round(v, decimals) for v in values]
