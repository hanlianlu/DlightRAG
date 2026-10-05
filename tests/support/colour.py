# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Colour arithmetic for the palette and contrast tests: WCAG contrast, OKLab, CVD simulation.

Distances are OKLab Euclidean distance times 100; colour-vision deficiencies are simulated with the
Machado, Oliveira and Fernandes (2009) matrices at severity 1.0, the model the data-viz palette
checks are calibrated to.
"""

import math
import re

type Rgba = tuple[float, float, float, float]

_MACHADO = {
    "protan": (
        (0.152286, 1.052583, -0.204868),
        (0.114503, 0.786281, 0.099216),
        (-0.003882, -0.048116, 1.051998),
    ),
    "deutan": (
        (0.367322, 0.860646, -0.227968),
        (0.280085, 0.672501, 0.047413),
        (-0.011820, 0.042940, 0.968881),
    ),
    "tritan": (
        (1.255528, -0.076749, -0.178779),
        (-0.078411, 0.930809, 0.147602),
        (0.004733, 0.691367, 0.303900),
    ),
}


def parse(text: str) -> Rgba:
    """Read a CSS colour: ``#rgb``, ``#rrggbb``, ``rgb(r g b / a%)`` or ``rgba(r, g, b, a)``."""
    text = text.strip().lower()
    if text.startswith("#"):
        digits = text[1:]
        if len(digits) == 3:
            digits = "".join(c * 2 for c in digits)
        red, green, blue = (int(digits[i : i + 2], 16) / 255 for i in (0, 2, 4))
        return (red, green, blue, 1.0)
    match = re.fullmatch(r"rgba?\(([^)]*)\)", text)
    if not match:
        raise ValueError(f"not a colour: {text!r}")
    parts = [p for p in re.split(r"[\s,/]+", match.group(1).strip()) if p]
    red, green, blue = (float(p) / 255 for p in parts[:3])
    alpha = 1.0
    if len(parts) > 3:
        alpha = float(parts[3].rstrip("%")) / (100 if parts[3].endswith("%") else 1)
    return (red, green, blue, alpha)


def over(top: Rgba, bottom: Rgba) -> Rgba:
    """Return ``top`` painted over ``bottom``."""
    alpha = top[3] + bottom[3] * (1 - top[3])
    if alpha == 0:
        return (0.0, 0.0, 0.0, 0.0)
    red, green, blue = (
        (top[i] * top[3] + bottom[i] * bottom[3] * (1 - top[3])) / alpha for i in range(3)
    )
    return (red, green, blue, alpha)


def _linear(channel: float) -> float:
    return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4


def luminance(colour: Rgba) -> float:
    r, g, b = (_linear(c) for c in colour[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: Rgba, b: Rgba) -> float:
    """The WCAG contrast ratio of two opaque colours."""
    high, low = sorted((luminance(a), luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def oklab(colour: Rgba, simulate: str | None = None) -> tuple[float, float, float]:
    r, g, b = (_linear(c) for c in colour[:3])
    if simulate:
        m = _MACHADO[simulate]
        r, g, b = (max(0.0, min(1.0, m[i][0] * r + m[i][1] * g + m[i][2] * b)) for i in range(3))
    long = math.cbrt(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    medium = math.cbrt(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    short = math.cbrt(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return (
        0.2104542553 * long + 0.7936177850 * medium - 0.0040720468 * short,
        1.9779984951 * long - 2.4285922050 * medium + 0.4505937099 * short,
        0.0259040371 * long + 0.7827717662 * medium - 0.8086757660 * short,
    )


def oklch(colour: Rgba) -> tuple[float, float, float]:
    """Return lightness, chroma and hue in degrees."""
    lightness, a, b = oklab(colour)
    return lightness, math.hypot(a, b), math.degrees(math.atan2(b, a)) % 360


def distance(a: Rgba, b: Rgba, simulate: str | None = None) -> float:
    """The OKLab distance times 100, under normal vision or a simulated deficiency."""
    return 100 * math.dist(oklab(a, simulate), oklab(b, simulate))


def hue_gap(a: Rgba, b: Rgba) -> float:
    gap = abs(oklch(a)[2] - oklch(b)[2]) % 360
    return min(gap, 360 - gap)
