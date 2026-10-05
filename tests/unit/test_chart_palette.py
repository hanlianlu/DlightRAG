# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The Mineral colours of the charts and the report: one source, checked against the product's own."""

import itertools
import json
import re
from pathlib import Path

import pytest

from tests.support import colour

_ROOT = Path(__file__).resolve().parents[2]
_CHART = _ROOT / "chart-render"
_PALETTE = json.loads((_CHART / "palette.json").read_text(encoding="utf-8"))
_REPORT_CSS = (_CHART / "report/report.css").read_text(encoding="utf-8")
_FOUNDATIONS = _ROOT / "frontend/design-system/foundations"
_PRODUCT_CSS = (_FOUNDATIONS / "color.css").read_text(encoding="utf-8")
_PRODUCT_GEOMETRY = (_FOUNDATIONS / "geometry.css").read_text(encoding="utf-8")
_MODES = ("light", "dark")

# What a chart can be drawn on: the page, a card, and a raised surface (stone 50, 200, 300; 950, 900,
# 800). Charts sit on the first two; the third is what a tooltip or a menu would be.
_SURFACES = {
    "light": ("#fafaf9", "#e7e5e4", "#d6d3d1"),
    "dark": ("#0c0a09", "#1c1917", "#292524"),
}
# Colours are one and the same family when their pairwise distance clears these floors (OKLab x 100).
_NORMAL_NEIGHBOUR = 15
_CVD_FLOOR = 8
# The categorical palette is good for this many series by colour alone: the first six are told apart
# under protanopia and deuteranopia in every pair.
_COLOUR_ALONE = 6


def _colours(mode: str, name: str) -> list[colour.Rgba]:
    return [colour.parse(c) for c in _PALETTE[mode][name]]


def _declarations(block: str) -> dict[str, str]:
    cleaned = re.sub(r"/\*.*?\*/", "", block, flags=re.DOTALL)
    return {
        name: " ".join(value.split()).lower()
        for name, value in re.findall(r"(--[\w-]+)\s*:\s*([^;]+);", cleaned)
    }


def _block(css: str, opener: str) -> str:
    """The body of the first rule that opens with ``opener``, braces matched."""
    start = css.index(opener) + len(opener)
    depth, end = 1, start
    while depth:
        depth += {"{": 1, "}": -1}.get(css[end], 0)
        end += 1
    return css[start : end - 1]


def _report_tokens(mode: str) -> dict[str, str]:
    light = _declarations(_block(_REPORT_CSS, ":root {"))
    if mode == "light":
        return light
    dark = _declarations(
        _block(_block(_REPORT_CSS, "@media (prefers-color-scheme: dark) {"), ":root {")
    )
    return {**light, **dark}


def _product_tokens(mode: str) -> dict[str, str]:
    """The product's colour tokens for a mode, aliases resolved to their literal values."""
    base = {
        **_declarations(_block(_PRODUCT_CSS, ":root {")),
        **_declarations(_block(_PRODUCT_GEOMETRY, ":root {")),
    }
    tokens = (
        {**base, **_declarations(_block(_PRODUCT_CSS, ":root[data-color-mode='light'] {"))}
        if mode == "light"
        else base
    )

    def resolve(value: str) -> str:
        while match := re.fullmatch(r"var\((--[\w-]+)\)", value):
            value = tokens[match.group(1)]
        return value

    return {
        name: resolve(value)
        for name, value in tokens.items()
        if name.startswith(("--color-", "--radius-"))
    }


def _same_colour(a: str, b: str) -> bool:
    return colour.parse(a) == pytest.approx(colour.parse(b), abs=1e-6)  # type: ignore[comparison-overlap]


@pytest.mark.parametrize("mode", _MODES)
def test_the_report_tokens_are_the_products(mode: str) -> None:
    ours, theirs = _report_tokens(mode), _product_tokens(mode)
    shared = sorted(n for n in ours if n.startswith(("--color-", "--radius-")) and n in theirs)

    assert len(shared) >= 25
    assert [n for n in shared if ours[n] != theirs[n]] == []
    # Every colour the report defines but the product does not is a tone, named as one.
    unknown = {n for n in ours if n.startswith("--color-") and n not in theirs}
    assert unknown == set()


def test_the_report_names_the_roles_the_owner_asked_for() -> None:
    required = {
        "--color-bg-base", "--color-bg-surface", "--color-bg-elevated", "--color-text-primary",
        "--color-text-secondary", "--color-text-muted", "--color-border-subtle",
        "--color-border-strong", "--color-accent-action", "--color-accent-text", "--color-danger",
        "--color-success", "--color-control-ring", "--radius-control", "--radius-card",
        "--radius-pill",
    }  # fmt: skip
    for mode in _MODES:
        assert required <= set(_report_tokens(mode))
    assert "color-scheme: light dark" in _REPORT_CSS


def test_the_stylesheet_has_no_colour_outside_its_token_blocks() -> None:
    body = re.sub(r"/\*.*?\*/", "", _REPORT_CSS, flags=re.DOTALL)
    for opener in (":root {", "@media (prefers-color-scheme: dark) {"):
        block = _block(body, opener)
        body = body.replace(block, "", 1)
    literals = re.findall(r"#[0-9a-fA-F]{3,8}\b|\brgba?\(|\bhsla?\(|\boklch\(|\blab\(", body)

    assert literals == []


@pytest.mark.parametrize("mode", _MODES)
def test_the_chart_roles_are_the_mineral_roles(mode: str) -> None:
    roles, ours, theirs = _PALETTE[mode]["roles"], _report_tokens(mode), _product_tokens(mode)
    same = {
        "background": "--color-bg-base",
        "surface": "--color-bg-surface",
        "text": "--color-text-primary",
        "textMuted": "--color-text-muted",
        "axisLine": "--color-border-strong",
        "gridLine": "--color-divider",
        "tooltipBorder": "--color-border-subtle",
        "inactive": "--color-text-dim",
    }
    for role, token in same.items():
        assert _same_colour(roles[role], theirs[token]), f"{mode} {role} is not {token}"
        if token in ours:
            assert ours[token] == theirs[token]
    tooltip = "--color-bg-subtle" if mode == "light" else "--color-bg-elevated"
    assert _same_colour(roles["tooltipBackground"], theirs[tooltip])
    assert _same_colour(roles["tooltipText"], theirs["--color-text-primary"])


@pytest.mark.parametrize("mode", _MODES)
def test_the_lead_colour_is_the_primary_accent_of_the_mode(mode: str) -> None:
    assert _same_colour(
        _PALETTE[mode]["categorical"][0], _product_tokens(mode)["--color-accent-action"]
    )
    assert _same_colour(
        _PALETTE[mode]["highlight"][0], _product_tokens(mode)["--color-accent-action"]
    )


@pytest.mark.parametrize("mode", _MODES)
def test_every_categorical_mark_holds_three_to_one_on_the_surfaces_a_chart_sits_on(
    mode: str,
) -> None:
    page, card, _raised = (colour.parse(s) for s in _SURFACES[mode])

    assert len(_PALETTE[mode]["categorical"]) == 8
    for mark in _colours(mode, "categorical"):
        assert colour.contrast(mark, page) >= 3
        assert colour.contrast(mark, card) >= 3


@pytest.mark.parametrize("mode", _MODES)
def test_neighbours_differ_in_lightness_and_in_hue_and_are_far_apart_even_without_colour_vision(
    mode: str,
) -> None:
    marks = _colours(mode, "categorical")

    for a, b in itertools.pairwise(marks):
        assert abs(colour.oklch(a)[0] - colour.oklch(b)[0]) >= 0.045
        if colour.oklch(a)[1] > 0.04 and colour.oklch(b)[1] > 0.04:
            assert colour.hue_gap(a, b) >= 25
        assert colour.distance(a, b) >= _NORMAL_NEIGHBOUR
        for kind in ("protan", "deutan", "tritan"):
            assert colour.distance(a, b, kind) >= _CVD_FLOOR


@pytest.mark.parametrize("mode", _MODES)
def test_no_red_sits_beside_a_green(mode: str) -> None:
    hues = [colour.oklch(c)[2] for c in _colours(mode, "categorical")]

    def red(hue: float) -> bool:
        return hue < 55 or hue > 340

    def green(hue: float) -> bool:
        return 115 < hue < 185

    for a, b in itertools.pairwise(hues):
        assert not (red(a) and green(b)) and not (green(a) and red(b))


@pytest.mark.parametrize("mode", _MODES)
def test_the_first_six_colours_are_told_apart_in_every_pair_under_protanopia_and_deuteranopia(
    mode: str,
) -> None:
    marks = _colours(mode, "categorical")[:_COLOUR_ALONE]

    for a, b in itertools.combinations(marks, 2):
        for kind in ("protan", "deutan"):
            assert colour.distance(a, b, kind) >= _CVD_FLOOR - 0.05
        assert colour.distance(a, b) >= 10


@pytest.mark.parametrize("mode", _MODES)
def test_the_sequential_ramp_is_one_hue_in_even_monotone_steps(mode: str) -> None:
    ramp = _colours(mode, "sequential")
    lightness = [colour.oklch(c)[0] for c in ramp]
    ordered = sorted(lightness, reverse=(mode == "light"))

    assert lightness == ordered
    for a, b in itertools.pairwise(ramp):
        assert 10 <= colour.distance(a, b) <= 20
    # The first step is the neutral stone; every step after it is the product's gold.
    assert colour.oklch(ramp[0])[1] < 0.02
    assert all(85 <= colour.oklch(c)[2] <= 95 for c in ramp[1:])


@pytest.mark.parametrize("mode", _MODES)
def test_the_diverging_ramp_has_a_neutral_middle_and_two_unlike_hues(mode: str) -> None:
    ramp = _colours(mode, "diverging")
    middle = len(ramp) // 2
    lightness = [colour.oklch(c)[0] for c in ramp]

    assert colour.oklch(ramp[middle])[1] < 0.02
    towards = lightness[: middle + 1]  # from the end of the first hue to the neutral middle
    away = lightness[middle:]
    assert towards == sorted(towards, reverse=(mode == "dark"))
    assert away == sorted(away, reverse=(mode == "light"))
    assert colour.hue_gap(ramp[0], ramp[-1]) >= 120
    for a, b in itertools.pairwise(ramp):
        assert colour.distance(a, b) >= 10
    assert _PALETTE[mode]["divergingEnds"] == [
        _PALETTE[mode]["diverging"][0],
        _PALETTE[mode]["diverging"][-1],
    ]


@pytest.mark.parametrize("mode", _MODES)
def test_the_highlight_palette_is_the_lead_colour_then_graded_stone(mode: str) -> None:
    marks = _colours(mode, "highlight")
    card = colour.parse(_SURFACES[mode][1])

    assert all(colour.oklch(c)[1] < 0.02 for c in marks[1:])
    greys = [colour.oklch(c)[0] for c in marks[1:4]]
    assert greys == sorted(greys, reverse=True)
    for a, b in itertools.pairwise(marks[1:4]):
        assert colour.distance(a, b) >= 8
    assert min(colour.contrast(c, card) for c in marks) >= 3


@pytest.mark.parametrize("mode", _MODES)
def test_chart_text_holds_its_contrast_on_every_surface(mode: str) -> None:
    roles = _PALETTE[mode]["roles"]
    for surface in map(colour.parse, _SURFACES[mode]):
        assert colour.contrast(colour.parse(roles["textMuted"]), surface) >= 4.5
        assert colour.contrast(colour.parse(roles["text"]), surface) >= 7


def test_the_tag_tones_are_marks_of_the_categorical_palette_and_stay_apart_for_every_reader() -> (
    None
):
    # Tone 1 to 6 are azurite, malachite, amethyst, aquamarine, cinnabar and slate: the six marks,
    # leaving out the accent and rhodonite, that stay apart in every pair for a deuteranope.
    slots = [1, 2, 4, 6, 5, 3]
    for mode in _MODES:
        tokens = _report_tokens(mode)
        tones = [colour.parse(tokens[f"--tone-{n}"]) for n in range(1, 7)]
        assert tones == [_colours(mode, "categorical")[i] for i in slots]
        for a, b in itertools.combinations(tones, 2):
            assert colour.distance(a, b) >= 11
            for kind in ("protan", "deutan"):
                assert colour.distance(a, b, kind) >= 4.9
