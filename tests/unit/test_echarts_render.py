# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""What ``echarts-render`` writes: the Mineral look, the palettes an option can ask for, its failures."""

import collections
import json
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

import echarts_render
import pytest
from PIL import Image

from tests.support.chart_tools import require_png_tools

_PALETTE = json.loads((Path(echarts_render.__file__).parent / "palette.json").read_text("utf-8"))
_LIGHT = _PALETTE["light"]


def _rgb(hex_colour: str) -> tuple[int, int, int]:
    return tuple(int(hex_colour[i : i + 2], 16) for i in (1, 3, 5))  # type: ignore[return-value]


def _bars(**extra: Any) -> dict[str, Any]:
    return {
        "title": {"text": "季度营收", "subtext": "单位：亿元 · 来源：公司财报"},
        "xAxis": {"type": "category", "data": ["一", "二", "三", "四"]},
        "yAxis": {"type": "value"},
        "series": [{"type": "bar", "data": [4.2, 4.8, 5.1, 6.0]}],
        **extra,
    }


def _render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, option: dict[str, Any], *flags: str
) -> Path:
    source, png = tmp_path / "option.json", tmp_path / "chart.png"
    source.write_text(json.dumps(option, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv", ["echarts-render", str(source), str(png), "--scale", "1", *flags]
    )
    echarts_render.main()
    return png


def _colours(png: Path) -> collections.Counter[tuple[int, int, int]]:
    pixels = Image.open(png).convert("RGB").get_flattened_data()
    return collections.Counter(cast(Iterable[tuple[int, int, int]], pixels))


def test_a_chart_is_drawn_in_the_first_palette_colour_on_the_mineral_background(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    require_png_tools()
    png = _render(tmp_path, monkeypatch, _bars())
    seen = _colours(png)
    background = _rgb(_LIGHT["roles"]["background"])
    image = Image.open(png).convert("RGB")

    assert image.getpixel((2, 2)) == background
    assert seen.most_common(1)[0][0] == background
    marks = [colour for colour, _ in seen.most_common(6) if colour != background]
    assert marks[0] == _rgb(_LIGHT["categorical"][0])


def test_the_highlight_palette_draws_the_first_series_in_gold_and_the_rest_in_stone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    require_png_tools()
    option = _bars(
        palette="highlight",
        xAxis={"type": "category", "data": ["一", "二", "三"]},
        series=[
            {"type": "bar", "name": "重点", "data": [5, 6, 7]},
            {"type": "bar", "name": "对照甲", "data": [3, 4, 3]},
            {"type": "bar", "name": "对照乙", "data": [2, 2, 3]},
        ],
    )
    seen = _colours(_render(tmp_path, monkeypatch, option))
    background = _rgb(_LIGHT["roles"]["background"])
    marks = {
        colour for colour, count in seen.most_common(12) if colour != background and count > 400
    }

    assert _rgb(_LIGHT["highlight"][0]) in marks
    assert _rgb(_LIGHT["highlight"][1]) in marks
    assert _rgb(_LIGHT["highlight"][2]) in marks
    assert _rgb(_LIGHT["categorical"][1]) not in marks


def test_a_series_colour_the_author_wrote_wins_over_the_palette(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    require_png_tools()
    option = _bars(
        series=[{"type": "bar", "data": [1, 2, 3, 4], "itemStyle": {"color": "#123456"}}]
    )
    seen = _colours(_render(tmp_path, monkeypatch, option))

    assert _rgb("#123456") in {c for c, count in seen.most_common(4)}


def test_an_unknown_palette_is_refused_in_one_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(SystemExit) as refused:
        _render(tmp_path, monkeypatch, _bars(palette="neon"))

    assert str(refused.value).startswith('echarts-render: "palette" must be one of categorical')


def test_an_option_echarts_cannot_draw_fails_with_its_message_and_does_not_hang(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    option = _bars(xAxis={"type": "nonexistent"})
    with pytest.raises(SystemExit) as refused:
        _render(tmp_path, monkeypatch, option)

    assert "ECharts could not draw this option" in str(refused.value)
    assert "xAxis.nonexistent" in str(refused.value)


def test_four_lines_get_their_own_dash_and_marker_and_the_svg_paints_the_mineral_roles() -> None:
    lines = [{"type": "line", "name": f"线{i}", "data": [i, i + 1, i + 2]} for i in range(4)]
    option = {"xAxis": {"type": "category", "data": ["a", "b", "c"]}, "yAxis": {}, "series": lines}
    svg = echarts_render._draw(option, 800, 500)

    assert re.search(r"stroke-dasharray", svg)
    assert svg.count("<path") > 8
    assert _LIGHT["roles"]["background"] in svg
    for colour in _LIGHT["categorical"][:4]:
        assert colour in svg


def test_the_html_page_builds_the_same_theme_from_the_same_files() -> None:
    page = echarts_render._html_page(_bars(palette="highlight"), "<svg></svg>", 800, 500)

    assert page.count("echarts.registerTheme") == 1
    assert "Theme.build(" in page and "Theme.pick(" in page
    assert _LIGHT["roles"]["background"] in page
    assert "@background" in page  # the structure still holds references, filled in the browser
    assert page.count("<script>") == 2 and page.count("</script>") == 2
