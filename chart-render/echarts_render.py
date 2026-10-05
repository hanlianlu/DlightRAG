#!/app/.venv/bin/python
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Render an Apache ECharts option as a PNG, with an SVG or a self-contained HTML page on request.

The image installs this file as ``echarts-render`` for the built-in ``charts`` Skill. Node draws
the chart with ECharts' server-side SVG renderer (``ssr.cjs``) and the resvg CLI rasterizes it.
The chart's look is the Mineral light theme: ``theme.json`` is its structure and ``palette.json``
its colours, which ``html-report`` shares; an option can ask for another palette by name.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, NoReturn

_HERE = Path(__file__).resolve().parent
# The theme's structure: its colours are "@role" references that palette.json fills in node.
_THEME = json.loads((_HERE / "theme.json").read_text(encoding="utf-8"))
_PALETTE = json.loads((_HERE / "palette.json").read_text(encoding="utf-8"))["light"]
# The palette names an option can ask for with "palette": categorical is the default.
PALETTES = ("categorical", *_THEME["palettes"])
_FONT = "Noto Sans SC"
# The image puts the font here; a checkout can point the variable at its own copy.
_FONT_DIR = os.environ.get("ECHARTS_RENDER_FONT_DIR", "/usr/local/share/fonts/noto-sans-sc")
# Every generic family is the one font the image ships: when none of an element's families exists,
# resvg falls back to its serif family, and with no font there it drops the text and still exits 0.
_FAMILY_FLAGS = (
    "--font-family",
    "--serif-family",
    "--sans-serif-family",
    "--monospace-family",
    "--cursive-family",
    "--fantasy-family",
)
# The series an option can describe in JSON alone: custom needs renderItem and map needs map data.
SERIES = frozenset(
    "line bar pie scatter effectScatter radar tree treemap sunburst boxplot candlestick heatmap "
    "parallel lines graph sankey chord funnel gauge pictorialBar themeRiver".split()
)
# zrender's own text-width table for ASCII (platform.js), which it falls back to on a server
# without a canvas; every other character counts as one em.
_ASCII_WIDTHS = "007LLmW'55;N0500LLLLLLLLLL00NNNLzWW\\\\WQb\\0FWLg\\bWb\\WQ\\WrWWQ000CL5LLFLL0LL**F*gLLLL5F0LF\\FFF5.5N"
# Vertical space the legend takes above the plot: its first row, and each row it wraps onto.
_LEGEND_ROW = 44
_LEGEND_WRAPPED_ROW = 34


def _fail(message: str) -> NoReturn:
    sys.exit(f"echarts-render: {message}")


def _as_list(value: Any) -> list[Any]:
    """Return what ECharts takes as one object or a list of them, as a list."""
    if isinstance(value, list):
        return value
    return [] if value is None else [value]


def _text_width(text: str, size: float) -> float:
    """Estimate a label's width the way zrender does without a canvas."""
    width = 0.0
    for char in text:
        if " " <= char <= "~":
            width += (ord(_ASCII_WIDTHS[ord(char) - 32]) - 20) / 100 * size
        else:
            width += size
    return width


def _read_option(source: str) -> tuple[str, dict[str, Any]]:
    """Read the option from a file or, for ``-``, from stdin; return its text and its value."""
    text = sys.stdin.read() if source == "-" else Path(source).read_text(encoding="utf-8")
    try:
        option = json.loads(
            text, parse_constant=lambda name: _fail(f"{name} is not a JSON value; write null")
        )
    except ValueError as error:
        _fail(f"{source} is not valid JSON: {error}")
    if not isinstance(option, dict):
        _fail("the option must be a JSON object")
    return text, option


def _drawable_series(option: dict[str, Any], text: str) -> list[dict[str, Any]]:
    """Refuse an option ECharts would draw as a blank picture, and return its series."""
    series = [s for s in _as_list(option.get("series")) if isinstance(s, dict)]
    for s in series:
        if s.get("type") not in SERIES:
            _fail(f"series type {json.dumps(s.get('type'))} cannot be drawn from JSON")
    if not option.get("dataset") and not any(s.get("data") or s.get("nodes") for s in series):
        _fail("no series has any data")
    if re.search(r'"formatter"\s*:\s*"\s*(function\b|\(?[\w\s,]*\)?\s*=>)', text):
        _fail('a formatter must be a template string such as "{b}: {c}", not JavaScript')
    if option.get("palette", "categorical") not in PALETTES:
        _fail(
            f'"palette" must be one of {", ".join(PALETTES)}, not {json.dumps(option["palette"])}'
        )
    return series


def _legend_rows(legend: Any, series: list[dict[str, Any]], width: int) -> int:
    """Count the rows a legend wraps onto, from the theme's item sizes and its entry names."""
    style = _THEME["legend"]
    names = (legend if isinstance(legend, dict) else {}).get("data") or [
        entry.get("name", "") if isinstance(entry, dict) else ""
        for s in series
        for entry in (s.get("data") or [] if s.get("type") == "pie" else [s])
    ]
    rows, x = 1, 0.0
    for name in names:
        label = str(name.get("name", "") if isinstance(name, dict) else name)
        # A legend entry is its symbol, a gap of 5 pixels, and the label.
        item = style["itemWidth"] + 5 + _text_width(label, style["textStyle"]["fontSize"])
        if x > 0 and x + item > width - style["left"] - _THEME["grid"]["right"]:
            rows, x = rows + 1, 0.0
        x += item + style["itemGap"]
    return rows


def _lay_out(option: dict[str, Any], series: list[dict[str, Any]], width: int) -> None:
    """Give the chart the layout defaults ECharts leaves to the caller."""
    if "legend" not in option and sum(s.get("type") != "pie" for s in series) > 1:
        option["legend"] = {}  # several series need a legend, and ECharts draws none unless asked
    legend, grid = option.get("legend"), option.get("grid")
    if not (isinstance(grid, list) or (grid or {}).get("top") is not None):
        top = _THEME["grid"]["top"]
        if legend in (None, False) or (isinstance(legend, dict) and legend.get("show") is False):
            option["grid"] = {**(grid or {}), "top": top - _LEGEND_ROW}
        elif (rows := _legend_rows(legend, series, width)) > 1:
            # A legend that wraps would cover the axis name, so each extra row gets its space.
            option["grid"] = {**(grid or {}), "top": top + (rows - 1) * _LEGEND_WRAPPED_ROW}
    if any(
        isinstance(axis, dict) and axis.get("type") == "category"
        for axis in _as_list(option.get("yAxis"))
    ):
        for s in series:  # horizontal bars round the value end, not the top
            if s.get("type") == "bar":
                s["itemStyle"] = {"borderRadius": [0, 4, 4, 0], **s.get("itemStyle", {})}


def _draw(option: dict[str, Any], width: int, height: int) -> str:
    """Draw the option as an SVG on node."""
    # A toolbox's buttons do nothing in a picture.
    picture = {key: value for key, value in option.items() if key != "toolbox"}
    request = {"option": {**picture, "animation": False}, "width": width, "height": height}
    drawn = subprocess.run(  # noqa: S603 - argv list, no shell, a fixed executable
        ["node", str(_HERE / "ssr.cjs")],  # noqa: S607 - node is on the image's PATH
        input=json.dumps(request),
        capture_output=True,
        text=True,
    )
    if drawn.returncode or not drawn.stdout.startswith("<svg"):
        _fail(f"ECharts could not draw this option: {drawn.stderr.strip()[:300] or 'no output'}")
    return drawn.stdout


class RenderError(Exception):
    """A step of the pipeline failed; the message says what to fix."""


def echarts_library() -> Path:
    """Return the ECharts build: beside this file in the image, in node_modules in a checkout."""
    for path in (_HERE / "echarts.min.js", _HERE / "node_modules/echarts/dist/echarts.min.js"):
        if path.is_file():
            return path
    raise RenderError("echarts.min.js is missing: run `npm ci` in chart-render")


def rasterize(svg: str, out: Path, scale: float) -> str:
    """Write the SVG as a PNG and return the characters the font lacks, if any.

    The SVG goes to resvg on stdin and the PNG comes back on stdout; ``html-report`` previews its
    charts through the same step.
    """
    flags = ["--skip-system-fonts", "--use-fonts-dir", _FONT_DIR]
    for family in _FAMILY_FLAGS:
        flags += [family, _FONT]
    out.parent.mkdir(parents=True, exist_ok=True)
    # Resources resolve beside the PNG, which is what stops resvg warning that stdin has no directory.
    command = ["resvg", *flags, "--zoom", str(scale), "--resources-dir", str(out.parent), "-", "-c"]
    png = subprocess.run(  # noqa: S603 - argv list, no shell, resvg is on the image's PATH
        command, input=svg.encode(), capture_output=True
    )
    warnings = png.stderr.decode(errors="replace")
    # With the font in place resvg warns of nothing, and "No match for" means text was dropped.
    if png.returncode or not png.stdout.startswith(b"\x89PNG") or "No match for" in warnings:
        raise RenderError(
            f"resvg could not rasterize the chart: {warnings.strip()[:300] or 'empty output'}"
        )
    out.write_bytes(png.stdout)
    return "".join(dict.fromkeys(re.findall(r"No fonts with a (.)/U\+", warnings)))


def _presentation_attributes(svg: str) -> str:
    """Turn every ``style="..."`` into presentation attributes.

    An SVG Artifact opened on its own is served with a policy that blocks inline CSS, and
    presentation attributes are not CSS. ECharts puts only font-family, font-size and font-weight
    in a style.
    """

    def attributes(match: re.Match[str]) -> str:
        # Decode the attribute text before splitting at ";", which an entity ends in, and escape
        # each value again for the attribute it moves to.
        declarations = (d.split(":", 1) for d in html.unescape(match[1]).split(";") if ":" in d)
        return " ".join(
            f'{name.strip()}="{html.escape(value.strip())}"' for name, value in declarations
        )

    return re.sub(r'style="([^"]*)"', attributes, svg)


def embed_json(value: Any) -> str:
    """Return JSON that cannot end a script element: every ``<`` is escaped."""
    return json.dumps(value, ensure_ascii=False).replace("<", "\\u003c")


def _html_page(option: dict[str, Any], svg: str, width: int, height: int) -> str:
    """Return a page with ECharts inlined, so it needs no network.

    The SVG shows until the reader activates the page and ECharts replaces it. The page paints its
    own background, since a transparent one turns black in a dark frame. It builds the same theme
    ``ssr.cjs`` does, from the same ``theme.js``, so the page and the PNG cannot differ.
    """
    toolbox = option.get("toolbox")
    if isinstance(toolbox, dict) and isinstance(toolbox.get("feature"), dict):
        toolbox["feature"].pop("saveAsImage", None)  # the artifact frame cannot download

    def inline(path: Path) -> str:
        # Library text must not end its own script element.
        return path.read_text(encoding="utf-8").replace("</script", "<\\/script")

    try:
        library = inline(echarts_library())
    except RenderError as error:
        _fail(str(error))
    theme = inline(_HERE / "theme.js")
    return (
        '<!doctype html><meta charset="utf-8">'
        f"<style>html,body{{margin:0;background:{_PALETTE['roles']['background']}}}"
        f"#chart{{width:{width}px;max-width:100%;height:{height}px}}</style>"
        f'<div id="chart">{svg}</div>'
        f"<script>{library}</script>"
        f"<script>(()=>{{const module={{exports:{{}}}};{theme}\nconst Theme=module.exports;"
        f"const picked=Theme.pick({embed_json(option)});"
        f"const themes=Theme.build({embed_json(_THEME)},{embed_json(_PALETTE)});"
        "echarts.registerTheme('dlight',themes[picked.name]);"
        "const el=document.getElementById('chart');el.textContent='';"
        "const chart=echarts.init(el,'dlight',{renderer:'svg'});"
        "chart.setOption(Theme.decorate(picked.option,picked.name));"
        "addEventListener('resize',()=>chart.resize());})();</script>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="echarts-render", description=__doc__)
    parser.add_argument("option", help="the option as a JSON file, or - to read it from stdin")
    parser.add_argument("png", help="the PNG to write")
    parser.add_argument("--svg", help="also write the SVG here")
    parser.add_argument("--html", help="also write a self-contained interactive HTML page here")
    parser.add_argument("--width", type=int, default=800, help="chart width in SVG pixels")
    parser.add_argument("--height", type=int, default=500, help="chart height in SVG pixels")
    parser.add_argument("--scale", type=float, default=2, help="PNG pixels per SVG pixel")
    args = parser.parse_args()

    text, option = _read_option(args.option)
    series = _drawable_series(option, text)
    _lay_out(option, series, args.width)
    svg = _draw(option, args.width, args.height)
    try:
        missing = rasterize(svg, Path(args.png).resolve(), args.scale)
    except RenderError as error:
        _fail(str(error))
    if missing:
        print(
            f"echarts-render: no font has {missing}; those characters show as boxes",
            file=sys.stderr,
        )
    if args.svg:
        Path(args.svg).write_text(_presentation_attributes(svg), encoding="utf-8")
    if args.html:
        Path(args.html).write_text(
            _html_page(option, svg, args.width, args.height), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
