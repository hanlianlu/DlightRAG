#!/app/.venv/bin/python
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Build a self-contained interactive HTML report from a fragment of plain HTML.

The image installs this file as ``html-report`` for the built-in ``interactive-html`` Skill. The
fragment holds the report's prose, its slicers and pages, and one JSON block per chart; the build
checks it, then writes one document with the full ECharts build, the Mineral stylesheet and the
runtime inlined, which runs inside the Artifact sandbox without a network. ``--preview`` draws
every chart to PNG through the same code the browser runs, so the author can look before it ships.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import echarts_render

_HERE = Path(__file__).resolve().parent
_REPORT = _HERE / "report"
# The artifact limit of the product (active_html_max_bytes).
_MAX_BYTES = 20 * 1024 * 1024
_MAX_ERRORS = 10
_PREVIEW_WIDTHS = (360, 900)
_PREVIEW_SCALE = {360: 2, 900: 1}

# The series an option can draw from JSON alone (custom needs renderItem, map needs map data).
_SERIES = echarts_render.SERIES
_TIDY_SERIES = frozenset({"bar", "line", "scatter", "effectScatter", "pie", "funnel"})
_VOID = frozenset("area base br col embed hr img input link meta param source track wbr".split())
_HIDDEN_TEXT = frozenset({"script", "style", "title", "template", "noscript", "head"})
# Elements that end a line of text; an inline element, such as <b>, does not split a phrase.
_BLOCKS = frozenset(
    "address article aside blockquote br dd details div dl dt figcaption figure footer h1 h2 h3 h4 h5 h6 "
    "header hr li main nav ol p pre section summary table tbody td tfoot th thead tr ul".split()
)
_FORBIDDEN_ELEMENTS = {
    "link": "a <link> cannot load anything in the sandbox",
    "iframe": "a nested frame cannot load in the sandbox",
    "object": "an <object> cannot load in the sandbox",
    "embed": "an <embed> cannot load in the sandbox",
    "form": "a <form> cannot submit in the sandbox",
}
_URL_ATTRIBUTES = frozenset(
    "src href action data poster srcset xlink:href formaction ping background".split()
)
# What an author script must not touch: each is refused, or throws, inside the sandbox.
_FORBIDDEN_CALLS = (
    "localStorage",
    "sessionStorage",
    "document.cookie",
    "fetch(",
    "XMLHttpRequest",
    "WebSocket",
    "window.open",
    "alert(",
    "confirm(",
    "prompt(",
    "eval(",
    "new Function",
    "importScripts",
    "navigator.sendBeacon",
)
_DRAWING_HINTS = (
    re.compile(r"createElementNS\s*\([^)]*2000/svg", re.IGNORECASE),
    re.compile(r"<canvas", re.IGNORECASE),
    re.compile(r"getContext\s*\("),
    re.compile(r"\b(niceTicks|nice_ticks|lineChart|barChart|hbars|drawAxis|drawChart)\b"),
)
_PALETTES = echarts_render.PALETTES

# Subject-unrelated declarations a model likes to add. A source line, an as-of date, the method and
# the assumptions of THIS analysis are content and match none of these: every pattern names the
# declaration itself, never a word that real analysis also uses ("合规风险", "联合声明").
_BOILERPLATE: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "an AI or model credit",
        (
            r"AI\s*(?:生成|撰写|编写|创作|辅助生成)",
            r"(?:人工智能|大模型|大语言模型)\s*(?:生成|撰写|编写)",
            r"由[^，。；\n]{0,12}(?:AI|人工智能|大模型|大语言模型|LLM)[^，。；\n]{0,8}(?:生成|撰写|编写|创作)",
            r"本(?:报告|页面|文档|网页|内容)由[^，。；\n]{0,20}(?:生成|撰写|编写|编制|出具|制作)",
            r"AI[- ]generated",
            r"generated (?:by|with) (?:an? )?(?:AI|LLM|language model|ChatGPT|Claude|GPT)",
            r"(?:written|created|produced) (?:by|with) (?:an? )?(?:AI|LLM|ChatGPT|Claude)",
        ),
    ),
    (
        "a disclaimer",
        (
            r"仅供(?:参考|学习|研究|交流|内部)",
            r"不构成[^，。；\n]{0,10}(?:建议|承诺|要约|邀约)",
            r"免责(?:声明|条款)",
            r"disclaimer",
            r"for (?:reference|informational|information|educational) purposes only",
            r"not (?:an? )?(?:financial|investment|legal|medical|tax) advice",
        ),
    ),
    (
        "a privacy, compliance or copyright notice",
        (
            r"隐私(?:政策|声明|保护|条款)",
            r"个人信息保护",
            r"cookies?\b",
            r"GDPR",
            r"合规(?:声明|披露|说明)",
            r"privacy (?:policy|notice|statement)",
            r"版权(?:所有|声明)",
            r"©",
            r"all rights reserved",
        ),
    ),
    (
        "a footer credit",
        (
            r"生成(?:时间|于)",
            r"generated (?:on|at)\b",
            r"powered by",
            r"built with",
            r"技术支持[:：]",
        ),
    ),
    (
        "a call to action",
        (
            r"关注(?:我们|公众号)",
            r"欢迎(?:订阅|关注|转发|分享|联系)",
            r"联系我们",
            r"扫码",
            r"\bsubscribe\b",
            r"follow us",
            r"contact us",
        ),
    ),
)
_BOILERPLATE_RULES = tuple(
    (family, re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE))
    for family, patterns in _BOILERPLATE
)

_FONT_SIZE = re.compile(r"font-size\s*:\s*([\d.]+)\s*(px|pt|rem|em)\b", re.IGNORECASE)
_CSS_URL = re.compile(r"url\(\s*(['\"]?)(.*?)\1\s*\)", re.IGNORECASE)
_CSS_IMPORT = re.compile(r"@import\s+(?:url\()?\s*['\"]?([^'\")\s;]+)", re.IGNORECASE)
_FORMATTER_JS = re.compile(r'"formatter"\s*:\s*"\s*(function\b|\(?[\w\s,]*\)?\s*=>)')
_CJK = re.compile(r"[\u3400-\u9fff]")


class ReportError(Exception):
    """The fragment cannot become a report; the message says what to change."""


@dataclass
class _Element:
    tag: str
    attrs: dict[str, str]
    line: int


@dataclass
class _Page:
    id: str
    title: str
    line: int
    charts: int = 0
    tables: int = 0


@dataclass
class _Fragment:
    """What the parser found in the fragment, with the lines to point the author at."""

    figures: list[_Element] = field(default_factory=list)
    blocks: list[_Element] = field(default_factory=list)
    block_text: dict[int, str] = field(default_factory=dict)
    slicers: list[_Element] = field(default_factory=list)
    pages: list[_Page] = field(default_factory=list)
    scripts: list[tuple[str, int]] = field(default_factory=list)
    styles: list[tuple[str, int]] = field(default_factory=list)
    elements: list[_Element] = field(default_factory=list)
    text: list[tuple[str, int]] = field(default_factory=list)
    h1: str = ""


class _Parser(HTMLParser):
    """Walk the fragment once and keep what the checks need; the tree itself is not kept."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.found = _Fragment()
        self._stack: list[_Element] = []
        self._pages: list[tuple[int, _Page]] = []
        self._script: _Element | None = None
        self._script_text: list[str] = []
        self._style_line = 0
        self._style_text: list[str] = []
        self._h1_open = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        element = _Element(tag, {k: v or "" for k, v in attrs}, self.getpos()[0])
        self.found.elements.append(element)
        attributes = element.attrs
        if tag == "script":
            self._script, self._script_text = element, []
        elif tag == "style":
            self._style_line, self._style_text = element.line, []
        if "data-chart" in attributes and tag == "figure":
            self.found.figures.append(element)
            if self._pages:
                self._pages[-1][1].charts += 1
        if tag == "table" and self._pages:
            self._pages[-1][1].tables += 1
        if "data-slicer" in attributes:
            self.found.slicers.append(element)
        if "data-page" in attributes:
            page = _Page(attributes["data-page"], attributes.get("data-title", ""), element.line)
            self.found.pages.append(page)
            self._pages.append((len(self._stack), page))
        if tag == "h1" and not self.found.h1:
            self._h1_open = True
        if tag in _BLOCKS:
            self.found.text.append(("\n", element.line))
        if tag not in _VOID:
            self._stack.append(element)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in _VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag == "script" and self._script is not None:
            self._close_script()
        elif tag == "style":
            self.found.styles.append(("".join(self._style_text), self._style_line))
        elif tag == "h1":
            self._h1_open = False
        if tag in _BLOCKS:
            self.found.text.append(("\n", self.getpos()[0]))
        for depth in range(len(self._stack) - 1, -1, -1):
            if self._stack[depth].tag == tag:
                del self._stack[depth:]
                break
        while self._pages and self._pages[-1][0] >= len(self._stack):
            self._pages.pop()

    def _close_script(self) -> None:
        script, text = self._script, "".join(self._script_text)
        self._script = None
        if script is None:
            return
        kind = script.attrs.get("type", "").lower()
        if "src" in script.attrs:
            return
        if kind == "application/json":
            self.found.blocks.append(script)
            self.found.block_text[id(script)] = text
        elif kind in ("", "text/javascript", "module", "application/javascript"):
            self.found.scripts.append((text, script.line))

    def handle_data(self, data: str) -> None:
        if self._script is not None:
            self._script_text.append(data)
        elif self._stack and self._stack[-1].tag == "style":
            self._style_text.append(data)
        elif not any(e.tag in _HIDDEN_TEXT for e in self._stack):
            self.found.text.append((data, self.getpos()[0]))
            if self._h1_open and not self.found.h1:
                self.found.h1 = data.strip()


def _parse(source: str) -> _Fragment:
    parser = _Parser()
    parser.feed(source)
    parser.close()
    return parser.found


@dataclass
class _Chart:
    """One chart: its figure, its parsed block, and the slicers it lists."""

    id: str
    line: int
    figure: _Element | None
    block_line: int | None = None
    spec: dict[str, Any] | None = None

    @property
    def block(self) -> dict[str, Any]:
        """The parsed block; the checks only reach for it once it parsed."""
        if self.spec is None:
            raise ReportError(f'chart "{self.id}" has no usable JSON block')
        return self.spec

    @property
    def option(self) -> dict[str, Any]:
        return self.spec["option"] if self.spec else {}

    @property
    def filters(self) -> list[str]:
        return self.spec.get("filters", []) if self.spec else []

    @property
    def rows(self) -> list[dict[str, Any]]:
        dataset = self.option.get("dataset")
        first = dataset[0] if isinstance(dataset, list) and dataset else dataset
        source = first.get("source") if isinstance(first, dict) else None
        return [row for row in source if isinstance(row, dict)] if isinstance(source, list) else []

    @property
    def series(self) -> list[dict[str, Any]]:
        value = self.option.get("series")
        items = value if isinstance(value, list) else [] if value is None else [value]
        return [s for s in items if isinstance(s, dict)]


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return [] if value is None else [value]


@dataclass
class _Report:
    """The checked fragment: what was found, the errors, and the warnings."""

    fragment: _Fragment
    charts: dict[str, _Chart] = field(default_factory=dict)
    slicers: dict[str, dict[str, Any]] = field(default_factory=dict)
    datasets: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    used_datasets: set[str] = field(default_factory=set)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)


def _reject_constant(name: str) -> Any:
    raise ValueError(f"{name} is not a JSON value; write null")


def _read_datasets(report: _Report) -> None:
    """Read the shared ``data-*`` blocks: rows that several charts name with ``dataset.from``."""
    seen: dict[str, int] = {}
    for block in report.fragment.blocks:
        block_id = block.attrs.get("id", "")
        if not block_id.startswith("data-"):
            continue
        name = block_id.removeprefix("data-")
        if name in seen:
            report.error(f'duplicate block id "{block_id}" (lines {seen[name]} and {block.line})')
            continue
        seen[name] = block.line
        try:
            rows = json.loads(
                report.fragment.block_text[id(block)], parse_constant=_reject_constant
            )
        except ValueError as error:
            report.error(f'data block "{name}" (line {block.line}) is not valid JSON ({error})')
            continue
        if not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows):
            report.error(f'data block "{name}" (line {block.line}) must be a list of row objects')
            continue
        report.datasets[name] = rows


def _resolve_datasets(report: _Report, chart: _Chart) -> None:
    """Replace each ``dataset.from`` of a chart with the rows of the block it names."""
    entries = chart.block["option"].get("dataset")
    for entry in entries if isinstance(entries, list) else [entries]:
        if not isinstance(entry, dict) or "from" not in entry:
            continue
        name = entry.pop("from")
        if "source" in entry:
            report.error(f'chart "{chart.id}": dataset has both "from" and "source"; use one')
        elif name not in report.datasets:
            report.error(
                f'chart "{chart.id}": dataset.from "{name}" has no <script type="application/json" '
                f'id="data-{name}"> block; add it, or put the rows in dataset.source'
            )
        else:
            entry["source"] = report.datasets[name]
            report.used_datasets.add(name)


def _read_blocks(report: _Report) -> None:
    fragment = report.fragment
    seen: dict[str, int] = {}
    for figure in fragment.figures:
        chart_id = figure.attrs["data-chart"]
        if chart_id in seen:
            report.error(
                f'duplicate chart id "{chart_id}" (lines {seen[chart_id]} and {figure.line}); '
                "give every figure its own data-chart"
            )
            continue
        seen[chart_id] = figure.line
        report.charts[chart_id] = _Chart(chart_id, figure.line, figure)
    ids: dict[str, int] = {}
    for block in fragment.blocks:
        block_id = block.attrs.get("id", "")
        if not block_id.startswith("chart-"):
            continue
        chart_id = block_id.removeprefix("chart-")
        if chart_id in ids:
            report.error(
                f'duplicate block id "{block_id}" (lines {ids[chart_id]} and {block.line}); '
                "each chart has one JSON block"
            )
            continue
        ids[chart_id] = block.line
        chart = report.charts.get(chart_id)
        if chart is None:
            report.error(
                f'<script id="{block_id}"> (line {block.line}) has no <figure class="chart" '
                f'data-chart="{chart_id}">; add the figure where the chart goes, or remove the block'
            )
            continue
        chart.block_line = block.line
        try:
            spec = json.loads(fragment.block_text[id(block)], parse_constant=_reject_constant)
        except json.JSONDecodeError as error:
            # The error counts lines from the first character after the script tag.
            where = f"line {block.line + error.lineno - 1}, column {error.colno}"
            report.error(
                f'chart "{chart_id}" ({where}): the JSON block is not valid ({error.msg}); '
                "JSON allows no comments, trailing commas, NaN or functions"
            )
            continue
        except ValueError as error:
            report.error(
                f'chart "{chart_id}" (line {block.line}): the JSON block is not valid ({error}); '
                "JSON allows no comments, trailing commas, NaN or functions"
            )
            continue
        if not isinstance(spec, dict) or not isinstance(spec.get("option"), dict):
            report.error(
                f'chart "{chart_id}" (line {block.line}): the block must be an object with an '
                '"option" object'
            )
            continue
        chart.spec = spec
        _resolve_datasets(report, chart)
    for chart_id, chart in report.charts.items():
        if chart.block_line is None:
            report.error(
                f'<figure data-chart="{chart_id}"> (line {chart.line}) has no <script '
                f'type="application/json" id="chart-{chart_id}">; add the block or remove the figure'
            )


def _read_slicers(report: _Report) -> None:
    seen: dict[str, int] = {}
    for element in report.fragment.slicers:
        attrs, line = element.attrs, element.line
        slicer_id = attrs["data-slicer"]
        if not slicer_id:
            report.error(f"slicer (line {line}) has an empty data-slicer; give it an id")
            continue
        if slicer_id in seen:
            report.error(
                f'duplicate slicer id "{slicer_id}" (lines {seen[slicer_id]} and {line}); '
                "slicer ids must be unique"
            )
            continue
        seen[slicer_id] = line
        kind = attrs.get("data-type", "filter")
        mode = attrs.get("data-mode", "single")
        ui = attrs.get("data-ui", "auto")
        problems = []
        if kind not in ("filter", "metric"):
            problems.append(f'data-type "{kind}" must be filter or metric')
        if mode not in ("single", "multi"):
            problems.append(f'data-mode "{mode}" must be single or multi')
        if ui not in ("auto", "chips", "select"):
            problems.append(f'data-ui "{ui}" must be auto, chips or select')
        slicer: dict[str, Any] = {"id": slicer_id, "type": kind, "mode": mode, "line": line}
        if kind == "metric":
            options = _json_attribute(attrs.get("data-options"), problems, "data-options")
            good = (
                isinstance(options, list)
                and options
                and all(isinstance(o, dict) and o.get("label") and o.get("y") for o in options)
            )
            if not good:
                problems.append(
                    'a metric slicer needs data-options=\'[{"label":"销售额","y":"revenue"}, ...]\''
                )
            else:
                labels = [o["label"] for o in options]
                if len(set(labels)) != len(labels):
                    problems.append("data-options labels must be unique")
            slicer["options"] = options if good else []
        else:
            slicer["all"] = attrs.get("data-all") != "false"
            slicer["field"] = attrs.get("data-field", "")
            if not slicer["field"]:
                problems.append('a filter slicer needs data-field="the row field to filter on"')
            if attrs.get("data-values"):
                values = _json_attribute(attrs["data-values"], problems, "data-values")
                slicer["values"] = [str(v) for v in values] if isinstance(values, list) else []
        for problem in problems:
            report.error(f'slicer "{slicer_id}" (line {line}): {problem}')
        report.slicers[slicer_id] = slicer


def _json_attribute(raw: str | None, problems: list[str], name: str) -> Any:
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except ValueError as error:
        problems.append(f"{name} is not valid JSON ({error})")
        return None


def _row_fields(chart: _Chart) -> list[str]:
    fields: dict[str, None] = {}
    for row in chart.rows:
        fields.update(dict.fromkeys(row))
    return list(fields)


def _check_chart(report: _Report, chart: _Chart) -> None:
    if chart.spec is None:
        return
    name = f'chart "{chart.id}"'
    spec = chart.spec
    unknown = sorted(set(spec) - {"option", "filters", "aspect", "minHeight", "orient"})
    if unknown:
        report.warn(
            f"{name}: unknown key {', '.join(map(repr, unknown))}; the block takes option, "
            "filters, aspect, minHeight and orient"
        )
    filters = spec.get("filters", [])
    if not isinstance(filters, list) or not all(isinstance(f, str) for f in filters):
        report.error(f'{name}: "filters" must be a list of slicer ids')
        return
    for slicer_id in filters:
        if slicer_id not in report.slicers:
            report.error(
                f'{name} lists filter "{slicer_id}", but no slicer declares it; add '
                f'<div class="slicer" data-slicer="{slicer_id}" data-field="..."></div>'
            )
    if spec.get("orient", "auto") not in ("auto", "keep"):
        report.error(f'{name}: "orient" must be auto or keep')
    aspect = spec.get("aspect", "16:9")
    if not re.fullmatch(r"\d+(\.\d+)?:\d+(\.\d+)?", str(aspect)):
        report.error(f'{name}: "aspect" must look like "16:9", not {json.dumps(aspect)}')
    min_height = spec.get("minHeight", 240)
    if isinstance(min_height, bool) or not isinstance(min_height, int | float) or min_height <= 0:
        report.error(f'{name}: "minHeight" must be a number of pixels')
    option = chart.option
    palette = option.get("palette", "categorical")
    if palette not in _PALETTES:
        report.error(f'{name}: "palette" must be one of {", ".join(_PALETTES)}')
    series = chart.series
    for position, entry in enumerate(series):
        if entry.get("type") not in _SERIES:
            report.error(
                f"{name}: series {position} has type {json.dumps(entry.get('type'))}, which cannot "
                "be drawn from JSON; use bar, line, pie, scatter and the other types in the charts "
                "Skill (custom and map are not available)"
            )
    text = json.dumps(option, ensure_ascii=False)
    if _FORMATTER_JS.search(text):
        report.error(
            f'{name}: a formatter must be a template string such as "{{b}}: {{c}}", not JavaScript'
        )
    if not chart.rows and not any(s.get("data") or s.get("nodes") for s in series):
        report.error(
            f"{name} has no data; put rows in option.dataset.source (a list of objects) or give "
            "a series its data"
        )
    _check_fields(report, chart)
    if "{metric}" in text and not any(
        report.slicers.get(f, {}).get("type") == "metric" for f in filters
    ):
        report.error(f'{name} uses {{metric}} but lists no metric slicer in "filters"')


def _fields_of(report: _Report, chart: _Chart, value: str) -> list[str]:
    """The row fields an encode value stands for: ``{metric}`` stands for every option's field."""
    if value != "{metric}":
        return [value]
    for slicer_id in chart.filters:
        slicer = report.slicers.get(slicer_id)
        if slicer and slicer["type"] == "metric":
            return [o["y"] for o in slicer["options"]]
    return []


def _check_fields(report: _Report, chart: _Chart) -> None:
    fields = _row_fields(chart)
    name = f'chart "{chart.id}"'
    known = f"the rows have {', '.join(fields)}" if fields else "the chart has no rows"
    for position, entry in enumerate(chart.series):
        wanted: list[tuple[str, str]] = []
        for key, value in (entry.get("encode") or {}).items():
            if isinstance(value, str) and key in ("x", "y", "itemName", "value"):
                wanted += [(f"encode.{key}", f) for f in _fields_of(report, chart, value)]
        if isinstance(entry.get("seriesBy"), str):
            wanted.append(("seriesBy", entry["seriesBy"]))
        if wanted and entry.get("type") not in _TIDY_SERIES and "seriesBy" in entry:
            report.error(
                f"{name}: series {position} uses seriesBy, which bar, line and scatter take"
            )
        for where, field_name in wanted:
            if field_name not in fields:
                report.error(
                    f'{name}: series {position} names field "{field_name}" in {where}, which no '
                    f"row has ({known})"
                )
    for slicer_id in chart.filters:
        slicer = report.slicers.get(slicer_id)
        if (
            slicer
            and slicer["type"] == "filter"
            and slicer["field"]
            and slicer["field"] not in fields
        ):
            report.error(
                f'slicer "{slicer_id}" (line {slicer["line"]}) filters on field "{slicer["field"]}", '
                f'which no row of chart "{chart.id}" has ({known})'
            )


def _expanded_series(chart: _Chart) -> int:
    rows = chart.rows
    count = 0
    for entry in chart.series:
        by = entry.get("seriesBy")
        if isinstance(by, str):
            order = entry.get("seriesOrder")
            values = order if isinstance(order, list) else {str(r.get(by)) for r in rows if by in r}
            count += len(values)
        else:
            count += 1
    return count


def _categories(chart: _Chart) -> list[str]:
    axis = _as_list(chart.option.get("xAxis"))
    if axis and isinstance(axis[0], dict) and isinstance(axis[0].get("data"), list):
        return [str(c) for c in axis[0]["data"]]
    seen: dict[str, None] = {}
    for entry in chart.series:
        encode = entry.get("encode") or {}
        field_name = encode.get("x") if isinstance(encode, dict) else None
        if isinstance(field_name, str):
            seen.update(dict.fromkeys(str(r[field_name]) for r in chart.rows if field_name in r))
    return list(seen)


def _warn_chart(report: _Report, chart: _Chart) -> None:
    if chart.spec is None:
        return
    name = f'chart "{chart.id}"'
    title = _as_list(chart.option.get("title"))
    first = title[0] if title and isinstance(title[0], dict) else {}
    if not first.get("text"):
        report.warn(f"{name} has no title.text; give it a title that says what it shows")
    if not first.get("subtext"):
        report.warn(
            f"{name} has no title.subtext; say the unit and the source in words, such as "
            '"单位：万元 · 来源：内部销售数据"'
        )
    if _expanded_series(chart) > 8:
        report.warn(
            f"{name} draws more than 8 series; keep the top ones and fold the rest into 其他"
        )
    y_axes = [a for a in _as_list(chart.option.get("yAxis")) if isinstance(a, dict)]
    if sum(a.get("type", "value") == "value" for a in y_axes) >= 2:
        report.warn(f"{name} has two value axes; use one, or split it into two charts")
    series = chart.series
    x_axis = _as_list(chart.option.get("xAxis"))
    vertical = (
        series
        and all(s.get("type") == "bar" for s in series)
        and x_axis
        and isinstance(x_axis[0], dict)
        and x_axis[0].get("type", "category") == "category"
        and not any(isinstance(a, dict) and a.get("type") == "category" for a in y_axes)
    )
    if vertical and chart.spec.get("orient", "auto") == "auto":
        categories = _categories(chart)
        if len(categories) > 8 or max((len(c) for c in categories), default=0) > 10:
            report.warn(
                f"{name} is a vertical bar chart with long or many category labels; the runtime lays "
                'it out horizontally on narrow widths (set "orient": "keep" to stop that)'
            )


def _check_markup(report: _Report) -> None:
    fragment = report.fragment
    for element in fragment.elements:
        tag, attrs, line = element.tag, element.attrs, element.line
        if tag in _FORBIDDEN_ELEMENTS:
            report.error(f"<{tag}> (line {line}): {_FORBIDDEN_ELEMENTS[tag]}; remove it")
        if tag == "a" and "download" in attrs:
            report.error(f"<a download> (line {line}) cannot save a file in the sandbox; remove it")
        if tag == "script" and "src" in attrs:
            report.error(
                f'<script src="{attrs["src"][:40]}"> (line {line}) cannot load in the sandbox; '
                "write the code inline"
            )
        for key, value in attrs.items():
            if key in _URL_ATTRIBUTES and value.strip() and not _allowed_url(value):
                report.error(
                    f'{key}="{value.strip()[:50]}" (line {line}) cannot work in the sandbox: only '
                    "data: URLs and #fragments do; write an address as plain text instead"
                )
            if key == "style":
                _check_css(report, value, line)
    for css, line in fragment.styles:
        _check_css(report, css, line)


def _allowed_url(value: str) -> bool:
    value = value.strip().lower()
    return value.startswith("data:") or value.startswith("#")


def _check_css(report: _Report, css: str, line: int) -> None:
    for _quote, target in _CSS_URL.findall(css):
        if target and not _allowed_url(target):
            report.error(
                f"url({target[:50]}) (line {line}) cannot load in the sandbox; use a data: URL "
                "or draw it with CSS"
            )
    for target in _CSS_IMPORT.findall(css):
        if not _allowed_url(target):
            report.error(
                f"@import {target[:50]} (line {line}) cannot load in the sandbox; inline the CSS"
            )
    for number, unit in _FONT_SIZE.findall(css):
        pixels = float(number) * {"px": 1.0, "pt": 4 / 3, "rem": 16.0, "em": 16.0}[unit.lower()]
        if pixels < 11:
            report.warn(
                f"text under 11px (font-size: {number}{unit}, line {line}); keep text at 11px or more"
            )


def _check_scripts(report: _Report) -> None:
    for text, line in report.fragment.scripts:
        for call in _FORBIDDEN_CALLS:
            if call in text:
                at = line + text[: text.index(call)].count("\n")
                report.error(
                    f'script (line {at}) uses "{call}", which the sandbox refuses; compute with '
                    "plain JavaScript and show results in the page"
                )
        for hint in _DRAWING_HINTS:
            if hint.search(text):
                report.warn(
                    f"script (line {line}) draws by hand ({hint.pattern.split('(')[0][:24]}...): "
                    "use an ECharts chart: the runtime draws, themes and resizes it"
                )
                break


def _check_pages(report: _Report) -> None:
    pages = report.fragment.pages
    seen: dict[str, int] = {}
    for page in pages:
        if page.id in seen:
            report.error(
                f'duplicate page id "{page.id}" (lines {seen[page.id]} and {page.line}); '
                "data-page values must be unique"
            )
        seen.setdefault(page.id, page.line)
        if not page.charts and not page.tables:
            report.warn(
                f'page "{page.id}" (line {page.line}) has no chart and no table; fold it into another page'
            )
    if len(pages) == 1:
        report.warn(
            f'only one section declares data-page ("{pages[0].id}", line {pages[0].line}); pages need '
            "at least two, so remove the attribute or add the others"
        )


def _check_boilerplate(report: _Report) -> None:
    joined = ""
    starts: list[tuple[int, int]] = []
    for text, line in report.fragment.text:
        starts.append((len(joined), line))
        joined += text
    for family, rule in _BOILERPLATE_RULES:
        for match in rule.finditer(joined):
            line = next((ln for at, ln in reversed(starts) if at <= match.start()), 0)
            report.warn(
                f'boilerplate: "{match.group(0).strip()}" (line {line}) is {family}; delete it, '
                "the reader did not ask for it"
            )


def _check_usage(report: _Report) -> None:
    for name in report.datasets.keys() - report.used_datasets:
        report.warn(f'data block "{name}" is used by no chart; remove it, or add dataset.from')
    listed = {f for chart in report.charts.values() for f in chart.filters}
    for slicer_id, slicer in report.slicers.items():
        if slicer_id not in listed:
            report.warn(
                f'slicer "{slicer_id}" (line {slicer["line"]}) is listed in no chart\'s "filters"; '
                "list it where it should apply, or remove it"
            )


def _slicer_defs(report: _Report, chart: _Chart) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for slicer_id in chart.filters:
        slicer = report.slicers.get(slicer_id)
        if slicer is not None:
            out[slicer_id] = {k: v for k, v in slicer.items() if k != "line"}
    return out


def _runtime_spec(report: _Report, chart: _Chart) -> dict[str, Any]:
    return {**chart.block, "id": chart.id, "slicers": _slicer_defs(report, chart)}


def _node(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run chart jobs through the node renderer: the browser's own ``prepare``, drawn as SVG."""
    try:
        drawn = subprocess.run(  # noqa: S603 - argv list, no shell, a fixed executable
            ["node", str(_REPORT / "preview.cjs")],  # noqa: S607 - node is on the image's PATH
            input=json.dumps({"jobs": jobs}),
            capture_output=True,
            text=True,
            timeout=300,
        )
    except FileNotFoundError as error:
        raise ReportError("node is not on PATH, and the charts are checked with it") from error
    except subprocess.TimeoutExpired as error:
        raise ReportError(
            "drawing the charts took more than five minutes; cut their rows"
        ) from error
    if drawn.returncode:
        raise ReportError(f"the chart renderer failed: {drawn.stderr.strip()[:300]}")
    return json.loads(drawn.stdout)


def _check_drawing(report: _Report) -> None:
    """Draw every chart at 360 and 900 for each state of its first filter, and keep ECharts' message."""
    jobs = [
        {
            "id": c.id,
            "spec": _runtime_spec(report, c),
            "widths": list(_PREVIEW_WIDTHS),
            "sweep": True,
            "svg": False,
        }
        for c in report.charts.values()
        if c.spec is not None
    ]
    if not jobs:
        return
    reported: set[tuple[str, str]] = set()
    for result in _node(jobs):
        chart_id = result["id"]
        where = f'chart "{chart_id}" at {result["width"]}px'
        if result.get("state"):
            where += f" with {json.dumps(result['state'], ensure_ascii=False)}"
        if "error" in result and (chart_id, result["error"]) not in reported:
            reported.add((chart_id, result["error"]))
            report.error(
                f"{where}: ECharts cannot draw it ({result['error'][:200]}); fix the option"
            )
        elif result.get("empty") and not result.get("state"):
            key = (chart_id, "empty")
            if key not in reported:
                reported.add(key)
                report.error(
                    f'chart "{chart_id}" draws nothing at its default slicer state; check that its '
                    "rows and series fields agree"
                )


def _check(source: str) -> _Report:
    report = _Report(_parse(source))
    _read_datasets(report)
    _read_blocks(report)
    _read_slicers(report)
    for chart in report.charts.values():
        _check_chart(report, chart)
    _check_markup(report)
    _check_scripts(report)
    _check_pages(report)
    _check_usage(report)
    for chart in report.charts.values():
        _warn_chart(report, chart)
    _check_boilerplate(report)
    if not report.errors:
        _check_drawing(report)
    return report


def _escape_script(text: str) -> str:
    return text.replace("</script", "<\\/script")


def _runtime() -> str:
    """The runtime script: theme.js, core.js and runtime.js in one function, plus the theme data."""
    data = {
        "structure": json.loads((_HERE / "theme.json").read_text(encoding="utf-8")),
        "palette": json.loads((_HERE / "palette.json").read_text(encoding="utf-8")),
    }
    theme = (_HERE / "theme.js").read_text(encoding="utf-8")
    core = (_REPORT / "core.js").read_text(encoding="utf-8")
    runtime = (_REPORT / "runtime.js").read_text(encoding="utf-8")
    return _escape_script(
        "(function(){'use strict';\n"
        "const __mods={};\n"
        "const require=(name)=>__mods[name.slice(name.lastIndexOf('/')+1)];\n"
        "const __def=(name,factory)=>{const module={exports:{}};factory(module,module.exports);"
        "__mods[name]=module.exports;};\n"
        f"__def('theme.js',(module,exports)=>{{{theme}\n}});\n"
        f"__def('core.js',(module,exports)=>{{{core}\n}});\n"
        f"const DATA={echarts_render.embed_json(data)};\n"
        f"{runtime}\n"
        "})();"
    )


def _document(fragment: str, found: _Fragment, title: str | None) -> str:
    visible = "".join(text for text, _ in found.text)
    han, latin = len(_CJK.findall(visible)), len(re.findall(r"[A-Za-z]", visible))
    chinese = han >= 2 and han >= 0.15 * (han + latin)
    try:
        library = _escape_script(echarts_render.echarts_library().read_text(encoding="utf-8"))
    except echarts_render.RenderError as error:
        raise ReportError(str(error)) from error
    css = (_REPORT / "report.css").read_text(encoding="utf-8")
    heading = (title or found.h1 or "Report").replace("&", "&amp;").replace("<", "&lt;")
    return (
        '<!doctype html>\n<html lang="' + ("zh-CN" if chinese else "en") + '">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<meta name="color-scheme" content="light dark">\n'
        f"<title>{heading}</title>\n"
        f"<style>\n{css}</style>\n"
        f"<script>{library}</script>\n"
        f"<script>\n{_runtime()}\n</script>\n"
        '</head>\n<body>\n<div class="report">\n'
        f"{fragment}\n"
        "</div>\n</body>\n</html>\n"
    )


def _preview(report: _Report, directory: Path, dark: bool) -> None:
    """Write DIR/<chart-id>-360.png and -900.png: each chart body at that container width."""
    jobs = [
        {
            "id": chart.id,
            "spec": _runtime_spec(report, chart),
            "width": width,
            "mode": "dark" if dark else "light",
        }
        for chart in report.charts.values()
        if chart.spec is not None
        for width in _PREVIEW_WIDTHS
    ]
    for result in _node(jobs):
        if "error" in result:
            raise ReportError(f'chart "{result["id"]}": {result["error"]}')
        target = directory / f"{result['id']}-{result['width']}.png"
        try:
            missing = echarts_render.rasterize(
                result["svg"], target, _PREVIEW_SCALE[result["width"]]
            )
        except echarts_render.RenderError as error:
            raise ReportError(f"preview: {error}") from error
        if missing:
            print(
                f"html-report: warning: no font has {missing}; those characters show as boxes",
                file=sys.stderr,
            )


def build(source: Path, out: Path, title: str | None, preview: Path | None, dark: bool) -> int:
    """Check the fragment and write the report; return the process exit code."""
    try:
        fragment = source.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        print(f"html-report: cannot read {source}: {error}", file=sys.stderr)
        return 1
    try:
        report = _check(fragment)
        for warning in report.warnings:
            print(f"html-report: warning: {warning}", file=sys.stderr)
        if report.errors:
            shown = (
                report.errors
                if len(report.errors) <= _MAX_ERRORS
                else report.errors[: _MAX_ERRORS - 1]
            )
            for error in shown:
                print(f"html-report: {error}", file=sys.stderr)
            if len(shown) < len(report.errors):
                print(
                    f"html-report: and {len(report.errors) - len(shown)} more problems; fix these first",
                    file=sys.stderr,
                )
            return 1
        if preview is not None:
            _preview(report, preview, dark)
        document = _document(fragment, report.fragment, title)
    except ReportError as error:
        print(f"html-report: {error}", file=sys.stderr)
        return 1
    size = len(document.encode("utf-8"))
    if size > _MAX_BYTES:
        print(
            f"html-report: the report is {size / 2**20:.1f} MiB, over the 20 MiB artifact limit; "
            "cut the rows embedded in the charts",
            file=sys.stderr,
        )
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(document, encoding="utf-8")
    pages = len(report.fragment.pages) if len(report.fragment.pages) > 1 else 1
    print(
        f"html-report: wrote {out} ({size / 2**20:.1f} MiB, {pages} page{'s' if pages != 1 else ''}, "
        f"{len(report.charts)} charts, {len(report.slicers)} slicers, {len(report.warnings)} warnings)"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="html-report", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build", help="check a fragment and build the report")
    build_parser.add_argument("source", type=Path, help="the HTML fragment")
    build_parser.add_argument("out", type=Path, help="the report to write")
    build_parser.add_argument("--title", help="the document title (default: the first h1)")
    build_parser.add_argument(
        "--preview", type=Path, help="write each chart as PNG into this directory"
    )
    build_parser.add_argument(
        "--preview-dark", action="store_true", help="draw the previews in Mineral dark"
    )
    args = parser.parse_args(argv)
    return build(args.source, args.out, args.title, args.preview, args.preview_dark)


if __name__ == "__main__":
    sys.exit(main())
