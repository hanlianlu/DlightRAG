#!/usr/bin/env python3
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Build a self-contained interactive HTML report from a fragment of plain HTML.

The image installs this file as ``html-report`` for the built-in ``interactive-html`` Skill. The
fragment holds the report's prose, its slicers and pages, and one JSON block per chart; the build
checks it, then writes one document with the full ECharts build, the Mineral stylesheet and the
runtime inlined, which runs inside the Artifact sandbox without a network. ``--preview`` draws
every chart to PNG through the same code the browser runs, so the author can look before it ships.

The document carries the fragment it was built from, deflated in its head, and ``source`` writes
that back: an Agent Workspace does not outlive its Run and a Resource cannot be edited, so a later
turn that must change a report would otherwise have no source to edit.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import re
import subprocess
import sys
import zlib
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
# The most series the palette tells apart by colour alone, measured under colour-vision deficiency.
_COLOUR_SERIES = 6
# ECharts writes {a} {b} {c} {d} {e} and {value} in its own templates, so no slicer may be called so.
_ECHARTS_PLACEHOLDERS = frozenset({"a", "b", "c", "d", "e", "value"})
_SLICER_TYPES = ("filter", "metric", "slider")
_TOKEN = re.compile(r"\{([^{}|]+)\}")
_PREVIEW_WIDTHS = (360, 900)
_PREVIEW_SCALE = {360: 2, 900: 1}
# The fragment a report was built from rides in the report under this id; `source` reads it back.
_SOURCE_ID = "report-source"
# What the embedded block says of itself: its version, how the fragment is encoded, and the digest
# that proves a round trip. A later version may add a codec; an older reader refuses rather than guesses.
_SOURCE_FORMAT = 1
_SOURCE_CODEC = "zlib+base64"

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
#: Calls the sandbox refuses: (the name the refusal prints, the pattern that matches the call).
#: The pattern starts at an identifier boundary, so a longer word that happens to end in one —
#: `prefetch(` is not `fetch(`, `retrieval()` is not `eval(` — is the author's own function, not the
#: refused API.
_FORBIDDEN_CALLS = (
    ("localStorage", re.compile(r"(?<![\w$.])localStorage\b")),
    ("sessionStorage", re.compile(r"(?<![\w$.])sessionStorage\b")),
    ("document.cookie", re.compile(r"document\s*\.\s*cookie\b")),
    ("fetch(", re.compile(r"(?<![\w$.])fetch\s*\(")),
    ("XMLHttpRequest", re.compile(r"(?<![\w$.])XMLHttpRequest\b")),
    ("WebSocket", re.compile(r"(?<![\w$.])WebSocket\b")),
    ("window.open", re.compile(r"window\s*\.\s*open\s*\(")),
    ("alert(", re.compile(r"(?<![\w$.])alert\s*\(")),
    ("confirm(", re.compile(r"(?<![\w$.])confirm\s*\(")),
    ("prompt(", re.compile(r"(?<![\w$.])prompt\s*\(")),
    ("eval(", re.compile(r"(?<![\w$.])eval\s*\(")),
    ("new Function", re.compile(r"new\s+Function\b")),
    ("importScripts", re.compile(r"(?<![\w$.])importScripts\b")),
    ("navigator.sendBeacon", re.compile(r"navigator\s*\.\s*sendBeacon\b")),
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
    # The data-page section this element sits in, or None when it sits outside every page.
    page: str | None = None


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
        enclosing = self._pages[-1][1].id if self._pages else None
        element = _Element(tag, {k: v or "" for k, v in attrs}, self.getpos()[0], enclosing)
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
    # The page the figure sits in, or None outside every page.
    page: str | None = None
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
    def unfiltered(self) -> list[str]:
        """The slicers the chart says it does not obey, however the block writes them down."""
        raw = self.spec.get("unfiltered", []) if self.spec else []
        if isinstance(raw, dict):
            return [key for key in raw if isinstance(key, str)]
        return [item for item in raw if isinstance(item, str)] if isinstance(raw, list) else []

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
        report.charts[chart_id] = _Chart(chart_id, figure.line, figure, figure.page)
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
        if slicer_id in _ECHARTS_PLACEHOLDERS:
            problems.append(
                f'"{slicer_id}" is one of ECharts\' own template placeholders ({{a}} {{b}} {{c}} '
                "{d} {e} {value}), which a {id} token would replace; choose a longer id"
            )
        if kind not in _SLICER_TYPES:
            problems.append(f'data-type "{kind}" must be filter, metric or slider')
        if mode not in ("single", "multi"):
            problems.append(f'data-mode "{mode}" must be single or multi')
        if ui not in ("auto", "chips", "select"):
            problems.append(f'data-ui "{ui}" must be auto, chips or select')
        slicer: dict[str, Any] = {"id": slicer_id, "type": kind, "mode": mode, "line": line}
        slicer["page"] = element.page
        slicer["label"] = attrs.get("data-label") or slicer_id
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
        elif kind == "slider":
            _read_slider(attrs, slicer, problems)
        else:
            slicer["all"] = attrs.get("data-all") != "false"
            slicer["field"] = attrs.get("data-field", "")
            if attrs.get("data-values"):
                values = _json_attribute(attrs["data-values"], problems, "data-values")
                slicer["values"] = [str(v) for v in values] if isinstance(values, list) else []
            if not slicer["field"] and "values" not in slicer:
                problems.append(
                    'a filter slicer needs data-field="the row field to filter on", or '
                    'data-values=\'["a","b"]\' when only a script or a {id} token reads it'
                )
        for problem in problems:
            report.error(f'slicer "{slicer_id}" (line {line}): {problem}')
        report.slicers[slicer_id] = slicer


def _number_attribute(attrs: dict[str, str], name: str) -> float | None:
    raw = attrs.get(name, "").strip()
    try:
        value = float(raw)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _read_slider(attrs: dict[str, str], slicer: dict[str, Any], problems: list[str]) -> None:
    """Read a slider's range, step, default and unit; say what is wrong with any of them."""
    low = _number_attribute(attrs, "data-min")
    high = _number_attribute(attrs, "data-max")
    step = _number_attribute(attrs, "data-step")
    if low is None or high is None or step is None:
        problems.append(
            'a slider needs numbers in data-min, data-max and data-step, such as data-min="-1" '
            'data-max="2" data-step="0.25"'
        )
        return
    if high <= low:
        problems.append(f"data-max {attrs['data-max']} must be above data-min {attrs['data-min']}")
        return
    if step <= 0 or step > high - low:
        problems.append(
            f"data-step {attrs['data-step']} must be above 0 and no more than the range "
            f"({high - low:g}); it is the size of one notch"
        )
        return
    value = _number_attribute(attrs, "data-value")
    if "data-value" in attrs and value is None:
        problems.append(f'data-value "{attrs["data-value"]}" is not a number')
    elif value is not None and not low <= value <= high:
        problems.append(f"data-value {attrs['data-value']} is outside data-min..data-max")
    slicer.update(
        min=low,
        max=high,
        step=step,
        value=low if value is None else value,
        unit=attrs.get("data-unit", ""),
    )


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
    unknown = sorted(
        set(spec) - {"option", "filters", "unfiltered", "aspect", "minHeight", "orient"}
    )
    if unknown:
        report.warn(
            f"{name}: unknown key {', '.join(map(repr, unknown))}; the block takes option, "
            "filters, unfiltered, aspect, minHeight and orient"
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
        elif report.slicers[slicer_id]["type"] == "slider":
            report.error(
                f'{name} lists slider "{slicer_id}" in "filters", but a slider filters no rows; '
                f"write {{{slicer_id}}} in the chart's text, or compute the rows in a script"
            )
        elif (
            report.slicers[slicer_id]["type"] == "filter" and not report.slicers[slicer_id]["field"]
        ):
            report.error(
                f'{name} lists slicer "{slicer_id}", which has no data-field to filter its rows by'
            )
    _check_unfiltered(report, chart, name)
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
    _check_wiring(report, chart)
    if (
        "{metric}" in text
        and "metric" not in report.slicers
        and not any(report.slicers.get(f, {}).get("type") == "metric" for f in filters)
    ):
        report.error(f'{name} uses {{metric}} but lists no metric slicer in "filters"')


def _metric_slicer(report: _Report, chart: _Chart, name: str) -> dict[str, Any] | None:
    """The metric slicer a ``{name}`` token stands for: the one so named, or for ``{metric}`` the one the chart lists."""
    slicer = report.slicers.get(name)
    if slicer is None and name == "metric":
        slicer = next(
            (
                report.slicers[i]
                for i in chart.filters
                if report.slicers.get(i, {}).get("type") == "metric"
            ),
            None,
        )
    return slicer if slicer is not None and slicer["type"] == "metric" else None


def _fields_of(report: _Report, chart: _Chart, value: str) -> list[str]:
    """The row fields an encode value stands for: a metric slicer's token stands for every option's field."""
    token = _TOKEN.fullmatch(value)
    if token is None:
        return [value]
    slicer = _metric_slicer(report, chart, token.group(1))
    if slicer is not None:
        return [o["y"] for o in slicer["options"]]
    # {metric} with no metric slicer is reported once, by the chart check; any other value is a field name.
    return [] if token.group(1) == "metric" and "metric" not in report.slicers else [value]


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


def _check_unfiltered(report: _Report, chart: _Chart, name: str) -> None:
    """Read ``unfiltered``: the slicers the chart says it does not obey, and why, if it says why."""
    raw = chart.block.get("unfiltered", [])
    if isinstance(raw, dict):
        if not all(isinstance(reason, str) and reason.strip() for reason in raw.values()):
            report.error(
                f'{name}: every "unfiltered" reason must say something; drop the empty one'
            )
            return
    elif not isinstance(raw, list) or not all(isinstance(f, str) for f in raw):
        report.error(
            f'{name}: "unfiltered" must be a list of slicer ids, or an object of id to reason'
        )
        return
    fields = _row_fields(chart)
    for slicer_id in chart.unfiltered:
        slicer = report.slicers.get(slicer_id)
        if slicer is None:
            report.error(
                f'{name} is unfiltered by "{slicer_id}", but no slicer declares it; remove it, or '
                f'add <div class="slicer" data-slicer="{slicer_id}" data-field="..."></div>'
            )
        elif slicer_id in chart.filters:
            report.error(
                f'{name} names "{slicer_id}" in both "filters" and "unfiltered"; a chart either '
                "obeys a slicer or does not"
            )
        elif slicer["type"] != "filter" or not slicer["field"] or slicer["field"] not in fields:
            report.error(
                f'{name} is unfiltered by "{slicer_id}", which could never have reached it: say so '
                "only for a filter slicer whose field its rows carry"
            )


def _reaches(slicer: dict[str, Any], chart: _Chart) -> bool:
    """Whether the reader can work a slicer where the chart stands: shared anywhere, a page's own only there."""
    return slicer["page"] is None or chart.page is None or slicer["page"] == chart.page


def _check_wiring(report: _Report, chart: _Chart) -> None:
    """A chart that draws rows a slicer could narrow must say whether it obeys that slicer."""
    fields = _row_fields(chart)
    for slicer_id, slicer in report.slicers.items():
        if (
            slicer["type"] != "filter"
            or not slicer["field"]
            or slicer["field"] not in fields
            or slicer_id in chart.filters
            or slicer_id in chart.unfiltered
            or not _reaches(slicer, chart)
        ):
            continue
        report.error(
            f'chart "{chart.id}" (line {chart.line}) draws rows with field "{slicer["field"]}", '
            f'which slicer "{slicer_id}" (line {slicer["line"]}) filters, but its "filters" does not '
            f'list it; add "{slicer_id}" to "filters", or write "unfiltered": ["{slicer_id}"] if the '
            f"chart is meant to keep every {slicer['field']}"
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
    drawn = _expanded_series(chart)
    if drawn > _COLOUR_SERIES:
        report.warn(
            f"{name} draws {drawn} series, more than colour alone can tell apart ({_COLOUR_SERIES}); "
            "keep the top five and fold the rest into 其他, or highlight one series"
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
        for call, pattern in _FORBIDDEN_CALLS:
            match = pattern.search(text)
            if match is not None:
                at = line + text[: match.start()].count("\n")
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
    written = " ".join(
        json.dumps(chart.option, ensure_ascii=False) for chart in report.charts.values()
    )
    scripts = "\n".join(text for text, _ in report.fragment.scripts)
    for slicer_id, slicer in report.slicers.items():
        named_by_script = re.search(rf"""['"`]{re.escape(slicer_id)}['"`]""", scripts)
        if slicer_id not in listed and f"{{{slicer_id}}}" not in written and not named_by_script:
            report.warn(
                f'slicer "{slicer_id}" (line {slicer["line"]}) is used by nothing; list it in a '
                f"chart's \"filters\", write {{{slicer_id}}} in a chart's text, or read it in a "
                f"script with Report.slicer('{slicer_id}')"
            )


def _slicer_defs(report: _Report, chart: _Chart) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for slicer_id in chart.filters:
        slicer = report.slicers.get(slicer_id)
        if slicer is not None:
            out[slicer_id] = {k: v for k, v in slicer.items() if k not in ("line", "page")}
    return out


def _runtime_spec(report: _Report, chart: _Chart) -> dict[str, Any]:
    return {**chart.block, "id": chart.id, "slicers": _slicer_defs(report, chart)}


def _is_chinese(found: _Fragment) -> bool:
    """Whether the fragment's own words are Chinese: han characters are a real share of its letters."""
    visible = "".join(text for text, _ in found.text)
    han, latin = len(_CJK.findall(visible)), len(re.findall(r"[A-Za-z]", visible))
    return han >= 2 and han >= 0.15 * (han + latin)


#: The document language a caller declares is a BCP-47-shaped tag. The toolkit's own runtime words
#: ship in English and Chinese; any other tag is written on `<html lang>` and read back by `check`,
#: while the runtime label strings fall back to English. The page's chrome language is the caller's
#: business: a projected report ships the English pack around a small-language reader's prose.
_LANGUAGE_TAG = re.compile(r"^[A-Za-z]{2,3}(?:-[A-Za-z0-9]{1,8})*$")


def _language_tag(value: object) -> str:
    """A BCP-47-shaped language tag as given, trimmed, or ``''`` when the value is not one."""
    if not isinstance(value, str):
        return ""
    text = value.strip().replace("_", "-")
    return text if _LANGUAGE_TAG.fullmatch(text) else ""


def _declared_language(lang: str | None) -> str:
    """The document's declared language tag, or ``''`` when the caller declared none."""
    return _language_tag(lang)


def _is_chinese_document(found: _Fragment, lang: str | None = None) -> bool:
    """Whether the document is Chinese: the declared language wins over the fragment's own words."""
    declared = _declared_language(lang)
    if declared:
        return declared.lower().startswith("zh")
    return _is_chinese(found)


#: The document's own declared language, written into its `<html lang>` at build time. `check` reads
#: it back so a page built with a declared language rebuilds to itself; without it the fragment's own
#: words decide, as they did before the flag existed.
_DOCUMENT_LANG = re.compile(r"<html\b[^>]*\blang\s*=\s*(?:\"([^\"]*)\"|'([^']*)')", re.IGNORECASE)


def _document_language(document: str) -> str | None:
    """The BCP-47 tag a built document declares on its `<html>` tag, or None if it declares none."""
    match = _DOCUMENT_LANG.search(document)
    if match is None:
        return None
    value = match.group(1) if match.group(1) is not None else match.group(2) or ""
    return _language_tag(value) or None


def _node_context(report: _Report, lang: str | None = None) -> dict[str, Any]:
    """What every chart job carries so a {id} token reads as it does in the browser."""
    return {
        "slicers": {
            slicer_id: {k: v for k, v in slicer.items() if k not in ("line", "page")}
            for slicer_id, slicer in report.slicers.items()
        },
        "lang": "zh" if _is_chinese_document(report.fragment, lang) else "en",
    }


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


def _check_drawing(report: _Report, lang: str | None = None) -> None:
    """Draw every chart at 360 and 900 for each state of its first filter, and keep ECharts' message."""
    jobs = [
        {
            "id": c.id,
            "spec": _runtime_spec(report, c),
            "widths": list(_PREVIEW_WIDTHS),
            "sweep": True,
            "svg": False,
            **_node_context(report, lang),
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


def _check(source: str, lang: str | None = None) -> _Report:
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
        _check_drawing(report, lang)
    if not report.errors:
        _check_wiring_responds(report, lang)
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


def _embed_source(fragment: str, title: str | None) -> str:
    """The fragment, deflated and base64-encoded, as the block a report carries in its head.

    Deflated because the fragment holds the rows, and the report has a size limit; base64 because
    the fragment holds anything a document may, including the end tag of a script. The digest is
    what makes the round trip a fact rather than a hope.
    """
    raw = fragment.encode("utf-8")
    payload = {
        "format": _SOURCE_FORMAT,
        "codec": _SOURCE_CODEC,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "title": title,
        "source": base64.b64encode(zlib.compress(raw, 9)).decode("ascii"),
    }
    return (
        f'<script type="application/json" id="{_SOURCE_ID}">'
        f"{_escape_script(json.dumps(payload, ensure_ascii=False))}</script>"
    )


def _read_source(document: str) -> dict[str, Any]:
    """The fragment a built report carries, verified against the digest the block states."""
    match = re.search(
        rf'<script type="application/json" id="{_SOURCE_ID}">(.*?)</script>', document, re.DOTALL
    )
    if match is None:
        raise ReportError(
            f"this document carries no {_SOURCE_ID} block, so the fragment it was built from is not "
            "in it; a report built before the source was embedded has none, and a hand-edited one "
            "may have lost it"
        )
    try:
        payload = json.loads(match.group(1))
    except ValueError as error:
        raise ReportError(
            f"the {_SOURCE_ID} block is not valid JSON ({error}); build the report again"
        ) from error
    if not isinstance(payload, dict):
        raise ReportError(f"the {_SOURCE_ID} block is not an object; build the report again")
    if payload.get("format") != _SOURCE_FORMAT or payload.get("codec") != _SOURCE_CODEC:
        raise ReportError(
            f"the {_SOURCE_ID} block is in format {payload.get('format')!r} with codec "
            f"{payload.get('codec')!r}, which this build does not read; use the html-report that "
            "wrote the report, or build it again"
        )
    try:
        raw = zlib.decompress(base64.b64decode(str(payload["source"]), validate=True))
    except (KeyError, ValueError, zlib.error) as error:
        raise ReportError(
            f"the {_SOURCE_ID} block is damaged ({error}); build the report again"
        ) from error
    if hashlib.sha256(raw).hexdigest() != payload.get("sha256"):
        raise ReportError(
            f"the {_SOURCE_ID} block does not match the digest it carries, so the document was "
            "edited after it was built; take the fragment from the report as it was published"
        )
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ReportError(f"the {_SOURCE_ID} block is not valid UTF-8 ({error})") from error
    return {"text": text, "title": payload.get("title"), "bytes": len(raw)}


def _document(fragment: str, found: _Fragment, title: str | None, lang: str | None = None) -> str:
    document_lang = _declared_language(lang) or ("zh-CN" if _is_chinese(found) else "en")
    try:
        library = _escape_script(echarts_render.echarts_library().read_text(encoding="utf-8"))
    except echarts_render.RenderError as error:
        raise ReportError(str(error)) from error
    css = (_REPORT / "report.css").read_text(encoding="utf-8")
    heading = (title or found.h1 or "Report").replace("&", "&amp;").replace("<", "&lt;")
    return (
        '<!doctype html>\n<html lang="' + document_lang + '">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<meta name="color-scheme" content="light dark">\n'
        f"<title>{heading}</title>\n"
        f"<style>\n{css}</style>\n"
        f"<script>{library}</script>\n"
        f"<script>\n{_runtime()}\n</script>\n"
        f"{_embed_source(fragment, title)}\n"
        '</head>\n<body>\n<div class="report">\n'
        f"{fragment}\n"
        "</div>\n</body>\n</html>\n"
    )


def _check_wiring_responds(report: _Report, lang: str | None = None) -> None:
    """Ask every chart that lists a slicer whether its drawing moves when that slicer does."""
    jobs = [
        {
            "id": chart.id,
            "spec": _runtime_spec(report, chart),
            "width": _PREVIEW_WIDTHS[-1],
            "sweep": True,
            "render": False,
            **_node_context(report, lang),
        }
        for chart in report.charts.values()
        if chart.spec is not None and chart.filters
    ]
    if not jobs:
        return
    standing: dict[str, str] = {}
    moved: dict[str, dict[str, tuple[Any, str]]] = {}
    for result in _node(jobs):
        if "error" in result:
            continue  # what ECharts cannot draw is already an error, from _check_drawing
        state = result.get("state")
        if not state:
            standing[result["id"]] = result["digest"]
            continue
        ((slicer_id, value),) = state.items()
        moved.setdefault(result["id"], {})[slicer_id] = (value, result["digest"])
    for chart_id, by_slicer in moved.items():
        for slicer_id, (value, digest) in by_slicer.items():
            if digest != standing.get(chart_id):
                continue
            report.warn(
                f'chart "{chart_id}" lists "{slicer_id}" but draws the same with it set to '
                f"{json.dumps(value, ensure_ascii=False)}; the slicer moves no row it draws, so "
                'point the chart at the rows it narrows, or drop it from "filters"'
            )


def _preview_name(result: dict[str, Any]) -> str:
    """A preview's file name: the chart and its width, plus the slicer state it was drawn in."""
    state = result.get("state")
    if not state:
        return f"{result['id']}-{result['width']}.png"
    ((slicer_id, value),) = state.items()
    if isinstance(value, list):
        value = "+".join(str(part) for part in value)
    chosen = re.sub(r"[^\w.+-]+", "_", str(value)).strip("_")
    return f"{result['id']}-{result['width']}@{slicer_id}={chosen}.png"


def _preview(report: _Report, directory: Path, dark: bool, lang: str | None = None) -> None:
    """Write each chart at DIR/<chart-id>-360.png and -900.png, plus its <chart>-900@<id>=<value> states."""
    jobs = [
        {
            "id": chart.id,
            "spec": _runtime_spec(report, chart),
            "width": width,
            # The wide preview also draws the chart at a non-default state of every slicer it lists,
            # so the author sees with their own eyes that it follows the controls.
            "sweep": width == _PREVIEW_WIDTHS[-1] and bool(chart.filters),
            "mode": "dark" if dark else "light",
            **_node_context(report, lang),
        }
        for chart in report.charts.values()
        if chart.spec is not None
        for width in _PREVIEW_WIDTHS
    ]
    for result in _node(jobs):
        if "error" in result:
            raise ReportError(f'chart "{result["id"]}": {result["error"]}')
        target = directory / _preview_name(result)
        try:
            missing = echarts_render.rasterize(
                result["svg"], target, _PREVIEW_SCALE[result["width"]]
            )
        except echarts_render.RenderError as error:
            # The report is the deliverable and the preview is an aid, so a rasteriser that is
            # missing (or a chart it cannot draw) skips the previews, says so in the warnings the
            # build already reports, and still writes the document. Losing a whole report to a
            # missing optional tool is the defect this replaces.
            report.warn(f"previews skipped: {error}")
            # The collected warnings were printed before the charts were drawn, so this one is said
            # here, in the same words and on the same stream; the closing summary still counts it.
            print(f"html-report: warning: previews skipped: {error}", file=sys.stderr)
            directory.mkdir(parents=True, exist_ok=True)
            continue
        if missing:
            print(
                f"html-report: warning: no font has {missing}; those characters show as boxes",
                file=sys.stderr,
            )


def build(
    source: Path,
    out: Path,
    title: str | None,
    preview: Path | None,
    dark: bool,
    lang: str | None = None,
) -> int:
    """Check the fragment and write the report; return the process exit code.

    ``lang`` is the document's declared BCP-47 language tag. A caller that owns the reader's language
    (a report projected from a case, for instance) passes it so the page's own ``<html lang>``, its
    runtime labels and its chart previews follow the reader rather than the share of Han characters
    the fragment happens to contain. Without it the fragment's own words decide, as before.
    """
    try:
        fragment = source.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        print(f"html-report: cannot read {source}: {error}", file=sys.stderr)
        return 1
    try:
        report = _check(fragment, lang)
        for warning in report.warnings:
            print(f"html-report: warning: {warning}", file=sys.stderr)
        if not report.charts:
            print(
                "html-report: warning: the fragment declares no chart; a report with no chart is "
                "not a delivered visual report (add a paired figure, or write Markdown instead)",
                file=sys.stderr,
            )
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
            _preview(report, preview, dark, lang)
        document = _document(fragment, report.fragment, title, lang)
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


def source(report: Path, out: Path) -> int:
    """Write the fragment a built report carries; return the process exit code."""
    try:
        document = report.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        print(f"html-report: cannot read {report}: {error}", file=sys.stderr)
        return 1
    try:
        found = _read_source(document)
    except ReportError as error:
        print(f"html-report: {error}", file=sys.stderr)
        return 1
    if str(out) == "-":
        sys.stdout.write(found["text"])
    else:
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(found["text"], encoding="utf-8")
        except OSError as error:
            print(f"html-report: cannot write {out}: {error}", file=sys.stderr)
            return 1
        print(
            f"html-report: wrote {out} ({found['bytes']} bytes, the fragment this report was built "
            "from); edit it and build again"
        )
    if found["title"]:
        print(
            f'html-report: the report was built with --title "{found["title"]}"; pass the same flag '
            "to keep the document title",
            file=sys.stderr,
        )
    return 0


def check(report: Path) -> int:
    """Verify a finished report: the fragment it carries rebuilds to it, and it holds a chart.

    Every refusal is one line naming the rebuild, because the Agent can still correct it. The
    comparison is byte-identical because the document is deterministic (ADR 0038): no footer, no
    timestamp, no mode from the environment.
    """
    try:
        document = report.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        print(f"html-report: cannot read {report}: {error}", file=sys.stderr)
        return 1
    try:
        found = _read_source(document)
        declared = _document_language(document)
        spec = _check(found["text"], declared)
    except ReportError as error:
        print(f"html-report: {error}", file=sys.stderr)
        return 1
    if spec.errors:
        for error in spec.errors[:_MAX_ERRORS]:
            print(f"html-report: {error}", file=sys.stderr)
        print(
            "html-report: the fragment this report carries does not build; edit report/src.html "
            "and run html-report build again",
            file=sys.stderr,
        )
        return 1
    if not spec.charts:
        print(
            "html-report: this report carries no chart; a page with no paired chart is not a "
            "delivered visual report. Add a paired chart and build again",
            file=sys.stderr,
        )
        return 1
    try:
        rebuilt = _document(found["text"], spec.fragment, found["title"], declared)
    except ReportError as error:
        print(f"html-report: {error}", file=sys.stderr)
        return 1
    if rebuilt != document:
        print(
            "html-report: this document was edited after it was built (or built by another "
            "html-report); edit the fragment and run html-report build again, never patch the "
            "report by hand",
            file=sys.stderr,
        )
        return 1
    pages = len(spec.fragment.pages) if len(spec.fragment.pages) > 1 else 1
    print(
        f"html-report: {report} is the toolkit's report ({len(spec.charts)} charts, "
        f"{len(spec.slicers)} slicers, {pages} page{'s' if pages != 1 else ''}), built from the "
        "fragment it carries"
    )
    return 0


_BUILD_RULES = """\
slicers (<div class="slicer" data-slicer="ID" ...>; ID is not a, b, c, d, e or value):
  filter (default)  data-field="row field" narrows the rows of every chart that obeys ID (below);
                    data-values='["a","b"]' fixes the choices (alone, when only a script or a {ID}
                    token reads it); data-mode="multi", data-all="false", data-ui="chips|select"
  metric            data-options='[{"label":"Sales","y":"revenue"}]' swaps the plotted field
  slider            data-min, data-max, data-step (numbers), data-value (default: data-min), data-unit

a filter slicer reaches every chart whose rows carry its field. List ID in the chart's "filters", or
name it in "unfiltered" (a list of ids, or an object of id to reason) when the chart keeps every value
on purpose: a chart that does neither is refused, and one that lists a slicer moving no row it draws
is warned about. An unfiltered chart says so under its title, so the reader is not left wondering why
it alone stands still. A slicer inside a data-page section reaches that page, and content outside
every page, and no other.

a slider is a real <input type=range>. Its value is a Number: Report.slicer(ID).value() returns it,
.set(n) clamps to min..max and snaps to the step. It is shown with as many decimals as data-step (or
data-min) has, a + before a positive number when data-min is below zero, then the unit: min -1,
step 0.25, unit pp shows 0.5 as +0.50pp. The report:slicer event fires on every input, once per
animation frame at most, with the latest value. A slider filters no rows: do not list it in "filters".

tokens: {ID} in any string of a chart's option, except the rows (dataset, a series' data), becomes
what slicer ID holds now: a slider's number as above, a metric's label, a filter's chosen value
(several joined by 、 or ", "; nothing chosen: 全部 or All, by the report's language).
{metric} also names the metric slicer the chart lists when no slicer has the id metric, and in a
series' encode a metric slicer's token is the row field of its chosen option. A {name} that is no
slicer id stays as written.

a slicer is in use when a chart lists it in "filters", a chart's text holds its {ID}, or an author
script names its id in quotes; the build warns about any other.
"""

_SOURCE_RULES = """\
every report carries the fragment it was built from, deflated in its head, so a later turn that must
change a report does not need the workspace that built it: reopen the published report (its artifact:
link names the resource_id), materialize it, and write the fragment back with

  html-report source report.html report/src.html

which writes it byte for byte, and `-` in place of the path writes it to standard output. Edit the
fragment and build again; the built file is never patched by hand. A document with no such block, one
whose block this build cannot read, or one whose bytes do not match the digest the block carries is
refused, and the call says which.
"""


_CHECK_RULES = """\
verify a finished report: it carries the fragment it was built from (html-report build embeds it),
that fragment rebuilds to these exact bytes, and the page holds at least one paired chart (a
<figure class="chart" data-chart="ID"> with its <script type="application/json" id="chart-ID">
block). Run it before attach_artifact: a document with no source block, a document edited after it
was built, and a page with no chart are refused, and the call names the rebuild that fixes it.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="html-report", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser(
        "build",
        help="check a fragment and build the report",
        epilog=_BUILD_RULES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    build_parser.add_argument("source", type=Path, help="the HTML fragment")
    build_parser.add_argument("out", type=Path, help="the report to write")
    build_parser.add_argument("--title", help="the document title (default: the first h1)")
    build_parser.add_argument(
        "--preview", type=Path, help="write each chart as PNG into this directory"
    )
    build_parser.add_argument(
        "--preview-dark", action="store_true", help="draw the previews in Mineral dark"
    )
    build_parser.add_argument(
        "--lang",
        default=None,
        help=(
            "the document's declared BCP-47 language tag, e.g. en, zh-CN or ja-JP (default: inferred "
            "from the fragment's own words); pass it when the caller knows the reader, so "
            "<html lang>, the runtime's labels and the previews follow the reader. The toolkit's "
            "runtime words ship in English and Chinese; any other tag is declared and read back, "
            "with the runtime label strings in English"
        ),
    )
    source_parser = commands.add_parser(
        "source",
        help="write the fragment a built report carries",
        epilog=_SOURCE_RULES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source_parser.add_argument("report", type=Path, help="a report this command built")
    source_parser.add_argument("out", type=Path, help="the fragment to write, or - for stdout")
    check_parser = commands.add_parser(
        "check",
        help="verify a finished report was built by the toolkit and carries a paired chart",
        epilog=_CHECK_RULES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    check_parser.add_argument("report", type=Path, help="a report this command built")
    args = parser.parse_args(argv)
    if getattr(args, "lang", None) is not None and not _language_tag(args.lang):
        parser.error(
            f"--lang {args.lang!r} is not a BCP-47 language tag (examples: en, zh-CN, ja-JP)"
        )
    if args.command == "source":
        return source(args.report, args.out)
    if args.command == "check":
        return check(args.report)
    return build(args.source, args.out, args.title, args.preview, args.preview_dark, args.lang)


if __name__ == "__main__":
    sys.exit(main())
