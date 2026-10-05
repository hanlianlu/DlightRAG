# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""``html-report build``: what it refuses, what it warns about, and the document it writes."""

import json
import re
from pathlib import Path
from typing import Any

import echarts_render
import html_report
import pytest
from PIL import Image

from tests.support.chart_tools import require_png_tools
from tests.support.report_examples import examples

_ROWS = [
    {"region": "华东", "quarter": "2023Q1", "revenue": 10, "profit": 1},
    {"region": "华东", "quarter": "2023Q2", "revenue": 12, "profit": 2},
    {"region": "华北", "quarter": "2023Q1", "revenue": 8, "profit": 1},
    {"region": "华北", "quarter": "2023Q2", "revenue": 9, "profit": 1},
]
_OPTION: dict[str, Any] = {
    "title": {"text": "季度销售额", "subtext": "单位：万元 · 来源：内部销售数据"},
    "dataset": {"source": _ROWS},
    "xAxis": {"type": "category"},
    "yAxis": {"type": "value"},
    "series": [{"type": "bar", "seriesBy": "region", "encode": {"x": "quarter", "y": "revenue"}}],
}
_REGION = '<div class="slicer" data-slicer="region" data-field="region" data-label="地区"></div>'
_METRIC = (
    '<div class="slicer" data-slicer="metric" data-type="metric" data-label="指标" '
    'data-options=\'[{"label":"销售额","y":"revenue"},{"label":"利润","y":"profit"}]\'></div>'
)


def chart(chart_id: str = "sales", option: dict[str, Any] | None = None, **block: Any) -> str:
    """A figure and its JSON block, as an author writes them."""
    spec = {"option": option if option is not None else _OPTION, **block}
    return (
        f'<figure class="chart" data-chart="{chart_id}"></figure>\n'
        f'<script type="application/json" id="chart-{chart_id}">{json.dumps(spec, ensure_ascii=False)}</script>\n'
    )


def report(*parts: str) -> str:
    return (
        '<header class="report-head"><h1>销售报告</h1></header>\n<main>\n'
        + "\n".join(parts)
        + "\n</main>\n"
    )


class Build:
    """The outcome of one build: exit code, what it printed and where it wrote."""

    def __init__(self, code: int, out: str, err: str, path: Path) -> None:
        self.code, self.out, self.err, self.path = code, out, err, path

    @property
    def errors(self) -> list[str]:
        return [
            line
            for line in self.err.splitlines()
            if line.startswith("html-report: ") and "warning:" not in line
        ]

    @property
    def warnings(self) -> list[str]:
        return [line for line in self.err.splitlines() if line.startswith("html-report: warning: ")]

    @property
    def document(self) -> str:
        return self.path.read_text(encoding="utf-8")


@pytest.fixture
def build(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    def run(source: str, *flags: str) -> Build:
        fragment, out = tmp_path / "report.src.html", tmp_path / "out" / "report.html"
        fragment.write_text(source, encoding="utf-8")
        code = html_report.main(["build", str(fragment), str(out), *flags])
        captured = capsys.readouterr()
        return Build(code, captured.out, captured.err, out)

    return run


def failed(result: Build, *needles: str) -> None:
    """A refusal: exit 1, no document, only ``html-report:`` lines, naming what to fix."""
    assert result.code == 1, result.err
    assert not result.path.exists()
    assert result.errors, result.err
    assert 0 < len(result.errors) <= 10
    assert all(
        line.startswith("html-report: ")
        for line in result.err.splitlines()
        if "warning:" not in line
    )
    for needle in needles:
        assert needle in "\n".join(result.errors), f"{needle!r} not in {result.errors}"


def warned(result: Build, *needles: str) -> None:
    """A warning: the build still succeeds, and the warning names what to change."""
    assert result.code == 0, result.err
    assert result.path.exists()
    for needle in needles:
        assert needle in "\n".join(result.warnings), f"{needle!r} not in {result.warnings}"


# ---- a valid fragment, and the document it becomes ---------------------------------------------


def test_a_valid_fragment_builds_and_prints_one_summary_line(build) -> None:
    result = build(report(_REGION, _METRIC, chart("sales", filters=["region"])))

    assert result.code == 0, result.err
    assert re.fullmatch(
        r"html-report: wrote \S+ \(\d+\.\d MiB, 1 page, 1 charts, 2 slicers, \d+ warnings\)\n",
        result.out,
    )
    assert result.out.count("\n") == 1


def test_the_document_is_complete_utf8_with_the_runtime_before_the_fragment(build) -> None:
    result = build(report(chart()))
    document = result.document

    assert document.startswith("<!doctype html>")
    assert '<meta charset="utf-8">' in document
    assert '<html lang="zh-CN">' in document
    assert "<title>销售报告</title>" in document
    assert result.path.read_bytes().decode("utf-8") == document
    assert document.index("window.Report") < document.index('data-chart="sales"')
    assert document.index("<style>") < document.index("<script>")
    assert document.count('id="chart-sales"') == 1
    assert "季度销售额" in document


def test_echarts_is_inlined_exactly_once(build) -> None:
    library = echarts_render.echarts_library().read_text(encoding="utf-8")
    marker = library[200:320]
    document = build(report(chart("a"), chart("b"))).document

    assert document.count(marker) == 1
    assert document.count("<script") == len(re.findall(r"</script>", document)) == 2 + 2


def test_an_end_tag_inside_inlined_text_cannot_end_its_script(
    build, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    library = tmp_path / "echarts.min.js"
    library.write_text("var echarts={};/* </script><b>x</b> */", encoding="utf-8")
    monkeypatch.setattr(echarts_render, "echarts_library", lambda: library)

    # The node check needs the real library, so the shape of the document is what is tested.
    text = html_report._escape_script(library.read_text(encoding="utf-8"))

    assert "</script" not in text
    assert "<\\/script><b>x</b>" in text


def test_data_embedded_in_the_runtime_cannot_end_its_script() -> None:
    runtime = html_report._runtime()

    assert "</script" not in runtime
    assert echarts_render.embed_json({"x": "</script><b>"}) == '{"x": "\\u003c/script>\\u003cb>"}'


def test_the_title_is_the_option_or_the_first_heading(build) -> None:
    assert "<title>用我的标题</title>" in build(report(chart()), "--title", "用我的标题").document
    assert "<title>销售报告</title>" in build(report(chart())).document
    assert "<title>Report</title>" in build("<main>" + chart() + "</main>").document


def test_an_english_report_is_marked_english(build) -> None:
    document = build(
        '<header class="report-head"><h1>Sales</h1></header>'
        + chart(
            option={**_OPTION, "title": {"text": "Revenue", "subtext": "Unit: USD · Source: ERP"}}
        )
    ).document

    assert '<html lang="en">' in document


def test_a_report_over_the_artifact_limit_is_refused(
    build, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(html_report, "_MAX_BYTES", 1024)

    failed(build(report(chart())), "20 MiB")


@pytest.mark.parametrize("example", sorted(examples()))
def test_the_toolkit_adds_no_footer_credit_watermark_or_timestamp(build, example: str) -> None:
    fragment = examples()[example].read_text(encoding="utf-8")
    document = build(fragment).document

    # The body is the author's fragment in one wrapper, with nothing before or after it.
    assert (
        document.split("<body>", 1)[1]
        == f'\n<div class="report">\n{fragment}\n</div>\n</body>\n</html>\n'
    )
    # What the toolkit writes itself, its stylesheet and runtime, says nothing a reader would see
    # as a credit, a disclaimer, a notice or a stamp: the same table the build warns by.
    chrome = (html_report._REPORT / "report.css").read_text("utf-8") + html_report._runtime()
    assert "<footer" not in chrome
    for family, rule in html_report._BOILERPLATE_RULES:
        assert not rule.search(chrome), family
    assert not re.search(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}", chrome)


# ---- --preview ---------------------------------------------------------------------------------


def test_preview_writes_each_chart_at_360_and_900(build, tmp_path: Path) -> None:
    require_png_tools()
    folder = tmp_path / "preview"
    result = build(
        report(
            chart("sales"),
            chart(
                "mix",
                option={
                    **_OPTION,
                    "series": [
                        {"type": "pie", "encode": {"itemName": "region", "value": "revenue"}}
                    ],
                },
            ),
        ),
        "--preview",
        str(folder),
    )

    assert result.code == 0, result.err
    assert sorted(p.name for p in folder.iterdir()) == [
        "mix-360.png",
        "mix-900.png",
        "sales-360.png",
        "sales-900.png",
    ]
    narrow, wide = Image.open(folder / "sales-360.png"), Image.open(folder / "sales-900.png")
    assert (narrow.width, wide.width) == (720, 900)
    assert all(p.stat().st_size < 150_000 for p in folder.iterdir())


def test_preview_is_light_by_default_and_dark_on_request(build, tmp_path: Path) -> None:
    require_png_tools()
    palette = json.loads((Path(html_report.__file__).parent / "palette.json").read_text("utf-8"))

    def corner(flag: tuple[str, ...]) -> tuple[int, int, int]:
        folder = tmp_path / f"p{len(flag)}"
        build(report(chart()), "--preview", str(folder), *flag)
        return Image.open(folder / "sales-900.png").convert("RGB").getpixel((1, 1))  # type: ignore[return-value]

    def rgb(hex_colour: str) -> tuple[int, ...]:
        return tuple(int(hex_colour[i : i + 2], 16) for i in (1, 3, 5))

    assert corner(()) == rgb(palette["light"]["roles"]["surface"])
    assert corner(("--preview-dark",)) == rgb(palette["dark"]["roles"]["surface"])


# ---- errors: each one refuses the build and says what to change ---------------------------------


def test_error_invalid_json_in_a_chart_block(build) -> None:
    source = report(
        '<figure class="chart" data-chart="x"></figure>\n<script type="application/json" id="chart-x">\n{"option": {"series": [}\n</script>'
    )

    failed(build(source), 'chart "x" (line 5, column 24)', "not valid", "Expecting value")


def test_error_nan_is_not_a_json_value(build) -> None:
    block = json.dumps({"option": _OPTION}).replace('"revenue": 10', '"revenue": NaN', 1)
    source = report(
        '<figure class="chart" data-chart="x"></figure>\n<script type="application/json" id="chart-x">'
        + block
        + "</script>"
    )

    failed(build(source), "NaN is not a JSON value; write null")


def test_error_a_chart_block_without_a_figure(build) -> None:
    source = report(
        chart("sales") + '<script type="application/json" id="chart-orphan">{"option": {}}</script>'
    )

    failed(build(source), 'id="chart-orphan"', "has no <figure", 'data-chart="orphan"')


def test_error_a_figure_without_a_block(build) -> None:
    source = report(chart("sales") + '<figure class="chart" data-chart="lonely"></figure>')

    failed(build(source), 'data-chart="lonely"', 'id="chart-lonely"')


@pytest.mark.parametrize(
    ("body", "needle"),
    [
        (chart("a") + '<figure data-chart="a"></figure>', 'duplicate chart id "a"'),
        (
            chart("a") + '<script type="application/json" id="chart-a">{"option": {}}</script>',
            'duplicate block id "chart-a"',
        ),
        (
            '<section data-page="p"><table></table></section><section data-page="p"><table></table></section>',
            'duplicate page id "p"',
        ),
        (_REGION + _REGION, 'duplicate slicer id "region"'),
    ],
)
def test_error_duplicate_ids(build, body: str, needle: str) -> None:
    failed(
        build(report(chart("sales", filters=["region"]) if "region" in body else "", body)), needle
    )


@pytest.mark.parametrize("kind", ["custom", "map"])
def test_error_a_series_type_that_cannot_be_drawn(build, kind: str) -> None:
    option = {**_OPTION, "series": [{"type": kind, "data": [1]}]}

    failed(build(report(chart(option=option))), 'chart "sales"', f'"{kind}"', "cannot be drawn")


def test_error_an_option_echarts_cannot_draw_names_the_chart_width_and_its_message(build) -> None:
    option = {
        "xAxis": {"type": "nonexistent"},
        "yAxis": {},
        "series": [{"type": "bar", "data": [1, 2]}],
    }

    failed(
        build(report(chart(option=option))),
        'chart "sales" at 360px',
        "ECharts cannot draw it",
        "xAxis.nonexistent",
    )


def test_error_the_drawing_check_runs_for_every_state_of_the_first_filter(build) -> None:
    # Only the 华北 rows hold a field the second series reads, so only that state fails to draw.
    rows = [
        {"region": "华东", "quarter": "Q1", "v": 1, "marker": None},
        {"region": "华北", "quarter": "Q1", "v": 2, "marker": 5},
    ]
    option = {
        "dataset": {"source": rows},
        "xAxis": {"type": "category"},
        "yAxis": {"type": "value"},
        "series": [
            {
                "type": "bar",
                "seriesBy": "region",
                "encode": {"x": "quarter", "y": "v"},
                "markLine": {"data": [{"type": "no-such-kind"}]},
            }
        ],
    }
    source = report(_REGION, chart("sales", option=option, filters=["region"]))

    failed(build(source), 'chart "sales"', "ECharts cannot draw it")


def test_error_a_chart_with_no_data(build) -> None:
    option = {
        "xAxis": {"type": "category"},
        "yAxis": {},
        "series": [{"type": "bar"}],
        "dataset": {"source": []},
    }

    failed(build(report(chart(option=option))), 'chart "sales" has no data', "dataset.source")


def test_error_a_filter_naming_an_undeclared_slicer(build) -> None:
    failed(
        build(report(chart(filters=["area"]))),
        'chart "sales" lists filter "area"',
        'data-slicer="area"',
    )


def test_error_a_slicer_field_that_no_filtered_chart_has(build) -> None:
    source = report(
        '<div class="slicer" data-slicer="region" data-field="area"></div>',
        chart(filters=["region"]),
    )

    failed(
        build(source), 'slicer "region"', 'field "area"', 'chart "sales"', "the rows have region"
    )


@pytest.mark.parametrize(
    ("series", "needle"),
    [
        (
            {"type": "bar", "seriesBy": "area", "encode": {"x": "quarter", "y": "revenue"}},
            'field "area" in seriesBy',
        ),
        ({"type": "bar", "encode": {"x": "season", "y": "revenue"}}, 'field "season" in encode.x'),
        ({"type": "bar", "encode": {"x": "quarter", "y": "revenu"}}, 'field "revenu" in encode.y'),
    ],
)
def test_error_a_field_the_series_names_that_no_row_has(
    build, series: dict[str, Any], needle: str
) -> None:
    option = {**_OPTION, "series": [series]}

    failed(build(report(chart(option=option))), 'chart "sales": series 0', needle, "the rows have")


def test_error_the_metric_slicer_options_are_checked_for_every_encoded_field(build) -> None:
    metric = _METRIC.replace('"y":"profit"', '"y":"margin"')
    option = {
        **_OPTION,
        "series": [
            {"type": "bar", "seriesBy": "region", "encode": {"x": "quarter", "y": "{metric}"}}
        ],
    }

    failed(
        build(report(metric, chart(option=option, filters=["metric"]))),
        'field "margin" in encode.y',
    )


def test_error_metric_without_a_metric_slicer(build) -> None:
    option = {
        **_OPTION,
        "series": [
            {"type": "bar", "seriesBy": "region", "encode": {"x": "quarter", "y": "{metric}"}}
        ],
    }

    failed(build(report(chart(option=option))), "uses {metric} but lists no metric slicer")


@pytest.mark.parametrize(
    ("markup", "needle"),
    [
        ('<img src="https://example.com/a.png">', 'src="https://example.com/a.png"'),
        ('<a href="https://example.com">x</a>', 'href="https://example.com"'),
        (
            '<div style="background:url(https://example.com/a.png)"></div>',
            "url(https://example.com/a.png)",
        ),
        (
            "<style>@import 'https://example.com/a.css';</style>",
            "@import https://example.com/a.css",
        ),
        ('<link rel="stylesheet" href="a.css">', "<link>"),
        ('<iframe src="data:text/html,x"></iframe>', "<iframe>"),
        ('<object data="data:text/html,x"></object>', "<object>"),
        ('<embed src="data:text/html,x">', "<embed>"),
        ('<form action="#"><input></form>', "<form>"),
        ('<a href="#x" download>x</a>', "<a download>"),
        ('<script src="https://example.com/a.js"></script>', "<script src"),
    ],
)
def test_error_markup_the_sandbox_refuses(build, markup: str, needle: str) -> None:
    failed(build(report(chart(), markup)), needle)


@pytest.mark.parametrize(
    "call",
    [
        "localStorage.getItem('a')",
        "sessionStorage.setItem('a', 1)",
        "document.cookie = 'a=1'",
        "fetch('/x')",
        "new XMLHttpRequest()",
        "new WebSocket('wss://x')",
        "window.open('/x')",
        "alert('hi')",
        "confirm('sure?')",
        "prompt('name?')",
        "eval('1+1')",
        "new Function('return 1')",
        "importScripts('a.js')",
        "navigator.sendBeacon('/x')",
    ],
)
def test_error_a_script_calls_what_the_sandbox_refuses(build, call: str) -> None:
    failed(
        build(report(chart(), f"<script>\nReport.ready(() => {{ {call}; }});\n</script>")),
        "which the sandbox refuses",
    )


def test_error_an_unknown_palette(build) -> None:
    failed(
        build(report(chart(option={**_OPTION, "palette": "neon"}))),
        '"palette" must be one of categorical',
    )


@pytest.mark.parametrize(
    ("slicer", "needle"),
    [
        ('<div class="slicer" data-slicer="s"></div>', "needs data-field"),
        ('<div class="slicer" data-slicer="s" data-type="metric"></div>', "needs data-options"),
        (
            '<div class="slicer" data-slicer="s" data-field="region" data-mode="both"></div>',
            'data-mode "both"',
        ),
        (
            '<div class="slicer" data-slicer="s" data-field="region" data-values="[oops"></div>',
            "data-values is not valid JSON",
        ),
    ],
)
def test_error_a_malformed_slicer(build, slicer: str, needle: str) -> None:
    failed(build(report(slicer, chart())), 'slicer "s"', needle)


def test_error_a_dataset_from_a_block_that_is_missing(build) -> None:
    option = {**_OPTION, "dataset": {"from": "nowhere"}}

    failed(build(report(chart(option=option))), 'dataset.from "nowhere"', 'id="data-nowhere"')


def test_a_dataset_can_be_shared_by_name(build) -> None:
    option = {**_OPTION, "dataset": {"from": "sales"}}
    block = f'<script type="application/json" id="data-sales">{json.dumps(_ROWS, ensure_ascii=False)}</script>'
    result = build(report(chart("a", option=option), chart("b", option=option), block))

    assert result.code == 0, result.err
    assert not result.warnings


def test_at_most_ten_error_lines_are_printed(build) -> None:
    figures = "".join(f'<figure data-chart="c{n}"></figure>' for n in range(14))
    result = build(report(figures))

    failed(result)
    assert len(result.err.splitlines()) == 10
    assert "and 5 more problems" in result.err


# ---- warnings: the build still succeeds --------------------------------------------------------


def test_warning_a_chart_without_a_title_or_a_subtitle(build) -> None:
    option = {k: v for k, v in _OPTION.items() if k != "title"}
    option_no_sub = {**_OPTION, "title": {"text": "季度销售额"}}

    warned(build(report(chart("a", option=option))), 'chart "a" has no title.text')
    warned(
        build(report(chart("b", option=option_no_sub))),
        'chart "b" has no title.subtext',
        "the unit and the source",
    )


def test_warning_more_series_than_colour_alone_tells_apart(build) -> None:
    def drawn(count: int) -> Any:
        rows = [{"region": f"区{n}", "quarter": "Q1", "revenue": n} for n in range(count)]
        return build(report(chart(option={**_OPTION, "dataset": {"source": rows}})))

    warned(drawn(7), "draws 7 series", "colour alone", "其他")
    assert not drawn(6).warnings


def test_warning_two_value_axes(build) -> None:
    option = {**_OPTION, "yAxis": [{"type": "value"}, {"type": "value"}]}

    warned(build(report(chart(option=option))), "two value axes")


def test_warning_a_vertical_bar_chart_with_long_or_many_labels_says_the_runtime_lays_it_down(
    build,
) -> None:
    long = [{"k": "新能源汽车及零部件制造", "v": 1}, {"k": "半导体", "v": 2}]
    option = {
        "title": _OPTION["title"],
        "dataset": {"source": long},
        "xAxis": {"type": "category"},
        "yAxis": {"type": "value"},
        "series": [{"type": "bar", "encode": {"x": "k", "y": "v"}}],
    }
    many = {**option, "dataset": {"source": [{"k": f"类{n}", "v": n} for n in range(9)]}}

    warned(
        build(report(chart(option=option))),
        "lays it out horizontally on narrow widths",
    )
    warned(build(report(chart(option=many))), "lays it out horizontally")
    assert not build(report(chart(option=option, orient="keep"))).warnings


def test_warning_a_page_with_no_chart_and_no_table(build) -> None:
    pages = '<section data-page="a">words</section><section data-page="b">' + chart() + "</section>"

    warned(build(report(pages)), 'page "a"', "no chart and no table")


def test_warning_a_single_page_that_declares_data_page(build) -> None:
    warned(
        build(report('<section data-page="only">' + chart() + "</section>")),
        "only one section declares data-page",
        "at least two",
    )


@pytest.mark.parametrize(
    "script",
    [
        "const s = document.createElementNS('http://www.w3.org/2000/svg', 'svg');",
        "const c = document.createElement('canvas'); c.getContext('2d');",
        "function niceTicks(min, max) { return []; }",
        "const x = canvas.getContext('2d');",
    ],
)
def test_warning_hand_drawing_in_an_author_script(build, script: str) -> None:
    warned(
        build(report(chart(), f"<script>{script}</script>")),
        "use an ECharts chart: the runtime draws, themes and resizes it",
    )


def test_warning_a_canvas_element_in_the_markup(build) -> None:
    result = build(
        report(
            chart(),
            "<script>document.body.insertAdjacentHTML('beforeend', '<canvas></canvas>')</script>",
        )
    )

    warned(result, "use an ECharts chart")


@pytest.mark.parametrize(
    "css",
    [
        "<style>.small { font-size: 10px; }</style>",
        "<style>.small { font-size: 0.5rem; }</style>",
        '<p style="font-size: 9px">tiny</p>',
    ],
)
def test_warning_text_under_11px_set_by_author_css(build, css: str) -> None:
    warned(build(report(chart(), css)), "text under 11px")


def test_the_example_text_sizes_at_or_over_11px_are_not_warned_about(build) -> None:
    assert not build(
        report(chart(), "<style>.a { font-size: 11px } .b { font-size: 0.875rem }</style>")
    ).warnings


def test_warning_a_slicer_or_a_data_block_nothing_uses(build) -> None:
    warned(build(report(_REGION, chart())), 'slicer "region"', "listed in no chart")
    block = '<script type="application/json" id="data-spare">[{"a": 1}]</script>'
    warned(build(report(chart(), block)), 'data block "spare" is used by no chart')


def test_warnings_do_not_fail_the_build_and_are_counted_in_the_summary(build) -> None:
    option = {k: v for k, v in _OPTION.items() if k != "title"}
    result = build(report(chart(option=option)))

    assert result.code == 0
    assert len(result.warnings) == 2
    assert "2 warnings" in result.out


@pytest.mark.parametrize("name", sorted(examples()))
def test_each_golden_example_builds_without_a_warning_and_stays_small_enough_to_read(
    build, name: str
) -> None:
    example = examples()[name]
    result = build(example.read_text(encoding="utf-8"))

    assert result.code == 0, result.err
    assert result.warnings == []
    assert example.stat().st_size < 12 * 1024
