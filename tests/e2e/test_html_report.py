# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Chromium coverage for ``html-report``: the golden examples, built by the real CLI, run in the
product's own artifact sandbox at phone, tablet and desktop widths, in both colour schemes.

Set ``REPORT_SHOTS_DIR`` to keep a full-page screenshot of every example for review.
"""

import itertools
import json
import os
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path
from typing import Any

import echarts_render
import html_report
import pytest
from PIL import Image
from playwright.sync_api import Browser

from tests.e2e import report_probe as probe
from tests.e2e.report_frame import ReportPage
from tests.support import colour
from tests.support.report_examples import examples

pytestmark = pytest.mark.e2e

_PALETTE = json.loads((Path(html_report.__file__).parent / "palette.json").read_text("utf-8"))
_WIDTHS = (360, 390, 820, 1280)
_SCHEMES = ("light", "dark")

# What each example draws at its default slicer state: chart id -> series count, first series type.
_CHARTS = {
    "brief": {"trend": (2, "line")},
    "dashboard": {"trend": (4, "line"), "product": (2, "bar"), "mix": (1, "pie")},
    "scenario": {"path": (4, "line")},
    "multipage": {
        "total": (2, "bar"),
        "flows": (2, "line"),
        "balance": (1, "bar"),
        "rd": (1, "bar"),
        "margin": (1, "bar"),
        "output": (1, "bar"),
        "share": (1, "pie"),
    },
}
_PAGES = {
    "multipage": {
        "overview": ["total"],
        "trend": ["flows", "balance"],
        "detail": ["rd", "margin", "output", "share"],
    }
}
_CARDS = ".kpi, figure.chart, .callout, .card, .table-wrap, .slicer-bar, .tabs"


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    folder = tmp_path_factory.mktemp("reports")
    out = {}
    for name in _CHARTS:
        out[name] = folder / f"{name}.html"
        assert html_report.main(["build", str(examples()[name]), str(out[name])]) == 0
    return out


@pytest.fixture
def open_report(browser: Browser) -> Iterator[Any]:
    opened: list[ReportPage] = []

    def open_(path: Path, width: int, **kwargs: Any) -> ReportPage:
        page = ReportPage(browser, path, width=width, **kwargs)
        opened.append(page)
        return page

    yield open_
    for page in opened:
        page.close()


def show(page: ReportPage, page_id: str) -> None:
    page.frame.click(f'button[role="tab"][data-tab="{page_id}"]')
    page.page.wait_for_timeout(450)


def shoot(page: ReportPage, name: str) -> None:
    folder = os.environ.get("REPORT_SHOTS_DIR")
    if folder:
        Path(folder).mkdir(parents=True, exist_ok=True)
        page.screenshot(Path(folder) / f"{name}.png")


def visible_figures(page: ReportPage) -> list[str]:
    return page.eval(
        "() => [...document.querySelectorAll('figure[data-chart]')]"
        ".filter((f) => f.offsetParent !== null).map((f) => f.dataset.chart)"
    )


def each_page(example: str, page: ReportPage) -> Iterator[tuple[str | None, list[str]]]:
    """Visit every page of a report: its id (None for a single page) and the charts it shows."""
    if example not in _PAGES:
        yield None, list(_CHARTS[example])
        return
    for page_id, charts in _PAGES[example].items():
        show(page, page_id)
        yield page_id, charts


def test_every_example_the_skill_shows_a_model_is_opened_here() -> None:
    assert sorted(_CHARTS) == sorted(examples())


# ---- the report runs clean inside the sandbox, and every chart fits -----------------------------


@pytest.mark.parametrize("width", _WIDTHS)
@pytest.mark.parametrize("example", list(_CHARTS))
def test_the_report_runs_clean_and_every_chart_fits_its_figure(
    built: dict[str, Path], open_report, example: str, width: int
) -> None:
    page = open_report(built[example], width)
    caption_sizes: dict[str, float] = {}

    for page_id, charts in each_page(example, page):
        label = f"{example}@{width}{'/' + page_id if page_id else ''}"
        assert page.eval("() => document.documentElement.scrollWidth <= window.innerWidth"), label
        assert sorted(visible_figures(page)) == sorted(charts), label
        for chart in charts:
            facts = probe.chart_facts(page, chart)
            expected = _CHARTS[example][chart]
            assert facts is not None, f"{label}: {chart} is not an ECharts instance"
            assert (len(facts["series"]), facts["series"][0]["type"]) == expected, (
                f"{label}: {chart}"
            )
            assert facts["width"] >= facts["body"][0] - 1, (
                f"{label}: {chart} is narrower than its figure"
            )
            assert facts["height"] >= 200, f"{label}: {chart} is {facts['height']}px tall"
            if width == 360 and facts["plot"]:
                assert facts["plot"][0] >= 0.55 * facts["width"], (
                    f"{label}: {chart} plot {facts['plot']}"
                )
            if facts["axisFont"]:
                title = page.eval(
                    "(id) => parseFloat(getComputedStyle(document.querySelector("
                    '`figure[data-chart="${id}"] .chart-title`)).fontSize)',
                    chart,
                )
                caption_sizes[chart] = title
                assert facts["legendFont"] <= facts["axisFont"] * 1.25, label
                assert title <= facts["axisFont"] * 1.25, f"{label}: {chart} title {title}px"
        too_small = [i for i in probe.text_scan(page) if i["size"] < 11]
        assert too_small == [], f"{label}: text under 11px: {too_small[:3]}"

    assert page.observed.clean(), page.observed
    assert page.violations() == []


@pytest.mark.parametrize("example", list(_CHARTS))
def test_echarts_is_in_the_document_once_and_the_report_adds_one_global(
    built: dict[str, Path], open_report, example: str
) -> None:
    text = built[example].read_text(encoding="utf-8")
    marker = echarts_render.echarts_library().read_text(encoding="utf-8")[200:320]
    page = open_report(built[example], 1280)

    assert text.count(marker) == 1
    assert (
        page.eval(
            "() => [...document.scripts].filter((s) => s.textContent.includes('registerTheme')).length"
        )
        >= 1
    )
    assert page.eval("() => typeof Report === 'object' && typeof echarts === 'object'")
    assert page.eval("() => Object.keys(Report).sort().join()") == "chart,ready,slicer"
    assert page.eval("() => Object.isFrozen(Report)")


@pytest.mark.parametrize("example", list(_CHARTS))
def test_a_hydrated_report_shows_the_authors_words_and_its_controls_and_no_notice_credit_or_stamp(
    built: dict[str, Path], open_report, example: str
) -> None:
    page = open_report(built[example], 1280)

    for page_id, _charts in each_page(example, page):
        text = page.eval("() => document.body.innerText")
        for family, rule in html_report._BOILERPLATE_RULES:
            found = rule.search(text)
            assert found is None, f"{example}/{page_id}: {family}: {found and found.group(0)!r}"
        assert page.eval("() => document.querySelectorAll('footer').length") == 0


# ---- the Mineral look, in both colour schemes ---------------------------------------------------


@pytest.mark.parametrize("scheme", _SCHEMES)
@pytest.mark.parametrize("width", (390, 1280))
@pytest.mark.parametrize("example", list(_CHARTS))
def test_text_marks_cards_and_focus_meet_the_mineral_floors(
    built: dict[str, Path], open_report, example: str, width: int, scheme: str
) -> None:
    page = open_report(built[example], width, scheme=scheme)  # type: ignore[arg-type]
    roles = _PALETTE[scheme]["roles"]

    for page_id, charts in each_page(example, page):
        label = f"{example}@{width}/{scheme}{'/' + page_id if page_id else ''}"
        assert probe.failing_text(page) == [], label
        # Cards are tone and a hairline, never a shadow.
        shadows = page.eval(
            f"() => [...document.querySelectorAll('{_CARDS}')]"
            ".filter((n) => getComputedStyle(n).boxShadow !== 'none').map((n) => n.className)"
        )
        assert shadows == [], label
        surface = colour.parse(probe.tokens(page, ["--color-bg-surface"])["--color-bg-surface"])
        assert colour.parse(roles["surface"]) == pytest.approx(surface, abs=1e-3)  # type: ignore[comparison-overlap]
        for chart in charts:
            facts = probe.chart_facts(page, chart)
            assert facts is not None
            count = (
                len(facts["series"])
                if facts["series"][0]["type"] != "pie"
                else len(facts["series"][0]["data"])
            )
            marks = [colour.parse(c) for c in facts["palette"][:count]]
            assert marks, label
            for mark in marks:
                assert colour.contrast(mark, surface) >= 3, f"{label}: {chart} mark {mark}"
        shoot(page, f"{example}{'-' + page_id if page_id else ''}-{scheme}-{width}")

    page_bg = page.eval("() => getComputedStyle(document.body).backgroundColor")
    base = probe.tokens(page, ["--color-bg-base"])["--color-bg-base"]
    assert colour.parse(page_bg) == pytest.approx(colour.parse(base), abs=1e-3)  # type: ignore[comparison-overlap]
    assert page.eval("() => getComputedStyle(document.documentElement).colorScheme") == "light dark"


@pytest.mark.parametrize("scheme", _SCHEMES)
@pytest.mark.parametrize("width", (390, 1280))
@pytest.mark.parametrize("example", ["dashboard", "multipage", "scenario"])
def test_every_control_shows_a_two_pixel_ring_two_pixels_off_in_the_ring_colour(
    built: dict[str, Path], open_report, example: str, width: int, scheme: str
) -> None:
    page = open_report(built[example], width, scheme=scheme)  # type: ignore[arg-type]
    ring = colour.parse(probe.tokens(page, ["--color-control-ring"])["--color-control-ring"])
    page.page.keyboard.press("Tab")
    seen: list[int] = []
    for _ in range(40):
        found = probe.focus_ring(page)
        if found is None or found["index"] in seen:
            break
        seen.append(found["index"])
        assert found["style"] == "solid" and found["width"] == 2 and found["offset"] == 2, found
        painted = colour.parse(found["colour"])
        assert painted == pytest.approx(ring, abs=0.02), found  # type: ignore[comparison-overlap]
        # The ring is drawn outside its control, over what is behind the control's parent.
        around = tuple(c / 255 for c in found["behind"][:3])
        surface = (*around, 1.0)
        assert colour.contrast(colour.over(painted, surface), surface) >= 3, found
        page.page.keyboard.press("Tab")
    assert len(seen) >= 4, seen


def test_the_reset_control_and_a_selected_slicer_keep_their_contrast(
    built: dict[str, Path], open_report
) -> None:
    for scheme in _SCHEMES:
        page = open_report(built["dashboard"], 390, scheme=scheme)  # type: ignore[arg-type]
        page.frame.click('button.chip[data-value="华东"]')
        page.page.wait_for_timeout(300)

        assert page.eval("() => document.querySelector('.slicer-reset.on') !== null")
        assert probe.failing_text(page) == [], scheme


def test_tag_tones_are_marks_of_the_palette_and_stay_apart_for_a_deuteranope(
    built: dict[str, Path], open_report
) -> None:
    for scheme in _SCHEMES:
        page = open_report(built["multipage"], 1280, scheme=scheme)  # type: ignore[arg-type]
        names = [f"--tone-{n}" for n in range(1, 7)]
        tones = [colour.parse(v) for v in probe.tokens(page, names).values()]
        for a, b in itertools.combinations(tones, 2):
            assert colour.distance(a, b) >= 11
            for kind in ("protan", "deutan"):
                assert colour.distance(a, b, kind) >= 4.9
        shapes = page.eval(
            "() => [1, 2, 3, 4, 5, 6].map((n) => { const t = document.createElement('span'); t.className = 'tag';"
            " t.dataset.tone = String(n); document.body.append(t); const s = getComputedStyle(t, '::before');"
            " const out = s.clipPath + '|' + s.borderRadius + '|' + s.backgroundColor; t.remove(); return out; })"
        )
        assert len(set(shapes)) == 6


# ---- slicers ------------------------------------------------------------------------------------


def trend(page: ReportPage) -> dict[str, Any]:
    facts = probe.chart_facts(page, "trend")
    assert facts is not None
    return facts


def settle(page: ReportPage) -> None:
    page.page.wait_for_timeout(350)


def test_choosing_a_value_changes_the_charts_data_and_reset_restores_it(
    built: dict[str, Path], open_report
) -> None:
    page = open_report(built["dashboard"], 1280)
    events = page.eval(
        "() => { window.__events = []; document.addEventListener('report:slicer', (e) => window.__events.push(e.detail)); return 0; }"
    )
    before = trend(page)
    assert [s["name"] for s in before["series"]] == ["华东", "华北", "华南", "西南"]
    assert (
        page.eval("() => document.querySelector('.slicer-reset').classList.contains('on')") is False
    )

    page.frame.click('button.chip[data-value="华东"]')
    settle(page)
    only = trend(page)
    assert [s["name"] for s in only["series"]] == ["华东"]  # no stale series remain
    assert only["series"][0]["data"] == [530, 575, 604, 667]
    assert only["series"][0]["color"] == before["series"][0]["color"]
    assert (
        page.eval(
            "() => document.querySelector('button.chip[data-value=\"华东\"]').getAttribute('aria-pressed')"
        )
        == "true"
    )
    assert (
        page.eval("() => document.querySelector('.slicer-reset').classList.contains('on')") is True
    )
    assert page.eval("() => window.__events") == [{"id": "region", "value": "华东"}]

    page.frame.click(".slicer-reset")
    settle(page)
    restored = trend(page)
    assert [s["name"] for s in restored["series"]] == ["华东", "华北", "华南", "西南"]
    assert restored["series"] == before["series"]
    assert page.eval("() => Report.slicer('region').value()") is None
    assert events == 0


def test_a_series_keeps_its_colour_when_the_filter_leaves_it_alone(
    built: dict[str, Path], open_report
) -> None:
    page = open_report(built["dashboard"], 1280)
    all_colours = {s["name"]: s["color"] for s in trend(page)["series"]}

    page.frame.click('button.chip[data-value="华南"]')
    settle(page)

    assert trend(page)["series"][0]["color"] == all_colours["华南"]


def test_the_metric_slicer_swaps_the_plotted_dimension_and_the_title(
    built: dict[str, Path], open_report
) -> None:
    page = open_report(built["dashboard"], 1280)
    assert (
        page.eval(
            "() => document.querySelector('figure[data-chart=trend] .chart-title').textContent"
        )
        == "季度销售额"
    )
    revenue = trend(page)["series"][0]["data"]

    page.frame.click('button.chip[data-value="利润"]')
    settle(page)

    profit = trend(page)["series"][0]["data"]
    assert profit != revenue and profit == [102, 111, 120, 138]
    assert (
        page.eval(
            "() => document.querySelector('figure[data-chart=trend] .chart-title').textContent"
        )
        == "季度利润"
    )
    assert (
        page.eval("() => document.querySelector('figure[data-chart=mix] .chart-title').textContent")
        == "产品构成（利润）"
    )
    pie = probe.chart_facts(page, "mix")
    assert pie is not None and sum(d["value"] for d in pie["series"][0]["data"]) == 1492


def test_a_filter_that_leaves_no_rows_shows_the_empty_state_in_every_chart(
    built: dict[str, Path], open_report
) -> None:
    page = open_report(built["dashboard"], 390)

    page.eval("() => Report.slicer('region').set('不存在')")
    settle(page)

    texts = page.eval(
        "() => [...document.querySelectorAll('.chart-empty')].filter((n) => !n.hidden).map((n) => n.textContent)"
    )
    assert texts == ["没有符合当前筛选的数据"] * 3
    assert page.eval(
        "() => [...document.querySelectorAll('.chart-body')].every((b) => b.offsetHeight >= 200)"
    )
    assert page.observed.clean()
    assert probe.failing_text(page) == []

    page.frame.click(".slicer-reset")
    settle(page)
    assert page.eval("() => [...document.querySelectorAll('.chart-empty')].every((n) => n.hidden)")
    assert len(trend(page)["series"]) == 4


def test_choosing_a_chip_moves_nothing(built: dict[str, Path], open_report) -> None:
    page = open_report(built["dashboard"], 390)
    boxes = "() => [...document.querySelectorAll('.slicer-bar *, figure.chart')].map((n) => { const b = n.getBoundingClientRect(); return [Math.round(b.x), Math.round(b.y), Math.round(b.width)]; })"
    before = page.eval(boxes)

    page.frame.click('button.chip[data-value="华北"]')
    settle(page)
    after = page.eval(boxes)

    assert before == after


def test_a_slicer_of_many_values_is_a_select_and_a_multi_slicer_toggles(
    tmp_path: Path, open_report
) -> None:
    rows = [{"city": f"城{n}", "v": n + 1} for n in range(8)]
    option = {
        "title": {"text": "各城市", "subtext": "单位：个 · 来源：测试数据"},
        "dataset": {"source": rows},
        "xAxis": {"type": "category"},
        "yAxis": {"type": "value"},
        "series": [{"type": "bar", "name": "数量", "encode": {"x": "city", "y": "v"}}],
    }
    fragment = (
        "<h1>城市</h1>"
        '<div class="slicer" data-slicer="one" data-field="city" data-label="城市"></div>'
        '<div class="slicer" data-slicer="many" data-field="city" data-mode="multi" data-label="多选"></div>'
        '<figure class="chart" data-chart="c"></figure>'
        f'<script type="application/json" id="chart-c">{json.dumps({"filters": ["one", "many"], "option": option}, ensure_ascii=False)}</script>'
    )
    source, out = tmp_path / "s.html", tmp_path / "o.html"
    source.write_text(fragment, encoding="utf-8")
    assert html_report.main(["build", str(source), str(out)]) == 0
    page = open_report(out, 1280)

    assert page.eval("() => document.querySelector('[data-slicer=one] select') !== null")
    assert page.eval("() => document.querySelectorAll('[data-slicer=one] option').length") == 9
    assert page.eval("() => document.querySelector('[data-slicer=many] select') === null")
    page.frame.select_option("[data-slicer=one] select", "城3")
    settle(page)
    assert probe.chart_facts(page, "c")["series"][0]["data"] == [4]  # type: ignore[index]
    page.frame.select_option("[data-slicer=one] select", "")
    page.frame.click('[data-slicer=many] button.chip[data-value="城1"]')
    page.frame.click('[data-slicer=many] button.chip[data-value="城5"]')
    settle(page)
    assert probe.chart_facts(page, "c")["series"][0]["data"] == [2, 6]  # type: ignore[index]
    assert page.eval("() => Report.slicer('many').value()") == ["城1", "城5"]
    page.frame.click('[data-slicer=many] button.chip[data-value="城1"]')
    settle(page)
    assert page.eval("() => Report.slicer('many').value()") == ["城5"]


# ---- sliders and the scenario page --------------------------------------------------------------


def spread(page: ReportPage) -> Any:
    return page.frame.locator("[data-slicer=spread] input[type=range]")


def pixel_at(page: ReportPage, x: float, y: float) -> tuple[int, int, int]:
    """The colour painted at a point of the page, read from a screenshot."""
    png = page.page.screenshot(clip={"x": x - 1, "y": y - 1, "width": 3, "height": 3})
    return Image.open(BytesIO(png)).convert("RGB").getpixel((1, 1))  # type: ignore[return-value]


def ratio_of(page: ReportPage) -> float:
    return float(
        page.eval(
            "() => document.querySelector('[data-slicer=spread] input').style.getPropertyValue('--ratio')"
        )
    )


@pytest.mark.parametrize("scheme", _SCHEMES)
@pytest.mark.parametrize("width", (360, 1280))
def test_a_slider_is_a_real_range_input_with_its_value_and_its_ends_shown(
    built: dict[str, Path], open_report, width: int, scheme: str
) -> None:
    page = open_report(built["scenario"], width, scheme=scheme)  # type: ignore[arg-type]

    facts = page.eval(
        """() => {
      const input = document.querySelector('[data-slicer=spread] input');
      const box = (el) => { const b = el.getBoundingClientRect(); return [b.left, b.right, b.top, b.height]; };
      const root = document.querySelector('[data-slicer=spread]');
      const value = root.querySelector('.slicer-value');
      return {
        type: input.type, min: input.min, max: input.max, step: input.step, number: input.valueAsNumber,
        valuetext: input.getAttribute('aria-valuetext'),
        label: document.getElementById(input.getAttribute('aria-labelledby')).textContent,
        shown: value.textContent, ends: [...root.querySelectorAll('.slicer-ends span')].map((n) => n.textContent),
        figures: getComputedStyle(value).fontVariantNumeric,
        label_box: box(root.querySelector('.slicer-label')), value_box: box(value), input_box: box(input),
        api: [typeof Report.slicer('spread').value(), Report.slicer('spread').value()],
      };
    }"""
    )

    assert (facts["type"], facts["min"], facts["max"], facts["step"], facts["number"]) == (
        "range", "-1", "2", "0.25", 0.5,
    )  # fmt: skip
    assert facts["valuetext"] == facts["shown"] == "+0.50pp"
    assert facts["label"] == "利差变化（瑞典减中国）"
    assert facts["ends"] == ["-1.00pp", "+2.00pp"]
    assert "tabular-nums" in facts["figures"]
    assert facts["api"] == ["number", 0.5]
    # The label sits on the left and the value on the right, over a track as wide as the slicer.
    assert facts["label_box"][0] < facts["value_box"][0]
    assert abs(facts["value_box"][1] - facts["input_box"][1]) <= 1
    assert facts["input_box"][2] > facts["value_box"][2]
    # A thumb a finger can find: 44px tall on a phone, and the thumb itself holds 3:1 on its card.
    assert facts["input_box"][3] >= (44 if width < 720 else 36)
    tokens = probe.tokens(page, ["--color-bg-surface", "--color-accent-action"])
    surface, accent = (
        colour.parse(tokens[n]) for n in ("--color-bg-surface", "--color-accent-action")
    )
    box = spread(page).bounding_box()
    centre = pixel_at(
        page, box["x"] + 10 + (box["width"] - 20) * ratio_of(page), box["y"] + box["height"] / 2
    )
    assert colour.parse(f"rgb({centre[0]} {centre[1]} {centre[2]})") == pytest.approx(
        accent, abs=0.02
    )  # type: ignore[comparison-overlap]
    assert colour.contrast(accent, surface) >= 3
    assert probe.failing_text(page) == []
    assert page.observed.clean() and page.violations() == []


@pytest.mark.parametrize("scheme", _SCHEMES)
@pytest.mark.parametrize("width", (360, 1280))
def test_a_slider_moves_by_arrow_key_and_by_tap_and_shows_each_value(
    built: dict[str, Path], open_report, width: int, scheme: str
) -> None:
    page = open_report(built["scenario"], width, scheme=scheme, touch=True)  # type: ignore[arg-type]
    shown = "() => document.querySelector('[data-slicer=spread] .slicer-value').textContent"
    valuetext = (
        "() => document.querySelector('[data-slicer=spread] input').getAttribute('aria-valuetext')"
    )

    spread(page).focus()
    page.page.keyboard.press("ArrowRight")
    assert page.eval("() => Report.slicer('spread').value()") == 0.75
    assert page.eval(shown) == page.eval(valuetext) == "+0.75pp"
    page.page.keyboard.press("ArrowLeft")
    page.page.keyboard.press("ArrowLeft")
    assert page.eval(shown) == "+0.25pp"
    page.page.keyboard.press("Home")
    assert (page.eval(shown), ratio_of(page)) == ("-1.00pp", 0)
    page.page.keyboard.press("End")
    assert (page.eval(shown), ratio_of(page)) == ("+2.00pp", 1)

    box = spread(page).bounding_box()
    thumb = 20
    page.page.touchscreen.tap(
        box["x"] + thumb / 2 + (box["width"] - thumb) * 0.75, box["y"] + box["height"] / 2
    )
    page.page.wait_for_timeout(100)
    assert abs(page.eval("() => Report.slicer('spread').value()") - 1.25) <= 0.25
    assert page.eval("() => Report.slicer('spread').value() % 0.25") == 0, "a tap lands on a notch"
    assert abs(ratio_of(page) - 0.75) <= 0.1

    # Reset appears once the slider has left its default, and takes it back.
    on = "() => document.querySelector('[data-slicer=spread] .slicer-reset').classList.contains('on')"
    assert page.eval(on)
    page.frame.click("[data-slicer=spread] .slicer-reset")
    assert page.eval("() => Report.slicer('spread').value()") == 0.5
    assert page.eval(shown) == "+0.50pp" and not page.eval(on)


def test_a_slider_event_goes_out_once_a_frame_with_the_latest_number(
    built: dict[str, Path], open_report
) -> None:
    page = open_report(built["scenario"], 1280)
    page.eval(
        "() => { window.__events = []; document.addEventListener('report:slicer', (e) => window.__events.push(e.detail)); }"
    )

    page.eval(
        "() => { const s = Report.slicer('spread'); s.set(0); s.set(0.3); s.set(0.8); s.set(99); }"
    )
    page.page.wait_for_timeout(120)
    assert page.eval("() => window.__events") == [{"id": "spread", "value": 2}]

    # A value that is no number is refused, and the one a slider already holds says nothing.
    page.eval("() => { const s = Report.slicer('spread'); s.set('x'); s.set(2); s.set(''); }")
    page.page.wait_for_timeout(120)
    assert page.eval("() => window.__events") == [{"id": "spread", "value": 2}]
    assert page.eval("() => Report.slicer('spread').value()") == 2
    # Snapped to the step, and held to the range.
    page.eval("() => Report.slicer('spread').set(-7)")
    assert page.eval("() => Report.slicer('spread').value()") == -1
    page.eval("() => Report.slicer('spread').set(0.6)")
    assert page.eval("() => Report.slicer('spread').value()") == 0.5
    # A chip of a filter slicer still answers at once, and only for a choice it offers.
    page.eval("() => { window.__events.length = 0; Report.slicer('scenario').set('乐观'); }")
    assert page.eval("() => window.__events") == [{"id": "scenario", "value": "乐观"}]
    page.eval("() => Report.slicer('scenario').set('不存在')")
    assert page.eval("() => Report.slicer('scenario').value()") == "乐观"


def path_rows(page: ReportPage) -> list[dict[str, Any]]:
    return page.eval("() => Report.chart('path').rows()")


@pytest.mark.parametrize("width", (360, 1280))
def test_the_scenario_script_redraws_the_band_and_the_numbers_when_a_control_moves(
    built: dict[str, Path], open_report, width: int
) -> None:
    page = open_report(built["scenario"], width)
    say = lambda name: page.eval("(id) => document.getElementById(id).textContent", name)  # noqa: E731
    subtitle = "() => document.querySelector('figure[data-chart=path] .chart-subtitle').textContent"

    assert (say("mid"), say("range"), say("change")) == ("0.665", "0.606–0.725", "-0.3%")
    assert "利差 +0.50pp，通胀预期 2.00%" in page.eval(subtitle)

    spread(page).focus()
    page.page.keyboard.press("End")
    settle(page)
    rows = path_rows(page)
    assert rows[12]["mid"] == pytest.approx(0.6903, abs=1e-4)
    assert (say("mid"), say("range"), say("change")) == ("0.690", "0.628–0.752", "+3.5%")
    assert "利差 +2.00pp，通胀预期 2.00%" in page.eval(subtitle)
    drawn = probe.chart_facts(page, "path")
    assert drawn is not None and drawn["series"][0]["data"][12] == rows[12]["mid"]

    # A chip presets both sliders, and the sliders go on to move the chart from there.
    page.frame.click("[data-slicer=scenario] button.chip[data-value='乐观']")
    settle(page)
    both = "() => [Report.slicer('spread').value(), Report.slicer('inflation').value()]"
    assert page.eval(both) == [1.5, 1.5]
    assert say("change") == "+2.7%" and path_rows(page)[12]["mid"] == pytest.approx(
        0.6853, abs=1e-4
    )
    page.frame.click("[data-slicer=scenario] button.chip[data-value='悲观']")
    settle(page)
    assert page.eval(both) == [-0.5, 3]
    assert say("change") == "-3.7%" and "通胀预期 3.00%" in page.eval(subtitle)
    assert page.observed.clean()


@pytest.mark.parametrize("scheme", _SCHEMES)
def test_the_band_is_two_stacked_areas_under_the_central_line_and_only_the_line_has_a_legend(
    built: dict[str, Path], open_report, scheme: str
) -> None:
    page = open_report(built["scenario"], 1280, scheme=scheme)  # type: ignore[arg-type]

    option = page.eval(
        "() => Report.chart('path').instance.getOption().series.map((s) => ({name: s.name, stack: s.stack || null, fill: (s.areaStyle || {}).opacity ?? null, line: (s.lineStyle || {}).opacity ?? null, z: s.z}))"
    )
    assert [o["name"] for o in option] == ["中位路径", "区间下限", "不确定区间", "区间上限"]
    assert [o["stack"] for o in option] == [None, "band", "band", None]
    assert option[1]["fill"] == 0 and option[2]["fill"] == pytest.approx(0.28)
    assert option[1]["line"] == option[2]["line"] == option[3]["line"] == 0
    assert option[0]["z"] > option[2]["z"]
    # Every month the band sits around the path: low, mid, high.
    for row in path_rows(page):
        assert row["low"] <= row["mid"] <= row["high"]
        assert row["low"] + row["width"] == pytest.approx(row["high"], abs=2e-4)
    # The shaded area is drawn, wide, and the legend names the line and the band only.
    area = page.eval(
        """() => {
      const plot = document.querySelector('figure[data-chart=path] .chart-body svg');
      const filled = [...plot.querySelectorAll('path')].filter((p) => Math.abs(parseFloat(p.getAttribute('fill-opacity') ?? '1') - 0.28) < 0.01);
      const box = filled.map((p) => p.getBoundingClientRect()).sort((a, b) => b.width - a.width)[0];
      return {count: filled.length, width: box ? box.width : 0, height: box ? box.height : 0, plot: plot.getBoundingClientRect().width,
        words: [...plot.querySelectorAll('text')].map((t) => t.textContent)};
    }"""
    )
    assert area["count"] >= 1 and area["width"] >= 0.7 * area["plot"] and area["height"] > 40
    assert "中位路径" in area["words"] and "不确定区间" in area["words"]
    assert not {"区间下限", "区间上限"} & set(area["words"])
    # Hovering reads the path and both ends of the band, not the band's width.
    page.eval(
        "() => Report.chart('path').instance.dispatchAction({type: 'showTip', seriesIndex: 0, dataIndex: 6})"
    )
    page.page.wait_for_timeout(250)
    tip = page.eval(
        "() => [...document.querySelectorAll('figure[data-chart=path] .chart-body div')]"
        ".filter((d) => d.style.position === 'absolute' && !d.querySelector('svg')).map((d) => d.innerText).join('|')"
    )
    assert all(name in tip for name in ("中位路径", "区间下限", "区间上限"))
    assert "不确定区间" not in tip


# ---- pages --------------------------------------------------------------------------------------


@pytest.mark.parametrize("width", (390, 1280))
def test_tabs_show_the_right_page_by_click_and_by_keyboard_and_sync_the_hash(
    built: dict[str, Path], open_report, width: int
) -> None:
    page = open_report(built["multipage"], width)
    state = (
        "() => [...document.querySelectorAll('[role=tab]')].map((t) => [t.dataset.tab, t.getAttribute('aria-selected'), t.tabIndex,"
        " document.getElementById(t.getAttribute('aria-controls')).hidden])"
    )
    assert page.eval("() => document.querySelectorAll('[role=tablist]').length") == 1
    assert page.eval(state) == [
        ["overview", "true", 0, False],
        ["trend", "false", -1, True],
        ["detail", "false", -1, True],
    ]
    # Charts of a hidden page are not drawn until it is first shown.
    assert page.eval("() => Report.chart('flows').instance") is None

    page.frame.click("[role=tab][data-tab=trend]")
    settle(page)
    assert page.eval("() => location.hash") == "#trend"
    assert page.eval(state)[1][1:3] == ["true", 0]
    flows = probe.chart_facts(page, "flows")
    assert flows is not None and flows["width"] >= flows["body"][0] - 1

    page.frame.focus("[role=tab][data-tab=trend]")
    page.page.keyboard.press("ArrowRight")
    settle(page)
    assert page.eval("() => document.activeElement.dataset.tab") == "detail"
    assert page.eval("() => location.hash") == "#detail"
    page.page.keyboard.press("ArrowRight")  # wraps
    assert page.eval("() => document.activeElement.dataset.tab") == "overview"
    page.page.keyboard.press("End")
    assert page.eval("() => document.activeElement.dataset.tab") == "detail"
    page.page.keyboard.press("Home")
    assert page.eval("() => document.activeElement.dataset.tab") == "overview"
    page.page.keyboard.press("ArrowLeft")  # wraps backwards
    assert page.eval("() => document.activeElement.dataset.tab") == "detail"
    settle(page)
    rd = probe.chart_facts(page, "rd")
    assert rd is not None and rd["width"] >= rd["body"][0] - 1 and rd["height"] >= 200
    assert page.observed.clean()


def test_the_hash_is_honoured_on_load(built: dict[str, Path], browser: Browser) -> None:
    context = browser.new_context(viewport={"width": 900, "height": 800})
    page = context.new_page()
    page.goto(built["multipage"].as_uri() + "#detail")
    page.wait_for_function("() => document.documentElement.dataset.reportReady === 'true'")

    assert (
        page.eval_on_selector("[role=tab][aria-selected=true]", "(t) => t.dataset.tab") == "detail"
    )
    assert page.eval_on_selector("section[data-page=overview]", "(s) => s.hidden") is True
    context.close()


def test_the_shared_slicer_shows_only_where_a_chart_it_filters_is_in_view(
    built: dict[str, Path], open_report
) -> None:
    page = open_report(built["multipage"], 1280)
    bar_hidden = "() => document.querySelector('.slicer-bar').hidden"

    assert page.eval(bar_hidden) is False
    show(page, "trend")
    assert page.eval(bar_hidden) is False
    show(page, "detail")
    assert page.eval(bar_hidden) is True
    page.frame.click("[role=tab][data-tab=overview]")
    page.frame.click('button.chip[data-value="自瑞进口"]')
    settle(page)
    assert [s["name"] for s in probe.chart_facts(page, "total")["series"]] == ["自瑞进口"]  # type: ignore[index]
    show(page, "trend")
    assert [s["name"] for s in probe.chart_facts(page, "flows")["series"]] == ["自瑞进口"]  # type: ignore[index]


def test_the_tabs_and_the_filters_stay_in_view_while_the_page_scrolls(
    built: dict[str, Path], open_report
) -> None:
    page = open_report(built["multipage"], 390)

    page.eval("() => window.scrollTo(0, 900)")
    settle(page)
    bar = page.eval("() => document.querySelector('.slicer-bar').getBoundingClientRect().top")
    tabs = page.eval(
        "() => { const t = document.querySelector('[role=tablist]').getBoundingClientRect(); return [t.top, t.height]; }"
    )
    height = page.eval("() => document.querySelector('.slicer-bar').getBoundingClientRect().height")

    assert abs(bar) <= 1
    assert abs(tabs[0] - height) <= 1.5
    assert page.eval(
        "() => [...document.querySelectorAll('[role=tab]')].every((t) => t.offsetHeight >= 36)"
    )


# ---- the runtime, resizing, motion, theme and print --------------------------------------------


def test_rotating_a_phone_lays_every_visible_chart_out_again_at_its_new_width(
    built: dict[str, Path], open_report
) -> None:
    page = open_report(built["dashboard"], 390, height=800)
    portrait = {c: probe.chart_facts(page, c) for c in ("trend", "product", "mix")}
    assert all(f is not None and f["profile"] == "narrow" for f in portrait.values())

    page.page.set_viewport_size({"width": 844, "height": 390})
    page.page.wait_for_timeout(700)

    for chart, before in portrait.items():
        after = probe.chart_facts(page, chart)
        assert after is not None and before is not None
        assert after["width"] >= before["width"], chart
        assert after["width"] >= after["body"][0] - 1, chart
    wide = probe.chart_facts(page, "trend")
    assert wide is not None and wide["profile"] in ("medium", "wide")
    assert wide["width"] > portrait["trend"]["width"] + 200  # type: ignore[index]
    page.page.set_viewport_size({"width": 390, "height": 800})
    page.page.wait_for_timeout(700)
    assert probe.chart_facts(page, "trend")["profile"] == "narrow"  # type: ignore[index]
    assert page.eval("() => document.documentElement.scrollWidth <= window.innerWidth")


def test_a_chart_that_throws_degrades_alone(
    built: dict[str, Path], open_report, tmp_path: Path
) -> None:
    text = built["dashboard"].read_text(encoding="utf-8")
    broken = text.replace(
        '"series": [{"type": "pie", "encode"',
        '"xAxis": {"type": "nonexistent"}, "series": [{"type": "pie", "encode"',
        1,
    )
    assert broken != text
    corrupt = broken.replace('id="chart-product">', 'id="chart-product">{ not json ', 1)
    path = tmp_path / "broken.html"
    path.write_text(corrupt, encoding="utf-8")

    page = open_report(path, 1280)

    errors = page.eval(
        "() => [...document.querySelectorAll('.chart-error')].filter((n) => !n.hidden).map((n) => [n.closest('figure').dataset.chart, n.textContent])"
    )
    assert sorted(e[0] for e in errors) == ["mix", "product"]
    assert all(e[1].startswith("This chart could not be drawn: ") for e in errors)
    assert probe.chart_facts(page, "trend") is not None and len(trend(page)["series"]) == 4
    assert page.eval("() => typeof Report.chart('trend').instance.getOption") == "function"


def test_the_report_follows_the_system_colour_scheme_while_it_is_open(
    built: dict[str, Path], open_report
) -> None:
    page = open_report(built["dashboard"], 1280, scheme="light")
    light = _PALETTE["light"]["categorical"][0]
    dark = _PALETTE["dark"]["categorical"][0]
    assert trend(page)["series"][0]["color"] == light

    page.page.emulate_media(color_scheme="dark")
    page.page.wait_for_timeout(600)

    assert trend(page)["series"][0]["color"] == dark
    background = colour.parse(page.eval("() => getComputedStyle(document.body).backgroundColor"))
    assert background == pytest.approx(
        colour.parse(_PALETTE["dark"]["roles"]["background"]), abs=1e-3
    )  # type: ignore[comparison-overlap]
    assert probe.failing_text(page) == []

    page.page.emulate_media(color_scheme="light")
    page.page.wait_for_timeout(600)
    assert trend(page)["series"][0]["color"] == light


def test_animation_is_off_under_reduced_motion(built: dict[str, Path], browser: Browser) -> None:
    context = browser.new_context(viewport={"width": 900, "height": 700}, reduced_motion="reduce")
    page = context.new_page()
    page.goto(built["brief"].as_uri())
    page.wait_for_function("() => document.documentElement.dataset.reportReady === 'true'")

    assert page.evaluate("() => Report.chart('trend').instance.getOption().animation") is False
    assert page.evaluate(
        "() => getComputedStyle(document.querySelector('.tab, .chip, .kpi') || document.body).transitionDuration"
    ) in ("0s", "")
    context.close()


def test_every_page_is_shown_when_printed(built: dict[str, Path], open_report) -> None:
    page = open_report(built["multipage"], 900)

    page.page.emulate_media(media="print")
    page.page.wait_for_timeout(700)

    assert page.eval(
        "() => [...document.querySelectorAll('[data-page]')].every((s) => getComputedStyle(s).display !== 'none')"
    )
    assert (
        page.eval("() => getComputedStyle(document.querySelector('[role=tablist]')).display")
        == "none"
    )
    assert all(
        probe.chart_facts(page, c) is not None for c in ("total", "flows", "balance", "rd", "share")
    )


def test_the_public_api_reads_and_changes_rows_and_slicers(
    built: dict[str, Path], open_report
) -> None:
    page = open_report(built["dashboard"], 1280)

    assert page.eval("() => Report.chart('nowhere')") is None
    assert page.eval("() => Report.slicer('nowhere')") is None
    assert page.eval("() => Report.chart('trend').rows().length") == 32
    page.eval("() => Report.slicer('metric').set('利润')")
    settle(page)
    assert page.eval("() => Report.slicer('metric').value()") == "利润"
    page.eval(
        "() => Report.slicer('metric').set('不是指标')"
    )  # an option that does not exist is ignored
    assert page.eval("() => Report.slicer('metric').value()") == "利润"

    page.eval(
        "() => ['trend', 'product', 'mix'].forEach((id) => { const c = Report.chart(id);"
        " c.setRows(c.rows().filter((r) => r.region === '华东')); })"
    )
    settle(page)
    assert [s["name"] for s in trend(page)["series"]] == ["华东"]
    # The slicer offers the values the rows have now.
    assert (
        page.eval("() => document.querySelectorAll('button.chip[data-value=\"华北\"]').length") == 0
    )
    page.eval("() => Report.chart('trend').refresh()")
    ready = page.eval("() => new Promise((resolve) => Report.ready(() => resolve('ran')))")
    assert ready == "ran"


def test_a_tap_shows_the_tooltip_and_the_legend_toggles_a_series(
    built: dict[str, Path], open_report
) -> None:
    page = open_report(built["dashboard"], 390, touch=True)
    point = page.eval(
        "() => { const i = Report.chart('product').instance; const b = document.querySelector('figure[data-chart=product] .chart-body').getBoundingClientRect();"
        " const x = i.convertToPixel({xAxisIndex: 0}, '2023Q2'); const y = i.convertToPixel({yAxisIndex: 0}, 200); return [b.x + x, b.y + y]; }"
    )
    page.page.touchscreen.tap(point[0], point[1])
    page.page.wait_for_timeout(500)

    tip = page.eval(
        "() => [...document.querySelectorAll('figure[data-chart=product] .chart-body div')].map((d) => d.textContent).find((t) => t.includes('2023Q2') && t.length < 80)"
    )
    assert tip and "2023Q2" in tip

    page.eval(
        "() => Report.chart('product').instance.dispatchAction({type: 'legendToggleSelect', name: 'A'})"
    )
    page.page.wait_for_timeout(300)
    assert (
        page.eval("() => Report.chart('product').instance.getOption().legend[0].selected.A")
        is False
    )


def test_a_single_chart_page_from_echarts_render_runs_in_the_sandbox_and_paints_the_palette(
    tmp_path: Path, open_report
) -> None:
    option = {
        "xAxis": {"type": "category", "data": ["甲", "乙"]},
        "yAxis": {"type": "value"},
        "series": [{"type": "bar", "data": [1, 2]}],
    }
    svg = echarts_render._draw(option, 600, 400)
    page_html = tmp_path / "page.html"
    page_html.write_text(echarts_render._html_page(option, svg, 600, 400), encoding="utf-8")

    page = open_report(page_html, 700, ready=False)  # this page has no report runtime to wait for
    page.frame.wait_for_function(
        "() => typeof echarts === 'object' && echarts.getInstanceByDom(document.getElementById('chart'))"
    )
    drawn = page.eval("() => document.getElementById('chart').innerHTML")

    assert _PALETTE["light"]["categorical"][0] in drawn
    assert page.observed.clean() and page.violations() == []
