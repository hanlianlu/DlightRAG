"""Self-test of the checker on its own fixtures (fixtures/build.py): the page built to satisfy every check must pass
them, and each broken page must fail exactly the checks it was built to break. Run: .venv/bin/python selftest_checker.py"""

import json
import sys
from pathlib import Path

import checker
from frame import load_product_frame
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
FIX = HERE / "fixtures" / "out"
OUT = HERE / "results" / "selftest"

EXPECT = {
    "good-dashboard.html": {
        "pass": [
            "load_ok",
            "no_external",
            "charts_via_echarts_runtime",
            "no_horizontal_overflow",
            "pages_or_sections",
            "slicer_changes_chart",
            "legend_not_dominant",
            "chart_min_size",
            "min_font",
            "tap_targets",
            "controls_change_output",
            "no_boilerplate",
            "timeline_or_events",
            "assumptions_stated",
            "no_text_overlap",
        ],
        "fail": ["restraint_no_tabs_no_slicers"],
    },
    "bad-hand-rolled.html": {
        "pass": [],
        "fail": [
            "load_ok",
            "no_external",
            "charts_via_echarts_runtime",
            "no_horizontal_overflow",
            "min_font",
            "tap_targets",
            "no_boilerplate",
            "pages_or_sections",
            "dark_scheme_legible",
        ],
    },
    "static-image.html": {
        "pass": [
            "load_ok",
            "no_external",
            "no_boilerplate",
            "restraint_no_tabs_no_slicers",
            "dark_scheme_legible",
        ],
        "fail": ["charts_via_echarts_runtime", "min_font"],
    },
    "bad-overlap.html": {
        "pass": ["load_ok", "charts_via_echarts_runtime"],
        "fail": ["no_text_overlap", "legend_not_dominant"],
    },
    "iframe-charts.html": {
        "pass": ["load_ok", "dark_scheme_legible"],
        "fail": ["charts_via_echarts_runtime", "min_font", "legend_not_dominant"],
    },
    # sliders and scenario buttons decide the charts through the runtime's event path (a frame later, then a debounce): every control has an effect
    "scenario-sliders.html": {
        "pass": [
            "load_ok",
            "no_external",
            "charts_via_echarts_runtime",
            "no_horizontal_overflow",
            "slicer_changes_chart",
            "slicer_coverage",
            "controls_change_output",
            "chart_min_size",
            "min_font",
            "tap_targets",
        ],
        "fail": ["restraint_no_tabs_no_slicers", "pages_or_sections"],
    },
}
EXPECT["scenario-sliders-slow.html"] = EXPECT[
    "scenario-sliders.html"
]  # the same page with a handler that redraws after 650 ms


def interaction_log(name: str) -> list[dict]:
    """What the checker's own interaction pass concluded for every control of a fixture, at one viewport."""
    product = load_product_frame()
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            sess = checker.Session(browser, product, 1280, 800)
            try:
                sess.load((FIX / name).read_text(encoding="utf-8"))
                sess.events.clear()
                pages, _ = checker.find_tabs(sess)
                return checker.run_interactions(sess, pages, set())
            finally:
                sess.close()
        finally:
            browser.close()


def scenario_interactions(problems: list[str], name: str, slow: bool) -> None:
    """Every scenario button and every slider of the page changes a chart, and the page is back at its base after each.

    Then the same pass with the waiting taken out (the page measured the instant after the control, as the checker once did) is printed, and on
    the SLOW page it must not see any slider change a chart: that proves the fixture is still the shape that caught the bug, and not a page
    whose charts follow instantly. (On the 100 ms page this depends on how fast the machine is, and a scenario button can pick up the late
    effect of the control before it: the other face of the same bug. Both are printed, not asserted.) On the slow page the effect lands after
    the first wait, so it must be found by the second look (`late_effect`).
    """
    print(f"== {name}: the interaction pass, control by control", flush=True)
    tried = [e for e in interaction_log(name) if "chart_data_changed" in e]
    for e in tried:
        print(
            f"   {e['kind']:7s} {e['label']:10s} charts changed {e['chart_data_changed']}, restored {(e.get('reset') or {}).get('restored')}"
            + (", landed late" if e.get("late_effect") else "")
        )
    if len(tried) != 5 or sum(1 for e in tried if e["kind"] == "range") != 3:
        problems.append(
            f"{name}: expected 2 scenario buttons and 3 sliders to be tried, got {[(e['kind'], e['label']) for e in tried]}"
        )
    for e in tried:
        if not e["chart_data_changed"]:
            problems.append(
                f"{name}: {e['kind']} '{e['label']}' changed no chart although its effect lands a frame and a debounce later"
            )
        if not (e.get("reset") or {}).get("restored"):
            problems.append(
                f"{name}: {e['kind']} '{e['label']}' was not put back, so the next control would start from another state"
            )
        if slow and not e.get("late_effect"):
            problems.append(
                f"{name}: {e['kind']} '{e['label']}' should have been found by the second look (its chart follows after 650 ms)"
            )
        if not slow and e.get("late_effect"):
            problems.append(
                f"{name}: {e['kind']} '{e['label']}' needed the second look although its chart follows after 100 ms"
            )
    saved = checker.settled_snapshot
    checker.settled_snapshot = lambda sess, **_kw: checker._snapshot(sess)  # noqa: SLF001
    try:
        blind = [e for e in interaction_log(name) if "chart_data_changed" in e]
    finally:
        checker.settled_snapshot = saved
    print(
        f"   without the waiting, controls that appear to change a chart: {[e['label'] for e in blind if e['chart_data_changed']] or 'none'}"
    )
    changed = [e["label"] for e in blind if e["kind"] == "range" and e["chart_data_changed"]]
    if slow and changed:
        problems.append(
            f"{name}: measured without waiting, the sliders {changed} still change a chart: the fixture no longer catches a checker that does not wait"
        )


def main() -> int:
    problems = []
    wanted = sys.argv[1:]
    for name, expect in EXPECT.items():
        if wanted and name not in wanted:
            continue
        print(f"== {name}", flush=True)
        doc = checker.check_file(
            FIX / name,
            OUT / name.removesuffix(".html"),
            viewports=(360, 1280),
            heights=(800,),
            log=lambda s: None,
        )
        (OUT / name.removesuffix(".html") / "checks.json").write_text(
            json.dumps(doc, ensure_ascii=False, indent=1)
        )
        checks = doc["checks"]
        for key in expect["pass"]:
            if checks[key]["status"] != "pass":
                problems.append(
                    f"{name}: {key} expected pass, got {checks[key]['status']}: {checks[key]['reason']}"
                )
        for key in expect["fail"]:
            if checks[key]["status"] != "fail":
                problems.append(
                    f"{name}: {key} expected fail, got {checks[key]['status']}: {checks[key]['reason']}"
                )
        for key, value in checks.items():
            print(f"   {key:30s} {value['status']:5s} {value['reason'][:170]}")
    for name, slow in (("scenario-sliders.html", False), ("scenario-sliders-slow.html", True)):
        if not wanted or name in wanted:
            scenario_interactions(problems, name, slow)
    print()
    if problems:
        print("\n".join(problems))
        return 1
    print("checker self-test ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
