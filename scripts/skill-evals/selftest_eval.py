"""Offline self-test of the evaluation logic: what the numbers MEAN, checked against hand-computed scenarios.

No stack, no browser. Run: .venv/bin/python selftest_eval.py

  * routing: the precision / recall / forbidden-load formulas (with `allow`, with absent skills, with failed Runs), the per-case
    budgets and their precedence, the verdict kinds, the report text carries the definitions;
  * tasks: `html_published`, adoption of saved Runs, "judged by an older checker" detection, the table and its counts;
  * compare: directions (a lower forbidden-load rate is the better one), the changed-check list, older label shapes;
  * summary, variant helpers (SKILL.md names, label handling), the Run lock.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import textwrap
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import common
import compare
import routing
import summary
import tasks
import variant
from checker import CHECKER_VERSION
from driver import RunRecord


def _first_line(child: subprocess.Popen[str]) -> str:
    """The first line a child process prints (it says it holds the lock)."""
    if child.stdout is None:
        raise AssertionError("the child has no stdout pipe")
    return child.stdout.readline().strip()


def check(condition: object, message: object = "") -> None:
    """An assertion that is not an `assert` statement: it survives `python -O` and the repo's security gate (S101) for scripts/."""
    if not condition:
        raise AssertionError(message)


HERE = Path(__file__).resolve().parent
CATALOG = {"x", "y", "z", "w"}


def record_with(
    loaded: list[str], status: str = "succeeded", stopped_by: str = "all_loaded"
) -> RunRecord:
    return RunRecord(
        status=status,
        stopped_by=stopped_by,
        skills_loaded=[
            {"seq": i + 1, "batch": i + 1, "name": n, "path": "SKILL.md", "is_error": False}
            for i, n in enumerate(loaded)
        ],
    )


def sample(
    case: dict[str, Any],
    settings: dict[str, Any],
    loaded: list[str],
    n: int,
    status: str = "cancelled",
) -> dict[str, Any]:
    rec = record_with(loaded, status=status)
    return {
        "case": case["id"],
        "n": n,
        "status": status,
        "stopped_by": "all_loaded",
        "elapsed_s": 1.0,
        "tool_count": len(loaded),
        "tool_names": ["load_skill"] * len(loaded),
        "tool_calls": [],
        "notes": [],
        **routing.score_sample(case, settings, CATALOG, rec),
    }


def opts(**kw: Any) -> argparse.Namespace:
    base = {"max_tool_calls": None, "max_seconds": None, "skill_batches": None}
    base.update(kw)
    return argparse.Namespace(**base)


def scenario() -> tuple[
    list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]
]:
    cases = [
        {
            "id": "A",
            "q": "a",
            "load": ["x", "y", "q"],
            "allow": ["z"],
            "forbid": ["w"],
        },  # q is absent from the catalog
        {"id": "B", "q": "b", "load": [], "forbid": ["x"]},
        {"id": "C", "q": "c", "load": ["z"], "forbid": []},
        {
            "id": "D",
            "q": "d",
            "load": [],
            "forbid": ["q"],
        },  # forbids only a skill the catalog lacks
    ]
    settings = {c["id"]: routing.case_settings(c, CATALOG, opts()) for c in cases}
    by_case = {
        "A": [
            sample(cases[0], settings["A"], ["x", "y"], 1),
            sample(cases[0], settings["A"], ["x", "z"], 2),
            sample(cases[0], settings["A"], ["x", "w"], 3),
        ],
        "B": [
            sample(cases[1], settings["B"], [], 1),
            sample(cases[1], settings["B"], ["x"], 2),
            sample(cases[1], settings["B"], [], 3, status="failed"),
        ],
        "C": [sample(cases[2], settings["C"], ["z", "w"], 1)],
        "D": [sample(cases[3], settings["D"], [], 1), sample(cases[3], settings["D"], ["w"], 2)],
    }
    return cases, settings, by_case


def test_routing_metrics_match_a_hand_computed_scenario() -> None:
    cases, settings, by_case = scenario()
    m = routing.routing_metrics(
        cases, by_case, settings, CATALOG | set(), ["x", "y", "z", "w", "q"]
    )
    per = m["per_skill"]
    # x: expected in A (3 samples), loaded in all 3.   TP 3 (A) , FP 1 (B2: loaded x though B forbids it)
    check(per["x"]["recall"] == {"k": 3, "n": 3}, per["x"])
    check(
        (per["x"]["precision"]["tp"], per["x"]["precision"]["fp"], per["x"]["precision"]["allowed"])
        == (3, 1, 0),
        per["x"],
    )
    check(
        per["x"]["forbidden_load"] == {"k": 1, "n": 2},
        "B forbids x: 2 scored samples (the failed Run is left out), x loaded in 1",
    )
    # y: expected in A, loaded once
    check(per["y"]["recall"] == {"k": 1, "n": 3})
    check((per["y"]["precision"]["tp"], per["y"]["precision"]["fp"]) == (1, 0))
    # z: expected in C (1/1); A's z is allowed: in neither TP nor FP
    check(per["z"]["recall"] == {"k": 1, "n": 1})
    check(
        (per["z"]["precision"]["tp"], per["z"]["precision"]["fp"], per["z"]["precision"]["allowed"])
        == (1, 0, 1),
        per["z"],
    )
    # w: never expected; loaded in A3 (forbidden there), C1 and D2 (neither load nor allow): 3 FP
    check(per["w"]["recall"] == {"k": 0, "n": 0})
    check((per["w"]["precision"]["tp"], per["w"]["precision"]["fp"]) == (0, 3), per["w"])
    check(per["w"]["forbidden_load"] == {"k": 1, "n": 3}, "A forbids w: 3 samples, loaded in 1")
    # q is absent: never counted
    check(per["q"]["in_catalog"] is False or "q" not in CATALOG)
    o = m["overall"]
    check(o["recall"] == {"k": 5, "n": 7}, o["recall"])
    check(
        (o["precision"]["tp"], o["precision"]["n"], o["precision"]["allowed"]) == (5, 9, 1),
        o["precision"],
    )
    check(
        o["forbidden_load"] == {"k": 2, "n": 5},
        "samples of A (3) and B (2 scored); D forbids only an absent skill and does not count",
    )
    check(o["excluded_failed"] == 1)


def test_the_forbidden_load_of_an_absent_skill_is_na_not_zero() -> None:
    cases, settings, by_case = scenario()
    m = routing.routing_metrics(cases, by_case, settings, CATALOG, ["x", "q"])
    check(
        m["per_skill"]["q"]["forbidden_load"] == {"k": 0, "n": 0},
        "q is absent: no sample could violate it",
    )


def test_case_verdict_kinds() -> None:
    cases, settings, by_case = scenario()
    a = routing.case_verdict(cases[0], settings["A"], by_case["A"])
    check(
        (a["kind"], a["ok"], a["n"], a["label"]) == ("full", 1, 3, "PARTIAL 1/3"), a
    )  # needs x AND y (q is absent), no w
    b = routing.case_verdict(cases[1], settings["B"], by_case["B"])
    check((b["ok"], b["n"], b["failed_runs"]) == (1, 2, 1) and b["label"] == "PARTIAL 1/2", b)
    only_absent = {"id": "E", "q": "e", "load": ["q"], "forbid": []}
    st = routing.case_settings(only_absent, CATALOG, opts())
    check(
        routing.case_verdict(only_absent, st, [sample(only_absent, st, [], 1)])["kind"] == "absent"
    )
    forbid_only = {"id": "F", "q": "f", "load": ["q"], "forbid": ["w"]}
    st = routing.case_settings(forbid_only, CATALOG, opts())
    v = routing.case_verdict(forbid_only, st, [sample(forbid_only, st, ["w"], 1)])
    check(v["kind"] == "forbid-only" and v["label"] == "forbid-only 0/1", v)
    absent_forbid = {"id": "G", "q": "g", "load": ["q"], "forbid": ["q2"]}
    st = routing.case_settings(absent_forbid, CATALOG, opts())
    check(
        routing.case_verdict(absent_forbid, st, [sample(absent_forbid, st, [], 1)])["kind"]
        == "absent",
        "a forbid of an absent skill judges nothing",
    )


def test_a_sample_that_loaded_a_skill_the_catalog_lacks_is_noted_not_scored() -> None:
    case = {"id": "A", "q": "a", "load": ["x", "q"], "forbid": []}
    st = routing.case_settings(case, CATALOG, opts())
    s = sample(case, st, ["q", "x"], 1)
    check(s["attempted_absent"] == ["q"] and s["loaded"] == ["x"] and s["recall_ok"] is True)


def test_budgets_cases_own_fields_cli_overrides_and_defaults() -> None:
    plain = routing.case_settings(
        {"id": "p", "q": "p", "load": ["x"], "forbid": []}, CATALOG, opts()
    )
    check((plain["max_tool_calls"], plain["max_seconds"]) == (8, 180.0))
    own = routing.case_settings(
        {
            "id": "o",
            "q": "o",
            "load": ["x"],
            "forbid": [],
            "max_tool_calls": 40,
            "max_seconds": 420,
        },
        CATALOG,
        opts(),
    )
    check((own["max_tool_calls"], own["max_seconds"]) == (40, 420.0))
    forced = routing.case_settings(
        {
            "id": "o",
            "q": "o",
            "load": ["x"],
            "forbid": [],
            "max_tool_calls": 40,
            "max_seconds": 420,
        },
        CATALOG,
        opts(max_tool_calls=5, max_seconds=60.0, skill_batches=2),
    )
    check(
        (forced["max_tool_calls"], forced["max_seconds"], forced["batches"]) == (5, 60.0, 2),
        "the command line overrides the case",
    )
    stop = routing.early_stop_for(forced)
    check(stop.batches == 2 and stop.required == ("x",))


def test_required_skills_are_the_catalog_subset_of_load_and_history_is_counted() -> None:
    st = routing.case_settings(
        {
            "id": "r",
            "q": "r",
            "load": ["interactive-html", "x"],
            "forbid": ["w"],
            "allow": ["z"],
            "history": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}],
        },
        CATALOG,
        opts(),
    )
    check(
        st["required"] == ["x"]
        and st["absent"] == ["interactive-html"]
        and st["history_turns"] == 2
        and st["allow"] == ["z"]
    )


def test_a_saved_sample_of_an_edited_case_is_not_reused() -> None:
    case = {"id": "r", "q": "q1", "load": ["x"], "forbid": []}
    st = routing.case_settings(case, CATALOG, opts())
    edited = {**case, "q": "q2"}
    check(
        routing.case_fingerprint(case, st)
        != routing.case_fingerprint(edited, routing.case_settings(edited, CATALOG, opts()))
    )
    widened = routing.case_settings({**case, "max_tool_calls": 40}, CATALOG, opts())
    check(
        routing.case_fingerprint(case, st) != routing.case_fingerprint(case, widened),
        "a changed budget is a changed definition",
    )


def test_routing_report_prints_the_formulas_and_the_note() -> None:
    cases, settings, by_case = scenario()
    meta = {"catalog": sorted(CATALOG), "answering_model": "m", "query_model_preset": "deepseek"}
    doc = {
        "label": "t",
        "meta": meta,
        "params": {"samples": 3, "skill_batches": None},
        "cases": [
            {
                "id": c["id"],
                "q": c["q"],
                "settings": settings[c["id"]],
                "samples": by_case[c["id"]],
                "verdict": routing.case_verdict(c, settings[c["id"]], by_case[c["id"]]),
            }
            for c in cases
        ],
        "metrics": routing.routing_metrics(
            cases, by_case, settings, CATALOG, ["x", "y", "z", "w", "q"]
        ),
    }
    md = routing.render_routing_md(doc)
    for needle in (
        "precision(s)** = TP / (TP + FP)",
        "allow",
        "forbidden-load rate(s)",
        "Stop rule",
        "1 failed Run(s) left out",
        "PARTIAL 1/3",
        "allowed 1",
    ):
        check(needle in md, needle)


def run_dir_with(
    root: Path,
    label: str,
    name: str,
    record: RunRecord,
    checks: dict[str, Any] | None = None,
    version: str = CHECKER_VERSION,
    extra_files: dict[str, str] | None = None,
) -> Path:
    d = root / label / "tasks" / name
    d.mkdir(parents=True)
    for art in record.artifacts:
        if art.get("saved_as"):
            (d / art["saved_as"]).parent.mkdir(parents=True, exist_ok=True)
            (d / art["saved_as"]).write_text("<html>report</html>")
    (d / "run.json").write_text(json.dumps(record.to_json()))
    (d / "tools.json").write_text("{}")
    (d / "answer.md").write_text(record.answer)
    for rel, text in (extra_files or {}).items():
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text(text)
    if checks is not None:
        (d / "checks.json").write_text(
            json.dumps({"checker_version": version, "checks": checks, "report": True})
        )
    return d


def html_artifact(name: str = "report.html", size: int = 60_000) -> dict[str, Any]:
    return {
        "resource_id": "artifact-1",
        "media_type": "text/html",
        "filename": name,
        "byte_size": size,
        "status": "available",
        "saved_as": f"artifacts/{name}",
    }


def call_record(name: str, path: str, failed: bool, preview: str = "") -> dict[str, Any]:
    return {
        "seq": 1,
        "batch": 1,
        "name": name,
        "args": {"path": path},
        "is_error": failed,
        "result_preview": preview,
    }


def test_html_published_pass_and_every_way_to_fail() -> None:
    ok = RunRecord(
        status="succeeded",
        artifacts=[html_artifact()],
        tool_calls=[call_record("attach_artifact", "report.html", False)],
    )
    check(
        tasks.html_published_check(ok)["status"] == "pass"
        and "report.html" in tasks.html_published_check(ok)["reason"]
    )
    png = RunRecord(
        status="succeeded",
        artifacts=[
            {"filename": "c.png", "media_type": "image/png", "status": "available", "byte_size": 9}
        ],
        tool_calls=[call_record("attach_artifact", "c.png", False)],
    )
    r = tasks.html_published_check(png)
    check(r["status"] == "fail" and "c.png" in r["reason"] and "no HTML" in r["reason"], r)
    pasted_html = "```html\n<!doctype html><html>" + "x" * 2000 + "</html>\n```"
    failed = RunRecord(
        status="succeeded",
        answer=pasted_html,
        artifacts=[],
        tool_calls=[
            call_record(
                "attach_artifact",
                "r.html",
                True,
                "unsafe_file: the Artifact root contains an unsafe entry",
            )
        ],
    )
    r = tasks.html_published_check(failed)
    check(
        r["status"] == "fail"
        and "attach_artifact" in r["reason"]
        and "unsafe_file" in r["reason"]
        and "answer text only" in r["reason"],
        r,
    )
    never = RunRecord(status="succeeded", answer="done", artifacts=[], tool_calls=[])
    check("no attach_artifact call" in tasks.html_published_check(never)["reason"])
    dead = RunRecord(status="cancelled", stopped_by="timeout", artifacts=[])
    check("cancelled" in tasks.html_published_check(dead)["reason"])
    issue = RunRecord(
        status="succeeded",
        artifacts=[{"filename": "r.html", "media_type": "text/html", "status": "unavailable"}],
        artifact_outcome={
            "status": "failed",
            "issues": [{"kind": "too_large", "description": "over 20 MiB"}],
        },
    )
    r = tasks.html_published_check(issue)
    check(
        r["status"] == "fail" and "too_large" in r["reason"],
        "an artifact the product marked unavailable is not published",
    )


def test_adoption_copies_the_run_but_not_its_old_judging() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        old, new = common.RESULTS, tasks.RESULTS
        common.RESULTS = tasks.RESULTS = root
        try:
            rec = RunRecord(status="succeeded", artifacts=[html_artifact()])
            run_dir_with(
                root,
                "base",
                "t2-multipage-1",
                rec,
                checks={"load_ok": {"status": "pass", "reason": ""}},
                version="1",
                extra_files={
                    "report.html": "stale",
                    "shots/a.png": "x",
                    "contact.png": "x",
                    "artifacts/from-answer.html": "pasted",
                },
            )
            out = root / "new"
            (out / "tasks").mkdir(parents=True)
            adopted = tasks.adopt_runs(
                out, ["base:t2-multipage"], {"t2-multipage", "t5-brief"}, lambda s: None
            )
            check(adopted == {"base": ["t2-multipage-1"]})
            dst = out / "tasks" / "t2-multipage-1"
            names = sorted(str(p.relative_to(dst)) for p in dst.rglob("*") if p.is_file())
            check(
                names
                == ["adopted.json", "answer.md", "artifacts/report.html", "run.json", "tools.json"],
                names,
            )
            check(json.loads((dst / "adopted.json").read_text())["judged_before_by"] == "1")
            check(tasks.needs_judging(dst, recheck=False), "no checks.json yet")
            (dst / "run.json").write_text("changed")
            tasks.adopt_runs(out, ["base:t2-multipage"], {"t2-multipage"}, lambda s: None)
            check(
                (dst / "run.json").read_text() == "changed",
                "a Run already in the label is never overwritten",
            )
            for bad in (["base:nope"], ["nolabel"], ["ghost:t2-multipage"]):
                try:
                    tasks.adopt_runs(out, bad, {"t2-multipage"}, lambda s: None)
                except SystemExit:
                    continue
                raise AssertionError(f"{bad} must be refused")
        finally:
            common.RESULTS, tasks.RESULTS = old, new


def test_judging_is_out_of_date_when_the_checker_version_differs() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        check(tasks.needs_judging(d, recheck=False), "no checks.json")
        (d / "checks.json").write_text(
            json.dumps({"checker_version": "1", "checks": {"html_published": {}}})
        )
        check(tasks.needs_judging(d, recheck=False), "judged by version 1")
        (d / "checks.json").write_text(
            json.dumps({"checker_version": CHECKER_VERSION, "checks": {}})
        )
        check(tasks.needs_judging(d, recheck=False), "html_published missing")
        (d / "checks.json").write_text(
            json.dumps({"checker_version": CHECKER_VERSION, "checks": {"html_published": {}}})
        )
        check(not tasks.needs_judging(d, recheck=False))
        check(tasks.needs_judging(d, recheck=True), "--recheck forces it")
        (d / "checks.json").write_text("{not json")
        check(tasks.needs_judging(d, recheck=False))


def make_task_label(
    root: Path, label: str, published: bool, with_pasted: bool = False, adopted: bool = False
) -> None:
    """A label with one t1 Run (judged) and one t5 Run, without any browser."""
    task_t1 = {
        "id": "t1-sales",
        "kind": "filterable-dashboard",
        "viewports": [390],
        "must": [
            "html_published",
            "slicer_changes_chart",
            "charts_via_echarts_runtime",
            "no_boilerplate",
        ],
    }
    pass_, fail_ = {"status": "pass", "reason": "ok"}, {"status": "fail", "reason": "bad"}
    arts = [html_artifact()] if published else []
    rec = RunRecord(
        status="succeeded",
        elapsed_s=100.0,
        artifacts=arts,
        answer="done",
        usage={"input_tokens": 1, "output_tokens": 2},
        tool_calls=[
            {
                "seq": 1,
                "batch": 1,
                "name": "load_skill",
                "args": {"name": "interactive-html"},
                "is_error": False,
            },
            {
                "seq": 2,
                "batch": 2,
                "name": "load_skill",
                "args": {"name": "interactive-html", "path": "references/x.md"},
                "is_error": False,
            },
        ],
        skills_loaded=[
            {
                "seq": 1,
                "batch": 1,
                "name": "interactive-html",
                "path": "SKILL.md",
                "is_error": False,
            },
            {
                "seq": 2,
                "batch": 2,
                "name": "interactive-html",
                "path": "references/x.md",
                "is_error": False,
            },
        ],
        tool_count=2,
    )
    pub = tasks.html_published_check(rec)
    checks = {"html_published": pub, "skills_loaded": tasks.skills_check(tasks.tool_summary(rec))}
    if published:
        checks.update(
            {
                "slicer_changes_chart": pass_,
                "charts_via_echarts_runtime": fail_ if not adopted else pass_,
                "no_boilerplate": {
                    "status": "fail",
                    "reason": "1 hit",
                    "details": {
                        "hits": [
                            {
                                "phrase": "仅供参考",
                                "category": "disclaimer",
                                "element": "p",
                                "context": "...仅供参考...",
                                "path": "p",
                                "sentence": "仅供参考。",
                            }
                        ]
                    },
                },
                "pages_or_sections": fail_,
                "load_ok": pass_,
            }
        )
    d = run_dir_with(root, label, "t1-sales-1", rec, checks=checks)
    doc = json.loads((d / "checks.json").read_text())
    doc.update(
        {
            "report": published,
            "task": {
                "id": task_t1["id"],
                "verdict": tasks.verdict(task_t1, checks)[0],
                "failing": tasks.verdict(task_t1, checks)[1],
            },
            "pasted_html": {"chars": 5000, "checks": {}} if with_pasted else None,
        }
    )
    (d / "checks.json").write_text(json.dumps(doc))
    if adopted:
        (d / "adopted.json").write_text(json.dumps({"from_label": "base"}))
    (root / label / "tasks-meta.json").write_text(
        json.dumps(
            {
                "answering_model": "m",
                "catalog": ["interactive-html", "charts"],
                "query_model_preset": "deepseek",
            }
        )
    )


def test_the_tasks_table_counts_only_what_a_task_asks_for_and_marks_adoption() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        make_task_label(root, "lab", published=True, adopted=True)
        spec = common.load_cases(common.DEFAULT_CASES)
        results = tasks.collect_results(root / "lab", spec["tasks"])
        md = tasks.render_tasks_md(
            "lab",
            json.loads((root / "lab" / "tasks-meta.json").read_text()),
            spec["tasks"],
            results,
        )
        header = [line for line in md.splitlines() if line.startswith("| task-n")][0]
        check(
            header.split(" | ")[3].strip() == "html_published",
            "html_published is the first check column",
        )
        row = [line for line in md.splitlines() if line.startswith("| t1-sales-1")][0]
        check("(adopted)" in row and "**pass**" in row)
        summary_part = md.split("## Summary", 1)[1].split("## Why", 1)[0]
        check(
            "pages_or_sections" not in summary_part,
            "t1 no longer asks for pages: its failing pages check is not counted",
        )
        check("no_boilerplate | 0 | 1 | 0" in summary_part)
        check(
            "interactive-html" in md and "references/x.md" in md,
            "who loaded what, with the reference file",
        )


def test_a_run_without_a_published_report_fails_on_html_published_and_is_not_judged_further() -> (
    None
):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        make_task_label(root, "lab", published=False, with_pasted=True)
        spec = common.load_cases(common.DEFAULT_CASES)
        results = tasks.collect_results(root / "lab", spec["tasks"])
        res = results["t1-sales-1"]
        check(res["verdict"] == "FAIL" and res["failing"][0] == "html_published", res["failing"])
        md = tasks.render_tasks_md("lab", {}, spec["tasks"], results)
        check("Pasted, not published" in md)
        row = [line for line in md.splitlines() if line.startswith("| t1-sales-1")][0]
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        check(
            cells.count("-") == len(tasks.ALL_CHECKS) - 1 and "**FAIL**" in cells,
            ("everything that needs a report is shown as not evaluated", cells),
        )
        why = md.split("## Why the must-checks fail", 1)[1].split("\n## ", 1)[0]
        check("html_published" in why and "slicer_changes_chart" not in why)


def test_compare_directions_and_changed_checks() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        old = common.RESULTS
        common.RESULTS = root
        try:

            def routing_doc(
                label: str,
                recall: tuple[int, int],
                precision: tuple[int, int],
                forb: tuple[int, int],
            ) -> dict[str, Any]:
                m = {
                    "recall": {"k": recall[0], "n": recall[1]},
                    "precision": {
                        "k": precision[0],
                        "n": precision[1],
                        "tp": precision[0],
                        "fp": precision[1] - precision[0],
                        "allowed": 0,
                    },
                    "forbidden_load": {"k": forb[0], "n": forb[1]},
                }
                return {
                    "label": label,
                    "meta": {"catalog": ["charts"], "answering_model": "m"},
                    "params": {"samples": 3, "cases_file_fingerprint": "f1"},
                    "metrics": {
                        "per_skill": {"charts": {"in_catalog": True, **m}},
                        "overall": {**m, "forbidden_load": m["forbidden_load"]},
                    },
                    "cases": [
                        {
                            "id": "r01",
                            "settings": {},
                            "samples": [],
                            "verdict": {
                                "ok": recall[0] // 4,
                                "n": 3,
                                "kind": "full",
                                "label": f"PARTIAL {recall[0] // 4}/3",
                            },
                        }
                    ],
                }

            for label, doc in (
                ("a", routing_doc("a", (6, 12), (6, 8), (1, 10))),
                ("b", routing_doc("b", (9, 12), (6, 12), (3, 10))),
            ):
                (root / label).mkdir()
                (root / label / "routing.json").write_text(json.dumps(doc))
            text = compare.compare_labels("a", "b", common.DEFAULT_CASES, color=False)
            lines = {
                line.split()[0]: line
                for line in text.splitlines()
                if line.strip() and line.split()[0] in ("charts", "OVERALL")
            }
            check("▲ +25pp" in lines["charts"], "recall 50% -> 75% is better")
            check("▼ -25pp" in lines["charts"], "precision 75% -> 50% is worse")
            check(
                text.count("▼ +20pp") == 2,
                "forbidden-load 10% -> 30% rose: that is WORSE (per skill and overall)",
            )
            check(
                "routing   recall 50% → 75% ▲ | precision 75% → 50% ▼ | forbidden-load 10% → 30% ▼"
                in text,
                text.split("ROUTING")[0],
            )
            colored = compare.compare_labels("a", "b", common.DEFAULT_CASES, color=True)
            check("\033[32m▲" in colored and "\033[31m▼" in colored)

            # tasks: the same label content twice, B better on html_published
            make_task_label(root, "ta", published=False)
            make_task_label(root, "tb", published=True)
            for label in ("ta", "tb"):
                spec = common.load_cases(common.DEFAULT_CASES)
                results = tasks.collect_results(root / label, spec["tasks"])
                (root / label / "tasks.json").write_text(
                    json.dumps(tasks.tasks_json(label, results))
                )
            text = compare.compare_labels("ta", "tb", common.DEFAULT_CASES, color=False)
            check("▲ t1-sales" in text and "html_published*" in text, text)
            check(
                "pages_or_sections" not in text,
                "t1 does not ask for pages: its failing pages check is not a result (as in tasks.md)",
            )
            check("who loaded what" in text and "interactive-html" in text)
        finally:
            common.RESULTS = old


def routing_doc_from(
    label: str,
    cases: list[dict[str, Any]],
    settings: dict[str, dict[str, Any]],
    by_case: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """A routing.json built with the real scoring functions, so compare is tested on what the harness writes."""
    metrics = routing.routing_metrics(cases, by_case, settings, CATALOG, ["x", "y", "z", "w", "q"])
    return {
        "label": label,
        "meta": {"catalog": sorted(CATALOG), "answering_model": "m"},
        "params": {"samples": 3, "cases_file_fingerprint": "f"},
        "metrics": metrics,
        "cases": [
            {
                "id": c["id"],
                "q": c["q"],
                "settings": settings[c["id"]],
                "samples": by_case[c["id"]],
                "verdict": routing.case_verdict(c, settings[c["id"]], by_case[c["id"]]),
            }
            for c in cases
        ],
    }


def test_compare_on_a_subset_is_like_for_like_and_sample_counts_are_not_a_change() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        old = common.RESULTS
        common.RESULTS = root
        try:
            cases, settings, by_case = scenario()
            by_case = {
                **by_case,
                "C": [sample(cases[2], settings["C"], ["z"], n) for n in (1, 2, 3)],
            }  # C passes three times in the full run
            full = routing_doc_from("full", cases, settings, by_case)
            # B ran only cases A and C: one sample of A (the first: both skills loaded) and one of C
            sub_by_case = {"A": by_case["A"][:1], "C": [sample(cases[2], settings["C"], ["z"], 1)]}
            sub = routing_doc_from(
                "sub", [cases[0], cases[2]], {k: settings[k] for k in ("A", "C")}, sub_by_case
            )
            for label, doc in (("full", full), ("sub", sub)):
                (root / label).mkdir()
                (root / label / "routing.json").write_text(json.dumps(doc))
            text = compare.compare_labels("full", "sub", common.DEFAULT_CASES, color=False)
            check("A ran 4 case(s) and B 2: compared on the 2 they share" in text, text)
            check(
                "both sides' metrics are recomputed over the 2 compared case(s): A, C" in text, text
            )
            row = {
                line.split()[0]: line
                for line in text.splitlines()
                if line.strip() and line.split()[0] in ("x", "y", "z")
            }
            # A's metrics are recomputed over cases A and C only: y is expected in A's 3 samples and loaded in 1; B's single sample loaded it
            check(
                "33% (1/3)" in row["y"] and "100% (1/1)" in row["y"] and "▲ +67pp" in row["y"],
                row["y"],
            )
            # x is forbidden in case B alone, which B did not run: the restricted forbidden-load of x is empty on both sides, not 1/2
            check("50%" not in row["x"] and row["x"].rstrip().endswith("-"), row["x"])
            # case A moved (1/3 -> 1/1); case C is the same pass-rate with other sample counts (3/3 vs 1/1): not a change
            moved = text.split("cases whose verdict moved:", 1)[1].split("unchanged", 1)[0]
            check(
                "PARTIAL 1/3" in moved
                and "PASS 1/1" in moved
                and "▲" in moved
                and "\n    C " not in moved,
                moved,
            )
            check("unchanged (1): C" in text, text)
            # the same label against itself on the shared cases shows nothing moved
            same = compare.compare_labels("full", "full", common.DEFAULT_CASES, color=False)
            check("no routing case changed its verdict" in same and "compared on" not in same)
        finally:
            common.RESULTS = old


def test_compare_leaves_out_a_case_that_was_edited_between_the_labels() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        old = common.RESULTS
        common.RESULTS = root
        try:
            cases, settings, by_case = scenario()
            first = routing_doc_from("first", cases, settings, by_case)
            edited_cases = [
                {**cases[0], "q": "a, reworded"},
                *cases[1:],
            ]  # the message of case A changed; nothing else did
            edited_settings = {
                c["id"]: routing.case_settings(c, CATALOG, opts()) for c in edited_cases
            }
            second = routing_doc_from("second", edited_cases, edited_settings, by_case)
            budget_cases = [
                cases[0],
                {**cases[1], "max_tool_calls": 40},
                *cases[2:],
            ]  # the budget of case B was raised: also not the same case
            budget_settings = {
                c["id"]: routing.case_settings(c, CATALOG, opts()) for c in budget_cases
            }
            third = routing_doc_from("third", budget_cases, budget_settings, by_case)
            for label, doc in (("first", first), ("second", second), ("third", third)):
                (root / label).mkdir()
                (root / label / "routing.json").write_text(json.dumps(doc))
            text = compare.compare_labels("first", "second", common.DEFAULT_CASES, color=False)
            check(
                "left out because the case itself was edited between A and B (message, expectations, budget or history): A"
                in text,
                text,
            )
            check("recomputed over the 3 compared case(s): B, C, D" in text, text)
            check(
                "A ran 4 case(s) and B 4" not in text,
                "the same four cases were run: only the edited one is left out",
            )
            text = compare.compare_labels("first", "third", common.DEFAULT_CASES, color=False)
            check(
                "budget or history): B" in text
                and "recomputed over the 3 compared case(s): A, C, D" in text,
                text,
            )
            same = compare.compare_labels("first", "first", common.DEFAULT_CASES, color=False)
            check(
                "left out because" not in same
                and "recomputed" not in same
                and "unchanged (4)" in same,
                same,
            )
        finally:
            common.RESULTS = old


def test_compare_on_a_subset_of_tasks_counts_only_the_tasks_both_ran() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        old = common.RESULTS
        common.RESULTS = root
        try:
            spec = common.load_cases(common.DEFAULT_CASES)
            make_task_label(root, "ta", published=True)  # a t1 Run
            make_task_label(root, "tb", published=True)
            # B also has a t5 Run that A lacks
            t5 = next(t for t in spec["tasks"] if t["id"] == "t5-brief")
            rec = RunRecord(
                status="succeeded",
                elapsed_s=10.0,
                artifacts=[html_artifact()],
                answer="x",
                tool_count=1,
            )
            run_dir_with(
                root,
                "tb",
                "t5-brief-1",
                rec,
                checks={
                    "html_published": tasks.html_published_check(rec),
                    "restraint_no_tabs_no_slicers": {"status": "pass", "reason": "ok"},
                },
            )
            for label in ("ta", "tb"):
                results = tasks.collect_results(root / label, spec["tasks"])
                (root / label / "tasks.json").write_text(
                    json.dumps(tasks.tasks_json(label, results))
                )
            text = compare.compare_labels("ta", "tb", common.DEFAULT_CASES, color=False)
            check(f"left out: {t5['id']} (only in B)" in text, text)
            header = [line for line in text.splitlines() if line.startswith("  check")][0]
            check("t1-sales" in header and "t5-brief" not in header, header)
            head = [line for line in text.splitlines() if line.strip().startswith("tasks ")][0]
            check(
                "Runs meeting every must-check 0/1 → 0/1" in head,
                (
                    "the headline counts the common task only (t1 fails charts_via_echarts_runtime on both sides)",
                    head,
                ),
            )
        finally:
            common.RESULTS = old


def test_summary_has_every_section() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        old, old_tasks = common.RESULTS, tasks.RESULTS
        common.RESULTS = tasks.RESULTS = root
        try:
            make_task_label(root, "lab", published=True)
            text = summary.build_summary("lab", common.DEFAULT_CASES)
            for needle in (
                "# SUMMARY: lab",
                "## Routing",
                "(no routing results in this label)",
                "## Tasks",
                "### Who loaded what (tasks",
                "## Boilerplate hits",
                "仅供参考",
            ):
                check(needle in text, needle)
            check("\n# Tasks" not in text, "embedded reports are demoted below the SUMMARY heading")
        finally:
            common.RESULTS, tasks.RESULTS = old, old_tasks


def test_skill_names_read_the_frontmatter_the_way_the_loader_names_a_skill() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "alpha").mkdir()
        (root / "alpha" / "SKILL.md").write_text(
            "---\nname: alpha\ndescription: Use when a.\n---\nbody"
        )
        (root / "beta-dir").mkdir()
        (root / "beta-dir" / "SKILL.md").write_text("---\nname: 'beta'\ndescription: x\n---\n")
        (root / "gamma").mkdir()
        (root / "gamma" / "SKILL.md").write_text("no frontmatter at all")
        (root / ".hidden").mkdir()
        (root / ".hidden" / "SKILL.md").write_text("---\nname: hidden\n---\n")
        (root / "notaskill").mkdir()
        check(variant.skill_names(root) == {"alpha": "alpha", "beta-dir": "beta", "gamma": "gamma"})


@contextlib.contextmanager
def scratch_results(root: Path) -> Iterator[None]:
    """Point the results directory and the Run lock at a scratch directory for the duration of a test."""
    saved = (common.RESULTS, common.STATE, common._LOCK_PATH)  # noqa: SLF001
    common.RESULTS, common.STATE, common._LOCK_PATH = (
        root / "results",
        root / "state",
        root / "state" / "runs.lock",
    )  # noqa: SLF001
    common.RESULTS.mkdir(parents=True, exist_ok=True)
    try:
        yield
    finally:
        common.RESULTS, common.STATE, common._LOCK_PATH = saved  # noqa: SLF001


def variant_opts(skills: Path, **kw: Any) -> argparse.Namespace:
    base: dict[str, Any] = dict(
        skills_dir=skills,
        toolkit_dir=None,
        what=[],
        cases_file=common.DEFAULT_CASES,
        cases="r01,n01",
        tasks="t1-sales",
        label="v1",
        resume=False,
        replace=False,
        down_after=False,
        verbose=False,
        no_color=True,
        against="base",
        model="deepseek",
        samples_routing=1,
        samples_tasks=1,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def make_skills(root: Path) -> Path:
    skills = root / "skills"
    (skills / "alpha").mkdir(parents=True)
    (skills / "alpha" / "SKILL.md").write_text(
        "---\nname: alpha\ndescription: Use when a.\n---\nbody"
    )
    return skills


def test_variant_label_handling() -> None:
    with tempfile.TemporaryDirectory() as tmp, scratch_results(Path(tmp)):
        root = common.RESULTS
        for bad in ("", "../x", "a/b", ".hidden", "with space", "x" * 80):
            try:
                variant.check_label(bad, resume=False, replace=False)
            except SystemExit as exc:
                check("plain name" in str(exc), exc)
            else:
                raise AssertionError(f"label {bad!r} must be refused")
        variant.check_label(
            "v1", resume=False, replace=False
        )  # does not exist yet: fine, and nothing is created by the check
        check(not (root / "v1").exists())
        out = variant.prepare_label("v1", replace=False)
        variant.check_label("v1", resume=False, replace=False)  # an empty directory is not a result
        (out / "x").write_text("1")
        try:
            variant.check_label("v1", resume=False, replace=False)
        except SystemExit as exc:
            check(
                "already exists" in str(exc) and "--resume" in str(exc) and "--replace" in str(exc)
            )
        else:
            raise AssertionError("an existing label must be refused")
        variant.check_label("v1", resume=True, replace=False)
        variant.check_label("v1", resume=False, replace=True)
        try:
            variant.check_label("v1", resume=True, replace=True)
        except SystemExit as exc:
            check("exclude each other" in str(exc))
        else:
            raise AssertionError("--resume with --replace must be refused")
        check(
            variant.prepare_label("v1", replace=False) == out and (out / "x").exists(),
            "resume keeps what is there",
        )
        with contextlib.redirect_stdout(io.StringIO()):
            fresh = variant.prepare_label("v1", replace=True)
        check(fresh.exists() and not (fresh / "x").exists())
        check(
            any(p.name.startswith("v1.old-") and (p / "x").exists() for p in root.iterdir()),
            "the old results are moved aside, not deleted",
        )


def test_variant_resume_refuses_another_variant() -> None:
    with tempfile.TemporaryDirectory() as tmp, scratch_results(Path(tmp)):
        out = variant.prepare_label("v1", replace=False)
        variant.check_resume(
            out, "aaa", None, "deepseek"
        )  # no variant.json yet (a label made by `routing` alone): nothing to compare
        (out / "variant.json").write_text(
            json.dumps({"skills_manifest": "aaa", "toolkit_manifest": None, "model": "deepseek"})
        )
        variant.check_resume(out, "aaa", None, "deepseek")
        for args, word in (
            (("bbb", None, "deepseek"), "Skill text"),
            (("aaa", "ttt", "deepseek"), "toolkit"),
            (("aaa", None, "glm"), "answering model"),
            (("bbb", "ttt", "glm"), "Skill text and toolkit and answering model"),
        ):
            try:
                variant.check_resume(out, *args)
            except SystemExit as exc:
                check(word in str(exc) and "--replace" in str(exc), exc)
            else:
                raise AssertionError(f"resuming with another {word} must be refused")


def test_a_refused_variant_leaves_results_untouched_so_the_same_command_can_be_run_again() -> None:
    with tempfile.TemporaryDirectory() as tmp, scratch_results(Path(tmp)):
        root = Path(tmp)
        skills = make_skills(root)
        saved = variant.make_ready
        calls: list[str] = []
        try:

            def refuse(*_a: Any, **_k: Any) -> dict[str, Any]:
                calls.append("make_ready")
                raise variant.VariantError(
                    "evaluate.py: the API does not list 'alpha' (a SKILL.md the loader rejects)"
                )

            variant.make_ready = refuse
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    variant.run_variant(variant_opts(skills))
                except SystemExit as exc:
                    check("does not list" in str(exc))
                else:
                    raise AssertionError(
                        "a stack that does not see the variant must stop the command"
                    )
            check(
                calls == ["make_ready"] and not list(common.RESULTS.iterdir()),
                "readiness failed: nothing may have been written to results/",
            )
            with common.runs_lock("proof the lock was released"):
                pass
            # an invalid selection stops before the stack is even looked at
            for bad in ({"cases": "r99"}, {"tasks": "t9"}):
                try:
                    variant.run_variant(variant_opts(skills, **bad))
                except SystemExit as exc:
                    check("unknown" in str(exc), exc)
                else:
                    raise AssertionError("an unknown case or task id must be refused")
            check(calls == ["make_ready"], "an invalid selection must not reach the stack")
            # a Skills directory without any SKILL.md is refused up front
            empty = root / "empty"
            empty.mkdir()
            try:
                variant.run_variant(variant_opts(empty))
            except SystemExit as exc:
                check("holds no <name>/SKILL.md" in str(exc))
            else:
                raise AssertionError("an empty skills directory must be refused")
        finally:
            variant.make_ready = saved


def test_a_variant_that_finds_the_lock_taken_writes_nothing() -> None:
    with tempfile.TemporaryDirectory() as tmp, scratch_results(Path(tmp)):
        root = Path(tmp)
        skills = make_skills(root)
        holder = subprocess.Popen(  # noqa: S603 - a child of this interpreter holding the lock
            [
                sys.executable,
                "-c",
                textwrap.dedent(f"""
                import sys, time
                sys.path.insert(0, {str(HERE)!r})
                import common
                from pathlib import Path
                common._LOCK_PATH = Path({str(common._LOCK_PATH)!r})
                common.STATE = Path({str(common.STATE)!r})
                with common.runs_lock("a running baseline"):
                    print("held", flush=True)
                    time.sleep(8)
            """),
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        saved = variant.make_ready
        reached: list[str] = []
        try:
            check(_first_line(holder) == "held")
            variant.make_ready = lambda *a, **k: reached.append("make_ready")  # type: ignore[assignment]
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    variant.run_variant(variant_opts(skills))
            except SystemExit as exc:
                check(
                    "another evaluation is submitting Runs" in str(exc)
                    and "a running baseline" in str(exc),
                    exc,
                )
            else:
                raise AssertionError("a variant must not run beside another evaluation")
            check(
                not reached and not list(common.RESULTS.iterdir()),
                "the stack must not be touched and no label directory may appear",
            )
        finally:
            variant.make_ready = saved
            holder.kill()
            holder.wait()


def test_a_variant_runs_both_evaluations_under_the_lock_and_a_failure_in_one_does_not_hide_the_table() -> (
    None
):
    with tempfile.TemporaryDirectory() as tmp, scratch_results(Path(tmp)):
        root = Path(tmp)
        skills = make_skills(root)
        (common.RESULTS / "base").mkdir()
        saved = (
            variant.make_ready,
            variant.routing.run_routing,
            variant.tasks.run_tasks,
            variant.summary.write_summary,
            variant.compare.compare_labels,
        )
        order: list[str] = []
        lock_held: list[bool] = []
        try:
            variant.make_ready = lambda *a, **k: {
                "was_down": False,
                "recreated": False,
                "before": "x",
                "after": "x",
            }  # type: ignore[assignment]

            def fake_routing(
                o: argparse.Namespace, log: Any = None, echo_report: bool = True
            ) -> dict[str, Any]:
                order.append(f"routing x{o.samples} {o.cases}")
                lock_held.append(common._lock_depth > 0)  # noqa: SLF001
                log("a routing log line")
                print("a stderr line from the driver", file=sys.stderr)
                raise RuntimeError("the provider fell over")

            def fake_tasks(
                o: argparse.Namespace, log: Any = None, echo_report: bool = True
            ) -> dict[str, Any]:
                order.append(f"tasks x{o.samples} {o.tasks}")
                lock_held.append(common._lock_depth > 0)  # noqa: SLF001
                return {}

            variant.routing.run_routing = fake_routing  # type: ignore[assignment]
            variant.tasks.run_tasks = fake_tasks  # type: ignore[assignment]
            variant.summary.write_summary = lambda label, cases_file: (
                common.RESULTS / label / "SUMMARY.md"
            )  # type: ignore[assignment]
            variant.compare.compare_labels = lambda a, b, cases_file, color=False, sections=(): (
                f"THE TABLE {a} vs {b} [{','.join(sections)}]\n"
            )  # type: ignore[assignment]
            out_text, err_text = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out_text), contextlib.redirect_stderr(err_text):
                rc = variant.run_variant(variant_opts(skills))
            check(order == ["routing x1 r01,n01", "tasks x1 t1-sales"], order)
            check(lock_held == [True, True], "both evaluations run under the Run lock")
            check(rc == 1, "a failed evaluation is a failed command")
            shown = out_text.getvalue()
            check(
                "THE TABLE base vs v1 [routing,tasks]" in shown,
                "the comparison is printed even though routing failed",
            )
            check(
                "routing: 2 case(s) x 1 sample(s)" in shown
                and "tasks: 1 task(s) x 1 sample(s)" in shown,
                shown,
            )
            check(
                "a routing log line" not in shown and "a stderr line from the driver" not in shown,
                "progress is kept out of the output",
            )
            check("FAILED routing: RuntimeError: the provider fell over" in err_text.getvalue())
            log = (common.RESULTS / "v1" / "logs" / "variant.log").read_text()
            check(
                "a routing log line" in log
                and "a stderr line from the driver" in log
                and "routing failed: RuntimeError" in log
            )
            meta = json.loads((common.RESULTS / "v1" / "variant.json").read_text())
            check(
                meta["failures"]
                and meta["model"] == "deepseek"
                and meta["skills_manifest"]
                and "finished_at" in meta,
                meta,
            )
            check(
                (common.RESULTS / "v1" / "skills-snapshot" / "alpha" / "SKILL.md").exists(),
                "the Skill text is kept next to its numbers",
            )
            check(
                (common.RESULTS / "v1" / "compare-vs-base.md")
                .read_text()
                .startswith("```text\nTHE TABLE base vs v1"),
                "the table is kept in the label",
            )
            with common.runs_lock("released"):
                pass
            # a second call with the same label is refused until --resume or --replace
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    variant.run_variant(variant_opts(skills))
            except SystemExit as exc:
                check("already exists" in str(exc))
            else:
                raise AssertionError(
                    "the label of an earlier variant must not be overwritten silently"
                )
            # evaluations can be chosen
            order.clear()
            shown_v2 = io.StringIO()
            with contextlib.redirect_stdout(shown_v2), contextlib.redirect_stderr(io.StringIO()):
                variant.run_variant(variant_opts(skills, label="v2", what=["tasks"]))
            check(order == ["tasks x1 t1-sales"], order)
            check(
                "THE TABLE base vs v2 [tasks]" in shown_v2.getvalue(),
                "the table covers only what was run",
            )
            # --verbose shows the progress and the driver's stderr on the terminal (and still keeps them in the log)
            verbose_err = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(verbose_err):
                variant.run_variant(
                    variant_opts(skills, label="v3", what=["routing"], verbose=True)
                )
            check(
                "a routing log line" in verbose_err.getvalue()
                and "a stderr line from the driver" in verbose_err.getvalue(),
                verbose_err.getvalue(),
            )
            check(
                "a routing log line" in (common.RESULTS / "v3" / "logs" / "variant.log").read_text()
            )
        finally:
            (
                variant.make_ready,
                variant.routing.run_routing,
                variant.tasks.run_tasks,
                variant.summary.write_summary,
                variant.compare.compare_labels,
            ) = saved


def test_the_runs_lock_is_reentrant_in_a_process_and_exclusive_across_processes() -> None:
    with tempfile.TemporaryDirectory() as tmp, scratch_results(Path(tmp)):
        tmp = str(common.STATE)
        holder = subprocess.Popen(  # noqa: S603 - a child of this interpreter holding the lock
            [
                sys.executable,
                "-c",
                textwrap.dedent(f"""
                import sys, time
                sys.path.insert(0, {str(HERE)!r})
                import common
                from pathlib import Path
                common._LOCK_PATH = Path({tmp!r}) / "runs.lock"
                common.STATE = Path({tmp!r})
                with common.runs_lock("the holder"):
                    print("held", flush=True)
                    time.sleep(8)
            """),
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        try:
            check(_first_line(holder) == "held")
            try:
                with common.runs_lock("the second"):
                    raise AssertionError("a second process must not get the lock")
            except SystemExit as exc:
                check(
                    "another evaluation is submitting Runs" in str(exc)
                    and "the holder" in str(exc),
                    exc,
                )
        finally:
            holder.kill()
            holder.wait()
        with common.runs_lock("outer"):
            with common.runs_lock("inner"):
                pass
        with common.runs_lock("again"):
            pass


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"ok   {test.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}")
    print(f"{len(tests) - failed}/{len(tests)} evaluation tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
