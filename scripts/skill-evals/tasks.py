"""The end-to-end `tasks` evaluation and the standalone `check` command (both need the browser).

A task Run lives in results/<label>/tasks/<task>-<n>/ :
  run.json tools.json answer.md artifacts/      the Run (written once, never changed by judging)
  checks.json                                   the verdict: the checker's document (when a report was published) plus
                                                html_published, skills_loaded and the task verdict; always written
  report.html shots/ contact*.png checker.log   the published report and what the checker saw (when one was published)
  adopted.json                                  present when the Run was copied from another label and judged again here

The label's tasks.md / tasks.json are always rebuilt from those directories, so a label can hold Runs of different origin.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import common
import driver
from checker import CHECKER_VERSION
from driver import Api, RunRecord, follow_run

HERE = Path(__file__).resolve().parent
RESULTS = common.RESULTS
HTML_TYPES = ("text/html", "application/xhtml+xml")
# Checks every report is held to whether or not the task asks for them (the task's `must` decides the verdict).
HYGIENE = (
    "load_ok",
    "no_external",
    "no_horizontal_overflow",
    "chart_min_size",
    "legend_not_dominant",
    "min_font",
    "tap_targets",
    "charts_via_echarts_runtime",
    "no_boilerplate",
    "no_text_overlap",
)
ALL_CHECKS = [
    "html_published",
    "pages_or_sections",
    "slicer_changes_chart",
    "controls_change_output",
    "charts_via_echarts_runtime",
    "timeline_or_events",
    "assumptions_stated",
    "restraint_no_tabs_no_slicers",
    "no_horizontal_overflow",
    "legend_not_dominant",
    "no_boilerplate",
    "chart_min_size",
    "no_text_overlap",
    "min_font",
    "tap_targets",
    "load_ok",
    "no_external",
]
MARK = {"pass": "pass", "fail": "FAIL", "n/a": "n/a", "info": "info"}


# ---------------------------------------------------------------------------------------------
# What the model did, from its tool calls
# ---------------------------------------------------------------------------------------------


def distinct_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The same phrase in the same sentence of the same element counts once, whichever patterns fired on it."""
    seen: set[tuple[str, str, str]] = set()
    out = []
    for h in hits:
        key = (h.get("path", ""), h.get("sentence", ""), h.get("phrase", ""))
        if key not in seen:
            seen.add(key)
            out.append(h)
    return out


def tool_summary(record: RunRecord) -> dict[str, Any]:
    calls = record.tool_calls
    names = Counter(c["name"] for c in calls)
    commands = [str(c["args"].get("command", "")) for c in calls if c["name"] == "bash"]

    def ran(pattern: str) -> list[dict[str, Any]]:
        rx = re.compile(pattern)
        return [
            {
                "seq": c["seq"],
                "ok": not c.get("is_error"),
                "cmd": str(c["args"].get("command", ""))[:140],
            }
            for c in calls
            if c["name"] == "bash" and rx.search(str(c["args"].get("command", "")))
        ]

    wrote = []
    for c in calls:
        if c["name"] == "write":
            path = str(
                c["args"].get("path")
                or c["args"].get("file_path")
                or c["args"].get("filename")
                or ""
            )
            wrote.append(
                {
                    "seq": c["seq"],
                    "path": path,
                    "chars": len(json.dumps(c["args"], ensure_ascii=False)),
                }
            )
    loads = record.skills_loaded
    return {
        "tool_count": record.tool_count,
        "tool_observations": record.tool_observation_count,
        "tools": dict(names),
        "bash_calls": names.get("bash", 0),
        "skills_loaded": [f"{s['name']}" for s in loads if s["path"] == "SKILL.md"],
        "skill_files_read": [f"{s['name']}:{s['path']}" for s in loads if s["path"] != "SKILL.md"],
        "skill_load_order": [f"{s['name']}:{s['path']}" for s in loads],
        "html_report_runs": ran(r"html[-_]report"),
        "echarts_render_runs": ran(r"echarts[-_]render"),
        "wrote_files": wrote,
        "html_in_bash": len([c for c in commands if ".html" in c]),
        "attached": [
            str(c["args"].get("path", "")) for c in calls if c["name"] == "attach_artifact"
        ],
        "web_searches": names.get("search_web", 0),
        "view_calls": names.get("view", 0),
        "notes": record.assistant_notes[:4],
    }


def html_artifacts(record: RunRecord) -> list[dict[str, Any]]:
    out = []
    for item in record.artifacts:
        media = str(item.get("media_type") or "").lower()
        name = str(item.get("filename") or "")
        if item.get("status") == "available" and (
            media in HTML_TYPES or name.lower().endswith((".html", ".htm"))
        ):
            out.append(item)
    return sorted(out, key=lambda i: int(i.get("byte_size") or 0), reverse=True)


_FENCE = re.compile(r"```(?:html|HTML)[^\n]*\n(.*?)```", re.DOTALL)


def html_from_answer(answer: str) -> str | None:
    """The longest fenced ```html block of the answer that is a whole page, for a Run that could not publish one."""
    blocks = [
        b
        for b in _FENCE.findall(answer or "")
        if re.search(r"<!doctype html|<html", b, re.IGNORECASE) and len(b) >= 1500
    ]
    return max(blocks, key=len) if blocks else None


def html_published_check(record: RunRecord) -> dict[str, Any]:
    """`html_published`: the Run published an HTML artifact (attach_artifact of an .html file).

    A page pasted only into the answer text does not count; neither does a Run that failed or published only other files.
    """
    published = html_artifacts(record)
    attaches = [c for c in record.tool_calls if c["name"] == "attach_artifact"]
    attempts = [
        {
            "path": str(c["args"].get("path", "")),
            "failed": bool(c.get("is_error")),
            "result": str(c.get("result_preview") or "")[:160],
        }
        for c in attaches
    ]
    details: dict[str, Any] = {
        "attach_attempts": attempts,
        "artifacts": [
            {
                "filename": a.get("filename"),
                "media_type": a.get("media_type"),
                "status": a.get("status"),
                "bytes": a.get("byte_size"),
            }
            for a in record.artifacts
        ],
    }
    if published:
        names = ", ".join(
            f"{a['filename']} ({int(a.get('byte_size') or 0) // 1024} kB)" for a in published
        )
        return {
            "status": "pass",
            "reason": f"published {names} through attach_artifact",
            "details": details,
        }
    why: list[str] = []
    if record.status != "succeeded":
        why.append(f"the Run {record.status} ({record.stopped_by}) and has no result")
    other = [a for a in record.artifacts if a.get("status") == "available"]
    if other:
        why.append(
            "it published "
            + ", ".join(f"{a.get('filename')} ({a.get('media_type')})" for a in other)
            + ", no HTML"
        )
    elif attempts:
        failed = [a for a in attempts if a["failed"]]
        if failed:
            why.append(
                f"{len(failed)} of {len(attempts)} attach_artifact call(s) failed: {failed[0]['result'][:110]!r}"
            )
        else:
            why.append(f"{len(attempts)} attach_artifact call(s) but no artifact in the result")
    elif record.status == "succeeded":
        why.append("no attach_artifact call")
    outcome = record.artifact_outcome or {}
    issues = outcome.get("issues") or []
    if issues:
        why.append(
            f"artifact issue: {issues[0].get('kind')}: {str(issues[0].get('description'))[:100]}"
        )
    pasted = html_from_answer(record.answer)
    if pasted:
        why.append(
            f"an HTML page of {len(pasted)} characters sits in the answer text only, which does not count"
        )
        details["pasted_html_chars"] = len(pasted)
    return {"status": "fail", "reason": "; ".join(why) or "no HTML artifact", "details": details}


# ---------------------------------------------------------------------------------------------
# One task Run: drive it, save it, judge it
# ---------------------------------------------------------------------------------------------


def save_run(record: RunRecord, run_dir: Path) -> None:
    """Write the Run's record, tool calls, answer and every artifact (the Run is written once and never changed by judging)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    art_dir = run_dir / "artifacts"
    clean = []
    for item in record.artifacts:
        data = item.pop("_bytes", None)
        entry = dict(item)
        if data is not None:
            art_dir.mkdir(exist_ok=True)
            name = Path(str(item.get("filename") or item.get("resource_id"))).name
            path = art_dir / name
            if path.exists():
                path = art_dir / f"{item['resource_id'][:8]}-{name}"
            path.write_bytes(data)
            entry["saved_as"] = str(path.relative_to(run_dir))
        clean.append(entry)
    record.artifacts = clean
    (run_dir / "run.json").write_text(json.dumps(record.to_json(), ensure_ascii=False, indent=1))
    (run_dir / "answer.md").write_text(record.answer or "")
    (run_dir / "tools.json").write_text(
        json.dumps(
            {"summary": tool_summary(record), "calls": record.tool_calls},
            ensure_ascii=False,
            indent=1,
        )
    )


def load_record(run_dir: Path) -> RunRecord:
    data = json.loads((run_dir / "run.json").read_text())
    return RunRecord(**{k: v for k, v in data.items() if k in RunRecord.__dataclass_fields__})


def published_html_paths(record: RunRecord, run_dir: Path) -> list[Path]:
    return [
        run_dir / a["saved_as"]
        for a in html_artifacts(record)
        if a.get("saved_as") and (run_dir / a["saved_as"]).exists()
    ]


def run_checker(
    path: Path, out_dir: Path, viewports: list[int], heights: str, log_path: Path
) -> dict[str, Any]:
    """The checker in a subprocess (own browser, isolated from a crash), result read from checks.json."""
    cmd = [
        sys.executable,
        str(HERE / "evaluate.py"),
        "check",
        str(path),
        "--viewports",
        ",".join(map(str, viewports)),
        "--heights",
        heights,
        "--out",
        str(out_dir),
    ]
    with log_path.open("w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=1800)  # noqa: S603 - this interpreter and this directory's evaluate.py
    checks = out_dir / "checks.json"
    if proc.returncode != 0 or not checks.exists():
        return {"error": f"checker exited {proc.returncode}; see {log_path.name}"}
    return json.loads(checks.read_text())


def verdict(task: dict[str, Any], checks: dict[str, Any]) -> tuple[str, list[str]]:
    failing = [c for c in task["must"] if checks.get(c, {}).get("status") != "pass"]
    return ("PASS" if not failing else "FAIL"), failing


def skills_check(summary: dict[str, Any]) -> dict[str, Any]:
    order = summary["skill_load_order"]
    return {
        "status": "info",
        "reason": ("loaded " + " -> ".join(order)) if order else "no Skill loaded",
        "details": {
            "skills": summary["skills_loaded"],
            "reference_files_read": summary["skill_files_read"],
            "order": order,
        },
    }


def judge_run(
    task: dict[str, Any], run_dir: Path, record: RunRecord, heights: str
) -> dict[str, Any]:
    """Judge one saved Run with the current checker and write its checks.json."""
    summary = tool_summary(record)
    htmls = published_html_paths(record, run_dir)
    pub = html_published_check(record)
    pasted_info: dict[str, Any] | None = None
    if htmls:
        (run_dir / "report.html").write_bytes(htmls[0].read_bytes())
        doc = run_checker(
            run_dir / "report.html", run_dir, task["viewports"], heights, run_dir / "checker.log"
        )
        doc["report"] = True
        doc["report_file"] = htmls[0].name
        for extra in htmls[1:4]:
            sub = run_dir / "other" / extra.stem
            run_checker(extra, sub, task["viewports"], "800", run_dir / f"checker-{extra.stem}.log")
    else:
        doc = {"report": False, "checks": {}}
        for stale in ("report.html", "contact.png"):
            (run_dir / stale).unlink(missing_ok=True)
        shutil.rmtree(run_dir / "shots", ignore_errors=True)
        pasted = html_from_answer(record.answer) if record.status == "succeeded" else None
        if pasted:
            (run_dir / "artifacts").mkdir(exist_ok=True)
            path = run_dir / "artifacts" / "from-answer.html"
            path.write_text(pasted, encoding="utf-8")
            info = run_checker(
                path, run_dir / "pasted", task["viewports"], "800", run_dir / "checker-pasted.log"
            )
            pasted_info = {
                "chars": len(pasted),
                "checks": {
                    k: {"status": v["status"], "reason": v["reason"]}
                    for k, v in (info.get("checks") or {}).items()
                },
                "note": "information only: a page pasted into the answer is not a published report",
            }
    doc.setdefault("checks", {})
    if "error" in doc:
        doc["checks"] = {}
    doc["checks"]["html_published"] = pub
    doc["checks"]["skills_loaded"] = skills_check(summary)
    doc["checker_version"] = doc.get("checker_version", CHECKER_VERSION)
    v, failing = verdict(task, doc["checks"])
    doc["task"] = {
        "id": task["id"],
        "kind": task["kind"],
        "must": task["must"],
        "verdict": v,
        "failing": failing,
    }
    doc["pasted_html"] = pasted_info
    (run_dir / "checks.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    return doc


def needs_judging(run_dir: Path, recheck: bool) -> bool:
    if recheck:
        return True
    path = run_dir / "checks.json"
    if not path.exists():
        return True
    try:
        doc = json.loads(path.read_text())
    except ValueError:
        return True
    return (
        doc.get("checker_version") != CHECKER_VERSION
        or "html_published" not in (doc.get("checks") or {})
        or "error" in doc
    )


def run_one_task(
    args: tuple[dict[str, Any], int, Path, argparse.Namespace, int, int, Any],
) -> dict[str, Any]:
    task, n, out_dir, opts, index, total, log = args
    run_dir = out_dir / "tasks" / f"{task['id']}-{n}"
    tag = f"[{index:>2}/{total}] {task['id']}-{n}"
    if (run_dir / "run.json").exists():
        record = load_record(run_dir)
        action = (
            "judged again"
            if needs_judging(run_dir, getattr(opts, "recheck", False))
            else "up to date"
        )
    else:
        api = Api()
        try:
            record = follow_run(
                api,
                task["q"],
                early_stop=None,
                timeout_s=opts.timeout,
                fetch_artifacts=True,
                label=f"{task['id']}-{n}",
            )
        finally:
            api.close()
        save_run(record, run_dir)
        action = "run"
        log(
            f"{tag} {record.status} ({record.stopped_by}) {record.elapsed_s:.0f}s tools={record.tool_count} "
            f"html={[Path(a['saved_as']).name for a in html_artifacts(record) if a.get('saved_as')]}"
            + (f" ERROR={record.error_message}" if record.status == "failed" else "")
        )
    if action != "up to date":
        doc = judge_run(task, run_dir, record, opts.heights)
        if action != "run":
            log(f"{tag} {action}: {doc['task']['verdict']}")
    return {"task": task["id"], "n": n, "action": action}


# ---------------------------------------------------------------------------------------------
# Adoption: judge another label's saved Runs again instead of running them again
# ---------------------------------------------------------------------------------------------


def adopt_runs(
    out_dir: Path, specs: list[str], task_ids: set[str], log: Any
) -> dict[str, list[str]]:
    """Copy the saved Runs named by `LABEL:task,task` into this label (never overwriting a Run already here)."""
    adopted: dict[str, list[str]] = {}
    for spec in specs:
        label, _, names = spec.partition(":")
        wanted = [t.strip() for t in names.split(",") if t.strip()]
        if not label or not wanted:
            sys.exit(f"evaluate.py: --adopt wants LABEL:task[,task], not '{spec}'")
        unknown = [t for t in wanted if t not in task_ids]
        if unknown:
            sys.exit(f"evaluate.py: --adopt names unknown task ids {unknown}")
        src_root = RESULTS / label / "tasks"
        if not src_root.is_dir():
            sys.exit(f"evaluate.py: results/{label} has no saved task Runs to adopt")
        for task_id in wanted:
            dirs = sorted(d for d in src_root.glob(f"{task_id}-*") if (d / "run.json").exists())
            if not dirs:
                sys.exit(f"evaluate.py: results/{label} has no saved Run of {task_id}")
            for src in dirs:
                dst = out_dir / "tasks" / src.name
                if dst.exists():
                    continue
                dst.mkdir(parents=True)
                record = load_record(src)
                for name in ("run.json", "tools.json", "answer.md"):
                    if (src / name).exists():
                        shutil.copy2(src / name, dst / name)
                for art in (
                    record.artifacts
                ):  # only what the Run published; not the files an earlier judging derived
                    saved = art.get("saved_as")
                    if saved and (src / saved).exists():
                        (dst / saved).parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src / saved, dst / saved)
                old = {}
                if (src / "checks.json").exists():
                    try:
                        old = {
                            "checker_version": json.loads((src / "checks.json").read_text()).get(
                                "checker_version", "1"
                            )
                        }
                    except ValueError:
                        pass
                (dst / "adopted.json").write_text(
                    json.dumps(
                        {
                            "from_label": label,
                            "from_dir": str(src),
                            "adopted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                            "judged_before_by": old.get("checker_version", "1"),
                            "judged_now_by": CHECKER_VERSION,
                        },
                        indent=1,
                    )
                )
                adopted.setdefault(label, []).append(src.name)
    for label, names in adopted.items():
        log(
            f"[evaluate] adopted {len(names)} saved Run(s) from '{label}': {', '.join(names)}; they are judged again, not run again"
        )
    return adopted


# ---------------------------------------------------------------------------------------------
# Reading a label back
# ---------------------------------------------------------------------------------------------


def load_run_result(run_dir: Path, task: dict[str, Any]) -> dict[str, Any] | None:
    if not (run_dir / "run.json").exists():
        return None
    record = load_record(run_dir)
    doc: dict[str, Any] = {}
    if (run_dir / "checks.json").exists():
        try:
            doc = json.loads((run_dir / "checks.json").read_text())
        except ValueError:
            doc = {}
    checks = doc.get("checks") or {}
    summary = tool_summary(record)
    adopted = (
        json.loads((run_dir / "adopted.json").read_text())
        if (run_dir / "adopted.json").exists()
        else None
    )
    n = int(run_dir.name.rsplit("-", 1)[1])
    if checks:
        v, failing = verdict(task, checks)
    else:
        v, failing = "NOT JUDGED", []
    return {
        "task": task["id"],
        "n": n,
        "run_dir": str(run_dir),
        "run_id": record.run_id,
        "status": record.status,
        "stopped_by": record.stopped_by,
        "elapsed_s": record.elapsed_s,
        "error": record.error_message,
        "usage": {
            k: record.usage.get(k) for k in ("input_tokens", "output_tokens", "total_tokens")
        },
        "summary": summary,
        "checks": checks,
        "verdict": v,
        "failing": failing,
        "report": bool(doc.get("report")),
        "report_error": doc.get("error"),
        "pasted_html": doc.get("pasted_html"),
        "answer_head": (record.answer or "")[:240].replace("\n", " "),
        "run_errors": doc.get("run_errors") or [],
        "probe_errors": doc.get("probe_errors") or [],
        "checker_version": doc.get("checker_version"),
        "adopted": adopted,
    }


def collect_results(out_dir: Path, tasks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for task in tasks:
        for run_dir in sorted((out_dir / "tasks").glob(f"{task['id']}-*")):
            res = load_run_result(run_dir, task)
            if res:
                results[f"{task['id']}-{res['n']}"] = res
    return results


# ---------------------------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------------------------


def _cell(task: dict[str, Any], name: str, res: dict[str, Any]) -> str:
    c = res["checks"].get(name)
    if c is None or (name not in task["must"] and name not in HYGIENE and name != "html_published"):
        return "-"  # a task-specific check on a task that does not ask for it, or not evaluated (no published report)
    mark = MARK[c["status"]] if c["status"] in MARK else str(c["status"])
    if name not in task["must"]:
        mark = mark.lower()
    if name == "no_boilerplate" and c["status"] == "fail":
        mark = f"FAIL ({len(distinct_hits(c.get('details', {}).get('hits', [])))})"
    return f"**{mark}**" if name in task["must"] else mark


def render_tasks_md(
    label: str,
    meta: dict[str, Any],
    tasks: list[dict[str, Any]],
    results: dict[str, dict[str, Any]],
) -> str:
    runs = [
        (t, results[k])
        for t in tasks
        for k in sorted(
            (k for k in results if results[k]["task"] == t["id"]), key=lambda k: results[k]["n"]
        )
    ]
    lines = [f"# Tasks: {label}", ""]
    per_task = Counter(t["id"] for t, _ in runs)
    adopted = sorted({r["adopted"]["from_label"] for _, r in runs if r.get("adopted")})
    lines.append(
        f"Answering model: `{meta.get('answering_model')}` (preset `{meta.get('query_model_preset')}`) | catalog: "
        f"{', '.join(meta.get('catalog', []))} | samples/task: {', '.join(f'{k} {v}' for k, v in per_task.items())} | "
        f"toolkit: {meta.get('toolkit_dir') or 'image default'} | checker v{CHECKER_VERSION}"
    )
    if adopted:
        lines.append("")
        lines.append(
            "Runs marked `(adopted)` were not run again: they are the saved Runs of "
            + ", ".join(f"`{a}`" for a in adopted)
            + ", judged here by the current checker."
        )
    lines.append("")
    lines.append(
        "`pass`/`FAIL`/`n/a` per deterministic check. **Bold** cells are that task's `must` list (they decide the verdict); plain lower-case "
        "cells are hygiene checks every report is held to; `-` is a check the task does not ask for, or one that could not be evaluated because "
        "the Run published no HTML. `FAIL (n)` under no_boilerplate is the number of high-confidence hits."
    )
    lines.append("")
    header = ["task-n", "run", "verdict"] + ALL_CHECKS
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    failures: list[str] = []
    for task, res in runs:
        v = res["verdict"]
        if res["report_error"]:
            v = "CHECK ERROR"
        run = f"{res['status']} {res['elapsed_s']:.0f}s" + (
            f" ({res['stopped_by']})" if res["stopped_by"] not in ("finished",) else ""
        )
        if res.get("adopted"):
            run += " (adopted)"
        if res["run_errors"]:
            run += " [checker incomplete: " + "; ".join(e["stage"] for e in res["run_errors"]) + "]"
        cells = [_cell(task, name, res) for name in ALL_CHECKS]
        lines.append(f"| {task['id']}-{res['n']} | {run} | {v} | " + " | ".join(cells) + " |")
        key = f"{task['id']}-{res['n']}"
        for name in res["failing"]:
            c = res["checks"].get(name)
            if name != "html_published" and not res["report"]:
                continue  # judged on nothing: the html_published line says why
            failures.append(
                f"- **{key} / {name}** ({c['status'] if c else 'not evaluated'}): {c['reason'] if c else 'not evaluated'}"
            )
        if res["report_error"]:
            failures.append(f"- **{key}**: the checker failed: {res['report_error']}")
        if res["status"] != "succeeded":
            failures.append(
                f"- **{key}**: Run {res['status']} ({res['stopped_by']}): {res.get('error')}"
            )
    lines.append("")
    # how many reports pass each check
    counts: dict[str, list[int]] = {}
    reports = hit_total = reports_with_hits = 0
    for task, res in runs:
        for name in ALL_CHECKS:
            c = res["checks"].get(name)
            if c is None or _cell(task, name, res) == "-":
                continue  # not evaluated, or a check this task does not ask for
            tally = counts.setdefault(name, [0, 0, 0])
            tally[{"pass": 0, "fail": 1}.get(c["status"], 2)] += 1
        if res["report"]:
            reports += 1
            hits = len(
                distinct_hits(
                    (res["checks"].get("no_boilerplate") or {}).get("details", {}).get("hits", [])
                )
            )
            hit_total += hits
            reports_with_hits += 1 if hits else 0
    lines.append(f"## Summary over {len(runs)} Run(s), {reports} with a published report")
    lines.append("")
    lines.append(
        "Counted per check over the Runs whose task asks for it (its `must`) or where it is a hygiene check, and that were evaluated."
    )
    lines.append("")
    lines.append("| check | pass | fail | n/a |")
    lines.append("|---|---|---|---|")
    for name in ALL_CHECKS:
        if name in counts:
            p_, f_, n_ = counts[name]
            lines.append(f"| {name} | {p_} | {f_} | {n_} |")
    lines.append("")
    lines.append(
        f"**Boilerplate: {reports_with_hits} of {reports} published report(s) carry at least one high-confidence hit ({hit_total} hits in total).**"
    )
    lines.append("")
    lines.append("## Why the must-checks fail")
    lines.append("")
    lines.extend(failures or ["(none)"])
    lines.append("")
    pasted = [(t, r) for t, r in runs if r.get("pasted_html")]
    if pasted:
        lines.append(
            "Pasted, not published (information only; these pages do not count as reports): "
            + "; ".join(
                f"{t['id']}-{r['n']} ({r['pasted_html']['chars']} characters in the answer)"
                for t, r in pasted
            )
            + "."
        )
        lines.append("")
    lines.append("## What the model did")
    lines.append("")
    lines.append(
        "| task-n | tool calls (bash) | skills loaded | skill reference files read | html-report | echarts-render | wrote files | attached | web searches | tokens in/out | wall |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for task, res in runs:
        s = res["summary"]
        u = res.get("usage") or {}
        lines.append(
            f"| {task['id']}-{res['n']} | {s['tool_count']} ({s['bash_calls']}) | {', '.join(s['skills_loaded']) or '-'} | "
            f"{', '.join(s['skill_files_read']) or '-'} | {len(s['html_report_runs']) or '-'} | {len(s['echarts_render_runs']) or '-'} | "
            f"{', '.join(Path(w['path']).name for w in s['wrote_files'])[:60] or '-'} | {', '.join(Path(a).name for a in s['attached'])[:50] or '-'} | "
            f"{s['web_searches'] or '-'} | {u.get('input_tokens')}/{u.get('output_tokens')} | {res['elapsed_s']:.0f}s |"
        )
    lines.append("")
    hits_lines = []
    for task, res in runs:
        nb = res["checks"].get("no_boilerplate") or {}
        for hit in distinct_hits((nb.get("details") or {}).get("hits", [])):
            hits_lines.append(
                f"- {task['id']}-{res['n']}: '{hit['phrase']}' [{hit['category']}] in `{hit['element']}`: ...{hit['context']}..."
            )
    lines.append("## Boilerplate hits")
    lines.append("")
    lines.extend(hits_lines or ["(none)"])
    lines.append("")
    return "\n".join(lines)


def tasks_json(label: str, results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The compact machine-readable form (compare reads it)."""
    runs = {}
    for key, res in results.items():
        runs[key] = {
            "task": res["task"],
            "n": res["n"],
            "status": res["status"],
            "stopped_by": res["stopped_by"],
            "elapsed_s": res["elapsed_s"],
            "usage": res["usage"],
            "verdict": res["verdict"],
            "failing": res["failing"],
            "report": res["report"],
            "skills_loaded": res["summary"]["skills_loaded"],
            "skill_files_read": res["summary"]["skill_files_read"],
            "tool_count": res["summary"]["tool_count"],
            "html_report_runs": len(res["summary"]["html_report_runs"]),
            "echarts_render_runs": len(res["summary"]["echarts_render_runs"]),
            "attached": res["summary"]["attached"],
            "adopted_from": (res.get("adopted") or {}).get("from_label"),
            "checker_version": res.get("checker_version"),
            "checks": {
                name: {
                    "status": c["status"],
                    "reason": c["reason"],
                    **({"heuristic": True} if c.get("heuristic") else {}),
                }
                for name, c in res["checks"].items()
            },
        }
    return {"label": label, "checker_version": CHECKER_VERSION, "runs": runs}


def write_outputs(
    out: Path, label: str, meta: dict[str, Any], tasks: list[dict[str, Any]]
) -> tuple[str, dict[str, dict[str, Any]]]:
    results = collect_results(out, tasks)
    md = render_tasks_md(label, meta, tasks, results)
    (out / "tasks.md").write_text(md)
    (out / "tasks.json").write_text(
        json.dumps(tasks_json(label, results), ensure_ascii=False, indent=1)
    )
    return md, results


# ---------------------------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------------------------


def run_tasks(
    opts: argparse.Namespace, log: Any = lambda s: print(s, flush=True), echo_report: bool = True
) -> dict[str, Any]:
    spec = common.load_cases(opts.cases_file)
    all_tasks = spec["tasks"]
    ids = {t["id"] for t in all_tasks}
    selected = all_tasks
    if opts.tasks:
        wanted = [t.strip() for t in opts.tasks.split(",") if t.strip()]
        unknown = [t for t in wanted if t not in ids]
        if unknown:
            sys.exit(f"evaluate.py: unknown task ids {unknown}")
        selected = [t for t in all_tasks if t["id"] in wanted]
    label = opts.label or time.strftime("tasks-%Y%m%d-%H%M%S")
    recheck, rerender = getattr(opts, "recheck", False), getattr(opts, "rerender", False)
    resume = opts.resume or recheck or rerender
    out = common.prepare_label_dir(label, resume, "tasks")
    (out / "tasks").mkdir(exist_ok=True)
    adopted = adopt_runs(out, opts.adopt, ids, log) if opts.adopt else {}
    adopted_ids = {name.rsplit("-", 1)[0] for names in adopted.values() for name in names}
    meta_path = out / "tasks-meta.json"
    by_id = {t["id"]: t for t in all_tasks}

    def saved_runs() -> list[tuple[dict[str, Any], int]]:
        found = []
        for d in sorted((out / "tasks").iterdir()):
            task_id, _, n = d.name.rpartition("-")
            if (d / "run.json").exists() and task_id in by_id and n.isdigit():
                found.append((by_id[task_id], int(n)))
        return found

    # What to do: rerender builds the report only; recheck judges every saved Run; otherwise the selected tasks are run (a Run already
    # saved is kept) and every other saved Run (adopted ones) is judged if its judging is out of date.
    if rerender:
        order: list[tuple[dict[str, Any], int]] = []
    elif recheck:
        order = saved_runs()
    else:
        fresh = [t for t in selected if t["id"] not in adopted_ids]
        order = [(t, n) for n in range(1, opts.samples + 1) for t in fresh]
        wanted_keys = {(t["id"], n) for t, n in order}
        order += [(t, n) for t, n in saved_runs() if (t["id"], n) not in wanted_keys]
    total = len(order)
    jobs = [
        (f"{t['id']}-{n}", (t, n, out, opts, i + 1, total, log)) for i, (t, n) in enumerate(order)
    ]
    to_run = [
        (t, n) for t, n in order if not (out / "tasks" / f"{t['id']}-{n}" / "run.json").exists()
    ]

    if to_run:
        with common.runs_lock(f"tasks {label}"):
            meta = common.stack_meta(Api())
            meta_path.write_text(
                json.dumps(
                    {
                        "label": label,
                        "adopted": adopted,
                        **{k: v for k, v in meta.items() if k != "catalog_detail"},
                    },
                    ensure_ascii=False,
                    indent=1,
                )
            )
            log(
                f"[evaluate] tasks '{label}': {len(to_run)} Run(s) to run, {len(jobs) - len(to_run)} saved to judge; "
                f"model={meta['answering_model']} catalog={meta['catalog']}"
            )
            common.run_in_pool(jobs, run_one_task, opts.concurrency)
    else:
        if meta_path.exists():
            meta = json.loads(
                meta_path.read_text()
            )  # the stack that produced the Runs, not whatever is up now
        else:  # nothing to run: the stack that produced the adopted Runs is the source label's
            src = next(iter(adopted), None)
            src_meta = RESULTS / src / "tasks-meta.json" if src else None
            meta = json.loads(src_meta.read_text()) if src_meta and src_meta.exists() else {}
            meta_path.write_text(
                json.dumps(
                    {"label": label, "adopted": adopted, **meta}, ensure_ascii=False, indent=1
                )
            )
        if jobs:
            log(
                f"[evaluate] tasks '{label}': {len(jobs)} saved Run(s), judged with checker v{CHECKER_VERSION} where out of date"
            )
            common.run_in_pool(jobs, run_one_task, opts.concurrency)
    md, results = write_outputs(out, label, meta, all_tasks)
    if echo_report:
        print("\n" + md)
    if driver.THROTTLE.events:
        log(f"[evaluate] {len(driver.THROTTLE.events)} back-off event(s) on provider limits")
    return {"label": label, "results": results, "meta": meta}


# ---------------------------------------------------------------------------------------------
# The standalone checker
# ---------------------------------------------------------------------------------------------


def checks_table(doc: dict[str, Any], must: list[str] | None = None) -> str:
    lines = []
    for name, c in doc["checks"].items():
        star = "*" if must and name in must else " "
        lines.append(
            f"{star}{name:<30} {MARK.get(c['status'], c['status']):<5} {c['reason'][:300]}"
        )
    return "\n".join(lines)


def run_check(opts: argparse.Namespace) -> int:
    import checker

    path: Path = opts.path
    if not path.exists():
        sys.exit(f"evaluate.py: {path} not found")
    viewports = tuple(int(v) for v in opts.viewports.split(",") if v.strip())
    heights = tuple(int(h) for h in opts.heights.split(",") if h.strip())
    must = None
    query = None
    if opts.kind:
        spec = common.load_cases(opts.cases_file)
        match = next(
            (t for t in spec["tasks"] if t["id"] == opts.kind or t["kind"] == opts.kind), None
        )
        if match is None:
            sys.exit(f"evaluate.py: no task with id or kind '{opts.kind}'")
        must, query = match["must"], match["q"]
    out = opts.out or (RESULTS / "check" / path.stem)
    out.mkdir(parents=True, exist_ok=True)
    print(
        f"[check] {path.name} ({path.stat().st_size} bytes) viewports={list(viewports)} heights={list(heights)} -> {out}",
        flush=True,
    )
    doc = checker.check_file(
        path,
        out,
        viewports=viewports,
        heights=heights,
        query=query,
        log=lambda s: print(s, flush=True),
    )
    (out / "checks.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    table = checks_table(doc, must)
    if doc.get("run_errors"):
        table = (
            f"WARNING: {len(doc['run_errors'])} stage(s) of the checker failed, so the checks below are INCOMPLETE: "
            + "; ".join(f"{e['stage']}: {e['error']}" for e in doc["run_errors"][:3])
            + "\n"
            + table
        )
    if doc.get("probe_errors"):
        table = (
            f"WARNING: {len(doc['probe_errors'])} in-page probe(s) threw, so some measurements below are missing: "
            + "; ".join(f"{e['probe']}: {e['error']}" for e in doc["probe_errors"][:3])
            + "\n"
            + table
        )
    (out / "checks.txt").write_text(table + "\n")
    print(table)
    if must:
        failing = [
            c
            for c in must
            if c != "html_published" and doc["checks"].get(c, {}).get("status") != "pass"
        ]
        note = (
            " (html_published is a property of the Run: not judged here)"
            if "html_published" in must
            else ""
        )
        print(
            f"\nmust ({opts.kind}): {'PASS' if not failing else 'FAIL: ' + ', '.join(failing)}{note}"
        )
    print(f"\nshots: {out / 'shots'}  contact sheet: {out / 'contact.png'}  ({doc['elapsed_s']}s)")
    return 0
