"""results/<label>/SUMMARY.md: everything about one label in one file (routing, tasks, boilerplate hits, who loaded what)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import common
import routing
import tasks
from checker import CHECKER_VERSION


def demote(markdown: str, by: int = 1) -> str:
    """Push every heading of an embedded report down `by` levels."""
    return re.sub(
        r"^(#{1,5}) ", lambda m: "#" * (len(m.group(1)) + by) + " ", markdown, flags=re.MULTILINE
    )


def loads_by_skill(doc: dict[str, Any]) -> list[str]:
    """For every skill: the cases whose samples loaded it, as k/n of the scored samples."""
    per: dict[str, dict[str, list[int]]] = {}
    for case in doc["cases"]:
        scored = [s for s in case["samples"] if routing.scorable(s)]
        for skill in sorted({name for s in scored for name in s["loaded"]}):
            k = sum(1 for s in scored if skill in s["loaded"])
            per.setdefault(skill, {})[case["id"]] = [k, len(scored)]
    lines = []
    for skill, cases in sorted(per.items()):
        lines.append(
            f"- `{skill}`: " + ", ".join(f"{cid} {k}/{n}" for cid, (k, n) in cases.items())
        )
    absent = sorted(
        {a for case in doc["cases"] for s in case["samples"] for a in s.get("attempted_absent", [])}
    )
    if absent:
        lines.append(
            "- attempted to load skills the catalog does not have: "
            + ", ".join(f"`{a}`" for a in absent)
        )
    return lines or ["- no skill was loaded"]


def header(label: str, routing_doc: dict[str, Any] | None, tasks_meta: dict[str, Any]) -> list[str]:
    meta = (routing_doc or {}).get("meta") or tasks_meta or {}
    rows = [
        (
            "answering model",
            f"`{meta.get('answering_model')}` (preset `{meta.get('query_model_preset')}`)",
        ),
        ("catalog", ", ".join(f"`{s}`" for s in meta.get("catalog", [])) or "?"),
        (
            "skills directory",
            f"`{meta.get('skills_dir')}` (manifest `{meta.get('skills_manifest')}`)",
        ),
        (
            "toolkit",
            f"`{meta.get('toolkit_dir') or 'the image default'}`"
            + (f" (manifest `{meta.get('toolkit_manifest')}`)" if meta.get("toolkit_dir") else ""),
        ),
        ("image", f"`{meta.get('image_id')}`"),
        ("checker", f"v{CHECKER_VERSION}"),
        ("started", str(meta.get("started_at"))),
    ]
    if routing_doc:
        rows.append(
            (
                "routing samples",
                f"{routing_doc['params']['samples']} per case; cases-file fingerprint `{routing_doc['params'].get('cases_file_fingerprint')}`",
            )
        )
    adopted = (tasks_meta or {}).get("adopted") or {}
    if adopted:
        rows.append(
            (
                "adopted task Runs",
                "; ".join(f"{', '.join(names)} from `{src}`" for src, names in adopted.items())
                + " (saved Runs judged again by the current checker, not run again)",
            )
        )
    return ["| | |", "|---|---|", *[f"| {k} | {v} |" for k, v in rows]]


def build_summary(label: str, cases_file: Path) -> str:
    out = common.RESULTS / label
    if not out.is_dir():
        raise SystemExit(f"evaluate.py: no results for label '{label}'")
    spec = common.load_cases(cases_file)
    routing_doc = (
        json.loads((out / "routing.json").read_text()) if (out / "routing.json").exists() else None
    )
    tasks_meta = (
        json.loads((out / "tasks-meta.json").read_text())
        if (out / "tasks-meta.json").exists()
        else {}
    )
    lines = [f"# SUMMARY: {label}", ""]
    lines.extend(header(label, routing_doc, tasks_meta))
    lines.append("")
    if routing_doc:
        lines.append("## Routing")
        lines.append("")
        lines.append(demote(routing.render_routing_md(routing_doc).split("\n", 2)[2], 1))
        lines.append("")
        lines.append("### Who loaded what (routing: cases whose samples loaded each skill, k/n)")
        lines.append("")
        lines.extend(loads_by_skill(routing_doc))
        lines.append("")
    else:
        lines.extend(["## Routing", "", "(no routing results in this label)", ""])
    results = tasks.collect_results(out, spec["tasks"])
    if results:
        meta = tasks_meta or (routing_doc or {}).get("meta") or {}
        md = tasks.render_tasks_md(label, meta, spec["tasks"], results)
        lines.append("## Tasks")
        lines.append("")
        lines.append(demote(md.split("\n", 2)[2], 1))
        lines.append("")
        lines.append("### Who loaded what (tasks: skills, reference files, toolkit commands)")
        lines.append("")
        lines.append(
            "| task-n | skills loaded (in order) | reference files read | html-report runs | echarts-render runs | tool calls |"
        )
        lines.append("|---|---|---|---|---|---|")
        for key in sorted(results, key=lambda k: (k.rsplit("-", 1)[0], results[k]["n"])):
            r = results[key]
            s = r["summary"]
            order = ", ".join(x.replace(":SKILL.md", "") for x in s["skill_load_order"]) or "none"
            lines.append(
                f"| {key} | {order} | {', '.join(s['skill_files_read']) or '-'} | {len(s['html_report_runs']) or '-'} | "
                f"{len(s['echarts_render_runs']) or '-'} | {s['tool_count']} |"
            )
        lines.append("")
    else:
        lines.extend(["## Tasks", "", "(no task results in this label)", ""])
    return "\n".join(lines)


def write_summary(label: str, cases_file: Path) -> Path:
    path = common.RESULTS / label / "SUMMARY.md"
    path.write_text(build_summary(label, cases_file))
    return path
