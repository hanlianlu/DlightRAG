"""`evaluate.py compare A B`: two labels side by side, the effect of a Skill change at a glance.

Three blocks: routing metrics per skill (recall, precision, forbidden-load rate), the routing cases whose verdict moved, and
every task check (pass-rate over that task's Runs, A -> B) with who loaded what. `▲` marks a change for the better, `▼` for the
worse (a lower forbidden-load rate is the better one), `·` no change. Plain aligned text: it reads in a terminal and in a chat.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import common
import routing as routing_mod
import tasks as tasks_mod

UP, DOWN, SAME = "▲", "▼", "·"
GREEN, RED, RESET = "\033[32m", "\033[31m", "\033[0m"


# ---------------------------------------------------------------------------------------------
# Loading a label (older harness versions wrote older shapes: they are read, not rejected)
# ---------------------------------------------------------------------------------------------


def load_label(label: str) -> dict[str, Any]:
    out = common.RESULTS / label
    if not out.is_dir():
        raise SystemExit(
            f"evaluate.py: no results for label '{label}' (results/{label} does not exist)"
        )
    data: dict[str, Any] = {
        "label": label,
        "dir": out,
        "routing": None,
        "tasks": None,
        "tasks_meta": {},
    }
    if (out / "routing.json").exists():
        data["routing"] = json.loads((out / "routing.json").read_text())
    if (out / "tasks.json").exists():
        raw = json.loads((out / "tasks.json").read_text())
        data["tasks"] = (
            raw if "runs" in raw else {"label": label, "checker_version": "1", "runs": raw}
        )
    if (out / "tasks-meta.json").exists():
        data["tasks_meta"] = json.loads((out / "tasks-meta.json").read_text())
    return data


def normalize_run(run: dict[str, Any]) -> dict[str, Any]:
    """Older labels carry fewer fields (no verdict, the tool facts under `summary`); fill what compare reads."""
    run = dict(run)
    summary = run.get("summary") or {}
    run["checks"] = run.get("checks") or {}
    run.setdefault("verdict", "?")
    run.setdefault("tool_count", summary.get("tool_count", 0))
    run.setdefault("skills_loaded", summary.get("skills_loaded", []))
    run.setdefault("skill_files_read", summary.get("skill_files_read", []))
    run.setdefault("html_report_runs", len(summary.get("html_report_runs", [])))
    run.setdefault("elapsed_s", 0)
    return run


def meta_of(data: dict[str, Any]) -> dict[str, Any]:
    return (data["routing"] or {}).get("meta") or data["tasks_meta"] or {}


# ---------------------------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------------------------


def frac(k: int, n: int) -> float | None:
    return k / n if n else None


def mark(a: float | None, b: float | None, higher_is_better: bool = True, eps: float = 1e-9) -> str:
    if a is None or b is None or abs(a - b) <= eps:
        return SAME
    better = (b > a) == higher_is_better
    return UP if better else DOWN


def colorize(text: str, on: bool) -> str:
    if not on:
        return text
    return text.replace(UP, f"{GREEN}{UP}{RESET}").replace(DOWN, f"{RED}{DOWN}{RESET}")


def table(rows: list[list[str]], aligns: str | None = None) -> list[str]:
    """Aligned plain-text columns; the first row is the header."""
    widths = [max(len(r[i]) for r in rows if i < len(r)) for i in range(max(len(r) for r in rows))]
    out = []
    for n, row in enumerate(rows):
        out.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip())
        if n == 0:
            out.append("  ".join("-" * widths[i] for i in range(len(widths))))
    return out


def pct_cell(m: dict[str, Any] | None) -> str:
    if not m or not m.get("n"):
        return "-"
    return f"{100 * m['k'] / m['n']:.0f}% ({m['k']}/{m['n']})"


def delta_cell(
    a: dict[str, Any] | None, b: dict[str, Any] | None, higher_is_better: bool = True
) -> str:
    ra = frac(a["k"], a["n"]) if a else None
    rb = frac(b["k"], b["n"]) if b else None
    if ra is None or rb is None:
        return (
            "new"
            if ra is None and rb is not None
            else ("gone" if rb is None and ra is not None else "")
        )
    symbol = mark(ra, rb, higher_is_better)
    diff = round((rb - ra) * 100)
    return f"{symbol} {diff:+d}pp" if symbol != SAME else SAME


# ---------------------------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------------------------

METRICS = (
    ("recall", "recall", True),
    ("precision", "precision", True),
    ("forbidden_load", "forbidden-load", False),
)


def overall_metric(metrics: dict[str, Any], key: str) -> dict[str, Any] | None:
    overall = metrics["overall"]
    if key == "forbidden_load":
        return overall.get("forbidden_load") or overall.get("forbidden_load_samples")
    return overall.get(key)


def restricted_metrics(doc: dict[str, Any], ids: set[str]) -> dict[str, Any] | None:
    """The metrics of a routing document recomputed over only the cases in `ids`; None when the document predates the settings it needs."""
    cases = [c for c in doc["cases"] if c["id"] in ids]
    if not cases or not all((c.get("settings") or {}).get("required") is not None for c in cases):
        return None
    return routing_mod.routing_metrics(
        cases,
        {c["id"]: c["samples"] for c in cases},
        {c["id"]: c["settings"] for c in cases},
        set(doc["meta"]["catalog"]),
        list(doc["metrics"]["per_skill"]),
    )


def case_definition(case: dict[str, Any]) -> tuple[Any, ...] | None:
    """What makes two samples of a case comparable: the message, the expectations and the budget (not the catalog-derived fields)."""
    st = case.get("settings") or {}
    if st.get("required") is None:
        return None  # an older label: its cases carry no settings
    return (
        case.get("q"),
        tuple(st.get("load", [])),
        tuple(st.get("allow", [])),
        tuple(st.get("forbid", [])),
        st.get("max_tool_calls"),
        st.get("max_seconds"),
        st.get("history_turns"),
    )


def routing_view(
    a: dict[str, Any], b: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], list[str], list[str]]:
    """What to compare: both labels' metrics over the cases both ran AND defined the same way (recomputed when that is not every case)."""
    da, db = a["routing"], b["routing"]
    ca, cb = {c["id"]: c for c in da["cases"]}, {c["id"]: c for c in db["cases"]}
    both = [i for i in ca if i in cb]
    edited = [
        i
        for i in both
        if case_definition(ca[i]) is not None
        and case_definition(cb[i]) is not None
        and case_definition(ca[i]) != case_definition(cb[i])
    ]
    shared = [i for i in both if i not in edited]
    if set(ca) == set(cb) and not edited:
        return da["metrics"], db["metrics"], [], shared
    ma, mb = restricted_metrics(da, set(shared)), restricted_metrics(db, set(shared))
    if ma is None or mb is None:
        return (
            da["metrics"],
            db["metrics"],
            [
                f"A ran {len(ca)} case(s) and B {len(cb)}, and the metrics cannot be restricted to the "
                f"{len(shared)} they share (an older label): the rows mix different cases"
            ],
            shared,
        )
    notes = []
    if set(ca) != set(cb):
        notes.append(
            f"A ran {len(ca)} case(s) and B {len(cb)}: compared on the {len(shared) + len(edited)} they share"
        )
    if edited:
        notes.append(
            f"left out because the case itself was edited between A and B (message, expectations, budget or history): {', '.join(edited)}"
        )
    notes.append(
        f"both sides' metrics are recomputed over the {len(shared)} compared case(s): {', '.join(shared)}"
    )
    return ma, mb, notes, shared


def routing_block(
    a: dict[str, Any],
    b: dict[str, Any],
    view: tuple[dict[str, Any], dict[str, Any], list[str], list[str]],
) -> list[str]:
    da, db = a["routing"], b["routing"]
    if not da or not db:
        missing = [d["label"] for d in (a, b) if not d["routing"]]
        return ["ROUTING", f"  not compared: no routing results in {', '.join(missing)}", ""]
    ma_all, mb_all, view_notes, shared = view
    lines = [
        "ROUTING  (sample level; precision counts `allow`ed loads in neither TP nor FP; see routing.md for the formulas)"
    ]
    skills = list(dict.fromkeys([*ma_all["per_skill"], *mb_all["per_skill"]]))
    rows = [["skill"] + [h for _, lbl, _ in METRICS for h in (f"{lbl} A", f"{lbl} B", "")]]
    for skill in skills:
        ma, mb = ma_all["per_skill"].get(skill), mb_all["per_skill"].get(skill)
        in_a, in_b = (ma or {}).get("in_catalog", True), (mb or {}).get("in_catalog", True)
        row = [
            skill
            + (
                ""
                if in_a and in_b
                else " (absent)"
                if not in_a and not in_b
                else " (absent in A)"
                if not in_a
                else " (absent in B)"
            )
        ]
        for key, _, hib in METRICS:
            xa, xb = (ma or {}).get(key), (mb or {}).get(key)
            row += [pct_cell(xa), pct_cell(xb), delta_cell(xa, xb, hib)]
        rows.append(row)
    row = ["OVERALL"]
    for key, _, hib in METRICS:
        xa, xb = overall_metric(ma_all, key), overall_metric(mb_all, key)
        row += [pct_cell(xa), pct_cell(xb), delta_cell(xa, xb, hib)]
    rows.append(row)
    lines += ["  " + line for line in table(rows)]
    notes = list(view_notes)
    if not da["params"].get("cases_file_fingerprint") or not db["params"].get(
        "cases_file_fingerprint"
    ):
        notes.append(
            "one side was produced by an older harness: its precision counted every load of an unexpected skill, not the `allow` rule"
        )
    elif (
        da["params"]["cases_file_fingerprint"] != db["params"]["cases_file_fingerprint"]
        and not view_notes
    ):
        notes.append(
            "cases.json was edited between A and B, but no case that both ran changed its definition"
        )
    for metrics, tag in ((ma_all, "A"), (mb_all, "B")):
        ex = metrics["overall"].get("excluded_failed")
        if ex:
            notes.append(f"{tag}: {ex} failed Run(s) left out")
    lines += [f"  note: {n}" for n in notes]
    # cases whose verdict moved: a different pass-rate or a different kind of verdict (the sample counts in the label text are not a change)
    ca = {c["id"]: c for c in da["cases"]}
    cb = {c["id"]: c for c in db["cases"]}
    moved, still = [], []
    for cid in shared:
        va, vb = (ca.get(cid) or {}).get("verdict"), (cb.get(cid) or {}).get("verdict")
        if va is None or vb is None:
            continue  # an older label that kept no verdict: nothing to compare
        ra = frac(va["ok"], va["n"]) if va.get("kind") == "full" else None
        rb = frac(vb["ok"], vb["n"]) if vb.get("kind") == "full" else None
        sym = mark(ra, rb)
        if va.get("kind") != vb.get("kind") or sym != SAME:
            moved.append(
                [
                    cid,
                    va["label"],
                    vb["label"],
                    sym,
                    _loaded_text(ca.get(cid)),
                    _loaded_text(cb.get(cid)),
                ]
            )
        else:
            still.append(cid)
    lines.append("")
    if moved:
        lines.append("  cases whose verdict moved:")
        lines += [
            "    " + line
            for line in table([["case", "A", "B", "", "A loaded", "B loaded"], *moved])
        ]
    else:
        lines.append("  no routing case changed its verdict")
    if still:
        lines.append(f"  unchanged ({len(still)}): {', '.join(still)}")
    lines.append("")
    return lines


def _loaded_text(case: dict[str, Any] | None) -> str:
    if not case:
        return "-"
    seen: dict[str, int] = {}
    scored = [s for s in case["samples"] if "loaded" in s and s.get("status") != "failed"]
    for s in scored:
        for name in s["loaded"]:
            seen[name] = seen.get(name, 0) + 1
    if not seen:
        return "none"
    return ", ".join(f"{name} {k}/{len(scored)}" for name, k in sorted(seen.items()))


# ---------------------------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------------------------


def runs_by_task(data: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for run in ((data["tasks"] or {}).get("runs") or {}).values():
        out.setdefault(run["task"], []).append(normalize_run(run))
    for runs in out.values():
        runs.sort(key=lambda r: r["n"])
    return out


def check_rate(runs: list[dict[str, Any]], name: str) -> tuple[int, int] | None:
    """(passes, evaluated) over the Runs where the check was evaluated (pass or fail); None when it never was."""
    k = n = 0
    for run in runs:
        c = (run.get("checks") or {}).get(name)
        if c and c["status"] in ("pass", "fail"):
            n += 1
            k += 1 if c["status"] == "pass" else 0
    return (k, n) if n else None


def run_skills(runs: list[dict[str, Any]]) -> str:
    seen: dict[str, int] = {}
    for run in runs:
        names = run.get("skills_loaded")
        if names is None:
            names = (run.get("summary") or {}).get("skills_loaded", [])
        for name in names:
            seen[name] = seen.get(name, 0) + 1
    return ", ".join(f"{name} {k}/{len(runs)}" for name, k in sorted(seen.items())) or "none"


def run_refs(runs: list[dict[str, Any]]) -> str:
    refs: dict[str, int] = {}
    for run in runs:
        files = run.get("skill_files_read")
        if files is None:
            files = (run.get("summary") or {}).get("skill_files_read", [])
        for f in files:
            refs[f] = refs.get(f, 0) + 1
    return ", ".join(f"{f} {k}/{len(runs)}" for f, k in sorted(refs.items())) or "none"


def tasks_block(
    a: dict[str, Any], b: dict[str, Any], spec_tasks: list[dict[str, Any]], stats: dict[str, Any]
) -> list[str]:
    ra, rb = runs_by_task(a), runs_by_task(b)
    if not ra or not rb:
        missing = [d["label"] for d in (a, b) if not runs_by_task(d)]
        return ["TASKS", f"  not compared: no task results in {', '.join(missing)}", ""]
    task_ids = [t["id"] for t in spec_tasks if t["id"] in ra and t["id"] in rb]
    only = [
        f"{t['id']} (only in {'A' if t['id'] in ra else 'B'})"
        for t in spec_tasks
        if (t["id"] in ra) != (t["id"] in rb)
    ]
    if not task_ids:
        return ["TASKS", f"  not compared: A and B ran no task in common ({', '.join(only)})", ""]
    must = {t["id"]: set(t["must"]) for t in spec_tasks}
    lines = [
        "TASKS  (pass-rate over the task's Runs where the check was evaluated, A→B; * = in the task's must list; ▲ better, ▼ worse, blank = same)"
    ]
    if only:
        lines.append(f"  note: compared on the tasks both labels ran; left out: {', '.join(only)}")
    names = tasks_mod.ALL_CHECKS
    changes: list[tuple[int, str, str, str, str, str]] = []
    rows = [["check", *task_ids]]
    verdict_row = ["VERDICT (PASS Runs)"]
    for tid in task_ids:

        def verdicts(runs: list[dict[str, Any]]) -> tuple[int, int] | None:
            judged = [r for r in runs if r["verdict"] in ("PASS", "FAIL")]
            return (
                (sum(1 for r in judged if r["verdict"] == "PASS"), len(judged)) if judged else None
            )

        xa, xb = verdicts(ra.get(tid, [])), verdicts(rb.get(tid, []))
        sa = f"{xa[0]}/{xa[1]}" if xa else "-"
        sb = f"{xb[0]}/{xb[1]}" if xb else "-"
        sym = mark(frac(*xa) if xa else None, frac(*xb) if xb else None)
        verdict_row.append(f"{sa}→{sb}" + ("" if sym == SAME else f" {sym}"))
        stats["pass_a"] = stats.get("pass_a", 0) + (xa[0] if xa else 0)
        stats["pass_b"] = stats.get("pass_b", 0) + (xb[0] if xb else 0)
        stats["runs_a"] = stats.get("runs_a", 0) + (xa[1] if xa else 0)
        stats["runs_b"] = stats.get("runs_b", 0) + (xb[1] if xb else 0)
    rows.append(verdict_row)
    for name in names:
        row = [name]
        shown = False
        for tid in task_ids:
            # the same rule as tasks.md: a task-specific check on a task that does not ask for it is not a result (a brief has no tabs to test)
            asked = (
                name in must.get(tid, set())
                or name in tasks_mod.HYGIENE
                or name == "html_published"
            )
            xa = check_rate(ra.get(tid, []), name) if asked else None
            xb = check_rate(rb.get(tid, []), name) if asked else None
            if xa is None and xb is None:
                row.append("-")
                continue
            shown = True
            sa = f"{xa[0]}/{xa[1]}" if xa else "-"
            sb = f"{xb[0]}/{xb[1]}" if xb else "-"
            sym = mark(frac(*xa) if xa else None, frac(*xb) if xb else None)
            star = "*" if name in must.get(tid, set()) else ""
            row.append(f"{sa}→{sb}" + ("" if sym == SAME else f" {sym}") + star)
            if sym != SAME:
                changes.append((0 if star else 1, sym, tid, name + star, sa, sb))
                stats["better" if sym == UP else "worse"] = (
                    stats.get("better" if sym == UP else "worse", 0) + 1
                )
        if shown:
            rows.append(row)
    lines += ["  " + line for line in table(rows)]
    lines.append("")
    if changes:
        lines.append("  changed checks (must-checks first):")
        for _, sym, tid, name, sa, sb in sorted(
            changes, key=lambda c: (c[0], c[1] != DOWN, c[2], c[3])
        ):
            lines.append(f"    {sym} {tid:<14} {name:<32} {sa} → {sb}")
    else:
        lines.append("  no task check changed")
    lines.append("")
    lines.append("  who loaded what (tasks):")
    for tid in task_ids:
        for tag, runs in (("A", ra.get(tid, [])), ("B", rb.get(tid, []))):
            if not runs:
                continue
            calls = statistics.median([r["tool_count"] for r in runs]) if runs else 0
            wall = statistics.median([r["elapsed_s"] for r in runs]) if runs else 0
            html_report = sum(r.get("html_report_runs", 0) for r in runs)
            lines.append(
                f"    {tid:<14} {tag}: skills {run_skills(runs)} | reference files {run_refs(runs)} | html-report runs {html_report} | "
                f"tool calls (median) {calls:g} | wall {wall:.0f}s (median)"
            )
    lines.append("")
    return lines


# ---------------------------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------------------------


def describe(data: dict[str, Any]) -> str:
    m = meta_of(data)
    toolkit = Path(m["toolkit_dir"]).name if m.get("toolkit_dir") else "image toolkit"
    return (
        f"{data['label']}: model {m.get('answering_model')} | skills {', '.join(m.get('catalog', [])) or '?'} "
        f"(#{m.get('skills_manifest') or '?'}) | {toolkit} | image {m.get('image_id') or '?'}"
    )


def compare_labels(
    label_a: str,
    label_b: str,
    cases_file: Path = common.DEFAULT_CASES,
    color: bool = False,
    sections: tuple[str, ...] = ("routing", "tasks"),
) -> str:
    """The comparison as text. `sections` limits it to what the caller ran (`variant routing` has nothing to say about tasks)."""
    a, b = load_label(label_a), load_label(label_b)
    spec = common.load_cases(cases_file)
    lines = [f"A  {describe(a)}", f"B  {describe(b)}"]
    ma, mb = meta_of(a), meta_of(b)
    warn = []
    if (
        ma.get("answering_model")
        and mb.get("answering_model")
        and ma["answering_model"] != mb["answering_model"]
    ):
        warn.append(
            f"the answering models differ ({ma['answering_model']} vs {mb['answering_model']}): the numbers mix two effects"
        )
    if ma.get("image_id") and mb.get("image_id") and ma["image_id"] != mb["image_id"]:
        warn.append("the application images differ")
    va, vb = (a["tasks"] or {}).get("checker_version"), (b["tasks"] or {}).get("checker_version")
    if va and vb and va != vb:
        warn.append(f"the task reports were judged by different checker versions (v{va} vs v{vb})")
    lines += [f"WARNING: {w}" for w in warn]
    lines.append("")
    stats: dict[str, Any] = {}
    view = routing_view(a, b) if "routing" in sections and a["routing"] and b["routing"] else None
    body: list[str] = []
    if "routing" in sections:
        body += routing_block(a, b, view)  # type: ignore[arg-type]  # view is only read when both have routing
    if "tasks" in sections:
        body += tasks_block(a, b, spec["tasks"], stats)
    lines += headline(view, stats)
    lines += body
    return colorize("\n".join(lines).rstrip() + "\n", color)


def headline(
    view: tuple[dict[str, Any], dict[str, Any], list[str], list[str]] | None, stats: dict[str, Any]
) -> list[str]:
    """The numbers to read first."""
    lines = ["HEADLINE"]
    if view:
        ma, mb = view[0], view[1]
        parts = []
        for key, name, hib in METRICS:
            xa, xb = overall_metric(ma, key), overall_metric(mb, key)
            ra, rb = (
                (frac(xa["k"], xa["n"]) if xa else None),
                (frac(xb["k"], xb["n"]) if xb else None),
            )
            sym = mark(ra, rb, hib)
            fmt = lambda r: "n/a" if r is None else f"{100 * r:.0f}%"  # noqa: E731
            parts.append(f"{name} {fmt(ra)} → {fmt(rb)}" + ("" if sym == SAME else f" {sym}"))
        lines.append("  routing   " + " | ".join(parts))
    if stats.get("runs_a") is not None:
        sym = mark(frac(stats["pass_a"], stats["runs_a"]), frac(stats["pass_b"], stats["runs_b"]))

        def runs(k: int, n: int) -> str:
            return (
                f"{k}/{n}" if n else "n/a"
            )  # a label written before verdicts existed has none to count

        lines.append(
            f"  tasks     Runs meeting every must-check {runs(stats['pass_a'], stats['runs_a'])} → {runs(stats['pass_b'], stats['runs_b'])}"
            + ("" if sym == SAME else f" {sym}")
            + f" | checks better {stats.get('better', 0)}, worse {stats.get('worse', 0)}"
        )
    lines.append("")
    return lines
