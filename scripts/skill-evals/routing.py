"""Skill LOADING: the routing evaluation (precision, recall, forbidden-load) and its report.

Case fields (cases.json `routing[]`; `_routing_semantics` there is the source of truth):
  q          the user's message
  history    earlier turns the message refers to ([{role, content}]), sent as the request's `history`
  load       Skills that must ALL be loaded before the Run is stopped
  allow      Skills that may be loaded without counting against precision
  forbid     Skills that must not be loaded
  max_tool_calls / max_seconds   the budget, when the model researches before it builds (default 8 calls, 180 s)

What is measured, per sample (a sample is one Run), is written into every routing.md and is pinned by DEFINITIONS below.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import common
import driver
from common import compress, pct
from driver import Api, EarlyStop, RunRecord, follow_run

DEFAULT_MAX_TOOL_CALLS = 8
DEFAULT_MAX_SECONDS = 180.0

STOP_LABEL = {
    "all_loaded": "loaded",
    "first_load_skill": "skill",
    "skill_batches": "skill",
    "forbidden_load": "FORBIDDEN",
    "max_tool_calls": "cap",
    "timeout": "time",
    "finished": "done",
    "failed": "FAILED",
    "cancelled": "cancelled",
}

DEFINITIONS = """\
**Definitions** (one sample = one Run; `loaded(s)` = the Run called `load_skill(name=s)`, any path):

- **Case sample passes** iff every skill of `load` that the catalog has was loaded and no skill of `forbid` was loaded. A skill the catalog lacks is *absent*: it is not waited for, not counted in recall, and the verdict ignores it.
- **recall(s)** = samples with `s` in `load` where `loaded(s)` / samples with `s` in `load`.
- **precision(s)** = TP / (TP + FP). TP = samples where `s` is loaded and `s` is in `load`. FP = samples where `s` is loaded and `s` is in neither `load` nor `allow` (a forbidden load is an FP too). A load of `s` with `s` in `allow` and not in `load` is counted in neither: it is shown as *allowed*. Overall precision = ΣTP / Σ(TP + FP) over the catalog's skills.
- **forbidden-load rate(s)** = samples with `s` in `forbid` where `loaded(s)` / samples with `s` in `forbid`. Overall = samples where any forbidden skill was loaded / samples whose case forbids at least one skill. A forbidden skill the catalog lacks cannot be loaded, so it is left out (`n/a`), and a case that forbids only absent skills does not count: the denominators grow when the skill exists.
- Runs that failed (provider error) carry no routing information and are left out of every number; their count is stated.
- **Stop rule**, first that applies: a `forbid` skill is loaded (`FORBIDDEN`); every `load` skill the catalog has is loaded (`loaded`); when the catalog has none of them, the first model turn that loads a skill (`skill`); the tool-call cap (`cap`); the time cap (`time`); or the Run finishes (`done`). A Run is never stopped before its decision is visible, and never waits for a skill that does not exist."""


# ---------------------------------------------------------------------------------------------
# Case settings
# ---------------------------------------------------------------------------------------------


def case_settings(
    case: dict[str, Any], catalog: set[str], opts: argparse.Namespace
) -> dict[str, Any]:
    """The effective settings of one case: its own fields, the command line's overrides, the defaults."""
    cli_calls, cli_seconds = (
        getattr(opts, "max_tool_calls", None),
        getattr(opts, "max_seconds", None),
    )
    return {
        "load": list(case["load"]),
        "allow": list(case.get("allow", [])),
        "forbid": list(case["forbid"]),
        "required": [s for s in case["load"] if s in catalog],
        "absent": [
            s for s in case["load"] + case["forbid"] + case.get("allow", []) if s not in catalog
        ],
        "max_tool_calls": cli_calls
        if cli_calls is not None
        else case.get("max_tool_calls", DEFAULT_MAX_TOOL_CALLS),
        "max_seconds": float(
            cli_seconds if cli_seconds is not None else case.get("max_seconds", DEFAULT_MAX_SECONDS)
        ),
        "batches": getattr(opts, "skill_batches", None),
        "history_turns": len(case.get("history") or []),
    }


def early_stop_for(settings: dict[str, Any]) -> EarlyStop:
    return EarlyStop(
        required=tuple(settings["required"]),
        forbidden=tuple(settings["forbid"]),
        batches=settings["batches"],
        max_tool_calls=settings["max_tool_calls"],
        max_seconds=settings["max_seconds"],
    )


def case_fingerprint(case: dict[str, Any], settings: dict[str, Any]) -> str:
    """Identifies what a saved sample was run against, so a resumed label never reuses a sample of an edited case."""
    return common.fingerprint(case["q"], case.get("history"), settings)


# ---------------------------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------------------------


def score_sample(
    case: dict[str, Any], settings: dict[str, Any], catalog: set[str], record: RunRecord
) -> dict[str, Any]:
    names: list[str] = []
    for item in record.skills_loaded:
        name = item.get("name")
        if name and name not in names:
            names.append(name)
    loaded = [s for s in names if s in catalog]
    load, allow, forbid = set(settings["load"]), set(settings["allow"]), set(settings["forbid"])
    wanted = [s for s in loaded if s in load]
    allowed = [s for s in loaded if s in allow and s not in load]
    unwanted = [s for s in loaded if s not in load and s not in allow]
    missing = [s for s in settings["required"] if s not in loaded]
    forbidden_hit = [s for s in loaded if s in forbid]
    return {
        "loaded": loaded,
        "loaded_paths": [
            f"{item['name']}:{item['path']}" for item in record.skills_loaded if item.get("name")
        ],
        "attempted_absent": [s for s in names if s not in catalog],
        "wanted": wanted,
        "allowed": allowed,
        "unwanted": unwanted,
        "missing": missing,
        "forbidden_hit": forbidden_hit,
        "recall_ok": (not missing) if settings["required"] else None,
        "forbid_ok": not forbidden_hit,
    }


def sample_passes(sample: dict[str, Any]) -> bool:
    return bool(sample["forbid_ok"]) and sample["recall_ok"] is not False


def scorable(sample: dict[str, Any]) -> bool:
    """A sample that carries routing information: it exists and its Run did not fail."""
    return "loaded" in sample and sample.get("status") != "failed"


def routing_metrics(
    cases: list[dict[str, Any]],
    samples_by_case: dict[str, list[dict[str, Any]]],
    settings_by_case: dict[str, dict[str, Any]],
    catalog: set[str],
    skills: list[str],
) -> dict[str, Any]:
    """recall / precision / forbidden-load per skill and overall, exactly as DEFINITIONS states."""
    per_skill: dict[str, dict[str, Any]] = {}
    absent_attempts = 0
    excluded = 0
    for case in cases:
        for sample in samples_by_case.get(case["id"], []):
            if not scorable(sample):
                excluded += 1
            else:
                absent_attempts += len(sample["attempted_absent"])
    for skill in skills:
        in_catalog = skill in catalog
        expected = hit = tp = fp = allowed = f_n = f_hit = 0
        for case in cases:
            st = settings_by_case[case["id"]]
            for sample in samples_by_case.get(case["id"], []):
                if not scorable(sample):
                    continue
                loaded = skill in sample["loaded"]
                if skill in st["load"] and in_catalog:
                    expected += 1
                    hit += 1 if loaded else 0
                if loaded:
                    if skill in st["load"]:
                        tp += 1
                    elif skill in st["allow"]:
                        allowed += 1
                    else:
                        fp += 1
                if skill in st["forbid"] and in_catalog:
                    f_n += 1
                    f_hit += 1 if loaded else 0
        per_skill[skill] = {
            "in_catalog": in_catalog,
            "recall": {"k": hit, "n": expected},
            "precision": {"k": tp, "n": tp + fp, "tp": tp, "fp": fp, "allowed": allowed},
            "forbidden_load": {"k": f_hit, "n": f_n},
        }
    present = [m for m in per_skill.values() if m["in_catalog"]]
    overall: dict[str, Any] = {
        "recall": {
            "k": sum(m["recall"]["k"] for m in present),
            "n": sum(m["recall"]["n"] for m in present),
        },
        "precision": {
            "k": sum(m["precision"]["tp"] for m in present),
            "n": sum(m["precision"]["n"] for m in present),
            "tp": sum(m["precision"]["tp"] for m in present),
            "fp": sum(m["precision"]["fp"] for m in present),
            "allowed": sum(m["precision"]["allowed"] for m in present),
        },
    }
    sample_n = sample_hit = 0
    for case in cases:
        if not [f for f in settings_by_case[case["id"]]["forbid"] if f in catalog]:
            continue
        for sample in samples_by_case.get(case["id"], []):
            if scorable(sample):
                sample_n += 1
                sample_hit += 1 if sample["forbidden_hit"] else 0
    overall["forbidden_load"] = {"k": sample_hit, "n": sample_n}
    overall["excluded_failed"] = excluded
    overall["absent_attempts"] = absent_attempts
    return {"per_skill": per_skill, "overall": overall}


def case_verdict(
    case: dict[str, Any], settings: dict[str, Any], samples: list[dict[str, Any]]
) -> dict[str, Any]:
    scored = [s for s in samples if scorable(s)]
    ok = sum(1 for s in scored if sample_passes(s))
    if settings["load"] and not settings["required"]:
        # every skill the case expects is absent: only the forbid side can be judged (and only for forbidden skills that exist)
        kind = (
            "forbid-only"
            if [f for f in settings["forbid"] if f not in settings["absent"]]
            else "absent"
        )
    else:
        kind = "full"
    if not scored:
        label = "no scored sample"
    elif kind == "absent":
        label = "n/a (skill absent)"
    elif kind == "forbid-only":
        label = f"forbid-only {ok}/{len(scored)}"
    else:
        label = f"{'PASS' if ok == len(scored) else 'FAIL' if ok == 0 else 'PARTIAL'} {ok}/{len(scored)}"
    return {
        "ok": ok,
        "n": len(scored),
        "kind": kind,
        "label": label,
        "failed_runs": len(samples) - len(scored),
    }


# ---------------------------------------------------------------------------------------------
# One sample
# ---------------------------------------------------------------------------------------------


def _brief_args(args: dict[str, Any], limit: int = 160) -> dict[str, Any]:
    out = {}
    for key, value in args.items():
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        out[key] = text if len(text) <= limit else text[:limit] + "..."
    return out


def run_sample(
    args: tuple[
        dict[str, Any], dict[str, Any], int, Path, set[str], int, int, Callable[[str], None]
    ],
) -> dict[str, Any]:
    case, settings, n, out_dir, catalog, index, total, log = args
    raw_path = out_dir / "runs" / f"{case['id']}-{n}.json"
    fp = case_fingerprint(case, settings)
    if raw_path.exists():
        saved = json.loads(raw_path.read_text())
        if saved.get("fingerprint") == fp:
            log(f"[{index:>3}/{total}] {case['id']}-{n} (resumed)")
            return saved
        log(
            f"[{index:>3}/{total}] {case['id']}-{n}: the saved sample is of another case definition; running it again"
        )
    api = Api()
    try:
        record = follow_run(
            api,
            case["q"],
            history=case.get("history"),
            early_stop=early_stop_for(settings),
            label=f"{case['id']}-{n}",
        )
    finally:
        api.close()
    scored = score_sample(case, settings, catalog, record)
    sample = {
        "case": case["id"],
        "n": n,
        "fingerprint": fp,
        "run_id": record.run_id,
        "status": record.status,
        "stopped_by": record.stopped_by,
        "error": record.error_message,
        "elapsed_s": record.elapsed_s,
        "attempts": record.attempts,
        "backoffs": record.backoffs,
        "tool_count": record.tool_count,
        "tool_names": [call["name"] for call in record.tool_calls],
        "tool_calls": [
            {
                "seq": c["seq"],
                "batch": c["batch"],
                "name": c["name"],
                "args": _brief_args(c["args"]),
            }
            for c in record.tool_calls
        ],
        "notes": record.assistant_notes[:3],
        **scored,
    }
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(sample, ensure_ascii=False, indent=1))
    log(
        f"[{index:>3}/{total}] {case['id']}-{n} loaded={scored['loaded'] or '-'} "
        f"stop={STOP_LABEL.get(record.stopped_by, record.stopped_by)} calls={record.tool_count} {record.elapsed_s:.0f}s"
        + (f" ERROR={record.error_message}" if record.status == "failed" else "")
    )
    return sample


# ---------------------------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------------------------


def _expected_cell(st: dict[str, Any]) -> str:
    parts = []
    if st["load"]:
        parts.append(("all of " if len(st["load"]) > 1 else "") + ", ".join(st["load"]))
    else:
        parts.append("no skill")
    if st["allow"]:
        parts.append(f"allow {', '.join(st['allow'])}")
    if st["forbid"]:
        parts.append(f"not {', '.join(st['forbid'])}")
    text = "; ".join(parts)
    if st["absent"]:
        text += f" (absent: {', '.join(sorted(set(st['absent'])))})"
    if st["max_tool_calls"] != DEFAULT_MAX_TOOL_CALLS or st["max_seconds"] != DEFAULT_MAX_SECONDS:
        text += f" [budget {st['max_tool_calls']} calls / {st['max_seconds']:.0f} s]"
    if st["history_turns"]:
        text += f" [+{st['history_turns']} history turns]"
    return text


def render_routing_md(doc: dict[str, Any]) -> str:
    meta, params, metrics = doc["meta"], doc["params"], doc["metrics"]
    catalog = set(meta["catalog"])
    lines = [f"# Routing: {doc['label']}", ""]
    lines.append(
        f"Answering model: `{meta.get('answering_model')}` (preset `{meta.get('query_model_preset')}`) | catalog: "
        f"{', '.join(sorted(catalog))} | samples/case: {params['samples']} | default budget {DEFAULT_MAX_TOOL_CALLS} calls / "
        f"{DEFAULT_MAX_SECONDS:.0f} s, per case where a case says so"
        + (
            f" | `--skill-batches {params['skill_batches']}` override"
            if params.get("skill_batches")
            else ""
        )
    )
    lines.append("")
    lines.append(DEFINITIONS)
    lines.append("")
    lines.append(
        "Cell legend: skills loaded in order (`-` none), then how the Run was stopped: `[loaded]` all required skills loaded, "
        "`[skill]` the first model turn that loads a skill, `[FORBIDDEN]` a forbidden skill was loaded, `[cap]` tool-call cap, "
        "`[time]` time cap, `[done]` the Run finished, `[FAILED]` the Run failed. `absent:x` = tried to load a skill the catalog does not have; "
        "`+x` = loaded a skill that is neither expected nor allowed."
    )
    lines.append("")
    lines.append("| case | query | expected | loaded per sample | verdict |")
    lines.append("|---|---|---|---|---|")
    for case in doc["cases"]:
        st = case["settings"]
        cells = []
        for sample in case["samples"]:
            if "loaded" not in sample:
                cells.append("ERROR")
                continue
            cell = ",".join(sample["loaded"]) or "-"
            if sample.get("unwanted"):
                cell += " " + " ".join(f"+{s}" for s in sample["unwanted"] if s not in st["forbid"])
            if sample["attempted_absent"]:
                cell += " absent:" + ",".join(sample["attempted_absent"])
            cell += f" [{STOP_LABEL.get(sample['stopped_by'], sample['stopped_by'])}]"
            cells.append(cell.replace("  ", " "))
        query = case["q"].replace("\n", " ")
        query = query[:34] + ("…" if len(query) > 34 else "")
        lines.append(
            f"| {case['id']} | {query} | {_expected_cell(st)} | {' / '.join(cells)} | {case['verdict']['label']} |"
        )
    lines.append("")
    lines.append("## Metrics (sample level)")
    lines.append("")
    lines.append(
        "| skill | in catalog | load recall | load precision (TP, FP, allowed) | forbidden-load rate |"
    )
    lines.append("|---|---|---|---|---|")
    for skill, m in metrics["per_skill"].items():
        p = m["precision"]
        prec = (
            f"{pct(p['k'], p['n'])} (TP {p['tp']}, FP {p['fp']}, allowed {p['allowed']})"
            if p["n"] or p["allowed"]
            else "n/a (never loaded)"
        )
        if not m["in_catalog"]:
            lines.append(f"| {skill} | NO (absent) | n/a | n/a | n/a (absent) |")
            continue
        lines.append(
            f"| {skill} | yes | {pct(m['recall']['k'], m['recall']['n'])} | {prec} | {pct(m['forbidden_load']['k'], m['forbidden_load']['n'])} |"
        )
    o = metrics["overall"]
    lines.append(
        f"| **overall** | | {pct(o['recall']['k'], o['recall']['n'])} | "
        f"{pct(o['precision']['k'], o['precision']['n'])} (TP {o['precision']['tp']}, FP {o['precision']['fp']}, allowed {o['precision']['allowed']}) | "
        f"{pct(o['forbidden_load']['k'], o['forbidden_load']['n'])} (samples) |"
    )
    lines.append("")
    notes = []
    if o.get("excluded_failed"):
        notes.append(f"{o['excluded_failed']} failed Run(s) left out of every number")
    if o.get("absent_attempts"):
        notes.append(
            f"{o['absent_attempts']} attempt(s) to load a skill the catalog does not have (not counted anywhere above)"
        )
    if notes:
        lines.append("Note: " + "; ".join(notes) + ".")
        lines.append("")
    instead = [
        c
        for c in doc["cases"]
        if c["settings"]["absent"]
        and any(s in c["settings"]["load"] for s in c["settings"]["absent"])
    ]
    if instead:
        lines.append("## What the model did instead (cases that expect a skill the catalog lacks)")
        lines.append("")
        for case in instead:
            lines.append(f"### {case['id']}: {case['q'].splitlines()[0][:70]}")
            for sample in case["samples"]:
                if "tool_names" not in sample:
                    continue
                wrote_html = any(
                    c["name"] in ("write", "bash")
                    and ".html" in json.dumps(c["args"], ensure_ascii=False)
                    for c in sample["tool_calls"]
                )
                lines.append(
                    f"- sample {sample['n']} [{STOP_LABEL.get(sample['stopped_by'], sample['stopped_by'])}, "
                    f"{sample['tool_count']} calls, {sample['elapsed_s']:.0f}s]: {compress(sample['tool_names'])}"
                    + (" | touched .html" if wrote_html else "")
                    + (f" | loaded {sample['loaded']}" if sample["loaded"] else "")
                )
                if sample.get("notes"):
                    lines.append(f"  - model said: {sample['notes'][0][:200]!r}")
            lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# The command
# ---------------------------------------------------------------------------------------------


def run_routing(
    opts: argparse.Namespace,
    log: Callable[[str], None] = lambda s: print(s, flush=True),
    echo_report: bool = True,
) -> dict[str, Any]:
    spec = common.load_cases(opts.cases_file)
    cases = spec["routing"]
    if opts.cases:
        wanted = [c.strip() for c in opts.cases.split(",") if c.strip()]
        unknown = [c for c in wanted if c not in {x["id"] for x in cases}]
        if unknown:
            sys.exit(f"evaluate.py: unknown case ids {unknown}")
        cases = [c for c in cases if c["id"] in wanted]
    label = opts.label or time.strftime("routing-%Y%m%d-%H%M%S")
    with common.runs_lock(f"routing {label}"):
        out = common.prepare_label_dir(label, opts.resume, "routing")
        api = Api()
        meta = common.stack_meta(api)
        catalog = set(meta["catalog"])
        settings = {c["id"]: case_settings(c, catalog, opts) for c in cases}
        params = {
            "samples": opts.samples,
            "skill_batches": opts.skill_batches,
            "cases": [c["id"] for c in cases],
            "cases_file_fingerprint": common.fingerprint(spec["routing"]),
            "max_tool_calls_cli": opts.max_tool_calls,
            "max_seconds_cli": opts.max_seconds,
        }
        (out / "meta.json").write_text(
            json.dumps({"label": label, "params": params, **meta}, ensure_ascii=False, indent=1)
        )
        log(
            f"[evaluate] routing '{label}': {len(cases)} cases x {opts.samples} samples; model={meta['answering_model']} catalog={sorted(catalog)}"
        )
        order = [
            (case, n) for n in range(1, opts.samples + 1) for case in cases
        ]  # interleaved: a partial run still covers every case
        total = len(order)
        jobs = [
            (f"{case['id']}-{n}", (case, settings[case["id"]], n, out, catalog, i + 1, total, log))
            for i, (case, n) in enumerate(order)
        ]
        results = common.run_in_pool(jobs, run_sample, opts.concurrency)
    samples_by_case: dict[str, list[dict[str, Any]]] = {c["id"]: [] for c in cases}
    for key, sample in results.items():
        samples_by_case[key.rsplit("-", 1)[0]].append(sample)
    for samples in samples_by_case.values():
        samples.sort(key=lambda s: s.get("n", 0))
    metrics = routing_metrics(cases, samples_by_case, settings, catalog, list(spec["skills"]))
    doc = {
        "label": label,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "meta": {k: v for k, v in meta.items() if k != "catalog_detail"},
        "params": params,
        "backoffs": driver.THROTTLE.events,
        "cases": [
            {
                "id": c["id"],
                "src": c.get("src"),
                "q": c["q"],
                "settings": settings[c["id"]],
                "samples": samples_by_case[c["id"]],
                "verdict": case_verdict(c, settings[c["id"]], samples_by_case[c["id"]]),
            }
            for c in cases
        ],
        "metrics": metrics,
    }
    (out / "routing.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    markdown = render_routing_md(doc)
    (out / "routing.md").write_text(markdown)
    if echo_report:
        print("\n" + markdown)
    if driver.THROTTLE.events:
        log(
            f"[evaluate] {len(driver.THROTTLE.events)} back-off event(s) on provider limits; see routing.json"
        )
    return doc
