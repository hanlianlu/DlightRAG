#!/usr/bin/env python3
"""Run the Skill routing eval against a live DlightRAG and keep the evidence.

The **routing** suite in `docs/skills-routing-eval.md` asks three questions of every case -- did the
right Skill load, did the run read the documents the Skill says that case needs, and did the
delivered work pass the Skill's own gates -- and it had no run log until 2026-10-14. Two real runs
on that date showed why the log matters: one never found the Skill at all, and one applied its
concepts with its own engine while every named gate went untriggered.

This driver answers the first question mechanically (`expected_loaded` / `forbidden_loaded`, read
off the Skill load events), records what the run read and executed so a reader can answer the second
(nothing here compares `references_seen` with the documents a case needs), and keeps the raw
material for the third. It starts one Answer run per case over the REST API, waits for the run to
settle, and writes the transcript and artifact listing to an output directory, plus a `score.json`
holding what the run shows: which Skills were loaded and how often, the Skill document names the
answer's prose mentions, how many messages and tool calls the run spent, and whether the run
executed the Skill's own validator.

Usage:
    python3 scripts/run_skill_routing_eval.py --base-url http://127.0.0.1:8110 --out /tmp/skill-eval

The endpoint is the REST surface (`make eval-serve` runs an app for it with no sign-in, on a database
of its own). Two things are deliberate: the run mode is explicit (`research`, because Skills exist only
there -- a run started without it never sees the catalog, which is a measurement of the harness rather
than of routing), and `requested_skill` is never set, because a run told which Skill to load measures
nothing about routing.

Exit status is 0 when every case settled, 1 otherwise, and 2 when `--only` matches no case; a case
that times out is reported and kept.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import re
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any

# The cases are those a synthetic pass ran in the repository on 2026-10-14, so the two settings can
# be compared directly: same query, same expectations, different delivery.
CASES: tuple[dict[str, str], ...] = (
    {
        "id": "report",
        "query": "给我一份 Volvo Cars 的 HTML 报告，要有财务数据和一张图。",
        "expect": "interactive-html",
        "forbidden": "",
    },
)

TERMINAL = {"succeeded", "failed", "cancelled", "canceled", "error"}


def _request(url: str, payload: dict[str, Any] | None = None, timeout: float = 60.0) -> dict:
    """One JSON request; raises for a transport or status failure with the body in the message."""
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(  # noqa: S310 - loopback URL supplied by the operator
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data else {},
        method="POST" if data else "GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:  # the body names the field the endpoint rejected
        raise RuntimeError(
            f"{url} -> {exc.code}: {exc.read().decode('utf-8', 'replace')[:300]}"
        ) from exc


def _start(base_url: str, query: str, mode: str = "research") -> str:
    """Start one run. `mode` is explicit because Skills exist only in Research runs."""
    payload: dict[str, Any] = {"query": query, "mode": mode}
    descriptor = _request(f"{base_url}/answer", payload)
    run_id = descriptor.get("run_id")
    if not run_id:
        raise RuntimeError(f"answer returned no run_id: {descriptor}")
    return str(run_id)


def _wait(base_url: str, run_id: str, *, timeout_seconds: float, poll_seconds: float) -> str:
    """Poll one run until it settles, returning its final status word."""
    deadline = time.monotonic() + timeout_seconds
    while True:
        state = _request(f"{base_url}/runs/{run_id}")
        status = str(state.get("status", "")).lower()
        if status in TERMINAL:
            return status
        if time.monotonic() > deadline:
            return f"timeout({status})"
        time.sleep(poll_seconds)


def _events(base_url: str, run_id: str) -> str:
    """The run's server-sent event stream as text; the tool record lives here, not in the transcript."""
    try:
        with urllib.request.urlopen(  # noqa: S310 - loopback URL supplied by the operator
            f"{base_url}/runs/{run_id}/events", timeout=120.0
        ) as response:
            return response.read().decode("utf-8", "replace")
    except urllib.error.URLError as exc:
        return f"events unavailable: {exc}"


def _settled(url: str, *, attempts: int = 6, pause: float = 10.0) -> dict:
    """Read a result surface, tolerating the store's own settle lag.

    A run reaches a terminal status a moment before its result row is stored, and the surfaces
    answer 409 ("not available until the run has a stored result") in that window. A harness that
    treats that as fatal loses a thirty-minute run to a race measured in seconds.
    """
    last: Exception | None = None
    for _ in range(attempts):
        try:
            return _request(url, timeout=120.0)
        except RuntimeError as exc:
            if "409" not in str(exc):
                raise
            last = exc
            time.sleep(pause)
    raise RuntimeError(f"{url} never settled: {last}")


def _text_of(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False) if content is not None else ""


def _score(transcript: dict, events: str) -> dict:
    """What the run actually did, read off the event stream and the transcript.

    The event stream is the authoritative record of tool calls: the transcript carries the prose
    turns, and its `tool_calls` are not the whole tool record — an early version of this scorer read
    the transcript alone and reported "no Skill loaded" for a run that had loaded one twelve times.
    So a Skill load is counted from `tool_name: load_skill` events and named by `object_label`,
    which is the Skill's own name -- by distinct call id, because each call also emits progress and
    end events that would otherwise triple every count.
    """
    messages = transcript.get("messages", [])
    # One call emits tool_start, tool_progress and tool_end, so events must be counted by call id:
    # counting occurrences inflated every tool count three-fold, and the first version of this
    # scorer published those numbers as call counts.
    tools: dict[str, set[str]] = {}
    labels: dict[str, str] = {}
    for match in re.finditer(r'"call_id":\s*"([^"]+)"[^}]*?"tool_name":\s*"([a-z_]+)"', events):
        call_id, tool = match.group(1), match.group(2)
        tools.setdefault(tool, set()).add(call_id)
    for match in re.finditer(r'"call_id":\s*"([^"]+)"[^}]*?"object_label":\s*"([^"]+)"', events):
        labels[match.group(1)] = match.group(2)
    loaded: dict[str, int] = {}
    for call_id in tools.get("load_skill", ()):
        label = labels.get(call_id)
        if label:
            loaded[label] = loaded.get(label, 0) + 1
    # "The verifier ran" is read the way the product reads it: a recorded command naming the verifier
    # the loaded Skill declares today. A read of the script names it too, and the declaration is this
    # checkout's own frontmatter, not a second copy of the command in this scorer.
    declared = _declared_verifiers()
    verifier_ran = any(
        label is not None and any(name in label for name in declared)
        for call_id, label in labels.items()
        if call_id in tools.get("bash", set()) | tools.get("shell", set())
    )
    prose = "\n".join(_text_of(m) for m in messages)
    return {
        "messages": len(messages),
        "loaded": loaded,
        "tools": {
            tool: len(ids) for tool, ids in sorted(tools.items(), key=lambda kv: -len(kv[1]))
        },
        "load_skill_calls": len(tools.get("load_skill", ())),
        "references_seen": sorted(set(re.findall(r"(?:references/)?([a-z-]+\.md)", prose))),
        "validator_run": verifier_ran,
        "answer_chars": len(_text_of(messages[-1])) if messages else 0,
    }


# The checkout under test: the Skills whose frontmatter names the verifier a run must execute.
BUILTIN_SKILLS = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "dlightrag"
    / "engine"
    / "agent"
    / "builtin_skills"
)


@functools.cache
def _declared_verifiers() -> tuple[str, ...]:
    """The verifier file name each built-in Skill declares, from this checkout's frontmatter.

    Read through the product's own discovery: the frontmatter is the
    Skill's claim about what verifies its delivery, and the product attests a Run by a recorded
    command naming that file. A scorer that hard-coded the name would drift the day the declaration
    moved, and would then report a faithful Run as having run no verifier.
    """
    from dlightrag.engine.agent.skills import SkillCatalog

    catalog = SkillCatalog.discover(builtin_root=BUILTIN_SKILLS)
    return tuple(
        sorted(
            {
                PurePosixPath(skill.attestation).name
                for skill in catalog.metadata
                if skill.attestation is not None
            }
        )
    )


# --- the routing suite -------------------------------------------------------------------------


def _run_routing(args: argparse.Namespace, out: Path, wanted: set[str]) -> int:
    out.mkdir(parents=True, exist_ok=True)
    cases = [c for c in CASES if not wanted or c["id"] in wanted]
    if not cases:
        known = ", ".join(c["id"] for c in CASES)
        print(f"no case matches --only {sorted(wanted)}; the cases are: {known}", file=sys.stderr)
        return 2

    failures = 0
    for case in cases:
        case_dir = out / case["id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{case['id']}] starting: {case['query'][:70]}", flush=True)
        try:
            run_id = _start(args.base_url, case["query"], mode=args.mode)
        except RuntimeError as exc:
            print(f"[{case['id']}] refused: {exc}", flush=True)
            failures += 1
            continue
        started = time.monotonic()
        status = _wait(args.base_url, run_id, timeout_seconds=args.timeout, poll_seconds=args.poll)
        elapsed = round(time.monotonic() - started)
        # Only a settled Run has a closed event stream: for a Run still executing, the stream keeps
        # its keepalive open and reading it blocks past the socket timeout, which is how a timed-out
        # case used to hang the harness instead of being reported.
        events = _events(args.base_url, run_id) if status.startswith("succeeded") else ""
        (case_dir / "events.txt").write_text(events, encoding="utf-8")
        transcript = _settled(f"{args.base_url}/answer/{run_id}/transcript")
        artifacts = _settled(f"{args.base_url}/answer/{run_id}/artifacts")
        (case_dir / "transcript.json").write_text(
            json.dumps(transcript, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        (case_dir / "artifacts.json").write_text(
            json.dumps(artifacts, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        if not status.startswith("succeeded"):
            print(
                f"[{case['id']}] run {run_id} did not settle within the wait ({status}); its records "
                "are kept and it can be collected later with the same run id",
                flush=True,
            )
        score = _score(transcript, events) | {
            "case": case["id"],
            "run_id": run_id,
            "status": status,
            "seconds": elapsed,
            "expected_skill": case["expect"],
            "forbidden_skill": case["forbidden"],
        }
        score["expected_loaded"] = case["expect"] in score["loaded"]
        score["forbidden_loaded"] = bool(case["forbidden"]) and case["forbidden"] in score["loaded"]
        (case_dir / "score.json").write_text(
            json.dumps(score, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(
            f"[{case['id']}] {status} in {elapsed}s | load_skill_calls={score['load_skill_calls']} "
            f"loaded={sorted(score['loaded'])} | expected_loaded={score['expected_loaded']} "
            f"forbidden_loaded={score['forbidden_loaded']} | validator_run={score['validator_run']}",
            flush=True,
        )
        if not status.startswith("succeeded"):
            failures += 1

    print(f"done: {len(cases) - failures}/{len(cases)} settled", flush=True)
    return 0 if failures == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Skill routing eval against a live API.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8110")
    parser.add_argument("--out", default=os.path.join(tempfile.gettempdir(), "skill-eval"))
    parser.add_argument("--timeout", type=float, default=5400.0, help="seconds per case")
    parser.add_argument("--poll", type=float, default=20.0, help="seconds between status reads")
    parser.add_argument("--only", default="", help="comma-separated case ids")
    parser.add_argument(
        "--mode",
        default="research",
        help="Answer mode; Skills (load_skill + catalog) exist only in research runs",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    wanted = {part.strip() for part in args.only.split(",") if part.strip()}
    return _run_routing(args, Path(args.out), wanted)


if __name__ == "__main__":
    raise SystemExit(main())
