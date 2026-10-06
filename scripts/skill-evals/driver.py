#!/usr/bin/env python3
"""A small client of the DlightRAG REST API for the eval harness.

Submits a research-mode Answer Run, follows it, and collects what the evaluations need: the ordered
tool calls with their arguments, the final answer, usage, wall time and the published artifacts.

The API facts this relies on (verified against a live stack, docs/interfaces.md and the routes):

  POST   /answer                          202 + descriptor; JSON, or multipart (request + attachments)
  GET    /runs/{id}                       status, phase, result (once terminal), error_kind/message
  DELETE /runs/{id}                       cancel: 202 while pending, 200 once terminal (there is NO
                                          POST /answer/{id}/cancel)
  GET    /answer/{id}/transcript?limit=N  the lane's last N (<= 100) committed messages. It is LIVE: an
                                          assistant message carries a whole batch of tool calls with
                                          their arguments the moment the model turn commits, before the
                                          tools run; tool results follow as `tool` messages.
  GET    /answer/{id}/artifacts           descriptors (409 until the Run has a stored result)
  GET    /answer/{id}/artifacts/{rid}     the bytes

Because the transcript keeps only the last 100 messages, a long Run's early calls would fall out of
one final read; the follower therefore merges a snapshot every poll, and cross-checks the count
against the Run's own `trace.tool_observations` at the end.

A Run is never left running: every exit path (exception, KeyboardInterrupt, SIGTERM) cancels the Runs
this process started.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import signal
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

DEFAULT_BASE_URL = os.environ.get(
    "EVAL_API_URL", f"http://127.0.0.1:{os.environ.get('EVAL_API_PORT', '18100')}"
)
TERMINAL = frozenset({"succeeded", "failed", "cancelled"})
# A provider's concurrency/quota refusal surfaces as a failed Run (or a refused submission), not as a
# status code the driver sees directly; these markers in the error text identify it.
_RATE_LIMIT = re.compile(
    r"\b(402|429|503)\b|rate.?limit|concurren|too many requests|quota|insufficient|overloaded|"
    r"capacity|throttl",
    re.IGNORECASE,
)
_ARG_PREVIEW_CHARS = 4000


class ApiError(RuntimeError):
    """The API refused or failed a request in a way the driver cannot continue from."""


class RateLimited(ApiError):
    """A 402/429-class refusal: the caller backs off and retries."""


# ---------------------------------------------------------------------------------------------
# Throttle: one shared "do not submit before" clock, so a rate limit pauses every worker.
# ---------------------------------------------------------------------------------------------


class Throttle:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._not_before = 0.0
        self.events: list[dict[str, Any]] = []

    def wait(self) -> None:
        while True:
            with self._lock:
                delay = self._not_before - time.monotonic()
            if delay <= 0:
                return
            time.sleep(min(delay, 5.0))

    def hit(self, delay: float, reason: str, label: str = "") -> None:
        """Pause every submission for `delay` seconds, and say so."""
        with self._lock:
            self._not_before = max(self._not_before, time.monotonic() + delay)
            self.events.append(
                {
                    "at": time.strftime("%H:%M:%S"),
                    "delay_s": delay,
                    "reason": reason[:200],
                    "label": label,
                }
            )
        print(
            f"[driver] BACKOFF {delay:.0f}s ({label or 'run'}): provider/API limit ({reason[:160]})",
            file=sys.stderr,
            flush=True,
        )


THROTTLE = Throttle()


# ---------------------------------------------------------------------------------------------
# The set of Runs this process started and has not seen terminal: cancelled on every exit path.
# ---------------------------------------------------------------------------------------------

_ACTIVE: dict[str, str] = {}
_ACTIVE_LOCK = threading.Lock()
_BASE_FOR_CLEANUP = DEFAULT_BASE_URL


def _track(run_id: str, base_url: str) -> None:
    with _ACTIVE_LOCK:
        _ACTIVE[run_id] = base_url


def _untrack(run_id: str) -> None:
    with _ACTIVE_LOCK:
        _ACTIVE.pop(run_id, None)


def cancel_active_runs() -> list[str]:
    """Cancel every Run this process still holds; returns their ids."""
    with _ACTIVE_LOCK:
        held = dict(_ACTIVE)
    cancelled = []
    for run_id, base_url in held.items():
        try:
            httpx.delete(f"{base_url}/runs/{run_id}", timeout=15)
            cancelled.append(run_id)
        except Exception as exc:  # noqa: BLE001 - best effort on the way out
            print(f"[driver] could not cancel {run_id}: {exc}", file=sys.stderr)
    return cancelled


def install_exit_hooks() -> None:
    atexit.register(cancel_active_runs)

    def _on_signal(signum: int, _frame: Any) -> None:
        cancelled = cancel_active_runs()
        print(f"[driver] signal {signum}: cancelled {len(cancelled)} run(s)", file=sys.stderr)
        raise SystemExit(128 + signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _on_signal)
        except ValueError:  # not the main thread
            pass


# ---------------------------------------------------------------------------------------------
# The API client
# ---------------------------------------------------------------------------------------------


class Api:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def _request(
        self, method: str, path: str, *, retries: int = 3, **kwargs: Any
    ) -> httpx.Response:
        last: Exception | None = None
        for attempt in range(retries + 1):
            try:
                response = self._client.request(method, path, **kwargs)
            except (httpx.TransportError, httpx.TimeoutException) as exc:
                last = exc
                time.sleep(1.5 * (attempt + 1))
                continue
            if response.status_code in (402, 429):
                raise RateLimited(
                    f"{method} {path}: HTTP {response.status_code} {response.text[:200]}"
                )
            if response.status_code >= 500 and attempt < retries:
                last = ApiError(
                    f"{method} {path}: HTTP {response.status_code} {response.text[:200]}"
                )
                time.sleep(1.5 * (attempt + 1))
                continue
            return response
        raise ApiError(f"{method} {path}: {last}")

    def health(self) -> dict[str, Any]:
        response = self._request("GET", "/health")
        return response.json() if response.status_code == 200 else {}

    def skills(self) -> list[dict[str, Any]]:
        response = self._request("GET", "/web/api/skills")
        if response.status_code != 200:
            raise ApiError(f"GET /web/api/skills: HTTP {response.status_code}")
        return list(response.json()["skills"])

    def submit(
        self,
        query: str,
        *,
        mode: str = "research",
        attachments: list[Path] | None = None,
        effort: str | None = None,
        history: list[dict[str, str]] | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"query": query, "mode": mode}
        if effort:
            body["effort"] = effort
        if history:
            body["history"] = (
                history  # prior {role, content} turns, for cases that refer back ("the analysis just now")
            )
        headers = {"Idempotency-Key": idempotency_key or f"eval-{uuid.uuid4()}"}
        if attachments:
            files: list[tuple[str, tuple[str | None, bytes | str, str]]] = [
                ("request", (None, json.dumps(body, ensure_ascii=False), "application/json"))
            ]
            for path in attachments:
                files.append(
                    ("attachments", (path.name, path.read_bytes(), "application/octet-stream"))
                )
            response = self._request("POST", "/answer", files=files, headers=headers)
        else:
            response = self._request("POST", "/answer", json=body, headers=headers)
        if response.status_code != 202:
            raise ApiError(f"POST /answer: HTTP {response.status_code} {response.text[:400]}")
        return dict(response.json())

    def run(self, run_id: str) -> dict[str, Any]:
        response = self._request("GET", f"/runs/{run_id}")
        if response.status_code != 200:
            raise ApiError(f"GET /runs/{run_id}: HTTP {response.status_code} {response.text[:200]}")
        return dict(response.json())

    def cancel(self, run_id: str) -> dict[str, Any]:
        response = self._request("DELETE", f"/runs/{run_id}")
        if response.status_code not in (200, 202):
            raise ApiError(
                f"DELETE /runs/{run_id}: HTTP {response.status_code} {response.text[:200]}"
            )
        return dict(response.json())

    def transcript(self, run_id: str, limit: int = 100) -> list[dict[str, Any]]:
        response = self._request("GET", f"/answer/{run_id}/transcript", params={"limit": limit})
        if response.status_code != 200:
            raise ApiError(f"GET transcript {run_id}: HTTP {response.status_code}")
        return list(response.json()["messages"])

    def artifacts(self, run_id: str) -> list[dict[str, Any]]:
        response = self._request("GET", f"/answer/{run_id}/artifacts")
        if response.status_code != 200:
            raise ApiError(
                f"GET artifacts {run_id}: HTTP {response.status_code} {response.text[:200]}"
            )
        return list(response.json().get("artifacts") or [])

    def artifact_bytes(self, run_id: str, resource_id: str) -> bytes:
        response = self._request("GET", f"/answer/{run_id}/artifacts/{resource_id}")
        if response.status_code != 200:
            raise ApiError(f"GET artifact {resource_id}: HTTP {response.status_code}")
        return response.content

    def list_runs(self) -> list[dict[str, Any]]:
        runs: list[dict[str, Any]] = []
        after: str | None = None
        while True:
            params: dict[str, Any] = {"limit": 100}
            if after:
                params["after"] = after
            batch = self._request("GET", "/runs", params=params).json()["runs"]
            runs.extend(batch)
            if len(batch) < 100:
                return runs
            after = batch[-1]["run_id"]


# ---------------------------------------------------------------------------------------------
# Transcript accumulation
# ---------------------------------------------------------------------------------------------


@dataclass
class ToolCall:
    seq: int  # 1-based order of requests in the Run
    batch: int  # 1-based index of the assistant message that requested it
    call_id: str
    name: str
    args: dict[str, Any]
    argument_error: str | None = None
    is_error: bool | None = None  # from the matching tool message, once it exists
    result_preview: str = ""
    result_chars: int = 0

    def short_args(self) -> dict[str, Any]:
        """Arguments with long strings (a whole HTML file in `write`) cut to a preview."""
        out: dict[str, Any] = {}
        for key, value in self.args.items():
            if isinstance(value, str) and len(value) > _ARG_PREVIEW_CHARS:
                out[key] = value[:_ARG_PREVIEW_CHARS] + f"... [{len(value)} chars]"
            else:
                out[key] = value
        return out


class Transcript:
    """Merges successive last-100 snapshots of the lane transcript into one ordered history."""

    def __init__(self) -> None:
        self.calls: dict[str, ToolCall] = {}
        self.order: list[str] = []
        self.batches = 0
        self.assistant_texts: list[str] = []
        self._seen_batches: set[tuple[str, ...]] = set()
        self._seen_texts: set[tuple[int, str]] = set()
        self.final_text = ""

    def merge(self, messages: list[dict[str, Any]]) -> None:
        for _index, message in enumerate(messages):
            role = message.get("role")
            if role == "assistant":
                tool_calls = message.get("tool_calls") or []
                ids = tuple(str(tc.get("id")) for tc in tool_calls)
                if tool_calls:
                    if ids not in self._seen_batches:
                        self._seen_batches.add(ids)
                        self.batches += 1
                        for tc in tool_calls:
                            call_id = str(tc.get("id"))
                            if call_id in self.calls:
                                continue
                            arguments = tc.get("arguments")
                            self.calls[call_id] = ToolCall(
                                seq=len(self.order) + 1,
                                batch=self.batches,
                                call_id=call_id,
                                name=str(tc.get("name")),
                                args=arguments
                                if isinstance(arguments, dict)
                                else {"_raw": arguments},
                                argument_error=tc.get("argument_error"),
                            )
                            self.order.append(call_id)
                        text = (message.get("content") or "").strip()
                        if text:
                            self.assistant_texts.append(text)
                else:
                    self.final_text = (message.get("content") or "").strip() or self.final_text
            elif role == "tool":
                call = self.calls.get(str(message.get("tool_call_id")))
                if call is not None and call.is_error is None:
                    content = message.get("content")
                    text = (
                        content
                        if isinstance(content, str)
                        else json.dumps(content, ensure_ascii=False)
                    )
                    call.is_error = bool(message.get("is_error"))
                    call.result_chars = len(text)
                    call.result_preview = text[:300]

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [self.calls[call_id] for call_id in self.order]

    def pending(self) -> int:
        return sum(1 for call in self.calls.values() if call.is_error is None)

    def skill_loads(self) -> list[dict[str, Any]]:
        return [
            {
                "seq": call.seq,
                "batch": call.batch,
                "name": call.args.get("name"),
                "path": call.args.get("path") or "SKILL.md",
                "is_error": call.is_error,
            }
            for call in self.tool_calls
            if call.name == "load_skill"
        ]


# ---------------------------------------------------------------------------------------------
# Following one Run
# ---------------------------------------------------------------------------------------------


@dataclass
class EarlyStop:
    """When a routing Run has shown its decision, or has used its budget: cancel it.

    The first rule that applies wins (`routing_stop` is the one place that says so):

      forbidden_load    a Skill named in `forbidden` was loaded (attempted loads count)
      skill_batches     `batches` is set (the explicit override) and that many model turns loaded a Skill;
                        with batches == 1 the reason is `first_load_skill`
      all_loaded        `required` is not empty and every Skill in it was loaded
      first_load_skill  neither `batches` nor `required` is set, and a Skill was loaded
      max_tool_calls    the Run requested `max_tool_calls` tool calls (counted per model turn: a batch is not split)
      timeout           `max_seconds` passed (checked by the follower)

    `required` holds the Skills of a case's `load` that the Run's catalog has: a Skill that does not exist cannot be
    waited for. A Skill counts as loaded once a `load_skill` call named it and did not fail.
    """

    required: tuple[str, ...] = ()
    forbidden: tuple[str, ...] = ()
    batches: int | None = None
    max_tool_calls: int | None = 8
    max_seconds: float = 180.0
    grace_seconds: float = 4.0  # after the decision, for the batch's tool results to commit


def routing_stop(calls: list[ToolCall], stop: EarlyStop) -> str:
    """The reason to stop a routing Run now, or "" to let it go on."""
    loads = [call for call in calls if call.name == "load_skill"]
    attempted = {str(call.args.get("name")) for call in loads}
    succeeded = {str(call.args.get("name")) for call in loads if call.is_error is not True}
    if stop.forbidden and attempted & set(stop.forbidden):
        return "forbidden_load"
    batches = len({call.batch for call in loads})
    if stop.batches is not None:
        if batches >= stop.batches:
            return "first_load_skill" if stop.batches == 1 else "skill_batches"
    elif stop.required:
        if set(stop.required) <= succeeded:
            return "all_loaded"
    elif batches >= 1:
        return "first_load_skill"
    if stop.max_tool_calls is not None and len(calls) >= stop.max_tool_calls:
        return "max_tool_calls"
    return ""


@dataclass
class RunRecord:
    run_id: str = ""
    query: str = ""
    mode: str = "research"
    submitted_at: str = ""
    elapsed_s: float = 0.0
    status: str = ""  # succeeded | failed | cancelled | error
    stopped_by: str = ""  # finished | first_load_skill | max_tool_calls | timeout | failed | error
    error_kind: str | None = None
    error_message: str | None = None
    attempts: int = 1
    backoffs: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    skills_loaded: list[dict[str, Any]] = field(default_factory=list)
    tool_count: int = 0
    tool_observation_count: int | None = None
    assistant_notes: list[str] = field(default_factory=list)
    answer: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    trace: dict[str, Any] = field(default_factory=dict)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    artifact_outcome: dict[str, Any] | None = None
    started_at: str | None = None
    finished_at: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _usage_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Pick the token numbers out of the result's usage (inclusive of child agents)."""
    usage = (
        (result.get("usage") or {}).get("inclusive_usage_details")
        or (result.get("usage") or {}).get("usage_details")
        or {}
    )
    return {
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "cached_tokens": usage.get("input_tokens_details.cached_tokens"),
        "reasoning_tokens": usage.get("output_tokens_details.reasoning_tokens"),
        "raw": result.get("usage"),
    }


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def follow_run(
    api: Api,
    query: str,
    *,
    mode: str = "research",
    attachments: list[Path] | None = None,
    history: list[dict[str, str]] | None = None,
    early_stop: EarlyStop | None = None,
    timeout_s: float = 900.0,
    poll_s: float = 1.0,
    label: str = "",
    fetch_artifacts: bool = False,
    max_backoffs: int = 4,
) -> RunRecord:
    """Submit one Run and follow it to its end (or to its early stop); never leave it running.

    A provider limit (a failed Run or a refused submission whose error reads like 402/429) pauses
    every worker through THROTTLE and resubmits, up to `max_backoffs` times.
    """
    record = RunRecord(query=query, mode=mode, submitted_at=_now_iso())
    delays = [20, 45, 90, 150, 240]
    for attempt in range(max_backoffs + 1):
        THROTTLE.wait()
        try:
            outcome = _follow_once(
                api,
                query,
                mode=mode,
                attachments=attachments,
                history=history,
                early_stop=early_stop,
                timeout_s=timeout_s,
                poll_s=poll_s,
                label=label,
                fetch_artifacts=fetch_artifacts,
            )
        except RateLimited as exc:
            outcome = RunRecord(query=query, mode=mode, status="failed", stopped_by="failed")
            outcome.error_kind, outcome.error_message = "rate_limited", str(exc)
        outcome.attempts = attempt + 1
        outcome.backoffs = record.backoffs
        record = outcome
        limited = record.status == "failed" and _RATE_LIMIT.search(
            f"{record.error_kind or ''} {record.error_message or ''}"
        )
        if not limited or attempt == max_backoffs:
            return record
        delay = delays[min(attempt, len(delays) - 1)]
        THROTTLE.hit(delay, f"{record.error_kind}: {record.error_message}", label)
        record.backoffs = [
            *record.backoffs,
            {
                "after_attempt": attempt + 1,
                "delay_s": delay,
                "reason": f"{record.error_kind}: {record.error_message}"[:300],
            },
        ]
        # The failed attempt's tool calls are not the next attempt's; start the record clean.
        record = RunRecord(
            query=query, mode=mode, submitted_at=record.submitted_at, backoffs=record.backoffs
        )
    return record


def _follow_once(
    api: Api,
    query: str,
    *,
    mode: str,
    attachments: list[Path] | None,
    history: list[dict[str, str]] | None,
    early_stop: EarlyStop | None,
    timeout_s: float,
    poll_s: float,
    label: str,
    fetch_artifacts: bool,
) -> RunRecord:
    record = RunRecord(query=query, mode=mode, submitted_at=_now_iso())
    started = time.monotonic()
    descriptor = api.submit(query, mode=mode, attachments=attachments, history=history)
    run_id = str(descriptor["run_id"])
    record.run_id = run_id
    _track(run_id, api.base_url)
    transcript = Transcript()
    limit_seconds = early_stop.max_seconds if early_stop else timeout_s
    decided_at: float | None = None
    stop_reason = ""
    status: dict[str, Any] = {}
    try:
        while True:
            status = api.run(run_id)
            state = status["status"]
            try:
                transcript.merge(api.transcript(run_id))
            except ApiError:
                pass  # a read between commits; the next poll has it
            elapsed = time.monotonic() - started
            if state in TERMINAL:
                stop_reason = (
                    "finished"
                    if state == "succeeded"
                    else ("failed" if state == "failed" else "cancelled")
                )
                break
            if early_stop is not None:
                reason = routing_stop(transcript.tool_calls, early_stop)
                if reason and decided_at is None:
                    decided_at = time.monotonic()
                    stop_reason = reason
                if decided_at is not None and (
                    transcript.pending() == 0
                    or time.monotonic() - decided_at >= early_stop.grace_seconds
                ):
                    break
            if elapsed >= limit_seconds:
                stop_reason = "timeout"
                break
            time.sleep(poll_s)
        if status.get("status") not in TERMINAL:
            _cancel_and_wait(api, run_id)
            status = api.run(run_id)
            try:
                transcript.merge(api.transcript(run_id))
            except ApiError:
                pass
        else:
            transcript.merge(api.transcript(run_id))
    finally:
        # Whatever happened above, the Run is not left running.
        try:
            final = api.run(run_id)
            if final["status"] not in TERMINAL:
                _cancel_and_wait(api, run_id)
        except Exception as exc:  # noqa: BLE001
            print(f"[driver] cleanup of {run_id} failed: {exc}", file=sys.stderr)
        else:
            _untrack(run_id)
    record.elapsed_s = round(time.monotonic() - started, 2)
    record.status = str(status.get("status"))
    record.stopped_by = stop_reason or "finished"
    record.error_kind = status.get("error_kind")
    record.error_message = status.get("error_message")
    record.started_at, record.finished_at = status.get("started_at"), status.get("finished_at")
    record.tool_calls = [
        {**{k: v for k, v in asdict(call).items() if k != "args"}, "args": call.short_args()}
        for call in transcript.tool_calls
    ]
    record.tool_count = len(transcript.order)
    record.skills_loaded = transcript.skill_loads()
    record.assistant_notes = transcript.assistant_texts
    result = status.get("result") or {}
    if result:
        record.answer = str(result.get("answer") or "")
        record.usage = _usage_summary(result)
        trace = result.get("trace") or {}
        record.trace = {
            key: trace.get(key)
            for key in (
                "agent_turns",
                "agent_effort",
                "agent_stop_reason",
                "prompt_cache",
                "web_search_cost_dollars",
            )
        }
        observations = trace.get("tool_observations")
        if isinstance(observations, list):
            record.tool_observation_count = len(observations)
        record.artifacts = [
            {k: v for k, v in item.items() if k not in ("url",)}
            for item in (result.get("artifacts") or [])
        ]
        outcome = result.get("artifact_outcome")
        record.artifact_outcome = dict(outcome) if isinstance(outcome, dict) else None
        if fetch_artifacts:
            record.artifacts = _download_artifacts(api, run_id, record.artifacts)
    if record.status == "failed" and not record.error_message:
        record.error_message = "Run failed without an error message"
    return record


def _cancel_and_wait(api: Api, run_id: str, wait_s: float = 90.0, repeat_s: float = 5.0) -> None:
    """Cancel, and keep asking every `repeat_s` until the Run is terminal (a cancel can be refused while a handoff starts)."""
    deadline = time.monotonic() + wait_s
    try:
        api.cancel(run_id)
    except ApiError as exc:
        print(f"[driver] cancel {run_id}: {exc}", file=sys.stderr)
    last = time.monotonic()
    while time.monotonic() < deadline:
        if api.run(run_id)["status"] in TERMINAL:
            return
        time.sleep(1.0)
        if time.monotonic() - last >= repeat_s:
            last = time.monotonic()
            try:
                api.cancel(run_id)
            except ApiError:
                pass
    print(
        f"[driver] WARNING: {run_id} still not terminal {wait_s:.0f}s after cancel", file=sys.stderr
    )


def _download_artifacts(
    api: Api, run_id: str, descriptors: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Attach the bytes of every available artifact (as `_bytes`, popped by the caller before JSON)."""
    out = []
    for descriptor in descriptors:
        item = dict(descriptor)
        if item.get("status") == "available" and item.get("resource_id"):
            try:
                item["_bytes"] = api.artifact_bytes(run_id, str(item["resource_id"]))
            except ApiError as exc:
                item["download_error"] = str(exc)
        out.append(item)
    return out


# ---------------------------------------------------------------------------------------------
# CLI: one-off Runs and the janitor
# ---------------------------------------------------------------------------------------------


def _cli() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="submit one research Run and follow it")
    run.add_argument("query")
    run.add_argument("--attach", action="append", type=Path, default=[])
    run.add_argument(
        "--early-stop",
        action="store_true",
        help="routing-style early stop (first load_skill batch, 8 calls, 180 s)",
    )
    run.add_argument("--timeout", type=float, default=900.0)
    run.add_argument("--out", type=Path, help="write the record (and artifacts) here")
    sub.add_parser("cancel-all", help="cancel every non-terminal Run of this stack")
    sub.add_parser("catalog", help="print the Skills the API reports")
    args = parser.parse_args()
    install_exit_hooks()
    api = Api(args.base_url)
    if args.cmd == "catalog":
        for skill in api.skills():
            print(f"{skill['name']:<20} {skill['source']:<8} {skill['description'][:100]}")
        return 0
    if args.cmd == "cancel-all":
        cancelled = 0
        for descriptor in api.list_runs():
            if descriptor["status"] not in TERMINAL:
                api.cancel(descriptor["run_id"])
                cancelled += 1
                print("cancelled", descriptor["run_id"])
        print(f"{cancelled} run(s) cancelled")
        return 0
    record = follow_run(
        api,
        args.query,
        attachments=args.attach or None,
        early_stop=EarlyStop() if args.early_stop else None,
        timeout_s=args.timeout,
        fetch_artifacts=args.out is not None,
        label="cli",
    )
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        for item in record.artifacts:
            data = item.pop("_bytes", None)
            if data is not None:
                name = Path(
                    str(item.get("filename") or item.get("path") or item["resource_id"])
                ).name
                (args.out / name).write_bytes(data)
        (args.out / "run.json").write_text(
            json.dumps(record.to_json(), ensure_ascii=False, indent=2)
        )
    summary = record.to_json()
    summary.pop("answer", None)
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:6000])
    print("\nANSWER:\n" + record.answer[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
