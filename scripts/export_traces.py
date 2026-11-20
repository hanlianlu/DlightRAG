#!/usr/bin/env python3
# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Export the agent traces owners contributed, as JSONL for training work. Repository host tool,
read-only.

It reads only what owners chose to contribute (Settings → Privacy, ADR 0041), and nothing else:
a Run is selected only if its owner contributes now and the Agent Session its routing row names
began at or after the moment they did. A Session begins at the earlier of its own row's creation
and the acceptance of its first remaining Run, so neither a queue wait nor retention pruning its
oldest Runs moves the start later. A Session is therefore exported whole or not at all, a
conversation begun before the permission is never read however long it goes on or is forked, and
a Run with no Session (a Retrieval, a Corpus Mutation) is not a conversation and is never read.
The rule is part of the selection query, and no option includes more: naming an owner or a Run
narrows what is exported and never widens it. For a development database, turn the switch on for
your own account.

Two modes share one record builder and that one rule (`make export-traces ARGS="..."` passes
the arguments on):

* Raw, the default: one record per Run on stdout or in `--out`, ids as stored, for an operator
  to inspect. `--out` is written with mode 0600, replacing a file already there, and its
  directory is left as it is. `--since` and `--until` bound when a Run was created.
* Dataset, `--dataset DIR`: `train.jsonl`, `eval.jsonl` and `datasheet.json`, a de-identified,
  split and reproducible training set (ADR 0044). It takes whole Sessions, every finished Run of
  each, and `--owner`, `--since` and `--until` apply to a Session as a unit, by when it began.
  Every owner, Session, Run, Entry and Child id, and the ids minted for a tool call's intent and
  attempt, a compaction, a steer and a fork's lane, becomes an HMAC-SHA256 under the operator's
  key, `DLIGHTRAG_EXPORT_KEY` (at least 32 characters, in `.env` or the environment), wherever
  the id stands, so the same key gives the same pseudonyms and datasets made with it link up. Text
  is not scrubbed. A Session goes to `eval` or `train` by a hash of its stored id, so re-exporting never
  moves one. Each dataset is a full re-export: an owner who turns the switch off is simply absent
  from the next. `--limit`, `--run-id`, `--dedupe` and `--out` do not apply to it, because each
  would cut a Session or leave the directory.

One line is one Run: the episode. What it was asked, how it was routed (mode, model
fingerprints, context policy revision, the entry it ended on, which a fork starts from), every
entry it wrote — the user message, each assistant message with the tool calls it made, the usage
it cost and the call that made it (ADR 0043), each tool result with the call it answers and when
that tool started and ended — the Session context it ran in, the sources its answer cites, the
evidence it gathered, the session notes it wrote, and the owner's judgements of its answers
(`labels`).

A Run's own entries are the ones that state it as their writer, and its record holds those and
nothing else its session holds. A conversation is one session across all its Runs, and a fork
opens another lane of the same session. Deleting a conversation, or retention pruning a Run,
removes the Run and leaves what it wrote in the session for as long as another Run routes there,
so the entries around a Run's own can be words of a Run its owner deleted. Which Run an entry
belongs to is never read from the clock, the lane or the Runs beside it, and an entry that
states no Run is read by none. Its routed children come with it, each with the entries of its
own session that the Run wrote, so a delegated episode is whole. Runs of one conversation share
a session id and every step names its parent entry, so a consumer rebuilds the turns before a
Run by following `parent_entry_id` into the other records of that session, all of them covered
by the same permission.

The one place words of a Run that is gone are read is `session.carried`. A Run ran on the chain
of entries above its first, up to the root of the session. An entry on it that no exported Run
wrote, because retention pruned its Run, its owner deleted it, or an option left it out, is
context the Run really saw, so the record carries that entry whole. The rest of the chain is
entries other exported Runs wrote, so a consumer rebuilds it by following `parent_entry_id` from
the Run's first step through the steps of the session's records and the Run's `carried`; the
record lists no chain of ids, which would grow with the square of a conversation. Words of a fork
the owner deleted are on no surviving Run's chain, and are not carried.

It is a host tool: it reads the deployment's PostgreSQL in one read-only, repeatable-read
transaction, so an owner who turns the switch off or keeps chatting while it runs changes nothing
it writes. It imports no application code and gives the server no export surface. The records
carry user content and retrieved passages, so decide where they may live before writing them
anywhere.

A child's `summary` is not exported. It is that child's last message word for word — the store
keeps it so a roster can draw a row without reading a transcript — and a record carrying it
beside the entries that already hold it repeats one passage twice in one episode, which teaches
a model the house style rather than the task.

Left out of every record: the provider's replay state (`provider_state`), the Run's events
(only the start and end of each tool call are read, for its timing), error messages (the kind
stays), the Run's stored result but for the sources its answer cites, and the passage text of
the evidence snapshots, each of which repeats the whole ledger so far. What the agent saw is in
its tool results, and the passages the answer cites are in `result.sources`.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, NamedTuple

import asyncpg

SUMMARY = "Export the agent traces owners contributed, as JSONL for training work"
ROOT = Path(__file__).resolve().parents[1]
#: The record shape this tool writes. A consumer reads it to know what it is parsing.
SCHEMA_VERSION = 5
#: Where the operator keeps the key that makes the pseudonyms of a dataset (ADR 0044).
KEY_NAME = "DLIGHTRAG_EXPORT_KEY"
KEY_MIN_CHARS = 32
#: The share of Sessions that go to `eval`, unless the operator names another.
EVAL_FRACTION = 0.10
SPLIT_VERSION = "dlightrag-split-v1"
_SPLIT_SALT = SPLIT_VERSION.encode() + b"\0"
_KEY_FINGERPRINT_MESSAGE = b"dlightrag-export-key-fingerprint-v1"
DATASET_FILES = ("train.jsonl", "eval.jsonl", "datasheet.json")
_FINISHED = ("succeeded", "failed", "cancelled")
_DEFAULT_PG = {
    "host": "localhost",
    "port": 5432,
    "user": "dlightrag",
    "password": "dlightrag",
    "database": "dlightrag",
}

# Which Runs may be read at all (ADR 0041, Conversation Contribution), decided by joins so that
# nothing reaches the rest of the export unless it passes: the Run's owner contributes now, and the
# Agent Session its routing row names began at or after the moment they did. A Session begins at
# the earlier of two times: its own row's `created_at`, which lasts as long as any Run of the
# Session does, so retention pruning a long Session's oldest Runs cannot move it later, and the
# earliest `created_at` among the Runs that routing still names it for, per owner, which covers
# the wait before the row is written. A Session is covered whole or not at all, and a Run with no
# routing row has no Session and is not read. The narrowing options below only ever narrow.
# `--since` and `--until` arrive as text and are read as timestamps here, because the driver binds
# a timestamp from a datetime and not a string.
#
# `$6` is the dataset mode (ADR 0044). There the Session is the unit: the bounds measure when it
# began, not when a Run was created, so a Session is in or out as a whole; and only finished Runs
# are read, because a Run still going has no end to learn from and its prepared input holds
# execution secrets (the salt of its resource handles, its connection bindings) that an export
# must never carry. The consent rule above is the same in both modes.
_RUNS_SQL = """
SELECT run.run_id, run.owner_id, run.run_kind, run.lane, run.submitted_by,
       run.access_scope_kind, run.prepared_input_json, run.accepted_input_json,
       run.request_fingerprint, run.status, run.phase, run.stop_reason,
       run.result_json -> 'sources' AS sources_json, run.error_kind,
       run.superseded_by_run_id, run.retention_seconds, run.purge_after,
       run.events_trimmed_at, run.created_at, run.started_at, run.finished_at
FROM dlightrag_runs AS run
JOIN dlightrag_privacy_settings AS consent
  ON consent.owner_id = run.owner_id
JOIN dlightrag_answer_run_routing AS routing
  ON routing.owner_id = run.owner_id AND routing.run_id = run.run_id
JOIN (
    SELECT member.owner_id, member.agent_session_id, min(accepted.created_at) AS first_accepted_at
    FROM dlightrag_answer_run_routing AS member
    JOIN dlightrag_runs AS accepted
      ON accepted.owner_id = member.owner_id AND accepted.run_id = member.run_id
    WHERE ($1::text IS NULL OR member.owner_id = $1)
    GROUP BY member.owner_id, member.agent_session_id
) AS session_start
  ON session_start.owner_id = routing.owner_id
 AND session_start.agent_session_id = routing.agent_session_id
LEFT JOIN dlightrag_agent_sessions AS session_row
  ON session_row.owner_id = routing.owner_id AND session_row.session_id = routing.agent_session_id
CROSS JOIN LATERAL (
    SELECT LEAST(session_start.first_accepted_at, session_row.created_at) AS began_at
) AS lifetime
WHERE consent.contribute
  AND lifetime.began_at >= consent.contribute_since
  AND (NOT $6::boolean OR run.status IN ('succeeded', 'failed', 'cancelled'))
  AND ($1::text IS NULL OR run.owner_id = $1)
  AND ($2::text IS NULL
       OR CASE WHEN $6::boolean THEN lifetime.began_at ELSE run.created_at END
          >= $2::text::timestamptz)
  AND ($3::text IS NULL
       OR CASE WHEN $6::boolean THEN lifetime.began_at ELSE run.created_at END
          < $3::text::timestamptz)
  AND ($4::uuid[] IS NULL OR run.run_id = ANY($4))
ORDER BY run.created_at, run.run_id
LIMIT $5
"""

# The owners who contribute now, read in the same snapshot as the Runs, so the datasheet's
# statement of its consent basis cannot disagree with its records.
_CONSENT_SQL = """
SELECT count(*)::int AS owners, min(contribute_since) AS earliest
FROM dlightrag_privacy_settings
WHERE contribute AND ($1::text IS NULL OR owner_id = $1)
"""

_ROUTING_SQL = """
SELECT run_id, owner_id, requested_mode, valid_modes, resolved_mode, model_fingerprints,
       context_policy_revision, agent_session_id, agent_lane_id, source_lane_id,
       fork_point_entry_id
FROM dlightrag_answer_run_routing
WHERE run_id = ANY($1::uuid[])
"""

# A Run's entries are the ones that state it as their writer (`run_id`, which the Session
# repository stamps on every entry it commits, a Child's included), and no others: the Session
# holds the entries of every Run that wrote into it, and of Runs that are gone, because deleting a
# conversation or pruning a Run removes the Run and leaves what it wrote for as long as another
# Run routes to the Session. An entry that states no Run is read by none.
_ENTRIES_SQL = """
SELECT run_id, session_id, sequence, entry_id, parent_entry_id, entry_type, schema_version,
       timestamp, payload_json
FROM dlightrag_agent_session_entries
WHERE run_id = ANY($1::uuid[])
ORDER BY session_id, sequence
"""

# The shape of every Session the Runs ran in, ids and writers only: enough to walk the chain above
# a Run's first entry and to tell which entries on it no exported Run wrote. The words of those
# are read afterwards, for the entries on a chain alone.
_SKELETON_SQL = """
SELECT entry_id, parent_entry_id, run_id
FROM dlightrag_agent_session_entries
WHERE (owner_id, session_id) IN (SELECT * FROM unnest($1::text[], $2::uuid[]))
"""

_CARRIED_SQL = """
SELECT run_id, session_id, sequence, entry_id, parent_entry_id, entry_type, schema_version,
       timestamp, payload_json
FROM dlightrag_agent_session_entries
WHERE entry_id = ANY($1::uuid[])
"""

# Only the two events that bound a tool call, and only the id that ties them to it: the rest of a
# Run's events are its streamed text and its progress, which the entries already hold.
_TOOL_EVENTS_SQL = """
SELECT run_id, event_type, payload ->> 'call_id' AS call_id, created_at
FROM dlightrag_run_events
WHERE (owner_id, run_id) IN (SELECT * FROM unnest($1::text[], $2::uuid[]))
  AND event_type IN ('tool_start', 'tool_end')
ORDER BY run_id, event_sequence
"""

# A snapshot of the ledger after one tool settlement: each row repeats everything admitted so far,
# so only what identifies a snapshot is read. The order ends in the primary key, because the rows
# of one settlement share a time and an ordinal.
_EVIDENCE_SQL = """
SELECT run_id, result_ordinal, content_digest, locator, locator_digest, created_at
FROM dlightrag_answer_evidence
WHERE run_id = ANY($1::uuid[])
ORDER BY run_id, created_at, session_id, intent_id, result_ordinal
"""

_CHILDREN_SQL = """
SELECT run_id, child_session_id, parent_session_id, parent_intent_id, parent_call_id, depth, status,
       objective, model_role, context_mode, summary, usage_json, plan_json, budget_json,
       tools_json, cancel_requested_at, created_at, updated_at
FROM dlightrag_answer_child_sessions
WHERE run_id = ANY($1::uuid[])
ORDER BY created_at, child_session_id
"""

_NOTES_SQL = """
SELECT written_by_run_id, session_id, relative_path, size_bytes, content, content_digest,
       revision, updated_at
FROM dlightrag_answer_session_notes
WHERE written_by_run_id = ANY($1::uuid[])
ORDER BY written_by_run_id, session_id, relative_path
"""

# The owner's judgement of each answered message: one row is kept per message, so what is read is
# the latest. `agent` is `main` or a Child's Session id, and `sequence` is the message's own place
# in that agent's Session.
_LABELS_SQL = """
SELECT run_id, agent, sequence, value, reasons, note, created_at
FROM dlightrag_answer_feedback
WHERE (owner_id, run_id) IN (SELECT * FROM unnest($1::text[], $2::uuid[]))
ORDER BY run_id, agent <> 'main', agent, sequence
"""


# ── reading the deployment's PostgreSQL ──────────────────────────────────────────────────


def read_env(repo_root: Path = ROOT) -> dict[str, str]:
    """Read DLIGHTRAG_* values from the repository .env, overlaid on os.environ."""
    values: dict[str, str] = {}
    env_file = repo_root / ".env"
    if env_file.is_file():
        for raw in env_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            values[key.strip()] = value.strip().strip('"').strip("'")
    for key, value in os.environ.items():
        if key.startswith("DLIGHTRAG_"):
            values[key] = value
    return values


def connection_settings(env: Mapping[str, str]) -> dict[str, Any]:
    """The PostgreSQL settings the deployment uses, so the tool reads the same database."""
    prefix = "DLIGHTRAG_STORAGE__POSTGRES__"

    def value(name: str) -> str:
        return env.get(f"{prefix}{name.upper()}") or str(_DEFAULT_PG[name])

    return {
        "host": value("host"),
        "port": int(value("port")),
        "user": value("user"),
        "password": value("password"),
        "database": value("database"),
    }


# ── assembling one episode ───────────────────────────────────────────────────────────────


def _json(value: object) -> object:
    if isinstance(value, (str, bytes, bytearray)):
        text = value.decode("utf-8", "replace") if isinstance(value, bytes | bytearray) else value
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    return value


def _text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", "replace")
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _own_summary(
    child: Mapping[str, Any],
    child_steps: Mapping[str, Sequence[Mapping[str, Any]]],
) -> str | None:
    """A child's summary when its own entries do not already say it, and None when they do.

    A child often concludes with the words its last entry holds, and the record then has the same
    passage twice; sometimes the summary is a document of its own that no entry carries, and
    dropping it would lose what the child produced. Null says which of the two this is.
    """
    summary = str(child.get("summary") or "")
    if not summary:
        return None
    said: set[str] = set()
    for entry in child_steps.get(str(child["child_session_id"]), []):
        payload = entry.get("payload_json")
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except ValueError:
                payload = None
        if isinstance(payload, Mapping):
            said.add(str(payload.get("content") or ""))
    return None if summary in said else summary


def _stamp(value: datetime | str | None) -> str | None:
    if value is None:
        return None
    return value.isoformat() if isinstance(value, datetime) else str(value)


def step(entry: Mapping[str, Any]) -> dict[str, Any]:
    """One committed entry as a training step, its payload kept whole but for the provider's
    replay state, which only the provider that wrote it can read.

    `schema_version` is the entry's own, the version of the payload format a reader is parsing.
    `tool` is filled in for a tool result by `_time_tools`.
    """
    payload = _json(entry.get("payload_json"))
    if isinstance(payload, dict):
        payload = {name: value for name, value in payload.items() if name != "provider_state"}
    return {
        "sequence": entry.get("sequence"),
        "entry_id": str(entry["entry_id"]),
        "parent_entry_id": (
            str(entry["parent_entry_id"]) if entry.get("parent_entry_id") else None
        ),
        "type": entry.get("entry_type"),
        "schema_version": entry.get("schema_version"),
        "timestamp": _stamp(entry.get("timestamp")),
        "payload": payload,
        "tool": None,
    }


def _interval_ms(started: datetime | None, ended: datetime | None) -> int | None:
    """Whole milliseconds from one moment to a later one, half a millisecond rounding up; None when
    either is unknown or they are out of order."""
    if started is None or ended is None or ended < started:
        return None
    return ((ended - started) // timedelta(microseconds=1) + 500) // 1000


def _tool_times(
    events: Iterable[Mapping[str, Any]],
) -> dict[str, tuple[datetime | None, datetime | None]]:
    """When each tool call started and ended, as its Run's events say and nothing else.

    Both events carry the id the model gave the call, which is what ties them to it. A call whose
    events are more than one start or more than one end is left out: a tool run again after a
    crash starts twice, and nothing says which start the end belongs to. A call with one event of
    the two has the time of that event and none for the other.
    """
    seen: dict[str, dict[str, list[datetime]]] = defaultdict(lambda: defaultdict(list))
    for event in events:
        if event.get("call_id") and event.get("event_type") in ("tool_start", "tool_end"):
            seen[str(event["call_id"])][str(event["event_type"])].append(event["created_at"])
    return {
        call: (
            moments["tool_start"][0] if moments["tool_start"] else None,
            moments["tool_end"][0] if moments["tool_end"] else None,
        )
        for call, moments in seen.items()
        if len(moments["tool_start"]) <= 1 and len(moments["tool_end"]) <= 1
    }


def _call_id(item: Mapping[str, Any]) -> str | None:
    payload = item.get("payload")
    call = payload.get("call_id") if isinstance(payload, Mapping) else None
    return str(call) if call else None


def _time_tools(steps: Iterable[dict[str, Any]], events: Iterable[Mapping[str, Any]]) -> None:
    """Say when each tool result's tool started and ended, and how long between.

    A Child's tool calls end up in the same events as its parent's, told apart by nothing but the
    call id, so an id two results of the Run claim belongs to neither: both get no time. The
    interval is the one between the two events, which the database stamps as each is written; it
    is longer by their commits than the tool's own run.
    """
    results = [item for item in steps if item["type"] == "tool_result"]
    times = _tool_times(events)
    claims = Counter(_call_id(item) for item in results)
    for item in results:
        call = _call_id(item)
        started, ended = (
            times.get(call, (None, None)) if call and claims[call] == 1 else (None, None)
        )
        item["tool"] = {
            "started_at": _stamp(started),
            "ended_at": _stamp(ended),
            "duration_ms": _interval_ms(started, ended),
        }


def _child_finish(
    child: Mapping[str, Any], entries: Sequence[Mapping[str, Any]]
) -> datetime | None:
    """When a Child that has finished ended: when it wrote its last entry, or, if it wrote none,
    when its row last changed. None for a Child that never reached an end."""
    if child.get("status") not in _FINISHED:
        return None
    return entries[-1].get("timestamp") if entries else child.get("updated_at")


def _child_record(
    child: Mapping[str, Any],
    *,
    child_steps: Mapping[str, Sequence[Mapping[str, Any]]],
    steps: list[dict[str, Any]],
) -> dict[str, Any]:
    """One Child as a nested episode: it starts when its row is made and, if it ended, ends when
    it wrote its last entry."""
    started = child.get("created_at")
    finished = _child_finish(child, child_steps.get(str(child["child_session_id"]), []))
    return {
        "child_session_id": str(child["child_session_id"]),
        "parent_session_id": (
            str(child["parent_session_id"]) if child.get("parent_session_id") else None
        ),
        "parent_call_id": child.get("parent_call_id"),
        "parent_intent_id": (
            str(child["parent_intent_id"]) if child.get("parent_intent_id") else None
        ),
        "depth": child.get("depth"),
        "status": child.get("status"),
        "objective": child.get("objective"),
        "model_role": child.get("model_role"),
        "summary": _own_summary(child, child_steps),
        "context_mode": child.get("context_mode"),
        "usage": _json(child.get("usage_json")),
        "created_at": _stamp(started),
        "started_at": _stamp(started),
        "finished_at": _stamp(finished),
        "duration_seconds": (finished - started).total_seconds() if started and finished else None,
        "cancel_requested_at": _stamp(child.get("cancel_requested_at")),
        "steps": steps,
    }


def _chain_above(first: Mapping[str, Any] | None, parents: Mapping[str, str | None]) -> list[str]:
    """The entries above a Run's first, root first: the context the Run was given.

    Each entry names its parent, so the chain is found by following them from the first entry's
    parent up to the entry that has none.
    """
    above: list[str] = []
    cursor = str(first["parent_entry_id"]) if first and first.get("parent_entry_id") else None
    while cursor is not None:
        above.append(cursor)
        cursor = parents.get(cursor)
    above.reverse()
    return above


def episode(
    run: Mapping[str, Any],
    *,
    routing: Mapping[str, Any] | None,
    steps: Sequence[Mapping[str, Any]],
    session_id: str | None,
    carried: Sequence[Mapping[str, Any]],
    children: Sequence[Mapping[str, Any]],
    child_steps: Mapping[str, Sequence[Mapping[str, Any]]],
    events: Sequence[Mapping[str, Any]],
    evidence: Sequence[Mapping[str, Any]],
    notes: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """One Run as one training record: the episode and everything that grounds it.

    An action is an assistant message, an observation the tool result that answers its call, and
    the end is the last assistant message with the Run's status and stop reason. The reward is
    kept as the owner gave it, in `labels`, and never reduced to a number.
    """
    started, finished = run.get("started_at"), run.get("finished_at")
    duration = (finished - started).total_seconds() if started and finished else None
    own = [step(entry) for entry in steps]
    spans = {
        str(child["child_session_id"]): [
            step(entry) for entry in child_steps.get(str(child["child_session_id"]), [])
        ]
        for child in children
    }
    _time_tools([*own, *(item for items in spans.values() for item in items)], events)
    sources = _json(run.get("sources_json"))
    return {
        "schema_version": SCHEMA_VERSION,
        # Which side of a dataset the Run's Session fell on; raw mode splits nothing.
        "split": None,
        "run": {
            "run_id": str(run["run_id"]),
            "owner_id": run.get("owner_id"),
            "kind": run.get("run_kind"),
            "lane": run.get("lane"),
            "submitted_by": run.get("submitted_by"),
            "access_scope_kind": run.get("access_scope_kind"),
            "status": run.get("status"),
            "phase": run.get("phase"),
            "stop_reason": run.get("stop_reason"),
            "error_kind": run.get("error_kind"),
            "superseded_by_run_id": (
                str(run["superseded_by_run_id"]) if run.get("superseded_by_run_id") else None
            ),
            "request_fingerprint": run.get("request_fingerprint"),
            "retention_seconds": run.get("retention_seconds"),
            "purge_after": _stamp(run.get("purge_after")),
            "events_trimmed_at": _stamp(run.get("events_trimmed_at")),
            "created_at": _stamp(run.get("created_at")),
            "started_at": _stamp(started),
            "finished_at": _stamp(finished),
            "duration_seconds": duration,
        },
        "input": {
            "prepared": _json(run.get("prepared_input_json")),
            "accepted": _json(run.get("accepted_input_json")),
        },
        "routing": (
            None
            if routing is None
            else {
                "requested_mode": routing.get("requested_mode"),
                "valid_modes": list(routing.get("valid_modes") or []),
                "resolved_mode": routing.get("resolved_mode"),
                "model_fingerprints": _json(routing.get("model_fingerprints")),
                "context_policy_revision": routing.get("context_policy_revision"),
                "agent_lane_id": routing.get("agent_lane_id"),
                "source_lane_id": routing.get("source_lane_id"),
                "fork_point_entry_id": routing.get("fork_point_entry_id"),
            }
        ),
        # The conversation this Run continues lives in its session, exported as the Runs that wrote
        # it: `steps` are this Run's own entries, and each names its parent entry, which is how a
        # consumer walks back into the Runs before it. `carried` holds, whole, the entries of the
        # chain above the Run's first entry that no exported Run wrote, root first, because the Run
        # saw them; the rest of the chain is in the steps of the Runs that did. No list of the
        # chain is kept: it would repeat, in every Run, every entry above it.
        "session": {
            "id": session_id,
            "head_entry_id": own[-1]["entry_id"] if own else None,
            "carried": [step(entry) for entry in carried],
        },
        "steps": own,
        "children": [
            _child_record(
                child, child_steps=child_steps, steps=spans[str(child["child_session_id"])]
            )
            for child in children
        ],
        # A snapshot of the evidence ledger after one tool call. Each repeats everything admitted
        # so far, so it is told apart by its digest and kept without its text.
        "evidence": [
            {
                "ordinal": row.get("result_ordinal"),
                "content_digest": row.get("content_digest"),
                "locator": _json(row.get("locator")),
                "locator_digest": row.get("locator_digest"),
                "created_at": _stamp(row.get("created_at")),
            }
            for row in evidence
        ],
        "session_notes": [
            {
                "path": row.get("relative_path"),
                "revision": row.get("revision"),
                "size_bytes": row.get("size_bytes"),
                "content": _text(row.get("content")),
                "content_digest": row.get("content_digest"),
                "updated_at": _stamp(row.get("updated_at")),
            }
            for row in notes
        ],
        # Only what the answer's citations need: the sources it cites, with the passages cited.
        # The rest of the stored result repeats the answer and the whole evidence.
        "result": None if sources is None else {"sources": sources},
        # The owner's own judgements of this Run's answers, as they gave them: the vote, the
        # reasons, their note, and when they first gave it. `agent` is `main` or a Child's Session
        # id, and `sequence` the step it judges in that agent's own steps.
        "labels": [
            {
                "agent": str(row["agent"]),
                "sequence": row["sequence"],
                "value": row["value"],
                "reasons": list(row.get("reasons") or []),
                "note": row.get("note"),
                "given_at": _stamp(row.get("created_at")),
            }
            for row in labels
        ],
    }


def without_repeats(runs: Iterable[Any]) -> list[Any]:
    """Keep one Run per request fingerprint, the newest, so a retry cannot weight twice.

    It is applied to the chosen Runs before anything else about them is read, so the entries of a
    Run it drops are not read as the Run's own, and are carried by the Runs that followed it.
    """
    newest: dict[object, Any] = {}
    ordered: list[Any] = []
    for run in runs:
        fingerprint = run["request_fingerprint"]
        if fingerprint is None:
            ordered.append(run)
            continue
        previous = newest.get(fingerprint)
        if previous is None:
            newest[fingerprint] = run
            ordered.append(run)
            continue
        if run["created_at"] >= previous["created_at"]:
            ordered[ordered.index(previous)] = run
            newest[fingerprint] = run
    return ordered


def as_jsonl(records: Iterable[dict[str, Any]]) -> str:
    """One record per line, keys sorted, so two exports of one database compare equal."""
    return "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
    )


# ── the export ───────────────────────────────────────────────────────────────────────────


class Consent(NamedTuple):
    """Who contributes, as one export saw it: the owners whose switch is on, and the earliest
    moment any of their current permissions began."""

    owners: int
    earliest: datetime | None


class Exported(NamedTuple):
    records: list[dict[str, Any]]
    consent: Consent


def _written(entries: Iterable[Any]) -> dict[str, dict[str, list[Any]]]:
    """What each chosen Run wrote, by the Session it wrote it in and in the Session's own order:
    its conversation's Session for `steps`, and each Child's own Session for that Child's."""
    written: dict[str, dict[str, list[Any]]] = {}
    for entry in entries:
        by_session = written.setdefault(str(entry["run_id"]), {})
        by_session.setdefault(str(entry["session_id"]), []).append(entry)
    return written


def _chains(
    routing: Mapping[str, Any],
    written: Mapping[str, Mapping[str, Sequence[Any]]],
    parents: Mapping[str, str | None],
) -> dict[str, list[str]]:
    """For each chosen Run, the entries above its first in its conversation's Session."""
    chains: dict[str, list[str]] = {}
    for run_id, row in routing.items():
        own = written.get(run_id, {}).get(str(row["agent_session_id"]), [])
        chains[run_id] = _chain_above(own[0] if own else None, parents)
    return chains


async def export(
    settings: Mapping[str, Any],
    *,
    owner: str | None,
    since: str | None,
    until: str | None,
    run_ids: Sequence[str] | None,
    limit: int | None,
    dedupe: bool,
    dataset: bool = False,
) -> Exported:
    connection = await asyncpg.connect(**settings)
    try:
        await connection.execute("SET default_transaction_read_only = on")
        # Every read sees one snapshot, taken when the Runs are chosen. An owner who turns the
        # switch off, or keeps chatting, while the export runs changes nothing it writes: the
        # entries a chosen Run wrote are the ones that existed when it was chosen.
        async with connection.transaction(isolation="repeatable_read", readonly=True):
            runs = await connection.fetch(
                _RUNS_SQL, owner, since, until, list(run_ids) if run_ids else None, limit, dataset
            )
            if dedupe:
                runs = without_repeats(runs)
            contributors = await connection.fetchrow(_CONSENT_SQL, owner)
            consent = Consent(contributors["owners"], contributors["earliest"])
            if not runs:
                return Exported([], consent)
            ids = [str(run["run_id"]) for run in runs]
            owners = [str(run["owner_id"]) for run in runs]
            routing = {str(row["run_id"]): row for row in await connection.fetch(_ROUTING_SQL, ids)}
            children = _group_by_run(await connection.fetch(_CHILDREN_SQL, ids))
            events = _group_by_run(await connection.fetch(_TOOL_EVENTS_SQL, owners, ids))
            evidence = _group_by_run(await connection.fetch(_EVIDENCE_SQL, ids))
            notes = _group_by_run(await connection.fetch(_NOTES_SQL, ids), key="written_by_run_id")
            labels = _group_by_run(await connection.fetch(_LABELS_SQL, owners, ids))
            written = _written(await connection.fetch(_ENTRIES_SQL, ids))

            # The chain above each Run's first entry, from the shape of its whole Session; and the
            # entries on those chains that no chosen Run wrote, whose words are read as well.
            sessions = sorted(
                {(str(row["owner_id"]), row["agent_session_id"]) for row in routing.values()}
            )
            skeleton = await connection.fetch(
                _SKELETON_SQL, [owner_id for owner_id, _ in sessions], [s for _, s in sessions]
            )
            parents = {
                str(row["entry_id"]): str(row["parent_entry_id"])
                if row["parent_entry_id"]
                else None
                for row in skeleton
            }
            writers = {
                str(row["entry_id"]): str(row["run_id"]) for row in skeleton if row["run_id"]
            }
            chains = _chains(routing, written, parents)
            chosen = set(ids)
            beyond = sorted(
                {
                    entry
                    for chain in chains.values()
                    for entry in chain
                    if writers.get(entry) not in chosen
                }
            )
            carried = (
                {str(row["entry_id"]): row for row in await connection.fetch(_CARRIED_SQL, beyond)}
                if beyond
                else {}
            )
    finally:
        await connection.close()

    records = []
    for run in runs:
        run_id = str(run["run_id"])
        row = routing.get(run_id)
        session = (
            str(row["agent_session_id"]) if row is not None and row["agent_session_id"] else None
        )
        its_entries = written.get(run_id, {})
        chain = chains.get(run_id, [])
        records.append(
            episode(
                run,
                routing=row,
                steps=its_entries.get(session, []) if session else [],
                session_id=session,
                carried=[carried[entry] for entry in chain if entry in carried],
                children=children.get(run_id, []),
                child_steps=its_entries,
                events=events.get(run_id, []),
                evidence=evidence.get(run_id, []),
                notes=notes.get(run_id, []),
                labels=labels.get(run_id, []),
            )
        )
    return Exported(records, consent)


def _counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """What the summary line reads, for any selection; no one has contributed yet means none."""
    return {
        "runs": len(records),
        "steps": sum(len(record["steps"]) for record in records),
        "children": sum(len(record["children"]) for record in records),
        "evidence": sum(len(record["evidence"]) for record in records),
        "notes": sum(len(record["session_notes"]) for record in records),
        "labels": sum(len(record["labels"]) for record in records),
    }


def _group_by_run(rows: Iterable[Any], *, key: str = "run_id") -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = {}
    for row in rows:
        grouped.setdefault(str(row[key]), []).append(row)
    return grouped


# ── de-identifying a dataset ─────────────────────────────────────────────────────────────

_UUID_BODY = r"[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}"
_UUID = re.compile(rf"(?<![0-9A-Fa-f]){_UUID_BODY}(?![0-9A-Fa-f])")
#: What can be told from a word and found inside text: a UUID, or a SHA-256 digest in hex, which is
#: what an owner id is. A longer run of hex is neither, so neither is cut out of it.
_TOKEN = re.compile(rf"(?<![0-9A-Fa-f])(?:{_UUID_BODY}|[0-9A-Fa-f]{{64}})(?![0-9A-Fa-f])")


class Pseudonyms:
    """HMAC-SHA256 of an id under the operator's key, as hex.

    The same key gives an id the same pseudonym in every dataset, so versions link up, and no one
    without the key can compute a pseudonym or undo one.
    """

    def __init__(self, key: str) -> None:
        self._key = key.encode("utf-8")
        self._made: dict[str, str] = {}

    def __call__(self, raw: str) -> str:
        pseudonym = self._made.get(raw)
        if pseudonym is None:
            pseudonym = hmac.digest(self._key, raw.encode("utf-8"), "sha256").hex()
            self._made[raw] = pseudonym
        return pseudonym

    @property
    def fingerprint(self) -> str:
        """Eight hex digits that say two datasets share a key and tell nothing about it."""
        return hmac.digest(self._key, _KEY_FINGERPRINT_MESSAGE, "sha256").hex()[:8]


#: The keys of an entry's payload that hold an id the Runtime minted: for a tool call's intent and
#: attempt, a compaction's projection and the entries it names, a steer. They are UUIDs, and
#: only a UUID is replaced, so a lane that is `main` stays what it is.
_MINTED_IN_PAYLOADS = (
    "intent_id",
    "attempt_id",
    "projection_id",
    "control_id",
    "covered_through_entry_id",
    "first_retained_entry_id",
)


class Identities:
    """The ids of one export, and what replaces them wherever they stand in a record.

    Only ids the export has met are replaced. A UUID and a SHA-256 digest in hex, which is what an
    owner id is, cannot be a word, so they are replaced inside text and tool arguments too, where
    the agent was shown or gave one (a path that names an owner's skill directory has the owner
    id in it), while a UUID or digest the product did not issue (a page's, a file's) is left as
    written. An id that is some other string, such as an owner id in a test fixture, could be a
    word inside a sentence, so it is replaced only where a value is exactly it. Text is otherwise
    untouched: no pattern finds a name.
    """

    def __init__(self, pseudonyms: Pseudonyms) -> None:
        self._pseudonyms = pseudonyms
        self._whole: dict[str, str] = {}
        self._inside: dict[str, str] = {}

    def learn(self, record: Mapping[str, Any]) -> None:
        """Take in every id a record names, in the places the export itself writes ids."""
        run, session = record["run"], record["session"]
        ids = [
            run["owner_id"],
            run["submitted_by"],
            run["run_id"],
            run["superseded_by_run_id"],
            session["id"],
            session["head_entry_id"],
        ]
        minted: list[Any] = []
        for given in record["input"].values():
            if isinstance(given, Mapping):
                ids += [given.get("parent_run_id"), given.get("agent_session_id")]
                minted += [given.get("agent_lane_id"), given.get("source_lane_id")]
        if record["routing"]:
            ids.append(record["routing"]["fork_point_entry_id"])
            minted += [record["routing"]["agent_lane_id"], record["routing"]["source_lane_id"]]
        entries = [*record["steps"], *session["carried"]]
        for child in record["children"]:
            ids += [child["child_session_id"], child["parent_session_id"]]
            minted.append(child["parent_intent_id"])
            entries += child["steps"]
        for entry in entries:
            ids += [entry["entry_id"], entry["parent_entry_id"]]
            if isinstance(entry["payload"], Mapping):
                minted += [entry["payload"].get(name) for name in _MINTED_IN_PAYLOADS]
        for raw in ids:
            if isinstance(raw, str) and raw:
                self._take(raw)
        for raw in minted:
            if isinstance(raw, str) and _UUID.fullmatch(raw):
                self._take(raw)

    def _take(self, raw: str) -> None:
        findable = _TOKEN.fullmatch(raw) is not None
        canonical = raw.lower() if findable else raw
        self._whole[canonical] = self._pseudonyms(canonical)
        if findable:
            self._inside[canonical] = self._whole[canonical]

    def apply(self, value: Any) -> Any:
        """The value with every id this export knows replaced, in strings and in object keys."""
        if isinstance(value, str):
            return self._text(value)
        if isinstance(value, Mapping):
            return {
                (self._text(name) if isinstance(name, str) else name): self.apply(item)
                for name, item in value.items()
            }
        if isinstance(value, list | tuple):
            return [self.apply(item) for item in value]
        return value

    def _text(self, text: str) -> str:
        whole = self._whole.get(text)
        if whole is not None:
            return whole
        if len(text) < 36:
            return text
        return _TOKEN.sub(self._token, text)

    def _token(self, match: re.Match[str]) -> str:
        found = match.group(0)
        return self._inside.get(found.lower(), found)


# ── packaging a dataset ──────────────────────────────────────────────────────────────────

_SELECTION_RULE = (
    "Whole Agent Sessions of the owners whose Conversation Contribution is on when the export "
    "reads the database (ADR 0041, default deny), and only a Session that began at or after the "
    "moment that owner's current permission began. A Session begins at the earlier of its own "
    "row's creation and the acceptance of its first remaining Run. Every finished Run "
    "(succeeded, failed or cancelled) of a selected Session is in. --owner, --since and --until "
    "narrow by Session, by when it began, and never widen. A dataset is a full re-export, not an "
    "increment: an owner who turned the switch off is absent from the next one."
)

_EXCLUDED = (
    "Owners who have not turned Conversation Contribution on, or have turned it off.",
    "Sessions that began before their owner's current permission, however long they went on or "
    "were forked.",
    "Runs with no Session: a Retrieval or a Corpus Mutation is not a conversation.",
    "Runs still queued or running when the export read the database: they have no end, and their "
    "prepared input holds execution secrets.",
    "Runs that retention pruned and conversations their owner deleted: those Runs are gone. "
    "Entries they wrote that remain in a Session another exported Run routes to appear only in "
    "`session.carried` of a Run whose context chain passes through them, never as a record. "
    "Words of a deleted fork that no surviving Run's chain passes through appear nowhere.",
    "`provider_state`, the provider's replay blobs.",
    "Run events other than the start and end of each tool call, which are read only to time it.",
    "Error messages; `error_kind` stays.",
    "The Run's stored result but for `result.sources`, the sources its answer cites with the "
    "passages cited.",
    "The passage text of the evidence snapshots: each repeats the whole ledger so far. What the "
    "agent saw is in its tool results, and what the answer cites is in `result.sources`.",
    "A Child's summary when its own entries already say it.",
)

_LIMITS = (
    "Text is not de-identified. Messages, tool arguments and results, notes, citations and "
    "labels are as written, with whatever names, addresses and other personal data they hold, "
    "and with third parties' content: pages the agent fetched, files the owner attached, "
    "personal connections such as Notion, and Profile Memory the agent recalled.",
    "Only the product's own ids are replaced: the ids of the owners in the dataset (SHA-256 "
    "digests) and of submitted_by, Session, Run, Entry and Child Session, and the ids the Runtime "
    "mints for a tool call's intent and attempt, a compaction's projection, a steer and a fork's "
    "lane, wherever they stand, text and tool arguments included: an owner id inside a path is "
    "replaced like any other. An id of some other shape would be a word inside a sentence, so it "
    "is replaced only where a value is exactly it. Anyone holding the key and a stored id can "
    "compute its pseudonym. Other ids stay as stored: tool call ids the provider issued, "
    "resource handles, corpus document, chunk and source ids, digests, and any UUID the product "
    "did not issue.",
    "A conversation its owner deleted can still appear in the dataset as context. When another "
    "conversation was forked from it, the deleted conversation's messages ride along as "
    "`session.carried` in the fork's records, because the fork ran on them. A deleted parent and "
    "a retention-pruned one look the same in the database, so the export cannot carry the one "
    "and withhold the other.",
    "One owner can have many Sessions, and the split is by Session, so it does not isolate "
    "people: one person's style and knowledge can be in train and in eval.",
    "A REST caller's `history` and a recalled Profile Memory can copy text from one Session into "
    "another, so the split does not isolate text either.",
    "There are no preference pairs. The product does not regenerate an answer, so a Run has the "
    "owner's judgement of its answer, when it has one, and no counterpart to compare it with.",
    "Labels are the owner's raw judgements, never a score, and most answers have none "
    "(`labels.unlabelled_share`).",
    "`session.carried` repeats in every record whose context chain passes through the entries.",
    "Timestamps come from two clocks, the database's and the workers' (see `timing`).",
)

_TIMING = {
    "run": (
        "created_at is when the Run was accepted, started_at when a worker began it and "
        "finished_at when it ended, on the database's clock (the start of the transaction that "
        "wrote them); duration_seconds is finished_at minus started_at."
    ),
    "step.timestamp": (
        "When the worker that wrote the entry built it, immediately before committing it, on "
        "that worker's clock and not the database's. `sequence`, not the timestamp, is the order "
        "of a Session's entries."
    ),
    "step.payload.call": (
        "On an assistant message or a compaction, passed through as stored (ADR 0043). queued_ms "
        "is how long the model request waited for a slot before it was sent, and duration_ms how "
        "long the attempt that produced the result took, from sending the request to the end of "
        "the response. Both are whole milliseconds off the worker's monotonic clock. Absent on a "
        "step written before they existed."
    ),
    "step.tool": (
        "On a tool result. started_at and ended_at are the database's times of the Run's "
        "tool_start and tool_end events for the call, matched by the call id; duration_ms is the "
        "whole milliseconds between them, half a millisecond rounding up. It is the interval "
        "between two events and so includes the commit of each, which makes it a little longer "
        "than the tool's own run. A call that does not have exactly one start and at most one "
        "end among its Run's events, or whose id another result of the Run also carries, has "
        "none of the three, and so does every call of a Run whose events retention trimmed "
        "(run.events_trimmed_at). A call with only one of the two events has that time and no "
        "duration."
    ),
    "child": (
        "started_at is when the Child's row was made, the same as created_at. finished_at is the "
        "timestamp of its last entry, or, for a Child that wrote none, the last change of its "
        "row, and is null for a Child that has not reached succeeded, failed or cancelled. "
        "duration_seconds is finished_at minus started_at."
    ),
}


_SPLIT_RULE = (
    "A Session goes to eval when u < f and to train otherwise, where f is eval_fraction and u is "
    "the first eight bytes of SHA-256 over `dlightrag-split-v1`, a zero byte and the stored "
    "Session id (its canonical lowercase text), read as a big-endian integer and divided by "
    "2**64. Every Run of a Session, and its Children, share its side. There is no secret and no "
    "stored list, so a Session keeps its side however much is exported after it, and raising f "
    "only moves Sessions from train to eval."
)


def split_of(session_id: str, eval_fraction: float) -> str:
    """The side a Session falls on, by `_SPLIT_RULE`. The comparison is of the integer against
    f * 2**64, which is the same as u < f without rounding u."""
    digest = hashlib.sha256(_SPLIT_SALT + session_id.encode("utf-8")).digest()
    return "eval" if int.from_bytes(digest[:8], "big") < eval_fraction * 2**64 else "train"


def _label_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    labels = [label for record in records for label in record["labels"]]
    values = Counter(label["value"] for label in labels)
    without = sum(1 for record in records if not record["labels"])
    return {
        "count": len(labels),
        "by_value": {"down": values["down"], "up": values["up"]},
        "by_agent": {
            "child": sum(1 for label in labels if label["agent"] != "main"),
            "main": sum(1 for label in labels if label["agent"] == "main"),
        },
        "by_reason": dict(
            sorted(Counter(reason for label in labels for reason in label["reasons"]).items())
        ),
        "with_note": sum(1 for label in labels if label["note"]),
        "runs_without_labels": without,
        "unlabelled_share": without / len(records) if records else None,
    }


def _dataset_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def part(name: str) -> dict[str, int]:
        held = [record for record in records if record["split"] == name]
        return {"runs": len(held), "sessions": len({record["session"]["id"] for record in held})}

    return {
        "owners": len({record["run"]["owner_id"] for record in records}),
        "sessions": len({record["session"]["id"] for record in records}),
        "runs": len(records),
        "children": sum(len(record["children"]) for record in records),
        "steps": sum(len(record["steps"]) for record in records),
        "child_steps": sum(len(child["steps"]) for r in records for child in r["children"]),
        "carried_entries": sum(len(record["session"]["carried"]) for record in records),
        "evidence_snapshots": sum(len(record["evidence"]) for record in records),
        "session_notes": sum(len(record["session_notes"]) for record in records),
        "by_mode": dict(
            sorted(
                Counter(
                    (record["routing"] or {}).get("resolved_mode") or "unresolved"
                    for record in records
                ).items()
            )
        ),
        "by_status": dict(sorted(Counter(r["run"]["status"] for r in records).items())),
        "by_split": {name: part(name) for name in ("train", "eval")},
    }


class Dataset(NamedTuple):
    """The bytes of each file of a dataset, and the datasheet they are described by."""

    files: dict[str, bytes]
    datasheet: dict[str, Any]


def package(
    exported: Exported,
    *,
    key: str,
    eval_fraction: float,
    owner: str | None,
    since: str | None,
    until: str | None,
    commit: str | None,
    generated_at: datetime,
) -> Dataset:
    """Turn what an export read into a dataset: split by Session, every id pseudonymised, sorted
    by (Session, creation, Run), and described by a datasheet whose counts are of these records.
    """
    pseudonyms = Pseudonyms(key)
    identities = Identities(pseudonyms)
    for record in exported.records:
        identities.learn(record)
    ordered = []
    for record in exported.records:
        record["split"] = split_of(record["session"]["id"], eval_fraction)
        order = (
            pseudonyms(record["session"]["id"]),
            datetime.fromisoformat(record["run"]["created_at"]),
            pseudonyms(record["run"]["run_id"]),
        )
        ordered.append((order, identities.apply(record)))
    ordered.sort(key=lambda item: item[0])
    records = [record for _, record in ordered]

    files = {
        "train.jsonl": as_jsonl(r for r in records if r["split"] == "train").encode("utf-8"),
        "eval.jsonl": as_jsonl(r for r in records if r["split"] == "eval").encode("utf-8"),
    }
    created = [datetime.fromisoformat(record["run"]["created_at"]) for record in records]
    sheet = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at.isoformat(),
        "exporter": {"commit": commit},
        "key_fingerprint": pseudonyms.fingerprint,
        "selection": {
            "rule": _SELECTION_RULE,
            "owner": pseudonyms(owner) if owner else None,
            "since": since,
            "until": until,
        },
        "consent": {
            "adr": "0041",
            "contributing_owners": exported.consent.owners,
            "earliest_contribute_since": _stamp(exported.consent.earliest),
        },
        "time_window": {
            "first_run_created_at": min(created).isoformat() if created else None,
            "last_run_created_at": max(created).isoformat() if created else None,
        },
        "counts": _dataset_counts(records),
        "labels": _label_counts(records),
        "split": {"version": SPLIT_VERSION, "eval_fraction": eval_fraction, "rule": _SPLIT_RULE},
        "files": {
            name: {
                "bytes": len(data),
                "records": data.count(b"\n"),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            for name, data in files.items()
        },
        "timing": _TIMING,
        "excluded": list(_EXCLUDED),
        "limits": list(_LIMITS),
    }
    files["datasheet.json"] = (
        json.dumps(sheet, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    return Dataset(files, sheet)


def _write_private(path: Path, data: bytes, *, replace: bool = False) -> None:
    """Write a file only its owner can read, which never holds the data with wider rights.

    A file of a dataset is new, and an existing one is an error. The raw export's `--out` replaces
    what is there, as it always did, and is made private whether or not it existed.
    """
    flags = os.O_WRONLY | os.O_CREAT | (os.O_TRUNC if replace else os.O_EXCL)
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        os.fchmod(handle.fileno(), 0o600)
        handle.write(data)


def write_dataset(directory: Path, dataset: Dataset) -> None:
    """Write a dataset into a directory only its owner can enter, the datasheet last, so a
    directory without one holds a dataset that did not finish."""
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    for name in DATASET_FILES:
        _write_private(directory / name, dataset.files[name])


def _exporter_commit() -> str | None:
    """The commit this tool runs from, when git can say; the datasheet records which code made a
    dataset."""
    git = shutil.which("git")
    if git is None:
        return None
    try:
        done = subprocess.run(  # noqa: S603 - fixed argv, no shell, a host ops tool
            [git, "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except OSError, subprocess.SubprocessError:
        return None
    commit = done.stdout.strip()
    return (
        commit
        if done.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", commit)
        else None
    )


#: What each option a dataset cannot honour says when it is given with `--dataset`.
_DATASET_REFUSES = {
    "--out": "a dataset is written to its own directory",
    "--limit": "a dataset takes whole Sessions, and a limit would cut one",
    "--run-id": "a dataset takes whole Sessions, and naming Runs would cut one",
    "--dedupe": "a dataset takes whole Sessions, and dropping a retry would cut one",
}


def _export_raw(args: argparse.Namespace, settings: Mapping[str, Any]) -> int:
    try:
        exported = asyncio.run(
            export(
                settings,
                owner=args.owner,
                since=args.since,
                until=args.until,
                run_ids=args.run_ids,
                limit=1000 if args.limit is None else args.limit,
                dedupe=args.dedupe,
            )
        )
    except (OSError, asyncpg.PostgresError) as error:
        print(f"export-traces: cannot read {settings['database']}: {error}", file=sys.stderr)
        return 1

    payload = as_jsonl(exported.records)
    if args.out is None:
        sys.stdout.write(payload)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        _write_private(args.out, payload.encode("utf-8"), replace=True)
    where = "stdout" if args.out is None else str(args.out)
    counts = _counts(exported.records)
    print(
        f"export-traces: {counts['runs']} run(s), {counts['steps']} step(s), "
        f"{counts['children']} child session(s), {counts['evidence']} evidence row(s), "
        f"{counts['notes']} note(s), {counts['labels']} label(s) → {where}",
        file=sys.stderr,
    )
    return 0


def _export_dataset(
    args: argparse.Namespace, env: Mapping[str, str], settings: Mapping[str, Any]
) -> int:
    key = env.get(KEY_NAME, "")
    if len(key) < KEY_MIN_CHARS:
        print(
            f"export-traces: --dataset needs {KEY_NAME}, at least {KEY_MIN_CHARS} characters, in "
            "the repository .env or the environment. Make one with `openssl rand -base64 32` and "
            "keep it: the same key gives the same pseudonyms, so datasets made with it link up.",
            file=sys.stderr,
        )
        return 2
    directory: Path = args.dataset
    if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
        print(
            f"export-traces: {directory} exists and is not an empty directory; "
            "a dataset is written to a new one",
            file=sys.stderr,
        )
        return 2

    try:
        exported = asyncio.run(
            export(
                settings,
                owner=args.owner,
                since=args.since,
                until=args.until,
                run_ids=None,
                limit=None,
                dedupe=False,
                dataset=True,
            )
        )
    except (OSError, asyncpg.PostgresError) as error:
        print(f"export-traces: cannot read {settings['database']}: {error}", file=sys.stderr)
        return 1

    dataset = package(
        exported,
        key=key,
        eval_fraction=EVAL_FRACTION if args.eval_fraction is None else args.eval_fraction,
        owner=args.owner,
        since=args.since,
        until=args.until,
        commit=_exporter_commit(),
        generated_at=datetime.now(UTC).replace(microsecond=0),
    )
    try:
        write_dataset(directory, dataset)
    except OSError as error:
        print(f"export-traces: cannot write {directory}: {error}", file=sys.stderr)
        return 1
    counts = dataset.datasheet["counts"]
    print(
        f"export-traces: dataset of {counts['runs']} run(s) in {counts['sessions']} session(s) "
        f"(train {counts['by_split']['train']['runs']}, eval {counts['by_split']['eval']['runs']}), "
        f"{counts['steps']} step(s) → {directory}",
        file=sys.stderr,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=SUMMARY, suggest_on_error=True)
    parser.add_argument("--out", type=Path, help="write here instead of stdout")
    parser.add_argument(
        "--dataset",
        type=Path,
        metavar="DIR",
        help=(
            "write a de-identified, split dataset of whole Sessions (train.jsonl, eval.jsonl, "
            f"datasheet.json) into this new or empty directory, under the key {KEY_NAME}"
        ),
    )
    parser.add_argument(
        "--eval-fraction",
        type=float,
        help=f"with --dataset: the share of Sessions held out as eval (default {EVAL_FRACTION})",
    )
    parser.add_argument(
        "--since",
        help="only Runs created at or after this ISO timestamp "
        "(with --dataset: only Sessions that began at or after it)",
    )
    parser.add_argument(
        "--until",
        help="only Runs created before this ISO timestamp "
        "(with --dataset: only Sessions that began before it)",
    )
    parser.add_argument("--owner", help="only this owner's contributed Runs")
    parser.add_argument(
        "--run-id", action="append", dest="run_ids", help="only this Run if contributed; repeats"
    )
    parser.add_argument("--limit", type=int, help="at most this many Runs (default 1000)")
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="keep one record per request fingerprint, the newest",
    )
    args = parser.parse_args(argv)

    if args.dataset is None:
        if args.eval_fraction is not None:
            parser.error("--eval-fraction applies to --dataset only")
    else:
        used = {
            "--out": args.out is not None,
            "--limit": args.limit is not None,
            "--run-id": bool(args.run_ids),
            "--dedupe": args.dedupe,
        }
        for flag, reason in _DATASET_REFUSES.items():
            if used[flag]:
                parser.error(f"{flag} cannot be used with --dataset: {reason}")
        if args.eval_fraction is not None and not 0 <= args.eval_fraction <= 1:
            parser.error("--eval-fraction must be between 0 and 1")

    env = read_env(ROOT)
    settings = connection_settings(env)
    if args.dataset is not None:
        return _export_dataset(args, env, settings)
    return _export_raw(args, settings)


if __name__ == "__main__":
    sys.exit(main())
