# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The trace export's rules that need no database: which retry stays, and what a Child says."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from scripts.export_traces import episode, without_repeats

START = datetime(2026, 4, 1, 10, 0, tzinfo=UTC)


def _entry(sequence: int, minutes: int, kind: str = "assistant_message") -> dict[str, Any]:
    return {
        "sequence": sequence,
        "entry_id": f"entry-{sequence}",
        "parent_entry_id": None,
        "entry_type": kind,
        "schema_version": 2,
        "timestamp": START + timedelta(minutes=minutes),
        "payload_json": '{"content": "step"}',
    }


def _run(run_id: str, *, minutes: int, fingerprint: str | None) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "owner_id": "owner",
        "run_kind": "answer",
        "lane": "main",
        "submitted_by": "owner",
        "access_scope_kind": "owner",
        "prepared_input_json": '{"q": "why"}',
        "accepted_input_json": None,
        "request_fingerprint": fingerprint,
        "status": "succeeded",
        "phase": "done",
        "stop_reason": None,
        "sources_json": None,
        "error_kind": None,
        "superseded_by_run_id": None,
        "retention_seconds": 31_536_000,
        "purge_after": START + timedelta(days=365),
        "events_trimmed_at": None,
        "created_at": START + timedelta(minutes=minutes),
        "started_at": START + timedelta(minutes=minutes),
        "finished_at": START + timedelta(minutes=minutes + 5),
    }


def test_a_retry_of_one_request_is_exported_once_its_newest() -> None:
    """A superseded attempt must not weight twice in a dataset."""
    first = _run("a", minutes=0, fingerprint="same")
    retry = _run("b", minutes=30, fingerprint="same")
    other = _run("c", minutes=40, fingerprint=None)

    kept = without_repeats([first, retry, other])

    assert [run["run_id"] for run in kept] == ["b", "c"]


def test_a_childs_summary_is_kept_only_when_no_entry_of_its_own_says_it() -> None:
    child = {
        "run_id": "a",
        "child_session_id": "child",
        "parent_session_id": "session",
        "parent_call_id": "call-1",
        "parent_intent_id": None,
        "depth": 1,
        "status": "succeeded",
        "objective": "Research the thing",
        "model_role": "query",
        "context_mode": "absolute",
        "summary": "step",
        "usage_json": '{"total_tokens": 9400}',
        "created_at": START,
        "updated_at": START,
        "cancel_requested_at": None,
    }

    record = episode(
        _run("a", minutes=0, fingerprint="fingerprint"),
        routing=None,
        steps=[_entry(1, 0, "user_message")],
        session_id="session",
        carried=[],
        children=[
            child,
            {**child, "child_session_id": "child-2", "summary": "a brief no entry carries"},
        ],
        child_steps={
            "child": [_entry(1, 2, "assistant_message")],
            "child-2": [_entry(1, 3, "assistant_message")],
        },
        events=[],
        evidence=[],
        notes=[],
        labels=[],
    )

    assert record["children"][0]["usage"] == {"total_tokens": 9400}
    assert record["children"][0]["summary"] is None, (
        "a summary its own entries already say is null, so no passage is carried twice"
    )
    assert record["children"][1]["summary"] == "a brief no entry carries", (
        "and a summary no entry holds is kept, because dropping it would lose what the child made"
    )
