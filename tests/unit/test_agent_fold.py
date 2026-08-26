# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Semantic Entry fold and complete exchange boundaries."""

from datetime import UTC, datetime

from dlightrag.agent.session.effects import ToolResultEntry
from dlightrag.agent.session.entries import (
    AssistantMessageEntry,
    CompactionEntry,
    ControlMessageEntry,
    ToolResultMessageEntry,
    UserMessageEntry,
)
from dlightrag.agent.session.fold import (
    exchange_starts,
    fold_entries,
    project_session_messages,
    select_compaction_boundary,
)
from dlightrag.agent.session.ids import EntryId, IntentId, ProjectionId, SessionId
from dlightrag.agent.session.projection import ContextProjection, projection_source_digest
from dlightrag.ai.messages import ToolCall


def _now():
    return datetime.now(UTC)


def _result(session_id: SessionId, call_id: str, source: int) -> ToolResultMessageEntry:
    return ToolResultMessageEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        timestamp=_now(),
        result=ToolResultEntry.text(
            tool_name="lookup", call_id=call_id, outcome="succeeded", text=f"result {source}"
        ),
        intent_id=IntentId.new(),
        source_index=source,
        contract_version=1,
        input_schema_digest="a" * 64,
        replay_policy="never",
        attempt_id=None,
        effective_input_digest="b" * 64,
    )


def test_fold_projects_only_conversation_semantics_in_source_order() -> None:
    session_id = SessionId.new()
    entries = (
        UserMessageEntry(
            entry_id=EntryId.new(), session_id=session_id, timestamp=_now(), content="question"
        ),
        AssistantMessageEntry(
            entry_id=EntryId.new(),
            session_id=session_id,
            timestamp=_now(),
            content="",
            stop_reason="tool_use",
            tool_calls=(ToolCall("c1", "lookup", {"value": "x"}),),
        ),
        _result(session_id, "c1", 0),
        ControlMessageEntry(
            entry_id=EntryId.new(),
            session_id=session_id,
            timestamp=_now(),
            control_id="s1",
            content="correct",
        ),
        CompactionEntry(
            entry_id=EntryId.new(),
            session_id=session_id,
            timestamp=_now(),
            projection_id=ProjectionId.new(),
            summary=None,
            covered_through_sequence=0,
            first_retained_sequence=1,
        ),
    )
    messages = fold_entries(entries)
    assert [message["role"] for message in messages] == [
        "user",
        "assistant",
        "tool",
        "user",
    ]
    assert messages[2]["tool_call_id"] == "c1"
    assert messages[3]["content"] == "correct"


def test_exchange_boundaries_never_split_assistant_from_ordered_results() -> None:
    session_id = SessionId.new()
    entries = (
        UserMessageEntry(
            entry_id=EntryId.new(), session_id=session_id, timestamp=_now(), content="question"
        ),
        AssistantMessageEntry(
            entry_id=EntryId.new(),
            session_id=session_id,
            timestamp=_now(),
            content="",
            stop_reason="tool_use",
            tool_calls=(
                ToolCall("c1", "lookup", {"value": "x"}),
                ToolCall("c2", "lookup", {"value": "y"}),
            ),
        ),
        _result(session_id, "c1", 0),
        _result(session_id, "c2", 1),
    )
    assert exchange_starts(entries) == (1,)
    assert select_compaction_boundary(entries, retained_tail_tokens=0) == 1


def test_projection_is_bound_to_physical_branch_entry_identity() -> None:
    session_id = SessionId.new()
    user = UserMessageEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        timestamp=_now(),
        sequence=1,
        content="old",
    )
    assistant = AssistantMessageEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        timestamp=_now(),
        sequence=2,
        content="new",
        stop_reason="stop",
    )
    projection = ContextProjection(
        projection_id=ProjectionId.new(),
        covered_through_sequence=1,
        first_retained_sequence=2,
        covered_through_entry_id=user.entry_id,
        first_retained_entry_id=assistant.entry_id,
        source_digest=projection_source_digest([user.entry_id]),
        summary='{"goal":"summary"}',
    )
    messages = project_session_messages((user, assistant), projection)
    assert messages[-1]["content"] == "new"
