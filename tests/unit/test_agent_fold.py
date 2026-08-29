# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Semantic Entry fold and complete exchange boundaries."""

from datetime import UTC, datetime

from dlightrag.engine.agent.session.effects import ToolResultEntry
from dlightrag.engine.agent.session.entries import (
    AssistantMessageEntry,
    CompactionEntry,
    ControlMessageEntry,
    ToolResultMessageEntry,
    UserMessageEntry,
)
from dlightrag.engine.agent.session.fold import (
    exchange_starts,
    fold_entries,
    host_turn_starts,
    project_session_messages,
    select_compaction_boundary,
)
from dlightrag.engine.agent.session.ids import EntryId, IntentId, ProjectionId, SessionId
from dlightrag.engine.agent.session.projection import ContextProjection, projection_source_digest
from dlightrag.engine.ai.messages import ToolCall


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


def test_fold_omits_incomplete_fast_host_users_from_authoritative_history() -> None:
    session_id = SessionId.new()
    succeeded = UserMessageEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        timestamp=_now(),
        content="successful question",
        acceptance_id="run-succeeded",
    )
    answer = AssistantMessageEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        timestamp=_now(),
        content="successful answer",
        stop_reason="stop",
        acceptance_id="run-succeeded",
    )
    failed = UserMessageEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        timestamp=_now(),
        content="failed question",
        acceptance_id="run-failed",
    )
    current = UserMessageEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        timestamp=_now(),
        content="current question",
        acceptance_id="run-current",
    )
    entries = (succeeded, answer, failed, current)

    assert [message["content"] for message in fold_entries(entries)] == [
        "successful question",
        "successful answer",
    ]
    assert [
        message["content"]
        for message in fold_entries(
            entries,
            included_incomplete_host_user_entry_id=current.entry_id,
        )
    ] == ["successful question", "successful answer", "current question"]


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


def test_direct_host_turns_are_complete_compaction_exchanges() -> None:
    session_id = SessionId.new()
    entries = (
        UserMessageEntry(
            entry_id=EntryId.new(), session_id=session_id, timestamp=_now(), content="old"
        ),
        AssistantMessageEntry(
            entry_id=EntryId.new(),
            session_id=session_id,
            timestamp=_now(),
            content="answer",
            stop_reason="stop",
        ),
        UserMessageEntry(
            entry_id=EntryId.new(), session_id=session_id, timestamp=_now(), content="current"
        ),
    )

    starts = host_turn_starts(entries)
    assert starts == (0, 2)
    assert select_compaction_boundary(entries, retained_tail_tokens=0, starts=starts) == 2


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
