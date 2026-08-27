# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Closed semantic Entry union contracts."""

from datetime import UTC, datetime
from typing import Any

import pytest

from dlightrag.engine.agent.session.effects import ToolResultEntry
from dlightrag.engine.agent.session.entries import (
    ENTRY_TYPE_TO_CLASS,
    AssistantMessageEntry,
    CompactionEntry,
    ControlMessageEntry,
    ToolResultMessageEntry,
    UserMessageEntry,
    decode_entry_payload,
)
from dlightrag.engine.agent.session.ids import (
    AttemptId,
    EntryId,
    IntentId,
    ProjectionId,
    SessionId,
)
from dlightrag.engine.ai.messages import ToolCall


def _common() -> dict[str, Any]:
    return {
        "entry_id": EntryId.new(),
        "session_id": SessionId.new(),
        "timestamp": datetime.now(UTC),
    }


def test_entry_union_contains_only_approved_semantic_variants() -> None:
    assert set(ENTRY_TYPE_TO_CLASS) == {
        "user_message",
        "assistant_message",
        "tool_result",
        "control_message",
        "compaction",
    }


def test_every_semantic_entry_round_trips_its_canonical_payload() -> None:
    session_id = SessionId.new()
    common = {
        "entry_id": EntryId.new(),
        "session_id": session_id,
        "sequence": 1,
        "timestamp": datetime.now(UTC),
    }
    intent_id = IntentId.new()
    variants = (
        UserMessageEntry(**common, content="question"),
        AssistantMessageEntry(
            **common,
            content="",
            stop_reason="tool_use",
            tool_calls=(ToolCall("c1", "lookup", {"value": "x"}),),
            acceptance_id="host-turn-1",
        ),
        ToolResultMessageEntry(
            **common,
            result=ToolResultEntry.text(
                tool_name="lookup", call_id="c1", outcome="succeeded", text="found"
            ),
            intent_id=intent_id,
            source_index=0,
            contract_version=2,
            input_schema_digest="a" * 64,
            replay_policy="never",
            attempt_id=AttemptId.new(),
            effective_input_digest="b" * 64,
        ),
        ControlMessageEntry(**common, control_id="s1", content="correct"),
        CompactionEntry(
            **common,
            projection_id=ProjectionId.new(),
            summary=None,
            covered_through_sequence=0,
            first_retained_sequence=1,
        ),
    )
    for entry in variants:
        decoded = decode_entry_payload(
            entry_type=entry.entry_type,
            entry_id=entry.entry_id,
            session_id=entry.session_id,
            sequence=entry.sequence,
            timestamp=entry.timestamp,
            payload=entry.canonical_payload(),
        )
        assert decoded == entry


def test_tool_result_requires_complete_executable_provenance() -> None:
    with pytest.raises(ValueError, match="provenance"):
        ToolResultMessageEntry(
            **_common(),
            result=ToolResultEntry.text(
                tool_name="lookup", call_id="c1", outcome="succeeded", text="found"
            ),
            intent_id=IntentId.new(),
            source_index=0,
            contract_version=1,
            input_schema_digest="",
            replay_policy="never",
            attempt_id=None,
            effective_input_digest="",
        )
