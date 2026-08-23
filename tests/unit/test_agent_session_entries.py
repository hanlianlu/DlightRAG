# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the finite journal entry union and canonical typed ids."""

from datetime import UTC, datetime
from typing import Any

import pytest

from dlightrag.agent.session.effects import EffectIntent, ToolResultEntry
from dlightrag.agent.session.entries import (
    ENTRY_TYPE_TO_CLASS,
    SESSION_ENTRY_SCHEMA_VERSION,
    AssistantMessageEntry,
    CompactionEntry,
    EffectIntentEntry,
    EffectResultEntry,
    ProfileFactEntry,
    RunSegmentEntry,
    SessionTerminalEntry,
    UserMessageEntry,
    entry_type_of,
    new_session_entry,
)
from dlightrag.agent.session.ids import (
    EntryId,
    IntentId,
    ProjectionId,
    SessionId,
    StageIntentId,
    deterministic_uuid,
    new_uuid7,
)
from dlightrag.ai.messages import ToolCall


def _now() -> datetime:
    return datetime.now(UTC)


def _entry(**overrides: Any) -> dict[str, Any]:
    return dict(
        entry_id=EntryId.new(),
        session_id=SessionId.new(),
        timestamp=_now(),
        **overrides,
    )


class TestCanonicalIds:
    def test_framework_ids_are_uuidv7(self) -> None:
        assert new_uuid7()[14] == "7"

    def test_deterministic_ids_are_stable_and_namespaced(self) -> None:
        first = deterministic_uuid(seed="run-1", name="fast")
        second = deterministic_uuid(seed="run-1", name="fast")
        other = deterministic_uuid(seed="run-1", name="fast:planner:1")
        assert first == second
        assert first != other
        assert first != deterministic_uuid(seed="run-2", name="fast")

    def test_ids_reject_non_canonical_uuids(self) -> None:
        for id_class in (SessionId, EntryId, IntentId, ProjectionId, StageIntentId):
            with pytest.raises(ValueError):
                id_class("not-a-uuid")
            with pytest.raises(ValueError):
                id_class("A0EEBC99-9C0B-4EF8-BB6D-6BB9BD380A11")  # uppercase form


class TestEntryUnion:
    def test_union_is_finite_and_matches_agent_session_variants(self) -> None:
        assert set(ENTRY_TYPE_TO_CLASS) == {
            "run_segment",
            "user_message",
            "assistant_message",
            "effect_intent",
            "effect_result",
            "context_injection",
            "compaction",
            "profile_fact",
            "session_terminal",
        }

    def test_unknown_entry_type_raises_before_store_use(self) -> None:
        with pytest.raises(ValueError):
            new_session_entry(entry_type="workspace", session_id=SessionId.new(), **{})

    def test_run_segment_records_selected_parent_head(self) -> None:
        entry = RunSegmentEntry(
            **_entry(),
            segment_id=EntryId.new().value,
            kind="resume",
            parent_head_id=EntryId.new().value,
        )
        assert entry.canonical_payload()["kind"] == "resume"
        assert entry.canonical_payload()["parent_head_id"] is not None

    def test_user_message_canonical_payload(self) -> None:
        entry = UserMessageEntry(**_entry(), content="what is the state?")
        assert entry.canonical_payload() == {"content": "what is the state?"}
        assert entry.to_canonical_json()["payload"]["content"] == "what is the state?"

    def test_assistant_entry_requires_calls_for_tool_use(self) -> None:
        with pytest.raises(ValueError):
            AssistantMessageEntry(**_entry(), content="done", stop_reason="tool_use")

    def test_assistant_entry_serializes_provider_neutral_calls(self) -> None:
        entry = AssistantMessageEntry(
            **_entry(),
            content="searching",
            stop_reason="tool_use",
            tool_calls=(ToolCall(id="c1", name="search_knowledge_base", arguments={"q": "x"}),),
            usage={"input_tokens": 10, "output_tokens": 4},
        )
        payload = entry.canonical_payload()
        assert payload["stop_reason"] == "tool_use"
        assert payload["tool_calls"][0]["name"] == "search_knowledge_base"
        assert payload["usage"] == {"input_tokens": 10, "output_tokens": 4}

    def test_effect_intent_entry_carries_all_replay_facts(self) -> None:
        intent = EffectIntent(
            intent_id=IntentId.new(),
            tool_name="search_knowledge_base",
            replay_policy="safe",
            contract_version=3,
            input_schema_digest="a" * 64,
            canonical_input='{"q":"x"}',
            source_call_id="c1",
        )
        entry = EffectIntentEntry(**_entry(), intent=intent)
        payload = entry.canonical_payload()
        assert payload["tool_name"] == "search_knowledge_base"
        assert payload["replay_policy"] == "safe"
        assert payload["contract_version"] == 3
        assert payload["input_schema_digest"] == "a" * 64

    def test_compaction_entry_requires_covered_prefix_before_retained_start(self) -> None:
        with pytest.raises(ValueError):
            CompactionEntry(
                **_entry(),
                projection_id=ProjectionId.new(),
                summary='{"goal":"g"}',
                covered_through_sequence=5,
                first_retained_sequence=5,
            )

    def test_profile_fact_and_terminal_entries_validate(self) -> None:
        fact = ProfileFactEntry(**_entry(), key="objective", value="answer the question")
        assert fact.canonical_payload() == {
            "key": "objective",
            "value": "answer the question",
        }
        terminal = SessionTerminalEntry(**_entry(), reason="cancelled")
        assert entry_type_of(terminal) == "session_terminal"

    def test_effect_result_outcome_is_deterministic(self) -> None:
        result = ToolResultEntry(
            tool_name="search_web",
            call_id="c2",
            outcome="unknown_tool",
            content='Tool "search_web" is not available.',
        )
        entry = EffectResultEntry(**_entry(), result=result)
        assert entry.canonical_payload()["outcome"] == "unknown_tool"

    def test_schema_version_is_current(self) -> None:
        assert SESSION_ENTRY_SCHEMA_VERSION == 1
        with pytest.raises(ValueError):
            UserMessageEntry(**_entry(), content="x", schema_version=99)
