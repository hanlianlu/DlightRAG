# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the pure session fold: entries to model context, byte-for-byte."""

from datetime import UTC, datetime
from typing import Any

import pytest
from dlightrag_agent.session.effects import EffectIntent, ToolResultEntry
from dlightrag_agent.session.entries import (
    AssistantMessageEntry,
    CompactionEntry,
    ContextInjectionEntry,
    EffectIntentEntry,
    EffectResultEntry,
    ProfileFactEntry,
    SessionTerminalEntry,
    UserMessageEntry,
)
from dlightrag_agent.session.fold import (
    SessionEpisode,
    exchange_starts,
    fold_entries,
    head_tail_text,
    select_compaction_boundary,
)
from dlightrag_agent.session.ids import EntryId, IntentId, ProjectionId, SessionId
from dlightrag_ai.messages import ToolCall


def _now() -> datetime:
    return datetime.now(UTC)


def _assistant(
    session_id: SessionId,
    *,
    content: str,
    calls: tuple[ToolCall, ...] = (),
    stop_reason: str = "stop",
) -> AssistantMessageEntry:
    return AssistantMessageEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        timestamp=_now(),
        content=content,
        stop_reason=stop_reason,  # type: ignore[arg-type]
        tool_calls=calls,
    )


def _result(
    session_id: SessionId,
    *,
    intent_id: IntentId | None,
    call_id: str,
    content: str,
    tool_name: str = "search_knowledge_base",
) -> EffectResultEntry:
    return EffectResultEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        timestamp=_now(),
        intent_id=intent_id,
        result=ToolResultEntry(
            tool_name=tool_name, call_id=call_id, outcome="succeeded", content=content
        ),
    )


def test_replay_fold_equals_live_fold_byte_for_byte() -> None:
    session_id = SessionId.new()
    call = ToolCall(id="c1", name="search_knowledge_base", arguments={"q": "x"})
    entries = [
        UserMessageEntry(
            entry_id=EntryId.new(), session_id=session_id, timestamp=_now(), content="question"
        ),
        _assistant(session_id, content="searching", calls=(call,), stop_reason="tool_use"),
        _result(session_id, intent_id=IntentId.new(), call_id="c1", content="found"),
        ContextInjectionEntry(
            entry_id=EntryId.new(),
            session_id=session_id,
            timestamp=_now(),
            content="control instruction",
        ),
        ProfileFactEntry(
            entry_id=EntryId.new(),
            session_id=session_id,
            timestamp=_now(),
            key="profile",
            value={"window": 1000},
        ),
    ]
    live = fold_entries(entries)
    replayed = fold_entries(tuple(entries))
    assert live == replayed
    assert [message["role"] for message in live] == ["user", "assistant", "tool", "user"]


def test_fold_matches_executor_message_shapes() -> None:
    session_id = SessionId.new()
    call = ToolCall(id="c1", name="search_knowledge_base", arguments={"q": "x"})
    assistant = _assistant(session_id, content="searching", calls=(call,), stop_reason="tool_use")
    result = _result(session_id, intent_id=IntentId.new(), call_id="c1", content="found")
    messages = fold_entries([assistant, result])

    # Same shapes the ToolTurnExecutor emits for a live turn.
    assert messages[0] == {
        "role": "assistant",
        "content": "searching",
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {
                    "name": "search_knowledge_base",
                    "arguments": '{"q":"x"}',
                },
            }
        ],
    }
    assert messages[1] == {
        "role": "tool",
        "tool_call_id": "c1",
        "name": "search_knowledge_base",
        "content": "found",
        "is_error": False,
    }


def test_error_results_fold_with_is_error() -> None:
    session_id = SessionId.new()
    result = EffectResultEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        timestamp=_now(),
        intent_id=None,
        result=ToolResultEntry(
            tool_name="search_web", call_id="c2", outcome="failed", content="boom"
        ),
    )
    (message,) = fold_entries([result])
    assert message["is_error"] is True


def test_compaction_entry_renders_deterministically_at_its_position() -> None:
    from dlightrag_agent.session.projection import CompactionSummary

    session_id = SessionId.new()
    summary = CompactionSummary(goal="answer q", progress="found sources").canonical_json()
    entries = [
        UserMessageEntry(
            entry_id=EntryId.new(), session_id=session_id, timestamp=_now(), content="q1"
        ),
        CompactionEntry(
            entry_id=EntryId.new(),
            session_id=session_id,
            timestamp=_now(),
            projection_id=ProjectionId.new(),
            summary=summary,
            covered_through_sequence=1,
            first_retained_sequence=2,
        ),
        UserMessageEntry(
            entry_id=EntryId.new(), session_id=session_id, timestamp=_now(), content="q2"
        ),
    ]
    messages = fold_entries(entries)
    assert messages[1]["role"] == "user"
    assert "answer q" in messages[1]["content"]
    assert fold_entries(entries) == fold_entries(tuple(entries))


def test_accounting_entries_produce_no_model_messages() -> None:
    session_id = SessionId.new()
    intent = EffectIntentEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        timestamp=_now(),
        intent=EffectIntent(
            intent_id=IntentId.new(),
            tool_name="search_knowledge_base",
            replay_policy="safe",
            contract_version=1,
            input_schema_digest="a" * 64,
            canonical_input='{"q":"x"}',
            source_call_id="c1",
        ),
    )
    terminal = SessionTerminalEntry(
        entry_id=EntryId.new(), session_id=session_id, timestamp=_now(), reason="completed"
    )
    assert fold_entries([intent, terminal]) == []


def test_exchange_starts_never_split_calls_from_results() -> None:
    session_id = SessionId.new()
    call_a = ToolCall(id="a", name="search_knowledge_base", arguments={"q": "1"})
    call_b = ToolCall(id="b", name="search_knowledge_base", arguments={"q": "2"})
    entries = [
        UserMessageEntry(
            entry_id=EntryId.new(), session_id=session_id, timestamp=_now(), content="q"
        ),
        _assistant(session_id, content="a+b", calls=(call_a, call_b), stop_reason="tool_use"),
        _result(session_id, intent_id=IntentId.new(), call_id="a", content="ra"),
        _result(session_id, intent_id=IntentId.new(), call_id="b", content="rb"),
        _assistant(session_id, content="done", stop_reason="stop"),
        UserMessageEntry(
            entry_id=EntryId.new(), session_id=session_id, timestamp=_now(), content="next"
        ),
    ]
    assert exchange_starts(entries) == (1,)


def test_compaction_boundary_keeps_whole_exchanges() -> None:
    session_id = SessionId.new()
    call = ToolCall(id="c", name="search_knowledge_base", arguments={"q": "x"})
    entries = [
        UserMessageEntry(
            entry_id=EntryId.new(), session_id=session_id, timestamp=_now(), content="start"
        ),
        _assistant(session_id, content="one", calls=(call,), stop_reason="tool_use"),
        _result(session_id, intent_id=IntentId.new(), call_id="c", content="r1"),
        _assistant(session_id, content="two", calls=(call,), stop_reason="tool_use"),
        _result(session_id, intent_id=IntentId.new(), call_id="c", content="r2"),
        _assistant(session_id, content="three", stop_reason="stop"),
    ]
    starts = exchange_starts(entries)
    assert starts == (1, 3)
    # A tiny budget keeps only the newest complete exchange; the assistant call
    # at index 3 is never split from its result at index 4.
    boundary = select_compaction_boundary(entries, retained_tail_tokens=4)
    assert boundary in starts
    assert boundary == 3

    # A huge budget keeps everything from the first entry.
    assert select_compaction_boundary(entries, retained_tail_tokens=10_000) == 1


def test_head_tail_text_bounds_oversized_bodies() -> None:
    body = "word " * 500
    bounded = head_tail_text(body, head_tokens=20, tail_tokens=20)
    assert bounded != body
    assert "…" in bounded
    assert bounded.startswith("word ")
    assert bounded.endswith("word ")
    # Small bodies pass through untouched.
    short = "hello"
    assert head_tail_text(short, head_tokens=20, tail_tokens=20) == short


class TestSessionEpisode:
    def test_replays_recent_tail_verbatim_and_trims_older_reasoning(self) -> None:
        episode = SessionEpisode(retained_tail_tokens=1)
        episode.record(
            [
                {"role": "assistant", "content": "old", "provider_state": {"reason": 1}},
                {
                    "role": "tool",
                    "tool_call_id": "c1",
                    "content": "old result",
                },
            ]
        )
        episode.record(
            [
                {"role": "assistant", "content": "new", "provider_state": {"reason": 2}},
            ]
        )
        messages = episode.messages()
        assert messages[-1]["content"] == "new"
        assert messages[-1]["provider_state"] == {"reason": 2}
        assert messages[0]["content"] == "old"
        assert "provider_state" not in messages[0]

    def test_canonical_round_trip_preserves_exchanges(self) -> None:
        episode = SessionEpisode(retained_tail_tokens=2000)
        episode.record([{"role": "assistant", "content": "a"}])
        state = episode.canonical_json()
        state["exchanges"][0][0]["content"] = "mutated"  # callers may adapt copies
        rebuilt = SessionEpisode.from_canonical_json(state, retained_tail_tokens=2000)
        assert rebuilt.messages()[0]["content"] == "mutated"

    def test_rejects_non_sequence_state(self) -> None:
        with pytest.raises(ValueError):
            SessionEpisode.from_canonical_json(
                {"exchanges": "not-a-sequence"}, retained_tail_tokens=10
            )


class TestSessionEpisodeReplayBudget:
    def test_a_short_run_replays_every_exchange_in_full(self) -> None:
        episode = SessionEpisode(retained_tail_tokens=20_000)
        episode.record(_exchange("first", reasoning="short"))
        episode.record(_exchange("second", reasoning="short"))

        assistants = [m for m in episode.messages() if m["role"] == "assistant"]

        assert [m["tool_calls"][0]["id"] for m in assistants] == ["first", "second"]
        assert all("provider_state" in m for m in assistants)

    def test_exchange_past_budget_keeps_call_without_reasoning(self) -> None:
        episode = SessionEpisode(retained_tail_tokens=20_000)
        episode.record(_exchange("first", reasoning="short"))
        episode.record(_exchange("second", reasoning="n" * 200_000))

        older, newer = (m for m in episode.messages() if m["role"] == "assistant")

        assert "provider_state" not in older
        assert "thought_signature" not in older["tool_calls"][0]
        assert older["tool_calls"][0]["id"] == "first"
        assert newer["provider_state"]
        assert newer["tool_calls"][0]["thought_signature"] == "signed"


def _exchange(call_id: str, *, reasoning: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": "search_web", "arguments": '{"query":"x"}'},
                    "thought_signature": "signed",
                }
            ],
            "provider_state": {"reasoning_content": reasoning},
        },
        {
            "role": "tool",
            "tool_call_id": call_id,
            "name": "search_web",
            "content": "Open web added 1 new passages.",
        },
    ]
