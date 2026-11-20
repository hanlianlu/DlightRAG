# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Bounded structured Tool facts: what a step card states without reading a result.

The facts layer is deliberately not text and not a raw payload: a Tool reports a few named
values a reader's card shows — a search's provider and sources, a read's line count — and the
bounds belong to the field, so one producer's malformed fact costs that key and not the Run.
"""

import datetime
import json
from dataclasses import replace
from typing import Any

import pytest

from dlightrag.engine.agent.session.effects import ToolResultEntry
from dlightrag.engine.agent.session.entries import (
    ToolResultMessageEntry,
    decode_entry_payload,
)
from dlightrag.engine.agent.session.ids import EntryId, IntentId, SessionId
from dlightrag.engine.agent.session.operation import ToolBatchItem
from dlightrag.engine.agent.session.runtime import _settlement_facts
from dlightrag.engine.agent.tool_content import ToolTextPart
from dlightrag.engine.agent.tools.contracts import (
    TOOL_FACTS_MAX_BYTES,
    TOOL_FACTS_MAX_CHARS,
    TOOL_FACTS_MAX_ITEMS,
    TOOL_FACTS_MAX_KEYS,
    ToolResult,
    bounded_tool_facts,
)

_NOW = datetime.datetime.now(datetime.UTC)


def _encoded_size(facts: dict[str, Any]) -> int:
    return len(
        json.dumps(facts, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    )


def _entry(
    facts: dict[str, Any] | None = None, *, tool_name: str = "read"
) -> ToolResultMessageEntry:
    """One settled Tool result entry, the shape the session Runtime writes."""
    return ToolResultMessageEntry(
        entry_id=EntryId.new(),
        session_id=SessionId.new(),
        timestamp=_NOW,
        result=ToolResultEntry(
            tool_name=tool_name,
            call_id="call-1",
            outcome="succeeded",
            parts=(ToolTextPart("result text"),),
            facts=facts,
        ),
        intent_id=None,
        source_index=0,
        contract_version=0,
        input_schema_digest="",
        replay_policy="never",
        attempt_id=None,
        effective_input_digest="",
    )


def _batch_item() -> ToolBatchItem:
    return ToolBatchItem(
        source_index=2,
        call_id="call-3",
        tool_name="read",
        disposition="executable",
        result_entry_id=EntryId.new(),
        intent_id=IntentId.new(),
        contract_version=1,
        input_schema_digest="a" * 64,
        effective_input_digest="b" * 64,
    )


def test_scalars_lists_and_one_record_level_survive_normalization() -> None:
    """A fact map is a small structure: a count, a link, and a list of source records."""
    facts = {
        "provider": "exa",
        "hits": 3,
        "degraded": False,
        "cost": 0.0123,
        "nothing": None,
        "sources": [
            {"url": "https://example.org/a", "domain": "example.org", "published_at": "2026-01-02"},
        ],
        "tags": ["one", "two"],
    }

    assert bounded_tool_facts(facts) == facts


def test_values_that_break_a_bound_drop_their_key() -> None:
    """One bad value costs its own key; the rest of the card still renders."""
    assert bounded_tool_facts(
        {
            "long": "x" * (TOOL_FACTS_MAX_CHARS + 1),
            "opaque": object(),
            "set": {1, 2},
            "nan": float("nan"),
            "infinity": float("inf"),
            "deep": {"one": {"two": {"three": 1}}},
            "short": "kept",
        }
    ) == {"short": "kept"}


def test_a_list_longer_than_its_bound_drops_its_key() -> None:
    assert bounded_tool_facts({"items": list(range(TOOL_FACTS_MAX_ITEMS + 1)), "kept": True}) == {
        "kept": True
    }


def test_a_non_string_key_is_not_a_fact() -> None:
    assert bounded_tool_facts({1: "one", "two": 2}) == {"two": 2}  # pyright: ignore[reportArgumentType]


def test_the_key_count_is_bounded() -> None:
    facts = {f"key{index}": index for index in range(TOOL_FACTS_MAX_KEYS + 4)}

    bounded = bounded_tool_facts(facts)

    assert bounded is not None
    assert len(bounded) == TOOL_FACTS_MAX_KEYS
    assert "key0" in bounded  # insertion order is the producer's priority


def test_an_empty_or_invalid_map_is_no_facts_at_all() -> None:
    assert bounded_tool_facts(None) is None
    assert bounded_tool_facts({}) is None
    assert bounded_tool_facts({"gone": object()}) is None


def test_the_byte_bound_trims_a_list_before_dropping_a_small_fact() -> None:
    """A step card would rather show the sources that fit than lose the provider beside them."""
    sources = [
        {"url": "https://example.org/" + f"{index:02d}" + "a" * 190, "title": "t" * 100}
        for index in range(10)
    ]

    bounded = bounded_tool_facts({"provider": "exa", "hits": 10, "sources": sources})

    assert bounded is not None
    assert bounded["provider"] == "exa"
    assert bounded["hits"] == 10
    assert isinstance(bounded["sources"], list)
    assert 0 < len(bounded["sources"]) < len(sources)
    assert _encoded_size(bounded) <= TOOL_FACTS_MAX_BYTES


def test_the_byte_bound_drops_the_largest_key_when_no_list_can_be_trimmed() -> None:
    facts = {f"key{index}": "x" * TOOL_FACTS_MAX_CHARS for index in range(TOOL_FACTS_MAX_KEYS)}

    bounded = bounded_tool_facts(facts)

    assert bounded is not None
    assert 0 < len(bounded) < len(facts)
    assert _encoded_size(bounded) <= TOOL_FACTS_MAX_BYTES


def test_a_tool_result_normalizes_the_facts_it_reports() -> None:
    assert ToolResult.text("done", facts={"kept": "yes", "dropped": object()}).facts == {
        "kept": "yes"
    }
    assert ToolResult.text("done", facts={"dropped": object()}).facts is None
    assert ToolResult.text("done").facts is None


def test_a_settled_entry_round_trips_its_facts_and_still_drops_details() -> None:
    """The Entry keeps the product's structure and not the Runtime's private details."""
    entry = _entry({"provider": "exa", "hits": 2})
    entry = replace(
        entry,
        result=replace(entry.result, details={"private": True}),
    )

    payload = entry.canonical_payload()

    assert payload["facts"] == {"provider": "exa", "hits": 2}
    assert "details" not in payload
    decoded = decode_entry_payload(
        entry_type="tool_result",
        entry_id=entry.entry_id,
        session_id=entry.session_id,
        sequence=1,
        timestamp=entry.timestamp,
        payload=payload,
    )
    assert isinstance(decoded, ToolResultMessageEntry)
    assert decoded.result.facts == {"provider": "exa", "hits": 2}
    assert decoded.result.details is None


def test_an_entry_without_facts_stores_no_facts_key_and_decodes_without_them() -> None:
    """A call no Tool reported facts for stores none, and reads back with none."""
    entry = _entry(tool_name="bash")
    payload = entry.canonical_payload()

    assert "facts" not in payload
    decoded = decode_entry_payload(
        entry_type="tool_result",
        entry_id=entry.entry_id,
        session_id=entry.session_id,
        sequence=1,
        timestamp=entry.timestamp,
        payload=payload,
    )
    assert isinstance(decoded, ToolResultMessageEntry)
    assert decoded.result.facts is None


def test_the_settlement_event_publishes_the_durable_facts() -> None:
    """The SSE frame and the transcript must state the same structure the Entry keeps."""
    result = ToolResultEntry(
        tool_name="read",
        call_id="call-3",
        outcome="succeeded",
        parts=(ToolTextPart("..."),),
        facts={"path": "notes/ledger.md", "lines_read": 40, "total_lines": 40, "truncated": False},
    )

    facts = _settlement_facts(_batch_item(), result, 120)

    assert facts["facts"] == {
        "path": "notes/ledger.md",
        "lines_read": 40,
        "total_lines": 40,
        "truncated": False,
    }
    assert facts["duration_ms"] == 120
    assert facts["tool_name"] == "read"


def test_a_settlement_without_facts_publishes_none() -> None:
    result = ToolResultEntry(
        tool_name="bash",
        call_id="call-4",
        outcome="succeeded",
        parts=(ToolTextPart("ok"),),
    )

    assert "facts" not in _settlement_facts(_batch_item(), result, None)


def test_replacing_private_details_keeps_the_facts() -> None:
    result = ToolResultEntry(
        tool_name="read",
        call_id="call-5",
        outcome="succeeded",
        parts=(ToolTextPart("..."),),
        details={"private": True},
        facts={"path": "a.md"},
    )

    durable = replace(result, details=None)

    assert durable.details is None
    assert durable.facts == {"path": "a.md"}


@pytest.mark.parametrize("limit", [1, TOOL_FACTS_MAX_ITEMS])
def test_list_bounds_are_inclusive(limit: int) -> None:
    facts = {"items": list(range(limit))}

    assert bounded_tool_facts(facts) == facts


class _RecordingRunSession:
    """The RunSession surface the answer sink writes through, recorded in memory."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    async def emit_tool_event(self, event_type: str, payload: dict[str, Any]) -> None:
        self.events.append((event_type, payload))


async def test_the_answer_sink_forwards_facts_onto_the_durable_tool_event() -> None:
    """The event log carries the same structure as the entry, so both transports can state it."""
    from dlightrag.engine.agent.session.ids import LaneId, OperationId
    from dlightrag.engine.agent.session.runtime import AgentSessionEvent
    from dlightrag.engine.answer.research.runtime import _answer_runtime_event_sink

    session = _RecordingRunSession()
    sink = _answer_runtime_event_sink(session)  # type: ignore[arg-type]
    await sink(
        AgentSessionEvent(
            kind="tool_result_committed",
            session_id=SessionId.new(),
            lane_id=LaneId.main(),
            operation_id=OperationId.new(),
            commit_sequence=4,
            data={
                "tool_name": "search_web",
                "call_id": "call-9",
                "source_index": 0,
                "outcome": "succeeded",
                "duration_ms": 812,
                "facts": {"provider": "exa", "hits": 2},
            },
        )
    )

    assert session.events == [
        (
            "tool_end",
            {
                "tool_name": "search_web",
                "call_id": "call-9",
                "source_index": 0,
                "outcome": "succeeded",
                "duration_ms": 812,
                "facts": {"provider": "exa", "hits": 2},
                "session_commit_sequence": 4,
            },
        )
    ]


async def test_the_answer_sink_still_drops_events_with_no_tool_settlement() -> None:
    from dlightrag.engine.agent.session.ids import LaneId, OperationId
    from dlightrag.engine.agent.session.runtime import AgentSessionEvent
    from dlightrag.engine.answer.research.runtime import _answer_runtime_event_sink

    session = _RecordingRunSession()
    sink = _answer_runtime_event_sink(session)  # type: ignore[arg-type]
    await sink(
        AgentSessionEvent(
            kind="model_start",
            session_id=SessionId.new(),
            lane_id=LaneId.main(),
            operation_id=OperationId.new(),
            commit_sequence=None,
            data={"attempt_id": "attempt-1"},
            ephemeral=True,
        )
    )

    assert session.events == []


def test_a_malformed_leaf_costs_the_fact_and_not_the_tool_call() -> None:
    """A value JSON or UTF-8 refuses is dropped: a settlement must never fail on a fact."""
    huge = 10**5000
    surrogate = json.loads('"\\ud800"')
    circular: list[Any] = []
    circular.append(circular)

    facts = bounded_tool_facts(
        {
            "huge": huge,
            "surrogate": surrogate,
            "circular": circular,
            "kept": 3,
        }
    )

    assert facts == {"circular": [[]], "kept": 3}
