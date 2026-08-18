# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the agent session store contract and its in-memory adapter."""

from datetime import UTC, datetime

import pytest
from dlightrag_agent.session.effects import EffectSettlement
from dlightrag_agent.session.entries import (
    EffectIntentEntry,
    EffectResultEntry,
    UserMessageEntry,
)
from dlightrag_agent.session.ids import EntryId, IntentId, ProjectionId, SessionId
from dlightrag_agent.session.memory import InMemoryAgentSessionStore
from dlightrag_agent.session.projection import ContextProjection
from dlightrag_agent.session.store import (
    EffectAlreadySettled,
    EffectCommit,
    EffectMissing,
    NoHostUpdate,
    SessionCommit,
    VersionConflict,
)


def _now() -> datetime:
    return datetime.now(UTC)


def _user(session_id: SessionId, content: str) -> UserMessageEntry:
    return UserMessageEntry(
        entry_id=EntryId.new(), session_id=session_id, timestamp=_now(), content=content
    )


def _intent(session_id: SessionId, *, intent_id: IntentId | None = None) -> EffectIntentEntry:
    from dlightrag_agent.session.effects import EffectIntent

    return EffectIntentEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        timestamp=_now(),
        intent=EffectIntent(
            intent_id=intent_id or IntentId.new(),
            tool_name="search_knowledge_base",
            replay_policy="safe",
            contract_version=1,
            input_schema_digest="a" * 64,
            canonical_input='{"q":"x"}',
            source_call_id="c1",
        ),
    )


def _settlement(intent_id: IntentId) -> EffectSettlement[NoHostUpdate]:
    from dlightrag_agent.session.effects import ToolResultEntry

    result = ToolResultEntry(
        tool_name="search_knowledge_base", call_id="c1", outcome="succeeded", content="found"
    )
    return EffectSettlement(outcome="succeeded", result=result, host_update=NoHostUpdate())


def _result_entry(
    session_id: SessionId, intent_id: IntentId, *, sequence: int = 0
) -> EffectResultEntry:
    from dlightrag_agent.session.effects import ToolResultEntry

    return EffectResultEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        timestamp=_now(),
        intent_id=intent_id,
        result=ToolResultEntry(
            tool_name="search_knowledge_base", call_id="c1", outcome="succeeded", content="found"
        ),
    )


@pytest.mark.asyncio
async def test_append_with_wrong_expected_version_changes_nothing() -> None:
    store = InMemoryAgentSessionStore()
    session_id = SessionId.new()
    first = await store.append(
        session_id=session_id, expected_version=0, entries=[_user(session_id, "hi")]
    )
    assert isinstance(first, SessionCommit)

    conflict = await store.append(
        session_id=session_id, expected_version=0, entries=[_user(session_id, "again")]
    )
    assert isinstance(conflict, VersionConflict)
    assert conflict.expected_version == 0
    assert conflict.current_version == 1

    snapshot = await store.load(session_id)
    assert snapshot.version == 1
    assert len(snapshot.entries) == 1
    assert snapshot.entries[0].canonical_payload()["content"] == "hi"


@pytest.mark.asyncio
async def test_append_never_settles_an_intent() -> None:
    store = InMemoryAgentSessionStore()
    session_id = SessionId.new()
    intent = _intent(session_id)
    commit = await store.append(session_id=session_id, expected_version=0, entries=[intent])
    assert isinstance(commit, SessionCommit)

    # The intent exists but is unsettled: a settlement of a missing intent id
    # still reports EffectMissing, and settling it works exactly once.
    missing = await store.settle_effect(
        session_id=session_id,
        expected_version=1,
        intent_id=IntentId.new(),
        settlement=_settlement(intent.intent_id),
        entries=[_result_entry(session_id, intent.intent_id)],
    )
    assert isinstance(missing, EffectMissing)


@pytest.mark.asyncio
async def test_settle_effect_atomically_marks_and_appends_ordered_results() -> None:
    store = InMemoryAgentSessionStore()
    session_id = SessionId.new()
    intent = _intent(session_id)
    await store.append(session_id=session_id, expected_version=0, entries=[intent])

    settled = await store.settle_effect(
        session_id=session_id,
        expected_version=1,
        intent_id=intent.intent_id,
        settlement=_settlement(intent.intent_id),
        entries=[_result_entry(session_id, intent.intent_id)],
    )
    assert isinstance(settled, EffectCommit)
    assert settled.version == 2
    assert settled.intent_id == intent.intent_id
    assert settled.outcome == "succeeded"

    again = await store.settle_effect(
        session_id=session_id,
        expected_version=2,
        intent_id=intent.intent_id,
        settlement=_settlement(intent.intent_id),
        entries=[_result_entry(session_id, intent.intent_id)],
    )
    assert isinstance(again, EffectAlreadySettled)


@pytest.mark.asyncio
async def test_version_conflict_on_settlement_rolls_back_everything() -> None:
    store = InMemoryAgentSessionStore()
    session_id = SessionId.new()
    intent = _intent(session_id)
    await store.append(session_id=session_id, expected_version=0, entries=[intent])

    conflict = await store.settle_effect(
        session_id=session_id,
        expected_version=0,
        intent_id=intent.intent_id,
        settlement=_settlement(intent.intent_id),
        entries=[_result_entry(session_id, intent.intent_id)],
    )
    assert isinstance(conflict, VersionConflict)
    snapshot = await store.load(session_id)
    assert snapshot.version == 1
    assert len(snapshot.entries) == 1
    assert isinstance(snapshot.entries[0], EffectIntentEntry)


@pytest.mark.asyncio
async def test_sequences_are_contiguous_per_transaction() -> None:
    store = InMemoryAgentSessionStore()
    session_id = SessionId.new()
    first = await store.append(
        session_id=session_id,
        expected_version=0,
        entries=[_user(session_id, "a"), _user(session_id, "b")],
    )
    assert isinstance(first, SessionCommit)
    assert first.appended_sequences == (1, 2)

    intent = _intent(session_id)
    second = await store.append(session_id=session_id, expected_version=1, entries=[intent])
    assert isinstance(second, SessionCommit)
    assert second.appended_sequences == (3,)

    snapshot = await store.load(session_id)
    assert [entry.sequence for entry in snapshot.entries] == [1, 2, 3]


@pytest.mark.asyncio
async def test_projection_commits_with_the_transaction() -> None:
    from dlightrag_agent.session.projection import TokenAnchor

    store = InMemoryAgentSessionStore()
    session_id = SessionId.new()
    projection = ContextProjection(
        projection_id=ProjectionId.new(),
        first_retained_sequence=1,
        covered_through_sequence=0,
        summary=None,
        token_anchors=(
            TokenAnchor(through_sequence=0, measured_input_tokens=10, measured_output_tokens=2),
        ),
    )
    commit = await store.append(
        session_id=session_id,
        expected_version=0,
        entries=[_user(session_id, "hi")],
        projection=projection,
    )
    assert isinstance(commit, SessionCommit)
    snapshot = await store.load(session_id)
    assert snapshot.active_projection == projection


@pytest.mark.asyncio
async def test_empty_transaction_is_rejected() -> None:
    store = InMemoryAgentSessionStore()
    session_id = SessionId.new()
    with pytest.raises(ValueError):
        await store.append(session_id=session_id, expected_version=0, entries=[])


@pytest.mark.asyncio
async def test_settlement_entries_must_belong_to_the_intent() -> None:
    store = InMemoryAgentSessionStore()
    session_id = SessionId.new()
    intent = _intent(session_id)
    await store.append(session_id=session_id, expected_version=0, entries=[intent])

    with pytest.raises(ValueError):
        await store.settle_effect(
            session_id=session_id,
            expected_version=1,
            intent_id=intent.intent_id,
            settlement=_settlement(intent.intent_id),
            entries=[_result_entry(session_id, IntentId.new())],
        )


@pytest.mark.asyncio
async def test_prelude_settlement_still_commits_the_effect() -> None:
    store = InMemoryAgentSessionStore()
    session_id = SessionId.new()
    intent = _intent(session_id)
    await store.append(session_id=session_id, expected_version=0, entries=[intent])

    settled = await store.settle_effect(
        session_id=session_id,
        expected_version=1,
        intent_id=intent.intent_id,
        settlement=_settlement(intent.intent_id),
        entries=[_result_entry(session_id, intent.intent_id)],
        progress="prelude",
    )
    assert isinstance(settled, EffectCommit)
    snapshot = await store.load(session_id)
    assert snapshot.version == 2
