# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Research turn commits record provider-measured token anchors."""

from dataclasses import dataclass
from datetime import UTC, datetime

from dlightrag.agent.session.entries import UserMessageEntry
from dlightrag.agent.session.ids import EntryId, ProjectionId, SessionId
from dlightrag.agent.session.projection import (
    CompactionSummary,
    ContextProjection,
    TokenAnchor,
    live_anchor,
    projection_source_digest,
)
from dlightrag.agent.session.store import SessionCommit
from dlightrag.agent.tools.contracts import ExecutedTurn
from dlightrag.ai.messages import AssistantTurn
from dlightrag.answer.executor import FetchedResourceBuffer, JournalRunBoundaries
from tests.in_memory_session_store import InMemoryAgentSessionStore


@dataclass
class _FakeSession:
    run_id: str
    owner_id: str = "owner"

    async def check_cancelled(self) -> None:
        return None

    async def enter_phase(self, _phase: str) -> None:
        return None


def _seed_projection() -> ContextProjection:
    return ContextProjection(
        projection_id=ProjectionId.new(),
        first_retained_sequence=1,
        covered_through_sequence=0,
        summary=None,
        token_anchors=(
            TokenAnchor(through_sequence=0, measured_input_tokens=0, measured_output_tokens=0),
        ),
    )


async def _seeded_boundaries(
    *,
    projection: ContextProjection | None = None,
) -> tuple[InMemoryAgentSessionStore, SessionId, JournalRunBoundaries]:
    store = InMemoryAgentSessionStore()
    session_id = SessionId.new()
    active = projection if projection is not None else _seed_projection()
    commit = await store.append(
        session_id=session_id,
        expected_version=0,
        entries=[
            UserMessageEntry(
                entry_id=EntryId.new(),
                session_id=session_id,
                timestamp=datetime.now(UTC),
                content="q",
            )
        ],
        projection=active,
    )
    assert isinstance(commit, SessionCommit)
    snapshot = await store.load(session_id)
    bounds = JournalRunBoundaries(
        session=_FakeSession(run_id=str(session_id.value)),  # type: ignore[arg-type]
        journal=store,  # type: ignore[arg-type]
        session_id=session_id,
        tools_by_name={},
        ledger_state=lambda: "{}",
        fetched_buffer=FetchedResourceBuffer(),
        run_id=str(session_id.value),
        initial_version=snapshot.version,
        last_sequence=snapshot.entries[-1].sequence,
        active_projection=snapshot.active_projection,
    )
    return store, session_id, bounds


def _turn(*, usage: dict[str, int] | None) -> ExecutedTurn:
    return ExecutedTurn(
        assistant=AssistantTurn(
            text="ok",
            tool_calls=(),
            stop_reason="stop",
            usage_details=usage,
        ),
        results=(),
        messages=[],
    )


async def test_commit_turn_anchors_usage_at_the_pre_call_sequence() -> None:
    store, session_id, bounds = await _seeded_boundaries()
    await bounds.commit_turn(
        _turn(usage={"prompt_tokens": 80, "completion_tokens": 12}),
        turn_number=1,
    )
    snapshot = await store.load(session_id)
    projection = snapshot.active_projection
    assert projection is not None
    assistant_sequence = next(
        entry.sequence for entry in snapshot.entries if entry.entry_type == "assistant_message"
    )
    assert live_anchor(projection, last_retained_sequence=assistant_sequence) == TokenAnchor(
        through_sequence=1,
        measured_input_tokens=80,
        measured_output_tokens=12,
    )
    assert projection.covered_through_sequence == 0
    assert projection.summary is None


async def test_commit_turn_without_usage_keeps_the_seed_projection() -> None:
    seed = _seed_projection()
    store, session_id, bounds = await _seeded_boundaries(projection=seed)
    await bounds.commit_turn(_turn(usage=None), turn_number=1)
    snapshot = await store.load(session_id)
    assert snapshot.active_projection == seed
    projection = snapshot.active_projection
    assert projection is not None
    assert live_anchor(projection, last_retained_sequence=9) is None


async def test_accounted_input_prefers_the_live_measured_anchor() -> None:
    store, session_id, bounds = await _seeded_boundaries()
    await bounds.commit_turn(
        _turn(usage={"prompt_tokens": 80, "completion_tokens": 12}),
        turn_number=1,
    )
    # The new assistant follows the pre-call anchor and remains in the
    # estimated unanchored tail.
    assert bounds.accounted_input(1_000) > 80
    snapshot = await store.load(session_id)
    assert snapshot.active_projection is not None


async def test_accounted_input_falls_back_to_the_estimate_without_usage() -> None:
    store, session_id, bounds = await _seeded_boundaries()
    await bounds.commit_turn(_turn(usage=None), turn_number=1)
    assert bounds.accounted_input(1_000) == 1_000


async def test_commit_compaction_writes_entry_and_projection_atomically() -> None:
    store, session_id, bounds = await _seeded_boundaries()
    summary = CompactionSummary(goal="answer", progress="three sources read").canonical_json()
    seeded = await store.load(session_id)
    covered_id = seeded.entries[0].entry_id
    projection = ContextProjection(
        projection_id=ProjectionId.new(),
        first_retained_sequence=2,
        covered_through_sequence=1,
        summary=summary,
        token_anchors=(),
        covered_through_entry_id=covered_id,
        source_digest=projection_source_digest((covered_id,)),
    )
    commit = await bounds.commit_compaction(projection=projection)
    assert commit.version == 2
    snapshot = await store.load(session_id)
    compaction_entries = [entry for entry in snapshot.entries if entry.entry_type == "compaction"]
    assert len(compaction_entries) == 1
    assert compaction_entries[0].sequence == 2
    assert snapshot.active_projection == projection
    # A second compaction keeps the chain going without a version conflict.
    second_projection = ContextProjection(
        projection_id=ProjectionId.new(),
        first_retained_sequence=3,
        covered_through_sequence=2,
        summary=summary,
        token_anchors=(),
        covered_through_entry_id=compaction_entries[0].entry_id,
        source_digest=projection_source_digest((covered_id, compaction_entries[0].entry_id)),
    )
    second = await bounds.commit_compaction(projection=second_projection)
    assert second.version == 3
