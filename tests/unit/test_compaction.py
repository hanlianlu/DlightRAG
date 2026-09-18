# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Automatic checkpoint compaction parsing and projection contracts."""

import pytest

from dlightrag.engine.agent.session.fold import host_turn_starts
from dlightrag.engine.agent.session.ids import LaneId, SessionId
from dlightrag.engine.agent.session.memory import MemoryAgentSessionRepository
from dlightrag.engine.agent.session.projection import CompactionSummary, render_compaction_summary
from dlightrag.engine.ai.capacity import ContextPolicy, ModelProfile
from dlightrag.engine.answer.compaction import (
    _MAX_DURABLE_HANDLES,
    CompactionCoordinator,
    CompactionUnavailable,
    parse_compaction_summary,
)
from dlightrag.engine.answer.continuation_handles import MAX_NAMED_SESSION_NOTES, MAX_SPILL_HANDLES
from dlightrag.engine.answer.fast import FastSessionHost


def test_compaction_summary_parser_preserves_typed_sections() -> None:
    summary = parse_compaction_summary(
        """## Goal
Ship the runtime.

## Constraints & Preferences
No generic workflow.

## Progress
Done: Runtime.

## Key Decisions
Total state.

## Next Steps
Review.

## Critical Context
Recovery uses the same interpreter.
"""
    )
    assert summary.goal == "Ship the runtime."
    assert summary.constraints_preferences == "No generic workflow."
    assert summary.progress == "Done: Runtime."
    assert summary.decisions == "Total state."
    assert summary.next_steps == "Review."
    assert summary.critical_context == "Recovery uses the same interpreter."


def test_compaction_summary_requires_a_goal() -> None:
    with pytest.raises(ValueError, match="goal"):
        parse_compaction_summary("## Progress\nNothing")


def test_unknown_compaction_sections_are_not_silently_dropped() -> None:
    summary = parse_compaction_summary("## Goal\nShip.\n\n## Extra\nKeep this.")
    assert "## Extra" in summary.critical_context
    assert "Keep this." in summary.critical_context


@pytest.mark.asyncio
async def test_coordinator_compacts_direct_fast_pairs_without_a_baseline_register() -> None:
    store = MemoryAgentSessionRepository[None]()

    async def no_result() -> None:
        return None

    session_id = SessionId.new()
    host = FastSessionHost(
        repository=store,
        initial_snapshot=await store.load(session_id),
        load_settled_result=no_result,
        fencing_epoch=1,
    )
    await host.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="old",
        idempotency_key="old-key",
        content="old question " * 200,
    )
    await host.complete(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="old",
        content="old answer " * 200,
    )
    current = await host.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="current",
        idempotency_key="current-key",
        content="current question",
    )

    async def stream_model(**_kwargs):
        yield "## Goal\nPreserve the old turn."

    coordinator = CompactionCoordinator(
        model_profile=ModelProfile(context_window_tokens=100_000),
        context_policy=ContextPolicy(
            requested_output_reserve_tokens=1_000,
            dynamic_context_reserve_tokens=1_000,
            retained_tail_tokens=0,
        ),
        stream_model=stream_model,
        exchange_starts_func=host_turn_starts,
    )
    snapshot = await store.load(session_id)
    projection, outcome = await coordinator.prepare(
        snapshot,
        tail_target_tokens=0,
        accounted_before=100,
        trace={},
    )

    assert projection.covered_through_entry_id == snapshot.tree.ancestry()[1].entry_id
    assert projection.first_retained_entry_id == current.user_entry_id
    assert outcome.covered_through_sequence == snapshot.tree.ancestry()[1].sequence


@pytest.mark.asyncio
async def test_compaction_keeps_the_runs_source_handles_re_readable() -> None:
    """A compacted transcript loses the passages it showed; their handles stay.

    The former framework field had no authority once temporary Tool Arguments were
    deleted, and the pair was deliberately emptied. The Evidence ledger is the record
    of what the run actually admitted, so handles come from there instead.
    """
    store = MemoryAgentSessionRepository[None]()

    async def no_result() -> None:
        return None

    session_id = SessionId.new()
    host = FastSessionHost(
        repository=store,
        initial_snapshot=await store.load(session_id),
        load_settled_result=no_result,
        fencing_epoch=1,
    )
    await host.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="old",
        idempotency_key="old-key",
        content="old question " * 200,
    )
    await host.complete(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="old",
        content="old answer " * 200,
    )
    await host.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="current",
        idempotency_key="current-key",
        content="current question",
    )

    async def stream_model(**_kwargs):
        yield "## Goal\nPreserve the old turn."

    coordinator = CompactionCoordinator(
        model_profile=ModelProfile(context_window_tokens=100_000),
        context_policy=ContextPolicy(
            requested_output_reserve_tokens=1_000,
            dynamic_context_reserve_tokens=1_000,
            retained_tail_tokens=0,
        ),
        stream_model=stream_model,
        exchange_starts_func=host_turn_starts,
    )
    projection, _outcome = await coordinator.prepare(
        await store.load(session_id),
        tail_target_tokens=0,
        accounted_before=100,
        durable_handles=("[1] report.pdf", "[1] report.pdf", "  ", "[2] memo.docx"),
        trace={},
    )

    assert projection.summary is not None
    summary = CompactionSummary.from_canonical_json(projection.summary)
    # Deduplicated, blank-free, and ordered as the ledger admitted them.
    assert summary.durable_handles == ["[1] report.pdf", "[2] memo.docx"]
    assert "durable handles (re-readable, not evidence)" in render_compaction_summary(
        projection.summary
    )


def test_the_spill_share_leaves_room_for_evidence_under_the_summary_cap() -> None:
    """Two numbers in two modules decide whether Evidence can be starved.

    Raising the spill share to the summary's whole cap would make a retrieval-heavy
    Run's citation handles silently disappear, which is the failure the reserved
    share exists to prevent, so the relationship is asserted rather than assumed.
    """
    assert MAX_SPILL_HANDLES + 1 <= _MAX_DURABLE_HANDLES


@pytest.mark.asyncio
async def test_the_summary_carries_the_bounded_deduplicated_run_notes() -> None:
    """One coordinator call is where a note list becomes committed continuation memory.

    The orchestrator test records the argument and the compose test bounds the
    lines, so without this the field could stop accepting notes and every test
    would stay green — the hole the spill handle test was written to close.
    """
    store = MemoryAgentSessionRepository[None]()

    async def no_result() -> None:
        return None

    session_id = SessionId.new()
    host = FastSessionHost(
        repository=store,
        initial_snapshot=await store.load(session_id),
        load_settled_result=no_result,
        fencing_epoch=1,
    )
    await host.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="old",
        idempotency_key="old-key",
        content="old question " * 200,
    )
    await host.complete(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="old",
        content="old answer " * 200,
    )
    await host.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="current",
        idempotency_key="current-key",
        content="current question",
    )

    async def stream_model(**_kwargs):
        yield "## Goal\nKeep working from the notes."

    coordinator = CompactionCoordinator(
        model_profile=ModelProfile(context_window_tokens=100_000),
        context_policy=ContextPolicy(
            requested_output_reserve_tokens=1_000,
            dynamic_context_reserve_tokens=1_000,
            retained_tail_tokens=0,
        ),
        stream_model=stream_model,
        exchange_starts_func=host_turn_starts,
    )
    notes = [f"[note] notes/{index}.md (12 bytes)" for index in range(MAX_NAMED_SESSION_NOTES + 4)]
    projection, _outcome = await coordinator.prepare(
        await store.load(session_id),
        tail_target_tokens=0,
        accounted_before=100,
        run_notes=(*notes, "  ", notes[0]),
        trace={},
    )

    summary_json = projection.summary
    assert summary_json is not None
    summary = CompactionSummary.from_canonical_json(summary_json)
    assert summary.run_notes == notes[:MAX_NAMED_SESSION_NOTES]
    assert "Session notes (re-readable, not evidence):" in render_compaction_summary(summary_json)


@pytest.mark.asyncio
async def test_a_prefix_with_no_complete_exchange_is_unavailable_not_a_failure() -> None:
    """Unavailability is a property of the Session, so the Run keeps going.

    Measured live: a Run whose retained tail was one oversized exchange was over the
    trigger again, and the projection's whole uncovered prefix held no exchange start,
    so every attempt — three of them, each with a smaller tail — failed identically and
    the Run died. Nothing about retrying could have helped, and the request was inside
    the hard input limit, so the honest answer is "no compaction is possible".
    """
    store = MemoryAgentSessionRepository[None]()

    async def no_result() -> None:
        return None

    session_id = SessionId.new()
    host = FastSessionHost(
        repository=store,
        initial_snapshot=await store.load(session_id),
        load_settled_result=no_result,
        fencing_epoch=1,
    )
    await host.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="current",
        idempotency_key="current-key",
        content="current question",
    )
    summarized = False

    async def stream_model(**_kwargs):
        nonlocal summarized
        summarized = True
        yield "## Goal\nnever reached"

    coordinator = CompactionCoordinator(
        model_profile=ModelProfile(context_window_tokens=100_000),
        context_policy=ContextPolicy(retained_tail_tokens=0),
        stream_model=stream_model,
        # Every exchange this Session holds is already retained, which is the state a
        # boundary pinned at an oversized newest exchange leaves behind.
        exchange_starts_func=lambda _entries: (),
    )

    with pytest.raises(CompactionUnavailable):
        await coordinator.prepare(
            await store.load(session_id),
            tail_target_tokens=0,
            accounted_before=100,
            trace={},
        )
    assert summarized is False
