# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Automatic checkpoint compaction parsing and projection contracts."""

import pytest

from dlightrag.engine.agent.session.fold import host_turn_starts
from dlightrag.engine.agent.session.ids import LaneId, SessionId
from dlightrag.engine.agent.session.memory import MemoryAgentSessionRepository
from dlightrag.engine.ai.capacity import ContextPolicy, ModelProfile
from dlightrag.engine.answer.compaction import CompactionCoordinator, parse_compaction_summary
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
            safety_reserve_tokens=100,
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
