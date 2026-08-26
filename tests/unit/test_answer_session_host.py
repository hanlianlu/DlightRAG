# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Fast Host turns share the canonical Agent Session tree without an Operation."""

from typing import Any, cast

import pytest

from dlightrag.agent.session.entries import AssistantMessageEntry, UserMessageEntry
from dlightrag.agent.session.ids import EntryId, LaneId, SessionId
from dlightrag.agent.session.memory import MemoryAgentSessionStore
from dlightrag.agent.session.plan import AgentRunPlan
from dlightrag.agent.session.registers import HostTurnReservation, decode_register
from dlightrag.agent.session.runtime import AgentSessionRuntime, OperationConflictError
from dlightrag.answer.session_host import FastSessionHost


def test_fast_host_turn_reservation_round_trips_closed_register_codec() -> None:
    value = HostTurnReservation(
        lane_id=LaneId.main(),
        reservation_id="run-1",
        idempotency_key="submission-1",
        user_entry_id=EntryId.new(),
    )
    assert (
        decode_register(
            kind="host_turn_reservation",
            payload=value.canonical_payload(),
        )
        == value
    )


@pytest.mark.asyncio
async def test_fast_turn_accepts_user_and_reservation_then_settles_assistant() -> None:
    store = MemoryAgentSessionStore[None]()
    host = FastSessionHost(transactions=store, load=store.load, fencing_epoch=1)
    session_id = SessionId.new()

    accepted = await host.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="run-1",
        idempotency_key="submission-1",
        content="question",
    )
    replay = await host.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="run-1",
        idempotency_key="submission-1",
        content="question",
    )
    assert accepted.created is True
    assert replay.created is False
    reserved = await store.load(session_id)
    assert [type(entry) for entry in reserved.tree.ancestry()] == [UserMessageEntry]
    assert any(isinstance(record.value, HostTurnReservation) for record in reserved.registers)

    await host.complete(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="run-1",
        content="answer",
        usage={"input_tokens": 2, "output_tokens": 1},
    )
    settled = await store.load(session_id)
    assert [type(entry) for entry in settled.tree.ancestry()] == [
        UserMessageEntry,
        AssistantMessageEntry,
    ]
    assert not any(isinstance(record.value, HostTurnReservation) for record in settled.registers)
    settled_replay = await host.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="run-1",
        idempotency_key="submission-1",
        content="question",
    )
    assert settled_replay.settled is True
    assert len((await store.load(session_id)).entries) == 2


@pytest.mark.asyncio
async def test_fast_failure_clears_reservation_but_keeps_unanswered_user() -> None:
    store = MemoryAgentSessionStore[None]()
    host = FastSessionHost(transactions=store, load=store.load, fencing_epoch=1)
    session_id = SessionId.new()
    await host.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="run-failed",
        idempotency_key="submission-failed",
        content="unanswered",
    )

    await host.fail(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="run-failed",
    )
    snapshot = await store.load(session_id)
    [entry] = snapshot.tree.ancestry()
    assert isinstance(entry, UserMessageEntry)
    assert entry.content == "unanswered"
    assert not any(isinstance(record.value, HostTurnReservation) for record in snapshot.registers)

    recovered = await host.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="run-failed",
        idempotency_key="submission-failed",
        content="unanswered",
    )
    assert recovered.created is False
    assert len((await store.load(session_id)).entries) == 1
    await host.fail(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="run-failed",
    )
    with pytest.raises(OperationConflictError, match="changed"):
        await host.accept(
            session_id=session_id,
            lane_id=LaneId.main(),
            reservation_id="run-failed",
            idempotency_key="submission-failed",
            content="different",
        )


@pytest.mark.asyncio
async def test_runtime_accept_rejects_a_fast_reservation_on_the_same_lane() -> None:
    store = MemoryAgentSessionStore[None]()
    session_id = SessionId.new()
    host = FastSessionHost(transactions=store, load=store.load, fencing_epoch=1)
    await host.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="run-fast",
        idempotency_key="submission-fast",
        content="fast question",
    )
    runtime = AgentSessionRuntime[None](
        transactions=store,
        load=store.load,
        effects=cast(Any, object()),
        tools=(),
        fencing_epoch=1,
    )

    with pytest.raises(OperationConflictError, match="active Host turn"):
        await runtime.accept(
            session_id=session_id,
            lane_id=LaneId.main(),
            idempotency_key="research-run",
            content="research question",
            plan=AgentRunPlan(
                model_role="query",
                context_policy_revision="test",
                tools=(),
            ),
        )
