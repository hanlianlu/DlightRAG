# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Fast Host turns share the canonical Agent Session tree without an Operation."""

from typing import Any, cast

import pytest

from dlightrag.answer.session_host import FastSessionHost
from dlightrag.engine.agent.session.entries import AssistantMessageEntry, UserMessageEntry
from dlightrag.engine.agent.session.ids import EntryId, LaneId, SessionId
from dlightrag.engine.agent.session.memory import MemoryAgentSessionRepository
from dlightrag.engine.agent.session.plan import AgentRunPlan
from dlightrag.engine.agent.session.registers import HostTurnReservation, decode_register
from dlightrag.engine.agent.session.runtime import AgentSessionRuntime, OperationConflictError


async def _no_settled_result() -> None:
    return None


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
    store = MemoryAgentSessionRepository[None]()
    settled_result: dict[str, Any] | None = None

    async def load_settled_result() -> dict[str, Any] | None:
        return settled_result

    host = FastSessionHost(
        transactions=store,
        load=store.load,
        load_settled_result=load_settled_result,
        fencing_epoch=1,
    )
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

    settled_result = {
        "answer": "answer",
        "contexts": {"chunks": []},
        "sources": [],
        "artifacts": [],
        "usage": {"input_tokens": 2, "output_tokens": 1},
    }
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
    assistant = settled.tree.ancestry()[-1]
    assert isinstance(assistant, AssistantMessageEntry)
    assert assistant.acceptance_id == "run-1"
    assert assistant.provider_state is None
    settled_replay = await host.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="run-1",
        idempotency_key="submission-1",
        content="question",
    )
    assert settled_replay.settled is True
    assert settled_replay.settled_payload == settled_result
    assert len((await store.load(session_id)).entries) == 2
    assert (
        await host.fail(
            session_id=session_id,
            lane_id=LaneId.main(),
            reservation_id="run-1",
        )
        is None
    )


@pytest.mark.asyncio
async def test_staged_fast_result_settles_active_reservation_without_regeneration() -> None:
    store = MemoryAgentSessionRepository[None]()
    settled_result: dict[str, Any] | None = None

    async def load_settled_result() -> dict[str, Any] | None:
        return settled_result

    host = FastSessionHost(
        transactions=store,
        load=store.load,
        load_settled_result=load_settled_result,
        fencing_epoch=1,
    )
    session_id = SessionId.new()
    await host.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="run-crashed",
        idempotency_key="submission-crashed",
        content="question",
    )
    settled_result = {
        "answer": "durable answer",
        "contexts": {"chunks": [{"chunk_id": "c1"}]},
        "sources": [{"id": "1"}],
        "artifacts": [{"resource_id": "report"}],
        "usage": {"input_tokens": 7, "output_tokens": 3},
    }

    assert (
        await host.fail(
            session_id=session_id,
            lane_id=LaneId.main(),
            reservation_id="run-crashed",
        )
        is None
    )
    staged = await store.load(session_id)
    assert any(isinstance(record.value, HostTurnReservation) for record in staged.registers)

    replay = await host.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="run-crashed",
        idempotency_key="submission-crashed",
        content="question",
    )

    assert replay.settled_payload == settled_result
    assert replay.progress_advanced is True
    snapshot = await store.load(session_id)
    assert [entry.entry_type for entry in snapshot.tree.ancestry()] == [
        "user_message",
        "assistant_message",
    ]
    assistant = snapshot.tree.ancestry()[-1]
    assert isinstance(assistant, AssistantMessageEntry)
    assert assistant.content == "durable answer"
    assert not any(isinstance(record.value, HostTurnReservation) for record in snapshot.registers)


@pytest.mark.asyncio
async def test_staged_fast_result_rejects_an_interleaved_lane_head() -> None:
    store = MemoryAgentSessionRepository[None]()
    settled_result: dict[str, Any] | None = None

    async def load_settled_result() -> dict[str, Any] | None:
        return settled_result

    host = FastSessionHost(
        transactions=store,
        load=store.load,
        load_settled_result=load_settled_result,
        fencing_epoch=1,
    )
    session_id = SessionId.new()
    await host.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="run-crashed",
        idempotency_key="submission-crashed",
        content="original question",
    )
    await host.fail(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="run-crashed",
    )
    await host.accept(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="run-other",
        idempotency_key="submission-other",
        content="interleaved question",
    )
    await host.complete(
        session_id=session_id,
        lane_id=LaneId.main(),
        reservation_id="run-other",
        content="interleaved answer",
    )
    settled_result = {
        "answer": "durable answer",
        "contexts": {"chunks": []},
        "sources": [],
        "artifacts": [],
        "usage": {},
    }

    with pytest.raises(OperationConflictError, match="accepted lane head"):
        await host.accept(
            session_id=session_id,
            lane_id=LaneId.main(),
            reservation_id="run-crashed",
            idempotency_key="submission-crashed",
            content="original question",
        )

    snapshot = await store.load(session_id)
    assert [getattr(entry, "acceptance_id", None) for entry in snapshot.tree.ancestry()] == [
        "run-crashed",
        "run-other",
        "run-other",
    ]


@pytest.mark.asyncio
async def test_fast_failure_clears_reservation_but_keeps_unanswered_user() -> None:
    store = MemoryAgentSessionRepository[None]()
    host = FastSessionHost(
        transactions=store,
        load=store.load,
        load_settled_result=_no_settled_result,
        fencing_epoch=1,
    )
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
    store = MemoryAgentSessionRepository[None]()
    session_id = SessionId.new()
    host = FastSessionHost(
        transactions=store,
        load=store.load,
        load_settled_result=_no_settled_result,
        fencing_epoch=1,
    )
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
