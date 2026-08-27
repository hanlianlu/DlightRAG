# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Canonical Memory Session transaction/store behavior."""

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from dlightrag.engine.agent.session.entries import UserMessageEntry
from dlightrag.engine.agent.session.ids import EntryId, IntentId, LaneId, OperationId, SessionId
from dlightrag.engine.agent.session.memory import MemoryAgentSessionRepository
from dlightrag.engine.agent.session.operation import OperationMeta, ReadyForProvider
from dlightrag.engine.agent.session.registers import (
    DeleteRegister,
    LaneHead,
    LaneState,
    OperationMetaRegister,
    OperationStateRegister,
    RegisterRecord,
    RegisterRef,
    SessionFault,
    SetRegister,
)
from dlightrag.engine.agent.session.transactions import (
    HostDeltaSettlement,
    RegisterConflict,
    RegisterExpectation,
    SessionTransaction,
    TransactionCommit,
)


def _user(session_id: SessionId, content: str) -> UserMessageEntry:
    return UserMessageEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        timestamp=datetime.now(UTC),
        content=content,
    )


async def _append_entries(
    store: MemoryAgentSessionRepository[None],
    *,
    session_id: SessionId,
    lane_id: LaneId,
    expected_head: RegisterRecord,
    entries: list[UserMessageEntry],
):
    snapshot = await store.load(session_id)
    lane = snapshot.tree.lane(lane_id)
    assert isinstance(expected_head.value, LaneHead)
    parent = expected_head.value.entry_id
    placed: list[UserMessageEntry] = []
    for entry in entries:
        item = replace(entry, parent_entry_id=parent)
        placed.append(item)
        parent = item.entry_id
    return await store.transact(
        session_id=session_id,
        fencing_epoch=1,
        transaction=SessionTransaction.from_parts(
            entries=placed,
            register_writes=[SetRegister(LaneHead(lane_id, parent))],
            expectations=[
                RegisterExpectation(expected_head.ref, expected_head.sequence),
                RegisterExpectation(lane.state.ref, lane.state.sequence),
            ],
        ),
    )


async def _fork_branch(
    store: MemoryAgentSessionRepository[None],
    *,
    session_id: SessionId,
    source_lane_id: LaneId,
    lane_id: LaneId,
) -> None:
    snapshot = await store.load(session_id)
    target = snapshot.tree.lane(source_lane_id).head_entry_id
    assert target is not None and snapshot.tree.is_stable_checkpoint(target)
    head = LaneHead(lane_id, target)
    state = LaneState(lane_id)
    outcome = await store.transact(
        session_id=session_id,
        fencing_epoch=1,
        transaction=SessionTransaction.from_parts(
            register_writes=[SetRegister(head), SetRegister(state)],
            expectations=[
                RegisterExpectation(head.ref, None),
                RegisterExpectation(state.ref, None),
            ],
        ),
    )
    assert isinstance(outcome, TransactionCommit)


async def _seed(store: MemoryAgentSessionRepository[None], session_id: SessionId) -> None:
    head = LaneHead(LaneId.main(), None)
    state = LaneState(LaneId.main())
    entry = _user(session_id, "root")
    outcome = await store.transact(
        session_id=session_id,
        fencing_epoch=1,
        transaction=SessionTransaction.from_parts(
            entries=[entry],
            register_writes=[
                SetRegister(LaneHead(LaneId.main(), entry.entry_id)),
                SetRegister(state),
            ],
            expectations=[
                RegisterExpectation(head.ref, None),
                RegisterExpectation(state.ref, None),
            ],
        ),
    )
    assert isinstance(outcome, TransactionCommit)


@pytest.mark.asyncio
async def test_exact_lane_cas_ignores_unrelated_branch_commit() -> None:
    store = MemoryAgentSessionRepository[None]()
    session_id = SessionId.new()
    await _seed(store, session_id)
    main = (await store.load(session_id)).tree.lane()
    branch_id = LaneId.new()
    await _fork_branch(
        store,
        session_id=session_id,
        source_lane_id=LaneId.main(),
        lane_id=branch_id,
    )
    branch = (await store.load(session_id)).tree.lane(branch_id)
    await _append_entries(
        store,
        session_id=session_id,
        lane_id=branch_id,
        expected_head=branch.head,
        entries=[_user(session_id, "branch")],
    )
    main_commit = await _append_entries(
        store,
        session_id=session_id,
        lane_id=LaneId.main(),
        expected_head=main.head,
        entries=[_user(session_id, "main")],
    )
    assert isinstance(main_commit, TransactionCommit)


@pytest.mark.asyncio
async def test_same_lane_stale_head_conflicts_without_writing() -> None:
    store = MemoryAgentSessionRepository[None]()
    session_id = SessionId.new()
    await _seed(store, session_id)
    stale = (await store.load(session_id)).tree.lane().head
    await _append_entries(
        store,
        session_id=session_id,
        lane_id=LaneId.main(),
        expected_head=stale,
        entries=[_user(session_id, "first")],
    )
    conflict = await _append_entries(
        store,
        session_id=session_id,
        lane_id=LaneId.main(),
        expected_head=stale,
        entries=[_user(session_id, "lost")],
    )
    assert isinstance(conflict, RegisterConflict)
    ancestry = (await store.load(session_id)).tree.ancestry()
    assert all(isinstance(entry, UserMessageEntry) for entry in ancestry)
    assert [entry.content for entry in ancestry if isinstance(entry, UserMessageEntry)] == [
        "root",
        "first",
    ]


def _operation_registers() -> tuple[OperationMetaRegister, OperationStateRegister]:
    operation_id = OperationId.new()
    return (
        OperationMetaRegister(
            OperationMeta(
                operation_id=operation_id,
                lane_id=LaneId.main(),
                idempotency_key="operation",
                acceptance_digest="a" * 64,
                plan_json="{}",
                plan_digest="b" * 64,
            )
        ),
        OperationStateRegister(ReadyForProvider(operation_id)),
    )


@pytest.mark.parametrize("immutable", ["operation_meta", "session_fault"])
def test_transaction_rejects_updates_to_immutable_registers(immutable: str) -> None:
    meta, _state = _operation_registers()
    value = meta if immutable == "operation_meta" else SessionFault("fault")
    with pytest.raises(ValueError, match="immutable"):
        SessionTransaction.from_parts(
            register_writes=[SetRegister(value)],
            expectations=[RegisterExpectation(value.ref, 1)],
        )


@pytest.mark.parametrize(
    "ref",
    [
        RegisterRef("operation_meta", "operation"),
        RegisterRef("operation_state", "operation"),
        RegisterRef("session_fault", "session"),
        RegisterRef("lane_head", LaneId.main().value),
        RegisterRef("lane_state", LaneId.main().value),
    ],
)
def test_transaction_rejects_deleting_permanent_or_main_registers(ref: RegisterRef) -> None:
    with pytest.raises(ValueError, match="cannot be deleted"):
        SessionTransaction.from_parts(
            register_writes=[DeleteRegister(ref)],
            expectations=[RegisterExpectation(ref, 1)],
        )


def test_transaction_rejects_archiving_main_lane() -> None:
    state = LaneState(LaneId.main(), archived=True)
    with pytest.raises(ValueError, match="main Lane cannot be archived"):
        SessionTransaction.from_parts(
            register_writes=[SetRegister(state)],
            expectations=[RegisterExpectation(state.ref, 1)],
        )


def test_entry_transaction_must_advance_lane_head_to_final_entry() -> None:
    session_id = SessionId.new()
    entry = _user(session_id, "unplaced")
    state = LaneState(LaneId.main())
    with pytest.raises(ValueError, match="advance a Lane Head"):
        SessionTransaction.from_parts(
            entries=[entry],
            register_writes=[SetRegister(state)],
            expectations=[RegisterExpectation(state.ref, 1)],
        )
    wrong_head = LaneHead(LaneId.main(), EntryId.new())
    with pytest.raises(ValueError, match="advance a Lane Head"):
        SessionTransaction.from_parts(
            entries=[entry],
            register_writes=[SetRegister(wrong_head)],
            expectations=[RegisterExpectation(wrong_head.ref, 1)],
        )


@pytest.mark.asyncio
async def test_memory_host_delta_is_exactly_once_under_register_cas() -> None:
    store = MemoryAgentSessionRepository[dict[str, str]]()
    session_id = SessionId.new()
    head = LaneHead(LaneId.main(), None)
    state = LaneState(LaneId.main())
    initial = await store.transact(
        session_id=session_id,
        fencing_epoch=1,
        transaction=SessionTransaction.from_parts(
            register_writes=[SetRegister(head), SetRegister(state)],
            expectations=[
                RegisterExpectation(head.ref, None),
                RegisterExpectation(state.ref, None),
            ],
        ),
    )
    assert isinstance(initial, TransactionCommit)
    intent_id = IntentId.new()
    transaction = SessionTransaction.from_parts(
        register_writes=[SetRegister(state)],
        expectations=[RegisterExpectation(state.ref, initial.commit_sequence)],
        host_delta=HostDeltaSettlement(intent_id, {"memory": "changed"}),
    )
    first = await store.transact(
        session_id=session_id,
        fencing_epoch=1,
        transaction=transaction,
    )
    replay = await store.transact(
        session_id=session_id,
        fencing_epoch=1,
        transaction=transaction,
    )
    assert isinstance(first, TransactionCommit)
    assert isinstance(replay, RegisterConflict)
    assert store.applied_host_deltas(session_id) == ((intent_id, {"memory": "changed"}),)
