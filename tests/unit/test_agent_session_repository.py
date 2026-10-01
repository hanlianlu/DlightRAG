# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Session snapshot and transaction rules; repository behavior is pinned on PostgreSQL."""

from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

import pytest

from dlightrag.engine.agent.session.entries import UserMessageEntry
from dlightrag.engine.agent.session.graph import AgentSessionGraph
from dlightrag.engine.agent.session.ids import EntryId, LaneId, OperationId, SessionId
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
from dlightrag.engine.agent.session.repository import AgentSessionSnapshot
from dlightrag.engine.agent.session.transactions import (
    RegisterExpectation,
    SessionTransaction,
    TransactionCommit,
)
from tests.in_memory_session_repository import MemoryAgentSessionRepository


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
async def test_snapshot_views_are_built_once_and_follow_each_snapshot() -> None:
    store = MemoryAgentSessionRepository[None]()
    session_id = SessionId.new()
    await _seed(store, session_id)
    branch_id = LaneId.new()
    await _fork_branch(
        store,
        session_id=session_id,
        source_lane_id=LaneId.main(),
        lane_id=branch_id,
    )
    await _append_entries(
        store,
        session_id=session_id,
        lane_id=LaneId.main(),
        expected_head=(await store.load(session_id)).tree.lane().head,
        entries=[_user(session_id, "main delta")],
    )
    snapshot = await store.load(session_id)

    # Each view is built once per immutable snapshot and reused by every query.
    assert snapshot.tree is snapshot.tree
    assert snapshot.graph is snapshot.graph
    assert snapshot.tree.graph is snapshot.tree.graph

    # A replaced snapshot starts without them and derives the same views.
    rebuilt = replace(snapshot)
    assert rebuilt.tree is not snapshot.tree
    assert rebuilt.tree == snapshot.tree
    assert rebuilt.graph == snapshot.graph
    assert rebuilt.tree.ancestry(branch_id) == snapshot.tree.ancestry(branch_id)

    # Selecting another Lane is a new snapshot whose graph follows that Lane.
    branch = replace(snapshot, selected_lane_id=branch_id)

    def contents(entries: tuple[object, ...]) -> list[str]:
        return [entry.content for entry in entries if isinstance(entry, UserMessageEntry)]

    assert contents(snapshot.graph.ancestry()) == ["root", "main delta"]
    assert contents(branch.graph.ancestry()) == ["root"]
    assert contents(snapshot.tree.ancestry(branch_id)) == ["root"]
    assert branch.graph.head_entry_id == snapshot.tree.lane(branch_id).head_entry_id


@pytest.mark.asyncio
async def test_snapshot_graph_reuses_the_lane_tree_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MemoryAgentSessionRepository[None]()
    session_id = SessionId.new()
    await _seed(store, session_id)
    loaded = await store.load(session_id)
    build = AgentSessionGraph.from_entries.__func__  # type: ignore[attr-defined]
    validations: list[SessionId] = []

    def counted(cls: type[AgentSessionGraph], session: SessionId, *args: Any, **kwargs: Any):
        validations.append(session)
        return build(cls, session, *args, **kwargs)

    monkeypatch.setattr(AgentSessionGraph, "from_entries", classmethod(counted))
    snapshot = replace(loaded)

    assert snapshot.graph.nodes is snapshot.tree.graph.nodes
    assert validations == [session_id]

    # Lane registers that cannot form a tree still leave the Entry graph.
    headless = replace(
        loaded,
        registers=tuple(
            record for record in loaded.registers if not isinstance(record.value, LaneState)
        ),
    )
    with pytest.raises(ValueError, match="incomplete"):
        _ = headless.tree
    assert headless.graph.entries == loaded.graph.entries
    assert headless.graph.head_entry_id == loaded.graph.head_entry_id


async def test_a_snapshot_rejects_an_entry_sequence_with_a_gap() -> None:
    store = MemoryAgentSessionRepository[None]()
    session_id = SessionId.new()
    await _seed(store, session_id)
    snapshot = await store.load(session_id)

    with pytest.raises(ValueError, match="gap-free"):
        AgentSessionSnapshot(
            session_id=session_id,
            commit_sequence=1,
            last_entry_sequence=1,
            entries=(replace(snapshot.entries[0], sequence=2),),
            registers=snapshot.registers,
        )


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
