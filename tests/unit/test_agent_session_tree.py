# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Immutable Entry Tree, Lane, fencing, and HostDelta contracts."""

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from dlightrag.agent.session.entries import AssistantMessageEntry, UserMessageEntry
from dlightrag.agent.session.ids import EntryId, IntentId, LaneId, SessionId
from dlightrag.agent.session.memory import MemoryAgentSessionRepository
from dlightrag.agent.session.registers import LaneHead, LaneState, SetRegister
from dlightrag.agent.session.transactions import (
    HostDeltaSettlement,
    RegisterExpectation,
    SessionTransaction,
    TransactionCommit,
    TransactionLeaseLost,
)
from dlightrag.ai.messages import ToolCall


def _user(session_id: SessionId, content: str) -> UserMessageEntry:
    return UserMessageEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        timestamp=datetime.now(UTC),
        content=content,
    )


async def _fork_branch(store, session_id: SessionId, lane_id: LaneId) -> None:
    snapshot = await store.load(session_id)
    target = snapshot.tree.lane().head_entry_id
    if target is None or not snapshot.tree.is_stable_checkpoint(target):
        raise ValueError("a Lane can fork only from a stable checkpoint")
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


async def _seed(store, session_id: SessionId, entry=None):
    entry = entry or _user(session_id, "root")
    head = LaneHead(LaneId.main(), entry.entry_id)
    state = LaneState(LaneId.main())
    outcome = await store.transact(
        session_id=session_id,
        fencing_epoch=1,
        transaction=SessionTransaction.from_parts(
            entries=[entry],
            register_writes=[SetRegister(head), SetRegister(state)],
            expectations=[
                RegisterExpectation(head.ref, None),
                RegisterExpectation(state.ref, None),
            ],
        ),
    )
    assert isinstance(outcome, TransactionCommit)


@pytest.mark.asyncio
async def test_fork_requires_a_stable_checkpoint() -> None:
    store = MemoryAgentSessionRepository[None]()
    session_id = SessionId.new()
    assistant = AssistantMessageEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        timestamp=datetime.now(UTC),
        content="",
        stop_reason="tool_use",
        tool_calls=(ToolCall("c1", "lookup", {}),),
    )
    await _seed(store, session_id, assistant)
    with pytest.raises(ValueError, match="stable checkpoint"):
        await _fork_branch(store, session_id, LaneId.new())


@pytest.mark.asyncio
async def test_archive_keeps_shared_entries_and_blocks_future_writes() -> None:
    store = MemoryAgentSessionRepository[None]()
    session_id = SessionId.new()
    await _seed(store, session_id)
    branch_id = LaneId.new()
    await _fork_branch(store, session_id, branch_id)
    before_archive = (await store.load(session_id)).tree.lane(branch_id)
    archived_state = replace(before_archive.state.value, archived=True)
    assert isinstance(archived_state, LaneState)
    archived = await store.transact(
        session_id=session_id,
        fencing_epoch=1,
        transaction=SessionTransaction.from_parts(
            register_writes=[SetRegister(archived_state)],
            expectations=[
                RegisterExpectation(before_archive.state.ref, before_archive.state.sequence)
            ],
        ),
    )
    assert isinstance(archived, TransactionCommit)
    snapshot = await store.load(session_id)
    assert snapshot.tree.lane(branch_id).archived
    assert len(snapshot.entries) == 1
    with pytest.raises(ValueError, match="archived"):
        branch = snapshot.tree.lane(branch_id)
        entry = replace(_user(session_id, "lost"), parent_entry_id=branch.head_entry_id)
        await store.transact(
            session_id=session_id,
            fencing_epoch=1,
            transaction=SessionTransaction.from_parts(
                entries=[entry],
                register_writes=[SetRegister(LaneHead(branch_id, entry.entry_id))],
                expectations=[
                    RegisterExpectation(branch.head.ref, branch.head.sequence),
                    RegisterExpectation(branch.state.ref, branch.state.sequence),
                ],
            ),
        )


@pytest.mark.asyncio
async def test_memory_transaction_commits_typed_host_delta_atomically() -> None:
    store = MemoryAgentSessionRepository[dict[str, str]]()
    session_id = SessionId.new()
    head = LaneHead(LaneId.main(), None)
    state = LaneState(LaneId.main())
    intent_id = IntentId.new()
    outcome = await store.transact(
        session_id=session_id,
        fencing_epoch=1,
        transaction=SessionTransaction.from_parts(
            register_writes=[SetRegister(head), SetRegister(state)],
            expectations=[
                RegisterExpectation(head.ref, None),
                RegisterExpectation(state.ref, None),
            ],
            host_delta=HostDeltaSettlement(intent_id, {"evidence": "added"}),
        ),
    )
    assert isinstance(outcome, TransactionCommit)
    assert store.applied_host_deltas(session_id) == ((intent_id, {"evidence": "added"}),)


@pytest.mark.asyncio
async def test_transferred_lease_fences_old_epoch() -> None:
    store = MemoryAgentSessionRepository[None](fencing_epoch=4)
    store.transfer_lease(5)
    session_id = SessionId.new()
    head = LaneHead(LaneId.main(), None)
    state = LaneState(LaneId.main())
    outcome = await store.transact(
        session_id=session_id,
        fencing_epoch=4,
        transaction=SessionTransaction.from_parts(
            register_writes=[SetRegister(head), SetRegister(state)],
            expectations=[
                RegisterExpectation(head.ref, None),
                RegisterExpectation(state.ref, None),
            ],
        ),
    )
    assert isinstance(outcome, TransactionLeaseLost)
    assert (await store.load(session_id)).commit_sequence == 0
