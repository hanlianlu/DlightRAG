# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Immutable Entry Tree, Lane, fencing, and HostDelta contracts."""

from datetime import UTC, datetime

import pytest

from dlightrag.agent.session.entries import AssistantMessageEntry, UserMessageEntry
from dlightrag.agent.session.ids import EntryId, IntentId, LaneId, SessionId
from dlightrag.agent.session.memory import MemoryAgentSessionStore
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
    store = MemoryAgentSessionStore[None]()
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
        await store.fork_lane(
            session_id=session_id,
            source_lane_id=LaneId.main(),
            lane_id=LaneId.new(),
        )


@pytest.mark.asyncio
async def test_archive_keeps_shared_entries_and_blocks_future_writes() -> None:
    store = MemoryAgentSessionStore[None]()
    session_id = SessionId.new()
    await _seed(store, session_id)
    branch_id = LaneId.new()
    await store.fork_lane(
        session_id=session_id,
        source_lane_id=LaneId.main(),
        lane_id=branch_id,
    )
    await store.archive_lane(session_id=session_id, lane_id=branch_id)
    snapshot = await store.load(session_id)
    assert snapshot.tree.lane(branch_id).archived
    assert len(snapshot.entries) == 1
    with pytest.raises(ValueError, match="archived"):
        await store.append_to_lane(
            session_id=session_id,
            lane_id=branch_id,
            expected_head=snapshot.tree.lane(branch_id).head,
            entries=[_user(session_id, "lost")],
        )


@pytest.mark.asyncio
async def test_memory_transaction_commits_typed_host_delta_atomically() -> None:
    store = MemoryAgentSessionStore[dict[str, str]]()
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
    store = MemoryAgentSessionStore[None](fencing_epoch=4)
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
