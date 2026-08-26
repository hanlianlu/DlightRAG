# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Seam tests for parent-linked Session Trees and exact Lane CAS."""

from datetime import UTC, datetime

import pytest

from dlightrag.agent.session.effects import ToolResultEntry
from dlightrag.agent.session.entries import (
    AssistantMessageEntry,
    EffectResultEntry,
    UserMessageEntry,
)
from dlightrag.agent.session.ids import EntryId, LaneId, SessionId
from dlightrag.agent.session.memory import MemoryAgentSessionStore
from dlightrag.agent.session.registers import LaneHead
from dlightrag.agent.session.transactions import (
    RegisterConflict,
    TransactionCommit,
    TransactionLeaseLost,
)
from dlightrag.ai.messages import ToolCall


def _now() -> datetime:
    return datetime.now(UTC)


def _user(session_id: SessionId, content: str) -> UserMessageEntry:
    return UserMessageEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        timestamp=_now(),
        content=content,
    )


def _user_contents(entries: tuple[object, ...]) -> list[object]:
    assert all(isinstance(entry, UserMessageEntry) for entry in entries)
    return [entry.content for entry in entries if isinstance(entry, UserMessageEntry)]


@pytest.mark.asyncio
async def test_forked_lanes_share_ancestry_and_advance_without_global_conflicts() -> None:
    store = MemoryAgentSessionStore[None]()
    session_id = SessionId.new()
    first = await store.append(
        session_id=session_id,
        expected_version=0,
        entries=[_user(session_id, "root")],
    )
    assert first.__class__.__name__ == "SessionCommit"
    root_snapshot = await store.load(session_id)
    main_at_root = root_snapshot.tree.lane()

    fork_id = LaneId.new()
    forked = await store.fork_lane(
        session_id=session_id,
        source_lane_id=LaneId.main(),
        lane_id=fork_id,
    )
    assert isinstance(forked, TransactionCommit)
    after_fork = await store.load(session_id)
    branch_at_root = after_fork.tree.lane(fork_id)

    main_commit = await store.append_to_lane(
        session_id=session_id,
        lane_id=LaneId.main(),
        expected_head=main_at_root.head,
        entries=[_user(session_id, "main future")],
    )
    assert isinstance(main_commit, TransactionCommit)

    # The unrelated main-Lane commit advanced the Session commit sequence but
    # did not invalidate the fork Lane's exact head token.
    branch_commit = await store.append_to_lane(
        session_id=session_id,
        lane_id=fork_id,
        expected_head=branch_at_root.head,
        entries=[_user(session_id, "branch future")],
    )
    assert isinstance(branch_commit, TransactionCommit)

    final = await store.load(session_id)
    assert _user_contents(final.tree.ancestry(LaneId.main())) == [
        "root",
        "main future",
    ]
    assert _user_contents(final.tree.ancestry(fork_id)) == [
        "root",
        "branch future",
    ]
    main_leaf = final.tree.lane().head_entry_id
    branch_leaf = final.tree.lane(fork_id).head_entry_id
    assert main_leaf is not None
    assert main_leaf != branch_leaf
    assert (
        final.graph.select_head(main_leaf).ancestry()[-1].parent_entry_id
        == root_snapshot.entries[0].entry_id
    )  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_same_lane_stale_head_is_an_exact_register_conflict() -> None:
    store = MemoryAgentSessionStore[None]()
    session_id = SessionId.new()
    await store.append(
        session_id=session_id,
        expected_version=0,
        entries=[_user(session_id, "root")],
    )
    stale = (await store.load(session_id)).tree.lane().head
    first = await store.append_to_lane(
        session_id=session_id,
        lane_id=LaneId.main(),
        expected_head=stale,
        entries=[_user(session_id, "first")],
    )
    assert isinstance(first, TransactionCommit)
    conflict = await store.append_to_lane(
        session_id=session_id,
        lane_id=LaneId.main(),
        expected_head=stale,
        entries=[_user(session_id, "lost")],
    )
    assert isinstance(conflict, RegisterConflict)
    assert conflict.ref == LaneHead(LaneId.main(), None).ref


@pytest.mark.asyncio
async def test_fork_rejects_a_head_with_unmatched_tool_calls() -> None:
    store = MemoryAgentSessionStore[None]()
    session_id = SessionId.new()
    await store.append(
        session_id=session_id,
        expected_version=0,
        entries=[_user(session_id, "search")],
    )
    head = (await store.load(session_id)).tree.lane().head
    assistant = AssistantMessageEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        timestamp=_now(),
        content="",
        stop_reason="tool_use",
        tool_calls=(ToolCall(id="call-1", name="search", arguments={"q": "x"}),),
    )
    await store.append_to_lane(
        session_id=session_id,
        lane_id=LaneId.main(),
        expected_head=head,
        entries=[assistant],
    )
    with pytest.raises(ValueError, match="stable checkpoint"):
        await store.fork_lane(
            session_id=session_id,
            source_lane_id=LaneId.main(),
            lane_id=LaneId.new(),
        )

    after_assistant = (await store.load(session_id)).tree.lane().head
    result = EffectResultEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        timestamp=_now(),
        result=ToolResultEntry.text(
            tool_name="search",
            call_id="call-1",
            outcome="succeeded",
            text="found",
        ),
    )
    await store.append_to_lane(
        session_id=session_id,
        lane_id=LaneId.main(),
        expected_head=after_assistant,
        entries=[result],
    )
    accepted = await store.fork_lane(
        session_id=session_id,
        source_lane_id=LaneId.main(),
        lane_id=LaneId.new(),
    )
    assert isinstance(accepted, TransactionCommit)


@pytest.mark.asyncio
async def test_archiving_a_lane_keeps_shared_entries_and_main_is_not_archivable() -> None:
    store = MemoryAgentSessionStore[None]()
    session_id = SessionId.new()
    await store.append(
        session_id=session_id,
        expected_version=0,
        entries=[_user(session_id, "root")],
    )
    lane_id = LaneId.new()
    await store.fork_lane(
        session_id=session_id,
        source_lane_id=LaneId.main(),
        lane_id=lane_id,
    )
    archived = await store.archive_lane(session_id=session_id, lane_id=lane_id)
    assert isinstance(archived, TransactionCommit)
    snapshot = await store.load(session_id)
    archived_lane = snapshot.tree.lane(lane_id)
    assert archived_lane.archived is True
    assert len(snapshot.entries) == 1
    with pytest.raises(ValueError, match="archived Lane"):
        await store.append_to_lane(
            session_id=session_id,
            lane_id=lane_id,
            expected_head=archived_lane.head,
            entries=[_user(session_id, "must not append")],
        )
    with pytest.raises(ValueError, match="main Lane"):
        await store.archive_lane(session_id=session_id, lane_id=LaneId.main())


@pytest.mark.asyncio
async def test_fork_rejects_an_empty_main_lane() -> None:
    from dlightrag.agent.session.registers import LaneHead, LaneState, SetRegister
    from dlightrag.agent.session.transactions import RegisterExpectation, SessionTransaction

    store = MemoryAgentSessionStore[None]()
    session_id = SessionId.new()
    head = LaneHead(LaneId.main(), None)
    state = LaneState(LaneId.main())
    await store.transact(
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
    with pytest.raises(ValueError, match="empty Head"):
        await store.fork_lane(
            session_id=session_id,
            source_lane_id=LaneId.main(),
            lane_id=LaneId.new(),
        )


def test_main_lane_registers_cannot_be_deleted_or_archived() -> None:
    from dlightrag.agent.session.registers import (
        DeleteRegister,
        LaneHead,
        LaneState,
        SetRegister,
    )
    from dlightrag.agent.session.transactions import RegisterExpectation, SessionTransaction

    head = LaneHead(LaneId.main(), None)
    with pytest.raises(ValueError, match="cannot be deleted"):
        SessionTransaction.from_parts(
            register_writes=[DeleteRegister(head.ref)],
            expectations=[RegisterExpectation(head.ref, 1)],
        )
    with pytest.raises(ValueError, match="cannot be archived"):
        SessionTransaction.from_parts(
            register_writes=[SetRegister(LaneState(LaneId.main(), archived=True))],
            expectations=[RegisterExpectation(LaneState(LaneId.main()).ref, 1)],
        )


@pytest.mark.asyncio
async def test_memory_transaction_rejects_unconsumed_host_delta_before_mutation() -> None:
    from dlightrag.agent.session.registers import LaneHead, LaneState, SetRegister
    from dlightrag.agent.session.transactions import RegisterExpectation, SessionTransaction

    store = MemoryAgentSessionStore[object]()
    session_id = SessionId.new()
    head = LaneHead(LaneId.main(), None)
    state = LaneState(LaneId.main())
    with pytest.raises(TypeError, match="HostDelta"):
        await store.transact(
            session_id=session_id,
            fencing_epoch=1,
            transaction=SessionTransaction.from_parts(
                register_writes=[SetRegister(head), SetRegister(state)],
                expectations=[
                    RegisterExpectation(head.ref, None),
                    RegisterExpectation(state.ref, None),
                ],
                host_delta=object(),
            ),
        )
    assert (await store.load(session_id)).commit_sequence == 0


def test_entry_transaction_requires_a_lane_head_advance() -> None:
    from dlightrag.agent.session.transactions import SessionTransaction

    with pytest.raises(ValueError, match="advance a Lane Head"):
        SessionTransaction.from_parts(
            entries=[_user(SessionId.new(), "orphan")],
        )


@pytest.mark.asyncio
async def test_transferred_lease_fences_old_transaction_epoch() -> None:
    store = MemoryAgentSessionStore[None](fencing_epoch=4)
    session_id = SessionId.new()
    store.transfer_lease(5)
    from dlightrag.agent.session.registers import LaneHead, LaneState, SetRegister
    from dlightrag.agent.session.transactions import RegisterExpectation, SessionTransaction

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
    assert (await store.load(session_id)).entries == ()
