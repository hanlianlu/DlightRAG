# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Immutable Entry Tree and Lane contracts."""

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from dlightrag.engine.agent.session.effects import ToolResultEntry
from dlightrag.engine.agent.session.entries import (
    AssistantMessageEntry,
    ToolResultMessageEntry,
    UserMessageEntry,
)
from dlightrag.engine.agent.session.fold import project_session_messages
from dlightrag.engine.agent.session.ids import AttemptId, EntryId, IntentId, LaneId, SessionId
from dlightrag.engine.agent.session.registers import LaneHead, LaneState, SetRegister
from dlightrag.engine.agent.session.transactions import (
    RegisterExpectation,
    SessionTransaction,
    TransactionCommit,
)
from dlightrag.engine.ai.fingerprints import ModelInvocationFingerprint
from dlightrag.engine.ai.messages import AssistantTurn, ToolCall
from dlightrag.engine.ai.providers.openai_response import response_input
from dlightrag.engine.ai.replay import bind_provider_replay, messages_for_model
from tests.in_memory_session_repository import MemoryAgentSessionRepository


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
async def test_response_fork_reconstructs_the_selected_local_branch() -> None:
    store = MemoryAgentSessionRepository[None]()
    session_id = SessionId.new()
    root = _user(session_id, "base question")
    native_call = {
        "id": "fc-base",
        "type": "function_call",
        "status": "completed",
        "call_id": "call-base",
        "name": "lookup",
        "arguments": '{"value":"base"}',
    }
    fingerprint = ModelInvocationFingerprint("openai", "gpt-test", None, "response")
    bound_call = bind_provider_replay(
        AssistantTurn(
            text="",
            stop_reason="tool_use",
            tool_calls=(ToolCall("call-base", "lookup", {"value": "base"}),),
            provider_state={"response_replay": {"v": 1, "items": [native_call]}},
        ),
        fingerprint,
    )
    call = AssistantMessageEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        parent_entry_id=root.entry_id,
        timestamp=datetime.now(UTC),
        content="",
        stop_reason="tool_use",
        tool_calls=bound_call.tool_calls,
        provider_state=bound_call.provider_state,
    )
    result = ToolResultMessageEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        parent_entry_id=call.entry_id,
        timestamp=datetime.now(UTC),
        result=ToolResultEntry.text(
            tool_name="lookup",
            call_id="call-base",
            outcome="succeeded",
            text="base result",
        ),
        intent_id=IntentId.new(),
        source_index=0,
        contract_version=1,
        input_schema_digest="a" * 64,
        replay_policy="never",
        attempt_id=AttemptId.new(),
        effective_input_digest="b" * 64,
    )
    native_answer = {
        "id": "msg-base",
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "output_text", "text": "base answer"}],
    }
    bound_answer = bind_provider_replay(
        AssistantTurn(
            text="base answer",
            tool_calls=(),
            stop_reason="stop",
            provider_state={"response_replay": {"v": 1, "items": [native_answer]}},
        ),
        fingerprint,
    )
    checkpoint = AssistantMessageEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        parent_entry_id=result.entry_id,
        timestamp=datetime.now(UTC),
        content="base answer",
        stop_reason="stop",
        provider_state=bound_answer.provider_state,
    )
    later_user = UserMessageEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        parent_entry_id=checkpoint.entry_id,
        timestamp=datetime.now(UTC),
        content="main-only question",
    )
    later_answer = AssistantMessageEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        parent_entry_id=later_user.entry_id,
        timestamp=datetime.now(UTC),
        content="main-only answer",
        stop_reason="stop",
    )
    main_head = LaneHead(LaneId.main(), later_answer.entry_id)
    main_state = LaneState(LaneId.main())
    committed = await store.transact(
        session_id=session_id,
        fencing_epoch=1,
        transaction=SessionTransaction.from_parts(
            entries=[root, call, result, checkpoint, later_user, later_answer],
            register_writes=[SetRegister(main_head), SetRegister(main_state)],
            expectations=[
                RegisterExpectation(main_head.ref, None),
                RegisterExpectation(main_state.ref, None),
            ],
        ),
    )
    assert isinstance(committed, TransactionCommit)

    branch_id = LaneId.new()
    branch_head = LaneHead(branch_id, checkpoint.entry_id)
    branch_state = LaneState(branch_id)
    forked = await store.transact(
        session_id=session_id,
        fencing_epoch=1,
        transaction=SessionTransaction.from_parts(
            register_writes=[SetRegister(branch_head), SetRegister(branch_state)],
            expectations=[
                RegisterExpectation(branch_head.ref, None),
                RegisterExpectation(branch_state.ref, None),
            ],
        ),
    )
    assert isinstance(forked, TransactionCommit)
    branch_question = UserMessageEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        parent_entry_id=checkpoint.entry_id,
        timestamp=datetime.now(UTC),
        content="branch question",
    )
    appended = await store.transact(
        session_id=session_id,
        fencing_epoch=1,
        transaction=SessionTransaction.from_parts(
            entries=[branch_question],
            register_writes=[SetRegister(LaneHead(branch_id, branch_question.entry_id))],
            expectations=[
                RegisterExpectation(
                    branch_head.ref, dict(forked.register_sequences)[branch_head.ref]
                )
            ],
        ),
    )
    assert isinstance(appended, TransactionCommit)

    snapshot = await store.load(session_id)
    canonical = project_session_messages(snapshot.tree.ancestry(branch_id), None)

    assert "main-only" not in str(canonical)
    prepared = messages_for_model(canonical, fingerprint)
    assert response_input(prepared) == [
        {"role": "user", "content": "base question"},
        native_call,
        {
            "type": "function_call_output",
            "call_id": "call-base",
            "output": "base result",
        },
        native_answer,
        {"role": "user", "content": "branch question"},
    ]


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


async def _commit_chain(store, session_id: SessionId, entries) -> None:
    """Commit entries as one chain with the Lane's head on the last."""
    head = LaneHead(LaneId.main(), entries[-1].entry_id)
    state = LaneState(LaneId.main())
    outcome = await store.transact(
        session_id=session_id,
        fencing_epoch=1,
        transaction=SessionTransaction.from_parts(
            entries=entries,
            register_writes=[SetRegister(head), SetRegister(state)],
            expectations=[
                RegisterExpectation(head.ref, None),
                RegisterExpectation(state.ref, None),
            ],
        ),
    )
    assert isinstance(outcome, TransactionCommit)


def _call_turn(session_id: SessionId, parent: EntryId, *call_ids: str) -> AssistantMessageEntry:
    return AssistantMessageEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        parent_entry_id=parent,
        timestamp=datetime.now(UTC),
        content="",
        stop_reason="tool_use",
        tool_calls=tuple(ToolCall(call_id, "lookup", {}) for call_id in call_ids),
    )


def _output(session_id: SessionId, parent: EntryId, call_id: str) -> ToolResultMessageEntry:
    return ToolResultMessageEntry(
        entry_id=EntryId.new(),
        session_id=session_id,
        parent_entry_id=parent,
        timestamp=datetime.now(UTC),
        result=ToolResultEntry.text(
            tool_name="lookup", call_id=call_id, outcome="succeeded", text="found"
        ),
        intent_id=IntentId.new(),
        source_index=0,
        contract_version=1,
        input_schema_digest="a" * 64,
        replay_policy="never",
        attempt_id=AttemptId.new(),
        effective_input_digest="b" * 64,
    )


@pytest.mark.asyncio
async def test_settled_ancestry_stops_before_a_turn_whose_calls_have_no_outputs() -> None:
    session_id = SessionId.new()
    question = _user(session_id, "compare the suppliers")
    first = _call_turn(session_id, question.entry_id, "c1")
    first_output = _output(session_id, first.entry_id, "c1")
    running = _call_turn(session_id, first_output.entry_id, "c2", "c3")
    c2_output = _output(session_id, running.entry_id, "c2")
    c3_output = _output(session_id, c2_output.entry_id, "c3")

    def ids(entries) -> list[EntryId]:
        return [entry.entry_id for entry in entries]

    # One of the two parallel calls has its output and the other does not: the whole turn waits.
    partial = MemoryAgentSessionRepository[None]()
    await _commit_chain(partial, session_id, [question, first, first_output, running, c2_output])
    tree = (await partial.load(session_id)).tree
    assert ids(tree.settled_ancestry()) == ids([question, first, first_output])
    assert not tree.is_stable_checkpoint(c2_output.entry_id)

    done = MemoryAgentSessionRepository[None]()
    await _commit_chain(
        done, session_id, [question, first, first_output, running, c2_output, c3_output]
    )
    tree = (await done.load(session_id)).tree
    assert ids(tree.settled_ancestry()) == ids(
        [question, first, first_output, running, c2_output, c3_output]
    )
    assert tree.is_stable_checkpoint(c3_output.entry_id)
