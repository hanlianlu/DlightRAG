# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Integration tests for server-owned Web conversation command events on PostgreSQL 18.

Exercises the real contract against a live database: the appended migration and its
constraints, the sequence counter commands share with turns, the merged history page and
its cursor, owner scoping, and the conversation cascade. Every test runs inside a
throwaway database created and dropped per test, so the configured administrative
database is never mutated. Connection settings come from the shared integration-test
PostgreSQL environment; skipped if unavailable.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from dlightrag.adapters.postgres.answer.goals import PGAnswerGoalStore
from dlightrag.adapters.postgres.runtime.run_store import PGRunStore
from dlightrag.adapters.postgres.web.web_conversations import (
    PGWebConversationStore,
)
from dlightrag.application.web_conversations.models import (
    AgentspaceSnapshotChanges,
    CommandTone,
    ConversationHistoryCursor,
    ConversationHistoryPage,
    ConversationHistoryPageRequest,
    ConversationQueueConflict,
    GoalRoundGuard,
    QueuedDeliveryGuard,
    queued_delivery_digest,
)
from dlightrag.engine.answer.goal_service import AnswerGoalService
from dlightrag.engine.answer.runs.envelope import accepted_input_envelope
from dlightrag.engine.runtime.records import (
    PreparedRunEnvelope,
    RunAccessScope,
    SessionWorkBusyError,
    run_request_fingerprint,
)
from tests.support.pg import (
    PG_CONN_KWARGS,
    drop_database,
    skip_without_postgres,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
]

_OWNER = "principal-alpha"
_OTHER_OWNER = "principal-beta"
_MAX_COMMANDS = 100


@pytest.fixture
async def db_name() -> AsyncIterator[str]:
    """Provision an isolated throwaway database and yield its name."""
    await skip_without_postgres()

    name = f"dlightrag_webcommands_{uuid.uuid4().hex[:12]}"
    admin = await asyncpg.connect(**PG_CONN_KWARGS)
    try:
        await admin.execute(f'CREATE DATABASE "{name}"')
    finally:
        await admin.close()
    try:
        yield name
    finally:
        await drop_database(name)


@pytest.fixture
async def pool(db_name: str) -> AsyncIterator[Any]:
    """Yield a pool bound to this test's throwaway database."""
    created = await asyncpg.create_pool(
        **{**PG_CONN_KWARGS, "database": db_name}, min_size=1, max_size=8
    )
    try:
        yield created
    finally:
        await created.close()


@pytest.fixture
async def runs(pool: Any) -> PGRunStore:
    store = PGRunStore(pool=pool)
    await store.initialize()
    return store


@pytest.fixture
async def store(pool: Any, runs: PGRunStore) -> PGWebConversationStore:
    created = PGWebConversationStore(pool=pool, run_store=runs)
    await created.initialize()
    return created


def _request(query: str = "why", **extra: Any) -> dict[str, Any]:
    return {
        "query": query,
        "workspaces": ["alpha"],
        "history": [],
        "agent_session_id": "00000000-0000-7000-8000-000000000001",
        "agent_lane_id": "main",
        **extra,
    }


def _turn_envelope(
    *, owner: str, request: dict[str, Any], submission_id: str, fingerprint: str
) -> PreparedRunEnvelope:
    return PreparedRunEnvelope(
        run_kind="answer",
        lane="query",
        submitted_by=owner,
        access_scope=RunAccessScope(kind="owner", scope_id=owner),
        submission_key=submission_id,
        request_fingerprint=fingerprint,
        payload=request,
        accepted_input=accepted_input_envelope(request),
        retention_seconds=365 * 24 * 60 * 60,
    )


async def _conversation(store: PGWebConversationStore, owner: str = _OWNER) -> str:
    row = await store.create_conversation(owner)
    return str(row["conversation_id"])


async def _submit(store: PGWebConversationStore, conversation_id: str, query: str) -> None:
    """Accept one submission so the conversation carries a turn at its allocated number."""
    request = {**_request(query), "agent_session_id": conversation_id}
    submission_id = str(uuid.uuid4())
    created = await store.create_answer_turn(
        principal_id=_OWNER,
        conversation_id=conversation_id,
        submission_id=submission_id,
        envelope=_turn_envelope(
            owner=_OWNER,
            request=request,
            submission_id=submission_id,
            fingerprint=run_request_fingerprint(request),
        ),
        run_id=str(uuid.uuid7()),
        title_hint="why",
    )
    assert created is not None


async def _append(
    store: PGWebConversationStore,
    conversation_id: str,
    *,
    name: str = "goal",
    input: str = "",
    result: str = "No goal in this conversation. /goal <objective> creates one.",
    tone: CommandTone = "success",
    result_data: dict[str, Any] | None = None,
    owner: str = _OWNER,
):
    return await store.append_command(
        owner,
        conversation_id,
        name=name,
        command_input=input,
        result=result,
        tone=tone,
        result_data=result_data,
    )


async def _count(pool: Any, table: str, **where: Any) -> int:
    clause = " AND ".join(f"{column} = ${index}" for index, column in enumerate(where, start=1))
    async with pool.acquire() as conn:
        return int(
            await conn.fetchval(
                f"SELECT count(*)::int FROM {table} "  # noqa: S608 - fixed table/column names
                + (f"WHERE {clause}" if clause else ""),
                *where.values(),
            )
        )


async def _history(
    store: PGWebConversationStore,
    conversation_id: str,
    *,
    limit: int,
    cursor: ConversationHistoryCursor | None = None,
    owner: str = _OWNER,
) -> ConversationHistoryPage | None:
    return await store.history_page(
        owner,
        conversation_id,
        page=ConversationHistoryPageRequest(limit=limit, cursor=cursor),
    )


# ---------------------------------------------------------------------------
# The shared counter and its constraints
# ---------------------------------------------------------------------------


async def test_a_command_allocates_the_conversations_shared_sequence(
    store: PGWebConversationStore, pool: Any
) -> None:
    conversation_id = await _conversation(store)

    first = await _append(
        store,
        conversation_id,
        result_data={"kind": "goal_status", "goal": None},
    )
    second = await _append(
        store,
        conversation_id,
        input="cap 12",
        result="Ongoing goal\nObjective: ship it\nRound: 0/12",
        result_data={
            "kind": "goal_status",
            "goal": {
                "objective": "ship it",
                "phase": "active",
                "max_rounds": 12,
                "rounds_started": 0,
                "blocked_reason": None,
            },
        },
    )
    legacy = await _append(store, conversation_id, input="clear", result="Goal cleared.")

    assert first is not None and second is not None and legacy is not None
    assert (first.sequence, second.sequence, legacy.sequence) == (1, 2, 3)
    assert (first.name, first.input, first.result, first.tone) == (
        "goal",
        "",
        "No goal in this conversation. /goal <objective> creates one.",
        "success",
    )
    assert first.result_data == {"kind": "goal_status", "goal": None}
    assert second.result_data is not None
    assert second.result_data["goal"]["max_rounds"] == 12
    assert second.result_data["goal"]["blocked_reason"] is None
    assert legacy.result_data is None
    assert first.created_at.tzinfo is not None
    head = await store.submission_seed(_OWNER, conversation_id, attachment_limit=0)
    assert head is not None and head.head.content_revision == 3
    assert await _count(pool, "web_conversation_commands") == 3


async def test_commands_and_turns_share_one_ordering_space(
    store: PGWebConversationStore, pool: Any, runs: PGRunStore
) -> None:
    conversation_id = await _conversation(store)
    await _submit(store, conversation_id, "first turn")
    claim = await runs.claim_next(worker_id="worker-order")
    assert claim is not None
    await runs.finish_success(
        owner_id=_OWNER,
        run_id=claim.run.run_id,
        worker_id="worker-order",
        fencing_epoch=claim.run.fencing_epoch,
        result={"answer": "done"},
    )
    await _append(store, conversation_id, input="pause", result="Paused goal")
    await _submit(store, conversation_id, "second turn")
    await _append(store, conversation_id, input="resume", result="Ongoing goal")

    page = await _history(store, conversation_id, limit=3)

    assert page is not None
    assert [turn.turn_number for turn in page.turns] == [3]
    assert [command.sequence for command in page.commands] == [2, 4]
    assert page.next_cursor == ConversationHistoryCursor(
        conversation_id=uuid.UUID(conversation_id),
        before_sequence=2,
    )
    second = await _history(store, conversation_id, limit=3, cursor=page.next_cursor)
    assert second is not None
    assert [turn.turn_number for turn in second.turns] == [1]
    assert second.commands == ()
    assert second.next_cursor is None
    # The stored ordering number is the turn number: one counter, no overlap, no gap.
    assert await _count(pool, "web_conversation_commands") == 2
    assert await _count(pool, "web_conversation_turns") == 2


async def test_command_history_pages_traverse_the_sequence_space_once(
    store: PGWebConversationStore,
) -> None:
    conversation_id = await _conversation(store)
    for index in range(1, 6):
        await _append(
            store,
            conversation_id,
            input=f"cap {index}",
            result=f"rounds {index}",
            result_data={"kind": "goal_invalid", "reason": "cap"},
        )

    seen: list[int] = []
    cursor: ConversationHistoryCursor | None = None
    while True:
        page = await _history(store, conversation_id, limit=2, cursor=cursor)
        assert page is not None
        assert page.turns == ()
        page_sequences = [command.sequence for command in page.commands]
        assert page_sequences == sorted(page_sequences)  # chronological in the model
        assert all(
            command.result_data == {"kind": "goal_invalid", "reason": "cap"}
            for command in page.commands
        )
        seen.extend(page_sequences)
        cursor = page.next_cursor
        if cursor is None:
            break

    assert sorted(seen) == [1, 2, 3, 4, 5]
    assert len(seen) == len(set(seen))
    assert len(seen) <= _MAX_COMMANDS


async def test_a_foreign_principal_never_appends_or_reads_commands(
    store: PGWebConversationStore,
) -> None:
    conversation_id = await _conversation(store, owner=_OWNER)
    first = await _append(store, conversation_id)
    assert first is not None

    assert await _append(store, conversation_id, owner=_OTHER_OWNER) is None
    assert await _history(store, conversation_id, limit=10, owner=_OTHER_OWNER) is None
    assert await _history(store, conversation_id, limit=10, owner=_OWNER) is not None


async def test_the_command_table_rejects_an_unknown_tone_and_a_foreign_conversation(
    store: PGWebConversationStore, pool: Any
) -> None:
    conversation_id = await _conversation(store)
    async with pool.acquire() as conn:
        with pytest.raises(asyncpg.exceptions.CheckViolationError):
            await conn.execute(
                """
                INSERT INTO web_conversation_commands
                    (principal_id, conversation_id, command_id, sequence, name, input, result, tone)
                VALUES ($1, $2::uuid, $3::uuid, 1, 'goal', '', 'text', 'maybe')
                """,
                _OWNER,
                conversation_id,
                str(uuid.uuid4()),
            )
        with pytest.raises(asyncpg.exceptions.ForeignKeyViolationError):
            await conn.execute(
                """
                INSERT INTO web_conversation_commands
                    (principal_id, conversation_id, command_id, sequence, name, input, result, tone)
                VALUES ($1, $2::uuid, $3::uuid, 1, 'goal', '', 'text', 'success')
                """,
                _OWNER,
                str(uuid.uuid4()),
                str(uuid.uuid4()),
            )


async def test_deleting_a_conversation_cascades_its_commands(
    store: PGWebConversationStore, pool: Any
) -> None:
    conversation_id = await _conversation(store)
    await _submit(store, conversation_id, "a turn")
    await _append(store, conversation_id)
    assert await _count(pool, "web_conversation_commands") == 1

    assert await store.delete_conversation(_OWNER, conversation_id) is True

    assert await _count(pool, "web_conversation_commands") == 0
    assert await _count(pool, "web_conversation_turns") == 0


# ---------------------------------------------------------------------------
# Messages queued behind a live Run
# ---------------------------------------------------------------------------


async def test_queued_messages_round_trip_in_position_order(
    store: PGWebConversationStore,
) -> None:
    conversation_id = await _conversation(store)
    first = await store.append_queued_message(_OWNER, conversation_id, content="first")
    second = await store.append_queued_message(_OWNER, conversation_id, content="second")
    assert first is not None and second is not None
    assert (first.position, second.position) == (1, 2)

    messages = await store.queue_messages(_OWNER, conversation_id)

    assert messages is not None
    assert [message.content for message in messages] == ["first", "second"]
    assert [message.position for message in messages] == [1, 2]
    assert first.created_at.tzinfo is not None


async def test_queued_messages_are_owner_scoped(
    store: PGWebConversationStore,
) -> None:
    conversation_id = await _conversation(store)
    assert await store.append_queued_message(_OWNER, conversation_id, content="mine")

    assert await store.queue_messages(_OTHER_OWNER, conversation_id) is None
    assert (
        await store.append_queued_message(_OTHER_OWNER, conversation_id, content="theirs") is None
    )


async def test_queued_message_edit_remove_and_reorder(
    store: PGWebConversationStore,
) -> None:
    conversation_id = await _conversation(store)
    first = await store.append_queued_message(_OWNER, conversation_id, content="a")
    second = await store.append_queued_message(_OWNER, conversation_id, content="b")
    third = await store.append_queued_message(_OWNER, conversation_id, content="c")
    assert first is not None and second is not None and third is not None

    edited = await store.edit_queued_message(
        _OWNER, conversation_id, first.message_id, content="a2"
    )
    assert edited is not None
    assert edited.content == "a2"
    assert edited.position == first.position

    reordered = await store.reorder_queued_messages(
        _OWNER,
        conversation_id,
        message_ids=[third.message_id, first.message_id, second.message_id],
    )
    assert reordered is not None
    assert [message.content for message in reordered] == ["c", "a2", "b"]
    assert [message.position for message in reordered] == [1, 2, 3]

    assert await store.remove_queued_message(_OWNER, conversation_id, second.message_id)
    assert not await store.remove_queued_message(_OWNER, conversation_id, second.message_id)

    messages = await store.queue_messages(_OWNER, conversation_id)
    assert messages is not None
    assert [message.content for message in messages] == ["c", "a2"]


async def test_a_stale_reorder_set_is_a_conflict(
    store: PGWebConversationStore,
) -> None:
    conversation_id = await _conversation(store)
    first = await store.append_queued_message(_OWNER, conversation_id, content="a")
    second = await store.append_queued_message(_OWNER, conversation_id, content="b")
    assert first is not None and second is not None

    with pytest.raises(ConversationQueueConflict):
        await store.reorder_queued_messages(_OWNER, conversation_id, message_ids=[first.message_id])
    with pytest.raises(ConversationQueueConflict):
        await store.reorder_queued_messages(
            _OWNER,
            conversation_id,
            message_ids=[first.message_id, first.message_id],
        )


async def test_deleting_a_conversation_drops_its_queue(
    store: PGWebConversationStore,
    pool: Any,
) -> None:
    conversation_id = await _conversation(store)
    assert await store.append_queued_message(_OWNER, conversation_id, content="a")
    assert await _count(pool, "web_conversation_queue") == 1

    assert await store.delete_conversation(_OWNER, conversation_id)

    assert await _count(pool, "web_conversation_queue") == 0


# ---------------------------------------------------------------------------
# Admission reads the live Run
# ---------------------------------------------------------------------------


async def test_live_answer_run_is_the_conversations_non_terminal_answer(
    store: PGWebConversationStore,
    pool: Any,
) -> None:
    conversation_id = await _conversation(store)
    assert await store.live_answer_run(_OWNER, conversation_id) is None

    await _submit(store, conversation_id, "why")
    live = await store.live_answer_run(_OWNER, conversation_id)

    assert live is not None
    assert live.status == "queued"
    # A human turn has no continuation kind; a goal round states the one pause needs.
    assert live.continuation_kind is None
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE dlightrag_runs"
            " SET accepted_input_json ="
            "   COALESCE(accepted_input_json, '{}'::jsonb) ||"
            '   \'{"continuation_kind": "goal_round"}\'::jsonb'
            " WHERE run_id = ("
            "   SELECT answer_run_id FROM web_conversation_turns"
            "   WHERE principal_id = $1 AND conversation_id = $2::text::uuid"
            " )",
            _OWNER,
            conversation_id,
        )
    live = await store.live_answer_run(_OWNER, conversation_id)
    assert live is not None and live.continuation_kind == "goal_round"
    assert await store.live_answer_run(_OTHER_OWNER, conversation_id) is None

    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE dlightrag_runs"
            " SET status = 'succeeded', prepared_input_json = NULL,"
            " result_json = '{}'::jsonb, finished_at = NOW()"
            " WHERE run_id = ("
            "   SELECT answer_run_id FROM web_conversation_turns"
            "   WHERE principal_id = $1 AND conversation_id = $2::text::uuid"
            " )",
            _OWNER,
            conversation_id,
        )

    assert await store.live_answer_run(_OWNER, conversation_id) is None


async def test_a_conversations_session_is_read_before_it_is_deleted(
    store: PGWebConversationStore,
) -> None:
    conversation_id = await _conversation(store)

    session_id = await store.conversation_agent_session(_OWNER, conversation_id)
    assert session_id is not None
    assert session_id in await store.agent_sessions(_OWNER)
    assert await store.conversation_agent_session(_OTHER_OWNER, conversation_id) is None

    assert await store.delete_conversation(_OWNER, conversation_id) is True
    assert await store.conversation_agent_session(_OWNER, conversation_id) is None
    assert session_id not in await store.agent_sessions(_OWNER)


async def _work_head(pool: Any, session_id: str) -> str | None:
    async with pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT active_run_id::text FROM dlightrag_session_work_heads"
            " WHERE owner_id = $1 AND session_id = $2::text::uuid",
            _OWNER,
            session_id,
        )


async def test_a_session_admits_one_live_conversation_run_at_a_time(
    store: PGWebConversationStore, pool: Any, runs: PGRunStore
) -> None:
    conversation_id = await _conversation(store)
    await _submit(store, conversation_id, "the first phase")
    active = await _work_head(pool, conversation_id)
    assert active is not None

    with pytest.raises(SessionWorkBusyError) as busy:
        await _submit(store, conversation_id, "a competing phase")
    assert busy.value.active_run_id == active

    # A different conversation is a different Session: it admits independently.
    other = await _conversation(store)
    await _submit(store, other, "another case")

    claim = await runs.claim_next(worker_id="worker-head")
    assert claim is not None
    assert claim.run.run_id == active
    await runs.finish_success(
        owner_id=_OWNER,
        run_id=active,
        worker_id="worker-head",
        fencing_epoch=claim.run.fencing_epoch,
        result={"answer": "phase done"},
    )

    assert await _work_head(pool, conversation_id) is None
    await _submit(store, conversation_id, "the next phase")
    assert await _work_head(pool, conversation_id) is not None


async def test_a_failed_or_deleted_conversation_run_releases_the_session_slot(
    store: PGWebConversationStore, pool: Any, runs: PGRunStore
) -> None:
    conversation_id = await _conversation(store)
    await _submit(store, conversation_id, "a phase that fails")
    active = await _work_head(pool, conversation_id)
    assert active is not None
    claim = await runs.claim_next(worker_id="worker-head")
    assert claim is not None
    await runs.finish_failure(
        owner_id=_OWNER,
        run_id=claim.run.run_id,
        worker_id="worker-head",
        fencing_epoch=claim.run.fencing_epoch,
        error_kind="ANSWER_STREAM_FAILED",
        error_message="phase failed",
    )
    assert await _work_head(pool, conversation_id) is None
    await _submit(store, conversation_id, "a retried phase")

    # Deleting the conversation while its Run is live takes the slot with it.
    assert await store.delete_conversation(_OWNER, conversation_id) is True
    assert await _work_head(pool, conversation_id) is None


async def _deliver_queue(
    store: PGWebConversationStore,
    conversation_id: str,
    guard: QueuedDeliveryGuard,
) -> Any:
    """Accept one coalesced queued turn through the same fenced transaction a delivery uses."""
    request = {**_request("queued words"), "agent_session_id": conversation_id}
    submission_id = str(uuid.uuid4())
    return await store.create_answer_turn(
        principal_id=_OWNER,
        conversation_id=conversation_id,
        submission_id=submission_id,
        envelope=_turn_envelope(
            owner=_OWNER,
            request=request,
            submission_id=submission_id,
            fingerprint=run_request_fingerprint(request),
        ),
        run_id=str(uuid.uuid7()),
        title_hint="why",
        queue_guard=guard,
    )


async def test_a_queued_delivery_is_fenced_by_the_queue_it_read(
    store: PGWebConversationStore, pool: Any
) -> None:
    conversation_id = await _conversation(store)
    first = await store.append_queued_message(_OWNER, conversation_id, content="the first words")
    second = await store.append_queued_message(_OWNER, conversation_id, content="the second words")
    assert first is not None and second is not None
    stale = QueuedDeliveryGuard(
        message_ids=(first.message_id, second.message_id),
        digest=queued_delivery_digest([first, second]),
    )

    # The reader edits a message after the delivery read the queue: the transaction must
    # refuse the stale prompt, leave the queue, and admit nothing.
    await store.edit_queued_message(
        _OWNER, conversation_id, second.message_id, content="edited words"
    )
    assert await _deliver_queue(store, conversation_id, stale) is None
    assert await _count(pool, "web_conversation_turns") == 0
    assert await _count(pool, "web_conversation_queue") == 2

    messages = await store.queue_messages(_OWNER, conversation_id)
    assert messages is not None
    fresh = QueuedDeliveryGuard(
        message_ids=tuple(message.message_id for message in messages),
        digest=queued_delivery_digest(messages),
    )
    accepted = await _deliver_queue(store, conversation_id, fresh)
    assert accepted is not None
    # The words and the Run that answers them committed together, and the rows are gone.
    assert await store.queue_messages(_OWNER, conversation_id) == ()
    assert await _count(pool, "web_conversation_turns") == 1


async def test_two_simultaneous_acceptances_admit_exactly_one(
    store: PGWebConversationStore, runs: PGRunStore
) -> None:
    conversation_id = await _conversation(store)

    results = await asyncio.gather(
        _submit(store, conversation_id, "the human phase"),
        _submit(store, conversation_id, "the goal round"),
        return_exceptions=True,
    )

    successes = [result for result in results if not isinstance(result, BaseException)]
    refused = [result for result in results if isinstance(result, SessionWorkBusyError)]
    assert len(successes) == 1
    assert len(refused) == 1


async def test_the_workspace_inspector_reads_heads_snapshots_and_file_history(
    store: PGWebConversationStore, pool: Any
) -> None:
    from dlightrag.engine.runtime.records import SessionWorkBusyError

    conversation_id = await _conversation(store)
    session_id = await store.conversation_agent_session(_OWNER, conversation_id)
    assert session_id is not None
    first = str(uuid.uuid7())
    second = str(uuid.uuid7())
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO dlightrag_agent_sessions"
            " (owner_id, session_id, lease_run_id, fencing_epoch)"
            " VALUES ($1, $2::uuid, $3::uuid, 1)"
            " ON CONFLICT (owner_id, session_id) DO NOTHING",
            _OWNER,
            uuid.UUID(session_id),
            uuid.UUID(first),
        )
        await conn.execute(
            "INSERT INTO dlightrag_session_workspace_heads"
            " (owner_id, session_id, lane_id, head_run_id, manifest_digest, revision)"
            " VALUES ($1, $2::uuid, 'main', $3::uuid, $4, 2)",
            _OWNER,
            uuid.UUID(session_id),
            uuid.UUID(second),
            "a" * 64,
        )
        await conn.execute(
            "INSERT INTO dlightrag_session_workspace_snapshots"
            " (owner_id, session_id, run_id, parent_run_id, manifest_digest, manifest_json)"
            " VALUES ($1, $2::uuid, $3::uuid, NULL, $4, $5::jsonb),"
            "        ($1, $2::uuid, $6::uuid, $3::uuid, $7, $8::jsonb)",
            _OWNER,
            uuid.UUID(session_id),
            uuid.UUID(first),
            "b" * 64,
            json.dumps([{"path": "scripts/model.py", "digest": "c" * 64, "size": 11}]),
            uuid.UUID(second),
            "d" * 64,
            json.dumps(
                [
                    {"path": "scripts/model.py", "digest": "c" * 64, "size": 11},
                    {"path": "artifacts/report.html", "digest": "e" * 64, "size": 20},
                ]
            ),
        )

    heads = await store.agentspace_heads(_OWNER, conversation_id)
    assert heads is not None and len(heads) == 1
    assert (heads[0].lane_id, heads[0].head_run_id, heads[0].revision) == ("main", second, 2)
    snapshots = await store.agentspace_snapshots(_OWNER, conversation_id, limit=10)
    assert snapshots is not None
    assert [(snapshot.run_id, snapshot.file_count) for snapshot in snapshots] == [
        (second, 2),
        (first, 1),
    ]
    # The list carries the diff each snapshot represents, so the timeline needs no second read.
    assert [snapshot.changes for snapshot in snapshots] == [
        AgentspaceSnapshotChanges(added=1, changed=0, removed=0),
        AgentspaceSnapshotChanges(added=1, changed=0, removed=0),
    ]
    files = await store.agentspace_snapshot_files(_OWNER, conversation_id, second)
    assert files is not None
    # The detail's markers are the list's counts resolved onto the files.
    assert {(file.relative_path, file.change) for file in files} == {
        ("scripts/model.py", "unchanged"),
        ("artifacts/report.html", "added"),
    }
    versions = await store.agentspace_file_versions(_OWNER, conversation_id, "scripts/model.py")
    assert versions is not None
    assert [(version.run_id, version.change) for version in versions] == [
        (first, "added"),
        (second, "unchanged"),
    ]
    assert await store.agentspace_heads(_OTHER_OWNER, conversation_id) is None

    # Reset is refused while the Session works, then clears every branch and snapshot.
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO dlightrag_session_work_heads"
            " (owner_id, session_id, active_run_id) VALUES ($1, $2::uuid, $3::uuid)",
            _OWNER,
            uuid.UUID(session_id),
            uuid.UUID(second),
        )
    with pytest.raises(SessionWorkBusyError):
        await store.reset_session_agentspace(_OWNER, conversation_id)
    assert await store.agentspace_heads(_OWNER, conversation_id) == (heads[0],)

    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE dlightrag_session_work_heads SET active_run_id = NULL"
            " WHERE owner_id = $1 AND session_id = $2::uuid",
            _OWNER,
            uuid.UUID(session_id),
        )
    assert await store.reset_session_agentspace(_OWNER, conversation_id) == session_id
    assert await store.agentspace_heads(_OWNER, conversation_id) == ()
    assert await store.agentspace_snapshots(_OWNER, conversation_id, limit=10) == ()


async def test_a_snapshot_diffs_against_its_parent_and_reads_a_pruned_parent_as_empty(
    store: PGWebConversationStore, pool: Any
) -> None:
    conversation_id = await _conversation(store)
    session_id = await store.conversation_agent_session(_OWNER, conversation_id)
    assert session_id is not None
    first, second, third = (str(uuid.uuid7()) for _ in range(3))

    def manifest(*files: tuple[str, str]) -> str:
        return json.dumps(
            [{"path": path, "digest": digest, "size": len(path)} for path, digest in files]
        )

    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO dlightrag_agent_sessions"
            " (owner_id, session_id, lease_run_id, fencing_epoch)"
            " VALUES ($1, $2::uuid, $3::uuid, 1)"
            " ON CONFLICT (owner_id, session_id) DO NOTHING",
            _OWNER,
            uuid.UUID(session_id),
            uuid.UUID(first),
        )
        await conn.execute(
            "INSERT INTO dlightrag_session_workspace_snapshots"
            " (owner_id, session_id, run_id, parent_run_id, manifest_digest, manifest_json)"
            " VALUES ($1, $2::uuid, $3::uuid, NULL, $4, $5::jsonb),"
            "        ($1, $2::uuid, $6::uuid, $3::uuid, $7, $8::jsonb),"
            "        ($1, $2::uuid, $9::uuid, $6::uuid, $10, $11::jsonb)",
            _OWNER,
            uuid.UUID(session_id),
            uuid.UUID(first),
            "a" * 64,
            manifest(("keep.md", "a" * 64), ("edit.md", "b" * 64), ("gone.md", "c" * 64)),
            uuid.UUID(second),
            "b" * 64,
            manifest(("keep.md", "a" * 64), ("edit.md", "d" * 64), ("new.md", "e" * 64)),
            uuid.UUID(third),
            "e" * 64,
            manifest(("keep.md", "a" * 64), ("edit.md", "d" * 64), ("new.md", "e" * 64)),
        )

    snapshots = await store.agentspace_snapshots(_OWNER, conversation_id, limit=10)
    assert snapshots is not None
    assert [(snapshot.run_id, snapshot.changes) for snapshot in snapshots] == [
        (third, AgentspaceSnapshotChanges(added=0, changed=0, removed=0)),
        (second, AgentspaceSnapshotChanges(added=1, changed=1, removed=1)),
        (first, AgentspaceSnapshotChanges(added=3, changed=0, removed=0)),
    ]
    files = await store.agentspace_snapshot_files(_OWNER, conversation_id, second)
    assert files is not None
    assert [(file.relative_path, file.change) for file in files] == [
        ("keep.md", "unchanged"),
        ("edit.md", "changed"),
        ("new.md", "added"),
    ]

    # Prune the middle snapshot the way the retained-history bound does. Its child still
    # names it in parent_run_id, but that parent manifest can no longer be read, so the
    # child is diffed against the empty one: every file it holds is added, none removed.
    async with pool.acquire() as conn:
        await conn.execute(
            "DELETE FROM dlightrag_session_workspace_snapshots"
            " WHERE owner_id = $1 AND session_id = $2::text::uuid AND run_id = $3::uuid",
            _OWNER,
            session_id,
            uuid.UUID(second),
        )

    files = await store.agentspace_snapshot_files(_OWNER, conversation_id, third)
    assert files is not None
    assert [(file.relative_path, file.change) for file in files] == [
        ("keep.md", "added"),
        ("edit.md", "added"),
        ("new.md", "added"),
    ]
    snapshots = await store.agentspace_snapshots(_OWNER, conversation_id, limit=10)
    assert snapshots is not None
    assert [(snapshot.run_id, snapshot.changes) for snapshot in snapshots] == [
        (third, AgentspaceSnapshotChanges(added=3, changed=0, removed=0)),
        (first, AgentspaceSnapshotChanges(added=3, changed=0, removed=0)),
    ]


async def test_the_goal_round_reads_its_scope_and_session_conversation(
    store: PGWebConversationStore,
) -> None:
    """A `/goal` round continues the newest turn's workspaces, found from its Session."""
    conversation_id = await _conversation(store)
    assert await store.newest_answer_workspaces(_OWNER, conversation_id) is None
    assert await store.conversation_for_session(_OWNER, conversation_id) == conversation_id

    await _submit(store, conversation_id, "first")

    assert await store.newest_answer_workspaces(_OWNER, conversation_id) == ("alpha",)
    assert (
        await store.conversation_for_session(_OWNER, "00000000-0000-7000-8000-0000000000ff") is None
    )


async def _answer_goal(pool: Any, conversation_id: str) -> Any:
    goals = AnswerGoalService(store=PGAnswerGoalStore(pool=pool))
    return await goals.create(
        owner_id=_OWNER,
        session_id=conversation_id,
        run_id="",
        objective="finish the model",
        max_rounds=4,
    )


def _goal_delivery(
    conversation_id: str, request_id: str, goal: Any, *, content: str
) -> tuple[dict[str, Any], str, str]:
    """One goal-round request, its submission key and its prompt digest."""
    import hashlib
    from uuid import NAMESPACE_URL, uuid5

    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    request = _request(
        content,
        agent_session_id=conversation_id,
        mode="research",
        continuation_kind="goal_round",
        goal_id=goal.id,
        goal_revision=goal.revision,
        goal_round=goal.rounds_started,
    )
    submission_id = str(
        uuid5(NAMESPACE_URL, f"dlightrag-goal-delivery:{conversation_id}:{request_id}")
    )
    return request, submission_id, digest


async def test_a_goal_round_delivery_re_checks_the_goal_in_the_transaction(
    pool: Any, store: Any
) -> None:
    """A reserved round is admitted, and the delivered continuation is consumed, atomically."""
    import hashlib

    conversation_id = await _conversation(store)
    goal = await _answer_goal(pool, conversation_id)
    content = '<goal_round>Objective: "finish the model"</goal_round>'
    request_id = str(uuid.uuid7())
    request = _request(
        content,
        agent_session_id=conversation_id,
        mode="research",
        continuation_kind="goal_round",
        goal_id=goal.id,
        goal_revision=goal.revision + 1,
        goal_round=1,
    )
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    assert await store.enqueue_goal_round(
        _OWNER,
        conversation_id,
        request_id=request_id,
        content=content,
        prompt_digest=digest,
        goal_id=goal.id,
        goal_revision=goal.revision + 1,
        goal_round=1,
    )
    submission_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"dlightrag-goal-delivery:{conversation_id}:{request_id}",
        )
    )
    assert not await store.submission_exists(_OWNER, conversation_id, submission_id)

    accepted = await store.create_answer_turn(
        principal_id=_OWNER,
        conversation_id=conversation_id,
        submission_id=submission_id,
        envelope=_turn_envelope(
            owner=_OWNER,
            request=request,
            submission_id=submission_id,
            fingerprint=run_request_fingerprint(request),
        ),
        run_id=str(uuid.uuid7()),
        title_hint=None,
        goal_round_guard=GoalRoundGuard(
            request_id=request_id,
            prompt_digest=digest,
            goal_id=goal.id,
            goal_revision=goal.revision + 1,
            goal_round=1,
        ),
    )

    assert accepted is not None
    assert await store.pending_goal_round(_OWNER, conversation_id) is None
    assert await store.submission_exists(_OWNER, conversation_id, submission_id)
    goals = AnswerGoalService(store=PGAnswerGoalStore(pool=pool))
    stored = await goals.get(owner_id=_OWNER, session_id=conversation_id)
    assert stored is not None
    assert (stored.id, stored.rounds_started, stored.phase.value) == (goal.id, 1, "active")


async def test_a_goal_round_delivery_refuses_a_mixed_reserved_pair(pool: Any, store: Any) -> None:
    """Revision and rounds must describe one state: a mixed pair is another writer's change."""
    import hashlib

    conversation_id = await _conversation(store)
    goals = AnswerGoalService(store=PGAnswerGoalStore(pool=pool))
    goal = await goals.create(
        owner_id=_OWNER,
        session_id=conversation_id,
        run_id="",
        objective="finish the model",
        max_rounds=4,
    )
    content = '<goal_round>Objective: "finish the model"</goal_round>'
    request_id = str(uuid.uuid7())
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    assert await store.enqueue_goal_round(
        _OWNER,
        conversation_id,
        request_id=request_id,
        content=content,
        prompt_digest=digest,
        goal_id=goal.id,
        goal_revision=goal.revision + 1,
        goal_round=1,
    )
    edited = await goals.update(
        owner_id=_OWNER,
        session_id=conversation_id,
        run_id="",
        goal_id=goal.id,
        revision=goal.revision,
        action="edit",
        objective="finish the revised model",
    )
    assert edited.revision == goal.revision + 1 and edited.rounds_started == 0
    request = _request(
        content,
        agent_session_id=conversation_id,
        mode="research",
        continuation_kind="goal_round",
        goal_id=goal.id,
        goal_revision=edited.revision,
        goal_round=1,
    )
    submission_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"dlightrag-goal-delivery:{conversation_id}:{request_id}",
        )
    )

    accepted = await store.create_answer_turn(
        principal_id=_OWNER,
        conversation_id=conversation_id,
        submission_id=submission_id,
        envelope=_turn_envelope(
            owner=_OWNER,
            request=request,
            submission_id=submission_id,
            fingerprint=run_request_fingerprint(request),
        ),
        run_id=str(uuid.uuid7()),
        title_hint=None,
        goal_round_guard=GoalRoundGuard(
            request_id=request_id,
            prompt_digest=digest,
            goal_id=goal.id,
            goal_revision=goal.revision + 1,
            goal_round=1,
        ),
    )

    assert accepted is None
    assert await store.pending_goal_round(_OWNER, conversation_id) is not None
    stored = await goals.get(owner_id=_OWNER, session_id=conversation_id)
    assert stored is not None and (stored.rounds_started, stored.revision) == (0, edited.revision)


async def test_a_reserved_goal_round_is_refused_after_the_cap_dropped(
    pool: Any, store: Any
) -> None:
    """A cap lowered after reservation refuses cleanly instead of raising the DB round check."""
    import hashlib

    conversation_id = await _conversation(store)
    goals = AnswerGoalService(store=PGAnswerGoalStore(pool=pool))
    goal = await goals.create(
        owner_id=_OWNER,
        session_id=conversation_id,
        run_id="",
        objective="finish the model",
        max_rounds=2,
    )
    started, _ = await goals.admit_round(
        owner_id=_OWNER,
        session_id=conversation_id,
        run_id="",
        goal_id=goal.id,
        revision=goal.revision,
    )
    lowered = await goals.update(
        owner_id=_OWNER,
        session_id=conversation_id,
        run_id="",
        goal_id=started.id,
        revision=started.revision,
        action="edit",
        max_rounds=1,
    )
    content = '<goal_round>Objective: "finish the model"</goal_round>'
    request_id = str(uuid.uuid7())
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    assert await store.enqueue_goal_round(
        _OWNER,
        conversation_id,
        request_id=request_id,
        content=content,
        prompt_digest=digest,
        goal_id=started.id,
        goal_revision=lowered.revision + 1,
        goal_round=2,
    )
    request = _request(
        content,
        agent_session_id=conversation_id,
        mode="research",
        continuation_kind="goal_round",
        goal_id=started.id,
        goal_revision=lowered.revision + 1,
        goal_round=2,
    )
    submission_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"dlightrag-goal-delivery:{conversation_id}:{request_id}",
        )
    )

    accepted = await store.create_answer_turn(
        principal_id=_OWNER,
        conversation_id=conversation_id,
        submission_id=submission_id,
        envelope=_turn_envelope(
            owner=_OWNER,
            request=request,
            submission_id=submission_id,
            fingerprint=run_request_fingerprint(request),
        ),
        run_id=str(uuid.uuid7()),
        title_hint=None,
        goal_round_guard=GoalRoundGuard(
            request_id=request_id,
            prompt_digest=digest,
            goal_id=started.id,
            goal_revision=lowered.revision + 1,
            goal_round=2,
        ),
    )

    assert accepted is None
    assert await store.pending_goal_round(_OWNER, conversation_id) is not None


@pytest.mark.parametrize("change", ["pause", "clear", "move"])
async def test_a_goal_round_delivery_refuses_a_goal_that_moved(
    pool: Any, store: Any, change: str
) -> None:
    """The transaction's own re-read refuses a continuation the Web read still thought current."""
    import hashlib

    conversation_id = await _conversation(store)
    goals = AnswerGoalService(store=PGAnswerGoalStore(pool=pool))
    goal = await goals.create(
        owner_id=_OWNER,
        session_id=conversation_id,
        run_id="",
        objective="finish the model",
        max_rounds=4,
    )
    started, _ = await goals.admit_round(
        owner_id=_OWNER,
        session_id=conversation_id,
        run_id="",
        goal_id=goal.id,
        revision=goal.revision,
    )
    content = '<goal_round>Objective: "finish the model"</goal_round>'
    request_id = str(uuid.uuid7())
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    request, submission_id, _ = _goal_delivery(
        conversation_id, request_id, started, content=content
    )
    assert await store.enqueue_goal_round(
        _OWNER,
        conversation_id,
        request_id=request_id,
        content=content,
        prompt_digest=digest,
        goal_id=started.id,
        goal_revision=started.revision,
        goal_round=started.rounds_started,
    )
    if change == "pause":
        await goals.update(
            owner_id=_OWNER,
            session_id=conversation_id,
            run_id="",
            goal_id=started.id,
            revision=started.revision,
            action="pause",
            human_control=True,
        )
    elif change == "clear":
        await goals.clear(owner_id=_OWNER, session_id=conversation_id, run_id="")
    else:
        await goals.admit_round(
            owner_id=_OWNER,
            session_id=conversation_id,
            run_id="",
            goal_id=started.id,
            revision=started.revision,
        )

    refused = await store.create_answer_turn(
        principal_id=_OWNER,
        conversation_id=conversation_id,
        submission_id=submission_id,
        envelope=_turn_envelope(
            owner=_OWNER,
            request=request,
            submission_id=submission_id,
            fingerprint=run_request_fingerprint(request),
        ),
        run_id=str(uuid.uuid7()),
        title_hint=None,
        goal_round_guard=GoalRoundGuard(
            request_id=request_id,
            prompt_digest=digest,
            goal_id=started.id,
            goal_revision=started.revision,
            goal_round=started.rounds_started,
        ),
    )

    assert refused is None
    assert await _count(pool, "web_conversation_turns") == 0
    assert await _count(pool, "dlightrag_runs") == 0
    assert await store.pending_goal_round(_OWNER, conversation_id) is not None


async def test_clear_goal_round_deletes_only_the_identity_it_read(store: Any) -> None:
    """A dropped continuation is deleted by identity: a stale clear leaves the newer row alone."""
    conversation_id = await _conversation(store)
    digest = "a" * 64
    request_id = str(uuid.uuid7())
    assert (
        await store.clear_goal_round(
            _OWNER, conversation_id, request_id=request_id, prompt_digest=digest
        )
        is False
    )

    assert await store.enqueue_goal_round(
        _OWNER,
        conversation_id,
        request_id=request_id,
        content="continue",
        prompt_digest=digest,
        goal_id="goal-1",
        goal_revision=2,
        goal_round=1,
    )
    # A clear that names a row this transaction did not read deletes nothing.
    assert (
        await store.clear_goal_round(
            _OWNER,
            conversation_id,
            request_id=str(uuid.uuid7()),
            prompt_digest=digest,
        )
        is False
    )
    pending = await store.pending_goal_round(_OWNER, conversation_id)
    assert pending is not None and pending.request_id == request_id

    assert (
        await store.clear_goal_round(
            _OWNER, conversation_id, request_id=request_id, prompt_digest=digest
        )
        is True
    )
    assert (
        await store.clear_goal_round(
            _OWNER, conversation_id, request_id=request_id, prompt_digest=digest
        )
        is False
    )
