# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Integration tests for the Web run-activity read on PostgreSQL 18.

Exercises the real contract against a live database: one hard page of the owner's
newest conversation-linked Answer runs, the conversation title join, owner scoping,
and the exclusion of runs that are not Web activity (a non-Answer run a turn links,
or an Answer run no turn links). Every test runs inside a throwaway database created
and dropped per test, so the configured administrative database is never mutated.
Connection settings come from the shared integration-test PostgreSQL environment;
skipped if unavailable.
"""

from __future__ import annotations

import datetime
import uuid
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from dlightrag.adapters.postgres.runtime.run_store import PGRunStore
from dlightrag.adapters.postgres.web.web_conversations import PGWebConversationStore
from dlightrag.application.web_conversations.models import WebRunActivity
from dlightrag.engine.answer.runs.envelope import accepted_input_envelope
from dlightrag.engine.runtime.records import (
    PreparedRunEnvelope,
    RunAccessScope,
    RunKind,
    run_request_fingerprint,
)
from tests.support.pg import (
    PG_CONN_KWARGS,
    drop_database,
    release_session_work,
    skip_without_postgres,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
]

_OWNER = "principal-alpha"
_OTHER_OWNER = "principal-beta"
_T0 = datetime.datetime(2026, 10, 24, 10, 0, 0, tzinfo=datetime.UTC)


@pytest.fixture
async def db_name() -> AsyncIterator[str]:
    """Provision an isolated throwaway database and yield its name."""
    await skip_without_postgres()

    name = f"dlightrag_runactivity_{uuid.uuid4().hex[:12]}"
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
    *,
    owner: str,
    request: dict[str, Any],
    submission_id: str,
    fingerprint: str,
    run_kind: RunKind = "answer",
) -> PreparedRunEnvelope:
    return PreparedRunEnvelope(
        run_kind=run_kind,
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


async def _submit(
    store: PGWebConversationStore,
    conversation_id: str,
    query: str,
    *,
    owner: str = _OWNER,
    title_hint: str | None = None,
    parent_run_id: str | None = None,
    continuation_kind: str | None = None,
    pool: Any | None = None,
) -> str:
    """Accept one submission as a linked turn and return the Run it created.

    A goal round is an ordinary Answer submission that names the Run it continues: the
    accepted envelope carries ``parent_run_id`` and ``continuation_kind``, which is
    exactly the durable fact the activity projection states.
    """
    request = {**_request(query), "agent_session_id": conversation_id}
    if parent_run_id is not None:
        request["parent_run_id"] = parent_run_id
    if continuation_kind is not None:
        request["continuation_kind"] = continuation_kind
    submission_id = str(uuid.uuid4())
    created = await store.create_answer_turn(
        principal_id=owner,
        conversation_id=conversation_id,
        submission_id=submission_id,
        envelope=_turn_envelope(
            owner=owner,
            request=request,
            submission_id=submission_id,
            fingerprint=run_request_fingerprint(request),
        ),
        run_id=str(uuid.uuid7()),
        title_hint=title_hint,
    )
    assert created is not None
    if pool is not None:
        # This suite reads activity across several turns of one Session; release each admission
        # the way a finished worker would so the next fixture turn is admissible.
        async with pool.acquire() as conn:
            await release_session_work(conn, owner_id=owner, run_id=created.turn.run.run_id)
    return created.turn.run.run_id


async def _insert_linked_turn(
    pool: Any,
    *,
    conversation_id: str,
    run_id: str,
    turn_number: int,
    owner: str = _OWNER,
) -> None:
    """Link an existing non-Answer run to a conversation the way a turn row does.

    Web acceptance only links Answer runs, so a non-Answer turn is seeded raw: it is
    exactly the row the read must tolerate and filter out.
    """
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                """
                INSERT INTO web_conversation_turns
                    (turn_id, principal_id, conversation_id, turn_number, submission_id, answer_run_id)
                VALUES ($1::text::uuid, $2, $3::text::uuid, $4, $5::text::uuid, $6::text::uuid)
                """,
                str(uuid.uuid4()),
                owner,
                conversation_id,
                turn_number,
                str(uuid.uuid4()),
                run_id,
            )
            await conn.execute(
                """
                UPDATE web_conversations
                SET content_revision = $3
                WHERE principal_id = $1 AND conversation_id = $2::text::uuid
                """,
                owner,
                conversation_id,
                turn_number,
            )


async def _accept_unlinked_answer(runs: PGRunStore, *, query: str, owner: str = _OWNER) -> str:
    """Accept an Answer run with no Web conversation turn, as REST/MCP acceptance does."""
    request = _request(query)
    run_id = str(uuid.uuid7())
    await runs.accept_run(
        envelope=_turn_envelope(
            owner=owner,
            request=request,
            submission_id=str(uuid.uuid4()),
            fingerprint=run_request_fingerprint(request),
        ),
        run_id=run_id,
    )
    return run_id


async def _set_run(
    pool: Any,
    run_id: str,
    *,
    created_at: datetime.datetime,
    owner: str = _OWNER,
    status: str = "queued",
    phase: str | None = None,
    started_at: datetime.datetime | None = None,
    finished_at: datetime.datetime | None = None,
) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """
            UPDATE dlightrag_runs
            SET created_at = $3, status = $4, phase = $5, started_at = $6, finished_at = $7
            WHERE owner_id = $1 AND run_id = $2::text::uuid
            """,
            owner,
            run_id,
            created_at,
            status,
            phase,
            started_at,
            finished_at,
        )


# ---------------------------------------------------------------------------
# The projection and its ordering
# ---------------------------------------------------------------------------


async def test_activity_reads_newest_first_and_joins_the_conversation_title(
    store: PGWebConversationStore, pool: Any
) -> None:
    titled = await _conversation(store)
    untitled = await _conversation(store)
    oldest = await _submit(store, titled, "first", title_hint="Volvo outlook", pool=pool)
    newest = await _submit(store, untitled, "second", title_hint=None, pool=pool)
    middle = await _submit(store, titled, "third", title_hint="ignored", pool=pool)

    await _set_run(pool, oldest, created_at=_T0)
    await _set_run(pool, middle, created_at=_T0 + datetime.timedelta(hours=1))
    started = _T0 + datetime.timedelta(hours=2)
    await _set_run(
        pool,
        newest,
        created_at=_T0 + datetime.timedelta(hours=2),
        status="running",
        phase="researching",
        started_at=started,
    )

    activity = await store.recent_answer_runs(_OWNER, limit=10)

    assert [item.run_id for item in activity] == [newest, middle, oldest]
    assert [item.conversation_id for item in activity] == [untitled, titled, titled]
    assert [item.conversation_title for item in activity] == [
        None,
        "Volvo outlook",
        "Volvo outlook",
    ]
    assert isinstance(activity[0], WebRunActivity)
    assert activity[0].status == "running"
    assert activity[0].phase == "researching"
    assert activity[0].created_at == _T0 + datetime.timedelta(hours=2)
    assert activity[0].started_at == started
    assert activity[0].finished_at is None
    assert activity[2].status == "queued"
    # An ordinary reader submission states no goal-continuation fact.
    assert [item.goal_round for item in activity] == [False, False, False]


async def test_activity_states_the_goal_round_fact_of_a_goal_round_run(
    store: PGWebConversationStore, pool: Any
) -> None:
    conversation = await _conversation(store)
    answered = await _submit(store, conversation, "answer the question", pool=pool)
    goal_round = await _submit(
        store,
        conversation,
        "continue toward the objective",
        parent_run_id=answered,
        continuation_kind="goal_round",
        pool=pool,
    )
    await _set_run(pool, answered, created_at=_T0)
    # A goal round's Run is a later ordinary Answer Run in the same conversation.
    await _set_run(pool, goal_round, created_at=_T0 + datetime.timedelta(minutes=1))

    activity = await store.recent_answer_runs(_OWNER, limit=10)

    assert [item.run_id for item in activity] == [goal_round, answered]
    assert [item.goal_round for item in activity] == [True, False]
    assert activity[0].goal_round is True
    assert activity[1].goal_round is False
    # The goal fact is additive: ordering and the conversation link are unchanged.
    assert [item.conversation_id for item in activity] == [conversation, conversation]


async def test_activity_orders_runs_of_one_instant_by_run_id_descending(
    store: PGWebConversationStore, pool: Any
) -> None:
    conversation = await _conversation(store)
    run_ids = [
        await _submit(store, conversation, f"query {index}", pool=pool) for index in range(3)
    ]
    for run_id in run_ids:
        await _set_run(pool, run_id, created_at=_T0)

    activity = await store.recent_answer_runs(_OWNER, limit=10)

    # Canonical UUID text order is the binary order the runs PK sorts by.
    assert [item.run_id for item in activity] == sorted(run_ids, reverse=True)


# ---------------------------------------------------------------------------
# Ownership and bounds
# ---------------------------------------------------------------------------


async def test_activity_is_owner_scoped_and_one_hard_page(
    store: PGWebConversationStore, pool: Any
) -> None:
    owned = await _conversation(store, _OWNER)
    foreign = await _conversation(store, _OTHER_OWNER)
    run_ids = [await _submit(store, owned, f"owned {index}", pool=pool) for index in range(3)]
    foreign_run = await _submit(store, foreign, "foreign", owner=_OTHER_OWNER, pool=pool)
    for index, run_id in enumerate(run_ids):
        await _set_run(pool, run_id, created_at=_T0 + datetime.timedelta(hours=index))
    await _set_run(
        pool,
        foreign_run,
        owner=_OTHER_OWNER,
        created_at=_T0 + datetime.timedelta(hours=10),
    )

    everything = await store.recent_answer_runs(_OWNER, limit=10)

    assert [item.run_id for item in everything] == list(reversed(run_ids))
    assert all(item.conversation_id == owned for item in everything)

    page = await store.recent_answer_runs(_OWNER, limit=2)
    assert [item.run_id for item in page] == [run_ids[2], run_ids[1]]

    other_owner = await store.recent_answer_runs(_OTHER_OWNER, limit=10)
    assert [item.run_id for item in other_owner] == [foreign_run]


# ---------------------------------------------------------------------------
# What is not Web run activity
# ---------------------------------------------------------------------------


async def test_activity_excludes_a_non_answer_run_and_an_unlinked_answer_run(
    store: PGWebConversationStore, runs: PGRunStore, pool: Any
) -> None:
    conversation = await _conversation(store)
    answer = await _submit(store, conversation, "linked answer")
    retrieval_request = _request("linked retrieval")
    retrieval = str(uuid.uuid7())
    await runs.accept_run(
        envelope=_turn_envelope(
            owner=_OWNER,
            request=retrieval_request,
            submission_id=str(uuid.uuid4()),
            fingerprint=run_request_fingerprint(retrieval_request),
            run_kind="retrieval",
        ),
        run_id=retrieval,
    )
    await _insert_linked_turn(
        pool,
        conversation_id=conversation,
        run_id=retrieval,
        turn_number=2,
    )
    unlinked = await _accept_unlinked_answer(runs, query="unlinked answer")
    await _set_run(pool, answer, created_at=_T0)
    await _set_run(pool, retrieval, created_at=_T0 + datetime.timedelta(hours=1))
    await _set_run(pool, unlinked, created_at=_T0 + datetime.timedelta(hours=2))

    activity = await store.recent_answer_runs(_OWNER, limit=10)

    assert [item.run_id for item in activity] == [answer]
