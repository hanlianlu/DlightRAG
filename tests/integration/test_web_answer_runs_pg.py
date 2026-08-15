# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Integration tests for run-linked Web conversations on PostgreSQL 18.

Exercises the real contract against a live database: the reset migration and its
foreign keys, the single transaction that creates a run, its uploaded bytes, and
its conversation turn, owner-wide submission idempotency and its conflicts,
concurrent replay, conversation deletion with ownership-safe artifact cleanup,
the retention exemption a successful linked turn grants, and the cascade that
removes a pruned failed run's visible turn.

Every test runs inside a throwaway database created and dropped per test, so the
developer's ``dlightrag`` database is never mutated.

Requires PostgreSQL at localhost:5432 (dlightrag/dlightrag); skipped otherwise.
"""

import asyncio
import uuid
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from dlightrag.runtime import (
    IdempotencyKeyConflict,
    PendingArtifact,
    PendingArtifactReference,
    answer_run_request_fingerprint,
    artifact_digest,
)
from dlightrag.storage.answer_runs import (
    PGAnswerRunStore,
)
from dlightrag.storage.web_conversations import (
    ConversationSubmissionConflict,
    PGWebConversationStore,
)
from tests.conftest import FingerprintingAnswerRunStore

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
]

_PG_CONN_KWARGS: dict[str, Any] = dict(
    host="localhost",
    port=5432,
    user="dlightrag",
    password="dlightrag",
    database="dlightrag",
)

_OWNER = "principal-alpha"
_OTHER_OWNER = "principal-beta"
_TTL_DAYS = 30
_MAX_TURNS = 100


async def _pg_available() -> bool:
    try:
        conn = await asyncpg.connect(**_PG_CONN_KWARGS)
        await conn.fetchval("SELECT 1")
        await conn.close()
        return True
    except Exception:
        return False


@pytest.fixture
async def db_name() -> AsyncIterator[str]:
    """Provision an isolated throwaway database and yield its name."""
    if not await _pg_available():
        pytest.skip("PostgreSQL not available")

    name = f"dlightrag_weblink_{uuid.uuid4().hex[:12]}"
    admin = await asyncpg.connect(**_PG_CONN_KWARGS)
    try:
        await admin.execute(f'CREATE DATABASE "{name}"')
    finally:
        await admin.close()
    try:
        yield name
    finally:
        admin = await asyncpg.connect(**_PG_CONN_KWARGS)
        try:
            await admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        finally:
            await admin.close()


@pytest.fixture
async def pool(db_name: str) -> AsyncIterator[Any]:
    """Yield a pool bound to this test's throwaway database."""
    created = await asyncpg.create_pool(
        **{**_PG_CONN_KWARGS, "database": db_name}, min_size=1, max_size=8
    )
    try:
        yield created
    finally:
        await created.close()


@pytest.fixture
async def runs(pool: Any) -> PGAnswerRunStore:
    store = FingerprintingAnswerRunStore(pool=pool)
    await store.initialize()
    return store


@pytest.fixture
async def store(pool: Any, runs: PGAnswerRunStore) -> PGWebConversationStore:
    created = PGWebConversationStore(pool=pool, run_store=runs)
    await created.initialize()
    return created


def _request(query: str = "why", **extra: Any) -> dict[str, Any]:
    return {"query": query, "workspaces": ["alpha"], "history": [], **extra}


async def _conversation(store: PGWebConversationStore, owner: str = _OWNER) -> str:
    row = await store.create_conversation(owner)
    return str(row["conversation_id"])


async def _submit(
    store: PGWebConversationStore,
    conversation_id: str,
    *,
    owner: str = _OWNER,
    submission_id: str | None = None,
    request: dict[str, Any] | None = None,
    artifacts: list[PendingArtifact] | None = None,
    references: list[PendingArtifactReference] | None = None,
    idempotency_fingerprint: str | None = None,
):
    effective_request = request if request is not None else _request()
    return await store.create_answer_turn(
        principal_id=owner,
        conversation_id=conversation_id,
        submission_id=submission_id or str(uuid.uuid4()),
        request=effective_request,
        idempotency_fingerprint=(
            idempotency_fingerprint
            if idempotency_fingerprint is not None
            else answer_run_request_fingerprint(effective_request)
        ),
        artifacts=artifacts or [],
        references=references or [],
        title_hint="why",
        max_turns=_MAX_TURNS,
        ttl_days=_TTL_DAYS,
    )


async def _finish(pool: Any, run_id: str, *, status: str, error: str | None = None) -> None:
    """Drive one run to a terminal state the way its worker eventually would."""
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE dlightrag_answer_runs "
            "SET status = $2, finished_at = NOW(), "
            "    result_json = CASE WHEN $2 = 'succeeded' "
            '        THEN \'{"answer": "done"}\'::jsonb ELSE NULL END, '
            "    error_kind = $3, error_message = $3 "
            "WHERE run_id = $1",
            uuid.UUID(run_id),
            status,
            error,
        )


async def _backdate_finish(pool: Any, run_id: str, *, days: int) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE dlightrag_answer_runs "
            "SET finished_at = NOW() - ($2 * INTERVAL '1 day') WHERE run_id = $1",
            uuid.UUID(run_id),
            days,
        )


async def _count(pool: Any, table: str, **where: Any) -> int:
    clause = " AND ".join(f"{column} = ${index}" for index, column in enumerate(where, start=1))
    async with pool.acquire() as conn:
        return int(
            await conn.fetchval(
                f"SELECT count(*)::int FROM {table}"  # noqa: S608 - fixed table/column names
                + (f" WHERE {clause}" if clause else ""),
                *where.values(),
            )
        )


# ---------------------------------------------------------------------------
# One transaction
# ---------------------------------------------------------------------------


async def test_a_submission_commits_the_run_bytes_and_turn_together(
    store: PGWebConversationStore, pool: Any
) -> None:
    conversation_id = await _conversation(store)
    content = b"chart-bytes"
    digest = artifact_digest(content)

    creation = await _submit(
        store,
        conversation_id,
        artifacts=[PendingArtifact(content=content)],
        references=[
            PendingArtifactReference(
                resource_id="attachment-1",
                reference_kind="current_attachment",
                ordinal=1,
                digest=digest,
                filename="chart.png",
                mime_type="image/png",
            )
        ],
    )

    assert creation is not None
    assert creation.replayed is False
    assert creation.turn.turn_number == 1
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT t.answer_run_id::text AS run_id, r.status "
            "FROM web_conversation_turns AS t "
            "JOIN dlightrag_answer_runs AS r "
            "  ON r.owner_id = t.principal_id AND r.run_id = t.answer_run_id "
            "WHERE t.turn_id = $1::text::uuid",
            creation.turn.turn_id,
        )
    assert row["run_id"] == creation.turn.answer_run_id
    assert row["status"] == "queued"
    assert await _count(pool, "dlightrag_answer_artifacts", owner_id=_OWNER, digest=digest) == 1
    assert await _count(pool, "dlightrag_answer_run_artifacts", owner_id=_OWNER) == 1


async def test_a_submission_to_an_unknown_conversation_writes_nothing(
    store: PGWebConversationStore, pool: Any
) -> None:
    creation = await _submit(store, str(uuid.uuid4()))

    assert creation is None
    assert await _count(pool, "dlightrag_answer_runs") == 0
    assert await _count(pool, "web_conversation_turns") == 0


async def test_a_foreign_conversation_is_never_written_to(
    store: PGWebConversationStore, pool: Any
) -> None:
    conversation_id = await _conversation(store, _OTHER_OWNER)

    creation = await _submit(store, conversation_id, owner=_OWNER)

    assert creation is None
    assert await _count(pool, "dlightrag_answer_runs") == 0


# ---------------------------------------------------------------------------
# Owner-wide submission idempotency
# ---------------------------------------------------------------------------


async def test_replaying_a_submission_returns_the_same_run_and_turn(
    store: PGWebConversationStore, pool: Any
) -> None:
    conversation_id = await _conversation(store)
    submission_id = str(uuid.uuid4())

    first = await _submit(store, conversation_id, submission_id=submission_id)
    second = await _submit(store, conversation_id, submission_id=submission_id)

    assert first is not None and second is not None
    assert second.replayed is True
    assert second.turn.turn_id == first.turn.turn_id
    assert second.turn.answer_run_id == first.turn.answer_run_id
    assert await _count(pool, "dlightrag_answer_runs") == 1
    assert await _count(pool, "web_conversation_turns") == 1


async def test_public_fingerprint_controls_atomic_web_replay(
    store: PGWebConversationStore, pool: Any
) -> None:
    conversation_id = await _conversation(store)
    submission_id = str(uuid.uuid4())
    public_fingerprint = answer_run_request_fingerprint(_request())

    first = await _submit(
        store,
        conversation_id,
        submission_id=submission_id,
        request=_request(pinned_models=[{"revision": "old"}]),
        idempotency_fingerprint=public_fingerprint,
    )
    replay = await _submit(
        store,
        conversation_id,
        submission_id=submission_id,
        request=_request(pinned_models=[{"revision": "new"}]),
        idempotency_fingerprint=public_fingerprint,
    )

    assert first is not None and replay is not None
    assert replay.replayed is True
    assert replay.turn.answer_run_id == first.turn.answer_run_id
    with pytest.raises(ConversationSubmissionConflict):
        await _submit(
            store,
            conversation_id,
            submission_id=submission_id,
            request=_request(pinned_models=[{"revision": "old"}]),
            idempotency_fingerprint=answer_run_request_fingerprint(_request("changed")),
        )
    assert await _count(pool, "dlightrag_answer_runs") == 1


async def test_concurrent_identical_submissions_create_exactly_one_turn(
    store: PGWebConversationStore, pool: Any
) -> None:
    conversation_id = await _conversation(store)
    submission_id = str(uuid.uuid4())

    results = await asyncio.gather(
        *(_submit(store, conversation_id, submission_id=submission_id) for _ in range(5))
    )

    run_ids = {result.turn.answer_run_id for result in results if result is not None}
    assert len(run_ids) == 1
    assert await _count(pool, "dlightrag_answer_runs") == 1
    assert await _count(pool, "web_conversation_turns") == 1


async def test_reusing_a_submission_with_different_input_is_a_conflict(
    store: PGWebConversationStore, pool: Any
) -> None:
    conversation_id = await _conversation(store)
    submission_id = str(uuid.uuid4())
    await _submit(store, conversation_id, submission_id=submission_id)

    with pytest.raises(ConversationSubmissionConflict):
        await _submit(
            store,
            conversation_id,
            submission_id=submission_id,
            request=_request("a different question"),
        )

    assert await _count(pool, "dlightrag_answer_runs") == 1


async def test_reusing_a_submission_in_another_conversation_is_a_conflict(
    store: PGWebConversationStore, pool: Any
) -> None:
    first_conversation = await _conversation(store)
    second_conversation = await _conversation(store)
    submission_id = str(uuid.uuid4())
    await _submit(store, first_conversation, submission_id=submission_id)

    with pytest.raises(ConversationSubmissionConflict):
        await _submit(store, second_conversation, submission_id=submission_id)

    assert await _count(pool, "web_conversation_turns") == 1


async def test_the_same_submission_in_two_conversations_at_once_is_a_conflict(
    store: PGWebConversationStore, pool: Any
) -> None:
    """A race loses to the unique key, and losing is a conflict, not a server fault.

    Both callers see no existing turn and both replay the one run the submission
    id owns, so the second turn insert is what discovers the reuse.
    """
    first_conversation = await _conversation(store)
    second_conversation = await _conversation(store)
    submission_id = str(uuid.uuid4())
    content = b"chart-bytes"
    references = [
        PendingArtifactReference(
            resource_id="attachment-1",
            reference_kind="current_attachment",
            ordinal=1,
            digest=artifact_digest(content),
            filename="chart.png",
            mime_type="image/png",
        )
    ]

    outcomes = await asyncio.gather(
        *(
            _submit(
                store,
                conversation_id,
                submission_id=submission_id,
                artifacts=[PendingArtifact(content=content)],
                references=list(references),
            )
            for conversation_id in (first_conversation, second_conversation)
        ),
        return_exceptions=True,
    )

    accepted = [item for item in outcomes if not isinstance(item, BaseException)]
    rejected = [item for item in outcomes if isinstance(item, BaseException)]
    assert len(accepted) == 1
    assert [type(error) for error in rejected] == [ConversationSubmissionConflict]
    assert await _count(pool, "dlightrag_answer_runs") == 1
    assert await _count(pool, "web_conversation_turns") == 1
    assert await _count(pool, "dlightrag_answer_artifacts") == 1
    assert await _count(pool, "dlightrag_answer_run_artifacts") == 1


async def test_the_submission_key_is_owner_wide_not_conversation_scoped(
    store: PGWebConversationStore, runs: PGAnswerRunStore
) -> None:
    """The run's idempotency key is the submission id in the owner's namespace."""
    conversation_id = await _conversation(store)
    submission_id = str(uuid.uuid4())

    creation = await _submit(store, conversation_id, submission_id=submission_id)

    assert creation is not None
    record = await runs.get_run(owner_id=_OWNER, run_id=creation.turn.answer_run_id)
    assert record is not None
    assert record.idempotency_key == submission_id


async def test_two_principals_may_use_the_same_submission_id(
    store: PGWebConversationStore, pool: Any
) -> None:
    submission_id = str(uuid.uuid4())
    mine = await _conversation(store, _OWNER)
    theirs = await _conversation(store, _OTHER_OWNER)

    await _submit(store, mine, owner=_OWNER, submission_id=submission_id)
    await _submit(store, theirs, owner=_OTHER_OWNER, submission_id=submission_id)

    assert await _count(pool, "web_conversation_turns") == 2
    assert await _count(pool, "dlightrag_answer_runs") == 2


async def test_a_run_created_outside_a_conversation_keeps_the_same_key_namespace(
    store: PGWebConversationStore, runs: PGAnswerRunStore
) -> None:
    conversation_id = await _conversation(store)
    submission_id = str(uuid.uuid4())
    await _submit(store, conversation_id, submission_id=submission_id)

    other_request = _request("something else")
    with pytest.raises(IdempotencyKeyConflict):
        await runs.create_run(
            owner_id=_OWNER,
            request=other_request,
            idempotency_fingerprint=answer_run_request_fingerprint(other_request),
            idempotency_key=submission_id,
        )


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


async def test_a_snapshot_projects_each_turn_from_its_run(
    store: PGWebConversationStore, pool: Any
) -> None:
    conversation_id = await _conversation(store)
    pending = await _submit(store, conversation_id, request=_request("first"))
    done = await _submit(store, conversation_id, request=_request("second"))
    assert pending is not None and done is not None
    await _finish(pool, done.turn.answer_run_id, status="succeeded")

    snapshot = await store.snapshot(
        _OWNER, conversation_id, ttl_days=_TTL_DAYS, max_turns=_MAX_TURNS
    )

    assert snapshot is not None
    assert [turn.turn_number for turn in snapshot.turns] == [1, 2]
    assert [turn.run.status for turn in snapshot.turns] == ["queued", "succeeded"]
    assert snapshot.turns[0].run.request["query"] == "first"
    assert snapshot.turns[1].run.result == {"answer": "done"}


async def test_a_run_is_only_findable_by_its_owner(store: PGWebConversationStore) -> None:
    conversation_id = await _conversation(store)
    creation = await _submit(store, conversation_id)
    assert creation is not None

    assert await store.find_turn_by_run(_OWNER, creation.turn.answer_run_id) is not None
    assert await store.find_turn_by_run(_OTHER_OWNER, creation.turn.answer_run_id) is None


# ---------------------------------------------------------------------------
# Deletion and retention
# ---------------------------------------------------------------------------


async def _hold_conversation_lock(conn: Any, conversation_id: str) -> Any:
    """Take the lock a submission holds, so a deleter has to wait behind it."""
    transaction = conn.transaction()
    await transaction.start()
    await conn.fetchrow(
        "SELECT 1 FROM web_conversations "
        "WHERE principal_id = $1 AND conversation_id = $2::text::uuid FOR UPDATE",
        _OWNER,
        conversation_id,
    )
    return transaction


async def _link_turn(conn: Any, runs: PGAnswerRunStore, conversation_id: str) -> str:
    """Create the run and its turn the way an accepted submission would."""
    request = _request("late")
    creation = await runs.create_run_in(
        conn,
        owner_id=_OWNER,
        request=request,
        idempotency_fingerprint=answer_run_request_fingerprint(request),
    )
    await conn.execute(
        "INSERT INTO web_conversation_turns "
        "(turn_id, principal_id, conversation_id, turn_number, submission_id, answer_run_id) "
        "VALUES ($1, $2, $3::text::uuid, 1, $4, $5::text::uuid)",
        uuid.uuid4(),
        _OWNER,
        conversation_id,
        uuid.uuid4(),
        creation.run.run_id,
    )
    return creation.run.run_id


async def test_deleting_a_conversation_never_orphans_a_turn_committed_behind_it(
    store: PGWebConversationStore, runs: PGAnswerRunStore, pool: Any
) -> None:
    """The deleter must snapshot run ids under the conversation lock, not before it."""
    conversation_id = await _conversation(store)

    async with pool.acquire() as holder:
        transaction = await _hold_conversation_lock(holder, conversation_id)
        deletion = asyncio.create_task(
            store.delete_conversation(_OWNER, conversation_id, ttl_days=_TTL_DAYS)
        )
        await asyncio.sleep(0.2)
        await _link_turn(holder, runs, conversation_id)
        await transaction.commit()

    assert await deletion is True
    assert await _count(pool, "web_conversation_turns") == 0
    assert await _count(pool, "dlightrag_answer_runs") == 0


async def test_deleting_every_conversation_never_orphans_a_turn_committed_behind_it(
    store: PGWebConversationStore, runs: PGAnswerRunStore, pool: Any
) -> None:
    conversation_id = await _conversation(store)

    async with pool.acquire() as holder:
        transaction = await _hold_conversation_lock(holder, conversation_id)
        deletion = asyncio.create_task(store.delete_all_conversations(_OWNER))
        await asyncio.sleep(0.2)
        await _link_turn(holder, runs, conversation_id)
        await transaction.commit()

    assert await deletion == 1
    assert await _count(pool, "web_conversation_turns") == 0
    assert await _count(pool, "dlightrag_answer_runs") == 0


async def test_deleting_a_conversation_deletes_its_runs_and_frees_its_bytes(
    store: PGWebConversationStore, pool: Any
) -> None:
    conversation_id = await _conversation(store)
    content = b"only-reference"
    digest = artifact_digest(content)
    creation = await _submit(
        store,
        conversation_id,
        artifacts=[PendingArtifact(content=content)],
        references=[
            PendingArtifactReference(
                resource_id="attachment-1",
                reference_kind="current_attachment",
                ordinal=1,
                digest=digest,
                filename="chart.png",
                mime_type="image/png",
            )
        ],
    )
    assert creation is not None
    await _finish(pool, creation.turn.answer_run_id, status="succeeded")

    assert await store.delete_conversation(_OWNER, conversation_id, ttl_days=_TTL_DAYS) is True

    assert await _count(pool, "dlightrag_answer_runs") == 0
    assert await _count(pool, "dlightrag_answer_run_artifacts") == 0
    assert await _count(pool, "dlightrag_answer_artifacts") == 0
    assert await _count(pool, "web_conversation_turns") == 0


async def test_deletion_keeps_bytes_another_run_still_references(
    store: PGWebConversationStore, pool: Any
) -> None:
    content = b"shared-bytes"
    digest = artifact_digest(content)
    reference = PendingArtifactReference(
        resource_id="attachment-1",
        reference_kind="current_attachment",
        ordinal=1,
        digest=digest,
        filename="chart.png",
        mime_type="image/png",
    )
    doomed = await _conversation(store)
    kept = await _conversation(store)
    await _submit(
        store, doomed, artifacts=[PendingArtifact(content=content)], references=[reference]
    )
    await _submit(store, kept, artifacts=[PendingArtifact(content=content)], references=[reference])

    await store.delete_conversation(_OWNER, doomed, ttl_days=_TTL_DAYS)

    assert await _count(pool, "dlightrag_answer_artifacts", owner_id=_OWNER, digest=digest) == 1
    assert await _count(pool, "dlightrag_answer_runs") == 1


async def test_deleting_every_conversation_deletes_every_linked_run(
    store: PGWebConversationStore, pool: Any
) -> None:
    mine = await _conversation(store, _OWNER)
    theirs = await _conversation(store, _OTHER_OWNER)
    await _submit(store, mine, owner=_OWNER)
    await _submit(store, theirs, owner=_OTHER_OWNER)

    assert await store.delete_all_conversations(_OWNER) == 1

    assert await _count(pool, "dlightrag_answer_runs", owner_id=_OWNER) == 0
    assert await _count(pool, "dlightrag_answer_runs", owner_id=_OTHER_OWNER) == 1


async def test_a_successful_linked_run_outlives_the_thirty_day_bound(
    store: PGWebConversationStore, runs: PGAnswerRunStore, pool: Any
) -> None:
    conversation_id = await _conversation(store)
    creation = await _submit(store, conversation_id)
    assert creation is not None
    await _finish(pool, creation.turn.answer_run_id, status="succeeded")
    await _backdate_finish(pool, creation.turn.answer_run_id, days=45)

    deletion = await runs.prune_expired_runs()

    assert deletion.runs == 0
    assert await _count(pool, "web_conversation_turns") == 1


async def test_an_expired_event_log_is_still_trimmed_for_a_linked_run(
    store: PGWebConversationStore, runs: PGAnswerRunStore, pool: Any
) -> None:
    """Retention exempts the run row, not its event log."""
    conversation_id = await _conversation(store)
    creation = await _submit(store, conversation_id)
    assert creation is not None
    await _finish(pool, creation.turn.answer_run_id, status="succeeded")
    await _backdate_finish(pool, creation.turn.answer_run_id, days=45)

    assert await runs.trim_expired_event_logs() == 1

    record = await runs.get_run(owner_id=_OWNER, run_id=creation.turn.answer_run_id)
    assert record is not None
    assert record.events_trimmed_at is not None
    assert record.result == {"answer": "done"}


@pytest.mark.parametrize("status", ["failed", "cancelled"])
async def test_a_failed_or_cancelled_linked_run_prunes_and_cascades_its_turn(
    store: PGWebConversationStore, runs: PGAnswerRunStore, pool: Any, status: str
) -> None:
    conversation_id = await _conversation(store)
    creation = await _submit(store, conversation_id)
    assert creation is not None
    await _finish(
        pool,
        creation.turn.answer_run_id,
        status=status,
        error="answer_stream_failed" if status == "failed" else None,
    )
    await _backdate_finish(pool, creation.turn.answer_run_id, days=45)

    deletion = await runs.prune_expired_runs()

    assert deletion.runs == 1
    assert await _count(pool, "web_conversation_turns") == 0
    assert await _count(pool, "web_conversations") == 1


async def test_an_unlinked_successful_run_still_prunes(
    store: PGWebConversationStore, runs: PGAnswerRunStore, pool: Any
) -> None:
    request = _request()
    creation = await runs.create_run(
        owner_id=_OWNER,
        request=request,
        idempotency_fingerprint=answer_run_request_fingerprint(request),
    )
    await _finish(pool, creation.run.run_id, status="succeeded")
    await _backdate_finish(pool, creation.run.run_id, days=45)

    assert (await runs.prune_expired_runs()).runs == 1


async def test_deleting_a_run_row_cascades_its_conversation_turn(
    store: PGWebConversationStore, runs: PGAnswerRunStore, pool: Any
) -> None:
    conversation_id = await _conversation(store)
    creation = await _submit(store, conversation_id)
    assert creation is not None

    await runs.delete_runs(owner_id=_OWNER, run_ids=[creation.turn.answer_run_id])

    assert await _count(pool, "web_conversation_turns") == 0


async def test_a_turn_cannot_reference_a_run_another_principal_owns(
    store: PGWebConversationStore, runs: PGAnswerRunStore, pool: Any
) -> None:
    """The foreign key carries the principal, so the link is owner-scoped."""
    conversation_id = await _conversation(store)
    request = _request()
    foreign = await runs.create_run(
        owner_id=_OTHER_OWNER,
        request=request,
        idempotency_fingerprint=answer_run_request_fingerprint(request),
    )

    async with pool.acquire() as conn:
        with pytest.raises(asyncpg.exceptions.ForeignKeyViolationError):
            await conn.execute(
                "INSERT INTO web_conversation_turns "
                "(turn_id, principal_id, conversation_id, turn_number, submission_id, "
                " answer_run_id) "
                "VALUES ($1::text::uuid, $2, $3::text::uuid, 1, $4::text::uuid, $5::text::uuid)",
                str(uuid.uuid4()),
                _OWNER,
                conversation_id,
                str(uuid.uuid4()),
                foreign.run.run_id,
            )


async def test_expired_conversations_prune_with_their_runs(
    store: PGWebConversationStore, pool: Any
) -> None:
    conversation_id = await _conversation(store)
    await _submit(store, conversation_id)
    async with pool.acquire() as conn:
        await conn.execute("UPDATE web_conversations SET updated_at = NOW() - INTERVAL '90 days'")

    assert await store.prune_expired(ttl_days=_TTL_DAYS) == 1

    assert await _count(pool, "dlightrag_answer_runs") == 0
    assert await _count(pool, "web_conversation_turns") == 0


async def test_trimming_the_conversation_window_removes_the_oldest_runs(
    store: PGWebConversationStore, pool: Any
) -> None:
    conversation_id = await _conversation(store)
    for index in range(3):
        await store.create_answer_turn(
            principal_id=_OWNER,
            conversation_id=conversation_id,
            submission_id=str(uuid.uuid4()),
            request=_request(f"question {index}"),
            idempotency_fingerprint=answer_run_request_fingerprint(_request(f"question {index}")),
            title_hint="why",
            max_turns=2,
            ttl_days=_TTL_DAYS,
        )

    assert await _count(pool, "web_conversation_turns") == 2
    assert await _count(pool, "dlightrag_answer_runs") == 2


# ---------------------------------------------------------------------------
# Retention/deletion lock order
# ---------------------------------------------------------------------------


async def test_conversation_deletion_takes_the_same_lock_order_as_run_retention(
    store: PGWebConversationStore, db_name: str, pool: Any
) -> None:
    """Both deletion paths must lock the run row before its conversation turn.

    Retention now runs on every process, so a conversation delete racing a prune
    is ordinary traffic. The blocking connection reproduces exactly what
    ``prune_expired_runs`` does: lock the expired run, then delete it and let the
    cascade take its turn. If conversation deletion removed the turn first, the
    two transactions would hold each other's next lock and PostgreSQL would abort
    one of them.
    """
    conversation_id = await _conversation(store)
    creation = await _submit(store, conversation_id)
    assert creation is not None
    run_uuid = uuid.UUID(creation.turn.answer_run_id)

    blocker = await asyncpg.connect(**{**_PG_CONN_KWARGS, "database": db_name})
    try:
        transaction = blocker.transaction()
        await transaction.start()
        await blocker.fetchrow(
            "SELECT run_id FROM dlightrag_answer_runs "
            "WHERE owner_id = $1 AND run_id = $2 FOR UPDATE",
            _OWNER,
            run_uuid,
        )

        deleting = asyncio.create_task(
            store.delete_conversation(_OWNER, conversation_id, ttl_days=_TTL_DAYS)
        )
        await asyncio.sleep(0.3)
        assert not deleting.done()

        await blocker.execute(
            "DELETE FROM dlightrag_answer_runs WHERE owner_id = $1 AND run_id = $2",
            _OWNER,
            run_uuid,
        )
        await transaction.commit()

        assert await asyncio.wait_for(deleting, timeout=10) is True
    finally:
        await blocker.close()

    assert await _count(pool, "web_conversations") == 0
    assert await _count(pool, "web_conversation_turns") == 0


# ---------------------------------------------------------------------------
# Reader validation
# ---------------------------------------------------------------------------


async def test_a_reader_validates_the_migrated_schema_without_ddl(pool: Any) -> None:
    writer = PGWebConversationStore(pool=pool)
    await writer.initialize()

    reader = PGWebConversationStore(pool=pool)
    await reader.initialize(validate_only=True)

    conversation_id = await _conversation(reader)
    creation = await _submit(reader, conversation_id)
    assert creation is not None
