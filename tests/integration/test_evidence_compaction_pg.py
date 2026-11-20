# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Evidence compaction: a finished Run's log keeps its identities, not its uncited text."""

from __future__ import annotations

import json
import uuid
from typing import Any

import asyncpg
import pytest

from dlightrag.adapters.postgres.runtime.run_store import PGRunStore
from tests.support.pg import drop_database, skip_without_postgres

_CITED = "web-cited"
_UNCITED = "web-uncited"
_OWNER = "owner-evidence-compaction"
_TEST_DATABASE = "dlightrag_evidence_compaction_test"


def _target() -> dict[str, Any]:
    from tests.support.pg import PG_CONN_KWARGS

    return dict(PG_CONN_KWARGS)


@pytest.fixture
async def pool() -> Any:
    await skip_without_postgres()
    admin = await asyncpg.connect(**_target())
    try:
        await admin.execute(f'CREATE DATABASE "{_TEST_DATABASE}"')
    except asyncpg.DuplicateDatabaseError:
        pass
    finally:
        await admin.close()
    created = await asyncpg.create_pool(
        **{**_target(), "database": _TEST_DATABASE}, min_size=1, max_size=4
    )
    try:
        yield created
    finally:
        await created.close()
        await drop_database(_TEST_DATABASE)


async def _store(pool: Any) -> PGRunStore:
    store = PGRunStore(pool=pool)
    await store.initialize()
    return store


def _payload() -> bytes:
    return json.dumps(
        {
            "contexts": {
                "chunks": [
                    {
                        "_source_reference_id": _CITED,
                        "chunk_id": "chunk-cited",
                        "content": "the passage the answer stands on",
                        "metadata": {"title": "Cited", "source_uri": "https://cited"},
                    },
                    {
                        "_source_reference_id": _UNCITED,
                        "chunk_id": "chunk-uncited",
                        "content": "a passage the answer never cited",
                        "metadata": {"title": "Other", "source_uri": "https://other"},
                    },
                ]
            },
            "source_ids": [_CITED, _UNCITED],
        },
        ensure_ascii=False,
    ).encode("utf-8")


async def _insert_running_run(pool: Any, run_id: uuid.UUID) -> None:
    """A Run of the same conversation that has not finished: never a compaction candidate."""
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO dlightrag_runs (
                owner_id, run_id, run_kind, lane, submitted_by, access_scope_kind,
                submission_key, prepared_input_json, accepted_input_json,
                request_fingerprint, status, phase, fencing_epoch,
                durable_progress_version, last_reclaim_progress_version,
                reclaims_without_progress, next_event_sequence, retention_seconds
            ) VALUES (
                $1, $2, 'answer', 'query', $1, 'owner', $3, '{}'::jsonb, '{}'::jsonb,
                $4, 'running', 'answering', 0, 0, 0, 0, 1, 86400
            )
            """,
            _OWNER,
            run_id,
            str(run_id),
            "e" * 64,
        )


async def _insert_run(pool: Any, run_id: uuid.UUID, *, finished_minutes_ago: int) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO dlightrag_runs (
                owner_id, run_id, run_kind, lane, submitted_by, access_scope_kind,
                submission_key, accepted_input_json, request_fingerprint, status, phase,
                result_json, fencing_epoch, durable_progress_version,
                last_reclaim_progress_version, reclaims_without_progress,
                next_event_sequence, retention_seconds, finished_at
            ) VALUES (
                $1, $2, 'answer', 'query', $1, 'owner', $3, '{}'::jsonb, $4, 'succeeded',
                'done', $5::jsonb, 0, 0, 0, 0, 1, 86400,
                NOW() - make_interval(mins => $6::int)
            )
            """,
            _OWNER,
            run_id,
            str(run_id),
            "f" * 64,
            json.dumps({"sources": [{"id": _CITED}]}),
            finished_minutes_ago,
        )


async def _insert_evidence(
    pool: Any, run_id: uuid.UUID, session_id: uuid.UUID, ordinal: int, *, minutes_ago: int
) -> None:
    payload = _payload()
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO dlightrag_answer_evidence (
                owner_id, run_id, session_id, intent_id, result_ordinal,
                content_digest, locator_digest, content, locator, created_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, '{}'::bytea,
                      NOW() - make_interval(mins => $9::int))
            """,
            _OWNER,
            run_id,
            session_id,
            uuid.uuid7(),
            ordinal,
            "a" * 64,
            "b" * 64,
            payload,
            minutes_ago,
        )


async def _rows(pool: Any, run_id: uuid.UUID) -> list[Any]:
    async with pool.acquire() as conn:
        return list(
            await conn.fetch(
                "SELECT result_ordinal, content, content_slimmed"
                " FROM dlightrag_answer_evidence"
                " WHERE owner_id = $1 AND run_id = $2 ORDER BY created_at",
                _OWNER,
                run_id,
            )
        )


async def test_a_finished_run_keeps_the_newest_row_and_the_cited_passage(pool: Any) -> None:
    store = await _store(pool)
    run_id, session_id = uuid.uuid7(), uuid.uuid7()
    await _insert_run(pool, run_id, finished_minutes_ago=60)
    for ordinal, minutes_ago in enumerate((30, 20, 10)):
        await _insert_evidence(pool, run_id, session_id, ordinal, minutes_ago=minutes_ago)

    assert await store.compact_evidence_logs() == 1

    rows = await _rows(pool, run_id)
    newest = rows[-1]
    assert newest["content_slimmed"] is False
    for row in rows[:-1]:
        assert row["content_slimmed"] is True
        cited, uncited = json.loads(bytes(row["content"]))["contexts"]["chunks"]
        assert cited["content"] == "the passage the answer stands on"
        assert "content" not in uncited
        assert uncited["metadata"] == {"title": "Other", "source_uri": "https://other"}
        assert uncited["chunk_id"] == "chunk-uncited"

    # A compacted Run is no longer a candidate, so draining is finite.
    assert await store.compact_evidence_logs() == 0


async def test_a_run_inside_the_grace_period_is_left_whole(pool: Any) -> None:
    store = await _store(pool)
    run_id, session_id = uuid.uuid7(), uuid.uuid7()
    await _insert_run(pool, run_id, finished_minutes_ago=1)
    await _insert_evidence(pool, run_id, session_id, 0, minutes_ago=30)
    await _insert_evidence(pool, run_id, session_id, 1, minutes_ago=20)

    assert await store.compact_evidence_logs() == 0

    assert all(not row["content_slimmed"] for row in await _rows(pool, run_id))


async def test_a_run_that_cited_nothing_keeps_only_its_newest_row_whole(pool: Any) -> None:
    store = await _store(pool)
    run_id, session_id = uuid.uuid7(), uuid.uuid7()
    await _insert_run(pool, run_id, finished_minutes_ago=60)
    async with pool.acquire() as conn:
        await conn.execute(
            'UPDATE dlightrag_runs SET result_json = \'{"status": "cancelled"}\'::jsonb'
            " WHERE owner_id = $1 AND run_id = $2",
            _OWNER,
            run_id,
        )
    for ordinal, minutes_ago in enumerate((30, 20, 10)):
        await _insert_evidence(pool, run_id, session_id, ordinal, minutes_ago=minutes_ago)

    assert await store.compact_evidence_logs() == 1

    rows = await _rows(pool, run_id)
    assert rows[-1]["content_slimmed"] is False
    for row in rows[:-1]:
        document = json.loads(bytes(row["content"]))
        assert all("content" not in chunk for chunk in document["contexts"]["chunks"])


async def test_a_new_run_of_the_same_session_is_untouched_by_compaction(pool: Any) -> None:
    """Compaction is run-scoped: a conversation that is answering right now keeps its rows."""
    store = await _store(pool)
    finished, running = uuid.uuid7(), uuid.uuid7()
    session_id = uuid.uuid7()
    await _insert_run(pool, finished, finished_minutes_ago=60)
    for ordinal, minutes_ago in enumerate((30, 20, 10)):
        await _insert_evidence(pool, finished, session_id, ordinal, minutes_ago=minutes_ago)
    # The live Run writes newer rows of the same Agent Session while compaction runs.
    await _insert_running_run(pool, running)
    for ordinal, minutes_ago in enumerate((1, 0)):
        await _insert_evidence(pool, running, session_id, ordinal, minutes_ago=minutes_ago)
    live_before = [bytes(row["content"]) for row in await _rows(pool, running)]

    assert await store.compact_evidence_logs() == 1

    # The finished Run's log compacted, newest row excepted, without regard to the live
    # Run's newer rows: reads are run-scoped, so each Run keeps what its own reader needs.
    own_rows = await _rows(pool, finished)
    assert own_rows[-1]["content_slimmed"] is False
    assert all(row["content_slimmed"] for row in own_rows[:-1])
    # The live Run's ledger is byte-for-byte what it wrote.
    assert [bytes(row["content"]) for row in await _rows(pool, running)] == live_before
    assert all(not row["content_slimmed"] for row in await _rows(pool, running))
    # A Run in progress is never a candidate, so it cannot be compacted mid-answer.
    assert await store.compact_evidence_logs() == 0


async def test_the_column_says_which_rows_lost_text_and_the_payload_is_the_document(
    pool: Any,
) -> None:
    """A SQL reader loads the payload straight from the row; the column marks its state."""
    store = await _store(pool)
    run_id, session_id = uuid.uuid7(), uuid.uuid7()
    await _insert_run(pool, run_id, finished_minutes_ago=60)
    for ordinal, minutes_ago in enumerate((30, 20, 10)):
        await _insert_evidence(pool, run_id, session_id, ordinal, minutes_ago=minutes_ago)

    assert await store.compact_evidence_logs() == 1

    rows = await _rows(pool, run_id)
    newest, older = rows[-1], rows[0]
    assert newest["content_slimmed"] is False
    assert (
        json.loads(bytes(newest["content"]))["contexts"]["chunks"][0]["content"]
        == "the passage the answer stands on"
    )
    assert older["content_slimmed"] is True
    cited, uncited = json.loads(bytes(older["content"]))["contexts"]["chunks"]
    assert cited["content"] == "the passage the answer stands on"
    assert "content" not in uncited


async def _insert_child(pool: Any, run_id: uuid.UUID, child_session_id: uuid.UUID) -> None:
    async with pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO dlightrag_answer_child_sessions (
                owner_id, run_id, child_session_id, parent_session_id, parent_call_id,
                status, depth, context_snapshot_json, host_state_json, fencing_epoch
            ) VALUES ($1, $2, $3, $4, $5, 'succeeded', 1, '{}'::jsonb, $6::jsonb, 0)
            """,
            _OWNER,
            run_id,
            child_session_id,
            uuid.uuid7(),
            "call-1",
            json.dumps({"sandbox": "live", "browser": {"page": "http://shop.example/details"}}),
        )
        await conn.execute(
            """
            INSERT INTO dlightrag_answer_child_operations (
                owner_id, run_id, child_session_id, operation_sequence, operation_id,
                idempotency_key, request_fingerprint, content, origin, status, outcome_json
            ) VALUES ($1, $2, $3, 1, $4, 'idem-1', $5, 'continue', 'user', 'succeeded',
                      $6::jsonb)
            """,
            _OWNER,
            run_id,
            child_session_id,
            uuid.uuid7(),
            "f" * 64,
            json.dumps({"receipt": "x" * 5000}),
        )


async def _child_state(pool: Any, run_id: uuid.UUID) -> tuple[Any, Any, Any]:
    async with pool.acquire() as conn:
        child = await conn.fetchrow(
            "SELECT host_state_json, status, usage_json"
            " FROM dlightrag_answer_child_sessions"
            " WHERE owner_id = $1 AND run_id = $2",
            _OWNER,
            run_id,
        )
        operation = await conn.fetchrow(
            "SELECT outcome_json, status, summary"
            " FROM dlightrag_answer_child_operations"
            " WHERE owner_id = $1 AND run_id = $2",
            _OWNER,
            run_id,
        )
    return child, operation, None


async def test_a_finished_run_gives_up_the_state_its_children_ran_on(pool: Any) -> None:
    """What a live Run's controls need is gone once no control can reach them again."""
    store = await _store(pool)
    run_id, child_id = uuid.uuid7(), uuid.uuid7()
    await _insert_run(pool, run_id, finished_minutes_ago=60)
    await _insert_child(pool, run_id, child_id)

    assert await store.clear_finished_run_working_state() == 1

    child, operation, _ = await _child_state(pool, run_id)
    assert json.loads(child["host_state_json"]) == {}
    assert json.loads(operation["outcome_json"]) == {}
    # What a reader surfaces stays.
    assert child["status"] == "succeeded"
    assert operation["status"] == "succeeded"
    assert await store.clear_finished_run_working_state() == 0


async def test_a_live_run_keeps_the_state_its_children_run_on(pool: Any) -> None:
    store = await _store(pool)
    run_id, child_id = uuid.uuid7(), uuid.uuid7()
    await _insert_run(pool, run_id, finished_minutes_ago=60)
    await _insert_child(pool, run_id, child_id)
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE dlightrag_runs SET status = 'running', finished_at = NULL,"
            " prepared_input_json = '{}'::jsonb"
            " WHERE owner_id = $1 AND run_id = $2",
            _OWNER,
            run_id,
        )

    assert await store.clear_finished_run_working_state() == 0

    child, operation, _ = await _child_state(pool, run_id)
    assert json.loads(child["host_state_json"]) != {}
    assert json.loads(operation["outcome_json"]) != {}
