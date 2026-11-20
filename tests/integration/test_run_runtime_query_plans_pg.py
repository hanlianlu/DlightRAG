# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Representative RunRuntime admission, claim, FIFO, and cursor query plans."""

from __future__ import annotations

import uuid

import pytest

from dlightrag.adapters.postgres.runtime.run_store import _COUNT_OWNER_ACTIVE_ANSWERS
from dlightrag.engine.runtime.policy import MAX_RECLAIMS_WITHOUT_PROGRESS
from tests.integration.run_runtime_pg_harness import isolated_run_runtime, run_envelope
from tests.support.pg import skip_without_postgres

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]


async def _plan(connection, sql: str, *args) -> str:
    rows = await connection.fetch(f"EXPLAIN (ANALYZE, BUFFERS, COSTS OFF) {sql}", *args)
    return "\n".join(str(row["QUERY PLAN"]) for row in rows)


async def test_runtime_backlog_queries_use_bounded_indexes() -> None:
    await skip_without_postgres()
    async with isolated_run_runtime("runtime_plans") as (store, pool):
        for index in range(40):
            await store.accept_run(
                envelope=run_envelope("retrieval", key=f"query-{index:03d}"),
                run_id=str(uuid.uuid7()),
            )
        for index in range(20):
            await store.accept_run(
                envelope=run_envelope(
                    "corpus_mutation",
                    key=f"mutation-{index:03d}",
                    workspace=f"workspace-{index % 5:03d}",
                ),
                run_id=str(uuid.uuid7()),
            )
        claimed = await store.claim_next(
            worker_id="event-writer", run_kinds=("retrieval",), lanes=("query",)
        )
        assert claimed is not None
        await store.append_event(
            owner_id=claimed.run.owner_id,
            run_id=claimed.run.run_id,
            worker_id="event-writer",
            fencing_epoch=claimed.run.fencing_epoch,
            phase="searching",
            event_type="progress",
            payload={"step": 1},
        )

        async with pool.acquire() as connection, connection.transaction():
            await connection.execute("ANALYZE dlightrag_runs")
            await connection.execute("ANALYZE dlightrag_run_events")
            await connection.execute("SET LOCAL enable_seqscan = off")
            admission_limit = await _plan(
                connection,
                "SELECT COUNT(*) FROM dlightrag_runs "
                "WHERE lane = $1 AND status IN ('queued', 'running')",
                "corpus_mutation",
            )
            mutation_fifo = await _plan(
                connection,
                "SELECT r.owner_id, r.run_id FROM dlightrag_runs r "
                "WHERE r.run_kind = ANY($2::text[]) AND r.lane = ANY($3::text[]) "
                "AND r.cancel_requested_at IS NULL "
                "AND (r.next_attempt_at IS NULL OR r.next_attempt_at <= NOW()) "
                "AND (r.status = 'queued' OR "
                "(r.status = 'running' AND r.lease_expires_at < NOW() "
                "AND r.reclaims_without_progress < $1)) "
                "AND (r.lane <> 'corpus_mutation' OR NOT EXISTS ("
                "SELECT 1 FROM dlightrag_runs earlier "
                "WHERE earlier.lane = 'corpus_mutation' "
                "AND earlier.owner_id = r.owner_id "
                "AND earlier.status IN ('queued', 'running') "
                "AND (earlier.created_at, earlier.run_id) < (r.created_at, r.run_id))) "
                "ORDER BY r.created_at, r.run_id LIMIT 1",
                MAX_RECLAIMS_WITHOUT_PROGRESS,
                ["corpus_mutation"],
                ["corpus_mutation"],
            )
            events = await _plan(
                connection,
                "SELECT event_sequence, event_type, payload, created_at "
                "FROM dlightrag_run_events "
                "WHERE owner_id = $1 AND run_id = $2 AND event_sequence > $3 "
                "ORDER BY event_sequence LIMIT 500",
                claimed.run.owner_id,
                uuid.UUID(claimed.run.run_id),
                0,
            )

        assert "idx_dlightrag_runs_claim" in admission_limit
        assert "idx_dlightrag_runs_claim" in mutation_fifo
        assert "idx_dlightrag_runs_mutation_fifo" in mutation_fifo
        assert "dlightrag_run_events_pkey" in events


# A deployment that tells no owners apart has one owner for every Run it ever accepted, so this
# owner's history is the table, and its few active Answers are what a limit counts.
_HISTORY = """
INSERT INTO dlightrag_runs (
    owner_id, run_id, run_kind, lane, submitted_by, access_scope_kind, submission_key,
    prepared_input_json, accepted_input_json, request_fingerprint, status, result_json,
    finished_at, retention_seconds, purge_after
)
SELECT $1, gen_random_uuid(), 'answer', 'query', $1, 'owner', $1 || '-done-' || n,
    NULL, '{}'::jsonb, 'fingerprint', 'succeeded', '{"answer": "x"}'::jsonb,
    NOW() - make_interval(secs => n), 31536000, NOW() + INTERVAL '300 days'
FROM generate_series(1, $2) AS n
"""
_ACTIVE = """
INSERT INTO dlightrag_runs (
    owner_id, run_id, run_kind, lane, submitted_by, access_scope_kind, submission_key,
    prepared_input_json, accepted_input_json, request_fingerprint, status, retention_seconds
)
SELECT $1, gen_random_uuid(), 'answer', 'query', $1, 'owner', $1 || '-live-' || n,
    '{"query": "q"}'::jsonb, '{}'::jsonb, 'fingerprint', $3, 31536000
FROM generate_series(1, $2) AS n
"""
_NEIGHBOURS = """
INSERT INTO dlightrag_runs (
    owner_id, run_id, run_kind, lane, submitted_by, access_scope_kind, submission_key,
    prepared_input_json, accepted_input_json, request_fingerprint, status, retention_seconds
)
SELECT 'neighbour-' || n, gen_random_uuid(), 'answer', 'query', 'neighbour-' || n, 'owner',
    'neighbour-' || n || '-live', '{"query": "q"}'::jsonb, '{}'::jsonb, 'fingerprint', 'queued',
    31536000
FROM generate_series(1, $1) AS n
"""


async def test_an_owners_active_answers_are_counted_without_walking_the_rest() -> None:
    await skip_without_postgres()
    async with isolated_run_runtime("owner_active_plan") as (_store, pool):
        async with pool.acquire() as connection:
            await connection.execute(_HISTORY, "the-deployment", 30_000)
            await connection.execute(_ACTIVE, "the-deployment", 4, "queued")
            await connection.execute(_ACTIVE, "another-owner", 3, "running")
            await connection.execute(_NEIGHBOURS, 8_000)
            # A maintained table, as autovacuum keeps it: an index-only scan is costed by how many
            # of its pages the visibility map shows all-visible.
            await connection.execute("VACUUM (ANALYZE) dlightrag_runs")

            plan = await _plan(connection, _COUNT_OWNER_ACTIVE_ANSWERS, "the-deployment")
            held = await connection.fetchval(_COUNT_OWNER_ACTIVE_ANSWERS, "the-deployment")

        assert held == 4
        assert "idx_dlightrag_runs_owner_active" in plan
        # The count reads the owner's active Answers and nothing of their history or of
        # anyone else's active Runs, which is what a filter would have thrown away.
        assert "Rows Removed by Filter" not in plan
