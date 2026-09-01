# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL-backed ingest job state.

Commit 3 adds the promotion trigger: one successful committed ingest window
atomically inserts a durable counter ledger event (idempotent per job/window),
increments the workspace registry's monotonic doc/chunk totals, evaluates the
configured promotion thresholds, idempotently enqueues one promotion job, and
flips promotion observability — all in the window's transaction, so a crash
between the window update and the counter can never double-count on replay.
"""

import json
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

from dlightrag.adapters.postgres.core._migrations import Migration, apply_migrations
from dlightrag.adapters.postgres.core._operations import PostgresOperationRunner
from dlightrag.adapters.postgres.corpus.workspace_write_gate import (
    _active_fence_seconds,
    workspace_write_gate,
)
from dlightrag.engine.rag.corpus.ingest_jobs import (
    JOB_ABANDONED_ERROR,
    JOB_LEASE_SECONDS,
    JOB_ORPHAN_AFTER_SECONDS,
    JOB_RETENTION_SECONDS,
    IngestJobSchemaError,
)

TABLE = "dlightrag_ingest_jobs"
# Caps every bulk statement here, not just the pruning ones.
_BATCH_LIMIT = 1000

_CREATE = """
CREATE TABLE IF NOT EXISTS dlightrag_ingest_jobs (
    job_id          TEXT PRIMARY KEY,
    workspace       TEXT NOT NULL,
    source_type     TEXT NOT NULL,
    status          TEXT NOT NULL,
    request_json    JSONB NOT NULL DEFAULT '{}'::jsonb,
    total_items     INTEGER NOT NULL DEFAULT 0,
    processed_items INTEGER NOT NULL DEFAULT 0,
    failed_items    INTEGER NOT NULL DEFAULT 0,
    current_window  INTEGER NOT NULL DEFAULT 0,
    result_json     JSONB NOT NULL DEFAULT '{}'::jsonb,
    errors          JSONB NOT NULL DEFAULT '[]'::jsonb,
    errors_truncated BOOLEAN NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at      TIMESTAMPTZ,
    finished_at     TIMESTAMPTZ,
    lease_owner     TEXT,
    lease_expires_at TIMESTAMPTZ
)
"""

_CREATE_INDEXES = (
    f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_workspace_created "
    f"ON {TABLE} (workspace, created_at DESC)",
    f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_status_updated ON {TABLE} (status, updated_at DESC)",
    f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_status_lease ON {TABLE} (status, lease_expires_at)",
)

_CREATE_ACTIVE_RETRY_UNIQUE = f"""
CREATE UNIQUE INDEX IF NOT EXISTS idx_{TABLE}_one_active_failed_retry
ON {TABLE} (workspace, source_type)
WHERE source_type = 'retry_failed' AND status IN ('queued', 'running')
"""

_RETRY_ITEMS_TABLE = "dlightrag_failed_retry_items"
_ADD_RETRY_COHORT_SEALED = f"""
ALTER TABLE {TABLE}
ADD COLUMN IF NOT EXISTS retry_cohort_sealed BOOLEAN NOT NULL DEFAULT FALSE
"""
_ADD_ERRORS_TRUNCATED = f"""
ALTER TABLE {TABLE}
ADD COLUMN IF NOT EXISTS errors_truncated BOOLEAN NOT NULL DEFAULT FALSE
"""
_FAIL_ACTIVE_LEGACY_RETRIES = f"""
UPDATE {TABLE}
SET status = 'failed',
    result_json = jsonb_build_object(
        'upgrade_interrupted', TRUE,
        'retried', total_items,
        'succeeded', processed_items,
        'failed', failed_items
    ),
    errors = (
        SELECT COALESCE(jsonb_agg(value ORDER BY ordinal), '[]'::jsonb)
        FROM (
            SELECT value, ordinal
            FROM jsonb_array_elements(
                errors || '["failed-document retry interrupted by durable-ledger upgrade"]'::jsonb
            ) WITH ORDINALITY AS entry(value, ordinal)
            ORDER BY ordinal
            LIMIT 200
        ) AS retained
    ),
    errors_truncated = errors_truncated OR jsonb_array_length(errors) >= 200,
    lease_owner = NULL,
    lease_expires_at = NULL,
    updated_at = clock_timestamp(),
    finished_at = clock_timestamp()
WHERE source_type = 'retry_failed'
  AND status IN ('queued', 'running')
  AND retry_cohort_sealed = FALSE
"""  # noqa: S608 - interpolates only the private table constant
_CREATE_RETRY_ITEMS = f"""
CREATE TABLE IF NOT EXISTS {_RETRY_ITEMS_TABLE} (
    job_id      TEXT NOT NULL REFERENCES {TABLE}(job_id) ON DELETE CASCADE,
    doc_id      TEXT NOT NULL,
    outcome     TEXT NOT NULL DEFAULT 'pending'
                CHECK (outcome IN ('pending', 'succeeded', 'failed')),
    summary_json JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (job_id, doc_id)
)
"""

# Durable idempotency ledger: one event per ingest job/window. Replayed
# windows (lease lost and reclaimed) hit the primary key and must not move the
# registry counters again.
_COUNTER_TABLE = "dlightrag_ingest_counters"
_CREATE_COUNTERS = """
CREATE TABLE IF NOT EXISTS dlightrag_ingest_counters (
    job_id      TEXT NOT NULL,
    window_number INTEGER NOT NULL,
    workspace   TEXT NOT NULL,
    docs        BIGINT NOT NULL,
    chunks      BIGINT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (job_id, window_number),
    CONSTRAINT dlightrag_ingest_counters_nonnegative
        CHECK (window_number > 0 AND docs >= 0 AND chunks >= 0)
)
"""

_INSERT_COUNTER_EVENT = """
INSERT INTO dlightrag_ingest_counters (job_id, window_number, workspace, docs, chunks)
VALUES ($1, $2, $3, $4, $5)
ON CONFLICT (job_id, window_number) DO NOTHING
RETURNING 1
"""

# Locks the job row and proves the current running lease before any ledger or
# progress mutation; a lost/expired lease or a non-running row refuses here.
_RECORD_WINDOW_GUARD = """
SELECT status, workspace
FROM dlightrag_ingest_jobs
WHERE job_id = $1
  AND status = 'running'
  AND lease_owner = $2
  AND lease_expires_at > clock_timestamp()
FOR UPDATE
"""

# One statement per successful window: monotonic totals, threshold evaluation,
# and promotion observability. `promotion_state` moves to 'pending' only from
# 'none' on a shared tier, so already-hot workspaces never re-enqueue and the
# pending state survives until the worker picks the job up.
_ADD_COUNTS_AND_TRIGGER = """
WITH updated AS (
    UPDATE dlightrag_workspace_meta
    SET ingested_docs_total = ingested_docs_total + $2,
        ingested_chunks_total = ingested_chunks_total + $3,
        promotion_state = CASE
            WHEN promotion_state = 'none' AND storage_tier = 'shared'
                 AND (($4::bigint IS NOT NULL
                       AND ingested_docs_total + $2 >= $4::bigint)
                      OR ($5::bigint IS NOT NULL
                          AND ingested_chunks_total + $3 >= $5::bigint))
            THEN 'pending' ELSE promotion_state END,
        updated_at = NOW()
    WHERE workspace = $1
    RETURNING workspace, promotion_state
)
INSERT INTO dlightrag_promotion_jobs (workspace)
SELECT workspace FROM updated WHERE promotion_state = 'pending'
ON CONFLICT (workspace) WHERE state IN ('pending', 'promoting', 'failed') DO NOTHING
"""

_RELEASE_RUNNING = """
UPDATE dlightrag_ingest_jobs
SET status = 'queued',
    lease_owner = NULL,
    lease_expires_at = NULL,
    updated_at = NOW()
WHERE job_id = $1
  AND status = 'running'
  AND lease_owner = $2
  AND lease_expires_at > clock_timestamp()
"""

# A queued job waiting out a promotion fence refreshes its liveness so the
# sweeper's orphan window (12 leases) never marks it abandoned while a
# long promotion renews its own lease.
_TOUCH_QUEUED = """
UPDATE dlightrag_ingest_jobs
SET updated_at = NOW()
WHERE job_id = $1
  AND status = 'queued'
  AND lease_owner IS NULL
RETURNING 1
"""

# Explicit user cancellation of a still-queued job is terminal: no lease
# exists yet, so the transition is ownerless and guarded only by the legal
# queued state. Shutdown cancellation never calls this — recovered rows stay
# queued for the next startup.
_CANCEL_QUEUED = """
WITH updated AS (
    UPDATE dlightrag_ingest_jobs
    SET status = 'failed',
        errors = (
            SELECT COALESCE(jsonb_agg(value ORDER BY ordinal), '[]'::jsonb)
            FROM (
                SELECT value, ordinal
                FROM jsonb_array_elements(errors || $2::jsonb)
                    WITH ORDINALITY AS entry(value, ordinal)
                ORDER BY ordinal
                LIMIT $3
            ) AS retained
        ),
        errors_truncated = errors_truncated
            OR jsonb_array_length(errors) + jsonb_array_length($2::jsonb) > $3,
        lease_owner = NULL,
        lease_expires_at = NULL,
        updated_at = NOW(),
        finished_at = NOW()
    WHERE job_id = $1
      AND status = 'queued'
      AND lease_owner IS NULL
    RETURNING 1
)
SELECT COUNT(*)::int FROM updated
"""

_FAIL_INVALID_RECOVERABLE = """
WITH updated AS (
    UPDATE dlightrag_ingest_jobs
    SET status = 'failed',
        errors = (
            SELECT COALESCE(jsonb_agg(value ORDER BY ordinal), '[]'::jsonb)
            FROM (
                SELECT value, ordinal
                FROM jsonb_array_elements(errors || $2::jsonb)
                    WITH ORDINALITY AS entry(value, ordinal)
                ORDER BY ordinal
                LIMIT $3
            ) AS retained
        ),
        errors_truncated = errors_truncated
            OR jsonb_array_length(errors) + jsonb_array_length($2::jsonb) > $3,
        lease_owner = NULL,
        lease_expires_at = NULL,
        updated_at = NOW(),
        finished_at = NOW()
    WHERE job_id = $1
      AND (
          (status = 'queued' AND lease_owner IS NULL)
          OR (status = 'running'
              AND (lease_expires_at IS NULL OR lease_expires_at < clock_timestamp()))
      )
    RETURNING 1
)
SELECT COUNT(*)::int FROM updated
"""

_INSERT = """
INSERT INTO dlightrag_ingest_jobs (job_id, workspace, source_type, status, request_json)
VALUES ($1, $2, $3, 'queued', $4::jsonb)
"""

_START_FAILED_RETRY = """
INSERT INTO dlightrag_ingest_jobs (job_id, workspace, source_type, status, request_json)
VALUES ($1, $2, 'retry_failed', 'queued', $3::jsonb)
ON CONFLICT (workspace, source_type)
WHERE source_type = 'retry_failed' AND status IN ('queued', 'running')
DO NOTHING
RETURNING
job_id, workspace, source_type, status, request_json, total_items,
processed_items, failed_items, current_window, result_json, errors,
errors_truncated,
created_at, updated_at, started_at, finished_at, lease_owner, lease_expires_at
"""

_GET_LATEST_FAILED_RETRY = """
SELECT
job_id, workspace, source_type, status, request_json, total_items,
processed_items, failed_items, current_window, result_json, errors,
errors_truncated,
created_at, updated_at, started_at, finished_at, lease_owner, lease_expires_at
FROM dlightrag_ingest_jobs
WHERE workspace = $1 AND source_type = 'retry_failed'
ORDER BY created_at DESC, job_id DESC
LIMIT 1
"""

_CLAIM_RUNNING = """
UPDATE dlightrag_ingest_jobs
SET status = 'running',
    lease_owner = $2,
    lease_expires_at = clock_timestamp() + ($3 * INTERVAL '1 second'),
    started_at = COALESCE(started_at, clock_timestamp()),
    updated_at = clock_timestamp()
WHERE job_id = $1
  AND status IN ('queued', 'running')
  AND (lease_owner = $2 OR lease_expires_at IS NULL OR lease_expires_at < clock_timestamp())
RETURNING
job_id, workspace, source_type, status, request_json, total_items,
processed_items, failed_items, current_window, result_json, errors,
errors_truncated,
created_at, updated_at, started_at, finished_at, lease_owner, lease_expires_at
"""

# Lock ownership without an expiry predicate first. PostgreSQL may evaluate an
# UPDATE predicate before it waits for a row lock, so the wall-clock expiry
# check must be a second statement executed after this lock is held.
_HEARTBEAT_LOCK = """
SELECT 1
FROM dlightrag_ingest_jobs
WHERE job_id = $1
  AND lease_owner = $2
  AND status = 'running'
FOR UPDATE
"""

_HEARTBEAT = """
WITH updated AS (
    UPDATE dlightrag_ingest_jobs
    SET lease_expires_at = clock_timestamp() + ($3 * INTERVAL '1 second'),
        updated_at = clock_timestamp()
    WHERE job_id = $1
      AND lease_owner = $2
      AND status = 'running'
      AND lease_expires_at > clock_timestamp()
    RETURNING 1
)
SELECT COUNT(*)::int FROM updated
"""

# Cap retained per-job error entries so the JSONB column (and status polls that
# deserialize it) stay bounded on large, high-failure ingests.
_MAX_JOB_ERRORS = 200

_RECORD_WINDOW = """
WITH updated AS (
    UPDATE dlightrag_ingest_jobs
    SET total_items = total_items + $2,
        processed_items = processed_items + $3,
        failed_items = failed_items + $4,
        current_window = $5,
        errors = (
            SELECT COALESCE(jsonb_agg(value ORDER BY ordinal), '[]'::jsonb)
            FROM (
                SELECT value, ordinal
                FROM jsonb_array_elements(errors || $6::jsonb)
                    WITH ORDINALITY AS entry(value, ordinal)
                ORDER BY ordinal
                LIMIT $9
            ) AS retained
        ),
        errors_truncated = errors_truncated
            OR jsonb_array_length(errors) + jsonb_array_length($6::jsonb) > $9,
        lease_expires_at = clock_timestamp() + ($8 * INTERVAL '1 second'),
        updated_at = clock_timestamp()
    WHERE job_id = $1
      AND lease_owner = $7
    RETURNING 1
)
SELECT COUNT(*)::int FROM updated
"""

_FINISH = """
WITH updated AS (
    UPDATE dlightrag_ingest_jobs
    SET status = CASE WHEN failed_items > 0 THEN 'partial' ELSE 'succeeded' END,
        result_json = $2::jsonb,
        lease_owner = NULL,
        lease_expires_at = NULL,
        updated_at = NOW(),
        finished_at = NOW()
    WHERE job_id = $1
      AND lease_owner = $3
      AND status = 'running'
      AND lease_expires_at > clock_timestamp()
    RETURNING 1
)
SELECT COUNT(*)::int FROM updated
"""

_SEAL_RETRY_COHORT_GUARD = """
SELECT retry_cohort_sealed
FROM dlightrag_ingest_jobs
WHERE job_id = $1
  AND source_type = 'retry_failed'
  AND status = 'running'
  AND lease_owner = $2
  AND lease_expires_at > clock_timestamp()
FOR UPDATE
"""

_INSERT_RETRY_COHORT = f"""
INSERT INTO {_RETRY_ITEMS_TABLE} (job_id, doc_id)
SELECT $1, doc_id FROM unnest($2::text[]) AS doc_id
ON CONFLICT (job_id, doc_id) DO NOTHING
"""  # noqa: S608 - interpolates only the private table constant

_MARK_RETRY_COHORT_SEALED = """
UPDATE dlightrag_ingest_jobs
SET retry_cohort_sealed = TRUE,
    lease_expires_at = clock_timestamp() + ($3 * INTERVAL '1 second'),
    updated_at = clock_timestamp()
WHERE job_id = $1
  AND source_type = 'retry_failed'
  AND lease_owner = $2
  AND status = 'running'
  AND lease_expires_at > clock_timestamp()
"""

_LIST_UNFINISHED_RETRY_ITEMS = f"""
SELECT item.doc_id
FROM {_RETRY_ITEMS_TABLE} AS item
JOIN dlightrag_ingest_jobs AS job ON job.job_id = item.job_id
WHERE item.job_id = $1
  AND item.outcome = 'pending'
  AND job.retry_cohort_sealed = TRUE
  AND job.status = 'running'
  AND job.lease_owner = $2
  AND job.lease_expires_at > clock_timestamp()
ORDER BY item.doc_id
"""  # noqa: S608 - interpolates only the private table constant

_RETRY_COHORT_STATE = """
SELECT retry_cohort_sealed
FROM dlightrag_ingest_jobs
WHERE job_id = $1
  AND source_type = 'retry_failed'
  AND status = 'running'
  AND lease_owner = $2
  AND lease_expires_at > clock_timestamp()
"""

_RECORD_RETRY_OUTCOME = f"""
WITH owner AS (
    SELECT 1 FROM dlightrag_ingest_jobs
    WHERE job_id = $1
      AND source_type = 'retry_failed'
      AND status = 'running'
      AND lease_owner = $5
      AND lease_expires_at > clock_timestamp()
    FOR UPDATE
), updated AS (
    UPDATE {_RETRY_ITEMS_TABLE}
    SET outcome = $3,
        summary_json = $4::jsonb,
        updated_at = NOW()
    WHERE job_id = $1
      AND doc_id = $2
      AND outcome = 'pending'
      AND EXISTS (SELECT 1 FROM owner)
    RETURNING 1
)
SELECT COUNT(*)::int FROM updated
"""  # noqa: S608 - interpolates only the private table constant

_LOCK_ACTIVE_JOB_FOR_CANCEL = """
SELECT source_type
FROM dlightrag_ingest_jobs
WHERE job_id = $1
  AND workspace = $2
  AND status IN ('queued', 'running')
FOR UPDATE
"""

_CANCEL_ACTIVE_ORDINARY = """
WITH updated AS (
    UPDATE dlightrag_ingest_jobs
    SET status = 'failed',
        result_json = result_json || jsonb_build_object('cancelled', TRUE),
        errors = (
            SELECT COALESCE(jsonb_agg(value ORDER BY ordinal), '[]'::jsonb)
            FROM (
                SELECT value, ordinal
                FROM jsonb_array_elements(errors || $3::jsonb)
                    WITH ORDINALITY AS entry(value, ordinal)
                ORDER BY ordinal LIMIT $4
            ) AS retained
        ),
        errors_truncated = errors_truncated
            OR jsonb_array_length(errors) + jsonb_array_length($3::jsonb) > $4,
        lease_owner = NULL,
        lease_expires_at = NULL,
        updated_at = clock_timestamp(),
        finished_at = clock_timestamp()
    WHERE job_id = $1
      AND workspace = $2
      AND source_type <> 'retry_failed'
      AND status IN ('queued', 'running')
    RETURNING 1
)
SELECT COUNT(*)::int FROM updated
"""

_CANCEL_FAILED_RETRY = f"""
WITH aggregate AS (
    SELECT COUNT(*)::int AS retried,
           COUNT(*) FILTER (WHERE outcome = 'succeeded')::int AS succeeded,
           COUNT(*) FILTER (WHERE outcome = 'failed')::int AS failed,
           COUNT(*) FILTER (WHERE outcome = 'pending')::int AS pending
    FROM {_RETRY_ITEMS_TABLE}
    WHERE job_id = $1
), succeeded_details AS (
    SELECT COALESCE(jsonb_agg(summary_json ORDER BY doc_id), '[]'::jsonb) AS value
    FROM (SELECT doc_id, summary_json FROM {_RETRY_ITEMS_TABLE}
          WHERE job_id = $1 AND outcome = 'succeeded'
          ORDER BY doc_id LIMIT 100) AS bounded
), failed_details AS (
    SELECT COALESCE(jsonb_agg(value ORDER BY doc_id), '[]'::jsonb) AS value
    FROM (
        SELECT doc_id, summary_json AS value FROM {_RETRY_ITEMS_TABLE}
        WHERE job_id = $1 AND outcome = 'failed'
        UNION ALL
        SELECT doc_id, jsonb_build_object(
            'doc_id', doc_id, 'reason', 'retry cancelled before completion'
        ) AS value FROM {_RETRY_ITEMS_TABLE}
        WHERE job_id = $1 AND outcome = 'pending'
        ORDER BY doc_id LIMIT 100
    ) AS bounded
), updated AS (
    UPDATE dlightrag_ingest_jobs AS job
    SET status = 'failed',
        total_items = aggregate.retried,
        processed_items = aggregate.succeeded,
        failed_items = aggregate.failed + aggregate.pending,
        result_json = jsonb_build_object(
            'retried', aggregate.retried,
            'succeeded', aggregate.succeeded,
            'failed', aggregate.failed + aggregate.pending,
            'succeeded_docs', succeeded_details.value,
            'failed_docs', failed_details.value,
            'details_truncated', aggregate.succeeded > 100
                OR aggregate.failed + aggregate.pending > 100,
            'cancelled', TRUE
        ),
        errors = (
            SELECT COALESCE(jsonb_agg(value ORDER BY ordinal), '[]'::jsonb)
            FROM (
                SELECT value, ordinal
                FROM jsonb_array_elements(errors || $2::jsonb)
                    WITH ORDINALITY AS entry(value, ordinal)
                ORDER BY ordinal LIMIT $4
            ) AS retained
        ),
        errors_truncated = errors_truncated
            OR jsonb_array_length(errors) + jsonb_array_length($2::jsonb) > $4,
        lease_owner = NULL,
        lease_expires_at = NULL,
        updated_at = clock_timestamp(),
        finished_at = clock_timestamp()
    FROM aggregate, succeeded_details, failed_details
    WHERE job.job_id = $1
      AND job.source_type = 'retry_failed'
      AND (
          ($3::text IS NULL AND job.status IN ('queued', 'running'))
          OR (job.status = 'running' AND job.lease_owner = $3
              AND job.lease_expires_at > clock_timestamp())
      )
    RETURNING 1
)
SELECT COUNT(*)::int FROM updated
"""  # noqa: S608 - interpolates only the private table constant

_FINISH_FAILED_RETRY = f"""
WITH aggregate AS (
    SELECT COUNT(*)::int AS retried,
           COUNT(*) FILTER (WHERE outcome = 'succeeded')::int AS succeeded,
           COUNT(*) FILTER (WHERE outcome = 'failed')::int AS failed,
           COUNT(*) FILTER (WHERE outcome = 'pending')::int AS pending
    FROM {_RETRY_ITEMS_TABLE}
    WHERE job_id = $1
), succeeded_details AS (
    SELECT COALESCE(jsonb_agg(summary_json ORDER BY doc_id), '[]'::jsonb) AS value
    FROM (SELECT doc_id, summary_json FROM {_RETRY_ITEMS_TABLE}
          WHERE job_id = $1 AND outcome = 'succeeded'
          ORDER BY doc_id LIMIT 100) AS bounded
), failed_details AS (
    SELECT COALESCE(jsonb_agg(summary_json ORDER BY doc_id), '[]'::jsonb) AS value
    FROM (SELECT doc_id, summary_json FROM {_RETRY_ITEMS_TABLE}
          WHERE job_id = $1 AND outcome = 'failed'
          ORDER BY doc_id LIMIT 100) AS bounded
), updated AS (
    UPDATE dlightrag_ingest_jobs AS job
    SET status = CASE WHEN aggregate.failed > 0 THEN 'partial' ELSE 'succeeded' END,
        total_items = aggregate.retried,
        processed_items = aggregate.succeeded,
        failed_items = aggregate.failed,
        current_window = 0,
        result_json = jsonb_build_object(
            'retried', aggregate.retried,
            'succeeded', aggregate.succeeded,
            'failed', aggregate.failed,
            'succeeded_docs', succeeded_details.value,
            'failed_docs', failed_details.value,
            'details_truncated', aggregate.succeeded > 100 OR aggregate.failed > 100
        ),
        lease_owner = NULL,
        lease_expires_at = NULL,
        updated_at = NOW(),
        finished_at = NOW()
    FROM aggregate, succeeded_details, failed_details
    WHERE job.job_id = $1
      AND job.source_type = 'retry_failed'
      AND job.retry_cohort_sealed = TRUE
      AND aggregate.pending = 0
      AND job.lease_owner = $2
      AND job.status = 'running'
      AND job.lease_expires_at > clock_timestamp()
    RETURNING 1
)
SELECT COUNT(*)::int FROM updated
"""  # noqa: S608 - interpolates only the private table constant

_FAIL = """
WITH updated AS (
    UPDATE dlightrag_ingest_jobs
    SET status = 'failed',
        errors = (
            SELECT COALESCE(jsonb_agg(value ORDER BY ordinal), '[]'::jsonb)
            FROM (
                SELECT value, ordinal
                FROM jsonb_array_elements(errors || $2::jsonb)
                    WITH ORDINALITY AS entry(value, ordinal)
                ORDER BY ordinal
                LIMIT $4
            ) AS retained
        ),
        errors_truncated = errors_truncated
            OR jsonb_array_length(errors) + jsonb_array_length($2::jsonb) > $4,
        lease_owner = NULL,
        lease_expires_at = NULL,
        updated_at = NOW(),
        finished_at = NOW()
    WHERE job_id = $1
      AND lease_owner = $3
      AND status = 'running'
      AND lease_expires_at > clock_timestamp()
    RETURNING 1
)
SELECT COUNT(*)::int FROM updated
"""

_GET = """
SELECT
job_id, workspace, source_type, status, request_json, total_items,
processed_items, failed_items, current_window, result_json, errors,
errors_truncated,
created_at, updated_at, started_at, finished_at, lease_owner, lease_expires_at
FROM dlightrag_ingest_jobs
WHERE job_id = $1
"""

_GET_ACTIVE_FAILED_RETRY = """
SELECT
job_id, workspace, source_type, status, request_json, total_items,
processed_items, failed_items, current_window, result_json, errors,
errors_truncated,
created_at, updated_at, started_at, finished_at, lease_owner, lease_expires_at
FROM dlightrag_ingest_jobs
WHERE workspace = $1
  AND source_type = 'retry_failed'
  AND status IN ('queued', 'running')
ORDER BY created_at DESC, job_id DESC
LIMIT 1
"""

_LIST_RECOVERABLE = """
SELECT
job_id, workspace, source_type, status, request_json, total_items,
processed_items, failed_items, current_window, result_json, errors,
errors_truncated,
created_at, updated_at, started_at, finished_at, lease_owner, lease_expires_at
FROM dlightrag_ingest_jobs
WHERE (
    status = 'queued'
    OR (status = 'running' AND (lease_expires_at IS NULL OR lease_expires_at < clock_timestamp()))
)
  AND (
      source_type = 'retry_failed'
      OR COALESCE(lease_expires_at, updated_at) >= NOW() - ($1 * INTERVAL '1 second')
  )
ORDER BY updated_at ASC
LIMIT $2
"""

_MARK_ABANDONED = """
WITH updated AS (
    UPDATE dlightrag_ingest_jobs
    SET status = 'failed',
        errors = (
            SELECT COALESCE(jsonb_agg(value ORDER BY ordinal), '[]'::jsonb)
            FROM (
                SELECT value, ordinal
                FROM jsonb_array_elements(errors || $2::jsonb)
                    WITH ORDINALITY AS entry(value, ordinal)
                ORDER BY ordinal
                LIMIT $4
            ) AS retained
        ),
        errors_truncated = errors_truncated
            OR jsonb_array_length(errors) + jsonb_array_length($2::jsonb) > $4,
        lease_owner = NULL,
        lease_expires_at = NULL,
        updated_at = NOW(),
        finished_at = NOW()
    WHERE job_id IN (
        SELECT job_id
        FROM dlightrag_ingest_jobs
        WHERE status IN ('queued', 'running')
          AND source_type <> 'retry_failed'
          AND COALESCE(lease_expires_at, updated_at) < NOW() - ($1 * INTERVAL '1 second')
        ORDER BY updated_at ASC
        LIMIT $3
    )
    RETURNING 1
)
SELECT COUNT(*)::int FROM updated
"""

_PRUNE_COMPLETED = """
WITH deleted AS (
    DELETE FROM dlightrag_ingest_jobs
    WHERE job_id IN (
        SELECT job_id
        FROM dlightrag_ingest_jobs
        WHERE status IN ('succeeded', 'partial', 'failed')
          AND COALESCE(finished_at, updated_at) < NOW() - ($1 * INTERVAL '1 second')
        ORDER BY COALESCE(finished_at, updated_at) ASC
        LIMIT $2
    )
    RETURNING 1
)
SELECT COUNT(*)::int FROM deleted
"""

_DELETE_WORKSPACE = """
WITH deleted AS (
    DELETE FROM dlightrag_ingest_jobs
    WHERE workspace = $1
    RETURNING 1
)
SELECT COUNT(*)::int FROM deleted
"""

# Orphaned counter rows are removed before the constraint lands so existing
# ledgers whose jobs were already pruned stay migration-compatible; afterwards
# the FK cascade keeps the ledger bounded for the life of the job.
_CREATE_COUNTERS_FK = """
DO $dlightrag$
BEGIN
    DELETE FROM dlightrag_ingest_counters c
    WHERE NOT EXISTS (SELECT 1 FROM dlightrag_ingest_jobs j WHERE j.job_id = c.job_id);
    IF NOT EXISTS (SELECT 1 FROM pg_catalog.pg_constraint
                   WHERE conrelid = 'dlightrag_ingest_counters'::regclass
                     AND conname = 'dlightrag_ingest_counters_job_fk') THEN
        ALTER TABLE dlightrag_ingest_counters
        ADD CONSTRAINT dlightrag_ingest_counters_job_fk
        FOREIGN KEY (job_id) REFERENCES dlightrag_ingest_jobs(job_id) ON DELETE CASCADE;
    END IF;
END
$dlightrag$
"""

_SCHEMA_MIGRATIONS = (
    Migration(
        "ingest_jobs",
        "Create ingest job state table",
        (_CREATE, *_CREATE_INDEXES),
    ),
    Migration(
        "ingest_counters",
        "Create the per-job/window counter idempotency ledger",
        (_CREATE_COUNTERS,),
    ),
    Migration(
        "ingest_counters_fk",
        "Cascade-delete counter ledger rows with their ingest jobs",
        (_CREATE_COUNTERS_FK,),
    ),
    Migration(
        "one_active_failed_retry",
        "Allow at most one active failed-document retry per workspace",
        (_CREATE_ACTIVE_RETRY_UNIQUE,),
    ),
    Migration(
        "failed_retry_items",
        "Seal failed retry cohorts and persist idempotent document outcomes",
        (_ADD_RETRY_COHORT_SEALED, _CREATE_RETRY_ITEMS),
    ),
    Migration(
        "ingest_job_error_bounds",
        "Backfill bounded-error tracking for databases created before the column",
        (_ADD_ERRORS_TRUNCATED,),
    ),
    Migration(
        "failed_retry_legacy_active",
        "Terminally park active retries created before the durable item ledger",
        (_FAIL_ACTIVE_LEGACY_RETRIES,),
    ),
)


class _RetryCohortLeaseExpired(RuntimeError):
    """Rollback sentinel for a cohort whose lease expires while it is inserted."""


class PGIngestJobStore(PostgresOperationRunner):
    """Durable ingest job state backed by PostgreSQL.

    ``promotion_doc_threshold``/``promotion_chunk_threshold`` enable the
    automatic promotion trigger; ``None`` (the default) keeps the counters
    monotonic but never enqueues promotion work.
    """

    def __init__(
        self,
        *,
        pool: Any = None,
        promotion_doc_threshold: int | None = None,
        promotion_chunk_threshold: int | None = None,
    ) -> None:
        super().__init__(pool=pool)
        self._promotion_doc_threshold = promotion_doc_threshold
        self._promotion_chunk_threshold = promotion_chunk_threshold

    async def initialize(self) -> None:
        async def _operation(conn: Any) -> None:
            await apply_migrations(
                conn,
                scope="ingest_jobs",
                migrations=_SCHEMA_MIGRATIONS,
                schema_error=IngestJobSchemaError,
            )

        await self._run(_operation)

    async def create(
        self,
        *,
        job_id: str,
        workspace: str,
        source_type: str,
        request: dict[str, Any],
    ) -> None:
        workspace_id = str(workspace).strip()
        if not workspace_id:
            raise ValueError("workspace cannot be empty")

        async def _operation(conn: Any) -> None:
            await conn.execute(_INSERT, job_id, workspace_id, source_type, json.dumps(request))

        await self._run(_operation)

    async def start_or_join_failed_retry(
        self,
        *,
        job_id: str,
        workspace: str,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        """Insert one retry or return the row that won the unique conflict."""
        workspace_id = str(workspace).strip()
        if not workspace_id:
            raise ValueError("workspace cannot be empty")

        async def _operation(conn: Any) -> Any:
            async with conn.transaction():
                row = await conn.fetchrow(
                    _START_FAILED_RETRY,
                    job_id,
                    workspace_id,
                    json.dumps(request),
                )
                if row is not None:
                    return row
                # This second statement gets a fresh READ COMMITTED snapshot
                # after a conflicting insert commits. Return that winner even
                # if its runner became terminal before this lookup.
                return await conn.fetchrow(_GET_LATEST_FAILED_RETRY, workspace_id)

        row = await self._run_once(_operation)
        if row is None:
            raise RuntimeError("failed retry start arbitration returned no job")
        return _serialize_row(row)

    async def claim_running(self, job_id: str, *, lease_owner: str, lease_seconds: int) -> bool:
        """Claim one job only while its workspace has no active write fence.

        The registry row is locked FOR SHARE in the same transaction as the
        claim, so a concurrent promotion fence acquisition serializes with the
        claim: either the claim sees the committed fence and refuses, or the
        fence sees the committed running claim and the promotion worker waits
        for the job to finish.
        """
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")

        async def _operation(conn: Any) -> bool:
            async with conn.transaction():
                # All progress/claim paths lock the job row before the workspace
                # registry row, preventing a lease-boundary lock inversion.
                job_row = await conn.fetchrow(
                    "SELECT workspace FROM dlightrag_ingest_jobs WHERE job_id = $1 FOR UPDATE",
                    job_id,
                )
                if job_row is None:
                    return False
                fence_row = await conn.fetchrow(
                    "SELECT write_fence_until > NOW() AS fenced "
                    "FROM dlightrag_workspace_meta WHERE workspace = $1 FOR SHARE",
                    job_row["workspace"],
                )
                if fence_row is not None and bool(fence_row["fenced"]):
                    return False
                return (
                    await conn.fetchrow(_CLAIM_RUNNING, job_id, lease_owner, lease_seconds)
                    is not None
                )

        return await self._run_once(_operation)

    async def release_running(self, job_id: str, *, lease_owner: str) -> bool:
        """Durably return one running job to queued so a later claim can retry."""
        owner_id = str(lease_owner).strip()
        if not owner_id:
            raise ValueError("lease_owner cannot be empty")

        async def _operation(conn: Any) -> str:
            return await conn.execute(_RELEASE_RUNNING, job_id, owner_id)

        return (await self._run_once(_operation)) != "UPDATE 0"

    async def touch_queued(self, job_id: str) -> bool:
        """Refresh one still-queued job's liveness; false if it is gone or claimed."""

        async def _operation(conn: Any) -> Any:
            return await conn.fetchval(_TOUCH_QUEUED, job_id)

        return (await self._run(_operation)) is not None

    async def cancel(
        self,
        job_id: str,
        *,
        workspace: str,
        error: str,
    ) -> bool:
        """Durably revoke and terminalize an active job from any API process."""
        workspace_id = str(workspace).strip()
        error_text = str(error).strip()
        if not workspace_id:
            raise ValueError("workspace cannot be empty")
        if not error_text:
            raise ValueError("cancel error cannot be empty")

        async def _operation(conn: Any) -> bool:
            async with conn.transaction():
                row = await conn.fetchrow(_LOCK_ACTIVE_JOB_FOR_CANCEL, job_id, workspace_id)
                if row is None:
                    return False
                if str(row["source_type"]) == "retry_failed":
                    updated = await conn.fetchval(
                        _CANCEL_FAILED_RETRY,
                        job_id,
                        json.dumps([error_text]),
                        None,
                        _MAX_JOB_ERRORS,
                    )
                else:
                    updated = await conn.fetchval(
                        _CANCEL_ACTIVE_ORDINARY,
                        job_id,
                        workspace_id,
                        json.dumps([error_text]),
                        _MAX_JOB_ERRORS,
                    )
                return int(updated or 0) > 0

        return await self._run_once(_operation)

    async def cancel_failed_retry(
        self,
        job_id: str,
        *,
        error: str,
        lease_owner: str | None,
    ) -> bool:
        """Cancel a retry while retaining exact settled ledger outcomes."""
        error_text = str(error).strip()
        if not error_text:
            raise ValueError("cancel error cannot be empty")

        async def _operation(conn: Any) -> int:
            updated = await conn.fetchval(
                _CANCEL_FAILED_RETRY,
                job_id,
                json.dumps([error_text]),
                lease_owner,
                _MAX_JOB_ERRORS,
            )
            return int(updated or 0)

        return (await self._run_once(_operation)) > 0

    async def cancel_queued(self, job_id: str, *, error: str) -> bool:
        """Terminally fail one still-queued job (explicit user cancellation)."""
        error_text = str(error).strip()
        if not error_text:
            raise ValueError("cancel error cannot be empty")

        async def _operation(conn: Any) -> int:
            updated = await conn.fetchval(
                _CANCEL_QUEUED, job_id, json.dumps([error_text]), _MAX_JOB_ERRORS
            )
            return int(updated or 0)

        return (await self._run_once(_operation)) > 0

    async def fail_invalid_recoverable(self, job_id: str, *, error: str) -> bool:
        """Fail malformed durable rows without entering corpus-write fencing."""
        safe_error = str(error)[:1024]

        async def _operation(conn: Any) -> int:
            value = await conn.fetchval(
                _FAIL_INVALID_RECOVERABLE,
                job_id,
                json.dumps([safe_error]),
                _MAX_JOB_ERRORS,
            )
            return int(value or 0)

        return await self._run_once(_operation) > 0

    async def is_workspace_fenced(self, workspace: str) -> bool:
        """Report whether a promotion write fence currently blocks writes."""
        workspace_id = str(workspace).strip()
        if not workspace_id:
            raise ValueError("workspace cannot be empty")

        async def _operation(conn: Any) -> bool:
            return (await _active_fence_seconds(conn, workspace_id)) > 0

        return await self._run(_operation)

    @asynccontextmanager
    async def workspace_write_gate(self, workspace: str) -> AsyncIterator[None]:
        """Gate one ingest run behind the promotion fence and drain protocol."""
        workspace_id = str(workspace).strip()
        if not workspace_id:
            raise ValueError("workspace cannot be empty")
        async with workspace_write_gate(workspace_id):
            yield None

    async def heartbeat(self, job_id: str, *, lease_owner: str, lease_seconds: int) -> bool:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")

        async def _operation(conn: Any) -> bool:
            async with conn.transaction():
                locked = await conn.fetchval(_HEARTBEAT_LOCK, job_id, lease_owner)
                if locked is None:
                    return False
                updated = await conn.fetchval(_HEARTBEAT, job_id, lease_owner, lease_seconds)
                return int(updated or 0) > 0

        return await self._run_once(_operation)

    async def record_window(
        self,
        job_id: str,
        *,
        total_delta: int,
        processed_delta: int,
        failed_delta: int,
        chunk_delta: int,
        current_window: int,
        errors: list[str],
        lease_owner: str,
        lease_seconds: int,
    ) -> bool:
        """Commit one ingest window: ledger-first, progress second.

        One transaction locks the job row and validates the current running
        lease (status + owner + unexpired lease), then inserts the per-window
        ledger event. Only a freshly inserted event applies job progress,
        errors, the registry counters, the threshold evaluation, and the
        promotion enqueue. A replayed window (same job/window key) conflicts
        on the ledger and merely heartbeats the lease — job totals and
        counters can never double-count on replay.
        """
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")

        async def _operation(conn: Any) -> bool:
            async with conn.transaction():
                guard = await conn.fetchrow(_RECORD_WINDOW_GUARD, job_id, lease_owner)
                if guard is None or str(guard["status"]) != "running":
                    return False
                workspace = str(guard["workspace"])
                inserted = await conn.fetchval(
                    _INSERT_COUNTER_EVENT,
                    job_id,
                    current_window,
                    workspace,
                    max(0, int(processed_delta)),
                    max(0, int(chunk_delta)),
                )
                if inserted is not None:
                    updated = await conn.fetchval(
                        _RECORD_WINDOW,
                        job_id,
                        total_delta,
                        processed_delta,
                        failed_delta,
                        current_window,
                        json.dumps(errors),
                        lease_owner,
                        lease_seconds,
                        _MAX_JOB_ERRORS,
                    )
                    if not int(updated or 0):
                        return False
                    await conn.execute(
                        _ADD_COUNTS_AND_TRIGGER,
                        workspace,
                        max(0, int(processed_delta)),
                        max(0, int(chunk_delta)),
                        self._promotion_doc_threshold,
                        self._promotion_chunk_threshold,
                    )
                    return True
                # Duplicate window: renew the lease only, never re-apply
                # totals, errors, or counters.
                renewed = await conn.fetchval(_HEARTBEAT, job_id, lease_owner, lease_seconds)
                return int(renewed or 0) > 0

        return await self._run_once(_operation)

    async def finish(self, job_id: str, *, result: dict[str, Any], lease_owner: str) -> bool:
        async def _operation(conn: Any) -> int:
            updated = await conn.fetchval(_FINISH, job_id, json.dumps(result), lease_owner)
            return int(updated or 0)

        return await self._run(_operation) > 0

    async def seal_failed_retry_cohort(
        self,
        job_id: str,
        *,
        doc_ids: Sequence[str],
        lease_owner: str,
    ) -> bool:
        stable_ids = tuple(dict.fromkeys(str(doc_id) for doc_id in doc_ids if doc_id))

        async def _operation(conn: Any) -> bool:
            async with conn.transaction():
                guard = await conn.fetchrow(_SEAL_RETRY_COHORT_GUARD, job_id, lease_owner)
                if guard is None:
                    return False
                if bool(guard["retry_cohort_sealed"]):
                    return True
                for offset in range(0, len(stable_ids), _BATCH_LIMIT):
                    await conn.execute(
                        _INSERT_RETRY_COHORT,
                        job_id,
                        list(stable_ids[offset : offset + _BATCH_LIMIT]),
                    )
                updated = await conn.execute(
                    _MARK_RETRY_COHORT_SEALED,
                    job_id,
                    lease_owner,
                    JOB_LEASE_SECONDS,
                )
                if updated == "UPDATE 0":
                    # Raising inside the transaction rolls back every inserted
                    # item before a waiting recovery claimant can acquire the row.
                    raise _RetryCohortLeaseExpired
                return True

        try:
            return await self._run_once(_operation)
        except _RetryCohortLeaseExpired:
            return False

    async def list_unfinished_failed_retry_items(
        self,
        job_id: str,
        *,
        lease_owner: str,
    ) -> tuple[str, ...] | None:
        async def _operation(conn: Any) -> tuple[str, ...] | None:
            state = await conn.fetchrow(_RETRY_COHORT_STATE, job_id, lease_owner)
            if state is None:
                return None
            if not bool(state["retry_cohort_sealed"]):
                return None
            rows = await conn.fetch(_LIST_UNFINISHED_RETRY_ITEMS, job_id, lease_owner)
            return tuple(str(row["doc_id"]) for row in rows)

        return await self._run(_operation)

    async def record_failed_retry_outcome(
        self,
        job_id: str,
        *,
        doc_id: str,
        outcome: str,
        summary: dict[str, Any],
        lease_owner: str,
    ) -> bool:
        if outcome not in {"succeeded", "failed"}:
            raise ValueError("failed retry outcome is invalid")
        compact = _compact_retry_summary(doc_id, outcome, summary)

        async def _operation(conn: Any) -> int:
            updated = await conn.fetchval(
                _RECORD_RETRY_OUTCOME,
                job_id,
                doc_id,
                outcome,
                json.dumps(compact),
                lease_owner,
            )
            return int(updated or 0)

        return await self._run_once(_operation) > 0

    async def finish_failed_retry(
        self,
        job_id: str,
        *,
        lease_owner: str,
    ) -> bool:
        """Finish from the durable item ledger, the sole totals authority."""

        async def _operation(conn: Any) -> int:
            updated = await conn.fetchval(_FINISH_FAILED_RETRY, job_id, lease_owner)
            return int(updated or 0)

        return await self._run_once(_operation) > 0

    async def fail(self, job_id: str, *, error: str, lease_owner: str) -> bool:
        async def _operation(conn: Any) -> int:
            updated = await conn.fetchval(
                _FAIL,
                job_id,
                json.dumps([error]),
                lease_owner,
                _MAX_JOB_ERRORS,
            )
            return int(updated or 0)

        return await self._run(_operation) > 0

    async def get(self, job_id: str) -> dict[str, Any] | None:
        async def _operation(conn: Any) -> Any:
            return await conn.fetchrow(_GET, job_id)

        row = await self._run(_operation)
        return _serialize_row(row) if row is not None else None

    async def get_active_failed_retry(self, workspace: str) -> dict[str, Any] | None:
        workspace_id = str(workspace).strip()
        if not workspace_id:
            raise ValueError("workspace cannot be empty")

        async def _operation(conn: Any) -> Any:
            return await conn.fetchrow(_GET_ACTIVE_FAILED_RETRY, workspace_id)

        row = await self._run(_operation)
        return _serialize_row(row) if row is not None else None

    async def list_recoverable(self) -> list[dict[str, Any]]:
        """Return queued/running jobs whose owner may still come back for them."""

        async def _operation(conn: Any) -> list[Any]:
            return await conn.fetch(_LIST_RECOVERABLE, JOB_ORPHAN_AFTER_SECONDS, _BATCH_LIMIT)

        rows = await self._run(_operation)
        return [_serialize_row(row) for row in rows]

    async def prune(self) -> dict[str, int]:
        """Fail jobs whose owner is gone for good and delete old finished rows."""

        async def _operation(conn: Any) -> dict[str, int]:
            failed = await conn.fetchval(
                _MARK_ABANDONED,
                JOB_ORPHAN_AFTER_SECONDS,
                json.dumps([JOB_ABANDONED_ERROR]),
                _BATCH_LIMIT,
                _MAX_JOB_ERRORS,
            )
            deleted = await conn.fetchval(_PRUNE_COMPLETED, JOB_RETENTION_SECONDS, _BATCH_LIMIT)
            return {
                "failed_abandoned": int(failed or 0),
                "deleted_completed": int(deleted or 0),
            }

        return await self._run(_operation)

    async def delete_for_workspace(self, workspace: str) -> int:
        """Delete all ingest job rows for a workspace."""
        workspace_id = str(workspace).strip()
        if not workspace_id:
            raise ValueError("workspace cannot be empty")

        async def _operation(conn: Any) -> int:
            deleted = await conn.fetchval(_DELETE_WORKSPACE, workspace_id)
            return int(deleted or 0)

        return await self._run(_operation)


def _compact_retry_summary(
    doc_id: str,
    outcome: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    compact: dict[str, Any] = {"doc_id": str(doc_id)[:255]}
    file_path = summary.get("file_path")
    if isinstance(file_path, str) and file_path:
        file_label = file_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        file_label = file_label.split("?", 1)[0].split("#", 1)[0]
        if file_label:
            compact["file_path"] = file_label[:255]
    if outcome == "failed":
        reason = summary.get("reason")
        compact["reason"] = str(reason or "retry ingestion failed")[:256]
    else:
        compact["replacement_count"] = max(0, min(1, int(summary.get("replacement_count") or 0)))
    return compact


def _serialize_row(row: Any) -> dict[str, Any]:
    data = dict(row)
    data["request"] = _json_value(data.pop("request_json", {}), default={})
    data["result"] = _json_value(data.pop("result_json", {}), default={})
    data["errors"] = _json_value(data.get("errors"), default=[])
    data["errors_truncated"] = bool(data.get("errors_truncated", False))
    return data


def _json_value(value: Any, *, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        return json.loads(value)
    return value


__all__ = [
    "PGIngestJobStore",
]
