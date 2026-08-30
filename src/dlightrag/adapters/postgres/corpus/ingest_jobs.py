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
from collections.abc import AsyncIterator
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
  AND lease_expires_at > NOW()
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

_INSERT = """
INSERT INTO dlightrag_ingest_jobs (job_id, workspace, source_type, status, request_json)
VALUES ($1, $2, $3, 'queued', $4::jsonb)
"""

_CLAIM_RUNNING = """
UPDATE dlightrag_ingest_jobs
SET status = 'running',
    lease_owner = $2,
    lease_expires_at = NOW() + ($3 * INTERVAL '1 second'),
    started_at = COALESCE(started_at, NOW()),
    updated_at = NOW()
WHERE job_id = $1
  AND status IN ('queued', 'running')
  AND (lease_owner = $2 OR lease_expires_at IS NULL OR lease_expires_at < NOW())
RETURNING
job_id, workspace, source_type, status, request_json, total_items,
processed_items, failed_items, current_window, result_json, errors,
errors_truncated,
created_at, updated_at, started_at, finished_at, lease_owner, lease_expires_at
"""

_HEARTBEAT = """
WITH updated AS (
    UPDATE dlightrag_ingest_jobs
    SET lease_expires_at = NOW() + ($3 * INTERVAL '1 second'),
        updated_at = NOW()
    WHERE job_id = $1
      AND lease_owner = $2
      AND status = 'running'
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
        lease_expires_at = NOW() + ($8 * INTERVAL '1 second'),
        updated_at = NOW()
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
    RETURNING 1
)
SELECT COUNT(*)::int FROM updated
"""

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

_LIST_RECOVERABLE = """
SELECT
job_id, workspace, source_type, status, request_json, total_items,
processed_items, failed_items, current_window, result_json, errors,
errors_truncated,
created_at, updated_at, started_at, finished_at, lease_owner, lease_expires_at
FROM dlightrag_ingest_jobs
WHERE (
    status = 'queued'
    OR (status = 'running' AND (lease_expires_at IS NULL OR lease_expires_at < NOW()))
)
  AND COALESCE(lease_expires_at, updated_at) >= NOW() - ($1 * INTERVAL '1 second')
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
        updated_at = NOW(),
        finished_at = NOW()
    WHERE job_id IN (
        SELECT job_id
        FROM dlightrag_ingest_jobs
        WHERE status IN ('queued', 'running')
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
)


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
                job_row = await conn.fetchrow(
                    "SELECT workspace FROM dlightrag_ingest_jobs WHERE job_id = $1",
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

        async def _operation(conn: Any) -> int:
            updated = await conn.fetchval(_HEARTBEAT, job_id, lease_owner, lease_seconds)
            return int(updated or 0)

        return await self._run(_operation) > 0

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
