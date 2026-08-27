# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL-backed ingest job state."""

import json
from typing import Any

from dlightrag.adapters.postgres._migrations import Migration, apply_migrations
from dlightrag.adapters.postgres._operations import PostgresOperationRunner
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

_SCHEMA_MIGRATIONS = (
    Migration(
        "0001_ingest_jobs",
        "Create ingest job state table",
        (_CREATE, *_CREATE_INDEXES),
    ),
)


class PGIngestJobStore(PostgresOperationRunner):
    """Durable ingest job state backed by PostgreSQL."""

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
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")

        async def _operation(conn: Any) -> Any:
            return await conn.fetchrow(_CLAIM_RUNNING, job_id, lease_owner, lease_seconds)

        return await self._run(_operation) is not None

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
        current_window: int,
        errors: list[str],
        lease_owner: str,
        lease_seconds: int,
    ) -> bool:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")

        async def _operation(conn: Any) -> int:
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
            return int(updated or 0)

        return await self._run(_operation) > 0

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
