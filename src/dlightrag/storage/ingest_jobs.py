# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PostgreSQL-backed ingest job state."""

from __future__ import annotations

import json
from typing import Any

from dlightrag.storage.migrations import Migration, apply_migrations
from dlightrag.utils import normalize_workspace

TABLE = "dlightrag_ingest_jobs"
DEFAULT_COMPLETED_JOB_TTL_SECONDS = 14 * 24 * 3600
DEFAULT_STALE_RUNNING_SECONDS = 24 * 3600
DEFAULT_PRUNE_BATCH_SIZE = 1000
ABANDONED_ERROR = "ingest job abandoned after process exit"

_CREATE = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    job_id          TEXT PRIMARY KEY,
    workspace       TEXT NOT NULL,
    source_type     TEXT NOT NULL,
    status          TEXT NOT NULL,
    request_json    JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    total_items     INTEGER NOT NULL DEFAULT 0,
    processed_items INTEGER NOT NULL DEFAULT 0,
    failed_items    INTEGER NOT NULL DEFAULT 0,
    current_window  INTEGER NOT NULL DEFAULT 0,
    result_json     JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    errors          JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at      TIMESTAMPTZ,
    finished_at     TIMESTAMPTZ
)
"""

_CREATE_INDEXES = (
    f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_workspace_created "
    f"ON {TABLE} (workspace, created_at DESC)",
    f"CREATE INDEX IF NOT EXISTS idx_{TABLE}_status_updated ON {TABLE} (status, updated_at DESC)",
)

_INSERT = f"""
INSERT INTO {TABLE} (job_id, workspace, source_type, status, request_json)
VALUES ($1, $2, $3, 'queued', $4::jsonb)
"""

_MARK_RUNNING = f"""
UPDATE {TABLE}
SET status = 'running',
    started_at = COALESCE(started_at, NOW()),
    updated_at = NOW()
WHERE job_id = $1
"""

_RECORD_WINDOW = f"""
UPDATE {TABLE}
SET total_items = total_items + $2,
    processed_items = processed_items + $3,
    failed_items = failed_items + $4,
    current_window = $5,
    errors = errors || $6::jsonb,
    updated_at = NOW()
WHERE job_id = $1
"""

_FINISH = f"""
UPDATE {TABLE}
SET status = 'succeeded',
    result_json = $2::jsonb,
    updated_at = NOW(),
    finished_at = NOW()
WHERE job_id = $1
"""

_FAIL = f"""
UPDATE {TABLE}
SET status = 'failed',
    errors = errors || $2::jsonb,
    updated_at = NOW(),
    finished_at = NOW()
WHERE job_id = $1
"""

_GET = f"""
SELECT job_id, workspace, source_type, status, request_json, total_items,
       processed_items, failed_items, current_window, result_json, errors,
       created_at, updated_at, started_at, finished_at
FROM {TABLE}
WHERE job_id = $1
"""

_LIST_RECOVERABLE = f"""
SELECT job_id, workspace, source_type, status, request_json, total_items,
       processed_items, failed_items, current_window, result_json, errors,
       created_at, updated_at, started_at, finished_at
FROM {TABLE}
WHERE status IN ('queued', 'running')
  AND updated_at >= NOW() - ($1 * INTERVAL '1 second')
ORDER BY updated_at ASC
LIMIT $2
"""

_MARK_ABANDONED = f"""
WITH updated AS (
    UPDATE {TABLE}
    SET status = 'failed',
        errors = errors || $2::jsonb,
        updated_at = NOW(),
        finished_at = NOW()
    WHERE job_id IN (
        SELECT job_id
        FROM {TABLE}
        WHERE status IN ('queued', 'running')
          AND updated_at < NOW() - ($1 * INTERVAL '1 second')
        ORDER BY updated_at ASC
        LIMIT $3
    )
    RETURNING 1
)
SELECT COUNT(*)::int FROM updated
"""

_PRUNE_COMPLETED = f"""
WITH deleted AS (
    DELETE FROM {TABLE}
    WHERE job_id IN (
        SELECT job_id
        FROM {TABLE}
        WHERE status IN ('succeeded', 'failed')
          AND COALESCE(finished_at, updated_at) < NOW() - ($1 * INTERVAL '1 second')
        ORDER BY COALESCE(finished_at, updated_at) ASC
        LIMIT $2
    )
    RETURNING 1
)
SELECT COUNT(*)::int FROM deleted
"""

_DELETE_WORKSPACE = f"""
WITH deleted AS (
    DELETE FROM {TABLE}
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


class PGIngestJobStore:
    """Durable ingest job state backed by PostgreSQL."""

    def __init__(self, *, pool: Any = None) -> None:
        self._pool = pool

    async def _run(self, operation):
        if self._pool is not None:
            async with self._pool.acquire() as conn:
                return await operation(conn)

        from dlightrag.storage.pool import pg_pool

        return await pg_pool.run(operation)

    async def initialize(self) -> None:
        async def _operation(conn: Any) -> None:
            await apply_migrations(
                conn,
                scope="ingest_jobs",
                migrations=_SCHEMA_MIGRATIONS,
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
        workspace_id = normalize_workspace(workspace)
        if not workspace_id:
            raise ValueError("workspace cannot be empty")

        async def _operation(conn: Any) -> None:
            await conn.execute(_INSERT, job_id, workspace_id, source_type, json.dumps(request))

        await self._run(_operation)

    async def mark_running(self, job_id: str) -> None:
        async def _operation(conn: Any) -> None:
            await conn.execute(_MARK_RUNNING, job_id)

        await self._run(_operation)

    async def record_window(
        self,
        job_id: str,
        *,
        total_delta: int,
        processed_delta: int,
        failed_delta: int,
        current_window: int,
        errors: list[str],
    ) -> None:
        async def _operation(conn: Any) -> None:
            await conn.execute(
                _RECORD_WINDOW,
                job_id,
                total_delta,
                processed_delta,
                failed_delta,
                current_window,
                json.dumps(errors),
            )

        await self._run(_operation)

    async def finish(self, job_id: str, *, result: dict[str, Any]) -> None:
        async def _operation(conn: Any) -> None:
            await conn.execute(_FINISH, job_id, json.dumps(result))

        await self._run(_operation)

    async def fail(self, job_id: str, *, error: str) -> None:
        async def _operation(conn: Any) -> None:
            await conn.execute(_FAIL, job_id, json.dumps([error]))

        await self._run(_operation)

    async def get(self, job_id: str) -> dict[str, Any] | None:
        async def _operation(conn: Any) -> Any:
            return await conn.fetchrow(_GET, job_id)

        row = await self._run(_operation)
        return _serialize_row(row) if row is not None else None

    async def list_recoverable(
        self,
        *,
        stale_running_seconds: int = DEFAULT_STALE_RUNNING_SECONDS,
        limit: int = DEFAULT_PRUNE_BATCH_SIZE,
    ) -> list[dict[str, Any]]:
        """Return queued/running jobs that are recent enough to recover."""
        if limit <= 0:
            raise ValueError("limit must be positive")

        async def _operation(conn: Any) -> list[Any]:
            return await conn.fetch(_LIST_RECOVERABLE, stale_running_seconds, limit)

        rows = await self._run(_operation)
        return [_serialize_row(row) for row in rows]

    async def prune(
        self,
        *,
        completed_ttl_seconds: int = DEFAULT_COMPLETED_JOB_TTL_SECONDS,
        stale_running_seconds: int = DEFAULT_STALE_RUNNING_SECONDS,
        batch_size: int = DEFAULT_PRUNE_BATCH_SIZE,
    ) -> dict[str, int]:
        """Mark abandoned in-flight jobs failed and delete old completed rows."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        async def _operation(conn: Any) -> dict[str, int]:
            failed = await conn.fetchval(
                _MARK_ABANDONED,
                stale_running_seconds,
                json.dumps([ABANDONED_ERROR]),
                batch_size,
            )
            deleted = await conn.fetchval(_PRUNE_COMPLETED, completed_ttl_seconds, batch_size)
            return {
                "failed_abandoned": int(failed or 0),
                "deleted_completed": int(deleted or 0),
            }

        return await self._run(_operation)

    async def delete_for_workspace(self, workspace: str) -> int:
        """Delete all ingest job rows for a workspace."""
        workspace_id = normalize_workspace(workspace)
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
    return data


def _json_value(value: Any, *, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, str):
        return json.loads(value)
    return value


__all__ = [
    "ABANDONED_ERROR",
    "DEFAULT_COMPLETED_JOB_TTL_SECONDS",
    "DEFAULT_PRUNE_BATCH_SIZE",
    "DEFAULT_STALE_RUNNING_SECONDS",
    "PGIngestJobStore",
]
