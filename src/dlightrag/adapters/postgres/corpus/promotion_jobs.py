# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Durable hot-workspace promotion jobs (schema + narrow adapter only).

Commit 1 installs the durable job schema and the adapter interfaces the
Commit 3 promotion worker will consume. Nothing here enqueues or drives work:
no triggers, no worker loop, no cutover behavior. The table survives crashes,
enforces its legal-state transitions, and exposes bounded claim scans.

Idempotency: at most one live/retrying job per workspace (partial unique
index). Leasing/fencing: only the current owner + generation may transition an
unexpired ``promoting`` lease; each claim bumps both ``attempt_count`` and the
generation. Failures carry an error and required next-retry timestamp. Partial
indexes bound pending, due-retry, and expired-lease claim scans.
"""

from typing import Any

from dlightrag.adapters.postgres.core._migrations import (
    Migration,
    TableRequirement,
    apply_migrations,
    verify_migrations,
)
from dlightrag.adapters.postgres.core._operations import PostgresOperationRunner
from dlightrag.engine.rag.workspace.ports import CorpusSchemaError

_JOBS_TABLE = "dlightrag_promotion_jobs"

_CREATE = """
CREATE TABLE IF NOT EXISTS dlightrag_promotion_jobs (
    job_id            BIGSERIAL PRIMARY KEY,
    workspace         TEXT NOT NULL,
    state             TEXT NOT NULL DEFAULT 'pending',
    attempt_count     INTEGER NOT NULL DEFAULT 0,
    lease_generation  BIGINT NOT NULL DEFAULT 0,
    lease_owner       TEXT,
    lease_until       TIMESTAMPTZ,
    last_error        TEXT,
    next_retry_at     TIMESTAMPTZ,
    promoted_at       TIMESTAMPTZ,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT dlightrag_promotion_jobs_state
        CHECK (state IN ('pending', 'promoting', 'done', 'failed')),
    CONSTRAINT dlightrag_promotion_jobs_attempts
        CHECK (attempt_count >= 0 AND lease_generation >= 0),
    CONSTRAINT dlightrag_promotion_jobs_leased
        CHECK (
            (state = 'promoting' AND lease_owner IS NOT NULL AND lease_until IS NOT NULL)
            OR
            (state <> 'promoting' AND lease_owner IS NULL AND lease_until IS NULL)
        ),
    CONSTRAINT dlightrag_promotion_jobs_failed_error
        CHECK ((state = 'failed') = (last_error IS NOT NULL)),
    CONSTRAINT dlightrag_promotion_jobs_done_time
        CHECK ((state = 'done') = (promoted_at IS NOT NULL)),
    CONSTRAINT dlightrag_promotion_jobs_retry_state
        CHECK ((state = 'failed') = (next_retry_at IS NOT NULL))
)
"""

# Retryable failures remain live: enqueue cannot bypass their backoff by
# inserting a fresh row. Done jobs are immutable history.
_ACTIVE_JOB_INDEX = """
CREATE UNIQUE INDEX IF NOT EXISTS uq_dlightrag_promotion_jobs_active
ON dlightrag_promotion_jobs (workspace)
WHERE state IN ('pending', 'promoting', 'failed')
"""

_CLAIM_INDEX = """
CREATE INDEX IF NOT EXISTS idx_dlightrag_promotion_jobs_claim
ON dlightrag_promotion_jobs (created_at, job_id)
WHERE state = 'pending'
"""

_RETRY_INDEX = """
CREATE INDEX IF NOT EXISTS idx_dlightrag_promotion_jobs_retry
ON dlightrag_promotion_jobs (next_retry_at, job_id)
WHERE state = 'failed'
"""

_LEASE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_dlightrag_promotion_jobs_lease
ON dlightrag_promotion_jobs (lease_until, job_id)
WHERE state = 'promoting'
"""

_ENQUEUE = """
INSERT INTO dlightrag_promotion_jobs (workspace)
VALUES ($1)
ON CONFLICT (workspace) WHERE state IN ('pending', 'promoting', 'failed') DO NOTHING
"""

_CLAIM_NEXT = """
WITH candidate AS (
    SELECT job_id
    FROM dlightrag_promotion_jobs
    WHERE state = 'pending'
       OR (state = 'failed' AND next_retry_at <= NOW())
       OR (state = 'promoting' AND lease_until <= NOW())
    ORDER BY COALESCE(next_retry_at, lease_until, created_at), job_id
    FOR UPDATE SKIP LOCKED
    LIMIT 1
)
UPDATE dlightrag_promotion_jobs AS job
SET state = 'promoting',
    lease_owner = $1,
    lease_until = $2::timestamptz,
    lease_generation = lease_generation + 1,
    attempt_count = attempt_count + 1,
    last_error = NULL,
    next_retry_at = NULL,
    promoted_at = NULL,
    updated_at = NOW()
FROM candidate
WHERE job.job_id = candidate.job_id AND $2::timestamptz > NOW()
RETURNING job.job_id, job.workspace, job.attempt_count, job.lease_generation
"""

_RENEW_LEASE = """
UPDATE dlightrag_promotion_jobs
SET lease_until = $4::timestamptz, updated_at = NOW()
WHERE job_id = $1
  AND state = 'promoting'
  AND lease_owner = $2
  AND lease_generation = $3
  AND lease_until > NOW()
  AND $4::timestamptz > NOW()
"""

_MARK_FAILED = """
UPDATE dlightrag_promotion_jobs
SET state = 'failed',
    last_error = $4,
    next_retry_at = $5::timestamptz,
    lease_owner = NULL,
    lease_until = NULL,
    updated_at = NOW()
WHERE job_id = $1
  AND state = 'promoting'
  AND lease_owner = $2
  AND lease_generation = $3
  AND lease_until > NOW()
"""

_MARK_DONE = """
UPDATE dlightrag_promotion_jobs
SET state = 'done',
    promoted_at = NOW(),
    last_error = NULL,
    next_retry_at = NULL,
    lease_owner = NULL,
    lease_until = NULL,
    updated_at = NOW()
WHERE job_id = $1
  AND state = 'promoting'
  AND lease_owner = $2
  AND lease_generation = $3
  AND lease_until > NOW()
"""

_SCHEMA_MIGRATIONS = (
    Migration(
        "promotion_jobs",
        "Create durable promotion-job table",
        (_CREATE,),
    ),
    Migration(
        "promotion_jobs_active_unique",
        "At most one live job per workspace",
        (_ACTIVE_JOB_INDEX,),
    ),
    Migration(
        "promotion_jobs_claim_index",
        "Bounded keyset claim index for pending jobs",
        (_CLAIM_INDEX,),
    ),
    Migration(
        "promotion_jobs_retry_index",
        "Bounded keyset index for failed retry jobs",
        (_RETRY_INDEX,),
    ),
    Migration(
        "promotion_jobs_lease_index",
        "Bounded keyset index for expired promotion leases",
        (_LEASE_INDEX,),
    ),
)

_SCHEMA_TABLES = (
    TableRequirement(
        name=_JOBS_TABLE,
        columns=(
            "job_id",
            "workspace",
            "state",
            "attempt_count",
            "lease_generation",
            "lease_owner",
            "lease_until",
            "last_error",
            "next_retry_at",
            "promoted_at",
            "created_at",
            "updated_at",
        ),
        primary_key=("job_id",),
        unique_indexes=("uq_dlightrag_promotion_jobs_active",),
        indexes=(
            "idx_dlightrag_promotion_jobs_claim",
            "idx_dlightrag_promotion_jobs_retry",
            "idx_dlightrag_promotion_jobs_lease",
        ),
        checks=(
            "dlightrag_promotion_jobs_state",
            "dlightrag_promotion_jobs_attempts",
            "dlightrag_promotion_jobs_leased",
            "dlightrag_promotion_jobs_failed_error",
            "dlightrag_promotion_jobs_done_time",
            "dlightrag_promotion_jobs_retry_state",
        ),
    ),
)


class PGPromotionJobStore(PostgresOperationRunner):
    """Durable promotion-job store for the Commit 3 promotion worker.

    Commit 1 installs the schema and the claim/transition interfaces; the
    worker that drives them does not exist yet.
    """

    async def initialize(self, *, validate_only: bool = False) -> None:
        """Create/verify the promotion-job schema."""

        async def _operation(conn: Any) -> None:
            if validate_only:
                await verify_migrations(
                    conn,
                    scope="promotion_jobs",
                    migrations=_SCHEMA_MIGRATIONS,
                    tables=_SCHEMA_TABLES,
                    schema_error=CorpusSchemaError,
                )
                return
            await apply_migrations(
                conn,
                scope="promotion_jobs",
                migrations=_SCHEMA_MIGRATIONS,
                schema_error=CorpusSchemaError,
            )

        await self._run(_operation)

    async def enqueue(self, workspace: str) -> bool:
        """Idempotently enqueue one workspace; false if a live/retrying job exists."""
        workspace_id = _nonempty(workspace, field="workspace")

        async def _operation(conn: Any) -> str:
            return await conn.execute(_ENQUEUE, workspace_id)

        return (await self._run(_operation)) != "INSERT 0 0"

    async def claim_next(
        self,
        *,
        owner: str,
        lease_until: Any,
    ) -> dict[str, Any] | None:
        """Lease one pending, due-retry, or expired-lease job.

        Every successful claim increments and returns ``lease_generation``.
        Commit 3 must carry that generation through every side effect and
        transition, so an expired stale worker cannot complete a newer claim.
        """
        owner_id = _nonempty(owner, field="lease owner")

        async def _operation(conn: Any) -> Any:
            return await conn.fetchrow(_CLAIM_NEXT, owner_id, lease_until)

        row = await self._run(_operation)
        return dict(row) if row is not None else None

    async def renew_lease(
        self,
        *,
        job_id: int,
        owner: str,
        lease_generation: int,
        lease_until: Any,
    ) -> bool:
        """Extend one still-current, unexpired fenced lease."""
        identity = _lease_identity(job_id, owner, lease_generation)

        async def _operation(conn: Any) -> str:
            return await conn.execute(_RENEW_LEASE, *identity, lease_until)

        return (await self._run(_operation)) != "UPDATE 0"

    async def mark_failed(
        self,
        *,
        job_id: int,
        owner: str,
        lease_generation: int,
        error: str,
        next_retry_at: Any,
    ) -> bool:
        """Schedule retry for one still-current, unexpired fenced lease."""
        identity = _lease_identity(job_id, owner, lease_generation)
        error_text = _nonempty(error, field="promotion error")
        if next_retry_at is None:
            raise ValueError("next_retry_at is required for a failed promotion job")

        async def _operation(conn: Any) -> str:
            return await conn.execute(
                _MARK_FAILED,
                *identity,
                error_text,
                next_retry_at,
            )

        return (await self._run(_operation)) != "UPDATE 0"

    async def mark_done(
        self,
        *,
        job_id: int,
        owner: str,
        lease_generation: int,
    ) -> bool:
        """Complete one still-current, unexpired fenced lease."""
        identity = _lease_identity(job_id, owner, lease_generation)

        async def _operation(conn: Any) -> str:
            return await conn.execute(_MARK_DONE, *identity)

        return (await self._run(_operation)) != "UPDATE 0"


def _nonempty(value: Any, *, field: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field} cannot be empty")
    return text


def _lease_identity(job_id: int, owner: str, lease_generation: int) -> tuple[int, str, int]:
    if isinstance(job_id, bool) or not isinstance(job_id, int) or job_id < 1:
        raise ValueError("job_id must be a positive integer")
    if (
        isinstance(lease_generation, bool)
        or not isinstance(lease_generation, int)
        or lease_generation < 1
    ):
        raise ValueError("lease_generation must be a positive integer")
    return job_id, _nonempty(owner, field="lease owner"), lease_generation


__all__ = ["PGPromotionJobStore"]
