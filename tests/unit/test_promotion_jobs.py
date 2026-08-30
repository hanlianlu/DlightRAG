# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Unit tests for the durable promotion-job schema and adapter interfaces."""

from typing import Any

import pytest

from dlightrag.adapters.postgres.corpus import promotion_jobs
from dlightrag.adapters.postgres.corpus.promotion_jobs import PGPromotionJobStore


class _Pool:
    def __init__(self, conn: _Conn) -> None:
        self._conn = conn

    def acquire(self) -> Any:
        return _Acquire(self._conn)


class _Acquire:
    def __init__(self, conn: _Conn) -> None:
        self._conn = conn

    async def __aenter__(self) -> _Conn:
        return self._conn

    async def __aexit__(self, *args: object) -> None:
        return None


class _Tx:
    async def __aenter__(self) -> _Tx:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None


class _Conn:
    def __init__(self, *, row: dict[str, Any] | None = None) -> None:
        self.executed: list[tuple[str, tuple[Any, ...]]] = []
        self._row = row

    def transaction(self) -> _Tx:
        return _Tx()

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        self.executed.append((query, args))
        return self._row

    async def fetchval(self, query: str, *args: Any) -> Any:
        self.executed.append((query, args))
        return None

    async def execute(self, query: str, *args: Any) -> str:
        self.executed.append((query, args))
        return "UPDATE 1"


def _store(conn: _Conn) -> PGPromotionJobStore:
    store = PGPromotionJobStore()
    store._operation_pool = _Pool(conn)  # type: ignore[attr-defined]
    return store


def _schema_sql() -> str:
    return "\n".join(
        stmt for migration in promotion_jobs._SCHEMA_MIGRATIONS for stmt in migration.statements
    )


def test_job_schema_declares_legal_recoverable_state_constraints() -> None:
    sql = _schema_sql()

    assert "state IN ('pending', 'promoting', 'done', 'failed')" in sql
    assert "lease_generation  BIGINT NOT NULL DEFAULT 0" in sql
    assert "attempt_count >= 0 AND lease_generation >= 0" in sql
    assert "state = 'promoting' AND lease_owner IS NOT NULL AND lease_until IS NOT NULL" in sql
    assert "state <> 'promoting' AND lease_owner IS NULL AND lease_until IS NULL" in sql
    assert "(state = 'failed') = (last_error IS NOT NULL)" in sql
    assert "(state = 'failed') = (next_retry_at IS NOT NULL)" in sql
    assert "(state = 'done') = (promoted_at IS NOT NULL)" in sql


def test_job_schema_declares_bounded_pending_retry_and_expired_lease_indexes() -> None:
    sql = _schema_sql()

    assert "WHERE state IN ('pending', 'promoting', 'failed')" in sql
    assert "ON dlightrag_promotion_jobs (created_at, job_id)\nWHERE state = 'pending'" in sql
    assert "ON dlightrag_promotion_jobs (next_retry_at, job_id)\nWHERE state = 'failed'" in sql
    assert "ON dlightrag_promotion_jobs (lease_until, job_id)\nWHERE state = 'promoting'" in sql


async def test_enqueue_is_idempotent_against_pending_promoting_and_retrying_jobs() -> None:
    conn = _Conn()
    store = _store(conn)

    await store.enqueue(" ws-a ")

    sql, args = conn.executed[-1]
    assert (
        "ON CONFLICT (workspace) WHERE state IN ('pending', 'promoting', 'failed') DO NOTHING"
        in sql
    )
    assert args == ("ws-a",)


async def test_claim_next_recovers_due_retry_and_expired_lease_with_new_generation() -> None:
    conn = _Conn(row={"job_id": 7, "workspace": "ws-a", "attempt_count": 3, "lease_generation": 2})
    store = _store(conn)

    claimed = await store.claim_next(owner="worker-1", lease_until="2026-04-01T00:00:00Z")

    assert claimed == {
        "job_id": 7,
        "workspace": "ws-a",
        "attempt_count": 3,
        "lease_generation": 2,
    }
    sql, args = conn.executed[-1]
    assert "state = 'failed' AND next_retry_at <= NOW()" in sql
    assert "state = 'promoting' AND lease_until <= NOW()" in sql
    assert "FOR UPDATE SKIP LOCKED" in sql
    assert "lease_generation = lease_generation + 1" in sql
    assert "$2::timestamptz > NOW()" in sql
    assert "job.lease_generation" in sql
    assert args == ("worker-1", "2026-04-01T00:00:00Z")


async def test_claim_next_returns_none_without_claimable_jobs() -> None:
    conn = _Conn(row=None)
    store = _store(conn)

    assert await store.claim_next(owner="worker-1", lease_until="2026-04-01T00:00:00Z") is None


async def test_renew_fail_and_done_require_current_generation_and_unexpired_lease() -> None:
    conn = _Conn()
    store = _store(conn)

    await store.renew_lease(
        job_id=7,
        owner="worker-1",
        lease_generation=2,
        lease_until="2026-04-01T01:00:00Z",
    )
    sql, args = conn.executed[-1]
    assert "lease_generation = $3" in sql
    assert "lease_until > NOW()" in sql
    assert "$4::timestamptz > NOW()" in sql
    assert args == (7, "worker-1", 2, "2026-04-01T01:00:00Z")

    await store.mark_failed(
        job_id=7,
        owner="worker-1",
        lease_generation=2,
        error="cutover invariant mismatch",
        next_retry_at="2026-04-02T00:00:00Z",
    )
    sql, args = conn.executed[-1]
    assert "SET state = 'failed'" in sql
    assert "lease_generation = $3" in sql
    assert "lease_until > NOW()" in sql
    assert args == (
        7,
        "worker-1",
        2,
        "cutover invariant mismatch",
        "2026-04-02T00:00:00Z",
    )

    await store.mark_done(job_id=7, owner="worker-1", lease_generation=2)
    sql, args = conn.executed[-1]
    assert "SET state = 'done'" in sql
    assert "promoted_at = NOW()" in sql
    assert "lease_generation = $3" in sql
    assert "lease_until > NOW()" in sql
    assert args == (7, "worker-1", 2)


async def test_transition_identity_and_retry_inputs_are_validated() -> None:
    store = _store(_Conn())

    with pytest.raises(ValueError, match="lease owner"):
        await store.claim_next(owner=" ", lease_until="2026-04-01T00:00:00Z")
    with pytest.raises(ValueError, match="job_id"):
        await store.mark_done(job_id=0, owner="worker", lease_generation=1)
    with pytest.raises(ValueError, match="lease_generation"):
        await store.mark_done(job_id=1, owner="worker", lease_generation=0)
    with pytest.raises(ValueError, match="next_retry_at"):
        await store.mark_failed(
            job_id=1,
            owner="worker",
            lease_generation=1,
            error="failed",
            next_retry_at=None,
        )
