# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for PostgreSQL-backed ingest job state."""

import json
from typing import Any

from dlightrag.storage.ingest_jobs import PGIngestJobStore


class _Acquire:
    def __init__(self, conn: _Conn) -> None:
        self._conn = conn

    async def __aenter__(self) -> _Conn:
        return self._conn

    async def __aexit__(self, *args: object) -> None:
        return None


class _Tx:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *args: object) -> None:
        return None


class _Pool:
    def __init__(self, conn: _Conn) -> None:
        self._conn = conn

    def acquire(self) -> _Acquire:
        return _Acquire(self._conn)


class _Conn:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[Any, ...]]] = []
        self.fetches: list[tuple[str, tuple[Any, ...]]] = []
        self.fetchrows: list[tuple[str, tuple[Any, ...]]] = []
        self.fetch_results: list[list[dict[str, Any]]] = []
        self.fetchvals: list[tuple[str, tuple[Any, ...]]] = []
        self.fetchval_results: list[int] = []
        self.applied: set[tuple[str, str]] = set()
        self.row: dict[str, Any] | None = None

    async def execute(self, query: str, *args: Any) -> None:
        self.executed.append((query, args))
        normalized = query.strip()
        if normalized.startswith("INSERT INTO dlightrag_schema_migrations"):
            self.applied.add((str(args[0]), str(args[1])))
        if normalized.startswith("INSERT INTO dlightrag_ingest_jobs"):
            self.row = {
                "job_id": args[0],
                "workspace": args[1],
                "source_type": args[2],
                "status": "queued",
                "request_json": args[3],
                "total_items": 0,
                "processed_items": 0,
                "failed_items": 0,
                "current_window": 0,
                "result_json": "{}",
                "errors": "[]",
                "created_at": None,
                "updated_at": None,
                "started_at": None,
                "finished_at": None,
                "lease_owner": None,
                "lease_expires_at": None,
            }
        elif "SET status = 'running'" in query and self.row is not None:
            self.row["status"] = "running"
        elif "total_items = total_items + $2" in query and self.row is not None:
            self.row["total_items"] += args[1]
            self.row["processed_items"] += args[2]
            self.row["failed_items"] += args[3]
            self.row["current_window"] = args[4]
            self.row["errors"] = json.dumps(json.loads(self.row["errors"]) + json.loads(args[5]))
        elif "SET status = 'succeeded'" in query and self.row is not None:
            self.row["status"] = "succeeded"
            self.row["result_json"] = args[1]

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        self.fetchrows.append((query, args))
        if "SET status = 'running'" in query and self.row is not None:
            self.row["status"] = "running"
            self.row["lease_owner"] = args[1]
            self.row["lease_expires_at"] = "future"
        return self.row

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.fetches.append((query, args))
        return self.fetch_results.pop(0) if self.fetch_results else []

    async def fetchval(self, query: str, *args: Any) -> int:
        self.fetchvals.append((query, args))
        if "total_items = total_items + $2" in query and self.row is not None:
            self.row["total_items"] += args[1]
            self.row["processed_items"] += args[2]
            self.row["failed_items"] += args[3]
            self.row["current_window"] = args[4]
            self.row["errors"] = json.dumps(json.loads(self.row["errors"]) + json.loads(args[5]))
            return 1
        if "SET status = 'succeeded'" in query and self.row is not None:
            self.row["status"] = "succeeded"
            self.row["result_json"] = args[1]
            self.row["lease_owner"] = None
            self.row["lease_expires_at"] = None
            return 1
        if "SET status = 'failed'" in query and self.row is not None:
            self.row["status"] = "failed"
            self.row["errors"] = json.dumps(json.loads(self.row["errors"]) + json.loads(args[1]))
            self.row["lease_owner"] = None
            self.row["lease_expires_at"] = None
            return 1
        if "SET lease_expires_at" in query and self.row is not None:
            self.row["lease_expires_at"] = "future"
            return 1
        return self.fetchval_results.pop(0) if self.fetchval_results else 0

    def transaction(self) -> _Tx:
        return _Tx()


async def test_ingest_job_store_records_window_progress_and_result() -> None:
    conn = _Conn()
    store = PGIngestJobStore(pool=_Pool(conn))

    await store.initialize()
    await store.create(
        job_id="job-1",
        workspace="default",
        source_type="s3",
        request={"bucket": "b", "prefix": "docs/"},
    )
    await store.claim_running("job-1", lease_owner="owner-1", lease_seconds=300)
    await store.record_window(
        "job-1",
        total_delta=64,
        processed_delta=63,
        failed_delta=1,
        current_window=1,
        errors=["s3://b/docs/bad.pdf: failed"],
        lease_owner="owner-1",
        lease_seconds=300,
    )
    await store.finish("job-1", result={"processed": 63}, lease_owner="owner-1")

    row = await store.get("job-1")

    assert row is not None
    assert row["status"] == "succeeded"
    assert row["workspace"] == "default"
    assert row["total_items"] == 64
    assert row["processed_items"] == 63
    assert row["failed_items"] == 1
    assert row["current_window"] == 1
    assert row["request"] == {"bucket": "b", "prefix": "docs/"}
    assert row["result"] == {"processed": 63}
    assert row["errors"] == ["s3://b/docs/bad.pdf: failed"]


async def test_ingest_job_store_claims_job_with_database_lease() -> None:
    conn = _Conn()
    conn.row = {
        "job_id": "job-1",
        "workspace": "default",
        "source_type": "s3",
        "status": "running",
        "request_json": "{}",
        "total_items": 0,
        "processed_items": 0,
        "failed_items": 0,
        "current_window": 0,
        "result_json": "{}",
        "errors": "[]",
        "created_at": None,
        "updated_at": None,
        "started_at": None,
        "finished_at": None,
        "lease_owner": None,
        "lease_expires_at": None,
    }
    store = PGIngestJobStore(pool=_Pool(conn))

    claimed = await store.claim_running("job-1", lease_owner="owner-1", lease_seconds=300)

    assert claimed is True
    query, args = conn.fetchrows[0]
    assert "UPDATE dlightrag_ingest_jobs" in query
    assert "lease_owner = $2" in query
    assert "lease_expires_at = NOW() + ($3 * INTERVAL '1 second')" in query
    assert "RETURNING" in query
    assert args == ("job-1", "owner-1", 300)


async def test_ingest_job_store_renews_owned_lease() -> None:
    conn = _Conn()
    conn.row = {
        "job_id": "job-1",
        "workspace": "default",
        "source_type": "s3",
        "status": "running",
        "request_json": "{}",
        "total_items": 0,
        "processed_items": 0,
        "failed_items": 0,
        "current_window": 0,
        "result_json": "{}",
        "errors": "[]",
        "created_at": None,
        "updated_at": None,
        "started_at": None,
        "finished_at": None,
        "lease_owner": "owner-1",
        "lease_expires_at": "soon",
    }
    store = PGIngestJobStore(pool=_Pool(conn))

    renewed = await store.heartbeat("job-1", lease_owner="owner-1", lease_seconds=300)

    assert renewed is True
    query, args = conn.fetchvals[0]
    assert "SET lease_expires_at = NOW() + ($3 * INTERVAL '1 second')" in query
    assert "lease_owner = $2" in query
    assert args == ("job-1", "owner-1", 300)


async def test_ingest_job_store_prunes_stale_jobs() -> None:
    conn = _Conn()
    conn.fetchval_results = [2, 3]
    store = PGIngestJobStore(pool=_Pool(conn))

    result = await store.prune(
        completed_ttl_seconds=14 * 24 * 3600,
        stale_running_seconds=24 * 3600,
        batch_size=500,
    )

    assert result == {"failed_abandoned": 2, "deleted_completed": 3}
    assert len(conn.fetchvals) == 2
    mark_query, mark_args = conn.fetchvals[0]
    assert "status IN ('queued', 'running')" in mark_query
    assert mark_args[0] == 24 * 3600
    assert json.loads(mark_args[1]) == ["ingest job abandoned after process exit"]
    assert mark_args[2] == 500
    delete_query, delete_args = conn.fetchvals[1]
    assert "status IN ('succeeded', 'failed')" in delete_query
    assert delete_args == (14 * 24 * 3600, 500)


async def test_ingest_job_store_lists_recoverable_jobs() -> None:
    conn = _Conn()
    conn.fetch_results = [
        [
            {
                "job_id": "job-1",
                "workspace": "project_a",
                "source_type": "s3",
                "status": "running",
                "request_json": json.dumps(
                    {
                        "workspace": "project_a",
                        "source_type": "s3",
                        "kwargs": {"bucket": "b", "prefix": "docs/"},
                    }
                ),
                "total_items": 128,
                "processed_items": 128,
                "failed_items": 0,
                "current_window": 2,
                "result_json": "{}",
                "errors": "[]",
                "created_at": None,
                "updated_at": None,
                "started_at": None,
                "finished_at": None,
            }
        ]
    ]
    store = PGIngestJobStore(pool=_Pool(conn))

    rows = await store.list_recoverable(stale_running_seconds=3600, limit=25)

    assert rows[0]["job_id"] == "job-1"
    assert rows[0]["workspace"] == "project_a"
    assert rows[0]["current_window"] == 2
    assert rows[0]["request"]["kwargs"] == {"bucket": "b", "prefix": "docs/"}
    query, args = conn.fetches[0]
    assert "status = 'queued'" in query
    assert "lease_expires_at < NOW()" in query
    assert args == (3600, 25)


async def test_ingest_job_store_deletes_workspace_jobs() -> None:
    conn = _Conn()
    conn.fetchval_results = [4]
    store = PGIngestJobStore(pool=_Pool(conn))

    deleted = await store.delete_for_workspace("Project A")

    assert deleted == 4
    query, args = conn.fetchvals[0]
    assert "DELETE FROM dlightrag_ingest_jobs" in query
    assert args == ("project_a",)
