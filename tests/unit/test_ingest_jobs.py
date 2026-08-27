# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for PostgreSQL-backed ingest job state."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

from dlightrag.adapters.postgres.corpus.ingest_jobs import PGIngestJobStore
from dlightrag.engine.rag.corpus.ingest_jobs import (
    JOB_LEASE_SECONDS,
    JOB_ORPHAN_AFTER_SECONDS,
    JOB_RETENTION_SECONDS,
)
from dlightrag.engine.rag.corpus.ingestion.jobs import IngestJobCoordinator


def _finished_status(row: dict[str, Any]) -> str:
    """Mirror _FINISH's CASE so the fake cannot drift from the real statement."""
    return "partial" if int(row.get("failed_items") or 0) > 0 else "succeeded"


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
                "errors_truncated": False,
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
        elif "THEN 'partial' ELSE 'succeeded'" in query and self.row is not None:
            self.row["status"] = _finished_status(self.row)
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
        if "dlightrag_schema_migrations" in query and "version" in query:
            scope = str(args[0])
            versions = sorted(
                version for applied_scope, version in self.applied if applied_scope == scope
            )
            return [{"version": version} for version in versions]
        return self.fetch_results.pop(0) if self.fetch_results else []

    async def fetchval(self, query: str, *args: Any) -> int:
        self.fetchvals.append((query, args))
        if "total_items = total_items + $2" in query and self.row is not None:
            self.row["total_items"] += args[1]
            self.row["processed_items"] += args[2]
            self.row["failed_items"] += args[3]
            self.row["current_window"] = args[4]
            retained = json.loads(self.row["errors"])
            incoming = json.loads(args[5])
            if "errors_truncated =" in query:
                self.row["errors_truncated"] = len(retained) + len(incoming) > args[8]
                retained = (retained + incoming)[: args[8]]
            else:
                retained += incoming
            self.row["errors"] = json.dumps(retained)
            return 1
        if "THEN 'partial' ELSE 'succeeded'" in query and self.row is not None:
            self.row["status"] = _finished_status(self.row)
            self.row["result_json"] = args[1]
            self.row["lease_owner"] = None
            self.row["lease_expires_at"] = None
            return 1
        if "SET status = 'failed'" in query and self.row is not None:
            self.row["status"] = "failed"
            retained = json.loads(self.row["errors"])
            incoming = json.loads(args[1])
            if "errors_truncated =" in query:
                self.row["errors_truncated"] = len(retained) + len(incoming) > args[3]
                retained = (retained + incoming)[: args[3]]
            else:
                retained += incoming
            self.row["errors"] = json.dumps(retained)
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
    # One of the 64 items failed, so the job must not call itself a success.
    assert row["status"] == "partial"
    assert row["workspace"] == "default"
    assert row["total_items"] == 64
    assert row["processed_items"] == 63
    assert row["failed_items"] == 1
    assert row["current_window"] == 1
    assert row["request"] == {"bucket": "b", "prefix": "docs/"}
    assert row["result"] == {"processed": 63}
    assert row["errors"] == ["s3://b/docs/bad.pdf: failed"]
    assert row["errors_truncated"] is False


async def test_ingest_job_store_caps_retained_errors_and_reports_truncation() -> None:
    conn = _Conn()
    store = PGIngestJobStore(pool=_Pool(conn))
    errors = [f"document-{index}.pdf: failed" for index in range(250)]

    await store.initialize()
    await store.create(
        job_id="job-many-errors",
        workspace="default",
        source_type="local",
        request={"path": "/inputs"},
    )
    await store.claim_running(
        "job-many-errors",
        lease_owner="owner-1",
        lease_seconds=300,
    )
    await store.record_window(
        "job-many-errors",
        total_delta=250,
        processed_delta=0,
        failed_delta=250,
        current_window=1,
        errors=errors,
        lease_owner="owner-1",
        lease_seconds=300,
    )

    row = await store.get("job-many-errors")

    assert row is not None
    assert row["failed_items"] == 250
    assert row["errors"] == errors[:200]
    assert row["errors_truncated"] is True


async def test_ingest_job_store_caps_terminal_failure_error() -> None:
    conn = _Conn()
    store = PGIngestJobStore(pool=_Pool(conn))
    retained_errors = [f"document-{index}.pdf: failed" for index in range(200)]

    await store.initialize()
    await store.create(
        job_id="job-terminal-error",
        workspace="default",
        source_type="local",
        request={"path": "/inputs"},
    )
    await store.claim_running(
        "job-terminal-error",
        lease_owner="owner-1",
        lease_seconds=300,
    )
    await store.record_window(
        "job-terminal-error",
        total_delta=200,
        processed_delta=0,
        failed_delta=200,
        current_window=1,
        errors=retained_errors,
        lease_owner="owner-1",
        lease_seconds=300,
    )
    await store.fail(
        "job-terminal-error",
        error="terminal worker failure",
        lease_owner="owner-1",
    )

    row = await store.get("job-terminal-error")

    assert row is not None
    assert row["errors"] == retained_errors
    assert row["errors_truncated"] is True


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

    result = await store.prune()

    assert result == {"failed_abandoned": 2, "deleted_completed": 3}
    assert len(conn.fetchvals) == 2
    mark_query, mark_args = conn.fetchvals[0]
    assert "status IN ('queued', 'running')" in mark_query
    assert "errors_truncated =" in mark_query
    # Liveness comes from the lease, so the reaper keys off it, not updated_at.
    assert "COALESCE(lease_expires_at, updated_at) <" in mark_query
    assert mark_args[0] == JOB_ORPHAN_AFTER_SECONDS
    assert json.loads(mark_args[1]) == ["ingest job abandoned after process exit"]
    assert mark_args[3] == 200
    delete_query, delete_args = conn.fetchvals[1]
    assert "status IN ('succeeded', 'partial', 'failed')" in delete_query
    assert delete_args[0] == JOB_RETENTION_SECONDS == 7 * 24 * 3600


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

    rows = await store.list_recoverable()

    assert rows[0]["job_id"] == "job-1"
    assert rows[0]["workspace"] == "project_a"
    assert rows[0]["current_window"] == 2
    assert rows[0]["request"]["kwargs"] == {"bucket": "b", "prefix": "docs/"}
    query, args = conn.fetches[0]
    assert "status = 'queued'" in query
    assert "lease_expires_at < NOW()" in query
    # Recovery and reaping split orphans on the same boundary: no gap, no overlap.
    assert "COALESCE(lease_expires_at, updated_at) >=" in query
    assert args[0] == JOB_ORPHAN_AFTER_SECONDS


async def test_ingest_job_store_deletes_workspace_jobs() -> None:
    conn = _Conn()
    conn.fetchval_results = [4]
    store = PGIngestJobStore(pool=_Pool(conn))

    deleted = await store.delete_for_workspace("project_a")

    assert deleted == 4
    query, args = conn.fetchvals[0]
    assert "DELETE FROM dlightrag_ingest_jobs" in query
    assert args == ("project_a",)


def test_the_orphan_window_leaves_room_for_a_live_lease_to_renew() -> None:
    """Reaping at the lease boundary would kill workers that are merely slow."""
    assert JOB_ORPHAN_AFTER_SECONDS > JOB_LEASE_SECONDS * 4
    assert JOB_ORPHAN_AFTER_SECONDS < JOB_RETENTION_SECONDS


async def test_a_job_that_lost_no_items_still_reports_success() -> None:
    """The CASE must not turn every finished job into a partial one."""
    conn = _Conn()
    store = PGIngestJobStore(pool=_Pool(conn))
    await store.create(job_id="job-1", workspace="default", source_type="s3", request={})
    await store.claim_running("job-1", lease_owner="owner-1", lease_seconds=300)
    await store.record_window(
        "job-1",
        total_delta=8,
        processed_delta=8,
        failed_delta=0,
        current_window=1,
        errors=[],
        lease_owner="owner-1",
        lease_seconds=300,
    )

    await store.finish("job-1", result={"processed": 8}, lease_owner="owner-1")

    row = await store.get("job-1")
    assert row is not None
    assert row["status"] == "succeeded"
    assert row["failed_items"] == 0


class _CoordinatorStore:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.recoverable_rows: list[dict[str, Any]] = []
        self.claim_results: dict[str, bool] = {}

    async def initialize(self) -> None:
        return None

    async def create(
        self,
        *,
        job_id: str,
        workspace: str,
        source_type: str,
        request: dict[str, Any],
    ) -> None:
        self.rows[job_id] = {
            "job_id": job_id,
            "workspace": workspace,
            "source_type": source_type,
            "status": "queued",
            "request": request,
            "total_items": 0,
            "processed_items": 0,
            "failed_items": 0,
            "current_window": 0,
            "result": {},
            "errors": [],
        }

    async def claim_running(
        self,
        job_id: str,
        *,
        lease_owner: str,
        lease_seconds: int,
    ) -> bool:
        if not self.claim_results.get(job_id, True):
            return False
        self.rows[job_id]["status"] = "running"
        self.rows[job_id]["lease_owner"] = lease_owner
        self.rows[job_id]["lease_seconds"] = lease_seconds
        return True

    async def heartbeat(
        self,
        job_id: str,
        *,
        lease_owner: str,
        lease_seconds: int,
    ) -> bool:
        row = self.rows.get(job_id)
        return bool(row and row.get("lease_owner") == lease_owner and lease_seconds > 0)

    async def record_window(
        self,
        job_id: str,
        *,
        total_delta: int,
        processed_delta: int,
        failed_delta: int,
        current_window: int,
        errors: list[str],
        lease_owner: str | None = None,
        lease_seconds: int | None = None,
    ) -> bool:
        row = self.rows[job_id]
        row["total_items"] += total_delta
        row["processed_items"] += processed_delta
        row["failed_items"] += failed_delta
        row["current_window"] = current_window
        row["errors"].extend(errors)
        return True

    async def finish(
        self,
        job_id: str,
        *,
        result: dict[str, Any],
        lease_owner: str | None = None,
    ) -> bool:
        self.rows[job_id]["status"] = "succeeded"
        self.rows[job_id]["result"] = result
        return True

    async def fail(
        self,
        job_id: str,
        *,
        error: str,
        lease_owner: str | None = None,
    ) -> bool:
        self.rows[job_id]["status"] = "failed"
        self.rows[job_id]["errors"].append(error)
        return True

    async def get(self, job_id: str) -> dict[str, Any] | None:
        return self.rows.get(job_id)

    async def list_recoverable(self) -> list[dict[str, Any]]:
        return list(self.recoverable_rows)

    async def prune(self) -> dict[str, int]:
        return {"failed_abandoned": 0, "deleted_completed": 0}

    async def delete_for_workspace(self, workspace: str) -> int:
        before = len(self.rows)
        self.rows = {
            job_id: row for job_id, row in self.rows.items() if row.get("workspace") != workspace
        }
        return before - len(self.rows)


def _coordinator(
    store: _CoordinatorStore,
    runtime: AsyncMock,
    *,
    input_root: Path,
) -> IngestJobCoordinator:
    return IngestJobCoordinator(
        AsyncMock(return_value=runtime),
        input_root=input_root,
        store=store,
    )


async def test_coordinator_recovers_from_the_recorded_window(tmp_path: Path) -> None:
    row = {
        "job_id": "job-1",
        "workspace": "project_a",
        "source_type": "s3",
        "status": "running",
        "request": {
            "workspace": "project_a",
            "source_type": "s3",
            "kwargs": {"bucket": "bucket", "prefix": "docs/"},
        },
        "total_items": 128,
        "processed_items": 128,
        "failed_items": 0,
        "current_window": 2,
        "errors": [],
        "result": {},
    }
    store = _CoordinatorStore()
    store.rows["job-1"] = dict(row)
    store.recoverable_rows = [row]
    runtime = AsyncMock()
    runtime.aingest.return_value = {"processed": 1, "errors": []}
    coordinator = _coordinator(store, runtime, input_root=tmp_path)

    await coordinator.start_recovery()
    result = await coordinator.await_job("job-1", timeout=1)
    await coordinator.close()

    assert result is not None
    assert result["processed_items"] == 129
    assert result["result"]["processed"] == 129
    assert runtime.aingest.await_args.kwargs["_resume_from_window"] == 2
    assert runtime.aingest.await_args.kwargs["bucket"] == "bucket"


async def test_coordinator_does_not_run_recovery_after_losing_the_claim(
    tmp_path: Path,
) -> None:
    row = {
        "job_id": "job-1",
        "workspace": "project_a",
        "source_type": "s3",
        "status": "running",
        "request": {
            "workspace": "project_a",
            "source_type": "s3",
            "kwargs": {"bucket": "bucket"},
        },
        "current_window": 0,
    }
    store = _CoordinatorStore()
    store.rows["job-1"] = dict(row)
    store.recoverable_rows = [row]
    store.claim_results["job-1"] = False
    runtime = AsyncMock()
    coordinator = _coordinator(store, runtime, input_root=tmp_path)

    await coordinator.start_recovery()
    await coordinator.await_job("job-1", timeout=1)
    await coordinator.close()

    runtime.aingest.assert_not_awaited()


async def test_coordinator_records_progress_and_cleans_completed_upload_batch(
    tmp_path: Path,
) -> None:
    store = _CoordinatorStore()
    runtime = AsyncMock()

    async def ingest(**kwargs: Any) -> dict[str, Any]:
        await kwargs["_progress_callback"](
            SimpleNamespace(
                total_delta=2,
                processed_delta=1,
                failed_delta=1,
                batch_index=0,
                errors=("bad.pdf: failed",),
            )
        )
        return {"processed": 1, "failed": 1}

    runtime.aingest.side_effect = ingest
    coordinator = _coordinator(store, runtime, input_root=tmp_path)
    staged_dir = tmp_path / "project_a" / "__uploads__" / "batch-1"
    staged_dir.mkdir(parents=True)
    (staged_dir / "report.pdf").write_text("pdf", encoding="utf-8")

    job = await coordinator.start_job(
        "project_a",
        "local",
        path=str(staged_dir),
        cleanup_paths=staged_dir,
    )
    result = await coordinator.await_job(job["job_id"], timeout=1)
    await coordinator.close()

    assert result is not None
    assert result["status"] == "succeeded"
    assert result["total_items"] == 2
    assert result["failed_items"] == 1
    assert result["request"]["cleanup_paths"] == [str(staged_dir)]
    assert not staged_dir.exists()


async def test_coordinator_close_keeps_running_upload_for_recovery(tmp_path: Path) -> None:
    store = _CoordinatorStore()
    runtime = AsyncMock()
    started = asyncio.Event()

    async def ingest(**_kwargs: Any) -> dict[str, Any]:
        started.set()
        await asyncio.Event().wait()
        return {"processed": 1}

    runtime.aingest.side_effect = ingest
    coordinator = _coordinator(store, runtime, input_root=tmp_path)
    staged_dir = tmp_path / "default" / "__uploads__" / "batch-1"
    staged_dir.mkdir(parents=True)
    (staged_dir / "report.pdf").write_text("pdf", encoding="utf-8")

    job = await coordinator.start_job(
        "default",
        "local",
        path=str(staged_dir),
        cleanup_paths=staged_dir,
    )
    await asyncio.wait_for(started.wait(), timeout=1)
    await coordinator.close()

    assert store.rows[job["job_id"]]["status"] == "running"
    assert staged_dir.exists()


async def test_coordinator_timeout_returns_running_job_without_cancelling_it(
    tmp_path: Path,
) -> None:
    store = _CoordinatorStore()
    runtime = AsyncMock()
    started = asyncio.Event()
    release = asyncio.Event()

    async def ingest(**_kwargs: Any) -> dict[str, Any]:
        started.set()
        await release.wait()
        return {"processed": 1}

    runtime.aingest.side_effect = ingest
    coordinator = _coordinator(store, runtime, input_root=tmp_path)
    job = await coordinator.start_job("default", "local", path=str(tmp_path / "report.pdf"))
    await asyncio.wait_for(started.wait(), timeout=1)

    running = await coordinator.await_job(job["job_id"], timeout=0)

    assert running is not None
    assert running["status"] == "running"
    assert coordinator.has_active_workspace_job("default")

    release.set()
    completed = await coordinator.await_job(job["job_id"], timeout=1)
    await coordinator.close()

    assert completed is not None
    assert completed["status"] == "succeeded"
