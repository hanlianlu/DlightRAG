# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for IngestTaskManager — background task lifecycle and SSE broadcasting."""

from __future__ import annotations

import asyncio

from dlightrag.core.ingest_tasks import IngestTaskManager, TaskInfo

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _fake_ingest_ok(**kwargs) -> dict:
    """Fake ingest function that succeeds."""
    await asyncio.sleep(0.01)
    return {"status": "ok", "files_ingested": 1}


async def _fake_ingest_fail(**kwargs) -> dict:
    """Fake ingest function that raises."""
    raise RuntimeError("ingestion boom")


# ---------------------------------------------------------------------------
# TestTaskInfo
# ---------------------------------------------------------------------------


class TestTaskInfo:
    def test_default_values(self) -> None:
        task = TaskInfo(task_id="t1", workspace="default", file_name="test.pdf")
        assert task.status == "queued"
        assert task.progress == 0.0
        assert task.error == ""
        assert task.result is None
        assert task.created_at > 0


# ---------------------------------------------------------------------------
# TestSubmit
# ---------------------------------------------------------------------------


class TestSubmit:
    async def test_submit_returns_task_id(self) -> None:
        mgr = IngestTaskManager(ingest_fn=_fake_ingest_ok)
        task_id = await mgr.submit(workspace="default", file_path="/tmp/test.pdf")
        assert isinstance(task_id, str)
        assert len(task_id) > 0

    async def test_submit_creates_task_info(self) -> None:
        mgr = IngestTaskManager(ingest_fn=_fake_ingest_ok)
        task_id = await mgr.submit(workspace="ws1", file_path="/tmp/doc.pdf")
        task = mgr.get_task(task_id)
        assert task is not None
        assert task.workspace == "ws1"
        assert task.file_name == "doc.pdf"


# ---------------------------------------------------------------------------
# TestTaskLifecycle
# ---------------------------------------------------------------------------


class TestTaskLifecycle:
    async def test_successful_task(self) -> None:
        mgr = IngestTaskManager(ingest_fn=_fake_ingest_ok)
        task_id = await mgr.submit(workspace="default", file_path="/tmp/test.pdf")

        # Wait for background task to complete
        await asyncio.sleep(0.1)

        task = mgr.get_task(task_id)
        assert task is not None
        assert task.status == "done"
        assert task.progress == 1.0
        assert task.finished_at > 0
        assert task.elapsed > 0
        assert task.error == ""
        assert task.result == {"status": "ok", "files_ingested": 1}

    async def test_failed_task(self) -> None:
        mgr = IngestTaskManager(ingest_fn=_fake_ingest_fail)
        task_id = await mgr.submit(workspace="default", file_path="/tmp/test.pdf")

        await asyncio.sleep(0.1)

        task = mgr.get_task(task_id)
        assert task is not None
        assert task.status == "failed"
        assert "ingestion boom" in task.error
        assert task.finished_at > 0

    async def test_get_active_tasks(self) -> None:
        mgr = IngestTaskManager(ingest_fn=_fake_ingest_ok)
        await mgr.submit(workspace="default", file_path="/tmp/a.pdf")
        await mgr.submit(workspace="default", file_path="/tmp/b.pdf")

        # Tasks may already be running or done
        # Just verify we can call the method
        active = mgr.get_active_tasks()
        assert isinstance(active, list)

    async def test_get_recent_tasks(self) -> None:
        mgr = IngestTaskManager(ingest_fn=_fake_ingest_ok)
        await mgr.submit(workspace="default", file_path="/tmp/test.pdf")
        await asyncio.sleep(0.1)

        recent = mgr.get_recent_tasks(max_age_s=60.0)
        assert len(recent) >= 1

    async def test_get_task_unknown(self) -> None:
        mgr = IngestTaskManager(ingest_fn=_fake_ingest_ok)
        assert mgr.get_task("nonexistent") is None


# ---------------------------------------------------------------------------
# TestSubscriber
# ---------------------------------------------------------------------------


class TestSubscriber:
    async def test_subscribe_yields_snapshot(self) -> None:
        mgr = IngestTaskManager(ingest_fn=_fake_ingest_ok)

        events: list[dict] = []
        sub = mgr.subscribe()
        # Get the first event (snapshot)
        event = await sub.__anext__()
        events.append(event)
        await sub.aclose()

        assert len(events) == 1
        assert events[0]["type"] == "snapshot"
        assert isinstance(events[0]["tasks"], list)

    async def test_subscribe_receives_task_created(self) -> None:
        mgr = IngestTaskManager(ingest_fn=_fake_ingest_ok)

        events: list[dict] = []
        sub = mgr.subscribe()

        # Get snapshot
        await sub.__anext__()

        # Submit a task (will push task_created to subscriber)
        await mgr.submit(workspace="default", file_path="/tmp/test.pdf")

        # Get the task_created event
        event = await asyncio.wait_for(sub.__anext__(), timeout=1.0)
        events.append(event)
        await sub.aclose()

        assert len(events) >= 1
        assert events[0]["type"] == "task_created"
        assert events[0]["task"]["file_name"] == "test.pdf"

    async def test_subscriber_cleanup_on_close(self) -> None:
        mgr = IngestTaskManager(ingest_fn=_fake_ingest_ok)
        assert len(mgr._subscribers) == 0

        sub = mgr.subscribe()
        await sub.__anext__()  # snapshot
        assert len(mgr._subscribers) == 1

        await sub.aclose()
        # Subscriber should be cleaned up
        assert len(mgr._subscribers) == 0


# ---------------------------------------------------------------------------
# TestTmpDirCleanup
# ---------------------------------------------------------------------------


class TestTmpDirCleanup:
    async def test_tmp_dir_cleaned_on_success(self, tmp_path) -> None:
        mgr = IngestTaskManager(ingest_fn=_fake_ingest_ok)
        tmp_dir = tmp_path / "upload"
        tmp_dir.mkdir()
        (tmp_dir / "test.pdf").write_bytes(b"fake")

        await mgr.submit(
            workspace="default",
            file_path=str(tmp_dir / "test.pdf"),
            tmp_dir=str(tmp_dir),
        )
        await asyncio.sleep(0.1)

        assert not tmp_dir.exists()

    async def test_tmp_dir_cleaned_on_failure(self, tmp_path) -> None:
        mgr = IngestTaskManager(ingest_fn=_fake_ingest_fail)
        tmp_dir = tmp_path / "upload"
        tmp_dir.mkdir()
        (tmp_dir / "test.pdf").write_bytes(b"fake")

        await mgr.submit(
            workspace="default",
            file_path=str(tmp_dir / "test.pdf"),
            tmp_dir=str(tmp_dir),
        )
        await asyncio.sleep(0.1)

        assert not tmp_dir.exists()


# ---------------------------------------------------------------------------
# TestTaskDict
# ---------------------------------------------------------------------------


class TestTaskDict:
    def test_serialization(self) -> None:
        task = TaskInfo(
            task_id="abc",
            workspace="ws1",
            file_name="test.pdf",
            status="running",
            step="parsing",
            progress=0.5,
        )
        d = IngestTaskManager._task_dict(task)
        assert d["task_id"] == "abc"
        assert d["workspace"] == "ws1"
        assert d["status"] == "running"
        assert d["progress"] == 0.5
        assert d["step"] == "parsing"
