# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""IngestTaskManager -- background ingestion task tracking with SSE progress broadcasting.

Manages asyncio background tasks for file ingestion with real-time progress
notification via subscriber queues. Decoupled from ServiceManager via ingest_fn
callable injection.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import time
import uuid
from asyncio import QueueFull
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)


@dataclass
class TaskInfo:
    """Snapshot of a single ingest task's lifecycle."""

    task_id: str
    workspace: str
    file_name: str
    status: Literal["queued", "running", "done", "failed"] = "queued"
    step: str = ""
    progress: float = 0.0
    detail: str = ""
    error: str = ""
    created_at: float = field(default_factory=time.time)
    finished_at: float = 0.0
    elapsed: float = 0.0
    result: dict | None = None


class IngestTaskManager:
    """Background ingest task manager with SSE subscriber broadcasting.

    Accepts an async ingest callable so this class has zero import coupling
    to ServiceManager or any domain store.

    Usage::

        mgr = IngestTaskManager(ingest_fn=service_manager.aingest)
        task_id = await mgr.submit(workspace="default", file_path="/tmp/doc.pdf")
        async for event in mgr.subscribe():
            ...
    """

    def __init__(self, ingest_fn: Callable) -> None:
        """Initialise with an async ingest callable.

        Args:
            ingest_fn: Async callable compatible with
                ``aingest(workspace, source_type, path, replace)``.
        """
        self._ingest_fn = ingest_fn
        self._tasks: dict[str, TaskInfo] = {}
        self._subscribers: list[asyncio.Queue] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def submit(
        self,
        workspace: str,
        file_path: str,
        *,
        replace: bool = False,
        tmp_dir: str | None = None,
    ) -> str:
        """Create and enqueue a background ingest task.

        Returns the task_id immediately; the actual ingestion runs in the
        background via ``asyncio.create_task``.
        """
        import os

        task_id = str(uuid.uuid4())
        file_name = os.path.basename(file_path)
        task = TaskInfo(
            task_id=task_id,
            workspace=workspace,
            file_name=file_name,
        )
        self._tasks[task_id] = task

        self._notify({"type": "task_created", "task": self._task_dict(task)})

        asyncio.create_task(
            self._run(
                task=task,
                workspace=workspace,
                file_path=file_path,
                replace=replace,
                tmp_dir=tmp_dir,
            ),
            name=f"ingest-{task_id}",
        )

        return task_id

    def get_task(self, task_id: str) -> TaskInfo | None:
        """Return the TaskInfo for a given task_id, or None if unknown."""
        return self._tasks.get(task_id)

    def get_active_tasks(self) -> list[TaskInfo]:
        """Return all tasks currently queued or running."""
        return [t for t in self._tasks.values() if t.status in ("queued", "running")]

    def get_recent_tasks(self, max_age_s: float = 60.0) -> list[TaskInfo]:
        """Return active tasks plus completed tasks finished within max_age_s seconds."""
        now = time.time()
        result: list[TaskInfo] = []
        for task in self._tasks.values():
            if task.status in ("queued", "running"):
                result.append(task)
            elif task.finished_at > 0 and (now - task.finished_at) <= max_age_s:
                result.append(task)
        return result

    async def subscribe(self) -> AsyncGenerator[dict[str, Any], None]:
        """Async generator yielding SSE-style event dicts.

        Immediately yields a ``snapshot`` event with all recent tasks, then
        yields task update events as they occur. Cleans up on generator close.
        """
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=100)
        self._subscribers.append(queue)
        try:
            # Yield current snapshot first so the client has full state
            yield {
                "type": "snapshot",
                "tasks": [self._task_dict(t) for t in self.get_recent_tasks()],
            }
            # Stream subsequent events
            while True:
                event = await queue.get()
                yield event
        finally:
            try:
                self._subscribers.remove(queue)
            except ValueError:
                pass

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    async def _run(
        self,
        task: TaskInfo,
        workspace: str,
        file_path: str,
        replace: bool,
        tmp_dir: str | None,
    ) -> None:
        """Execute the ingest callable in the background, updating TaskInfo state."""
        task.status = "running"
        self._notify({"type": "task_updated", "task": self._task_dict(task)})

        try:
            result = await self._ingest_fn(
                workspace=workspace,
                source_type="local",
                path=file_path,
                replace=replace,
            )
            task.status = "done"
            task.finished_at = time.time()
            task.elapsed = task.finished_at - task.created_at
            task.progress = 1.0
            task.result = result if isinstance(result, dict) else None
            self._notify({"type": "task_done", "task": self._task_dict(task)})
        except Exception as exc:  # noqa: BLE001
            task.status = "failed"
            task.finished_at = time.time()
            task.elapsed = task.finished_at - task.created_at
            task.error = str(exc)
            logger.exception(
                "Ingest task %s failed for workspace=%s file=%s",
                task.task_id,
                workspace,
                file_path,
            )
            self._notify({"type": "task_failed", "task": self._task_dict(task)})
        finally:
            if tmp_dir is not None:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    def _notify(self, event: dict[str, Any]) -> None:
        """Push event to all subscribers, dropping on full queues."""
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(event)
            except QueueFull:
                logger.debug("Dropping SSE event for slow subscriber: %s", event.get("type"))

    @staticmethod
    def _task_dict(task: TaskInfo) -> dict[str, Any]:
        """Serialize a TaskInfo to a plain dict for SSE transmission."""
        return {
            "task_id": task.task_id,
            "workspace": task.workspace,
            "file_name": task.file_name,
            "status": task.status,
            "step": task.step,
            "progress": task.progress,
            "detail": task.detail,
            "error": task.error,
            "created_at": task.created_at,
            "finished_at": task.finished_at,
            "elapsed": task.elapsed,
            "result": task.result,
        }
