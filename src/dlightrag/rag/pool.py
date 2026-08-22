# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Concurrent lifecycle owner for workspace RAG runtimes."""

import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from dlightrag.rag.lifecycle import await_shared_cleanup, defer_cancellation
from dlightrag.rag.ports import CorpusSchemaError, WorkspaceCorpusBackend
from dlightrag.rag.settings import RagSettings
from dlightrag.rag.workspace_rag import WorkspaceRag
from dlightrag.rag.workspaces import require_canonical_workspace_id

logger = logging.getLogger(__name__)

type WorkspaceBuilder = Callable[
    [str, RagSettings, WorkspaceCorpusBackend],
    Awaitable[WorkspaceRag],
]


class WorkspaceUnavailableError(RuntimeError):
    """One canonical workspace runtime is temporarily unavailable."""

    def __init__(self, detail: str | None = None) -> None:
        self.detail = detail or "Workspace is not available"
        super().__init__(self.detail)


class WorkspacePool:
    """Build, cache, warm, evict, and close workspace runtimes."""

    def __init__(
        self,
        *,
        settings_for: Callable[[str], RagSettings],
        backend_for: Callable[[str], WorkspaceCorpusBackend],
        build: WorkspaceBuilder,
        clock: Callable[[], float] = time.monotonic,
        initial_backoff_seconds: float = 15.0,
        max_backoff_seconds: float = 300.0,
        warm_concurrency: int = 8,
    ) -> None:
        self._settings_for = settings_for
        self._backend_for = backend_for
        self._build = build
        self._clock = clock
        self._initial_backoff = initial_backoff_seconds
        self._max_backoff = max_backoff_seconds
        self._warm_semaphore = asyncio.Semaphore(warm_concurrency)
        self._runtimes: dict[str, WorkspaceRag] = {}
        self._locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._backoff: dict[str, tuple[float, float]] = {}
        self._warmups: set[asyncio.Task[None]] = set()
        self._close_task: asyncio.Task[asyncio.CancelledError | None] | None = None
        self._closed = False

    async def get_pipeline_status(self, workspace_id: str) -> dict[str, Any] | None:
        """Read one loaded runtime's pipeline status without warming it."""
        workspace = require_canonical_workspace_id(workspace_id)
        if self._closed:
            return None
        async with self._locks[workspace]:
            runtime = self._runtimes.get(workspace)
            if runtime is None or self._closed:
                return None
            return await runtime.aget_pipeline_status()

    async def is_loaded(self, workspace_id: str) -> bool:
        """Return whether one runtime is loaded, synchronized with lifecycle changes."""
        workspace = require_canonical_workspace_id(workspace_id)
        if self._closed:
            return False
        async with self._locks[workspace]:
            return workspace in self._runtimes and not self._closed

    async def acquire(self, workspace_id: str) -> WorkspaceRag:
        """Return one runtime, constructing it once under a per-workspace lock."""
        workspace = require_canonical_workspace_id(workspace_id)
        if self._closed:
            raise WorkspaceUnavailableError("Workspace pool is closed")

        async with self._locks[workspace]:
            if self._closed:
                raise WorkspaceUnavailableError("Workspace pool is closed")
            loaded = self._runtimes.get(workspace)
            if loaded is not None:
                return loaded
            self._raise_during_backoff(workspace)
            try:
                runtime = await self._build(
                    workspace,
                    self._settings_for(workspace),
                    self._backend_for(workspace),
                )
            except CorpusSchemaError:
                raise
            except Exception as exc:
                failed = self._backoff.get(workspace)
                interval = (
                    self._initial_backoff
                    if failed is None
                    else min(failed[1] * 2, self._max_backoff)
                )
                self._backoff[workspace] = (self._clock(), interval)
                logger.warning(
                    "Workspace '%s' construction failed; retry in %.0fs",
                    workspace,
                    interval,
                    exc_info=True,
                )
                detail = str(exc)
                if exc.__cause__ is None:
                    detail = f"{type(exc).__name__}: {detail}"
                raise WorkspaceUnavailableError(
                    f"Workspace '{workspace}' is unavailable: {detail}"
                ) from exc
            if self._closed:
                await runtime.aclose()
                raise WorkspaceUnavailableError("Workspace pool is closed")
            self._runtimes[workspace] = runtime
            self._backoff.pop(workspace, None)
            return runtime

    async def warm(self, workspace_ids: Sequence[str]) -> None:
        """Warm canonical workspaces in a dedicated caller-owned task."""
        current = asyncio.current_task()
        if current is not None:
            self._warmups.add(current)

        async def _warm(workspace_id: str) -> None:
            async with self._warm_semaphore:
                await self.acquire(workspace_id)

        tasks = [
            asyncio.create_task(_warm(workspace)) for workspace in dict.fromkeys(workspace_ids)
        ]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        finally:
            if current is not None:
                self._warmups.discard(current)

    async def evict(self, workspace_id: str) -> None:
        """Close and remove one loaded runtime; a cold workspace is a no-op."""
        workspace = require_canonical_workspace_id(workspace_id)
        async with self._locks[workspace]:
            runtime = self._runtimes.get(workspace)
            if runtime is None:
                self._backoff.pop(workspace, None)
                return
            try:
                await runtime.aclose()
            finally:
                self._runtimes.pop(workspace, None)
                self._backoff.pop(workspace, None)

    async def aclose(self) -> None:
        """Cancel warmups, close all runtimes, and reject future acquires."""
        close_task = self._close_task
        if close_task is None:
            self._closed = True
            close_task = asyncio.create_task(self._close_resources())
            self._close_task = close_task

        resource_cancellation = await await_shared_cleanup(close_task)
        if resource_cancellation is not None:
            raise resource_cancellation

    async def _close_resources(self) -> asyncio.CancelledError | None:
        cancellation: asyncio.CancelledError | None = None
        warmups = list(self._warmups)
        for task in warmups:
            task.cancel()
        if warmups:
            await asyncio.gather(*warmups, return_exceptions=True)

        for workspace, lock in list(self._locks.items()):
            async with lock:
                runtime = self._runtimes.get(workspace)
                if runtime is None:
                    continue
                try:
                    await runtime.aclose()
                except asyncio.CancelledError as exc:
                    cancellation = defer_cancellation(cancellation, exc)
                except Exception:
                    logger.warning(
                        "Failed to close workspace '%s'",
                        workspace,
                        exc_info=True,
                    )
                finally:
                    self._runtimes.pop(workspace, None)
        self._backoff.clear()
        self._locks.clear()
        return cancellation

    def _raise_during_backoff(self, workspace: str) -> None:
        failed = self._backoff.get(workspace)
        if failed is None:
            return
        failed_at, interval = failed
        remaining = interval - (self._clock() - failed_at)
        if remaining > 0:
            raise WorkspaceUnavailableError(
                f"Workspace '{workspace}' in backoff (retry in {remaining:.0f}s)"
            )


__all__ = ["WorkspacePool", "WorkspaceUnavailableError"]
