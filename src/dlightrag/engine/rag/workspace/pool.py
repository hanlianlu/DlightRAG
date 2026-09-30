# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Concurrent lifecycle owner for workspace RAG runtimes."""

import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence

from dlightrag.engine.dependencies import (
    DependencyComponent,
    TransientDependencyError,
    classify_transient_dependency,
)
from dlightrag.engine.rag.workspace.lifecycle import await_shared_cleanup, defer_cancellation
from dlightrag.engine.rag.workspace.ports import CorpusSchemaError
from dlightrag.engine.rag.workspace.workspace_rag import WorkspaceRag
from dlightrag.engine.rag.workspace.workspaces import require_canonical_workspace_id

logger = logging.getLogger(__name__)

type WorkspaceBuilder = Callable[[str], Awaitable[WorkspaceRag]]
#: Told which dependency's outage keeps a workspace from being built.
type WorkspaceUnavailableCallback = Callable[[str, DependencyComponent], None]
#: Told a workspace was built, and which dependency's outage it recovered from, if any.
type WorkspaceAvailableCallback = Callable[[str, DependencyComponent | None], None]


class WorkspaceUnavailableError(TransientDependencyError):
    """One canonical workspace runtime is temporarily unavailable.

    ``component`` is the dependency whose outage keeps it from being built: a
    workspace's construction reaches corpus storage, and the model provider when
    it probes image embedding. Whoever needs the workspace waits for that one.
    """

    def __init__(
        self, detail: str | None = None, *, component: DependencyComponent = "corpus_storage"
    ) -> None:
        self.detail = detail or "Workspace is not available"
        super().__init__(component, self.detail)


class WorkspacePool:
    """Build, cache, warm, evict, and close workspace runtimes."""

    def __init__(
        self,
        *,
        build: WorkspaceBuilder,
        clock: Callable[[], float] = time.monotonic,
        initial_backoff_seconds: float = 15.0,
        max_backoff_seconds: float = 300.0,
        warm_concurrency: int = 8,
        on_workspace_unavailable: WorkspaceUnavailableCallback | None = None,
        on_workspace_available: WorkspaceAvailableCallback | None = None,
    ) -> None:
        self._build = build
        self._clock = clock
        self._initial_backoff = initial_backoff_seconds
        self._max_backoff = max_backoff_seconds
        self._warm_semaphore = asyncio.Semaphore(warm_concurrency)
        self._on_workspace_unavailable = on_workspace_unavailable
        self._on_workspace_available = on_workspace_available
        self._runtimes: dict[str, WorkspaceRag] = {}
        self._locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        # When the last build failed, the backoff it opened, and the dependency that failed it.
        self._backoff: dict[str, tuple[float, float, DependencyComponent]] = {}
        self._workspace_flights: dict[str, asyncio.Task[WorkspaceRag]] = {}
        self._close_task: asyncio.Task[asyncio.CancelledError | None] | None = None
        self._closed = False

    async def is_loaded(self, workspace_id: str) -> bool:
        """Return whether one runtime is loaded, synchronized with lifecycle changes."""
        workspace = require_canonical_workspace_id(workspace_id)
        if self._closed:
            return False
        async with self._locks[workspace]:
            return workspace in self._runtimes and not self._closed

    async def acquire(self, workspace_id: str) -> WorkspaceRag:
        """Return one runtime, joining pool-owned construction when it is cold."""
        workspace = require_canonical_workspace_id(workspace_id)
        runtime, flight = await self._runtime_or_flight(workspace)
        if runtime is not None:
            return runtime
        if flight is None:
            raise WorkspaceUnavailableError("Workspace runtime could not be initialized")
        return await self._await_workspace_flight(flight)

    async def warm(self, workspace_ids: Sequence[str]) -> None:
        """Warm each distinct canonical workspace through its pool-owned flight."""
        workspaces = tuple(
            dict.fromkeys(require_canonical_workspace_id(item) for item in workspace_ids)
        )
        flights: list[asyncio.Task[WorkspaceRag]] = []
        for workspace in workspaces:
            _, flight = await self._runtime_or_flight(workspace)
            if flight is not None:
                flights.append(flight)

        # Every flight is already running. Awaiting them in place avoids creating
        # caller-owned per-workspace tasks, while shield isolates this waiter.
        for flight in flights:
            await self._await_workspace_flight(flight)

    async def evict(self, workspace_id: str) -> None:
        """Close and remove one runtime, waiting for overlapping construction."""
        workspace = require_canonical_workspace_id(workspace_id)
        lock = self._locks[workspace]
        while True:
            async with lock:
                runtime = self._runtimes.get(workspace)
                if runtime is not None:
                    try:
                        await runtime.aclose()
                    finally:
                        if self._runtimes.get(workspace) is runtime:
                            self._runtimes.pop(workspace, None)
                        self._backoff.pop(workspace, None)
                    return
                flight = self._workspace_flights.get(workspace)
                if flight is None:
                    self._backoff.pop(workspace, None)
                    return

            # Eviction must not cancel pool-owned construction if its own caller
            # is cancelled. Once the flight settles, loop under the lifecycle lock
            # and close only the runtime that was actually published.
            try:
                await asyncio.gather(asyncio.shield(flight), return_exceptions=True)
            finally:
                if flight.done():
                    self._finish_workspace_flight(workspace, flight)

    async def aclose(self) -> None:
        """Cancel workspace flights, close runtimes, and reject future acquires."""
        close_task = self._close_task
        if close_task is None:
            self._closed = True
            close_task = asyncio.create_task(self._close_resources())
            self._close_task = close_task

        resource_cancellation = await await_shared_cleanup(close_task)
        if resource_cancellation is not None:
            raise resource_cancellation

    async def _runtime_or_flight(
        self,
        workspace: str,
    ) -> tuple[WorkspaceRag | None, asyncio.Task[WorkspaceRag] | None]:
        if self._closed:
            raise WorkspaceUnavailableError("Workspace pool is closed")

        async with self._locks[workspace]:
            if self._closed:
                raise WorkspaceUnavailableError("Workspace pool is closed")
            loaded = self._runtimes.get(workspace)
            if loaded is not None:
                return loaded, None
            self._raise_during_backoff(workspace)
            flight = self._workspace_flights.get(workspace)
            if flight is None:
                flight = asyncio.create_task(
                    self._run_workspace_flight(workspace),
                    name=f"workspace-warm:{workspace}",
                )
                self._workspace_flights[workspace] = flight
                flight.add_done_callback(
                    lambda completed, flight_workspace=workspace: self._finish_workspace_flight(
                        flight_workspace, completed
                    )
                )
            return None, flight

    async def _run_workspace_flight(self, workspace: str) -> WorkspaceRag:
        async with self._warm_semaphore:
            async with self._locks[workspace]:
                if self._closed:
                    raise WorkspaceUnavailableError("Workspace pool is closed")
                loaded = self._runtimes.get(workspace)
                if loaded is not None:
                    return loaded
                self._raise_during_backoff(workspace)
                try:
                    runtime = await self._build(workspace)
                except CorpusSchemaError:
                    raise
                except Exception as exc:
                    # A bare connection error or timeout is storage's: the model
                    # provider's calls name their own outages.
                    component = classify_transient_dependency(exc, component_hint="corpus_storage")
                    if component is None:
                        raise
                    failed = self._backoff.get(workspace)
                    interval = (
                        self._initial_backoff
                        if failed is None
                        else min(failed[1] * 2, self._max_backoff)
                    )
                    self._backoff[workspace] = (self._clock(), interval, component)
                    logger.warning(
                        "Workspace '%s' construction failed (%s unavailable); retry in %.0fs",
                        workspace,
                        component,
                        interval,
                        extra={"error_type": type(exc).__name__},
                    )
                    self._notify(self._on_workspace_unavailable, workspace, component)
                    raise WorkspaceUnavailableError(
                        f"Workspace '{workspace}' is temporarily unavailable", component=component
                    ) from exc
                if self._closed:
                    await self._close_unpublished_runtime(workspace, runtime)
                    raise WorkspaceUnavailableError("Workspace pool is closed")
                self._runtimes[workspace] = runtime
                recovered = self._backoff.pop(workspace, None)
                self._notify(
                    self._on_workspace_available,
                    workspace,
                    None if recovered is None else recovered[2],
                )
                return runtime

    async def _close_unpublished_runtime(
        self,
        workspace: str,
        runtime: WorkspaceRag,
    ) -> None:
        close_task = asyncio.create_task(runtime.aclose())
        try:
            await await_shared_cleanup(close_task)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning(
                "Failed to close unpublished workspace '%s'",
                workspace,
                exc_info=True,
            )

    async def _await_workspace_flight(
        self,
        flight: asyncio.Task[WorkspaceRag],
    ) -> WorkspaceRag:
        try:
            return await asyncio.shield(flight)
        except asyncio.CancelledError as exc:
            current = asyncio.current_task()
            caller_was_cancelled = current is not None and current.cancelling() > 0
            if self._closed and flight.cancelled() and not caller_was_cancelled:
                raise WorkspaceUnavailableError("Workspace pool is closed") from exc
            raise

    def _finish_workspace_flight(
        self,
        workspace: str,
        task: asyncio.Task[WorkspaceRag],
    ) -> None:
        if self._workspace_flights.get(workspace) is task:
            self._workspace_flights.pop(workspace, None)
        if not task.cancelled():
            # Pool ownership includes consuming every flight exception. Build
            # failures are logged at their translation site, not again here.
            task.exception()

    async def _close_resources(self) -> asyncio.CancelledError | None:
        cancellation: asyncio.CancelledError | None = None
        flights = list(self._workspace_flights.values())
        for flight in flights:
            flight.cancel()
        if flights:
            await asyncio.gather(*flights, return_exceptions=True)
        self._workspace_flights.clear()

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
                    if self._runtimes.get(workspace) is runtime:
                        self._runtimes.pop(workspace, None)
        self._backoff.clear()
        self._locks.clear()
        return cancellation

    @staticmethod
    def _notify[C: DependencyComponent | None](
        callback: Callable[[str, C], None] | None,
        workspace: str,
        component: C,
    ) -> None:
        if callback is None:
            return
        try:
            callback(workspace, component)
        except Exception:
            logger.warning("Workspace health callback failed", exc_info=True)

    def _raise_during_backoff(self, workspace: str) -> None:
        failed = self._backoff.get(workspace)
        if failed is None:
            return
        failed_at, interval, component = failed
        remaining = interval - (self._clock() - failed_at)
        if remaining > 0:
            raise WorkspaceUnavailableError(
                f"Workspace '{workspace}' in backoff (retry in {remaining:.0f}s)",
                component=component,
            )


__all__ = [
    "WorkspaceAvailableCallback",
    "WorkspacePool",
    "WorkspaceUnavailableCallback",
    "WorkspaceUnavailableError",
]
