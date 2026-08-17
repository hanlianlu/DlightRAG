# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Deduplicate a run's exact tool calls across its turns and its restarts."""

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from dlightrag_agent.tools import ToolResult


class ExactCallCache:
    """Run each distinct tool call once, so a repeat costs a turn and not a search.

    This is execution bookkeeping, not memory the model reads: it keys on exact
    arguments, and the episode is what shows the model which angles are spent.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._tasks: dict[str, asyncio.Future[ToolResult]] = {}
        self._closed = False

    def export_results(self) -> dict[str, dict[str, Any]]:
        """Return the calls this run already answered, newest state only.

        An in-flight, cancelled, or failed call is not a completed result: it is
        left out so a resumed run executes it again rather than replaying a
        half-finished one.
        """
        exported: dict[str, dict[str, Any]] = {}
        for key, task in self._tasks.items():
            if not task.done() or task.cancelled() or task.exception() is not None:
                continue
            result = task.result()
            exported[key] = {"content": result.content, "details": result.details}
        return exported

    def restore_results(self, results: Mapping[str, Mapping[str, Any]]) -> None:
        """Seed completed results so a resumed run does not re-execute them."""
        loop = asyncio.get_running_loop()
        for key, payload in results.items():
            details = payload.get("details")
            future: asyncio.Future[ToolResult] = loop.create_future()
            future.set_result(
                ToolResult(
                    content=str(payload.get("content") or ""),
                    details=dict(details) if isinstance(details, Mapping) else None,
                )
            )
            self._tasks[key] = future

    async def aclose(self) -> None:
        async with self._lock:
            if self._closed:
                return
            self._closed = True
            tasks, self._tasks = list(self._tasks.values()), {}
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def run(
        self,
        key: str,
        operation: Callable[[], Awaitable[ToolResult]],
    ) -> ToolResult:
        async with self._lock:
            if self._closed:
                raise RuntimeError("tool-call cache is closed")
            task = self._tasks.get(key)
            repeated = task is not None
            if task is None:
                task = asyncio.ensure_future(operation())
                self._tasks[key] = task
        try:
            result = await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.cancelled():
                async with self._lock:
                    if self._tasks.get(key) is task:
                        self._tasks.pop(key, None)
            raise
        except BaseException:
            async with self._lock:
                if self._tasks.get(key) is task:
                    self._tasks.pop(key, None)
            raise
        if repeated:
            return ToolResult(
                content="Equivalent tool call already executed; no new evidence was added.",
                details=result.details,
                cached=True,
            )
        return result


__all__ = ["ExactCallCache"]
