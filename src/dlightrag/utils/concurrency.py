# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Bounded concurrency utilities.

Provides ``bounded_gather`` — a drop-in replacement for
``asyncio.gather(*tasks, return_exceptions=True)`` that limits the
number of concurrently running coroutines using an asyncio.Queue +
TaskGroup worker-pool pattern.

Why not Semaphore + gather?
- ``gather`` creates all coroutines upfront (memory spike for large batches).
- Semaphore inside each coroutine means N coroutines are alive but only M run
  — wasteful for N >> M.
- ``return_exceptions=True`` silently swallows failures without logging.

The Queue + workers pattern ensures at most ``max_concurrent`` coroutines
exist at any time, with explicit per-task error logging.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

logger = logging.getLogger(__name__)


async def bounded_gather(
    coros: list[Coroutine[Any, Any, Any]],
    *,
    max_concurrent: int = 4,
    task_name: str = "task",
) -> list[Any]:
    """Run coroutines with bounded concurrency, preserving input order.

    Parameters
    ----------
    coros:
        Coroutines to execute. Each is consumed exactly once.
    max_concurrent:
        Maximum number of coroutines running at the same time.
    task_name:
        Label for log messages on failure (e.g. "ingestion", "download").

    Returns
    -------
    list[Any]
        Results in the same order as *coros*. Successful coroutines
        contribute their return value; failed ones contribute the
        ``Exception`` instance (same convention as
        ``asyncio.gather(return_exceptions=True)``).
    """
    if not coros:
        return []

    results: list[Any] = [None] * len(coros)
    queue: asyncio.Queue[tuple[int, Coroutine[Any, Any, Any]]] = asyncio.Queue()

    for idx, coro in enumerate(coros):
        queue.put_nowait((idx, coro))

    async def _worker() -> None:
        while True:
            try:
                idx, coro = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                results[idx] = await coro
            except Exception as exc:
                logger.warning("%s #%d failed: %s", task_name, idx, exc)
                results[idx] = exc

    async with asyncio.TaskGroup() as tg:
        for _ in range(min(max_concurrent, len(coros))):
            tg.create_task(_worker())

    return results
