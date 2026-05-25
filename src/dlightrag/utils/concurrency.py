# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Bounded concurrency utilities."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from typing import Any, cast

logger = logging.getLogger(__name__)

_MISSING = object()


async def bounded_gather(
    coros: Sequence[Coroutine[Any, Any, Any]],
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
    async def _await_coro(coro: Coroutine[Any, Any, Any]) -> Any:
        return await coro

    return await bounded_map(coros, _await_coro, max_concurrent=max_concurrent, task_name=task_name)


async def bounded_map[T, R](
    items: Sequence[T],
    worker: Callable[[T], Awaitable[R]],
    *,
    max_concurrent: int = 4,
    task_name: str = "task",
) -> list[R | Exception]:
    """Apply an async worker to items with bounded concurrency.

    Unlike ``bounded_gather``, this accepts plain input items and creates each
    worker coroutine only when a worker slot is available.
    """
    if not items:
        return []

    results: list[R | Exception | object] = [_MISSING] * len(items)
    queue: asyncio.Queue[tuple[int, T]] = asyncio.Queue()
    for idx, item in enumerate(items):
        queue.put_nowait((idx, item))

    async def _worker() -> None:
        while True:
            try:
                idx, item = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                results[idx] = await worker(item)
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                logger.warning("%s #%d failed: %s", task_name, idx, exc)
                results[idx] = exc if isinstance(exc, Exception) else Exception(str(exc))

    async with asyncio.TaskGroup() as tg:
        for _ in range(min(max(1, max_concurrent), len(items))):
            tg.create_task(_worker())

    final: list[R | Exception] = []
    for item in results:
        if item is _MISSING:
            final.append(RuntimeError("bounded_map worker returned no result"))
        else:
            final.append(cast(R | Exception, item))
    return final
