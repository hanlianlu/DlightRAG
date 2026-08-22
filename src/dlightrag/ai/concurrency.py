# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Bounded concurrency utilities."""

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable, Coroutine, Sequence
from typing import Any, cast

logger = logging.getLogger(__name__)


async def shutdown_async_callable(func: Any, *, graceful: bool = True) -> None:
    """Best-effort shutdown for LightRAG priority-queue wrapped callables."""
    shutdown = getattr(func, "shutdown", None)
    if not callable(shutdown):
        return
    result = shutdown(graceful=graceful)
    if inspect.isawaitable(result):
        await cast(Awaitable[Any], result)


async def bounded_gather(
    coros: Sequence[Coroutine[Any, Any, Any]],
    *,
    max_concurrent: int = 4,
    task_name: str = "task",
) -> list[Any]:
    """Run coroutines with bounded concurrency, preserving input order.

    Thin convenience wrapper over :func:`bounded_map` (identity worker).

    When to use which
    -----------------
    Prefer ``bounded_gather`` when you already hold a list of coroutines,
    especially heterogeneous or multi-argument calls that read most naturally
    as a comprehension, e.g. ``[score(start, batch) for start, batch in xs]``.
    Prefer :func:`bounded_map` when you start from plain data plus a single
    async worker (map semantics): it avoids the intermediate coroutine list
    and creates coroutines lazily, so only ``max_concurrent`` exist at once
    (cheaper for large inputs, no dangling "coroutine was never awaited"
    warnings if the run is cancelled). Both share the same concurrency
    mechanism and error convention, so the choice is about input shape, not
    behaviour.

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

    When to use which
    -----------------
    Prefer ``bounded_map`` when you start from plain data plus a single async
    worker (map semantics): only ``max_concurrent`` coroutines exist at once,
    which is cheaper for large inputs and avoids dangling "coroutine was never
    awaited" warnings on cancellation. Prefer :func:`bounded_gather` when you
    already hold a list of coroutines, especially heterogeneous or
    multi-argument calls that read most naturally as a comprehension. Both
    share the same concurrency mechanism and error convention, so the choice
    is about input shape, not behaviour.

    Results preserve input order; a failed item contributes its ``Exception``
    instead of its return value.
    """
    if not items:
        return []

    results: dict[int, R | Exception] = {}
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
            except Exception as exc:
                logger.warning("%s #%d failed: %s", task_name, idx, exc)
                results[idx] = exc

    async with asyncio.TaskGroup() as tg:
        for _ in range(min(max(1, max_concurrent), len(items))):
            tg.create_task(_worker())

    return [results[idx] for idx in range(len(items))]
