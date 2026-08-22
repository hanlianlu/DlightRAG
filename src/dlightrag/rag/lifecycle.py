# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared lifecycle helpers for core-owned resources."""

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Mapping
from typing import Any, cast

logger = logging.getLogger(__name__)


def defer_cancellation(
    first: asyncio.CancelledError | None,
    current: asyncio.CancelledError,
) -> asyncio.CancelledError:
    """Record cancellation while allowing the remaining cleanup to run."""
    task = asyncio.current_task()
    if task is not None:
        while task.cancelling():
            task.uncancel()
    return first if first is not None else current


async def await_shared_cleanup[T](task: asyncio.Task[T]) -> T:
    """Join one shared cleanup task while preserving caller cancellation priority."""
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError as exc:
            cancellation = defer_cancellation(cancellation, exc)
        except BaseException:
            break
    if cancellation is not None:
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.warning("Cleanup failed while the caller was cancelled", exc_info=error)
        raise cancellation
    return task.result()


def _unwrap_worker_pool(value: Any) -> Any:
    return getattr(value, "func", value)


def _collect_worker_pools(lightrag: Any) -> list[tuple[str, Any]]:
    """Enumerate the callables LightRAG queue-wraps.

    LightRAG wraps exactly two things: ``embedding_func.func`` and each role's
    ``_role_llm_states[role].wrapped``. The base ``llm_model_func`` is
    deliberately left unwrapped upstream, and DlightRAG never passes
    ``rerank_model_func``, so neither ever owns a pool.
    """
    funcs: list[tuple[str, Any]] = [
        ("embedding_func", _unwrap_worker_pool(getattr(lightrag, "embedding_func", None)))
    ]
    states = getattr(lightrag, "_role_llm_states", None)
    items = states.items() if isinstance(states, Mapping) else ()
    for role, state in items:
        funcs.append((f"role_llm.{role}", _unwrap_worker_pool(getattr(state, "wrapped", None))))
    return funcs


async def shutdown_lightrag_worker_pools(lightrag: Any, *, dry_run: bool = False) -> int:
    """Best-effort shutdown of LightRAG worker pools.

    Returns the number of unique shutdown-capable pools discovered during a
    dry-run, or the number successfully shut down in real mode.
    """
    if lightrag is None:
        return 0

    funcs = _collect_worker_pools(lightrag)
    shutdown_count = 0
    seen: set[int] = set()
    for label, func in funcs:
        if func is None or id(func) in seen:
            continue
        seen.add(id(func))
        if not callable(getattr(func, "shutdown", None)):
            continue
        if dry_run:
            shutdown_count += 1
            continue
        try:
            result = func.shutdown(graceful=True)
            if inspect.isawaitable(result):
                await cast(Awaitable[Any], result)
        except Exception:  # noqa: BLE001
            logger.debug("Failed to shutdown %s worker pool", label, exc_info=True)
        else:
            shutdown_count += 1

    return shutdown_count
