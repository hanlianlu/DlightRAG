# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared lifecycle helpers for LightRAG-owned resources."""

import logging
from collections.abc import Mapping
from typing import Any

from dlightrag.utils.concurrency import shutdown_async_callable

logger = logging.getLogger(__name__)


def _unwrap_worker_pool(value: Any) -> Any:
    return getattr(value, "func", value)


async def shutdown_lightrag_worker_pools(lightrag: Any, *, dry_run: bool = False) -> int:
    """Best-effort shutdown of LightRAG worker pools.

    Returns the number of unique shutdown-capable pools discovered.
    """
    if lightrag is None:
        return 0

    funcs: list[tuple[str, Any]] = []
    for attr in ("embedding_func", "llm_model_func", "rerank_model_func"):
        try:
            funcs.append((attr, _unwrap_worker_pool(getattr(lightrag, attr, None))))
        except Exception:  # noqa: BLE001
            logger.debug("Failed to collect %s worker pool", attr, exc_info=True)

    role_funcs = getattr(lightrag, "role_llm_funcs", None) or {}
    items = role_funcs.items() if isinstance(role_funcs, Mapping) else ()
    for role, func in items:
        label = f"role_llm_funcs.{role}"
        try:
            funcs.append((label, _unwrap_worker_pool(func)))
        except Exception:  # noqa: BLE001
            logger.debug("Failed to collect %s worker pool", label, exc_info=True)

    shutdown_count = 0
    seen: set[int] = set()
    for label, func in funcs:
        if func is None or id(func) in seen:
            continue
        seen.add(id(func))
        if not callable(getattr(func, "shutdown", None)):
            continue
        shutdown_count += 1
        if dry_run:
            continue
        try:
            await shutdown_async_callable(func)
        except Exception:  # noqa: BLE001
            logger.debug("Failed to shutdown %s worker pool", label, exc_info=True)

    return shutdown_count
