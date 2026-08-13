# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Transport-neutral execution helpers for public retrieve requests."""

from typing import Any

from dlightrag.core.client_requests import query_kwargs_from_payload
from dlightrag.core.servicemanager import RAGServiceManager


async def execute_retrieve(
    *,
    manager: RAGServiceManager,
    payload: Any,
    resolved_workspaces: list[str],
):
    """Execute a transport-normalized retrieve request against an authorized manager."""

    return await manager.aretrieve(
        payload.query,
        workspaces=resolved_workspaces,
        top_k=payload.top_k,
        chunk_top_k=payload.chunk_top_k,
        **query_kwargs_from_payload(payload),
    )


__all__ = ["execute_retrieve"]
