# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Transport-neutral execution helpers for public retrieve/answer requests."""

from typing import Any

from dlightrag.core.client_attachments import answer_link_resources
from dlightrag.core.client_contracts import conversation_history_as_dicts
from dlightrag.core.client_requests import query_kwargs_from_payload
from dlightrag.core.resources.models import ResourceInput
from dlightrag.core.scope import RequestScope
from dlightrag.core.servicemanager import RAGServiceManager


async def execute_retrieve(
    *,
    manager: RAGServiceManager,
    payload: Any,
    resolved_workspaces: list[str],
    scope: RequestScope,
):
    """Execute a transport-normalized retrieve request against an authorized manager."""

    return await manager.aretrieve(
        payload.query,
        workspaces=resolved_workspaces,
        top_k=payload.top_k,
        chunk_top_k=payload.chunk_top_k,
        scope=scope,
        **query_kwargs_from_payload(payload),
    )


async def execute_answer(
    *,
    manager: RAGServiceManager,
    payload: Any,
    resolved_workspaces: list[str],
    scope: RequestScope,
    resources: list[ResourceInput] | None = None,
):
    """Execute a transport-normalized answer request against an authorized manager.

    ``resources`` overrides attachment derivation for transports (REST multipart)
    that combine JSON link descriptors with uploaded files; when omitted the
    request's ``attachments`` link descriptors become the request-local resources.
    """

    if resources is None:
        resources = answer_link_resources(getattr(payload, "attachments", None))
    return await manager.aanswer(
        payload.query,
        workspaces=resolved_workspaces,
        top_k=payload.top_k,
        chunk_top_k=payload.chunk_top_k,
        semantic_highlights=payload.semantic_highlights,
        history=conversation_history_as_dicts(payload.history),
        resources=resources or None,
        scope=scope,
        **query_kwargs_from_payload(payload),
    )


__all__ = ["execute_answer", "execute_retrieve"]
