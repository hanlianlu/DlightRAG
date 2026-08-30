# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""MCP transport adapter for the runtime model catalogue."""

from __future__ import annotations

from typing import Annotated, Any

from mcp.types import ToolAnnotations
from pydantic import Field

from dlightrag.adapters.mcp import server as mcp_server
from dlightrag.adapters.mcp.server import mcp_app
from dlightrag.application.access import AccessAction, current_request_scope
from dlightrag.application.model_catalogue import (
    ModelCatalogueEntryNotFoundError,
    ModelCatalogueRevisionConflict,
    ModelCatalogueView,
)


def _view(view: ModelCatalogueView) -> dict[str, Any]:
    return {
        "revision": view.revision,
        "models": [
            {
                "provider": item.provider,
                "model": item.model,
                "base_url": item.base_url,
                "profile": dict(item.profile),
                "source": item.source,
            }
            for item in view.models
        ],
    }


@mcp_app.tool(
    name="get_model_catalogue",
    description=(
        "Return the effective runtime model catalogue and its optimistic-concurrency "
        "revision. Profiles include complete capacity, image, and reasoning facts."
    ),
    annotations=ToolAnnotations(read_only_hint=True),
)
async def get_model_catalogue_tool() -> dict[str, Any]:
    application = await mcp_server._ensure_application()
    return _view(application.model_catalogue.read())


@mcp_app.tool(
    name="upsert_model_catalogue_entry",
    description=(
        "Admin-only: publish one complete model endpoint profile. expected_revision must "
        "equal get_model_catalogue.revision; stale writes are rejected."
    ),
    annotations=ToolAnnotations(destructive_hint=False),
)
async def upsert_model_catalogue_entry_tool(
    entry: Annotated[
        dict[str, Any],
        Field(description="Complete provider/model/base_url/profile catalogue entry."),
    ],
    expected_revision: Annotated[
        str,
        Field(description="Exact effective revision returned by get_model_catalogue."),
    ],
) -> dict[str, Any]:
    application = await mcp_server._ensure_application()
    await mcp_server._enforce_access(
        AccessAction.MODEL_CATALOGUE_WRITE,
        application=application,
    )
    try:
        view = await application.model_catalogue.upsert(
            entry,
            expected_revision=expected_revision,
            actor=current_request_scope().user_id,
        )
    except ModelCatalogueRevisionConflict as exc:
        raise ValueError(str(exc)) from None
    return _view(view)


@mcp_app.tool(
    name="remove_model_catalogue_entry",
    description=(
        "Admin-only: restore a built-in endpoint to its default profile or delete a "
        "custom endpoint, guarded by the effective revision."
    ),
    annotations=ToolAnnotations(destructive_hint=True),
)
async def remove_model_catalogue_entry_tool(
    provider: Annotated[str, Field(description="Canonical provider name.")],
    model: Annotated[str, Field(description="Model id.")],
    expected_revision: Annotated[
        str,
        Field(description="Exact effective revision returned by get_model_catalogue."),
    ],
    base_url: Annotated[
        str | None,
        Field(default=None, description="Endpoint base URL; omit for provider default."),
    ] = None,
) -> dict[str, Any]:
    application = await mcp_server._ensure_application()
    await mcp_server._enforce_access(
        AccessAction.MODEL_CATALOGUE_WRITE,
        application=application,
    )
    try:
        view = await application.model_catalogue.remove(
            provider=provider,
            model=model,
            base_url=base_url,
            expected_revision=expected_revision,
            actor=current_request_scope().user_id,
        )
    except ModelCatalogueRevisionConflict as exc:
        raise ValueError(str(exc)) from None
    except ModelCatalogueEntryNotFoundError as exc:
        raise ValueError(str(exc.args[0])) from None
    return _view(view)


__all__ = [
    "get_model_catalogue_tool",
    "remove_model_catalogue_entry_tool",
    "upsert_model_catalogue_entry_tool",
]
