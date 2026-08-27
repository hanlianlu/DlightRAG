# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""MCP tools for inline retrieval."""

from __future__ import annotations

from typing import Annotated, Any

from mcp.types import ToolAnnotations
from pydantic import Field

from dlightrag.adapters.mcp import server as mcp_server
from dlightrag.adapters.mcp.contracts import (
    RetrieveInput,
)
from dlightrag.adapters.mcp.server import (
    QueryImagesParam,
    mcp_app,
)
from dlightrag.application.access import AccessAction
from dlightrag.application.retrieval import MetadataFilter, RetrieveProjection
from dlightrag.application.retrieval import RetrieveRequest as ServiceRequest


@mcp_app.tool(
    name="retrieve",
    description=(
        "Query the RAG knowledge base for relevant information. Supports structured "
        "metadata filters and default or selected workspaces for precise document lookups."
    ),
    annotations=ToolAnnotations(read_only_hint=True),
)
async def retrieve_tool(
    query: Annotated[str, Field(description="The search query")],
    top_k: Annotated[
        int | None,
        Field(default=None, description="Number of top results to return"),
    ] = None,
    chunk_top_k: Annotated[
        int | None,
        Field(default=None, description="Vector chunk candidate count override."),
    ] = None,
    bm25_query: Annotated[
        str | None,
        Field(
            default=None,
            max_length=1024,
            description=(
                "Optional lexical/BM25 query override. When omitted, BM25 uses the main query."
            ),
        ),
    ] = None,
    workspaces: Annotated[
        list[str] | None,
        Field(default=None, description="Workspace names to search. Omit for default."),
    ] = None,
    all_workspaces: Annotated[
        bool,
        Field(
            default=False,
            description="Search all workspaces visible to the current caller.",
        ),
    ] = False,
    filters: Annotated[
        dict[str, Any] | None,
        Field(default=None, description="Metadata filters for structured queries."),
    ] = None,
    query_images: QueryImagesParam = Field(default_factory=list),
) -> dict[str, Any]:
    args = RetrieveInput.model_validate(locals())
    application = await mcp_server._ensure_application()
    resolved_workspaces = await mcp_server._resolve_authorized_query_workspaces(
        application,
        workspaces=args.workspaces,
        all_workspaces=args.all_workspaces,
    )
    visual_workspaces = await mcp_server._authorized_workspace_names(
        AccessAction.WORKSPACE_READ_VISUAL_ASSET,
        resolved_workspaces,
        application=application,
    )
    result = await application.retrieval.retrieve(
        ServiceRequest(
            query=args.query,
            workspaces=tuple(resolved_workspaces),
            top_k=args.top_k,
            chunk_top_k=args.chunk_top_k,
            bm25_query=args.bm25_query,
            filters=MetadataFilter.model_validate(args.filters) if args.filters else None,
            query_images=tuple(
                image.model_dump(exclude_none=True) for image in args.query_images or ()
            ),
            projection=RetrieveProjection(
                downloadable_workspaces=frozenset(),
                visual_workspaces=frozenset(visual_workspaces),
            ),
        )
    )
    return {
        "contexts": result.contexts,
        "sources": list(result.sources),
        "trace": dict(result.trace),
        "image_descriptions": list(result.image_descriptions),
    }
