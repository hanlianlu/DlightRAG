# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""MCP server for agent integration (stdio + streamable-http).

Entry point: dlightrag-mcp
Primarily used by DeerFlow and other MCP-compatible agents for
retrieve() + lightweight ingest().
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP
from mcp.server.stdio import stdio_server
from mcp.types import ContentBlock, TextContent
from pydantic import Field

import dlightrag
from dlightrag.config import DlightragConfig, get_config, load_config, set_config
from dlightrag.core.client_contracts import (
    ConversationMessage,
    MetadataPolicy,
    SourceType,
    dump_optional_list,
)
from dlightrag.core.client_payloads import (
    answer_payload,
    retrieval_payload,
)
from dlightrag.core.client_requests import (
    ingest_kwargs_from_payload,
    managed_local_ingest_path,
    query_kwargs_from_payload,
)
from dlightrag.core.scope import RequestScope, current_request_scope, request_scope_context
from dlightrag.core.servicemanager import RAGServiceManager
from dlightrag.mcp.contracts import (
    AnswerInput,
    CreateWorkspaceInput,
    DeleteFilesInput,
    DeleteWorkspaceInput,
    IngestInput,
    IngestJobStatusInput,
    ListFilesInput,
    QueryImage,
    RetrieveInput,
)

logger = logging.getLogger(__name__)

MetadataPolicyParam = MetadataPolicy
SourceTypeParam = SourceType
QueryImagesParam = Annotated[
    list[QueryImage],
    Field(
        max_length=3,
        description="User-attached image URLs or data URI blocks (max 3)",
    ),
]
ReferencedImageIdsParam = Annotated[
    list[str],
    Field(description="Previously stored image ids to include in the query."),
]


class DlightRAGFastMCP(FastMCP):
    """FastMCP with DlightRAG's strict input and text-error contract."""

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> Sequence[ContentBlock] | dict[str, Any]:
        try:
            self._reject_unknown_arguments(name, arguments or {})
            return await super().call_tool(name, arguments or {})
        except Exception as exc:
            logger.exception("MCP tool '%s' failed: %s", name, exc)
            return [TextContent(type="text", text=f"Error: {exc}")]

    def _reject_unknown_arguments(self, name: str, arguments: Mapping[str, Any]) -> None:
        tool = self._tool_manager._tools.get(name)
        if tool is None:
            raise ValueError(f"Unknown tool: {name}")
        allowed = set(tool.parameters.get("properties", {}))
        unknown = sorted(set(arguments) - allowed)
        if unknown:
            raise ValueError(f"Unexpected argument(s) for {name}: {', '.join(unknown)}")


mcp_app = DlightRAGFastMCP(
    "dlightrag",
    log_level="INFO",
    warn_on_duplicate_tools=True,
)
server = mcp_app._mcp_server
server.version = dlightrag.__version__


def _get_config() -> DlightragConfig:
    return get_config()


_manager: RAGServiceManager | None = None


async def _ensure_manager() -> RAGServiceManager:
    global _manager
    if _manager is None:
        _manager = await RAGServiceManager.create()
    return _manager


def _normalize_workspace_argument(args: CreateWorkspaceInput) -> tuple[str, str]:
    from dlightrag.utils import normalize_workspace, validate_workspace_name

    label = validate_workspace_name(args.workspace)
    display_name = validate_workspace_name(args.display_name or label)
    return normalize_workspace(label), display_name


@mcp_app.tool(
    name="retrieve",
    description=(
        "Query the RAG knowledge base for relevant information. Supports structured "
        "metadata filters for precise document lookups."
    ),
)
async def retrieve_tool(
    query: Annotated[str, Field(description="The search query")],
    top_k: Annotated[
        int | None,
        Field(default=None, description="Number of top results to return"),
    ] = None,
    workspaces: Annotated[
        list[str] | None,
        Field(default=None, description="Workspace names to search. Omit for default."),
    ] = None,
    filters: Annotated[
        dict[str, Any] | None,
        Field(default=None, description="Metadata filters for structured queries."),
    ] = None,
    query_images: QueryImagesParam = Field(default_factory=list),
    session_id: Annotated[
        str | None,
        Field(default=None, description="Session id for image memory."),
    ] = None,
    referenced_image_ids: ReferencedImageIdsParam = Field(default_factory=list),
) -> dict[str, Any]:
    args = RetrieveInput.model_validate(locals())
    manager = await _ensure_manager()
    scope = current_request_scope().for_workspaces(args.workspaces)
    result = await manager.aretrieve(
        args.query,
        workspaces=args.workspaces,
        top_k=args.top_k,
        scope=scope,
        **query_kwargs_from_payload(args, include_multimodal_content=False),
    )
    return retrieval_payload(result)


@mcp_app.tool(
    name="answer",
    description=(
        "Ask a question and get an LLM-generated answer backed by retrieved context "
        "from the knowledge base."
    ),
)
async def answer_tool(
    query: Annotated[str, Field(description="The question to answer")],
    top_k: Annotated[
        int | None,
        Field(default=None, description="Retrieval candidate count override for this answer"),
    ] = None,
    chunk_top_k: Annotated[
        int | None,
        Field(default=None, description="Vector chunk candidate count override for this answer"),
    ] = None,
    answer_context_top_k: Annotated[
        int | None,
        Field(default=None, description="Maximum chunks included in the final answer prompt"),
    ] = None,
    workspaces: Annotated[
        list[str] | None,
        Field(default=None, description="Workspace names to search. Omit for default."),
    ] = None,
    filters: Annotated[
        dict[str, Any] | None,
        Field(default=None, description="Metadata filters for structured queries."),
    ] = None,
    conversation_history: Annotated[
        list[ConversationMessage] | None,
        Field(default=None, description="Prior conversation turns as role/content objects."),
    ] = None,
    query_images: QueryImagesParam = Field(default_factory=list),
    session_id: Annotated[
        str | None,
        Field(default=None, description="Session id for image memory."),
    ] = None,
    referenced_image_ids: ReferencedImageIdsParam = Field(default_factory=list),
) -> dict[str, Any]:
    args = AnswerInput.model_validate(locals())
    manager = await _ensure_manager()
    scope = current_request_scope().for_workspaces(args.workspaces)
    result = await manager.aanswer(
        args.query,
        conversation_history=dump_optional_list(args.conversation_history),
        workspaces=args.workspaces,
        top_k=args.top_k,
        chunk_top_k=args.chunk_top_k,
        answer_context_top_k=args.answer_context_top_k,
        scope=scope,
        **query_kwargs_from_payload(args, include_multimodal_content=False),
    )
    return answer_payload(result)


@mcp_app.tool(name="list_workspaces", description="List all registered workspaces.")
async def list_workspaces_tool() -> dict[str, Any]:
    manager = await _ensure_manager()
    records = await manager.list_workspace_records()
    return {
        "workspaces": [row["workspace"] for row in records],
        "records": records,
    }


@mcp_app.tool(name="create_workspace", description="Create an empty DlightRAG workspace.")
async def create_workspace_tool(
    workspace: Annotated[str, Field(description="Workspace name to create.")],
    display_name: Annotated[
        str | None,
        Field(default=None, description="Optional user-facing display name."),
    ] = None,
) -> dict[str, Any]:
    args = CreateWorkspaceInput.model_validate(locals())
    manager = await _ensure_manager()
    normalized_workspace, normalized_display_name = _normalize_workspace_argument(args)
    existing = await manager.list_workspaces()
    if normalized_workspace in existing:
        raise ValueError(f"Workspace '{normalized_display_name}' already exists")
    await manager.acreate_workspace(normalized_workspace, display_name=normalized_display_name)
    return {
        "workspace": normalized_workspace,
        "display_name": normalized_display_name,
        "created": True,
    }


@mcp_app.tool(name="delete_workspace", description="Delete/reset one DlightRAG workspace.")
async def delete_workspace_tool(
    workspace: Annotated[str, Field(description="Workspace name to delete.")],
    keep_files: Annotated[
        bool,
        Field(default=False, description="Keep source files on disk."),
    ] = False,
    dry_run: Annotated[
        bool,
        Field(default=False, description="Report what would be deleted without mutating storage."),
    ] = False,
) -> dict[str, Any]:
    args = DeleteWorkspaceInput.model_validate(locals())
    manager = await _ensure_manager()
    from dlightrag.utils import normalize_workspace, validate_workspace_name

    label = validate_workspace_name(args.workspace)
    normalized_workspace = normalize_workspace(label)
    result = await manager.areset(
        workspace=label,
        keep_files=args.keep_files,
        dry_run=args.dry_run,
    )
    return {
        "workspace": normalized_workspace,
        "deleted": not args.dry_run,
        "result": result,
    }


@mcp_app.tool(name="ingest", description="Ingest document(s) into the RAG knowledge base.")
async def ingest_tool(
    source_type: Annotated[SourceTypeParam, Field(description="Type of data source")],
    path: Annotated[
        str | None,
        Field(default=None, description="File or directory path for local source."),
    ] = None,
    container_name: Annotated[
        str | None,
        Field(default=None, description="Azure Blob container name."),
    ] = None,
    blob_path: Annotated[
        str | None,
        Field(default=None, description="Specific blob path for azure_blob."),
    ] = None,
    bucket: Annotated[
        str | None,
        Field(default=None, description="S3 bucket name."),
    ] = None,
    key: Annotated[
        str | None,
        Field(default=None, description="S3 object key, single object or prefix."),
    ] = None,
    prefix: Annotated[
        str | None,
        Field(default=None, description="Path/blob/key prefix filter."),
    ] = None,
    url: Annotated[
        str | None,
        Field(default=None, description="Public HTTPS document URL."),
    ] = None,
    urls: Annotated[
        list[str] | None,
        Field(default=None, description="Public HTTPS document URLs."),
    ] = None,
    filename: Annotated[
        str | None,
        Field(default=None, description="Parser filename for a single URL."),
    ] = None,
    source_uri: Annotated[
        str | None,
        Field(default=None, description="Stable source URI stored for a single URL."),
    ] = None,
    source_uris: Annotated[
        list[str] | None,
        Field(default=None, description="Stable source URIs stored for URL batches."),
    ] = None,
    replace: Annotated[
        bool | None,
        Field(default=None, description="Replace existing documents."),
    ] = None,
    workspace: Annotated[
        str | None,
        Field(default=None, description="Target workspace. Omit for default."),
    ] = None,
    title: Annotated[
        str | None,
        Field(default=None, description="Optional document title metadata."),
    ] = None,
    author: Annotated[
        str | None,
        Field(default=None, description="Optional document author metadata."),
    ] = None,
    metadata: Annotated[
        dict[str, Any] | None,
        Field(default=None, description="User metadata to attach to ingested documents."),
    ] = None,
    metadata_policy: Annotated[
        MetadataPolicyParam | None,
        Field(default=None, description="How undeclared user metadata fields are handled."),
    ] = None,
) -> dict[str, Any]:
    args = IngestInput.model_validate(locals())
    manager = await _ensure_manager()
    workspace_name = args.workspace or _get_config().workspace
    kwargs = ingest_kwargs_from_payload(args)
    if args.source_type == "local":
        kwargs["path"] = managed_local_ingest_path(
            source_type=args.source_type,
            path=kwargs.get("path"),
            input_dir=_get_config().input_dir_path,
            workspace=workspace_name,
        )
    return await manager.astart_ingest_job(workspace_name, source_type=args.source_type, **kwargs)


@mcp_app.tool(name="ingest_job_status", description="Return the status of an ingest job.")
async def ingest_job_status_tool(
    job_id: Annotated[str, Field(description="Ingest job id returned by the ingest tool.")],
) -> dict[str, Any]:
    args = IngestJobStatusInput.model_validate(locals())
    manager = await _ensure_manager()
    if not args.job_id:
        raise ValueError("job_id is required")
    result = await manager.get_ingest_job(args.job_id)
    if result is None:
        raise ValueError(f"Ingest job not found: {args.job_id}")
    return result


@mcp_app.tool(name="list_files", description="List all documents ingested in the knowledge base.")
async def list_files_tool(
    workspace: Annotated[
        str | None,
        Field(default=None, description="Workspace to list files from. Omit for default."),
    ] = None,
) -> dict[str, Any]:
    args = ListFilesInput.model_validate(locals())
    manager = await _ensure_manager()
    workspace_name = args.workspace or _get_config().workspace
    files = await manager.list_ingested_files(workspace_name)
    return {"files": files, "count": len(files), "workspace": workspace_name}


@mcp_app.tool(name="delete_files", description="Delete documents from the knowledge base.")
async def delete_files_tool(
    filenames: Annotated[
        list[str] | None,
        Field(default=None, description="List of filenames to delete."),
    ] = None,
    file_paths: Annotated[
        list[str] | None,
        Field(default=None, description="List of file paths to delete."),
    ] = None,
    workspace: Annotated[
        str | None,
        Field(default=None, description="Workspace to delete from. Omit for default."),
    ] = None,
) -> dict[str, Any]:
    args = DeleteFilesInput.model_validate(locals())
    manager = await _ensure_manager()
    workspace_name = args.workspace or _get_config().workspace
    results = await manager.delete_files(
        workspace_name,
        filenames=args.filenames,
        file_paths=args.file_paths,
    )
    return {"results": results, "workspace": workspace_name}


# ═══════════════════════════════════════════════════════════════════
# Server startup
# ═══════════════════════════════════════════════════════════════════


async def run_stdio() -> None:
    """Run MCP server over stdio transport."""
    await _ensure_manager()
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )
    finally:
        if _manager is not None:
            await _manager.close()


async def run_streamable_http(host: str, port: int) -> None:
    """Run MCP server over streamable-http transport.

    Bearer auth is enforced when ``auth_mode`` is not ``"none"``.
    Both simple (``DLIGHTRAG_API_AUTH_TOKEN``) and JWT
    (``DLIGHTRAG_JWT_VERIFICATION_KEY``) modes are supported via the shared
    ``verify_bearer_token`` dispatcher.
    Without auth, the server runs open — caller is responsible for binding
    to loopback or trusted network only. We log a loud warning in that case.
    """
    from contextlib import asynccontextmanager

    import uvicorn
    from fastapi import HTTPException
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from mcp.server.transport_security import TransportSecuritySettings
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse
    from starlette.routing import Mount

    cfg = _get_config()
    has_auth = cfg.auth_mode != "none"

    if not has_auth:
        if host not in ("127.0.0.1", "localhost", "::1"):
            logger.warning(
                "=" * 72 + "\nMCP streamable-http on host=%s:%d WITHOUT auth (auth_mode='none').\n"
                "If this bind reaches a non-loopback network, ANY client can call\n"
                "ingest, delete_files, retrieve, answer against EVERY workspace.\n"
                "Safe configurations:\n"
                "  (a) Set DLIGHTRAG_AUTH_MODE=simple + DLIGHTRAG_API_AUTH_TOKEN\n"
                "      — bearer token guards MCP and REST (single secret).\n"
                "  (b) Set DLIGHTRAG_AUTH_MODE=jwt + DLIGHTRAG_JWT_VERIFICATION_KEY\n"
                "      — JWT bearer auth guards MCP and REST (external issuer).\n"
                "  (c) Bind to 127.0.0.1 (loopback only).\n"
                "  (d) Map host port to 127.0.0.1 only (compose: '127.0.0.1:8101:8101')\n"
                "      — safe even with container-internal 0.0.0.0.\n" + "=" * 72,
                host,
                port,
            )
    else:
        logger.info("MCP streamable-http on %s:%d with bearer auth enabled", host, port)

    class MCPPathMiddleware:
        """Normalize the public MCP endpoint before Starlette's mount redirect."""

        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            if scope.get("type") == "http" and scope.get("path") == "/mcp":
                scope = dict(scope)
                scope["path"] = "/mcp/"
                if scope.get("raw_path") == b"/mcp":
                    scope["raw_path"] = b"/mcp/"
            await self.app(scope, receive, send)

    class BearerAuthMiddleware(BaseHTTPMiddleware):
        """Enforce Bearer auth on every request to the MCP transport.

        Delegates to ``verify_bearer_token`` for auth-mode dispatch.
        No-op when ``auth_mode='none'`` (operator opted out).
        """

        async def dispatch(self, request, call_next):
            if not has_auth:
                with request_scope_context(RequestScope.anonymous()):
                    return await call_next(request)
            header = request.headers.get("Authorization", "")
            if not header.startswith("Bearer "):
                return JSONResponse(
                    {"error": "Missing or invalid Authorization header"}, status_code=401
                )
            try:
                from dlightrag.api.auth import verify_bearer_token

                user = verify_bearer_token(
                    header[7:],
                    cfg,
                    default_user_id=request.headers.get("X-User-Id", "anonymous"),
                )
            except HTTPException as exc:
                return JSONResponse({"error": exc.detail}, status_code=exc.status_code)
            with request_scope_context(RequestScope.from_user(user)):
                return await call_next(request)

    await _ensure_manager()

    transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=cfg.mcp_allowed_hosts,
        allowed_origins=cfg.mcp_allowed_origins,
    )
    session_manager = StreamableHTTPSessionManager(
        app=server,
        json_response=True,
        stateless=True,
        security_settings=transport_security,
    )

    @asynccontextmanager
    async def lifespan(app):
        async with session_manager.run():
            yield

    starlette_app = Starlette(
        routes=[Mount("/mcp", app=session_manager.handle_request)],
        middleware=[
            Middleware(MCPPathMiddleware),
            Middleware(BearerAuthMiddleware),
        ],
        lifespan=lifespan,
    )

    config = uvicorn.Config(
        starlette_app,
        host=host,
        port=port,
        log_level="info",
    )
    uv_server = uvicorn.Server(config)

    try:
        await uv_server.serve()
    finally:
        if _manager is not None:
            await _manager.close()


def main() -> None:
    """Entry point for dlightrag-mcp."""
    import argparse

    parser = argparse.ArgumentParser(description="dlightrag MCP server")
    parser.add_argument("--env-file", help="Path to .env configuration file")
    args = parser.parse_args()

    if args.env_file:
        config = load_config(args.env_file)
        set_config(config)
    else:
        config = _get_config()

    logging.basicConfig(level=getattr(logging, config.log_level.upper(), logging.INFO))

    if config.mcp_transport == "streamable-http":
        logger.info(f"Starting MCP server (streamable-http) on {config.mcp_host}:{config.mcp_port}")
        asyncio.run(run_streamable_http(config.mcp_host, config.mcp_port))
    else:
        logger.info("Starting MCP server (stdio)")
        asyncio.run(run_stdio())


__all__ = ["main", "mcp_app", "server"]
