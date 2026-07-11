# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""MCP server for agent integration (stdio + streamable-http).

Entry point: dlightrag-mcp
Primarily used by DeerFlow and other MCP-compatible agents for
retrieve() + lightweight ingest().
"""

import asyncio
import logging
from collections.abc import Mapping, Sequence
from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP
from mcp.server.stdio import stdio_server
from mcp.types import ContentBlock, TextContent
from pydantic import Field

import dlightrag
from dlightrag.access_control import AccessAction, AccessDeniedError, access_control_from_config
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
    ingest_spec_from_payload,
    managed_local_ingest_documents,
    managed_local_ingest_path,
    query_kwargs_from_payload,
)
from dlightrag.core.query_workspaces import (
    NoQueryableWorkspacesError,
    resolve_query_workspaces,
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
            # FastMCP wraps tool-body errors as ToolError(...) from the original, so
            # inspect __cause__ as well. Surface user-facing validation/authorization
            # messages; hide unexpected internals behind a generic message.
            user_error = exc if isinstance(exc, ValueError | PermissionError) else exc.__cause__
            if isinstance(user_error, ValueError | PermissionError):
                logger.warning("MCP tool '%s' rejected: %s", name, user_error)
                return [TextContent(type="text", text=f"Error: {user_error}")]
            logger.exception("MCP tool '%s' failed", name)
            return [TextContent(type="text", text="Error: internal tool failure")]

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
        _manager = await RAGServiceManager.acreate()
    return _manager


def _normalize_workspace_argument(args: CreateWorkspaceInput) -> tuple[str, str]:
    from dlightrag.utils import normalize_workspace, validate_workspace_name

    label = validate_workspace_name(args.workspace)
    display_name = validate_workspace_name(args.display_name or label)
    return normalize_workspace(label), display_name


async def _enforce_access(action: str, workspace: str | None = None) -> None:
    try:
        await access_control_from_config(_get_config()).check(
            current_request_scope(),
            action,
            workspace=workspace,
        )
    except AccessDeniedError as exc:
        raise ValueError(str(exc)) from None


async def _enforce_workspaces_access(
    action: str,
    workspaces: list[str] | None,
) -> None:
    from dlightrag.utils import normalize_workspace

    for workspace in workspaces or [_get_config().workspace]:
        await _enforce_access(action, normalize_workspace(workspace))


async def _filter_workspace_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    workspaces = [str(row["workspace"]) for row in records]
    allowed = set(
        await access_control_from_config(_get_config()).filter_workspaces(
            current_request_scope(),
            AccessAction.WORKSPACE_QUERY,
            workspaces,
        )
    )
    return [row for row in records if str(row["workspace"]) in allowed]


async def _resolve_authorized_query_workspaces(
    manager: RAGServiceManager,
    *,
    workspaces: list[str] | None,
    all_workspaces: bool,
) -> list[str]:
    """Resolve MCP query targets after applying the current request ACL."""
    available: list[str] | None = None
    if all_workspaces:
        records = await manager.alist_workspace_records()
        visible = await _filter_workspace_records(records)
        available = [str(row["workspace"]) for row in visible]

    try:
        resolved = resolve_query_workspaces(
            default_workspace=_get_config().workspace,
            workspaces=workspaces,
            all_workspaces=all_workspaces,
            available_workspaces=available,
        )
    except NoQueryableWorkspacesError:
        raise PermissionError("No workspaces are available for query") from None

    if not all_workspaces:
        await _enforce_workspaces_access(AccessAction.WORKSPACE_QUERY, resolved)
    return resolved


@mcp_app.tool(
    name="retrieve",
    description=(
        "Query the RAG knowledge base for relevant information. Supports structured "
        "metadata filters and default or selected workspaces for precise document lookups."
    ),
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
    session_id: Annotated[
        str | None,
        Field(default=None, description="Session id for image memory."),
    ] = None,
    referenced_image_ids: ReferencedImageIdsParam = Field(default_factory=list),
) -> dict[str, Any]:
    args = RetrieveInput.model_validate(locals())
    manager = await _ensure_manager()
    resolved_workspaces = await _resolve_authorized_query_workspaces(
        manager,
        workspaces=args.workspaces,
        all_workspaces=args.all_workspaces,
    )
    scope = current_request_scope().for_workspaces(resolved_workspaces)
    result = await manager.aretrieve(
        args.query,
        workspaces=resolved_workspaces,
        top_k=args.top_k,
        chunk_top_k=args.chunk_top_k,
        scope=scope,
        **query_kwargs_from_payload(args, include_multimodal_content=False),
    )
    return retrieval_payload(result)


@mcp_app.tool(
    name="answer",
    description=(
        "Ask a question and get an LLM-generated answer backed by retrieved context "
        "from the default or selected workspaces in the knowledge base."
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
    semantic_highlights: Annotated[
        bool,
        Field(default=False, description="Include semantic highlight phrases in cited sources."),
    ] = False,
) -> dict[str, Any]:
    args = AnswerInput.model_validate(locals())
    manager = await _ensure_manager()
    resolved_workspaces = await _resolve_authorized_query_workspaces(
        manager,
        workspaces=args.workspaces,
        all_workspaces=args.all_workspaces,
    )
    scope = current_request_scope().for_workspaces(resolved_workspaces)
    result = await manager.aanswer(
        args.query,
        conversation_history=dump_optional_list(args.conversation_history),
        workspaces=resolved_workspaces,
        top_k=args.top_k,
        chunk_top_k=args.chunk_top_k,
        answer_context_top_k=args.answer_context_top_k,
        semantic_highlights=args.semantic_highlights,
        scope=scope,
        **query_kwargs_from_payload(args, include_multimodal_content=False),
    )
    return answer_payload(result)


@mcp_app.tool(
    name="list_workspaces",
    description=(
        "List workspaces visible to the current user. Returns workspace ids plus "
        "records containing workspace, display_name, embedding_model, created_at, "
        "and updated_at. Use display_name as the user-facing workspace label."
    ),
)
async def list_workspaces_tool() -> dict[str, Any]:
    manager = await _ensure_manager()
    records = await manager.alist_workspace_records()
    records = await _filter_workspace_records(records)
    return {
        "workspaces": [row["workspace"] for row in records],
        "records": records,
    }


@mcp_app.tool(
    name="create_workspace",
    description=(
        "Create and register an empty DlightRAG workspace. Optional display_name is "
        "the user-facing label; response returns normalized workspace id, display_name, "
        "and created."
    ),
)
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
    await _enforce_access(AccessAction.WORKSPACE_CREATE, normalized_workspace)
    existing = await manager.alist_workspaces()
    if normalized_workspace in existing:
        raise ValueError(f"Workspace '{normalized_display_name}' already exists")
    await manager.acreate_workspace(normalized_workspace, display_name=normalized_display_name)
    return {
        "workspace": normalized_workspace,
        "display_name": normalized_display_name,
        "created": True,
    }


@mcp_app.tool(
    name="delete_workspace",
    description=(
        "Delete/reset one DlightRAG workspace and remove its registry row. Supports "
        "dry_run and keep_files; response returns normalized workspace id, deleted, "
        "and result."
    ),
)
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
    await _enforce_access(AccessAction.WORKSPACE_DELETE, normalized_workspace)
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


@mcp_app.tool(
    name="ingest",
    description=(
        "Start a durable ingest job for local, URL, Azure Blob, or S3 documents into "
        "a workspace. Response includes job_id, status, and workspace."
    ),
)
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
    s3_region: Annotated[
        str | None,
        Field(default=None, description="S3 region name."),
    ] = None,
    s3_key: Annotated[
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
    documents: Annotated[
        list[dict[str, Any]] | None,
        Field(
            default=None,
            description=(
                "Explicit document manifest. Local documents use path, S3/Azure use key, "
                "URL documents use url. Document metadata overlays request metadata."
            ),
        ),
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
    retain_source_file: Annotated[
        bool | None,
        Field(default=None, description="Override remote source file retention for this ingest."),
    ] = None,
) -> dict[str, Any]:
    args = IngestInput.model_validate(locals())
    manager = await _ensure_manager()
    workspace_name = args.workspace or _get_config().workspace
    from dlightrag.utils import normalize_workspace

    workspace_name = normalize_workspace(workspace_name)
    await _enforce_access(AccessAction.WORKSPACE_INGEST, workspace_name)
    ingest_spec = ingest_spec_from_payload(args)
    if args.source_type == "local":
        path = managed_local_ingest_path(
            source_type=args.source_type,
            path=ingest_spec.path,
            input_dir=_get_config().input_dir_path,
            workspace=workspace_name,
        )
        managed_documents = managed_local_ingest_documents(
            source_type=args.source_type,
            documents=ingest_spec.documents,
            input_dir=_get_config().input_dir_path,
            workspace=workspace_name,
        )
        ingest_spec = ingest_spec.model_copy(update={"path": path, "documents": managed_documents})
    return await manager.astart_ingest_job(workspace_name, ingest_spec)


@mcp_app.tool(
    name="get_ingest_job",
    description=(
        "Return status for an ingest job_id returned by ingest, including the job workspace "
        "when available."
    ),
)
async def get_ingest_job_tool(
    job_id: Annotated[str, Field(description="Ingest job id returned by the ingest tool.")],
) -> dict[str, Any]:
    args = IngestJobStatusInput.model_validate(locals())
    manager = await _ensure_manager()
    if not args.job_id:
        raise ValueError("job_id is required")
    result = await manager.aget_ingest_job(args.job_id)
    if result is None:
        raise ValueError(f"Ingest job not found: {args.job_id}")
    workspace = result.get("workspace")
    await _enforce_access(AccessAction.JOB_READ, str(workspace) if workspace else None)
    return result


@mcp_app.tool(
    name="list_files",
    description=(
        "List documents ingested in one workspace. Response returns files, count, and workspace."
    ),
)
async def list_files_tool(
    workspace: Annotated[
        str | None,
        Field(default=None, description="Workspace to list files from. Omit for default."),
    ] = None,
) -> dict[str, Any]:
    args = ListFilesInput.model_validate(locals())
    manager = await _ensure_manager()
    workspace_name = args.workspace or _get_config().workspace
    await _enforce_access(AccessAction.WORKSPACE_LIST_FILES, workspace_name)
    files = await manager.alist_ingested_files(workspace_name)
    return {"files": files, "count": len(files), "workspace": workspace_name}


@mcp_app.tool(
    name="delete_files",
    description=(
        "Delete or dry_run matching documents from one workspace by filename or file_path. "
        "Response returns results and workspace."
    ),
)
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
    dry_run: Annotated[
        bool,
        Field(default=False, description="Report matching documents without deleting them."),
    ] = False,
) -> dict[str, Any]:
    args = DeleteFilesInput.model_validate(locals())
    manager = await _ensure_manager()
    workspace_name = args.workspace or _get_config().workspace
    await _enforce_access(AccessAction.WORKSPACE_DELETE_FILES, workspace_name)
    results = await manager.adelete_files(
        workspace_name,
        filenames=args.filenames,
        file_paths=args.file_paths,
        dry_run=args.dry_run,
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
            await _manager.aclose()


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
        enable_dns_rebinding_protection=cfg.mcp_dns_rebinding_protection,
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
            await _manager.aclose()


def main() -> None:
    """Entry point for dlightrag-mcp."""
    import argparse

    parser = argparse.ArgumentParser(
        description="dlightrag MCP server",
        suggest_on_error=True,
    )
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
