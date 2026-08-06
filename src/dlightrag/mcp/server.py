# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""MCP server for agent integration (stdio + streamable-http).

Entry point: dlightrag-mcp
Primarily used by DeerFlow and other MCP-compatible agents for
retrieve() + lightweight ingest().
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

from mcp import MCPError
from mcp.server import MCPServer
from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.settings import AuthSettings
from mcp.server.context import CallNext, HandlerResult, ServerRequestContext
from mcp.server.mcpserver import Context
from mcp.types import CallToolResult, InputRequiredResult, TextContent, ToolAnnotations
from pydantic import Field
from starlette.middleware.cors import CORSMiddleware
from starlette.types import ASGIApp

import dlightrag
from dlightrag.access_control import AccessAction, AccessDeniedError, access_control_from_config
from dlightrag.config import DlightragConfig, get_config
from dlightrag.core import access as core_access
from dlightrag.core.answer.capability import answer_image_capability_summary
from dlightrag.core.client_contracts import (
    MAX_HISTORY_MESSAGES,
    QueryImage,
    SourceType,
)
from dlightrag.core.client_execution import execute_answer, execute_retrieve
from dlightrag.core.client_payloads import (
    answer_payload,
    retrieval_payload,
)
from dlightrag.core.client_requests import (
    ingest_spec_from_payload,
    managed_local_ingest_documents,
    managed_local_ingest_path,
)
from dlightrag.core.request.workspaces import (
    NoQueryableWorkspacesError,
)
from dlightrag.core.scope import RequestScope, current_request_scope, request_scope_context
from dlightrag.core.servicemanager import RAGServiceManager
from dlightrag.mcp.auth import DlightRAGTokenVerifier
from dlightrag.mcp.contracts import (
    AnswerInput,
    ConversationMessage,
    CreateWorkspaceInput,
    DeleteFilesInput,
    DeleteWorkspaceInput,
    IngestInput,
    IngestJobStatusInput,
    ListFilesInput,
    RetrieveInput,
)

logger = logging.getLogger(__name__)

QueryImagesParam = Annotated[
    list[QueryImage],
    Field(
        max_length=3,
        description="User-attached image URLs or data URI blocks (max 3)",
    ),
]
HistoryParam = Annotated[
    list[ConversationMessage] | None,
    Field(
        max_length=MAX_HISTORY_MESSAGES,
        description=(
            "Prior conversation turns as role/content messages. Caller-owned and "
            "stateless: re-send each request; never stored server-side."
        ),
    ),
]


class DlightRAGMCPServer(MCPServer):
    """MCPServer with DlightRAG's strict input and text-error contract."""

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        context: Context | None = None,
    ) -> CallToolResult | InputRequiredResult:
        try:
            await self._reject_unknown_arguments(name, arguments or {})
            return await super().call_tool(name, arguments or {}, context=context)
        except MCPError:
            raise
        except Exception as exc:
            # MCPServer wraps tool-body errors as ToolError(...) from the original, so
            # inspect __cause__ as well. Surface user-facing validation/authorization
            # messages; hide unexpected internals behind a generic message.
            user_error = exc if isinstance(exc, ValueError | PermissionError) else exc.__cause__
            if isinstance(user_error, ValueError | PermissionError):
                logger.warning("MCP tool '%s' rejected: %s", name, user_error)
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Error: {user_error}")],
                    is_error=True,
                )
            logger.exception("MCP tool '%s' failed", name)
            return CallToolResult(
                content=[TextContent(type="text", text="Error: internal tool failure")],
                is_error=True,
            )

    async def _reject_unknown_arguments(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> None:
        tool = next((tool for tool in await self.list_tools() if tool.name == name), None)
        if tool is None:
            raise ValueError(f"Unknown tool: {name}")
        allowed = set(tool.input_schema.get("properties", {}))
        unknown = sorted(set(arguments) - allowed)
        if unknown:
            raise ValueError(f"Unexpected argument(s) for {name}: {', '.join(unknown)}")


class DlightRAGRequestScopeMiddleware:
    """Project an MCP OAuth principal into DlightRAG's request scope."""

    async def __call__(
        self,
        ctx: ServerRequestContext[Any, Any],
        call_next: CallNext,
    ) -> HandlerResult:
        access_token = get_access_token()
        scope = current_request_scope()
        if access_token is not None:
            scope = RequestScope(
                user_id=access_token.subject or access_token.client_id,
                auth_mode=_get_config().auth_mode,
                claims=dict(access_token.claims or {}),
            )
        with request_scope_context(scope):
            return await call_next(ctx)


def _get_config() -> DlightragConfig:
    return get_config()


_manager: RAGServiceManager | None = None


async def _ensure_manager() -> RAGServiceManager:
    global _manager
    if _manager is None:
        _manager = await RAGServiceManager.acreate()
    return _manager


async def _close_manager() -> None:
    global _manager
    manager, _manager = _manager, None
    if manager is not None:
        await manager.aclose()


@asynccontextmanager
async def _mcp_lifespan(_: MCPServer[Any]) -> AsyncIterator[None]:
    await _ensure_manager()
    try:
        yield
    finally:
        await _close_manager()


def _http_auth(
    config: DlightragConfig,
) -> tuple[AuthSettings | None, DlightRAGTokenVerifier | None]:
    if config.mcp_transport != "streamable-http" or config.auth_mode == "none":
        return None, None
    resource = config.mcp_resource_server_url
    auth = AuthSettings.model_validate(
        {
            "issuer_url": config.jwt_issuer or "http://localhost",
            "resource_server_url": resource,
        }
    )
    return auth, DlightRAGTokenVerifier(config, resource=resource)


_auth, _token_verifier = _http_auth(_get_config())
mcp_app = DlightRAGMCPServer(
    "dlightrag",
    version=dlightrag.__version__,
    log_level="INFO",
    warn_on_duplicate_tools=True,
    lifespan=_mcp_lifespan,
    middleware=[DlightRAGRequestScopeMiddleware()],
    auth=_auth,
    token_verifier=_token_verifier,
)


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


async def _filter_workspace_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return await core_access.filter_workspace_records(
        access_control_from_config(_get_config()),
        current_request_scope(),
        AccessAction.WORKSPACE_QUERY,
        records,
    )


async def _resolve_authorized_query_workspaces(
    manager: RAGServiceManager,
    *,
    workspaces: list[str] | None,
    all_workspaces: bool,
) -> list[str]:
    """Resolve MCP query targets after applying the current request ACL."""
    try:
        return await core_access.resolve_authorized_query_workspaces(
            access_control_from_config(_get_config()),
            current_request_scope(),
            manager,
            default_workspace=_get_config().workspace,
            workspaces=workspaces,
            all_workspaces=all_workspaces,
        )
    except NoQueryableWorkspacesError:
        raise PermissionError("No workspaces are available for query") from None
    except AccessDeniedError as exc:
        raise ValueError(str(exc)) from None


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
    manager = await _ensure_manager()
    resolved_workspaces = await _resolve_authorized_query_workspaces(
        manager,
        workspaces=args.workspaces,
        all_workspaces=args.all_workspaces,
    )
    scope = current_request_scope().for_workspaces(resolved_workspaces)
    result = await execute_retrieve(
        manager=manager,
        payload=args,
        resolved_workspaces=resolved_workspaces,
        scope=scope,
    )
    return retrieval_payload(result)


@mcp_app.tool(
    name="answer",
    description=(
        "Ask a question and get an LLM-generated answer backed by retrieved context "
        "from the default or selected workspaces in the knowledge base."
    ),
    annotations=ToolAnnotations(read_only_hint=True),
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
    query_images: QueryImagesParam = Field(default_factory=list),
    semantic_highlights: Annotated[
        bool,
        Field(default=False, description="Include semantic highlight phrases in cited sources."),
    ] = False,
    history: HistoryParam = None,
) -> dict[str, Any]:
    args = AnswerInput.model_validate(locals())
    manager = await _ensure_manager()
    resolved_workspaces = await _resolve_authorized_query_workspaces(
        manager,
        workspaces=args.workspaces,
        all_workspaces=args.all_workspaces,
    )
    scope = current_request_scope().for_workspaces(resolved_workspaces)
    result = await execute_answer(
        manager=manager,
        payload=args,
        resolved_workspaces=resolved_workspaces,
        scope=scope,
    )
    return answer_payload(result)


@mcp_app.tool(
    name="list_workspaces",
    description=(
        "List workspaces visible to the current user. Returns workspace ids plus "
        "records containing workspace, display_name, embedding_model, created_at, "
        "and updated_at. Use display_name as the user-facing workspace label."
    ),
    annotations=ToolAnnotations(read_only_hint=True),
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
    name="get_capabilities",
    description=(
        "Report deployment capabilities agents should honor. Returns "
        "answer_image_capability with status (supported/unsupported/unknown), "
        "effective_max_images (max images the answer model accepts; 0 means send none), "
        "configured_ceiling, and model — query images reach the answer model only when "
        "status is 'supported'."
    ),
    annotations=ToolAnnotations(read_only_hint=True),
)
async def get_capabilities_tool() -> dict[str, Any]:
    manager = await _ensure_manager()
    return {
        "answer_image_capability": answer_image_capability_summary(manager.answer_image_capability),
    }


@mcp_app.tool(
    name="create_workspace",
    description=(
        "Create and register an empty DlightRAG workspace. Optional display_name is "
        "the user-facing label; response returns normalized workspace id, display_name, "
        "and created."
    ),
    annotations=ToolAnnotations(
        read_only_hint=False,
        destructive_hint=False,
        idempotent_hint=False,
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
    annotations=ToolAnnotations(read_only_hint=False, destructive_hint=True),
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
        "a workspace. URL fetch endpoints, stable source identity, and durable download "
        "locators are separate; signed fetches require retention or a queryless locator. "
        "Response includes job_id, status, and workspace."
    ),
    annotations=ToolAnnotations(
        read_only_hint=False,
        destructive_hint=False,
        idempotent_hint=False,
    ),
)
async def ingest_tool(
    source_type: Annotated[SourceType, Field(description="Type of data source")],
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
        Field(
            default=None,
            description=(
                "Public or signed HTTPS fetch URL. A signed/query-bearing URL requires "
                "retention or a separate queryless download_uri."
            ),
        ),
    ] = None,
    urls: Annotated[
        list[str] | None,
        Field(
            default=None,
            description=(
                "Public or signed HTTPS fetch URLs. Signed/query-bearing entries require "
                "retention or matching queryless download_uris."
            ),
        ),
    ] = None,
    filename: Annotated[
        str | None,
        Field(default=None, description="Parser filename for a single URL."),
    ] = None,
    source_uri: Annotated[
        str | None,
        Field(
            default=None,
            description="Stable provenance identity for one URL; not a download address.",
        ),
    ] = None,
    source_uris: Annotated[
        list[str] | None,
        Field(
            default=None,
            description="Stable provenance identities for a URL batch; not download addresses.",
        ),
    ] = None,
    download_uri: Annotated[
        str | None,
        Field(
            default=None,
            description=(
                "Durable S3, Azure, or credential-free queryless public HTTPS locator "
                "for one fetched URL."
            ),
        ),
    ] = None,
    download_uris: Annotated[
        list[str] | None,
        Field(
            default=None,
            description=("Durable S3, Azure, or queryless public HTTPS locators for a URL batch."),
        ),
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
    retain_source_file: Annotated[
        bool | None,
        Field(
            default=None,
            description=(
                "Keep fetched bytes as the download source. Signed URL fetches require this "
                "unless a separate queryless durable locator is supplied."
            ),
        ),
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
    annotations=ToolAnnotations(read_only_hint=True),
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
    annotations=ToolAnnotations(read_only_hint=True),
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
    annotations=ToolAnnotations(read_only_hint=False, destructive_hint=True),
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


def create_mcp_http_app() -> ASGIApp:
    """Create the production Streamable HTTP ASGI app."""
    from mcp.server.transport_security import TransportSecuritySettings

    config = _get_config()
    transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=config.mcp_allowed_hosts,
        allowed_origins=config.mcp_allowed_origins,
    )
    http_app: ASGIApp = mcp_app.streamable_http_app(
        streamable_http_path="/mcp",
        json_response=True,
        stateless_http=True,
        transport_security=transport_security,
        host=config.mcp_host,
    )
    return CORSMiddleware(
        http_app,
        allow_origins=config.cors_allow_origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "Mcp-Method",
            "Mcp-Name",
            "Mcp-Protocol-Version",
        ],
    )


async def run_stdio() -> None:
    """Run MCP server over stdio transport."""
    await mcp_app.run_stdio_async()


async def run_streamable_http() -> None:
    """Run the MCP 2.0 Streamable HTTP server."""
    import uvicorn

    config = _get_config()
    uvicorn_config = uvicorn.Config(
        create_mcp_http_app(),
        host=config.mcp_host,
        port=config.mcp_port,
        log_level="info",
    )
    await uvicorn.Server(uvicorn_config).serve()


def run() -> None:
    """Run the configured MCP transport."""
    config = _get_config()
    logging.basicConfig(level=getattr(logging, config.log_level.upper(), logging.INFO))

    if config.mcp_transport == "streamable-http":
        logger.info(
            "Starting MCP server (streamable-http) on %s:%d",
            config.mcp_host,
            config.mcp_port,
        )
        asyncio.run(run_streamable_http())
    else:
        logger.info("Starting MCP server (stdio)")
        asyncio.run(run_stdio())


__all__ = ["create_mcp_http_app", "mcp_app", "run"]
