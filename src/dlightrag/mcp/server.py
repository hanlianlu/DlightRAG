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

from dlightrag_rag.contracts import SourceType
from dlightrag_rag.pool import WorkspaceUnavailableError
from dlightrag_rag.retrieval import MetadataFilter
from dlightrag_rag.workspaces import normalize_workspace, normalize_workspace_ids
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
from dlightrag.access import (
    AccessAction,
    AccessDeniedError,
    AccessGate,
    NoQueryableWorkspacesError,
    RequestScope,
    WorkspaceRecord,
    access_control_from_settings,
    current_request_scope,
    owner_id_from_principal,
    request_scope_context,
)
from dlightrag.answer.capability import answer_image_capability_summary
from dlightrag.answer.client_contracts import (
    MAX_HISTORY_MESSAGES,
    AnswerAttachmentLink,
    QueryImage,
    conversation_history_as_dicts,
)
from dlightrag.answer.errors import AnswerInputError, InvalidToolConfigurationError
from dlightrag.answer.resources.links import answer_link_resources
from dlightrag.answer.runs.results import project_answer_result
from dlightrag.application import Application
from dlightrag.config import DlightragConfig, get_config
from dlightrag.mcp.auth import DlightRAGTokenVerifier
from dlightrag.mcp.contracts import (
    AnswerInput,
    AnswerRunInput,
    ConversationMessage,
    CreateWorkspaceInput,
    DeleteFilesInput,
    DeleteWorkspaceInput,
    IngestInput,
    IngestJobStatusInput,
    ListFilesInput,
    RetrieveInput,
)
from dlightrag.model_settings import access_settings
from dlightrag.runtime import AnswerRunRecord, IdempotencyKeyConflict
from dlightrag.services.answers import AnswerRequest as ServiceAnswerRequest
from dlightrag.services.answers import AnswerRuntimeUnavailableError
from dlightrag.services.corpora import (
    ingest_spec_from_payload,
    managed_local_ingest_documents,
    managed_local_ingest_path,
)
from dlightrag.services.retrieval import (
    RetrievalTimeoutError,
    RetrieveProjection,
)
from dlightrag.services.retrieval import (
    RetrieveRequest as ServiceRequest,
)

logger = logging.getLogger(__name__)

QueryImagesParam = Annotated[
    list[QueryImage],
    Field(
        max_length=3,
        description="User-attached image URLs or data URI blocks (max 3)",
    ),
]
AttachmentsParam = Annotated[
    list[AnswerAttachmentLink],
    Field(
        description=(
            "Answer attachments as HTTPS link descriptors ({url, filename?}). MCP accepts "
            "links only; local paths, raw bytes, and base64 are rejected."
        ),
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
IdempotencyKeyParam = Annotated[
    str | None,
    Field(
        default=None,
        max_length=255,
        description=(
            "Optional replay key scoped to the calling identity. Repeating it with the "
            "same request returns the same run instead of starting a second one; "
            "repeating it with a different request is rejected."
        ),
    ),
]


def _owner_id() -> str:
    """Project the current MCP principal into the shared run owner namespace."""
    scope = current_request_scope()
    return owner_id_from_principal(
        auth_mode=scope.auth_mode,
        user_id=scope.user_id,
        issuer=str(scope.claims.get("iss") or "") or None,
    )


def _run_descriptor(record: AnswerRunRecord) -> dict[str, Any]:
    """Project one run's identity and lifecycle state for an MCP caller."""
    return {
        "run_id": record.run_id,
        "status": record.status,
        "cancel_requested": record.cancel_requested,
        "created_at": record.created_at.isoformat(),
    }


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
            # inspect __cause__ as well. Server misconfiguration is a server failure;
            # user-facing validation/authorization messages are surfaced as rejections;
            # unexpected internals hide behind a generic message.
            surfaced = (
                ValueError,
                PermissionError,
                InvalidToolConfigurationError,
                RetrievalTimeoutError,
                WorkspaceUnavailableError,
                AnswerRuntimeUnavailableError,
            )
            inner = exc if isinstance(exc, surfaced) else exc.__cause__
            if isinstance(inner, InvalidToolConfigurationError):
                logger.exception("MCP tool '%s' failed: %s", name, inner)
                text = f"Error [{inner.error_kind}]: {inner.public_message}"
            elif isinstance(
                inner,
                ValueError
                | PermissionError
                | RetrievalTimeoutError
                | WorkspaceUnavailableError
                | AnswerRuntimeUnavailableError,
            ):
                logger.warning("MCP tool '%s' rejected: %s", name, inner)
                text = (
                    f"Error [{inner.error_kind}]: {inner.public_message}"
                    if isinstance(inner, AnswerInputError)
                    else f"Error: {inner}"
                )
            else:
                logger.exception("MCP tool '%s' failed", name)
                text = "Error: internal tool failure"
            return CallToolResult(
                content=[TextContent(type="text", text=text)],
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
            application = await _ensure_application()
            scope = RequestScope(
                user_id=access_token.subject or access_token.client_id,
                auth_mode=application.config.auth_mode,
                claims=dict(access_token.claims or {}),
            )
        with request_scope_context(scope):
            return await call_next(ctx)


def _get_config() -> DlightragConfig:
    return get_config()


_application: Application | None = None


async def _ensure_application() -> Application:
    global _application
    if _application is None:
        _application = await Application.acreate()
    return _application


async def _close_application() -> None:
    global _application
    application, _application = _application, None
    if application is not None:
        await application.aclose()


@asynccontextmanager
async def _mcp_lifespan(_: MCPServer[Any]) -> AsyncIterator[None]:
    await _ensure_application()
    try:
        yield
    finally:
        await _close_application()


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
    from dlightrag.services.corpora import validate_workspace_name

    label = validate_workspace_name(args.workspace)
    display_name = validate_workspace_name(args.display_name or label)
    return normalize_workspace(label), display_name


async def _enforce_access(
    action: str,
    workspace: str | None = None,
    *,
    application: Application,
) -> None:
    try:
        await _access_gate(application).check(action, workspace=workspace)
    except AccessDeniedError as exc:
        raise ValueError(str(exc)) from None


def _access_gate(application: Application) -> AccessGate:
    return AccessGate(
        access_control_from_settings(access_settings(application.config)),
        current_request_scope(),
    )


async def _filter_workspace_records(
    records: list[WorkspaceRecord],
    *,
    application: Application,
) -> list[WorkspaceRecord]:
    return await _access_gate(application).filter_workspace_records(
        AccessAction.WORKSPACE_QUERY, records
    )


async def _authorized_workspace_names(
    action: str,
    workspaces: list[str],
    *,
    application: Application,
) -> set[str]:
    return await _access_gate(application).authorized_workspace_ids(action, workspaces)


async def _resolve_authorized_query_workspaces(
    application: Application,
    *,
    workspaces: list[str] | None,
    all_workspaces: bool,
) -> list[str]:
    """Resolve MCP query targets after applying the current request ACL."""
    try:
        return await _access_gate(application).resolve_query_workspaces(
            application.corpora,
            default_workspace=normalize_workspace(application.config.workspace),
            workspaces=normalize_workspace_ids(workspaces) if workspaces is not None else None,
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
    application = await _ensure_application()
    resolved_workspaces = await _resolve_authorized_query_workspaces(
        application,
        workspaces=args.workspaces,
        all_workspaces=args.all_workspaces,
    )
    visual_workspaces = await _authorized_workspace_names(
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


@mcp_app.tool(
    name="answer",
    description=(
        "Start an LLM-generated answer backed by retrieved context from the default or "
        "selected workspaces. Returns immediately with a run_id and its initial status; "
        "the answer itself is NOT returned here. Poll get_answer_run with that run_id "
        "until status is succeeded, failed, or cancelled, and call cancel_answer_run to "
        "stop a run you no longer need. A run survives this call, this connection, and a "
        "server restart."
    ),
    annotations=ToolAnnotations(read_only_hint=False, idempotent_hint=False),
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
    attachments: AttachmentsParam = Field(default_factory=list),
    semantic_highlights: Annotated[
        bool,
        Field(default=False, description="Include semantic highlight phrases in cited sources."),
    ] = False,
    history: HistoryParam = None,
    idempotency_key: IdempotencyKeyParam = None,
) -> dict[str, Any]:
    args = AnswerInput.model_validate(locals())
    application = await _ensure_application()
    max_attachments = application.config.answer.max_attachments
    if len(args.attachments) > max_attachments:
        raise ValueError(f"Too many attachments; at most {max_attachments} are allowed")
    resolved_workspaces = await _resolve_authorized_query_workspaces(
        application,
        workspaces=args.workspaces,
        all_workspaces=args.all_workspaces,
    )
    try:
        creation = await application.answers.create(
            request=ServiceAnswerRequest(
                query=args.query,
                workspaces=tuple(resolved_workspaces),
                top_k=args.top_k,
                chunk_top_k=args.chunk_top_k,
                filters=MetadataFilter.model_validate(args.filters) if args.filters else None,
                semantic_highlights=args.semantic_highlights,
                history=tuple(conversation_history_as_dicts(args.history) or ()),
                resources=tuple(answer_link_resources(args.attachments)),
            ),
            idempotency_key=args.idempotency_key,
            owner_id=_owner_id(),
        )
    except IdempotencyKeyConflict:
        raise ValueError(
            "idempotency_key was already used for a different answer request"
        ) from None
    return _run_descriptor(creation.run)


@mcp_app.tool(
    name="get_answer_run",
    description=(
        "Return the current state of an answer run started by the answer tool. status is "
        "queued, running, succeeded, failed, or cancelled; cancel_requested reports whether "
        "cancellation was asked for. A succeeded run carries result with answer, sources, "
        "contexts, references, answer_images, and image_descriptions. A failed run carries "
        "error_kind and error_message. An unknown run id, or one owned by another caller, "
        "is reported as not found."
    ),
    annotations=ToolAnnotations(read_only_hint=True),
)
async def get_answer_run_tool(
    run_id: Annotated[str, Field(description="Run id returned by the answer tool.")],
) -> dict[str, Any]:
    args = AnswerRunInput.model_validate(locals())
    application = await _ensure_application()
    record = await application.answers.get(owner_id=_owner_id(), run_id=args.run_id)
    if record is None:
        raise ValueError(f"Answer run not found: {args.run_id}")
    result: dict[str, Any] | None = None
    if record.result is not None:
        result = project_answer_result(
            record.result,
            visual_workspaces=await _authorized_workspace_names(
                AccessAction.WORKSPACE_READ_VISUAL_ASSET,
                [str(value) for value in (record.prepared_input or {}).get("workspaces") or ()],
                application=application,
            ),
        )
    return {
        **_run_descriptor(record),
        "phase": record.phase,
        "durable_progress_version": record.durable_progress_version,
        "result": result,
        "error_kind": record.error_kind,
        "error_message": record.error_message,
        "finished_at": record.finished_at.isoformat() if record.finished_at else None,
    }


@mcp_app.tool(
    name="cancel_answer_run",
    description=(
        "Request cancellation of an answer run started by the answer tool and return its "
        "state. A queued run is cancelled immediately; a running one is cancelled once its "
        "worker observes the request, so status may still be running with cancel_requested "
        "true. Cancelling an already finished run changes nothing. An unknown run id, or "
        "one owned by another caller, is reported as not found."
    ),
    annotations=ToolAnnotations(read_only_hint=False, idempotent_hint=True),
)
async def cancel_answer_run_tool(
    run_id: Annotated[str, Field(description="Run id returned by the answer tool.")],
) -> dict[str, Any]:
    args = AnswerRunInput.model_validate(locals())
    application = await _ensure_application()
    outcome = await application.answers.cancel(owner_id=_owner_id(), run_id=args.run_id)
    if outcome.run is None:
        raise ValueError(f"Answer run not found: {args.run_id}")
    return _run_descriptor(outcome.run)


@mcp_app.tool(
    name="list_answer_runs",
    description="List this caller's durable answer runs, oldest first.",
    annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True),
)
async def list_answer_runs_tool(
    after: Annotated[str | None, Field(default=None, description="Cursor run id")] = None,
    limit: Annotated[int, Field(default=50, description="Page size")] = 50,
) -> dict[str, Any]:
    application = await _ensure_application()
    rows = await application.answers.list(owner_id=_owner_id(), after_run_id=after, limit=limit)
    return {"runs": [_run_descriptor(record) for record in rows]}


@mcp_app.tool(
    name="list_answer_artifacts",
    description="List artifact descriptors for one owned answer run.",
    annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True),
)
async def list_answer_artifacts_tool(
    run_id: Annotated[str, Field(description="Run id")],
) -> dict[str, Any]:
    application = await _ensure_application()
    items = await application.answers.list_artifacts(owner_id=_owner_id(), run_id=run_id)
    return {
        "artifacts": [
            {
                "resource_id": item.resource_id,
                "kind": item.reference_kind,
                "filename": item.filename,
                "media_type": item.mime_type,
            }
            for item in items
        ]
    }


@mcp_app.tool(
    name="read_answer_artifact",
    description="Read up to 1 MiB of one artifact as base64, returning the next offset.",
    annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True),
)
async def read_answer_artifact_tool(
    run_id: Annotated[str, Field(description="Run id")],
    resource_id: Annotated[str, Field(description="Artifact resource id")],
    offset: Annotated[int, Field(default=0, description="Byte offset")] = 0,
    length: Annotated[int, Field(default=1_048_576, description="Max bytes")] = 1_048_576,
) -> dict[str, Any]:
    import base64

    application = await _ensure_application()
    chunk = await application.answers.read_artifact(
        owner_id=_owner_id(),
        run_id=run_id,
        resource_id=resource_id,
        offset=max(0, offset),
        length=min(max(0, length), 1_048_576),
    )
    if chunk is None:
        raise ValueError("artifact not found")
    return {
        "data": base64.b64encode(chunk).decode("ascii"),
        "offset": offset,
        "next_offset": offset + len(chunk),
        "bytes": len(chunk),
    }


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
    application = await _ensure_application()
    records = await application.corpora.alist_workspace_records()
    records = await _filter_workspace_records(records, application=application)
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
    application = await _ensure_application()
    capabilities = await application.answers.capabilities()
    return {
        "answer_image_capability": answer_image_capability_summary(capabilities.answer),
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
    application = await _ensure_application()
    normalized_workspace, normalized_display_name = _normalize_workspace_argument(args)
    await _enforce_access(
        AccessAction.WORKSPACE_CREATE,
        normalized_workspace,
        application=application,
    )
    existing = await application.corpora.list_workspaces()
    if normalized_workspace in existing:
        raise ValueError(f"Workspace '{normalized_display_name}' already exists")
    await application.corpora.create_workspace(
        normalized_workspace,
        display_name=normalized_display_name,
    )
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
    application = await _ensure_application()
    from dlightrag.services.corpora import validate_workspace_name

    label = validate_workspace_name(args.workspace)
    normalized_workspace = normalize_workspace(label)
    await _enforce_access(
        AccessAction.WORKSPACE_DELETE,
        normalized_workspace,
        application=application,
    )
    result = await application.corpora.reset(
        workspace_ids=(normalized_workspace,),
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
    application = await _ensure_application()
    workspace_name = args.workspace or application.config.workspace
    workspace_name = normalize_workspace(workspace_name)
    await _enforce_access(
        AccessAction.WORKSPACE_INGEST,
        workspace_name,
        application=application,
    )
    ingest_spec = ingest_spec_from_payload(args)
    if args.source_type == "local":
        path = managed_local_ingest_path(
            source_type=args.source_type,
            path=ingest_spec.path,
            input_dir=application.config.input_dir_path,
            workspace=workspace_name,
        )
        managed_documents = managed_local_ingest_documents(
            source_type=args.source_type,
            documents=ingest_spec.documents,
            input_dir=application.config.input_dir_path,
            workspace=workspace_name,
        )
        ingest_spec = ingest_spec.model_copy(update={"path": path, "documents": managed_documents})
    return await application.corpora.start_ingest_job(workspace_name, ingest_spec)


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
    application = await _ensure_application()
    if not args.job_id:
        raise ValueError("job_id is required")
    result = await application.corpora.get_ingest_job(args.job_id)
    if result is None:
        raise ValueError(f"Ingest job not found: {args.job_id}")
    workspace = result.get("workspace")
    workspace_id = normalize_workspace(str(workspace)) if workspace else None
    await _enforce_access(AccessAction.JOB_READ, workspace_id, application=application)
    return result


@mcp_app.tool(
    name="cancel_ingest_job",
    description=(
        "Stop a running ingest job. Documents already ingested are kept; "
        "unfinished ones end up failed and can be retried."
    ),
    annotations=ToolAnnotations(read_only_hint=False, idempotent_hint=True),
)
async def cancel_ingest_job_tool(
    job_id: Annotated[str, Field(description="Ingest job id returned by the ingest tool.")],
) -> dict[str, Any]:
    args = IngestJobStatusInput.model_validate(locals())
    application = await _ensure_application()
    if not args.job_id:
        raise ValueError("job_id is required")
    result = await application.corpora.get_ingest_job(args.job_id)
    if result is None:
        raise ValueError(f"Ingest job not found: {args.job_id}")
    workspace = result.get("workspace")
    workspace_id = normalize_workspace(str(workspace)) if workspace else None
    await _enforce_access(AccessAction.JOB_CANCEL, workspace_id, application=application)
    cancelled = await application.corpora.cancel_ingest_job(args.job_id)
    return cancelled if cancelled is not None else result


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
    application = await _ensure_application()
    workspace_name = normalize_workspace(args.workspace or application.config.workspace)
    await _enforce_access(
        AccessAction.WORKSPACE_LIST_FILES,
        workspace_name,
        application=application,
    )
    files = await application.corpora.list_ingested_files(workspace_name)
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
    application = await _ensure_application()
    workspace_name = normalize_workspace(args.workspace or application.config.workspace)
    await _enforce_access(
        AccessAction.WORKSPACE_DELETE_FILES,
        workspace_name,
        application=application,
    )
    results = await application.corpora.delete_files(
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
