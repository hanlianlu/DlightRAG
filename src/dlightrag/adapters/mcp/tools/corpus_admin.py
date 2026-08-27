# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""MCP tools for workspaces, capabilities, ingest, and files."""

from __future__ import annotations

from typing import Annotated, Any

from mcp.types import ToolAnnotations
from pydantic import Field

from dlightrag.adapters.mcp import server as mcp_server
from dlightrag.adapters.mcp.contracts import (
    CreateWorkspaceInput,
    DeleteFilesInput,
    DeleteWorkspaceInput,
    IngestInput,
    IngestJobStatusInput,
    ListFilesInput,
)
from dlightrag.adapters.mcp.server import (
    mcp_app,
)
from dlightrag.application.access import AccessAction
from dlightrag.application.answer_runs.capability import answer_image_capability_summary
from dlightrag.application.corpus_admin import (
    SourceType,
    ingest_spec_from_payload,
    managed_local_ingest_documents,
    managed_local_ingest_path,
    normalize_workspace,
    validate_workspace_name,
)


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
    application = await mcp_server._ensure_application()
    records = await application.corpora.alist_workspace_records()
    records = await mcp_server._filter_workspace_records(records, application=application)
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
    application = await mcp_server._ensure_application()
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
    application = await mcp_server._ensure_application()
    normalized_workspace, normalized_display_name = mcp_server._normalize_workspace_argument(args)
    await mcp_server._enforce_access(
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
    application = await mcp_server._ensure_application()

    label = validate_workspace_name(args.workspace)
    normalized_workspace = normalize_workspace(label)
    await mcp_server._enforce_access(
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
    application = await mcp_server._ensure_application()
    workspace_name = args.workspace or application.config.deployment.workspace
    workspace_name = normalize_workspace(workspace_name)
    await mcp_server._enforce_access(
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
    application = await mcp_server._ensure_application()
    if not args.job_id:
        raise ValueError("job_id is required")
    result = await application.corpora.get_ingest_job(args.job_id)
    if result is None:
        raise ValueError(f"Ingest job not found: {args.job_id}")
    workspace = result.get("workspace")
    workspace_id = normalize_workspace(str(workspace)) if workspace else None
    await mcp_server._enforce_access(AccessAction.JOB_READ, workspace_id, application=application)
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
    application = await mcp_server._ensure_application()
    if not args.job_id:
        raise ValueError("job_id is required")
    result = await application.corpora.get_ingest_job(args.job_id)
    if result is None:
        raise ValueError(f"Ingest job not found: {args.job_id}")
    workspace = result.get("workspace")
    workspace_id = normalize_workspace(str(workspace)) if workspace else None
    await mcp_server._enforce_access(AccessAction.JOB_CANCEL, workspace_id, application=application)
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
    application = await mcp_server._ensure_application()
    workspace_name = normalize_workspace(args.workspace or application.config.deployment.workspace)
    await mcp_server._enforce_access(
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
    application = await mcp_server._ensure_application()
    workspace_name = normalize_workspace(args.workspace or application.config.deployment.workspace)
    await mcp_server._enforce_access(
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
