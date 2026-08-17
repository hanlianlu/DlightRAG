# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""RAG operations API routes."""

import logging
from typing import Any

from dlightrag_rag.ingestion.uploads import (
    UploadTooLargeError,
    safe_upload_basename,
    write_upload_stream,
)
from dlightrag_rag.workspaces import normalize_workspace
from fastapi import APIRouter, Depends, HTTPException, Request
from starlette.datastructures import UploadFile as StarletteUploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException

from dlightrag.access import AccessAction, UserContext
from dlightrag.answer.sources import SourceDownloadLinkBuilder
from dlightrag.api.auth import get_current_user
from dlightrag.api.models import (
    IngestJobStatusResponse,
    IngestRequest,
    ResetRequest,
    ResetResponse,
    RetrievalResponse,
    RetrieveRequest,
    UploadIngestJobResponse,
)
from dlightrag.app_state import request_config
from dlightrag.core.client_contracts import IngestSpec
from dlightrag.core.client_execution import execute_retrieve
from dlightrag.core.client_payloads import retrieval_payload
from dlightrag.core.client_requests import (
    ingest_spec_from_payload,
    managed_local_ingest_documents,
    managed_local_ingest_path,
)

from .deps import (
    authorized_workspaces,
    enforce_access,
    get_manager,
    resolve_authorized_query_workspaces,
    resolve_workspace,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _job_response(job: dict[str, Any]) -> dict[str, Any]:
    job["status_url"] = f"/ingest/jobs/{job['job_id']}"
    return job


@router.post(
    "/ingest",
    response_model=IngestJobStatusResponse,
    status_code=202,
)
async def ingest(
    body: IngestRequest, request: Request, user: UserContext = Depends(get_current_user)
) -> dict[str, Any]:
    """Bulk document ingestion."""
    manager = get_manager(request)
    cfg = request_config(request)
    ws = resolve_workspace(body.workspace, request)
    await enforce_access(request, user, AccessAction.WORKSPACE_INGEST, workspace=ws)
    ingest_spec = ingest_spec_from_payload(body)
    if body.source_type == "local":
        try:
            path = managed_local_ingest_path(
                source_type=body.source_type,
                path=ingest_spec.path,
                input_dir=cfg.input_dir_path,
                workspace=ws,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from None
        documents = managed_local_ingest_documents(
            source_type=body.source_type,
            documents=ingest_spec.documents,
            input_dir=cfg.input_dir_path,
            workspace=ws,
        )
        ingest_spec = ingest_spec.model_copy(update={"path": path, "documents": documents})

    job = await manager.astart_ingest_job(ws, ingest_spec)
    return _job_response(job)


@router.get("/ingest/jobs/{job_id}", response_model=IngestJobStatusResponse)
async def get_ingest_job(
    job_id: str,
    request: Request,
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    """Return durable ingest job status."""
    manager = get_manager(request)
    job = await manager.aget_ingest_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Ingest job not found")
    workspace = job.get("workspace")
    workspace_id = normalize_workspace(str(workspace)) if workspace else None
    await enforce_access(
        request,
        user,
        AccessAction.JOB_READ,
        workspace=workspace_id,
    )
    return job


@router.post("/ingest/jobs/{job_id}/cancel", response_model=IngestJobStatusResponse)
async def cancel_ingest_job(
    job_id: str,
    request: Request,
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    """Stop a running ingest job, keeping whatever it already ingested."""
    manager = get_manager(request)
    job = await manager.aget_ingest_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Ingest job not found")
    workspace = job.get("workspace")
    workspace_id = normalize_workspace(str(workspace)) if workspace else None
    await enforce_access(
        request,
        user,
        AccessAction.JOB_CANCEL,
        workspace=workspace_id,
    )
    cancelled = await manager.acancel_ingest_job(job_id)
    return cancelled if cancelled is not None else job


@router.post("/retrieve", response_model=RetrievalResponse)
async def retrieve(
    body: RetrieveRequest, request: Request, user: UserContext = Depends(get_current_user)
) -> dict[str, Any]:
    """Retrieve contexts and sources without LLM answer generation."""
    manager = get_manager(request)
    resolved_workspaces = await resolve_authorized_query_workspaces(
        request,
        user,
        workspaces=body.workspaces,
        all_workspaces=body.all_workspaces,
    )
    downloadable_workspaces = await authorized_workspaces(
        request,
        user,
        resolved_workspaces,
        AccessAction.WORKSPACE_DOWNLOAD_SOURCE,
    )
    visual_workspaces = await authorized_workspaces(
        request,
        user,
        resolved_workspaces,
        AccessAction.WORKSPACE_READ_VISUAL_ASSET,
    )
    result = await execute_retrieve(
        manager=manager,
        payload=body,
        resolved_workspaces=resolved_workspaces,
    )
    link_builder = SourceDownloadLinkBuilder()
    return retrieval_payload(
        result,
        source_link_builder=link_builder,
        downloadable_workspaces=downloadable_workspaces,
        visual_workspaces=visual_workspaces,
    )


_ALLOWED_INGEST_PARTS = {"file", "workspace", "title", "author", "metadata"}
_MAX_INGEST_FORM_FIELDS = 8
_INGEST_FORM_FIELD_MAX_BYTES = 1024 * 1024


@router.post(
    "/ingest/blob",
    response_model=UploadIngestJobResponse,
    status_code=202,
)
async def ingest_blob(
    request: Request,
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    """Direct file upload ingestion via multipart/form-data.

    File is persisted to input_dir/<workspace>/<filename> for citation
    download links, then ingested via the local file pipeline.
    """
    import json as _json

    manager = get_manager(request)
    cfg = request_config(request)
    try:
        form = await request.form(
            max_files=2,
            max_fields=_MAX_INGEST_FORM_FIELDS,
            max_part_size=_INGEST_FORM_FIELD_MAX_BYTES,
        )
    except StarletteHTTPException as exc:
        detail = str(exc.detail)
        if exc.status_code == 400 and detail.startswith(
            ("Too many files.", "Too many fields.", "Part exceeded maximum size")
        ):
            raise HTTPException(status_code=413, detail=detail) from exc
        raise
    try:
        unexpected = sorted({key for key, _ in form.multi_items()} - _ALLOWED_INGEST_PARTS)
        if unexpected:
            raise HTTPException(
                status_code=400,
                detail=f"Unexpected multipart field(s): {', '.join(unexpected)}",
            )
        files = form.getlist("file")
        if len(files) != 1 or not isinstance(files[0], StarletteUploadFile):
            raise HTTPException(status_code=400, detail="Exactly one file is required")
        file = files[0]
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")

        def optional_text(name: str) -> str | None:
            value = form.get(name)
            if value in (None, ""):
                return None
            if not isinstance(value, str):
                raise HTTPException(status_code=400, detail=f"{name} must be a text field")
            return value

        workspace = optional_text("workspace")
        title = optional_text("title")
        author = optional_text("author")
        metadata = optional_text("metadata")
        ws = resolve_workspace(workspace, request)
        await enforce_access(request, user, AccessAction.WORKSPACE_INGEST, workspace=ws)

        try:
            safe_name = safe_upload_basename(file.filename)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid filename") from None

        target_dir = cfg.input_dir_path / ws
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / safe_name

        try:
            await write_upload_stream(file, target_path, max_bytes=cfg.max_upload_bytes)
        except UploadTooLargeError:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds maximum size of {cfg.max_upload_bytes} bytes",
            ) from None
    finally:
        await form.close()

    # Parse optional metadata JSON
    meta_dict: dict[str, Any] | None = None
    if metadata:
        try:
            meta_dict = _json.loads(metadata)
        except _json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid metadata JSON") from None

    kwargs: dict[str, Any] = {"source_type": "local", "path": str(target_path)}
    if title is not None:
        kwargs["title"] = title
    if author is not None:
        kwargs["author"] = author
    if meta_dict is not None:
        kwargs["metadata"] = meta_dict

    job = await manager.astart_ingest_job(ws, IngestSpec(**kwargs))
    job["uploaded_file"] = str(target_path)
    job["filename"] = safe_name
    return _job_response(job)


@router.post("/reset", response_model=ResetResponse)
async def reset_workspace(
    body: ResetRequest, request: Request, user: UserContext = Depends(get_current_user)
) -> dict[str, Any]:
    """Reset all RAG data for a workspace."""
    manager = get_manager(request)
    ws = resolve_workspace(body.workspace, request)
    await enforce_access(request, user, AccessAction.WORKSPACE_RESET, workspace=ws)
    return await manager.areset(
        workspace=ws,
        keep_files=body.keep_files,
        dry_run=body.dry_run,
    )
