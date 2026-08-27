# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""RAG operations API routes."""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from starlette.datastructures import UploadFile as StarletteUploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException

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
from dlightrag.api.payloads import metadata_filter_from_payload
from dlightrag.application.access import AccessAction, UserContext
from dlightrag.application.corpus_admin import (
    CorpusResetResult,
    IngestSpec,
    UnsafeUploadNameError,
    UploadTooLargeError,
    ingest_spec_from_payload,
    managed_local_ingest_documents,
    managed_local_ingest_path,
)
from dlightrag.application.retrieval import RetrieveProjection
from dlightrag.application.retrieval import RetrieveRequest as ServiceRequest
from dlightrag.rag.workspaces import normalize_workspace

from .deps import (
    authorized_workspaces,
    enforce_access,
    get_application,
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
    application = get_application(request)
    cfg = application.config
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

    job = await application.corpora.start_ingest_job(ws, ingest_spec)
    return _job_response(job)


@router.get("/ingest/jobs/{job_id}", response_model=IngestJobStatusResponse)
async def get_ingest_job(
    job_id: str,
    request: Request,
    user: UserContext = Depends(get_current_user),
) -> dict[str, Any]:
    """Return durable ingest job status."""
    application = get_application(request)
    job = await application.corpora.get_ingest_job(job_id)
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
    application = get_application(request)
    job = await application.corpora.get_ingest_job(job_id)
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
    cancelled = await application.corpora.cancel_ingest_job(job_id)
    return cancelled if cancelled is not None else job


@router.post("/retrieve", response_model=RetrievalResponse)
async def retrieve(
    body: RetrieveRequest, request: Request, user: UserContext = Depends(get_current_user)
) -> dict[str, Any]:
    """Retrieve contexts and sources without LLM answer generation."""
    application = get_application(request)
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
    result = await application.retrieval.retrieve(
        ServiceRequest(
            query=body.query,
            workspaces=tuple(resolved_workspaces),
            top_k=body.top_k,
            chunk_top_k=body.chunk_top_k,
            bm25_query=body.bm25_query,
            filters=metadata_filter_from_payload(body.filters),
            query_images=tuple(
                image.model_dump(exclude_none=True) for image in body.query_images or ()
            ),
            projection=RetrieveProjection(
                downloadable_workspaces=frozenset(downloadable_workspaces),
                visual_workspaces=frozenset(visual_workspaces),
                include_download_links=True,
            ),
        )
    )
    return {
        "contexts": result.contexts,
        "sources": list(result.sources),
        "trace": dict(result.trace),
        "image_descriptions": list(result.image_descriptions),
    }


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

    application = get_application(request)
    cfg = application.config
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
            target_path, safe_name = await application.corpora.stage_upload_stream(
                ws,
                filename=file.filename,
                reader=file,
                max_bytes=cfg.corpus.ingestion.max_upload_bytes,
            )
        except UnsafeUploadNameError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except UploadTooLargeError as exc:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds maximum size of {cfg.corpus.ingestion.max_upload_bytes} bytes",
            ) from exc
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

    job = await application.corpora.start_ingest_job(ws, IngestSpec(**kwargs))
    job["uploaded_file"] = str(target_path)
    job["filename"] = safe_name
    return _job_response(job)


@router.post("/reset", response_model=ResetResponse)
async def reset_workspace(
    body: ResetRequest, request: Request, user: UserContext = Depends(get_current_user)
) -> CorpusResetResult:
    """Reset all RAG data for a workspace."""
    application = get_application(request)
    ws = resolve_workspace(body.workspace, request)
    await enforce_access(request, user, AccessAction.WORKSPACE_RESET, workspace=ws)
    return await application.corpora.reset(
        workspace_ids=(ws,),
        keep_files=body.keep_files,
        dry_run=body.dry_run,
    )
