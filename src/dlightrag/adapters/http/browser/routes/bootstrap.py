# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Authenticated browser bootstrap contract shared by the page and Lit app."""

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request

from dlightrag.adapters.http.browser.attachment_models import SUPPORTED_DOCUMENT_EXTENSIONS
from dlightrag.adapters.http.browser.deps import (
    filter_web_workspace_records,
    get_application,
    get_workspace,
)
from dlightrag.application.access import AccessAction, WorkspaceRecord
from dlightrag.application.answer_runs.capability import ImageCapabilityStatus
from dlightrag.application.answer_runs.client_contracts import ClientContractModel
from dlightrag.application.corpus_admin import normalize_workspace

router = APIRouter()


class WebBootstrapUnavailableError(RuntimeError):
    """The workspace inventory required for browser startup is unavailable."""


class WebBootstrapWorkspace(ClientContractModel):
    workspace: str
    display_name: str
    embedding_model: str


class WebAttachmentBootstrap(ClientContractModel):
    count_limit: int
    image_max_bytes: int
    document_max_bytes: int
    extensions: list[str]
    image_capability: ImageCapabilityStatus
    image_limit: int
    accept: str


class WebBootstrap(ClientContractModel):
    contract_version: Literal[1] = 1
    workspaces: list[WebBootstrapWorkspace]
    primary_workspace: str
    active_workspaces: list[str]
    answer_attachments: WebAttachmentBootstrap
    active_html_preview_enabled: bool


def _workspace_contract(record: WorkspaceRecord) -> WebBootstrapWorkspace:
    workspace = str(record["workspace"])
    return WebBootstrapWorkspace(
        workspace=workspace,
        display_name=str(record.get("display_name") or workspace),
        embedding_model=str(record.get("embedding_model") or ""),
    )


async def build_web_bootstrap(
    request: Request,
    workspace: str,
) -> WebBootstrap:
    """Build the one authorized startup snapshot consumed by the browser."""
    application = get_application(request)
    capabilities = await application.answers.capabilities()
    records: list[WorkspaceRecord]
    try:
        records = await application.corpora.alist_workspace_records()
    except Exception as exc:
        raise WebBootstrapUnavailableError from exc
    records = await filter_web_workspace_records(
        request,
        AccessAction.WORKSPACE_QUERY,
        records,
    )
    workspaces = [_workspace_contract(record) for record in records]

    authorized = [record.workspace for record in workspaces]
    known = set(authorized)
    active_raw = request.cookies.get("dlightrag_workspace_ids", "")
    active = [normalize_workspace(item.strip()) for item in active_raw.split(",") if item.strip()]
    active = [item for item in active if item in known]

    primary = normalize_workspace(request.cookies.get("dlightrag_workspace", workspace))
    if not active:
        active = authorized
    if primary not in known:
        primary = "default" if "default" in known else (authorized[0] if authorized else "")

    capability = capabilities.answer
    if capability is None:
        capability_status = "unknown"
        effective_current_upload_limit = 0
    else:
        capability_status = capability.status
        effective_current_upload_limit = capability.effective_max_images

    extensions = sorted(SUPPORTED_DOCUMENT_EXTENSIONS)
    attachment_limit = application.config.answer.generation.max_attachment_bytes
    return WebBootstrap(
        workspaces=workspaces,
        primary_workspace=primary,
        active_workspaces=active,
        answer_attachments=WebAttachmentBootstrap(
            count_limit=application.config.answer.generation.max_attachments,
            image_max_bytes=attachment_limit,
            document_max_bytes=attachment_limit,
            extensions=extensions,
            image_capability=capability_status,
            image_limit=effective_current_upload_limit,
            accept=",".join(["image/*", *(f".{extension}" for extension in extensions)]),
        ),
        active_html_preview_enabled=(
            application.config.answer.conversations.active_html_preview_enabled
        ),
    )


@router.get("/bootstrap", response_model=WebBootstrap)
async def browser_bootstrap(
    request: Request,
    workspace: str = Depends(get_workspace),
) -> WebBootstrap:
    try:
        return await build_web_bootstrap(request, workspace)
    except WebBootstrapUnavailableError:
        raise HTTPException(
            status_code=503,
            detail="Web application bootstrap is unavailable",
        ) from None


__all__ = [
    "WebAttachmentBootstrap",
    "WebBootstrap",
    "WebBootstrapUnavailableError",
    "WebBootstrapWorkspace",
    "build_web_bootstrap",
    "router",
]
