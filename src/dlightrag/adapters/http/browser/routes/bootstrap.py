# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Authenticated browser bootstrap contract shared by the page and Lit app."""

import logging
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import Field

from dlightrag.adapters.http.browser.deps import (
    filter_web_workspace_records,
    get_application,
    get_web_access_gate,
    get_workspace,
)
from dlightrag.adapters.http.browser.workspace_models import (
    WebBootstrapWorkspace,
    project_workspace_records,
)
from dlightrag.application.access.control import AccessAction
from dlightrag.application.access.workspaces import WorkspaceRecord
from dlightrag.application.corpus_admin.workspace_catalog import (
    WORKSPACE_CATALOG_PAGE_MAX_LIMIT,
    WorkspaceCatalogPageRequest,
)
from dlightrag.application.policies.service import (
    AGENT_EFFORT_CEILING,
    ANSWER_FEEDBACK,
    PolicyUnreadableError,
)
from dlightrag.engine.answer.client_contracts import (
    AnswerEffort,
    ClientContractModel,
)
from dlightrag.engine.answer.image_capability import ImageCapabilityStatus
from dlightrag.engine.answer.resources.admission import READABLE_DOCUMENT_EXTENSIONS
from dlightrag.engine.rag.workspace.workspaces import (
    normalize_workspace,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# The page reports as a service of its own, so a collector's service map keeps the
# browser apart from the processes that serve it.
BROWSER_OBSERVABILITY_SERVICE = "dlightrag-web"


class WebBootstrapUnavailableError(RuntimeError):
    """The workspace inventory required for browser startup is unavailable."""


class WebAttachmentBootstrap(ClientContractModel):
    count_limit: int
    image_max_bytes: int
    document_max_bytes: int
    extensions: list[str]
    image_capability: ImageCapabilityStatus
    image_limit: int
    accept: str


class WebAnswerEffort(ClientContractModel):
    """The agent efforts this deployment offers the caller, and the default they run at if named."""

    levels: list[AnswerEffort]
    default: AnswerEffort | None = None


class WebBootstrapObservability(ClientContractModel):
    """Where the page sends errors, RUM, and session replay, once it is offered a channel."""

    url: str
    # The ingestion key is a credential: it rides under the SDK's own option spelling and
    # stays out of every repr.
    api_key: str = Field(serialization_alias="apiKey", repr=False)
    service: str
    # The share of sessions whose replay is recorded. Errors and RUM ignore it: the page
    # samples the recording once per session, and a session it passed over still reports.
    replay_sample_rate: float = 1.0


class WebBootstrap(ClientContractModel):
    contract_version: Literal[6] = 6
    workspaces: list[WebBootstrapWorkspace]
    workspaces_next_cursor: str | None = None
    primary_workspace: str
    active_workspaces: list[str]
    known_workspaces: list[str]
    # The deployment's configured Workspace, which Workspace Delete refuses.
    default_workspace: str
    answer_attachments: WebAttachmentBootstrap
    active_html_preview_enabled: bool
    agent_effort: WebAnswerEffort
    # Whether the `answer_feedback` Runtime Policy offers this caller the thumbs under an answer.
    answer_feedback_enabled: bool
    # The browser channel exists only where an operator set both its destination and its key;
    # unset, the snapshot omits the object and the page starts no SDK.
    observability: WebBootstrapObservability | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


async def build_web_bootstrap(
    request: Request,
    workspace: str,
) -> WebBootstrap:
    """Build the one authorized startup snapshot consumed by the browser."""
    application = get_application(request)
    default_workspace = application.config.deployment.workspace_id
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

    # Authorization inputs (known/active/primary) keep the full catalog. Only
    # the user-visible array is a bounded first page over the full catalog
    # ordering; its continuation cursor is minted from the full page so a later
    # load-more page never re-delivers rows the first page already displayed.
    try:
        catalog_page = await application.corpora.list_workspace_records_page(
            page=WorkspaceCatalogPageRequest(limit=WORKSPACE_CATALOG_PAGE_MAX_LIMIT)
        )
        page_records = await filter_web_workspace_records(
            request,
            AccessAction.WORKSPACE_QUERY,
            list(catalog_page.items),
        )
        workspaces = await project_workspace_records(
            get_web_access_gate(request), page_records, default_workspace=default_workspace
        )
        next_cursor = (
            application.corpora.workspace_catalog_cursor_codec.encode(catalog_page.next_cursor)
            if catalog_page.next_cursor is not None
            else None
        )
    except Exception as exc:
        if isinstance(exc, WebBootstrapUnavailableError):
            raise
        raise WebBootstrapUnavailableError from exc

    known = set(record["workspace"] for record in records)

    # Active/primary computation keeps the full authorized catalog; the bounded
    # workspaces array above is presentation-only.
    authorized_full = [record["workspace"] for record in records]
    active_raw = request.cookies.get("dlightrag_workspace_ids", "")
    active = [normalize_workspace(item.strip()) for item in active_raw.split(",") if item.strip()]
    active = [item for item in active if item in known]

    primary = normalize_workspace(request.cookies.get("dlightrag_workspace", workspace))
    if not active:
        active = authorized_full
    if primary not in known:
        primary = (
            default_workspace
            if default_workspace in known
            else (authorized_full[0] if authorized_full else "")
        )
    # Once the catalog traversal is exhausted, every authorized record the page
    # lacks (such as the synthetic default of a degraded registry) joins it in
    # order. Until then only the primary does: it is the Files panel's target, and
    # that panel offers what its record says the caller may change.
    shown = {item.workspace for item in workspaces}
    workspaces += await project_workspace_records(
        get_web_access_gate(request),
        [
            record
            for record in records
            if record["workspace"] not in shown
            and (next_cursor is None or record["workspace"] == primary)
        ],
        default_workspace=default_workspace,
    )

    capability = capabilities.answer
    if capability is None:
        capability_status = "unknown"
        effective_current_upload_limit = 0
    else:
        capability_status = capability.status
        effective_current_upload_limit = capability.effective_max_images

    # Both halves or neither: a destination without a key would have the page send
    # unauthenticated telemetry, and a key without a destination has nowhere to send it.
    browser_url = application.config.observability.browser_otlp_url
    browser_key = application.config.observability.browser_api_key
    observability = (
        WebBootstrapObservability(
            url=browser_url,
            api_key=browser_key,
            service=BROWSER_OBSERVABILITY_SERVICE,
            replay_sample_rate=application.config.observability.browser_replay_sample_rate,
        )
        if browser_url and browser_key
        else None
    )

    # The composer offers what an Answer Run reads; acceptance enforces the same list.
    extensions = list(READABLE_DOCUMENT_EXTENSIONS)
    attachment_limit = application.config.answer.generation.max_attachment_bytes
    # Read afresh, like every route that depends on a policy. A document that no longer
    # validates must not take the whole page down with it: the page loads with the thumbs off,
    # while the rating routes and the policy GET answer 503 until an administrator replaces it.
    user = getattr(request.state, "user_context", None)
    try:
        feedback_offered = await application.policies.value_for(ANSWER_FEEDBACK, user)
    except PolicyUnreadableError:
        logger.warning("Runtime policy %s is unreadable; no one is offered it", ANSWER_FEEDBACK)
        feedback_offered = False
    # The same for the effort ceiling: the page loads with no effort offered, and a submission
    # is refused until the document is replaced, since a default could admit what it forbids.
    try:
        effort_offer = await application.answers.agent_effort_offer(user)
        agent_effort = WebAnswerEffort(
            levels=list(effort_offer.levels),
            default=effort_offer.default,
        )
    except PolicyUnreadableError:
        logger.warning(
            "Runtime policy %s is unreadable; no effort is offered", AGENT_EFFORT_CEILING
        )
        agent_effort = WebAnswerEffort(levels=[], default=None)
    return WebBootstrap(
        workspaces=workspaces,
        workspaces_next_cursor=next_cursor,
        primary_workspace=primary,
        active_workspaces=active,
        known_workspaces=authorized_full,
        default_workspace=default_workspace,
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
        agent_effort=agent_effort,
        answer_feedback_enabled=feedback_offered,
        observability=observability,
    )


@router.get("/bootstrap", response_model=WebBootstrap)
async def browser_bootstrap(
    request: Request,
    response: Response,
    workspace: str = Depends(get_workspace),
) -> WebBootstrap:
    # Nothing here is page-load-invariant, and the browser channel's key is part of it.
    response.headers["Cache-Control"] = "private, no-store"
    try:
        return await build_web_bootstrap(request, workspace)
    except WebBootstrapUnavailableError:
        raise HTTPException(
            status_code=503,
            detail="Web application bootstrap is unavailable",
        ) from None


__all__ = [
    "BROWSER_OBSERVABILITY_SERVICE",
    "WebAnswerEffort",
    "WebAttachmentBootstrap",
    "WebBootstrap",
    "WebBootstrapObservability",
    "WebBootstrapUnavailableError",
    "build_web_bootstrap",
    "router",
]
