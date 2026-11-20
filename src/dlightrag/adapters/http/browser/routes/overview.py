# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Web route for the Inspector's Run overview: one Run's whole-Run dossier.

The durable transcript is per Session, so a Run-level panel cannot count what a reader needs from
one agent's page. The application read derives the overview once from the same merged index the
step list reads, gates the roster's labels and the capped findings' excerpts at emission, and
states only whole-Run numbers. This adapter is the wire shape: it projects the read into the
snake_case payload and derives the panel's cache validator from the index version the read states.
It reads nothing itself beyond the read the service hands it.
"""

from __future__ import annotations

import hashlib
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from dlightrag.adapters.http.browser.deps import (
    get_application,
    get_web_conversation_service,
)
from dlightrag.application.access.principal import (
    owner_id_from_user,
)
from dlightrag.application.answer_runs.overview import RunAgentRollup, RunFinding, RunTotals
from dlightrag.application.answer_runs.service import AnswerService
from dlightrag.application.web_conversations.service import WebConversationService

router = APIRouter()

#: The overview is a whole-Run read from a derived index whose version the panel holds: a browser
#: revalidates it each time rather than trusting a max-age, and an unchanged Run answers 304.
_OVERVIEW_CACHE = {"Cache-Control": "private, no-cache"}


def _etag(version: str) -> str:
    """A strong validator over the derived index version the overview was served from.

    The overview's body is a function of the Run version alone — it states no page and takes no
    narrowing — so the index version is the whole validator, and a Run that has not moved costs a
    round trip and no body.
    """
    digest = hashlib.sha256(version.encode("utf-8")).hexdigest()
    return f'"{digest[:32]}"'


def _totals_payload(totals: RunTotals) -> dict[str, Any]:
    """The whole Run's size, as the header band and the composition bar read it."""
    return {
        "steps": totals.steps,
        "kinds": dict(totals.kinds),
        "decisions": totals.decisions,
        "marks": dict(totals.marks),
        "tool_time_ms": totals.tool_time_ms,
    }


def _agent_payload(agent: RunAgentRollup) -> dict[str, Any]:
    """One agent's ledger row: its share of the Run, never a share of a page."""
    return {
        "agent": agent.agent,
        "label": agent.label,
        "order": agent.order,
        "steps": agent.steps,
        "tool_calls": agent.tool_calls,
        "marks": dict(agent.marks),
        "tool_time_ms": agent.tool_time_ms,
        "first_at": agent.first_at,
        "last_at": agent.last_at,
    }


def _finding_payload(finding: RunFinding) -> dict[str, Any]:
    """One finding row: the step's own facts and the one bounded line of its masked content."""
    return {
        "agent": finding.agent,
        "sequence": finding.sequence,
        "kind": finding.kind,
        "at": finding.at,
        "name": finding.name,
        "subject": finding.subject,
        "outcome": finding.outcome,
        "marks": list(finding.marks),
        "excerpt": finding.excerpt,
    }


@router.get("/answer/{run_id}/overview")
async def answer_run_overview(
    run_id: str,
    request: Request,
    conversation_service: WebConversationService = Depends(get_web_conversation_service),
) -> Response:
    """One Run's whole-Run overview: its totals, its agents, and its capped findings.

    The numbers are the read's, not the response's: this route projects what the whole-Run read
    derived and never reconstructs a total from the rows it happens to be sending. ``usage`` is
    stated as null when the Run result carries none, and a matching ``If-None-Match`` answers 304,
    so a panel that already holds an unchanged Run pays no body.

    The run is checked against the reader's conversations first, so an unknown or foreign Run is
    the same opaque 404 as every other answer read.
    """
    user = getattr(request.state, "user_context", None)
    if await conversation_service.turn_for_run(user, run_id) is None:
        raise HTTPException(status_code=404, detail="Answer run not found")
    application = get_application(request)
    answers: AnswerService = application.answers
    overview = await answers.run_overview(owner_id=owner_id_from_user(user), run_id=run_id)
    if overview is None:
        raise HTTPException(status_code=404, detail="Answer run not found")
    headers = {**_OVERVIEW_CACHE, "ETag": _etag(overview.version)}
    if request.headers.get("if-none-match") == headers["ETag"]:
        return Response(status_code=304, headers=headers)
    return JSONResponse(
        content={
            "run_id": overview.run_id,
            "totals": _totals_payload(overview.totals),
            "agents": [_agent_payload(agent) for agent in overview.agents],
            "findings": [_finding_payload(finding) for finding in overview.findings],
            "findings_total": overview.findings_total,
            "usage": None if overview.usage is None else dict(overview.usage),
        },
        headers=headers,
    )


__all__ = ["router"]
