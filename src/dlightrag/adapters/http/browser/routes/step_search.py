# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Web route for the Run-level step search: the one step list the Run panel still walks.

The overview states aggregates; when a reader types a query the panel switches to the matching
steps of the whole Run, across every agent. The application read derives the matches from the same
merged index the per-agent step list reads — excluding thinking, whose text is how a turn was
reached rather than an action the Run took — and pages them newest page first with the marks the
whole Run states. This adapter is the wire shape: one bounded line of each step's masked content,
the cursor the next page is read from, and none of the fields the retired timeline wire carried
that a search does not state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from fastapi.responses import JSONResponse

from dlightrag.adapters.http.browser.deps import (
    get_application,
    get_web_conversation_service,
)
from dlightrag.application.access.principal import (
    owner_id_from_user,
)
from dlightrag.application.answer_runs.service import AnswerService
from dlightrag.application.answer_runs.timeline import RunTimelinePosition, RunTimelineStep
from dlightrag.application.web_conversations.service import WebConversationService

from ._step_cursor import decode_step_cursor, encode_step_cursor

router = APIRouter()

STEP_SEARCH_DEFAULT_LIMIT = 100
STEP_SEARCH_MAX_LIMIT = 200
STEP_EXCERPT_CHARS = 120

#: A search is a different read every time — its page is a function of the words and the cursor —
#: so it is never stored by the browser. The panel coalesces its own reads instead.
_CACHE_SEARCH = {"Cache-Control": "private, no-store"}


def _excerpt(content: Any) -> str:
    """The one line a search row shows: the first, bounded, so a page never ships a payload."""
    if not isinstance(content, str):
        return ""
    return content.split("\n", 1)[0].strip()[:STEP_EXCERPT_CHARS]


def _step_marks(
    step: RunTimelineStep,
    marks: Mapping[tuple[str, int], tuple[str, ...]],
) -> Sequence[str]:
    """The marks this step states, or none when it is not a decision point.

    The marks are keyed by ``(agent, sequence)``, which a thinking step shares with the message it
    precedes. A search page holds no thinking step, and the thought is how the turn was reached
    rather than the step that made the decision, so it never carries the marks of the step beside
    it.
    """
    if step.kind == "thinking":
        return ()
    return marks.get((step.agent, step.sequence), ())


def _step_payload(step: RunTimelineStep, marks: Sequence[str]) -> dict[str, Any]:
    """One matching step of the page, in the shape the Run panel's search results read.

    A step's whole text is not here: a result row states a bounded ``excerpt``, and the full text
    stays where an agent's own page shows it. Where the step sits is stated by its agent, sequence,
    and kind, so a client can link the row with the message it came from without the retired wire's
    page index or call id.
    """
    return {
        "agent": step.agent,
        "agent_label": step.agent_label,
        "at": step.at,
        "kind": step.kind,
        "sequence": step.sequence,
        "name": step.name,
        "subject": step.subject,
        "outcome": step.outcome,
        "duration_ms": step.duration_ms,
        "excerpt": _excerpt(step.content),
        "marks": list(marks),
    }


@router.get("/answer/{run_id}/steps/search")
async def answer_run_step_search(
    run_id: str,
    request: Request,
    q: Annotated[str, Query(min_length=1, max_length=200)],
    conversation_service: WebConversationService = Depends(get_web_conversation_service),
    limit: Annotated[int, Query(ge=1, le=STEP_SEARCH_MAX_LIMIT)] = STEP_SEARCH_DEFAULT_LIMIT,
    before: Annotated[str | None, Query(min_length=1, max_length=200)] = None,
) -> Response:
    """One page of the Run's steps whose messages hold the search's words.

    ``q`` is the reader's query; ``before`` is the ``next_before`` the previous page stated, so
    this page is strictly older than the step the cursor names. Rows are the merge's order, oldest
    first within the page, and each states the marks the whole Run gives its step rather than the
    marks this page could see. ``matched`` sizes the whole result beside the page, so a page never
    reads as the Run's whole search.

    The Run is checked against the reader's conversations first, so an unknown or foreign Run is
    the same opaque 404 as every other answer read; a cursor this route did not state is refused
    before the read.
    """
    user = getattr(request.state, "user_context", None)
    if await conversation_service.turn_for_run(user, run_id) is None:
        raise HTTPException(status_code=404, detail="Answer run not found")
    boundary: RunTimelinePosition | None = None
    if before is not None:
        boundary = decode_step_cursor(before)
        if boundary is None:
            raise HTTPException(
                status_code=422, detail="`before` must be a cursor this route stated"
            )
    application = get_application(request)
    answers: AnswerService = application.answers
    read = await answers.search_run_steps(
        owner_id=owner_id_from_user(user),
        run_id=run_id,
        query=q,
        limit=limit,
        before=boundary,
    )
    if read is None:
        raise HTTPException(status_code=404, detail="Answer run not found")
    return JSONResponse(
        content={
            "run_id": run_id,
            "steps": [_step_payload(step, _step_marks(step, read.marks)) for step in read.page],
            "next_before": (
                encode_step_cursor(read.page[0]) if read.page and read.has_older else None
            ),
            "matched": read.matched,
            "terms": list(read.terms),
            "match": read.match,
        },
        headers=_CACHE_SEARCH,
    )


__all__ = ["router"]
