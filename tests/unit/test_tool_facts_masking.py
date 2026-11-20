# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Structured facts are a reader-facing carrier: their words pass the same gate as the result.

A fact is a shape, but a leaf can be the model's own text — a search query, a title a page
chose — so the reader's content gate applies to the leaves exactly as it applies to the
result they were derived from (ADR 0061).
"""

import datetime
import inspect
import json
from functools import partial
from typing import Any

import pytest

from dlightrag.adapters.http.browser import answer_events as web_events
from dlightrag.adapters.http.rest.routes import runs as rest_routes
from dlightrag.application.answer_runs.service import (
    _masked_store_message,
    _public_transcript_message,
)
from dlightrag.application.runs import RunEvent
from dlightrag.engine.agent.skill_protection import (
    MASK_PLACEHOLDER,
    ProtectedTextIndex,
)

_DOCUMENT = (
    "The run ledger is the first thing a calculation states: every input carries the period it "
    "was observed in, the source that stated it, and the mechanism by which it reaches the "
    "conclusion. An input nobody can date is an input nobody can defend."
)


def _index() -> ProtectedTextIndex:
    return ProtectedTextIndex.build([("method", "references/method.md", _DOCUMENT.encode())])


def _tool_message() -> dict[str, Any]:
    return {
        "sequence": 4,
        "role": "tool",
        "name": "search_knowledge_base",
        "tool_call_id": "call-1",
        "content": "Knowledge base added 2 new passages.",
        "facts": {"new_chunks": 2, "query": _DOCUMENT, "note": f"A note: {_DOCUMENT}"},
    }


def _event(sequence: int, event_type: str, payload: dict[str, Any]) -> RunEvent:
    return RunEvent(
        sequence=sequence,
        event_type=event_type,
        payload=payload,
        created_at=datetime.datetime.now(datetime.UTC),
    )


def test_transcript_projection_masks_facts_strings() -> None:
    index = _index()

    projected = _public_transcript_message(
        _tool_message(),
        protected_skill_documents=frozenset(),
        mask_reader_text=index.mask,
    )

    assert projected["facts"]["query"] == MASK_PLACEHOLDER
    assert projected["facts"]["note"] == f"A note: {MASK_PLACEHOLDER}"
    assert projected["facts"]["new_chunks"] == 2


def test_transcript_tail_masks_facts_strings() -> None:
    index = _index()

    projected = _masked_store_message(
        _tool_message(),
        protected=frozenset(),
        mask_reader_text=index.mask,
    )

    assert projected["facts"]["query"] == MASK_PLACEHOLDER
    assert projected["facts"]["note"] == f"A note: {MASK_PLACEHOLDER}"


@pytest.mark.parametrize("render", ["rest", "web"])
async def test_both_event_streams_mask_facts_strings(render: str) -> None:
    """A canonical frame and a browser frame carry the same facts and the same gate."""
    index = _index()
    payload = {
        "tool_name": "search_knowledge_base",
        "call_id": "call-1",
        "source_index": 0,
        "outcome": "succeeded",
        "facts": {"new_chunks": 2, "query": _DOCUMENT},
    }
    event = _event(7, "tool_end", payload)
    renderer: Any = (
        partial(
            rest_routes.run_frame,
            downloadable_workspaces=None,
            visual_workspaces=None,
            run_id="019893f4-0000-7000-8000-000000000001",
            run_kind="answer",
            mask_reader_text=index.mask,
        )
        if render == "rest"
        else partial(
            web_events.browser_frame,
            downloadable_workspaces=None,
            visual_workspaces=None,
            mask_reader_text=index.mask,
        )
    )
    rendered: Any = renderer(event)
    if inspect.isawaitable(rendered):
        rendered = await rendered
    frame: str = rendered
    data = json.loads(frame.split("data: ", 1)[1].strip())

    assert data["facts"]["query"] == MASK_PLACEHOLDER
    assert data["facts"]["new_chunks"] == 2
    assert _DOCUMENT not in frame
