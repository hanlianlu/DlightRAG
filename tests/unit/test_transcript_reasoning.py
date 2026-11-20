# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The model's reasoning reaches the Inspector activity page, and no other transcript surface.

The store projects an assistant entry's reasoning raw — it is not a reader edge — and the answer
service masks and bounds it for the activity page alone. Every other transcript reader (the child
observation, the transport tail, search hits, the step-search excerpt) states no reasoning at all
(ADR 0065, Considered options). The live `reasoning` stream is the other surface and is
masked at the SSE edges; these tests are about the retrospective page.
"""

import json
from typing import Any

import pytest

from dlightrag.adapters.postgres.runtime.run_store import _transcript_message
from dlightrag.application.answer_runs.service import (
    PUBLIC_REASONING_MAX_CHARS,
    _public_transcript_message,
)
from dlightrag.engine.agent.skill_protection import MASK_PLACEHOLDER, ProtectedTextIndex
from tests.unit.test_answer_service import (
    _NOW,
    _OWNER,
    _PROTECTED_PARAGRAPH,
    _protected_index,
    _record,
    _service,
    _Store,
)

_PRIVATE_REASONING = "The ledger note names the account; I will not quote it in the answer."

# A Run whose Session has stored turns is the branch of every transcript read that projects rows.
_STORED_RUN = {"agent_session_id": "0199a0a0-0000-7000-8000-000000000099"}


def _stored_assistant(reasoning: str | None = _PRIVATE_REASONING) -> dict[str, Any]:
    """One assistant message as the store projects it, optionally carrying reasoning."""
    message: dict[str, Any] = {
        "sequence": 7,
        "role": "assistant",
        "content": "Checked the filings.",
        "tool_calls": [],
        "at": "2026-10-05T12:00:00+00:00",
    }
    if reasoning is not None:
        message["reasoning"] = reasoning
    return message


@pytest.mark.parametrize(
    "payload_json",
    [
        {"content": "Checked the filings.", "stop_reason": "stop", "reasoning": _PRIVATE_REASONING},
        json.dumps(
            {
                "content": "Checked the filings.",
                "stop_reason": "stop",
                "reasoning": _PRIVATE_REASONING,
            }
        ),
    ],
    ids=["mapping", "json-text"],
)
def test_the_store_projects_assistant_reasoning_raw(payload_json: Any) -> None:
    """The store states what was captured: masking and bounding belong to the reader surface."""
    message = _transcript_message(
        payload_json=payload_json,
        entry_type="assistant_message",
        sequence=7,
        timestamp=_NOW,
    )

    assert message is not None
    assert message["reasoning"] == _PRIVATE_REASONING


def test_the_store_projects_an_absent_reasoning_as_empty_text() -> None:
    """An assistant turn that thought nothing still states the field every assistant message has."""
    message = _transcript_message(
        payload_json={"content": "ok", "stop_reason": "stop"},
        entry_type="assistant_message",
        sequence=7,
        timestamp=_NOW,
    )

    assert message is not None
    assert message["reasoning"] == ""


def test_the_store_keeps_protected_text_in_reasoning_unmasked() -> None:
    """Reasoning is stored text like any other: the store has no reader gate to apply."""
    message = _transcript_message(
        payload_json={
            "content": "Checked the filings.",
            "stop_reason": "stop",
            "reasoning": f"I read it: {_PROTECTED_PARAGRAPH}",
        },
        entry_type="assistant_message",
        sequence=7,
        timestamp=_NOW,
    )

    assert message is not None
    assert message["reasoning"] == f"I read it: {_PROTECTED_PARAGRAPH}"


@pytest.mark.parametrize("include_reasoning", [False, True], ids=["default", "activity"])
def test_the_default_projection_states_no_reasoning(include_reasoning: bool) -> None:
    projected = _public_transcript_message(
        _stored_assistant(),
        protected_skill_documents=frozenset(),
        include_reasoning=include_reasoning,
    )

    if include_reasoning:
        assert projected["reasoning"] == _PRIVATE_REASONING
    else:
        assert "reasoning" not in projected


async def test_the_activity_page_states_masked_reasoning() -> None:
    """The activity page is the reader surface; a protected document's words are gated there."""
    store = _Store(run=_record(status="running", accepted_input=_STORED_RUN))
    store.transcript_rows = (_stored_assistant(f"I read the document: {_PROTECTED_PARAGRAPH}"),)
    service = _service(store=store, protected_text=_protected_index().mask)

    page = await service.activity_page(owner_id=_OWNER, run_id="run-1", limit=10)

    assert page is not None
    (message,) = page.messages
    assert message["reasoning"] == f"I read the document: {MASK_PLACEHOLDER}"
    assert _PROTECTED_PARAGRAPH not in str(page.payload())


async def test_the_activity_page_bounds_reasoning_without_marking_the_cut() -> None:
    reasoning = "the ledger reconciles in two passes. " * 300
    assert len(reasoning) > PUBLIC_REASONING_MAX_CHARS
    store = _Store(run=_record(status="running", accepted_input=_STORED_RUN))
    store.transcript_rows = (_stored_assistant(reasoning),)
    service = _service(store=store, protected_text=ProtectedTextIndex.empty().mask)

    page = await service.activity_page(owner_id=_OWNER, run_id="run-1", limit=10)

    assert page is not None
    (message,) = page.messages
    assert PUBLIC_REASONING_MAX_CHARS == 4000
    assert message["reasoning"] == reasoning[:PUBLIC_REASONING_MAX_CHARS]
    assert not message["reasoning"].endswith("…")


@pytest.mark.parametrize("reasoning", [None, "", " \n\t "], ids=["absent", "empty", "blank"])
async def test_the_activity_page_states_no_reasoning_key_when_there_is_none(
    reasoning: str | None,
) -> None:
    store = _Store(run=_record(status="running", accepted_input=_STORED_RUN))
    store.transcript_rows = (_stored_assistant(reasoning),)
    service = _service(store=store, protected_text=ProtectedTextIndex.empty().mask)

    page = await service.activity_page(owner_id=_OWNER, run_id="run-1", limit=10)

    assert page is not None
    (message,) = page.messages
    assert "reasoning" not in message


async def test_the_transcript_tail_states_no_reasoning() -> None:
    """The tail is a transport-neutral reader surface: the store row's reasoning stops here."""

    store = _Store(run=_record(status="succeeded", accepted_input=_STORED_RUN))
    store.transcript_rows = (_stored_assistant(),)
    service = _service(store=store, protected_text=ProtectedTextIndex.empty().mask)

    tail = await service.transcript_tail(owner_id=_OWNER, run_id="run-1", limit=10)

    assert tail is not None
    (message,) = tail.messages
    assert "reasoning" not in message
    assert _PRIVATE_REASONING not in str(tail.messages)


async def test_the_transcript_tail_states_no_reasoning_from_a_fallback_history() -> None:
    """A Run with no stored turn yet projects its accepted history; no reasoning may ride along."""
    store = _Store(
        run=_record(
            status="succeeded",
            accepted_input={
                "query": "why?",
                "history": [
                    {
                        "role": "assistant",
                        "content": "The earlier answer.",
                        "reasoning": _PRIVATE_REASONING,
                    }
                ],
            },
            result={"answer": "The settled answer."},
        )
    )
    service = _service(store=store, protected_text=ProtectedTextIndex.empty().mask)

    tail = await service.transcript_tail(owner_id=_OWNER, run_id="run-1", limit=10)

    assert tail is not None
    assert all("reasoning" not in message for message in tail.messages)
    assert _PRIVATE_REASONING not in str(tail.messages)


async def test_the_child_observation_states_no_reasoning() -> None:
    store = _Store(run=_record(status="running"))
    store.child_session_rows["child-1"] = {
        "child_session_id": "child-1",
        "status": "running",
        "objective": "inspect sources",
    }
    store.transcript_rows = (_stored_assistant(),)
    service = _service(store=store, protected_text=ProtectedTextIndex.empty().mask)

    observation = await service.observe_child(
        owner_id=_OWNER, run_id="run-1", child_session_id="child-1"
    )

    assert observation is not None
    (message,) = observation.transcript
    assert "reasoning" not in message
    assert _PRIVATE_REASONING not in str(observation.payload())


async def test_transcript_search_hits_state_no_reasoning() -> None:
    store = _Store(run=_record(status="succeeded"))
    store.transcript_rows = ({**_stored_assistant(), "content": "Checked the filings."},)
    service = _service(store=store, protected_text=ProtectedTextIndex.empty().mask)

    answer = await service.search_run_traces(owner_id=_OWNER, run_id="run-1", query="filings")

    assert answer is not None
    _, hits, _, _ = answer
    assert hits
    assert all("reasoning" not in hit for hit in hits)
    assert _PRIVATE_REASONING not in str(hits)
