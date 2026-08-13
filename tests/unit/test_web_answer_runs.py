# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The Web durable Answer run contract.

Covers the atomic submission descriptor, its idempotent replay and conflict, the
owner-scoped status/cancel/event routes, run-artifact attachment reads, and the
projection every conversation read shares. Storage-level atomicity, foreign
keys, retention, and concurrency live in the PostgreSQL integration suite.
"""

import asyncio
import datetime
import json
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from httpx import ASGITransport, AsyncClient

from dlightrag.api.server import create_app
from dlightrag.storage.answer_runs import AnswerRunEvent, IdempotencyKeyConflict
from dlightrag.storage.web_conversations import ConversationSubmissionConflict
from dlightrag.web.answer_events import render_done_event, stream_answer_events
from dlightrag.web.conversations import WebConversationService, project_conversation_turn
from tests.unit.web.answer_run_fixtures import (
    RUN_ID,
    SUBMISSION_ID,
    TURN_ID,
    answer_run,
    linked_turn,
    run_request,
    stored_result,
    web_answer_submission,
)

_CID = "00000000-0000-0000-0000-000000000001"
_BODY = {
    "query": "What changed?",
    "workspaces": ["default"],
    "conversation_id": _CID,
    "submission_id": SUBMISSION_ID,
}


@pytest.fixture
def service() -> AsyncMock:
    created = AsyncMock()
    created.start_answer.return_value = web_answer_submission(conversation_id=_CID)
    created.turn_for_run.return_value = linked_turn()
    return created


@pytest.fixture
def manager() -> AsyncMock:
    created = AsyncMock()
    created.answer_image_capability = None
    created.alist_workspace_records.return_value = [{"workspace": "default"}]
    return created


@pytest.fixture
async def client(service: AsyncMock, manager: AsyncMock, test_config):
    application = create_app(include_web_app=True)
    manager.config = test_config
    application.state.manager = manager
    application.state.web_conversation_service = service
    transport = ASGITransport(app=application)
    async with AsyncClient(
        transport=transport,
        base_url="http://test",
        cookies={"dlightrag_workspace": "default"},
    ) as created:
        yield created


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------


async def test_submission_returns_202_with_the_durable_descriptor(
    client: AsyncClient, service: AsyncMock
) -> None:
    response = await client.post("/web/answer", json=_BODY)

    assert response.status_code == 202
    body = response.json()
    assert body["run_id"] == RUN_ID
    assert body["status"] == "queued"
    assert body["turn_id"] == TURN_ID
    assert body["events_url"] == f"/web/answer/{RUN_ID}/events"
    assert body["cancel_url"] == f"/web/answer/{RUN_ID}"
    assert body["conversation"]["conversation_id"] == _CID
    # The 202 body is the whole answer contract: nothing is streamed by the
    # request that created the run.
    assert "html" not in body
    assert response.headers["content-type"].startswith("application/json")


async def test_submission_passes_the_submission_id_as_the_run_key(
    client: AsyncClient, service: AsyncMock
) -> None:
    await client.post("/web/answer", json=_BODY)

    kwargs = service.start_answer.await_args.kwargs
    assert kwargs["submission_id"] == SUBMISSION_ID
    assert kwargs["conversation_id"] == _CID
    assert kwargs["query"] == "What changed?"
    assert list(kwargs["workspaces"]) == ["default"]


async def test_replaying_a_submission_returns_the_authoritative_run(
    client: AsyncClient, service: AsyncMock
) -> None:
    service.start_answer.return_value = web_answer_submission(
        conversation_id=_CID, run=answer_run(status="running")
    )

    response = await client.post("/web/answer", json=_BODY)

    # A replay is accepted work too, so it reports the same 202 as the original
    # submission and simply carries the run's current status.
    assert response.status_code == 202
    assert response.json()["run_id"] == RUN_ID
    assert response.json()["status"] == "running"


@pytest.mark.parametrize(
    "error",
    [
        ConversationSubmissionConflict("reused"),
        IdempotencyKeyConflict("reused"),
    ],
    ids=["different-conversation", "different-input"],
)
async def test_reusing_a_submission_with_different_input_is_409(
    client: AsyncClient, service: AsyncMock, error: Exception
) -> None:
    service.start_answer.side_effect = error

    response = await client.post("/web/answer", json=_BODY)

    assert response.status_code == 409


async def test_submission_to_an_unknown_conversation_is_404(
    client: AsyncClient, service: AsyncMock
) -> None:
    service.start_answer.return_value = None

    response = await client.post("/web/answer", json=_BODY)

    assert response.status_code == 404


async def test_an_empty_question_is_rejected_before_acceptance(
    client: AsyncClient, service: AsyncMock
) -> None:
    response = await client.post("/web/answer", json={**_BODY, "query": "  "})

    assert response.status_code == 422
    service.start_answer.assert_not_awaited()


# ---------------------------------------------------------------------------
# Status, cancellation, and events
# ---------------------------------------------------------------------------


async def test_status_projects_the_linked_turn(client: AsyncClient) -> None:
    response = await client.get(f"/web/answer/{RUN_ID}")

    assert response.status_code == 200
    body = response.json()
    assert body["answer_run_id"] == RUN_ID
    assert body["status"] == "queued"
    assert body["user_text"] == "What changed?"


@pytest.mark.parametrize("path", ["", "/events"])
async def test_a_run_this_principal_does_not_own_is_404(
    client: AsyncClient, service: AsyncMock, path: str
) -> None:
    service.turn_for_run.return_value = None

    response = await client.get(f"/web/answer/{RUN_ID}{path}")

    assert response.status_code == 404


async def test_cancelling_an_unowned_run_never_reaches_the_manager(
    client: AsyncClient, service: AsyncMock, manager: AsyncMock
) -> None:
    service.turn_for_run.return_value = None

    response = await client.delete(f"/web/answer/{RUN_ID}")

    assert response.status_code == 404
    manager.acancel_answer_run.assert_not_awaited()


async def test_cancelling_a_running_run_reports_the_pending_request(
    client: AsyncClient, manager: AsyncMock
) -> None:
    running = answer_run(status="running", cancel_requested_at=datetime.datetime.now(datetime.UTC))
    manager.acancel_answer_run.return_value = Mock(outcome="pending", run=running)

    response = await client.delete(f"/web/answer/{RUN_ID}")

    assert response.status_code == 202
    assert response.json()["cancel_requested"] is True
    assert response.json()["status"] == "running"


async def test_cancelling_a_terminal_run_is_a_200_no_op(
    client: AsyncClient, manager: AsyncMock
) -> None:
    manager.acancel_answer_run.return_value = Mock(
        outcome="already_terminal", run=answer_run(status="succeeded", result=stored_result())
    )

    response = await client.delete(f"/web/answer/{RUN_ID}")

    assert response.status_code == 200
    assert response.json()["status"] == "succeeded"


async def test_a_trimmed_event_log_is_410(client: AsyncClient, service: AsyncMock) -> None:
    service.turn_for_run.return_value = linked_turn(
        answer_run(status="succeeded", events_trimmed_at=datetime.datetime.now(datetime.UTC))
    )

    response = await client.get(f"/web/answer/{RUN_ID}/events")

    assert response.status_code == 410


@pytest.fixture
async def scoped_client(manager: AsyncMock, test_config):
    """A client whose conversation service is real, over a store that must not run."""
    store = AsyncMock()
    application = create_app(include_web_app=True)
    manager.config = test_config
    application.state.manager = manager
    application.state.web_conversation_service = WebConversationService(
        store=store, max_turns=100, ttl_days=30, max_attachments=6
    )
    transport = ASGITransport(app=application)
    async with AsyncClient(
        transport=transport,
        base_url="http://test",
        cookies={"dlightrag_workspace": "default"},
    ) as created:
        created.store = store  # type: ignore[attr-defined]
        yield created


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/web/answer/{run_id}"),
        ("DELETE", "/web/answer/{run_id}"),
        ("GET", "/web/answer/{run_id}/events"),
        ("GET", "/web/runs/{run_id}/attachments/1"),
        ("GET", "/web/runs/{run_id}/attachments/1/thumbnail"),
    ],
    ids=["status", "cancel", "events", "attachment", "thumbnail"],
)
@pytest.mark.parametrize(
    "run_id", ["not-a-uuid", "019", RUN_ID[:-1]], ids=["text", "short", "trunc"]
)
async def test_a_malformed_run_id_is_the_same_opaque_404(
    scoped_client: AsyncClient, manager: AsyncMock, method: str, path: str, run_id: str
) -> None:
    """An unparseable id is unknown, not a server fault, and never reaches storage."""
    response = await scoped_client.request(method, path.format(run_id=run_id))

    assert response.status_code == 404
    scoped_client.store.find_turn_by_run.assert_not_awaited()  # type: ignore[attr-defined]
    manager.acancel_answer_run.assert_not_awaited()
    manager.asubscribe_answer_run.assert_not_awaited()


@pytest.mark.parametrize(
    ("header", "query", "expected"),
    [
        (None, None, 0),
        ("7", None, 7),
        (None, "3", 3),
        ("", None, 0),
    ],
    ids=["no-cursor", "last-event-id", "after", "blank-last-event-id"],
)
async def test_the_resume_cursor_comes_from_either_form(
    client: AsyncClient,
    manager: AsyncMock,
    header: str | None,
    query: str | None,
    expected: int,
) -> None:
    async def _events(**_kwargs: Any):
        return _empty_events()

    manager.asubscribe_answer_run.side_effect = _events
    url = f"/web/answer/{RUN_ID}/events" + (f"?after={query}" if query is not None else "")

    await client.get(url, headers={"Last-Event-ID": header} if header is not None else None)

    assert manager.asubscribe_answer_run.await_args.kwargs["after_sequence"] == expected


@pytest.mark.parametrize(
    ("header", "query", "status"),
    [
        ("2", "5", 400),
        (None, "abc", 400),
        (None, "-1", 400),
    ],
    ids=["conflicting-cursors", "non-numeric", "negative"],
)
async def test_an_unusable_cursor_never_subscribes(
    client: AsyncClient,
    manager: AsyncMock,
    header: str | None,
    query: str | None,
    status: int,
) -> None:
    url = f"/web/answer/{RUN_ID}/events" + (f"?after={query}" if query is not None else "")

    response = await client.get(
        url, headers={"Last-Event-ID": header} if header is not None else None
    )

    assert response.status_code == status
    manager.asubscribe_answer_run.assert_not_awaited()


async def _empty_events():
    return
    yield  # pragma: no cover - generator marker


# ---------------------------------------------------------------------------
# Event projection
# ---------------------------------------------------------------------------


def _event(sequence: int, event_type: str, payload: dict[str, Any]) -> AnswerRunEvent:
    return AnswerRunEvent(
        sequence=sequence,
        event_type=event_type,  # type: ignore[arg-type]
        payload=payload,
        created_at=datetime.datetime.now(datetime.UTC),
    )


async def _frames(events: list[AnswerRunEvent]) -> list[str]:
    async def _iterate():
        for event in events:
            yield event

    return [frame async for frame in stream_answer_events(_iterate())]


async def test_every_frame_carries_its_durable_sequence_as_the_sse_id() -> None:
    frames = await _frames(
        [
            _event(1, "progress", {"phase": "planning"}),
            _event(2, "token", {"text": "Rev"}),
            _event(3, "reset", {}),
            _event(4, "token", {"text": "Revenue"}),
            _event(5, "done", {"status": "succeeded", "result": stored_result()}),
        ]
    )

    assert [frame.split("\n")[0] for frame in frames] == [
        "id: 1",
        "id: 2",
        "id: 3",
        "id: 4",
        "id: 5",
    ]
    assert [frame.split("\n")[1] for frame in frames] == [
        "event: progress",
        "event: token",
        "event: reset",
        "event: token",
        "event: done",
    ]


async def test_a_token_frame_carries_only_the_text() -> None:
    (frame,) = await _frames([_event(1, "token", {"text": "Rev"})])

    assert json.loads(frame.split("data: ", 1)[1].strip()) == "Rev"


async def test_a_quiet_run_keeps_its_connection_alive_until_the_next_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A queued or silent run yields comments, and the next real event still follows."""
    monkeypatch.setattr("dlightrag.web.answer_events.SSE_KEEPALIVE_SECONDS", 0.01)
    released = asyncio.Event()

    async def _iterate():
        await released.wait()
        yield _event(1, "token", {"text": "Rev"})

    stream = stream_answer_events(_iterate())
    keepalive = await anext(stream)
    released.set()
    event = await anext(stream)
    await stream.aclose()

    assert keepalive == ": keepalive\n\n"
    assert keepalive.startswith(":")  # a comment consumes no durable sequence
    assert event.startswith("id: 1\nevent: token\n")


async def test_closing_the_event_stream_detaches_without_cancelling(
    client: AsyncClient, manager: AsyncMock
) -> None:
    """Disconnecting is a transport decision, never a decision about the run."""
    detached = asyncio.Event()

    async def _iterate():
        try:
            yield _event(1, "progress", {"phase": "planning"})
            yield _event(2, "token", {"text": "Rev"})
        finally:
            detached.set()

    async def _events(**_kwargs: Any):
        return _iterate()

    manager.asubscribe_answer_run.side_effect = _events

    async with client.stream("GET", f"/web/answer/{RUN_ID}/events") as response:
        assert response.status_code == 200
        async for _line in response.aiter_lines():
            break

    assert detached.is_set()
    manager.acancel_answer_run.assert_not_awaited()


async def test_a_failed_run_becomes_a_public_browser_error() -> None:
    (frame,) = await _frames(
        [_event(1, "error", {"kind": "answer_stream_failed", "message": "Service error."})]
    )

    payload = json.loads(frame.split("data: ", 1)[1].strip())
    assert payload == {"message": "Service error.", "error_kind": "answer_stream_failed"}


def test_the_done_event_is_derived_from_the_canonical_result() -> None:
    done = render_done_event(
        {"status": "succeeded", "result": stored_result()},
        downloadable_workspaces={"default"},
        visual_workspaces={"default"},
    )

    assert done.status == "succeeded"
    assert done.answer == "Revenue increased [1]."
    assert "/web/files/raw/report?workspace=default" in done.html
    assert "citation-badge" in done.html


def test_a_cancelled_run_carries_no_answer() -> None:
    done = render_done_event(
        {"status": "cancelled"}, downloadable_workspaces=None, visual_workspaces=None
    )

    assert done.status == "cancelled"
    assert done.html == ""
    assert done.answer == ""


# ---------------------------------------------------------------------------
# Turn projection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", ["queued", "running"])
def test_a_pending_turn_is_visible_without_an_answer(status: str) -> None:
    turn = project_conversation_turn(linked_turn(answer_run(status=status)))  # type: ignore[arg-type]

    assert turn.status == status
    assert turn.answer_run_id == RUN_ID
    assert turn.user_text == "What changed?"
    assert turn.assistant_text == ""
    assert turn.answer_html == ""
    assert turn.cancel_requested is False


def test_a_pending_turn_reports_a_requested_cancellation() -> None:
    turn = project_conversation_turn(
        linked_turn(
            answer_run(status="running", cancel_requested_at=datetime.datetime.now(datetime.UTC))
        )
    )

    assert turn.cancel_requested is True


def test_a_failed_turn_stays_visible_with_its_public_error() -> None:
    turn = project_conversation_turn(
        linked_turn(
            answer_run(
                status="failed",
                error_kind="answer_stream_failed",
                error_message="Service error.",
            )
        )
    )

    assert turn.status == "failed"
    assert turn.error_kind == "answer_stream_failed"
    assert turn.error_message == "Service error."
    assert turn.answer_html == ""


def test_a_succeeded_turn_renders_from_the_run_result() -> None:
    turn = project_conversation_turn(
        linked_turn(answer_run(status="succeeded", result=stored_result())),
        downloadable_workspaces={"default"},
        visual_workspaces={"default"},
    )

    assert turn.assistant_text == "Revenue increased [1]."
    assert "/web/files/raw/report?workspace=default" in turn.answer_html
    # The turn model never re-exposes stored source or principal state.
    assert "answer_sources" not in turn.model_dump()
    assert "principal_id" not in turn.model_dump_json()


def test_uploads_are_addressed_through_their_run() -> None:
    request = run_request(
        attachments=[
            {
                "digest": "a" * 64,
                "filename": "chart.png",
                "mime_type": "image/png",
                "ordinal": 1,
                "byte_size": 9,
            }
        ]
    )

    turn = project_conversation_turn(linked_turn(answer_run(request=request)))

    attachment = turn.user_attachments[0]
    assert attachment.url == f"/web/runs/{RUN_ID}/attachments/1"
    assert attachment.thumbnail_url == f"/web/runs/{RUN_ID}/attachments/1/thumbnail"
    assert attachment.kind == "image"
    assert attachment.byte_size == 9
    assert "digest" not in attachment.model_dump()
