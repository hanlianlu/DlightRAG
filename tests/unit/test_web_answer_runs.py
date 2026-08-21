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
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from httpx import ASGITransport, AsyncClient

from dlightrag.access import owner_id_from_user
from dlightrag.api.answer_stream import follow_run_frames
from dlightrag.api.server import create_app
from dlightrag.runtime import AnswerRunEvent, IdempotencyKeyConflict
from dlightrag.web.answer_events import browser_frame, render_done_event
from dlightrag.web.conversation_models import (
    ConversationSnapshot,
    ConversationSubmissionConflict,
)
from dlightrag.web.conversations import WebConversationService, project_conversation_turn
from dlightrag.web.routes import chat as chat_routes
from tests.unit.conftest import answer_capability_view
from tests.unit.web.answer_run_fixtures import (
    RUN_ID,
    SUBMISSION_ID,
    TURN_ID,
    FakeAnswers,
    answer_run,
    answer_turn_creation,
    input_artifact,
    linked_turn,
    run_request,
    stored_result,
    web_answer_submission,
)

_CID = "00000000-0000-0000-0000-000000000001"
_ANONYMOUS = owner_id_from_user(None)
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
def application_double() -> AsyncMock:
    created = AsyncMock()
    capability_view = answer_capability_view()
    created.answers = SimpleNamespace(
        capabilities=capability_view.read,
        cancel=AsyncMock(),
        subscribe=MagicMock(),
    )
    created.corpora = SimpleNamespace(
        alist_workspace_records=AsyncMock(return_value=[{"workspace": "default"}])
    )
    return created


@pytest.fixture
async def client(service: AsyncMock, application_double: AsyncMock, test_config):
    application = create_app(include_web_app=True)
    application_double.config = test_config
    application_double.web_conversations = service
    application.state.application = application_double
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
    response = await client.post("/web/api/answer", json=_BODY)

    assert response.status_code == 202
    body = response.json()
    assert body["run_id"] == RUN_ID
    assert body["status"] == "queued"
    assert body["turn_id"] == TURN_ID
    assert body["events_url"] == f"/web/api/answer/{RUN_ID}/events"
    assert body["cancel_url"] == f"/web/api/answer/{RUN_ID}"
    assert body["conversation"]["conversation_id"] == _CID
    # The 202 body is the whole answer contract: nothing is streamed by the
    # request that created the run.
    assert "html" not in body
    assert response.headers["content-type"].startswith("application/json")


async def test_submission_passes_the_submission_id_as_the_run_key(
    client: AsyncClient, service: AsyncMock
) -> None:
    await client.post("/web/api/answer", json=_BODY)

    kwargs = service.start_answer.await_args.kwargs
    assert kwargs["submission_id"] == SUBMISSION_ID
    assert kwargs["conversation_id"] == _CID
    assert kwargs["query"] == "What changed?"
    assert list(kwargs["workspaces"]) == ["default"]


async def test_first_submission_can_atomically_create_its_conversation(
    client: AsyncClient, service: AsyncMock
) -> None:
    created_conversation_id = "00000000-0000-4000-8000-000000000099"
    service.start_answer.return_value = web_answer_submission(
        conversation_id=created_conversation_id
    )

    response = await client.post(
        "/web/api/answer",
        json={key: value for key, value in _BODY.items() if key != "conversation_id"},
    )

    assert response.status_code == 202
    assert response.json()["conversation"]["conversation_id"] == created_conversation_id
    assert service.start_answer.await_args.kwargs["conversation_id"] is None


async def test_submission_canonicalizes_display_workspaces_before_access_and_service(
    client: AsyncClient,
    service: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    enforce = AsyncMock()
    monkeypatch.setattr(chat_routes, "enforce_web_access", enforce)

    response = await client.post(
        "/web/api/answer",
        json={**_BODY, "workspaces": ["Finance Reports"]},
    )

    assert response.status_code == 202
    enforce.assert_awaited_once()
    assert enforce.await_args is not None
    assert enforce.await_args.args[2] == "finance_reports"
    assert service.start_answer.await_args.kwargs["workspaces"] == ["finance_reports"]


async def test_replaying_a_submission_returns_the_authoritative_run(
    client: AsyncClient, service: AsyncMock
) -> None:
    service.start_answer.return_value = web_answer_submission(
        conversation_id=_CID, run=answer_run(status="running")
    )

    response = await client.post("/web/api/answer", json=_BODY)

    # A replay is accepted work too, so it reports the same 202 as the original
    # submission and simply carries the run's current status.
    assert response.status_code == 202
    assert response.json()["run_id"] == RUN_ID
    assert response.json()["status"] == "running"


async def test_service_replays_before_preparing_resolved_run_input() -> None:
    now = datetime.datetime.now(datetime.UTC)
    store = AsyncMock()
    store.snapshot.return_value = ConversationSnapshot(
        principal_id="anonymous",
        conversation_id=_CID,
        content_revision=1,
        title="Conversation",
        created_at=now,
        updated_at=now,
        turns=(),
    )
    existing = answer_turn_creation(
        conversation_id=_CID,
        run=answer_run(status="running"),
        replayed=True,
    )
    store.replay_answer_turn.return_value = existing
    answers = FakeAnswers()
    service = WebConversationService(
        store=store,
        answers=answers,
        max_attachments=6,
    )

    submission = await service.start_answer(
        None,
        conversation_id=_CID,
        submission_id=SUBMISSION_ID,
        query="What changed?",
        workspaces=["default"],
    )

    assert submission is not None
    assert submission.run.status == "running"
    assert answers.prepared == []
    store.create_answer_turn.assert_not_awaited()


async def test_first_submission_uses_one_stable_server_conversation_and_atomic_store_write() -> (
    None
):
    store = AsyncMock()
    store.replay_answer_turn.return_value = None

    async def create_turn(**kwargs):
        return answer_turn_creation(conversation_id=kwargs["conversation_id"])

    store.create_answer_turn.side_effect = create_turn
    answers = FakeAnswers()
    service = WebConversationService(
        store=store,
        answers=answers,
        max_attachments=6,
    )

    submission = await service.start_answer(
        None,
        conversation_id=None,
        submission_id=SUBMISSION_ID,
        query="What changed?",
        workspaces=["default"],
    )

    assert submission is not None
    generated_id = submission.conversation.conversation_id
    assert generated_id
    store.snapshot.assert_not_awaited()
    assert store.replay_answer_turn.await_args.kwargs["conversation_id"] == generated_id
    assert store.create_answer_turn.await_args.kwargs["conversation_id"] == generated_id
    assert store.create_answer_turn.await_args.kwargs["create_conversation"] is True


async def test_replaying_first_submission_returns_its_created_conversation_before_preparation() -> (
    None
):
    store = AsyncMock()

    async def replay_turn(**kwargs):
        return answer_turn_creation(
            conversation_id=kwargs["conversation_id"],
            run=answer_run(status="running"),
            replayed=True,
        )

    store.replay_answer_turn.side_effect = replay_turn
    answers = FakeAnswers()
    service = WebConversationService(
        store=store,
        answers=answers,
        max_attachments=6,
    )

    submission = await service.start_answer(
        None,
        conversation_id=None,
        submission_id=SUBMISSION_ID,
        query="What changed?",
        workspaces=["default"],
    )

    assert submission is not None
    assert (
        submission.conversation.conversation_id
        == (store.replay_answer_turn.await_args.kwargs["conversation_id"])
    )
    assert answers.prepared == []
    store.snapshot.assert_not_awaited()
    store.create_answer_turn.assert_not_awaited()


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

    response = await client.post("/web/api/answer", json=_BODY)

    assert response.status_code == 409


async def test_submission_to_an_unknown_conversation_is_404(
    client: AsyncClient, service: AsyncMock
) -> None:
    service.start_answer.return_value = None

    response = await client.post("/web/api/answer", json=_BODY)

    assert response.status_code == 404


async def test_an_empty_question_is_rejected_before_acceptance(
    client: AsyncClient, service: AsyncMock
) -> None:
    response = await client.post("/web/api/answer", json={**_BODY, "query": "  "})

    assert response.status_code == 422
    service.start_answer.assert_not_awaited()


# ---------------------------------------------------------------------------
# Status, cancellation, and events
# ---------------------------------------------------------------------------


async def test_status_projects_the_linked_turn(client: AsyncClient) -> None:
    response = await client.get(f"/web/api/answer/{RUN_ID}")

    assert response.status_code == 200
    body = response.json()
    assert body["answer_run_id"] == RUN_ID
    assert body["status"] == "queued"
    assert body["user_text"] == "What changed?"


@pytest.mark.parametrize("path", ["", "/events", "/report"])
async def test_a_run_this_principal_does_not_own_is_404(
    client: AsyncClient, service: AsyncMock, path: str
) -> None:
    service.turn_for_run.return_value = None

    response = await client.get(f"/web/api/answer/{RUN_ID}{path}")

    assert response.status_code == 404


async def test_the_report_route_returns_structured_safe_presentation(
    client: AsyncClient, service: AsyncMock, application_double: AsyncMock
) -> None:
    result = stored_result(answer="")
    result["primary_report"] = "primary_report"
    service.turn_for_run.return_value = linked_turn(answer_run(status="succeeded", result=result))
    application_double.answers.read_artifact = AsyncMock(return_value=b"# Title\n\nBody")

    response = await client.get(f"/web/api/answer/{RUN_ID}/report")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["answer_text"] == "# Title\n\nBody"
    assert "<h1>Title</h1>" in response.json()["answer_html"]
    assert "<p>Body</p>" in response.json()["answer_html"]
    application_double.answers.read_artifact.assert_awaited_once()


async def test_the_report_route_is_404_without_a_handle(
    client: AsyncClient, service: AsyncMock
) -> None:
    service.turn_for_run.return_value = linked_turn(
        answer_run(status="succeeded", result=stored_result())
    )

    response = await client.get(f"/web/api/answer/{RUN_ID}/report")

    assert response.status_code == 404


async def test_cancelling_an_unowned_run_never_reaches_answer_service(
    client: AsyncClient, service: AsyncMock, application_double: AsyncMock
) -> None:
    service.turn_for_run.return_value = None

    response = await client.delete(f"/web/api/answer/{RUN_ID}")

    assert response.status_code == 404
    application_double.answers.cancel.assert_not_awaited()


async def test_cancelling_a_running_run_reports_the_pending_request(
    client: AsyncClient, application_double: AsyncMock
) -> None:
    running = answer_run(status="running", cancel_requested_at=datetime.datetime.now(datetime.UTC))
    application_double.answers.cancel.return_value = Mock(outcome="pending", run=running)

    response = await client.delete(f"/web/api/answer/{RUN_ID}")

    assert response.status_code == 202
    assert response.json()["cancel_requested"] is True
    assert response.json()["status"] == "running"


async def test_cancelling_a_terminal_run_is_a_200_no_op(
    client: AsyncClient, application_double: AsyncMock
) -> None:
    application_double.answers.cancel.return_value = Mock(
        outcome="already_terminal", run=answer_run(status="succeeded", result=stored_result())
    )

    response = await client.delete(f"/web/api/answer/{RUN_ID}")

    assert response.status_code == 200
    assert response.json()["status"] == "succeeded"


async def test_a_trimmed_event_log_is_410(client: AsyncClient, service: AsyncMock) -> None:
    service.turn_for_run.return_value = linked_turn(
        answer_run(status="succeeded", events_trimmed_at=datetime.datetime.now(datetime.UTC))
    )

    response = await client.get(f"/web/api/answer/{RUN_ID}/events")

    assert response.status_code == 410


@pytest.fixture
async def scoped_client(application_double: AsyncMock, test_config):
    """A client whose conversation service is real, over a store that must not run."""
    store = AsyncMock()
    application = create_app(include_web_app=True)
    application_double.config = test_config
    application_double.web_conversations = WebConversationService(
        store=store,
        answers=FakeAnswers(),
        max_attachments=6,
    )
    application.state.application = application_double
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
        ("GET", "/web/api/answer/{run_id}"),
        ("DELETE", "/web/api/answer/{run_id}"),
        ("GET", "/web/api/answer/{run_id}/events"),
        ("GET", "/web/api/runs/{run_id}/attachments/1"),
        ("GET", "/web/api/runs/{run_id}/attachments/1/thumbnail"),
    ],
    ids=["status", "cancel", "events", "attachment", "thumbnail"],
)
@pytest.mark.parametrize(
    "run_id", ["not-a-uuid", "019", RUN_ID[:-1]], ids=["text", "short", "trunc"]
)
async def test_a_malformed_run_id_is_the_same_opaque_404(
    scoped_client: AsyncClient, application_double: AsyncMock, method: str, path: str, run_id: str
) -> None:
    """An unparseable id is unknown, not a server fault, and never reaches storage."""
    response = await scoped_client.request(method, path.format(run_id=run_id))

    assert response.status_code == 404
    scoped_client.store.find_turn_by_run.assert_not_awaited()  # type: ignore[attr-defined]
    application_double.answers.cancel.assert_not_awaited()
    application_double.answers.subscribe.assert_not_called()


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
    application_double: AsyncMock,
    header: str | None,
    query: str | None,
    expected: int,
) -> None:
    def _events(**_kwargs: Any):
        return _empty_events()

    application_double.answers.subscribe.side_effect = _events
    url = f"/web/api/answer/{RUN_ID}/events" + (f"?after={query}" if query is not None else "")

    await client.get(url, headers={"Last-Event-ID": header} if header is not None else None)

    assert application_double.answers.subscribe.call_args.kwargs["after_sequence"] == expected


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
    application_double: AsyncMock,
    header: str | None,
    query: str | None,
    status: int,
) -> None:
    url = f"/web/api/answer/{RUN_ID}/events" + (f"?after={query}" if query is not None else "")

    response = await client.get(
        url, headers={"Last-Event-ID": header} if header is not None else None
    )

    assert response.status_code == status
    application_double.answers.subscribe.assert_not_called()


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

    return [frame async for frame in follow_run_frames(_iterate(), browser_frame)]


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


async def test_closing_the_event_stream_detaches_without_cancelling(
    client: AsyncClient, application_double: AsyncMock
) -> None:
    """Disconnecting is a transport decision, never a decision about the run."""
    detached = asyncio.Event()

    async def _iterate():
        try:
            yield _event(1, "progress", {"phase": "planning"})
            yield _event(2, "token", {"text": "Rev"})
        finally:
            detached.set()

    def _events(**_kwargs: Any):
        return _iterate()

    application_double.answers.subscribe.side_effect = _events

    async with client.stream("GET", f"/web/api/answer/{RUN_ID}/events") as response:
        assert response.status_code == 200
        async for _line in response.aiter_lines():
            break

    assert detached.is_set()
    application_double.answers.cancel.assert_not_awaited()


async def test_a_failed_run_becomes_a_public_browser_error() -> None:
    (frame,) = await _frames(
        [_event(1, "error", {"kind": "answer_stream_failed", "message": "Service error."})]
    )

    payload = json.loads(frame.split("data: ", 1)[1].strip())
    assert payload == {"message": "Service error.", "error_kind": "answer_stream_failed"}


def test_the_done_event_is_derived_from_the_canonical_result() -> None:
    result = stored_result()
    result["answer_images"] = [
        {
            "id": "chart",
            "chunk_id": "chunk-1",
            "workspace": "default",
            "source_ref": "1",
            "label": "Chart",
            "answer_image_sent": True,
        }
    ]
    done = render_done_event(
        {"status": "succeeded", "result": result},
        downloadable_workspaces={"default"},
        visual_workspaces={"default"},
    )

    assert done.status == "succeeded"
    assert done.presentation is not None
    assert done.presentation.answer_text == "Revenue increased [1]."
    assert done.presentation.primary_report is None
    assert done.presentation.sources[0].download_url == (
        "/web/api/files/raw/report?workspace=default"
    )
    assert "citation-badge" in done.presentation.answer_html
    assert done.presentation.answer_images[0].url.startswith("/web/api/images/default/chunk-1")


def test_a_cancelled_run_carries_no_answer() -> None:
    done = render_done_event(
        {"status": "cancelled"}, downloadable_workspaces=None, visual_workspaces=None
    )

    assert done.status == "cancelled"
    assert done.presentation is None


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
    assert turn.presentation is None
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
    assert turn.presentation is None


def test_a_succeeded_turn_renders_from_the_run_result() -> None:
    turn = project_conversation_turn(
        linked_turn(answer_run(status="succeeded", result=stored_result())),
        downloadable_workspaces={"default"},
        visual_workspaces={"default"},
    )

    assert turn.assistant_text == "Revenue increased [1]."
    assert turn.presentation is not None
    assert turn.presentation.primary_report is None
    assert turn.presentation.sources[0].download_url == (
        "/web/api/files/raw/report?workspace=default"
    )
    # The turn model never re-exposes stored source or principal state.
    assert "answer_sources" not in turn.model_dump()
    assert "principal_id" not in turn.model_dump_json()


def test_a_succeeded_turn_exposes_the_primary_report_handle() -> None:
    result = stored_result()
    result["primary_report"] = "primary_report"
    turn = project_conversation_turn(
        linked_turn(answer_run(status="succeeded", result=result)),
    )

    assert turn.presentation is not None
    assert turn.presentation.primary_report == "primary_report"
    assert turn.assistant_text == "Revenue increased [1]."


def test_the_done_event_carries_the_primary_report_handle() -> None:
    result = stored_result()
    result["primary_report"] = "primary_report"
    done = render_done_event(
        {"status": "succeeded", "result": result},
        downloadable_workspaces=None,
        visual_workspaces=None,
    )

    assert done.presentation is not None
    assert done.presentation.primary_report == "primary_report"
    assert "primary_report" not in done.presentation.answer_html


async def test_terminal_attachment_is_read_through_the_answer_service() -> None:
    store = AsyncMock()
    store.find_turn_by_run.return_value = linked_turn(
        answer_run(status="succeeded", result=stored_result())
    )
    answers = FakeAnswers({(RUN_ID, 0): input_artifact(content=b"content")})
    service = WebConversationService(
        store=store,
        answers=answers,
        max_attachments=6,
    )

    attachment = await service.attachment(None, RUN_ID, 0)

    assert attachment is not None
    assert attachment.content == b"content"
    assert answers.reads == [(_ANONYMOUS, RUN_ID, 0)]


async def test_an_unowned_run_never_reads_an_input_artifact() -> None:
    store = AsyncMock()
    store.find_turn_by_run.return_value = None
    answers = FakeAnswers({(RUN_ID, 0): input_artifact(content=b"content")})
    service = WebConversationService(
        store=store,
        answers=answers,
        max_attachments=6,
    )

    assert await service.attachment(None, RUN_ID, 0) is None
    assert answers.reads == []


async def test_history_attachments_load_from_the_run_that_accepted_them() -> None:
    origin_run_id = "019893f4-0000-7000-8000-0000000000ff"
    now = datetime.datetime.now(datetime.UTC)
    store = AsyncMock()
    store.snapshot.return_value = ConversationSnapshot(
        principal_id="anonymous",
        conversation_id=_CID,
        content_revision=1,
        title="Conversation",
        created_at=now,
        updated_at=now,
        turns=(
            linked_turn(
                answer_run(
                    status="succeeded",
                    run_id=origin_run_id,
                    request=run_request(
                        attachments=[
                            {
                                "digest": "b" * 64,
                                "filename": "prior.png",
                                "mime_type": "image/png",
                                "ordinal": 3,
                                "byte_size": 5,
                            }
                        ]
                    ),
                    result=stored_result(),
                )
            ),
        ),
    )
    store.replay_answer_turn.return_value = None
    store.create_answer_turn.return_value = None
    answers = FakeAnswers(
        {
            (origin_run_id, 3): input_artifact(
                content=b"prior-bytes",
                ordinal=3,
                filename="prior.png",
                mime_type="image/png",
                digest="b" * 64,
            )
        }
    )
    service = WebConversationService(
        store=store,
        answers=answers,
        max_attachments=6,
    )

    await service.start_answer(
        None,
        conversation_id=_CID,
        submission_id=SUBMISSION_ID,
        query="And now?",
        workspaces=["default"],
    )

    request = answers.prepared[0]
    # The carried reference is re-ordinalled for this run, while its bytes stay
    # addressed by the run and ordinal that accepted them.
    (carried,) = request.history_resources
    assert (carried.source_ordinal, carried.digest) == (3, "b" * 64)
    assert carried.run_id == origin_run_id


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
    assert attachment.url == f"/web/api/runs/{RUN_ID}/attachments/1"
    assert attachment.thumbnail_url == f"/web/api/runs/{RUN_ID}/attachments/1/thumbnail"
    assert attachment.kind == "image"
    assert attachment.byte_size == 9
    assert "digest" not in attachment.model_dump()
