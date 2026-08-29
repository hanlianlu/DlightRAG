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

from dlightrag.adapters.http.browser.answer_events import browser_frame, render_done_event
from dlightrag.adapters.http.browser.conversations import project_conversation_turn
from dlightrag.adapters.http.browser.routes import chat as chat_routes
from dlightrag.adapters.http.server import create_app
from dlightrag.adapters.http.streaming.answer_stream import follow_run_frames
from dlightrag.application.access import owner_id_from_user
from dlightrag.application.web_conversations import (
    CarriedAttachment,
    ConversationHead,
    ConversationSubmissionConflict,
    RecoveryTurnBatch,
    SubmissionSeed,
    WebConversationService,
)
from dlightrag.engine.runtime import AnswerRunEvent, IdempotencyKeyConflict
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
_REPORT_RESOURCE = "artifact-report"


def _with_primary_report(result: dict[str, Any], *, answer: str | None = None) -> dict[str, Any]:
    result["answer"] = (
        answer
        if answer is not None
        else (f"Revenue increased [1]. [View report](artifact:{_REPORT_RESOURCE})")
    )
    result["artifacts"] = [
        {
            "resource_id": _REPORT_RESOURCE,
            "role": "primary_report",
            "media_type": "text/markdown",
            "label": "View report",
            "filename": "report.md",
            "byte_size": 13,
            "digest": "a" * 64,
            "presentation": "markdown",
            "status": "available",
        }
    ]
    return result


@pytest.fixture
def service() -> AsyncMock:
    created = AsyncMock()
    created.start_answer.return_value = web_answer_submission(conversation_id=_CID)
    created.continue_answer.return_value = web_answer_submission(conversation_id=_CID)
    created.turn_for_run.return_value = linked_turn(conversation_id=_CID)
    return created


@pytest.fixture
def application_double() -> AsyncMock:
    created = AsyncMock()
    capability_view = answer_capability_view()
    created.answers = SimpleNamespace(
        capabilities=capability_view.read,
        cancel=AsyncMock(),
        steer=AsyncMock(
            return_value=SimpleNamespace(run_id=RUN_ID, control_sequence=1, kind="steer")
        ),
        children=AsyncMock(return_value=()),
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
    store = AsyncMock()
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


async def test_durable_recovery_reads_more_than_100_succeeded_turns_in_bounded_pages() -> None:
    now = datetime.datetime.now(datetime.UTC)
    store = AsyncMock()
    store.replay_answer_turn.return_value = None
    store.submission_seed.return_value = SubmissionSeed(
        head=ConversationHead(
            principal_id="anonymous",
            conversation_id=_CID,
            agent_session_id=_CID,
            agent_lane_id="main",
            content_revision=207,
            title="Conversation",
            created_at=now,
            updated_at=now,
        )
    )
    durable = [
        linked_turn(
            answer_run(
                status=("failed" if number in {50, 150} else "succeeded"),
                run_id=f"019893f4-0000-7000-8000-{number:012d}",
                request=run_request(query=f"q{number}"),
                result=(stored_result(f"a{number}") if number not in {50, 150} else None),
                error_kind=("failed" if number in {50, 150} else None),
                error_message=("failed" if number in {50, 150} else None),
            ),
            turn_number=number,
        )
        for number in range(1, 208)
    ]

    async def recovery_page(_principal: str, _conversation: str, *, page):
        selected = sorted(durable, key=lambda turn: turn.turn_number, reverse=True)
        if page.before_turn_number is not None:
            selected = [turn for turn in selected if turn.turn_number < page.before_turn_number]
        fetched = selected[: page.limit + 1]
        return RecoveryTurnBatch(
            tuple(fetched[: page.limit]), len(fetched) > page.limit, len(fetched)
        )

    store.recovery_page.side_effect = recovery_page
    store.create_answer_turn.return_value = None
    answers = FakeAnswers()
    service = WebConversationService(store=store, answers=answers, max_attachments=6)

    await service.start_answer(
        None,
        conversation_id=_CID,
        submission_id=SUBMISSION_ID,
        query="next",
        workspaces=["default"],
    )

    contents = [message["content"] for message in answers.prepared[0].history]
    assert len(contents) == 410
    assert contents[0] == "q1"
    assert contents[-1] == "a207"
    assert "q50" not in contents and "q150" not in contents
    assert store.recovery_page.await_count == 4
    assert all(call.kwargs["page"].limit == 64 for call in store.recovery_page.await_args_list)


@pytest.mark.parametrize(
    ("kind", "same_conversation", "include_answer"),
    [("follow_up", True, True), ("fork", False, False)],
)
async def test_service_continuation_uses_shared_answer_contract_and_branch_target(
    kind: str, same_conversation: bool, include_answer: bool
) -> None:
    store = AsyncMock()
    store.find_turn_by_run.return_value = linked_turn(
        answer_run(status="succeeded"), conversation_id=_CID
    )
    answers = AsyncMock()
    answers.continuation_request.return_value = SimpleNamespace()
    marker = web_answer_submission(conversation_id=_CID)
    answers.accept.return_value = marker
    service = WebConversationService(store=store, answers=answers, max_attachments=6)

    result = await service.continue_answer(
        None,
        parent_run_id=RUN_ID,
        submission_id=SUBMISSION_ID,
        query="What next?",
        kind=kind,
        authorized_workspaces=("default",),
    )

    assert result is marker
    assert answers.continuation_request.await_args.kwargs["include_answer"] is include_answer
    assert answers.continuation_request.await_args.kwargs["authorized_workspaces"] == ("default",)
    acceptor = answers.accept.await_args.kwargs["acceptor"]
    assert (acceptor.conversation_id == _CID) is same_conversation
    assert acceptor.create_conversation is (not same_conversation)


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
    store.submission_seed.assert_not_awaited()
    assert store.replay_answer_turn.await_args.kwargs["conversation_id"] == generated_id
    assert store.create_answer_turn.await_args.kwargs["conversation_id"] == generated_id
    assert store.create_answer_turn.await_args.kwargs["create_conversation"] is True
    assert answers.prepared[0].agent_session_id == generated_id
    assert answers.prepared[0].agent_lane_id == "main"


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
    store.submission_seed.assert_not_awaited()
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
# Status, controls, continuation, cancellation, and events
# ---------------------------------------------------------------------------


async def test_web_projects_resume_steer_and_child_roster(
    client: AsyncClient, application_double: AsyncMock
) -> None:
    resumed = await client.post(f"/web/api/answer/{RUN_ID}/resume")
    assert resumed.status_code == 200
    assert resumed.json()["answer_run_id"] == RUN_ID

    steered = await client.post(
        f"/web/api/answer/{RUN_ID}/steer", json={"content": "Focus on risks"}
    )
    assert steered.status_code == 202
    assert steered.json()["control_sequence"] == 1

    children = await client.get(f"/web/api/answer/{RUN_ID}/children")
    assert children.status_code == 200
    assert children.json() == {"run_id": RUN_ID, "children": []}
    application_double.answers.steer.assert_awaited_once()


@pytest.mark.parametrize(
    ("operation", "kind"),
    [("follow-up", "follow_up"), ("fork", "fork")],
)
async def test_web_continuations_return_a_linked_descriptor(
    client: AsyncClient,
    service: AsyncMock,
    operation: str,
    kind: str,
) -> None:
    response = await client.post(
        f"/web/api/answer/{RUN_ID}/{operation}",
        json={"content": "What next?", "submission_id": SUBMISSION_ID},
    )

    assert response.status_code == 202
    assert response.json()["conversation"]["conversation_id"] == _CID
    assert service.continue_answer.await_args.kwargs["kind"] == kind
    assert service.continue_answer.await_args.kwargs["parent_run_id"] == RUN_ID


async def test_status_projects_the_linked_turn(client: AsyncClient) -> None:
    response = await client.get(f"/web/api/answer/{RUN_ID}")

    assert response.status_code == 200
    body = response.json()
    assert body["answer_run_id"] == RUN_ID
    assert body["status"] == "queued"
    assert body["user_text"] == "What changed?"


@pytest.mark.parametrize("path", ["", "/events", "/artifacts/missing/presentation"])
async def test_a_run_this_principal_does_not_own_is_404(
    client: AsyncClient, service: AsyncMock, path: str
) -> None:
    service.turn_for_run.return_value = None

    response = await client.get(f"/web/api/answer/{RUN_ID}{path}")

    assert response.status_code == 404


async def test_general_artifact_route_returns_markdown_presentation(
    client: AsyncClient, service: AsyncMock, application_double: AsyncMock
) -> None:
    result = _with_primary_report(
        stored_result(answer=""), answer=f"[View report](artifact:{_REPORT_RESOURCE})"
    )
    service.turn_for_run.return_value = linked_turn(answer_run(status="succeeded", result=result))
    application_double.answers.read_artifact = AsyncMock(return_value=b"# Title\n\nBody")

    response = await client.get(
        f"/web/api/answer/{RUN_ID}/artifacts/{_REPORT_RESOURCE}/presentation"
    )

    assert response.status_code == 200
    assert response.headers["cache-control"] == "private, no-store"
    body = response.json()
    assert body["answer_text"] == "# Title\n\nBody"
    assert "<h1>Title</h1>" in body["parts"][0]["html"]
    assert "<p>Body</p>" in body["parts"][0]["html"]
    assert body["artifacts"][0]["role"] == "primary_report"
    application_double.answers.read_artifact.assert_awaited_once()


async def test_browser_artifact_data_is_attachment_nosniff_and_no_store(
    client: AsyncClient, service: AsyncMock, application_double: AsyncMock
) -> None:
    result = _with_primary_report(stored_result())
    result["artifacts"][0]["media_type"] = "text/html"
    result["artifacts"][0]["presentation"] = "html"
    result["artifacts"][0]["filename"] = "report.html"
    service.turn_for_run.return_value = linked_turn(answer_run(status="succeeded", result=result))
    application_double.answers.artifact_size = AsyncMock(return_value=13)

    async def stream():
        yield b"<h1>HTML</h1>"

    application_double.answers.open_artifact = AsyncMock(return_value=stream())

    response = await client.get(f"/web/api/answer/{RUN_ID}/artifacts/{_REPORT_RESOURCE}")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/octet-stream")
    assert response.headers["content-disposition"].startswith("attachment")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "private, no-store"


async def test_browser_svg_artifact_is_inline_only_under_an_inert_document_policy(
    client: AsyncClient, service: AsyncMock, application_double: AsyncMock
) -> None:
    result = _with_primary_report(stored_result())
    result["artifacts"][0].update(
        media_type="image/svg+xml", presentation="image", filename="chart.svg"
    )
    service.turn_for_run.return_value = linked_turn(answer_run(status="succeeded", result=result))
    application_double.answers.artifact_size = AsyncMock(return_value=46)

    async def stream():
        yield b'<svg xmlns="http://www.w3.org/2000/svg"></svg>'

    application_double.answers.open_artifact = AsyncMock(return_value=stream())

    response = await client.get(f"/web/api/answer/{RUN_ID}/artifacts/{_REPORT_RESOURCE}")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/svg+xml")
    assert response.headers["content-disposition"].startswith("inline")
    assert response.headers["content-security-policy"] == (
        "sandbox; default-src 'none'; img-src data:"
    )
    assert response.headers["x-content-type-options"] == "nosniff"


async def test_general_artifact_presentation_is_404_without_a_descriptor(
    client: AsyncClient, service: AsyncMock
) -> None:
    service.turn_for_run.return_value = linked_turn(
        answer_run(status="succeeded", result=stored_result())
    )

    response = await client.get(f"/web/api/answer/{RUN_ID}/artifacts/missing/presentation")

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


def test_memory_operation_frame_is_allowlisted_and_marks_only_live_events() -> None:
    payload = {
        "operation": "remember",
        "outcome": "changed",
        "change_id": "change-1",
        "memory_ids": ["memory-1"],
        "body": "Use Chinese.",
        "owner_id": "must-not-leak",
    }
    replay = browser_frame(_event(4, "memory_operation_settled", payload), live_after=4)
    live = browser_frame(_event(5, "memory_operation_settled", payload), live_after=4)

    replay_payload = json.loads(replay.split("data: ", 1)[1].strip())
    live_payload = json.loads(live.split("data: ", 1)[1].strip())
    assert replay_payload["live"] is False
    assert live_payload["live"] is True
    assert "owner_id" not in live_payload


async def test_tool_progress_frame_projects_metadata_without_raw_output() -> None:
    (frame,) = await _frames(
        [
            _event(
                1,
                "tool_progress",
                {
                    "tool_name": "bash",
                    "elapsed_ms": 125.0,
                    "output_bytes": 4096,
                    "spill_state": "staging",
                    "stdout": "secret output",
                    "stderr": "secret error",
                },
            )
        ]
    )

    assert json.loads(frame.split("data: ", 1)[1].strip()) == {
        "tool_name": "bash",
        "elapsed_ms": 125.0,
        "output_bytes": 4096,
        "spill_state": "staging",
    }


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
    result["evidence_images"] = [
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
    assert done.presentation.artifacts == []
    assert done.presentation.sources[0].download_url == (
        "/web/api/files/raw/report?workspace=default"
    )
    assert "citation-badge" in done.presentation.parts[0].html
    assert done.presentation.evidence_images[0].url.startswith("/web/api/images/default/chunk-1")


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
    assert turn.presentation.artifacts == []
    assert turn.presentation.sources[0].download_url == (
        "/web/api/files/raw/report?workspace=default"
    )
    # The turn model never re-exposes stored source or principal state.
    assert "answer_sources" not in turn.model_dump()
    assert "principal_id" not in turn.model_dump_json()


def test_a_succeeded_turn_exposes_primary_report_as_an_artifact_part() -> None:
    result = _with_primary_report(stored_result())
    turn = project_conversation_turn(
        linked_turn(answer_run(status="succeeded", result=result)),
    )

    assert turn.presentation is not None
    assert turn.presentation.artifacts[0].role == "primary_report"
    assert turn.presentation.parts[-1].artifact is not None
    assert turn.presentation.parts[-1].artifact.resource_id == _REPORT_RESOURCE
    assert turn.assistant_text == result["answer"]


def test_the_done_event_carries_primary_report_only_as_an_artifact() -> None:
    result = _with_primary_report(stored_result())
    done = render_done_event(
        {"status": "succeeded", "result": result},
        downloadable_workspaces=None,
        visual_workspaces=None,
        run_id=RUN_ID,
    )

    assert done.presentation is not None
    assert done.presentation.artifacts[0].role == "primary_report"
    assert done.presentation.artifacts[0].data_url is not None
    assert done.presentation.artifacts[0].data_url.startswith("/web/api/answer/")
    assert "primary_report" not in done.presentation.model_dump()


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
    store.submission_seed.return_value = SubmissionSeed(
        head=ConversationHead(
            principal_id="anonymous",
            conversation_id=_CID,
            agent_session_id=_CID,
            agent_lane_id="main",
            content_revision=1,
            title="Conversation",
            created_at=now,
            updated_at=now,
        ),
        attachments=(
            CarriedAttachment(
                run_id=origin_run_id,
                source_ordinal=3,
                digest="b" * 64,
                filename="prior.png",
                mime_type="image/png",
                byte_size=5,
            ),
        ),
    )
    store.recovery_page.return_value = RecoveryTurnBatch((), False, 0)
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


async def test_terminal_turns_project_history_from_the_accepted_envelope() -> None:
    """Blocker 1 regression: the terminal transition clears prepared input,

    so Web continuity must project the prior query from the durable accepted
    envelope instead of the cleared prepared input.
    """
    import dataclasses

    now = datetime.datetime.now(datetime.UTC)
    store = AsyncMock()
    store.submission_seed.return_value = SubmissionSeed(
        head=ConversationHead(
            principal_id="anonymous",
            conversation_id=_CID,
            agent_session_id=_CID,
            agent_lane_id="main",
            content_revision=1,
            title="Conversation",
            created_at=now,
            updated_at=now,
        )
    )
    recovered_turn = linked_turn(
        dataclasses.replace(
            answer_run(
                status="succeeded",
                accepted={
                    "query": "What changed?",
                    "workspaces": ["default"],
                    "mode": "auto",
                    "attachments": [],
                },
                result=stored_result(),
            ),
            prepared_input=None,
        )
    )
    store.recovery_page.return_value = RecoveryTurnBatch((recovered_turn,), False, 1)
    store.replay_answer_turn.return_value = None
    store.create_answer_turn.return_value = None
    answers = FakeAnswers()
    service = WebConversationService(store=store, answers=answers, max_attachments=6)

    await service.start_answer(
        None,
        conversation_id=_CID,
        submission_id=SUBMISSION_ID,
        query="And now?",
        workspaces=["default"],
    )

    request = answers.prepared[0]
    assert [message["content"] for message in request.history] == [
        "What changed?",
        "Revenue increased [1].",
    ]


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
