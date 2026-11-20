# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the Web conversation lifecycle routes and their failure contract."""

import datetime
import json
from typing import Any, cast
from unittest.mock import AsyncMock
from uuid import UUID

import asyncpg
import pytest
from httpx import ASGITransport, AsyncClient

from dlightrag.adapters.http.server import create_app
from dlightrag.adapters.postgres.web import web_conversations as pg_web
from dlightrag.application.answer_runs.service import AgentControlReceipt, AnswerService
from dlightrag.application.config.sections import DlightragConfig
from dlightrag.application.errors import ApplicationInputError
from dlightrag.application.runs import RunView
from dlightrag.application.web_conversations.models import (
    CommandTone,
    ConversationCommand,
    ConversationCursor,
    ConversationCursorCodec,
    ConversationGoal,
    ConversationHead,
    ConversationHistoryCursor,
    ConversationHistoryCursorCodec,
    ConversationHistoryPage,
    ConversationHistoryPageRequest,
    ConversationPage,
    ConversationPageRequest,
    ConversationQueueConflict,
    ConversationSessionBusy,
    ConversationSummary,
    LinkedTurn,
    LiveAnswerRun,
    QueuedMessage,
    SubmissionSeed,
    UnknownConversationCommand,
    WebRunActivity,
)
from dlightrag.application.web_conversations.service import WebAnswerDelivery
from dlightrag.engine.answer.goal import GOAL_NOT_FOUND, GoalError
from tests.support.application_double import application_double
from tests.support.config_helpers import mutate_config
from tests.support.pg_fakes import FakePool
from tests.unit.conftest import answer_capabilities
from tests.unit.web.answer_run_fixtures import FakeAnswers, web_answer_submission

_CID = "00000000-0000-0000-0000-000000000001"
_MID = "00000000-0000-0000-0000-000000000002"


# ---------------------------------------------------------------------------
# Route-level fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def application(test_config: DlightragConfig) -> Any:
    application = application_double(test_config)
    application.answers.capabilities.return_value = answer_capabilities()
    return application


@pytest.fixture
def conversation_service(application: Any) -> Any:
    """The Application's autospecced WebConversationService, as the routes reach it."""
    service = application.web_conversations
    now = datetime.datetime(2026, 7, 12, tzinfo=datetime.UTC)
    summary = {
        "conversation_id": _CID,
        "title": None,
        "created_at": now,
        "updated_at": now,
    }
    service.create.return_value = summary
    service.cursor_codec = ConversationCursorCodec(b"unit-test-cursor-secret")
    service.list.return_value = ConversationPage(
        items=(
            ConversationSummary(
                conversation_id=_CID,
                title=None,
                created_at=now,
                updated_at=now,
            ),
        ),
        next_cursor=None,
        fetched_rows=1,
    )
    service.history_cursor_codec = ConversationHistoryCursorCodec(
        b"unit-test-history-cursor-secret"
    )
    service.history.return_value = ConversationHistoryPage(
        conversation=ConversationHead(
            principal_id="anonymous",
            conversation_id=_CID,
            content_revision=0,
            title=None,
            created_at=now,
            updated_at=now,
            agent_session_id=_CID,
            agent_lane_id="main",
        ),
        turns=(),
        next_cursor=None,
        fetched_rows=0,
    )
    service.rename.return_value = {**summary, "title": "Renamed chat"}
    service.delete.return_value = True
    service.delete_all.return_value = 2
    service.start_answer.return_value = web_answer_submission(conversation_id=_CID)
    service.thumbnail.return_value = (b"derived-thumbnail", "image/jpeg")
    return service


@pytest.fixture
async def conversation_client(application: Any, conversation_service: Any):
    application.corpora.alist_workspace_records.return_value = [{"workspace": "default"}]
    app = create_app()
    app.state.application = application
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
async def cookie_conversation_client(
    test_config: DlightragConfig,
    application: Any,
    conversation_service: Any,
):
    mutate_config(test_config, "access.auth_mode", "simple")
    mutate_config(test_config, "access.api_token", "secret-token")
    app = create_app()
    app.state.application = application
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="https://app.example.com",
        follow_redirects=False,
    ) as client:
        login = await client.post("/web/login", data={"token": "secret-token", "next": "/web/"})
        assert login.status_code == 303
        yield client


# ---------------------------------------------------------------------------
# Lifecycle routes
# ---------------------------------------------------------------------------


async def test_create_ignores_client_identity_and_returns_server_uuid(
    conversation_client: AsyncClient,
    conversation_service: Any,
) -> None:
    response = await conversation_client.post(
        "/web/api/conversations",
        json={"principal_id": "attacker", "conversation_id": "client-selected"},
    )

    assert response.status_code == 201
    UUID(response.json()["conversation_id"])
    conversation_service.create.assert_awaited_once()
    (user,) = conversation_service.create.await_args.args
    assert user.user_id == "anonymous"


async def test_list_returns_only_service_projection(
    conversation_client: AsyncClient,
    conversation_service: Any,
) -> None:
    response = await conversation_client.get("/web/api/conversations")

    assert response.status_code == 200
    assert response.json()["next_cursor"] is None
    assert len(response.json()["items"]) == 1
    conversation_service.list.assert_awaited_once()


async def test_list_cursor_round_trips_as_paired_ordering_facts(
    conversation_client: AsyncClient,
    conversation_service: Any,
) -> None:
    updated_at = datetime.datetime(2026, 7, 12, 3, 4, 5, 123456, tzinfo=datetime.UTC)
    next_cursor = ConversationCursor(updated_at=updated_at, conversation_id=UUID(_CID))
    conversation_service.list.return_value = ConversationPage(
        items=(),
        next_cursor=next_cursor,
        fetched_rows=2,
    )
    first = await conversation_client.get("/web/api/conversations?limit=1")
    token = first.json()["next_cursor"]
    assert isinstance(token, str)

    conversation_service.list.return_value = ConversationPage(
        items=(),
        next_cursor=None,
        fetched_rows=0,
    )
    second = await conversation_client.get(
        "/web/api/conversations",
        params={"limit": 1, "cursor": token},
    )

    assert second.status_code == 200
    request = conversation_service.list.await_args.kwargs["page"]
    assert request == ConversationPageRequest(limit=1, cursor=next_cursor)


@pytest.mark.parametrize(
    ("query", "expected_status"),
    (
        pytest.param("limit=0", 422, id="limit-too-small"),
        pytest.param("limit=101", 422, id="limit-too-large"),
        pytest.param("limit=all", 422, id="limit-not-integer"),
        pytest.param("cursor=not-a-cursor", 422, id="malformed-cursor"),
    ),
)
async def test_invalid_page_inputs_are_422_before_the_service_runs(
    conversation_client: AsyncClient,
    conversation_service: Any,
    query: str,
    expected_status: int,
) -> None:
    response = await conversation_client.get(f"/web/api/conversations?{query}")

    assert response.status_code == expected_status
    conversation_service.list.assert_not_awaited()


async def test_tampered_cursor_is_422_before_the_service_runs(
    conversation_client: AsyncClient,
    conversation_service: Any,
) -> None:
    cursor = conversation_service.cursor_codec.encode(
        ConversationCursor(
            updated_at=datetime.datetime(2026, 7, 12, tzinfo=datetime.UTC),
            conversation_id=UUID(_CID),
        )
    )
    replacement = "A" if cursor[-1] != "A" else "B"

    response = await conversation_client.get(
        "/web/api/conversations",
        params={"cursor": f"{cursor[:-1]}{replacement}"},
    )

    assert response.status_code == 422
    conversation_service.list.assert_not_awaited()


async def test_history_of_other_principal_is_404(
    conversation_client: AsyncClient,
    conversation_service: Any,
) -> None:
    conversation_service.history.return_value = None

    response = await conversation_client.get(f"/web/api/conversations/{_CID}/history")

    assert response.status_code == 404


async def test_history_cursor_round_trips_and_cross_conversation_is_422(
    conversation_client: AsyncClient,
    conversation_service: Any,
) -> None:
    cursor = ConversationHistoryCursor(conversation_id=UUID(_CID), before_sequence=71)
    current = conversation_service.history.return_value
    conversation_service.history.return_value = ConversationHistoryPage(
        conversation=current.conversation,
        turns=(),
        next_cursor=cursor,
        fetched_rows=3,
    )
    first = await conversation_client.get(f"/web/api/conversations/{_CID}/history?limit=2")
    token = first.json()["next_cursor"]
    assert isinstance(token, str)

    conversation_service.history.return_value = ConversationHistoryPage(
        conversation=current.conversation,
        turns=(),
        next_cursor=None,
        fetched_rows=0,
    )
    second = await conversation_client.get(
        f"/web/api/conversations/{_CID}/history",
        params={"limit": 2, "cursor": token},
    )
    assert second.status_code == 200
    assert conversation_service.history.await_args.kwargs["page"] == (
        ConversationHistoryPageRequest(limit=2, cursor=cursor)
    )

    other = conversation_service.history_cursor_codec.encode(
        ConversationHistoryCursor(
            conversation_id=UUID("00000000-0000-0000-0000-000000000002"),
            before_sequence=71,
        )
    )
    conversation_service.history.reset_mock()
    mismatch = await conversation_client.get(
        f"/web/api/conversations/{_CID}/history", params={"cursor": other}
    )
    assert mismatch.status_code == 422
    conversation_service.history.assert_not_awaited()


async def test_tampered_history_cursor_is_422(
    conversation_client: AsyncClient,
    conversation_service: Any,
) -> None:
    token = conversation_service.history_cursor_codec.encode(
        ConversationHistoryCursor(conversation_id=UUID(_CID), before_sequence=2)
    )
    replacement = "A" if token[-1] != "A" else "B"
    response = await conversation_client.get(
        f"/web/api/conversations/{_CID}/history",
        params={"cursor": f"{token[:-1]}{replacement}"},
    )
    assert response.status_code == 422
    conversation_service.history.assert_not_awaited()


async def test_rename_validates_trimmed_title(conversation_client: AsyncClient) -> None:
    response = await conversation_client.patch(
        f"/web/api/conversations/{_CID}",
        json={"title": "   "},
    )

    assert response.status_code == 422


async def test_rename_normalizes_whitespace(
    conversation_client: AsyncClient,
    conversation_service: Any,
) -> None:
    response = await conversation_client.patch(
        f"/web/api/conversations/{_CID}",
        json={"title": "  Renamed\n chat  "},
    )

    assert response.status_code == 200
    assert conversation_service.rename.await_args.args[-1] == "Renamed chat"


async def test_delete_returns_204(
    conversation_client: AsyncClient,
    conversation_service: Any,
) -> None:
    response = await conversation_client.delete(f"/web/api/conversations/{_CID}")

    assert response.status_code == 204
    assert response.content == b""
    conversation_service.delete.assert_awaited_once()


async def test_delete_all_returns_204_when_no_conversations_exist(
    conversation_client: AsyncClient,
    conversation_service: Any,
) -> None:
    conversation_service.delete_all.return_value = 0

    response = await conversation_client.delete("/web/api/conversations")

    assert response.status_code == 204
    assert response.content == b""
    conversation_service.delete_all.assert_awaited_once()


async def test_delete_has_no_messages_subroute(conversation_client: AsyncClient) -> None:
    response = await conversation_client.delete(f"/web/api/conversations/{_CID}/messages")

    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Cross-origin protection
# ---------------------------------------------------------------------------


_COOKIE_MUTATIONS = (
    pytest.param("POST", "/web/api/conversations", None, "create", 201, id="create"),
    pytest.param(
        "PATCH",
        f"/web/api/conversations/{_CID}",
        {"title": "Renamed chat"},
        "rename",
        200,
        id="rename",
    ),
    pytest.param(
        "DELETE",
        f"/web/api/conversations/{_CID}",
        None,
        "delete",
        204,
        id="delete",
    ),
)


@pytest.mark.parametrize(
    ("method", "path", "body", "service_method", "status_code"), _COOKIE_MUTATIONS
)
async def test_cookie_lifecycle_mutations_accept_exact_same_origin(
    cookie_conversation_client: AsyncClient,
    conversation_service: Any,
    method: str,
    path: str,
    body: dict[str, str] | None,
    service_method: str,
    status_code: int,
) -> None:
    response = await cookie_conversation_client.request(
        method,
        path,
        json=body,
        headers={"Origin": "https://app.example.com"},
    )

    assert response.status_code == status_code
    getattr(conversation_service, service_method).assert_awaited_once()


@pytest.mark.parametrize(
    ("method", "path", "body", "service_method", "_status_code"), _COOKIE_MUTATIONS
)
async def test_cookie_lifecycle_mutations_reject_sibling_origin_before_service(
    cookie_conversation_client: AsyncClient,
    conversation_service: Any,
    method: str,
    path: str,
    body: dict[str, str] | None,
    service_method: str,
    _status_code: int,
) -> None:
    response = await cookie_conversation_client.request(
        method,
        path,
        json=body,
        headers={"Origin": "https://evil.example.com"},
    )

    assert response.status_code == 403
    getattr(conversation_service, service_method).assert_not_awaited()


async def test_cookie_lifecycle_mutation_rejects_missing_origin(
    cookie_conversation_client: AsyncClient,
    conversation_service: Any,
) -> None:
    response = await cookie_conversation_client.post("/web/api/conversations")

    assert response.status_code == 403
    conversation_service.create.assert_not_awaited()


async def test_bearer_lifecycle_mutation_does_not_require_browser_origin(
    cookie_conversation_client: AsyncClient,
    conversation_service: Any,
) -> None:
    response = await cookie_conversation_client.post(
        "/web/api/conversations",
        headers={"Authorization": "Bearer secret-token"},
    )

    assert response.status_code == 201
    conversation_service.create.assert_awaited_once()


_WEB_ANSWER_BODY = {
    "query": "hello",
    "conversation_id": _CID,
    "submission_id": "00000000-0000-4000-8000-000000000099",
}


@pytest.mark.parametrize("content_type", ["application/json", "text/plain"])
async def test_cookie_web_answer_accepts_exact_origin_independent_of_content_type(
    cookie_conversation_client: AsyncClient,
    conversation_service: Any,
    content_type: str,
) -> None:
    conversation_service.start_answer.return_value = None

    response = await cookie_conversation_client.post(
        "/web/api/answer",
        content=json.dumps(_WEB_ANSWER_BODY),
        headers={"Content-Type": content_type, "Origin": "https://app.example.com"},
    )

    assert response.status_code == 404
    conversation_service.start_answer.assert_awaited_once()


@pytest.mark.parametrize(
    "origin",
    [
        pytest.param("https://evil.example.com", id="sibling-origin"),
        pytest.param("https://app.example.com/path", id="malformed-origin"),
        pytest.param(None, id="missing-origin"),
    ],
)
async def test_cookie_web_answer_rejects_non_exact_origin_before_service(
    cookie_conversation_client: AsyncClient,
    conversation_service: Any,
    origin: str | None,
) -> None:
    conversation_service.start_answer.return_value = None
    headers = {"Content-Type": "text/plain"}
    if origin is not None:
        headers["Origin"] = origin

    response = await cookie_conversation_client.post(
        "/web/api/answer",
        content=json.dumps(_WEB_ANSWER_BODY),
        headers=headers,
    )

    assert response.status_code == 403
    conversation_service.start_answer.assert_not_awaited()


async def test_bearer_web_answer_does_not_require_browser_origin(
    cookie_conversation_client: AsyncClient,
    conversation_service: Any,
) -> None:
    conversation_service.start_answer.return_value = None

    response = await cookie_conversation_client.post(
        "/web/api/answer",
        json=_WEB_ANSWER_BODY,
        headers={"Authorization": "Bearer secret-token"},
    )

    assert response.status_code == 404
    conversation_service.start_answer.assert_awaited_once()


@pytest.mark.parametrize(
    ("method", "path"),
    [
        pytest.param("POST", "/web/api/files/upload", id="file-upload"),
        pytest.param(
            "DELETE",
            "/web/api/files?workspace=default&file_path=report.pdf",
            id="file-delete",
        ),
        pytest.param("POST", "/web/api/workspaces/create", id="workspace-create"),
        pytest.param("POST", "/web/api/workspaces/delete", id="workspace-delete"),
    ],
)
async def test_cookie_web_mutations_reject_missing_origin(
    cookie_conversation_client: AsyncClient,
    method: str,
    path: str,
) -> None:
    response = await cookie_conversation_client.request(method, path)

    assert response.status_code == 403


# ---------------------------------------------------------------------------
# Delivery to a live Run
# ---------------------------------------------------------------------------


async def test_the_answer_route_returns_a_queue_receipt(
    conversation_client: AsyncClient,
    conversation_service: Any,
) -> None:
    conversation_service.start_answer.return_value = WebAnswerDelivery(
        mode="queue", queued=_queued_message(content="continue")
    )

    response = await conversation_client.post(
        "/web/api/answer",
        json={**_WEB_ANSWER_BODY, "delivery": "queue"},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["kind"] == "queue"
    assert body["message"]["content"] == "continue"
    assert conversation_service.start_answer.await_args.kwargs["delivery"] == "queue"


async def test_the_answer_route_returns_a_steer_receipt(
    conversation_client: AsyncClient,
    conversation_service: Any,
) -> None:
    conversation_service.start_answer.return_value = WebAnswerDelivery(
        mode="steer", steered_run_id="run-live", control_sequence=4
    )

    response = await conversation_client.post(
        "/web/api/answer",
        json={**_WEB_ANSWER_BODY, "delivery": "steer"},
    )

    assert response.status_code == 202
    body = response.json()
    assert (body["kind"], body["run_id"], body["control_sequence"]) == ("steer", "run-live", 4)
    assert body["message"] is None


async def test_the_answer_route_rejects_an_unknown_delivery(
    conversation_client: AsyncClient,
    conversation_service: Any,
) -> None:
    response = await conversation_client.post(
        "/web/api/answer",
        json={**_WEB_ANSWER_BODY, "delivery": "later"},
    )

    assert response.status_code == 422
    conversation_service.start_answer.assert_not_awaited()


# ---------------------------------------------------------------------------
# Delivery to a live Run
# ---------------------------------------------------------------------------


def _submission_seed(
    conversation_id: str = _CID, *, parent_run_id: str | None = None
) -> SubmissionSeed:
    now = datetime.datetime(2026, 7, 12, tzinfo=datetime.UTC)
    return SubmissionSeed(
        head=ConversationHead(
            principal_id="anonymous",
            conversation_id=conversation_id,
            content_revision=0,
            title=None,
            created_at=now,
            updated_at=now,
            agent_session_id=conversation_id,
            agent_lane_id="main",
        ),
        parent_run_id=parent_run_id,
    )


def _delivery_service(store: AsyncMock, answers: AsyncMock, workspace_heads: Any = None) -> Any:
    from dlightrag.application.web_conversations.service import WebConversationService

    return WebConversationService(
        store=store,
        answers=answers,
        max_attachments=6,
        cursor_secret=b"unit-test-cursor-secret",
        workspace_heads=workspace_heads,
    )


class _HeadReclaimer:
    """Records what a deletion asked the disk to reclaim."""

    def __init__(self) -> None:
        self.sessions: list[tuple[str, str]] = []
        self.owners: list[str] = []
        self.raises: Exception | None = None

    async def reclaim_session(self, *, owner_id: str, session_id: str) -> None:
        if self.raises is not None:
            raise self.raises
        self.sessions.append((owner_id, session_id))

    async def reclaim_owner(self, *, owner_id: str) -> None:
        if self.raises is not None:
            raise self.raises
        self.owners.append(owner_id)


async def test_a_send_while_a_run_is_live_is_queued() -> None:
    store = AsyncMock()
    store.replay_answer_turn.return_value = None
    store.submission_seed.return_value = _submission_seed()
    store.live_answer_run.return_value = LiveAnswerRun(run_id="run-live", status="running")
    store.append_queued_message.return_value = _queued_message(content="continue")
    answers = AsyncMock(spec=AnswerService)
    service = _delivery_service(store, answers)

    result = await service.start_answer(
        None,
        conversation_id=_CID,
        submission_id="00000000-0000-4000-8000-000000000099",
        query="continue",
        workspaces=["default"],
    )

    assert isinstance(result, WebAnswerDelivery)
    assert result.mode == "queue"
    assert result.queued is not None and result.queued.content == "continue"
    store.append_queued_message.assert_awaited_once()
    answers.accept.assert_not_awaited()
    answers.steer.assert_not_awaited()


async def test_a_steer_delivery_injects_into_the_live_run() -> None:
    store = AsyncMock()
    store.replay_answer_turn.return_value = None
    store.submission_seed.return_value = _submission_seed()
    store.live_answer_run.return_value = LiveAnswerRun(run_id="run-live", status="running")
    answers = AsyncMock(spec=AnswerService)
    answers.steer.return_value = AgentControlReceipt(
        run_id="run-live", control_sequence=4, kind="steer"
    )
    service = _delivery_service(store, answers)

    result = await service.start_answer(
        None,
        conversation_id=_CID,
        submission_id="00000000-0000-4000-8000-000000000099",
        query="not that",
        workspaces=["default"],
        delivery="steer",
    )

    assert isinstance(result, WebAnswerDelivery)
    assert (result.mode, result.steered_run_id, result.control_sequence) == (
        "steer",
        "run-live",
        4,
    )
    assert answers.steer.await_args.kwargs["instruction"] == "not that"
    store.append_queued_message.assert_not_awaited()
    answers.accept.assert_not_awaited()


async def test_a_closed_steer_falls_back_to_the_queue() -> None:
    store = AsyncMock()
    store.replay_answer_turn.return_value = None
    store.submission_seed.return_value = _submission_seed()
    store.live_answer_run.return_value = LiveAnswerRun(run_id="run-live", status="running")
    store.append_queued_message.return_value = _queued_message(content="not that")
    answers = AsyncMock(spec=AnswerService)
    answers.steer.return_value = None
    service = _delivery_service(store, answers)

    result = await service.start_answer(
        None,
        conversation_id=_CID,
        submission_id="00000000-0000-4000-8000-000000000099",
        query="not that",
        workspaces=["default"],
        delivery="steer",
    )

    assert isinstance(result, WebAnswerDelivery)
    assert result.mode == "queue"
    store.append_queued_message.assert_awaited_once()
    answers.accept.assert_not_awaited()


async def test_a_send_without_a_live_run_is_accepted() -> None:
    store = AsyncMock()
    store.replay_answer_turn.return_value = None
    store.submission_seed.return_value = _submission_seed()
    store.live_answer_run.return_value = None
    answers = AsyncMock(spec=AnswerService)
    answers.accept.return_value = None
    service = _delivery_service(store, answers)

    result = await service.start_answer(
        None,
        conversation_id=_CID,
        submission_id="00000000-0000-4000-8000-000000000099",
        query="a new question",
        workspaces=["default"],
    )

    assert result is None
    answers.accept.assert_awaited_once()
    store.append_queued_message.assert_not_awaited()
    answers.steer.assert_not_awaited()


async def test_attachments_are_refused_while_a_run_is_live() -> None:
    from types import SimpleNamespace

    store = AsyncMock()
    store.replay_answer_turn.return_value = None
    store.submission_seed.return_value = _submission_seed()
    store.live_answer_run.return_value = LiveAnswerRun(run_id="run-live", status="running")
    answers = AsyncMock(spec=AnswerService)
    service = _delivery_service(store, answers)
    attachment = SimpleNamespace(
        filename="a.pdf",
        mime_type="application/pdf",
        ordinal=0,
        byte_size=3,
        content_sha256="0" * 64,
        attachment_bytes=b"abc",
    )

    with pytest.raises(ApplicationInputError):
        await service.start_answer(
            None,
            conversation_id=_CID,
            submission_id="00000000-0000-4000-8000-000000000099",
            query="with a file",
            workspaces=["default"],
            attachments=(attachment,),  # type: ignore[arg-type]
        )

    answers.accept.assert_not_awaited()
    store.append_queued_message.assert_not_awaited()


# ---------------------------------------------------------------------------
# Delivering the queued messages after a settled Run
# ---------------------------------------------------------------------------


async def test_a_settled_run_delivers_the_queue_as_one_turn() -> None:
    from types import SimpleNamespace

    store = AsyncMock()
    store.find_turn_by_run.return_value = SimpleNamespace(conversation_id=_CID)
    store.submission_seed.return_value = _submission_seed(parent_run_id="run-live")
    store.queue_messages.return_value = (
        _queued_message(message_id=_MID, content="first", position=1),
        _queued_message(
            message_id="00000000-0000-0000-0000-000000000003",
            content="second",
            position=2,
        ),
    )
    answers = AsyncMock(spec=AnswerService)
    answers.accept.return_value = object()
    service = _delivery_service(store, answers)

    delivered = await service.accept_queued_turn(
        "owner-1", parent_run_id="run-live", workspaces=("default",)
    )

    assert delivered is True
    request = answers.accept.await_args.kwargs["request"]
    assert request.parent_run_id == "run-live"
    assert request.agent_session_id == _CID
    assert "answer them together" in request.query
    assert request.query.index("first") < request.query.index("second")
    assert answers.accept.await_args.kwargs["owner_id"] == "owner-1"
    assert answers.accept.await_args.kwargs["caller"] is None
    from dlightrag.application.web_conversations.models import queued_delivery_digest

    guard = answers.accept.await_args.kwargs["acceptor"].queue_guard
    assert guard is not None
    assert guard.message_ids == (_MID, "00000000-0000-0000-0000-000000000003")
    assert guard.digest == queued_delivery_digest(store.queue_messages.return_value)


async def test_a_retried_delivery_reuses_the_same_submission_id() -> None:
    from types import SimpleNamespace

    store = AsyncMock()
    store.find_turn_by_run.return_value = SimpleNamespace(conversation_id=_CID)
    store.submission_seed.return_value = _submission_seed(parent_run_id="run-live")
    store.queue_messages.return_value = (_queued_message(content="first", position=1),)
    answers = AsyncMock(spec=AnswerService)
    answers.accept.return_value = object()
    service = _delivery_service(store, answers)

    await service.accept_queued_turn("owner-1", parent_run_id="run-live", workspaces=("default",))
    await service.accept_queued_turn("owner-1", parent_run_id="run-live", workspaces=("default",))

    keys = [call.kwargs["idempotency_key"] for call in answers.accept.await_args_list]
    assert keys[0] == keys[1]
    assert str(UUID(keys[0])) == keys[0]


async def test_no_queued_messages_leave_the_settlement_to_the_goal() -> None:
    from types import SimpleNamespace

    store = AsyncMock()
    store.find_turn_by_run.return_value = SimpleNamespace(conversation_id=_CID)
    store.queue_messages.return_value = ()
    answers = AsyncMock(spec=AnswerService)
    service = _delivery_service(store, answers)

    delivered = await service.accept_queued_turn(
        "owner-1", parent_run_id="run-live", workspaces=("default",)
    )

    assert delivered is False
    answers.accept.assert_not_awaited()


async def test_a_refused_accept_delivers_nothing() -> None:
    from types import SimpleNamespace

    store = AsyncMock()
    store.find_turn_by_run.return_value = SimpleNamespace(conversation_id=_CID)
    store.submission_seed.return_value = _submission_seed(parent_run_id="run-live")
    store.queue_messages.return_value = (_queued_message(content="first"),)
    answers = AsyncMock(spec=AnswerService)
    answers.accept.return_value = None
    service = _delivery_service(store, answers)

    delivered = await service.accept_queued_turn(
        "owner-1", parent_run_id="run-live", workspaces=("default",)
    )

    assert delivered is False


async def test_a_run_with_no_conversation_delivers_nothing() -> None:
    store = AsyncMock()
    store.find_turn_by_run.return_value = None
    answers = AsyncMock(spec=AnswerService)
    service = _delivery_service(store, answers)

    assert not await service.accept_queued_turn(
        "owner-1", parent_run_id="run-live", workspaces=("default",)
    )
    store.queue_messages.assert_not_awaited()


# ---------------------------------------------------------------------------
# Storage availability signalling
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "path", "store_method"),
    (
        pytest.param("GET", "/web/api/conversations", "list_conversations", id="read"),
        pytest.param("POST", "/web/api/conversations", "create_conversation", id="mutation"),
    ),
)
async def test_store_unavailability_returns_retryable_503(
    test_config: DlightragConfig,
    method: str,
    path: str,
    store_method: str,
) -> None:
    from dlightrag.application.web_conversations.models import WebConversationUnavailableError
    from dlightrag.application.web_conversations.service import WebConversationService

    store = AsyncMock()
    getattr(store, store_method).side_effect = WebConversationUnavailableError()
    application = create_app()
    service = WebConversationService(
        store=store,
        answers=FakeAnswers(),
        max_attachments=6,
        cursor_secret=b"unit-test-cursor-secret",
    )
    application.state.application = application_double(test_config, web_conversations=service)
    transport = ASGITransport(app=application)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.request(method, path)

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Web conversation storage is unavailable",
        "error_type": "unavailable",
    }


@pytest.mark.parametrize(
    "shutdown_error",
    (
        pytest.param(
            asyncpg.exceptions.AdminShutdownError("administrative shutdown"),
            id="admin-shutdown",
        ),
        pytest.param(
            asyncpg.exceptions.CrashShutdownError("crash shutdown"),
            id="crash-shutdown",
        ),
    ),
)
async def test_postgres_adapter_translates_shutdown_errors(shutdown_error: Exception) -> None:
    from dlightrag.adapters.postgres.web.web_conversations import PGWebConversationStore
    from dlightrag.application.web_conversations.models import WebConversationUnavailableError

    class Acquire:
        async def __aenter__(self):
            raise shutdown_error

        async def __aexit__(self, *_exc: object) -> bool:
            return False

    class Pool:
        def acquire(self) -> Acquire:
            return Acquire()

    store = PGWebConversationStore(pool=Pool())

    with pytest.raises(WebConversationUnavailableError):
        await store._run_read(AsyncMock())


@pytest.mark.parametrize(
    "store_error",
    (
        pytest.param(
            asyncpg.exceptions.UniqueViolationError("duplicate key"),
            id="unique-violation",
        ),
        pytest.param(asyncpg.exceptions.CheckViolationError("check failed"), id="constraint"),
        pytest.param(asyncpg.exceptions.DataError("invalid data"), id="data"),
        pytest.param(ValueError("broken projection"), id="value"),
        pytest.param(RuntimeError("broken adapter"), id="programmer"),
    ),
)
async def test_data_and_programmer_errors_are_not_mislabeled_as_store_unavailability(
    test_config: DlightragConfig,
    store_error: Exception,
) -> None:
    from dlightrag.application.web_conversations.service import WebConversationService

    store = AsyncMock()
    store.list_conversations.side_effect = store_error
    application = create_app()
    service = WebConversationService(
        store=store,
        answers=FakeAnswers(),
        max_attachments=6,
        cursor_secret=b"unit-test-cursor-secret",
    )
    application.state.application = application_double(test_config, web_conversations=service)
    transport = ASGITransport(app=application)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        with pytest.raises(type(store_error), match=str(store_error)):
            await client.get("/web/api/conversations")


async def test_the_queue_service_trims_content_and_delegates_the_order() -> None:
    from dlightrag.application.access.principal import (
        owner_id_from_user,
    )
    from dlightrag.application.web_conversations.service import WebConversationService

    store = AsyncMock()
    store.append_queued_message.return_value = _queued_message(content="hold this")
    service = WebConversationService(
        store=store,
        answers=FakeAnswers(),
        max_attachments=6,
        cursor_secret=b"unit-test-cursor-secret",
    )

    created = await service.enqueue(None, _CID, "  hold this  ")

    assert created is not None
    store.append_queued_message.assert_awaited_once_with(
        owner_id_from_user(None), _CID, content="hold this"
    )

    with pytest.raises(ApplicationInputError):
        await service.enqueue(None, _CID, "   ")
    store.append_queued_message.assert_awaited_once()

    store.reorder_queued_messages.return_value = ()
    await service.reorder_queue(None, _CID, [_MID])
    store.reorder_queued_messages.assert_awaited_once_with(
        owner_id_from_user(None), _CID, message_ids=[_MID]
    )
    # The set check lives in the store under its row lock; the service adds no second read.
    store.queue_messages.assert_not_awaited()


async def test_application_page_uses_extra_row_proof_and_last_returned_item_cursor() -> None:
    from dlightrag.application.access.principal import (
        owner_id_from_user,
    )
    from dlightrag.application.web_conversations.models import ConversationRowPage
    from dlightrag.application.web_conversations.service import WebConversationService

    now = datetime.datetime(2026, 7, 12, tzinfo=datetime.UTC)
    rows = tuple(
        {
            "conversation_id": f"00000000-0000-0000-0000-{index:012d}",
            "title": None,
            "created_at": now,
            "updated_at": now - datetime.timedelta(seconds=index),
        }
        for index in range(1, 3)
    )
    store = AsyncMock()
    store.list_conversations.return_value = ConversationRowPage(
        items=rows,
        has_more=True,
        fetched_rows=3,
    )
    service = WebConversationService(
        store=store,
        answers=FakeAnswers(),
        max_attachments=6,
        cursor_secret=b"unit-test-cursor-secret",
    )

    result = await service.list(None, page=ConversationPageRequest(limit=2))

    assert len(result.items) == 2
    assert result.fetched_rows == 3
    assert result.next_cursor == ConversationCursor(
        updated_at=rows[-1]["updated_at"],
        conversation_id=UUID(str(rows[-1]["conversation_id"])),
    )
    store.list_conversations.assert_awaited_once_with(
        owner_id_from_user(None),
        page=ConversationPageRequest(limit=2),
    )


def test_cursor_codec_round_trips_and_rejects_tampered_or_foreign_tokens() -> None:
    secret = b"unit-test-cursor-secret"
    codec = ConversationCursorCodec(secret)
    cursor = ConversationCursor(
        updated_at=datetime.datetime(2026, 7, 12, 3, 4, 5, 123456, tzinfo=datetime.UTC),
        conversation_id=UUID(_CID),
    )
    encoded = codec.encode(cursor)
    history = ConversationHistoryCursorCodec(secret).encode(
        ConversationHistoryCursor(conversation_id=UUID(_CID), before_sequence=2)
    )

    assert codec.decode(encoded) == cursor
    for value in (("A" if encoded[0] != "A" else "B") + encoded[1:], history):
        with pytest.raises(ValueError, match="invalid conversation page cursor"):
            codec.decode(value)


def test_history_cursor_codec_round_trips_the_shared_sequence() -> None:
    codec = ConversationHistoryCursorCodec(b"unit-test-history-cursor-secret")
    cursor = ConversationHistoryCursor(conversation_id=UUID(_CID), before_sequence=17)

    assert codec.decode(codec.encode(cursor)) == cursor


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_history_cursor_requires_a_positive_sequence(value: Any) -> None:
    with pytest.raises(ValueError, match="sequence"):
        ConversationHistoryCursor(
            conversation_id=UUID(_CID),
            before_sequence=value,
        )


@pytest.mark.parametrize("limit", [0, 101, True, 1.5])
def test_application_page_limit_is_hard_bounded(limit: object) -> None:
    with pytest.raises(ValueError, match="conversation page limit"):
        ConversationPageRequest(limit=limit)  # type: ignore[arg-type]


def test_browser_contracts_forbid_extra_fields_and_normalize_titles() -> None:
    from pydantic import ValidationError

    from dlightrag.adapters.http.browser.conversation_models import RenameConversationRequest

    request = RenameConversationRequest(title="  Quarterly\n review  ")

    assert request.title == "Quarterly review"
    with pytest.raises(ValidationError):
        RenameConversationRequest.model_validate({"title": "Valid", "principal_id": "p1"})


# ---------------------------------------------------------------------------
# Server-owned commands
# ---------------------------------------------------------------------------

_COMMAND_CREATED_AT = datetime.datetime(2026, 7, 12, 3, 4, 5, tzinfo=datetime.UTC)
_COMMAND_HELP = (
    "Commands: /goal edit <objective>, /goal cap N, /goal pause, /goal resume, /goal clear"
)


def _persisted_command(
    *,
    command_id: str = "00000000-0000-0000-0000-0000000000c1",
    sequence: int = 3,
    name: str = "goal",
    input: str = "pause",
    result: str = "Paused goal\nObjective: finish it\nRound: 1/8\n" + _COMMAND_HELP,
    tone: CommandTone = "success",
    result_data: dict[str, Any] | None = None,
) -> ConversationCommand:
    return ConversationCommand(
        command_id=command_id,
        sequence=sequence,
        name=name,
        input=input,
        result=result,
        tone=tone,
        created_at=_COMMAND_CREATED_AT,
        result_data=result_data,
    )


async def test_history_returns_persisted_commands_beside_turns(
    conversation_client: AsyncClient,
    conversation_service: Any,
) -> None:
    current = conversation_service.history.return_value
    conversation_service.history.return_value = ConversationHistoryPage(
        conversation=current.conversation,
        turns=(),
        commands=(
            _persisted_command(
                result_data={
                    "kind": "goal_status",
                    "goal": {
                        "objective": "finish it",
                        "phase": "paused",
                        "max_rounds": 8,
                        "rounds_started": 1,
                        "blocked_reason": None,
                    },
                }
            ),
        ),
        next_cursor=None,
        fetched_rows=1,
    )

    response = await conversation_client.get(f"/web/api/conversations/{_CID}/history")

    assert response.status_code == 200
    assert response.json()["commands"] == [
        {
            "command_id": "00000000-0000-0000-0000-0000000000c1",
            "sequence": 3,
            "name": "goal",
            "input": "pause",
            "result": "Paused goal\nObjective: finish it\nRound: 1/8\n" + _COMMAND_HELP,
            "tone": "success",
            "created_at": "2026-07-12T03:04:05Z",
            "result_data": {
                "kind": "goal_status",
                "goal": {
                    "objective": "finish it",
                    "phase": "paused",
                    "max_rounds": 8,
                    "rounds_started": 1,
                    "blocked_reason": None,
                },
            },
        }
    ]


async def test_history_renders_a_command_without_structured_data_as_null(
    conversation_client: AsyncClient,
    conversation_service: Any,
) -> None:
    """A row persisted before structured results existed still renders its fallback text."""
    current = conversation_service.history.return_value
    conversation_service.history.return_value = ConversationHistoryPage(
        conversation=current.conversation,
        turns=(),
        commands=(_persisted_command(),),
        next_cursor=None,
        fetched_rows=1,
    )

    response = await conversation_client.get(f"/web/api/conversations/{_CID}/history")

    assert response.status_code == 200
    assert response.json()["commands"][0]["result_data"] is None


async def test_posting_a_command_returns_the_persisted_event(
    conversation_client: AsyncClient,
    conversation_service: Any,
) -> None:
    command = _persisted_command(
        input="cap 12",
        result="Ongoing goal\nObjective: finish it\nRound: 0/12\n" + _COMMAND_HELP,
        result_data={
            "kind": "goal_status",
            "goal": {
                "objective": "finish it",
                "phase": "active",
                "max_rounds": 12,
                "rounds_started": 0,
                "blocked_reason": None,
            },
        },
    )
    conversation_service.command.return_value = command

    response = await conversation_client.post(
        f"/web/api/conversations/{_CID}/commands",
        json={"name": "goal", "input": "cap 12", "workspaces": ["alpha", "beta"]},
    )

    assert response.status_code == 200
    body = response.json()["command"]
    assert body["command_id"] == command.command_id
    assert body["sequence"] == command.sequence
    assert conversation_service.command.await_args.kwargs["workspaces"] == ["alpha", "beta"]
    assert body["name"] == "goal"
    assert body["input"] == "cap 12"
    assert body["result"] == command.result
    assert body["tone"] == "success"
    assert body["created_at"] == "2026-07-12T03:04:05Z"
    assert body["result_data"] == command.result_data
    call = conversation_service.command.await_args.args
    assert call[1:] == (_CID, "goal", "cap 12")


async def test_a_command_for_an_unknown_conversation_is_404(
    conversation_client: AsyncClient,
    conversation_service: Any,
) -> None:
    conversation_service.command.return_value = None

    response = await conversation_client.post(
        f"/web/api/conversations/{_CID}/commands",
        json={"name": "goal", "input": ""},
    )

    assert response.status_code == 404
    conversation_service.command.assert_awaited_once()


async def test_an_unregistered_command_is_a_conflict(
    conversation_client: AsyncClient,
    conversation_service: Any,
) -> None:
    conversation_service.command.side_effect = UnknownConversationCommand("bogus")

    response = await conversation_client.post(
        f"/web/api/conversations/{_CID}/commands",
        json={"name": "bogus", "input": ""},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "unknown conversation command: bogus"


@pytest.mark.parametrize(
    "body",
    (
        pytest.param({"name": "goal"}, id="missing-input"),
        pytest.param({"input": "pause"}, id="missing-name"),
        pytest.param({"name": "", "input": "pause"}, id="empty-name"),
        pytest.param({"name": "goal", "input": "pause", "extra": 1}, id="extra-field"),
    ),
)
async def test_a_malformed_command_body_is_422_before_the_service_runs(
    conversation_client: AsyncClient,
    conversation_service: Any,
    body: dict[str, Any],
) -> None:
    response = await conversation_client.post(
        f"/web/api/conversations/{_CID}/commands",
        json=body,
    )

    assert response.status_code == 422
    conversation_service.command.assert_not_awaited()


async def test_command_text_passes_the_reader_content_gate(
    conversation_client: AsyncClient,
    conversation_service: Any,
    application: Any,
) -> None:
    from types import SimpleNamespace

    application.skill_protection = SimpleNamespace(
        mask=lambda text: text.replace("SEALED-DOC", "[masked]")
    )
    conversation_service.command.return_value = _persisted_command(
        input="edit SEALED-DOC",
        result="Ongoing goal\nObjective: SEALED-DOC\nRound: 0/8\n" + _COMMAND_HELP,
        result_data={
            "kind": "goal_status",
            "goal": {
                "objective": "SEALED-DOC",
                "phase": "active",
                "max_rounds": 8,
                "rounds_started": 0,
                "blocked_reason": None,
            },
        },
    )

    response = await conversation_client.post(
        f"/web/api/conversations/{_CID}/commands",
        json={"name": "goal", "input": "edit SEALED-DOC"},
    )

    assert response.status_code == 200
    body = response.json()["command"]
    assert body["input"] == "edit [masked]"
    assert "SEALED-DOC" not in body["result"]
    # The browser localizes from result_data, so its prose passes the same gate while the
    # enum protocol fields the renderer matches stay intact.
    assert body["result_data"]["goal"]["objective"] == "[masked]"
    assert body["result_data"]["kind"] == "goal_status"
    assert body["result_data"]["goal"]["phase"] == "active"


# ---------------------------------------------------------------------------
# PostgreSQL history merge and command append
# ---------------------------------------------------------------------------


class _Transaction:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *_exc: Any) -> None:
        return None


class _HistoryConnection:
    """Answers the head, the two stream pages, and the append statement in-process."""

    def __init__(
        self,
        *,
        turns: tuple[dict[str, Any], ...] = (),
        commands: tuple[dict[str, Any], ...] = (),
        appended: dict[str, Any] | None = None,
    ) -> None:
        self.turns = turns
        self.commands = commands
        self.appended = appended
        self.fetches: list[tuple[str, tuple[Any, ...]]] = []
        self.append_args: tuple[Any, ...] = ()

    def transaction(self, **_kwargs: Any) -> _Transaction:
        return _Transaction()

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        assert query == pg_web._GET_CONVERSATION  # pyright: ignore[reportPrivateUsage]
        return {
            "principal_id": "owner",
            "conversation_id": _CID,
            "agent_session_id": _CID,
            "agent_lane_id": "main",
            "content_revision": 12,
            "title": None,
            "created_at": _COMMAND_CREATED_AT,
            "updated_at": _COMMAND_CREATED_AT,
        }

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.fetches.append((query, args))
        before, limit = args[2], args[3]
        if query == pg_web._GET_TURNS_PAGE:  # pyright: ignore[reportPrivateUsage]
            source, key = self.turns, "turn_number"
        elif query == pg_web._GET_COMMANDS_PAGE:  # pyright: ignore[reportPrivateUsage]
            source, key = self.commands, "sequence"
        else:
            raise AssertionError(f"unexpected history query: {query}")
        rows = [row for row in source if before is None or row[key] < before]
        rows.sort(key=lambda row: row[key], reverse=True)
        return rows[: int(limit)]


class _AppendConnection(_HistoryConnection):
    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        assert query == pg_web._APPEND_COMMAND  # pyright: ignore[reportPrivateUsage]
        self.append_args = args
        return self.appended


def _store_for(conn: Any) -> pg_web.PGWebConversationStore:
    store = pg_web.PGWebConversationStore(pool=cast(Any, FakePool(conn)))
    store._initialized = True  # pyright: ignore[reportPrivateUsage]
    return store


def _linked_turn_marker(row: dict[str, Any]) -> LinkedTurn:
    return LinkedTurn(
        turn_id=str(row["turn_number"]),
        turn_number=int(row["turn_number"]),
        submission_id="00000000-0000-4000-8000-000000000099",
        created_at=_COMMAND_CREATED_AT,
        run=RunView(
            run_id="00000000-0000-4000-8000-000000000098",
            run_kind="answer",
            lane="query",
            submitted_by="owner",
            access_scope_kind="owner",
            access_scope_id="owner",
            status="succeeded",
            phase=None,
            durable_progress_version=1,
            next_event_sequence=1,
            events_trimmed_at=None,
            cancel_requested=False,
            result=None,
            error_kind=None,
            error_message=None,
            created_at=_COMMAND_CREATED_AT,
            started_at=None,
            finished_at=None,
            request={},
        ),
    )


def _command_marker(row: dict[str, Any]) -> ConversationCommand:
    return ConversationCommand(
        command_id=f"command-{row['sequence']}",
        sequence=int(row["sequence"]),
        name="goal",
        input="",
        result="",
        tone="success",
        created_at=_COMMAND_CREATED_AT,
    )


def _markers(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    monkeypatch.setattr(
        pg_web,
        "_conversation_head",
        lambda row: SimpleNamespace(content_revision=row["content_revision"]),
    )
    monkeypatch.setattr(pg_web, "_linked_turn", _linked_turn_marker)
    monkeypatch.setattr(pg_web, "_command", _command_marker)


async def test_history_page_merges_turns_and_commands_by_their_shared_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _markers(monkeypatch)
    conn = _HistoryConnection(
        turns=({"turn_number": 10}, {"turn_number": 7}),
        commands=({"sequence": 12}, {"sequence": 9}, {"sequence": 3}),
    )

    page = await _store_for(conn).history_page(
        "owner",
        _CID,
        page=ConversationHistoryPageRequest(limit=3),
    )

    assert page is not None
    # Newest-first internally, chronological in the model.
    assert [turn.turn_number for turn in page.turns] == [10]
    assert [command.sequence for command in page.commands] == [9, 12]
    assert page.next_cursor == ConversationHistoryCursor(
        conversation_id=UUID(_CID),
        before_sequence=9,
    )
    assert page.fetched_rows == 5
    assert [query for query, _args in conn.fetches] == [
        pg_web._GET_TURNS_PAGE,  # pyright: ignore[reportPrivateUsage]
        pg_web._GET_COMMANDS_PAGE,  # pyright: ignore[reportPrivateUsage]
    ]
    for _query, args in conn.fetches:
        assert args[0] == "owner"
        assert args[1] == _CID
        assert args[2] is None
        assert args[3] == 4  # limit + 1 on each stream


async def test_history_page_continues_from_the_oldest_included_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _markers(monkeypatch)
    conn = _HistoryConnection(
        turns=({"turn_number": 10}, {"turn_number": 7}),
        commands=({"sequence": 12}, {"sequence": 9}, {"sequence": 3}),
    )

    page = await _store_for(conn).history_page(
        "owner",
        _CID,
        page=ConversationHistoryPageRequest(
            limit=3,
            cursor=ConversationHistoryCursor(
                conversation_id=UUID(_CID),
                before_sequence=9,
            ),
        ),
    )

    assert page is not None
    assert [turn.turn_number for turn in page.turns] == [7]
    assert [command.sequence for command in page.commands] == [3]
    assert page.next_cursor is None
    assert page.fetched_rows == 2
    for _query, args in conn.fetches:
        assert args[2] == 9


async def test_history_page_returns_both_streams_empty_past_the_oldest_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _markers(monkeypatch)
    conn = _HistoryConnection(
        turns=({"turn_number": 10},),
        commands=({"sequence": 12},),
    )

    page = await _store_for(conn).history_page(
        "owner",
        _CID,
        page=ConversationHistoryPageRequest(
            limit=3,
            cursor=ConversationHistoryCursor(
                conversation_id=UUID(_CID),
                before_sequence=2,
            ),
        ),
    )

    assert page is not None
    assert page.turns == ()
    assert page.commands == ()
    assert page.next_cursor is None
    assert page.fetched_rows == 0


async def test_append_command_allocates_its_row_and_projects_it() -> None:
    appended = {
        "command_id": "00000000-0000-0000-0000-0000000000c1",
        "sequence": 7,
        "name": "goal",
        "input": "pause",
        "result": "Paused goal",
        "tone": "success",
        "created_at": _COMMAND_CREATED_AT,
        # asyncpg hands a JSONB column back as text unless a codec is registered.
        "result_data": '{"kind": "goal_status", "goal": null}',
    }
    conn = _AppendConnection(appended=appended)

    command = await _store_for(conn).append_command(
        "owner",
        _CID,
        name="goal",
        command_input="pause",
        result="Paused goal",
        tone="success",
        result_data={"kind": "goal_status", "goal": None},
    )

    assert command == ConversationCommand(
        command_id="00000000-0000-0000-0000-0000000000c1",
        sequence=7,
        name="goal",
        input="pause",
        result="Paused goal",
        tone="success",
        created_at=_COMMAND_CREATED_AT,
        result_data={"kind": "goal_status", "goal": None},
    )
    (
        owner,
        conversation_id,
        command_id,
        name,
        command_input,
        result,
        tone,
        result_data_json,
    ) = conn.append_args
    assert (owner, conversation_id, name, command_input, result, tone) == (
        "owner",
        _CID,
        "goal",
        "pause",
        "Paused goal",
        "success",
    )
    assert json.loads(result_data_json) == {"kind": "goal_status", "goal": None}
    UUID(str(command_id))


async def test_append_command_without_structured_data_persists_null() -> None:
    appended = {
        "command_id": "00000000-0000-0000-0000-0000000000c1",
        "sequence": 8,
        "name": "goal",
        "input": "",
        "result": "No goal",
        "tone": "success",
        "created_at": _COMMAND_CREATED_AT,
        "result_data": None,
    }
    conn = _AppendConnection(appended=appended)

    command = await _store_for(conn).append_command(
        "owner",
        _CID,
        name="goal",
        command_input="",
        result="No goal",
        tone="success",
    )

    assert command is not None and command.result_data is None
    assert conn.append_args[-1] is None


async def test_append_command_reports_a_missing_conversation() -> None:
    conn = _AppendConnection(appended=None)

    command = await _store_for(conn).append_command(
        "owner",
        _CID,
        name="goal",
        command_input="",
        result="No goal in this conversation.",
        tone="success",
    )

    assert command is None


def test_a_stored_command_row_projects_its_fields() -> None:
    row = {
        "command_id": "00000000-0000-0000-0000-0000000000c1",
        "sequence": 4,
        "name": "goal",
        "input": "cap 12",
        "result": "Ongoing goal",
        "tone": "error",
        "created_at": _COMMAND_CREATED_AT,
        "result_data": '{"kind": "goal_invalid", "reason": "cap"}',
    }

    assert pg_web._command(row) == ConversationCommand(  # pyright: ignore[reportPrivateUsage]
        command_id="00000000-0000-0000-0000-0000000000c1",
        sequence=4,
        name="goal",
        input="cap 12",
        result="Ongoing goal",
        tone="error",
        created_at=_COMMAND_CREATED_AT,
        result_data={"kind": "goal_invalid", "reason": "cap"},
    )
    # A row persisted before structured data existed reads back as no payload at all.
    assert (
        pg_web._command({**row, "result_data": None}).result_data  # pyright: ignore[reportPrivateUsage]
        is None
    )
    with pytest.raises(RuntimeError, match="unknown tone"):
        pg_web._command({**row, "tone": "maybe"})  # pyright: ignore[reportPrivateUsage]


# ---------------------------------------------------------------------------
# Goal bar
# ---------------------------------------------------------------------------


def _conversation_goal() -> ConversationGoal:
    return ConversationGoal(
        conversation_id=_CID,
        goal_id="goal-1",
        revision=3,
        objective="finish the sealed model",
        phase="blocked",
        max_rounds=8,
        rounds_started=3,
        blocked_reason="the filings portal is unreachable",
    )


async def test_the_goal_bar_reads_the_current_goal(
    conversation_client: AsyncClient,
    conversation_service: Any,
) -> None:
    conversation_service.goal.return_value = _conversation_goal()

    response = await conversation_client.get(f"/web/api/conversations/{_CID}/goal")

    assert response.status_code == 200
    assert response.json() == {
        "conversation_id": _CID,
        "goal_id": "goal-1",
        "revision": 3,
        "objective": "finish the sealed model",
        "phase": "blocked",
        "max_rounds": 8,
        "rounds_started": 3,
        "blocked_reason": "the filings portal is unreachable",
    }


async def test_a_goal_command_carries_its_intent_and_refusal(
    conversation_client: AsyncClient,
    conversation_service: Any,
) -> None:
    conversation_service.goal_command.return_value = _conversation_goal()

    response = await conversation_client.post(
        f"/web/api/conversations/{_CID}/goal",
        json={"action": "edit", "objective": "accept the enterprise-plus-option framing"},
    )

    assert response.status_code == 200
    assert response.json()["objective"] == "finish the sealed model"
    conversation_service.goal_command.assert_awaited_once()
    kwargs = conversation_service.goal_command.await_args.kwargs
    assert kwargs["action"] == "edit"
    assert kwargs["objective"] == "accept the enterprise-plus-option framing"
    assert kwargs["workspaces"] is None

    conversation_service.goal_command.reset_mock()
    created = await conversation_client.post(
        f"/web/api/conversations/{_CID}/goal",
        json={"action": "create", "objective": "ship it", "workspaces": ["alpha"]},
    )
    assert created.status_code == 200
    assert conversation_service.goal_command.await_args.kwargs["workspaces"] == ["alpha"]

    conversation_service.goal_command.side_effect = GoalError(
        GOAL_NOT_FOUND, "this conversation carries no goal"
    )
    refused = await conversation_client.post(
        f"/web/api/conversations/{_CID}/goal", json={"action": "resume"}
    )
    assert refused.status_code == 409
    assert refused.json()["kind"] == GOAL_NOT_FOUND


async def test_a_goal_command_accepts_a_cap_above_the_old_ceiling(
    conversation_client: AsyncClient,
    conversation_service: Any,
) -> None:
    conversation_service.goal_command.return_value = _conversation_goal()

    response = await conversation_client.post(
        f"/web/api/conversations/{_CID}/goal",
        json={"action": "edit", "max_goal_rounds": 1_000},
    )

    assert response.status_code == 200
    assert conversation_service.goal_command.await_args.kwargs["max_rounds"] == 1_000


async def test_clearing_a_goal_answers_null(
    conversation_client: AsyncClient,
    conversation_service: Any,
) -> None:
    conversation_service.goal_command.return_value = None

    response = await conversation_client.post(
        f"/web/api/conversations/{_CID}/goal", json={"action": "clear"}
    )

    assert response.status_code == 200
    assert response.json() is None


def test_a_goal_round_prompt_is_shown_as_words_not_markup() -> None:
    from dlightrag.adapters.http.browser.conversations import reader_visible_query

    prompt = (
        '<goal_round>\nObjective: "Smoke: write a two-sentence memo."\nRound: 2/8\n'
        "Continue working...</goal_round>"
    )
    assert reader_visible_query(prompt) == "Goal round 2/8: Smoke: write a two-sentence memo."
    assert reader_visible_query("what is two plus two?") == "what is two plus two?"


_PROTECTED_GOAL_TEXT = (
    "When the evidence does not support the claim, the sentence is removed instead of softened, "
    "and the gap is named in the same paragraph. The check runs again on the final draft in the "
    "same order, so a reader can follow what changed between two versions without reading the "
    "whole ledger. Nothing is filled in from memory: an absence is stated, not guessed, because "
    "a confident sentence that no evidence carries is the one failure this method exists to "
    "prevent."
)


def _protected_index():
    from dlightrag.engine.agent.skill_protection import ProtectedTextIndex

    return ProtectedTextIndex.build(
        [("method", "references/method.md", _PROTECTED_GOAL_TEXT.encode("utf-8"))]
    )


def _goal_written_from_a_protected_document() -> ConversationGoal:
    return ConversationGoal(
        conversation_id=_CID,
        goal_id="goal-1",
        revision=3,
        objective=f"apply the method: {_PROTECTED_GOAL_TEXT}",
        phase="blocked",
        max_rounds=8,
        rounds_started=3,
        blocked_reason=f"stopped: {_PROTECTED_GOAL_TEXT}",
    )


def test_the_goal_projection_masks_its_reader_facing_prose() -> None:
    """A goal's objective and blocked reason are reader text, not model-internal state."""
    from dlightrag.adapters.http.browser.conversations import project_conversation_goal
    from dlightrag.engine.agent.skill_protection import MASK_PLACEHOLDER

    index = _protected_index()
    goal = _goal_written_from_a_protected_document()

    projected = project_conversation_goal(goal, mask_reader_text=index.mask)

    assert _PROTECTED_GOAL_TEXT not in projected.objective
    assert MASK_PLACEHOLDER in projected.objective
    assert projected.blocked_reason is not None
    assert _PROTECTED_GOAL_TEXT not in projected.blocked_reason
    assert MASK_PLACEHOLDER in projected.blocked_reason
    assert projected.conversation_id == _CID
    assert projected.phase == "blocked"
    # The record keeps the model's own words: masking happens only where a reader looks.
    assert _PROTECTED_GOAL_TEXT in goal.objective


async def test_the_goal_bar_routes_apply_the_reader_content_gate(
    conversation_client: AsyncClient,
    conversation_service: Any,
    application: Any,
) -> None:
    from types import SimpleNamespace

    from dlightrag.engine.agent.skill_protection import MASK_PLACEHOLDER

    application.skill_protection = SimpleNamespace(mask=_protected_index().mask)
    conversation_service.goal.return_value = _goal_written_from_a_protected_document()
    conversation_service.goal_command.return_value = _goal_written_from_a_protected_document()

    read = await conversation_client.get(f"/web/api/conversations/{_CID}/goal")
    assert read.status_code == 200
    read_body = read.json()
    assert _PROTECTED_GOAL_TEXT not in read_body["objective"]
    assert MASK_PLACEHOLDER in read_body["objective"]
    assert _PROTECTED_GOAL_TEXT not in read_body["blocked_reason"]
    assert MASK_PLACEHOLDER in read_body["blocked_reason"]

    written = await conversation_client.post(
        f"/web/api/conversations/{_CID}/goal", json={"action": "resume"}
    )
    assert written.status_code == 200
    written_body = written.json()
    assert _PROTECTED_GOAL_TEXT not in written_body["objective"]
    assert MASK_PLACEHOLDER in written_body["objective"]
    assert _PROTECTED_GOAL_TEXT not in written_body["blocked_reason"]
    assert MASK_PLACEHOLDER in written_body["blocked_reason"]


async def test_a_goal_command_cannot_name_an_unauthorized_workspace(
    conversation_client: AsyncClient,
    conversation_service: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A goal command is a write path: its client-named workspaces need the send-time access check."""
    from fastapi import HTTPException

    from dlightrag.adapters.http.browser.routes import conversations as conversation_routes

    async def deny(_request: Any, action: str, workspace: str | None) -> None:
        assert action == "workspace.query"
        assert workspace == "finance_q4"
        raise HTTPException(status_code=403, detail="Workspace access denied")

    monkeypatch.setattr(conversation_routes, "enforce_web_access", deny)

    response = await conversation_client.post(
        f"/web/api/conversations/{_CID}/goal",
        json={
            "action": "create",
            "objective": "read the restricted corpus",
            "workspaces": ["finance_q4"],
        },
    )

    assert response.status_code == 403
    conversation_service.goal_command.assert_not_awaited()


# ---------------------------------------------------------------------------
# Server-owned command execution
# ---------------------------------------------------------------------------


def _command_service(
    store: AsyncMock,
    *,
    goal_store: Any = None,
    goal_rounds: Any = None,
    goal_round_canceller: Any = None,
    parent_run_id: str | None = "run-parent",
) -> Any:
    from dlightrag.application.access.principal import (
        owner_id_from_user,
    )
    from dlightrag.application.web_conversations.models import SubmissionSeed
    from dlightrag.application.web_conversations.service import WebConversationService
    from dlightrag.engine.answer.goal_service import AnswerGoalService
    from tests.support.goals import FakeGoalStore

    goals = goal_store if goal_store is not None else FakeGoalStore()
    store.submission_seed.return_value = SubmissionSeed(
        head=ConversationHead(
            principal_id=owner_id_from_user(None),
            conversation_id=_CID,
            content_revision=0,
            title=None,
            created_at=_COMMAND_CREATED_AT,
            updated_at=_COMMAND_CREATED_AT,
            agent_session_id=_CID,
            agent_lane_id="main",
        ),
        parent_run_id=parent_run_id,
    )
    store.append_command.return_value = _persisted_command()
    store.newest_answer_workspaces.return_value = ("default",)
    store.pending_goal_round.return_value = None
    store.live_answer_run.return_value = None
    store.submission_exists.return_value = False
    return WebConversationService(
        store=store,
        answers=FakeAnswers(),
        max_attachments=6,
        cursor_secret=b"unit-test-cursor-secret",
        goals=AnswerGoalService(store=goals),
        goal_rounds=goal_rounds,
        goal_round_canceller=goal_round_canceller,
    )


class FakeGoalRounds:
    """The driver's human-control entry points, recorded instead of starting a Run."""

    def __init__(self, *, continues: bool = True, raises: Exception | None = None) -> None:
        self.continues = continues
        self.raises = raises
        #: ``(entry_point, owner_id, session_id, last_identifier)`` per call, in order.
        self.calls: list[tuple[str, str, str, str]] = []
        #: The workspaces the created-goal path carried, for the carry test.
        self.workspaces: tuple[str, ...] = ()

    async def continue_created_goal(
        self,
        *,
        owner_id: str,
        session_id: str,
        conversation_id: str,
        workspaces: tuple[str, ...] = (),
    ) -> bool:
        if self.raises is not None:
            raise self.raises
        self.calls.append(("created", owner_id, session_id, conversation_id))
        self.workspaces = tuple(workspaces)
        return self.continues

    async def continue_after_resume(
        self, *, owner_id: str, session_id: str, parent_run_id: str
    ) -> bool:
        if self.raises is not None:
            raise self.raises
        self.calls.append(("resumed", owner_id, session_id, parent_run_id))
        return self.continues


async def test_a_resume_command_asks_the_driver_to_continue_the_goal_now() -> None:
    from dlightrag.application.access.principal import (
        owner_id_from_user,
    )
    from tests.support.goals import FakeGoalStore

    goal_store = FakeGoalStore()
    store = AsyncMock()
    rounds = FakeGoalRounds()
    service = _command_service(store, goal_store=goal_store, goal_rounds=rounds)

    await service.command(None, _CID, "goal", "ship the report")
    await service.command(None, _CID, "goal", "pause")
    rounds.calls.clear()

    await service.command(None, _CID, "goal", "resume")

    assert rounds.calls == [("resumed", owner_id_from_user(None), _CID, "run-parent")]


async def test_a_create_command_starts_the_first_round_now() -> None:
    from dlightrag.application.access.principal import (
        owner_id_from_user,
    )
    from tests.support.goals import FakeGoalStore

    goal_store = FakeGoalStore()
    store = AsyncMock()
    rounds = FakeGoalRounds()
    service = _command_service(store, goal_store=goal_store, goal_rounds=rounds)

    await service.command(None, _CID, "goal", "ship the report")

    assert rounds.calls == [("created", owner_id_from_user(None), _CID, _CID)]
    assert rounds.workspaces == ("default",)
    goal = goal_store.rows[(owner_id_from_user(None), _CID)]
    assert goal.phase.value == "active"
    assert store.append_command.await_args.kwargs["tone"] == "success"


async def test_a_create_command_keeps_the_goal_when_no_round_can_start() -> None:
    """The driver refusing or failing must not undo a goal the human already created."""
    from tests.support.goals import FakeGoalStore

    goal_store = FakeGoalStore()
    store = AsyncMock()
    rounds = FakeGoalRounds(raises=RuntimeError("session busy"))
    service = _command_service(store, goal_store=goal_store, goal_rounds=rounds)

    await service.command(None, _CID, "goal", "ship the report")

    from dlightrag.application.access.principal import (
        owner_id_from_user,
    )

    assert (owner_id_from_user(None), _CID) in goal_store.rows
    assert store.append_command.await_args.kwargs["tone"] == "success"


async def test_only_create_and_resume_ask_the_driver_to_start_a_round() -> None:
    from tests.support.goals import FakeGoalStore

    store = AsyncMock()
    rounds = FakeGoalRounds()
    service = _command_service(
        store, goal_store=FakeGoalStore(), goal_rounds=rounds, parent_run_id=None
    )

    for raw in ("edit ship it well", "cap 12", "pause", "clear"):
        await service.command(None, _CID, "goal", raw)

    assert rounds.calls == []


@pytest.mark.parametrize("raw", ["pause", "edit ship it well", "clear"])
async def test_a_goal_command_that_ends_the_old_objective_drops_the_continuation_behind_it(
    raw: str,
) -> None:
    """Pausing, editing (the old objective's continuation) and clearing all withdraw it."""
    from dlightrag.application.access.principal import owner_id_from_user
    from tests.support.goals import FakeGoalStore

    store = AsyncMock()
    service = _command_service(store, goal_store=FakeGoalStore())
    store.pending_goal_round.return_value = _pending_goal_round()

    await service.command(None, _CID, "goal", "ship the report")
    store.clear_goal_round.reset_mock()

    await service.command(None, _CID, "goal", raw)

    store.clear_goal_round.assert_awaited_once_with(
        owner_id_from_user(None),
        _CID,
        request_id=_PENDING_REQUEST_ID,
        prompt_digest=_PENDING_PROMPT_DIGEST,
    )


async def test_pausing_an_active_goal_cancels_its_running_goal_round() -> None:
    """The owner's pause stops now: a live goal round is asked to cancel, not left running."""
    from dlightrag.application.access.principal import (
        owner_id_from_user,
    )
    from dlightrag.application.web_conversations.models import LiveAnswerRun
    from tests.support.goals import FakeGoalStore

    store = AsyncMock()
    canceller = AsyncMock()
    canceller.cancel.return_value = True
    service = _command_service(store, goal_store=FakeGoalStore(), goal_round_canceller=canceller)
    store.live_answer_run.return_value = LiveAnswerRun(
        run_id="run-goal", status="running", continuation_kind="goal_round"
    )

    await service.command(None, _CID, "goal", "ship the report")
    await service.command(None, _CID, "goal", "pause")

    canceller.cancel.assert_awaited_once_with(owner_id=owner_id_from_user(None), run_id="run-goal")
    # The phase change is what stands; the cancellation is best effort beside it.
    assert store.append_command.await_args.kwargs["result_data"]["goal"]["phase"] == "paused"
    assert store.append_command.await_args.kwargs["tone"] == "success"


async def test_pausing_never_cancels_a_live_human_turn() -> None:
    from dlightrag.application.web_conversations.models import LiveAnswerRun
    from tests.support.goals import FakeGoalStore

    store = AsyncMock()
    canceller = AsyncMock()
    service = _command_service(store, goal_store=FakeGoalStore(), goal_round_canceller=canceller)
    store.live_answer_run.return_value = LiveAnswerRun(run_id="run-human", status="running")

    await service.command(None, _CID, "goal", "ship the report")
    await service.command(None, _CID, "goal", "pause")

    canceller.cancel.assert_not_awaited()


async def test_pausing_is_idle_safe_when_no_run_is_live() -> None:
    from tests.support.goals import FakeGoalStore

    store = AsyncMock()
    canceller = AsyncMock()
    service = _command_service(store, goal_store=FakeGoalStore(), goal_round_canceller=canceller)
    store.live_answer_run.return_value = None

    await service.command(None, _CID, "goal", "ship the report")
    await service.command(None, _CID, "goal", "pause")

    canceller.cancel.assert_not_awaited()
    assert store.append_command.await_args.kwargs["result_data"]["goal"]["phase"] == "paused"


async def test_a_failed_cancellation_leaves_the_goal_paused() -> None:
    """Best effort: a Run that refuses or hangs the cancellation cannot un-pause the goal."""
    from dlightrag.application.web_conversations.models import LiveAnswerRun
    from tests.support.goals import FakeGoalStore

    store = AsyncMock()
    canceller = AsyncMock()
    canceller.cancel.side_effect = RuntimeError("cancel refused")
    service = _command_service(store, goal_store=FakeGoalStore(), goal_round_canceller=canceller)
    store.live_answer_run.return_value = LiveAnswerRun(
        run_id="run-goal", status="running", continuation_kind="goal_round"
    )

    await service.command(None, _CID, "goal", "ship the report")
    await service.command(None, _CID, "goal", "pause")

    assert store.append_command.await_args.kwargs["result_data"]["goal"]["phase"] == "paused"
    assert store.append_command.await_args.kwargs["tone"] == "success"


async def test_a_bare_goal_command_reads_the_state_with_no_goal() -> None:
    store = AsyncMock()
    service = _command_service(store)

    command = await service.command(None, _CID, "goal", "")

    assert command is not None
    assert store.append_command.await_args.args[1] == _CID
    assert store.append_command.await_args.kwargs == {
        "name": "goal",
        "command_input": "",
        "result": "No goal in this conversation. /goal <objective> creates one.",
        "tone": "success",
        "result_data": {"kind": "goal_status", "goal": None},
    }


async def test_the_goal_status_text_names_the_canonical_blocked_sentence() -> None:
    from dlightrag.application.access.principal import (
        owner_id_from_user,
    )
    from dlightrag.engine.answer.goal import Goal, GoalPhase
    from tests.support.goals import FakeGoalStore

    goal_store = FakeGoalStore()
    goal_store.rows[(owner_id_from_user(None), _CID)] = Goal(
        id="goal-1",
        revision=4,
        objective="ship the memo",
        phase=GoalPhase.BLOCKED,
        max_rounds=7,
        rounds_started=3,
        blocked_reason="the portal is unreachable",
    )
    store = AsyncMock()
    service = _command_service(store, goal_store=goal_store)

    await service.command(None, _CID, "goal", "")

    assert store.append_command.await_args.kwargs["result"] == (
        "Blocked goal\n"
        "Objective: ship the memo\n"
        "Round: 3/7\n"
        "Blocked: the portal is unreachable\n" + _COMMAND_HELP
    )
    assert store.append_command.await_args.kwargs["tone"] == "success"
    assert store.append_command.await_args.kwargs["result_data"] == {
        "kind": "goal_status",
        "goal": {
            "objective": "ship the memo",
            "phase": "blocked",
            "max_rounds": 7,
            "rounds_started": 3,
            "blocked_reason": "the portal is unreachable",
        },
    }


async def test_goal_commands_create_edit_cap_pause_resume_and_clear() -> None:
    from dlightrag.application.access.principal import (
        owner_id_from_user,
    )
    from tests.support.goals import FakeGoalStore

    goal_store = FakeGoalStore()
    store = AsyncMock()
    service = _command_service(store, goal_store=goal_store)

    for raw in ("ship the report", "cap 12", "edit ship it well", "pause", "resume", "clear"):
        await service.command(None, _CID, "goal", raw)

    from dlightrag.engine.answer.goal import DEFAULT_MAX_ROUNDS

    results = [call.kwargs["result"] for call in store.append_command.await_args_list]
    assert results[0] == (
        f"Ongoing goal\nObjective: ship the report\nRound: 0/{DEFAULT_MAX_ROUNDS}\n" + _COMMAND_HELP
    )
    assert results[1] == "Ongoing goal\nObjective: ship the report\nRound: 0/12\n" + _COMMAND_HELP
    assert results[2] == "Ongoing goal\nObjective: ship it well\nRound: 0/12\n" + _COMMAND_HELP
    assert results[3] == (
        "Paused goal\nObjective: ship it well\nRound: 0/12\n"
        "Paused. The goal round in progress was cancelled; no next round will start. "
        "Your own running turn is not stopped.\n" + _COMMAND_HELP
    )
    assert results[4] == "Ongoing goal\nObjective: ship it well\nRound: 0/12\n" + _COMMAND_HELP
    assert results[5] == "Goal cleared."
    assert [call.kwargs["tone"] for call in store.append_command.await_args_list] == [
        "success",
    ] * 6
    payloads = [call.kwargs["result_data"] for call in store.append_command.await_args_list]
    assert payloads[0] == {
        "kind": "goal_status",
        "goal": {
            "objective": "ship the report",
            "phase": "active",
            "max_rounds": DEFAULT_MAX_ROUNDS,
            "rounds_started": 0,
            "blocked_reason": None,
        },
    }
    assert payloads[1]["goal"]["max_rounds"] == 12
    assert payloads[2]["goal"]["objective"] == "ship it well"
    assert payloads[3]["goal"]["phase"] == "paused"
    assert payloads[4]["goal"]["phase"] == "active"
    assert payloads[5] == {"kind": "goal_cleared"}
    assert (owner_id_from_user(None), _CID) not in goal_store.rows


async def test_refused_goal_commands_are_persisted_with_an_error_tone() -> None:
    store = AsyncMock()
    service = _command_service(store)

    for raw in ("pause", "cap 0", "cap", "edit", "cap 9007199254740992"):
        await service.command(None, _CID, "goal", raw)

    calls = store.append_command.await_args_list
    assert [call.kwargs["result"] for call in calls] == [
        "this conversation carries no goal",
        "Add a positive number: /goal cap 5",
        "Add a positive number: /goal cap 5",
        "Add an objective: /goal <objective>",
        "Add a positive number: /goal cap 5",
    ]
    assert [call.kwargs["tone"] for call in calls] == ["error"] * 5
    assert [call.kwargs["result_data"] for call in calls] == [
        {
            "kind": "goal_error",
            "code": "goal_not_found",
            "message": "this conversation carries no goal",
        },
        {"kind": "goal_invalid", "reason": "cap"},
        {"kind": "goal_invalid", "reason": "cap"},
        {"kind": "goal_invalid", "reason": "objective"},
        {"kind": "goal_invalid", "reason": "cap"},
    ]


async def test_a_goal_command_keeps_the_callers_input_exactly_as_sent() -> None:
    from dlightrag.application.access.principal import (
        owner_id_from_user,
    )
    from tests.support.goals import FakeGoalStore

    goal_store = FakeGoalStore()
    store = AsyncMock()
    service = _command_service(store, goal_store=goal_store)
    raw = "  ship   the   plan  "

    await service.command(None, _CID, "goal", raw)

    assert store.append_command.await_args.kwargs["command_input"] == raw
    assert goal_store.rows[(owner_id_from_user(None), _CID)].objective == "ship   the   plan"


async def test_a_command_for_a_conversation_this_principal_does_not_own_is_a_miss() -> None:
    store = AsyncMock()
    service = _command_service(store)
    store.submission_seed.return_value = None

    command = await service.command(None, _CID, "goal", "")

    assert command is None
    store.append_command.assert_not_awaited()


async def test_an_unregistered_command_is_refused_before_any_store_read() -> None:
    store = AsyncMock()
    service = _command_service(store)

    with pytest.raises(UnknownConversationCommand):
        await service.command(None, _CID, "not-a-command", "")

    store.submission_seed.assert_not_awaited()
    store.append_command.assert_not_awaited()


# ---------------------------------------------------------------------------
# Run activity (read-only projection)
# ---------------------------------------------------------------------------

_ACTIVITY_CREATED_AT = datetime.datetime(2026, 10, 24, 10, 11, 12, 123456, tzinfo=datetime.UTC)
_ACTIVITY_STARTED_AT = datetime.datetime(2026, 10, 24, 10, 11, 13, tzinfo=datetime.UTC)
_ACTIVITY_FINISHED_AT = datetime.datetime(2026, 10, 24, 10, 13, 14, 500000, tzinfo=datetime.UTC)


def _run_activity(
    *,
    run_id: str = "00000000-0000-0000-0000-0000000000a1",
    conversation_title: str | None = "Volvo outlook",
    status: Any = "running",
    phase: str | None = "researching",
    finished_at: datetime.datetime | None = None,
    goal_round: bool = False,
) -> WebRunActivity:
    return WebRunActivity(
        run_id=run_id,
        conversation_id=_CID,
        conversation_title=conversation_title,
        status=status,
        phase=phase,
        created_at=_ACTIVITY_CREATED_AT,
        started_at=_ACTIVITY_STARTED_AT,
        finished_at=finished_at,
        goal_round=goal_round,
    )


def _activity_service(store: AsyncMock) -> Any:
    from dlightrag.application.web_conversations.service import WebConversationService

    return WebConversationService(
        store=store,
        answers=FakeAnswers(),
        max_attachments=6,
        cursor_secret=b"unit-test-cursor-secret",
    )


async def test_run_activity_returns_the_frozen_wire_projection(
    conversation_client: AsyncClient,
    conversation_service: Any,
) -> None:
    conversation_service.run_activity.return_value = (_run_activity(),)

    response = await conversation_client.get("/web/api/run-activity")

    assert response.status_code == 200
    assert response.json() == {
        "items": [
            {
                "run_id": "00000000-0000-0000-0000-0000000000a1",
                "conversation_id": _CID,
                "conversation_title": "Volvo outlook",
                "status": "running",
                "phase": "researching",
                "created_at": "2026-10-24T10:11:12.123456Z",
                "started_at": "2026-10-24T10:11:13Z",
                "finished_at": None,
                "goal_round": False,
            }
        ]
    }


async def test_run_activity_states_the_goal_round_fact_without_raw_continuation_fields(
    conversation_client: AsyncClient,
    conversation_service: Any,
) -> None:
    conversation_service.run_activity.return_value = (
        _run_activity(run_id="00000000-0000-0000-0000-0000000000b2", goal_round=True),
    )

    response = await conversation_client.get("/web/api/run-activity")

    assert response.status_code == 200
    (item,) = response.json()["items"]
    assert item["goal_round"] is True
    # The projection states the derived fact only: no raw continuation field crosses.
    assert "continuation_kind" not in item
    assert "parent_run_id" not in item


async def test_run_activity_renders_a_terminal_run_with_an_untitled_conversation(
    conversation_client: AsyncClient,
    conversation_service: Any,
) -> None:
    conversation_service.run_activity.return_value = (
        _run_activity(
            status="succeeded",
            phase=None,
            conversation_title=None,
            finished_at=_ACTIVITY_FINISHED_AT,
        ),
    )

    response = await conversation_client.get("/web/api/run-activity")

    assert response.status_code == 200
    (item,) = response.json()["items"]
    assert item["status"] == "succeeded"
    assert item["phase"] is None
    assert item["conversation_title"] is None
    assert item["started_at"] == "2026-10-24T10:11:13Z"
    assert item["finished_at"] == "2026-10-24T10:13:14.500000Z"


async def test_run_activity_is_an_empty_page_and_forwards_the_principal_and_default_limit(
    conversation_client: AsyncClient,
    conversation_service: Any,
) -> None:
    conversation_service.run_activity.return_value = ()

    response = await conversation_client.get("/web/api/run-activity")

    assert response.status_code == 200
    assert response.json() == {"items": []}
    conversation_service.run_activity.assert_awaited_once()
    (user,) = conversation_service.run_activity.await_args.args
    assert user.user_id == "anonymous"
    assert conversation_service.run_activity.await_args.kwargs == {"limit": 30}


async def test_run_activity_forwards_an_explicit_limit(
    conversation_client: AsyncClient,
    conversation_service: Any,
) -> None:
    conversation_service.run_activity.return_value = ()

    response = await conversation_client.get("/web/api/run-activity?limit=7")

    assert response.status_code == 200
    assert conversation_service.run_activity.await_args.kwargs == {"limit": 7}


@pytest.mark.parametrize(
    "query",
    (
        pytest.param("limit=0", id="limit-too-small"),
        pytest.param("limit=101", id="limit-too-large"),
        pytest.param("limit=all", id="limit-not-integer"),
    ),
)
async def test_invalid_run_activity_limits_are_422_before_the_service_runs(
    conversation_client: AsyncClient,
    conversation_service: Any,
    query: str,
) -> None:
    response = await conversation_client.get(f"/web/api/run-activity?{query}")

    assert response.status_code == 422
    conversation_service.run_activity.assert_not_awaited()


async def test_run_activity_store_unavailability_returns_the_shared_503(
    test_config: DlightragConfig,
) -> None:
    from dlightrag.application.access.principal import (
        owner_id_from_user,
    )
    from dlightrag.application.web_conversations.models import WebConversationUnavailableError

    store = AsyncMock()
    store.recent_answer_runs.side_effect = WebConversationUnavailableError()
    application = create_app()
    application.state.application = application_double(
        test_config, web_conversations=_activity_service(store)
    )
    transport = ASGITransport(app=application)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/web/api/run-activity")

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Web conversation storage is unavailable",
        "error_type": "unavailable",
    }
    # A real service forwards the authenticated principal and the bounded page size.
    store.recent_answer_runs.assert_awaited_once_with(owner_id_from_user(None), limit=30)


@pytest.mark.parametrize("limit", (0, 101))
async def test_the_service_rejects_an_out_of_bounds_limit_for_an_internal_caller(
    limit: int,
) -> None:
    store = AsyncMock()
    service = _activity_service(store)

    with pytest.raises(ValueError, match="run activity limit"):
        await service.run_activity(None, limit=limit)

    store.recent_answer_runs.assert_not_awaited()


# ---------------------------------------------------------------------------
# Messages queued behind a live Run
# ---------------------------------------------------------------------------


def _queued_message(
    *,
    message_id: str = _MID,
    content: str = "hold this",
    position: int = 1,
) -> QueuedMessage:
    return QueuedMessage(
        message_id=message_id,
        content=content,
        position=position,
        created_at=datetime.datetime(2026, 7, 12, 3, 4, 5, tzinfo=datetime.UTC),
    )


async def test_queue_lists_messages_in_the_order_they_were_queued(
    conversation_client: AsyncClient,
    conversation_service: Any,
) -> None:
    conversation_service.queue.return_value = (
        _queued_message(content="first", position=1),
        _queued_message(
            message_id="00000000-0000-0000-0000-000000000003",
            content="second",
            position=2,
        ),
    )

    response = await conversation_client.get(f"/web/api/conversations/{_CID}/queue")

    assert response.status_code == 200
    body = response.json()["items"]
    assert [item["content"] for item in body] == ["first", "second"]
    assert [item["position"] for item in body] == [1, 2]
    assert body[0]["message_id"] == _MID
    assert body[0]["created_at"] == "2026-07-12T03:04:05Z"
    (user, conversation_id) = conversation_service.queue.await_args.args
    assert user.user_id == "anonymous"
    assert conversation_id == _CID


async def test_queue_enqueue_forwards_the_body_and_returns_the_row(
    conversation_client: AsyncClient,
    conversation_service: Any,
) -> None:
    conversation_service.enqueue.return_value = _queued_message(content="hold this")

    response = await conversation_client.post(
        f"/web/api/conversations/{_CID}/queue",
        json={"content": "  hold this\n"},
    )

    assert response.status_code == 201
    assert response.json()["message"]["content"] == "hold this"
    call = conversation_service.enqueue.await_args.args
    # Trimming and refusing a blank are the application's rule, not the transport's.
    assert call[1:] == (_CID, "  hold this\n")


async def test_queue_enqueue_for_an_unknown_conversation_is_404(
    conversation_client: AsyncClient,
    conversation_service: Any,
) -> None:
    conversation_service.enqueue.return_value = None

    response = await conversation_client.post(
        f"/web/api/conversations/{_CID}/queue",
        json={"content": "hold this"},
    )

    assert response.status_code == 404
    conversation_service.enqueue.assert_awaited_once()


async def test_queue_edit_and_remove_round_trip(
    conversation_client: AsyncClient,
    conversation_service: Any,
) -> None:
    conversation_service.edit_queued.return_value = _queued_message(content="rewritten")

    edited = await conversation_client.patch(
        f"/web/api/conversations/{_CID}/queue/{_MID}",
        json={"content": "rewritten"},
    )

    assert edited.status_code == 200
    assert edited.json()["message"]["content"] == "rewritten"
    call = conversation_service.edit_queued.await_args.args
    assert call[1:] == (_CID, _MID, "rewritten")

    conversation_service.remove_queued.return_value = True
    removed = await conversation_client.delete(f"/web/api/conversations/{_CID}/queue/{_MID}")
    assert removed.status_code == 204
    call = conversation_service.remove_queued.await_args.args
    assert call[1:] == (_CID, _MID)

    conversation_service.remove_queued.return_value = False
    missing = await conversation_client.delete(f"/web/api/conversations/{_CID}/queue/{_MID}")
    assert missing.status_code == 404


async def test_queue_reorder_maps_uuids_to_the_service_contract(
    conversation_client: AsyncClient,
    conversation_service: Any,
) -> None:
    ordered = [
        _queued_message(message_id=_MID, content="second", position=1),
        _queued_message(
            message_id="00000000-0000-0000-0000-000000000003",
            content="first",
            position=2,
        ),
    ]
    conversation_service.reorder_queue.return_value = tuple(ordered)

    response = await conversation_client.put(
        f"/web/api/conversations/{_CID}/queue/order",
        json={"message_ids": [item.message_id for item in ordered]},
    )

    assert response.status_code == 200
    assert [item["content"] for item in response.json()["items"]] == ["second", "first"]
    call = conversation_service.reorder_queue.await_args.args
    assert call[1] == _CID
    assert call[2] == [item.message_id for item in ordered]


async def test_queue_reorder_conflict_is_409(
    conversation_client: AsyncClient,
    conversation_service: Any,
) -> None:
    conversation_service.reorder_queue.side_effect = ConversationQueueConflict()

    response = await conversation_client.put(
        f"/web/api/conversations/{_CID}/queue/order",
        json={"message_ids": [_MID]},
    )

    assert response.status_code == 409


@pytest.mark.parametrize(
    "body",
    (
        pytest.param({"content": ""}, id="empty"),
        pytest.param({}, id="missing"),
        pytest.param({"content": "x" * 20_001}, id="too-long"),
    ),
)
async def test_a_malformed_queue_message_is_422_before_the_service_runs(
    conversation_client: AsyncClient,
    conversation_service: Any,
    body: dict[str, Any],
) -> None:
    response = await conversation_client.post(
        f"/web/api/conversations/{_CID}/queue",
        json=body,
    )

    assert response.status_code == 422
    conversation_service.enqueue.assert_not_awaited()


async def test_a_blank_queued_message_is_refused_by_the_service(
    conversation_client: AsyncClient,
    conversation_service: Any,
) -> None:
    """The strip-and-refuse rule is the application's, so every transport gets it once."""
    conversation_service.enqueue.side_effect = ApplicationInputError(
        "a queued message must not be blank"
    )

    response = await conversation_client.post(
        f"/web/api/conversations/{_CID}/queue",
        json={"content": "   \n"},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "a queued message must not be blank"
    conversation_service.enqueue.assert_awaited_once()


async def test_queue_reorder_needs_at_least_one_message_id(
    conversation_client: AsyncClient,
    conversation_service: Any,
) -> None:
    response = await conversation_client.put(
        f"/web/api/conversations/{_CID}/queue/order",
        json={"message_ids": []},
    )

    assert response.status_code == 422
    conversation_service.reorder_queue.assert_not_awaited()


async def test_queue_content_passes_the_reader_content_gate(
    conversation_client: AsyncClient,
    conversation_service: Any,
    application: Any,
) -> None:
    from types import SimpleNamespace

    application.skill_protection = SimpleNamespace(
        mask=lambda text: text.replace("SEALED-DOC", "[masked]")
    )
    conversation_service.queue.return_value = (_queued_message(content="SEALED-DOC"),)

    response = await conversation_client.get(f"/web/api/conversations/{_CID}/queue")

    assert response.status_code == 200
    assert response.json()["items"][0]["content"] == "[masked]"


# ---------------------------------------------------------------------------
# Deleting a conversation reclaims its Session's head trees
# ---------------------------------------------------------------------------


async def test_deleting_a_conversation_reclaims_its_session_heads() -> None:
    store = AsyncMock()
    store.conversation_agent_session.return_value = "session-1"
    store.delete_conversation.return_value = True
    heads = _HeadReclaimer()
    service = _delivery_service(store, AsyncMock(spec=AnswerService), heads)

    assert await service.delete(None, _CID) is True

    from dlightrag.application.access.principal import (
        owner_id_from_user,
    )

    owner = owner_id_from_user(None)
    assert heads.sessions == [(owner, "session-1")]
    store.delete_conversation.assert_awaited_once_with(owner, _CID)


async def test_a_refused_or_missing_conversation_reclaims_nothing() -> None:
    store = AsyncMock()
    store.conversation_agent_session.return_value = None
    store.delete_conversation.return_value = True
    heads = _HeadReclaimer()
    service = _delivery_service(store, AsyncMock(spec=AnswerService), heads)
    assert await service.delete(None, _CID) is True
    assert heads.sessions == []

    store.conversation_agent_session.return_value = "session-1"
    store.delete_conversation.return_value = False
    assert await service.delete(None, _CID) is False
    assert heads.sessions == []


async def test_a_reclaim_failure_never_undoes_the_deletion() -> None:
    store = AsyncMock()
    store.conversation_agent_session.return_value = "session-1"
    store.delete_conversation.return_value = True
    heads = _HeadReclaimer()
    heads.raises = RuntimeError("volume gone")
    service = _delivery_service(store, AsyncMock(spec=AnswerService), heads)

    assert await service.delete(None, _CID) is True


async def test_deleting_every_conversation_reclaims_every_owner_head() -> None:
    store = AsyncMock()
    store.agent_sessions.return_value = ("session-1", "session-2")
    store.delete_all_conversations.return_value = 2
    heads = _HeadReclaimer()
    service = _delivery_service(store, AsyncMock(spec=AnswerService), heads)

    assert await service.delete_all(None) == 2

    from dlightrag.application.access.principal import (
        owner_id_from_user,
    )

    assert heads.owners == [owner_id_from_user(None)]


async def test_the_answer_route_reports_a_busy_session_with_its_kind(
    conversation_client: AsyncClient,
    conversation_service: Any,
) -> None:
    conversation_service.start_answer.side_effect = ConversationSessionBusy("run-live")

    response = await conversation_client.post(
        "/web/api/answer",
        json={**_WEB_ANSWER_BODY, "delivery": "queue"},
    )

    assert response.status_code == 409
    body = response.json()
    assert body["kind"] == "submission_conflict"
    assert body["error_kind"] == "session_busy"
    assert "agent lane" in body["message"]


# ---------------------------------------------------------------------------
# Workspace inspector and reset
# ---------------------------------------------------------------------------


def _workspace_fixture() -> tuple[Any, Any]:
    from types import SimpleNamespace

    head = SimpleNamespace(
        lane_id="main",
        head_run_id="run-head",
        manifest_digest="a" * 64,
        revision=2,
        committed_at=datetime.datetime(2026, 8, 1, tzinfo=datetime.UTC),
    )
    snapshot = SimpleNamespace(
        run_id="run-head",
        parent_run_id=None,
        manifest_digest="a" * 64,
        file_count=2,
        changes=SimpleNamespace(added=1, changed=1, removed=0),
        created_at=datetime.datetime(2026, 8, 1, tzinfo=datetime.UTC),
    )
    return head, snapshot


async def test_the_workspace_route_projects_heads_and_snapshots(
    conversation_client: AsyncClient,
    conversation_service: Any,
) -> None:
    head, snapshot = _workspace_fixture()
    conversation_service.agentspace.return_value = ((head,), (snapshot,))

    response = await conversation_client.get(f"/web/api/conversations/{_CID}/workspace")

    assert response.status_code == 200
    body = response.json()
    assert body["heads"][0]["lane_id"] == "main"
    assert body["heads"][0]["revision"] == 2
    assert body["snapshots"][0]["file_count"] == 2
    # The diff timeline reads each snapshot's counts without a second request, and a
    # snapshot's parent being gone is carried as counts, not as an absent field.
    assert body["snapshots"][0]["changes"] == {"added": 1, "changed": 1, "removed": 0}


async def test_the_workspace_snapshot_detail_marks_how_each_file_changed(
    conversation_client: AsyncClient,
    conversation_service: Any,
) -> None:
    from types import SimpleNamespace

    head, snapshot = _workspace_fixture()
    conversation_service.agentspace.return_value = ((head,), (snapshot,))
    conversation_service.agentspace_snapshot.return_value = (
        SimpleNamespace(
            relative_path="scripts/model.py",
            content_digest="c" * 64,
            size_bytes=11,
            change="changed",
        ),
        SimpleNamespace(
            relative_path="artifacts/report.html",
            content_digest="e" * 64,
            size_bytes=20,
            change="added",
        ),
    )

    response = await conversation_client.get(
        f"/web/api/conversations/{_CID}/workspace/snapshots/run-head"
    )

    assert response.status_code == 200
    body = response.json()
    # The detail reuses the list's snapshot view, so the counts are on this payload too.
    assert body["snapshot"]["changes"] == {"added": 1, "changed": 1, "removed": 0}
    assert [(file["relative_path"], file["change"]) for file in body["files"]] == [
        ("scripts/model.py", "changed"),
        ("artifacts/report.html", "added"),
    ]


async def test_the_workspace_routes_are_404_for_an_unknown_conversation(
    conversation_client: AsyncClient,
    conversation_service: Any,
) -> None:
    conversation_service.agentspace.return_value = None
    conversation_service.agentspace_file.return_value = None

    assert (
        await conversation_client.get(f"/web/api/conversations/{_CID}/workspace")
    ).status_code == 404
    assert (
        await conversation_client.get(f"/web/api/conversations/{_CID}/workspace/snapshots/run-1")
    ).status_code == 404
    assert (
        await conversation_client.get(
            f"/web/api/conversations/{_CID}/workspace/files", params={"path": "model.py"}
        )
    ).status_code == 404


async def test_the_workspace_file_route_names_the_run_that_changed_it(
    conversation_client: AsyncClient,
    conversation_service: Any,
) -> None:
    from types import SimpleNamespace

    conversation_service.agentspace_file.return_value = (
        SimpleNamespace(
            run_id="run-1",
            content_digest="c" * 64,
            size_bytes=11,
            created_at=datetime.datetime(2026, 8, 1, tzinfo=datetime.UTC),
            change="added",
        ),
    )

    response = await conversation_client.get(
        f"/web/api/conversations/{_CID}/workspace/files", params={"path": "scripts/model.py"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["relative_path"] == "scripts/model.py"
    assert body["versions"][0]["change"] == "added"
    assert body["versions"][0]["run_id"] == "run-1"


async def test_resetting_the_workspace_is_a_human_control_at_rest(
    conversation_client: AsyncClient,
    conversation_service: Any,
) -> None:
    conversation_service.reset_agentspace.return_value = True

    response = await conversation_client.post(f"/web/api/conversations/{_CID}/workspace/reset")

    assert response.status_code == 204
    conversation_service.reset_agentspace.assert_awaited_once()


async def test_resetting_a_busy_workspace_is_the_typed_refusal(
    conversation_client: AsyncClient,
    conversation_service: Any,
) -> None:
    conversation_service.reset_agentspace.side_effect = ConversationSessionBusy("run-live")

    response = await conversation_client.post(f"/web/api/conversations/{_CID}/workspace/reset")

    assert response.status_code == 409
    body = response.json()
    assert body["kind"] == "submission_conflict"
    assert body["error_kind"] == "session_busy"


async def test_the_service_reset_reclaims_the_session_trees_after_clearing_rows() -> None:
    from dlightrag.application.access.principal import (
        owner_id_from_user,
    )

    store = AsyncMock()
    store.reset_session_agentspace.return_value = "session-1"
    heads = _HeadReclaimer()
    service = _delivery_service(store, AsyncMock(spec=AnswerService), heads)

    assert await service.reset_agentspace(None, _CID) is True

    assert heads.sessions == [(owner_id_from_user(None), "session-1")]


async def test_a_reset_that_lost_the_session_still_removes_nothing_else() -> None:
    store = AsyncMock()
    store.reset_session_agentspace.return_value = None
    heads = _HeadReclaimer()
    service = _delivery_service(store, AsyncMock(spec=AnswerService), heads)

    assert await service.reset_agentspace(None, _CID) is False
    assert heads.sessions == []


# ---------------------------------------------------------------------------
# A goal round is delivered as the conversation's next turn (Phase 4a)
# ---------------------------------------------------------------------------


def _goal_round_service(
    store: AsyncMock,
    answers: AsyncMock,
    *,
    phase: Any,
    rounds_started: int = 4,
    goal_id: str = "goal-1",
    releases: AsyncMock | None = None,
) -> Any:
    store.submission_exists.return_value = False
    from types import SimpleNamespace

    from dlightrag.application.web_conversations.service import WebConversationService

    goal = SimpleNamespace(id=goal_id, phase=phase, rounds_started=rounds_started, revision=4)

    async def _get(*, owner_id: str, session_id: str) -> Any:
        del owner_id, session_id
        return goal

    goals = SimpleNamespace(get=_get, release_round=releases or AsyncMock())
    return WebConversationService(
        store=store,
        answers=answers,
        max_attachments=6,
        cursor_secret=b"unit-test-cursor-secret",
        goals=cast(Any, goals),
    )


_PENDING_REQUEST_ID = "00000000-0000-4000-8000-0000000000ab"
_PENDING_PROMPT_DIGEST = "a" * 64


def _pending_goal_round() -> Any:
    from types import SimpleNamespace

    return SimpleNamespace(
        request_id=_PENDING_REQUEST_ID,
        content="<goal_round>Objective: finish the model</goal_round>",
        prompt_digest=_PENDING_PROMPT_DIGEST,
        workspaces=("default",),
        goal_id="goal-1",
        goal_revision=4,
        goal_round=4,
        created_at=datetime.datetime(2026, 8, 1, tzinfo=datetime.UTC),
    )


async def test_a_goal_round_is_delivered_as_the_next_turn_at_rest() -> None:
    from types import SimpleNamespace

    from dlightrag.application.web_conversations.models import GoalRoundGuard

    store = AsyncMock()
    store.find_turn_by_run.return_value = SimpleNamespace(conversation_id=_CID)
    store.queue_messages.return_value = ()
    store.pending_goal_round.return_value = _pending_goal_round()
    store.submission_seed.return_value = _submission_seed(parent_run_id="run-parent")
    answers = AsyncMock(spec=AnswerService)
    answers.accept.return_value = object()
    service = _goal_round_service(store, answers, phase="active")
    from dlightrag.engine.answer.goal import GoalPhase

    service = _goal_round_service(store, answers, phase=GoalPhase.ACTIVE)

    delivered = await service.accept_queued_turn(
        "owner-1", parent_run_id="run-parent", workspaces=("default",)
    )

    assert delivered is True
    request = answers.accept.await_args.kwargs["request"]
    assert request.continuation_kind == "goal_round"
    assert request.mode == "research"
    assert request.goal_id == "goal-1"
    assert request.goal_revision == 4
    assert request.goal_round == 4
    assert "finish the model" in request.query
    acceptor = answers.accept.await_args.kwargs["acceptor"]
    assert isinstance(acceptor.goal_round_guard, GoalRoundGuard)
    assert acceptor.goal_round_guard.prompt_digest == "a" * 64
    assert acceptor.goal_round_guard.goal_id == "goal-1"
    assert acceptor.goal_round_guard.goal_revision == 4
    assert acceptor.goal_round_guard.goal_round == 4


async def test_a_paused_goal_drops_its_pending_round_instead_of_continuing() -> None:
    from types import SimpleNamespace

    from dlightrag.engine.answer.goal import GoalPhase

    store = AsyncMock()
    store.find_turn_by_run.return_value = SimpleNamespace(conversation_id=_CID)
    store.queue_messages.return_value = ()
    store.pending_goal_round.return_value = _pending_goal_round()
    store.submission_seed.return_value = _submission_seed(parent_run_id="run-parent")
    answers = AsyncMock(spec=AnswerService)
    service = _goal_round_service(store, answers, phase=GoalPhase.PAUSED)

    delivered = await service.accept_queued_turn(
        "owner-1", parent_run_id="run-parent", workspaces=("default",)
    )

    assert delivered is False
    answers.accept.assert_not_awaited()
    store.clear_goal_round.assert_awaited_once_with(
        "owner-1",
        _CID,
        request_id=_PENDING_REQUEST_ID,
        prompt_digest=_PENDING_PROMPT_DIGEST,
    )


async def test_human_queued_words_win_over_a_pending_goal_round() -> None:
    from types import SimpleNamespace

    store = AsyncMock()
    store.find_turn_by_run.return_value = SimpleNamespace(conversation_id=_CID)
    store.queue_messages.return_value = (_queued_message(content="mine first"),)
    store.submission_seed.return_value = _submission_seed(parent_run_id="run-parent")
    answers = AsyncMock(spec=AnswerService)
    answers.accept.return_value = object()
    from dlightrag.engine.answer.goal import GoalPhase

    service = _goal_round_service(store, answers, phase=GoalPhase.ACTIVE)

    assert await service.accept_queued_turn(
        "owner-1", parent_run_id="run-parent", workspaces=("default",)
    )
    acceptor = answers.accept.await_args.kwargs["acceptor"]
    assert acceptor.goal_round_guard is None
    assert acceptor.queue_guard is not None
    store.pending_goal_round.assert_not_awaited()


async def test_a_dropped_pending_round_releases_the_admission_it_burned() -> None:
    """A continuation dropped for a paused goal must not count against the round cap."""
    from types import SimpleNamespace

    from dlightrag.engine.answer.goal import GoalPhase

    store = AsyncMock()
    store.find_turn_by_run.return_value = SimpleNamespace(conversation_id=_CID)
    store.queue_messages.return_value = ()
    store.pending_goal_round.return_value = _pending_goal_round()
    store.submission_seed.return_value = _submission_seed(parent_run_id="run-parent")
    answers = AsyncMock(spec=AnswerService)
    releases = AsyncMock()
    service = _goal_round_service(store, answers, phase=GoalPhase.PAUSED, releases=releases)

    delivered = await service.accept_queued_turn(
        "owner-1", parent_run_id="run-parent", workspaces=("default",)
    )

    assert delivered is False
    answers.accept.assert_not_awaited()
    store.clear_goal_round.assert_awaited_once_with(
        "owner-1",
        _CID,
        request_id=_PENDING_REQUEST_ID,
        prompt_digest=_PENDING_PROMPT_DIGEST,
    )
    releases.assert_awaited_once_with(
        owner_id="owner-1", session_id=_CID, run_id="", goal_id="goal-1", revision=4
    )


async def test_a_continuation_from_a_replaced_goal_is_dropped_without_a_release() -> None:
    from types import SimpleNamespace

    from dlightrag.engine.answer.goal import GoalPhase

    store = AsyncMock()
    store.find_turn_by_run.return_value = SimpleNamespace(conversation_id=_CID)
    store.queue_messages.return_value = ()
    store.pending_goal_round.return_value = _pending_goal_round()
    store.submission_seed.return_value = _submission_seed(parent_run_id="run-parent")
    answers = AsyncMock(spec=AnswerService)
    releases = AsyncMock()
    service = _goal_round_service(
        store, answers, phase=GoalPhase.ACTIVE, goal_id="goal-replacement", releases=releases
    )

    assert not await service.accept_queued_turn(
        "owner-1", parent_run_id="run-parent", workspaces=("default",)
    )

    store.clear_goal_round.assert_awaited_once_with(
        "owner-1",
        _CID,
        request_id=_PENDING_REQUEST_ID,
        prompt_digest=_PENDING_PROMPT_DIGEST,
    )
    releases.assert_not_awaited()


async def test_a_delivery_refused_by_the_fenced_transaction_refunds_the_round() -> None:
    """The transaction's own re-read is authoritative: a refused row is dropped and refunded."""
    from types import SimpleNamespace

    from dlightrag.engine.answer.goal import GoalPhase

    store = AsyncMock()
    store.find_turn_by_run.return_value = SimpleNamespace(conversation_id=_CID)
    store.queue_messages.return_value = ()
    store.pending_goal_round.return_value = _pending_goal_round()
    store.submission_seed.return_value = _submission_seed(parent_run_id="run-parent")
    store.clear_goal_round.return_value = True
    store.submission_exists.return_value = False
    answers = AsyncMock(spec=AnswerService)
    answers.accept.return_value = None
    releases = AsyncMock()
    service = _goal_round_service(
        store, answers, phase=GoalPhase.ACTIVE, rounds_started=4, releases=releases
    )

    assert not await service.accept_queued_turn(
        "owner-1", parent_run_id="run-parent", workspaces=("default",)
    )

    store.clear_goal_round.assert_awaited_once_with(
        "owner-1",
        _CID,
        request_id=_PENDING_REQUEST_ID,
        prompt_digest=_PENDING_PROMPT_DIGEST,
    )
    releases.assert_awaited_once()


async def test_a_delivery_that_committed_a_turn_is_never_refunded() -> None:
    """A row gone because its turn committed is not a dropped round: the cap stays spent."""
    from types import SimpleNamespace

    from dlightrag.engine.answer.goal import GoalPhase

    store = AsyncMock()
    store.find_turn_by_run.return_value = SimpleNamespace(conversation_id=_CID)
    store.queue_messages.return_value = ()
    store.pending_goal_round.return_value = _pending_goal_round()
    store.submission_seed.return_value = _submission_seed(parent_run_id="run-parent")
    store.clear_goal_round.return_value = True
    answers = AsyncMock(spec=AnswerService)
    answers.accept.return_value = None
    releases = AsyncMock()
    service = _goal_round_service(
        store, answers, phase=GoalPhase.ACTIVE, rounds_started=4, releases=releases
    )
    store.submission_exists.return_value = True

    assert not await service.accept_queued_turn(
        "owner-1", parent_run_id="run-parent", workspaces=("default",)
    )

    releases.assert_not_awaited()


async def test_a_drop_that_deleted_nothing_does_not_refund() -> None:
    """The clear reports deletion; a row that was already consumed cannot be refunded twice."""
    from types import SimpleNamespace

    from dlightrag.engine.answer.goal import GoalPhase

    store = AsyncMock()
    store.find_turn_by_run.return_value = SimpleNamespace(conversation_id=_CID)
    store.queue_messages.return_value = ()
    store.pending_goal_round.return_value = _pending_goal_round()
    store.submission_seed.return_value = _submission_seed(parent_run_id="run-parent")
    store.clear_goal_round.return_value = False
    store.submission_exists.return_value = False
    answers = AsyncMock(spec=AnswerService)
    releases = AsyncMock()
    service = _goal_round_service(
        store, answers, phase=GoalPhase.PAUSED, rounds_started=4, releases=releases
    )

    assert not await service.accept_queued_turn(
        "owner-1", parent_run_id="run-parent", workspaces=("default",)
    )

    releases.assert_not_awaited()


def _identity_clear(
    rows: dict[str, Any],
    calls: list[dict[str, str]],
):
    """A store double whose delete names the row: only its own request identity can remove it."""

    async def clear(
        principal_id: str,
        conversation_id: str,
        *,
        request_id: str,
        prompt_digest: str,
    ) -> bool:
        del principal_id, conversation_id
        calls.append({"request_id": request_id, "prompt_digest": prompt_digest})
        current = rows["value"]
        if (
            current is None
            or current.request_id != request_id
            or current.prompt_digest != prompt_digest
        ):
            return False
        rows["value"] = None
        return True

    return clear


async def test_a_stale_refund_cannot_delete_a_newer_reservation() -> None:
    """The attempt's stale read names its own row; a newer reservation is left untouched."""
    from types import SimpleNamespace

    from dlightrag.engine.answer.goal import GoalPhase

    stale = _pending_goal_round()
    newer = _pending_goal_round()
    newer.request_id = "00000000-0000-4000-8000-0000000000cd"
    newer.prompt_digest = "b" * 64
    rows: dict[str, Any] = {"value": newer}
    calls: list[dict[str, str]] = []
    store = AsyncMock()
    store.find_turn_by_run.return_value = SimpleNamespace(conversation_id=_CID)
    store.queue_messages.return_value = ()
    store.pending_goal_round.return_value = stale
    store.clear_goal_round.side_effect = _identity_clear(rows, calls)
    releases = AsyncMock()
    service = _goal_round_service(
        store, AsyncMock(spec=AnswerService), phase=GoalPhase.PAUSED, releases=releases
    )

    assert not await service.accept_queued_turn(
        "owner-1", parent_run_id="run-parent", workspaces=("default",)
    )

    assert calls == [{"request_id": stale.request_id, "prompt_digest": stale.prompt_digest}], (
        "the delete names the row this attempt read"
    )
    assert rows["value"] is newer, "a newer reservation is not this attempt's to delete"
    releases.assert_not_awaited()


async def test_a_refund_deletes_and_releases_only_its_own_reservation() -> None:
    """Now that the delete is by identity, the attempt that still owns its row refunds it."""
    from types import SimpleNamespace

    from dlightrag.engine.answer.goal import GoalPhase

    pending = _pending_goal_round()
    rows: dict[str, Any] = {"value": pending}
    calls: list[dict[str, str]] = []
    store = AsyncMock()
    store.find_turn_by_run.return_value = SimpleNamespace(conversation_id=_CID)
    store.queue_messages.return_value = ()
    store.pending_goal_round.return_value = pending
    store.clear_goal_round.side_effect = _identity_clear(rows, calls)
    store.submission_seed.return_value = _submission_seed(parent_run_id="run-parent")
    releases = AsyncMock()
    service = _goal_round_service(
        store, AsyncMock(spec=AnswerService), phase=GoalPhase.PAUSED, releases=releases
    )

    assert not await service.accept_queued_turn(
        "owner-1", parent_run_id="run-parent", workspaces=("default",)
    )

    assert calls == [{"request_id": pending.request_id, "prompt_digest": pending.prompt_digest}]
    assert rows["value"] is None
    releases.assert_awaited_once_with(
        owner_id="owner-1", session_id=_CID, run_id="", goal_id="goal-1", revision=4
    )


async def test_a_withdrawal_clears_exactly_its_own_reservation() -> None:
    from types import SimpleNamespace

    store = AsyncMock()
    store.find_turn_by_run.return_value = SimpleNamespace(conversation_id=_CID)
    store.clear_goal_round.return_value = True
    service = _delivery_service(store, AsyncMock(spec=AnswerService))

    assert (
        await service.withdraw_goal_round(
            owner_id="owner-1",
            parent_run_id="run-parent",
            request_id="request-mine",
            prompt_digest="c" * 64,
        )
        is True
    )

    store.clear_goal_round.assert_awaited_once_with(
        "owner-1",
        _CID,
        request_id="request-mine",
        prompt_digest="c" * 64,
    )


async def test_a_withdrawal_of_a_consumed_or_replaced_reservation_returns_false() -> None:
    """A gone row is a delivery's or a newer attempt's: report it and release nothing."""
    from types import SimpleNamespace

    store = AsyncMock()
    store.find_turn_by_run.return_value = SimpleNamespace(conversation_id=_CID)
    store.clear_goal_round.return_value = False
    service = _delivery_service(store, AsyncMock(spec=AnswerService))

    assert (
        await service.withdraw_goal_round(
            owner_id="owner-1",
            parent_run_id="run-parent",
            request_id="request-old",
            prompt_digest="d" * 64,
        )
        is False
    )

    store.clear_goal_round.assert_awaited_once_with(
        "owner-1",
        _CID,
        request_id="request-old",
        prompt_digest="d" * 64,
    )


async def test_a_web_conversation_holds_and_reports_a_goal_round_request() -> None:
    from types import SimpleNamespace

    store = AsyncMock()
    store.find_turn_by_run.return_value = SimpleNamespace(conversation_id=_CID)
    store.enqueue_goal_round.return_value = True
    answers = AsyncMock(spec=AnswerService)
    service = _delivery_service(store, answers)

    reservation = await service.request_goal_round(
        owner_id="owner-1",
        parent_run_id="run-parent",
        prompt="<goal_round>continue</goal_round>",
        workspaces=("alpha",),
        goal_id="goal-1",
        goal_revision=3,
        goal_round=2,
    )
    assert reservation is not None
    call = store.enqueue_goal_round.await_args
    assert call.args[:2] == ("owner-1", _CID)
    assert call.kwargs["request_id"] == reservation.request_id
    assert call.kwargs["prompt_digest"] == reservation.prompt_digest
    assert call.kwargs["content"] == "<goal_round>continue</goal_round>"
    assert call.kwargs["workspaces"] == ("alpha",)
    assert call.kwargs["goal_id"] == "goal-1"
    assert call.kwargs["goal_revision"] == 3
    assert call.kwargs["goal_round"] == 2
    assert len(call.kwargs["prompt_digest"]) == 64

    store.pending_goal_round.return_value = _pending_goal_round()
    assert await service.goal_round_pending(owner_id="owner-1", parent_run_id="run-parent")
    store.pending_goal_round.return_value = None
    assert not await service.goal_round_pending(owner_id="owner-1", parent_run_id="run-parent")


async def test_a_run_without_a_conversation_has_no_goal_round_request() -> None:
    store = AsyncMock()
    store.find_turn_by_run.return_value = None
    answers = AsyncMock(spec=AnswerService)
    service = _delivery_service(store, answers)

    assert (
        await service.request_goal_round(
            owner_id="owner-1",
            parent_run_id="run-rest",
            prompt="<goal_round>x</goal_round>",
            goal_id="goal-1",
            goal_revision=1,
            goal_round=1,
        )
        is None
    )
    store.enqueue_goal_round.assert_not_awaited()


async def test_a_created_goal_starts_a_round_in_a_fresh_conversation() -> None:
    """The no-parent path: a goal round is a research Web turn, not a continuation of a Run."""
    from dlightrag.application.web_conversations.service import WebConversationService

    store = AsyncMock()
    store.submission_seed.return_value = _submission_seed(parent_run_id=None)
    answers = AsyncMock(spec=AnswerService)
    answers.accept.return_value = object()
    service = WebConversationService(
        store=store,
        answers=answers,
        max_attachments=6,
        cursor_secret=b"unit-test-cursor-secret",
    )

    started = await service.accept_goal_round_now(
        owner_id="owner-1",
        conversation_id=_CID,
        prompt='<goal_round>\nObjective: "finish the model"\nRound: 1/16\nContinue.</goal_round>',
        workspaces=("alpha",),
        goal_id="goal-1",
        goal_revision=2,
        goal_round=1,
    )

    assert started is True
    kwargs = answers.accept.await_args.kwargs
    request = kwargs["request"]
    assert request.continuation_kind == "goal_round"
    assert request.mode == "research"
    assert request.parent_run_id is None
    assert request.workspaces == ("alpha",)
    assert request.goal_id == "goal-1"
    assert request.goal_revision == 2
    assert request.goal_round == 1
    assert "finish the model" in request.query
    assert len(kwargs["idempotency_key"]) == 36

    # The same round prompt replays: one submission id, so a retried start cannot open a second turn.
    await service.accept_goal_round_now(
        owner_id="owner-1",
        conversation_id=_CID,
        prompt=request.query,
        goal_id="goal-1",
        goal_revision=2,
        goal_round=1,
    )
    assert answers.accept.await_args_list[1].kwargs["idempotency_key"] == kwargs["idempotency_key"]


async def test_the_composer_selection_is_the_first_round_workspaces() -> None:
    from tests.support.goals import FakeGoalStore

    goal_store = FakeGoalStore()
    store = AsyncMock()
    rounds = FakeGoalRounds()
    service = _command_service(store, goal_store=goal_store, goal_rounds=rounds)
    store.newest_answer_workspaces.return_value = ("stale-turn",)

    await service.command(None, _CID, "goal", "ship it", workspaces=("alpha", "beta"))

    assert rounds.workspaces == ("alpha", "beta")


async def test_an_empty_composer_selection_keeps_the_newest_turn() -> None:
    from tests.support.goals import FakeGoalStore

    goal_store = FakeGoalStore()
    store = AsyncMock()
    rounds = FakeGoalRounds()
    service = _command_service(store, goal_store=goal_store, goal_rounds=rounds)
    store.newest_answer_workspaces.return_value = ("alpha",)

    await service.command(None, _CID, "goal", "ship it", workspaces=())

    assert rounds.workspaces == ("alpha",)


async def test_the_goal_bar_create_carries_the_supplied_workspaces() -> None:
    from dlightrag.application.access.principal import (
        owner_id_from_user,
    )
    from tests.support.goals import FakeGoalStore

    goal_store = FakeGoalStore()
    store = AsyncMock()
    rounds = FakeGoalRounds()
    service = _command_service(store, goal_store=goal_store, goal_rounds=rounds)

    await service.goal_command(
        None, _CID, action="create", objective="ship it", workspaces=("gamma",)
    )

    assert rounds.workspaces == ("gamma",)
    assert (owner_id_from_user(None), _CID) in goal_store.rows


async def test_the_first_round_carries_the_newest_turns_workspaces() -> None:
    """A `/goal` round continues the scope the conversation last ran in, not a guessed default."""
    from tests.support.goals import FakeGoalStore

    goal_store = FakeGoalStore()
    store = AsyncMock()
    rounds = FakeGoalRounds()
    service = _command_service(store, goal_store=goal_store, goal_rounds=rounds)
    store.newest_answer_workspaces.return_value = ("alpha", "beta")

    await service.command(None, _CID, "goal", "ship it")

    assert rounds.workspaces == ("alpha", "beta")


async def test_the_first_round_falls_back_to_default_with_no_earlier_turn() -> None:
    from tests.support.goals import FakeGoalStore

    goal_store = FakeGoalStore()
    store = AsyncMock()
    rounds = FakeGoalRounds()
    service = _command_service(store, goal_store=goal_store, goal_rounds=rounds)
    store.newest_answer_workspaces.return_value = None

    await service.command(None, _CID, "goal", "ship it")

    assert rounds.workspaces == ("default",)


async def test_continue_session_goal_starts_the_round_for_its_conversation() -> None:
    """The crash-recovery hook: an active goal with no Run starts its round in its own Web turn."""
    store = AsyncMock()
    store.conversation_for_session.return_value = _CID
    rounds = FakeGoalRounds()
    service = _command_service(store, goal_rounds=rounds)
    store.newest_answer_workspaces.return_value = ("alpha",)

    assert await service.continue_session_goal(owner_id="owner-1", session_id=_CID)
    assert rounds.calls == [("created", "owner-1", _CID, _CID)]
    assert rounds.workspaces == ("alpha",)


async def test_continue_session_goal_is_false_without_a_web_conversation() -> None:
    store = AsyncMock()
    store.conversation_for_session.return_value = None
    rounds = FakeGoalRounds()
    service = _command_service(store, goal_rounds=rounds)

    assert not await service.continue_session_goal(owner_id="owner-1", session_id=_CID)
    assert rounds.calls == []


async def test_a_goal_round_start_that_cannot_happen_is_a_false() -> None:
    from dlightrag.application.web_conversations.service import WebConversationService
    from dlightrag.engine.runtime.records import SessionWorkBusyError

    store = AsyncMock()
    store.submission_seed.return_value = _submission_seed(parent_run_id=None)
    answers = AsyncMock(spec=AnswerService)
    answers.accept.side_effect = SessionWorkBusyError("run-live")
    service = WebConversationService(
        store=store,
        answers=answers,
        max_attachments=6,
        cursor_secret=b"unit-test-cursor-secret",
    )

    assert not await service.accept_goal_round_now(
        owner_id="owner-1",
        conversation_id=_CID,
        prompt="<goal_round>continue</goal_round>",
        goal_id="goal-1",
        goal_revision=2,
        goal_round=1,
    )

    store.submission_seed.return_value = None
    assert not await service.accept_goal_round_now(
        owner_id="owner-1",
        conversation_id=_CID,
        prompt="<goal_round>continue</goal_round>",
        goal_id="goal-1",
        goal_revision=2,
        goal_round=1,
    )
