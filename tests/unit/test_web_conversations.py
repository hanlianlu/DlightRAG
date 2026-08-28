# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the Web conversation lifecycle routes and their failure contract."""

import datetime
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import asyncpg
import pytest
from httpx import ASGITransport, AsyncClient

from dlightrag.adapters.http.server import create_app
from dlightrag.application.config import DlightragConfig
from dlightrag.application.web_conversations import (
    ConversationCursor,
    ConversationCursorCodec,
    ConversationPage,
    ConversationPageRequest,
    ConversationSnapshot,
    ConversationSummary,
)
from tests.config_helpers import mutate_config
from tests.unit.conftest import answer_capability_view
from tests.unit.web.answer_run_fixtures import FakeAnswers, web_answer_submission

_CID = "00000000-0000-0000-0000-000000000001"


# ---------------------------------------------------------------------------
# Route-level fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def conversation_service() -> AsyncMock:
    service = AsyncMock()
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
    service.snapshot.return_value = ConversationSnapshot(
        principal_id="anonymous",
        conversation_id=_CID,
        content_revision=0,
        title=None,
        created_at=now,
        updated_at=now,
        agent_session_id=_CID,
        agent_lane_id="main",
        turns=(),
    )
    service.rename.return_value = {**summary, "title": "Renamed chat"}
    service.delete.return_value = True
    service.delete_all.return_value = 2
    service.start_answer.return_value = web_answer_submission(conversation_id=_CID)
    service.thumbnail.return_value = (b"derived-thumbnail", "image/jpeg")
    return service


@pytest.fixture
async def conversation_client(conversation_service: AsyncMock):
    application = create_app(include_web_app=True)
    application.state.application = AsyncMock()
    application.state.application.web_conversations = conversation_service
    application.state.application.config = DlightragConfig()
    application.state.application.answers.capabilities = answer_capability_view().read
    application.state.application.corpora.alist_workspace_records.return_value = [
        {"workspace": "default"}
    ]
    transport = ASGITransport(app=application)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
async def cookie_conversation_client(
    test_config: DlightragConfig,
    conversation_service: AsyncMock,
):
    mutate_config(test_config, "access.auth_mode", "simple")
    mutate_config(test_config, "access.api_token", "secret-token")
    application = create_app(include_web_app=True)
    application.state.application = AsyncMock(config=test_config)
    application.state.application.web_conversations = conversation_service
    application.state.application.answers.capabilities = answer_capability_view().read
    transport = ASGITransport(app=application)
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
    conversation_service: AsyncMock,
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
    conversation_service: AsyncMock,
) -> None:
    response = await conversation_client.get("/web/api/conversations")

    assert response.status_code == 200
    assert response.json()["next_cursor"] is None
    assert len(response.json()["items"]) == 1
    conversation_service.list.assert_awaited_once()


async def test_list_cursor_round_trips_as_paired_ordering_facts(
    conversation_client: AsyncClient,
    conversation_service: AsyncMock,
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
    conversation_service: AsyncMock,
    query: str,
    expected_status: int,
) -> None:
    response = await conversation_client.get(f"/web/api/conversations?{query}")

    assert response.status_code == expected_status
    conversation_service.list.assert_not_awaited()


async def test_tampered_cursor_is_422_before_the_service_runs(
    conversation_client: AsyncClient,
    conversation_service: AsyncMock,
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
    conversation_service: AsyncMock,
) -> None:
    conversation_service.snapshot.return_value = None

    response = await conversation_client.get(f"/web/api/conversations/{_CID}/history")

    assert response.status_code == 404


async def test_rename_validates_trimmed_title(conversation_client: AsyncClient) -> None:
    response = await conversation_client.patch(
        f"/web/api/conversations/{_CID}",
        json={"title": "   "},
    )

    assert response.status_code == 422


async def test_rename_normalizes_whitespace(
    conversation_client: AsyncClient,
    conversation_service: AsyncMock,
) -> None:
    response = await conversation_client.patch(
        f"/web/api/conversations/{_CID}",
        json={"title": "  Renamed\n chat  "},
    )

    assert response.status_code == 200
    assert conversation_service.rename.await_args.args[-1] == "Renamed chat"


async def test_delete_returns_204(
    conversation_client: AsyncClient,
    conversation_service: AsyncMock,
) -> None:
    response = await conversation_client.delete(f"/web/api/conversations/{_CID}")

    assert response.status_code == 204
    assert response.content == b""
    conversation_service.delete.assert_awaited_once()


async def test_delete_all_returns_204_when_no_conversations_exist(
    conversation_client: AsyncClient,
    conversation_service: AsyncMock,
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
    conversation_service: AsyncMock,
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
    conversation_service: AsyncMock,
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
    conversation_service: AsyncMock,
) -> None:
    response = await cookie_conversation_client.post("/web/api/conversations")

    assert response.status_code == 403
    conversation_service.create.assert_not_awaited()


async def test_bearer_lifecycle_mutation_does_not_require_browser_origin(
    cookie_conversation_client: AsyncClient,
    conversation_service: AsyncMock,
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
    conversation_service: AsyncMock,
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
    conversation_service: AsyncMock,
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
    conversation_service: AsyncMock,
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
    method: str,
    path: str,
    store_method: str,
) -> None:
    from dlightrag.application.web_conversations import (
        WebConversationService,
        WebConversationUnavailableError,
    )

    store = AsyncMock()
    getattr(store, store_method).side_effect = WebConversationUnavailableError()
    application = create_app(include_web_app=True)
    service = WebConversationService(
        store=store,
        answers=FakeAnswers(),
        max_attachments=6,
    )
    application.state.application = SimpleNamespace(web_conversations=service)
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
    from dlightrag.application.web_conversations import WebConversationUnavailableError

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
    store_error: Exception,
) -> None:
    from dlightrag.application.web_conversations import WebConversationService

    store = AsyncMock()
    store.list_conversations.side_effect = store_error
    application = create_app(include_web_app=True)
    service = WebConversationService(
        store=store,
        answers=FakeAnswers(),
        max_attachments=6,
    )
    application.state.application = SimpleNamespace(web_conversations=service)
    transport = ASGITransport(app=application)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        with pytest.raises(type(store_error), match=str(store_error)):
            await client.get("/web/api/conversations")


async def test_application_page_uses_extra_row_proof_and_last_returned_item_cursor() -> None:
    from dlightrag.application.access import owner_id_from_user
    from dlightrag.application.web_conversations import (
        ConversationRowPage,
        WebConversationService,
    )

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
    service = WebConversationService(store=store, answers=FakeAnswers(), max_attachments=6)

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


def test_cursor_codec_rejects_tampering_and_round_trips_canonical_facts() -> None:
    codec = ConversationCursorCodec(b"unit-test-cursor-secret")
    cursor = ConversationCursor(
        updated_at=datetime.datetime(2026, 7, 12, 3, 4, 5, 123456, tzinfo=datetime.UTC),
        conversation_id=UUID(_CID),
    )
    encoded = codec.encode(cursor)

    assert codec.decode(encoded) == cursor
    replacement = "A" if encoded[-1] != "A" else "B"
    with pytest.raises(ValueError, match="invalid conversation page cursor"):
        codec.decode(f"{encoded[:-1]}{replacement}")


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
