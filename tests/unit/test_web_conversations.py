# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for durable Web conversation lifecycle adapters and routes."""

import datetime
import io
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID

import asyncpg
import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from dlightrag.api.auth import UserContext
from dlightrag.api.server import create_app
from dlightrag.config import DlightragConfig
from dlightrag.storage.web_conversations import ConversationSnapshot, StoredConversationImage
from dlightrag.utils.images import thumbnail_bytes


@pytest.fixture
def conversation_service() -> AsyncMock:
    service = AsyncMock()
    now = datetime.datetime(2026, 7, 12, tzinfo=datetime.UTC)
    summary = {
        "conversation_id": "00000000-0000-0000-0000-000000000001",
        "title": None,
        "created_at": now,
        "updated_at": now,
    }
    service.create.return_value = summary
    service.list.return_value = [summary]
    service.history.return_value = {
        "conversation": summary,
        "turns": [],
    }
    service.rename.return_value = {**summary, "title": "Renamed chat"}
    service.delete.return_value = True
    service.image.return_value = SimpleNamespace(
        image_id="00000000-0000-0000-0000-000000000002",
        mime_type="image/png",
        image_bytes=b"png-bytes",
    )
    return service


@pytest.fixture
async def conversation_client(conversation_service: AsyncMock):
    application = create_app(include_web=True)
    application.state.web_conversation_service = conversation_service
    transport = ASGITransport(app=application)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
async def cookie_conversation_client(
    test_config: DlightragConfig,
    conversation_service: AsyncMock,
):
    test_config.auth_mode = "simple"
    test_config.api_auth_token = "secret-token"
    application = create_app(include_web=True)
    application.state.web_conversation_service = conversation_service
    application.state.manager = AsyncMock(config=test_config)
    transport = ASGITransport(app=application)
    async with AsyncClient(
        transport=transport,
        base_url="https://app.example.com",
        follow_redirects=False,
    ) as client:
        login = await client.post(
            "/web/login",
            data={"token": "secret-token", "next": "/web/"},
        )
        assert login.status_code == 303
        yield client


async def test_create_ignores_client_identity_and_returns_server_uuid(
    conversation_client: AsyncClient,
    conversation_service: AsyncMock,
) -> None:
    response = await conversation_client.post(
        "/web/conversations",
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
    response = await conversation_client.get("/web/conversations")

    assert response.status_code == 200
    assert len(response.json()) == 1
    conversation_service.list.assert_awaited_once()


async def test_history_of_other_principal_is_404(
    conversation_client: AsyncClient,
    conversation_service: AsyncMock,
) -> None:
    conversation_service.history.return_value = None

    response = await conversation_client.get(
        "/web/conversations/00000000-0000-0000-0000-000000000001/history"
    )

    assert response.status_code == 404


async def test_rename_validates_trimmed_title(conversation_client: AsyncClient) -> None:
    response = await conversation_client.patch(
        "/web/conversations/00000000-0000-0000-0000-000000000001",
        json={"title": "   "},
    )

    assert response.status_code == 422


async def test_rename_normalizes_whitespace(
    conversation_client: AsyncClient,
    conversation_service: AsyncMock,
) -> None:
    response = await conversation_client.patch(
        "/web/conversations/00000000-0000-0000-0000-000000000001",
        json={"title": "  Renamed\n chat  "},
    )

    assert response.status_code == 200
    assert conversation_service.rename.await_args.args[-1] == "Renamed chat"


async def test_delete_returns_204(
    conversation_client: AsyncClient,
    conversation_service: AsyncMock,
) -> None:
    response = await conversation_client.delete(
        "/web/conversations/00000000-0000-0000-0000-000000000001"
    )

    assert response.status_code == 204
    assert response.content == b""
    conversation_service.delete.assert_awaited_once()


async def test_delete_has_no_messages_subroute(conversation_client: AsyncClient) -> None:
    response = await conversation_client.delete(
        "/web/conversations/00000000-0000-0000-0000-000000000001/messages"
    )

    assert response.status_code == 404


async def test_scoped_image_response_is_private(
    conversation_client: AsyncClient,
    conversation_service: AsyncMock,
) -> None:
    response = await conversation_client.get(
        "/web/conversations/00000000-0000-0000-0000-000000000001/images/"
        "00000000-0000-0000-0000-000000000002"
    )

    assert response.status_code == 200
    assert response.content == b"png-bytes"
    assert response.headers["content-type"] == "image/png"
    assert response.headers["cache-control"] == "private, max-age=3600"


async def test_scoped_image_of_other_principal_is_404(
    conversation_client: AsyncClient,
    conversation_service: AsyncMock,
) -> None:
    conversation_service.image.return_value = None

    response = await conversation_client.get(
        "/web/conversations/00000000-0000-0000-0000-000000000001/images/"
        "00000000-0000-0000-0000-000000000002"
    )

    assert response.status_code == 404


async def test_scoped_thumbnail_response_is_private_and_immutable(
    conversation_client: AsyncClient,
    conversation_service: AsyncMock,
) -> None:
    conversation_service.thumbnail.return_value = SimpleNamespace(
        image_id="00000000-0000-0000-0000-000000000002",
        mime_type="image/jpeg",
        image_bytes=b"derived-thumbnail",
    )

    response = await conversation_client.get(
        "/web/conversations/00000000-0000-0000-0000-000000000001/images/"
        "00000000-0000-0000-0000-000000000002/thumbnail"
    )

    assert response.status_code == 200
    assert response.content == b"derived-thumbnail"
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["cache-control"] == "private, max-age=86400, immutable"
    conversation_service.thumbnail.assert_awaited_once()


async def test_scoped_thumbnail_failure_does_not_fall_back_to_original(
    conversation_client: AsyncClient,
    conversation_service: AsyncMock,
) -> None:
    conversation_service.thumbnail.return_value = None

    response = await conversation_client.get(
        "/web/conversations/00000000-0000-0000-0000-000000000001/images/"
        "00000000-0000-0000-0000-000000000002/thumbnail"
    )

    assert response.status_code == 404
    assert response.content != b"png-bytes"
    conversation_service.thumbnail.assert_awaited_once()


async def test_scoped_thumbnail_requires_web_auth(
    test_config: DlightragConfig,
    conversation_service: AsyncMock,
) -> None:
    test_config.auth_mode = "simple"
    test_config.api_auth_token = "secret-token"
    application = create_app(include_web=True)
    application.state.web_conversation_service = conversation_service
    transport = ASGITransport(app=application)
    async with AsyncClient(
        transport=transport,
        base_url="https://app.example.com",
        follow_redirects=False,
    ) as client:
        response = await client.get(
            "/web/conversations/00000000-0000-0000-0000-000000000001/images/"
            "00000000-0000-0000-0000-000000000002/thumbnail"
        )

    assert response.status_code == 303
    assert response.headers["location"].startswith("/web/login")
    conversation_service.thumbnail.assert_not_awaited()


_COOKIE_MUTATIONS = (
    pytest.param("POST", "/web/conversations", None, "create", 201, id="create"),
    pytest.param(
        "PATCH",
        "/web/conversations/00000000-0000-0000-0000-000000000001",
        {"title": "Renamed chat"},
        "rename",
        200,
        id="rename",
    ),
    pytest.param(
        "DELETE",
        "/web/conversations/00000000-0000-0000-0000-000000000001",
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
    response = await cookie_conversation_client.post("/web/conversations")

    assert response.status_code == 403
    conversation_service.create.assert_not_awaited()


async def test_bearer_lifecycle_mutation_does_not_require_browser_origin(
    cookie_conversation_client: AsyncClient,
    conversation_service: AsyncMock,
) -> None:
    response = await cookie_conversation_client.post(
        "/web/conversations",
        headers={"Authorization": "Bearer secret-token"},
    )

    assert response.status_code == 201
    conversation_service.create.assert_awaited_once()


_WEB_ANSWER_BODY = {
    "query": "hello",
    "conversation_id": "00000000-0000-0000-0000-000000000001",
    "submission_id": "00000000-0000-4000-8000-000000000099",
}


@pytest.mark.parametrize("content_type", ["application/json", "text/plain"])
async def test_cookie_web_answer_accepts_exact_origin_independent_of_content_type(
    cookie_conversation_client: AsyncClient,
    conversation_service: AsyncMock,
    content_type: str,
) -> None:
    conversation_service.prepare_answer.return_value = None

    response = await cookie_conversation_client.post(
        "/web/answer",
        content=json.dumps(_WEB_ANSWER_BODY),
        headers={
            "Content-Type": content_type,
            "Origin": "https://app.example.com",
        },
    )

    assert response.status_code == 404
    conversation_service.prepare_answer.assert_awaited_once()


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
    conversation_service.prepare_answer.return_value = None
    headers = {"Content-Type": "text/plain"}
    if origin is not None:
        headers["Origin"] = origin

    response = await cookie_conversation_client.post(
        "/web/answer",
        content=json.dumps(_WEB_ANSWER_BODY),
        headers=headers,
    )

    assert response.status_code == 403
    conversation_service.prepare_answer.assert_not_awaited()


async def test_bearer_web_answer_does_not_require_browser_origin(
    cookie_conversation_client: AsyncClient,
    conversation_service: AsyncMock,
) -> None:
    conversation_service.prepare_answer.return_value = None

    response = await cookie_conversation_client.post(
        "/web/answer",
        json=_WEB_ANSWER_BODY,
        headers={"Authorization": "Bearer secret-token"},
    )

    assert response.status_code == 404
    conversation_service.prepare_answer.assert_awaited_once()


@pytest.mark.parametrize(
    ("method", "path"),
    [
        pytest.param("POST", "/web/files/upload", id="file-upload"),
        pytest.param(
            "DELETE",
            "/web/files?workspace=default&file_path=report.pdf",
            id="file-delete",
        ),
        pytest.param("POST", "/web/workspaces/create", id="workspace-create"),
        pytest.param("POST", "/web/workspaces/delete", id="workspace-delete"),
    ],
)
async def test_cookie_web_mutations_reject_missing_origin(
    cookie_conversation_client: AsyncClient,
    method: str,
    path: str,
) -> None:
    response = await cookie_conversation_client.request(method, path)

    assert response.status_code == 403


@pytest.mark.parametrize(
    ("method", "path", "store_method"),
    (
        pytest.param("GET", "/web/conversations", "list_conversations", id="read"),
        pytest.param("POST", "/web/conversations", "create_conversation", id="mutation"),
    ),
)
async def test_store_unavailability_returns_retryable_503(
    method: str,
    path: str,
    store_method: str,
) -> None:
    from dlightrag.web.conversations import WebConversationService

    store = AsyncMock()
    getattr(store, store_method).side_effect = ConnectionError("database unavailable")
    application = create_app(include_web=True)
    application.state.web_conversation_service = WebConversationService(
        store=store,
        max_turns=100,
        ttl_days=30,
    )
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
async def test_postgres_shutdown_returns_retryable_503(shutdown_error: Exception) -> None:
    from dlightrag.web.conversations import WebConversationService

    store = AsyncMock()
    store.list_conversations.side_effect = shutdown_error
    application = create_app(include_web=True)
    application.state.web_conversation_service = WebConversationService(
        store=store,
        max_turns=100,
        ttl_days=30,
    )
    transport = ASGITransport(app=application)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/web/conversations")

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Web conversation storage is unavailable",
        "error_type": "unavailable",
    }


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
    from dlightrag.web.conversations import WebConversationService

    store = AsyncMock()
    store.list_conversations.side_effect = store_error
    application = create_app(include_web=True)
    application.state.web_conversation_service = WebConversationService(
        store=store,
        max_turns=100,
        ttl_days=30,
    )
    transport = ASGITransport(app=application)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        with pytest.raises(type(store_error), match=str(store_error)):
            await client.get("/web/conversations")


def test_browser_contracts_forbid_extra_fields_and_normalize_titles() -> None:
    from pydantic import ValidationError

    from dlightrag.web.conversation_models import RenameConversationRequest

    request = RenameConversationRequest(title="  Quarterly\n review  ")

    assert request.title == "Quarterly review"
    with pytest.raises(ValidationError):
        RenameConversationRequest.model_validate({"title": "Valid", "principal_id": "p1"})


def _conversation_row() -> dict[str, object]:
    now = datetime.datetime(2026, 7, 12, tzinfo=datetime.UTC)
    return {
        "conversation_id": "00000000-0000-0000-0000-000000000001",
        "title": "Quarterly review",
        "content_revision": 7,
        "created_at": now,
        "updated_at": now,
    }


def _conversation_snapshot() -> ConversationSnapshot:
    now = datetime.datetime(2026, 7, 12, tzinfo=datetime.UTC)
    return ConversationSnapshot(
        principal_id="stored-principal",
        conversation_id="00000000-0000-0000-0000-000000000001",
        content_revision=7,
        title="Quarterly review",
        created_at=now,
        updated_at=now,
        history=(
            {
                "turn_id": "00000000-0000-0000-0000-000000000010",
                "turn_number": 1,
                "user_text": "What changed?",
                "assistant_text": "Revenue increased [1].",
                "answer_sources": {
                    "sources": [
                        {
                            "id": "1",
                            "title": "Report",
                            "type": "document",
                            "source_uri": "local://report.pdf",
                            "download_url": "/web/files/raw/report?workspace=default",
                        }
                    ],
                    "answer_images": [],
                },
                "queried_workspaces": ["default"],
                "created_at": now,
                "images": [
                    {
                        "image_id": "00000000-0000-0000-0000-000000000020",
                        "ordinal": 1,
                        "mime_type": "image/png",
                    }
                ],
            },
        ),
    )


@pytest.fixture
def conversation_store() -> AsyncMock:
    store = AsyncMock()
    store.create_conversation.return_value = _conversation_row()
    store.list_conversations.return_value = [_conversation_row()]
    store.snapshot.return_value = _conversation_snapshot()
    store.rename_conversation.return_value = _conversation_row()
    store.delete_conversation.return_value = True
    return store


@pytest.fixture
def service_under_test(conversation_store: AsyncMock):
    from dlightrag.web.conversations import WebConversationService

    return WebConversationService(
        store=conversation_store,
        max_turns=100,
        ttl_days=30,
    )


@pytest.fixture
def jwt_user() -> UserContext:
    return UserContext(
        user_id="alice",
        auth_mode="jwt",
        claims={"iss": "https://issuer.example"},
    )


async def test_service_derives_principal_for_each_lifecycle_operation(
    service_under_test,
    conversation_store: AsyncMock,
    jwt_user: UserContext,
) -> None:
    from dlightrag.web.principal import principal_id_from_user

    expected_principal = principal_id_from_user(jwt_user)

    await service_under_test.create(jwt_user)
    await service_under_test.list(jwt_user)
    await service_under_test.rename(
        jwt_user,
        "00000000-0000-0000-0000-000000000001",
        "Quarterly review",
    )
    await service_under_test.delete(
        jwt_user,
        "00000000-0000-0000-0000-000000000001",
    )

    assert conversation_store.create_conversation.await_args.args == (expected_principal,)
    assert conversation_store.list_conversations.await_args.args == (expected_principal,)
    assert conversation_store.rename_conversation.await_args.args[:2] == (
        expected_principal,
        "00000000-0000-0000-0000-000000000001",
    )
    assert conversation_store.delete_conversation.await_args.args[:2] == (
        expected_principal,
        "00000000-0000-0000-0000-000000000001",
    )


async def test_history_projects_safe_images_sources_and_rendered_answer(
    service_under_test,
    conversation_store: AsyncMock,
    jwt_user: UserContext,
) -> None:
    history = await service_under_test.history(
        jwt_user,
        "00000000-0000-0000-0000-000000000001",
    )

    assert history is not None
    turn = history.turns[0]
    image = turn.user_images[0]
    expected_url = (
        "/web/conversations/00000000-0000-0000-0000-000000000001/images/"
        "00000000-0000-0000-0000-000000000020"
    )
    assert image.url == expected_url
    assert image.thumbnail_url == expected_url + "/thumbnail"
    assert image.label == "Turn 1, image 1"
    assert turn.answer_sources["sources"][0]["download_url"].startswith("/web/")
    assert "citation-badge" in turn.answer_html
    assert "image_bytes" not in turn.model_dump_json()
    assert "principal_id" not in turn.model_dump_json()
    conversation_store.list_conversations.assert_not_awaited()


async def test_history_thumbnail_is_principal_scoped_and_resource_bounded(
    service_under_test,
    conversation_store: AsyncMock,
    jwt_user: UserContext,
) -> None:
    source = Image.effect_noise((1600, 1200), 100).convert("RGB")
    original_buffer = io.BytesIO()
    source.save(original_buffer, format="PNG")
    original = original_buffer.getvalue()
    conversation_store.get_image.return_value = StoredConversationImage(
        image_id="00000000-0000-0000-0000-000000000020",
        mime_type="image/png",
        image_bytes=original,
    )

    thumbnail = await service_under_test.thumbnail(
        jwt_user,
        "00000000-0000-0000-0000-000000000001",
        "00000000-0000-0000-0000-000000000020",
    )

    assert thumbnail is not None
    assert thumbnail.mime_type in {"image/jpeg", "image/png"}
    assert len(thumbnail.image_bytes) <= 128 * 1024
    assert len(thumbnail.image_bytes) < len(original)
    with Image.open(io.BytesIO(thumbnail.image_bytes)) as derived:
        assert max(derived.size) <= 320
        assert derived.format in {"JPEG", "PNG"}
    from dlightrag.web.principal import principal_id_from_user

    conversation_store.get_image.assert_awaited_once_with(
        principal_id_from_user(jwt_user),
        "00000000-0000-0000-0000-000000000001",
        "00000000-0000-0000-0000-000000000020",
        ttl_days=30,
    )


async def test_history_thumbnail_generation_failure_returns_none(
    service_under_test,
    conversation_store: AsyncMock,
    jwt_user: UserContext,
) -> None:
    original = b"durable-original-but-not-a-decodable-image"
    conversation_store.get_image.return_value = StoredConversationImage(
        image_id="00000000-0000-0000-0000-000000000020",
        mime_type="image/png",
        image_bytes=original,
    )

    thumbnail = await service_under_test.thumbnail(
        jwt_user,
        "00000000-0000-0000-0000-000000000001",
        "00000000-0000-0000-0000-000000000020",
    )

    assert thumbnail is None


def test_bounded_thumbnail_handles_valid_cmyk_jpeg() -> None:
    source = Image.new("CMYK", (640, 480), (0, 127, 127, 0))
    original_buffer = io.BytesIO()
    source.save(original_buffer, format="JPEG")

    payload, mime_type = thumbnail_bytes(
        original_buffer.getvalue(),
        max_px=320,
        max_bytes=128 * 1024,
    )

    assert mime_type in {"image/jpeg", "image/png"}
    assert len(payload) <= 128 * 1024
    with Image.open(io.BytesIO(payload)) as derived:
        assert max(derived.size) <= 320


async def test_prepare_answer_uses_one_snapshot_and_text_only_messages(
    service_under_test,
    conversation_store: AsyncMock,
    jwt_user: UserContext,
) -> None:
    prepared = await service_under_test.prepare_answer(
        jwt_user,
        "00000000-0000-0000-0000-000000000001",
    )

    assert prepared is not None
    assert prepared.content_revision == 7
    assert prepared.text_history == (
        {"role": "user", "content": "What changed?"},
        {"role": "assistant", "content": "Revenue increased [1]."},
    )
    conversation_store.snapshot.assert_awaited_with(
        prepared.principal_id,
        prepared.conversation_id,
        ttl_days=30,
        max_turns=100,
    )


async def test_prepare_answer_submission_replay_lookup_is_bounded_one_shot(
    service_under_test,
    conversation_store: AsyncMock,
    jwt_user: UserContext,
) -> None:
    conversation_store.find_committed_turn.return_value = None

    prepared = await service_under_test.prepare_answer(
        jwt_user,
        "00000000-0000-0000-0000-000000000001",
        "00000000-0000-4000-8000-000000000099",
    )

    assert prepared is not None
    conversation_store.find_committed_turn.assert_awaited_once()


async def test_commit_answer_maps_validated_images_and_revision(
    service_under_test,
    conversation_store: AsyncMock,
) -> None:
    from dlightrag.storage.web_conversations import CommitTurnResult
    from dlightrag.web.attachment_models import ValidatedWebImage
    from dlightrag.web.conversations import PreparedWebConversation

    conversation_store.commit_turn.return_value = CommitTurnResult(
        saved=False, reason="conversation_changed", summary=None, turn_id=None
    )
    prepared = PreparedWebConversation(
        principal_id="principal-hash",
        conversation_id="00000000-0000-0000-0000-000000000001",
        content_revision=7,
        text_history=(),
    )
    image = ValidatedWebImage(
        image_id="00000000-0000-0000-0000-000000000020",
        ordinal=1,
        mime_type="image/png",
        image_bytes=b"png",
        data_uri="data:image/png;base64,cG5n",
        content_sha256="digest",
    )

    result = await service_under_test.commit_answer(
        prepared,
        submission_id="00000000-0000-4000-8000-000000000098",
        user_text="Question",
        assistant_text="Answer",
        answer_sources={"sources": [], "answer_images": []},
        queried_workspaces=["default"],
        images=(image,),
        image_descriptions={"1": "diagram"},
    )

    assert result.reason == "conversation_changed"
    call = conversation_store.commit_turn.await_args.kwargs
    assert call["expected_revision"] == 7
    assert call["principal_id"] == "principal-hash"
    assert call["images"][0].image_bytes == b"png"
    assert call["images"][0].vlm_description == "diagram"


async def test_commit_answer_maps_sparse_descriptions_by_stable_ordinal(
    service_under_test,
    conversation_store: AsyncMock,
) -> None:
    from dlightrag.storage.web_conversations import CommitTurnResult
    from dlightrag.web.attachment_models import ValidatedWebImage
    from dlightrag.web.conversations import PreparedWebConversation

    conversation_store.commit_turn.return_value = CommitTurnResult(
        saved=True, reason=None, summary=None, turn_id="turn"
    )
    prepared = PreparedWebConversation(
        principal_id="principal",
        conversation_id="00000000-0000-0000-0000-000000000001",
        content_revision=1,
        text_history=(),
    )
    images = tuple(
        ValidatedWebImage(
            image_id=f"00000000-0000-0000-0000-00000000002{ordinal}",
            ordinal=ordinal,
            mime_type="image/png",
            image_bytes=b"png",
            data_uri="data:image/png;base64,cG5n",
            content_sha256=f"digest-{ordinal}",
        )
        for ordinal in (1, 2)
    )

    await service_under_test.commit_answer(
        prepared,
        submission_id="00000000-0000-4000-8000-000000000099",
        user_text="Question",
        assistant_text="Answer",
        answer_sources={},
        queried_workspaces=["default"],
        images=images,
        image_descriptions={"2": "Image 2: second"},
    )

    pending = conversation_store.commit_turn.await_args.kwargs["images"]
    assert [image.vlm_description for image in pending] == [None, "Image 2: second"]


async def test_commit_answer_persists_document_planner_digest(
    service_under_test,
    conversation_store: AsyncMock,
) -> None:
    from dlightrag.storage.web_conversations import CommitTurnResult
    from dlightrag.web.attachment_models import validate_web_documents
    from dlightrag.web.conversations import PreparedWebConversation

    conversation_store.commit_turn.return_value = CommitTurnResult(
        saved=True, reason=None, summary=None, turn_id="turn"
    )
    prepared = PreparedWebConversation(
        principal_id="principal",
        conversation_id="00000000-0000-0000-0000-000000000001",
        content_revision=1,
        text_history=(),
    )
    (document,) = validate_web_documents([("report.docx", "application/octet-stream", b"document")])

    await service_under_test.commit_answer(
        prepared,
        submission_id="00000000-0000-4000-8000-000000000099",
        user_text="Question",
        assistant_text="Answer",
        answer_sources={},
        queried_workspaces=["default"],
        images=(),
        image_descriptions={},
        documents=(document,),
        document_parse_summaries={document.attachment_id: "Structured planner digest"},
    )

    pending = conversation_store.commit_turn.await_args.kwargs["attachments"]
    assert pending[0].parse_summary == "Structured planner digest"


async def test_commit_answer_reconciles_lost_commit_acknowledgement(
    service_under_test,
    conversation_store: AsyncMock,
) -> None:
    from dlightrag.storage.web_conversations import CommitTurnResult
    from dlightrag.web.conversations import PreparedWebConversation

    committed = CommitTurnResult(
        saved=True,
        reason=None,
        summary=None,
        turn_id="turn",
        current_image_ids=("stored-image",),
        assistant_text="Stored answer",
        answer_sources={"sources": []},
        replayed=True,
    )
    conversation_store.commit_turn.side_effect = ConnectionError("ack lost")
    conversation_store.find_committed_turn.return_value = committed
    prepared = PreparedWebConversation(
        principal_id="principal",
        conversation_id="00000000-0000-0000-0000-000000000001",
        content_revision=1,
        text_history=(),
    )

    result = await service_under_test.commit_answer(
        prepared,
        submission_id="00000000-0000-4000-8000-000000000099",
        user_text="Question",
        assistant_text="Answer",
        answer_sources={},
        queried_workspaces=["default"],
        images=(),
        image_descriptions={},
    )

    assert result == committed
    conversation_store.find_committed_turn.assert_awaited_once_with(
        "principal",
        "00000000-0000-0000-0000-000000000001",
        "00000000-0000-4000-8000-000000000099",
        ttl_days=30,
        retry=False,
    )


async def test_commit_answer_returns_unknown_after_bounded_reconciliation_timeout(
    service_under_test,
    conversation_store: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    from dlightrag.web.conversations import PreparedWebConversation

    monkeypatch.setattr("dlightrag.web.conversations._COMMIT_ATTEMPT_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr("dlightrag.web.conversations._RECONCILE_ATTEMPT_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr("dlightrag.web.conversations._RECONCILE_ATTEMPTS", 1)
    conversation_store.commit_turn.side_effect = ConnectionError("ack lost")
    conversation_store.find_committed_turn.side_effect = asyncio.TimeoutError
    prepared = PreparedWebConversation(
        principal_id="principal",
        conversation_id="00000000-0000-0000-0000-000000000001",
        content_revision=1,
        text_history=(),
    )

    result = await asyncio.wait_for(
        service_under_test.commit_answer(
            prepared,
            submission_id="00000000-0000-4000-8000-000000000099",
            user_text="Question",
            assistant_text="Answer",
            answer_sources={},
            queried_workspaces=["default"],
            images=(),
            image_descriptions={},
        ),
        timeout=0.1,
    )

    assert result.saved is False
    assert result.reason == "commit_outcome_unknown"
    assert result.current_image_ids == ()


async def test_initialize_applies_schema_then_global_prune(
    service_under_test,
    conversation_store: AsyncMock,
) -> None:
    await service_under_test.initialize()

    conversation_store.initialize.assert_awaited_once_with()
    conversation_store.prune_expired.assert_awaited_once_with(ttl_days=30)


def test_conversation_turn_projects_document_references() -> None:
    from dlightrag.web.conversations import _conversation_turn

    row = {
        "turn_id": "00000000-0000-0000-0000-000000000010",
        "turn_number": 2,
        "user_text": "see attached",
        "assistant_text": "answer",
        "answer_sources": {},
        "queried_workspaces": ["default"],
        "images": [],
        "attachments": [
            {
                "attachment_id": "00000000-0000-0000-0000-000000000011",
                "ordinal": 1,
                "filename": "report.pdf",
                "mime_type": "application/pdf",
                "byte_size": 8,
                "content_sha256": "abc",
                "parse_summary": "report summary",
            }
        ],
        "created_at": "2026-07-20T00:00:00Z",
    }

    turn = _conversation_turn("00000000-0000-0000-0000-000000000001", row)

    assert turn.user_documents[0].filename == "report.pdf"
    assert turn.user_documents[0].url.endswith("/documents/00000000-0000-0000-0000-000000000011")


def test_pending_conversation_attachment_shape() -> None:
    from dlightrag.storage.web_conversations import PendingConversationAttachment

    item = PendingConversationAttachment(
        attachment_id="00000000-0000-0000-0000-000000000011",
        ordinal=1,
        filename="report.pdf",
        mime_type="application/pdf",
        suffix=".pdf",
        attachment_bytes=b"%PDF",
        content_sha256="abc",
        parse_summary=None,
    )

    assert item.byte_size == 4


async def test_prepare_answer_turn_passes_documents_to_web_planner(
    service_under_test,
    conversation_store: AsyncMock,
) -> None:
    from dlightrag.core.request.planner import QueryPlan
    from dlightrag.web.conversations import PreparedWebConversation

    conversation_store.list_image_catalog.return_value = []
    prepared = PreparedWebConversation(
        principal_id="local",
        conversation_id="00000000-0000-0000-0000-000000000001",
        content_revision=0,
        text_history=(),
    )
    manager = AsyncMock()
    manager.answer_image_capability = SimpleNamespace(effective_max_images=3)
    manager.adescribe_query_images.return_value = {}
    manager.aplan_web_conversation_query.return_value = QueryPlan(
        original_query="what does this say?",
        standalone_query="what does this say?",
        selected_history_image_ids=(),
        selected_history_attachment_ids=(),
    )

    turn = await service_under_test.prepare_answer_turn(
        manager=manager,
        prepared=prepared,
        query="what does this say?",
        current_images=[],
        current_documents=[],
        workspaces=["default"],
    )

    manager.aplan_web_conversation_query.assert_awaited_once()
    manager.aplan_query.assert_not_awaited()
    assert turn.composer_context_chunks == ()
    assert turn.attachment_resolution_status == "ok"


async def test_prepare_answer_turn_merges_current_document_context(
    service_under_test,
    conversation_store: AsyncMock,
    test_config: DlightragConfig,
) -> None:
    from dlightrag.core.request.attachments import (
        ParsedAttachmentBundle,
        build_text_attachment_chunk,
    )
    from dlightrag.core.request.planner import QueryPlan
    from dlightrag.web.attachment_models import validate_web_documents
    from dlightrag.web.conversations import PreparedWebConversation

    conversation_store.list_image_catalog.return_value = []
    prepared = PreparedWebConversation(
        principal_id="local",
        conversation_id="00000000-0000-0000-0000-000000000001",
        content_revision=0,
        text_history=(),
    )
    (document,) = validate_web_documents([("notes.md", "text/markdown", b"# Termination\nclause")])

    manager = AsyncMock()
    manager.answer_image_capability = SimpleNamespace(effective_max_images=0)
    manager.adescribe_query_images.return_value = {}
    manager.aplan_web_conversation_query.return_value = QueryPlan(
        original_query="what does this say?",
        standalone_query="what does this say?",
        selected_history_attachment_ids=(),
    )
    query_embedder = SimpleNamespace(
        dimension=2,
        image_enabled=False,
        aembed_query=AsyncMock(side_effect=AssertionError("full pass must skip dense query")),
    )
    manager.aget_composer_processing_resources.return_value = SimpleNamespace(
        lightrag=object(),
        config=test_config,
        robust_document_embedder=query_embedder,
        direct_image_embedding_enabled=False,
        model_bundle=SimpleNamespace(vlm_identity={}, extract_identity={}),
        rerank_func=None,
    )

    async def _select(**kwargs: Any):
        assert kwargs["current_dense_rankings"] == []
        assert kwargs["history_dense_rankings"] == []
        return kwargs["current_rows"], {"composer_evidence_strategy": "full"}

    manager._aselect_web_composer_evidence.side_effect = _select

    chunk = build_text_attachment_chunk(
        attachment_id=document.attachment_id,
        filename=document.filename,
        chunk_id=f"att:{document.attachment_id}:1",
        chunk_index=1,
        content="termination clause",
    )
    row = chunk.to_context_row()

    # Current documents are parsed before planning; fake that seam so the core
    # digest selector sees the bundle and the same chunk reaches the answer.
    async def _fake_parse(**_kwargs: object):
        bundle = ParsedAttachmentBundle(
            chunks=[chunk],
            trace={
                "attachment_parse_cache_hit": False,
                "attachment_analysis_outcome": "success",
                "attachment_analysis_error": None,
                "attachment_mm_chunk_count": 0,
                "attachment_vector_cache_hits": 0,
                "attachment_vector_cache_misses": 1,
                "attachment_embedding_fused": 0,
                "attachment_embedding_text": 1,
                "attachment_embedding_fallback": 0,
                "attachment_embedding_failed": 0,
            },
        )
        return [chunk], [], [(document.attachment_id, bundle)]

    service_under_test._parse_attachment_documents = _fake_parse  # type: ignore[method-assign]

    turn = await service_under_test.prepare_answer_turn(
        manager=manager,
        prepared=prepared,
        query="what does this say?",
        current_images=[],
        current_documents=[document],
        workspaces=["default"],
    )

    manager.aplan_web_conversation_query.assert_awaited_once()
    catalog = manager.aplan_web_conversation_query.await_args.kwargs["current_attachment_catalog"]
    assert catalog[0]["attachment_id"] == document.attachment_id
    assert catalog[0]["filename"] == "notes.md"
    # The planner sees the parsed content preview (document-aware planning), not "".
    assert catalog[0]["parse_summary"] == "termination clause"
    assert len(turn.composer_context_chunks) == 1
    selected_row = turn.composer_context_chunks[0]
    assert selected_row["chunk_id"] == row["chunk_id"]
    assert selected_row["reference_id"] == f"composer_{document.attachment_id.replace('-', '')}"
    assert selected_row["full_doc_id"] == document.attachment_id
    assert selected_row["metadata"]["attachment_scope"] == "current"
    assert turn.composer_evidence_trace["composer_evidence_strategy"] == "full"
    assert turn.composer_evidence_trace["composer_dense_status"] == "full_pass_not_needed"
    assert turn.composer_evidence_trace["composer_dense_current_chunks"] == 0
    assert turn.composer_evidence_trace["composer_dense_history_chunks"] == 0
    assert turn.composer_evidence_trace["composer_dense_chunks"] == 0
    query_embedder.aembed_query.assert_not_awaited()
    processing_trace = turn.composer_evidence_trace["attachment_processing"]
    assert processing_trace[0]["attachment_id"] == document.attachment_id
    assert processing_trace[0]["attachment_analysis_outcome"] == "success"
    assert processing_trace[0]["attachment_embedding_text"] == 1
    assert turn.attachment_resolution_status == "ok"


async def test_parse_attachment_documents_preserves_resource_identity_order_and_trace(
    service_under_test,
) -> None:
    from dlightrag.core.request.attachments import (
        AttachmentContextChunk,
        ParsedAttachmentBundle,
    )
    from dlightrag.web.conversations import PreparedWebConversation

    prepared = PreparedWebConversation(
        principal_id="local",
        conversation_id="00000000-0000-0000-0000-000000000001",
        content_revision=0,
        text_history=(),
    )
    documents = [
        SimpleNamespace(
            attachment_id="att-a",
            filename="a.pdf",
            document_bytes=b"a",
            content_sha256="sha-a",
        ),
        SimpleNamespace(
            attachment_id="att-b",
            filename="b.pdf",
            document_bytes=b"b",
            content_sha256="sha-b",
        ),
    ]

    class _AttachmentService:
        async def achunks_for_attachment(self, **kwargs: Any):
            attachment_id = kwargs["attachment_id"]
            chunk = AttachmentContextChunk(
                chunk_id=f"chunk-{attachment_id}",
                attachment_id=attachment_id,
                filename=kwargs["filename"],
                chunk_index=1,
                content=attachment_id,
            )
            return ParsedAttachmentBundle(chunks=[chunk]), {
                "attachment_parse_cache_hit": attachment_id == "att-a",
                "attachment_analysis_outcome": "success",
                "attachment_analysis_error": None,
                "attachment_mm_chunk_count": 1,
                "attachment_vector_cache_hits": 1,
                "attachment_vector_cache_misses": 0,
                "attachment_embedding_fused": 1,
                "attachment_embedding_text": 0,
                "attachment_embedding_fallback": 0,
                "attachment_embedding_failed": 0,
            }

    attachment_service = _AttachmentService()

    chunks, errors, bundles = await service_under_test._parse_attachment_documents(
        attachment_service=attachment_service,
        prepared=prepared,
        documents=documents,
    )

    assert errors == []
    assert [chunk.attachment_id for chunk in chunks] == ["att-a", "att-b"]
    assert [attachment_id for attachment_id, _bundle in bundles] == ["att-a", "att-b"]
    assert bundles[0][1].trace["attachment_parse_cache_hit"] is True
    assert bundles[1][1].trace["attachment_embedding_fused"] == 1


async def test_parse_attachment_documents_continues_after_invalid_parser_hint(
    service_under_test,
) -> None:
    from dlightrag.core.request.attachments import (
        AttachmentContextChunk,
        ParsedAttachmentBundle,
    )
    from dlightrag.web.conversations import PreparedWebConversation

    prepared = PreparedWebConversation(
        principal_id="local",
        conversation_id="00000000-0000-0000-0000-000000000001",
        content_revision=0,
        text_history=(),
    )
    documents = [
        SimpleNamespace(
            attachment_id="att-invalid",
            filename="report.[unknown].txt",
            document_bytes=b"bad hint",
            content_sha256="sha-invalid",
        ),
        SimpleNamespace(
            attachment_id="att-valid",
            filename="notes.txt",
            document_bytes=b"valid",
            content_sha256="sha-valid",
        ),
    ]

    class _AttachmentService:
        async def achunks_for_attachment(self, **kwargs: Any):
            attachment_id = kwargs["attachment_id"]
            if attachment_id == "att-invalid":
                return ParsedAttachmentBundle(chunks=[]), {
                    "attachment_parse_error": "ValueError",
                    "attachment_analysis_outcome": "degraded",
                    "attachment_parse_cache_hit": False,
                }
            chunk = AttachmentContextChunk(
                chunk_id="chunk-valid",
                attachment_id=attachment_id,
                filename=kwargs["filename"],
                chunk_index=1,
                content="valid",
            )
            return ParsedAttachmentBundle(chunks=[chunk]), {
                "attachment_parse_cache_hit": False,
                "attachment_analysis_outcome": "intentionally_disabled",
            }

    attachment_service = _AttachmentService()

    chunks, errors, bundles = await service_under_test._parse_attachment_documents(
        attachment_service=attachment_service,
        prepared=prepared,
        documents=documents,
    )

    assert [chunk.attachment_id for chunk in chunks] == ["att-valid"]
    assert errors == [{"attachment_id": "att-invalid", "error": "ValueError"}]
    assert [attachment_id for attachment_id, _bundle in bundles] == [
        "att-invalid",
        "att-valid",
    ]
    assert bundles[0][1].trace["attachment_analysis_outcome"] == "degraded"


async def test_prepare_answer_turn_preserves_selected_history_document_order(
    service_under_test,
    conversation_store: AsyncMock,
) -> None:
    from dlightrag.core.request.attachments import build_text_attachment_chunk
    from dlightrag.core.request.planner import QueryPlan
    from dlightrag.storage.web_conversations import StoredConversationAttachment
    from dlightrag.web.conversations import PreparedWebConversation

    conversation_store.list_image_catalog.return_value = []
    prepared = PreparedWebConversation(
        principal_id="local",
        conversation_id="00000000-0000-0000-0000-000000000001",
        content_revision=0,
        text_history=(),
    )
    selected_ids = ("att-a", "att-b", "att-missing")
    document_a = StoredConversationAttachment(
        attachment_id="att-a",
        filename="a.pdf",
        mime_type="application/pdf",
        suffix=".pdf",
        attachment_bytes=b"a",
        content_sha256="sha-a",
    )
    document_b = StoredConversationAttachment(
        attachment_id="att-b",
        filename="b.pdf",
        mime_type="application/pdf",
        suffix=".pdf",
        attachment_bytes=b"b",
        content_sha256="sha-b",
    )
    # PostgreSQL ANY(...) does not preserve the planner's relevance order.
    conversation_store.fetch_documents_by_ids.return_value = [document_b, document_a]

    manager = AsyncMock()
    manager.answer_image_capability = SimpleNamespace(effective_max_images=0)
    manager.adescribe_query_images.return_value = {}
    manager.aplan_web_conversation_query.return_value = QueryPlan(
        original_query="compare those reports",
        standalone_query="compare reports a and b",
        selected_history_attachment_ids=selected_ids,
    )

    async def _select(**kwargs: Any):
        assert kwargs["history_dense_rankings"] == kwargs["history_rows"]
        return kwargs["history_rows"], {"composer_evidence_strategy": "full"}

    manager._aselect_web_composer_evidence.side_effect = _select

    class _AttachmentService:
        async def adense_rankings(
            self, _query: str, current_rows: list[Any], history_rows: list[Any]
        ):
            return current_rows, history_rows, {}

    attachment_service = _AttachmentService()
    service_under_test._get_query_attachment_service = (  # type: ignore[method-assign]
        lambda _resources, _prepared: attachment_service
    )
    parsed_order: list[str] = []

    async def _fake_parse(*, documents, **_kwargs: object):
        parsed_order.extend(document.attachment_id for document in documents)
        chunks = [
            build_text_attachment_chunk(
                attachment_id=document.attachment_id,
                filename=document.filename,
                chunk_id=f"chunk-{document.attachment_id}",
                chunk_index=index,
                content=document.filename,
            )
            for index, document in enumerate(documents, start=1)
        ]
        return chunks, [], []

    service_under_test._parse_attachment_documents = _fake_parse  # type: ignore[method-assign]

    turn = await service_under_test.prepare_answer_turn(
        manager=manager,
        prepared=prepared,
        query="compare those reports",
        current_images=[],
        current_documents=[],
        workspaces=["default"],
    )

    assert parsed_order == ["att-a", "att-b"]
    assert [row["full_doc_id"] for row in turn.composer_context_chunks] == [
        "att-a",
        "att-b",
    ]
    assert [row["reference_id"] for row in turn.composer_context_chunks] == [
        "composer_atta",
        "composer_attb",
    ]
    assert all(
        row["metadata"]["attachment_scope"] == "history" for row in turn.composer_context_chunks
    )
    assert turn.history_attachments_selected == 2
    assert turn.attachment_resolution_status == "degraded"


async def test_prepare_answer_turn_degrades_on_current_document_parse_error(
    service_under_test,
    conversation_store: AsyncMock,
) -> None:
    from dlightrag.core.request.planner import QueryPlan
    from dlightrag.web.attachment_models import validate_web_documents
    from dlightrag.web.conversations import PreparedWebConversation

    conversation_store.list_image_catalog.return_value = []
    prepared = PreparedWebConversation(
        principal_id="local",
        conversation_id="00000000-0000-0000-0000-000000000001",
        content_revision=0,
        text_history=(),
    )
    (document,) = validate_web_documents(
        [("broken.docx", "application/octet-stream", b"not-a-docx")]
    )

    manager = AsyncMock()
    manager.answer_image_capability = SimpleNamespace(effective_max_images=0)
    manager.adescribe_query_images.return_value = {}
    manager.aplan_web_conversation_query.return_value = QueryPlan(
        original_query="what does this say?",
        standalone_query="what does this say?",
        selected_history_attachment_ids=(),
    )

    async def _fake_parse(**_kwargs: object):
        # Parser failed for the only document: no chunks, one scoped error.
        return [], [{"attachment_id": document.attachment_id, "error": "ValueError"}], []

    service_under_test._parse_attachment_documents = _fake_parse  # type: ignore[method-assign]

    turn = await service_under_test.prepare_answer_turn(
        manager=manager,
        prepared=prepared,
        query="what does this say?",
        current_images=[],
        current_documents=[document],
        workspaces=["default"],
    )

    assert turn.composer_context_chunks == ()
    assert turn.attachment_resolution_status == "degraded"
