# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the durable Web conversation service and its lifecycle routes."""

import datetime
import io
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import asyncpg
import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from dlightrag.api.auth import UserContext
from dlightrag.api.server import create_app
from dlightrag.config import DlightragConfig
from dlightrag.storage.web_conversations import (
    ConversationSnapshot,
    StoredConversationAttachment,
)
from dlightrag.utils.images import thumbnail_bytes

_CID = "00000000-0000-0000-0000-000000000001"
_AID = "00000000-0000-0000-0000-000000000020"


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
    service.list.return_value = [summary]
    service.history.return_value = {"conversation": summary, "turns": []}
    service.rename.return_value = {**summary, "title": "Renamed chat"}
    service.delete.return_value = True
    service.delete_all.return_value = 2
    service.attachment.return_value = SimpleNamespace(
        attachment_id=_AID,
        filename="chart.png",
        mime_type="image/png",
        attachment_bytes=b"png-bytes",
    )
    service.thumbnail.return_value = (b"derived-thumbnail", "image/jpeg")
    return service


@pytest.fixture
async def conversation_client(conversation_service: AsyncMock):
    application = create_app(include_web_app=True)
    application.state.web_conversation_service = conversation_service
    application.state.manager = AsyncMock()
    application.state.manager.alist_workspace_records.return_value = [{"workspace": "default"}]
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
    application = create_app(include_web_app=True)
    application.state.web_conversation_service = conversation_service
    application.state.manager = AsyncMock(config=test_config, answer_image_capability=None)
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

    response = await conversation_client.get(f"/web/conversations/{_CID}/history")

    assert response.status_code == 404


async def test_rename_validates_trimmed_title(conversation_client: AsyncClient) -> None:
    response = await conversation_client.patch(
        f"/web/conversations/{_CID}",
        json={"title": "   "},
    )

    assert response.status_code == 422


async def test_rename_normalizes_whitespace(
    conversation_client: AsyncClient,
    conversation_service: AsyncMock,
) -> None:
    response = await conversation_client.patch(
        f"/web/conversations/{_CID}",
        json={"title": "  Renamed\n chat  "},
    )

    assert response.status_code == 200
    assert conversation_service.rename.await_args.args[-1] == "Renamed chat"


async def test_delete_returns_204(
    conversation_client: AsyncClient,
    conversation_service: AsyncMock,
) -> None:
    response = await conversation_client.delete(f"/web/conversations/{_CID}")

    assert response.status_code == 204
    assert response.content == b""
    conversation_service.delete.assert_awaited_once()


async def test_delete_all_returns_204_when_no_conversations_exist(
    conversation_client: AsyncClient,
    conversation_service: AsyncMock,
) -> None:
    conversation_service.delete_all.return_value = 0

    response = await conversation_client.delete("/web/conversations")

    assert response.status_code == 204
    assert response.content == b""
    conversation_service.delete_all.assert_awaited_once()


async def test_delete_has_no_messages_subroute(conversation_client: AsyncClient) -> None:
    response = await conversation_client.delete(f"/web/conversations/{_CID}/messages")

    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Unified attachment download + thumbnail routes
# ---------------------------------------------------------------------------


async def test_scoped_attachment_response_is_private(
    conversation_client: AsyncClient,
) -> None:
    response = await conversation_client.get(f"/web/conversations/{_CID}/attachments/{_AID}")

    assert response.status_code == 200
    assert response.content == b"png-bytes"
    assert response.headers["content-type"] == "image/png"
    assert response.headers["cache-control"] == "private, max-age=3600"
    assert response.headers["x-content-type-options"] == "nosniff"


async def test_scoped_document_attachment_sets_content_disposition(
    conversation_client: AsyncClient,
    conversation_service: AsyncMock,
) -> None:
    conversation_service.attachment.return_value = SimpleNamespace(
        attachment_id=_AID,
        filename="report.pdf",
        mime_type="application/pdf",
        attachment_bytes=b"%PDF-1.4",
    )

    response = await conversation_client.get(f"/web/conversations/{_CID}/attachments/{_AID}")

    assert response.status_code == 200
    assert response.headers["content-disposition"] == 'attachment; filename="report.pdf"'


async def test_scoped_attachment_of_other_principal_is_404(
    conversation_client: AsyncClient,
    conversation_service: AsyncMock,
) -> None:
    conversation_service.attachment.return_value = None

    response = await conversation_client.get(f"/web/conversations/{_CID}/attachments/{_AID}")

    assert response.status_code == 404


async def test_scoped_thumbnail_response_is_private_and_immutable(
    conversation_client: AsyncClient,
    conversation_service: AsyncMock,
) -> None:
    response = await conversation_client.get(
        f"/web/conversations/{_CID}/attachments/{_AID}/thumbnail"
    )

    assert response.status_code == 200
    assert response.content == b"derived-thumbnail"
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["cache-control"] == "private, max-age=86400, immutable"
    assert response.headers["x-content-type-options"] == "nosniff"
    conversation_service.thumbnail.assert_awaited_once()


async def test_scoped_thumbnail_failure_does_not_fall_back_to_original(
    conversation_client: AsyncClient,
    conversation_service: AsyncMock,
) -> None:
    conversation_service.thumbnail.return_value = None

    response = await conversation_client.get(
        f"/web/conversations/{_CID}/attachments/{_AID}/thumbnail"
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
    application = create_app(include_web_app=True)
    application.state.web_conversation_service = conversation_service
    transport = ASGITransport(app=application)
    async with AsyncClient(
        transport=transport,
        base_url="https://app.example.com",
        follow_redirects=False,
    ) as client:
        response = await client.get(f"/web/conversations/{_CID}/attachments/{_AID}/thumbnail")

    assert response.status_code == 303
    assert response.headers["location"].startswith("/web/login")
    conversation_service.thumbnail.assert_not_awaited()


# ---------------------------------------------------------------------------
# Cross-origin protection
# ---------------------------------------------------------------------------


_COOKIE_MUTATIONS = (
    pytest.param("POST", "/web/conversations", None, "create", 201, id="create"),
    pytest.param(
        "PATCH",
        f"/web/conversations/{_CID}",
        {"title": "Renamed chat"},
        "rename",
        200,
        id="rename",
    ),
    pytest.param(
        "DELETE",
        f"/web/conversations/{_CID}",
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
    "conversation_id": _CID,
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
        headers={"Content-Type": content_type, "Origin": "https://app.example.com"},
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


# ---------------------------------------------------------------------------
# Storage availability signalling
# ---------------------------------------------------------------------------


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
    application = create_app(include_web_app=True)
    application.state.web_conversation_service = WebConversationService(
        store=store,
        max_turns=100,
        ttl_days=30,
        max_attachments=6,
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
    application = create_app(include_web_app=True)
    application.state.web_conversation_service = WebConversationService(
        store=store,
        max_turns=100,
        ttl_days=30,
        max_attachments=6,
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
    application = create_app(include_web_app=True)
    application.state.web_conversation_service = WebConversationService(
        store=store,
        max_turns=100,
        ttl_days=30,
        max_attachments=6,
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


# ---------------------------------------------------------------------------
# Service-level fixtures
# ---------------------------------------------------------------------------


def _conversation_row() -> dict[str, object]:
    now = datetime.datetime(2026, 7, 12, tzinfo=datetime.UTC)
    return {
        "conversation_id": _CID,
        "title": "Quarterly review",
        "content_revision": 7,
        "created_at": now,
        "updated_at": now,
    }


def _conversation_snapshot() -> ConversationSnapshot:
    now = datetime.datetime(2026, 7, 12, tzinfo=datetime.UTC)
    return ConversationSnapshot(
        principal_id="stored-principal",
        conversation_id=_CID,
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
                            "workspace": "default",
                            "document_id": "report",
                            "chunks": [],
                        }
                    ]
                },
                "queried_workspaces": ["default"],
                "created_at": now,
                "attachments": [
                    {
                        "attachment_id": _AID,
                        "ordinal": 1,
                        "filename": "chart.png",
                        "mime_type": "image/png",
                        "byte_size": 9,
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
    store.delete_all_conversations.return_value = 2
    store.prune_expired.return_value = 0
    return store


@pytest.fixture
def service_under_test(conversation_store: AsyncMock):
    from dlightrag.web.conversations import WebConversationService

    return WebConversationService(
        store=conversation_store, max_turns=100, ttl_days=30, max_attachments=6
    )


@pytest.fixture
def jwt_user() -> UserContext:
    return UserContext(
        user_id="alice",
        auth_mode="jwt",
        claims={"iss": "https://issuer.example"},
    )


# ---------------------------------------------------------------------------
# Service behaviour
# ---------------------------------------------------------------------------


async def test_service_derives_principal_for_each_lifecycle_operation(
    service_under_test,
    conversation_store: AsyncMock,
    jwt_user: UserContext,
) -> None:
    from dlightrag.web.principal import principal_id_from_user

    expected_principal = principal_id_from_user(jwt_user)

    await service_under_test.create(jwt_user)
    await service_under_test.list(jwt_user)
    await service_under_test.rename(jwt_user, _CID, "Quarterly review")
    await service_under_test.delete(jwt_user, _CID)
    await service_under_test.delete_all(jwt_user)

    assert conversation_store.create_conversation.await_args.args == (expected_principal,)
    assert conversation_store.list_conversations.await_args.args == (expected_principal,)
    assert conversation_store.rename_conversation.await_args.args[:2] == (
        expected_principal,
        _CID,
    )
    assert conversation_store.delete_conversation.await_args.args[:2] == (
        expected_principal,
        _CID,
    )
    assert conversation_store.delete_all_conversations.await_args.args == (expected_principal,)


async def test_history_projects_safe_attachments_sources_and_rendered_answer(
    service_under_test,
    conversation_store: AsyncMock,
    jwt_user: UserContext,
) -> None:
    history = await service_under_test.history(jwt_user, _CID)

    assert history is not None
    turn = history.turns[0]
    attachment = turn.user_attachments[0]
    expected_url = f"/web/conversations/{_CID}/attachments/{_AID}"
    assert attachment.url == expected_url
    assert attachment.kind == "image"
    assert attachment.thumbnail_url == expected_url + "/thumbnail"
    assert attachment.label == "Turn 1, attachment 1"
    assert "/web/files/raw/report?workspace=default" in turn.answer_html
    assert "citation-badge" in turn.answer_html
    assert "answer_sources" not in turn.model_dump()
    assert "queried_workspaces" not in turn.model_dump()
    assert "attachment_bytes" not in turn.model_dump_json()
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
    conversation_store.get_attachment.return_value = StoredConversationAttachment(
        attachment_id=_AID,
        filename="chart.png",
        mime_type="image/png",
        suffix=".png",
        attachment_bytes=original,
    )

    thumbnail = await service_under_test.thumbnail(jwt_user, _CID, _AID)

    assert thumbnail is not None
    payload, mime_type = thumbnail
    assert mime_type in {"image/jpeg", "image/png"}
    assert len(payload) <= 128 * 1024
    assert len(payload) < len(original)
    with Image.open(io.BytesIO(payload)) as derived:
        assert max(derived.size) <= 320
        assert derived.format in {"JPEG", "PNG"}
    from dlightrag.web.principal import principal_id_from_user

    conversation_store.get_attachment.assert_awaited_once_with(
        principal_id_from_user(jwt_user),
        _CID,
        _AID,
        ttl_days=30,
    )


async def test_history_thumbnail_generation_failure_returns_none(
    service_under_test,
    conversation_store: AsyncMock,
    jwt_user: UserContext,
) -> None:
    conversation_store.get_attachment.return_value = StoredConversationAttachment(
        attachment_id=_AID,
        filename="chart.png",
        mime_type="image/png",
        suffix=".png",
        attachment_bytes=b"durable-original-but-not-a-decodable-image",
    )

    thumbnail = await service_under_test.thumbnail(jwt_user, _CID, _AID)

    assert thumbnail is None


async def test_history_thumbnail_of_document_attachment_returns_none(
    service_under_test,
    conversation_store: AsyncMock,
    jwt_user: UserContext,
) -> None:
    conversation_store.get_attachment.return_value = StoredConversationAttachment(
        attachment_id=_AID,
        filename="report.pdf",
        mime_type="application/pdf",
        suffix=".pdf",
        attachment_bytes=b"%PDF-1.4",
    )

    thumbnail = await service_under_test.thumbnail(jwt_user, _CID, _AID)

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


async def test_prepare_answer_uses_one_snapshot_text_history_and_manifest(
    service_under_test,
    conversation_store: AsyncMock,
    jwt_user: UserContext,
) -> None:
    prepared = await service_under_test.prepare_answer(jwt_user, _CID)

    assert prepared is not None
    assert prepared.content_revision == 7
    assert prepared.text_history == (
        {"role": "user", "content": "What changed?"},
        {"role": "assistant", "content": "Revenue increased [1]."},
    )
    assert prepared.attachment_manifest[0]["attachment_id"] == _AID
    assert prepared.attachment_manifest[0]["filename"] == "chart.png"
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
        _CID,
        "00000000-0000-4000-8000-000000000099",
    )

    assert prepared is not None
    conversation_store.find_committed_turn.assert_awaited_once()


async def test_build_answer_resources_orders_current_then_history_attachments(
    service_under_test,
    jwt_user: UserContext,
) -> None:
    from dlightrag.web.attachment_models import validate_web_attachments

    prepared = await service_under_test.prepare_answer(jwt_user, _CID)
    assert prepared is not None
    image_buffer = io.BytesIO()
    Image.new("RGB", (4, 4), "white").save(image_buffer, format="PNG")
    (current,) = validate_web_attachments(
        [("now.png", "image/png", image_buffer.getvalue())],
        max_attachments=6,
        max_attachment_bytes=15 * 1024 * 1024,
        max_total_attachment_bytes=128 * 1024 * 1024,
    )

    resources = service_under_test.build_answer_resources(prepared, (current,))

    assert len(resources) == 2
    # Current-turn attachment carries inline bytes; prior attachment is a lazy loader.
    assert resources[0].content is not None
    assert resources[0].filename == "now.png"
    assert resources[1].content is None
    assert resources[1].loader is not None


async def test_build_answer_resources_caps_history_by_configured_limit(
    conversation_store: AsyncMock,
    jwt_user: UserContext,
) -> None:
    from dlightrag.web.attachment_models import validate_web_attachments
    from dlightrag.web.conversations import WebConversationService

    service = WebConversationService(
        store=conversation_store, max_turns=100, ttl_days=30, max_attachments=2
    )
    prepared = await service.prepare_answer(jwt_user, _CID)
    assert prepared is not None
    prepared = replace(
        prepared,
        attachment_manifest=tuple(
            {"attachment_id": f"prior-{i}", "filename": f"p{i}.pdf", "mime_type": "application/pdf"}
            for i in range(4)
        ),
    )
    image_buffer = io.BytesIO()
    Image.new("RGB", (4, 4), "white").save(image_buffer, format="PNG")
    (current,) = validate_web_attachments(
        [("now.png", "image/png", image_buffer.getvalue())],
        max_attachments=2,
        max_attachment_bytes=15 * 1024 * 1024,
        max_total_attachment_bytes=128 * 1024 * 1024,
    )

    resources = service.build_answer_resources(prepared, (current,))

    # Lowered limit of 2: one current attachment leaves room for exactly one
    # prior lazy resource, keeping the most recent manifest entry.
    assert len(resources) == 2
    assert resources[0].filename == "now.png"
    assert resources[1].content is None
    assert resources[1].loader is not None


async def test_commit_answer_maps_validated_attachments_and_revision(
    service_under_test,
    conversation_store: AsyncMock,
) -> None:
    from dlightrag.storage.web_conversations import CommitTurnResult
    from dlightrag.web.attachment_models import validate_web_attachments
    from dlightrag.web.conversations import PreparedWebConversation

    conversation_store.commit_turn.return_value = CommitTurnResult(
        saved=False, reason="conversation_changed", summary=None, turn_id=None
    )
    prepared = PreparedWebConversation(
        principal_id="principal-hash",
        conversation_id=_CID,
        content_revision=7,
        text_history=(),
    )
    image_buffer = io.BytesIO()
    Image.new("RGB", (4, 4), "white").save(image_buffer, format="PNG")
    raw = image_buffer.getvalue()
    (attachment,) = validate_web_attachments(
        [("chart.png", "image/png", raw)],
        max_attachments=6,
        max_attachment_bytes=15 * 1024 * 1024,
        max_total_attachment_bytes=128 * 1024 * 1024,
    )

    result = await service_under_test.commit_answer(
        prepared,
        submission_id="00000000-0000-4000-8000-000000000098",
        user_text="Question",
        assistant_text="Answer",
        answer_sources={"sources": [], "answer_images": []},
        queried_workspaces=["default"],
        attachments=(attachment,),
    )

    assert result.reason == "conversation_changed"
    call = conversation_store.commit_turn.await_args.kwargs
    assert call["expected_revision"] == 7
    assert call["principal_id"] == "principal-hash"
    pending = call["attachments"]
    assert pending[0].attachment_bytes == raw
    assert pending[0].filename == "chart.png"
    assert pending[0].suffix == ".png"


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
        current_attachment_ids=("stored-attachment",),
        assistant_text="Stored answer",
        answer_sources={"sources": []},
        replayed=True,
    )
    conversation_store.commit_turn.side_effect = ConnectionError("ack lost")
    conversation_store.find_committed_turn.return_value = committed
    prepared = PreparedWebConversation(
        principal_id="principal",
        conversation_id=_CID,
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
        attachments=(),
    )

    assert result == committed
    conversation_store.find_committed_turn.assert_awaited_once_with(
        "principal",
        _CID,
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
        conversation_id=_CID,
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
            attachments=(),
        ),
        timeout=0.1,
    )

    assert result.saved is False
    assert result.reason == "commit_outcome_unknown"
    assert result.current_attachment_ids == ()


async def test_initialize_applies_schema_then_global_prune(
    service_under_test,
    conversation_store: AsyncMock,
) -> None:
    await service_under_test.initialize()

    conversation_store.initialize.assert_awaited_once_with(validate_only=False)
    conversation_store.prune_expired.assert_awaited_once_with(ttl_days=30, batch_size=500)
    await service_under_test.aclose()


async def test_reader_initialize_validates_schema_without_migrating(
    conversation_store: AsyncMock,
) -> None:
    from dlightrag.web.conversations import WebConversationService

    service = WebConversationService(
        store=conversation_store,
        max_turns=30,
        ttl_days=30,
        max_attachments=4,
        validate_schema_only=True,
    )

    await service.initialize()

    conversation_store.initialize.assert_awaited_once_with(validate_only=True)
    await service.aclose()


async def test_initialize_runs_periodic_prune_until_closed(
    service_under_test,
    conversation_store: AsyncMock,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    periodic_prune = asyncio.Event()
    calls = 0

    async def prune_expired(*, ttl_days: int, batch_size: int) -> int:
        nonlocal calls
        calls += 1
        if calls >= 2:
            periodic_prune.set()
        return 0

    conversation_store.prune_expired.side_effect = prune_expired
    monkeypatch.setattr("dlightrag.web.conversations._PRUNE_INTERVAL_SECONDS", 0.001)

    await service_under_test.initialize()
    await asyncio.wait_for(periodic_prune.wait(), timeout=0.1)
    await service_under_test.aclose()

    assert calls >= 2
    assert all(
        call.kwargs == {"ttl_days": 30, "batch_size": 500}
        for call in conversation_store.prune_expired.await_args_list
    )


# ---------------------------------------------------------------------------
# Projection helpers
# ---------------------------------------------------------------------------


def test_conversation_turn_projects_document_and_image_attachments() -> None:
    from dlightrag.web.conversations import _conversation_turn

    row = {
        "turn_id": "00000000-0000-0000-0000-000000000010",
        "turn_number": 2,
        "user_text": "see attached",
        "assistant_text": "answer",
        "answer_sources": {},
        "queried_workspaces": ["default"],
        "attachments": [
            {
                "attachment_id": "00000000-0000-0000-0000-000000000011",
                "ordinal": 1,
                "filename": "report.pdf",
                "mime_type": "application/pdf",
                "byte_size": 8,
            },
            {
                "attachment_id": "00000000-0000-0000-0000-000000000012",
                "ordinal": 2,
                "filename": "chart.png",
                "mime_type": "image/png",
                "byte_size": 9,
            },
        ],
        "created_at": "2026-07-20T00:00:00Z",
    }

    turn = _conversation_turn(_CID, row)

    document, image = turn.user_attachments
    assert document.kind == "document"
    assert document.filename == "report.pdf"
    assert document.url.endswith("/attachments/00000000-0000-0000-0000-000000000011")
    assert document.thumbnail_url is None
    assert image.kind == "image"
    assert image.thumbnail_url is not None
    assert image.thumbnail_url.endswith(
        "/attachments/00000000-0000-0000-0000-000000000012/thumbnail"
    )


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
    )

    assert item.byte_size == 4
