# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for durable Web conversation lifecycle adapters and routes."""

import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from httpx import ASGITransport, AsyncClient

from dlightrag.api.auth import UserContext
from dlightrag.api.server import create_app
from dlightrag.storage.web_conversations import ConversationSnapshot


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
    assert image.thumbnail_url == expected_url
    assert image.label == "Turn 1, image 1"
    assert turn.answer_sources["sources"][0]["download_url"].startswith("/web/")
    assert "citation-badge" in turn.answer_html
    assert "image_bytes" not in turn.model_dump_json()
    assert "principal_id" not in turn.model_dump_json()


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


async def test_initialize_applies_schema_then_global_prune(
    service_under_test,
    conversation_store: AsyncMock,
) -> None:
    await service_under_test.initialize()

    conversation_store.initialize.assert_awaited_once_with()
    conversation_store.prune_expired.assert_awaited_once_with(ttl_days=30)
