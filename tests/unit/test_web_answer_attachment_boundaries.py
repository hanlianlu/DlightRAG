# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Boundary and security tests for Web-only Composer documents.

These cover three seams that keep Composer documents a Web-only
feature:

* the public REST ``/retrieve`` contract never accepts or emits attachment
  inputs/chunks (review finding I4 boundary);
* the Web answer layer projects a conversation-scoped download URL onto
  attachment sources (Task 8 Step 3);
* the conversation document download route is principal-scoped, returning 404
  to a caller who does not own the attachment (review finding I4 isolation).
"""

from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from dlightrag.api.auth import UserContext, get_current_user
from dlightrag.api.server import create_app
from dlightrag.citations.schemas import ChunkSnippet, SourceReferencePayload
from dlightrag.config import DlightragConfig
from dlightrag.storage.web_conversations import StoredConversationAttachment
from dlightrag.web.answer_events import _apply_attachment_source_links
from dlightrag.web.conversations import WebConversationService
from dlightrag.web.principal import principal_id_from_user

_ANON = UserContext(user_id="anonymous", auth_mode="none")


# ---------------------------------------------------------------------------
# Public /retrieve contract boundary
# ---------------------------------------------------------------------------


@pytest.fixture
def _retrieve_app(test_config: DlightragConfig) -> Iterator[FastAPI]:
    app = create_app(include_web=False)
    app.dependency_overrides[get_current_user] = lambda: _ANON
    yield app
    app.dependency_overrides.clear()


@pytest.fixture
async def api_client(_retrieve_app: FastAPI) -> Any:
    transport = ASGITransport(app=_retrieve_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


async def test_public_retrieve_contract_does_not_accept_documents(
    api_client: AsyncClient,
) -> None:
    """The public retrieve contract must reject or ignore attachment inputs.

    ``RetrieveRequest`` forbids extra fields, so a ``documents`` payload is
    rejected with 422 and the Web attachment pipeline is never reachable from
    the public REST surface.
    """
    response = await api_client.post(
        "/retrieve",
        json={"query": "hello", "documents": [{"filename": "report.pdf"}]},
    )

    assert response.status_code in {200, 422}
    if response.status_code == 200:
        payload = response.json()
        assert "documents" not in payload
        assert all(
            (chunk.get("metadata") or {}).get("source_type") != "web_attachment"
            for chunk in payload.get("contexts", {}).get("chunks", [])
        )


# ---------------------------------------------------------------------------
# Attachment source download-link projection (Task 8 Step 3)
# ---------------------------------------------------------------------------


def test_apply_attachment_source_links_projects_scoped_download_url() -> None:
    attachment_id = "98ec1e3a-1187-454b-8929-743bd5bc7d4b"
    attachment = SourceReferencePayload(
        id="att-1",
        title="report.pdf",
        source_uri=f"web-attachment://{attachment_id}",
        download_url=None,
        chunks=[
            ChunkSnippet(
                chunk_id="c1",
                content="figure",
                image_url="/web/images/__web_attachment__/c1?size=full",
                thumbnail_url="/web/images/__web_attachment__/c1?size=thumb",
            )
        ],
    )
    workspace = SourceReferencePayload(
        id="1",
        title="ledger.pdf",
        source_uri="s3://bucket/ledger.pdf",
        download_url="/web/files/raw/doc-ledger?workspace=finance",
        chunks=[
            ChunkSnippet(
                chunk_id="w1",
                content="evidence",
                image_url="/web/images/finance/w1?size=full",
                thumbnail_url="/web/images/finance/w1?size=thumb",
            )
        ],
    )

    projected = _apply_attachment_source_links([attachment, workspace], conversation_id="conv-9")

    assert projected[0].id == "att-1"
    download_url = projected[0].download_url
    assert download_url is not None
    assert download_url == f"/web/conversations/conv-9/documents/{attachment_id}"
    assert "att-1" not in download_url
    assert projected[0].chunks is not None
    assert projected[0].chunks[0].image_url is None
    assert projected[0].chunks[0].thumbnail_url is None
    # Workspace source is untouched.
    assert projected[1] == workspace
    assert projected[1].download_url == "/web/files/raw/doc-ledger?workspace=finance"
    assert projected[1].chunks is not None
    assert projected[1].chunks[0].image_url == "/web/images/finance/w1?size=full"


# ---------------------------------------------------------------------------
# Cross-principal download isolation (review finding I4)
# ---------------------------------------------------------------------------


class _ScopedAttachmentStore:
    """Fake store that only returns an attachment to its owning principal."""

    def __init__(self, *, owner_principal: str, attachment: StoredConversationAttachment) -> None:
        self._owner_principal = owner_principal
        self._attachment = attachment

    async def get_attachment(
        self,
        principal_id: str,
        conversation_id: str,
        attachment_id: str,
        *,
        ttl_days: int,
    ) -> StoredConversationAttachment | None:
        if principal_id != self._owner_principal:
            return None
        return self._attachment


async def test_document_download_is_cross_principal_isolated() -> None:
    owner = UserContext(user_id="owner", auth_mode="jwt", claims={"iss": "https://idp"})
    attacker = UserContext(user_id="attacker", auth_mode="jwt", claims={"iss": "https://idp"})
    owner_principal = principal_id_from_user(owner)
    attacker_principal = principal_id_from_user(attacker)
    assert owner_principal != attacker_principal

    attachment = StoredConversationAttachment(
        attachment_id="att-1",
        filename="report.pdf",
        mime_type="application/pdf",
        suffix=".pdf",
        attachment_bytes=b"%PDF-1.4\n",
    )
    store = _ScopedAttachmentStore(owner_principal=owner_principal, attachment=attachment)
    service = WebConversationService(store=store, max_turns=10, ttl_days=7)  # type: ignore[arg-type]

    owned = await service.document(owner, "conv-1", "att-1")
    assert owned is attachment

    denied = await service.document(attacker, "conv-1", "att-1")
    assert denied is None


async def test_document_route_returns_404_for_unowned_attachment() -> None:
    service = AsyncMock()
    service.document.return_value = None
    application = create_app(include_web=True)
    application.state.web_conversation_service = service
    transport = ASGITransport(app=application)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/web/conversations/00000000-0000-0000-0000-000000000001/documents/"
            "00000000-0000-0000-0000-000000000002"
        )

    assert response.status_code == 404
    service.document.assert_awaited_once()
