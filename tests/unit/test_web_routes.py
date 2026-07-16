# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for WebGUI route endpoints."""

import base64
import datetime
import html
import io
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock

import jwt
import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image

from dlightrag.api.server import create_app
from dlightrag.config import DlightragConfig
from dlightrag.core.answer_capability import AnswerImageCapability
from dlightrag.core.answer_turn import PreparedAnswerTurn
from dlightrag.storage.web_conversations import CommitTurnResult
from dlightrag.web.conversations import PreparedWebConversation

if TYPE_CHECKING:
    from dlightrag.core.servicemanager import RAGServiceManager


def _fake_manager(**attrs: object) -> RAGServiceManager:
    return cast("RAGServiceManager", SimpleNamespace(**attrs))


CONVERSATION_ID = "11111111-1111-4111-8111-111111111111"
SUBMISSION_ID = "22222222-2222-4222-8222-222222222222"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_manager():
    """Create a mock RAGServiceManager for web route tests."""
    manager = AsyncMock()
    manager.is_ready.return_value = True
    manager.is_degraded.return_value = False
    manager.get_warnings.return_value = []
    manager.answer_image_capability = AnswerImageCapability(
        status="supported",
        configured_ceiling=8,
        effective_max_images=8,
        provider="test",
        base_url=None,
        model="test-model",
        failure_kind=None,
    )
    manager.alist_workspaces = AsyncMock(return_value=["default", "test_ws"])
    manager.alist_workspace_records = AsyncMock(
        return_value=[
            {
                "workspace": "default",
                "display_name": "Default",
                "embedding_model": "voyage-multimodal-3.5",
            },
            {
                "workspace": "test_ws",
                "display_name": "Test Workspace",
                "embedding_model": "voyage-multimodal-3.5",
            },
        ]
    )
    manager.alist_ingested_files = AsyncMock(
        return_value=[{"filename": "test.pdf", "file_path": "/tmp/test.pdf"}]
    )
    manager.aget_pipeline_status = AsyncMock(
        return_value={"busy": False, "pending_enqueues": 0, "latest_message": ""}
    )
    manager.aget_file_panel_snapshot = AsyncMock(
        return_value={
            "files": [{"filename": "test.pdf", "file_path": "/tmp/test.pdf"}],
            "pipeline_status": {"busy": False, "pending_enqueues": 0, "latest_message": ""},
        }
    )
    manager.adelete_files = AsyncMock(return_value=[])
    manager.aingest = AsyncMock()
    manager.astart_ingest_job = AsyncMock(return_value={"job_id": "job-1", "status": "queued"})
    return manager


@pytest.fixture
def web_app(mock_manager, test_config: DlightragConfig):
    """Create the FastAPI app with web routes enabled and manager set."""
    application = create_app(include_web=True)
    mock_manager.config = test_config
    application.state.manager = mock_manager
    conversation_service = AsyncMock()
    conversation_service.prepare_answer = AsyncMock(
        return_value=PreparedWebConversation(
            principal_id="principal",
            conversation_id=CONVERSATION_ID,
            content_revision=0,
            text_history=({"role": "user", "content": "Earlier"},),
        )
    )

    async def _prepare_answer_turn(*, manager, prepared, query, current_images, workspaces=None):
        return PreparedAnswerTurn(
            current_query=query,
            retrieval_query=query,
            text_history=tuple(prepared.text_history),
            materialized_query_images=tuple(current_images),
        )

    conversation_service.prepare_answer_turn = AsyncMock(side_effect=_prepare_answer_turn)
    conversation_service.commit_answer = AsyncMock(
        return_value=CommitTurnResult(
            saved=True,
            reason=None,
            summary=None,
            turn_id="00000000-0000-0000-0000-000000000010",
        )
    )
    application.state.web_conversation_service = conversation_service
    return application


@pytest.fixture
async def client(web_app):
    """Create httpx async client for web route testing."""
    transport = ASGITransport(app=web_app)
    async with AsyncClient(
        transport=transport,
        base_url="http://test",
        cookies={"dlightrag_workspace": "default"},
        follow_redirects=False,
    ) as c:
        yield c


async def test_web_lifespan_initializes_one_app_scoped_conversation_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dlightrag.core.servicemanager import RAGServiceManager

    manager = AsyncMock()
    conversation_service = AsyncMock()
    application = create_app(include_web=True)
    installed_service = application.state.web_conversation_service
    application.state.web_conversation_service = conversation_service
    monkeypatch.setattr(RAGServiceManager, "acreate", AsyncMock(return_value=manager))

    async with application.router.lifespan_context(application):
        conversation_service.initialize.assert_awaited_once_with()
        assert application.state.web_conversation_service is conversation_service

    assert installed_service is not conversation_service
    manager.aclose.assert_awaited_once_with()


async def test_web_static_assets_are_not_browser_persistent(client):
    resp = await client.get("/static/generated/js/main.js")

    assert resp.status_code == 200
    assert resp.headers["cache-control"] == "no-cache, no-store, must-revalidate"
    assert resp.headers["pragma"] == "no-cache"
    assert resp.headers["expires"] == "0"
    assert "DOMContentLoaded" in resp.text


def _configure_web_manager(manager, cfg: DlightragConfig):
    manager.config = cfg
    manager.aget_pipeline_status = AsyncMock(
        return_value={"busy": False, "pending_enqueues": 0, "latest_message": ""}
    )
    return manager


def _web_client_for(cfg: DlightragConfig, manager):
    application = create_app(include_web=True)
    application.state.manager = _configure_web_manager(manager, cfg)
    transport = ASGITransport(app=application)
    return AsyncClient(
        transport=transport,
        base_url="http://test",
        cookies={"dlightrag_workspace": "default"},
        follow_redirects=False,
    )


# ---------------------------------------------------------------------------
# TestWebAuth
# ---------------------------------------------------------------------------


class TestWebAuth:
    """Web routes follow global auth_mode."""

    async def test_simple_missing_auth_redirects_browser_get(
        self, test_config: DlightragConfig, mock_manager
    ) -> None:
        test_config.auth_mode = "simple"
        test_config.api_auth_token = "secret-token"

        async with _web_client_for(test_config, mock_manager) as c:
            resp = await c.get("/web/")

        assert resp.status_code == 303
        assert resp.headers["location"].startswith("/web/login")

    async def test_source_download_login_redirect_preserves_workspace(
        self, test_config: DlightragConfig, mock_manager
    ) -> None:
        from urllib.parse import parse_qs, urlsplit

        test_config.auth_mode = "simple"
        test_config.api_auth_token = "secret-token"

        async with _web_client_for(test_config, mock_manager) as client:
            response = await client.get(
                "/web/files/raw/doc-report",
                params={"workspace": "finance"},
            )

        assert response.status_code == 303
        query = parse_qs(urlsplit(response.headers["location"]).query)
        assert query["next"] == ["/web/files/raw/doc-report?workspace=finance"]

    async def test_simple_invalid_bearer_rejected(
        self, test_config: DlightragConfig, mock_manager
    ) -> None:
        test_config.auth_mode = "simple"
        test_config.api_auth_token = "secret-token"

        async with _web_client_for(test_config, mock_manager) as c:
            resp = await c.get(
                "/web/files",
                headers={"Authorization": "Bearer wrong-token"},
            )

        assert resp.status_code == 401

    async def test_simple_login_sets_cookie_and_grants_access(
        self, test_config: DlightragConfig, mock_manager
    ) -> None:
        test_config.auth_mode = "simple"
        test_config.api_auth_token = "secret-token"

        async with _web_client_for(test_config, mock_manager) as c:
            login = await c.post(
                "/web/login",
                data={"token": "secret-token", "next": "/web/"},
            )
            resp = await c.get("/web/")

        assert login.status_code == 303
        assert "dlightrag_web_auth=" in login.headers["set-cookie"]
        assert resp.status_code == 200

    async def test_simple_login_cookie_downloads_source_without_bearer(
        self, test_config: DlightragConfig, mock_manager
    ) -> None:
        from dlightrag.core.source_download import LocalDownloadTarget

        test_config.auth_mode = "simple"
        test_config.api_auth_token = "secret-token"
        source = test_config.input_dir_path / "default" / "notes.md"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("downloadable notes", encoding="utf-8")
        mock_manager.aprepare_source_download.return_value = LocalDownloadTarget(
            path=source.resolve(),
            media_type="text/markdown",
            filename="notes.md",
        )

        async with _web_client_for(test_config, mock_manager) as c:
            await c.post(
                "/web/login",
                data={"token": "secret-token", "next": "/web/"},
            )
            response = await c.get(
                "/web/files/raw/doc-notes",
                params={"workspace": "default"},
            )
            rest_response = await c.get(
                "/files/raw/doc-notes",
                params={"workspace": "default"},
            )

        assert response.status_code == 200
        assert response.content == b"downloadable notes"
        assert rest_response.status_code == 401
        mock_manager.aprepare_source_download.assert_awaited_once_with("default", "doc-notes")

    async def test_login_redirect_rejects_external_next(
        self, test_config: DlightragConfig, mock_manager
    ) -> None:
        test_config.auth_mode = "simple"
        test_config.api_auth_token = "secret-token"

        async with _web_client_for(test_config, mock_manager) as c:
            resp = await c.post(
                "/web/login",
                data={"token": "secret-token", "next": "https://evil.example/"},
            )

        assert resp.status_code == 303
        assert resp.headers["location"] == "/web/"

    async def test_invalid_auth_cookie_is_cleared(
        self, test_config: DlightragConfig, mock_manager
    ) -> None:
        test_config.auth_mode = "simple"
        test_config.api_auth_token = "secret-token"

        async with _web_client_for(test_config, mock_manager) as c:
            c.cookies.set("dlightrag_web_auth", "not base64!")
            resp = await c.get("/web/")

        assert resp.status_code == 303
        assert resp.headers["location"].startswith("/web/login")
        assert "dlightrag_web_auth=" in resp.headers["set-cookie"]

    async def test_bearer_header_grants_web_access(
        self, test_config: DlightragConfig, mock_manager
    ) -> None:
        test_config.auth_mode = "simple"
        test_config.api_auth_token = "secret-token"

        async with _web_client_for(test_config, mock_manager) as c:
            resp = await c.get(
                "/web/files",
                headers={"Authorization": "Bearer secret-token"},
            )

        assert resp.status_code == 200

    async def test_jwt_invalid_bearer_rejected(
        self, test_config: DlightragConfig, mock_manager
    ) -> None:
        test_config.auth_mode = "jwt"
        test_config.jwt_verification_key = "test-jwt-verification-key-for-web-route-tests"

        async with _web_client_for(test_config, mock_manager) as c:
            resp = await c.get(
                "/web/files",
                headers={"Authorization": "Bearer not-a-jwt"},
            )

        assert resp.status_code == 401

    async def test_jwt_bearer_header_grants_web_access(
        self, test_config: DlightragConfig, mock_manager
    ) -> None:
        test_config.auth_mode = "jwt"
        test_config.jwt_verification_key = "test-jwt-verification-key-for-web-route-tests"
        token = jwt.encode(
            {
                "sub": "user-1",
                "exp": datetime.datetime.now(datetime.UTC) + datetime.timedelta(minutes=5),
            },
            "test-jwt-verification-key-for-web-route-tests",
            algorithm="HS256",
        )

        async with _web_client_for(test_config, mock_manager) as c:
            resp = await c.get(
                "/web/files",
                headers={"Authorization": f"Bearer {token}"},
            )

        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# TestWebIndex
# ---------------------------------------------------------------------------


class TestWebIndex:
    """Tests for GET /web/ — main page."""

    async def test_returns_html(self, client: AsyncClient, test_config: DlightragConfig) -> None:
        resp = await client.get("/web/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    async def test_contains_workspace_name(
        self, client: AsyncClient, test_config: DlightragConfig
    ) -> None:
        resp = await client.get("/web/")
        assert resp.status_code == 200
        assert "default" in resp.text

    async def test_index_renders_refresh_persistent_workspace_selector(
        self, client: AsyncClient, test_config: DlightragConfig
    ) -> None:
        resp = await client.get("/web/")

        assert resp.status_code == 200
        assert 'id="workspace-selector"' in resp.text
        assert "data-all=" in resp.text
        assert "data-active=" in resp.text
        assert "Test Workspace" in resp.text
        assert 'id="ws-add-btn"' not in resp.text

    async def test_index_renders_primary_workspace_for_last_selected_workspace(
        self, client: AsyncClient, test_config: DlightragConfig
    ) -> None:
        client.cookies.set("dlightrag_workspace", "test_ws")
        client.cookies.set("dlightrag_workspace_ids", "default,test_ws")

        resp = await client.get("/web/")

        assert resp.status_code == 200
        assert 'data-primary="test_ws"' in resp.text

    async def test_web_default_search_scope_is_all_authorized_workspaces(
        self, client: AsyncClient, test_config: DlightragConfig
    ) -> None:
        response = await client.get("/web/")

        assert response.status_code == 200
        active_match = re.search(r"data-active='([^']+)'", response.text)
        assert active_match is not None
        assert json.loads(html.unescape(active_match.group(1))) == ["default", "test_ws"]
        assert 'data-primary="default"' in response.text
        assert "Search in:" in response.text
        assert "Files in:" in response.text

    async def test_web_invalid_saved_scope_falls_back_to_all_authorized_workspaces(
        self, client: AsyncClient, test_config: DlightragConfig
    ) -> None:
        client.cookies.set("dlightrag_workspace_ids", "deleted")

        response = await client.get("/web/")

        active_match = re.search(r"data-active='([^']+)'", response.text)
        assert active_match is not None
        assert json.loads(html.unescape(active_match.group(1))) == ["default", "test_ws"]

    async def test_files_primary_target_remains_independent_of_saved_search_scope(
        self, client: AsyncClient, test_config: DlightragConfig
    ) -> None:
        client.cookies.set("dlightrag_workspace", "default")
        client.cookies.set("dlightrag_workspace_ids", "test_ws")

        response = await client.get("/web/")

        assert 'data-primary="default"' in response.text
        active_match = re.search(r"data-active='([^']+)'", response.text)
        assert active_match is not None
        assert json.loads(html.unescape(active_match.group(1))) == ["test_ws"]

    async def test_index_projects_configured_query_image_policy(
        self, client: AsyncClient, test_config: DlightragConfig, web_app
    ) -> None:
        test_config.query_images.max_upload_bytes = 12_345
        web_app.state.manager.config = test_config

        resp = await client.get("/web/")

        assert resp.status_code == 200
        assert 'data-max-upload-bytes="12345"' in resp.text

    async def test_index_projects_supported_capability_effective_limit(
        self, client: AsyncClient, test_config: DlightragConfig, web_app
    ) -> None:
        test_config.query_images.max_current_images = 3
        web_app.state.manager.config = test_config
        web_app.state.manager.answer_image_capability = AnswerImageCapability(
            status="supported",
            configured_ceiling=8,
            effective_max_images=2,
            provider="test",
            base_url=None,
            model="test-model",
            failure_kind=None,
        )

        resp = await client.get("/web/")

        assert resp.status_code == 200
        assert 'data-answer-image-capability="supported"' in resp.text
        # min(max_current_images=3, effective_max_images=2) == 2
        assert 'data-effective-current-upload-limit="2"' in resp.text

    async def test_index_unknown_capability_disables_upload(
        self, client: AsyncClient, test_config: DlightragConfig, web_app
    ) -> None:
        web_app.state.manager.config = test_config
        web_app.state.manager.answer_image_capability = AnswerImageCapability(
            status="unknown",
            configured_ceiling=8,
            effective_max_images=0,
            provider="test",
            base_url=None,
            model="test-model",
            failure_kind="timeout",
        )

        resp = await client.get("/web/")

        assert resp.status_code == 200
        assert 'data-answer-image-capability="unknown"' in resp.text
        assert 'data-effective-current-upload-limit="0"' in resp.text

    def test_web_markup_keeps_behavior_in_static_js(self) -> None:
        web_root = Path(__file__).parents[2] / "src" / "dlightrag" / "web"
        checked = list((web_root / "templates").rglob("*.html")) + [web_root / "deps.py"]

        offenders: list[str] = []
        for path in checked:
            text = path.read_text()
            for marker in ("onclick=", "onchange=", "style="):
                if marker in text:
                    offenders.append(f"{path.relative_to(web_root)}:{marker}")

        assert offenders == []


class TestWebAnswer:
    """Tests for POST /web/answer."""

    async def test_answer_stream_uses_private_prepared_manager_boundary(
        self,
        client: AsyncClient,
        test_config: DlightragConfig,
        web_app,
    ) -> None:
        async def mock_tokens():
            yield "Answer"

        class PreparedManager:
            def __init__(self) -> None:
                self.config = test_config
                self.answer_image_capability = None
                self._aanswer_stream_prepared = AsyncMock(
                    return_value=({"chunks": []}, mock_tokens())
                )

        manager = PreparedManager()
        web_app.state.manager = manager

        resp = await client.post(
            "/web/answer",
            json={
                "query": "hello",
                "conversation_id": CONVERSATION_ID,
                "submission_id": SUBMISSION_ID,
            },
        )

        assert resp.status_code == 200
        assert "event: done" in resp.text
        assert '"answer": "Answer"' in resp.text
        assert "Service error" not in resp.text
        manager._aanswer_stream_prepared.assert_awaited_once()
        await_args = manager._aanswer_stream_prepared.await_args
        assert await_args is not None
        turn = await_args.args[0]
        assert turn.current_query == "hello"
        assert turn.retrieval_query == "hello"
        assert turn.text_history == ({"role": "user", "content": "Earlier"},)

    async def test_answer_rejects_legacy_conversation_fields(
        self,
        client: AsyncClient,
        test_config: DlightragConfig,
        web_app,
    ) -> None:
        resp = await client.post(
            "/web/answer",
            json={
                "query": "hello",
                "conversation_id": CONVERSATION_ID,
                "submission_id": SUBMISSION_ID,
                "session_id": "session-1",
            },
        )

        assert resp.status_code == 422

    async def test_highlights_use_keyword_llm_role(
        self,
        client: AsyncClient,
        test_config: DlightragConfig,
        web_app,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        async def keyword_llm(*, messages, **kwargs):
            return '{"phrases": ["Evidence"], "confidence": 1.0}'

        keyword_called = False
        highlights_called = False

        def fake_get_keyword_model_func(cfg: DlightragConfig):
            nonlocal keyword_called
            keyword_called = True
            return keyword_llm

        async def fake_extract_highlights_for_sources(*, sources, answer_text, llm_func, **kwargs):
            from dlightrag.citations.schemas import SourceReferencePayload

            nonlocal highlights_called
            highlights_called = True
            assert llm_func is keyword_llm
            assert isinstance(sources[0], SourceReferencePayload)
            assert sources[0].download_url == "/web/files/raw/doc-report?workspace=default"
            sources[0].chunks[0].highlight_phrases = ["Evidence"]
            return sources

        def fail_query_model(*args, **kwargs):
            raise AssertionError("highlighter must not use query role")

        async def mock_tokens():
            yield "Evidence [1-1]."

        class PublicOnlyManager:
            def __init__(self) -> None:
                self.config = test_config
                self.answer_image_capability = None
                self._aanswer_stream_prepared = AsyncMock(
                    return_value=(
                        {
                            "chunks": [
                                {
                                    "chunk_id": "c1",
                                    "reference_id": "1",
                                    "full_doc_id": "doc-report",
                                    "content": "Evidence in cited chunk.",
                                    "file_path": "/docs/report.pdf",
                                    "_workspace": "default",
                                    "metadata": {
                                        "source_uri": "s3://bucket/report.pdf",
                                        "source_download_locator": "s3://bucket/report.pdf",
                                    },
                                }
                            ]
                        },
                        mock_tokens(),
                    )
                )

        test_config.citations.highlights.enabled = True
        manager = PublicOnlyManager()
        web_app.state.manager = manager
        monkeypatch.setattr(
            "dlightrag.models.llm.get_keyword_model_func",
            fake_get_keyword_model_func,
            raising=False,
        )
        monkeypatch.setattr("dlightrag.models.llm.get_query_model_func", fail_query_model)
        monkeypatch.setattr(
            "dlightrag.core.answer_highlights.extract_highlights_for_sources",
            fake_extract_highlights_for_sources,
        )

        resp = await client.post(
            "/web/answer",
            json={
                "query": "hello",
                "conversation_id": CONVERSATION_ID,
                "submission_id": SUBMISSION_ID,
            },
        )

        assert resp.status_code == 200
        assert "event: highlights" in resp.text
        assert keyword_called is True
        assert highlights_called is True

    async def test_query_only_user_does_not_receive_download_affordance(
        self,
        client: AsyncClient,
        test_config: DlightragConfig,
        web_app,
    ) -> None:
        from dlightrag.access_control import AccessAction, AccessDeniedError

        class QueryOnlyAccess:
            async def check(self, user, action, *, workspace=None):
                if action != AccessAction.WORKSPACE_QUERY:
                    raise AccessDeniedError("denied")

            async def filter_workspaces(self, user, action, workspaces):
                if action == AccessAction.WORKSPACE_QUERY:
                    return list(workspaces)
                return []

        async def mock_tokens():
            yield "Evidence [1-1]."

        manager = SimpleNamespace(
            config=test_config,
            answer_image_capability=None,
            _aanswer_stream_prepared=AsyncMock(
                return_value=(
                    {
                        "chunks": [
                            {
                                "chunk_id": "c1",
                                "reference_id": "1",
                                "full_doc_id": "doc-report",
                                "content": "Evidence",
                                "file_path": "report.pdf",
                                "_workspace": "default",
                                "metadata": {
                                    "source_uri": "local://default/report.pdf",
                                    "source_download_locator": "/private/report.pdf",
                                },
                            }
                        ]
                    },
                    mock_tokens(),
                )
            ),
        )
        web_app.state.manager = manager
        web_app.state.access_control = QueryOnlyAccess()

        response = await client.post(
            "/web/answer",
            json={
                "query": "hello",
                "conversation_id": CONVERSATION_ID,
                "submission_id": SUBMISSION_ID,
            },
        )

        assert response.status_code == 200
        assert "event: done" in response.text
        assert "source-dl-icon" not in response.text
        assert "/web/files/raw" not in response.text


class TestWebSSEBoundary:
    """Tests for SSE framing and browser HTML fragment safety."""

    def test_sse_event_json_encodes_payload(self) -> None:
        from dlightrag.web.events import AnswerDoneEvent
        from dlightrag.web.sse import sse_event

        event = sse_event(
            "done", AnswerDoneEvent(html="<b>x</b>", answer="x", conversation_saved=False)
        )

        assert event.startswith("event: done\n")
        assert event.endswith("\n\n")
        data_line = next(line for line in event.splitlines() if line.startswith("data: "))
        assert json.loads(data_line.removeprefix("data: ")) == {
            "html": "<b>x</b>",
            "answer": "x",
            "current_image_ids": [],
            "image_descriptions": {},
            "answer_images": [],
            "answer_blocks": [],
            "conversation_saved": False,
        }

    async def test_answer_done_html_strips_unsafe_urls(
        self,
        client: AsyncClient,
        test_config: DlightragConfig,
        web_app,
    ) -> None:
        async def mock_tokens():
            yield "Answer [1-1]."

        class PublicOnlyManager:
            def __init__(self) -> None:
                self.config = test_config
                self.answer_image_capability = None
                self._aanswer_stream_prepared = AsyncMock(
                    return_value=(
                        {
                            "chunks": [
                                {
                                    "chunk_id": "c1",
                                    "reference_id": "1",
                                    "full_doc_id": "doc-report",
                                    "content": "Evidence in cited chunk.",
                                    "file_path": "/docs/report.pdf",
                                    "image_url": "javascript:alert(1)",
                                    "thumbnail_url": "javascript:alert(2)",
                                    "_answer_image_sent": True,
                                    "_workspace": "default",
                                    "metadata": {
                                        "source_uri": "s3://bucket/report.pdf",
                                        "source_download_locator": "s3://bucket/report.pdf",
                                    },
                                }
                            ]
                        },
                        mock_tokens(),
                    )
                )

        web_app.state.manager = PublicOnlyManager()

        resp = await client.post(
            "/web/answer",
            json={
                "query": "hello",
                "conversation_id": CONVERSATION_ID,
                "submission_id": SUBMISSION_ID,
            },
        )

        assert resp.status_code == 200
        assert "event: done" in resp.text
        assert "answer-content" in resp.text
        assert "source-data" in resp.text
        assert "javascript:" not in resp.text
        assert "onerror" not in resp.text.lower()
        assert "<script" not in resp.text.lower()

    async def test_missing_source_document_id_uses_structured_answer_error(
        self,
        client: AsyncClient,
        test_config: DlightragConfig,
        web_app,
    ) -> None:
        async def mock_tokens():
            yield "Answer [1-1]."

        class PublicOnlyManager:
            def __init__(self) -> None:
                self.config = test_config
                self.answer_image_capability = None
                self._aanswer_stream_prepared = AsyncMock(
                    return_value=(
                        {
                            "chunks": [
                                {
                                    "chunk_id": "c1",
                                    "reference_id": "1",
                                    "content": "Evidence in cited chunk.",
                                    "file_path": "report.pdf",
                                    "_workspace": "default",
                                    "metadata": {
                                        "source_uri": "bynder://asset/1",
                                        "source_download_locator": "file://private/report.pdf",
                                    },
                                }
                            ]
                        },
                        mock_tokens(),
                    )
                )

        web_app.state.manager = PublicOnlyManager()

        response = await client.post(
            "/web/answer",
            json={
                "query": "hello",
                "conversation_id": CONVERSATION_ID,
                "submission_id": SUBMISSION_ID,
            },
        )

        assert response.status_code == 200
        assert "event: error" in response.text
        assert "Service error. Please try again." in response.text
        assert "event: done" not in response.text
        assert "file://private/report.pdf" not in response.text


class TestWebAnswerAdapter:
    """The route is a thin adapter over the browser answer presenter."""

    async def test_answer_route_delegates_to_stream_presenter(
        self,
        client: AsyncClient,
        test_config: DlightragConfig,
        web_app,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured = {}

        async def fake_stream_answer_events(**kwargs):
            captured.update(kwargs)
            yield 'event: done\ndata: {"html": "", "answer": "ok"}\n\n'

        monkeypatch.setattr(
            "dlightrag.web.routes.chat.stream_answer_events",
            fake_stream_answer_events,
            raising=False,
        )
        web_app.state.manager.config = test_config
        image_buffer = io.BytesIO()
        Image.new("RGB", (1, 1), "white").save(image_buffer, format="PNG")
        image_b64 = base64.b64encode(image_buffer.getvalue()).decode()

        resp = await client.post(
            "/web/answer",
            json={
                "query": "hello",
                "conversation_id": CONVERSATION_ID,
                "submission_id": SUBMISSION_ID,
                "images": [image_b64],
                "workspaces": ["default", "test_ws"],
            },
        )

        assert resp.status_code == 200
        assert captured["manager"] is web_app.state.manager
        assert captured["cfg"] is test_config
        assert captured["turn"].current_query == "hello"
        assert captured["turn"].text_history == ({"role": "user", "content": "Earlier"},)
        assert list(captured["turn"].materialized_query_images) == [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
        ]
        assert captured["workspaces"] == ["default", "test_ws"]
        assert captured["workspace"] == "default"
        assert "session_id" not in captured
        assert "conversation_history" not in captured

    async def test_answer_route_rejects_untyped_history_messages(
        self,
        client: AsyncClient,
        test_config: DlightragConfig,
        web_app,
    ) -> None:
        web_app.state.manager.config = test_config

        resp = await client.post(
            "/web/answer",
            json={
                "query": "hello",
                "conversation_id": CONVERSATION_ID,
                "submission_id": SUBMISSION_ID,
                "conversation_history": [{"role": "human", "content": "previous"}],
            },
        )

        assert resp.status_code == 422

    async def test_answer_requires_owned_conversation_before_model_call(
        self, client: AsyncClient, web_app
    ) -> None:
        web_app.state.web_conversation_service.prepare_answer.return_value = None

        response = await client.post(
            "/web/answer",
            json={
                "query": "hello",
                "conversation_id": CONVERSATION_ID,
                "submission_id": SUBMISSION_ID,
            },
        )

        assert response.status_code == 404
        web_app.state.manager._aanswer_stream_prepared.assert_not_awaited()

    async def test_invalid_image_is_rejected_instead_of_skipped(
        self, client: AsyncClient, test_config: DlightragConfig, web_app
    ) -> None:
        web_app.state.manager.config = test_config

        response = await client.post(
            "/web/answer",
            json={
                "query": "hello",
                "conversation_id": CONVERSATION_ID,
                "submission_id": SUBMISSION_ID,
                "images": [base64.b64encode(b"not an image").decode()],
            },
        )

        assert response.status_code == 422

    async def test_answer_requires_browser_submission_id(
        self, client: AsyncClient, test_config: DlightragConfig, web_app
    ) -> None:
        web_app.state.manager.config = test_config

        response = await client.post(
            "/web/answer", json={"query": "hello", "conversation_id": CONVERSATION_ID}
        )

        assert response.status_code == 422

    async def test_answer_rejects_configured_image_count_before_model(
        self, client: AsyncClient, test_config: DlightragConfig, web_app
    ) -> None:
        test_config.query_images.max_current_images = 1
        web_app.state.manager.config = test_config
        image_buffer = io.BytesIO()
        Image.new("RGB", (1, 1), "white").save(image_buffer, format="PNG")
        image_b64 = base64.b64encode(image_buffer.getvalue()).decode()

        response = await client.post(
            "/web/answer",
            json={
                "query": "hello",
                "conversation_id": CONVERSATION_ID,
                "submission_id": SUBMISSION_ID,
                "images": [image_b64, image_b64],
            },
        )

        assert response.status_code == 422
        web_app.state.manager._aanswer_stream_prepared.assert_not_awaited()

    async def test_answer_rejects_body_over_route_limit_before_json_materialization(
        self, client: AsyncClient, test_config: DlightragConfig, web_app
    ) -> None:
        test_config.query_images.max_current_images = 0
        web_app.state.manager.config = test_config

        response = await client.post(
            "/web/answer",
            content=b"{" + (b"x" * 70_000) + b"}",
            headers={"Content-Type": "application/json"},
        )

        assert response.status_code == 413
        web_app.state.web_conversation_service.prepare_answer.assert_not_awaited()

    async def test_answer_body_limit_uses_configured_image_count_and_bytes(
        self, client: AsyncClient, test_config: DlightragConfig, web_app
    ) -> None:
        test_config.query_images.max_current_images = 1
        test_config.query_images.max_upload_bytes = 3
        web_app.state.manager.config = test_config
        exact_limit = (64 * 1024) + 4

        exact = await client.post(
            "/web/answer",
            content=b"x" * exact_limit,
            headers={"Content-Type": "application/json"},
        )
        over = await client.post(
            "/web/answer",
            content=b"x" * (exact_limit + 1),
            headers={"Content-Type": "application/json"},
        )

        assert exact.status_code == 422
        assert over.status_code == 413

    async def test_answer_image_validation_uses_configured_exact_byte_limit(
        self, client: AsyncClient, test_config: DlightragConfig, web_app, monkeypatch
    ) -> None:
        async def fake_stream_answer_events(**_kwargs):
            yield "event: done\ndata: {}\n\n"

        monkeypatch.setattr(
            "dlightrag.web.routes.chat.stream_answer_events",
            fake_stream_answer_events,
            raising=False,
        )
        image_buffer = io.BytesIO()
        Image.new("RGB", (1, 1), "white").save(image_buffer, format="PNG")
        raw = image_buffer.getvalue()
        payload = base64.b64encode(raw).decode()
        test_config.query_images.max_upload_bytes = len(raw)
        web_app.state.manager.config = test_config

        exact = await client.post(
            "/web/answer",
            json={
                "query": "hello",
                "conversation_id": CONVERSATION_ID,
                "submission_id": SUBMISSION_ID,
                "images": [payload],
            },
        )
        test_config.query_images.max_upload_bytes = len(raw) - 1
        over = await client.post(
            "/web/answer",
            json={
                "query": "hello",
                "conversation_id": CONVERSATION_ID,
                "submission_id": "33333333-3333-4333-8333-333333333333",
                "images": [payload],
            },
        )

        assert exact.status_code == 200
        assert over.status_code == 422

    async def test_answer_rejects_invalid_base64_before_model(
        self, client: AsyncClient, test_config: DlightragConfig, web_app
    ) -> None:
        web_app.state.manager.config = test_config

        response = await client.post(
            "/web/answer",
            json={
                "query": "hello",
                "conversation_id": CONVERSATION_ID,
                "submission_id": SUBMISSION_ID,
                "images": ["%%%"],
            },
        )

        assert response.status_code == 422
        web_app.state.manager._aanswer_stream_prepared.assert_not_awaited()

    async def test_same_submission_replay_returns_stored_answer_before_model(
        self, client: AsyncClient, test_config: DlightragConfig, web_app
    ) -> None:
        web_app.state.manager.config = test_config
        web_app.state.web_conversation_service.prepare_answer.return_value = (
            PreparedWebConversation(
                principal_id="a" * 64,
                conversation_id=CONVERSATION_ID,
                content_revision=1,
                text_history=(),
                committed_submission=CommitTurnResult(
                    saved=True,
                    reason=None,
                    summary=None,
                    turn_id="turn",
                    current_image_ids=("stored-image",),
                    assistant_text="Stored answer",
                    answer_sources={"sources": [], "answer_images": []},
                    replayed=True,
                ),
            )
        )

        response = await client.post(
            "/web/answer",
            json={
                "query": "hello",
                "conversation_id": CONVERSATION_ID,
                "submission_id": SUBMISSION_ID,
            },
        )

        assert response.status_code == 200
        assert "Stored answer" in response.text
        assert "stored-image" in response.text
        web_app.state.manager._aanswer_stream_prepared.assert_not_awaited()


# ---------------------------------------------------------------------------
# TestWebFiles
# ---------------------------------------------------------------------------


class TestWebFiles:
    """Tests for GET /web/files and DELETE /web/files."""

    async def test_file_list_returns_html(
        self, client: AsyncClient, test_config: DlightragConfig
    ) -> None:
        resp = await client.get("/web/files")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    async def test_file_list_derives_display_name_from_path(
        self, client: AsyncClient, test_config: DlightragConfig, mock_manager
    ) -> None:
        mock_manager.aget_file_panel_snapshot = AsyncMock(
            return_value={
                "files": [
                    {"doc_id": "d1", "file_path": "/tmp/reports/q4.pdf", "status": "processed"}
                ],
                "pipeline_status": {"busy": False, "pending_enqueues": 0},
            }
        )

        resp = await client.get("/web/files")

        assert resp.status_code == 200
        assert ">q4.pdf</span>" in resp.text
        assert 'title="/tmp/reports/q4.pdf"' in resp.text

    async def test_file_list_uses_file_panel_snapshot_for_cold_workspace(
        self, client: AsyncClient, test_config: DlightragConfig, mock_manager
    ) -> None:
        mock_manager.alist_workspaces = AsyncMock(return_value=["default", "cold_ws"])
        mock_manager.aget_file_panel_snapshot = AsyncMock(
            return_value={
                "files": [
                    {"doc_id": "d1", "file_path": "/tmp/cold/report.pdf", "status": "processed"}
                ],
                "pipeline_status": {"busy": False, "pending_enqueues": 0},
            }
        )
        mock_manager.alist_ingested_files = AsyncMock(return_value=[])
        mock_manager.aget_pipeline_status = AsyncMock(return_value={"busy": False})

        resp = await client.get("/web/files", params={"workspace": "cold-ws"})

        assert resp.status_code == 200
        assert ">report.pdf</span>" in resp.text
        mock_manager.aget_file_panel_snapshot.assert_awaited_once_with("cold_ws")
        mock_manager.alist_ingested_files.assert_not_awaited()
        mock_manager.aget_pipeline_status.assert_not_awaited()

    async def test_file_list_rejects_stale_workspace(
        self, client: AsyncClient, test_config: DlightragConfig, mock_manager
    ) -> None:
        mock_manager.alist_workspaces = AsyncMock(return_value=["default"])

        resp = await client.get("/web/files", params={"workspace": "deleted_ws"})

        assert resp.status_code == 409
        assert "Workspace no longer exists" in resp.text
        mock_manager.aget_file_panel_snapshot.assert_not_awaited()
        mock_manager.alist_ingested_files.assert_not_awaited()
        mock_manager.aget_pipeline_status.assert_not_awaited()

    async def test_file_list_rejects_stale_workspace_even_with_registered_cookie(
        self, client: AsyncClient, test_config: DlightragConfig, mock_manager
    ) -> None:
        mock_manager.alist_workspaces = AsyncMock(return_value=["default", "test_ws"])
        client.cookies.set("dlightrag_workspace", "test_ws")

        resp = await client.get("/web/files", params={"workspace": "deleted_ws"})

        assert resp.status_code == 409
        assert "Workspace no longer exists" in resp.text
        mock_manager.aget_file_panel_snapshot.assert_not_awaited()
        mock_manager.alist_ingested_files.assert_not_awaited()
        mock_manager.aget_pipeline_status.assert_not_awaited()

    async def test_file_list_canonicalizes_requested_workspace(
        self, client: AsyncClient, test_config: DlightragConfig, mock_manager
    ) -> None:
        mock_manager.alist_workspaces = AsyncMock(return_value=["default", "test_fallback_ws"])

        resp = await client.get("/web/files", params={"workspace": "test-fallback-ws"})

        assert resp.status_code == 200
        mock_manager.aget_file_panel_snapshot.assert_awaited_once_with("test_fallback_ws")

    async def test_file_list_rejects_stale_workspace_without_default(
        self, client: AsyncClient, test_config: DlightragConfig, mock_manager
    ) -> None:
        mock_manager.alist_workspaces = AsyncMock(return_value=["other_ws"])

        resp = await client.get("/web/files", params={"workspace": "deleted_ws"})

        assert resp.status_code == 409
        assert "Workspace no longer exists" in resp.text
        mock_manager.aget_file_panel_snapshot.assert_not_awaited()
        mock_manager.alist_ingested_files.assert_not_awaited()

    async def test_ingest_status_rejects_stale_workspace(
        self, client: AsyncClient, test_config: DlightragConfig, mock_manager
    ) -> None:
        mock_manager.alist_workspaces = AsyncMock(return_value=["default"])

        resp = await client.get("/web/ingest-status", params={"workspace": "deleted_ws"})

        assert resp.status_code == 409
        assert "Workspace no longer exists" in resp.text
        mock_manager.aget_pipeline_status.assert_not_awaited()

    async def test_ingest_status_done_preserves_panel_content_container(
        self, client: AsyncClient, test_config: DlightragConfig
    ) -> None:
        resp = await client.get("/web/ingest-status", params={"workspace": "default"})

        assert resp.status_code == 200
        assert resp.headers["hx-retarget"] == "#panel-content"
        assert resp.headers["hx-reswap"] == "innerHTML"

    async def test_upload_preserves_filename_for_directory_ingest(
        self, client: AsyncClient, test_config: DlightragConfig, mock_manager
    ) -> None:
        resp = await client.post(
            "/web/files/upload",
            files=[("files", ("report.pdf", b"%PDF-fake", "application/pdf"))],
        )

        assert resp.status_code == 200
        mock_manager.aingest.assert_not_awaited()
        mock_manager.astart_ingest_job.assert_awaited_once()
        call = mock_manager.astart_ingest_job.await_args
        assert call.args[0] == "default"
        ingest_spec = call.args[1]
        assert ingest_spec.source_type == "local"
        upload_dir = Path(ingest_spec.path)
        assert upload_dir.is_dir()
        assert (upload_dir / "report.pdf").read_bytes() == b"%PDF-fake"

    async def test_upload_rejects_stale_workspace(
        self, client: AsyncClient, test_config: DlightragConfig, mock_manager
    ) -> None:
        mock_manager.alist_workspaces = AsyncMock(return_value=["default"])

        resp = await client.post(
            "/web/files/upload",
            data={"workspace": "deleted_ws"},
            files=[("files", ("report.pdf", b"%PDF-fake", "application/pdf"))],
        )

        assert resp.status_code == 409
        assert "Workspace no longer exists" in resp.text
        mock_manager.aingest.assert_not_awaited()
        mock_manager.astart_ingest_job.assert_not_awaited()

    @pytest.mark.parametrize(
        "filename",
        [
            "/tmp/evil.pdf",
            "../evil.pdf",
            r"..\evil.pdf",
            r"folder\..\evil.pdf",
            r"C:\Users\me\secret.pdf",
        ],
    )
    def test_safe_relative_path_rejects_unsafe_paths(self, filename: str) -> None:
        from dlightrag.core.ingestion.uploads import safe_upload_relative_path

        with pytest.raises(ValueError):
            safe_upload_relative_path(filename)

    async def test_delete_files(
        self, client: AsyncClient, test_config: DlightragConfig, mock_manager
    ) -> None:
        resp = await client.request(
            "DELETE",
            "/web/files",
            params={"file_path": "/tmp/test.pdf"},
        )
        assert resp.status_code == 200
        mock_manager.adelete_files.assert_awaited_once()

    async def test_delete_files_rejects_stale_workspace(
        self, client: AsyncClient, test_config: DlightragConfig, mock_manager
    ) -> None:
        mock_manager.alist_workspaces = AsyncMock(return_value=["default"])

        resp = await client.request(
            "DELETE",
            "/web/files",
            params={"workspace": "deleted_ws", "file_path": "/tmp/test.pdf"},
        )

        assert resp.status_code == 409
        assert "Workspace no longer exists" in resp.text
        mock_manager.adelete_files.assert_not_awaited()


# ---------------------------------------------------------------------------
# TestWebWorkspaceCreateDelete
# ---------------------------------------------------------------------------


class TestWebWorkspaceCreate:
    """Tests for POST /web/workspaces/create."""

    async def test_create_workspace(
        self, client: AsyncClient, test_config: DlightragConfig, mock_manager
    ) -> None:
        mock_manager.acreate_workspace = AsyncMock()
        # First call (duplicate check): workspace does not exist yet
        # Second call (post-create list): includes the new workspace
        mock_manager.alist_workspaces = AsyncMock(
            side_effect=[["default", "test_ws"], ["default", "test_ws", "new_workspace"]]
        )
        resp = await client.post(
            "/web/workspaces/create",
            data={"workspace_name": "new workspace"},
        )
        assert resp.status_code == 200
        assert "workspaceCreated" in resp.headers["hx-trigger"]
        set_cookies = resp.headers.get_list("set-cookie")
        assert any(
            cookie.startswith("dlightrag_workspace=new_workspace;") for cookie in set_cookies
        )
        assert any(
            cookie.startswith("dlightrag_workspace_ids=new_workspace;") for cookie in set_cookies
        )
        mock_manager.acreate_workspace.assert_awaited_once_with(
            "new_workspace",
            display_name="new workspace",
        )

    async def test_create_workspace_empty_name(
        self, client: AsyncClient, test_config: DlightragConfig
    ) -> None:
        resp = await client.post(
            "/web/workspaces/create",
            data={"workspace_name": ""},
        )
        assert resp.status_code == 400
        assert 'class="file-error"' in resp.text

    async def test_create_workspace_duplicate(
        self, client: AsyncClient, test_config: DlightragConfig, mock_manager
    ) -> None:
        resp = await client.post(
            "/web/workspaces/create",
            data={"workspace_name": "default"},
        )
        assert resp.status_code == 409

    async def test_create_workspace_forbidden_chars(
        self, client: AsyncClient, test_config: DlightragConfig
    ) -> None:
        resp = await client.post(
            "/web/workspaces/create",
            data={"workspace_name": "bad/name"},
        )
        assert resp.status_code == 400

    async def test_create_workspace_too_long(
        self, client: AsyncClient, test_config: DlightragConfig
    ) -> None:
        resp = await client.post(
            "/web/workspaces/create",
            data={"workspace_name": "a" * 65},
        )
        assert resp.status_code == 400


class TestWebWorkspaceDelete:
    """Tests for POST /web/workspaces/delete."""

    async def test_delete_workspace(
        self, client: AsyncClient, test_config: DlightragConfig, mock_manager
    ) -> None:
        mock_manager.areset = AsyncMock(return_value={"workspaces": {}, "total_errors": 0})
        mock_manager.alist_workspaces = AsyncMock(return_value=["default"])
        resp = await client.post(
            "/web/workspaces/delete",
            data={"workspace_name": "test-ws", "confirm_name": "test-ws"},
        )
        assert resp.status_code == 200
        assert "workspaceDeleted" in resp.headers["hx-trigger"]
        assert "dlightrag_workspace=default" in resp.headers["set-cookie"]
        mock_manager.areset.assert_awaited_once_with(workspace="test_ws")

    async def test_delete_default_workspace_selects_first_remaining_workspace(
        self, client: AsyncClient, test_config: DlightragConfig, mock_manager
    ) -> None:
        mock_manager.areset = AsyncMock(return_value={"workspaces": {}, "total_errors": 0})
        mock_manager.alist_workspaces = AsyncMock(return_value=["research"])

        resp = await client.post(
            "/web/workspaces/delete",
            data={"workspace_name": "default", "confirm_name": "default"},
        )

        assert resp.status_code == 200
        trigger = json.loads(resp.headers["hx-trigger"])
        assert trigger["workspaceDeleted"] == {
            "workspace": "default",
            "next_workspace": "research",
        }
        set_cookies = resp.headers.get_list("set-cookie")
        assert any(cookie.startswith("dlightrag_workspace=research;") for cookie in set_cookies)
        assert any(cookie.startswith("dlightrag_workspace_ids=research;") for cookie in set_cookies)

    async def test_delete_hyphen_workspace_emits_canonical_workspace(
        self, client: AsyncClient, test_config: DlightragConfig, mock_manager
    ) -> None:
        mock_manager.areset = AsyncMock(return_value={"workspaces": {}, "total_errors": 0})
        mock_manager.alist_workspaces = AsyncMock(return_value=["default"])

        resp = await client.post(
            "/web/workspaces/delete",
            data={"workspace_name": "test-fallback-ws", "confirm_name": "test-fallback-ws"},
        )

        assert resp.status_code == 200
        trigger = json.loads(resp.headers["hx-trigger"])
        assert trigger["workspaceDeleted"] == {
            "workspace": "test_fallback_ws",
            "next_workspace": "default",
        }
        mock_manager.areset.assert_awaited_once_with(workspace="test_fallback_ws")

    async def test_delete_workspace_confirm_mismatch(
        self, client: AsyncClient, test_config: DlightragConfig
    ) -> None:
        resp = await client.post(
            "/web/workspaces/delete",
            data={"workspace_name": "test-ws", "confirm_name": "wrong"},
        )
        assert resp.status_code == 400

    async def test_delete_workspace_empty_name(
        self, client: AsyncClient, test_config: DlightragConfig
    ) -> None:
        resp = await client.post(
            "/web/workspaces/delete",
            data={"workspace_name": "", "confirm_name": ""},
        )
        assert resp.status_code == 400


class TestSourcePanelTemplate:
    """Tests for source panel rendering contracts."""

    def test_zero_page_idx_is_rendered_as_page_zero(self) -> None:
        from dlightrag.web.deps import templates

        html = templates.env.get_template("partials/source_panel.html").render(
            sources=[
                {
                    "id": "1",
                    "title": "Doc",
                    "path": "/tmp/doc.pdf",
                    "chunks": [
                        {
                            "chunk_idx": 1,
                            "page_idx": 0,
                            "content": "first page",
                        }
                    ],
                }
            ]
        )

        assert "p.0" in html
        assert "#1" not in html

    def test_markdown_source_keeps_visible_download_action_when_url_exists(self) -> None:
        from dlightrag.web.deps import templates

        rendered = templates.env.get_template("partials/source_panel.html").render(
            sources=[
                {
                    "id": "1",
                    "title": "notes.md",
                    "download_url": "/web/files/raw/doc-notes?workspace=default",
                    "chunks": [],
                }
            ]
        )

        assert "notes.md" in rendered
        assert 'href="/web/files/raw/doc-notes?workspace=default"' in rendered
        assert 'aria-label="Download source"' in rendered


async def test_web_answer_done_builder_projects_http_source_payloads(
    tmp_path: Path,
) -> None:
    from dlightrag.citations.schemas import SourceReferencePayload
    from dlightrag.web import answer_events
    from dlightrag.web.safe_html import safe_source_panel

    contexts = {
        "chunks": [
            {
                "chunk_id": "c1",
                "reference_id": "1",
                "full_doc_id": "doc-report",
                "file_path": "report.pdf",
                "content": "Evidence",
                "_workspace": "finance",
                "metadata": {
                    "source_uri": "s3://bucket/report.pdf",
                    "source_download_locator": "s3://bucket/report.pdf",
                    "source_file_name": "report.pdf",
                },
            }
        ]
    }
    manager = _fake_manager(config=SimpleNamespace(workspace="default"))
    cfg = SimpleNamespace(input_dir_path=tmp_path / "inputs")

    payload = await answer_events._build_answer_done_payload(
        clean_answer="Answer [1-1].",
        contexts=contexts,
        image_descriptions={},
        manager=manager,
        cfg=cfg,
        workspace="default",
    )

    assert len(payload.sources) == 1
    source = payload.sources[0]
    assert isinstance(source, SourceReferencePayload)
    expected_url = "/web/files/raw/doc-report?workspace=finance"
    assert source.download_url == expected_url
    assert payload.done.html.count(f'href="{expected_url}"') == 1
    assert safe_source_panel(sources=payload.sources).count(f'href="{expected_url}"') == 1


async def test_web_answer_done_builder_extracts_images_before_public_projection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from dlightrag.citations.schemas import SourceReference, SourceReferencePayload
    from dlightrag.web import answer_events

    calls: list[str] = []

    def capture_images(sources, *, contexts):  # noqa: ANN001, ANN202
        assert isinstance(sources[0], SourceReference)
        calls.append("images")
        return []

    def capture_projection(  # noqa: ANN001, ANN202
        sources,
        *,
        resolver,
        downloadable_workspaces=None,
    ):
        assert isinstance(sources[0], SourceReference)
        assert downloadable_workspaces is None
        calls.append("projection")
        return [
            SourceReferencePayload(
                id=sources[0].id,
                title=sources[0].title,
                source_uri=sources[0].source_uri,
                download_url="/web/files/raw/doc-report?workspace=default",
                chunks=sources[0].chunks,
            )
        ]

    monkeypatch.setattr(answer_events, "answer_images_from_sources", capture_images)
    monkeypatch.setattr(answer_events, "project_source_payloads", capture_projection)
    contexts = {
        "chunks": [
            {
                "chunk_id": "c1",
                "reference_id": "1",
                "full_doc_id": "doc-report",
                "file_path": "report.pdf",
                "content": "Evidence",
                "_workspace": "default",
                "metadata": {
                    "source_uri": "local://default/report.pdf",
                    "source_download_locator": "report.pdf",
                },
            }
        ]
    }
    manager = _fake_manager(config=SimpleNamespace(workspace="default"))
    cfg = SimpleNamespace(input_dir_path=tmp_path / "inputs")

    await answer_events._build_answer_done_payload(
        clean_answer="Answer [1-1].",
        contexts=contexts,
        image_descriptions={},
        manager=manager,
        cfg=cfg,
        workspace="default",
    )

    assert calls == ["images", "projection"]
