# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for WebGUI route endpoints."""

from __future__ import annotations

import base64
import datetime
import json
from pathlib import Path
from unittest.mock import AsyncMock

import jwt
import pytest
from httpx import ASGITransport, AsyncClient

from dlightrag.api.server import create_app
from dlightrag.config import DlightragConfig

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
    manager.list_workspaces = AsyncMock(return_value=["default", "test_ws"])
    manager.list_workspace_records = AsyncMock(
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
    manager.list_ingested_files = AsyncMock(
        return_value=[{"filename": "test.pdf", "file_path": "/tmp/test.pdf"}]
    )
    manager.get_pipeline_status = AsyncMock(
        return_value={"busy": False, "pending_enqueues": 0, "latest_message": ""}
    )
    manager.delete_files = AsyncMock(return_value=[])
    manager.aingest = AsyncMock()
    manager.astart_ingest_job = AsyncMock(return_value={"job_id": "job-1", "status": "queued"})
    manager._astart_staged_local_ingest_job = AsyncMock(
        return_value={"job_id": "job-1", "status": "queued"}
    )
    return manager


@pytest.fixture
def web_app(mock_manager):
    """Create the FastAPI app with web routes enabled and manager set."""
    application = create_app(include_web=True)
    application.state.manager = mock_manager
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


async def test_web_static_assets_are_not_browser_persistent(client):
    resp = await client.get("/static/js/main.js")

    assert resp.status_code == 200
    assert resp.headers["cache-control"] == "no-cache, no-store, must-revalidate"
    assert resp.headers["pragma"] == "no-cache"
    assert resp.headers["expires"] == "0"
    assert "DOMContentLoaded" in resp.text


def _configure_web_manager(manager, cfg: DlightragConfig):
    manager.config = cfg
    manager.get_pipeline_status = AsyncMock(
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

        assert resp.status_code == 403

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

    async def test_answer_stream_uses_public_manager_methods(
        self,
        client: AsyncClient,
        test_config: DlightragConfig,
        web_app,
    ) -> None:
        async def mock_tokens():
            yield "Answer"

        class PublicOnlyManager:
            def __init__(self) -> None:
                self.config = test_config
                self.aanswer_stream = AsyncMock(return_value=({"chunks": []}, mock_tokens()))

        manager = PublicOnlyManager()
        web_app.state.manager = manager

        resp = await client.post("/web/answer", json={"query": "hello"})

        assert resp.status_code == 200
        assert "event: done" in resp.text
        assert '"answer": "Answer"' in resp.text
        assert "Service error" not in resp.text
        manager.aanswer_stream.assert_awaited_once()

    async def test_answer_stream_reads_session_images_through_public_manager_api(
        self,
        client: AsyncClient,
        test_config: DlightragConfig,
        web_app,
    ) -> None:
        class TokenStream:
            answer = "Answer"
            current_image_ids = ["img_0"]
            image_descriptions = ["Uploaded diagram"]

            def __aiter__(self):
                return self

            async def __anext__(self):
                raise StopAsyncIteration

        class PublicOnlyManager:
            def __init__(self) -> None:
                self.config = test_config
                self.aanswer_stream = AsyncMock(return_value=({"chunks": []}, TokenStream()))
                self.get_session_image_data = AsyncMock(return_value=["data:image/png;base64,abc"])

        manager = PublicOnlyManager()
        web_app.state.manager = manager

        resp = await client.post(
            "/web/answer",
            json={"query": "hello", "session_id": "session-1"},
        )

        assert resp.status_code == 200
        assert "event: done" in resp.text
        assert "Uploaded diagram" in resp.text
        manager.get_session_image_data.assert_awaited_once()
        args = manager.get_session_image_data.await_args
        assert args is not None
        assert args.args == ("session-1", ["img_0"])
        assert args.kwargs["scope"].session_key("session-1")

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
            nonlocal highlights_called
            highlights_called = True
            assert llm_func is keyword_llm
            return sources

        def fail_query_model(*args, **kwargs):
            raise AssertionError("highlighter must not use query role")

        async def mock_tokens():
            yield "Evidence [1-1]."

        class PublicOnlyManager:
            def __init__(self) -> None:
                self.config = test_config
                self.aanswer_stream = AsyncMock(
                    return_value=(
                        {
                            "chunks": [
                                {
                                    "chunk_id": "c1",
                                    "reference_id": "1",
                                    "content": "Evidence in cited chunk.",
                                    "file_path": "/docs/report.pdf",
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
            "dlightrag.web.answer_events.extract_highlights_for_sources",
            fake_extract_highlights_for_sources,
        )

        resp = await client.post("/web/answer", json={"query": "hello"})

        assert resp.status_code == 200
        assert "event: highlights" in resp.text
        assert keyword_called is True
        assert highlights_called is True


class TestWebSSEBoundary:
    """Tests for SSE framing and browser HTML fragment safety."""

    def test_sse_event_json_encodes_payload(self) -> None:
        from dlightrag.web.events import AnswerDoneEvent
        from dlightrag.web.sse import sse_event

        event = sse_event("done", AnswerDoneEvent(html="<b>x</b>", answer="x"))

        assert event.startswith("event: done\n")
        assert event.endswith("\n\n")
        data_line = next(line for line in event.splitlines() if line.startswith("data: "))
        assert json.loads(data_line.removeprefix("data: ")) == {
            "html": "<b>x</b>",
            "answer": "x",
            "current_image_ids": [],
            "image_descriptions": [],
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
                self.aanswer_stream = AsyncMock(
                    return_value=(
                        {
                            "chunks": [
                                {
                                    "chunk_id": "c1",
                                    "reference_id": "1",
                                    "content": "Evidence in cited chunk.",
                                    "file_path": "/docs/report.pdf",
                                    "image_url": "javascript:alert(1)",
                                    "thumbnail_url": "javascript:alert(2)",
                                    "_answer_image_sent": True,
                                }
                            ]
                        },
                        mock_tokens(),
                    )
                )

        web_app.state.manager = PublicOnlyManager()

        resp = await client.post("/web/answer", json={"query": "hello"})

        assert resp.status_code == 200
        assert "event: done" in resp.text
        assert "answer-content" in resp.text
        assert "source-data" in resp.text
        assert "javascript:" not in resp.text
        assert "onerror" not in resp.text.lower()
        assert "<script" not in resp.text.lower()


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
        image_b64 = base64.b64encode(b"tiny-image").decode()

        resp = await client.post(
            "/web/answer",
            json={
                "query": "hello",
                "images": [image_b64],
                "workspaces": ["default", "test_ws"],
                "conversation_history": [{"role": "user", "content": "previous"}],
                "session_id": "session-1",
            },
        )

        assert resp.status_code == 200
        assert captured["manager"] is web_app.state.manager
        assert captured["cfg"] is test_config
        assert captured["query"] == "hello"
        assert captured["query_images"] == [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
        ]
        assert captured["workspaces"] == ["default", "test_ws"]
        assert captured["workspace"] == "default"
        assert captured["session_id"] == "session-1"
        assert captured["conversation_history"] == [{"role": "user", "content": "previous"}]

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
                "conversation_history": [{"role": "human", "content": "previous"}],
            },
        )

        assert resp.status_code == 422


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
        mock_manager.list_ingested_files = AsyncMock(
            return_value=[
                {"doc_id": "d1", "file_path": "/tmp/reports/q4.pdf", "status": "processed"}
            ]
        )

        resp = await client.get("/web/files")

        assert resp.status_code == 200
        assert ">q4.pdf</span>" in resp.text
        assert 'title="/tmp/reports/q4.pdf"' in resp.text

    async def test_file_list_rejects_stale_workspace(
        self, client: AsyncClient, test_config: DlightragConfig, mock_manager
    ) -> None:
        mock_manager.list_workspaces = AsyncMock(return_value=["default"])

        resp = await client.get("/web/files", params={"workspace": "deleted_ws"})

        assert resp.status_code == 409
        assert "Workspace no longer exists" in resp.text
        mock_manager.list_ingested_files.assert_not_awaited()
        mock_manager.get_pipeline_status.assert_not_awaited()

    async def test_file_list_rejects_stale_workspace_even_with_registered_cookie(
        self, client: AsyncClient, test_config: DlightragConfig, mock_manager
    ) -> None:
        mock_manager.list_workspaces = AsyncMock(return_value=["default", "test_ws"])
        client.cookies.set("dlightrag_workspace", "test_ws")

        resp = await client.get("/web/files", params={"workspace": "deleted_ws"})

        assert resp.status_code == 409
        assert "Workspace no longer exists" in resp.text
        mock_manager.list_ingested_files.assert_not_awaited()
        mock_manager.get_pipeline_status.assert_not_awaited()

    async def test_file_list_canonicalizes_requested_workspace(
        self, client: AsyncClient, test_config: DlightragConfig, mock_manager
    ) -> None:
        mock_manager.list_workspaces = AsyncMock(return_value=["default", "test_fallback_ws"])

        resp = await client.get("/web/files", params={"workspace": "test-fallback-ws"})

        assert resp.status_code == 200
        mock_manager.list_ingested_files.assert_awaited_once_with("test_fallback_ws")
        mock_manager.get_pipeline_status.assert_awaited_once_with("test_fallback_ws")

    async def test_file_list_rejects_stale_workspace_without_default(
        self, client: AsyncClient, test_config: DlightragConfig, mock_manager
    ) -> None:
        mock_manager.list_workspaces = AsyncMock(return_value=["other_ws"])

        resp = await client.get("/web/files", params={"workspace": "deleted_ws"})

        assert resp.status_code == 409
        assert "Workspace no longer exists" in resp.text
        mock_manager.list_ingested_files.assert_not_awaited()

    async def test_ingest_status_rejects_stale_workspace(
        self, client: AsyncClient, test_config: DlightragConfig, mock_manager
    ) -> None:
        mock_manager.list_workspaces = AsyncMock(return_value=["default"])

        resp = await client.get("/web/ingest-status", params={"workspace": "deleted_ws"})

        assert resp.status_code == 409
        assert "Workspace no longer exists" in resp.text
        mock_manager.get_pipeline_status.assert_not_awaited()

    async def test_upload_preserves_filename_for_directory_ingest(
        self, client: AsyncClient, test_config: DlightragConfig, mock_manager
    ) -> None:
        resp = await client.post(
            "/web/files/upload",
            files=[("files", ("report.pdf", b"%PDF-fake", "application/pdf"))],
        )

        assert resp.status_code == 200
        mock_manager.aingest.assert_not_awaited()
        mock_manager.astart_ingest_job.assert_not_awaited()
        mock_manager._astart_staged_local_ingest_job.assert_awaited_once()
        call = mock_manager._astart_staged_local_ingest_job.await_args
        assert call.args == ("default",)
        upload_dir = Path(call.kwargs["path"])
        assert upload_dir.is_dir()
        assert (upload_dir / "report.pdf").read_bytes() == b"%PDF-fake"

    async def test_upload_rejects_stale_workspace(
        self, client: AsyncClient, test_config: DlightragConfig, mock_manager
    ) -> None:
        mock_manager.list_workspaces = AsyncMock(return_value=["default"])

        resp = await client.post(
            "/web/files/upload",
            data={"workspace": "deleted_ws"},
            files=[("files", ("report.pdf", b"%PDF-fake", "application/pdf"))],
        )

        assert resp.status_code == 409
        assert "Workspace no longer exists" in resp.text
        mock_manager.aingest.assert_not_awaited()
        mock_manager.astart_ingest_job.assert_not_awaited()
        mock_manager._astart_staged_local_ingest_job.assert_not_awaited()

    def test_safe_relative_path_rejects_absolute_paths(self) -> None:
        from dlightrag.web.routes.files import _safe_relative_path

        with pytest.raises(ValueError):
            _safe_relative_path("/tmp/evil.pdf")

    async def test_delete_files(
        self, client: AsyncClient, test_config: DlightragConfig, mock_manager
    ) -> None:
        resp = await client.request(
            "DELETE",
            "/web/files",
            params={"file_path": "/tmp/test.pdf"},
        )
        assert resp.status_code == 200
        mock_manager.delete_files.assert_awaited_once()

    async def test_delete_files_rejects_stale_workspace(
        self, client: AsyncClient, test_config: DlightragConfig, mock_manager
    ) -> None:
        mock_manager.list_workspaces = AsyncMock(return_value=["default"])

        resp = await client.request(
            "DELETE",
            "/web/files",
            params={"workspace": "deleted_ws", "file_path": "/tmp/test.pdf"},
        )

        assert resp.status_code == 409
        assert "Workspace no longer exists" in resp.text
        mock_manager.delete_files.assert_not_awaited()


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
        mock_manager.list_workspaces = AsyncMock(
            side_effect=[["default", "test_ws"], ["default", "test_ws", "new_workspace"]]
        )
        resp = await client.post(
            "/web/workspaces/create",
            data={"workspace_name": "new workspace"},
        )
        assert resp.status_code == 200
        assert "workspaceCreated" in resp.headers["hx-trigger"]
        assert "new_workspace" in resp.headers["set-cookie"]
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
        mock_manager.list_workspaces = AsyncMock(return_value=["default"])
        resp = await client.post(
            "/web/workspaces/delete",
            data={"workspace_name": "test-ws", "confirm_name": "test-ws"},
        )
        assert resp.status_code == 200
        assert "workspaceDeleted" in resp.headers["hx-trigger"]
        assert "dlightrag_workspace=default" in resp.headers["set-cookie"]
        mock_manager.areset.assert_awaited_once_with(workspace="test_ws")

    async def test_delete_hyphen_workspace_emits_canonical_workspace(
        self, client: AsyncClient, test_config: DlightragConfig, mock_manager
    ) -> None:
        mock_manager.areset = AsyncMock(return_value={"workspaces": {}, "total_errors": 0})
        mock_manager.list_workspaces = AsyncMock(return_value=["default"])

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
