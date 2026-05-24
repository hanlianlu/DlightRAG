# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for WebGUI route endpoints."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from dlightrag.api.server import create_app
from dlightrag.config import DlightragConfig
from dlightrag.core.retrieval.protocols import RetrievalResult

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
    manager.list_workspaces = AsyncMock(return_value=["default", "test-ws"])
    manager.list_ingested_files = AsyncMock(
        return_value=[{"filename": "test.pdf", "file_path": "/tmp/test.pdf"}]
    )
    manager.delete_files = AsyncMock(return_value=[])
    manager.get_llm_func.return_value = AsyncMock(return_value="rewritten")
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
                self.aplan_query = AsyncMock(
                    return_value=MagicMock(original_query="hello", standalone_query="hello")
                )
                self.aretrieve = AsyncMock(
                    return_value=RetrievalResult(answer=None, contexts={"chunks": []})
                )
                self.agenerate_stream_from_contexts = AsyncMock(
                    return_value=({"chunks": []}, mock_tokens())
                )

        manager = PublicOnlyManager()
        web_app.state.manager = manager

        resp = await client.post("/web/answer", json={"query": "hello"})

        assert resp.status_code == 200
        assert "event: done" in resp.text
        assert '"answer": "Answer"' in resp.text
        assert "Service error" not in resp.text
        manager.aplan_query.assert_awaited_once()
        manager.agenerate_stream_from_contexts.assert_awaited_once()


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


# ---------------------------------------------------------------------------
# TestWebWorkspaces
# ---------------------------------------------------------------------------


class TestWebWorkspaces:
    """Tests for GET /web/workspaces and POST /web/workspaces/switch."""

    async def test_workspace_list(self, client: AsyncClient, test_config: DlightragConfig) -> None:
        resp = await client.get("/web/workspaces")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]

    async def test_switch_workspace_redirects(
        self, client: AsyncClient, test_config: DlightragConfig
    ) -> None:
        resp = await client.post(
            "/web/workspaces/switch",
            data={"workspace": "test-ws"},
        )
        assert resp.status_code in {302, 303, 307}
        assert resp.headers.get("location", "").endswith("/web/")

    async def test_switch_workspace_ignores_unknown_tampered_workspace(
        self, client: AsyncClient, test_config: DlightragConfig
    ) -> None:
        resp = await client.post(
            "/web/workspaces/switch",
            data={"workspace": "evil\r\nSet-Cookie: injected=1"},
        )
        cookie = resp.headers.get("set-cookie", "")

        assert resp.status_code == 303
        assert "dlightrag_workspace=default" in cookie
        assert "injected" not in cookie


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
        assert "text/html" in resp.headers["content-type"]
        mock_manager.acreate_workspace.assert_awaited_once_with("new_workspace")

    async def test_create_workspace_empty_name(
        self, client: AsyncClient, test_config: DlightragConfig
    ) -> None:
        resp = await client.post(
            "/web/workspaces/create",
            data={"workspace_name": ""},
        )
        assert resp.status_code == 400

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
        mock_manager.areset.assert_awaited_once_with(workspace="test_ws")

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


# ---------------------------------------------------------------------------
# TestIngestProgress
# ---------------------------------------------------------------------------


class TestIngestProgress:
    """Tests for GET /web/ingest/progress SSE endpoint."""

    async def test_no_task_manager_returns_error(
        self, client: AsyncClient, test_config: DlightragConfig, web_app
    ) -> None:
        # Ensure no ingest_task_manager is set
        if hasattr(web_app.state, "ingest_task_manager"):
            del web_app.state.ingest_task_manager
        resp = await client.get("/web/ingest/progress")
        assert resp.status_code == 200
        assert "error" in resp.text
