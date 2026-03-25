# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for WebGUI route endpoints."""

from __future__ import annotations

from unittest.mock import AsyncMock

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


# ---------------------------------------------------------------------------
# TestWebWorkspaceCreateDelete
# ---------------------------------------------------------------------------


class TestWebWorkspaceCreate:
    """Tests for POST /web/workspaces/create."""

    async def test_create_workspace(
        self, client: AsyncClient, test_config: DlightragConfig, mock_manager
    ) -> None:
        mock_manager._get_service = AsyncMock()
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
        mock_manager._get_service.assert_awaited_once_with("new_workspace")

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
