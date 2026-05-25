# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for WebGUI route endpoints."""

from __future__ import annotations

from pathlib import Path
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
    manager.delete_files = AsyncMock(return_value=[])
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
                self.aanswer_stream = AsyncMock(
                    return_value=({"chunks": []}, mock_tokens())
                )

        manager = PublicOnlyManager()
        web_app.state.manager = manager

        resp = await client.post("/web/answer", json={"query": "hello"})

        assert resp.status_code == 200
        assert "event: done" in resp.text
        assert '"answer": "Answer"' in resp.text
        assert "Service error" not in resp.text
        manager.aanswer_stream.assert_awaited_once()

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
            "dlightrag.web.routes.chat.extract_highlights_for_sources",
            fake_extract_highlights_for_sources,
        )

        resp = await client.post("/web/answer", json={"query": "hello"})

        assert resp.status_code == 200
        assert "event: highlights" in resp.text
        assert keyword_called is True
        assert highlights_called is True


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
        assert "title=\"/tmp/reports/q4.pdf\"" in resp.text

    async def test_upload_preserves_filename_for_directory_ingest(
        self, client: AsyncClient, test_config: DlightragConfig, mock_manager
    ) -> None:
        async def ingest_upload_dir(workspace: str, source_type: str, **kwargs):
            upload_dir = Path(kwargs["path"])
            assert workspace == "default"
            assert source_type == "local"
            assert upload_dir.is_dir()
            assert (upload_dir / "report.pdf").read_bytes() == b"%PDF-fake"
            return {"processed": 1, "results": []}

        mock_manager.aingest = AsyncMock(side_effect=ingest_upload_dir)

        resp = await client.post(
            "/web/files/upload",
            files=[("files", ("report.pdf", b"%PDF-fake", "application/pdf"))],
        )

        assert resp.status_code == 200
        mock_manager.aingest.assert_awaited_once()

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
    """Tests for removed side-panel workspace routes."""

    async def test_workspace_side_panel_route_is_not_exposed(
        self, client: AsyncClient, test_config: DlightragConfig
    ) -> None:
        resp = await client.get("/web/workspaces")
        assert resp.status_code == 404

    async def test_cookie_workspace_switch_route_is_not_exposed(
        self, client: AsyncClient, test_config: DlightragConfig
    ) -> None:
        resp = await client.post(
            "/web/workspaces/switch",
            data={"workspace": "test-ws"},
        )
        assert resp.status_code == 404


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
    """Tests for removed ingest-progress frontend route."""

    async def test_progress_endpoint_is_not_exposed_without_frontend_consumer(
        self, client: AsyncClient, test_config: DlightragConfig, web_app
    ) -> None:
        resp = await client.get("/web/ingest/progress")
        assert resp.status_code == 404


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
