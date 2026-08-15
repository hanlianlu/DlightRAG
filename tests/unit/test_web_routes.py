# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for WebGUI route endpoints."""

import datetime
import html
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock

import jwt
import pytest
from httpx import ASGITransport, AsyncClient

from dlightrag.api.server import create_app
from dlightrag.config import DlightragConfig
from dlightrag.core.answer.capability import AnswerImageCapability
from dlightrag.web.attachment_models import SUPPORTED_DOCUMENT_EXTENSIONS

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
    application = create_app(include_web_app=True)
    mock_manager.config = test_config
    application.state.manager = mock_manager
    conversation_service = AsyncMock()
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
    application = create_app(include_web_app=True)
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


async def test_vendored_assets_allow_revalidation_caching(client):
    resp = await client.get("/static/vendor/mathjax/tex-mml-svg.js")

    assert resp.status_code == 200
    # Immutable vendored assets are not marked no-store, so the browser can
    # revalidate (304) instead of re-downloading the multi-MB MathJax payload.
    assert "no-store" not in resp.headers.get("cache-control", "")


def _configure_web_manager(manager, cfg: DlightragConfig):
    manager.config = cfg
    manager.aget_pipeline_status = AsyncMock(
        return_value={"busy": False, "pending_enqueues": 0, "latest_message": ""}
    )
    return manager


def _web_client_for(cfg: DlightragConfig, manager):
    application = create_app(include_web_app=True)
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

    async def test_index_projects_answer_attachment_byte_policy(
        self, client: AsyncClient, test_config: DlightragConfig, web_app
    ) -> None:
        test_config.answer.max_attachment_bytes = 12_345
        web_app.state.manager.config = test_config

        resp = await client.get("/web/")

        assert resp.status_code == 200
        assert 'data-attachment-image-max-bytes="12345"' in resp.text

    async def test_index_projects_supported_capability_effective_limit(
        self, client: AsyncClient, test_config: DlightragConfig, web_app
    ) -> None:
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
        assert 'data-attachment-image-capability="supported"' in resp.text
        assert 'data-attachment-image-limit="2"' in resp.text

    async def test_index_reprobes_unknown_capability_before_projecting_uploads(
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

        async def recover_capability() -> None:
            web_app.state.manager.answer_image_capability = AnswerImageCapability(
                status="supported",
                configured_ceiling=8,
                effective_max_images=2,
                provider="test",
                base_url=None,
                model="test-model",
                failure_kind=None,
            )

        web_app.state.manager._maybe_reprobe_answer_image_capability = AsyncMock(
            side_effect=recover_capability
        )

        resp = await client.get("/web/")

        assert resp.status_code == 200
        assert 'data-attachment-image-capability="supported"' in resp.text
        assert 'data-attachment-image-limit="2"' in resp.text
        web_app.state.manager._maybe_reprobe_answer_image_capability.assert_awaited_once()

    async def test_chat_template_projects_document_attachment_limits(
        self, client: AsyncClient
    ) -> None:
        resp = await client.get("/web/")

        assert resp.status_code == 200
        assert 'data-attachment-count-limit="6"' in resp.text
        assert 'data-attachment-document-max-bytes="104857600"' in resp.text
        extensions_match = re.search(r"data-attachment-extensions='([^']+)'", resp.text)
        accept_match = re.search(r'id="attachment-input"[^>]*accept="([^"]+)"', resp.text)
        assert extensions_match is not None
        assert accept_match is not None

        expected_extensions = sorted(SUPPORTED_DOCUMENT_EXTENSIONS)
        assert json.loads(html.unescape(extensions_match.group(1))) == expected_extensions
        assert accept_match.group(1) == ",".join(
            ["image/*", *(f".{extension}" for extension in expected_extensions)]
        )

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
        assert resp.json() == {"workspace": "new_workspace", "display_name": "new workspace"}
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

    async def test_create_workspace_duplicate(
        self, client: AsyncClient, test_config: DlightragConfig, mock_manager
    ) -> None:
        resp = await client.post(
            "/web/workspaces/create",
            data={"workspace_name": "default"},
        )
        assert resp.status_code == 409

    @pytest.mark.parametrize(
        "workspace_name",
        [
            pytest.param("", id="empty_name"),
            pytest.param("bad/name", id="forbidden_chars"),
            pytest.param("a" * 65, id="too_long"),
        ],
    )
    async def test_create_workspace_invalid_name(
        self,
        client: AsyncClient,
        test_config: DlightragConfig,
        workspace_name: str,
    ) -> None:
        resp = await client.post(
            "/web/workspaces/create",
            data={"workspace_name": workspace_name},
        )
        assert resp.status_code == 400
        assert resp.json()["error"]


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
        assert resp.json() == {"workspace": "test_ws", "next_workspace": "default"}
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
        assert resp.json() == {"workspace": "default", "next_workspace": "research"}
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
        assert resp.json() == {"workspace": "test_fallback_ws", "next_workspace": "default"}
        mock_manager.areset.assert_awaited_once_with(workspace="test_fallback_ws")

    @pytest.mark.parametrize(
        ("workspace_name", "confirm_name"),
        [
            pytest.param("test-ws", "wrong", id="confirm_mismatch"),
            pytest.param("", "", id="empty_name"),
        ],
    )
    async def test_delete_workspace_invalid(
        self,
        client: AsyncClient,
        test_config: DlightragConfig,
        workspace_name: str,
        confirm_name: str,
    ) -> None:
        resp = await client.post(
            "/web/workspaces/delete",
            data={"workspace_name": workspace_name, "confirm_name": confirm_name},
        )
        assert resp.status_code == 400


class TestSourcePanelTemplate:
    """Tests for source panel rendering contracts."""

    def test_page_number_is_rendered(self) -> None:
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
                            "page_number": 1,
                            "content": "first page",
                        }
                    ],
                }
            ]
        )

        assert "p.1" in html
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
