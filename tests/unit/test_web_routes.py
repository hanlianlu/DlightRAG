# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for WebGUI route endpoints."""

import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock

import jwt
import pytest
from httpx import ASGITransport, AsyncClient

from dlightrag.answer.capability import AnswerImageCapability
from dlightrag.api.server import create_app
from dlightrag.config import DlightragConfig
from dlightrag.web.attachment_models import SUPPORTED_DOCUMENT_EXTENSIONS
from tests.unit.conftest import answer_capability_view

if TYPE_CHECKING:
    from dlightrag.application import Application


def _fake_application(**attrs: object) -> Application:
    return cast("Application", SimpleNamespace(**attrs))


CONVERSATION_ID = "11111111-1111-4111-8111-111111111111"
SUBMISSION_ID = "22222222-2222-4222-8222-222222222222"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_application():
    """Create an Application-shaped Web route test double."""
    application_double = AsyncMock()
    capability_view = answer_capability_view(
        AnswerImageCapability(
            status="supported",
            configured_ceiling=8,
            effective_max_images=8,
            provider="test",
            base_url=None,
            model="test-model",
            failure_kind=None,
        )
    )
    application_double.answers = SimpleNamespace(capabilities=capability_view.read)
    corpora = SimpleNamespace()
    corpora.list_workspaces = AsyncMock(return_value=["default", "test_ws"])
    corpora.alist_workspace_records = AsyncMock(
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
    corpora.list_ingested_files = AsyncMock(
        return_value=[{"filename": "test.pdf", "file_path": "/tmp/test.pdf"}]
    )
    corpora.get_pipeline_status = AsyncMock(
        return_value={"busy": False, "pending_enqueues": 0, "latest_message": ""}
    )
    corpora.file_panel_snapshot = AsyncMock(
        return_value={
            "files": [{"filename": "test.pdf", "file_path": "/tmp/test.pdf"}],
            "pipeline_status": {"busy": False, "pending_enqueues": 0, "latest_message": ""},
        }
    )
    corpora.delete_files = AsyncMock(return_value=[])
    corpora.start_ingest_job = AsyncMock(return_value={"job_id": "job-1", "status": "queued"})
    corpora.prepare_source_download = AsyncMock()
    corpora.get_visual_asset = AsyncMock()
    corpora.create_workspace = AsyncMock()
    corpora.reset = AsyncMock(return_value={"workspaces": {}, "total_errors": 0})
    application_double.corpora = corpora
    return application_double


@pytest.fixture
def web_app(mock_application, test_config: DlightragConfig):
    """Create the FastAPI app with its Application-shaped double installed."""
    application = create_app(include_web_app=True)
    mock_application.config = test_config
    application.state.application = mock_application
    conversation_service = AsyncMock()
    mock_application.web_conversations = conversation_service
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
    from dlightrag.application import Application

    application_double = AsyncMock()
    conversation_service = AsyncMock()
    application = create_app(include_web_app=True)
    application_double.health = MagicMock()
    application_double.web_conversations = conversation_service
    monkeypatch.setattr(Application, "acreate", AsyncMock(return_value=application_double))

    async with application.router.lifespan_context(application):
        assert application.state.application.web_conversations is conversation_service

    conversation_service.aclose.assert_not_awaited()
    application_double.aclose.assert_awaited_once_with()


async def test_vite_hashed_assets_are_immutable(client):
    from dlightrag.web.static_files import APP_DIR

    asset = next((APP_DIR / "assets").glob("app-*.js"))
    response = await client.get(f"/static/app/assets/{asset.name}")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "public, max-age=31536000, immutable"


async def test_vendored_assets_allow_revalidation_caching(client):
    resp = await client.get("/static/vendor/mathjax/tex-mml-svg.js")

    assert resp.status_code == 200
    # Immutable vendored assets are not marked no-store, so the browser can
    # revalidate (304) instead of re-downloading the multi-MB MathJax payload.
    assert "no-store" not in resp.headers.get("cache-control", "")


def _configure_web_application(application_double, cfg: DlightragConfig):
    application_double.config = cfg
    application_double.corpora.get_pipeline_status = AsyncMock(
        return_value={"busy": False, "pending_enqueues": 0, "latest_message": ""}
    )
    return application_double


def _web_client_for(cfg: DlightragConfig, application_double):
    application = create_app(include_web_app=True)
    application.state.application = _configure_web_application(application_double, cfg)
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
        self, test_config: DlightragConfig, mock_application
    ) -> None:
        test_config.auth_mode = "simple"
        test_config.api_auth_token = "secret-token"

        async with _web_client_for(test_config, mock_application) as c:
            resp = await c.get("/web/")

        assert resp.status_code == 303
        assert resp.headers["location"].startswith("/web/login")

    async def test_conversation_route_login_redirect_preserves_deep_link(
        self, test_config: DlightragConfig, mock_application
    ) -> None:
        from urllib.parse import parse_qs, urlsplit

        test_config.auth_mode = "simple"
        test_config.api_auth_token = "secret-token"
        path = f"/web/conversations/{CONVERSATION_ID}"

        async with _web_client_for(test_config, mock_application) as client:
            response = await client.get(path)

        assert response.status_code == 303
        query = parse_qs(urlsplit(response.headers["location"]).query)
        assert query["next"] == [path]

    async def test_source_download_login_redirect_preserves_workspace(
        self, test_config: DlightragConfig, mock_application
    ) -> None:
        from urllib.parse import parse_qs, urlsplit

        test_config.auth_mode = "simple"
        test_config.api_auth_token = "secret-token"

        async with _web_client_for(test_config, mock_application) as client:
            response = await client.get(
                "/web/api/files/raw/doc-report",
                params={"workspace": "finance"},
            )

        assert response.status_code == 303
        query = parse_qs(urlsplit(response.headers["location"]).query)
        assert query["next"] == ["/web/api/files/raw/doc-report?workspace=finance"]

    async def test_simple_invalid_bearer_rejected(
        self, test_config: DlightragConfig, mock_application
    ) -> None:
        test_config.auth_mode = "simple"
        test_config.api_auth_token = "secret-token"

        async with _web_client_for(test_config, mock_application) as c:
            resp = await c.get(
                "/web/api/files",
                headers={"Authorization": "Bearer wrong-token"},
            )

        assert resp.status_code == 401

    async def test_simple_login_page_is_static_and_no_store(
        self, test_config: DlightragConfig, mock_application
    ) -> None:
        test_config.auth_mode = "simple"
        test_config.api_auth_token = "secret-token"

        async with _web_client_for(test_config, mock_application) as client:
            response = await client.get(
                "/web/login",
                params={"next": f"/web/conversations/{CONVERSATION_ID}"},
            )

        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-cache, no-store, must-revalidate"
        assert 'action="/web/login"' in response.text
        assert "/static/app/assets/login-" in response.text
        assert "secret-token" not in response.text

    async def test_invalid_paste_token_redirects_to_generic_static_error(
        self, test_config: DlightragConfig, mock_application
    ) -> None:
        from urllib.parse import parse_qs, urlsplit

        test_config.auth_mode = "simple"
        test_config.api_auth_token = "secret-token"
        target = f"/web/conversations/{CONVERSATION_ID}"

        async with _web_client_for(test_config, mock_application) as client:
            response = await client.post(
                "/web/login",
                data={"token": "wrong-token", "next": target},
            )

        assert response.status_code == 303
        query = parse_qs(urlsplit(response.headers["location"]).query)
        assert query == {"next": [target], "error": ["Authentication failed"]}

    async def test_simple_login_sets_cookie_and_grants_access(
        self, test_config: DlightragConfig, mock_application
    ) -> None:
        test_config.auth_mode = "simple"
        test_config.api_auth_token = "secret-token"

        async with _web_client_for(test_config, mock_application) as c:
            login = await c.post(
                "/web/login",
                data={"token": "secret-token", "next": "/web/"},
            )
            resp = await c.get("/web/")

        assert login.status_code == 303
        assert "dlightrag_web_auth=" in login.headers["set-cookie"]
        assert resp.status_code == 200

    async def test_simple_login_cookie_downloads_source_without_bearer(
        self, test_config: DlightragConfig, mock_application
    ) -> None:
        from dlightrag.services.errors import LocalDownloadTarget

        test_config.auth_mode = "simple"
        test_config.api_auth_token = "secret-token"
        source = test_config.input_dir_path / "default" / "notes.md"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("downloadable notes", encoding="utf-8")
        mock_application.corpora.prepare_source_download.return_value = LocalDownloadTarget(
            path=source.resolve(),
            media_type="text/markdown",
            filename="notes.md",
        )

        async with _web_client_for(test_config, mock_application) as c:
            await c.post(
                "/web/login",
                data={"token": "secret-token", "next": "/web/"},
            )
            response = await c.get(
                "/web/api/files/raw/doc-notes",
                params={"workspace": "default"},
            )
            rest_response = await c.get(
                "/files/raw/doc-notes",
                params={"workspace": "default"},
            )

        assert response.status_code == 200
        assert response.content == b"downloadable notes"
        assert rest_response.status_code == 401
        mock_application.corpora.prepare_source_download.assert_awaited_once_with(
            "default", "doc-notes"
        )

    async def test_login_redirect_rejects_external_next(
        self, test_config: DlightragConfig, mock_application
    ) -> None:
        test_config.auth_mode = "simple"
        test_config.api_auth_token = "secret-token"

        async with _web_client_for(test_config, mock_application) as c:
            resp = await c.post(
                "/web/login",
                data={"token": "secret-token", "next": "https://evil.example/"},
            )

        assert resp.status_code == 303
        assert resp.headers["location"] == "/web/"

    async def test_invalid_auth_cookie_is_cleared(
        self, test_config: DlightragConfig, mock_application
    ) -> None:
        test_config.auth_mode = "simple"
        test_config.api_auth_token = "secret-token"

        async with _web_client_for(test_config, mock_application) as c:
            c.cookies.set("dlightrag_web_auth", "not base64!")
            resp = await c.get("/web/")

        assert resp.status_code == 303
        assert resp.headers["location"].startswith("/web/login")
        assert "dlightrag_web_auth=" in resp.headers["set-cookie"]

    async def test_bearer_header_grants_web_access(
        self, test_config: DlightragConfig, mock_application
    ) -> None:
        test_config.auth_mode = "simple"
        test_config.api_auth_token = "secret-token"

        async with _web_client_for(test_config, mock_application) as c:
            resp = await c.get(
                "/web/api/files",
                headers={"Authorization": "Bearer secret-token"},
            )

        assert resp.status_code == 200

    async def test_jwt_invalid_bearer_rejected(
        self, test_config: DlightragConfig, mock_application
    ) -> None:
        test_config.auth_mode = "jwt"
        test_config.jwt_verification_key = "test-jwt-verification-key-for-web-route-tests"

        async with _web_client_for(test_config, mock_application) as c:
            resp = await c.get(
                "/web/api/files",
                headers={"Authorization": "Bearer not-a-jwt"},
            )

        assert resp.status_code == 401

    async def test_jwt_bearer_header_grants_web_access(
        self, test_config: DlightragConfig, mock_application
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

        async with _web_client_for(test_config, mock_application) as c:
            resp = await c.get(
                "/web/api/files",
                headers={"Authorization": f"Bearer {token}"},
            )

        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# TestWebIndex
# ---------------------------------------------------------------------------


class TestWebIndex:
    """Tests for the Vite-owned application document."""

    async def test_returns_no_store_vite_html(self, client: AsyncClient) -> None:
        response = await client.get("/web/")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert response.headers["cache-control"] == "no-cache, no-store, must-revalidate"
        assert "<dl-app>" in response.text
        assert "/static/app/assets/app-" in response.text
        assert "__THEME_INIT__" not in response.text

    async def test_explicit_conversation_route_serves_the_same_application_document(
        self, client: AsyncClient
    ) -> None:
        index = await client.get("/web/")
        conversation = await client.get(f"/web/conversations/{CONVERSATION_ID}")

        assert conversation.status_code == 200
        assert conversation.text == index.text

    async def test_unknown_web_page_does_not_fall_through_to_the_shell(
        self, client: AsyncClient
    ) -> None:
        response = await client.get("/web/not-a-page")

        assert response.status_code == 404

    def test_vite_app_source_keeps_behavior_out_of_static_html(self) -> None:
        frontend = Path(__file__).parents[2] / "frontend"
        checked = [frontend / "index.html", frontend / "login.html"]

        offenders: list[str] = []
        for path in checked:
            text = path.read_text()
            for marker in ("onclick=", "onchange=", "style="):
                if marker in text:
                    offenders.append(f"{path.name}:{marker}")

        assert offenders == []


# ---------------------------------------------------------------------------
# TestWebBootstrap
# ---------------------------------------------------------------------------


class TestWebBootstrap:
    async def test_returns_one_typed_authorized_startup_snapshot(
        self, client: AsyncClient, test_config: DlightragConfig
    ) -> None:
        response = await client.get("/web/api/bootstrap")

        assert response.status_code == 200
        assert response.json() == {
            "contract_version": 1,
            "workspaces": [
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
            ],
            "primary_workspace": "default",
            "active_workspaces": ["default", "test_ws"],
            "answer_attachments": {
                "count_limit": 6,
                "image_max_bytes": 104_857_600,
                "document_max_bytes": 104_857_600,
                "extensions": sorted(SUPPORTED_DOCUMENT_EXTENSIONS),
                "image_capability": "supported",
                "image_limit": 8,
                "accept": ",".join(
                    [
                        "image/*",
                        *(f".{extension}" for extension in sorted(SUPPORTED_DOCUMENT_EXTENSIONS)),
                    ]
                ),
            },
        }

    async def test_filters_saved_scope_and_primary_through_authorized_workspaces(
        self, client: AsyncClient
    ) -> None:
        client.cookies.set("dlightrag_workspace", "deleted")
        client.cookies.set("dlightrag_workspace_ids", "test_ws,deleted")

        response = await client.get("/web/api/bootstrap")

        assert response.status_code == 200
        assert response.json()["primary_workspace"] == "default"
        assert response.json()["active_workspaces"] == ["test_ws"]

    async def test_machine_snapshot_fails_closed_when_workspace_inventory_is_unavailable(
        self, client: AsyncClient, mock_application
    ) -> None:
        mock_application.corpora.alist_workspace_records.side_effect = RuntimeError("database down")

        bootstrap = await client.get("/web/api/bootstrap")
        app_page = await client.get("/web/")

        assert bootstrap.status_code == 503
        assert bootstrap.json() == {
            "detail": "Web application bootstrap is unavailable",
            "error_type": "unavailable",
        }
        assert app_page.status_code == 200
        assert "<dl-app>" in app_page.text

    @pytest.mark.parametrize(
        "old_path",
        [
            "/web/answer",
            "/web/conversations",
            "/web/files",
            "/web/ingest-status",
            "/web/workspaces/create",
        ],
    )
    async def test_old_browser_data_paths_have_no_compatibility_alias(
        self, client: AsyncClient, old_path: str
    ) -> None:
        response = await client.get(old_path)

        assert response.status_code == 404


# ---------------------------------------------------------------------------
# TestWebFiles
# ---------------------------------------------------------------------------


class TestWebFiles:
    """Tests for GET /web/api/files and DELETE /web/api/files."""

    async def test_file_list_returns_typed_json(
        self, client: AsyncClient, test_config: DlightragConfig
    ) -> None:
        resp = await client.get("/web/api/files")

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/json")
        assert resp.json()["workspace"] == "default"
        assert resp.json()["files"] == [{"file_name": "test.pdf", "file_path": "/tmp/test.pdf"}]
        assert resp.json()["ingest"]["busy"] is False

    async def test_file_list_fails_closed_when_snapshot_is_unavailable(
        self, client: AsyncClient, mock_application
    ) -> None:
        mock_application.corpora.file_panel_snapshot.side_effect = RuntimeError("database down")

        response = await client.get("/web/api/files")

        assert response.status_code == 503
        assert response.json() == {
            "detail": "Files are temporarily unavailable",
            "error_type": "unavailable",
        }

    async def test_file_list_derives_display_name_from_path(
        self, client: AsyncClient, test_config: DlightragConfig, mock_application
    ) -> None:
        mock_application.corpora.file_panel_snapshot = AsyncMock(
            return_value={
                "files": [
                    {"doc_id": "d1", "file_path": "/tmp/reports/q4.pdf", "status": "processed"}
                ],
                "pipeline_status": {"busy": False, "pending_enqueues": 0},
            }
        )

        resp = await client.get("/web/api/files")

        assert resp.status_code == 200
        assert resp.json()["files"] == [{"file_name": "q4.pdf", "file_path": "/tmp/reports/q4.pdf"}]

    async def test_file_list_uses_file_panel_snapshot_for_cold_workspace(
        self, client: AsyncClient, test_config: DlightragConfig, mock_application
    ) -> None:
        mock_application.corpora.list_workspaces = AsyncMock(return_value=["default", "cold_ws"])
        mock_application.corpora.file_panel_snapshot = AsyncMock(
            return_value={
                "files": [
                    {"doc_id": "d1", "file_path": "/tmp/cold/report.pdf", "status": "processed"}
                ],
                "pipeline_status": {"busy": False, "pending_enqueues": 0},
            }
        )
        mock_application.corpora.list_ingested_files = AsyncMock(return_value=[])
        mock_application.corpora.get_pipeline_status = AsyncMock(return_value={"busy": False})

        resp = await client.get("/web/api/files", params={"workspace": "cold-ws"})

        assert resp.status_code == 200
        assert resp.json()["files"] == [
            {"file_name": "report.pdf", "file_path": "/tmp/cold/report.pdf"}
        ]
        mock_application.corpora.file_panel_snapshot.assert_awaited_once_with("cold_ws")
        mock_application.corpora.list_ingested_files.assert_not_awaited()
        mock_application.corpora.get_pipeline_status.assert_not_awaited()

    async def test_file_list_rejects_stale_workspace(
        self, client: AsyncClient, test_config: DlightragConfig, mock_application
    ) -> None:
        mock_application.corpora.list_workspaces = AsyncMock(return_value=["default"])

        resp = await client.get("/web/api/files", params={"workspace": "deleted_ws"})

        assert resp.status_code == 409
        assert "Workspace no longer exists" in resp.text
        mock_application.corpora.file_panel_snapshot.assert_not_awaited()
        mock_application.corpora.list_ingested_files.assert_not_awaited()
        mock_application.corpora.get_pipeline_status.assert_not_awaited()

    async def test_file_list_rejects_stale_workspace_even_with_registered_cookie(
        self, client: AsyncClient, test_config: DlightragConfig, mock_application
    ) -> None:
        mock_application.corpora.list_workspaces = AsyncMock(return_value=["default", "test_ws"])
        client.cookies.set("dlightrag_workspace", "test_ws")

        resp = await client.get("/web/api/files", params={"workspace": "deleted_ws"})

        assert resp.status_code == 409
        assert "Workspace no longer exists" in resp.text
        mock_application.corpora.file_panel_snapshot.assert_not_awaited()
        mock_application.corpora.list_ingested_files.assert_not_awaited()
        mock_application.corpora.get_pipeline_status.assert_not_awaited()

    async def test_file_list_canonicalizes_requested_workspace(
        self, client: AsyncClient, test_config: DlightragConfig, mock_application
    ) -> None:
        mock_application.corpora.list_workspaces = AsyncMock(
            return_value=["default", "test_fallback_ws"]
        )

        resp = await client.get("/web/api/files", params={"workspace": "test-fallback-ws"})

        assert resp.status_code == 200
        mock_application.corpora.file_panel_snapshot.assert_awaited_once_with("test_fallback_ws")

    async def test_file_list_rejects_stale_workspace_without_default(
        self, client: AsyncClient, test_config: DlightragConfig, mock_application
    ) -> None:
        mock_application.corpora.list_workspaces = AsyncMock(return_value=["other_ws"])

        resp = await client.get("/web/api/files", params={"workspace": "deleted_ws"})

        assert resp.status_code == 409
        assert "Workspace no longer exists" in resp.text
        mock_application.corpora.file_panel_snapshot.assert_not_awaited()
        mock_application.corpora.list_ingested_files.assert_not_awaited()

    async def test_ingest_status_rejects_stale_workspace(
        self, client: AsyncClient, test_config: DlightragConfig, mock_application
    ) -> None:
        mock_application.corpora.list_workspaces = AsyncMock(return_value=["default"])

        resp = await client.get("/web/api/ingest-status", params={"workspace": "deleted_ws"})

        assert resp.status_code == 409
        assert "Workspace no longer exists" in resp.text
        mock_application.corpora.get_pipeline_status.assert_not_awaited()

    async def test_ingest_status_returns_typed_idle_state(
        self, client: AsyncClient, test_config: DlightragConfig
    ) -> None:
        resp = await client.get("/web/api/ingest-status", params={"workspace": "default"})

        assert resp.status_code == 200
        assert resp.json() == {
            "busy": False,
            "message": "",
            "progress_percent": None,
            "current_batch": None,
            "total_batches": None,
            "documents": None,
            "pending_enqueues": 0,
        }
        assert "hx-retarget" not in resp.headers
        assert "hx-reswap" not in resp.headers

    async def test_ingest_status_normalizes_progress_and_queue(
        self, client: AsyncClient, mock_application
    ) -> None:
        mock_application.corpora.get_pipeline_status = AsyncMock(
            return_value={
                "busy": True,
                "latest_message": "Embedding",
                "docs": 9,
                "batchs": 4,
                "cur_batch": 2,
                "pending_enqueues": 3,
            }
        )

        response = await client.get("/web/api/ingest-status", params={"workspace": "default"})

        assert response.status_code == 200
        assert response.json() == {
            "busy": True,
            "message": "Embedding",
            "progress_percent": 50,
            "current_batch": 2,
            "total_batches": 4,
            "documents": 9,
            "pending_enqueues": 3,
        }

    async def test_upload_preserves_filename_for_directory_ingest(
        self, client: AsyncClient, test_config: DlightragConfig, mock_application, tmp_path: Path
    ) -> None:
        upload_dir = tmp_path / "uploads"
        upload_dir.mkdir()
        saved = upload_dir / "report.pdf"
        saved.write_bytes(b"%PDF-fake")

        async def fake_stage_batch(workspace, files, *, per_file_max_bytes, batch_max_bytes):
            del workspace, files, per_file_max_bytes, batch_max_bytes
            return upload_dir, [saved]

        mock_application.corpora.stage_upload_batch = fake_stage_batch
        resp = await client.post(
            "/web/api/files/upload",
            files=[("files", ("report.pdf", b"%PDF-fake", "application/pdf"))],
        )

        assert resp.status_code == 200
        assert resp.json()["workspace"] == "default"
        assert resp.json()["file_count"] == 1
        assert resp.json()["queued"] is False
        assert resp.json()["ingest"] == {
            "busy": True,
            "message": "Starting ingest...",
            "progress_percent": None,
            "current_batch": None,
            "total_batches": None,
            "documents": None,
            "pending_enqueues": 0,
        }
        mock_application.corpora.start_ingest_job.assert_awaited_once()
        call = mock_application.corpora.start_ingest_job.await_args
        assert call.args[0] == "default"
        ingest_spec = call.args[1]
        assert ingest_spec.source_type == "local"
        upload_dir = Path(ingest_spec.path)
        assert upload_dir.is_dir()
        assert (upload_dir / "report.pdf").read_bytes() == b"%PDF-fake"

    async def test_upload_rejects_stale_workspace(
        self, client: AsyncClient, test_config: DlightragConfig, mock_application
    ) -> None:
        mock_application.corpora.list_workspaces = AsyncMock(return_value=["default"])

        resp = await client.post(
            "/web/api/files/upload",
            data={"workspace": "deleted_ws"},
            files=[("files", ("report.pdf", b"%PDF-fake", "application/pdf"))],
        )

        assert resp.status_code == 409
        assert "Workspace no longer exists" in resp.text
        mock_application.corpora.start_ingest_job.assert_not_awaited()

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
        from dlightrag_rag.ingestion.uploads import safe_upload_relative_path

        with pytest.raises(ValueError):
            safe_upload_relative_path(filename)

    async def test_delete_files(
        self, client: AsyncClient, test_config: DlightragConfig, mock_application
    ) -> None:
        resp = await client.request(
            "DELETE",
            "/web/api/files",
            params={"file_path": "/tmp/test.pdf"},
        )
        assert resp.status_code == 200
        assert resp.json()["workspace"] == "default"
        assert resp.json()["files"] == [{"file_name": "test.pdf", "file_path": "/tmp/test.pdf"}]
        mock_application.corpora.delete_files.assert_awaited_once()

    async def test_delete_files_rejects_stale_workspace(
        self, client: AsyncClient, test_config: DlightragConfig, mock_application
    ) -> None:
        mock_application.corpora.list_workspaces = AsyncMock(return_value=["default"])

        resp = await client.request(
            "DELETE",
            "/web/api/files",
            params={"workspace": "deleted_ws", "file_path": "/tmp/test.pdf"},
        )

        assert resp.status_code == 409
        assert "Workspace no longer exists" in resp.text
        mock_application.corpora.delete_files.assert_not_awaited()


# ---------------------------------------------------------------------------
# TestWebWorkspaceCreateDelete
# ---------------------------------------------------------------------------


class TestWebWorkspaceCreate:
    """Tests for POST /web/api/workspaces/create."""

    async def test_create_workspace(
        self, client: AsyncClient, test_config: DlightragConfig, mock_application
    ) -> None:
        mock_application.corpora.create_workspace = AsyncMock()
        # First call (duplicate check): workspace does not exist yet
        # Second call (post-create list): includes the new workspace
        mock_application.corpora.list_workspaces = AsyncMock(
            side_effect=[["default", "test_ws"], ["default", "test_ws", "new_workspace"]]
        )
        resp = await client.post(
            "/web/api/workspaces/create",
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
        mock_application.corpora.create_workspace.assert_awaited_once_with(
            "new_workspace",
            display_name="new workspace",
        )

    async def test_create_workspace_duplicate(
        self, client: AsyncClient, test_config: DlightragConfig, mock_application
    ) -> None:
        resp = await client.post(
            "/web/api/workspaces/create",
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
            "/web/api/workspaces/create",
            data={"workspace_name": workspace_name},
        )
        assert resp.status_code == 400
        assert resp.json()["error"]


class TestWebWorkspaceDelete:
    """Tests for POST /web/api/workspaces/delete."""

    async def test_delete_workspace(
        self, client: AsyncClient, test_config: DlightragConfig, mock_application
    ) -> None:
        mock_application.corpora.reset = AsyncMock(
            return_value={"workspaces": {}, "total_errors": 0}
        )
        mock_application.corpora.list_workspaces = AsyncMock(return_value=["default"])
        resp = await client.post(
            "/web/api/workspaces/delete",
            data={"workspace_name": "test-ws", "confirm_name": "test-ws"},
        )
        assert resp.status_code == 200
        assert resp.json() == {"workspace": "test_ws", "next_workspace": "default"}
        assert "dlightrag_workspace=default" in resp.headers["set-cookie"]
        mock_application.corpora.reset.assert_awaited_once_with(workspace_ids=("test_ws",))

    async def test_delete_default_workspace_selects_first_remaining_workspace(
        self, client: AsyncClient, test_config: DlightragConfig, mock_application
    ) -> None:
        mock_application.corpora.reset = AsyncMock(
            return_value={"workspaces": {}, "total_errors": 0}
        )
        mock_application.corpora.list_workspaces = AsyncMock(return_value=["research"])

        resp = await client.post(
            "/web/api/workspaces/delete",
            data={"workspace_name": "default", "confirm_name": "default"},
        )

        assert resp.status_code == 200
        assert resp.json() == {"workspace": "default", "next_workspace": "research"}
        set_cookies = resp.headers.get_list("set-cookie")
        assert any(cookie.startswith("dlightrag_workspace=research;") for cookie in set_cookies)
        assert any(cookie.startswith("dlightrag_workspace_ids=research;") for cookie in set_cookies)

    async def test_delete_hyphen_workspace_emits_canonical_workspace(
        self, client: AsyncClient, test_config: DlightragConfig, mock_application
    ) -> None:
        mock_application.corpora.reset = AsyncMock(
            return_value={"workspaces": {}, "total_errors": 0}
        )
        mock_application.corpora.list_workspaces = AsyncMock(return_value=["default"])

        resp = await client.post(
            "/web/api/workspaces/delete",
            data={"workspace_name": "test-fallback-ws", "confirm_name": "test-fallback-ws"},
        )

        assert resp.status_code == 200
        assert resp.json() == {"workspace": "test_fallback_ws", "next_workspace": "default"}
        mock_application.corpora.reset.assert_awaited_once_with(workspace_ids=("test_fallback_ws",))

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
            "/web/api/workspaces/delete",
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
                    "download_url": "/web/api/files/raw/doc-notes?workspace=default",
                    "chunks": [],
                }
            ]
        )

        assert "notes.md" in rendered
        assert 'href="/web/api/files/raw/doc-notes?workspace=default"' in rendered
        assert 'aria-label="Download source"' in rendered
