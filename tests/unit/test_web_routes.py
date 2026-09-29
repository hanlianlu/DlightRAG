# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for WebGUI route endpoints."""

import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import jwt
import pytest
from httpx import ASGITransport, AsyncClient

from dlightrag.adapters.http.browser.attachment_models import SUPPORTED_DOCUMENT_EXTENSIONS
from dlightrag.adapters.http.server import create_app
from dlightrag.application.access import DEPLOYMENT_OWNER_ID
from dlightrag.application.answer_runs.service import AgentEffortOffer
from dlightrag.application.config import DlightragConfig
from dlightrag.application.corpus_admin import (
    FilePanelCursor,
    FilePanelCursorCodec,
    FilePanelPageRequest,
    UploadLimits,
    WorkspaceCatalogCursor,
    WorkspaceCatalogCursorCodec,
    WorkspaceCatalogPage,
    WorkspaceNotFoundError,
)
from dlightrag.application.runs import RunAdmissionLimitExceededError
from dlightrag.engine.agent.skills import owner_skill_root
from dlightrag.engine.answer.image_capability import AnswerImageCapability
from tests.config_helpers import mutate_config
from tests.support.application_double import application_double
from tests.unit.conftest import answer_capabilities

CONVERSATION_ID = "11111111-1111-4111-8111-111111111111"
SUBMISSION_ID = "22222222-2222-4222-8222-222222222222"
BUILTIN_SKILL_COUNCIL = {
    "name": "council",
    "description": (
        "Council recipe for independent Child Session investigations and one curated "
        "cross-examination. Load when independent scrutiny would materially improve a "
        "contested, high-stakes, or multi-source answer, or when the user asks for "
        "independent critique. Skip ordinary factual or trivial questions. User veto, "
        "cancellation, and scope constraints win."
    ),
    "source": "builtin",
}
BUILTIN_SKILL_CREATOR = {
    "name": "skill-creator",
    "description": (
        "Guide the user through creating, improving, or removing personal DlightRAG skills. "
        "Use when the user wants to make, edit, or delete a skill "
        "(interview → draft → publish_skill)."
    ),
    "source": "builtin",
}

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_application(test_config: DlightragConfig):
    """Create an Application-shaped Web route test double."""
    application = application_double(test_config)
    application.answers.capabilities.return_value = answer_capabilities(
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
    # The bootstrap offers exactly what this deployment can apply.
    application.answers.agent_effort_offer.return_value = AgentEffortOffer(
        ("low", "high", "max"), None
    )
    corpora = application.corpora
    corpora.list_workspaces.return_value = ["default", "test_ws"]
    corpora.workspace_exists.return_value = True
    corpora.file_panel_cursor_codec = FilePanelCursorCodec(b"web-file-panel-test-secret")
    corpora.alist_workspace_records.return_value = [
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
    corpora.list_workspace_records_page.return_value = WorkspaceCatalogPage(
        items=(
            {
                "workspace": "default",
                "display_name": "Default",
                "embedding_model": "voyage-multimodal-3.5",
                "created_at": None,
                "updated_at": None,
            },
            {
                "workspace": "test_ws",
                "display_name": "Test Workspace",
                "embedding_model": "voyage-multimodal-3.5",
                "created_at": None,
                "updated_at": None,
            },
        ),
        next_cursor=None,
        fetched_rows=2,
    )
    corpora.workspace_catalog_cursor_codec = WorkspaceCatalogCursorCodec(
        b"web-workspace-catalog-test-secret"
    )
    corpora.file_panel_snapshot.return_value = {
        "files": [{"filename": "test.pdf", "file_path": "/tmp/test.pdf"}],
        "next_cursor": None,
        "fetched_rows": 1,
    }
    corpora.failed_file_snapshot.return_value = {
        "failed": [],
        "next_cursor": None,
        "fetched_rows": 0,
    }
    corpus_run = SimpleNamespace(
        run_id="0199a0a0-0000-7000-8000-0000000000bb",
        run_kind="corpus_mutation",
        lane="corpus_mutation",
        status="queued",
    )
    corpus_mutations = application.corpus_mutations
    for accept in (
        corpus_mutations.create_retry,
        corpus_mutations.create_staged_batch,
        corpus_mutations.create_delete,
        corpus_mutations.create_reset,
        corpus_mutations.create_workspace_delete,
    ):
        accept.return_value = SimpleNamespace(run=corpus_run)
    corpus_mutations.upload_limits = UploadLimits(
        file_bytes=100 * 1024 * 1024, request_bytes=512 * 1024 * 1024
    )
    application.runs.get.return_value = None
    application.runs.get_global.return_value = None
    application.runs.resume_repair.return_value = True
    return application


@pytest.fixture
def web_app(mock_application):
    """Create the FastAPI app with its Application-shaped double installed."""
    application = create_app(include_web_app=True)
    application.state.application = mock_application
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
    monkeypatch: pytest.MonkeyPatch, test_config: DlightragConfig
) -> None:
    from dlightrag.adapters.http import server as api_server

    application = application_double(test_config)
    conversation_service = application.web_conversations
    app = create_app(include_web_app=True)
    monkeypatch.setattr(
        api_server,
        "create_application",
        AsyncMock(return_value=application),
    )

    async with app.router.lifespan_context(app):
        assert app.state.application.web_conversations is conversation_service

    conversation_service.aclose.assert_not_awaited()
    application.aclose.assert_awaited_once_with()


async def test_skills_endpoint_merges_owner_skills(
    client, test_config: DlightragConfig, tmp_path: Path
) -> None:
    mutate_config(test_config, "answer.agent.skills_root", str(tmp_path / "global"))
    mutate_config(test_config, "answer.agent.owner_skills_root", str(tmp_path / "owners"))
    (tmp_path / "global" / "review").mkdir(parents=True)
    (tmp_path / "global" / "review" / "SKILL.md").write_text(
        "---\nname: review\ndescription: Global review.\n---\nbody",
        encoding="utf-8",
    )
    owner_dir = owner_skill_root(tmp_path / "owners", DEPLOYMENT_OWNER_ID)
    (owner_dir / "mine").mkdir(parents=True)
    (owner_dir / "mine" / "SKILL.md").write_text(
        "---\nname: mine\ndescription: My skill.\n---\nbody",
        encoding="utf-8",
    )

    response = await client.get("/web/api/skills")

    assert response.status_code == 200
    assert response.json() == {
        "skills": [
            BUILTIN_SKILL_COUNCIL,
            {"name": "mine", "description": "My skill.", "source": "owner"},
            {"name": "review", "description": "Global review.", "source": "global"},
            BUILTIN_SKILL_CREATOR,
        ]
    }


async def test_skills_endpoint_lists_discovered_global_skills(
    client, test_config: DlightragConfig, tmp_path: Path
) -> None:
    mutate_config(test_config, "answer.agent.skills_root", str(tmp_path / "global"))
    mutate_config(test_config, "answer.agent.owner_skills_root", str(tmp_path / "owners"))
    review = tmp_path / "global" / "review"
    review.mkdir(parents=True)
    (review / "SKILL.md").write_text(
        "---\nname: review\ndescription: Review plans.\n---\nbody",
        encoding="utf-8",
    )

    response = await client.get("/web/api/skills")

    assert response.status_code == 200
    assert response.json() == {
        "skills": [
            BUILTIN_SKILL_COUNCIL,
            {"name": "review", "description": "Review plans.", "source": "global"},
            BUILTIN_SKILL_CREATOR,
        ]
    }


async def test_skills_endpoint_lists_builtin_for_empty_filesystem_roots(
    client, test_config: DlightragConfig, tmp_path: Path
) -> None:
    mutate_config(test_config, "answer.agent.skills_root", str(tmp_path / "global"))
    mutate_config(test_config, "answer.agent.owner_skills_root", str(tmp_path / "owners"))

    response = await client.get("/web/api/skills")

    assert response.status_code == 200
    assert response.json() == {"skills": [BUILTIN_SKILL_COUNCIL, BUILTIN_SKILL_CREATOR]}


async def test_answer_rejects_unknown_requested_skill(
    client, test_config: DlightragConfig, tmp_path: Path
) -> None:
    mutate_config(test_config, "answer.agent.skills_root", str(tmp_path / "global"))
    mutate_config(test_config, "answer.agent.owner_skills_root", str(tmp_path / "owners"))

    response = await client.post(
        "/web/api/answer",
        json={
            "query": "Check this plan",
            "workspaces": ["default"],
            "submission_id": SUBMISSION_ID,
            "requested_skill": "does-not-exist",
        },
    )

    assert response.status_code == 422
    assert response.json()["kind"] == "invalid_request"


async def test_answer_with_requested_skill_forces_research_mode(
    client, mock_application, test_config: DlightragConfig, tmp_path: Path
) -> None:
    mutate_config(test_config, "answer.agent.skills_root", str(tmp_path / "global"))
    mutate_config(test_config, "answer.agent.owner_skills_root", str(tmp_path / "owners"))
    review = tmp_path / "global" / "review"
    review.mkdir(parents=True)
    (review / "SKILL.md").write_text(
        "---\nname: review\ndescription: Review plans.\n---\nbody",
        encoding="utf-8",
    )
    mock_application.web_conversations.start_answer.return_value = None

    response = await client.post(
        "/web/api/answer",
        json={
            "query": "Check this plan",
            "workspaces": ["default"],
            "submission_id": SUBMISSION_ID,
            "mode": "auto",
            "requested_skill": "review",
        },
    )

    assert response.status_code == 404  # service returns None → conversation_missing
    call = mock_application.web_conversations.start_answer.await_args
    assert call.kwargs["mode"] == "research"
    assert call.kwargs["requested_skill"] == "review"


async def test_vite_hashed_assets_are_immutable(client):
    from dlightrag.adapters.http.browser.static_files import APP_DIR

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


def _web_client_for(application: Any) -> AsyncClient:
    app = create_app(include_web_app=True)
    app.state.application = application
    transport = ASGITransport(app=app)
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
        mutate_config(test_config, "access.auth_mode", "simple")
        mutate_config(test_config, "access.api_token", "secret-token")

        async with _web_client_for(mock_application) as c:
            resp = await c.get("/web/")

        assert resp.status_code == 303
        assert resp.headers["location"].startswith("/web/login")

    async def test_conversation_route_login_redirect_preserves_deep_link(
        self, test_config: DlightragConfig, mock_application
    ) -> None:
        from urllib.parse import parse_qs, urlsplit

        mutate_config(test_config, "access.auth_mode", "simple")
        mutate_config(test_config, "access.api_token", "secret-token")
        path = f"/web/conversations/{CONVERSATION_ID}"

        async with _web_client_for(mock_application) as client:
            response = await client.get(path)

        assert response.status_code == 303
        query = parse_qs(urlsplit(response.headers["location"]).query)
        assert query["next"] == [path]

    @pytest.mark.parametrize(
        ("method", "path"),
        [
            ("GET", "/web/api/bootstrap"),
            ("POST", "/web/api/conversations"),
            # A source download link is a Web API route too: never sent to the login page.
            ("GET", "/web/api/files/raw/doc-report?workspace=finance"),
        ],
    )
    async def test_web_api_refuses_a_missing_session_in_the_error_envelope(
        self, test_config: DlightragConfig, mock_application, method: str, path: str
    ) -> None:
        mutate_config(test_config, "access.auth_mode", "simple")
        mutate_config(test_config, "access.api_token", "secret-token")

        async with _web_client_for(mock_application) as client:
            response = await client.request(method, path)

        assert response.status_code == 401
        assert response.json() == {"detail": "Authentication required", "error_type": "auth"}
        assert "location" not in response.headers

    @pytest.mark.parametrize("method", ["GET", "POST"])
    async def test_expired_session_cookie_is_a_401_for_the_app_and_a_login_for_a_page(
        self, test_config: DlightragConfig, mock_application, method: str
    ) -> None:
        import base64
        from urllib.parse import parse_qs, urlsplit

        key = "test-jwt-verification-key-for-web-route-tests"
        mutate_config(test_config, "access.auth_mode", "jwt")
        mutate_config(test_config, "access.jwt_verification_key", key)
        expired = jwt.encode(
            {
                "sub": "user-1",
                "exp": datetime.datetime.now(datetime.UTC) - datetime.timedelta(minutes=1),
            },
            key,
            algorithm="HS256",
        )
        page = f"/web/conversations/{CONVERSATION_ID}"

        async with _web_client_for(mock_application) as client:
            client.cookies.set(
                "dlightrag_web_auth",
                base64.urlsafe_b64encode(expired.encode()).decode().rstrip("="),
            )
            api = await client.request(method, "/web/api/conversations")
            navigation = await client.get(page)

        assert api.status_code == 401
        assert api.json() == {"detail": "Token expired", "error_type": "auth"}
        # Only a page load drops the cookie, so a request in flight cannot sign out another tab.
        assert "set-cookie" not in api.headers
        assert navigation.status_code == 303
        assert parse_qs(urlsplit(navigation.headers["location"]).query)["next"] == [page]
        assert "dlightrag_web_auth=" in navigation.headers["set-cookie"]

    async def test_simple_invalid_bearer_rejected(
        self, test_config: DlightragConfig, mock_application
    ) -> None:
        mutate_config(test_config, "access.auth_mode", "simple")
        mutate_config(test_config, "access.api_token", "secret-token")

        async with _web_client_for(mock_application) as c:
            resp = await c.get(
                "/web/api/files",
                headers={"Authorization": "Bearer wrong-token"},
            )

        assert resp.status_code == 401

    async def test_simple_login_page_is_static_and_no_store(
        self, test_config: DlightragConfig, mock_application
    ) -> None:
        mutate_config(test_config, "access.auth_mode", "simple")
        mutate_config(test_config, "access.api_token", "secret-token")

        async with _web_client_for(mock_application) as client:
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

        mutate_config(test_config, "access.auth_mode", "simple")
        mutate_config(test_config, "access.api_token", "secret-token")
        target = f"/web/conversations/{CONVERSATION_ID}"

        async with _web_client_for(mock_application) as client:
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
        mutate_config(test_config, "access.auth_mode", "simple")
        mutate_config(test_config, "access.api_token", "secret-token")

        async with _web_client_for(mock_application) as c:
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
        from dlightrag.application.corpus_admin import LocalDownloadTarget

        mutate_config(test_config, "access.auth_mode", "simple")
        mutate_config(test_config, "access.api_token", "secret-token")
        source = test_config.input_dir_path / "default" / "notes.md"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("downloadable notes", encoding="utf-8")
        mock_application.corpora.prepare_source_download.return_value = LocalDownloadTarget(
            path=source.resolve(),
            media_type="text/markdown",
            filename="notes.md",
        )

        async with _web_client_for(mock_application) as c:
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
        mutate_config(test_config, "access.auth_mode", "simple")
        mutate_config(test_config, "access.api_token", "secret-token")

        async with _web_client_for(mock_application) as c:
            resp = await c.post(
                "/web/login",
                data={"token": "secret-token", "next": "https://evil.example/"},
            )

        assert resp.status_code == 303
        assert resp.headers["location"] == "/web/"

    async def test_invalid_auth_cookie_is_cleared(
        self, test_config: DlightragConfig, mock_application
    ) -> None:
        mutate_config(test_config, "access.auth_mode", "simple")
        mutate_config(test_config, "access.api_token", "secret-token")

        async with _web_client_for(mock_application) as c:
            c.cookies.set("dlightrag_web_auth", "not base64!")
            resp = await c.get("/web/")

        assert resp.status_code == 303
        assert resp.headers["location"].startswith("/web/login")
        assert "dlightrag_web_auth=" in resp.headers["set-cookie"]

    async def test_bearer_header_grants_web_access(
        self, test_config: DlightragConfig, mock_application
    ) -> None:
        mutate_config(test_config, "access.auth_mode", "simple")
        mutate_config(test_config, "access.api_token", "secret-token")

        async with _web_client_for(mock_application) as c:
            resp = await c.get(
                "/web/api/files",
                headers={"Authorization": "Bearer secret-token"},
            )

        assert resp.status_code == 200

    async def test_jwt_invalid_bearer_rejected(
        self, test_config: DlightragConfig, mock_application
    ) -> None:
        mutate_config(test_config, "access.auth_mode", "jwt")
        mutate_config(
            test_config,
            "access.jwt_verification_key",
            "test-jwt-verification-key-for-web-route-tests",
        )

        async with _web_client_for(mock_application) as c:
            resp = await c.get(
                "/web/api/files",
                headers={"Authorization": "Bearer not-a-jwt"},
            )

        assert resp.status_code == 401

    async def test_jwt_bearer_header_grants_web_access(
        self, test_config: DlightragConfig, mock_application
    ) -> None:
        mutate_config(test_config, "access.auth_mode", "jwt")
        mutate_config(
            test_config,
            "access.jwt_verification_key",
            "test-jwt-verification-key-for-web-route-tests",
        )
        token = jwt.encode(
            {
                "sub": "user-1",
                "exp": datetime.datetime.now(datetime.UTC) + datetime.timedelta(minutes=5),
            },
            "test-jwt-verification-key-for-web-route-tests",
            algorithm="HS256",
        )

        async with _web_client_for(mock_application) as c:
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
        assert "DlightRAG" in response.text
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

    async def test_design_system_route_serves_its_vite_entry(self, client: AsyncClient) -> None:
        response = await client.get("/web/design-system")

        assert response.status_code == 200
        assert "DlightRAG Design System" in response.text
        assert "/static/app/assets/design-system-" in response.text
        assert "<dl-app>" not in response.text

    async def test_product_showcase_route_serves_its_vite_entry(self, client: AsyncClient) -> None:
        response = await client.get("/web/product-showcase")

        assert response.status_code == 200
        assert "DlightRAG Product Showcase" in response.text
        assert "/static/app/assets/product-showcase-" in response.text
        assert "<dl-app>" not in response.text

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
            "contract_version": 3,
            "personal_mcp_connections": True,
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
            "workspaces_next_cursor": None,
            "primary_workspace": "default",
            "active_workspaces": ["default", "test_ws"],
            "known_workspaces": ["default", "test_ws"],
            "default_workspace": "default",
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
            "active_html_preview_enabled": True,
            "agent_effort": {
                "levels": ["low", "high", "max"],
                "default": None,
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
        # The 503 is the inventory's outage, not a call the real signature refused.
        mock_application.corpora.alist_workspace_records.assert_awaited_once_with()

    @pytest.mark.parametrize(
        "old_path",
        [
            "/web/answer",
            "/web/conversations",
            "/web/files",
            "/web/ingest-status",
            "/web/workspaces/create",
            # The positional attachment address was replaced by the run-resource
            # address; the old shape is not kept as an alias.
            "/web/api/runs/019893f4-0000-7000-8000-000000000001/attachments/0",
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
        self,
        client: AsyncClient,
        test_config: DlightragConfig,
        mock_application,
    ) -> None:
        resp = await client.get("/web/api/files")

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/json")
        assert resp.json()["workspace"] == "default"
        assert resp.json()["files"] == [{"file_name": "test.pdf", "file_path": "/tmp/test.pdf"}]
        assert resp.json()["next_cursor"] is None
        mock_application.corpora.file_panel_snapshot.assert_awaited_once_with(
            "default", page=FilePanelPageRequest()
        )
        mock_application.corpora.workspace_exists.assert_awaited_once_with("default")
        mock_application.corpora.list_workspaces.assert_not_awaited()

    @pytest.mark.parametrize("limit", [1, 100])
    async def test_file_list_accepts_bounded_explicit_limits(
        self,
        client: AsyncClient,
        mock_application,
        limit: int,
    ) -> None:
        response = await client.get("/web/api/files", params={"limit": limit})

        assert response.status_code == 200
        mock_application.corpora.file_panel_snapshot.assert_awaited_once_with(
            "default", page=FilePanelPageRequest(limit=limit)
        )

    @pytest.mark.parametrize("limit", ["0", "101", "not-an-int"])
    async def test_file_list_rejects_invalid_limits_before_route_storage(
        self,
        client: AsyncClient,
        mock_application,
        limit: str,
    ) -> None:
        response = await client.get("/web/api/files", params={"limit": limit})

        assert response.status_code == 422
        mock_application.corpora.workspace_exists.assert_not_awaited()
        mock_application.corpora.file_panel_snapshot.assert_not_awaited()

    async def test_file_list_cursor_round_trip_and_cross_workspace_rejection(
        self, client: AsyncClient, mock_application
    ) -> None:
        timestamp = datetime.datetime(2026, 3, 4, 5, 6, 7, 123456)
        continuation = FilePanelCursor(
            workspace="default",
            updated_at=timestamp,
            doc_id="doc-50",
        )
        mock_application.corpora.file_panel_snapshot.return_value = {
            "files": [],
            "next_cursor": continuation,
            "fetched_rows": 51,
        }

        first = await client.get("/web/api/files")
        token = first.json()["next_cursor"]
        assert isinstance(token, str)

        mock_application.corpora.file_panel_snapshot.reset_mock()
        second = await client.get("/web/api/files", params={"cursor": token})
        assert second.status_code == 200
        mock_application.corpora.file_panel_snapshot.assert_awaited_once_with(
            "default",
            page=FilePanelPageRequest(cursor=continuation),
        )

        mock_application.corpora.file_panel_snapshot.reset_mock()
        foreign = mock_application.corpora.file_panel_cursor_codec.encode(
            FilePanelCursor(workspace="finance", updated_at=None, doc_id="doc")
        )
        response = await client.get("/web/api/files", params={"cursor": foreign})
        assert response.status_code == 422
        mock_application.corpora.file_panel_snapshot.assert_not_awaited()

        failed_view = mock_application.corpora.file_panel_cursor_codec.encode(
            FilePanelCursor(
                workspace="default", updated_at=None, doc_id="doc-failed", view="failed"
            )
        )
        response = await client.get("/web/api/files", params={"cursor": failed_view})
        assert response.status_code == 422
        mock_application.corpora.file_panel_snapshot.assert_not_awaited()

    async def test_file_list_rejects_tampered_cursor_before_storage(
        self, client: AsyncClient, mock_application
    ) -> None:
        token = mock_application.corpora.file_panel_cursor_codec.encode(
            FilePanelCursor(workspace="default", updated_at=None, doc_id="doc")
        )

        response = await client.get("/web/api/files", params={"cursor": token + "x"})

        assert response.status_code == 422
        mock_application.corpora.file_panel_snapshot.assert_not_awaited()

    async def test_file_workspace_registry_outage_fails_open_without_catalog_scan(
        self, client: AsyncClient, mock_application
    ) -> None:
        mock_application.corpora.workspace_exists.side_effect = RuntimeError("registry down")

        response = await client.get("/web/api/files", params={"workspace": "cold-ws"})

        assert response.status_code == 200
        # The route failed open on the registry's outage, not on a refused call.
        mock_application.corpora.workspace_exists.assert_awaited_once_with("cold_ws")
        mock_application.corpora.file_panel_snapshot.assert_awaited_once()
        mock_application.corpora.list_workspaces.assert_not_awaited()

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
        # The 503 is the snapshot's outage, not a call the real signature refused.
        mock_application.corpora.file_panel_snapshot.assert_awaited_once()

    async def test_file_list_derives_display_name_from_path(
        self, client: AsyncClient, test_config: DlightragConfig, mock_application
    ) -> None:
        mock_application.corpora.file_panel_snapshot.return_value = {
            "files": [{"doc_id": "d1", "file_path": "/tmp/reports/q4.pdf", "status": "processed"}],
        }

        resp = await client.get("/web/api/files")

        assert resp.status_code == 200
        assert resp.json()["files"] == [{"file_name": "q4.pdf", "file_path": "/tmp/reports/q4.pdf"}]

    async def test_file_list_uses_file_panel_snapshot_for_cold_workspace(
        self, client: AsyncClient, test_config: DlightragConfig, mock_application
    ) -> None:
        mock_application.corpora.workspace_exists.return_value = True
        mock_application.corpora.file_panel_snapshot.return_value = {
            "files": [{"doc_id": "d1", "file_path": "/tmp/cold/report.pdf", "status": "processed"}],
        }

        resp = await client.get("/web/api/files", params={"workspace": "cold-ws"})

        assert resp.status_code == 200
        assert resp.json()["files"] == [
            {"file_name": "report.pdf", "file_path": "/tmp/cold/report.pdf"}
        ]
        mock_application.corpora.file_panel_snapshot.assert_awaited_once_with(
            "cold_ws", page=FilePanelPageRequest()
        )

    async def test_file_list_rejects_stale_workspace(
        self, client: AsyncClient, test_config: DlightragConfig, mock_application
    ) -> None:
        mock_application.corpora.workspace_exists.return_value = False

        resp = await client.get("/web/api/files", params={"workspace": "deleted_ws"})

        assert resp.status_code == 409
        assert "Workspace no longer exists" in resp.text
        mock_application.corpora.file_panel_snapshot.assert_not_awaited()

    async def test_file_list_rejects_stale_workspace_even_with_registered_cookie(
        self, client: AsyncClient, test_config: DlightragConfig, mock_application
    ) -> None:
        mock_application.corpora.workspace_exists.return_value = False
        client.cookies.set("dlightrag_workspace", "test_ws")

        resp = await client.get("/web/api/files", params={"workspace": "deleted_ws"})

        assert resp.status_code == 409
        assert "Workspace no longer exists" in resp.text
        mock_application.corpora.file_panel_snapshot.assert_not_awaited()

    async def test_file_list_canonicalizes_requested_workspace(
        self, client: AsyncClient, test_config: DlightragConfig, mock_application
    ) -> None:
        mock_application.corpora.workspace_exists.return_value = True

        resp = await client.get("/web/api/files", params={"workspace": "test-fallback-ws"})

        assert resp.status_code == 200
        mock_application.corpora.file_panel_snapshot.assert_awaited_once_with(
            "test_fallback_ws", page=FilePanelPageRequest()
        )

    async def test_file_list_rejects_stale_workspace_without_default(
        self, client: AsyncClient, test_config: DlightragConfig, mock_application
    ) -> None:
        mock_application.corpora.workspace_exists.return_value = False

        resp = await client.get("/web/api/files", params={"workspace": "deleted_ws"})

        assert resp.status_code == 409
        assert "Workspace no longer exists" in resp.text
        mock_application.corpora.file_panel_snapshot.assert_not_awaited()

    async def test_failed_files_page_projects_bounded_rows(
        self, client: AsyncClient, mock_application
    ) -> None:
        timestamp = datetime.datetime(2026, 8, 31, 21, 36, 15)
        continuation = FilePanelCursor(
            workspace="default",
            updated_at=timestamp,
            doc_id="doc-failed",
            view="failed",
        )
        mock_application.corpora.failed_file_snapshot.return_value = {
            "failed": [
                {
                    "doc_id": "doc-failed",
                    "file_path": "/books/failed.pdf",
                    "error": "embedding failed: public diagnostic",
                    "updated_at": timestamp.isoformat(),
                }
            ],
            "next_cursor": continuation,
            "fetched_rows": 2,
        }
        response = await client.get("/web/api/files/failed")

        assert response.status_code == 200
        payload = response.json()
        assert payload["failed"] == [
            {
                "document_id": "doc-failed",
                "file_name": "failed.pdf",
                "error": payload["failed"][0]["error"],
                "updated_at": timestamp.isoformat(),
            }
        ]
        diagnostic = payload["failed"][0]["error"]
        assert diagnostic.startswith("embedding failed")
        assert "secret" not in diagnostic
        assert "password" not in diagnostic
        assert "/srv/private" not in diagnostic
        assert len(diagnostic) <= 512
        assert isinstance(payload["next_cursor"], str)
        assert "active_recovery" not in payload
        mock_application.corpora.failed_file_snapshot.assert_awaited_once_with(
            "default",
            page=FilePanelPageRequest(limit=5),
        )

    async def test_failed_file_retry_accepts_a_durable_run(
        self, client: AsyncClient, mock_application
    ) -> None:
        response = await client.post("/web/api/files/retry")

        assert response.status_code == 202
        assert response.json()["run_kind"] == "corpus_mutation"
        mock_application.corpus_mutations.create_retry.assert_awaited_once_with(
            workspace="default",
            selector="all_retryable",
            submitted_by=DEPLOYMENT_OWNER_ID,
        )

    async def test_failed_file_retry_projects_the_admission_limit(
        self, client: AsyncClient, mock_application
    ) -> None:
        mock_application.corpus_mutations.create_retry.side_effect = (
            RunAdmissionLimitExceededError()
        )

        response = await client.post("/web/api/files/retry")

        assert response.status_code == 503
        assert response.json()["detail"] == "Deployment-wide nonterminal admission limit reached"

    async def test_corpus_receipt_uses_same_origin_browser_run_urls(
        self, client: AsyncClient
    ) -> None:
        response = await client.post("/web/api/files/retry")

        assert response.status_code == 202
        run_id = response.json()["run_id"]
        assert response.json()["status_url"] == f"/web/api/corpus-runs/{run_id}"
        assert response.json()["events_url"] == f"/web/api/corpus-runs/{run_id}/events"
        assert response.json()["cancel_url"] == f"/web/api/corpus-runs/{run_id}"

    async def test_browser_reads_authorized_corpus_status_with_repair_guidance(
        self, client: AsyncClient, mock_application
    ) -> None:
        now = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)
        record = SimpleNamespace(
            run_id="0199a0a0-0000-7000-8000-0000000000bb",
            run_kind="corpus_mutation",
            lane="corpus_mutation",
            status="running",
            phase="waiting_for_repair",
            durable_progress_version=3,
            cancel_requested=False,
            result=None,
            error_kind=None,
            error_message=None,
            repair_reason="Upstream outcome is uncertain.",
            repair_remedy="Inspect and repair, then resume.",
            created_at=now,
            started_at=now,
            finished_at=None,
            access_scope_kind="workspace",
            access_scope_id="default",
            events_trimmed_at=None,
            request_input=lambda: {"action": "delete"},
        )
        mock_application.runs.get_global.return_value = record

        response = await client.get(f"/web/api/corpus-runs/{record.run_id}")

        assert response.status_code == 200
        assert response.json()["phase"] == "waiting_for_repair"
        assert response.json()["repair_reason"] == "Upstream outcome is uncertain."
        assert response.json()["repair_remedy"] == "Inspect and repair, then resume."
        assert response.json()["resume_url"] == f"/web/api/corpus-runs/{record.run_id}/resume"

    async def test_browser_resumes_the_same_authorized_corpus_run(
        self, client: AsyncClient, mock_application
    ) -> None:
        now = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)
        waiting = SimpleNamespace(
            run_id="0199a0a0-0000-7000-8000-0000000000bb",
            run_kind="corpus_mutation",
            lane="corpus_mutation",
            status="running",
            phase="waiting_for_repair",
            durable_progress_version=3,
            cancel_requested=False,
            result=None,
            error_kind=None,
            error_message=None,
            repair_reason="Inspect upstream state.",
            repair_remedy="Repair it, then resume.",
            created_at=now,
            started_at=now,
            finished_at=None,
            access_scope_kind="workspace",
            access_scope_id="default",
            events_trimmed_at=None,
            request_input=lambda: {"action": "delete"},
        )
        queued = SimpleNamespace(**{**waiting.__dict__, "status": "queued", "phase": None})
        mock_application.runs.get_global.return_value = waiting
        mock_application.runs.get.return_value = queued

        response = await client.post(f"/web/api/corpus-runs/{waiting.run_id}/resume")

        assert response.status_code == 202
        assert response.json()["run_id"] == waiting.run_id
        assert response.json()["status"] == "queued"
        mock_application.runs.resume_repair.assert_awaited_once_with(
            owner_id="default", run_id=waiting.run_id
        )

    async def test_failed_files_rejects_processed_view_cursor(
        self, client: AsyncClient, mock_application
    ) -> None:
        token = mock_application.corpora.file_panel_cursor_codec.encode(
            FilePanelCursor(workspace="default", updated_at=None, doc_id="doc-processed")
        )
        mock_application.corpora.failed_file_snapshot.reset_mock()

        response = await client.get("/web/api/files/failed", params={"cursor": token})

        assert response.status_code == 422
        mock_application.corpora.failed_file_snapshot.assert_not_awaited()

    async def test_upload_accepts_one_durable_batch_run(
        self, client: AsyncClient, mock_application, tmp_path: Path
    ) -> None:
        source = tmp_path / "report.pdf"
        source.write_bytes(b"%PDF-fake")
        mock_application.corpus_mutations.stage_uploads.return_value = [
            SimpleNamespace(
                path=source, filename="report.pdf", size_bytes=9, content_sha256="a" * 64
            )
        ]

        response = await client.post(
            "/web/api/files/upload",
            files=[("files", ("report.pdf", b"%PDF-fake", "application/pdf"))],
        )

        assert response.status_code == 202
        body = response.json()
        assert body["run_kind"] == "corpus_mutation"
        assert body["file_count"] == 1
        mock_application.corpus_mutations.stage_uploads.assert_awaited_once()
        mock_application.corpus_mutations.create_staged_batch.assert_awaited_once()
        # The Run is accepted under the id its files were staged for, never one
        # derived from their paths.
        staged_for = mock_application.corpus_mutations.stage_uploads.await_args.kwargs["run_id"]
        accepted = mock_application.corpus_mutations.create_staged_batch.await_args.kwargs
        assert accepted["run_id"] == staged_for
        mock_application.corpus_mutations.discard_staged_run.assert_not_awaited()

    async def test_single_upload_forwards_digest_verification_to_the_shared_stager(
        self, client: AsyncClient, mock_application, tmp_path: Path
    ) -> None:
        source = tmp_path / "report.pdf"
        source.write_bytes(b"content")
        digest = "a" * 64
        mock_application.corpus_mutations.stage_uploads.return_value = [
            SimpleNamespace(path=source, filename="report.pdf", size_bytes=7, content_sha256=digest)
        ]

        response = await client.post(
            "/web/api/files/upload",
            data={"content_sha256": digest},
            files=[("files", ("report.pdf", b"content", "application/pdf"))],
        )

        assert response.status_code == 202
        assert (
            mock_application.corpus_mutations.stage_uploads.await_args.kwargs["content_sha256"]
            == digest
        )

    async def test_upload_maps_a_cap_refusal_to_413_and_discards_the_stage(
        self, client: AsyncClient, mock_application
    ) -> None:
        from dlightrag.application.corpus_admin import UploadTooLargeError

        mock_application.corpus_mutations.stage_uploads.side_effect = UploadTooLargeError(
            "upload exceeds 536870912 bytes"
        )

        response = await client.post(
            "/web/api/files/upload",
            files=[
                ("files", ("first.pdf", b"first", "application/pdf")),
                ("files", ("second.pdf", b"second", "application/pdf")),
            ],
        )

        assert response.status_code == 413
        assert response.json()["detail"] == (
            "Upload exceeds limit (100 MB per file, 512 MB per request)"
        )
        mock_application.corpus_mutations.create_staged_batch.assert_not_awaited()
        mock_application.corpus_mutations.discard_staged_run.assert_awaited_once()

    async def test_upload_projects_the_admission_limit_and_discards_the_stage(
        self, client: AsyncClient, mock_application, tmp_path: Path
    ) -> None:
        source = tmp_path / "report.pdf"
        source.write_bytes(b"content")
        mock_application.corpus_mutations.stage_uploads.return_value = [
            SimpleNamespace(
                path=source, filename="report.pdf", size_bytes=7, content_sha256="a" * 64
            )
        ]
        mock_application.corpus_mutations.create_staged_batch.side_effect = (
            RunAdmissionLimitExceededError()
        )

        response = await client.post(
            "/web/api/files/upload",
            files=[("files", ("report.pdf", b"content", "application/pdf"))],
        )

        assert response.status_code == 503
        assert response.json()["detail"] == "Deployment-wide nonterminal admission limit reached"
        mock_application.corpus_mutations.discard_staged_run.assert_awaited_once()

    async def test_upload_discards_the_stage_when_run_acceptance_fails(
        self, client: AsyncClient, mock_application, tmp_path: Path
    ) -> None:
        source = tmp_path / "report.pdf"
        source.write_bytes(b"content")
        mock_application.corpus_mutations.stage_uploads.return_value = [
            SimpleNamespace(
                path=source, filename="report.pdf", size_bytes=7, content_sha256="a" * 64
            )
        ]
        mock_application.corpus_mutations.create_staged_batch.side_effect = RuntimeError(
            "admission unavailable"
        )

        response = await client.post(
            "/web/api/files/upload",
            files=[("files", ("report.pdf", b"content", "application/pdf"))],
        )

        assert response.status_code == 503
        # An untyped failure never shows its own text.
        assert response.json()["detail"] == "Upload could not be accepted. Please retry."
        # The failure is acceptance's, not a call the real signature refused.
        mock_application.corpus_mutations.create_staged_batch.assert_awaited_once()
        mock_application.corpus_mutations.discard_staged_run.assert_awaited_once()

    async def test_upload_rejects_stale_workspace(
        self, client: AsyncClient, test_config: DlightragConfig, mock_application
    ) -> None:
        mock_application.corpora.workspace_exists.return_value = False

        resp = await client.post(
            "/web/api/files/upload",
            data={"workspace": "deleted_ws"},
            files=[("files", ("report.pdf", b"%PDF-fake", "application/pdf"))],
        )

        assert resp.status_code == 409
        assert "Workspace no longer exists" in resp.text
        mock_application.corpus_mutations.stage_uploads.assert_not_awaited()

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
        from dlightrag.engine.rag.corpus.ingestion.uploads import safe_upload_relative_path

        with pytest.raises(ValueError):
            safe_upload_relative_path(filename)

    async def test_delete_files_accepts_a_durable_run(
        self, client: AsyncClient, mock_application
    ) -> None:
        response = await client.request(
            "DELETE",
            "/web/api/files",
            params={"file_path": "/tmp/test.pdf"},
        )
        assert response.status_code == 202
        assert response.json()["run_kind"] == "corpus_mutation"
        mock_application.corpus_mutations.create_delete.assert_awaited_once_with(
            workspace="default",
            file_paths=["/tmp/test.pdf"],
            submitted_by=DEPLOYMENT_OWNER_ID,
        )

    async def test_delete_files_reports_an_invalid_identifier_as_a_client_error(
        self, client: AsyncClient, mock_application
    ) -> None:
        mock_application.corpus_mutations.create_delete.side_effect = ValueError(
            "delete requires an exact identifier"
        )

        response = await client.request("DELETE", "/web/api/files", params={"file_path": " "})

        assert response.status_code == 400
        assert response.json()["detail"] == "delete requires an exact identifier"

    async def test_delete_files_projects_the_admission_limit(
        self, client: AsyncClient, mock_application
    ) -> None:
        mock_application.corpus_mutations.create_delete.side_effect = (
            RunAdmissionLimitExceededError()
        )

        response = await client.request(
            "DELETE",
            "/web/api/files",
            params={"file_path": "/tmp/test.pdf"},
        )

        assert response.status_code == 503
        assert response.json()["detail"] == "Deployment-wide nonterminal admission limit reached"

    async def test_delete_files_rejects_stale_workspace(
        self, client: AsyncClient, test_config: DlightragConfig, mock_application
    ) -> None:
        mock_application.corpora.workspace_exists.return_value = False

        resp = await client.request(
            "DELETE",
            "/web/api/files",
            params={"workspace": "deleted_ws", "file_path": "/tmp/test.pdf"},
        )

        assert resp.status_code == 409
        assert "Workspace no longer exists" in resp.text
        mock_application.corpus_mutations.create_delete.assert_not_awaited()


# ---------------------------------------------------------------------------
# TestWebWorkspaceCreateDelete
# ---------------------------------------------------------------------------


class TestWebWorkspaceCreate:
    """Tests for POST /web/api/workspaces/create."""

    async def test_create_workspace(
        self, client: AsyncClient, test_config: DlightragConfig, mock_application
    ) -> None:
        # The registry owns uniqueness; the catalog is read once, for the cookies.
        mock_application.corpora.list_workspaces.return_value = [
            "default",
            "test_ws",
            "new_workspace",
        ]
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
        from dlightrag.application.corpus_admin import WorkspaceExistsError

        mock_application.corpora.create_workspace.side_effect = WorkspaceExistsError(
            "Workspace 'default' already exists"
        )
        resp = await client.post(
            "/web/api/workspaces/create",
            data={"workspace_name": "default"},
        )
        assert resp.status_code == 409
        assert resp.json()["detail"] == "Workspace 'default' already exists"

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
        assert resp.json()["detail"]


async def test_bootstrap_falls_back_to_the_configured_default_workspace(
    client: AsyncClient, test_config: DlightragConfig
) -> None:
    """A deployment whose default is not literally "default" must not scope to "default"."""
    mutate_config(test_config, "deployment.workspace", "Test WS")
    client.cookies.delete("dlightrag_workspace")

    unscoped = (await client.get("/web/api/bootstrap")).json()
    client.cookies.set("dlightrag_workspace", "deleted_ws")
    stale = (await client.get("/web/api/bootstrap")).json()

    assert unscoped["default_workspace"] == "test_ws"
    assert unscoped["primary_workspace"] == "test_ws"
    assert stale["primary_workspace"] == "test_ws"


async def test_bootstrap_bounds_the_visible_array_but_keeps_full_authorization_inputs(
    client: AsyncClient, mock_application
) -> None:
    mock_application.corpora.list_workspace_records_page.return_value = WorkspaceCatalogPage(
        items=(
            {
                "workspace": "default",
                "display_name": "Default",
                "embedding_model": "voyage-multimodal-3.5",
                "created_at": None,
                "updated_at": None,
            },
        ),
        next_cursor=WorkspaceCatalogCursor(after_workspace="default"),
        fetched_rows=2,
    )
    mock_application.corpora.alist_workspace_records.return_value = [
        {
            "workspace": "default",
            "display_name": "Default",
            "embedding_model": "voyage-multimodal-3.5",
        },
        {
            "workspace": "finance",
            "display_name": "Finance",
            "embedding_model": "voyage-multimodal-3.5",
        },
        {
            "workspace": "research",
            "display_name": "Research",
            "embedding_model": "voyage-multimodal-3.5",
        },
    ]
    client.cookies.delete("dlightrag_workspace")

    resp = await client.get("/web/api/bootstrap")

    assert resp.status_code == 200
    body = resp.json()
    assert [item["workspace"] for item in body["workspaces"]] == ["default"]
    assert body["workspaces_next_cursor"] is not None
    # The full catalog still powers the authorization inputs: without cookies,
    # every visible workspace is active, not just the bounded display page.
    assert body["active_workspaces"] == ["default", "finance", "research"]
    assert body["known_workspaces"] == ["default", "finance", "research"]
    assert body["primary_workspace"] == "default"


async def test_web_workspaces_page_roundtrips_an_opaque_cursor(
    client: AsyncClient, mock_application
) -> None:
    mock_application.corpora.list_workspace_records_page.return_value = WorkspaceCatalogPage(
        items=(
            {
                "workspace": "finance",
                "display_name": "Finance",
                "embedding_model": "voyage-multimodal-3.5",
                "created_at": None,
                "updated_at": None,
            },
        ),
        next_cursor=WorkspaceCatalogCursor(after_workspace="finance"),
        fetched_rows=2,
    )

    resp = await client.get("/web/api/workspaces")

    assert resp.status_code == 200
    body = resp.json()
    assert body["workspaces"] == [
        {
            "workspace": "finance",
            "display_name": "Finance",
            "embedding_model": "voyage-multimodal-3.5",
        }
    ]
    assert body["next_cursor"] is not None

    second = await client.get("/web/api/workspaces", params={"cursor": body["next_cursor"]})
    assert second.status_code == 200
    assert second.json()["workspaces"][0]["workspace"] == "finance"


async def test_web_workspaces_page_rejects_tampered_cursor_before_storage(
    client: AsyncClient, mock_application
) -> None:
    resp = await client.get("/web/api/workspaces", params={"cursor": "AAAA.tampered"})

    assert resp.status_code == 422
    mock_application.corpora.list_workspace_records_page.assert_not_awaited()


async def test_reset_workspace_accepts_a_durable_corpus_run(
    client: AsyncClient, mock_application
) -> None:
    response = await client.post(
        "/web/api/workspaces/reset",
        data={"workspace_name": "test-ws", "confirm_name": "test-ws"},
    )

    assert response.status_code == 202
    assert response.json()["run_kind"] == "corpus_mutation"
    assert response.json()["workspace"] == "test_ws"
    mock_application.corpus_mutations.create_reset.assert_awaited_once_with(
        workspace="test_ws",
        submitted_by=DEPLOYMENT_OWNER_ID,
    )


@pytest.mark.parametrize(
    "path",
    ["/web/api/workspaces/create", "/web/api/workspaces/reset", "/web/api/workspaces/delete"],
    ids=["create", "reset", "delete"],
)
async def test_workspace_commands_on_a_read_only_replica_pass_the_refusal_through(
    client: AsyncClient, mock_application, path: str
) -> None:
    from dlightrag.application.corpus_admin import CorpusMutationUnavailableError

    refusal = CorpusMutationUnavailableError(request="the request")
    mock_application.corpora.create_workspace.side_effect = refusal
    mock_application.corpus_mutations.create_reset.side_effect = refusal
    mock_application.corpus_mutations.create_workspace_delete.side_effect = refusal

    response = await client.post(
        path, data={"workspace_name": "test-ws", "confirm_name": "test-ws"}
    )

    assert response.status_code == 503
    assert response.json() == {"detail": str(refusal), "error_type": "unavailable"}


@pytest.mark.parametrize(
    ("path", "failing", "status", "detail", "logged"),
    [
        (
            "/web/api/workspaces/create",
            "create_workspace",
            500,
            "Failed to create workspace; see server logs for details.",
            "Workspace creation failed",
        ),
        (
            "/web/api/workspaces/reset",
            "create_reset",
            503,
            "Failed to accept Corpus Reset; see server logs for details.",
            "Workspace reset Run acceptance failed",
        ),
        (
            "/web/api/workspaces/delete",
            "create_workspace_delete",
            503,
            "Failed to accept Workspace Delete; see server logs for details.",
            "Workspace Delete Run acceptance failed",
        ),
    ],
    ids=["create", "reset", "delete"],
)
async def test_workspace_commands_hide_untyped_failures_behind_logged_advice(
    client: AsyncClient,
    mock_application,
    caplog: pytest.LogCaptureFixture,
    path: str,
    failing: str,
    status: int,
    detail: str,
    logged: str,
) -> None:
    failure = RuntimeError("socket closed at 10.0.0.7")
    service = (
        mock_application.corpora
        if failing == "create_workspace"
        else mock_application.corpus_mutations
    )
    getattr(service, failing).side_effect = failure

    with caplog.at_level("ERROR"):
        response = await client.post(
            path, data={"workspace_name": "test-ws", "confirm_name": "test-ws"}
        )

    assert response.status_code == status
    assert response.json()["detail"] == detail
    assert "10.0.0.7" not in response.text
    assert logged in caplog.text
    # The advice covers the service's own failure, not a call the real signature refused.
    getattr(service, failing).assert_awaited_once()


async def test_workspace_commands_require_a_name(client: AsyncClient) -> None:
    response = await client.post("/web/api/workspaces/reset", data={"workspace_name": " "})

    assert response.status_code == 400
    assert response.json() == {
        "detail": "Workspace name cannot be empty",
        "error_type": "validation",
    }


async def test_reset_workspace_projects_the_admission_limit(
    client: AsyncClient, mock_application
) -> None:
    mock_application.corpus_mutations.create_reset.side_effect = RunAdmissionLimitExceededError()

    response = await client.post(
        "/web/api/workspaces/reset",
        data={"workspace_name": "test-ws", "confirm_name": "test-ws"},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Deployment-wide nonterminal admission limit reached"


async def test_delete_workspace_accepts_a_durable_run_behind_the_delete_permission(
    client: AsyncClient, mock_application, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dlightrag.adapters.http.browser.routes import workspaces as workspace_routes

    checked: list[tuple[str, str]] = []

    async def record_access(_request: Any, action: str, workspace: str) -> None:
        checked.append((action, workspace))

    monkeypatch.setattr(workspace_routes, "enforce_web_access", record_access)

    response = await client.post(
        "/web/api/workspaces/delete",
        data={"workspace_name": "test-ws", "confirm_name": "test-ws"},
    )

    assert response.status_code == 202
    assert response.json()["run_kind"] == "corpus_mutation"
    assert response.json()["workspace"] == "test_ws"
    assert checked == [("workspace.delete", "test_ws")]
    mock_application.corpus_mutations.create_workspace_delete.assert_awaited_once_with(
        workspace="test_ws",
        submitted_by=DEPLOYMENT_OWNER_ID,
    )
    mock_application.corpus_mutations.create_reset.assert_not_awaited()


async def test_delete_workspace_requires_a_matching_confirmation(
    client: AsyncClient, mock_application
) -> None:
    response = await client.post(
        "/web/api/workspaces/delete",
        data={"workspace_name": "test-ws", "confirm_name": "other"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Confirmation name does not match"
    mock_application.corpus_mutations.create_workspace_delete.assert_not_awaited()


async def test_delete_workspace_reports_a_workspace_that_is_already_gone(
    client: AsyncClient, mock_application
) -> None:
    # Corpus Mutation acceptance owns the existence check; the route only maps it.
    mock_application.corpus_mutations.create_workspace_delete.side_effect = WorkspaceNotFoundError(
        "Workspace no longer exists"
    )

    response = await client.post(
        "/web/api/workspaces/delete",
        data={"workspace_name": "test-ws", "confirm_name": "test-ws"},
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Workspace no longer exists", "error_type": "not_found"}


async def test_delete_workspace_explains_why_the_default_is_kept(
    client: AsyncClient, mock_application
) -> None:
    mock_application.corpus_mutations.create_workspace_delete.side_effect = ValueError(
        "The default workspace cannot be deleted; reset its corpus instead."
    )

    response = await client.post(
        "/web/api/workspaces/delete",
        data={"workspace_name": "default", "confirm_name": "default"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "The default workspace cannot be deleted; reset its corpus instead."
    )


class TestSourcePresentation:
    """Tests for structured source presentation contracts."""

    def test_page_number_and_download_are_projected(self) -> None:
        from dlightrag.adapters.http.browser.presentation import build_answer_presentation
        from dlightrag.engine.answer.citations.contracts import ChunkSnippet, SourceReferencePayload

        presentation = build_answer_presentation(
            answer="Answer [1].",
            sources=[
                SourceReferencePayload(
                    id="1",
                    title="notes.md",
                    source_uri="local://default/notes.md",
                    download_url="/web/api/files/raw/doc-notes?workspace=default",
                    chunks=[
                        ChunkSnippet(
                            chunk_id="chunk-1",
                            chunk_idx=1,
                            page_number=1,
                            content="first page",
                        )
                    ],
                )
            ],
            evidence_images=[],
        )

        source = presentation.sources[0]
        assert source.title == "notes.md"
        assert source.download_url == "/web/api/files/raw/doc-notes?workspace=default"
        assert source.chunks[0].page_number == 1
        assert "first page" in source.chunks[0].content_html

    async def test_the_effort_offer_states_exactly_what_the_deployment_offers(
        self, client: AsyncClient, web_app: Any
    ) -> None:
        """The control renders the deployment's own offer, never a fixed three levels."""
        application = web_app.state.application
        application.answers.agent_effort_offer.return_value = AgentEffortOffer(("low",), None)

        response = await client.get("/web/api/bootstrap")

        assert response.status_code == 200
        assert response.json()["agent_effort"] == {"levels": ["low"], "default": None}

    @pytest.mark.parametrize(
        ("method", "path", "body"),
        [
            ("POST", "/web/api/files/upload", None),
            ("POST", "/web/api/files/retry", {"workspace": "default"}),
            (
                "DELETE",
                "/web/api/files?workspace=default&file_path=report.pdf",
                None,
            ),
        ],
        ids=["upload", "retry", "delete"],
    )
    async def test_a_read_only_replica_reports_the_role_not_a_generic_failure(
        self,
        client: AsyncClient,
        mock_application,
        method: str,
        path: str,
        body: dict[str, object] | None,
    ) -> None:
        """The Files panel's write controls meet the role refusal, not a 500.

        Every write route catches broadly to turn infrastructure failures into advice.
        A read-only replica's typed refusal must pass through those catches, so the panel
        can say why the deployment declined instead of "please try again".
        """
        from dlightrag.application.corpus_admin import CorpusMutationUnavailableError

        refusal = CorpusMutationUnavailableError(request="the upload")
        mock_application.corpus_mutations.stage_uploads.side_effect = refusal
        mock_application.corpus_mutations.create_retry.side_effect = refusal
        mock_application.corpus_mutations.create_delete.side_effect = refusal

        if method == "POST" and path.endswith("upload"):
            response = await client.post(
                path, files=[("files", ("report.pdf", b"%PDF-fake", "application/pdf"))]
            )
        elif body is None:
            response = await client.request(method, path)
        else:
            response = await client.request(method, path, json=body)

        assert response.status_code == 503
        payload = response.json()
        assert payload["error_type"] == "unavailable"
        assert "read-only replica" in payload["detail"]


_NO_CANONICAL_ID = "1" * 64  # gains a leading underscore: 65 characters


@pytest.mark.parametrize(
    ("path", "params"),
    [
        (f"/web/api/images/{_NO_CANONICAL_ID}/chunk-1", None),
        ("/web/api/files", {"workspace": _NO_CANONICAL_ID}),
        ("/web/api/files/raw/doc-1", {"workspace": _NO_CANONICAL_ID}),
    ],
    ids=["image", "files", "download"],
)
async def test_browser_routes_refuse_a_workspace_name_without_a_canonical_id(
    client: AsyncClient, mock_application, path: str, params: dict[str, str] | None
) -> None:
    """Formerly a 500 or 503: the name normalized, but to no canonical id."""
    response = await client.get(path, params=params)

    assert response.status_code == 422
    assert response.json()["error_type"] == "validation"
    assert _NO_CANONICAL_ID not in response.text
    mock_application.corpora.workspace_exists.assert_not_awaited()


async def test_a_workspace_cookie_without_a_canonical_id_falls_back_to_the_default(
    client: AsyncClient, mock_application
) -> None:
    client.cookies.set("dlightrag_workspace", _NO_CANONICAL_ID)

    response = await client.get("/web/api/files")

    assert response.status_code == 200
    assert response.json()["workspace"] == "default"
