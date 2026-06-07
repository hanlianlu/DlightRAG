# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for FastAPI REST server endpoints and auth middleware."""

from __future__ import annotations

import datetime
from unittest.mock import AsyncMock

import jwt
import pytest
from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient

from dlightrag.api.auth import UserContext, get_current_user, verify_bearer_token
from dlightrag.api.server import create_app
from dlightrag.config import DlightragConfig
from dlightrag.core.retrieval.protocols import RetrievalResult
from dlightrag.core.servicemanager import RAGServiceUnavailableError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_ANON = UserContext(user_id="anonymous", auth_mode="none")
app: FastAPI


@pytest.fixture
def _api_app(test_config: DlightragConfig) -> FastAPI:
    """Create the API app after test_config has installed the singleton."""
    global app
    app = create_app(include_web=False)
    yield app
    app.dependency_overrides.clear()
    if hasattr(app.state, "manager"):
        del app.state.manager


@pytest.fixture
def mock_config(_api_app: FastAPI, test_config: DlightragConfig):
    """Override auth dependency to allow all requests (auth_mode=none)."""
    _api_app.dependency_overrides[get_current_user] = lambda: _ANON
    yield test_config
    _api_app.dependency_overrides.pop(get_current_user, None)


@pytest.fixture
def mock_config_no_auth_override(test_config: DlightragConfig):
    """Provide config WITHOUT overriding auth — real auth logic runs."""
    yield test_config


@pytest.fixture
def mock_service():
    """Create a mock RAGService."""
    service = AsyncMock()
    service.aingest = AsyncMock(return_value={"status": "success", "processed": 1})
    service.aretrieve = AsyncMock(
        return_value=RetrievalResult(answer="42", contexts={"chunks": []})
    )
    service.aanswer = AsyncMock(
        return_value=RetrievalResult(answer="The answer is 42", contexts={"chunks": []})
    )
    service.alist_ingested_files = AsyncMock(return_value=[])
    service.adelete_files = AsyncMock(return_value=[{"status": "deleted"}])
    return service


@pytest.fixture
def mock_manager(mock_service):
    """Create a mock RAGServiceManager that delegates to mock_service."""
    manager = AsyncMock()
    manager.aingest = mock_service.aingest
    manager.astart_ingest_job = AsyncMock(
        return_value={
            "job_id": "job-1",
            "workspace": "default",
            "source_type": "s3",
            "status": "queued",
        }
    )
    manager.get_ingest_job = AsyncMock(
        return_value={
            "job_id": "job-1",
            "workspace": "default",
            "source_type": "s3",
            "status": "running",
            "processed_items": 64,
        }
    )
    manager.aretrieve = mock_service.aretrieve
    manager.aanswer = mock_service.aanswer
    manager.list_ingested_files = mock_service.alist_ingested_files
    manager.delete_files = mock_service.adelete_files
    manager.list_workspaces = AsyncMock(return_value=["default"])
    manager.list_workspace_records = AsyncMock(
        return_value=[
            {
                "workspace": "default",
                "display_name": "default",
                "embedding_model": "voyage-multimodal-3.5",
                "created_at": None,
                "updated_at": None,
            }
        ]
    )
    manager.acreate_workspace = AsyncMock()
    manager.areset = AsyncMock(return_value={"workspaces": {"old_ws": {}}, "total_errors": 0})
    manager.is_ready = lambda: True
    manager.is_degraded = lambda: False
    manager.get_warnings = lambda: []
    manager.get_error_info = lambda: {"last_error": None, "timestamp": None, "retry_after": 30.0}
    manager.close = AsyncMock()
    return manager


@pytest.fixture
def _patch_manager(_api_app: FastAPI, mock_manager):
    """Set mock manager on app.state."""
    _api_app.state.manager = mock_manager
    yield
    if hasattr(_api_app.state, "manager"):
        del _api_app.state.manager


@pytest.fixture
async def client(_api_app: FastAPI):
    """Create httpx async client for testing."""
    transport = ASGITransport(app=_api_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


# ---------------------------------------------------------------------------
# TestAuthMiddleware
# ---------------------------------------------------------------------------


def test_api_server_has_no_eager_module_app() -> None:
    """The ASGI app is exposed through factories only."""
    import dlightrag.api.server as server

    assert not hasattr(server, "app")


class TestAuthMiddleware:
    """Test pluggable auth (none / simple / jwt)."""

    async def test_no_token_configured_passes(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager
        resp = await client.get("/health")
        assert resp.status_code == 200

    @pytest.mark.usefixtures("_patch_manager")
    async def test_simple_valid_token_passes(
        self, client: AsyncClient, mock_config_no_auth_override: DlightragConfig
    ) -> None:
        cfg = mock_config_no_auth_override
        cfg.api_auth_token = "secret-token"
        cfg.auth_mode = "simple"
        resp = await client.get(
            "/files",
            headers={"Authorization": "Bearer secret-token"},
        )
        assert resp.status_code == 200

    @pytest.mark.usefixtures("_patch_manager")
    async def test_simple_missing_auth_header_401(
        self, client: AsyncClient, mock_config_no_auth_override: DlightragConfig
    ) -> None:
        cfg = mock_config_no_auth_override
        cfg.api_auth_token = "secret-token"
        cfg.auth_mode = "simple"
        resp = await client.get("/files")
        assert resp.status_code == 401


class TestWorkspaceLifecycleAPI:
    """Workspace lifecycle API uses the durable manager registry."""

    async def test_list_workspaces_returns_records(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager

        resp = await client.get("/workspaces")

        assert resp.status_code == 200
        body = resp.json()
        assert body["workspaces"] == ["default"]
        assert body["records"][0]["display_name"] == "default"
        mock_manager.list_workspace_records.assert_awaited_once()

    async def test_create_workspace_registers_empty_workspace(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager
        mock_manager.list_workspaces = AsyncMock(return_value=["default"])

        resp = await client.post(
            "/workspaces",
            json={"workspace": "New Workspace", "display_name": "New Workspace"},
        )

        assert resp.status_code == 201
        assert resp.json() == {
            "workspace": "new_workspace",
            "display_name": "New Workspace",
            "created": True,
        }
        mock_manager.acreate_workspace.assert_awaited_once_with(
            "new_workspace",
            display_name="New Workspace",
        )

    async def test_create_workspace_rejects_duplicate(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager
        mock_manager.list_workspaces = AsyncMock(return_value=["default"])

        resp = await client.post("/workspaces", json={"workspace": "default"})

        assert resp.status_code == 409
        mock_manager.acreate_workspace.assert_not_awaited()

    async def test_delete_workspace_resets_and_removes_registry_row(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager

        resp = await client.delete("/workspaces/Old Workspace?keep_files=true&dry_run=true")

        assert resp.status_code == 200
        body = resp.json()
        assert body["workspace"] == "old_workspace"
        assert body["deleted"] is False
        mock_manager.areset.assert_awaited_once_with(
            workspace="Old Workspace",
            keep_files=True,
            dry_run=True,
        )

    @pytest.mark.usefixtures("_patch_manager")
    async def test_simple_wrong_scheme_401(
        self, client: AsyncClient, mock_config_no_auth_override: DlightragConfig
    ) -> None:
        cfg = mock_config_no_auth_override
        cfg.api_auth_token = "secret-token"
        cfg.auth_mode = "simple"
        resp = await client.get(
            "/files",
            headers={"Authorization": "Basic abc123"},
        )
        assert resp.status_code == 401

    @pytest.mark.usefixtures("_patch_manager")
    async def test_simple_invalid_token_403(
        self, client: AsyncClient, mock_config_no_auth_override: DlightragConfig
    ) -> None:
        cfg = mock_config_no_auth_override
        cfg.api_auth_token = "secret-token"
        cfg.auth_mode = "simple"
        resp = await client.get(
            "/files",
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp.status_code == 403

    @pytest.mark.parametrize(
        "method,path,body",
        [
            ("POST", "/ingest", {"source_type": "local", "path": "/tmp/f.pdf"}),
            ("POST", "/retrieve", {"query": "hello"}),
            ("POST", "/answer", {"query": "hello", "stream": False}),
            ("DELETE", "/files", {"filenames": ["f.pdf"]}),
        ],
    )
    async def test_endpoint_requires_auth(
        self,
        method: str,
        path: str,
        body: dict,
        client: AsyncClient,
        mock_config_no_auth_override: DlightragConfig,
    ) -> None:
        cfg = mock_config_no_auth_override
        cfg.api_auth_token = "secret-token"
        cfg.auth_mode = "simple"
        resp = await client.request(method, path, json=body)
        assert resp.status_code == 401

    @pytest.mark.usefixtures("_patch_manager")
    async def test_auth_mode_none_allows_all(
        self, client: AsyncClient, mock_config_no_auth_override: DlightragConfig
    ) -> None:
        cfg = mock_config_no_auth_override
        cfg.auth_mode = "none"
        resp = await client.get("/files")
        assert resp.status_code == 200

    @pytest.mark.usefixtures("_patch_manager")
    async def test_token_requires_explicit_simple_auth_mode(
        self, test_config: DlightragConfig
    ) -> None:
        """Setting api_auth_token without auth_mode is a config error."""
        test_config.api_auth_token = "my-token"
        test_config.auth_mode = "none"
        with pytest.raises(ValueError, match="auth_mode='simple'"):
            test_config._validate_auth_mode()


# ---------------------------------------------------------------------------
# TestJWTAuth
# ---------------------------------------------------------------------------

_JWT_SECRET = "test-jwt-secret-key-for-unit-tests"


class TestJWTAuth:
    """Test JWT authentication strategy."""

    @pytest.mark.usefixtures("_patch_manager")
    async def test_jwt_valid_token(
        self, client: AsyncClient, mock_config_no_auth_override: DlightragConfig
    ) -> None:
        cfg = mock_config_no_auth_override
        cfg.auth_mode = "jwt"
        cfg.jwt_secret = _JWT_SECRET
        cfg.jwt_algorithm = "HS256"

        payload = {
            "sub": "user-42",
            "exp": datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1),
        }
        token = jwt.encode(payload, _JWT_SECRET, algorithm="HS256")

        resp = await client.get(
            "/files",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200

    @pytest.mark.usefixtures("_patch_manager")
    async def test_jwt_expired_token(
        self, client: AsyncClient, mock_config_no_auth_override: DlightragConfig
    ) -> None:
        cfg = mock_config_no_auth_override
        cfg.auth_mode = "jwt"
        cfg.jwt_secret = _JWT_SECRET
        cfg.jwt_algorithm = "HS256"

        payload = {
            "sub": "user-42",
            "exp": datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=1),
        }
        token = jwt.encode(payload, _JWT_SECRET, algorithm="HS256")

        resp = await client.get(
            "/files",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# TestVerifyBearerToken
# ---------------------------------------------------------------------------


class TestVerifyBearerToken:
    """Unit tests for verify_bearer_token() — no FastAPI dependency needed."""

    def test_simple_valid_token(self, test_config: DlightragConfig) -> None:
        test_config.auth_mode = "simple"
        test_config.api_auth_token = "secret-token"
        ctx = verify_bearer_token("secret-token", test_config)
        assert ctx.user_id == "anonymous"
        assert ctx.auth_mode == "simple"

    def test_simple_invalid_token_raises_403(self, test_config: DlightragConfig) -> None:
        test_config.auth_mode = "simple"
        test_config.api_auth_token = "secret-token"
        with pytest.raises(HTTPException, match="Invalid token"):
            verify_bearer_token("wrong-token", test_config)

    def test_simple_empty_token_raises_403(self, test_config: DlightragConfig) -> None:
        test_config.auth_mode = "simple"
        test_config.api_auth_token = "secret-token"
        with pytest.raises(HTTPException, match="Invalid token"):
            verify_bearer_token("", test_config)

    def test_simple_default_user_id(self, test_config: DlightragConfig) -> None:
        test_config.auth_mode = "simple"
        test_config.api_auth_token = "secret-token"
        ctx = verify_bearer_token("secret-token", test_config, default_user_id="user-99")
        assert ctx.user_id == "user-99"

    def test_jwt_valid_token(self, test_config: DlightragConfig) -> None:
        test_config.auth_mode = "jwt"
        test_config.jwt_secret = _JWT_SECRET
        test_config.jwt_algorithm = "HS256"

        payload = {
            "sub": "user-42",
            "exp": datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1),
        }
        token = jwt.encode(payload, _JWT_SECRET, algorithm="HS256")
        ctx = verify_bearer_token(token, test_config)
        assert ctx.user_id == "user-42"
        assert ctx.auth_mode == "jwt"

    def test_jwt_expired_token_raises_401(self, test_config: DlightragConfig) -> None:
        test_config.auth_mode = "jwt"
        test_config.jwt_secret = _JWT_SECRET
        test_config.jwt_algorithm = "HS256"

        payload = {
            "sub": "user-42",
            "exp": datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=1),
        }
        token = jwt.encode(payload, _JWT_SECRET, algorithm="HS256")
        with pytest.raises(HTTPException, match="Token expired"):
            verify_bearer_token(token, test_config)

    def test_jwt_missing_sub_claim_raises_401(self, test_config: DlightragConfig) -> None:
        test_config.auth_mode = "jwt"
        test_config.jwt_secret = _JWT_SECRET
        test_config.jwt_algorithm = "HS256"

        payload = {
            "exp": datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1),
        }
        token = jwt.encode(payload, _JWT_SECRET, algorithm="HS256")
        with pytest.raises(HTTPException, match="missing 'sub' claim"):
            verify_bearer_token(token, test_config)

    def test_jwt_wrong_secret_raises_401(self, test_config: DlightragConfig) -> None:
        test_config.auth_mode = "jwt"
        test_config.jwt_secret = _JWT_SECRET
        test_config.jwt_algorithm = "HS256"

        payload = {"sub": "user-42"}
        token = jwt.encode(
            payload,
            "wrong-secret-different-key-for-unit-tests",
            algorithm="HS256",
        )
        with pytest.raises(HTTPException, match="Invalid token"):
            verify_bearer_token(token, test_config)


# ---------------------------------------------------------------------------
# TestIngestEndpoint
# ---------------------------------------------------------------------------


class TestIngestEndpoint:
    """Test /ingest validation and routing."""

    @pytest.mark.usefixtures("_patch_manager")
    async def test_local_requires_path(
        self, client: AsyncClient, mock_config: DlightragConfig
    ) -> None:
        resp = await client.post("/ingest", json={"source_type": "local"})
        assert resp.status_code == 422

    @pytest.mark.usefixtures("_patch_manager")
    async def test_azure_blob_requires_container(
        self, client: AsyncClient, mock_config: DlightragConfig
    ) -> None:
        resp = await client.post("/ingest", json={"source_type": "azure_blob"})
        assert resp.status_code == 422

    @pytest.mark.usefixtures("_patch_manager")
    async def test_s3_requires_bucket_and_key_or_prefix(
        self, client: AsyncClient, mock_config: DlightragConfig
    ) -> None:
        resp = await client.post("/ingest", json={"source_type": "s3"})
        assert resp.status_code == 422

    async def test_local_success(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager
        resp = await client.post(
            "/ingest",
            json={"source_type": "local", "path": "/data/file.pdf"},
        )
        assert resp.status_code == 200
        mock_manager.aingest.assert_awaited_once()

    async def test_local_directory_defaults_to_background_job(
        self,
        client: AsyncClient,
        mock_config: DlightragConfig,
        mock_manager,
        tmp_path,
    ) -> None:
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        mock_manager.astart_ingest_job.return_value = {
            "job_id": "job-1",
            "workspace": "default",
            "source_type": "local",
            "status": "queued",
        }
        app.state.manager = mock_manager

        resp = await client.post(
            "/ingest",
            json={"source_type": "local", "path": str(docs_dir)},
        )

        assert resp.status_code == 202
        body = resp.json()
        assert body["job_id"] == "job-1"
        assert body["status_url"] == "/ingest/jobs/job-1"
        mock_manager.astart_ingest_job.assert_awaited_once_with(
            "default",
            source_type="local",
            path=str(docs_dir),
        )
        mock_manager.aingest.assert_not_awaited()

    async def test_single_s3_key_can_force_background_job(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager

        resp = await client.post(
            "/ingest",
            json={
                "source_type": "s3",
                "bucket": "my-bucket",
                "key": "docs/file.pdf",
                "wait": False,
            },
        )

        assert resp.status_code == 202
        assert resp.json()["job_id"] == "job-1"
        mock_manager.astart_ingest_job.assert_awaited_once_with(
            "default",
            source_type="s3",
            bucket="my-bucket",
            key="docs/file.pdf",
        )
        mock_manager.aingest.assert_not_awaited()

    async def test_ingest_forwards_metadata_contract(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager
        resp = await client.post(
            "/ingest",
            json={
                "source_type": "local",
                "path": "/data/file.pdf",
                "title": "Field Notes",
                "author": "Ada",
                "metadata": {"project": "apollo"},
                "metadata_policy": "reject_unknown",
            },
        )
        assert resp.status_code == 200
        call_kwargs = mock_manager.aingest.call_args.kwargs
        assert call_kwargs["title"] == "Field Notes"
        assert call_kwargs["author"] == "Ada"
        assert call_kwargs["metadata"] == {"project": "apollo"}
        assert call_kwargs["metadata_policy"] == "reject_unknown"

    async def test_azure_blob_success(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager
        resp = await client.post(
            "/ingest",
            json={
                "source_type": "azure_blob",
                "container_name": "my-container",
                "blob_path": "docs/file.pdf",
            },
        )
        assert resp.status_code == 200

    async def test_s3_prefix_success(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager
        resp = await client.post(
            "/ingest",
            json={
                "source_type": "s3",
                "bucket": "my-bucket",
                "prefix": "docs/",
            },
        )
        assert resp.status_code == 202
        body = resp.json()
        assert body["job_id"] == "job-1"
        assert body["status_url"] == "/ingest/jobs/job-1"
        mock_manager.astart_ingest_job.assert_awaited_once_with(
            "default",
            source_type="s3",
            bucket="my-bucket",
            prefix="docs/",
        )
        mock_manager.aingest.assert_not_awaited()

    async def test_get_ingest_job_status(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager

        resp = await client.get("/ingest/jobs/job-1")

        assert resp.status_code == 200
        assert resp.json()["processed_items"] == 64
        mock_manager.get_ingest_job.assert_awaited_once_with("job-1")

    @pytest.mark.usefixtures("_patch_manager")
    async def test_s3_key_and_prefix_mutually_exclusive(
        self, client: AsyncClient, mock_config: DlightragConfig
    ) -> None:
        resp = await client.post(
            "/ingest",
            json={
                "source_type": "s3",
                "bucket": "my-bucket",
                "key": "docs/file.pdf",
                "prefix": "docs/",
            },
        )
        assert resp.status_code == 400

    @pytest.mark.usefixtures("_patch_manager")
    async def test_azure_blob_path_and_prefix_mutually_exclusive(
        self, client: AsyncClient, mock_config: DlightragConfig
    ) -> None:
        resp = await client.post(
            "/ingest",
            json={
                "source_type": "azure_blob",
                "container_name": "c",
                "blob_path": "docs/file.pdf",
                "prefix": "docs/",
            },
        )
        assert resp.status_code == 400

    async def test_ingest_with_workspace(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager
        resp = await client.post(
            "/ingest",
            json={"source_type": "local", "path": "/data/file.pdf", "workspace": "project-x"},
        )
        assert resp.status_code == 200
        call_kwargs = mock_manager.aingest.call_args
        assert call_kwargs[0][0] == "project_x"  # normalized: hyphens → underscores

    async def test_ingest_service_unavailable_503(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        mock_manager.aingest = AsyncMock(side_effect=RAGServiceUnavailableError("RAG not ready"))
        app.state.manager = mock_manager
        resp = await client.post(
            "/ingest",
            json={"source_type": "local", "path": "/data/file.pdf"},
        )
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# TestRetrieveEndpoint
# ---------------------------------------------------------------------------


class TestRetrieveEndpoint:
    """Test /retrieve endpoint."""

    async def test_retrieve_success(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager
        resp = await client.post("/retrieve", json={"query": "What is RAG?"})
        assert resp.status_code == 200
        body = resp.json()
        assert "answer" in body
        assert "contexts" in body
        assert "sources" in body

    async def test_retrieve_rejects_mode_field(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager
        resp = await client.post(
            "/retrieve",
            json={"query": "hello", "mode": "local"},
        )
        assert resp.status_code == 422
        mock_manager.aretrieve.assert_not_called()

    async def test_retrieve_forwards_multimodal_content(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager
        multimodal_content = [{"type": "image", "img_path": "/tmp/query.png"}]
        resp = await client.post(
            "/retrieve",
            json={"query": "Find the matching drawing", "multimodal_content": multimodal_content},
        )
        assert resp.status_code == 200
        assert mock_manager.aretrieve.call_args.kwargs["multimodal_content"] == multimodal_content


# ---------------------------------------------------------------------------
# TestHealthEndpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Test /health endpoint."""

    async def test_health_returns_status(
        self,
        client: AsyncClient,
        mock_config: DlightragConfig,
        mock_manager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class MockConnection:
            async def fetchval(self, query: str) -> int:
                return 1

            async def close(self) -> None:
                return None

        async def fake_connect(**kwargs):
            return MockConnection()

        import asyncpg

        monkeypatch.setattr(asyncpg, "connect", fake_connect)
        app.state.manager = mock_manager
        resp = await client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert "rag_initialized" in body
        assert "storage" in body


# ---------------------------------------------------------------------------
# TestHealthEndpointEnhanced
# ---------------------------------------------------------------------------


class TestHealthEndpointEnhanced:
    """Test enhanced /health endpoint with degraded state."""

    async def test_health_shows_degraded(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        mock_manager.is_degraded = lambda: True
        mock_manager.get_warnings = lambda: ["Embedding unreachable"]
        app.state.manager = mock_manager
        resp = await client.get("/health")
        body = resp.json()
        assert body["status"] == "degraded"
        assert "Embedding unreachable" in body["warnings"]

    async def test_health_healthy_no_warnings(
        self,
        client: AsyncClient,
        mock_config: DlightragConfig,
        mock_manager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class MockConnection:
            async def fetchval(self, query: str) -> int:
                return 1

            async def close(self) -> None:
                return None

        async def fake_connect(**kwargs):
            return MockConnection()

        import asyncpg

        monkeypatch.setattr(asyncpg, "connect", fake_connect)
        mock_manager.is_degraded = lambda: False
        mock_manager.get_warnings = lambda: []
        app.state.manager = mock_manager
        resp = await client.get("/health")
        body = resp.json()
        assert body["status"] == "healthy"
        assert "warnings" not in body


# ---------------------------------------------------------------------------
# TestDeleteEndpoint
# ---------------------------------------------------------------------------


class TestDeleteEndpoint:
    """Test DELETE /files endpoint."""

    async def test_delete_by_filenames(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager
        resp = await client.request(
            "DELETE",
            "/files",
            json={"filenames": ["report.pdf"]},
        )
        assert resp.status_code == 200
        mock_manager.delete_files.assert_awaited_once()

    async def test_delete_by_file_paths(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager
        resp = await client.request(
            "DELETE",
            "/files",
            json={"file_paths": ["/storage/report.pdf"]},
        )
        assert resp.status_code == 200

    async def test_delete_with_workspace(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager
        resp = await client.request(
            "DELETE",
            "/files",
            json={"filenames": ["report.pdf"], "workspace": "project-y"},
        )
        assert resp.status_code == 200
        call_kwargs = mock_manager.delete_files.call_args
        assert call_kwargs[0][0] == "project_y"  # normalized: hyphens → underscores


# ---------------------------------------------------------------------------
# TestAnswerEndpoint
# ---------------------------------------------------------------------------


class TestAnswerEndpoint:
    """Test /answer endpoint."""

    async def test_answer_success(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager
        resp = await client.post("/answer", json={"query": "What is RAG?", "stream": False})
        assert resp.status_code == 200
        body = resp.json()
        assert "answer" in body
        assert "contexts" in body
        assert "sources" in body
        assert body["answer"] == "The answer is 42"

    async def test_answer_forwards_explicit_filters(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager
        resp = await client.post(
            "/answer",
            json={
                "query": "What did Ada write?",
                "stream": False,
                "filters": {"doc_author": "Ada"},
            },
        )
        assert resp.status_code == 200
        filters = mock_manager.aanswer.call_args.kwargs["filters"]
        assert filters.doc_author == "Ada"

    async def test_answer_forwards_answer_context_limits(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager
        resp = await client.post(
            "/answer",
            json={
                "query": "What is RAG?",
                "stream": False,
                "answer_candidate_top_k": 12,
                "answer_context_top_k": 4,
            },
        )
        assert resp.status_code == 200
        call_kwargs = mock_manager.aanswer.call_args.kwargs
        assert call_kwargs["answer_candidate_top_k"] == 12
        assert call_kwargs["answer_context_top_k"] == 4

    async def test_answer_service_unavailable_503(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        mock_manager.aanswer = AsyncMock(side_effect=RAGServiceUnavailableError("RAG not ready"))
        app.state.manager = mock_manager
        resp = await client.post("/answer", json={"query": "hello", "stream": False})
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# TestFilesEndpoint
# ---------------------------------------------------------------------------


class TestFilesEndpoint:
    """Test GET /files endpoint."""

    @pytest.mark.usefixtures("_patch_manager")
    async def test_list_files_success(
        self, client: AsyncClient, mock_config: DlightragConfig
    ) -> None:
        resp = await client.get("/files")
        assert resp.status_code == 200
        body = resp.json()
        assert "files" in body
        assert "count" in body

    async def test_list_files_count_matches(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        mock_manager.list_ingested_files = AsyncMock(return_value=["a.pdf", "b.pdf", "c.pdf"])
        app.state.manager = mock_manager
        resp = await client.get("/files")
        assert resp.status_code == 200
        body = resp.json()
        assert body["count"] == 3
        assert len(body["files"]) == 3

    async def test_list_files_with_workspace(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager
        resp = await client.get("/files?workspace=project-z")
        assert resp.status_code == 200
        call_kwargs = mock_manager.list_ingested_files.call_args
        assert call_kwargs[0][0] == "project_z"  # normalized: hyphens → underscores


# ---------------------------------------------------------------------------
# TestAnswerStreamMode
# ---------------------------------------------------------------------------


class TestAnswerStreamMode:
    """Test POST /answer with stream=true SSE mode."""

    async def test_stream_returns_sse_content_type(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        async def mock_tokens():
            for t in ["Hello", " world"]:
                yield t

        mock_manager.aanswer_stream = AsyncMock(return_value=({"chunks": []}, mock_tokens()))
        app.state.manager = mock_manager
        resp = await client.post("/answer", json={"query": "test", "stream": True})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    async def test_stream_forwards_explicit_filters(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        async def mock_tokens():
            yield "Hello"

        mock_manager.aanswer_stream = AsyncMock(return_value=({"chunks": []}, mock_tokens()))
        app.state.manager = mock_manager
        resp = await client.post(
            "/answer",
            json={
                "query": "Stream filtered",
                "stream": True,
                "filters": {"doc_title": "Manual"},
            },
        )
        assert resp.status_code == 200
        filters = mock_manager.aanswer_stream.call_args.kwargs["filters"]
        assert filters.doc_title == "Manual"

    async def test_stream_forwards_answer_context_limits(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        async def mock_tokens():
            yield "Hello"

        mock_manager.aanswer_stream = AsyncMock(return_value=({"chunks": []}, mock_tokens()))
        app.state.manager = mock_manager
        resp = await client.post(
            "/answer",
            json={
                "query": "Stream with limits",
                "stream": True,
                "answer_candidate_top_k": 16,
                "answer_context_top_k": 5,
            },
        )
        assert resp.status_code == 200
        call_kwargs = mock_manager.aanswer_stream.call_args.kwargs
        assert call_kwargs["answer_candidate_top_k"] == 16
        assert call_kwargs["answer_context_top_k"] == 5

    async def test_stream_event_sequence(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        """Verify context -> token(s) -> done event order."""

        async def mock_tokens():
            for t in ["Hi", " there"]:
                yield t

        mock_manager.aanswer_stream = AsyncMock(
            return_value=({"chunks": [{"id": "c1"}]}, mock_tokens())
        )
        app.state.manager = mock_manager
        resp = await client.post("/answer", json={"query": "test", "stream": True})
        lines = [line for line in resp.text.split("\n") if line.startswith("data: ")]

        import json as json_mod

        events = [json_mod.loads(line.removeprefix("data: ")) for line in lines]
        assert events[0]["type"] == "context"
        assert events[0]["data"] == {"chunks": [{"id": "c1"}]}
        assert events[-1]["type"] == "done"
        assert events[-1]["answer"] == "Hi there"
        token_events = [e for e in events if e["type"] == "token"]
        assert len(token_events) == 2
        assert token_events[0]["content"] == "Hi"
        assert token_events[1]["content"] == " there"

    async def test_stream_error_during_iteration(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        """Error mid-stream produces error event."""

        async def mock_tokens():
            yield "start"
            raise RuntimeError("LLM exploded")

        mock_manager.aanswer_stream = AsyncMock(return_value=({"chunks": []}, mock_tokens()))
        app.state.manager = mock_manager
        resp = await client.post("/answer", json={"query": "test", "stream": True})
        lines = [line for line in resp.text.split("\n") if line.startswith("data: ")]

        import json as json_mod

        events = [json_mod.loads(line.removeprefix("data: ")) for line in lines]
        error_events = [e for e in events if e["type"] == "error"]
        assert len(error_events) == 1
        assert "Internal server error" in error_events[0]["message"]

    async def test_stream_service_unavailable_503(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        """Pre-stream errors return normal HTTP 503."""
        mock_manager.aanswer_stream = AsyncMock(
            side_effect=RAGServiceUnavailableError("RAG not ready")
        )
        app.state.manager = mock_manager
        resp = await client.post("/answer", json={"query": "hello", "stream": True})
        assert resp.status_code == 503

    async def test_stream_false_returns_json(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        """stream=false returns normal JSON response."""
        app.state.manager = mock_manager
        resp = await client.post("/answer", json={"query": "test", "stream": False})
        assert resp.status_code == 200
        body = resp.json()
        assert "answer" in body
        assert body["answer"] == "The answer is 42"

    async def test_missing_stream_defaults_to_streaming(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        """REST /answer streams by default; stream=false opts into JSON."""

        async def mock_tokens():
            yield "Hello"

        mock_manager.aanswer_stream = AsyncMock(return_value=({"chunks": []}, mock_tokens()))
        app.state.manager = mock_manager
        resp = await client.post("/answer", json={"query": "test"})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        mock_manager.aanswer_stream.assert_awaited_once()
