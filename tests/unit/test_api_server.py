# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for FastAPI REST server endpoints and auth middleware."""

import datetime
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import AsyncMock

import jwt
import pytest
from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient

from dlightrag.api import auth as auth_module
from dlightrag.api.auth import UserContext, get_current_user, verify_bearer_token
from dlightrag.api.server import create_app
from dlightrag.citations.schemas import ChunkSnippet, SourceReference
from dlightrag.config import (
    AccessControlConfig,
    AccessControlRuleConfig,
    DlightragConfig,
    set_config,
)
from dlightrag.core.client_contracts import IngestSpec
from dlightrag.core.retrieval.protocols import RetrievalResult
from dlightrag.core.servicemanager import RAGServiceUnavailableError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_ANON = UserContext(user_id="anonymous", auth_mode="none")
app: FastAPI


def _finance_source() -> SourceReference:
    return SourceReference(
        id="1",
        title="report.pdf",
        source_uri="s3://bucket/report.pdf",
        workspace="finance",
        document_id="doc-report",
        download_locator="s3://bucket/report.pdf",
    )


def _finance_source_context() -> dict[str, object]:
    return {
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


@pytest.fixture
def _api_app(test_config: DlightragConfig) -> Iterator[FastAPI]:
    """Create the API app after test_config has installed the singleton."""
    global app
    app = create_app(include_web=False)
    yield app
    app.dependency_overrides.clear()
    if hasattr(app.state, "manager"):
        del app.state.manager


@pytest.fixture
def mock_config(_api_app: FastAPI, test_config: DlightragConfig) -> Iterator[DlightragConfig]:
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
def mock_manager(mock_service, test_config):
    """Create a mock RAGServiceManager that delegates to mock_service."""
    manager = AsyncMock()
    manager.config = test_config
    manager.aingest = mock_service.aingest
    manager.astart_ingest_job = AsyncMock(
        return_value={
            "job_id": "job-1",
            "workspace": "default",
            "source_type": "s3",
            "status": "queued",
        }
    )
    manager.aget_ingest_job = AsyncMock(
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
    manager.alist_ingested_files = mock_service.alist_ingested_files
    manager.adelete_files = mock_service.adelete_files
    manager.alist_workspaces = AsyncMock(return_value=["default"])
    manager.alist_workspace_records = AsyncMock(
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
    from dlightrag.core.answer.capability import AnswerImageCapability

    manager.answer_image_capability = AnswerImageCapability(
        status="supported",
        configured_ceiling=8,
        effective_max_images=8,
        provider="test",
        base_url=None,
        model="test-model",
        failure_kind=None,
    )
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

    async def test_routes_use_app_scoped_config_after_singleton_changes(
        self,
        client: AsyncClient,
        _api_app: FastAPI,
        mock_config: DlightragConfig,
        mock_manager,
    ) -> None:
        _api_app.state.manager = mock_manager
        mock_config.workspace = "app_ws"
        singleton_config = mock_config.model_copy(deep=True)
        singleton_config.workspace = "singleton_ws"
        set_config(singleton_config)

        resp = await client.get("/files")

        assert resp.status_code == 200
        mock_manager.alist_ingested_files.assert_awaited_once_with("app_ws")

    async def test_list_workspaces_returns_records(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager

        resp = await client.get("/workspaces")

        assert resp.status_code == 200
        body = resp.json()
        assert body["workspaces"] == ["default"]
        assert body["records"][0]["display_name"] == "default"
        mock_manager.alist_workspace_records.assert_awaited_once()

    async def test_create_workspace_registers_empty_workspace(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager
        mock_manager.alist_workspaces = AsyncMock(return_value=["default"])

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
        mock_manager.alist_workspaces = AsyncMock(return_value=["default"])

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
    async def test_simple_invalid_token_401(
        self, client: AsyncClient, mock_config_no_auth_override: DlightragConfig
    ) -> None:
        cfg = mock_config_no_auth_override
        cfg.api_auth_token = "secret-token"
        cfg.auth_mode = "simple"
        resp = await client.get(
            "/files",
            headers={"Authorization": "Bearer wrong-token"},
        )
        assert resp.status_code == 401

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

_JWT_VERIFICATION_KEY = "test-jwt-verification-key-for-unit-tests"


class TestJWTAuth:
    """Test JWT authentication strategy."""

    @pytest.mark.usefixtures("_patch_manager")
    async def test_jwt_valid_token(
        self, client: AsyncClient, mock_config_no_auth_override: DlightragConfig
    ) -> None:
        cfg = mock_config_no_auth_override
        cfg.auth_mode = "jwt"
        cfg.jwt_verification_key = _JWT_VERIFICATION_KEY
        cfg.jwt_algorithm = "HS256"

        payload = {
            "sub": "user-42",
            "exp": datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1),
        }
        token = jwt.encode(payload, _JWT_VERIFICATION_KEY, algorithm="HS256")

        resp = await client.get(
            "/files",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200

    @pytest.mark.usefixtures("_patch_manager")
    async def test_jwt_claims_access_control_denies_unmapped_workspace(
        self, client: AsyncClient, mock_config_no_auth_override: DlightragConfig, mock_manager
    ) -> None:
        cfg = mock_config_no_auth_override
        cfg.auth_mode = "jwt"
        cfg.jwt_verification_key = _JWT_VERIFICATION_KEY
        cfg.jwt_algorithm = "HS256"
        cfg.access_control = AccessControlConfig(
            mode="jwt_claims",
            rules=[
                AccessControlRuleConfig(
                    claim="groups",
                    value="finance-rag-readers",
                    workspaces=["finance"],
                    actions=["workspace.query"],
                )
            ],
        )
        token = jwt.encode(
            {
                "sub": "user-42",
                "groups": ["legal-rag-readers"],
                "exp": datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1),
            },
            _JWT_VERIFICATION_KEY,
            algorithm="HS256",
        )

        resp = await client.post(
            "/retrieve",
            json={"query": "hello", "workspaces": ["finance"]},
            headers={"Authorization": f"Bearer {token}"},
        )

        assert resp.status_code == 403
        mock_manager.aretrieve.assert_not_awaited()

    @pytest.mark.usefixtures("_patch_manager")
    async def test_jwt_claims_access_control_allows_mapped_workspace(
        self, client: AsyncClient, mock_config_no_auth_override: DlightragConfig, mock_manager
    ) -> None:
        cfg = mock_config_no_auth_override
        cfg.auth_mode = "jwt"
        cfg.jwt_verification_key = _JWT_VERIFICATION_KEY
        cfg.jwt_algorithm = "HS256"
        cfg.access_control = AccessControlConfig(
            mode="jwt_claims",
            rules=[
                AccessControlRuleConfig(
                    claim="groups",
                    value="finance-rag-readers",
                    workspaces=["finance"],
                    actions=["workspace.query"],
                )
            ],
        )
        token = jwt.encode(
            {
                "sub": "user-42",
                "groups": ["finance-rag-readers"],
                "exp": datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1),
            },
            _JWT_VERIFICATION_KEY,
            algorithm="HS256",
        )

        resp = await client.post(
            "/retrieve",
            json={"query": "hello", "workspaces": ["finance"]},
            headers={"Authorization": f"Bearer {token}"},
        )

        assert resp.status_code == 200
        mock_manager.aretrieve.assert_awaited_once()

    @pytest.mark.usefixtures("_patch_manager")
    @pytest.mark.parametrize(
        ("groups", "expected_status"),
        [
            (["finance-rag-readers"], 200),
            (["legal-rag-readers"], 403),
        ],
    )
    async def test_all_workspaces_is_relative_to_query_authorization(
        self,
        client: AsyncClient,
        _api_app: FastAPI,
        mock_config_no_auth_override: DlightragConfig,
        mock_manager,
        groups: list[str],
        expected_status: int,
    ) -> None:
        registered = [f"ws_{index:02d}" for index in range(14)]
        allowed = registered[:10]
        mock_manager.alist_workspace_records.return_value = [
            {"workspace": workspace} for workspace in registered
        ]
        mock_config_no_auth_override.access_control = AccessControlConfig(
            mode="jwt_claims",
            rules=[
                AccessControlRuleConfig(
                    claim="groups",
                    value="finance-rag-readers",
                    workspaces=allowed,
                    actions=["workspace.query"],
                )
            ],
        )
        _api_app.dependency_overrides[get_current_user] = lambda: UserContext(
            user_id="alice",
            auth_mode="jwt",
            claims={"groups": groups},
        )

        response = await client.post(
            "/answer",
            json={"query": "hello", "stream": False, "all_workspaces": True},
        )

        assert response.status_code == expected_status
        if expected_status == 200:
            assert mock_manager.aanswer.await_args.kwargs["workspaces"] == allowed
        else:
            mock_manager.aanswer.assert_not_awaited()

    @pytest.mark.usefixtures("_patch_manager")
    async def test_jwt_expired_token(
        self, client: AsyncClient, mock_config_no_auth_override: DlightragConfig
    ) -> None:
        cfg = mock_config_no_auth_override
        cfg.auth_mode = "jwt"
        cfg.jwt_verification_key = _JWT_VERIFICATION_KEY
        cfg.jwt_algorithm = "HS256"

        payload = {
            "sub": "user-42",
            "exp": datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=1),
        }
        token = jwt.encode(payload, _JWT_VERIFICATION_KEY, algorithm="HS256")

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
        test_config.jwt_verification_key = _JWT_VERIFICATION_KEY
        test_config.jwt_algorithm = "HS256"

        payload = {
            "sub": "user-42",
            "exp": datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1),
        }
        token = jwt.encode(payload, _JWT_VERIFICATION_KEY, algorithm="HS256")
        ctx = verify_bearer_token(token, test_config)
        assert ctx.user_id == "user-42"
        assert ctx.auth_mode == "jwt"

    def test_jwt_jwks_url_validates_issuer_and_audience(
        self,
        test_config: DlightragConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        test_config.auth_mode = "jwt"
        test_config.jwt_jwks_url = "https://login.example.com/discovery/keys"
        test_config.jwt_issuer = "https://login.example.com/tenant/v2.0"
        test_config.jwt_audience = "api://dlightrag"
        test_config.jwt_algorithm = "HS256"

        payload = {
            "sub": "user-42",
            "iss": test_config.jwt_issuer,
            "aud": test_config.jwt_audience,
            "groups": ["finance-rag-readers"],
            "exp": datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1),
        }
        jwks_secret = "jwks-secret-for-unit-tests-32-bytes"
        token = jwt.encode(payload, jwks_secret, algorithm="HS256", headers={"kid": "key-1"})

        class FakeJwksClient:
            def get_signing_key_from_jwt(self, raw_token: str):
                assert raw_token == token
                return SimpleNamespace(key=jwks_secret)

        monkeypatch.setattr(auth_module, "_jwks_client", lambda _url: FakeJwksClient())

        ctx = verify_bearer_token(token, test_config)

        assert ctx.user_id == "user-42"
        assert ctx.claims["groups"] == ["finance-rag-readers"]

    def test_jwt_jwks_url_rejects_wrong_audience(
        self,
        test_config: DlightragConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        test_config.auth_mode = "jwt"
        test_config.jwt_jwks_url = "https://login.example.com/discovery/keys"
        test_config.jwt_issuer = "https://login.example.com/tenant/v2.0"
        test_config.jwt_audience = "api://dlightrag"
        test_config.jwt_algorithm = "HS256"

        payload = {
            "sub": "user-42",
            "iss": test_config.jwt_issuer,
            "aud": "api://other",
            "exp": datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1),
        }
        jwks_secret = "jwks-secret-for-unit-tests-32-bytes"
        token = jwt.encode(payload, jwks_secret, algorithm="HS256", headers={"kid": "key-1"})

        class FakeJwksClient:
            def get_signing_key_from_jwt(self, raw_token: str):
                assert raw_token == token
                return SimpleNamespace(key=jwks_secret)

        monkeypatch.setattr(auth_module, "_jwks_client", lambda _url: FakeJwksClient())

        with pytest.raises(HTTPException, match="Invalid token"):
            verify_bearer_token(token, test_config)

    def test_jwt_jwks_url_accepts_any_of_multiple_audiences(
        self,
        test_config: DlightragConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        test_config.auth_mode = "jwt"
        test_config.jwt_jwks_url = "https://login.example.com/discovery/keys"
        test_config.jwt_issuer = "https://login.example.com/tenant/v2.0"
        test_config.jwt_audience = ["api://dlightrag", "proxy-client-id"]
        test_config.jwt_algorithm = "HS256"

        payload = {
            "sub": "user-42",
            "iss": test_config.jwt_issuer,
            "aud": "proxy-client-id",
            "exp": datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1),
        }
        jwks_secret = "jwks-secret-for-unit-tests-32-bytes"
        token = jwt.encode(payload, jwks_secret, algorithm="HS256", headers={"kid": "key-1"})

        class FakeJwksClient:
            def get_signing_key_from_jwt(self, raw_token: str):
                assert raw_token == token
                return SimpleNamespace(key=jwks_secret)

        monkeypatch.setattr(auth_module, "_jwks_client", lambda _url: FakeJwksClient())

        ctx = verify_bearer_token(token, test_config)

        assert ctx.user_id == "user-42"

    def test_jwt_expired_token_raises_401(self, test_config: DlightragConfig) -> None:
        test_config.auth_mode = "jwt"
        test_config.jwt_verification_key = _JWT_VERIFICATION_KEY
        test_config.jwt_algorithm = "HS256"

        payload = {
            "sub": "user-42",
            "exp": datetime.datetime.now(datetime.UTC) - datetime.timedelta(hours=1),
        }
        token = jwt.encode(payload, _JWT_VERIFICATION_KEY, algorithm="HS256")
        with pytest.raises(HTTPException, match="Token expired"):
            verify_bearer_token(token, test_config)

    def test_jwt_missing_sub_claim_raises_401(self, test_config: DlightragConfig) -> None:
        test_config.auth_mode = "jwt"
        test_config.jwt_verification_key = _JWT_VERIFICATION_KEY
        test_config.jwt_algorithm = "HS256"

        payload = {
            "exp": datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=1),
        }
        token = jwt.encode(payload, _JWT_VERIFICATION_KEY, algorithm="HS256")
        with pytest.raises(HTTPException, match="missing 'sub' claim"):
            verify_bearer_token(token, test_config)

    def test_jwt_wrong_verification_key_raises_401(self, test_config: DlightragConfig) -> None:
        test_config.auth_mode = "jwt"
        test_config.jwt_verification_key = _JWT_VERIFICATION_KEY
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
    async def test_s3_requires_bucket(
        self, client: AsyncClient, mock_config: DlightragConfig
    ) -> None:
        resp = await client.post("/ingest", json={"source_type": "s3"})
        assert resp.status_code == 422

    @pytest.mark.usefixtures("_patch_manager")
    async def test_url_requires_url_or_urls(
        self, client: AsyncClient, mock_config: DlightragConfig
    ) -> None:
        resp = await client.post("/ingest", json={"source_type": "url"})
        assert resp.status_code == 422

    @pytest.mark.usefixtures("_patch_manager")
    async def test_url_rejects_both_url_and_urls(
        self, client: AsyncClient, mock_config: DlightragConfig
    ) -> None:
        resp = await client.post(
            "/ingest",
            json={
                "source_type": "url",
                "url": "https://api.bynder.com/docs/getting-started",
                "urls": ["https://api.bynder.com/docs/other"],
            },
        )
        assert resp.status_code == 422

    async def test_local_defaults_to_background_job(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        path = mock_config.input_dir_path / "default" / "file.pdf"
        app.state.manager = mock_manager
        resp = await client.post(
            "/ingest",
            json={"source_type": "local", "path": "file.pdf"},
        )
        assert resp.status_code == 202
        assert resp.json()["job_id"] == "job-1"
        mock_manager.astart_ingest_job.assert_awaited_once_with(
            "default",
            IngestSpec(source_type="local", path=str(path)),
        )
        mock_manager.aingest.assert_not_awaited()

    async def test_local_path_must_be_under_input_dir(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager
        resp = await client.post(
            "/ingest",
            json={"source_type": "local", "path": "/data/file.pdf"},
        )
        assert resp.status_code == 400
        assert "relative to input_dir" in resp.json()["detail"]
        mock_manager.astart_ingest_job.assert_not_awaited()
        mock_manager.aingest.assert_not_awaited()

    async def test_local_path_rejects_traversal(self, client: AsyncClient, mock_manager) -> None:
        app.state.manager = mock_manager
        resp = await client.post(
            "/ingest",
            json={
                "source_type": "local",
                "path": "../default/file.pdf",
                "workspace": "project-x",
            },
        )

        assert resp.status_code == 400
        assert "relative to input_dir" in resp.json()["detail"]
        mock_manager.astart_ingest_job.assert_not_awaited()
        mock_manager.aingest.assert_not_awaited()

    async def test_local_directory_defaults_to_background_job(
        self,
        client: AsyncClient,
        mock_config: DlightragConfig,
        mock_manager,
    ) -> None:
        docs_dir = mock_config.input_dir_path / "default" / "docs"
        docs_dir.mkdir(parents=True)
        mock_manager.astart_ingest_job.return_value = {
            "job_id": "job-1",
            "workspace": "default",
            "source_type": "local",
            "status": "queued",
        }
        app.state.manager = mock_manager

        resp = await client.post(
            "/ingest",
            json={"source_type": "local", "path": "docs"},
        )

        assert resp.status_code == 202
        body = resp.json()
        assert body["job_id"] == "job-1"
        assert body["status_url"] == "/ingest/jobs/job-1"
        mock_manager.astart_ingest_job.assert_awaited_once_with(
            "default",
            IngestSpec(source_type="local", path=str(docs_dir)),
        )
        mock_manager.aingest.assert_not_awaited()

    async def test_single_s3_key_defaults_to_background_job(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager

        resp = await client.post(
            "/ingest",
            json={
                "source_type": "s3",
                "bucket": "my-bucket",
                "s3_key": "docs/file.pdf",
            },
        )

        assert resp.status_code == 202
        assert resp.json()["job_id"] == "job-1"
        mock_manager.astart_ingest_job.assert_awaited_once_with(
            "default",
            IngestSpec(source_type="s3", bucket="my-bucket", s3_key="docs/file.pdf"),
        )
        mock_manager.aingest.assert_not_awaited()

    async def test_s3_bucket_only_ingests_whole_bucket(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager

        resp = await client.post(
            "/ingest",
            json={"source_type": "s3", "bucket": "my-bucket"},
        )

        assert resp.status_code == 202
        assert resp.json()["job_id"] == "job-1"
        mock_manager.astart_ingest_job.assert_awaited_once_with(
            "default",
            IngestSpec(source_type="s3", bucket="my-bucket"),
        )
        mock_manager.aingest.assert_not_awaited()

    async def test_url_ingest_defaults_to_background_job(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager

        resp = await client.post(
            "/ingest",
            json={
                "source_type": "url",
                "url": "https://api.bynder.com/docs/getting-started",
                "filename": "getting-started.html",
            },
        )

        assert resp.status_code == 202
        mock_manager.astart_ingest_job.assert_awaited_once_with(
            "default",
            IngestSpec(
                source_type="url",
                url="https://api.bynder.com/docs/getting-started",
                filename="getting-started.html",
            ),
        )
        mock_manager.aingest.assert_not_awaited()

    async def test_url_ingest_accepts_stable_source_uri(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager

        resp = await client.post(
            "/ingest",
            json={
                "source_type": "url",
                "url": "https://cdn.example.com/download?id=asset-1&signature=secret",
                "filename": "asset.pdf",
                "source_uri": "bynder://asset/asset-1",
            },
        )

        assert resp.status_code == 202
        mock_manager.astart_ingest_job.assert_awaited_once_with(
            "default",
            IngestSpec(
                source_type="url",
                url="https://cdn.example.com/download?id=asset-1&signature=secret",
                filename="asset.pdf",
                source_uri="bynder://asset/asset-1",
            ),
        )
        mock_manager.aingest.assert_not_awaited()

    async def test_url_ingest_accepts_download_uri(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager

        resp = await client.post(
            "/ingest",
            json={
                "source_type": "url",
                "url": "https://fetch.example.com/download?id=asset-1&signature=secret",
                "filename": "asset.pdf",
                "source_uri": "bynder://asset/asset-1",
                "download_uri": "https://cdn.example.com/assets/asset-1.pdf",
            },
        )

        assert resp.status_code == 202
        mock_manager.astart_ingest_job.assert_awaited_once_with(
            "default",
            IngestSpec(
                source_type="url",
                url="https://fetch.example.com/download?id=asset-1&signature=secret",
                filename="asset.pdf",
                source_uri="bynder://asset/asset-1",
                download_uri="https://cdn.example.com/assets/asset-1.pdf",
            ),
        )
        mock_manager.aingest.assert_not_awaited()

    async def test_blob_upload_stages_file_for_local_ingest(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        mock_manager.config = mock_config
        mock_manager.astart_ingest_job.return_value = {
            "job_id": "job-1",
            "workspace": "default",
            "source_type": "local",
            "status": "queued",
        }
        app.state.manager = mock_manager

        resp = await client.post(
            "/ingest/blob",
            files={"file": ("report.pdf", b"%PDF-fake", "application/pdf")},
        )

        assert resp.status_code == 202
        body = resp.json()
        assert body["job_id"] == "job-1"
        assert body["filename"] == "report.pdf"
        call_args = mock_manager.astart_ingest_job.call_args
        assert call_args.args[0] == "default"
        ingest_spec = call_args.args[1]
        assert ingest_spec.source_type == "local"
        assert ingest_spec.path.startswith(str(mock_config.input_dir_path / "default"))
        mock_manager.aingest.assert_not_awaited()

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
        assert resp.status_code == 202
        mock_manager.astart_ingest_job.assert_awaited_once_with(
            "default",
            IngestSpec(
                source_type="azure_blob",
                container_name="my-container",
                blob_path="docs/file.pdf",
            ),
        )
        mock_manager.aingest.assert_not_awaited()

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
            IngestSpec(source_type="s3", bucket="my-bucket", prefix="docs/"),
        )
        mock_manager.aingest.assert_not_awaited()

    async def test_get_get_ingest_job(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager

        resp = await client.get("/ingest/jobs/job-1")

        assert resp.status_code == 200
        assert resp.json()["processed_items"] == 64
        mock_manager.aget_ingest_job.assert_awaited_once_with("job-1")

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
        assert resp.status_code == 422

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
        assert resp.status_code == 422

    async def test_ingest_with_workspace(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        path = mock_config.input_dir_path / "project_x" / "file.pdf"
        app.state.manager = mock_manager
        resp = await client.post(
            "/ingest",
            json={
                "source_type": "local",
                "path": "file.pdf",
                "workspace": "project-x",
            },
        )
        assert resp.status_code == 202
        call_kwargs = mock_manager.astart_ingest_job.call_args
        assert call_kwargs[0][0] == "project_x"  # normalized: hyphens → underscores
        assert call_kwargs.args[1].path == str(path)
        mock_manager.aingest.assert_not_awaited()

    async def test_ingest_service_unavailable_503(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        mock_manager.astart_ingest_job = AsyncMock(
            side_effect=RAGServiceUnavailableError("RAG not ready")
        )
        app.state.manager = mock_manager
        resp = await client.post(
            "/ingest",
            json={"source_type": "local", "path": "file.pdf"},
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
        assert mock_manager.aretrieve.call_args.kwargs["chunk_top_k"] is None

    async def test_retrieve_projects_source_workspace_without_internal_fields(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        mock_manager.aretrieve = AsyncMock(
            return_value=RetrievalResult(contexts={"chunks": [_finance_source_context()]})
        )
        app.state.manager = mock_manager

        response = await client.post(
            "/retrieve",
            json={"query": "report", "workspaces": ["finance"]},
        )

        assert response.status_code == 200
        source = response.json()["sources"][0]
        assert source["source_uri"] == "s3://bucket/report.pdf"
        assert source["download_url"] == "/files/raw/doc-report?workspace=finance"
        assert {"workspace", "download_locator", "path", "url"}.isdisjoint(source)

    async def test_retrieve_omits_download_link_without_download_permission(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        class QueryOnlyAccess:
            async def check(self, user, action, *, workspace=None):
                return None

            async def filter_workspaces(self, user, action, workspaces):
                if action == "workspace.download_source":
                    return []
                return list(workspaces)

        mock_manager.aretrieve = AsyncMock(
            return_value=RetrievalResult(contexts={"chunks": [_finance_source_context()]})
        )
        app.state.manager = mock_manager
        app.state.access_control = QueryOnlyAccess()

        try:
            response = await client.post(
                "/retrieve",
                json={"query": "report", "workspaces": ["finance"]},
            )
        finally:
            del app.state.access_control

        assert response.status_code == 200
        assert response.json()["sources"][0]["download_url"] is None

    async def test_retrieve_all_workspaces_uses_all_visible_records(
        self,
        client: AsyncClient,
        mock_config: DlightragConfig,
        mock_manager,
    ) -> None:
        mock_manager.alist_workspace_records.return_value = [
            {"workspace": "default"},
            {"workspace": "research_notes"},
        ]
        app.state.manager = mock_manager

        response = await client.post(
            "/retrieve",
            json={"query": "hello", "all_workspaces": True},
        )

        assert response.status_code == 200
        assert mock_manager.aretrieve.await_args.kwargs["workspaces"] == [
            "default",
            "research_notes",
        ]

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

    async def test_retrieve_forwards_chunk_top_k_field(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager
        resp = await client.post(
            "/retrieve",
            json={"query": "hello", "chunk_top_k": 5},
        )
        assert resp.status_code == 200
        assert mock_manager.aretrieve.call_args.kwargs["chunk_top_k"] == 5


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
        from unittest.mock import AsyncMock

        from dlightrag.storage.pool import pg_pool

        monkeypatch.setattr(pg_pool, "run_once", AsyncMock(return_value=1))
        app.state.manager = mock_manager
        resp = await client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert "rag_initialized" in body
        assert "storage" in body
        cap = body["answer_image_capability"]
        assert cap["status"] == "supported"
        assert cap["effective_max_images"] == 8
        assert cap["configured_ceiling"] == 8
        assert cap["model"] == "test-model"


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
        from unittest.mock import AsyncMock

        from dlightrag.storage.pool import pg_pool

        monkeypatch.setattr(pg_pool, "run_once", AsyncMock(return_value=1))
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
        mock_manager.adelete_files.assert_awaited_once()

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
        call_kwargs = mock_manager.adelete_files.call_args
        assert call_kwargs[0][0] == "project_y"  # normalized: hyphens → underscores

    async def test_delete_forwards_dry_run(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager
        resp = await client.request(
            "DELETE",
            "/files",
            json={"filenames": ["report.pdf"], "dry_run": True},
        )
        assert resp.status_code == 200
        assert mock_manager.adelete_files.call_args.kwargs["dry_run"] is True


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

    async def test_non_stream_answer_projects_source_workspace_without_internal_fields(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        mock_manager.aanswer = AsyncMock(
            return_value=RetrievalResult(
                answer="Answer [1-1].",
                sources=[_finance_source()],
            )
        )
        app.state.manager = mock_manager

        response = await client.post(
            "/answer",
            json={"query": "report", "stream": False, "workspaces": ["finance"]},
        )

        assert response.status_code == 200
        source = response.json()["sources"][0]
        assert source["source_uri"] == "s3://bucket/report.pdf"
        assert source["download_url"] == "/files/raw/doc-report?workspace=finance"
        assert {"workspace", "download_locator", "path", "url"}.isdisjoint(source)

    async def test_answer_includes_structured_images_and_blocks(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        mock_manager.aanswer = AsyncMock(
            return_value=RetrievalResult(
                answer="Diagram below [1-1].",
                contexts={
                    "chunks": [
                        {
                            "chunk_id": "fig-1",
                            "reference_id": "1",
                            "file_path": "/private/report.pdf",
                            "content": "Figure evidence",
                            "image_data": "base64-payload",
                            "_workspace": "default",
                        }
                    ],
                },
                sources=[
                    SourceReference(
                        id="1",
                        title="report.pdf",
                        source_uri="s3://bucket/report.pdf",
                        workspace="default",
                        document_id="doc-report",
                        download_locator="s3://bucket/report.pdf",
                        chunks=[
                            ChunkSnippet(
                                chunk_id="fig-1",
                                chunk_idx=1,
                                content="Figure evidence",
                                image_url="/images/default/fig-1?size=full",
                                thumbnail_url="/images/default/fig-1?size=thumb",
                            )
                        ],
                    )
                ],
                answer_images=[
                    {
                        "id": "fig-1",
                        "chunk_id": "fig-1",
                        "source_ref": "1-1",
                        "url": "/images/default/fig-1?size=full",
                        "thumbnail_url": "/images/default/fig-1?size=thumb",
                        "label": "report.pdf",
                    }
                ],
                answer_blocks=[
                    {"type": "markdown", "text": "Diagram below [1-1]."},
                    {"type": "image_ref", "image_id": "fig-1"},
                ],
            )
        )
        app.state.manager = mock_manager

        resp = await client.post("/answer", json={"query": "show diagram", "stream": False})

        assert resp.status_code == 200
        body = resp.json()
        assert body["answer_images"][0]["chunk_id"] == "fig-1"
        assert body["answer_blocks"] == [
            {"type": "markdown", "text": "Diagram below [1-1]."},
            {"type": "image_ref", "image_id": "fig-1"},
        ]

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
                "chunk_top_k": 12,
                "answer_context_top_k": 4,
            },
        )
        assert resp.status_code == 200
        call_kwargs = mock_manager.aanswer.call_args.kwargs
        assert call_kwargs["chunk_top_k"] == 12
        assert call_kwargs["answer_context_top_k"] == 4
        assert call_kwargs["semantic_highlights"] is False

    async def test_answer_forwards_semantic_highlights_opt_in(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager
        resp = await client.post(
            "/answer",
            json={
                "query": "What is RAG?",
                "stream": False,
                "semantic_highlights": True,
            },
        )
        assert resp.status_code == 200
        assert mock_manager.aanswer.call_args.kwargs["semantic_highlights"] is True

    async def test_answer_rejects_conversation_history_and_accepts_query_image_blocks(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager
        history = [
            {"role": "user", "content": [{"type": "text", "text": "previous"}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "dlightrag-image://img_1"},
                    }
                ],
            },
        ]
        query_images = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]

        rejected = await client.post(
            "/answer",
            json={
                "query": "What is shown?",
                "stream": False,
                "conversation_history": history,
                "query_images": query_images,
            },
        )

        assert rejected.status_code == 422
        mock_manager.aanswer.assert_not_awaited()

        resp = await client.post(
            "/answer",
            json={
                "query": "What is shown?",
                "stream": False,
                "query_images": query_images,
            },
        )

        assert resp.status_code == 200
        call_kwargs = mock_manager.aanswer.call_args.kwargs
        assert "conversation_history" not in call_kwargs
        assert call_kwargs["query_images"] == query_images

    async def test_answer_accepts_caller_history(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager
        history = [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "Paris."},
        ]

        resp = await client.post(
            "/answer",
            json={"query": "And its population?", "stream": False, "history": history},
        )

        assert resp.status_code == 200
        assert mock_manager.aanswer.call_args.kwargs["history"] == history

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
        mock_manager.alist_ingested_files = AsyncMock(return_value=["a.pdf", "b.pdf", "c.pdf"])
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
        call_kwargs = mock_manager.alist_ingested_files.call_args
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

    async def test_stream_sources_event_projects_source_workspace(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        async def mock_tokens():
            yield "Answer [1-1]."

        mock_manager.aanswer_stream = AsyncMock(
            return_value=({"chunks": [_finance_source_context()]}, mock_tokens())
        )
        app.state.manager = mock_manager

        response = await client.post(
            "/answer",
            json={"query": "report", "stream": True, "workspaces": ["finance"]},
        )

        import json as json_mod

        events = [
            json_mod.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        sources_event = next(event for event in events if event["type"] == "sources")
        source = sources_event["data"][0]
        assert source["source_uri"] == "s3://bucket/report.pdf"
        assert source["download_url"] == "/files/raw/doc-report?workspace=finance"
        assert {"workspace", "download_locator", "path", "url"}.isdisjoint(source)

    async def test_stream_all_workspaces_uses_visible_records(
        self,
        client: AsyncClient,
        mock_config: DlightragConfig,
        mock_manager,
    ) -> None:
        async def mock_tokens():
            yield "answer"

        mock_manager.alist_workspace_records.return_value = [
            {"workspace": "default"},
            {"workspace": "research_notes"},
        ]
        mock_manager.aanswer_stream = AsyncMock(return_value=({"chunks": []}, mock_tokens()))
        app.state.manager = mock_manager

        response = await client.post(
            "/answer",
            json={"query": "hello", "stream": True, "all_workspaces": True},
        )

        assert response.status_code == 200
        await_args = mock_manager.aanswer_stream.await_args
        assert await_args is not None
        assert await_args.kwargs["workspaces"] == [
            "default",
            "research_notes",
        ]

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
                "chunk_top_k": 16,
                "answer_context_top_k": 5,
            },
        )
        assert resp.status_code == 200
        call_kwargs = mock_manager.aanswer_stream.call_args.kwargs
        assert call_kwargs["chunk_top_k"] == 16
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
        assert events[0]["data"] == {
            "chunks": [{"chunk_id": "c1", "reference_id": "", "file_path": "", "content": ""}],
            "entities": [],
            "relationships": [],
        }
        assert events[-1]["type"] == "done"
        assert events[-1]["answer"] == "Hi there"
        token_events = [e for e in events if e["type"] == "token"]
        assert len(token_events) == 2
        assert token_events[0]["content"] == "Hi"
        assert token_events[1]["content"] == " there"

    async def test_stream_done_includes_structured_images_and_blocks(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        async def mock_tokens():
            yield "Diagram below [1-1]."

        contexts = {
            "chunks": [
                {
                    "chunk_id": "fig-1",
                    "reference_id": "1",
                    "full_doc_id": "doc-report",
                    "file_path": "/private/report.pdf",
                    "content": "Figure evidence",
                    "image_data": "base64-payload",
                    "_workspace": "default",
                    "metadata": {
                        "source_uri": "s3://bucket/report.pdf",
                        "source_download_locator": "s3://bucket/report.pdf",
                    },
                }
            ]
        }
        mock_manager.aanswer_stream = AsyncMock(return_value=(contexts, mock_tokens()))
        app.state.manager = mock_manager

        resp = await client.post("/answer", json={"query": "show diagram", "stream": True})

        import json as json_mod

        events = [
            json_mod.loads(line.removeprefix("data: "))
            for line in resp.text.split("\n")
            if line.startswith("data: ")
        ]
        done = events[-1]
        assert done["type"] == "done"
        assert done["answer_images"][0]["chunk_id"] == "fig-1"
        assert done["answer_blocks"] == [
            {"type": "markdown", "text": "Diagram below [1-1]."},
            {"type": "image_ref", "image_id": "fig-1"},
        ]

    async def test_stream_semantic_highlights_default_off(
        self,
        client: AsyncClient,
        mock_config: DlightragConfig,
        mock_manager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Streaming REST answer does not enrich highlights unless requested."""
        from dlightrag.api.routes import rag as rag_routes

        async def mock_tokens():
            yield "Market growth improved [1-1]."

        async def fail_enrich(*args, **kwargs):
            raise AssertionError("semantic highlights should be opt-in for REST streaming")

        monkeypatch.setattr(
            rag_routes,
            "enrich_semantic_highlights",
            fail_enrich,
            raising=False,
        )
        mock_manager.aanswer_stream = AsyncMock(
            return_value=(
                {
                    "chunks": [
                        {
                            "chunk_id": "c1",
                            "reference_id": "1",
                            "full_doc_id": "doc-report",
                            "file_path": "/docs/report.pdf",
                            "content": "The report says market growth improved.",
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
        app.state.manager = mock_manager

        resp = await client.post("/answer", json={"query": "test", "stream": True})

        assert resp.status_code == 200
        assert "semantic highlights should be opt-in" not in resp.text

    async def test_stream_semantic_highlights_are_opt_in(
        self,
        client: AsyncClient,
        mock_config: DlightragConfig,
        mock_manager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Streaming REST answer can opt into source highlight phrases."""
        from dlightrag.api.routes import rag as rag_routes

        async def mock_tokens():
            yield "Market growth improved [1-1]."

        async def fake_enrich(
            sources: list[SourceReference],
            answer_text: str | None,
            config: DlightragConfig,
        ) -> list[SourceReference]:
            assert answer_text == "Market growth improved [1-1]."
            chunks = sources[0].chunks
            assert chunks is not None
            chunks[0].highlight_phrases = ["market growth"]
            return sources

        monkeypatch.setattr(
            rag_routes,
            "enrich_semantic_highlights",
            fake_enrich,
            raising=False,
        )
        mock_manager.aanswer_stream = AsyncMock(
            return_value=(
                {
                    "chunks": [
                        {
                            "chunk_id": "c1",
                            "reference_id": "1",
                            "full_doc_id": "doc-report",
                            "file_path": "/docs/report.pdf",
                            "content": "The report says market growth improved.",
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
        app.state.manager = mock_manager

        resp = await client.post(
            "/answer",
            json={"query": "test", "stream": True, "semantic_highlights": True},
        )

        import json as json_mod

        events = [
            json_mod.loads(line.removeprefix("data: "))
            for line in resp.text.split("\n")
            if line.startswith("data: ")
        ]
        sources_event = next(event for event in events if event["type"] == "sources")
        assert sources_event["data"][0]["chunks"][0]["highlight_phrases"] == ["market growth"]

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

    async def test_stream_setup_error_becomes_sse_error_event(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        """Setup errors during streaming surface as an SSE error event (200), not HTTP 503.

        The root trace span now wraps the whole streamed response, so
        ``aanswer_stream`` runs inside the SSE generator; its errors are streamed
        as an error event instead of a pre-stream HTTP status.
        """
        mock_manager.aanswer_stream = AsyncMock(
            side_effect=RAGServiceUnavailableError("RAG not ready")
        )
        app.state.manager = mock_manager
        resp = await client.post("/answer", json={"query": "hello", "stream": True})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        assert "Internal server error during streaming" in resp.text

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

    async def test_answer_image_capability_error_maps_to_400(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        """A non-streaming capability rejection surfaces as HTTP 400 + stable error_kind."""
        from dlightrag.core.answer.errors import CURRENT_IMAGES_UNSUPPORTED, AnswerImageError

        mock_manager.aanswer = AsyncMock(
            side_effect=AnswerImageError(
                "[IMAGES_NOT_SUPPORTED_BY_MODEL] no vision",
                error_kind=CURRENT_IMAGES_UNSUPPORTED,
            )
        )
        app.state.manager = mock_manager
        resp = await client.post("/answer", json={"query": "hi", "stream": False})
        assert resp.status_code == 400
        body = resp.json()
        assert body["error_type"] == "validation"
        assert body["error_kind"] == CURRENT_IMAGES_UNSUPPORTED

    async def test_stream_capability_error_becomes_structured_sse_error(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        """A capability rejection during streaming surfaces as a classified SSE error."""
        import json as json_mod

        from dlightrag.core.answer.errors import ANSWER_IMAGE_CAPABILITY_UNKNOWN, AnswerImageError

        mock_manager.aanswer_stream = AsyncMock(
            side_effect=AnswerImageError(
                "[ANSWER_IMAGE_CAPABILITY_UNKNOWN] unknown",
                error_kind=ANSWER_IMAGE_CAPABILITY_UNKNOWN,
            )
        )
        app.state.manager = mock_manager
        resp = await client.post("/answer", json={"query": "hi", "stream": True})
        assert resp.status_code == 200
        events = [
            json_mod.loads(line.removeprefix("data: "))
            for line in resp.text.split("\n")
            if line.startswith("data: ")
        ]
        error_events = [e for e in events if e["type"] == "error"]
        assert len(error_events) == 1
        assert error_events[0]["error_kind"] == ANSWER_IMAGE_CAPABILITY_UNKNOWN
        assert "Internal server error" not in error_events[0]["message"]


class TestAPIContracts:
    """Request and response contracts are explicit in OpenAPI."""

    def test_answer_stream_sources_event_uses_contract_model(self) -> None:
        import json

        from dlightrag.api.events import AnswerSourcesStreamEvent, sse_data_event
        from dlightrag.citations.schemas import SourceReferencePayload

        frame = sse_data_event(
            AnswerSourcesStreamEvent(
                data=[
                    SourceReferencePayload(
                        id="1",
                        title="report.pdf",
                        source_uri="local://default/report.pdf",
                        download_url="/files/raw/report.pdf?workspace=default",
                    )
                ]
            )
        )

        assert frame.startswith("data: ")
        payload = json.loads(frame.removeprefix("data: ").strip())
        assert payload == {
            "type": "sources",
            "data": [
                {
                    "id": "1",
                    "title": "report.pdf",
                    "source_uri": "local://default/report.pdf",
                    "download_url": "/files/raw/report.pdf?workspace=default",
                }
            ],
        }

    async def test_openapi_exposes_pydantic_response_models(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager

        resp = await client.get("/openapi.json")

        assert resp.status_code == 200
        spec = resp.json()
        schemas = spec["components"]["schemas"]
        assert "RetrievalResponse" in schemas
        assert "AnswerResponse" in schemas
        ingest_properties = schemas["IngestRequest"]["properties"]
        assert "download_uri" in ingest_properties
        assert "download_uris" in ingest_properties
        assert "download_url" not in ingest_properties
        assert "download_urls" not in ingest_properties
        assert (
            spec["paths"]["/retrieve"]["post"]["responses"]["200"]["content"]["application/json"][
                "schema"
            ]["$ref"]
            == "#/components/schemas/RetrievalResponse"
        )
        assert (
            spec["paths"]["/workspaces"]["get"]["responses"]["200"]["content"]["application/json"][
                "schema"
            ]["$ref"]
            == "#/components/schemas/WorkspacesResponse"
        )


class TestMetadataAPI:
    @pytest.mark.usefixtures("_patch_manager")
    async def test_get_metadata_hides_internal_paths_and_download_locator(
        self,
        client: AsyncClient,
        mock_config: DlightragConfig,
        mock_manager,
    ) -> None:
        mock_manager.aget_metadata = AsyncMock(
            return_value={
                "workspace": "default",
                "doc_id": "doc-1",
                "filename": "report.pdf",
                "file_path": "/srv/dlightrag/inputs/default/report.pdf",
                "source_uri": "bynder://asset/1",
                "download_locator": "https://cdn.example.com/assets/1.pdf",
            }
        )

        response = await client.get("/metadata/doc-1")

        assert response.status_code == 200
        assert response.json() == {
            "doc_id": "doc-1",
            "metadata": {
                "filename": "report.pdf",
                "source_uri": "bynder://asset/1",
            },
        }
