# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for FastAPI REST server endpoints and auth middleware."""

import asyncio
import contextlib
import datetime
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import jwt
import pytest
from dlightrag_rag.retrieval import RetrievalResult
from fastapi import FastAPI, HTTPException
from httpx import ASGITransport, AsyncClient, Response

from dlightrag.api import auth as auth_module
from dlightrag.api.auth import UserContext, get_current_user, verify_bearer_token
from dlightrag.api.server import create_app
from dlightrag.application import ApplicationHealth
from dlightrag.citations.schemas import SourceReference
from dlightrag.config import (
    AccessControlConfig,
    AccessControlRuleConfig,
    DlightragConfig,
    set_config,
)
from dlightrag.core.answer_runs.results import AnswerResult
from dlightrag.core.client_contracts import IngestSpec
from dlightrag.core.servicemanager import RAGServiceUnavailableError
from dlightrag.runtime import AnswerRunRecord, RunCreation
from tests.unit.conftest import prepare_test_answer_run_input

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


def _queued_run_record() -> AnswerRunRecord:
    now = datetime.datetime(2026, 8, 13, tzinfo=datetime.UTC)
    return AnswerRunRecord(
        owner_id="owner",
        run_id="0199a0a0-0000-7000-8000-0000000000aa",
        idempotency_key=None,
        request={"query": "hi", "workspaces": ["default"]},
        status="queued",
        phase=None,
        stop_reason=None,
        completed_turns=0,
        cancel_requested_at=None,
        lease_owner=None,
        lease_expires_at=None,
        fencing_epoch=0,
        recovery_count=0,
        next_event_sequence=1,
        events_trimmed_at=None,
        result=None,
        error_kind=None,
        error_message=None,
        created_at=now,
        updated_at=now,
        started_at=None,
        finished_at=None,
    )


@pytest.fixture
def _api_app(test_config: DlightragConfig) -> Iterator[FastAPI]:
    """Create the API app after test_config has installed the singleton."""
    global app
    app = create_app(include_web_app=False)
    yield app
    app.dependency_overrides.clear()
    if hasattr(app.state, "manager"):
        del app.state.manager
    if hasattr(app.state, "health"):
        del app.state.health


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
    service.aretrieve = AsyncMock(return_value=RetrievalResult(contexts={"chunks": []}))
    service.aanswer = AsyncMock(
        return_value=AnswerResult(answer="The answer is 42", contexts={"chunks": []})
    )
    service.alist_ingested_files = AsyncMock(return_value=[])
    service.adelete_files = AsyncMock(return_value=[{"status": "deleted"}])
    return service


@pytest.fixture
def mock_manager(_api_app: FastAPI, mock_service, test_config):
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
            "lease_owner": None,
            "lease_expires_at": None,
        }
    )
    manager.aget_ingest_job = AsyncMock(
        return_value={
            "job_id": "job-1",
            "workspace": "default",
            "source_type": "s3",
            "status": "running",
            "processed_items": 64,
            "lease_owner": "worker-7",
            "lease_expires_at": "2026-08-05T00:00:00+00:00",
        }
    )
    manager.aretrieve = mock_service.aretrieve
    manager.aanswer = mock_service.aanswer
    manager.astart_answer_run = AsyncMock(
        return_value=RunCreation(run=_queued_run_record(), replayed=False)
    )
    manager.aprepare_answer_run_input = AsyncMock(side_effect=prepare_test_answer_run_input)
    manager.aget_answer_run = AsyncMock(return_value=_queued_run_record())
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
    from dlightrag.adapters.postgres.corpus import PGReadinessProbe
    from dlightrag.core.answer.capability import answer_image_capability_summary

    manager.health = ApplicationHealth(
        readiness_probe=PGReadinessProbe(test_config),
    )
    manager.health.mark_ready()
    manager.health.set_answer_image_capability(
        answer_image_capability_summary(manager.answer_image_capability)
    )
    _api_app.state.health = manager.health
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
            ("POST", "/answer", {"query": "hello"}),
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
            (["finance-rag-readers"], 202),
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
            json={"query": "hello", "all_workspaces": True},
        )

        assert response.status_code == expected_status
        if expected_status == 202:
            run_input = mock_manager.astart_answer_run.await_args.kwargs["request"]
            assert list(run_input.workspaces) == allowed
        else:
            mock_manager.astart_answer_run.assert_not_awaited()

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

    @pytest.mark.parametrize("source_type", ["local", "azure_blob", "s3", "url"])
    @pytest.mark.usefixtures("_patch_manager")
    async def test_source_requires_identity(
        self,
        client: AsyncClient,
        mock_config: DlightragConfig,
        source_type: str,
    ) -> None:
        resp = await client.post("/ingest", json={"source_type": source_type})
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

    async def test_blob_upload_stages_file_for_local_ingest(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        mock_manager.config = mock_config
        mock_manager.astart_ingest_job.return_value = {
            "job_id": "job-1",
            "workspace": "default",
            "source_type": "local",
            "status": "queued",
            "lease_owner": None,
            "lease_expires_at": None,
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
        assert "lease_owner" not in body
        call_args = mock_manager.astart_ingest_job.call_args
        assert call_args.args[0] == "default"
        ingest_spec = call_args.args[1]
        assert ingest_spec.source_type == "local"
        assert ingest_spec.path.startswith(str(mock_config.input_dir_path / "default"))
        mock_manager.aingest.assert_not_awaited()

    async def test_get_get_ingest_job(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager

        resp = await client.get("/ingest/jobs/job-1")

        assert resp.status_code == 200
        body = resp.json()
        assert body["processed_items"] == 64
        # Queue bookkeeping stays server-side.
        assert "lease_owner" not in body
        assert "lease_expires_at" not in body
        mock_manager.aget_ingest_job.assert_awaited_once_with("job-1")

    async def test_cancel_ingest_job(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager
        mock_manager.acancel_ingest_job = AsyncMock(
            return_value={
                "job_id": "job-1",
                "workspace": "default",
                "source_type": "s3",
                "status": "failed",
                "processed_items": 64,
            }
        )

        resp = await client.post("/ingest/jobs/job-1/cancel")

        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "failed"
        # Cancelling stops further work; it never unwinds what already landed.
        assert body["processed_items"] == 64
        mock_manager.acancel_ingest_job.assert_awaited_once_with("job-1")

    async def test_cancel_unknown_ingest_job_is_404(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager
        mock_manager.aget_ingest_job = AsyncMock(return_value=None)
        mock_manager.acancel_ingest_job = AsyncMock()

        resp = await client.post("/ingest/jobs/nope/cancel")

        assert resp.status_code == 404
        mock_manager.acancel_ingest_job.assert_not_awaited()

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
        assert "answer" not in body
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

    async def test_retrieve_omits_download_and_visual_links_without_permissions(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        class QueryOnlyAccess:
            async def check(self, user, action, *, workspace=None):
                return None

            async def filter_workspaces(self, user, action, workspaces):
                if action in {"workspace.download_source", "workspace.read_visual_asset"}:
                    return []
                return list(workspaces)

        mock_manager.aretrieve = AsyncMock(
            return_value=RetrievalResult(
                contexts={"chunks": [{**_finance_source_context(), "image_data": "bytes"}]}
            )
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
        assert "image_url" not in response.json()["contexts"]["chunks"][0]

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

    async def test_retrieve_uses_shared_executor(
        self,
        client: AsyncClient,
        mock_config: DlightragConfig,
        mock_manager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from dlightrag.api.routes import rag as rag_routes

        execute = AsyncMock(return_value=RetrievalResult(contexts={"chunks": []}))
        monkeypatch.setattr(rag_routes, "execute_retrieve", execute)
        app.state.manager = mock_manager

        resp = await client.post("/retrieve", json={"query": "hello"})

        assert resp.status_code == 200
        execute.assert_awaited_once()
        mock_manager.aretrieve.assert_not_awaited()


# ---------------------------------------------------------------------------
# TestHealthEndpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """``/health`` is liveness only: in-process facts, never a database probe."""

    async def test_health_returns_status_without_probing_postgres(
        self,
        client: AsyncClient,
        mock_config: DlightragConfig,
        mock_manager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from dlightrag.adapters.postgres._pool import pg_pool

        probe = AsyncMock(return_value="off")
        monkeypatch.setattr(pg_pool, "run_once", probe)
        app.state.manager = mock_manager
        resp = await client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert "rag_initialized" in body
        assert "storage" in body
        assert "postgres" not in body
        probe.assert_not_awaited()
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
        mock_manager.health.mark_degraded("Embedding unreachable")
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
    ) -> None:
        app.state.manager = mock_manager
        resp = await client.get("/health")
        body = resp.json()
        assert body["status"] == "healthy"
        assert "warnings" not in body


# ---------------------------------------------------------------------------
# TestReadinessEndpoint
# ---------------------------------------------------------------------------


class TestReadinessEndpoint:
    """Test strict traffic-readiness semantics independently from /health."""

    async def test_ready_returns_200_without_authentication(
        self,
        client: AsyncClient,
        mock_config_no_auth_override: DlightragConfig,
        mock_manager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from dlightrag.adapters.postgres._pool import pg_pool

        mock_config_no_auth_override.auth_mode = "simple"
        mock_config_no_auth_override.api_auth_token = "required-elsewhere"
        probe = AsyncMock(return_value="off")
        monkeypatch.setattr(pg_pool, "run_once", probe)
        app.state.manager = mock_manager

        response = await client.get("/ready")

        assert response.status_code == 200
        assert response.json() == {"status": "ready", "service_role": "writer"}
        probe.assert_awaited_once()

    async def test_not_ready_manager_returns_503_without_probing_postgres(
        self,
        client: AsyncClient,
        mock_config: DlightragConfig,
        mock_manager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from dlightrag.adapters.postgres._pool import pg_pool

        mock_manager.health.mark_closed()
        probe = AsyncMock(return_value="off")
        monkeypatch.setattr(pg_pool, "run_once", probe)
        app.state.manager = mock_manager

        response = await client.get("/ready")

        assert response.status_code == 503
        assert response.json() == {
            "status": "not_ready",
            "service_role": "writer",
            "detail": "RAG service is not ready",
        }
        probe.assert_not_awaited()

    async def test_reader_requires_writable_domain_session(
        self,
        client: AsyncClient,
        mock_config: DlightragConfig,
        mock_manager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from dlightrag.adapters.postgres._pool import pg_pool

        mock_config.service_role = "reader"
        monkeypatch.setattr(pg_pool, "run_once", AsyncMock(return_value="on"))
        app.state.manager = mock_manager

        response = await client.get("/ready")

        assert response.status_code == 503
        assert response.json() == {
            "status": "not_ready",
            "service_role": "reader",
            "detail": "DlightRAG domain database session is not writable",
        }

    async def test_writer_requires_writable_domain_session(
        self,
        client: AsyncClient,
        mock_config: DlightragConfig,
        mock_manager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from dlightrag.adapters.postgres._pool import pg_pool

        monkeypatch.setattr(pg_pool, "run_once", AsyncMock(return_value="on"))
        app.state.manager = mock_manager

        response = await client.get("/ready")

        assert response.status_code == 503
        assert response.json() == {
            "status": "not_ready",
            "service_role": "writer",
            "detail": "DlightRAG domain database session is not writable",
        }

    async def test_reader_requires_read_only_corpus_session(
        self,
        client: AsyncClient,
        mock_config: DlightragConfig,
        mock_manager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import dlightrag.adapters.postgres.corpus as corpus_module
        from dlightrag.adapters.postgres._pool import pg_pool
        from dlightrag.adapters.postgres.corpus import PGReadinessProbe

        mock_config.service_role = "reader"
        mock_manager.health = ApplicationHealth(readiness_probe=PGReadinessProbe(mock_config))
        mock_manager.health.mark_ready()
        app.state.health = mock_manager.health
        monkeypatch.setattr(pg_pool, "run_once", AsyncMock(return_value="off"))
        monkeypatch.setattr(
            corpus_module,
            "verify_reader_corpus_session",
            AsyncMock(side_effect=RuntimeError("corpus pool is not read-only")),
        )
        app.state.manager = mock_manager

        response = await client.get("/ready")

        assert response.status_code == 503
        assert response.json() == {
            "status": "not_ready",
            "service_role": "reader",
            "detail": "Reader corpus database session is not read-only or is unavailable",
        }

    async def test_reader_is_ready_with_writable_domain_and_read_only_corpus(
        self,
        client: AsyncClient,
        mock_config: DlightragConfig,
        mock_manager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import dlightrag.adapters.postgres.corpus as corpus_module
        from dlightrag.adapters.postgres._pool import pg_pool
        from dlightrag.adapters.postgres.corpus import PGReadinessProbe

        mock_config.service_role = "reader"
        mock_manager.health = ApplicationHealth(readiness_probe=PGReadinessProbe(mock_config))
        mock_manager.health.mark_ready()
        app.state.health = mock_manager.health
        monkeypatch.setattr(pg_pool, "run_once", AsyncMock(return_value="off"))
        corpus_probe = AsyncMock()
        monkeypatch.setattr(corpus_module, "verify_reader_corpus_session", corpus_probe)
        app.state.manager = mock_manager

        response = await client.get("/ready")

        assert response.status_code == 200
        assert response.json() == {"status": "ready", "service_role": "reader"}
        corpus_probe.assert_awaited_once_with()

    async def test_repeated_polls_reuse_one_cached_probe(
        self,
        client: AsyncClient,
        mock_config: DlightragConfig,
        mock_manager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from dlightrag.adapters.postgres._pool import pg_pool

        probe = AsyncMock(return_value="off")
        monkeypatch.setattr(pg_pool, "run_once", probe)
        app.state.manager = mock_manager

        for _ in range(5):
            assert (await client.get("/ready")).status_code == 200

        probe.assert_awaited_once()

    async def test_concurrent_cold_polls_share_one_probe(
        self,
        client: AsyncClient,
        mock_config: DlightragConfig,
        mock_manager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A burst against a cold cache costs one round trip, not one per caller."""
        from dlightrag.adapters.postgres._pool import pg_pool

        started = asyncio.Event()
        release = asyncio.Event()

        async def _slow_probe(*_args: object, **_kwargs: object) -> str:
            started.set()
            await release.wait()
            return "off"

        probe = AsyncMock(side_effect=_slow_probe)
        monkeypatch.setattr(pg_pool, "run_once", probe)
        app.state.manager = mock_manager

        polls = [asyncio.create_task(client.get("/ready")) for _ in range(5)]
        await started.wait()
        release.set()
        responses = await asyncio.gather(*polls)

        assert [response.status_code for response in responses] == [200] * 5
        probe.assert_awaited_once()

    async def test_one_abandoned_poller_never_cancels_the_shared_probe(
        self,
        client: AsyncClient,
        mock_config: DlightragConfig,
        mock_manager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from dlightrag.adapters.postgres._pool import pg_pool

        started = asyncio.Event()
        release = asyncio.Event()
        completed = False

        async def _slow_probe(*_args: object, **_kwargs: object) -> str:
            nonlocal completed
            started.set()
            await release.wait()
            completed = True
            return "off"

        monkeypatch.setattr(pg_pool, "run_once", AsyncMock(side_effect=_slow_probe))
        app.state.manager = mock_manager

        abandoned = asyncio.create_task(client.get("/ready"))
        waiting = asyncio.create_task(client.get("/ready"))
        await started.wait()
        abandoned.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await abandoned
        release.set()

        assert (await waiting).status_code == 200
        assert completed is True

    async def test_the_cached_verdict_expires(
        self,
        client: AsyncClient,
        mock_manager,
    ) -> None:
        probe = AsyncMock(return_value=None)
        health = ApplicationHealth(readiness_probe=probe, readiness_cache_seconds=0.0)
        health.mark_ready()
        app.state.health = health
        app.state.manager = mock_manager

        await client.get("/ready")
        await client.get("/ready")

        assert probe.await_count == 2

    async def test_a_not_ready_transition_invalidates_the_cached_verdict(
        self,
        client: AsyncClient,
        mock_manager,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Startup and schema transitions must never be served from a stale verdict."""
        from dlightrag.adapters.postgres._pool import pg_pool

        probe = AsyncMock(return_value="off")
        monkeypatch.setattr(pg_pool, "run_once", probe)
        app.state.manager = mock_manager

        assert (await client.get("/ready")).status_code == 200
        mock_manager.health.mark_degraded("temporary startup failure")
        assert (await client.get("/ready")).status_code == 503
        mock_manager.health.mark_ready()
        assert (await client.get("/ready")).status_code == 200

        assert probe.await_count == 2


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
    """POST /answer admission: what the run's immutable input may carry."""

    async def test_answer_forwards_explicit_filters(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager
        resp = await client.post(
            "/answer",
            json={
                "query": "What did Ada write?",
                "filters": {"author": "Ada"},
            },
        )
        assert resp.status_code == 202
        run_input = mock_manager.astart_answer_run.await_args.kwargs["request"]
        assert run_input.filters == {"author": "Ada"}

    async def test_answer_forwards_answer_context_limits(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager
        resp = await client.post(
            "/answer",
            json={
                "query": "What is RAG?",
                "chunk_top_k": 12,
            },
        )
        assert resp.status_code == 202
        run_input = mock_manager.astart_answer_run.await_args.kwargs["request"]
        assert run_input.chunk_top_k == 12
        assert run_input.semantic_highlights is False

    async def test_answer_forwards_semantic_highlights_opt_in(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager
        resp = await client.post(
            "/answer",
            json={
                "query": "What is RAG?",
                "semantic_highlights": True,
            },
        )
        assert resp.status_code == 202
        run_input = mock_manager.astart_answer_run.await_args.kwargs["request"]
        assert run_input.semantic_highlights is True

    async def test_answer_rejects_query_images_and_accepts_attachment_links(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager
        query_images = [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]

        rejected = await client.post(
            "/answer",
            json={"query": "What is shown?", "query_images": query_images},
        )
        assert rejected.status_code == 422
        mock_manager.astart_answer_run.assert_not_awaited()

        resp = await client.post(
            "/answer",
            json={
                "query": "What is shown?",
                "attachments": [{"url": "https://example.com/report.pdf", "filename": "r.pdf"}],
            },
        )

        assert resp.status_code == 202
        run_input = mock_manager.astart_answer_run.await_args.kwargs["request"]
        assert [link.url for link in run_input.links] == ["https://example.com/report.pdf"]
        assert run_input.links[0].filename == "r.pdf"
        assert run_input.attachments == ()

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
            json={"query": "And its population?", "history": history},
        )

        assert resp.status_code == 202
        run_input = mock_manager.astart_answer_run.await_args.kwargs["request"]
        assert list(run_input.history) == history

    async def test_json_enforces_link_count_limit(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        mock_config.answer.max_attachments = 2
        app.state.manager = mock_manager

        resp = await client.post(
            "/answer",
            json={
                "query": "q",
                "attachments": [{"url": f"https://example.com/{index}.pdf"} for index in range(3)],
            },
        )

        assert resp.status_code == 413
        assert "2" in resp.json()["detail"]
        mock_manager.astart_answer_run.assert_not_awaited()

    async def test_answer_service_unavailable_503(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        mock_manager.astart_answer_run = AsyncMock(
            side_effect=RAGServiceUnavailableError("RAG not ready")
        )
        app.state.manager = mock_manager
        resp = await client.post("/answer", json={"query": "hello"})
        assert resp.status_code == 503


# ---------------------------------------------------------------------------
# TestAnswerMultipart
# ---------------------------------------------------------------------------


class TestAnswerMultipart:
    """POST /answer multipart: one JSON request part plus repeated attachment files."""

    async def test_multipart_mixes_links_and_files(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        import json as json_mod

        app.state.manager = mock_manager
        request_part = json_mod.dumps(
            {
                "query": "compare",
                "attachments": [{"url": "https://example.com/a.pdf"}],
            }
        )
        resp = await client.post(
            "/answer",
            data={"request": request_part},
            files=[("attachments", ("report.pdf", b"%PDF-body", "application/pdf"))],
        )

        assert resp.status_code == 202
        call = mock_manager.astart_answer_run.await_args.kwargs
        run_input = call["request"]
        assert [link.url for link in run_input.links] == ["https://example.com/a.pdf"]
        assert call["attachment_bytes"] == [b"%PDF-body"]
        assert run_input.attachments[0].filename == "report.pdf"
        assert run_input.attachments[0].mime_type == "application/pdf"

    async def test_multipart_accepts_maximum_unicode_history(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        import json as json_mod

        from dlightrag.core.client_contracts import (
            MAX_HISTORY_CONTENT_CHARS,
            MAX_HISTORY_MESSAGES,
        )

        app.state.manager = mock_manager
        history = [
            {
                "role": "user" if index % 2 == 0 else "assistant",
                "content": "\U0001f642" * MAX_HISTORY_CONTENT_CHARS,
            }
            for index in range(MAX_HISTORY_MESSAGES)
        ]

        response = await client.post(
            "/answer",
            data={
                "request": json_mod.dumps(
                    {"query": "continue", "history": history},
                    ensure_ascii=False,
                )
            },
            files=[("attachments", ("note.txt", b"evidence", "text/plain"))],
        )

        assert response.status_code == 202
        run_input = mock_manager.astart_answer_run.await_args.kwargs["request"]
        assert len(run_input.history) == MAX_HISTORY_MESSAGES

    async def test_multipart_requires_exactly_one_request_part(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        import json as json_mod

        app.state.manager = mock_manager
        missing = await client.post(
            "/answer", files=[("attachments", ("a.txt", b"x", "text/plain"))]
        )
        assert missing.status_code == 400

        duplicate = await client.post(
            "/answer",
            data={"request": json_mod.dumps({"query": "q"})},
            files=[
                (
                    "request",
                    (
                        "r.json",
                        json_mod.dumps({"query": "q2"}),
                        "application/json",
                    ),
                )
            ],
        )
        assert duplicate.status_code == 400
        mock_manager.astart_answer_run.assert_not_awaited()

    async def test_multipart_rejects_wrong_part_name(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        import json as json_mod

        app.state.manager = mock_manager
        resp = await client.post(
            "/answer",
            data={"request": json_mod.dumps({"query": "q"})},
            files=[("documents", ("a.txt", b"x", "text/plain"))],
        )

        assert resp.status_code == 400
        mock_manager.astart_answer_run.assert_not_awaited()

    async def test_multipart_malformed_request_part_is_422(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        app.state.manager = mock_manager
        resp = await client.post(
            "/answer",
            data={"request": "{not json"},
            files=[("attachments", ("a.txt", b"x", "text/plain"))],
        )

        assert resp.status_code == 422
        mock_manager.astart_answer_run.assert_not_awaited()

    async def test_multipart_enforces_count_limit(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        import json as json_mod

        mock_config.answer.max_attachments = 2
        app.state.manager = mock_manager
        resp = await client.post(
            "/answer",
            data={"request": json_mod.dumps({"query": "q"})},
            files=[("attachments", (f"f{index}.txt", b"x", "text/plain")) for index in range(3)],
        )

        assert resp.status_code == 413
        mock_manager.astart_answer_run.assert_not_awaited()

    async def test_multipart_enforces_per_item_limit(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        import json as json_mod

        mock_config.answer.max_attachment_bytes = 8
        app.state.manager = mock_manager
        resp = await client.post(
            "/answer",
            data={"request": json_mod.dumps({"query": "q"})},
            files=[("attachments", ("big.bin", b"x" * 64, "application/octet-stream"))],
        )

        assert resp.status_code == 413
        mock_manager.astart_answer_run.assert_not_awaited()

    async def test_multipart_enforces_total_limit(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        import json as json_mod

        mock_config.answer.max_total_attachment_bytes = 16
        app.state.manager = mock_manager
        resp = await client.post(
            "/answer",
            data={"request": json_mod.dumps({"query": "q"})},
            files=[
                ("attachments", ("a.bin", b"x" * 10, "application/octet-stream")),
                ("attachments", ("b.bin", b"y" * 10, "application/octet-stream")),
            ],
        )

        assert resp.status_code == 413
        mock_manager.astart_answer_run.assert_not_awaited()


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
    """Runtime failure mappings the app still owns for its non-durable routes."""

    async def test_rejected_metadata_is_a_client_error_not_a_500(
        self, client: AsyncClient, mock_config: DlightragConfig, mock_manager
    ) -> None:
        """Metadata validation happens below the request model, so it needs its own mapping."""
        from dlightrag_rag.retrieval.metadata_fields import MetadataValidationError

        mock_manager.aupdate_metadata = AsyncMock(
            side_effect=MetadataValidationError("title is a built-in metadata field")
        )
        app.state.manager = mock_manager
        resp = await client.post("/metadata/doc-1", json={"metadata": {"title": "X"}})
        assert resp.status_code == 400
        assert resp.json()["error_type"] == "validation"


class TestAPIContracts:
    """Request and response contracts are explicit in OpenAPI."""

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
        assert "AnswerRunDescriptor" in schemas
        assert "AnswerRunStatusResponse" in schemas
        assert "AnswerRunStatus" not in schemas
        assert "AnswerRunPhase" not in schemas
        assert schemas["AnswerRunDescriptor"]["properties"]["status"] == {
            "type": "string",
            "enum": ["queued", "running", "succeeded", "failed", "cancelled"],
            "title": "Status",
        }
        phase_schema = schemas["AnswerRunStatusResponse"]["properties"]["phase"]
        assert phase_schema["anyOf"][0] == {
            "type": "string",
            "enum": ["planning", "searching", "researching", "generating"],
        }
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
    async def test_search_route_is_not_shadowed_by_the_doc_id_route(
        self,
        client: AsyncClient,
        mock_config: DlightragConfig,
        mock_manager,
    ) -> None:
        """`/metadata/search` is a literal path, so it must be declared first."""
        mock_manager.asearch_metadata = AsyncMock(return_value=["doc-1"])
        app.state.manager = mock_manager

        resp = await client.post("/metadata/search", json={"custom": {"department": "legal"}})

        assert resp.status_code == 200
        assert resp.json()["document_ids"] == ["doc-1"]

    @pytest.mark.usefixtures("_patch_manager")
    async def test_unknown_filter_name_is_rejected_not_ignored(
        self,
        client: AsyncClient,
        mock_config: DlightragConfig,
        mock_manager,
    ) -> None:
        """A dropped filter name would match every document instead of failing."""
        mock_manager.asearch_metadata = AsyncMock(return_value=["doc-1"])
        app.state.manager = mock_manager

        resp = await client.post("/metadata/search", json={"nonsense": "x"})

        assert resp.status_code == 422
        mock_manager.asearch_metadata.assert_not_awaited()


# ---------------------------------------------------------------------------
# Request body limits
# ---------------------------------------------------------------------------


def _echo_length_app(max_bytes: int) -> FastAPI:
    from starlette.requests import Request

    from dlightrag.api.middleware import RequestBodyLimitMiddleware

    application = FastAPI()

    @application.post("/probe")
    async def probe(request: Request) -> dict[str, int]:
        return {"received": len(await request.body())}

    @application.post("/answer")
    @application.post("/web/answer")
    async def answer(request: Request) -> dict[str, int]:
        return {"received": len(await request.body())}

    application.add_middleware(
        RequestBodyLimitMiddleware,
        max_bytes=max_bytes,
        multipart_path_max_bytes={"/answer": max_bytes, "/web/answer": max_bytes},
    )
    return application


async def _post(app: FastAPI, path: str = "/probe", **kwargs: Any) -> Response:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as client:
        return await client.post(path, **kwargs)


@pytest.mark.asyncio
async def test_a_declared_oversize_body_is_refused_before_the_route_runs() -> None:
    response = await _post(
        _echo_length_app(100),
        content=b"x" * 200,
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 413, response.text
    assert response.json()["error_type"] == "validation"


@pytest.mark.asyncio
async def test_an_undeclared_body_returns_413_at_the_cap() -> None:
    async def chunks():
        for _ in range(10):
            yield b"x" * 50

    response = await _post(
        _echo_length_app(100),
        content=chunks(),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 413
    assert response.json()["error_type"] == "validation"


@pytest.mark.asyncio
async def test_an_undeclared_mislabeled_body_still_returns_413_at_the_cap() -> None:
    async def chunks():
        for _ in range(10):
            yield b"x" * 50

    response = await _post(
        _echo_length_app(100),
        content=chunks(),
        headers={"content-type": "text/plain"},
    )

    assert response.status_code == 413
    assert response.json()["error_type"] == "validation"


@pytest.mark.asyncio
async def test_an_unmapped_multipart_upload_uses_the_default_cap() -> None:
    response = await _post(
        _echo_length_app(100),
        files={"f": ("big.bin", b"x" * 400)},
    )

    assert response.status_code == 413
    assert response.json()["error_type"] == "validation"


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/answer", "/web/answer"])
async def test_a_chunked_answer_multipart_is_refused_at_receive_layer(path: str) -> None:
    async def chunks():
        for _ in range(10):
            yield b"x" * 50

    response = await _post(
        _echo_length_app(100),
        path,
        content=chunks(),
        headers={"content-type": "multipart/form-data; boundary=test"},
    )

    assert response.status_code == 413
    assert response.json()["error_type"] == "validation"


@pytest.mark.asyncio
async def test_real_app_returns_413_for_chunked_answer_multipart_overflow(
    mock_config: DlightragConfig,
) -> None:
    mock_config.answer.max_total_attachment_bytes = 64
    set_config(mock_config)

    async def chunks():
        yield (
            b"--test\r\n"
            b'Content-Disposition: form-data; name="attachments"; filename="huge.bin"\r\n'
            b"Content-Type: application/octet-stream\r\n\r\n"
        )
        for _ in range(105):
            yield b"x" * 65_536
        yield b"\r\n--test--\r\n"

    application = create_app(include_web_app=False)
    manager = AsyncMock()
    manager.astart_ingest_job.return_value = {
        "job_id": "job-overflow",
        "workspace": "default",
        "source_type": "local",
        "status": "queued",
        "lease_owner": None,
        "lease_expires_at": None,
    }
    application.state.manager = manager
    response = await _post(
        application,
        "/answer",
        content=chunks(),
        headers={
            "content-type": "multipart/form-data; boundary=test",
            "origin": "https://example.test",
            "x-request-id": "body-limit-test",
        },
    )

    assert response.status_code == 413
    assert response.json()["error_type"] == "validation"
    assert response.headers["x-request-id"] == "body-limit-test"
    assert response.headers["access-control-allow-origin"] == "*"


@pytest.mark.asyncio
async def test_real_app_caps_chunked_ingest_multipart_before_parsing(
    mock_config: DlightragConfig,
) -> None:
    mock_config.max_upload_size_mb = 8
    mock_config.max_upload_bytes = 1024 * 1024
    set_config(mock_config)

    async def chunks():
        yield (
            b"--test\r\n"
            b'Content-Disposition: form-data; name="file"; filename="huge.bin"\r\n'
            b"Content-Type: application/octet-stream\r\n\r\n"
        )
        for _ in range(50):
            yield b"x" * 65_536
        yield b"\r\n--test--\r\n"

    application = create_app(include_web_app=False)
    manager = AsyncMock()
    manager.astart_ingest_job.return_value = {
        "job_id": "job-overflow",
        "workspace": "default",
        "source_type": "local",
        "status": "queued",
        "lease_owner": None,
        "lease_expires_at": None,
    }
    application.state.manager = manager
    response = await _post(
        application,
        "/ingest/blob",
        content=chunks(),
        headers={"content-type": "multipart/form-data; boundary=test"},
    )

    assert response.status_code == 413, response.text
    assert response.json()["error_type"] == "validation"
    assert response.json()["detail"] == "Request body is too large"
    manager.astart_ingest_job.assert_not_awaited()


@pytest.mark.asyncio
async def test_ingest_blob_authenticates_before_parsing_multipart(
    mock_config: DlightragConfig,
) -> None:
    mock_config.auth_mode = "simple"
    mock_config.api_auth_token = "secret-token"
    set_config(mock_config)
    application = create_app(include_web_app=False)
    application.state.manager = AsyncMock()

    response = await _post(
        application,
        "/ingest/blob",
        content=b"malformed multipart body",
        headers={"content-type": "multipart/form-data; boundary=missing"},
    )

    assert response.status_code == 401
    assert "Authorization" in response.json()["detail"]


@pytest.mark.asyncio
async def test_multipart_header_does_not_raise_json_route_body_cap(
    mock_config: DlightragConfig,
) -> None:
    mock_config.max_upload_size_mb = 32
    set_config(mock_config)

    async def chunks():
        yield b'--test\r\nContent-Disposition: form-data; name="junk"\r\n\r\n'
        for _ in range(200):
            yield b"x" * 65_536
        yield b"\r\n--test--\r\n"

    application = create_app(include_web_app=False)
    application.state.manager = AsyncMock()
    response = await _post(
        application,
        "/retrieve",
        content=chunks(),
        headers={"content-type": "multipart/form-data; boundary=test"},
    )

    assert response.status_code == 413
    assert response.json()["detail"] == "Request body is too large"


@pytest.mark.asyncio
async def test_the_app_admits_answer_history_with_the_shared_body_cap(
    mock_config: DlightragConfig,
) -> None:
    set_config(mock_config)

    response = await _post(
        create_app(include_web_app=False),
        content=b'{"query":"' + b"x" * (1024 * 1024) + b'"}',
        headers={"content-type": "application/json"},
    )

    assert response.status_code != 413


@pytest.mark.asyncio
async def test_the_app_still_refuses_a_body_over_the_shared_json_budget(
    mock_config: DlightragConfig,
) -> None:
    from dlightrag.core.client_contracts import (
        MAX_HISTORY_CONTENT_CHARS,
        MAX_HISTORY_MESSAGES,
        MAX_QUERY_IMAGES,
    )

    set_config(mock_config)
    history_bytes = MAX_HISTORY_MESSAGES * MAX_HISTORY_CONTENT_CHARS * 4
    image_bytes = MAX_QUERY_IMAGES * (((mock_config.answer.image_max_bytes + 2) // 3) * 4)
    over_budget = max(history_bytes, image_bytes) + 2 * 1024 * 1024

    response = await _post(
        create_app(include_web_app=False),
        content=b'{"query":"' + b"x" * over_budget + b'"}',
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 413


@pytest.mark.asyncio
async def test_the_app_admits_the_fixed_retrieve_image_contract(
    mock_config: DlightragConfig,
) -> None:
    from dlightrag.core.client_contracts import MAX_QUERY_IMAGES

    set_config(mock_config)
    image_sized_body = (
        MAX_QUERY_IMAGES * (((mock_config.answer.image_max_bytes + 2) // 3) * 4) - 4096
    )

    response = await _post(
        create_app(include_web_app=False),
        content=b'{"query":"' + b"x" * image_sized_body + b'"}',
        headers={"content-type": "application/json"},
    )

    assert response.status_code != 413


@pytest.mark.asyncio
async def test_a_route_that_rejects_an_oversized_upload_is_not_reported_as_an_auth_failure(
    mock_config: DlightragConfig,
) -> None:
    set_config(mock_config)
    application = create_app(include_web_app=False)

    @application.post("/probe")
    async def probe() -> None:
        raise HTTPException(status_code=413, detail="too many documents")

    response = await _post(application)

    assert response.status_code == 413
    assert response.json()["error_type"] == "validation"


def test_body_limit_split_preserves_non_limit_exception_group_members() -> None:
    from dlightrag.api.middleware import _RequestBodyTooLarge, _split_body_too_large

    matched, remainder = _split_body_too_large(
        ExceptionGroup(
            "mixed",
            [_RequestBodyTooLarge(), ExceptionGroup("server", [RuntimeError("boom")])],
        )
    )

    assert matched is not None
    assert isinstance(remainder, BaseExceptionGroup)
    server_group = remainder.exceptions[0]
    assert isinstance(server_group, BaseExceptionGroup)
    assert isinstance(server_group.exceptions[0], RuntimeError)
    assert str(server_group.exceptions[0]) == "boom"


def test_body_limit_strips_root_path_only_at_segment_boundary() -> None:
    from dlightrag.api.middleware import _request_path

    assert _request_path({"path": "/answer", "root_path": "/a"}) == "/answer"
    assert _request_path({"path": "/api/answer", "root_path": "/api"}) == "/answer"
