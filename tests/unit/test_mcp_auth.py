# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""MCP 2.0 OAuth resource-server authentication contracts."""

from typing import Any

import jwt
import pytest
from mcp.server.auth.middleware.auth_context import auth_context_var
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from mcp.server.auth.provider import AccessToken

from dlightrag.access import UserContext, current_request_scope
from dlightrag.config import DlightragConfig, set_config
from dlightrag.mcp import auth as mcp_auth
from dlightrag.mcp.server import DlightRAGRequestScopeMiddleware


@pytest.mark.asyncio
async def test_mcp_access_token_preserves_identity_claims_and_scopes(
    monkeypatch: pytest.MonkeyPatch,
    test_config: DlightragConfig,
) -> None:
    claims: dict[str, Any] = {
        "sub": "alice",
        "azp": "agent-client",
        "scope": "openid dlightrag:query",
        "exp": 2_000_000_000,
        "groups": ["finance-rag-readers"],
    }
    monkeypatch.setattr(
        mcp_auth,
        "authenticate_bearer_token",
        lambda token, settings: UserContext(user_id="alice", auth_mode="jwt", claims=claims),
    )
    config = test_config.model_copy(update={"auth_mode": "jwt", "jwt_verification_key": "test-key"})
    set_config(config)
    verifier = mcp_auth.DlightRAGTokenVerifier(config)

    token = await verifier.verify_token("signed-token")

    assert token is not None
    assert token.token == "signed-token"
    assert token.client_id == "agent-client"
    assert token.subject == "alice"
    assert token.scopes == ["openid", "dlightrag:query"]
    assert token.expires_at == 2_000_000_000
    assert token.claims == claims

    auth_token = auth_context_var.set(AuthenticatedUser(token))

    async def capture_scope(ctx: Any) -> None:
        scope = current_request_scope()
        assert scope.user_id == "alice"
        assert scope.auth_mode == "jwt"
        assert scope.claims == claims

    previous = current_request_scope()
    try:
        await DlightRAGRequestScopeMiddleware()(object(), capture_scope)  # type: ignore[arg-type]
        assert current_request_scope() == previous
    finally:
        auth_context_var.reset(auth_token)


async def test_mcp_request_scope_restores_prior_scope_when_handler_fails(
    test_config: DlightragConfig,
) -> None:
    set_config(
        test_config.model_copy(update={"auth_mode": "jwt", "jwt_verification_key": "test-key"})
    )
    token = AccessToken(
        token="signed-token",
        client_id="client",
        scopes=[],
        subject="alice",
        claims={"sub": "alice"},
    )
    auth_token = auth_context_var.set(AuthenticatedUser(token))
    prior = current_request_scope()

    async def fail(ctx: Any) -> None:
        assert current_request_scope().user_id == "alice"
        raise RuntimeError("boom")

    try:
        with pytest.raises(RuntimeError, match="boom"):
            await DlightRAGRequestScopeMiddleware()(object(), fail)  # type: ignore[arg-type]
        assert current_request_scope() == prior
    finally:
        auth_context_var.reset(auth_token)


@pytest.mark.asyncio
async def test_mcp_oauth_requires_exact_resource_audience(test_config: DlightragConfig) -> None:
    signing_secret = "test-signing-secret-at-least-32-bytes"
    config = test_config.model_copy(
        update={
            "auth_mode": "jwt",
            "jwt_verification_key": signing_secret,
            "jwt_algorithm": "HS256",
            "jwt_audience": "api://rest",
        }
    )
    verifier = mcp_auth.DlightRAGTokenVerifier(
        config,
        resource="https://rag.example.com",
    )
    matching = jwt.encode(
        {"sub": "alice", "aud": "https://rag.example.com"},
        signing_secret,
        algorithm="HS256",
    )
    wrong = jwt.encode(
        {"sub": "alice", "aud": "api://rest"},
        signing_secret,
        algorithm="HS256",
    )

    assert await verifier.verify_token(matching) is not None
    assert await verifier.verify_token(wrong) is None
