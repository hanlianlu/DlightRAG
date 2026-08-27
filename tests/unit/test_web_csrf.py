# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""CSRF double-submit and origin hardening for state-changing web routes."""

from datetime import UTC, datetime, timedelta

import jwt
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from dlightrag.application.config import DlightragConfig
from dlightrag.web.auth import WEB_CSRF_COOKIE, WebAuthMiddleware


def _bearer(cfg: DlightragConfig) -> str:
    return jwt.encode(
        {
            "sub": "user-1",
            "iss": "https://issuer.example.com",
            "exp": datetime.now(UTC) + timedelta(minutes=5),
        },
        cfg.access.jwt_verification_key or "test-key",
        algorithm="HS256",
    )


def _app(cfg: DlightragConfig) -> FastAPI:
    app = FastAPI()
    app.add_middleware(WebAuthMiddleware, config_getter=lambda: cfg)

    @app.get("/web/")
    async def home(request: Request) -> dict[str, str]:
        return {"ok": "get"}

    @app.post("/web/")
    async def mutate(request: Request) -> dict[str, str]:
        return {"ok": "post"}

    @app.post("/web/login")
    async def login_route() -> dict[str, str]:
        return {"ok": "login"}

    return app


def _jwt_config() -> DlightragConfig:
    return DlightragConfig(  # pyright: ignore[reportCallIssue, reportArgumentType]
        access={
            "auth_mode": "jwt",
            "jwt_verification_key": "test-key",
        },
    )


def _client_with_session(cfg: DlightragConfig) -> tuple[TestClient, str, str]:
    """Return a client, its auth header, and the issued csrf token cookie value."""
    client = TestClient(_app(cfg))
    response = client.get("/web/", headers={"Authorization": f"Bearer {_bearer(cfg)}"})
    assert response.status_code == 200
    csrf_token = client.cookies.get(WEB_CSRF_COOKIE)
    assert csrf_token
    return client, f"Bearer {_bearer(cfg)}", csrf_token


class TestWebCsrfHardening:
    def test_get_issues_the_double_submit_cookie(self) -> None:
        client, _auth, csrf_token = _client_with_session(_jwt_config())
        assert csrf_token

    def test_mutation_without_header_is_rejected(self) -> None:
        client, auth, csrf_token = _client_with_session(_jwt_config())
        response = client.post("/web/", headers={"Authorization": auth})
        assert response.status_code == 403

    def test_mutation_with_matching_token_succeeds(self) -> None:
        client, auth, csrf_token = _client_with_session(_jwt_config())
        response = client.post(
            "/web/",
            headers={"Authorization": auth, "X-CSRF-Token": csrf_token},
        )
        assert response.status_code == 200
        assert response.json() == {"ok": "post"}

    def test_cookieless_scripted_client_is_left_to_bearer_credentials(self) -> None:
        client = TestClient(_app(_jwt_config()))
        response = client.post(
            "/web/", headers={"Authorization": f"Bearer {_bearer(_jwt_config())}"}
        )
        assert response.status_code == 200

    def test_cross_origin_mutation_is_rejected(self) -> None:
        client, auth, csrf_token = _client_with_session(_jwt_config())
        response = client.post(
            "/web/",
            headers={
                "Authorization": auth,
                "X-CSRF-Token": csrf_token,
                "Origin": "https://evil.example.com",
            },
        )
        assert response.status_code == 403

    def test_same_origin_mutation_succeeds(self) -> None:
        client, auth, csrf_token = _client_with_session(_jwt_config())
        response = client.post(
            "/web/",
            headers={
                "Authorization": auth,
                "X-CSRF-Token": csrf_token,
                "Origin": "http://testserver",
            },
        )
        assert response.status_code == 200

    def test_login_post_rejects_cross_origin_browsers(self) -> None:
        client = TestClient(_app(_jwt_config()))
        response = client.post(
            "/web/login",
            data={"token": "anything", "next": "/web/"},
            headers={"Origin": "https://evil.example.com"},
        )
        assert response.status_code == 403

    def test_logout_post_rejects_cross_origin_browsers(self) -> None:
        client = TestClient(_app(_jwt_config()))
        response = client.post(
            "/web/logout",
            headers={"Origin": "https://evil.example.com"},
        )
        assert response.status_code == 403
