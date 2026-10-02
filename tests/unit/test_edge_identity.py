# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Edge-asserted Web identity against a live issuer: discovery, keys, middleware, config."""

import json
import threading
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from jwt.algorithms import RSAAlgorithm

from dlightrag.adapters.http.browser.auth import WebAuthMiddleware
from dlightrag.adapters.http.browser.edge_identity import EdgeIdentityError, authenticate_edge
from dlightrag.application.access import AuthenticationSettings
from dlightrag.application.config import DlightragConfig

AUDIENCE = "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0"
_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(scope="module")
def issuer() -> Iterator[str]:
    """An identity provider that publishes its keys the way OpenID discovery says.

    Under ``/impostor`` it serves a discovery document that names another issuer.
    """
    jwk = {**RSAAlgorithm.to_jwk(_KEY.public_key(), as_dict=True), "kid": "k1", "alg": "RS256"}
    documents: dict[str, object] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            document = documents.get(self.path)
            body = json.dumps(document).encode()
            self.send_response(200 if document else 404)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return None

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    base = f"http://127.0.0.1:{server.server_port}"
    discovery = {"issuer": base, "jwks_uri": f"{base}/certs"}
    documents.update(
        {
            "/.well-known/openid-configuration": discovery,
            "/certs": {"keys": [jwk]},
            "/impostor/.well-known/openid-configuration": discovery,
        }
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield base
    finally:
        server.shutdown()


def _token(
    issuer: str,
    *,
    audience: str = AUDIENCE,
    expires_in: timedelta = timedelta(minutes=5),
) -> str:
    return jwt.encode(
        {
            "sub": "edge-user-1",
            "iss": issuer,
            "aud": audience,
            "email": "edge-user-1@example.com",
            "exp": datetime.now(UTC) + expires_in,
            "iat": datetime.now(UTC),
        },
        _KEY,
        algorithm="RS256",
        headers={"kid": "k1"},
    )


def _request(
    *, headers: dict[str, str] | None = None, cookies: dict[str, str] | None = None
) -> Request:
    header_list = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    if cookies:
        cookie_header = "; ".join(f"{k}={v}" for k, v in cookies.items())
        header_list.append((b"cookie", cookie_header.encode()))
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/web/",
        "headers": header_list,
        "query_string": b"",
        "server": ("testserver", 80),
        "scheme": "http",
        "client": ("127.0.0.1", 1234),
        "root_path": "",
        "app": None,
        "state": {},
    }
    return Request(scope)


def _verifier(issuer: str) -> AuthenticationSettings:
    return AuthenticationSettings(mode="jwt", jwt_issuer=issuer, jwt_audience=AUDIENCE)


@pytest.mark.parametrize(
    ("edge", "carrier"),
    [
        ("cloudflare", lambda token: {"headers": {"Cf-Access-Jwt-Assertion": token}}),
        ("cloudflare", lambda token: {"cookies": {"CF_Authorization": token}}),
        ("azure", lambda token: {"headers": {"X-MS-TOKEN-AAD-ID-TOKEN": token}}),
        ("aws", lambda token: {"headers": {"Authorization": f"Bearer {token}"}}),
    ],
)
def test_each_edge_verifies_the_token_where_it_rides(issuer: str, edge: str, carrier) -> None:
    user = authenticate_edge(
        _request(**carrier(_token(issuer))), edge=edge, settings=_verifier(issuer)
    )

    assert (user.user_id, user.auth_mode, user.claims["iss"]) == ("edge-user-1", "jwt", issuer)


def test_a_missing_or_unverifiable_edge_token_is_refused(issuer: str) -> None:
    good = _token(issuer)
    cases = [
        ("cloudflare", {}, "missing_credential"),
        # Azure's principal header is unsigned; without the ID token there is no caller.
        ("azure", {"headers": {"X-MS-CLIENT-PRINCIPAL": "eyJhIjoxfQ"}}, "missing_credential"),
        (
            "cloudflare",
            {"headers": {"Cf-Access-Jwt-Assertion": _token(issuer, audience="other")}},
            "invalid_credential",
        ),
        (
            "cloudflare",
            {"headers": {"Cf-Access-Jwt-Assertion": _token("https://other.example")}},
            "invalid_credential",
        ),
        (
            "cloudflare",
            {
                "headers": {
                    "Cf-Access-Jwt-Assertion": good[:-4]
                    + ("AAAA" if good[-4:] != "AAAA" else "BBBB")
                }
            },
            "invalid_credential",
        ),
        (
            "cloudflare",
            {
                "headers": {
                    "Cf-Access-Jwt-Assertion": _token(issuer, expires_in=timedelta(seconds=-60))
                }
            },
            "expired_credential",
        ),
    ]

    for edge, carrier, kind in cases:
        with pytest.raises(EdgeIdentityError) as raised:
            authenticate_edge(_request(**carrier), edge=edge, settings=_verifier(issuer))
        assert raised.value.kind == kind


def test_a_discovery_document_naming_another_issuer_is_a_broken_verifier(issuer: str) -> None:
    impostor = f"{issuer}/impostor"

    with pytest.raises(EdgeIdentityError) as raised:
        authenticate_edge(
            _request(headers={"Cf-Access-Jwt-Assertion": _token(impostor)}),
            edge="cloudflare",
            settings=_verifier(impostor),
        )
    assert raised.value.kind == "misconfigured"


class TestEdgeIdentityConfig:
    def test_edge_requires_jwt_and_an_issuer_and_audience(self) -> None:
        with pytest.raises(ValueError, match="auth_mode='jwt'"):
            DlightragConfig(  # pyright: ignore[reportCallIssue, reportArgumentType]
                access={"auth_mode": "none", "web_identity": {"edge": "cloudflare"}},
            )
        with pytest.raises(ValueError, match="requires an issuer and an audience"):
            DlightragConfig(  # pyright: ignore[reportCallIssue, reportArgumentType]
                access={
                    "auth_mode": "jwt",
                    "jwt_verification_key": "static-key",
                    "web_identity": {"edge": "cloudflare"},
                },
            )

    def test_audience_accepts_a_json_array_string(self) -> None:
        config = DlightragConfig(  # pyright: ignore[reportCallIssue, reportArgumentType]
            access={
                "auth_mode": "jwt",
                "jwt_issuer": "https://team.example",
                "jwt_audience": "api",
                "web_identity": {"edge": "cloudflare", "audience": '["a", "b"]'},
            },
        )
        assert config.access.web_identity.audience == ["a", "b"]


class TestWebEdgeMiddleware:
    def _app(self, cfg: DlightragConfig) -> FastAPI:
        app = FastAPI()
        app.add_middleware(WebAuthMiddleware, config_getter=lambda: cfg)

        @app.get("/web/")
        async def home(request: Request) -> dict[str, object]:
            user = request.state.user_context
            assert user is not None
            return {"user_id": user.user_id, "auth_mode": user.auth_mode, "iss": user.claims["iss"]}

        return app

    def _config(self, issuer: str, **web_identity: str) -> DlightragConfig:
        return DlightragConfig(  # pyright: ignore[reportCallIssue, reportArgumentType]
            access={
                "auth_mode": "jwt",
                "jwt_issuer": issuer,
                "jwt_audience": AUDIENCE,
                "web_identity": {"edge": "cloudflare", **web_identity},
            },
        )

    def test_the_web_verifies_the_edge_token_with_the_api_issuer(self, issuer: str) -> None:
        client = TestClient(self._app(self._config(issuer)))

        response = client.get("/web/", headers={"Cf-Access-Jwt-Assertion": _token(issuer)})

        assert response.status_code == 200
        assert response.json() == {"user_id": "edge-user-1", "auth_mode": "jwt", "iss": issuer}

    def test_the_web_may_name_an_audience_of_its_own(self, issuer: str) -> None:
        client = TestClient(self._app(self._config(issuer, audience="web-client")))

        ours = client.get(
            "/web/", headers={"Cf-Access-Jwt-Assertion": _token(issuer, audience="web-client")}
        )
        api = client.get("/web/", headers={"Cf-Access-Jwt-Assertion": _token(issuer)})

        assert (ours.status_code, api.status_code) == (200, 401)

    def test_missing_assertion_is_401_with_no_login_redirect(self, issuer: str) -> None:
        client = TestClient(self._app(self._config(issuer)))
        for response in (
            client.get("/web/", follow_redirects=False),
            client.get("/web/api/bootstrap", follow_redirects=False),
            client.post("/web/api/conversations", follow_redirects=False),
        ):
            assert response.status_code == 401
            assert response.json() == {"detail": "Authentication required", "error_type": "auth"}

    def test_paste_cookie_is_ignored_when_edge_is_configured(self, issuer: str) -> None:
        import base64

        client = TestClient(self._app(self._config(issuer)))
        # A pasted-token cookie must not satisfy the edge-configured surface.
        pasted = base64.urlsafe_b64encode(b"irrelevant").decode().rstrip("=")
        response = client.get("/web/", cookies={"dlightrag_web_auth": pasted})
        assert response.status_code == 401

    def test_paste_path_stays_when_no_edge_is_configured(self) -> None:
        cfg = DlightragConfig(  # pyright: ignore[reportCallIssue, reportArgumentType]
            access={"auth_mode": "jwt", "jwt_verification_key": "some-key"},
        )
        client = TestClient(self._app(cfg))
        response = client.get("/web/", follow_redirects=False)
        # Without an edge, a browser GET goes to the login page.
        assert response.status_code == 303
        assert response.headers["location"].startswith("/web/login")
