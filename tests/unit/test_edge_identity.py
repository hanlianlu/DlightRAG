# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Cloudflare Access edge identity: verification, middleware, and config."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from dlightrag.config import DlightragConfig, WebIdentitySettings
from dlightrag.web.auth import WebAuthMiddleware
from dlightrag.web.edge_identity import (
    AzureEasyAuthProvider,
    CloudflareAccessProvider,
    EdgeIdentityError,
    edge_identity_provider,
)

TEAM_ISSUER = "https://dlightrag-team.cloudflareaccess.com"
AUD_TAG = "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6q7r8s9t0"


def _rsa_key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _access_token(
    key: rsa.RSAPrivateKey,
    *,
    issuer: str = TEAM_ISSUER,
    audience: str = AUD_TAG,
    subject: str = "edge-user-1",
    expires_in: timedelta = timedelta(minutes=5),
) -> str:
    return jwt.encode(
        {
            "sub": subject,
            "iss": issuer,
            "aud": audience,
            "email": "edge-user-1@example.com",
            "identity_nonce": "session-nonce",
            "exp": datetime.now(UTC) + expires_in,
            "iat": datetime.now(UTC),
        },
        key,
        algorithm="RS256",
        headers={"kid": "test-key"},
    )


@pytest.fixture
def signing_key() -> rsa.RSAPrivateKey:
    return _rsa_key()


@pytest.fixture
def jwks(monkeypatch: pytest.MonkeyPatch, signing_key: rsa.RSAPrivateKey) -> None:
    public_pem = (
        signing_key.public_key()
        .public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
        .decode()
    )
    from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey

    public_key = serialization.load_pem_public_key(public_pem.encode())
    assert isinstance(public_key, RSAPublicKey)

    class FakeJWKSClient:
        def get_signing_key_from_jwt(self, raw_token: str):
            return SimpleNamespace(key=public_key)

    monkeypatch.setattr("dlightrag.web.edge_identity._jwks_client", lambda _url: FakeJWKSClient())


def _settings(**overrides: object) -> WebIdentitySettings:
    fields: dict[str, object] = {
        "edge": "cloudflare",
        "issuer": TEAM_ISSUER,
        "audience": AUD_TAG,
    }
    return WebIdentitySettings.model_validate({**fields, **overrides})


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


class TestCloudflareAccessProvider:
    def test_header_assertion_verifies(self, signing_key, jwks) -> None:
        provider = CloudflareAccessProvider(_settings())
        identity = provider.authenticate(
            _request(headers={"Cf-Access-Jwt-Assertion": _access_token(signing_key)})
        )
        assert identity.subject == "edge-user-1"
        assert identity.issuer == TEAM_ISSUER
        assert identity.claims["identity_nonce"] == "session-nonce"

    def test_cookie_falls_back_to_the_authorization_jwt(self, signing_key, jwks) -> None:
        provider = CloudflareAccessProvider(_settings())
        identity = provider.authenticate(
            _request(cookies={"CF_Authorization": _access_token(signing_key)})
        )
        assert identity.subject == "edge-user-1"

    def test_missing_credential_is_a_401_rejection(self, jwks) -> None:
        provider = CloudflareAccessProvider(_settings())
        with pytest.raises(EdgeIdentityError) as raised:
            provider.authenticate(_request())
        assert raised.value.kind == "missing_credential"
        assert raised.value.kind in {
            "missing_credential",
            "invalid_credential",
            "expired_credential",
        }

    def test_wrong_issuer_is_rejected(self, signing_key, jwks) -> None:
        provider = CloudflareAccessProvider(_settings())
        with pytest.raises(EdgeIdentityError, match="Invalid edge credential"):
            provider.authenticate(
                _request(
                    headers={
                        "Cf-Access-Jwt-Assertion": _access_token(
                            signing_key, issuer="https://other-team.cloudflareaccess.com"
                        )
                    }
                )
            )

    def test_wrong_audience_is_rejected(self, signing_key, jwks) -> None:
        provider = CloudflareAccessProvider(_settings())
        with pytest.raises(EdgeIdentityError, match="Invalid edge credential"):
            provider.authenticate(
                _request(
                    headers={
                        "Cf-Access-Jwt-Assertion": _access_token(
                            signing_key, audience="other-aud-tag"
                        )
                    }
                )
            )

    def test_expired_token_is_rejected(self, signing_key, jwks) -> None:
        provider = CloudflareAccessProvider(_settings())
        with pytest.raises(EdgeIdentityError) as raised:
            provider.authenticate(
                _request(
                    headers={
                        "Cf-Access-Jwt-Assertion": _access_token(
                            signing_key, expires_in=timedelta(seconds=-60)
                        )
                    }
                )
            )
        assert raised.value.kind == "expired_credential"

    def test_tampered_signature_is_rejected(self, signing_key, jwks) -> None:
        provider = CloudflareAccessProvider(_settings())
        raw = _access_token(signing_key)
        tampered = raw[:-4] + ("AAAA" if raw[-4:] != "AAAA" else "BBBB")
        with pytest.raises(EdgeIdentityError, match="Invalid edge credential"):
            provider.authenticate(_request(headers={"Cf-Access-Jwt-Assertion": tampered}))

    def test_factory_rejects_unknown_edges(self) -> None:
        with pytest.raises(EdgeIdentityError) as raised:
            edge_identity_provider(_settings(edge="aws"))  # implemented in a later task
        assert raised.value.kind == "misconfigured"


class TestAzureEasyAuthProvider:
    AAD_ISSUER = "https://login.microsoftonline.com/test-tenant/v2.0"
    AAD_AUDIENCE = "api-client-id-1234"

    def _provider(self) -> AzureEasyAuthProvider:
        return AzureEasyAuthProvider(
            WebIdentitySettings.model_validate(
                {"edge": "azure", "issuer": self.AAD_ISSUER, "audience": self.AAD_AUDIENCE}
            )
        )

    def _id_token(
        self,
        key: rsa.RSAPrivateKey,
        *,
        issuer: str = AAD_ISSUER,
        audience: str = AAD_AUDIENCE,
    ) -> str:
        return jwt.encode(
            {
                "sub": "aad-user-object-id",
                "iss": issuer,
                "aud": audience,
                "name": "Edge User",
                "exp": datetime.now(UTC) + timedelta(minutes=5),
                "iat": datetime.now(UTC),
            },
            key,
            algorithm="RS256",
            headers={"kid": "test-key"},
        )

    def test_id_token_verifies_and_principal_is_enrichment_only(self, signing_key, jwks) -> None:
        import base64
        import json as _json

        provider = self._provider()
        principal = base64.b64encode(
            _json.dumps(
                {"auth_typ": "aad", "claims": [{"typ": "sub", "val": "aad-user-object-id"}]}
            ).encode()
        ).decode()
        identity = provider.authenticate(
            _request(
                headers={
                    "X-MS-TOKEN-AAD-ID-TOKEN": self._id_token(signing_key),
                    "X-MS-CLIENT-PRINCIPAL": principal,
                }
            )
        )
        assert identity.subject == "aad-user-object-id"
        assert identity.issuer == self.AAD_ISSUER
        assert identity.display_claims == {
            "auth_typ": "aad",
            "claims": [{"typ": "sub", "val": "aad-user-object-id"}],
        }
        # Enrichment never leaks into the authoritative claims.
        assert "auth_typ" not in identity.claims

    def test_principal_header_alone_is_rejected(self, jwks) -> None:
        provider = self._provider()
        with pytest.raises(EdgeIdentityError) as raised:
            provider.authenticate(_request(headers={"X-MS-CLIENT-PRINCIPAL": "eyJhIjoxfQ"}))
        assert raised.value.kind == "missing_credential"
        assert raised.value.kind in {
            "missing_credential",
            "invalid_credential",
            "expired_credential",
        }

    def test_wrong_tenant_issuer_is_rejected(self, signing_key, jwks) -> None:
        provider = self._provider()
        with pytest.raises(EdgeIdentityError, match="Invalid edge credential"):
            provider.authenticate(
                _request(
                    headers={
                        "X-MS-TOKEN-AAD-ID-TOKEN": self._id_token(
                            signing_key,
                            issuer="https://login.microsoftonline.com/other-tenant/v2.0",
                        )
                    }
                )
            )

    def test_wrong_audience_is_rejected(self, signing_key, jwks) -> None:
        provider = self._provider()
        with pytest.raises(EdgeIdentityError, match="Invalid edge credential"):
            provider.authenticate(
                _request(
                    headers={
                        "X-MS-TOKEN-AAD-ID-TOKEN": self._id_token(
                            signing_key, audience="other-client-id"
                        )
                    }
                )
            )

    def test_non_v2_issuer_requires_explicit_jwks_url(self) -> None:
        with pytest.raises(EdgeIdentityError) as raised:
            AzureEasyAuthProvider(
                WebIdentitySettings.model_validate(
                    {
                        "edge": "azure",
                        "issuer": "https://sts.windows.net/test-tenant/",
                        "audience": self.AAD_AUDIENCE,
                    }
                )
            )
        assert raised.value.kind == "misconfigured"


class TestEdgeIdentityConfig:
    def test_edge_requires_issuer_and_audience(self) -> None:
        with pytest.raises(ValueError, match="requires web_identity.issuer"):
            WebIdentitySettings(edge="cloudflare", audience=AUD_TAG)
        with pytest.raises(ValueError, match="requires web_identity.audience"):
            WebIdentitySettings(edge="cloudflare", issuer=TEAM_ISSUER)

    def test_edge_requires_jwt_auth_mode(self) -> None:
        with pytest.raises(ValueError, match="auth_mode='jwt'"):
            DlightragConfig(
                auth_mode="none",
                web_identity=_settings(),
            )

    def test_audience_accepts_a_json_array_string(self) -> None:
        settings = WebIdentitySettings(edge="cloudflare", issuer=TEAM_ISSUER, audience='["a", "b"]')
        assert settings.audience == ["a", "b"]


class TestWebEdgeMiddleware:
    def _app(self, cfg: DlightragConfig) -> FastAPI:
        app = FastAPI()
        app.add_middleware(WebAuthMiddleware, config_getter=lambda: cfg)

        @app.get("/web/")
        async def home(request: Request) -> dict[str, object]:
            user = request.state.user_context
            assert user is not None
            return {
                "user_id": user.user_id,
                "auth_mode": user.auth_mode,
                "iss": user.claims.get("iss"),
            }

        return app

    def _jwt_config(self) -> DlightragConfig:
        return DlightragConfig(
            auth_mode="jwt",
            jwt_verification_key="some-key-for-rest-bearers",
            web_identity=_settings(),
        )

    def test_valid_assertion_projects_the_edge_owner(self, signing_key, jwks) -> None:
        client = TestClient(self._app(self._jwt_config()))
        response = client.get(
            "/web/",
            headers={"Cf-Access-Jwt-Assertion": _access_token(signing_key)},
        )
        assert response.status_code == 200
        assert response.json() == {
            "user_id": "edge-user-1",
            "auth_mode": "jwt",
            "iss": TEAM_ISSUER,
        }

    def test_missing_assertion_is_401_with_no_login_redirect(self, jwks) -> None:
        client = TestClient(self._app(self._jwt_config()))
        response = client.get("/web/")
        assert response.status_code == 401
        assert response.text == "Authentication required"

    def test_paste_cookie_is_ignored_when_edge_is_configured(self, signing_key, jwks) -> None:
        import base64

        client = TestClient(self._app(self._jwt_config()))
        # A pasted-token cookie must not satisfy the edge-configured surface.
        pasted = base64.urlsafe_b64encode(b"irrelevant").decode().rstrip("=")
        response = client.get("/web/", cookies={"dlightrag_web_auth": pasted})
        assert response.status_code == 401

    def test_paste_path_stays_when_no_edge_is_configured(self) -> None:
        cfg = DlightragConfig(auth_mode="jwt", jwt_verification_key="some-key")
        client = TestClient(self._app(cfg))
        response = client.get("/web/", follow_redirects=False)
        # The legacy path redirects a browser GET to the login page.
        assert response.status_code == 303
        assert response.headers["location"].startswith("/web/login")
