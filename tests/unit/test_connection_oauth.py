# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""SDK OAuth integration uses an in-process AS and MCP, never real services."""

import base64
import hashlib
import json
from urllib.parse import parse_qs, urlsplit

import httpx2
import pytest
from pydantic import SecretStr

from dlightrag.application.connections import ConnectionPolicy
from tests.support.dns import public_dns


class FakeAuthorizationServer:
    def __init__(self):
        self.requests = []
        self.authorization: dict[str, list[str]] = {}
        self.token_target = "https://as.example/token"
        self.resource = "https://mcp.example/mcp"
        self.fail_discovery = False
        self.scope = "read"
        self.granted_scope = "read"
        self.authorization_endpoint = "https://as.example/authorize"

    def __call__(self, request):
        self.requests.append(request)
        assert request.url.host == "93.184.216.34"
        assert request.extensions["sni_hostname"] == request.headers["host"]
        assert "cookie" not in request.headers
        host, path = request.headers["host"], request.url.path
        if path.startswith("/.well-known/oauth-protected-resource"):
            assert "authorization" not in request.headers
            return httpx2.Response(
                200,
                json={
                    "resource": self.resource,
                    "authorization_servers": ["https://as.example"],
                    "scopes_supported": self.scope.split(),
                },
            )
        if host == "as.example":
            if path.startswith("/.well-known/"):
                return httpx2.Response(
                    200,
                    json={
                        "issuer": "https://as.example",
                        "authorization_endpoint": self.authorization_endpoint,
                        "token_endpoint": self.token_target,
                        "registration_endpoint": "https://as.example/register",
                        "response_types_supported": ["code"],
                        "code_challenge_methods_supported": ["S256"],
                        "token_endpoint_auth_methods_supported": ["client_secret_basic"],
                    },
                )
            if path == "/register":
                return httpx2.Response(
                    201,
                    json={
                        **json.loads(request.content),
                        "client_id": "test-client",
                        "client_secret": "test-client-secret",
                        "token_endpoint_auth_method": "client_secret_basic",
                    },
                )
            if path == "/token":
                body = parse_qs(request.content.decode())
                assert body["code"] == ["test-code"]
                assert body["resource"] == ["https://mcp.example/mcp"]
                assert (
                    request.headers["authorization"]
                    == "Basic " + base64.b64encode(b"test-client:test-client-secret").decode()
                )
                challenge = (
                    base64.urlsafe_b64encode(
                        hashlib.sha256(body["code_verifier"][0].encode()).digest()
                    )
                    .decode()
                    .rstrip("=")
                )
                assert self.authorization["code_challenge"] == [challenge]
                return httpx2.Response(
                    200,
                    json={
                        "access_token": "test-access-token",
                        "refresh_token": "test-refresh-token",
                        "token_type": "Bearer",
                        "expires_in": 300,
                        "scope": self.granted_scope,
                    },
                )
        if host == "mcp.example" and path == "/mcp":
            if request.headers.get("authorization") != "Bearer test-access-token":
                return httpx2.Response(
                    401,
                    headers={
                        "www-authenticate": f'Bearer resource_metadata="https://mcp.example/.well-known/oauth-protected-resource", scope="{self.scope}"'
                    },
                )
            if self.fail_discovery:
                return httpx2.Response(500)
            if request.method != "POST":
                return httpx2.Response(405)
            body = json.loads(request.content)
            if "id" not in body:
                return httpx2.Response(202)
            result = (
                {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "fake", "version": "1"},
                }
                if body["method"] == "initialize"
                else {"tools": [{"name": "read", "inputSchema": {"type": "object"}}]}
            )
            return httpx2.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})
        raise AssertionError(f"Unexpected fixture target {host} {path}")


@pytest.mark.asyncio
async def test_sdk_authorization_pkce_resource_and_token_storage(monkeypatch, caplog):
    from dlightrag.adapters.mcp.oauth import PersonalOAuthClient

    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    server = FakeAuthorizationServer()
    saved = []

    async def redirect(url):
        server.authorization = parse_qs(urlsplit(url).query)

    async def callback():
        return SecretStr(
            json.dumps(
                {
                    "code": "test-code",
                    "state": server.authorization["state"][0],
                    "iss": "https://as.example",
                }
            )
        )

    async def save(secret):
        saved.append(secret)

    result = await PersonalOAuthClient(
        transport_factory=lambda: httpx2.MockTransport(server)
    ).authorize(
        endpoint="https://mcp.example/mcp",
        callback_url="https://app.example/web/oauth/connections/mcp/callback",
        policy=ConnectionPolicy(),
        redirect=redirect,
        callback=callback,
        save=save,
    )
    assert result.tools[0]["name"] == "read"
    assert result.scopes == ("read",)
    assert len(saved) == 2
    credentials = json.loads(result.credentials.get_secret_value())
    assert credentials["tokens"]["refresh_token"] == "test-refresh-token"
    assert credentials["client_info"]["client_id"] == "test-client"
    assert "test-client-secret" not in caplog.text
    assert "test-access-token" not in caplog.text
    assert "test-code" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "target",
    [
        "http://as.example/token",
        "https://127.0.0.1/token",
        "https://as.example/token?access_token=secret",
        "https://user:secret@as.example/token",
    ],
)
async def test_sdk_token_targets_are_admitted_before_credentials_leave(target, monkeypatch):
    from dlightrag.adapters.mcp.oauth import PersonalOAuthClient
    from dlightrag.application.connections import ConnectionsError

    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    server = FakeAuthorizationServer()
    server.token_target = target

    async def redirect(url):
        server.authorization = parse_qs(urlsplit(url).query)

    async def callback():
        return SecretStr(
            json.dumps({"code": "test-code", "state": server.authorization["state"][0]})
        )

    async def save(secret):
        pass

    with pytest.raises(ConnectionsError):
        await PersonalOAuthClient(transport_factory=lambda: httpx2.MockTransport(server)).authorize(
            endpoint="https://mcp.example/mcp",
            callback_url="https://app.example/web/oauth/connections/mcp/callback",
            policy=ConnectionPolicy(require_https=False),
            redirect=redirect,
            callback=callback,
            save=save,
        )
    assert not any(r.url.path == "/token" for r in server.requests)


@pytest.mark.asyncio
async def test_sdk_rejects_resource_audience_mismatch_before_registration(monkeypatch):
    from dlightrag.adapters.mcp.oauth import PersonalOAuthClient
    from dlightrag.application.connections import ConnectionsError

    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    server = FakeAuthorizationServer()
    server.resource = "https://different.example/mcp"

    async def forbidden(*args):
        raise AssertionError("No authorization or storage on mismatched resource")

    with pytest.raises(ConnectionsError):
        await PersonalOAuthClient(transport_factory=lambda: httpx2.MockTransport(server)).authorize(
            endpoint="https://mcp.example/mcp",
            callback_url="https://app.example/web/oauth/connections/mcp/callback",
            policy=ConnectionPolicy(),
            redirect=forbidden,
            callback=forbidden,
            save=forbidden,
        )
    assert not any(r.headers["host"] == "as.example" for r in server.requests)


@pytest.mark.asyncio
async def test_token_scope_outside_provider_consent_is_rejected_before_mcp_repost(monkeypatch):
    from dlightrag.adapters.mcp.oauth import PersonalOAuthClient
    from dlightrag.application.connections import ConnectionsError

    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    server = FakeAuthorizationServer()
    server.granted_scope = "read write"

    async def redirect(url):
        server.authorization = parse_qs(urlsplit(url).query)

    async def callback():
        return SecretStr(
            json.dumps({"code": "test-code", "state": server.authorization["state"][0]})
        )

    async def save(secret):
        pass

    with pytest.raises(ConnectionsError):
        await PersonalOAuthClient(transport_factory=lambda: httpx2.MockTransport(server)).authorize(
            endpoint="https://mcp.example/mcp",
            callback_url="https://app.example/web/oauth/connections/mcp/callback",
            policy=ConnectionPolicy(),
            redirect=redirect,
            callback=callback,
            save=save,
        )
    assert not any(
        r.headers.get("authorization") == "Bearer test-access-token" for r in server.requests
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "target",
    [
        "https://127.0.0.1/authorize",
        "http://as.example/authorize",
        "https://as.example/authorize#fragment",
    ],
)
async def test_browser_authorization_target_is_admitted_before_settings_redirect(
    target, monkeypatch
):
    from dlightrag.adapters.mcp.oauth import PersonalOAuthClient
    from dlightrag.application.connections import ConnectionsError

    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    server = FakeAuthorizationServer()
    server.authorization_endpoint = target
    redirects = []

    async def redirect(url):
        redirects.append(url)

    async def callback():
        raise AssertionError("Rejected target must not wait for callback")

    async def save(secret):
        pass

    with pytest.raises(ConnectionsError):
        await PersonalOAuthClient(transport_factory=lambda: httpx2.MockTransport(server)).authorize(
            endpoint="https://mcp.example/mcp",
            callback_url="https://app.example/web/oauth/connections/mcp/callback",
            policy=ConnectionPolicy(require_https=False),
            redirect=redirect,
            callback=callback,
            save=save,
        )
    assert redirects == []


@pytest.mark.asyncio
async def test_sdk_token_redirect_preserves_basic_and_rechecks_pinning(monkeypatch):
    from dlightrag.adapters.mcp.oauth import PersonalOAuthClient

    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    server = FakeAuthorizationServer()
    redirected = []

    def handler(request):
        if request.url.path == "/token":
            return httpx2.Response(
                307, headers={"location": "/exchange", "set-cookie": "must-not-forward=secret"}
            )
        if request.url.path == "/exchange":
            redirected.append(request)
            assert request.headers["host"] == "as.example"
            assert request.extensions["sni_hostname"] == "as.example"
            assert request.url.host == "93.184.216.34"
            assert "cookie" not in request.headers
            request = httpx2.Request(
                request.method,
                request.url.copy_with(path="/token"),
                headers=request.headers,
                content=request.content,
                extensions=request.extensions,
            )
        return server(request)

    async def redirect(url):
        server.authorization = parse_qs(urlsplit(url).query)

    async def callback():
        return SecretStr(
            json.dumps({"code": "test-code", "state": server.authorization["state"][0]})
        )

    async def save(secret):
        pass

    result = await PersonalOAuthClient(
        transport_factory=lambda: httpx2.MockTransport(handler)
    ).authorize(
        endpoint="https://mcp.example/mcp",
        callback_url="https://app.example/web/oauth/connections/mcp/callback",
        policy=ConnectionPolicy(),
        redirect=redirect,
        callback=callback,
        save=save,
    )
    assert result.tools[0]["name"] == "read"
    assert len(redirected) == 1


@pytest.mark.asyncio
async def test_oauth_catalogue_and_logs_cannot_echo_flow_credentials(monkeypatch, caplog):
    import logging

    from dlightrag.adapters.mcp.oauth import PersonalOAuthClient

    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    caplog.set_level(logging.DEBUG)
    server = FakeAuthorizationServer()

    def handler(request):
        response = server(request)
        if (
            request.method == "POST"
            and request.headers["host"] == "mcp.example"
            and request.headers.get("authorization")
        ):
            body = json.loads(request.content)
            if body.get("method") == "tools/list":
                return httpx2.Response(
                    200,
                    json={
                        "jsonrpc": "2.0",
                        "id": body["id"],
                        "result": {
                            "tools": [
                                {
                                    "name": "read",
                                    "description": "test-access-token test-refresh-token test-client-secret test-code",
                                    "inputSchema": {"type": "object"},
                                }
                            ]
                        },
                    },
                )
        return response

    async def redirect(url):
        server.authorization = parse_qs(urlsplit(url).query)

    async def callback():
        return SecretStr(
            json.dumps({"code": "test-code", "state": server.authorization["state"][0]})
        )

    async def save(secret):
        pass

    result = await PersonalOAuthClient(
        transport_factory=lambda: httpx2.MockTransport(handler)
    ).authorize(
        endpoint="https://mcp.example/mcp",
        callback_url="https://app.example/web/oauth/connections/mcp/callback",
        policy=ConnectionPolicy(),
        redirect=redirect,
        callback=callback,
        save=save,
    )
    for secret in ("test-access-token", "test-refresh-token", "test-client-secret", "test-code"):
        assert secret not in str(result.tools)
        assert secret not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metadata_target",
    [
        "https://127.0.0.1/metadata",
        "https://169.254.169.254/metadata",
        "http://as.example/metadata",
    ],
)
async def test_sdk_challenged_metadata_target_is_rejected_before_http(metadata_target, monkeypatch):
    from dlightrag.adapters.mcp.oauth import PersonalOAuthClient
    from dlightrag.application.connections import ConnectionsError

    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    calls = []

    def handler(request):
        calls.append(request)
        assert request.headers["host"] == "mcp.example"
        return httpx2.Response(
            401, headers={"www-authenticate": f'Bearer resource_metadata="{metadata_target}"'}
        )

    async def forbidden(*args):
        raise AssertionError("No registration or redirect for rejected metadata")

    with pytest.raises(ConnectionsError):
        await PersonalOAuthClient(
            transport_factory=lambda: httpx2.MockTransport(handler)
        ).authorize(
            endpoint="https://mcp.example/mcp",
            callback_url="https://app.example/web/oauth/connections/mcp/callback",
            policy=ConnectionPolicy(require_https=False),
            redirect=forbidden,
            callback=forbidden,
            save=forbidden,
        )
    assert len(calls) == 1


def refresh_credentials():
    return SecretStr(
        json.dumps(
            {
                "tokens": {
                    "access_token": "expired-token",
                    "refresh_token": "refresh-secret",
                    "token_type": "Bearer",
                    "expires_in": 1,
                    "scope": "read",
                },
                "expires_at": 1,
                "client_info": {
                    "client_id": "client",
                    "client_secret": "client-secret",
                    "redirect_uris": ["https://app.example/web/oauth/connections/mcp/callback"],
                    "token_endpoint_auth_method": "client_secret_basic",
                },
                "oauth_metadata": {
                    "issuer": "https://as.example",
                    "authorization_endpoint": "https://as.example/authorize",
                    "token_endpoint": "https://as.example/token",
                    "response_types_supported": ["code"],
                },
                "resource_metadata": {
                    "resource": "https://mcp.example/mcp",
                    "authorization_servers": ["https://as.example"],
                },
            }
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome", ["success", "scope", "rejected", "redirect", "cross-origin", "loop", "cas"]
)
async def test_sdk_refresh_preflight_sends_only_token_request_and_never_consent(
    outcome, monkeypatch
):
    from dlightrag.adapters.mcp.oauth import PersonalOAuthClient
    from dlightrag.application.connections import ConnectionsError

    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    requests, saved = [], []

    def remote(request):
        requests.append(request)
        assert request.headers["host"] == "as.example"
        assert request.url.path in {"/token", "/rotated-token"}
        body = parse_qs(request.content.decode())
        assert body["grant_type"] == ["refresh_token"]
        assert body["refresh_token"] == ["refresh-secret"]
        assert body["resource"] == ["https://mcp.example/mcp"]
        if outcome in {"cross-origin", "loop"}:
            return httpx2.Response(
                307,
                headers={
                    "location": "https://other.example/token"
                    if outcome == "cross-origin"
                    else "https://as.example/token"
                },
            )
        if outcome == "redirect" and request.url.path == "/token":
            return httpx2.Response(307, headers={"location": "https://as.example/rotated-token"})
        if outcome == "rejected":
            return httpx2.Response(400, json={"error": "invalid_grant"})
        return httpx2.Response(
            200,
            json={
                "access_token": "new-token",
                "token_type": "Bearer",
                "expires_in": 300,
                **({"scope": "read write"} if outcome == "scope" else {}),
            },
        )

    async def save(value):
        if outcome == "cas":
            raise ConnectionsError("Lease or secret version changed")
        saved.append(value)

    client = PersonalOAuthClient(transport_factory=lambda: httpx2.MockTransport(remote))
    if outcome in {"scope", "rejected", "cross-origin", "loop", "cas"}:
        with pytest.raises(ConnectionsError):
            await client.refresh(
                endpoint="https://mcp.example/mcp",
                credentials=refresh_credentials(),
                scopes=("read",),
                policy=ConnectionPolicy(),
                save=save,
            )
        assert saved == []
    else:
        result = await client.refresh(
            endpoint="https://mcp.example/mcp",
            credentials=refresh_credentials(),
            scopes=("read",),
            policy=ConnectionPolicy(),
            save=save,
        )
        raw = json.loads(result.get_secret_value())
        assert raw["tokens"]["access_token"] == "new-token"
        assert raw["tokens"]["refresh_token"] == "refresh-secret"
        assert raw["tokens"]["scope"] == "read"
        assert raw["expires_at"] > 1
        assert len(saved) == 1
        assert raw["oauth_metadata"]["token_endpoint"] == "https://as.example/token"
    assert len(requests) == (6 if outcome == "loop" else 2 if outcome == "redirect" else 1)
