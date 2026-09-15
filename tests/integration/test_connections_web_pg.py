# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Real Web authentication/CSRF and real PostgreSQL owner lifecycle, fake MCP."""

from types import SimpleNamespace

import jwt
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from dlightrag.adapters.http.browser.auth import WebAuthMiddleware
from dlightrag.adapters.http.browser.routes.connections import router
from dlightrag.adapters.postgres.connections import PGConnectionsStore
from dlightrag.application.config import DlightragConfig
from dlightrag.application.connections import Connections
from tests.integration.run_runtime_pg_harness import isolated_run_runtime
from tests.integration.test_connections_pg import FakeMcp


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["jwt", "none", "simple"])
async def test_web_owner_lifecycle_and_csrf(mode, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = DlightragConfig(
        _env_file=None,
        models={
            "chat": {
                "roles": {
                    role: {"model": "fixture-model"}
                    for role in ("extract", "query", "keyword", "vlm")
                }
            }
        },
        access={
            "auth_mode": mode,
            "jwt_verification_key": "test-only-web-jwt-key-not-for-production",
            "api_token": "test-only-simple-token" if mode == "simple" else None,
        },
    )
    async with isolated_run_runtime("connections_web") as (_, pool):
        store = PGConnectionsStore(pool=pool)
        await store.initialize(validate_only=False)
        app = FastAPI()
        app.include_router(router, prefix="/web/api")
        app.state.application = SimpleNamespace(connections=Connections(store=store, mcp=FakeMcp()))
        app.add_middleware(WebAuthMiddleware, config_getter=lambda: config)

        def auth(subject):
            if mode == "none":
                return {}
            token = (
                "test-only-simple-token"
                if mode == "simple"
                else jwt.encode(
                    {"iss": "fixture-issuer", "sub": subject},
                    "test-only-web-jwt-key-not-for-production",
                    algorithm="HS256",
                )
            )
            return {"Authorization": f"Bearer {token}"}

        async with AsyncClient(
            transport=ASGITransport(app), base_url="http://test", headers=auth("a")
        ) as client:
            initial = await client.get("/web/api/connections/mcp")
            if mode == "simple":
                assert initial.status_code == 403
                return
            assert initial.status_code == 200
            body = {
                "expected_revision": initial.json()["revision"],
                "label": "Fixture",
                "endpoint": "https://example.com/mcp",
            }
            blocked = await client.post(
                "/web/api/connections/mcp", json=body, headers={"Origin": "https://evil.example"}
            )
            assert blocked.status_code == 403
            headers = {
                "Origin": "http://test",
                "X-CSRF-Token": client.cookies.get("dlightrag_web_csrf", ""),
            }
            created = await client.post("/web/api/connections/mcp", json=body, headers=headers)
            assert created.status_code == 200
            assert not created.json()["connections"][0]["enabled"]
            identity = created.json()["connections"][0]["connection_id"]
            if mode == "jwt":
                async with AsyncClient(
                    transport=ASGITransport(app), base_url="http://test", headers=auth("b")
                ) as other:
                    assert (await other.get("/web/api/connections/mcp")).json()["connections"] == []
                    denied = await other.post(
                        f"/web/api/connections/mcp/{identity}/probe",
                        json={"expected_revision": "0"},
                        headers={"X-CSRF-Token": other.cookies.get("dlightrag_web_csrf", "") or ""},
                    )
                    assert denied.status_code == 404
            invalid = await client.put(
                f"/web/api/connections/mcp/{identity}/bearer",
                headers=headers,
                json={"expected_revision": "0", "bearer": {"secret": "do-not-echo-fixture"}},
            )
            assert invalid.status_code == 422
            assert "do-not-echo-fixture" not in invalid.text


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["jwt", "none"])
async def test_web_sdk_oauth_authenticated_callback_strips_query_and_returns_fixed_settings(
    tmp_path, monkeypatch, mode
):
    import asyncio
    from urllib.parse import parse_qs, urlsplit

    import httpx2

    from dlightrag.adapters.http.browser.routes.connections import callback_router
    from dlightrag.adapters.mcp.oauth import PersonalOAuthClient
    from dlightrag.application.connections import ConnectionPolicy
    from tests.integration.test_connection_authorization_pg import cipher
    from tests.support.dns import public_dns
    from tests.unit.test_connection_oauth import FakeAuthorizationServer

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    config = DlightragConfig(
        _env_file=None,
        models={
            "chat": {
                "roles": {
                    role: {"model": "fixture-model"}
                    for role in ("extract", "query", "keyword", "vlm")
                }
            }
        },
        access={
            "auth_mode": mode,
            "jwt_verification_key": "test-only-web-jwt-key-not-for-production",
        },
    )

    def auth(subject):
        if mode == "none":
            return {}
        return {
            "Authorization": "Bearer "
            + jwt.encode(
                {"iss": "fixture-issuer", "sub": subject},
                "test-only-web-jwt-key-not-for-production",
                algorithm="HS256",
            )
        }

    server = FakeAuthorizationServer()
    async with isolated_run_runtime("oauth_web") as (_, pool):
        store = PGConnectionsStore(pool=pool)
        await store.initialize(validate_only=False)
        service = Connections(
            store=store,
            mcp=FakeMcp(),
            cipher=cipher(),
            policy=ConnectionPolicy(
                oauth_callback_url="https://app.example/web/oauth/connections/mcp/callback"
            ),
            oauth=PersonalOAuthClient(transport_factory=lambda: httpx2.MockTransport(server)),
        )
        app = FastAPI()
        app.include_router(router, prefix="/web/api")
        app.include_router(callback_router, prefix="/web")
        app.state.application = SimpleNamespace(connections=service)
        app.add_middleware(WebAuthMiddleware, config_getter=lambda: config)
        scopes = []

        async def capture(scope, receive, send):
            await app(scope, receive, send)
            scopes.append(scope)

        async with AsyncClient(
            transport=ASGITransport(capture), base_url="https://app.example", headers=auth("a")
        ) as client:
            try:
                view = (await client.get("/web/api/connections/mcp")).json()
                headers = {
                    "Origin": "https://app.example",
                    "X-CSRF-Token": client.cookies.get("dlightrag_web_csrf", ""),
                }
                view = (
                    await client.post(
                        "/web/api/connections/mcp",
                        headers=headers,
                        json={
                            "expected_revision": view["revision"],
                            "label": "A",
                            "endpoint": "https://mcp.example/mcp",
                        },
                    )
                ).json()
                identity = view["connections"][0]["connection_id"]
                path = f"/web/api/connections/mcp/{identity}/oauth"
                body = {"expected_revision": view["revision"]}
                assert (
                    await client.post(path, json=body, headers={"Origin": "https://evil.example"})
                ).status_code == 403
                start = await client.post(path, json=body, headers=headers)
                assert start.status_code == 200
                server.authorization = parse_qs(urlsplit(start.json()["authorization_url"]).query)
                params = {
                    "state": server.authorization["state"][0],
                    "code": "test-code",
                    "iss": "https://as.example",
                }
                callback = "/web/oauth/connections/mcp/callback"
                if mode == "jwt":
                    async with AsyncClient(
                        transport=ASGITransport(capture), base_url="https://app.example"
                    ) as anonymous:
                        missing = await anonymous.get(callback, params=params)
                    assert missing.status_code == 303
                    assert "test-code" not in missing.headers["location"]
                    assert "state=" not in missing.headers["location"]
                    other = await client.get(callback, params=params, headers=auth("b"))
                    assert (
                        other.headers["location"]
                        == "/web/?settings=connections&authorization=restart"
                    )
                accepted = await client.get(callback, params=params)
                assert accepted.status_code == 303
                assert accepted.headers["location"] == "/web/?settings=connections"
                assert accepted.headers["cache-control"] == "no-store"
                assert accepted.headers["referrer-policy"] == "no-referrer"
                assert scopes[-1]["query_string"] == b""
                duplicate = await client.get(callback, params=params)
                assert duplicate.headers["location"].endswith("authorization=restart")
                for _ in range(100):
                    view = (await client.get("/web/api/connections/mcp")).json()
                    if view["connections"][0]["authentication"] == "oauth":
                        break
                    await asyncio.sleep(0.02)
                assert view["connections"][0]["authorization_status"] == "succeeded"
                assert not any(
                    secret in str(view)
                    for secret in ["test-access-token", "test-code", "test-client-secret"]
                )
            finally:
                await service.aclose()


@pytest.mark.asyncio
async def test_published_client_metadata_is_fetchable_without_a_session(tmp_path, monkeypatch):
    """An authorization server has no cookie, so the document must be public, minimal, and exact."""
    from dlightrag.adapters.http.browser.routes.connections import callback_router
    from dlightrag.application.connections import ConnectionPolicy

    monkeypatch.chdir(tmp_path)
    config = DlightragConfig(
        _env_file=None,
        models={
            "chat": {
                "roles": {
                    role: {"model": "fixture-model"}
                    for role in ("extract", "query", "keyword", "vlm")
                }
            }
        },
        access={
            "auth_mode": "jwt",
            "jwt_verification_key": "test-only-web-jwt-key-not-for-production",
        },
    )
    callback = "https://app.example/web/oauth/connections/mcp/callback"
    metadata_url = "https://app.example/web/oauth/connections/mcp/client-metadata"

    def app_for(service):
        app = FastAPI()
        app.include_router(callback_router, prefix="/web")
        app.state.application = SimpleNamespace(connections=service)
        app.add_middleware(WebAuthMiddleware, config_getter=lambda: config)
        return app

    async with isolated_run_runtime("oauth_metadata") as (_, pool):
        store = PGConnectionsStore(pool=pool)
        await store.initialize(validate_only=False)
        published = Connections(
            store=store, mcp=FakeMcp(), policy=ConnectionPolicy(oauth_callback_url=callback)
        )
        async with AsyncClient(
            transport=ASGITransport(app_for(published)), base_url="https://app.example"
        ) as client:
            response = await client.get("/web/oauth/connections/mcp/client-metadata")
            assert response.status_code == 200
            assert response.json() == {
                "client_id": metadata_url,
                "client_name": "DlightRAG personal Connection",
                "redirect_uris": [callback],
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "none",
            }
            assert response.headers["cache-control"] == "public, max-age=300"

        unpublished = Connections(store=store, mcp=FakeMcp(), policy=ConnectionPolicy())
        async with AsyncClient(
            transport=ASGITransport(app_for(unpublished)), base_url="https://app.example"
        ) as client:
            assert (
                await client.get("/web/oauth/connections/mcp/client-metadata")
            ).status_code == 404
