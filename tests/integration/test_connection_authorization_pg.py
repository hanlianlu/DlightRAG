# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Authorization through Connections, actual PostgreSQL and fake remote transports."""

import base64
import json

import pytest
from pydantic import SecretStr

from dlightrag.adapters.postgres.connections import PGConnectionsStore
from dlightrag.application.connections import ConnectionCommand, Connections, ConnectionsError
from dlightrag.application.connections.credentials import CredentialCipher
from tests.integration.run_runtime_pg_harness import isolated_run_runtime
from tests.integration.test_connections_pg import FakeMcp


def cipher():
    return CredentialCipher(
        SecretStr(
            json.dumps(
                {"active": "test", "keys": {"test": base64.urlsafe_b64encode(b"t" * 32).decode()}}
            )
        )
    )


@pytest.mark.asyncio
async def test_authenticated_endpoint_candidate_keeps_old_live_until_discovery_cas():
    class Mcp(FakeMcp):
        fail = False
        seen = []

        async def discover(self, *, endpoint, bearer, policy):
            self.seen.append((endpoint, bearer.get_secret_value() if bearer else None))
            if self.fail:
                raise ConnectionsError("fixture failure")
            return await super().discover(endpoint=endpoint, bearer=bearer, policy=policy)

    async with isolated_run_runtime("authorization_candidate") as (_, pool):
        store = PGConnectionsStore(pool=pool)
        await store.initialize(validate_only=False)
        mcp = Mcp()
        service = Connections(store=store, mcp=mcp, cipher=cipher())
        owner = dict(owner_id="a", auth_mode="jwt")
        view = await service.change(
            **owner,
            expected_revision="0",
            command=ConnectionCommand(kind="create", label="A", endpoint="https://old.example/mcp"),
        )
        identity = view.connections[0].connection_id
        view = await service.replace_bearer(
            **owner, connection_id=identity, bearer=SecretStr("old-token")
        )
        view = await service.change(
            **owner,
            expected_revision=view.revision,
            command=ConnectionCommand(kind="enable", connection_id=identity, consent_version=1),
        )
        mcp.fail = True
        with pytest.raises(ConnectionsError):
            await service.replace_bearer(
                **owner,
                connection_id=identity,
                expected_revision=view.revision,
                bearer=SecretStr("new-token"),
                endpoint="https://new.example/mcp",
            )
        assert await service.read(**owner) == view
        mcp.fail = False
        updated = await service.replace_bearer(
            **owner,
            connection_id=identity,
            expected_revision=view.revision,
            bearer=SecretStr("new-token"),
            endpoint="https://new.example/mcp",
        )
        assert updated.connections[0].enabled
        assert updated.connections[0].endpoint == "https://new.example/mcp"
        assert updated.connections[0].activation_epoch == view.connections[0].activation_epoch
        assert ("https://new.example/mcp", "old-token") not in mcp.seen
        assert mcp.seen[-1] == ("https://new.example/mcp", "new-token")


@pytest.mark.asyncio
async def test_sdk_flow_callback_other_worker_is_encrypted_owner_bound_and_once(monkeypatch):
    import asyncio
    from urllib.parse import parse_qs, urlsplit

    import httpx2

    from dlightrag.adapters.mcp.oauth import PersonalOAuthClient
    from dlightrag.application.connections import ConnectionPolicy
    from tests.unit.test_connection_oauth import FakeAuthorizationServer
    from tests.unit.test_connections_transport import public_dns

    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    server = FakeAuthorizationServer()
    async with isolated_run_runtime("oauth_inbox") as (_, pool):
        store = PGConnectionsStore(pool=pool)
        await store.initialize(validate_only=False)
        await PGConnectionsStore(pool=pool).initialize(validate_only=True)
        await store.start_notifications()
        await store.wait_refresh(2)
        policy = ConnectionPolicy(
            oauth_callback_url="https://app.example/web/oauth/connections/mcp/callback"
        )

        async def admitted_fake(request):
            async with pool.acquire() as conn:
                # Other callback transactions can briefly overlap. A transaction
                # held by THIS network operation cannot finish until we return.
                count = 1
                for _ in range(50):
                    count = await conn.fetchval(
                        "SELECT count(*) FROM pg_stat_activity WHERE datname=current_database() AND pid<>pg_backend_pid() AND state='idle in transaction'"
                    )
                    if count == 0:
                        break
                    await asyncio.sleep(0.01)
                assert count == 0
            return server(request)

        a = Connections(
            store=store,
            mcp=FakeMcp(),
            cipher=cipher(),
            policy=policy,
            oauth=PersonalOAuthClient(
                transport_factory=lambda: httpx2.MockTransport(admitted_fake)
            ),
        )
        b = Connections(
            store=PGConnectionsStore(pool=pool), mcp=FakeMcp(), cipher=cipher(), policy=policy
        )
        owner = dict(owner_id="a", auth_mode="jwt")
        draft = await a.change(
            **owner,
            expected_revision="0",
            command=ConnectionCommand(kind="create", label="A", endpoint="https://old.example/mcp"),
        )
        identity = draft.connections[0].connection_id
        draft = await a.replace_bearer(
            **owner, connection_id=identity, bearer=SecretStr("old-token")
        )
        draft = await a.change(
            **owner,
            expected_revision=draft.revision,
            command=ConnectionCommand(kind="enable", connection_id=identity, consent_version=1),
        )
        try:
            start = await a.begin_authorization(
                **owner,
                connection_id=identity,
                expected_revision=draft.revision,
                endpoint="https://mcp.example/mcp",
            )
            server.authorization = parse_qs(urlsplit(start.authorization_url).query)
            state = server.authorization["state"][0]
            with pytest.raises(ConnectionsError):
                await b.authorization_callback(
                    owner_id="b", auth_mode="jwt", state=state, code="test-code"
                )
            with pytest.raises(ConnectionsError):
                await b.authorization_callback(**owner, state="wrong-state", code="test-code")
            deposited = await asyncio.gather(
                b.authorization_callback(
                    **owner, state=state, code="test-code", issuer="https://as.example"
                ),
                a.authorization_callback(
                    **owner, state=state, code="test-code", issuer="https://as.example"
                ),
                return_exceptions=True,
            )
            assert sum(result is None for result in deposited) == 1
            assert sum(isinstance(result, ConnectionsError) for result in deposited) == 1
            with pytest.raises(ConnectionsError):
                await b.authorization_callback(**owner, state=state, code="test-code")
            view = draft
            for _ in range(100):
                view = await a.read(**owner)
                if view.connections[0].authentication == "oauth":
                    break
                await asyncio.sleep(0.02)
            assert view.connections[0].authentication == "oauth"
            assert view.connections[0].tools[0].remote_name == "read"
            async with pool.acquire() as conn:
                flows = await conn.fetch("SELECT * FROM dlightrag_connection_oauth_flows")
                grants = await conn.fetch("SELECT * FROM dlightrag_connection_grants")
            assert "test-code" not in str(flows)
            assert state not in str(flows)
            assert "test-access-token" not in str(grants)
            assert "test-client-secret" not in str(grants)
            assert flows[0]["encrypted_result"] is None
            assert flows[0]["consumed_at"] is not None
            assert view.connections[0].enabled
            assert view.connections[0].activation_epoch == draft.connections[0].activation_epoch
            assert view.connections[0].endpoint == "https://mcp.example/mcp"
            assert not any("old-token" in str(request.headers) for request in server.requests)
            for scope in ("read", "read write"):
                old_generation = view.connections[0].generation
                server.scope = server.granted_scope = scope
                start = await a.begin_authorization(
                    **owner, connection_id=identity, expected_revision=view.revision
                )
                server.authorization = parse_qs(urlsplit(start.authorization_url).query)
                await b.authorization_callback(
                    **owner,
                    state=server.authorization["state"][0],
                    code="test-code",
                    issuer="https://as.example",
                )
                for _ in range(100):
                    view = await a.read(**owner)
                    if view.connections[0].generation > old_generation:
                        break
                    await asyncio.sleep(0.02)
                assert view.connections[0].generation > old_generation
            async with pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT grant_id,status,encrypted_envelope,consented_scopes FROM dlightrag_connection_grants"
                )
            assert len({row["grant_id"] for row in rows}) == 4
            assert sum(row["status"] == "active" for row in rows) == 1
            assert all(
                row["encrypted_envelope"] is None for row in rows if row["status"] == "retired"
            )
            assert json.loads(
                next(row["consented_scopes"] for row in rows if row["status"] == "active")
            ) == ["read", "write"]
        finally:
            await a.aclose()
            await b.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    ["expired", "dead-worker", "denied", "issuer", "restart", "stale-candidate", "discovery"],
)
async def test_authorization_failure_never_retires_enabled_head_and_requires_restart(
    failure, monkeypatch
):
    import asyncio
    from urllib.parse import parse_qs, urlsplit

    import httpx2

    from dlightrag.adapters.mcp.oauth import PersonalOAuthClient
    from dlightrag.application.connections import ConnectionPolicy
    from tests.unit.test_connection_oauth import FakeAuthorizationServer
    from tests.unit.test_connections_transport import public_dns

    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    server = FakeAuthorizationServer()
    async with isolated_run_runtime("oauth_failure") as (_, pool):
        store = PGConnectionsStore(pool=pool)
        await store.initialize(validate_only=False)
        policy = ConnectionPolicy(
            oauth_callback_url="https://app.example/web/oauth/connections/mcp/callback"
        )
        a = Connections(
            store=store,
            mcp=FakeMcp(),
            cipher=cipher(),
            policy=policy,
            oauth=PersonalOAuthClient(transport_factory=lambda: httpx2.MockTransport(server)),
        )
        b = Connections(
            store=PGConnectionsStore(pool=pool), mcp=FakeMcp(), cipher=cipher(), policy=policy
        )
        owner = dict(owner_id="a", auth_mode="jwt")
        view = await a.change(
            **owner,
            expected_revision="0",
            command=ConnectionCommand(kind="create", label="A", endpoint="https://old.example/mcp"),
        )
        identity = view.connections[0].connection_id
        view = await a.replace_bearer(
            **owner, connection_id=identity, bearer=SecretStr("old-token")
        )
        view = await a.change(
            **owner,
            expected_revision=view.revision,
            command=ConnectionCommand(kind="enable", connection_id=identity, consent_version=1),
        )
        try:
            start = await a.begin_authorization(
                **owner,
                connection_id=identity,
                expected_revision=view.revision,
                endpoint="https://mcp.example/mcp",
            )
            server.authorization = parse_qs(urlsplit(start.authorization_url).query)
            state = server.authorization["state"][0]
            if failure in {"expired", "dead-worker"}:
                field = "expires_at" if failure == "expired" else "flow_lease_expires_at"
                async with pool.acquire() as conn:
                    await conn.execute(
                        f"UPDATE dlightrag_connection_oauth_flows SET {field}=clock_timestamp()-interval '1 second'"
                    )
                with pytest.raises(ConnectionsError):
                    await b.authorization_callback(**owner, state=state, code="test-code")
            elif failure == "restart":
                await a.begin_authorization(
                    **owner,
                    connection_id=identity,
                    expected_revision=view.revision,
                    endpoint="https://mcp.example/mcp",
                )
                with pytest.raises(ConnectionsError):
                    await b.authorization_callback(**owner, state=state, code="test-code")
            else:
                if failure == "stale-candidate":
                    view = await a.change(
                        **owner,
                        expected_revision=view.revision,
                        command=ConnectionCommand(
                            kind="edit", connection_id=identity, label="Changed"
                        ),
                    )
                server.fail_discovery = failure == "discovery"
                await b.authorization_callback(
                    **owner,
                    state=state,
                    code=None if failure == "denied" else "test-code",
                    error="access_denied" if failure == "denied" else None,
                    issuer="https://wrong.example" if failure == "issuer" else "https://as.example",
                )
                observed = await a.read(**owner)
                for _ in range(100):
                    observed = await a.read(**owner)
                    if observed.connections[0].authorization_status == "failed":
                        break
                    await asyncio.sleep(0.02)
                assert observed.connections[0].authorization_status == "failed"
            observed = await a.read(**owner)
            assert observed.connections[0].enabled
            assert observed.connections[0].endpoint == "https://old.example/mcp"
            assert observed.connections[0].generation == view.connections[0].generation
            async with pool.acquire() as conn:
                assert (
                    await conn.fetchval(
                        "SELECT count(*) FROM dlightrag_connection_grants WHERE status='active' AND encrypted_envelope IS NOT NULL"
                    )
                    == 1
                )
            if failure in {"expired", "dead-worker", "denied", "issuer", "restart"}:
                assert not any(r.url.path == "/token" for r in server.requests)
            assert not any("old-token" in str(r.headers) for r in server.requests)
        finally:
            await a.aclose()
            await b.aclose()
