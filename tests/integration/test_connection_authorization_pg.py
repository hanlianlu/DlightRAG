# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Authorization through Connections, actual PostgreSQL and fake remote transports."""

import base64
import json
from contextlib import asynccontextmanager
from dataclasses import replace
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from dlightrag.adapters.postgres.connections import PGConnectionsStore
from dlightrag.application.connections import ConnectionCommand, Connections, ConnectionsError
from dlightrag.application.connections.credentials import CredentialCipher
from tests.integration.run_runtime_pg_harness import isolated_run_runtime
from tests.integration.test_connections_pg import FakeMcp, stored_catalogue
from tests.support.pg import notification_hub


def cipher():
    return CredentialCipher(
        SecretStr(
            json.dumps(
                {"active": "test", "keys": {"test": base64.urlsafe_b64encode(b"t" * 32).decode()}}
            )
        )
    )


_CALLBACK_URL = "https://app.example/web/oauth/connections/mcp/callback"
_OWNER = {"owner_id": "a"}


@asynccontextmanager
async def _authorizing_owner(prefix, server, *, override: str | None = _CALLBACK_URL):
    """Connections for one owner with an enabled bearer Connection and a draft beside it.

    ``a`` initiates authorizations against ``server``; ``b`` is a second worker that
    receives the provider callback, as a load balancer may route it.
    """
    import httpx2

    from dlightrag.adapters.mcp.oauth import PersonalOAuthClient
    from dlightrag.application.connections import ConnectionPolicy

    async with isolated_run_runtime(prefix) as (_, pool):
        store = PGConnectionsStore(pool=pool)
        await store.initialize(validate_only=False)
        policy = ConnectionPolicy(oauth_callback_url=override)
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
        view = await a.change(
            **_OWNER,
            expected_revision="0",
            command=ConnectionCommand(
                kind="create", label="Target", endpoint="https://old.example/mcp"
            ),
        )
        target = view.connections[0].connection_id
        view = await a.replace_bearer(**_OWNER, connection_id=target, bearer=SecretStr("old-token"))
        view = await a.change(
            **_OWNER,
            expected_revision=view.revision,
            command=ConnectionCommand(kind="enable", connection_id=target, consent_version=1),
        )
        view = await a.change(
            **_OWNER,
            expected_revision=view.revision,
            command=ConnectionCommand(
                kind="create", label="Other", endpoint="https://other.example/mcp"
            ),
        )
        other = next(item.connection_id for item in view.connections if item.label == "Other")
        try:
            yield SimpleNamespace(
                pool=pool, store=store, policy=policy, a=a, b=b, target=target, other=other
            )
        finally:
            await a.aclose()
            await b.aclose()


async def _begin(fixture, server, *, reached=_CALLBACK_URL):
    """Begin authorizing the target at the owner's current revision; return the SDK state."""
    from urllib.parse import parse_qs, urlsplit

    current = await fixture.a.read(**_OWNER)
    start = await fixture.a.begin_authorization(
        **_OWNER,
        connection_id=fixture.target,
        expected_revision=current.revision,
        callback_url=reached,
        endpoint="https://mcp.example/mcp",
    )
    server.authorization = parse_qs(urlsplit(start.authorization_url).query)
    return server.authorization["state"][0]


async def _finished_flow(pool):
    """The latest authorization flow row once its initiator has finished with it."""
    import asyncio

    async with asyncio.timeout(10):
        while True:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT * FROM dlightrag_connection_oauth_flows ORDER BY expires_at DESC LIMIT 1"
                )
            if row is not None and row["finished_at"] is not None:
                return row
            await asyncio.sleep(0.02)


def _item(view, connection_id):
    return next(item for item in view.connections if item.connection_id == connection_id)


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
        owner = dict(owner_id="a")
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
    from tests.support.dns import public_dns
    from tests.unit.test_connection_oauth import FakeAuthorizationServer

    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    server = FakeAuthorizationServer()
    async with isolated_run_runtime("oauth_inbox") as (_, pool), notification_hub(pool) as hub:
        store = PGConnectionsStore(pool=pool, notifications=hub)
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
        owner = dict(owner_id="a")
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
                callback_url=_CALLBACK_URL,
                endpoint="https://mcp.example/mcp",
            )
            server.authorization = parse_qs(urlsplit(start.authorization_url).query)
            state = server.authorization["state"][0]
            with pytest.raises(ConnectionsError):
                await b.authorization_callback(owner_id="b", state=state, code="test-code")
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
            assert (await stored_catalogue(store))[0].remote_name == "read"
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
                    **owner,
                    connection_id=identity,
                    expected_revision=view.revision,
                    callback_url=_CALLBACK_URL,
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
            await store.stop_notifications()
            await a.aclose()
            await b.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    ["expired", "dead-worker", "denied", "issuer", "restart", "discovery"],
)
async def test_authorization_failure_never_retires_enabled_head_and_requires_restart(
    failure, monkeypatch
):
    import asyncio
    from urllib.parse import parse_qs, urlsplit

    import httpx2

    from dlightrag.adapters.mcp.oauth import PersonalOAuthClient
    from dlightrag.application.connections import ConnectionPolicy
    from tests.support.dns import public_dns
    from tests.unit.test_connection_oauth import FakeAuthorizationServer

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
        owner = dict(owner_id="a")
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
                callback_url=_CALLBACK_URL,
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
                    callback_url=_CALLBACK_URL,
                    endpoint="https://mcp.example/mcp",
                )
                with pytest.raises(ConnectionsError):
                    await b.authorization_callback(**owner, state=state, code="test-code")
            else:
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
            # A failed flow hands its head's refresh claim back; a restarted one keeps it.
            if failure == "restart":
                assert await store.claim(worker_id="loop", lease_seconds=30) is None
            else:
                await _finished_flow(pool)
                claim = await store.claim(worker_id="loop", lease_seconds=30)
                assert claim is not None and claim.connection.connection_id == identity
        finally:
            await a.aclose()
            await b.aclose()


@pytest.mark.asyncio
async def test_a_local_deployment_authorizes_through_its_loopback_callback(monkeypatch):
    """The callback follows the address the browser reached, with no configuration.

    It is where the provider sends that browser back, never a host DlightRAG calls, so a
    Compose deployment reached at localhost authorizes over plain HTTP there, although the
    outbound network policy refuses loopback and HTTP endpoints.
    """
    from tests.support.dns import public_dns
    from tests.unit.test_connection_oauth import FakeAuthorizationServer

    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    server = FakeAuthorizationServer()
    callback = "http://localhost:8100/web/oauth/connections/mcp/callback"
    async with _authorizing_owner("oauth_loopback", server, override=None) as fixture:
        await _begin(fixture, server, reached=callback)
        assert server.authorization["redirect_uri"] == [callback]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", ["other-refresh", "other-command", "own-refresh", "refresh-in-flight"]
)
async def test_authorization_completes_through_changes_made_elsewhere(change, monkeypatch):
    """Only a change to the authorized Connection itself can void its authorization.

    A background refresh or a Settings command on another of the owner's Connections leaves
    the authorized head where the flow began, and so does the head's own background refresh:
    none runs while the flow is live, and one already running when it began cannot publish.
    """
    from tests.support.dns import public_dns
    from tests.unit.test_connection_oauth import FakeAuthorizationServer

    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    server = FakeAuthorizationServer()
    async with _authorizing_owner("oauth_elsewhere", server) as fixture:
        store, policy = fixture.store, fixture.policy
        in_flight = None
        if change == "refresh-in-flight":
            in_flight = await store.claim(worker_id="loop", lease_seconds=30)
            assert in_flight is not None
            assert in_flight.connection.connection_id == fixture.target
        state = await _begin(fixture, server)
        if change == "other-refresh":
            claim = await store.claim(
                worker_id="loop", lease_seconds=30, owner_id="a", connection_id=fixture.other
            )
            assert claim is not None
            assert await store.publish(
                claim=claim, catalogue=None, error="discovery", retry_seconds=300, policy=policy
            )
        elif change == "other-command":
            await fixture.a.change(
                **_OWNER,
                expected_revision=(await fixture.a.read(**_OWNER)).revision,
                command=ConnectionCommand(
                    kind="edit", connection_id=fixture.other, label="Renamed"
                ),
            )
        elif change == "own-refresh":
            # The target is enabled and due: a refresh loop would publish whatever it claims.
            claim = await store.claim(worker_id="loop", lease_seconds=30)
            if claim is not None:
                await store.publish(
                    claim=claim,
                    catalogue=claim.connection.catalogue,
                    error=None,
                    retry_seconds=300,
                    policy=policy,
                )
        else:
            assert in_flight is not None
            assert not await store.publish(
                claim=in_flight,
                catalogue=in_flight.connection.catalogue,
                error=None,
                retry_seconds=300,
                policy=policy,
            )
        await fixture.b.authorization_callback(
            **_OWNER, state=state, code="test-code", issuer="https://as.example"
        )
        await _finished_flow(fixture.pool)
        target = _item(await fixture.a.read(**_OWNER), fixture.target)
        assert target.authorization_status == "succeeded"
        assert target.authentication == "oauth"
        assert target.endpoint == "https://mcp.example/mcp"
        assert target.status == "ready"


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["edit", "disable", "bearer", "delete"])
async def test_authorization_publishes_nothing_once_its_connection_changed(change, monkeypatch):
    """A change to the authorized Connection wins, and Settings says the flow must restart.

    The flow completes its token exchange and discovery against the head revision it began
    at; the owner's later edit, disable, credential, or delete moved that head, so the flow
    publishes no Grant or generation and ends as ``changed`` instead of a generic failure.
    """
    from tests.support.dns import public_dns
    from tests.unit.test_connection_oauth import FakeAuthorizationServer

    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    server = FakeAuthorizationServer()
    async with _authorizing_owner("oauth_changed", server) as fixture:
        a = fixture.a
        state = await _begin(fixture, server)
        revision = (await a.read(**_OWNER)).revision
        if change == "bearer":
            changed = await a.replace_bearer(
                **_OWNER,
                connection_id=fixture.target,
                expected_revision=revision,
                bearer=SecretStr("new-token"),
                endpoint="https://old.example/mcp",
            )
        else:
            changed = await a.change(
                **_OWNER,
                expected_revision=revision,
                command=ConnectionCommand(
                    kind=change,
                    connection_id=fixture.target,
                    label="Changed" if change == "edit" else None,
                ),
            )
        await fixture.b.authorization_callback(
            **_OWNER, state=state, code="test-code", issuer="https://as.example"
        )
        flow = await _finished_flow(fixture.pool)
        assert flow["outcome"] == "changed"
        assert any(r.url.path == "/token" for r in server.requests)
        async with fixture.pool.acquire() as conn:
            assert (
                await conn.fetchval(
                    "SELECT count(*) FROM dlightrag_connection_grants WHERE kind='oauth'"
                )
                == 0
            )
        after = await a.read(**_OWNER)
        if change == "delete":
            assert fixture.target not in {item.connection_id for item in after.connections}
        else:
            expected = replace(_item(changed, fixture.target), authorization_status="changed")
            assert _item(after, fixture.target) == expected
