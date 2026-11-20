# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Personal Connections behavior through the owner module and real PostgreSQL."""

import uuid

import pytest

from dlightrag.adapters.postgres.connections import PGConnectionsStore
from dlightrag.application.connections.models import (
    ConnectionCommand,
    ConnectionsError,
    McpDiscovery,
)
from dlightrag.application.connections.service import Connections
from dlightrag.engine.credential_cipher import CredentialCipher
from tests.integration.run_runtime_pg_harness import drop_owned_database, isolated_run_runtime
from tests.support.pg import PG_CONN_KWARGS, drop_scratch_database, notification_hub


class FakeMcp:
    async def call(self, **kwargs):
        raise AssertionError("Management must not dispatch")

    async def discover(self, *, endpoint, access, policy, **kwargs):
        return McpDiscovery(
            tools=(
                {
                    "name": "read",
                    "description": "Read a fixture",
                    "input_schema": {"type": "object"},
                },
            ),
            instructions=None,
        )


@pytest.mark.asyncio
async def test_owned_database_drops_with_a_backend_of_this_role_attached():
    """A connection still attached at teardown must not turn the drop into a permission error.

    `DROP DATABASE ... WITH (FORCE)` needs `pg_signal_backend` for another role's backend, which
    the development role lacks, so a listener or worker still detaching failed teardown with an
    unrelated `InsufficientPrivilegeError`.
    """
    import asyncpg as _asyncpg

    database = f"dlightrag_drop_probe_{uuid.uuid4().hex[:8]}"
    admin = await _asyncpg.connect(**PG_CONN_KWARGS)
    await admin.execute(f'CREATE DATABASE "{database}"')
    attached = await _asyncpg.connect(**{**PG_CONN_KWARGS, "database": database})
    try:
        await drop_owned_database(admin, database)
        remaining = await admin.fetchval(
            "select count(*) from pg_database where datname = $1", database
        )
        assert remaining == 0, "the database must be gone even though a backend was attached"
    finally:
        await attached.close()
        await drop_scratch_database(admin, database)
        await admin.close()


async def stored_catalogue(store, owner_id: str = "a", index: int = 0):
    """The published catalogue is agent-facing; the Settings projection no longer carries it."""
    _revision, items = await store.read(owner_id)
    return items[index].catalogue


@pytest.mark.asyncio
async def test_owner_draft_probe_consent_and_revision():
    async with isolated_run_runtime("connections") as (_, pool):
        store = PGConnectionsStore(pool=pool)
        await store.initialize(validate_only=False)
        connections = Connections(store=store, mcp=FakeMcp(), cipher=CredentialCipher(None))
        empty = await connections.read(owner_id="a")
        draft = await connections.change(
            owner_id="a",
            expected_revision=empty.revision,
            command=ConnectionCommand(
                kind="create", label="Fixture", endpoint="https://example.com/mcp"
            ),
        )
        item = draft.connections[0]
        assert item.enabled is False
        assert (await connections.read(owner_id="b")).connections == ()
        with pytest.raises(ConnectionsError, match="not found"):
            await connections.change(
                owner_id="b",
                expected_revision="0",
                command=ConnectionCommand(kind="probe", connection_id=item.connection_id),
            )
        ready = await connections.change(
            owner_id="a",
            expected_revision=draft.revision,
            command=ConnectionCommand(kind="probe", connection_id=item.connection_id),
        )
        assert (await stored_catalogue(store))[0].remote_name == "read"
        with pytest.raises(ConnectionsError, match="consent"):
            await connections.change(
                owner_id="a",
                expected_revision=ready.revision,
                command=ConnectionCommand(kind="enable", connection_id=item.connection_id),
            )
        enabled = await connections.change(
            owner_id="a",
            expected_revision=ready.revision,
            command=ConnectionCommand(
                kind="enable", connection_id=item.connection_id, consent_version=1
            ),
        )
        assert enabled.connections[0].enabled
        with pytest.raises(ConnectionsError, match="revision"):
            await connections.change(
                owner_id="a",
                expected_revision=ready.revision,
                command=ConnectionCommand(kind="disable", connection_id=item.connection_id),
            )
        disabled = await connections.change(
            owner_id="a",
            expected_revision=enabled.revision,
            command=ConnectionCommand(kind="disable", connection_id=item.connection_id),
        )
        assert not disabled.connections[0].enabled
        assert disabled.connections[0].activation_epoch > enabled.connections[0].activation_epoch


@pytest.mark.asyncio
async def test_bearer_encrypted_owner_bound_and_rotatable_without_echo():
    import json
    from dataclasses import asdict

    from pydantic import SecretStr

    from dlightrag.engine.credential_cipher import CredentialCipher
    from tests.unit.test_connections_config import KEYRING

    class AuthMcp(FakeMcp):
        observed = None

        async def discover(self, *, endpoint, access, policy, **kwargs):
            self.observed = access.token if access is not None else None
            return await super().discover(endpoint=endpoint, access=access, policy=policy, **kwargs)

    async with isolated_run_runtime("connection_secrets") as (_, pool):
        store = PGConnectionsStore(pool=pool)
        await store.initialize(validate_only=False)
        mcp = AuthMcp()
        service = Connections(store=store, mcp=mcp, cipher=CredentialCipher(SecretStr(KEYRING)))
        view = await service.change(
            owner_id="a",
            expected_revision="0",
            command=ConnectionCommand(
                kind="create", label="External", endpoint="https://example.com/mcp"
            ),
        )
        identity = view.connections[0].connection_id
        secret = SecretStr("test-only-personal-token")
        view = await service.replace_bearer(
            owner_id="a",
            connection_id=identity,
            bearer=secret,
            expected_revision=view.revision,
        )
        assert mcp.observed == secret
        assert view.connections[0].authentication == "bearer"
        with pytest.raises(ConnectionsError) as edit_error:
            await service.change(
                owner_id="a",
                expected_revision=view.revision,
                command=ConnectionCommand(
                    kind="edit",
                    connection_id=identity,
                    endpoint="https://another-audience.example/mcp",
                ),
            )
        assert edit_error.value.kind == "requires_reauthorization"
        assert await service.read(owner_id="a") == view
        assert "test-only-personal-token" not in json.dumps(asdict(view))
        assert "grant_id" not in json.dumps(asdict(view))
        # Persistence is an accepted seam: prove the live stored record is ciphertext.
        _, stored = await store.read("a")
        assert "test-only-personal-token" not in repr(stored)
        assert "test-only-personal-token" not in (stored[0].envelope or "")
        with pytest.raises(ConnectionsError, match="not found"):
            await service.replace_bearer(
                owner_id="b",
                connection_id=identity,
                bearer=secret,
                expected_revision="0",
            )
        no_key = Connections(store=store, mcp=mcp, cipher=CredentialCipher(None))
        with pytest.raises(ConnectionsError, match="deployment"):
            await no_key.replace_bearer(
                owner_id="a",
                connection_id=identity,
                bearer=secret,
                expected_revision=view.revision,
            )


@pytest.mark.asyncio
async def test_refresh_auto_admits_new_tools_and_preserves_last_good_on_fault():
    import asyncio

    from dlightrag.application.connections.policy import ConnectionPolicy

    class ChangingMcp(FakeMcp):
        changed = False

        async def discover(self, **kwargs):
            discovery = await super().discover(**kwargs)
            tools = list(discovery.tools)
            if self.changed:
                tools[0]["input_schema"] = {
                    "type": "object",
                    "properties": {"new_argument": {"type": "string"}},
                }
                tools.append(
                    {
                        "name": "write",
                        "description": "New external write",
                        "input_schema": {"type": "object"},
                    }
                )
            return McpDiscovery(tools=tuple(tools), instructions=discovery.instructions)

    async with (
        isolated_run_runtime("connection_refresh") as (_, pool),
        notification_hub(pool) as hub,
    ):
        mcp = ChangingMcp()
        store = PGConnectionsStore(pool=pool, notifications=hub)
        service = Connections(
            store=store,
            mcp=mcp,
            policy=ConnectionPolicy(refresh_seconds=1),
            cipher=CredentialCipher(None),
        )
        await service.start()
        try:
            draft = await service.change(
                owner_id="a",
                expected_revision="0",
                command=ConnectionCommand(
                    kind="create", label="Fixture", endpoint="https://example.com/mcp"
                ),
            )
            identity = draft.connections[0].connection_id
            ready = await service.change(
                owner_id="a",
                expected_revision=draft.revision,
                command=ConnectionCommand(kind="probe", connection_id=identity),
            )
            enabled = await service.change(
                owner_id="a",
                expected_revision=ready.revision,
                command=ConnectionCommand(kind="enable", connection_id=identity, consent_version=1),
            )
            enabled_catalogue = await stored_catalogue(store)
            mcp.changed = True
            async with asyncio.timeout(5):
                while True:
                    catalogue = await stored_catalogue(store)
                    if len(catalogue) == 2:
                        break
                    await asyncio.sleep(0.05)
            current = await service.read(owner_id="a")
            assert (
                current.connections[0].activation_epoch == enabled.connections[0].activation_epoch
            )
            assert current.connections[0].generation > enabled.connections[0].generation
            assert catalogue[0].input_schema == {
                "type": "object",
                "properties": {"new_argument": {"type": "string"}},
            }
            assert catalogue[0].local_name == enabled_catalogue[0].local_name
        finally:
            await service.aclose()


@pytest.mark.asyncio
async def test_expired_refresh_and_disabled_worker_cannot_publish():
    from dlightrag.application.connections.policy import ConnectionPolicy

    async with isolated_run_runtime("connection_claims") as (_, pool):
        store = PGConnectionsStore(pool=pool)
        await store.initialize(validate_only=False)
        await store.initialize(validate_only=True)
        service = Connections(store=store, mcp=FakeMcp(), cipher=CredentialCipher(None))
        draft = await service.change(
            owner_id="a",
            expected_revision="0",
            command=ConnectionCommand(
                kind="create", label="Fixture", endpoint="https://example.com/mcp"
            ),
        )
        identity = draft.connections[0].connection_id
        claim = await store.claim(
            worker_id="worker-a", lease_seconds=30, owner_id="a", connection_id=identity
        )
        assert claim is not None
        assert (
            await store.claim(
                worker_id="worker-b", lease_seconds=30, owner_id="a", connection_id=identity
            )
            is None
        )
        disabled = await service.change(
            owner_id="a",
            expected_revision=draft.revision,
            command=ConnectionCommand(kind="disable", connection_id=identity),
        )
        assert not await store.publish(
            claim=claim, catalogue=(), error=None, retry_seconds=1, policy=ConnectionPolicy()
        )
        assert (await service.read(owner_id="a")) == disabled


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad",
    [
        [{"name": "duplicate", "input_schema": {"type": "object"}}] * 2,
        [{"name": "invalid name", "input_schema": {"type": "object"}}],
        [{"name": "invalid", "input_schema": {"type": "nonsense"}}],
        [{"name": "large", "description": "x" * 8193, "input_schema": {"type": "object"}}],
    ],
)
async def test_bad_catalogue_preserves_entire_last_good_generation(bad):
    class FaultMcp(FakeMcp):
        fault = False

        async def discover(self, **kwargs):
            if self.fault:
                return McpDiscovery(tools=tuple(bad), instructions=None)
            return await super().discover(**kwargs)

    async with isolated_run_runtime("connection_catalogue") as (_, pool):
        store = PGConnectionsStore(pool=pool)
        await store.initialize(validate_only=False)
        mcp = FaultMcp()
        service = Connections(store=store, mcp=mcp, cipher=CredentialCipher(None))
        draft = await service.change(
            owner_id="a",
            expected_revision="0",
            command=ConnectionCommand(
                kind="create", label="Fixture", endpoint="https://example.com/mcp"
            ),
        )
        identity = draft.connections[0].connection_id
        ready = await service.change(
            owner_id="a",
            expected_revision=draft.revision,
            command=ConnectionCommand(kind="probe", connection_id=identity),
        )
        ready_catalogue = await stored_catalogue(store)
        mcp.fault = True
        fault = await service.change(
            owner_id="a",
            expected_revision=ready.revision,
            command=ConnectionCommand(kind="probe", connection_id=identity),
        )
        assert fault.connections[0].status == "degraded"
        assert fault.connections[0].generation == ready.connections[0].generation
        assert await stored_catalogue(store) == ready_catalogue


@pytest.mark.asyncio
async def test_a_grant_whose_key_is_lost_needs_authorization_and_a_new_bearer_restores_it():
    import json

    from pydantic import SecretStr

    from tests.unit.test_connections_config import KEYRING

    async with isolated_run_runtime("connection_rotation") as (_, pool):
        store = PGConnectionsStore(pool=pool)
        await store.initialize(validate_only=False)
        service = Connections(
            store=store, mcp=FakeMcp(), cipher=CredentialCipher(SecretStr(KEYRING))
        )
        view = await service.change(
            owner_id="a",
            expected_revision="0",
            command=ConnectionCommand(
                kind="create", label="Fixture", endpoint="https://example.com/mcp"
            ),
        )
        identity = view.connections[0].connection_id
        view = await service.replace_bearer(
            owner_id="a", connection_id=identity, bearer=SecretStr("fixture")
        )
        # The ring was lost: the next writer starts with a fresh key of its own.
        fresh_ring = {
            "active": "fresh",
            "keys": {"fresh": "YmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmJiYmI="},
        }
        restarted = Connections(
            store=store, mcp=FakeMcp(), cipher=CredentialCipher(SecretStr(json.dumps(fresh_ring)))
        )

        probed = await restarted.change(
            owner_id="a",
            expected_revision=view.revision,
            command=ConnectionCommand(kind="probe", connection_id=identity),
        )
        restored = await restarted.replace_bearer(
            owner_id="a", connection_id=identity, bearer=SecretStr("replacement")
        )

        assert probed.connections[0].status == "needs-auth"
        assert restored.connections[0].status == view.connections[0].status
        # Maintenance passes over the grant no key opens instead of failing on it.
        assert (await restarted.maintain())["reencrypted"] == 0


@pytest.mark.asyncio
async def test_expired_worker_loses_publication_to_new_claim():
    import asyncio

    from dlightrag.application.connections.policy import ConnectionPolicy

    async with isolated_run_runtime("connection_expiry") as (_, pool):
        store = PGConnectionsStore(pool=pool)
        await store.initialize(validate_only=False)
        service = Connections(store=store, mcp=FakeMcp(), cipher=CredentialCipher(None))
        view = await service.change(
            owner_id="a",
            expected_revision="0",
            command=ConnectionCommand(
                kind="create", label="Fixture", endpoint="https://example.com/mcp"
            ),
        )
        identity = view.connections[0].connection_id
        old = await store.claim(
            worker_id="a", owner_id="a", connection_id=identity, lease_seconds=0.05
        )
        assert old is not None
        await asyncio.sleep(0.08)
        new = await store.claim(
            worker_id="b", owner_id="a", connection_id=identity, lease_seconds=30
        )
        assert new is not None and new.epoch > old.epoch
        assert await store.publish(
            claim=new, catalogue=(), error=None, retry_seconds=1, policy=ConnectionPolicy()
        )
        assert not await store.publish(
            claim=old, catalogue=(), error=None, retry_seconds=1, policy=ConnectionPolicy()
        )
        assert (await service.read(owner_id="a")).connections[0].generation == 1


@pytest.mark.asyncio
async def test_next_refresh_counts_only_heads_a_background_claim_could_take():
    """The idle refresh loops sleep until this report, so it must never name a head they skip.

    A disabled head is never refreshed in the background, and a head under a live claim, a
    refresh's or a pending authorization's, comes back through its holder's publication.
    """
    from dlightrag.application.connections.models import OAuthFlow
    from dlightrag.application.connections.policy import ConnectionPolicy

    async with isolated_run_runtime("connection_schedule") as (_, pool):
        store = PGConnectionsStore(pool=pool)
        await store.initialize(validate_only=False)
        policy = ConnectionPolicy(refresh_seconds=600)
        service = Connections(
            store=store, mcp=FakeMcp(), policy=policy, cipher=CredentialCipher(None)
        )
        owner = dict(owner_id="a")
        assert await store.seconds_until_refresh() is None
        view = await service.change(
            **owner,
            expected_revision="0",
            command=ConnectionCommand(
                kind="create", label="Fixture", endpoint="https://example.com/mcp"
            ),
        )
        identity = view.connections[0].connection_id
        view = await service.change(
            **owner,
            expected_revision=view.revision,
            command=ConnectionCommand(kind="probe", connection_id=identity),
        )
        assert await store.seconds_until_refresh() is None
        view = await service.change(
            **owner,
            expected_revision=view.revision,
            command=ConnectionCommand(kind="enable", connection_id=identity, consent_version=1),
        )
        assert await store.seconds_until_refresh() == 0.0
        claim = await store.claim(worker_id="loop", lease_seconds=30)
        assert claim is not None
        assert await store.seconds_until_refresh() is None
        assert await store.publish(
            claim=claim,
            catalogue=claim.connection.catalogue,
            error=None,
            retry_seconds=policy.refresh_seconds,
            policy=policy,
        )
        due = await store.seconds_until_refresh()
        assert due is not None and 590 < due <= 600
        await store.create_oauth_flow(
            flow=OAuthFlow("flow", "a", identity, "worker", "https://example.com/mcp"),
            expected_revision=(await service.read(**owner)).revision,
            lifetime=60,
            lease=10,
        )
        assert await store.seconds_until_refresh() is None


@pytest.mark.asyncio
async def test_endpoint_candidate_failure_keeps_enabled_head_and_success_preserves_epoch():
    class EndpointMcp(FakeMcp):
        async def discover(self, **kwargs):
            if kwargs["endpoint"] == "https://unavailable.example/mcp":
                raise ConnectionsError("fixture fault")
            return await super().discover(**kwargs)

    async with isolated_run_runtime("connection_endpoint") as (_, pool):
        store = PGConnectionsStore(pool=pool)
        await store.initialize(validate_only=False)
        service = Connections(store=store, mcp=EndpointMcp(), cipher=CredentialCipher(None))
        draft = await service.change(
            owner_id="a",
            expected_revision="0",
            command=ConnectionCommand(
                kind="create", label="Fixture", endpoint="https://example.com/mcp"
            ),
        )
        identity = draft.connections[0].connection_id
        ready = await service.change(
            owner_id="a",
            expected_revision=draft.revision,
            command=ConnectionCommand(kind="probe", connection_id=identity),
        )
        enabled = await service.change(
            owner_id="a",
            expected_revision=ready.revision,
            command=ConnectionCommand(kind="enable", connection_id=identity, consent_version=1),
        )
        with pytest.raises(ConnectionsError):
            await service.change(
                owner_id="a",
                expected_revision=enabled.revision,
                command=ConnectionCommand(
                    kind="edit", connection_id=identity, endpoint="https://unavailable.example/mcp"
                ),
            )
        assert await service.read(owner_id="a") == enabled
        changed = await service.change(
            owner_id="a",
            expected_revision=enabled.revision,
            command=ConnectionCommand(
                kind="edit", connection_id=identity, endpoint="https://next.example/mcp"
            ),
        )
        assert changed.connections[0].enabled
        assert changed.connections[0].activation_epoch == enabled.connections[0].activation_epoch
        assert changed.connections[0].generation > enabled.connections[0].generation


@pytest.mark.asyncio
async def test_revoke_erases_live_grant_and_delete_tombstones_only_this_owner():
    from pydantic import SecretStr

    from dlightrag.engine.credential_cipher import CredentialCipher
    from tests.unit.test_connections_config import KEYRING

    async with isolated_run_runtime("connection_revoke") as (_, pool):
        store = PGConnectionsStore(pool=pool)
        await store.initialize(validate_only=False)
        service = Connections(
            store=store, mcp=FakeMcp(), cipher=CredentialCipher(SecretStr(KEYRING))
        )
        draft = await service.change(
            owner_id="a",
            expected_revision="0",
            command=ConnectionCommand(
                kind="create", label="Fixture", endpoint="https://example.com/mcp"
            ),
        )
        other = await service.change(
            owner_id="b",
            expected_revision="0",
            command=ConnectionCommand(
                kind="create", label="Other", endpoint="https://example.com/mcp"
            ),
        )
        identity = draft.connections[0].connection_id
        ready = await service.replace_bearer(
            owner_id="a", connection_id=identity, bearer=SecretStr("fixture")
        )
        enabled = await service.change(
            owner_id="a",
            expected_revision=ready.revision,
            command=ConnectionCommand(kind="enable", connection_id=identity, consent_version=1),
        )
        revoked = await service.change(
            owner_id="a",
            expected_revision=enabled.revision,
            command=ConnectionCommand(kind="revoke", connection_id=identity),
        )
        assert not revoked.connections[0].enabled
        assert revoked.connections[0].status == "revoked"
        assert revoked.connections[0].activation_epoch > enabled.connections[0].activation_epoch
        assert (await store.read("a"))[1][0].envelope is None
        with pytest.raises(ConnectionsError):
            await service.change(
                owner_id="a",
                expected_revision=revoked.revision,
                command=ConnectionCommand(kind="enable", connection_id=identity, consent_version=1),
            )
        deleted = await service.change(
            owner_id="a",
            expected_revision=revoked.revision,
            command=ConnectionCommand(kind="delete", connection_id=identity),
        )
        assert deleted.connections == ()
        assert await service.read(owner_id="b") == other


@pytest.mark.asyncio
async def test_a_header_credential_and_stated_headers_round_trip_to_dispatch():
    """The whole plane for one endpoint: two facts stored, both reconstructed at every request.

    A header credential's name and prefix live inside the sealed envelope -- no column holds
    them -- and a Connection's stated headers live on the head, unencrypted because they are not
    secrets. This proves each survives storage, and that discovery is spoken to with both.
    """
    import json
    from dataclasses import asdict

    from pydantic import SecretStr

    from tests.unit.test_connections_config import KEYRING

    class RecordingMcp(FakeMcp):
        seen = []

        async def discover(self, *, endpoint, access, policy, **kwargs):
            self.seen.append((access, kwargs))
            return await super().discover(endpoint=endpoint, access=access, policy=policy, **kwargs)

    async with isolated_run_runtime("connection_header_credential") as (_, pool):
        store = PGConnectionsStore(pool=pool)
        await store.initialize(validate_only=False)
        mcp = RecordingMcp()
        service = Connections(store=store, mcp=mcp, cipher=CredentialCipher(SecretStr(KEYRING)))
        stated = (("X-Lark-MCP-Allowed-Tools", "create-doc,fetch-doc"),)
        view = await service.change(
            owner_id="a",
            expected_revision="0",
            command=ConnectionCommand(
                kind="create",
                label="Feishu/Lark",
                endpoint="https://mcp.feishu.cn/mcp",
                endpoint_headers=stated,
            ),
        )
        identity = view.connections[0].connection_id
        assert view.connections[0].endpoint_headers == stated

        token = SecretStr("t-fixture-tenant-access-token")
        view = await service.replace_header_credential(
            owner_id="a",
            connection_id=identity,
            expected_revision=view.revision,
            header="X-Lark-MCP-TAT",
            prefix="",
            token=token,
        )
        # The refresh the credential save asks for reads the endpoint with both facts.
        sent, transport = mcp.seen[-1]
        assert sent is not None and sent.token == token
        assert sent.header == "X-Lark-MCP-TAT"
        assert sent.prefix == ""
        assert transport["endpoint_headers"] == stated
        assert view.connections[0].authentication == "header"

        # Neither the token nor the header it travels in reaches a redacted projection or the
        # stored row: the name is part of the credential, so it is part of the ciphertext.
        assert "t-fixture-tenant-access-token" not in json.dumps(asdict(view))
        _, stored = await store.read("a")
        assert "t-fixture-tenant-access-token" not in repr(stored)
        assert "X-Lark-MCP-TAT" not in repr(stored)
        assert stored[0].endpoint_headers == stated

        # Stated headers are configuration, so they change without a new credential, and the
        # form of every refusal is the Connections error the route already reports.
        with pytest.raises(ConnectionsError) as refused:
            await service.change(
                owner_id="a",
                expected_revision=view.revision,
                command=ConnectionCommand(
                    kind="edit",
                    connection_id=identity,
                    endpoint_headers=(("Authorization", "Bearer fixture"),),
                ),
            )
        assert refused.value.status == 422
        assert "fixture" not in str(refused.value)

        view = await service.change(
            owner_id="a",
            expected_revision=view.revision,
            command=ConnectionCommand(
                kind="edit",
                connection_id=identity,
                endpoint_headers=(("X-Lark-MCP-Allowed-Tools", "fetch-doc"),),
            ),
        )
        assert view.connections[0].endpoint_headers == (("X-Lark-MCP-Allowed-Tools", "fetch-doc"),)

        # The credential's own header is refused as a stated header: the credential is applied
        # last, and nothing may quietly overwrite it.
        with pytest.raises(ConnectionsError) as collision:
            await service.change(
                owner_id="a",
                expected_revision=view.revision,
                command=ConnectionCommand(
                    kind="edit",
                    connection_id=identity,
                    endpoint_headers=(("x-lark-mcp-tat", "fixture"),),
                ),
            )
        assert collision.value.status == 422


@pytest.mark.asyncio
async def test_a_server_that_publishes_no_tools_says_so_instead_of_ready():
    """An endpoint that answers with an empty catalogue is observed as offering nothing.

    This is the shape a request header produces: a server whose tool allowlist is stated and
    empty (or wrong) answers `tools/list` with nothing at all. `ready` would tell the owner the
    Connection works while the Agent gets no tools from it, so the observation says what
    happened, and the next discovery that finds tools says `ready` again.
    """

    class EmptyThenFull(FakeMcp):
        empty = True

        async def discover(self, **kwargs):
            self.calls = getattr(self, "calls", 0) + 1
            if self.empty:
                return McpDiscovery(tools=(), instructions=None)
            return await super().discover(**kwargs)

    async with isolated_run_runtime("connection_no_tools") as (_, pool):
        store = PGConnectionsStore(pool=pool)
        await store.initialize(validate_only=False)
        mcp = EmptyThenFull()
        service = Connections(store=store, mcp=mcp, cipher=CredentialCipher(None))
        view = await service.change(
            owner_id="a",
            expected_revision="0",
            command=ConnectionCommand(
                kind="create", label="Feishu/Lark", endpoint="https://mcp.feishu.cn/mcp"
            ),
        )
        identity = view.connections[0].connection_id
        view = await service.change(
            owner_id="a",
            expected_revision=view.revision,
            command=ConnectionCommand(kind="probe", connection_id=identity),
        )
        assert view.connections[0].status == "no-tools"

        mcp.empty = False
        view = await service.change(
            owner_id="a",
            expected_revision=view.revision,
            command=ConnectionCommand(kind="probe", connection_id=identity),
        )
        assert view.connections[0].status == "ready"


@pytest.mark.asyncio
async def test_every_path_that_publishes_a_catalogue_reports_an_empty_one_as_no_tools():
    """The observation rule holds wherever a catalogue is published, not only on a refresh.

    A credential save that carries an endpoint and an endpoint edit both publish what discovery
    found. Each one that found nothing has to say so: `ready` would tell the owner the Connection
    works while the Agent gets no tools from it.
    """
    from pydantic import SecretStr

    from tests.unit.test_connections_config import KEYRING

    class Nothing(FakeMcp):
        async def discover(self, **kwargs):
            return McpDiscovery(tools=(), instructions=None)

    async with isolated_run_runtime("connection_empty_catalogue") as (_, pool):
        store = PGConnectionsStore(pool=pool)
        await store.initialize(validate_only=False)
        service = Connections(
            store=store, mcp=Nothing(), cipher=CredentialCipher(SecretStr(KEYRING))
        )
        view = await service.change(
            owner_id="a",
            expected_revision="0",
            command=ConnectionCommand(
                kind="create", label="Docs", endpoint="https://one.example/mcp"
            ),
        )
        identity = view.connections[0].connection_id
        for command in (
            ConnectionCommand(kind="probe", connection_id=identity),
            ConnectionCommand(kind="enable", connection_id=identity, consent_version=1),
        ):
            view = await service.change(
                owner_id="a", expected_revision=view.revision, command=command
            )
        assert view.connections[0].status == "no-tools"

        # A credential save that carries the endpoint publishes its own discovery, and an
        # enabled Connection that now finds nothing is observed the same way.
        view = await service.replace_bearer(
            owner_id="a",
            connection_id=identity,
            bearer=SecretStr("fixture-token"),
            expected_revision=view.revision,
            endpoint="https://one.example/mcp",
        )
        assert view.connections[0].status == "no-tools"

        # An endpoint edit of an unauthenticated Connection publishes the candidate it probed.
        draft = await service.change(
            owner_id="a",
            expected_revision=view.revision,
            command=ConnectionCommand(
                kind="create", label="Other", endpoint="https://two.example/mcp"
            ),
        )
        other = next(item for item in draft.connections if item.connection_id != identity)
        edited = await service.change(
            owner_id="a",
            expected_revision=draft.revision,
            command=ConnectionCommand(
                kind="edit", connection_id=other.connection_id, endpoint="https://three.example/mcp"
            ),
        )
        current = next(
            item for item in edited.connections if item.connection_id == other.connection_id
        )
        assert current.status == "no-tools"
