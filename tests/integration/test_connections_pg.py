# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Personal Connections behavior through the owner module and real PostgreSQL."""

import pytest

from dlightrag.adapters.postgres.connections import PGConnectionsStore
from dlightrag.application.connections import ConnectionCommand, Connections, ConnectionsError
from tests.integration.run_runtime_pg_harness import isolated_run_runtime


class FakeMcp:
    async def call(self, **kwargs):
        raise AssertionError("Management must not dispatch")

    async def discover(self, *, endpoint, bearer, policy):
        return [
            {"name": "read", "description": "Read a fixture", "input_schema": {"type": "object"}}
        ]


@pytest.mark.asyncio
async def test_owner_draft_probe_consent_and_revision():
    async with isolated_run_runtime("connections") as (_, pool):
        store = PGConnectionsStore(pool=pool)
        await store.initialize(validate_only=False)
        connections = Connections(store=store, mcp=FakeMcp())
        empty = await connections.read(owner_id="a", auth_mode="jwt")
        draft = await connections.change(
            owner_id="a",
            auth_mode="jwt",
            expected_revision=empty.revision,
            command=ConnectionCommand(
                kind="create", label="Fixture", endpoint="https://example.com/mcp"
            ),
        )
        item = draft.connections[0]
        assert item.enabled is False
        assert (await connections.read(owner_id="b", auth_mode="jwt")).connections == ()
        with pytest.raises(ConnectionsError, match="not found"):
            await connections.change(
                owner_id="b",
                auth_mode="jwt",
                expected_revision="0",
                command=ConnectionCommand(kind="probe", connection_id=item.connection_id),
            )
        with pytest.raises(ConnectionsError, match="unavailable"):
            await connections.read(owner_id="shared", auth_mode="simple")
        ready = await connections.change(
            owner_id="a",
            auth_mode="jwt",
            expected_revision=draft.revision,
            command=ConnectionCommand(kind="probe", connection_id=item.connection_id),
        )
        assert ready.connections[0].tools[0].remote_name == "read"
        with pytest.raises(ConnectionsError, match="consent"):
            await connections.change(
                owner_id="a",
                auth_mode="jwt",
                expected_revision=ready.revision,
                command=ConnectionCommand(kind="enable", connection_id=item.connection_id),
            )
        enabled = await connections.change(
            owner_id="a",
            auth_mode="jwt",
            expected_revision=ready.revision,
            command=ConnectionCommand(
                kind="enable", connection_id=item.connection_id, consent_version=1
            ),
        )
        assert enabled.connections[0].enabled
        with pytest.raises(ConnectionsError, match="revision"):
            await connections.change(
                owner_id="a",
                auth_mode="jwt",
                expected_revision=ready.revision,
                command=ConnectionCommand(kind="disable", connection_id=item.connection_id),
            )
        disabled = await connections.change(
            owner_id="a",
            auth_mode="jwt",
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

    from dlightrag.application.connections.credentials import CredentialCipher
    from tests.unit.test_connections_config import KEYRING

    class AuthMcp(FakeMcp):
        observed = None

        async def discover(self, *, endpoint, bearer, policy):
            self.observed = bearer
            return await super().discover(endpoint=endpoint, bearer=bearer, policy=policy)

    async with isolated_run_runtime("connection_secrets") as (_, pool):
        store = PGConnectionsStore(pool=pool)
        await store.initialize(validate_only=False)
        mcp = AuthMcp()
        service = Connections(store=store, mcp=mcp, cipher=CredentialCipher(SecretStr(KEYRING)))
        view = await service.change(
            owner_id="a",
            auth_mode="none",
            expected_revision="0",
            command=ConnectionCommand(
                kind="create", label="External", endpoint="https://example.com/mcp"
            ),
        )
        identity = view.connections[0].connection_id
        secret = SecretStr("test-only-personal-token")
        view = await service.replace_bearer(
            owner_id="a",
            auth_mode="none",
            connection_id=identity,
            bearer=secret,
            expected_revision=view.revision,
        )
        assert mcp.observed == secret
        assert view.connections[0].authentication == "bearer"
        with pytest.raises(ConnectionsError) as edit_error:
            await service.change(
                owner_id="a",
                auth_mode="none",
                expected_revision=view.revision,
                command=ConnectionCommand(
                    kind="edit",
                    connection_id=identity,
                    endpoint="https://another-audience.example/mcp",
                ),
            )
        assert edit_error.value.kind == "requires_reauthorization"
        assert await service.read(owner_id="a", auth_mode="none") == view
        assert "test-only-personal-token" not in json.dumps(asdict(view))
        assert "grant_id" not in json.dumps(asdict(view))
        # Persistence is an accepted seam: prove the live stored record is ciphertext.
        _, stored = await store.read("a")
        assert "test-only-personal-token" not in repr(stored)
        assert "test-only-personal-token" not in (stored[0].envelope or "")
        with pytest.raises(ConnectionsError, match="not found"):
            await service.replace_bearer(
                owner_id="b",
                auth_mode="jwt",
                connection_id=identity,
                bearer=secret,
                expected_revision="0",
            )
        no_key = Connections(store=store, mcp=mcp)
        with pytest.raises(ConnectionsError, match="deployment"):
            await no_key.replace_bearer(
                owner_id="a",
                auth_mode="none",
                connection_id=identity,
                bearer=secret,
                expected_revision=view.revision,
            )


@pytest.mark.asyncio
async def test_refresh_auto_admits_new_tools_and_preserves_last_good_on_fault():
    import asyncio

    from dlightrag.application.connections import ConnectionPolicy

    class ChangingMcp(FakeMcp):
        changed = False

        async def discover(self, **kwargs):
            tools = await super().discover(**kwargs)
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
            return tools

    async with isolated_run_runtime("connection_refresh") as (_, pool):
        mcp = ChangingMcp()
        store = PGConnectionsStore(pool=pool)
        service = Connections(store=store, mcp=mcp, policy=ConnectionPolicy(refresh_seconds=1))
        await service.start()
        try:
            draft = await service.change(
                owner_id="a",
                auth_mode="jwt",
                expected_revision="0",
                command=ConnectionCommand(
                    kind="create", label="Fixture", endpoint="https://example.com/mcp"
                ),
            )
            identity = draft.connections[0].connection_id
            ready = await service.change(
                owner_id="a",
                auth_mode="jwt",
                expected_revision=draft.revision,
                command=ConnectionCommand(kind="probe", connection_id=identity),
            )
            enabled = await service.change(
                owner_id="a",
                auth_mode="jwt",
                expected_revision=ready.revision,
                command=ConnectionCommand(kind="enable", connection_id=identity, consent_version=1),
            )
            mcp.changed = True
            async with asyncio.timeout(5):
                while True:
                    current = await service.read(owner_id="a", auth_mode="jwt")
                    if len(current.connections[0].tools) == 2:
                        break
                    await asyncio.sleep(0.05)
            assert (
                current.connections[0].activation_epoch == enabled.connections[0].activation_epoch
            )
            assert current.connections[0].generation > enabled.connections[0].generation
            assert current.connections[0].tools[0].input_schema == {
                "type": "object",
                "properties": {"new_argument": {"type": "string"}},
            }
            assert (
                current.connections[0].tools[0].local_name
                == enabled.connections[0].tools[0].local_name
            )
        finally:
            await service.aclose()


@pytest.mark.asyncio
async def test_expired_refresh_and_disabled_worker_cannot_publish():
    from dlightrag.application.connections import ConnectionPolicy

    async with isolated_run_runtime("connection_claims") as (_, pool):
        store = PGConnectionsStore(pool=pool)
        await store.initialize(validate_only=False)
        await store.initialize(validate_only=True)
        service = Connections(store=store, mcp=FakeMcp())
        draft = await service.change(
            owner_id="a",
            auth_mode="jwt",
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
            auth_mode="jwt",
            expected_revision=draft.revision,
            command=ConnectionCommand(kind="disable", connection_id=identity),
        )
        assert not await store.publish(
            claim=claim, catalogue=(), error=None, retry_seconds=1, policy=ConnectionPolicy()
        )
        assert (await service.read(owner_id="a", auth_mode="jwt")) == disabled


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
            return bad if self.fault else await super().discover(**kwargs)

    async with isolated_run_runtime("connection_catalogue") as (_, pool):
        store = PGConnectionsStore(pool=pool)
        await store.initialize(validate_only=False)
        mcp = FaultMcp()
        service = Connections(store=store, mcp=mcp)
        draft = await service.change(
            owner_id="a",
            auth_mode="jwt",
            expected_revision="0",
            command=ConnectionCommand(
                kind="create", label="Fixture", endpoint="https://example.com/mcp"
            ),
        )
        identity = draft.connections[0].connection_id
        ready = await service.change(
            owner_id="a",
            auth_mode="jwt",
            expected_revision=draft.revision,
            command=ConnectionCommand(kind="probe", connection_id=identity),
        )
        mcp.fault = True
        fault = await service.change(
            owner_id="a",
            auth_mode="jwt",
            expected_revision=ready.revision,
            command=ConnectionCommand(kind="probe", connection_id=identity),
        )
        assert fault.connections[0].status == "degraded"
        assert fault.connections[0].generation == ready.connections[0].generation
        assert fault.connections[0].tools == ready.connections[0].tools


@pytest.mark.asyncio
async def test_unreadable_existing_grant_cannot_be_overwritten():
    import json

    from pydantic import SecretStr

    from dlightrag.application.connections.credentials import CredentialCipher
    from tests.unit.test_connections_config import KEYRING

    async with isolated_run_runtime("connection_rotation") as (_, pool):
        store = PGConnectionsStore(pool=pool)
        await store.initialize(validate_only=False)
        service = Connections(
            store=store, mcp=FakeMcp(), cipher=CredentialCipher(SecretStr(KEYRING))
        )
        view = await service.change(
            owner_id="a",
            auth_mode="jwt",
            expected_revision="0",
            command=ConnectionCommand(
                kind="create", label="Fixture", endpoint="https://example.com/mcp"
            ),
        )
        identity = view.connections[0].connection_id
        view = await service.replace_bearer(
            owner_id="a", auth_mode="jwt", connection_id=identity, bearer=SecretStr("fixture")
        )
        ring = json.loads(KEYRING)
        ring["keys"]["new"] = ring["keys"].pop("test")
        ring["active"] = "new"
        unreadable = Connections(
            store=store, mcp=FakeMcp(), cipher=CredentialCipher(SecretStr(json.dumps(ring)))
        )
        with pytest.raises(ConnectionsError, match="deployment"):
            await unreadable.replace_bearer(
                owner_id="a",
                auth_mode="jwt",
                connection_id=identity,
                bearer=SecretStr("replacement"),
            )
        assert await service.read(owner_id="a", auth_mode="jwt") == view


@pytest.mark.asyncio
async def test_expired_worker_loses_publication_to_new_claim():
    import asyncio

    from dlightrag.application.connections import ConnectionPolicy

    async with isolated_run_runtime("connection_expiry") as (_, pool):
        store = PGConnectionsStore(pool=pool)
        await store.initialize(validate_only=False)
        service = Connections(store=store, mcp=FakeMcp())
        view = await service.change(
            owner_id="a",
            auth_mode="jwt",
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
        assert (await service.read(owner_id="a", auth_mode="jwt")).connections[0].generation == 1


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
        service = Connections(store=store, mcp=EndpointMcp())
        draft = await service.change(
            owner_id="a",
            auth_mode="jwt",
            expected_revision="0",
            command=ConnectionCommand(
                kind="create", label="Fixture", endpoint="https://example.com/mcp"
            ),
        )
        identity = draft.connections[0].connection_id
        ready = await service.change(
            owner_id="a",
            auth_mode="jwt",
            expected_revision=draft.revision,
            command=ConnectionCommand(kind="probe", connection_id=identity),
        )
        enabled = await service.change(
            owner_id="a",
            auth_mode="jwt",
            expected_revision=ready.revision,
            command=ConnectionCommand(kind="enable", connection_id=identity, consent_version=1),
        )
        with pytest.raises(ConnectionsError):
            await service.change(
                owner_id="a",
                auth_mode="jwt",
                expected_revision=enabled.revision,
                command=ConnectionCommand(
                    kind="edit", connection_id=identity, endpoint="https://unavailable.example/mcp"
                ),
            )
        assert await service.read(owner_id="a", auth_mode="jwt") == enabled
        changed = await service.change(
            owner_id="a",
            auth_mode="jwt",
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

    from dlightrag.application.connections.credentials import CredentialCipher
    from tests.unit.test_connections_config import KEYRING

    async with isolated_run_runtime("connection_revoke") as (_, pool):
        store = PGConnectionsStore(pool=pool)
        await store.initialize(validate_only=False)
        service = Connections(
            store=store, mcp=FakeMcp(), cipher=CredentialCipher(SecretStr(KEYRING))
        )
        draft = await service.change(
            owner_id="a",
            auth_mode="jwt",
            expected_revision="0",
            command=ConnectionCommand(
                kind="create", label="Fixture", endpoint="https://example.com/mcp"
            ),
        )
        other = await service.change(
            owner_id="b",
            auth_mode="jwt",
            expected_revision="0",
            command=ConnectionCommand(
                kind="create", label="Other", endpoint="https://example.com/mcp"
            ),
        )
        identity = draft.connections[0].connection_id
        ready = await service.replace_bearer(
            owner_id="a", auth_mode="jwt", connection_id=identity, bearer=SecretStr("fixture")
        )
        enabled = await service.change(
            owner_id="a",
            auth_mode="jwt",
            expected_revision=ready.revision,
            command=ConnectionCommand(kind="enable", connection_id=identity, consent_version=1),
        )
        revoked = await service.change(
            owner_id="a",
            auth_mode="jwt",
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
                auth_mode="jwt",
                expected_revision=revoked.revision,
                command=ConnectionCommand(kind="enable", connection_id=identity, consent_version=1),
            )
        deleted = await service.change(
            owner_id="a",
            auth_mode="jwt",
            expected_revision=revoked.revision,
            command=ConnectionCommand(kind="delete", connection_id=identity),
        )
        assert deleted.connections == ()
        assert await service.read(owner_id="b", auth_mode="jwt") == other
