# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Grant refresh and retained metadata lifecycle against actual PostgreSQL."""

import asyncio
import json
from dataclasses import replace
from typing import Literal
from urllib.parse import parse_qs

import httpx2
import pytest

from dlightrag.adapters.mcp.oauth import PersonalOAuthClient
from dlightrag.adapters.postgres.connections import PGConnectionsStore
from dlightrag.application.connections import ConnectionCommand, ConnectionPolicy, Connections
from tests.integration.run_runtime_pg_harness import isolated_run_runtime
from tests.integration.test_connection_authorization_pg import cipher
from tests.integration.test_connection_binding_pg import enabled_connection
from tests.integration.test_connections_pg import stored_catalogue
from tests.support.dns import public_dns
from tests.unit.test_connection_oauth import refresh_credentials


async def oauth_connection(pool, remote):
    _, store, mcp, view = await enabled_connection(pool)
    identity = view.connections[0].connection_id
    key, envelope = cipher().encrypt(
        refresh_credentials(), owner_id="a", connection_id=identity, grant_id="grant"
    )
    await store.publish_authorization(
        owner_id="a",
        connection_id=identity,
        expected_revision=view.revision,
        endpoint="https://mcp.example/mcp",
        grant_id="grant",
        kind="oauth",
        key_id=key,
        envelope=envelope,
        scopes=("read",),
        catalogue=await stored_catalogue(store),
        policy=ConnectionPolicy(),
    )
    service = Connections(
        store=store,
        mcp=mcp,
        cipher=cipher(),
        oauth=PersonalOAuthClient(transport_factory=lambda: httpx2.MockTransport(remote)),
    )
    return service, store, mcp, identity


@pytest.mark.asyncio
@pytest.mark.parametrize("race", ["selection", "cas"])
async def test_cosmetic_reencryption_preserves_live_rotating_refresh(race, monkeypatch):
    import base64

    from pydantic import SecretStr

    from dlightrag.application.connections.credentials import CredentialCipher

    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    async with isolated_run_runtime("rotation_refresh") as (_, pool):
        entered, release = asyncio.Event(), asyncio.Event()
        refresh_tokens = []

        async def remote(request):
            assert request.headers["host"] == "as.example"
            token = parse_qs(request.content.decode())["refresh_token"][0]
            refresh_tokens.append(token)
            assert token == ("refresh-secret" if len(refresh_tokens) == 1 else "rotated-refresh-1")
            entered.set()
            await release.wait()
            return httpx2.Response(
                200,
                json={
                    "access_token": "fresh-token",
                    "refresh_token": f"rotated-refresh-{len(refresh_tokens)}",
                    "token_type": "Bearer",
                    "expires_in": 300,
                    "scope": "read",
                },
            )

        service, store, mcp, identity = await oauth_connection(pool, remote)
        rotated = CredentialCipher(
            SecretStr(
                json.dumps(
                    {
                        "active": "next",
                        "keys": {
                            "test": base64.urlsafe_b64encode(b"t" * 32).decode(),
                            "next": base64.urlsafe_b64encode(b"n" * 32).decode(),
                        },
                    }
                )
            )
        )
        maintenance = Connections(store=PGConnectionsStore(pool=pool), mcp=mcp, cipher=rotated)
        # Select before the refresh claim: its version/envelope stay unchanged
        # when the lease is claimed, so only the actual CAS lease guard can win.
        (candidate,) = await store.rotation_candidates(active_key_id="next", limit=100)
        view = await service.read(owner_id="a", auth_mode="jwt")
        task = asyncio.create_task(
            service.change(
                owner_id="a",
                auth_mode="jwt",
                expected_revision=view.revision,
                command=ConnectionCommand(kind="probe", connection_id=identity),
            )
        )
        try:
            await asyncio.wait_for(entered.wait(), 2)
            if race == "selection":
                assert await store.rotation_candidates(active_key_id="next", limit=100) == ()
                assert (await maintenance.maintain())["reencrypted"] == 0
            else:
                key, envelope = rotated.encrypt(
                    refresh_credentials(), owner_id="a", connection_id=identity, grant_id="grant"
                )
                assert not await store.reencrypt_grant(
                    grant=candidate, key_id=key, envelope=envelope
                )
            release.set()
            view = await asyncio.wait_for(task, 3)
            assert view.connections[0].status == "ready"
            assert (await maintenance.maintain())["reencrypted"] == 1
            _, (item,) = await store.read("a")
            assert item.envelope is not None and item.secret_version == 3
            credentials = json.loads(
                rotated.decrypt(
                    item.envelope, owner_id="a", connection_id=identity, grant_id="grant"
                ).get_secret_value()
            )
            assert credentials["tokens"]["refresh_token"] == "rotated-refresh-1"
            # Deterministically advance credential expiry, not a sleep or an
            # invented production TTL. The next SDK refresh must use the new token.
            credentials["expires_at"] = 1
            key, envelope = rotated.encrypt(
                SecretStr(json.dumps(credentials)),
                owner_id="a",
                connection_id=identity,
                grant_id="grant",
            )
            async with pool.acquire() as conn:
                await conn.execute(
                    "UPDATE dlightrag_connection_grants SET encrypted_envelope=$1 WHERE grant_id='grant'",
                    envelope,
                )
            next_worker = Connections(
                store=PGConnectionsStore(pool=pool),
                mcp=mcp,
                cipher=rotated,
                oauth=PersonalOAuthClient(transport_factory=lambda: httpx2.MockTransport(remote)),
            )
            view = await next_worker.change(
                owner_id="a",
                auth_mode="jwt",
                expected_revision=view.revision,
                command=ConnectionCommand(kind="probe", connection_id=identity),
            )
            assert view.connections[0].status == "ready"
            assert refresh_tokens == ["refresh-secret", "rotated-refresh-1"]
            assert (await maintenance.maintain())["reencrypted"] == 0
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "scope", "rejected"])
async def test_probe_automatically_refreshes_same_grant_without_network_transaction(
    outcome, monkeypatch
):
    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    async with isolated_run_runtime("grant_refresh") as (_, pool):
        requests = []

        async def remote(request):
            requests.append(request)
            assert request.headers["host"] == "as.example"
            async with pool.acquire() as conn:
                assert (
                    await conn.fetchval(
                        "SELECT count(*) FROM pg_stat_activity WHERE datname=current_database() AND state='idle in transaction'"
                    )
                    == 0
                )
            if outcome == "rejected":
                return httpx2.Response(400, json={"error": "invalid_grant"})
            return httpx2.Response(
                200,
                json={
                    "access_token": "new-token",
                    "token_type": "Bearer",
                    "expires_in": 300,
                    "scope": "read write" if outcome == "scope" else "read",
                },
            )

        service, store, mcp, identity = await oauth_connection(pool, remote)
        view = await service.read(owner_id="a", auth_mode="jwt")
        calls = mcp.calls
        view = await service.change(
            owner_id="a",
            auth_mode="jwt",
            expected_revision=view.revision,
            command=ConnectionCommand(kind="probe", connection_id=identity),
        )
        assert len(requests) == 1
        assert view.connections[0].status == ("ready" if outcome == "success" else "needs-auth")
        assert mcp.calls == calls + (outcome == "success")
        _, (item,) = await store.read("a")
        assert item.grant_id == "grant"
        assert item.secret_version == (2 if outcome == "success" else 1)
        async with pool.acquire() as conn:
            assert (
                await conn.fetchval(
                    "SELECT refresh_owner FROM dlightrag_connection_grants WHERE grant_id='grant'"
                )
                is None
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("winner", ["revoke", "rotation", "takeover"])
async def test_expiring_grant_lease_and_fenced_cas_cannot_overwrite_new_authority(winner):
    async with isolated_run_runtime("grant_cas") as (_, pool):
        service, a, _, identity = await oauth_connection(pool, lambda _: None)
        b = PGConnectionsStore(pool=pool)
        args = dict(
            owner_id="a",
            connection_id=identity,
            grant_id="grant",
            endpoint="https://mcp.example/mcp",
        )
        first = await a.claim_grant_refresh(**args, lease_seconds=30, worker_id="a")
        assert first is not None
        assert await b.claim_grant_refresh(**args, lease_seconds=30, worker_id="b") is None
        if winner == "revoke":
            view = await service.read(owner_id="a", auth_mode="jwt")
            await service.change(
                owner_id="a",
                auth_mode="jwt",
                expected_revision=view.revision,
                command=ConnectionCommand(kind="revoke", connection_id=identity),
            )
        elif winner == "rotation":
            async with pool.acquire() as conn:
                await conn.execute(
                    "UPDATE dlightrag_connection_grants SET refresh_expires_at=now()-interval '1 second'"
                )
            assert await b.reencrypt_grant(
                grant=first.grant, key_id="new", envelope="rotated-ciphertext"
            )
        else:
            async with pool.acquire() as conn:
                await conn.execute(
                    "UPDATE dlightrag_connection_grants SET refresh_expires_at=now()-interval '1 second'"
                )
            second = await b.claim_grant_refresh(**args, lease_seconds=30, worker_id="b")
            assert second is not None and second.grant.refresh_epoch > first.grant.refresh_epoch
            await a.release_grant_refresh(claim=first)
            assert await a.claim_grant_refresh(**args, lease_seconds=30, worker_id="c") is None
            assert await b.save_grant_refresh(
                claim=second, key_id="test", envelope="winning-ciphertext"
            )
        assert not await a.save_grant_refresh(
            claim=first, key_id="test", envelope="stale-ciphertext"
        )
        async with pool.acquire() as conn:
            stored = await conn.fetchval(
                "SELECT encrypted_envelope FROM dlightrag_connection_grants WHERE grant_id='grant'"
            )
        assert (
            stored
            == {"revoke": None, "rotation": "rotated-ciphertext", "takeover": "winning-ciphertext"}[
                winner
            ]
        )


@pytest.mark.asyncio
async def test_maintenance_retains_pins_then_run_cascade_releases_tombstone():
    from uuid import uuid7

    from tests.integration.run_runtime_pg_harness import run_envelope

    async with isolated_run_runtime("connection_gc") as (runs, pool):
        service, store, mcp, view = await enabled_connection(pool)
        bound = await service.bind_research(owner_id="a", auth_mode="jwt")
        envelope = run_envelope("answer", key="pin", owner="a", mode="research")
        envelope = replace(
            envelope,
            payload={
                **envelope.payload,
                "run_connection_bindings": [b.as_json() for b in bound.bindings],
            },
        )
        accepted = await runs.accept_run(
            envelope=envelope, run_id=str(uuid7()), connection_bindings=bound.bindings
        )
        identity = view.connections[0].connection_id
        for _ in range(3):
            view = await service.change(
                owner_id="a",
                auth_mode="jwt",
                expected_revision=view.revision,
                command=ConnectionCommand(kind="probe", connection_id=identity),
            )
        assert (await service.maintain())["collected"] == 3
        assert await store.pinned_catalogues(
            owner_id="a", run_id=accepted.run.run_id, bindings=bound.bindings
        ) == await stored_catalogue(store)
        view = await service.change(
            owner_id="a",
            auth_mode="jwt",
            expected_revision=view.revision,
            command=ConnectionCommand(kind="delete", connection_id=identity),
        )
        await service.maintain()
        async with pool.acquire() as conn:
            assert await conn.fetchval("SELECT count(*) FROM dlightrag_connection_generations") == 2
            assert await conn.fetchval("SELECT count(*) FROM dlightrag_answer_connection_pins") == 1
            await conn.execute("DELETE FROM dlightrag_runs WHERE run_id=$1", accepted.run.run_id)
        await service.maintain()
        async with pool.acquire() as conn:
            for table in (
                "dlightrag_answer_connection_pins",
                "dlightrag_connection_generations",
                "dlightrag_connection_heads",
            ):
                assert await conn.fetchval(f"SELECT count(*) FROM {table}") == 0
        assert mcp.calls == 4


@pytest.mark.asyncio
async def test_keyring_maintenance_is_cas_safe_and_old_key_can_be_removed():
    import base64

    from pydantic import SecretStr

    from dlightrag.application.connections.credentials import CredentialCipher

    async with isolated_run_runtime("connection_keys") as (_, pool):
        service, store, mcp, identity = await oauth_connection(pool, lambda _: None)
        keys = {
            "test": base64.urlsafe_b64encode(b"t" * 32).decode(),
            "next": base64.urlsafe_b64encode(b"n" * 32).decode(),
        }
        rotated = CredentialCipher(SecretStr(json.dumps({"active": "next", "keys": keys})))
        workers = [
            Connections(store=PGConnectionsStore(pool=pool), mcp=mcp, cipher=rotated)
            for _ in range(2)
        ]
        results = await asyncio.gather(*(worker.maintain() for worker in workers))
        assert sum(result["reencrypted"] for result in results) == 1
        _, (item,) = await store.read("a")
        assert item.envelope is not None and item.grant_id is not None
        assert item.secret_version == 2
        only_new = CredentialCipher(
            SecretStr(json.dumps({"active": "next", "keys": {"next": keys["next"]}}))
        )
        assert (
            only_new.decrypt(
                item.envelope, owner_id="a", connection_id=identity, grant_id=item.grant_id
            )
            == refresh_credentials()
        )
        assert (await workers[0].maintain())["reencrypted"] == 0
        view = await service.read(owner_id="a", auth_mode="jwt")
        await service.change(
            owner_id="a",
            auth_mode="jwt",
            expected_revision=view.revision,
            command=ConnectionCommand(kind="revoke", connection_id=identity),
        )
        async with pool.acquire() as conn:
            assert (
                await conn.fetchval(
                    "SELECT encrypted_envelope FROM dlightrag_connection_grants WHERE grant_id='grant'"
                )
                is None
            )


@pytest.mark.asyncio
async def test_gc_preserves_live_claims_and_expires_callback_ciphertext():
    from dlightrag.application.connections.models import OAuthFlow

    async with isolated_run_runtime("connection_gc_claim") as (_, pool):
        service, store, _, view = await enabled_connection(pool)
        identity = view.connections[0].connection_id
        claim = await store.claim(
            worker_id="busy", lease_seconds=30, owner_id="a", connection_id=identity
        )
        assert claim is not None
        assert (await service.maintain())["collected"] == 0
        flow = OAuthFlow(
            "flow", "a", identity, "flow-worker", "https://example.com/mcp", view.revision
        )
        await store.create_oauth_flow(flow=flow, lifetime=30, lease=10)
        await store.oauth_credentials(flow=flow, envelope="encrypted-fixture")
        assert (await service.maintain())["collected"] == 0
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE dlightrag_connection_oauth_flows SET expires_at=now()-interval '1 second'"
            )
            await conn.execute(
                "UPDATE dlightrag_connection_heads SET refresh_expires_at=now()-interval '1 second'"
            )
        assert (await service.maintain())["collected"] == 2
        assert not await store.publish(
            claim=claim,
            catalogue=claim.connection.catalogue,
            error=None,
            retry_seconds=30,
            policy=ConnectionPolicy(),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome",
    [
        "concurrent",
        "revoke",
        "rotation",
        "replacement",
        "scope",
        "revoked-first",
        "cancelled-first",
    ],
)
async def test_sdk_effect_refresh_is_fenced_before_and_after_remote_io(outcome, monkeypatch):
    import base64

    from pydantic import SecretStr

    from dlightrag.application.connections.credentials import CredentialCipher
    from tests.integration.test_connection_dispatch_pg import dispatch_fixture
    from tests.unit.test_connections_config import KEYRING

    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    async with isolated_run_runtime("refresh_effect") as (runs, pool):
        first = await dispatch_fixture(runs, pool, authenticated=True)
        _, store, mcp, view, bound, claim, _, runtime = first
        ring = CredentialCipher(SecretStr(KEYRING))
        _, (item,) = await store.read("a")
        assert item.grant_id is not None
        raw = (
            refresh_credentials()
            .get_secret_value()
            .replace("https://mcp.example/mcp", item.endpoint)
        )
        key_id, envelope = ring.encrypt(
            SecretStr(raw), owner_id="a", connection_id=item.connection_id, grant_id=item.grant_id
        )
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE dlightrag_connection_grants SET kind='oauth',consented_scopes='[\"read\"]',key_id=$1,encrypted_envelope=$2",
                key_id,
                envelope,
            )
        entered, release = asyncio.Event(), asyncio.Event()
        requests = []

        async def remote(request):
            requests.append(request)
            assert request.headers["host"] == "as.example"
            async with pool.acquire() as conn:
                assert (
                    await conn.fetchval(
                        "SELECT count(*) FROM pg_stat_activity WHERE datname=current_database() AND state='idle in transaction'"
                    )
                    == 0
                )
            entered.set()
            await release.wait()
            return httpx2.Response(
                200,
                json={
                    "access_token": "fresh-token",
                    "refresh_token": "rotated-refresh-token",
                    "token_type": "Bearer",
                    "expires_in": 300,
                    "scope": "read write" if outcome == "scope" else "read",
                },
            )

        def worker():
            return Connections(
                store=PGConnectionsStore(pool=pool),
                mcp=mcp,
                cipher=ring,
                oauth=PersonalOAuthClient(transport_factory=lambda: httpx2.MockTransport(remote)),
            )

        a, b = worker(), worker()
        (tool,) = await a.restore_research(bindings=bound.bindings, claim=claim)
        if outcome == "revoked-first":
            await a.change(
                owner_id="a",
                auth_mode="jwt",
                expected_revision=view.revision,
                command=ConnectionCommand(kind="revoke", connection_id=item.connection_id),
            )
        if outcome == "cancelled-first":
            async with pool.acquire() as conn:
                await conn.execute("UPDATE dlightrag_runs SET cancel_requested_at=now()")
        task = asyncio.ensure_future(
            tool.execute(tool.input_model.model_validate({"path": "x"}), runtime)
        )
        other = None
        rotating = None
        if outcome not in {"revoked-first", "cancelled-first"}:
            await asyncio.wait_for(entered.wait(), 2)
            if outcome == "concurrent":
                second = await dispatch_fixture(
                    runs,
                    pool,
                    connection=(a, store, mcp, view),
                    key="second",
                    worker="second-worker",
                )
                (second_tool,) = await b.restore_research(
                    bindings=second[4].bindings, claim=second[5]
                )
                other = asyncio.ensure_future(
                    second_tool.execute(
                        second_tool.input_model.model_validate({"path": "x"}), second[7]
                    )
                )
                await asyncio.sleep(0.05)
                assert len(requests) == 1
            elif outcome == "revoke":
                await b.change(
                    owner_id="a",
                    auth_mode="jwt",
                    expected_revision=view.revision,
                    command=ConnectionCommand(kind="revoke", connection_id=item.connection_id),
                )
            elif outcome == "replacement":
                await b.replace_bearer(
                    owner_id="a",
                    auth_mode="jwt",
                    connection_id=item.connection_id,
                    expected_revision=view.revision,
                    endpoint=item.endpoint,
                    bearer=SecretStr("replacement-token"),
                )
            elif outcome == "rotation":
                keys = json.loads(KEYRING)
                keys["active"] = "next"
                keys["keys"]["next"] = base64.urlsafe_b64encode(b"n" * 32).decode()
                rotating = Connections(
                    store=PGConnectionsStore(pool=pool),
                    mcp=mcp,
                    cipher=CredentialCipher(SecretStr(json.dumps(keys))),
                )
                assert (await rotating.maintain())["reencrypted"] == 0
            release.set()
        result = await asyncio.wait_for(task, 3)
        if other is not None:
            assert not (await asyncio.wait_for(other, 3)).is_error
        assert result.is_error == (outcome not in {"concurrent", "rotation"})
        assert len(requests) == (0 if outcome in {"revoked-first", "cancelled-first"} else 1)
        assert mcp.call.await_count == (
            2 if outcome == "concurrent" else 1 if outcome == "rotation" else 0
        )
        _, (final,) = await store.read("a")
        if outcome == "concurrent":
            assert final.secret_version == item.secret_version + 1
            assert mcp.call.call_args.kwargs["bearer"].get_secret_value() == "fresh-token"
        elif outcome == "rotation":
            assert rotating is not None
            assert (await rotating.maintain())["reencrypted"] == 1
            _, (final,) = await store.read("a")
            assert final.envelope and '"key_id": "next"' in final.envelope
        elif outcome == "replacement":
            assert final.grant_id != item.grant_id
            async with pool.acquire() as conn:
                assert (
                    await conn.fetchval(
                        "SELECT encrypted_envelope FROM dlightrag_connection_grants WHERE grant_id=$1",
                        item.grant_id,
                    )
                    is None
                )
        await a.aclose()
        await b.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation", ["refresh", "rotation", "revoke", "replacement", "cancel", "disable_enable"]
)
@pytest.mark.parametrize("child", [False, True], ids=["parent", "child"])
async def test_in_flight_http_call_tracks_authorization_not_secret_version(
    mutation, child, monkeypatch
):
    import base64
    import time
    import uuid

    from pydantic import SecretStr

    from dlightrag.adapters.mcp.personal_http import PersonalMcpClient
    from dlightrag.application.connections.credentials import CredentialCipher
    from tests.integration.test_connection_dispatch_pg import dispatch_fixture
    from tests.unit.test_connections_config import KEYRING

    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    async with isolated_run_runtime("dispatch_lifetime") as (runs, pool):
        setup, store, fake_mcp, view, bound, claim, _, runtime = await dispatch_fixture(
            runs, pool, authenticated=True
        )
        if child:
            parent_scope, child_scope = uuid.UUID(runtime.execution_scope), uuid.uuid7()
            async with pool.acquire() as conn:
                await conn.execute(
                    """INSERT INTO dlightrag_answer_child_sessions
                    (owner_id,run_id,child_session_id,parent_session_id,parent_call_id,status,depth,context_snapshot_json,
                     lease_owner,lease_expires_at,fencing_epoch)
                    VALUES('a',$1,$2,$3,'spawn','running',1,'{}','worker',now()+interval '1 minute',7)""",
                    uuid.UUID(claim.run_id),
                    child_scope,
                    parent_scope,
                )
                await conn.execute(
                    "INSERT INTO dlightrag_agent_sessions(owner_id,session_id,lease_run_id,fencing_epoch) VALUES('a',$1,$2,7)",
                    child_scope,
                    uuid.UUID(claim.run_id),
                )
                await conn.execute(
                    """INSERT INTO dlightrag_agent_session_registers SELECT owner_id,$1,register_kind,register_key,sequence,payload_json
                    FROM dlightrag_agent_session_registers WHERE session_id=$2""",
                    child_scope,
                    parent_scope,
                )
            runtime = replace(runtime, execution_scope=str(child_scope), fencing_epoch=7)
        keys = json.loads(KEYRING)
        keys["keys"]["next"] = base64.urlsafe_b64encode(b"n" * 32).decode()
        ring = CredentialCipher(SecretStr(json.dumps(keys)))
        _, (item,) = await store.read("a")
        assert item.grant_id is not None
        credentials = json.loads(refresh_credentials().get_secret_value())
        credentials["resource_metadata"]["resource"] = item.endpoint
        credentials["expires_at"] = time.time() + 300
        credentials["tokens"]["access_token"] = "initial-token"

        async def set_credentials():
            assert item.grant_id is not None
            key, envelope = ring.encrypt(
                SecretStr(json.dumps(credentials)),
                owner_id="a",
                connection_id=item.connection_id,
                grant_id=item.grant_id,
            )
            async with pool.acquire() as conn:
                await conn.execute(
                    "UPDATE dlightrag_connection_grants SET kind='oauth',consented_scopes='[\"read\"]',key_id=$1,encrypted_envelope=$2",
                    key,
                    envelope,
                )

        await set_credentials()
        entered, release, closed, changed = (asyncio.Event() for _ in range(4))
        checked = asyncio.Queue[bool]()
        effects, refreshes = [], []

        class ObservedStore(PGConnectionsStore):
            async def dispatch_alive(self, **kwargs):
                # Observe the real PG watch query after the mutation barrier;
                # never substitute a fake authorization result or rely on sleep.
                after_change = changed.is_set()
                alive = await super().dispatch_alive(**kwargs)
                if after_change:
                    checked.put_nowait(alive)
                return alive

        async def remote(request):
            async with pool.acquire() as conn:
                assert (
                    await conn.fetchval(
                        "SELECT count(*) FROM pg_stat_activity WHERE datname=current_database() AND state='idle in transaction'"
                    )
                    == 0
                )
            if request.headers["host"] == "as.example":
                refreshes.append(parse_qs(request.content.decode())["refresh_token"][0])
                assert refreshes == ["refresh-secret"]
                return httpx2.Response(
                    200,
                    json={
                        "access_token": "fresh-token",
                        "refresh_token": "rotated-refresh",
                        "token_type": "Bearer",
                        "expires_in": 300,
                        "scope": "read",
                    },
                )
            assert request.headers["host"] == "example.com"
            body = json.loads(request.content)
            if "id" not in body:
                return httpx2.Response(202)
            if body["method"] == "initialize":
                result = {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "fake", "version": "1"},
                }
            else:
                assert body["method"] == "tools/call"
                assert body["params"]["arguments"] == {"path": "x"}
                effects.append(request.headers["authorization"])
                if len(effects) == 1:
                    entered.set()
                    try:
                        await release.wait()
                    finally:
                        closed.set()
                result = {"content": [{"type": "text", "text": "written"}], "isError": False}
            return httpx2.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})

        mcp = PersonalMcpClient(transport_factory=lambda: httpx2.MockTransport(remote))
        a = Connections(store=ObservedStore(pool=pool), mcp=mcp, cipher=ring)
        b = Connections(
            store=PGConnectionsStore(pool=pool),
            mcp=fake_mcp,
            cipher=ring,
            oauth=PersonalOAuthClient(transport_factory=lambda: httpx2.MockTransport(remote)),
        )
        (tool,) = await a.restore_research(bindings=bound.bindings, claim=claim)
        task = asyncio.ensure_future(
            tool.execute(tool.input_model.model_validate({"path": "x"}), runtime)
        )
        try:
            await asyncio.wait_for(entered.wait(), 2)
            if mutation == "refresh":
                # Expire the credential while the gate-first HTTP effect is held.
                credentials["expires_at"] = 1
                await set_credentials()
                second = await dispatch_fixture(
                    runs,
                    pool,
                    connection=(b, store, fake_mcp, view),
                    key="second",
                    worker="second-worker",
                )
                worker = Connections(
                    store=PGConnectionsStore(pool=pool),
                    mcp=mcp,
                    cipher=ring,
                    oauth=PersonalOAuthClient(
                        transport_factory=lambda: httpx2.MockTransport(remote)
                    ),
                )
                (second_tool,) = await worker.restore_research(
                    bindings=second[4].bindings, claim=second[5]
                )
                result = await second_tool.execute(
                    second_tool.input_model.model_validate({"path": "x"}), second[7]
                )
                assert not result.is_error
                _, (updated,) = await store.read("a")
                assert (
                    updated.envelope is not None
                    and updated.secret_version == item.secret_version + 1
                )
                assert (
                    json.loads(
                        ring.decrypt(
                            updated.envelope,
                            owner_id="a",
                            connection_id=item.connection_id,
                            grant_id=item.grant_id,
                        ).get_secret_value()
                    )["tokens"]["refresh_token"]
                    == "rotated-refresh"
                )
            elif mutation == "rotation":
                keys["active"] = "next"
                rotating = Connections(
                    store=PGConnectionsStore(pool=pool),
                    mcp=fake_mcp,
                    cipher=CredentialCipher(SecretStr(json.dumps(keys))),
                )
                assert (await rotating.maintain())["reencrypted"] == 1
            elif mutation == "replacement":
                await b.replace_bearer(
                    owner_id="a",
                    auth_mode="jwt",
                    connection_id=item.connection_id,
                    expected_revision=view.revision,
                    bearer=SecretStr("replacement-token"),
                )
            elif mutation == "cancel":
                async with pool.acquire() as conn:
                    await conn.execute(
                        "UPDATE dlightrag_answer_child_sessions SET cancel_requested_at=now()"
                        if child
                        else "UPDATE dlightrag_runs SET cancel_requested_at=now()"
                    )
            else:
                kinds: tuple[Literal["revoke", "disable", "enable"], ...] = (
                    ("revoke",) if mutation == "revoke" else ("disable", "enable")
                )
                for kind in kinds:
                    view = await b.change(
                        owner_id="a",
                        auth_mode="jwt",
                        expected_revision=view.revision,
                        command=ConnectionCommand(
                            kind=kind,
                            connection_id=item.connection_id,
                            consent_version=1 if kind == "enable" else None,
                        ),
                    )
            changed.set()
            if mutation in {"refresh", "rotation"}:
                assert await asyncio.wait_for(checked.get(), 2)
                assert not task.done() and not closed.is_set()
                release.set()
                result = await asyncio.wait_for(task, 2)
                assert not result.is_error and result.text_content == "written"
            else:
                result = await asyncio.wait_for(task, 2)
                assert result.is_error and "unknown" in result.text_content
                assert closed.is_set()
                (restored,) = await a.restore_research(bindings=bound.bindings, claim=claim)
                denied = await restored.execute(
                    restored.input_model.model_validate({"path": "x"}), runtime
                )
                assert denied.is_error and "No call was sent" in denied.text_content
            assert effects == (
                ["Bearer initial-token", "Bearer fresh-token"]
                if mutation == "refresh"
                else ["Bearer initial-token"]
            )
            assert refreshes == (["refresh-secret"] if mutation == "refresh" else [])
        finally:
            release.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await a.aclose()
            await b.aclose()


@pytest.mark.asyncio
async def test_reader_validates_without_writer_gc_and_writer_shutdown_is_restartable():
    async with isolated_run_runtime("connection_lifecycle") as (_, pool):
        service, store, mcp, view = await enabled_connection(pool)
        await service.change(
            owner_id="a",
            auth_mode="jwt",
            expected_revision=view.revision,
            command=ConnectionCommand(
                kind="disable", connection_id=view.connections[0].connection_id
            ),
        )
        reader = Connections(store=PGConnectionsStore(pool=pool), mcp=mcp)
        await reader.start(validate_only=True)
        await asyncio.sleep(0.1)
        async with pool.acquire() as conn:
            assert await conn.fetchval("SELECT count(*) FROM dlightrag_connection_generations") == 2
        await reader.aclose()
        for _ in range(2):
            await service.start()
            count = 0
            for _ in range(100):
                async with pool.acquire() as conn:
                    count = await conn.fetchval(
                        "SELECT count(*) FROM dlightrag_connection_generations"
                    )
                if count == 1:
                    break
                await asyncio.sleep(0.01)
            assert count == 1
            await service.stop_refresh()
            await service.aclose()
        await store.initialize(validate_only=True)


@pytest.mark.asyncio
async def test_notification_reconnect_wakes_a_scan_after_listener_connection_loss():
    async with isolated_run_runtime("connection_notify") as (_, pool):
        store = PGConnectionsStore(pool=pool)
        await store.initialize(validate_only=False)
        await store.start_notifications()
        try:
            await store.wait_refresh(2)
            async with pool.acquire() as conn:
                pid = None
                for _ in range(100):
                    pid = await conn.fetchval(
                        "SELECT pid FROM pg_stat_activity WHERE datname=current_database() AND pid<>pg_backend_pid() AND query ILIKE 'LISTEN%' LIMIT 1"
                    )
                    if pid is not None:
                        break
                    await asyncio.sleep(0.01)
                assert pid is not None
                assert await conn.fetchval("SELECT pg_terminate_backend($1)", pid)
            # Reconnection's startup wake is observable without a new NOTIFY.
            started = asyncio.get_running_loop().time()
            await store.wait_refresh(5)
            assert asyncio.get_running_loop().time() - started < 4
        finally:
            await store.stop_notifications()


@pytest.mark.asyncio
async def test_owner_authorization_quota_is_durable_across_workers():
    from dlightrag.application.connections import ConnectionsError
    from dlightrag.application.connections.models import OAuthFlow
    from tests.integration.test_connections_pg import FakeMcp

    async with isolated_run_runtime("oauth_quota") as (_, pool):
        a, b = PGConnectionsStore(pool=pool), PGConnectionsStore(pool=pool)
        await a.initialize(validate_only=False)
        service = Connections(store=a, mcp=FakeMcp())
        view = await service.read(owner_id="a", auth_mode="jwt")
        revision = view.revision
        for number in range(5):
            view = await service.change(
                owner_id="a",
                auth_mode="jwt",
                expected_revision=revision,
                command=ConnectionCommand(
                    kind="create", label=str(number), endpoint="https://example.com/mcp"
                ),
            )
            revision = view.revision
        for number, item in enumerate(view.connections):
            flow = OAuthFlow(
                str(number), "a", item.connection_id, "worker-a", item.endpoint, revision
            )
            if number < 4:
                await a.create_oauth_flow(flow=flow, lifetime=30, lease=10)
            else:
                with pytest.raises(ConnectionsError, match="quota"):
                    await b.create_oauth_flow(flow=flow, lifetime=30, lease=10)
        async with pool.acquire() as conn:
            assert await conn.fetchval("SELECT count(*) FROM dlightrag_connection_oauth_flows") == 4
