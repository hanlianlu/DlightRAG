# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Personal Research bindings at the owner and actual PostgreSQL acceptance seams."""

from typing import Any

import pytest

from dlightrag.adapters.postgres.connections import PGConnectionsStore
from dlightrag.application.connections import ConnectionCommand, Connections
from tests.integration.run_runtime_pg_harness import isolated_run_runtime
from tests.integration.test_connections_pg import stored_catalogue


class CatalogueMcp:
    def __init__(self):
        from unittest.mock import AsyncMock

        self.call = AsyncMock(side_effect=AssertionError("Binding must not dispatch"))

    calls = 0
    description = "Accepted description"

    async def discover(self, **kwargs):
        self.calls += 1
        return [
            {
                "name": "read",
                "description": self.description,
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
            }
        ]


async def enabled_connection(pool, owner="a"):
    store = PGConnectionsStore(pool=pool)
    await store.initialize(validate_only=False)
    mcp = CatalogueMcp()
    service = Connections(store=store, mcp=mcp)
    view = await service.change(
        owner_id=owner,
        auth_mode="jwt",
        expected_revision="0",
        command=ConnectionCommand(
            kind="create", label="Fixture", endpoint="https://example.com/mcp"
        ),
    )
    identity = view.connections[0].connection_id
    for command in (
        ConnectionCommand(kind="probe", connection_id=identity),
        ConnectionCommand(kind="enable", connection_id=identity, consent_version=1),
    ):
        view = await service.change(
            owner_id=owner, auth_mode="jwt", expected_revision=view.revision, command=command
        )
    return service, store, mcp, view


@pytest.mark.asyncio
async def test_bind_research_is_owner_scoped_schema_exact_and_network_free():
    async with isolated_run_runtime("binding") as (_, pool):
        service, store, mcp, view = await enabled_connection(pool)
        bound = await service.bind_research(owner_id="a", auth_mode="jwt")
        assert mcp.calls == 1
        assert len(bound.bindings) == len(bound.tools) == 1
        assert bound.bindings[0].generation == view.connections[0].generation
        assert (
            bound.tools[0].definition.parameters == (await stored_catalogue(store))[0].input_schema
        )
        assert bound.tools[0].description == "Accepted description"
        assert bound.tools[0].replay_policy == "never"
        assert (await service.bind_research(owner_id="b", auth_mode="jwt")).tools == ()
        assert (await service.bind_research(owner_id="a", auth_mode="simple")).tools == ()
        assert (
            await service.bind_research(owner_id="a", auth_mode="none")
        ).bindings == bound.bindings
        assert mcp.calls == 1


@pytest.mark.asyncio
async def test_actual_accept_run_pins_restore_old_generation_and_prevent_gc():
    from dataclasses import replace
    from unittest.mock import AsyncMock
    from uuid import uuid7

    import asyncpg

    from dlightrag.engine.agent.session.ids import IntentId
    from dlightrag.engine.agent.tools import ToolRuntime
    from dlightrag.engine.answer.execution.connection_binding import (
        ResearchToolClaim,
        StaleConnectionBindingError,
    )
    from tests.integration.run_runtime_pg_harness import run_envelope

    async with isolated_run_runtime("binding_accept") as (runs, pool):
        service, store, mcp, view = await enabled_connection(pool)
        bound = await service.bind_research(owner_id="a", auth_mode="jwt")
        envelope = run_envelope("answer", key="r1", owner="a", mode="research")
        envelope = replace(
            envelope,
            payload={
                **envelope.payload,
                "run_connection_bindings": [b.as_json() for b in bound.bindings],
            },
        )
        r1 = await runs.accept_run(
            envelope=envelope, run_id=str(uuid7()), connection_bindings=bound.bindings
        )
        mcp.description = "Future description"
        await service.change(
            owner_id="a",
            auth_mode="jwt",
            expected_revision=view.revision,
            command=ConnectionCommand(
                kind="probe", connection_id=view.connections[0].connection_id
            ),
        )
        future = await service.bind_research(owner_id="a", auth_mode="jwt")
        assert future.tools[0].name == bound.tools[0].name
        assert future.tools[0].description == "Future description"
        assert future.bindings[0].generation > bound.bindings[0].generation
        r2_envelope = replace(
            envelope,
            submission_key="r2",
            payload={
                **envelope.payload,
                "run_connection_bindings": [b.as_json() for b in future.bindings],
            },
        )
        r2 = await runs.create_run(
            envelope=r2_envelope, run_id=str(uuid7()), connection_bindings=future.bindings
        )
        assert r2.run.run_id != r1.run.run_id
        replay = await runs.accept_run(
            envelope=envelope, run_id=str(uuid7()), connection_bindings=future.bindings
        )
        assert replay.replayed and replay.run.run_id == r1.run.run_id
        with pytest.raises(StaleConnectionBindingError):
            await runs.accept_run(
                envelope=replace(envelope, submission_key="stale"),
                run_id=str(uuid7()),
                connection_bindings=bound.bindings,
            )
        async with pool.acquire() as conn:
            assert await conn.fetchval("SELECT count(*) FROM dlightrag_answer_connection_pins") == 2
            assert (
                await conn.fetchval(
                    "SELECT generation FROM dlightrag_answer_connection_pins WHERE run_id=$1",
                    r1.run.run_id,
                )
                == bound.bindings[0].generation
            )
            assert (
                await conn.fetchval(
                    "SELECT generation FROM dlightrag_answer_connection_pins WHERE run_id=$1",
                    r2.run.run_id,
                )
                == future.bindings[0].generation
            )
            assert await conn.fetchval("SELECT count(*) FROM dlightrag_runs") == 2
            with pytest.raises(asyncpg.ForeignKeyViolationError):
                await conn.execute(
                    "DELETE FROM dlightrag_connection_generations WHERE owner_id='a' AND generation=$1",
                    bound.bindings[0].generation,
                )
        restarted = Connections(store=PGConnectionsStore(pool=pool), mcp=mcp)
        cancelled = AsyncMock()
        claim = ResearchToolClaim("a", r1.run.run_id, "worker", 7, cancelled)
        restored = await restarted.restore_research(bindings=bound.bindings, claim=claim)
        assert restored[0].definition == bound.tools[0].definition
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="accepted tool schema"):
            restored[0].input_model.model_validate({"owner_id": "b"})
        assert (
            await restored[0].execute(
                restored[0].input_model.model_validate({"path": "x"}),
                ToolRuntime("call", restored[0].name, IntentId.new(), "scope", AsyncMock()),
            )
        ).is_error
        assert mcp.calls == 2
        with pytest.raises(ValueError):
            await restarted.restore_research(
                bindings=bound.bindings, claim=replace(claim, owner_id="b")
            )
        await store.initialize(validate_only=True)


@pytest.mark.asyncio
async def test_web_turn_run_and_pins_share_actual_transaction_and_cascade():
    from dataclasses import replace
    from uuid import uuid7

    import asyncpg

    from dlightrag.adapters.postgres.web.web_conversations import PGWebConversationStore
    from tests.integration.run_runtime_pg_harness import run_envelope

    async with isolated_run_runtime("binding_web") as (runs, pool):
        service, _, _, _ = await enabled_connection(pool)
        bound = await service.bind_research(owner_id="a", auth_mode="jwt")
        web = PGWebConversationStore(pool=pool, run_store=runs)
        await web.initialize()
        envelope = run_envelope("answer", key=str(uuid7()), owner="a", mode="research")
        envelope = replace(
            envelope,
            payload={
                **envelope.payload,
                "run_connection_bindings": [b.as_json() for b in bound.bindings],
            },
        )
        conversation_id = str(uuid7())
        kwargs: dict[str, Any] = dict(
            principal_id="a",
            conversation_id=conversation_id,
            submission_id=envelope.submission_key,
            envelope=envelope,
            run_id=str(uuid7()),
            title_hint="Atomic",
            create_conversation=True,
            connection_bindings=bound.bindings,
        )
        async with pool.acquire() as conn:
            await conn.execute("""CREATE FUNCTION fail_pin() RETURNS trigger LANGUAGE plpgsql AS $$ BEGIN RAISE EXCEPTION 'pin failure'; END $$;
                CREATE TRIGGER fail_pin BEFORE INSERT ON dlightrag_answer_connection_pins FOR EACH ROW EXECUTE FUNCTION fail_pin();""")
        with pytest.raises(asyncpg.RaiseError, match="pin failure"):
            await web.create_answer_turn(**kwargs)
        async with pool.acquire() as conn:
            assert await conn.fetchval("SELECT count(*) FROM dlightrag_runs") == 0
            assert await conn.fetchval("SELECT count(*) FROM web_conversations") == 0
            assert await conn.fetchval("SELECT count(*) FROM dlightrag_answer_connection_pins") == 0
            await conn.execute("DROP TRIGGER fail_pin ON dlightrag_answer_connection_pins")
        result = await web.create_answer_turn(**kwargs)
        assert result is not None and not result.replayed
        replay = await web.create_answer_turn(**kwargs)
        assert (
            replay is not None
            and replay.replayed
            and replay.turn.run.run_id == result.turn.run.run_id
        )
        async with pool.acquire() as conn:
            assert await conn.fetchval("SELECT count(*) FROM dlightrag_answer_connection_pins") == 1
            assert await conn.fetchval("SELECT count(*) FROM web_conversations") == 1
            await conn.execute("DELETE FROM dlightrag_runs WHERE owner_id='a'")
            assert await conn.fetchval("SELECT count(*) FROM dlightrag_answer_connection_pins") == 0


@pytest.mark.asyncio
async def test_snapshot_publication_gc_accept_race_cannot_leave_dangling_pin():
    import asyncio
    from dataclasses import replace
    from uuid import uuid7

    from dlightrag.engine.answer.execution.connection_binding import StaleConnectionBindingError
    from tests.integration.run_runtime_pg_harness import run_envelope

    async with isolated_run_runtime("binding_gc") as (runs, pool):
        service, _, _, view = await enabled_connection(pool)
        bound = await service.bind_research(owner_id="a", auth_mode="jwt")
        old = bound.bindings[0]
        await service.change(
            owner_id="a",
            auth_mode="jwt",
            expected_revision=view.revision,
            command=ConnectionCommand(kind="probe", connection_id=old.connection_id),
        )
        envelope = run_envelope("answer", key="race", owner="a", mode="research")
        envelope = replace(
            envelope, payload={**envelope.payload, "run_connection_bindings": [old.as_json()]}
        )
        async with pool.acquire() as gc:
            async with gc.transaction():
                await gc.execute(
                    "SELECT 1 FROM dlightrag_connection_generations WHERE owner_id='a' AND generation=$1 FOR UPDATE",
                    old.generation,
                )
                await gc.execute(
                    "DELETE FROM dlightrag_connection_generations WHERE owner_id='a' AND generation=$1",
                    old.generation,
                )
                accept = asyncio.create_task(
                    runs.accept_run(
                        envelope=envelope, run_id=str(uuid7()), connection_bindings=bound.bindings
                    )
                )
                # It must reject the stale head even while GC holds the old generation.
                with pytest.raises(StaleConnectionBindingError):
                    await asyncio.wait_for(accept, timeout=2)
        async with pool.acquire() as conn:
            assert await conn.fetchval("SELECT count(*) FROM dlightrag_runs") == 0
            assert await conn.fetchval("SELECT count(*) FROM dlightrag_answer_connection_pins") == 0
        fresh = await service.bind_research(owner_id="a", auth_mode="jwt")
        fresh_envelope = replace(
            envelope,
            payload={
                **envelope.payload,
                "run_connection_bindings": [b.as_json() for b in fresh.bindings],
            },
        )
        await runs.create_run(
            envelope=fresh_envelope, run_id=str(uuid7()), connection_bindings=fresh.bindings
        )
        async with pool.acquire() as conn:
            assert (
                await conn.fetchval("SELECT generation FROM dlightrag_answer_connection_pins")
                == fresh.bindings[0].generation
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["application", "rest", "mcp", "web"])
@pytest.mark.parametrize(
    "identity,mode,expected",
    [
        ("none", "research", True),
        ("alice", "research", True),
        ("bob", "research", False),
        ("simple", "research", False),
        ("alice", "fast", False),
    ],
)
async def test_same_owner_binding_across_application_rest_mcp_and_web(
    test_config, monkeypatch, surface, identity, mode, expected
):
    import json
    from unittest.mock import AsyncMock
    from uuid import uuid7

    from httpx import ASGITransport, AsyncClient
    from mcp.types import CallToolResult, TextContent

    from dlightrag.adapters.http.rest.auth import get_current_user
    from dlightrag.adapters.http.server import create_app
    from dlightrag.adapters.mcp import server as mcp_server
    from dlightrag.adapters.postgres.web.web_conversations import PGWebConversationStore
    from dlightrag.application.access import (
        RequestScope,
        UserContext,
        owner_id_from_user,
        request_scope_context,
    )
    from dlightrag.application.answer_runs import AnswerRequest
    from dlightrag.application.config import set_config
    from dlightrag.application.web_conversations import WebConversationService
    from dlightrag.engine.answer.execution.input import AnswerRunInput
    from tests.integration.test_answer_run_api_pg import (
        _ALICE,
        _ANON,
        _BOB,
        _StoreBackedApplication,
    )

    async with isolated_run_runtime("binding_surfaces") as (runs, pool):
        connection_owner = owner_id_from_user(_ANON if identity == "none" else _ALICE)
        connections, _, mcp, _ = await enabled_connection(pool, connection_owner)
        application = _StoreBackedApplication(
            runs, test_config, bind_research=connections.bind_research
        )
        user = (
            _ANON
            if identity == "none"
            else _ALICE
            if identity == "alice"
            else _BOB
            if identity == "bob"
            else UserContext(user_id="shared", auth_mode="simple")
        )
        owner = owner_id_from_user(user)
        set_config(test_config)
        if surface == "application":
            created = await application.answers.create(
                request=AnswerRequest(query="q", workspaces=("default",), mode=mode),
                owner_id=owner,
                auth_mode=user.auth_mode,
            )
            run_id = created.run.run_id
        elif surface == "rest":
            app = create_app(include_web_app=False)
            app.state.application = application
            app.dependency_overrides[get_current_user] = lambda: user
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.post("/answer", json={"query": "q", "mode": mode})
                assert response.status_code == 202, response.text
                run_id = response.json()["run_id"]
        elif surface == "mcp":
            monkeypatch.setattr(
                mcp_server, "_ensure_application", AsyncMock(return_value=application)
            )
            with request_scope_context(
                RequestScope(user_id=user.user_id, auth_mode=user.auth_mode, claims=user.claims)
            ):
                result = await mcp_server.mcp_app.call_tool("answer", {"query": "q", "mode": mode})
                assert isinstance(result, CallToolResult)
                assert isinstance(result.content[0], TextContent)
                assert not result.is_error, result
                run_id = json.loads(result.content[0].text)["run_id"]
        else:
            web = WebConversationService(
                store=PGWebConversationStore(pool=pool, run_store=runs),
                answers=application.answers,
                max_attachments=8,
                cursor_secret=b"test",
            )
            result = await web.start_answer(
                user,
                conversation_id=None,
                submission_id=str(uuid7()),
                query="q",
                workspaces=("default",),
                mode=mode,
            )
            assert result is not None
            run_id = result.run.run_id
        record = await runs.get_run(owner_id=owner, run_id=run_id)
        assert record is not None
        decoded = AnswerRunInput.from_prepared_input(record.prepared_input)
        binding = await connections.bind_research(owner_id=connection_owner, auth_mode="jwt")
        assert decoded.run_connection_bindings == (binding.bindings if expected else ())
        assert decoded.agent_run_plan is not None
        personal = [t for t in decoded.agent_run_plan.tools if t.name == binding.tools[0].name]
        assert len(personal) == int(expected)
        if personal:
            from dlightrag.engine.agent.session.plan import AgentToolPlan

            assert personal[0] == AgentToolPlan.from_tool(binding.tools[0])
        async with pool.acquire() as conn:
            assert await conn.fetchval(
                "SELECT count(*) FROM dlightrag_answer_connection_pins WHERE owner_id=$1 AND run_id=$2",
                owner,
                record.run_id,
            ) == int(expected)
        assert mcp.calls == 1


@pytest.mark.asyncio
async def test_acceptance_locks_head_before_publication_and_gc_retains_pin():
    import asyncio
    from dataclasses import replace
    from uuid import uuid7

    import asyncpg

    from tests.integration.run_runtime_pg_harness import run_envelope

    async with isolated_run_runtime("binding_lock") as (runs, pool):
        service, _, _, view = await enabled_connection(pool)
        bound = await service.bind_research(owner_id="a", auth_mode="jwt")
        old = bound.bindings[0]
        envelope = run_envelope("answer", key="locked", owner="a", mode="research")
        envelope = replace(
            envelope, payload={**envelope.payload, "run_connection_bindings": [old.as_json()]}
        )
        async with pool.acquire() as accepting, pool.acquire() as observer:
            publication = None
            try:
                async with accepting.transaction():
                    await runs.create_run_in(
                        accepting,
                        envelope=envelope,
                        run_id=str(uuid7()),
                        connection_bindings=bound.bindings,
                    )
                    publication = asyncio.create_task(
                        service.change(
                            owner_id="a",
                            auth_mode="jwt",
                            expected_revision=view.revision,
                            command=ConnectionCommand(
                                kind="probe", connection_id=old.connection_id
                            ),
                        )
                    )
                    async with asyncio.timeout(3):
                        while not await observer.fetchval("""SELECT EXISTS(SELECT 1 FROM pg_stat_activity
                            WHERE datname=current_database() AND pid<>pg_backend_pid()
                            AND wait_event_type='Lock' AND query LIKE '%dlightrag_connection_heads%')"""):
                            await asyncio.sleep(0.01)
                    # None of the Run or pins is visible before the accepting commit.
                    assert await observer.fetchval("SELECT count(*) FROM dlightrag_runs") == 0
                    assert (
                        await observer.fetchval(
                            "SELECT count(*) FROM dlightrag_answer_connection_pins"
                        )
                        == 0
                    )
                    assert not publication.done()
                await asyncio.wait_for(publication, 3)
            finally:
                if publication is not None and not publication.done():
                    publication.cancel()
                    await asyncio.gather(publication, return_exceptions=True)
            with pytest.raises(asyncpg.ForeignKeyViolationError):
                await observer.execute(
                    "DELETE FROM dlightrag_connection_generations WHERE owner_id='a' AND generation=$1",
                    old.generation,
                )
            assert (
                await observer.fetchval("SELECT generation FROM dlightrag_answer_connection_pins")
                == old.generation
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["disable", "revoke", "delete"])
async def test_disable_reenable_or_retirement_cannot_accept_old_activation(mutation):
    from dataclasses import replace
    from uuid import uuid7

    from dlightrag.engine.answer.execution.connection_binding import StaleConnectionBindingError
    from tests.integration.run_runtime_pg_harness import run_envelope

    async with isolated_run_runtime("binding_epoch") as (runs, pool):
        service, _, _, view = await enabled_connection(pool)
        old = await service.bind_research(owner_id="a", auth_mode="jwt")
        current = await service.change(
            owner_id="a",
            auth_mode="jwt",
            expected_revision=view.revision,
            command=ConnectionCommand(kind=mutation, connection_id=old.bindings[0].connection_id),
        )
        if mutation == "disable":
            await service.change(
                owner_id="a",
                auth_mode="jwt",
                expected_revision=current.revision,
                command=ConnectionCommand(
                    kind="enable", connection_id=old.bindings[0].connection_id, consent_version=1
                ),
            )
        envelope = run_envelope("answer", key="stale_epoch", owner="a", mode="research")
        envelope = replace(
            envelope,
            payload={
                **envelope.payload,
                "run_connection_bindings": [b.as_json() for b in old.bindings],
            },
        )
        with pytest.raises(StaleConnectionBindingError):
            await runs.accept_run(
                envelope=envelope, run_id=str(uuid7()), connection_bindings=old.bindings
            )
        async with pool.acquire() as conn:
            assert await conn.fetchval("SELECT count(*) FROM dlightrag_runs") == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["accept_run", "create_run", "create_run_in"])
@pytest.mark.parametrize("invalid", ["owner", "digest", "omitted_pins", "fast", "simple"])
async def test_every_accepting_path_validates_normalized_binding_before_commit(path, invalid):
    from dataclasses import replace
    from uuid import uuid7

    from dlightrag.engine.answer.execution.connection_binding import StaleConnectionBindingError
    from tests.integration.run_runtime_pg_harness import run_envelope

    async with isolated_run_runtime("binding_invalid") as (runs, pool):
        service, _, _, _ = await enabled_connection(pool)
        bound = await service.bind_research(owner_id="a", auth_mode="jwt")
        bindings = bound.bindings
        if invalid == "owner":
            bindings = (replace(bindings[0], owner_id="b"),)
        elif invalid == "digest":
            bindings = (replace(bindings[0], catalogue_digest="0" * 64),)
        envelope = run_envelope(
            "answer", key="invalid", owner="a", mode="fast" if invalid == "fast" else "research"
        )
        envelope = replace(
            envelope,
            payload={
                **envelope.payload,
                "auth_mode": "simple" if invalid == "simple" else "jwt",
                "run_connection_bindings": [b.as_json() for b in bindings],
            },
        )
        if invalid == "omitted_pins":
            bindings = ()
        with pytest.raises((ValueError, StaleConnectionBindingError)):
            if path == "create_run_in":
                async with pool.acquire() as conn, conn.transaction():
                    await runs.create_run_in(
                        conn, envelope=envelope, run_id=str(uuid7()), connection_bindings=bindings
                    )
            else:
                await getattr(runs, path)(
                    envelope=envelope, run_id=str(uuid7()), connection_bindings=bindings
                )
        async with pool.acquire() as conn:
            assert await conn.fetchval("SELECT count(*) FROM dlightrag_runs") == 0
            assert await conn.fetchval("SELECT count(*) FROM dlightrag_answer_connection_pins") == 0
