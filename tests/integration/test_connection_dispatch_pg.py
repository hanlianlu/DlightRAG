# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Owner Connections foreground effects against real PostgreSQL authority rows."""

import json
import uuid
from dataclasses import replace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from dlightrag.engine.agent.session.ids import IntentId
from dlightrag.engine.agent.tools import ToolResult, ToolRuntime
from dlightrag.engine.answer.execution.connection_binding import ResearchToolClaim
from tests.integration.run_runtime_pg_harness import isolated_run_runtime, run_envelope
from tests.integration.test_connection_binding_pg import enabled_connection


async def dispatch_fixture(
    runs, pool, *, seed=True, authenticated=False, connection=None, key="effect", worker="worker"
):
    service, store, mcp, view = connection or await enabled_connection(pool)
    if authenticated:
        from pydantic import SecretStr

        from dlightrag.application.connections import ConnectionCommand, Connections
        from dlightrag.application.connections.credentials import CredentialCipher
        from tests.unit.test_connections_config import KEYRING

        service = Connections(store=store, mcp=mcp, cipher=CredentialCipher(SecretStr(KEYRING)))
        view = await service.replace_bearer(
            owner_id="a",
            auth_mode="jwt",
            connection_id=view.connections[0].connection_id,
            expected_revision=view.revision,
            bearer=SecretStr("dispatch-test-only-token"),
        )
        view = await service.change(
            owner_id="a",
            auth_mode="jwt",
            expected_revision=view.revision,
            command=ConnectionCommand(
                kind="enable", connection_id=view.connections[0].connection_id, consent_version=1
            ),
        )
    mcp.call = AsyncMock(return_value=ToolResult.text("written"))
    bound = await service.bind_research(owner_id="a", auth_mode="jwt")
    envelope = run_envelope("answer", key=key, owner="a", mode="research")
    envelope = replace(
        envelope,
        payload={
            **envelope.payload,
            "run_connection_bindings": [b.as_json() for b in bound.bindings],
        },
    )
    await runs.accept_run(
        envelope=envelope, run_id=str(uuid.uuid7()), connection_bindings=bound.bindings
    )
    claimed = await runs.claim_next(worker_id=worker)
    assert claimed is not None
    run = claimed.run
    claim = ResearchToolClaim("a", run.run_id, worker, run.fencing_epoch, AsyncMock())
    (tool,) = await service.restore_research(bindings=bound.bindings, claim=claim)
    runtime = ToolRuntime(
        call_id="call",
        tool_name=tool.name,
        intent_id=IntentId.new(),
        execution_scope=str(envelope.payload["agent_session_id"]),
        fencing_epoch=run.fencing_epoch,
        _update_sink=AsyncMock(),
    )
    from dlightrag.engine.agent.session.effects import canonical_json, schema_digest
    from dlightrag.engine.agent.session.ids import AttemptId, EntryId, OperationId
    from dlightrag.engine.agent.session.operation import (
        ToolBatchItem,
        ToolBatchPlan,
        ToolEffectPending,
    )
    from dlightrag.engine.agent.session.registers import OperationStateRegister, ToolArguments

    item = ToolBatchItem(
        0,
        "call",
        tool.name,
        "executable",
        EntryId.new(),
        runtime.intent_id,
        "never",
        tool.contract_version,
        tool.input_schema_digest,
        schema_digest({"path": "x"}),
    )
    state = ToolEffectPending(
        OperationId.new(), 1, ToolBatchPlan(EntryId.new(), (item,)), 0, AttemptId.new()
    )
    if not seed:
        return service, store, mcp, view, bound, claim, tool, runtime
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO dlightrag_agent_sessions(owner_id,session_id,lease_run_id,fencing_epoch) VALUES('a',$1,$2,$3)",
            uuid.UUID(runtime.execution_scope),
            uuid.UUID(run.run_id),
            run.fencing_epoch,
        )
        for reg in (
            OperationStateRegister(state),
            ToolArguments(runtime.intent_id, canonical_json({"path": "x"})),
        ):
            await conn.execute(
                "INSERT INTO dlightrag_agent_session_registers VALUES('a',$1,$2,$3,1,$4::jsonb)",
                uuid.UUID(runtime.execution_scope),
                reg.ref.kind,
                reg.ref.key,
                json.dumps(reg.canonical_payload()),
            )
    return service, store, mcp, view, bound, claim, tool, runtime


@pytest.mark.asyncio
async def test_committed_pending_owner_binding_dispatches_once():
    async with isolated_run_runtime("dispatch") as (runs, pool):
        service, store, mcp, view, bound, claim, tool, runtime = await dispatch_fixture(runs, pool)
        result = await tool.execute(tool.input_model.model_validate({"path": "x"}), runtime)
        assert result.text_content == "written"
        assert not result.is_error
        mcp.call.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "denial",
    [
        "revoke",
        "disable_enable",
        "owner",
        "parent_fence",
        "parent_expired",
        "parent_cancel",
        "session_fence",
        "no_pending",
        "settled",
        "arguments",
        "call_id",
        "scope",
        "pin",
    ],
)
async def test_dispatch_authority_denials_send_zero_calls(denial):
    from dlightrag.application.connections import ConnectionCommand

    async with isolated_run_runtime("dispatch_deny") as (runs, pool):
        service, store, mcp, view, bound, claim, tool, runtime = await dispatch_fixture(runs, pool)
        if denial in {"revoke", "disable_enable"}:
            kinds: list[Any] = ["revoke"] if denial == "revoke" else ["disable", "enable"]
            for kind in kinds:
                view = await service.change(
                    owner_id="a",
                    auth_mode="jwt",
                    expected_revision=view.revision,
                    command=ConnectionCommand(
                        kind=kind,
                        connection_id=view.connections[0].connection_id,
                        consent_version=1 if kind == "enable" else None,
                    ),
                )
        async with pool.acquire() as conn:
            if denial.startswith("parent_"):
                change = {
                    "parent_fence": "fencing_epoch=fencing_epoch+1",
                    "parent_expired": "lease_expires_at=now()-interval '1 second'",
                    "parent_cancel": "cancel_requested_at=now()",
                }[denial]
                await conn.execute("UPDATE dlightrag_runs SET " + change)
            elif denial == "session_fence":
                runtime = replace(runtime, fencing_epoch=runtime.fencing_epoch + 1)
            elif denial in {"no_pending", "settled"}:
                await conn.execute(
                    "DELETE FROM dlightrag_agent_session_registers WHERE register_kind='operation_state'"
                )
            elif denial == "pin":
                await conn.execute("DELETE FROM dlightrag_answer_connection_pins")
        if denial == "owner":
            # Trusted claim is not replaced with any argument-supplied owner.
            with pytest.raises(ValueError):
                await service.restore_research(
                    bindings=bound.bindings, claim=replace(claim, owner_id="b")
                )
            assert mcp.call.await_count == 0
            return
        if denial == "scope":
            runtime = replace(runtime, execution_scope=str(uuid.uuid7()))
        if denial == "call_id":
            runtime = replace(runtime, call_id="another")
        result = await tool.execute(
            tool.input_model.model_validate({"path": "other" if denial == "arguments" else "x"}),
            runtime,
        )
        assert result.is_error and "No call was sent" in result.text_content
        mcp.call.assert_not_awaited()


@pytest.mark.asyncio
async def test_gate_first_revoke_is_in_flight_cancelled_unknown_without_network_transaction():
    import asyncio

    from dlightrag.application.connections import ConnectionCommand

    async with isolated_run_runtime("dispatch_race") as (runs, pool):
        service, store, mcp, view, bound, claim, tool, runtime = await dispatch_fixture(runs, pool)
        entered = asyncio.Event()
        closed = asyncio.Event()

        async def call(**kwargs):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                closed.set()

        mcp.call.side_effect = call
        await store.start_notifications()
        task = asyncio.ensure_future(
            tool.execute(tool.input_model.model_validate({"path": "x"}), runtime)
        )
        try:
            await asyncio.wait_for(entered.wait(), 2)
            # Completes while network is parked: no gate tx/row lock survives I/O.
            await asyncio.wait_for(
                service.change(
                    owner_id="a",
                    auth_mode="jwt",
                    expected_revision=view.revision,
                    command=ConnectionCommand(
                        kind="revoke", connection_id=view.connections[0].connection_id
                    ),
                ),
                2,
            )
            result = await asyncio.wait_for(task, 2)
            assert result.is_error and "unknown" in result.text_content
            assert closed.is_set()
            mcp.call.assert_awaited_once()
            result = await tool.execute(tool.input_model.model_validate({"path": "x"}), runtime)
            assert result.is_error
            mcp.call.assert_awaited_once()
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await store.stop_notifications()
            await service.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation", [None, "fence", "expired", "cancel", "parent_cancel", "worker"]
)
async def test_interactive_child_requires_own_live_fence_and_parent(mutation):
    async with isolated_run_runtime("dispatch_child") as (runs, pool):
        service, store, mcp, view, bound, claim, tool, runtime = await dispatch_fixture(runs, pool)
        parent = uuid.UUID(runtime.execution_scope)
        child = uuid.uuid7()
        async with pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO dlightrag_answer_child_sessions
                (owner_id,run_id,child_session_id,parent_session_id,parent_call_id,status,depth,context_snapshot_json,
                 lease_owner,lease_expires_at,fencing_epoch)
                VALUES('a',$1,$2,$3,'spawn','running',1,'{}','worker',now()+interval '1 minute',7)""",
                uuid.UUID(claim.run_id),
                child,
                parent,
            )
            # Seed existing typed pending registers for the child effect, not new authority.
            await conn.execute(
                "INSERT INTO dlightrag_agent_sessions(owner_id,session_id,lease_run_id,fencing_epoch) VALUES('a',$1,$2,7)",
                child,
                uuid.UUID(claim.run_id),
            )
            await conn.execute(
                """INSERT INTO dlightrag_agent_session_registers SELECT owner_id,$1,register_kind,register_key,sequence,payload_json
                FROM dlightrag_agent_session_registers WHERE session_id=$2""",
                child,
                parent,
            )
            if mutation == "parent_cancel":
                await conn.execute("UPDATE dlightrag_runs SET cancel_requested_at=now()")
            elif mutation in {"expired", "cancel", "worker"}:
                clause = {
                    "expired": "lease_expires_at=now()-interval '1 second'",
                    "cancel": "cancel_requested_at=now()",
                    "worker": "lease_owner='other'",
                }[mutation]
                await conn.execute("UPDATE dlightrag_answer_child_sessions SET " + clause)
        runtime = replace(
            runtime, execution_scope=str(child), fencing_epoch=6 if mutation == "fence" else 7
        )
        result = await tool.execute(tool.input_model.model_validate({"path": "x"}), runtime)
        assert result.is_error == (mutation is not None)
        assert mcp.call.await_count == (0 if mutation else 1)


@pytest.mark.asyncio
async def test_one_broken_connection_reports_degraded_without_disabling_other_tools():
    from dlightrag.application.connections import ConnectionsError

    async with isolated_run_runtime("dispatch_fault") as (runs, pool):
        service, store, mcp, view, bound, claim, tool, runtime = await dispatch_fixture(runs, pool)
        mcp.call.side_effect = ConnectionsError("untrusted secret diagnostic", 401)
        result = await tool.execute(tool.input_model.model_validate({"path": "x"}), runtime)
        assert result.is_error and "secret diagnostic" not in result.text_content
        assert "final Answer" in result.text_content and "Do not retry" in result.text_content
        observed = await service.read(owner_id="a", auth_mode="jwt")
        assert observed.connections[0].status == "needs-auth"
        assert observed.connections[0].enabled
        assert observed.connections[0].generation == bound.bindings[0].generation


@pytest.mark.asyncio
@pytest.mark.parametrize("crash_point", ["pending", "dispatched"])
async def test_real_pg_runtime_restart_settles_unknown_never_redispatches(crash_point):
    import asyncio

    from dlightrag.adapters.postgres.answer.session_repository import PGAgentSessionRepository
    from dlightrag.application.connections import Connections
    from dlightrag.engine.agent.session.entries import ToolResultMessageEntry
    from dlightrag.engine.agent.session.ids import LaneId, SessionId
    from dlightrag.engine.agent.session.operation import OperationCompleted, ToolEffectPending
    from dlightrag.engine.agent.session.runtime import AgentSessionRuntime
    from dlightrag.engine.ai.messages import ToolCall
    from tests.unit.test_agent_session_runtime import _assistant, _Effects, _plan

    async with isolated_run_runtime("dispatch_restart") as (runs, pool):
        service, store, mcp, view, bound, claim, tool, runtime = await dispatch_fixture(
            runs, pool, seed=False
        )
        entered = asyncio.Event()

        async def call(**kwargs):
            entered.set()
            await asyncio.Event().wait()

        mcp.call.side_effect = call

        from dlightrag.engine.agent.session.effects import ToolResultEntry
        from dlightrag.engine.agent.session.runtime import ToolEffectResult
        from dlightrag.engine.runtime.settlements import EffectHostUpdate

        class Effects:
            def __init__(self, turns):
                self.fake = _Effects(turns)
                self.executed_sources = []

            async def assemble_request(self, context):
                return await self.fake.assemble_request(context)

            async def call_provider(self, context, request, attempt_id, emit_ephemeral):
                return await self.fake.call_provider(context, request, attempt_id, emit_ephemeral)

            async def compact(self, context, attempt) -> Any:
                return await self.fake.compact(context, attempt)

            async def execute_tool(
                self, context, item, arguments, attempt_id, emit_ephemeral
            ) -> ToolEffectResult[EffectHostUpdate]:
                self.executed_sources.append(item.source_index)
                if crash_point == "pending":
                    entered.set()
                    await asyncio.Event().wait()
                # Ordinary runtime commits Pending before invoking this public tool.
                result = await tool.execute(
                    tool.input_model.model_validate(arguments),
                    replace(runtime, intent_id=item.intent_id, call_id=item.call_id),
                )
                return ToolEffectResult(
                    ToolResultEntry.text(
                        tool_name=item.tool_name,
                        call_id=item.call_id,
                        outcome="failed" if result.is_error else "succeeded",
                        text=result.text_content,
                    ),
                    host_delta=None,
                )

        def repository(claim):
            return PGAgentSessionRepository(
                pool=pool,
                owner_id=claim.owner_id,
                run_id=uuid.UUID(claim.run_id),
                worker_id=claim.worker_id,
                lease_owner=claim.worker_id,
                fencing_epoch=claim.fencing_epoch,
            )

        repo = repository(claim)
        first = AgentSessionRuntime(
            repository=repo,
            effects=Effects([_assistant(ToolCall("c", tool.name, {"path": "x"}))]),
            tools=[tool],
            fencing_epoch=claim.fencing_epoch,
        )
        session_id = SessionId(runtime.execution_scope)
        accepted = await first.accept(
            session_id=session_id,
            lane_id=LaneId.main(),
            idempotency_key="restart",
            content="q",
            plan=_plan(tool),
        )
        task = asyncio.create_task(
            first.drive(session_id=session_id, operation_id=accepted.operation_id)
        )
        await asyncio.wait_for(entered.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert isinstance(
            (await first.restore(session_id=session_id, operation_id=accepted.operation_id)).state,
            ToolEffectPending,
        )
        assert mcp.call.await_count == (1 if crash_point == "dispatched" else 0)
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE dlightrag_runs SET lease_expires_at=now()-interval '1 second'"
            )
        reclaimed = await runs.claim_next(worker_id="replacement")
        assert reclaimed is not None
        fresh_claim = replace(
            claim, worker_id="replacement", fencing_epoch=reclaimed.run.fencing_epoch
        )
        fresh = Connections(store=store, mcp=mcp)
        (restored,) = await fresh.restore_research(bindings=bound.bindings, claim=fresh_claim)
        effects = Effects(
            [_assistant(text="Could not complete the requested external write; outcome unknown.")]
        )
        second_repo = repository(fresh_claim)
        second = AgentSessionRuntime(
            repository=second_repo,
            effects=effects,
            tools=[restored],
            fencing_epoch=fresh_claim.fencing_epoch,
        )
        final = await asyncio.wait_for(
            second.drive(session_id=session_id, operation_id=accepted.operation_id), 2
        )
        assert isinstance(final.state, OperationCompleted)
        results = [
            entry.result
            for entry in (await second_repo.load(session_id)).entries
            if isinstance(entry, ToolResultMessageEntry)
        ]
        assert [result.outcome for result in results] == ["outcome_unknown"]
        assert effects.executed_sources == []
        assert mcp.call.await_count == (1 if crash_point == "dispatched" else 0)


@pytest.mark.asyncio
async def test_revoke_wins_locked_pg_gate_race_and_causes_zero_dispatch():
    import asyncio

    from dlightrag.application.connections import ConnectionCommand

    async with isolated_run_runtime("dispatch_revoke_first") as (runs, pool):
        service, store, mcp, view, bound, claim, tool, runtime = await dispatch_fixture(runs, pool)
        async with pool.acquire() as blocker, pool.acquire() as observer:
            async with blocker.transaction():
                await blocker.execute("SELECT 1 FROM dlightrag_connection_heads FOR UPDATE")
                revoke = asyncio.create_task(
                    service.change(
                        owner_id="a",
                        auth_mode="jwt",
                        expected_revision=view.revision,
                        command=ConnectionCommand(
                            kind="revoke", connection_id=bound.bindings[0].connection_id
                        ),
                    )
                )
                async with asyncio.timeout(2):
                    while not await observer.fetchval("""SELECT EXISTS(SELECT 1 FROM pg_stat_activity WHERE datname=current_database()
                        AND wait_event_type='Lock' AND query LIKE '%FOR UPDATE OF h%')"""):
                        await asyncio.sleep(0.01)
                dispatch = asyncio.ensure_future(
                    tool.execute(tool.input_model.model_validate({"path": "x"}), runtime)
                )
                async with asyncio.timeout(2):
                    while (
                        await observer.fetchval("""SELECT count(*) FROM pg_stat_activity WHERE datname=current_database()
                        AND wait_event_type='Lock' AND query LIKE '%dlightrag_connection_heads%'""")
                        < 2
                    ):
                        await asyncio.sleep(0.01)
                assert not dispatch.done()
                mcp.call.assert_not_awaited()
            await asyncio.wait_for(revoke, 2)
            result = await asyncio.wait_for(dispatch, 2)
            assert result.is_error and "No call was sent" in result.text_content
            mcp.call.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", ["parent", "lease", "shutdown"])
async def test_in_flight_local_index_cancels_network_and_releases_resources(cancel):
    import asyncio

    async with isolated_run_runtime("dispatch_cancel") as (runs, pool):
        service, store, mcp, view, bound, claim, tool, runtime = await dispatch_fixture(runs, pool)
        entered, closed = asyncio.Event(), asyncio.Event()

        async def call(**kwargs):
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                closed.set()

        mcp.call.side_effect = call
        task = asyncio.ensure_future(
            tool.execute(tool.input_model.model_validate({"path": "x"}), runtime)
        )
        await asyncio.wait_for(entered.wait(), 2)
        if cancel == "shutdown":
            await service.aclose()
        else:
            async with pool.acquire() as conn:
                await conn.execute(
                    "UPDATE dlightrag_runs SET "
                    + (
                        "cancel_requested_at=now()"
                        if cancel == "parent"
                        else "lease_expires_at=now()-interval '1 second'"
                    )
                )
        result = await asyncio.wait_for(task, 2)
        assert result.is_error and "unknown" in result.text_content
        assert closed.is_set()
        mcp.call.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("grant_state", [None, "retired", "audience", "ciphertext", "oauth"])
async def test_static_grant_authority_and_audience_gate_before_io(grant_state):
    async with isolated_run_runtime("dispatch_grant") as (runs, pool):
        service, store, mcp, view, bound, claim, tool, runtime = await dispatch_fixture(
            runs, pool, authenticated=True
        )
        if grant_state:
            async with pool.acquire() as conn:
                change = {
                    "retired": "status='retired'",
                    "audience": "audience_digest='another'",
                    "ciphertext": "encrypted_envelope=NULL",
                    "oauth": "kind='oauth'",
                }[grant_state]
                await conn.execute("UPDATE dlightrag_connection_grants SET " + change)
        result = await tool.execute(tool.input_model.model_validate({"path": "x"}), runtime)
        assert result.is_error == (grant_state is not None)
        assert mcp.call.await_count == (0 if grant_state else 1)
        if grant_state is None:
            assert (
                mcp.call.call_args.kwargs["bearer"].get_secret_value() == "dispatch-test-only-token"
            )
        assert "dispatch-test-only-token" not in result.text_content
        assert "dispatch-test-only-token" not in repr(
            await service.read(owner_id="a", auth_mode="jwt")
        )


@pytest.mark.asyncio
async def test_owner_gate_reaches_real_sdk_fake_http_static_call(monkeypatch):
    import httpx2
    from pydantic import SecretStr

    from dlightrag.adapters.mcp.personal_http import PersonalMcpClient
    from dlightrag.application.connections import Connections
    from dlightrag.application.connections.credentials import CredentialCipher
    from tests.support.dns import public_dns
    from tests.unit.test_connections_config import KEYRING

    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    async with isolated_run_runtime("dispatch_http") as (runs, pool):
        service, store, mcp, view, bound, claim, tool, runtime = await dispatch_fixture(
            runs, pool, authenticated=True
        )
        requests = []

        def handler(request):
            assert request.headers["authorization"] == "Bearer dispatch-test-only-token"
            assert request.headers["host"] == "example.com"
            assert request.url.host == "93.184.216.34"
            body = json.loads(request.content)
            if "id" not in body:
                return httpx2.Response(202)
            if body["method"] == "initialize":
                result = {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "fixture", "version": "1"},
                }
            else:
                requests.append(body)
                assert body["method"] == "tools/call" and body["params"]["name"] == "read"
                result = {"content": [{"type": "text", "text": "written dispatch-test-only-token"}]}
            return httpx2.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})

        client = PersonalMcpClient(transport_factory=lambda: httpx2.MockTransport(handler))
        fresh = Connections(store=store, mcp=client, cipher=CredentialCipher(SecretStr(KEYRING)))
        (restored,) = await fresh.restore_research(bindings=bound.bindings, claim=claim)
        result = await restored.execute(restored.input_model.model_validate({"path": "x"}), runtime)
        assert result.text_content == "written [redacted]"
        assert len(requests) == 1


@pytest.mark.asyncio
async def test_research_runtime_continues_other_tools_and_reports_unavailable_part():
    from pydantic import BaseModel

    from dlightrag.adapters.postgres.answer.session_repository import PGAgentSessionRepository
    from dlightrag.application.connections import ConnectionsError
    from dlightrag.engine.agent.session.effects import ToolResultEntry
    from dlightrag.engine.agent.session.entries import AssistantMessageEntry, ToolResultMessageEntry
    from dlightrag.engine.agent.session.ids import LaneId, SessionId
    from dlightrag.engine.agent.session.operation import OperationCompleted
    from dlightrag.engine.agent.session.plan import AgentRunPlan
    from dlightrag.engine.agent.session.registers import RequestSnapshot
    from dlightrag.engine.agent.session.runtime import AgentSessionRuntime, ToolEffectResult
    from dlightrag.engine.agent.tools import AgentTool
    from dlightrag.engine.ai.messages import AssistantTurn, ToolCall
    from dlightrag.engine.runtime.settlements import EffectHostUpdate

    async with isolated_run_runtime("dispatch_continue") as (runs, pool):
        service, store, mcp, view, bound, claim, tool, runtime = await dispatch_fixture(
            runs, pool, seed=False
        )
        mcp.call.side_effect = ConnectionsError("private remote failure")

        class Args(BaseModel):
            pass

        healthy = AgentTool(
            "healthy",
            "Independent built-in",
            Args,
            AsyncMock(return_value=ToolResult.text("healthy result")),
        )
        tools = {tool.name: tool, healthy.name: healthy}

        class Effects:
            async def assemble_request(self, context):
                self.results = [
                    entry.result
                    for entry in context.snapshot.entries
                    if isinstance(entry, ToolResultMessageEntry)
                ]
                return RequestSnapshot.from_values(
                    operation_id=context.operation_id,
                    turn_number=getattr(context.state, "turn_count", 0) + 1,
                    plan_digest=context.meta.plan_digest,
                    model_role="query",
                    messages=[],
                    tools=[],
                    tool_choice="auto",
                    max_tokens=256,
                )

            async def call_provider(self, context, request, attempt_id, emit_ephemeral):
                if not self.results:
                    return AssistantTurn(
                        text="",
                        tool_calls=(
                            ToolCall("bad", tool.name, {"path": "x"}),
                            ToolCall("good", "healthy", {}),
                        ),
                        stop_reason="tool_use",
                    )
                assert [result.outcome for result in self.results] == ["failed", "succeeded"]
                assert "final Answer" in self.results[0].text_content
                assert "healthy result" == self.results[1].text_content
                return AssistantTurn(
                    text="Healthy part completed; external requested part unavailable, outcome unknown.",
                    tool_calls=(),
                    stop_reason="stop",
                )

            async def compact(self, context, attempt) -> Any:
                raise AssertionError("No compaction needed")

            async def execute_tool(
                self, context, item, arguments, attempt_id, emit_ephemeral
            ) -> ToolEffectResult[EffectHostUpdate]:
                selected = tools[item.tool_name]
                result = await selected.execute(
                    selected.input_model.model_validate(arguments),
                    replace(
                        runtime,
                        intent_id=item.intent_id,
                        call_id=item.call_id,
                        tool_name=item.tool_name,
                    ),
                )
                return ToolEffectResult(
                    result=ToolResultEntry.text(
                        tool_name=item.tool_name,
                        call_id=item.call_id,
                        outcome="failed" if result.is_error else "succeeded",
                        text=result.text_content,
                    ),
                    host_delta=None,
                )

        repo = PGAgentSessionRepository(
            pool=pool,
            owner_id="a",
            run_id=uuid.UUID(claim.run_id),
            worker_id="worker",
            lease_owner="worker",
            fencing_epoch=claim.fencing_epoch,
        )
        agent = AgentSessionRuntime(
            repository=repo,
            effects=Effects(),
            tools=list(tools.values()),
            fencing_epoch=claim.fencing_epoch,
        )
        session_id = SessionId(runtime.execution_scope)
        accepted = await agent.accept(
            session_id=session_id,
            lane_id=LaneId.main(),
            idempotency_key="continue",
            content="Do both parts",
            plan=AgentRunPlan.from_tools(
                list(tools.values()), model_role="query", context_policy_revision="context-v1"
            ),
        )
        final = await agent.drive(session_id=session_id, operation_id=accepted.operation_id)
        assert isinstance(final.state, OperationCompleted)
        snapshot = await repo.load(session_id)
        assistant = [
            entry for entry in snapshot.entries if isinstance(entry, AssistantMessageEntry)
        ][-1]
        assert "external requested part unavailable" in assistant.content
        mcp.call.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("expired", [False, True])
async def test_oauth_dispatch_uses_only_live_access_token_without_refresh_or_redirect(expired):
    import time

    from pydantic import SecretStr

    from dlightrag.application.connections.credentials import CredentialCipher
    from tests.unit.test_connections_config import KEYRING

    async with isolated_run_runtime("oauth_dispatch") as (runs, pool):
        service, store, mcp, view, bound, claim, tool, runtime = await dispatch_fixture(
            runs, pool, authenticated=True
        )
        _, stored = await store.read("a")
        grant = stored[0].grant_id
        assert grant is not None
        key_id, envelope = CredentialCipher(SecretStr(KEYRING)).encrypt(
            SecretStr(
                json.dumps(
                    {
                        "tokens": {
                            "token_type": "Bearer",
                            "access_token": "oauth-test-access",
                            "refresh_token": "must-not-refresh",
                        },
                        "expires_at": time.time() + (-10 if expired else 300),
                    }
                )
            ),
            owner_id="a",
            connection_id=stored[0].connection_id,
            grant_id=grant,
        )
        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE dlightrag_connection_grants SET kind='oauth',encrypted_envelope=$1,key_id=$2 WHERE owner_id='a' AND grant_id=$3",
                envelope,
                key_id,
                grant,
            )
        result = await tool.execute(tool.input_model.model_validate({"path": "x"}), runtime)
        if expired:
            assert result.is_error
            mcp.call.assert_not_awaited()
            assert (await service.read(owner_id="a", auth_mode="jwt")).connections[
                0
            ].status == "needs-auth"
        else:
            assert not result.is_error
            mcp.call.assert_awaited_once()
            assert mcp.call.call_args.kwargs["bearer"].get_secret_value() == "oauth-test-access"
        await service.aclose()
