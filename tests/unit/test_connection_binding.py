# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Durable binding input and Answer acceptance interfaces, without providers."""

from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from dlightrag.application.connections import BoundResearchConnections, ConnectionsError
from dlightrag.engine.agent.tools import ToolDeclaration
from dlightrag.engine.answer.execution.connection_binding import (
    RunConnectionBinding,
    StaleConnectionBindingError,
    decode_connection_bindings,
)
from dlightrag.engine.answer.execution.input import AnswerRunInput
from tests.unit.test_answer_service import (
    _record,
    _request,
    _Resources,
    _Retrieval,
    _service,
    _Store,
)


class Arguments(BaseModel):
    path: str


def bound(generation=1):
    return BoundResearchConnections(
        tools=(
            ToolDeclaration(
                name="mcp_fixture",
                description=f"generation {generation}",
                input_model=Arguments,
            ),
        ),
        bindings=(RunConnectionBinding("owner-1", "a" * 32, generation, 1, "b" * 64),),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["auto", "research"])
async def test_acceptance_pins_tools_and_input_for_research_capable_modes(mode):
    store = _Store()
    binding = bound()
    binder = AsyncMock(return_value=binding)
    await _service(store=store, bind_research=binder).create(
        request=_request(mode=mode), owner_id="owner-1", auth_mode="jwt"
    )
    binder.assert_awaited_once_with(owner_id="owner-1", auth_mode="jwt")
    accepted = store.created[0]
    decoded = AnswerRunInput.from_prepared_input(accepted["prepared_input"])
    assert decoded.run_connection_bindings == accepted["connection_bindings"] == binding.bindings
    assert decoded.agent_run_plan is not None
    assert (
        next(
            t.definition["description"]
            for t in decoded.agent_run_plan.tools
            if t.name == "mcp_fixture"
        )
        == "generation 1"
    )


@pytest.mark.asyncio
async def test_fast_never_binds_and_keyed_replay_keeps_original_pins():
    binder = AsyncMock(side_effect=AssertionError("must not bind"))
    store = _Store()
    service = _service(store=store, bind_research=binder)
    await service.create(request=_request(mode="fast"), owner_id="owner-1")
    assert store.created[0]["connection_bindings"] == ()
    assert (
        AnswerRunInput.from_prepared_input(
            store.created[0]["prepared_input"]
        ).run_connection_bindings
        == ()
    )
    from dlightrag.engine.runtime.records import RunCreation

    replay_store = _Store(replay=RunCreation(run=_record(), replayed=True))
    await _service(store=replay_store, bind_research=binder).create(
        request=_request(mode="research"), owner_id="owner-1", idempotency_key="original"
    )
    assert replay_store.created == []
    binder.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("churn", [False, True])
async def test_stale_binding_rebuild_is_bounded_and_does_not_repeat_resource_preparation(churn):
    class ChurningStore(_Store):
        attempts = 0

        async def create_run(self, **kwargs):
            self.attempts += 1
            if self.attempts == 1 or churn:
                raise StaleConnectionBindingError("snapshot changed")
            return await super().create_run(**kwargs)

    store = ChurningStore()
    resources = _Resources()
    retrieval = _Retrieval()
    binder = AsyncMock(side_effect=[bound(1), bound(2)])
    service = _service(store=store, resources=resources, retrieval=retrieval, bind_research=binder)
    if churn:
        with pytest.raises(ConnectionsError, match="changed repeatedly") as error:
            await service.create(request=_request(mode="research"), owner_id="owner-1")
        assert error.value.status == 409 and not store.created
    else:
        await service.create(request=_request(mode="research"), owner_id="owner-1")
        accepted = store.created[0]
        assert accepted["connection_bindings"] == bound(2).bindings
        plan = AnswerRunInput.from_prepared_input(accepted["prepared_input"]).agent_run_plan
        assert plan is not None
        assert (
            next(t.definition["description"] for t in plan.tools if t.name == "mcp_fixture")
            == "generation 2"
        )
    assert binder.await_count == store.attempts == 2
    assert retrieval.calls.count("schema_for") == 1
    assert resources.calls.count("resolve") == resources.calls.count("pin_current_image_links") == 1


@pytest.mark.parametrize(
    "change",
    [
        {"bearer": "forbidden"},
        {"owner_id": ""},
        {"generation": True},
        {"activation_epoch": 0},
        {"catalogue_digest": "bad"},
    ],
)
def test_binding_wire_rejects_extra_secret_fields_and_invalid_pins(change):
    with pytest.raises((ValueError, TypeError)):
        decode_connection_bindings([{**bound().bindings[0].as_json(), **change}])


def test_binding_wire_is_bounded_and_rejects_duplicates():
    binding = bound().bindings[0].as_json()
    for payload in (None, {}, [binding, binding], [binding] * 101):
        with pytest.raises(ValueError):
            decode_connection_bindings(payload)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["fast", "research"])
async def test_executor_resolver_claim_comes_from_run_session_not_prepared_arguments(mode):
    from typing import Any, cast
    from unittest.mock import MagicMock

    from dlightrag.engine.runtime.coordinator import RunSession
    from dlightrag.engine.runtime.errors import RunExecutionError
    from tests.in_memory_session_repository import MemoryAgentSessionRepository
    from tests.unit.test_answer_executor import _executor

    accepted_store = _Store()
    binding = bound()
    await _service(store=accepted_store, bind_research=AsyncMock(return_value=binding)).create(
        request=_request(mode="research"), owner_id="owner-1", auth_mode="jwt"
    )
    payload = {
        **accepted_store.created[0]["prepared_input"],
        "owner_id": "model-forged",
        "worker_id": "model-forged",
        "fencing_epoch": 999,
    }
    request = AnswerRunInput.from_prepared_input(payload)
    resolver = AsyncMock(return_value=binding.tools)
    executor = _executor()
    executor._connection_tool_resolver = resolver
    executor.validate_pinned_model_profiles = MagicMock(
        return_value={p.role: p.profile for p in request.pinned_models}
    )
    executor._store.load_routing = AsyncMock(return_value=MagicMock(resolved_mode=mode))
    executor.prepare_orchestrated_run = AsyncMock(
        side_effect=RunExecutionError("test_stop", "Prepared")
    )
    session = MagicMock(
        owner_id="owner-1",
        run_id="trusted-run",
        worker_id="trusted-worker",
        fencing_epoch=7,
        prepared_input=payload,
    )
    session.check_cancelled = AsyncMock()
    session.enter_phase = AsyncMock()
    session.execution.session_repository = MemoryAgentSessionRepository[Any](fencing_epoch=11)
    session.execution.fencing_epoch = 11
    with pytest.raises(RunExecutionError) as error:
        await executor.execute(cast(RunSession, session))
    assert error.value.kind == "test_stop", repr(error.value.__cause__)
    assert executor.prepare_orchestrated_run.await_args is not None
    if mode == "fast":
        resolver.assert_not_awaited()
        assert executor.prepare_orchestrated_run.await_args.kwargs["connection_tools"] == ()
    else:
        resolver.assert_awaited_once()
        assert resolver.await_args is not None
        claim = resolver.await_args.kwargs["claim"]
        assert (claim.owner_id, claim.run_id, claim.worker_id, claim.fencing_epoch) == (
            "owner-1",
            "trusted-run",
            "trusted-worker",
            7,
        )
        assert claim.check_cancelled is session.check_cancelled
        assert resolver.await_args.kwargs["bindings"] == binding.bindings
        assert (
            executor.prepare_orchestrated_run.await_args.kwargs["connection_tools"] == binding.tools
        )


@pytest.mark.asyncio
async def test_concurrent_key_winner_replays_before_stale_rebind():
    from dlightrag.engine.runtime.records import RunCreation

    class ConcurrentStore(_Store):
        async def create_run(self, **kwargs):
            self._replay = RunCreation(run=_record(), replayed=True)
            raise StaleConnectionBindingError("A concurrent original acceptance won")

    store = ConcurrentStore()
    binder = AsyncMock(side_effect=[bound(), AssertionError("Replay must not rebind")])
    result = await _service(store=store, bind_research=binder).create(
        request=_request(mode="research"), owner_id="owner-1", idempotency_key="same"
    )
    assert result.replayed
    binder.assert_awaited_once()
    assert not store.created
