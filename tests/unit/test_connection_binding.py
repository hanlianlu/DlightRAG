# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Durable binding input and Answer acceptance interfaces, without providers."""

from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

from dlightrag.application.answer_runs import AnswerConnectionsChangedError
from dlightrag.application.connections import BoundResearchConnections
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

        async def accept_run(self, **kwargs):
            self.attempts += 1
            if self.attempts == 1 or churn:
                raise StaleConnectionBindingError("snapshot changed")
            return await super().accept_run(**kwargs)

    store = ChurningStore()
    resources = _Resources()
    retrieval = _Retrieval()
    binder = AsyncMock(side_effect=[bound(1), bound(2)])
    service = _service(store=store, resources=resources, retrieval=retrieval, bind_research=binder)
    if churn:
        with pytest.raises(AnswerConnectionsChangedError):
            await service.create(request=_request(mode="research"), owner_id="owner-1")
        assert not store.created
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


class _ModelCalled(Exception):
    """Ends a driven Run at its first model call."""


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["fast", "research"])
async def test_only_research_offers_connection_tools_restored_under_the_session_claim(
    mode, tmp_path
):
    """Research restores its pinned Connection tools under the claim of the worker running it.

    The claim comes from the RunSession, never from prepared input a caller or model could
    shape, and the restored tools are what the Research model is offered. A Run routed to
    Fast never restores them.
    """
    from types import SimpleNamespace
    from typing import Any, cast
    from unittest.mock import MagicMock

    from dlightrag.engine.ai.capacity import ModelProfile
    from dlightrag.engine.answer.capabilities import RequestModelContext
    from dlightrag.engine.runtime.coordinator import RunSession
    from dlightrag.engine.runtime.errors import RunExecutionError
    from dlightrag.engine.runtime.progress import StageCommit
    from tests.in_memory_session_repository import MemoryAgentSessionRepository
    from tests.support.workspace_store import InMemoryWorkspaceStore
    from tests.unit.test_answer_executor import _executor

    executor = _executor()
    accepted_store = _Store()
    binding = bound()
    service = _service(store=accepted_store, bind_research=AsyncMock(return_value=binding))
    # Acceptance pins the Agent Plan this executor composes, as the composition root wires it.
    service._research_tool_declarations = executor.research_tool_declarations
    await service.create(request=_request(mode="research"), owner_id="owner-1", auth_mode="jwt")
    payload = {
        **accepted_store.created[0]["prepared_input"],
        "owner_id": "model-forged",
        "worker_id": "model-forged",
        "fencing_epoch": 999,
    }
    request = AnswerRunInput.from_prepared_input(payload)

    resolver = AsyncMock(return_value=tuple(tool.bind(AsyncMock()) for tool in binding.tools))
    executor._connection_tool_resolver = resolver
    executor.validate_pinned_model_profiles = MagicMock(
        return_value={p.role: p.profile for p in request.pinned_models}
    )
    executor._store.load_routing = AsyncMock(return_value=MagicMock(resolved_mode=mode))
    executor._store.list_child_sessions = AsyncMock(return_value=[])
    executor._store.load_pending_agent_controls = AsyncMock(return_value=[])
    profile = ModelProfile(context_window_tokens=1_000_000)
    models = RequestModelContext(extract=profile, query=profile, vlm=profile)
    executor._capabilities.request_model_context = MagicMock(return_value=models)
    executor._resources.resolve = AsyncMock(
        return_value=SimpleNamespace(
            models=models,
            current_images=[],
            web_sources=None,
            registry=None,
            resource_manifest=(),
            image_budget=None,
            query_images=None,
            current_image_count=0,
        )
    )
    executor._planning = _Retrieval()
    executor._workspace_root_setting = str(tmp_path / "workspaces")
    (tmp_path / "corpus").mkdir()
    executor._working_dir = str(tmp_path / "corpus")

    offered: list[set[str]] = []

    class _ResearchModel:
        async def __call__(self, **kwargs: Any) -> Any:
            offered.append({tool.name for tool in kwargs["tools"]})
            raise _ModelCalled

        async def stream_text(self, **_kwargs: Any) -> Any:
            raise AssertionError("the first model call ends the Run")

    class _FastGeneration:
        async def generate_stream(self, *_args: Any, **kwargs: Any) -> Any:
            offered.append({tool.name for tool in kwargs.get("tools") or ()})
            raise _ModelCalled

    executor._models.query_tool_model = MagicMock(return_value=_ResearchModel())
    executor._models.answer_synthesizer = MagicMock(return_value=_FastGeneration())

    session = MagicMock(
        owner_id="owner-1",
        run_id="trusted-run",
        worker_id="trusted-worker",
        fencing_epoch=7,
        prepared_input=payload,
        durable_progress_version=0,
        workspace_epoch=None,
        checkpoint=None,
    )
    for method in ("check_cancelled", "enter_phase", "emit_token", "flush_tokens", "reset_output"):
        setattr(session, method, AsyncMock())
    progress = MagicMock()
    progress.load_stage = AsyncMock(return_value=None)
    progress.settle_stage = AsyncMock(
        return_value=StageCommit(progress_version=1, stage_intent_id=MagicMock(), evidence_count=0)
    )
    session.execution.session_repository = MemoryAgentSessionRepository[Any](fencing_epoch=11)
    session.execution.workspace_store = InMemoryWorkspaceStore()
    session.execution.progress_store = progress
    session.execution.fencing_epoch = 11

    with pytest.raises(RunExecutionError):
        await executor.execute(cast(RunSession, session))

    assert len(offered) == 1, "the Run never reached its model"
    if mode == "fast":
        resolver.assert_not_awaited()
        assert offered == [set()]
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
        assert "mcp_fixture" in offered[0]


@pytest.mark.asyncio
async def test_concurrent_key_winner_replays_before_stale_rebind():
    from dlightrag.engine.runtime.records import RunCreation

    class ConcurrentStore(_Store):
        async def accept_run(self, **kwargs):
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


class _HeadLocks:
    """The accepting transaction as the pin writer sees it; no Connection head exists."""

    def __init__(self) -> None:
        self.statements: list[str] = []

    async def fetchrow(self, query: str, *args: object) -> None:
        self.statements.append(query)
        return None


async def _validate_pins(conn: _HeadLocks, *, auth_mode: str, mode: str, pinned: bool) -> None:
    from dlightrag.adapters.postgres.connections import PGConnectionPinWriter

    bindings = bound().bindings if pinned else ()
    await PGConnectionPinWriter.validate_in(
        conn,
        owner_id="owner-1",
        payload={
            "auth_mode": auth_mode,
            "mode": mode,
            "run_connection_bindings": [binding.as_json() for binding in bindings],
        },
        bindings=bindings,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("auth_mode", "mode"), [("simple", "research"), ("simple", "auto"), ("jwt", "fast")]
)
async def test_pin_writer_refuses_pins_for_an_ineligible_acceptance(auth_mode, mode):
    """Only a personal owner's non-Fast acceptance may pin Connections.

    Answer acceptance never binds for a shared ``simple`` owner or a Fast Run, so pins that
    arrive anyway are refused before any head is locked; the same acceptance without pins
    is an ordinary Run and passes untouched.
    """
    refused = _HeadLocks()
    with pytest.raises(ValueError, match="eligible Research acceptance"):
        await _validate_pins(refused, auth_mode=auth_mode, mode=mode, pinned=True)
    assert refused.statements == []
    unpinned = _HeadLocks()
    await _validate_pins(unpinned, auth_mode=auth_mode, mode=mode, pinned=False)
    assert unpinned.statements == []


@pytest.mark.asyncio
@pytest.mark.parametrize("auth_mode", ["jwt", "none"])
async def test_pin_writer_checks_an_eligible_acceptance_against_its_heads(auth_mode):
    heads = _HeadLocks()
    with pytest.raises(StaleConnectionBindingError):
        await _validate_pins(heads, auth_mode=auth_mode, mode="research", pinned=True)
    assert len(heads.statements) == 1
