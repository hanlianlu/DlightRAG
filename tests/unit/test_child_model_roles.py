# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Five public child selectors preserve objectives and effective model bindings."""

from dataclasses import asdict, replace
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from dlightrag.engine.agent.session.fold import PriorTurns
from dlightrag.engine.agent.session.ids import OperationId, SessionId
from dlightrag.engine.agent.session.memory import MemoryAgentSessionRepository
from dlightrag.engine.ai.capacity import CONTEXT_POLICY_REVISION, ModelCapabilityError, ModelProfile
from dlightrag.engine.ai.catalog import current_model_catalog_revision
from dlightrag.engine.ai.fingerprints import model_fingerprint
from dlightrag.engine.ai.messages import AssistantTurn
from dlightrag.engine.ai.reasoning import (
    ReasoningLevel,
    best_effort_reasoning_profile,
    merge_reasoning_kwargs,
    resolve_reasoning,
)
from dlightrag.engine.ai.settings import (
    CHAT_MODEL_SELECTORS,
    ModelRoleOverrides,
    ModelRoleSettings,
    ModelSettings,
)
from dlightrag.engine.answer.capabilities import RequestModelContext
from dlightrag.engine.answer.execution.input import (
    PinnedModelProfile,
    child_model_guidance,
    model_reasoning_settings,
)
from dlightrag.engine.answer.research.runtime import FetchedResourceBuffer, run_child_session
from dlightrag.engine.answer.tools.subagents import (
    ChildRequest,
    SpawnAgentInput,
    SubagentHost,
    subagent_tools,
)
from dlightrag.engine.rag.retrieval import RetrievalOptions
from dlightrag.engine.runtime.errors import IncompatibleActiveRunError
from dlightrag.engine.runtime.settlements import EffectHostUpdate
from tests.unit.test_answer_executor import _executor
from tests.unit.test_answer_model_runtime import _runtime
from tests.unit.test_subagents import _context_snapshot, _FakeSession


@pytest.mark.parametrize("role", CHAT_MODEL_SELECTORS)
def test_spawn_accepts_five_selectors_with_arbitrary_objective(role):
    objective = "Write a sonnet about an imaginary clock, then critique its rhythm."
    args = SpawnAgentInput.model_validate(
        {"children": [{"objective": objective, "model_role": role}]}
    )
    assert args.children[0].objective == objective
    assert args.children[0].model_role == role


def test_spawn_rejects_unknown_selector_and_defaults_to_query():
    assert ChildRequest(objective="Anything").model_role == "query"
    with pytest.raises(ValidationError):
        ChildRequest.model_validate({"objective": "Anything", "model_role": "strongest"})


def _roles():
    return ModelRoleSettings(
        default=ModelSettings(model="general", api_key="fake", reasoning="low"),
        roles=ModelRoleOverrides(
            query=ModelSettings(model="strongest", api_key="fake", reasoning="high"),
            keyword=ModelSettings(model="unauthorized-override"),
            extract=ModelSettings(
                model="routine", api_key="fake", reasoning="low", agentic_reasoning="medium"
            ),
            vlm=ModelSettings(model="visual", api_key="fake", agentic_reasoning="max"),
        ),
    )


def _pins(roles):
    return tuple(
        PinnedModelProfile(
            role=role,
            fingerprint=model_fingerprint(roles.resolve(role)),
            profile=ModelProfile(
                context_window_tokens=100_000,
                max_output_tokens=64_000,
                supports_images=role == "vlm",
                reasoning=best_effort_reasoning_profile("openrouter"),
            ),
            reasoning_settings=model_reasoning_settings(roles.resolve(role)),
        )
        for role in CHAT_MODEL_SELECTORS
    )


def test_runtime_default_is_not_query_and_complete_auth_fallback_is_preserved():
    runtime = _runtime()
    runtime._settings = replace(runtime._settings, model_roles=_roles())
    with patch("dlightrag.engine.answer.model_runtime.ToolModel") as wrapper:
        for role, name, reasoning in [
            ("default", "general", "low"),
            ("query", "strongest", "high"),
            ("keyword", "general", "low"),
            ("vlm", "visual", "max"),
            ("extract", "routine", "medium"),
        ]:
            runtime.tool_model(role)  # type: ignore[arg-type]
            settings = wrapper.call_args.args[0]
            assert settings.model == name
            assert settings.effective_agentic_reasoning == reasoning
    with pytest.raises(ValueError, match="selector"):
        runtime.tool_model("unknown")  # type: ignore[arg-type]


def test_spawn_guidance_uses_serialized_effective_profiles_not_tier_guarantees():
    pins = _pins(_roles())
    restored = tuple(PinnedModelProfile.from_json(pin.as_json()) for pin in pins)
    assert restored == pins
    tool = subagent_tools(host=SubagentHost(model_guidance=child_model_guidance(restored)))[0]
    assert tool.contract_version == 5
    assert tool.input_model is SpawnAgentInput
    assert "default: general-purpose tier below query" in tool.description
    assert "query: preferred strongest reasoning tier" in tool.description
    assert "model=general; images=False" in tool.description
    assert "model=visual; images=True" in tool.description
    assert "agentic_reasoning_request=medium" in tool.description
    assert "not a universal capability guarantee" in tool.description
    narrowed = tuple(
        replace(pin, profile=replace(pin.profile, supports_images=False))
        if pin.role == "vlm"
        else pin
        for pin in pins
    )
    assert "model=visual; images=False" in child_model_guidance(narrowed)


async def _prepared_executor(monkeypatch, agent_effort: ReasoningLevel | None = None):
    roles = _roles()
    pins = _pins(roles)
    profiles = {cast(Any, pin.role): pin.profile for pin in pins}
    models = RequestModelContext(
        extract=profiles["extract"], query=profiles["query"], vlm=profiles["vlm"]
    )
    provider = AsyncMock()
    provider.complete_tool_turn.return_value = AssistantTurn(
        text="a sonnet", tool_calls=(), stop_reason="stop"
    )
    monkeypatch.setattr("dlightrag.engine.ai.tool_model.get_provider", lambda *a, **kw: provider)
    monkeypatch.setattr("dlightrag.engine.ai.completion.get_provider", lambda *a, **kw: provider)
    runtime = _runtime()
    runtime._settings = replace(runtime._settings, model_roles=roles)
    executor = _executor()
    executor._models = runtime
    executor._model_fingerprint_for_role = lambda role: model_fingerprint(roles.resolve(role))
    cast(Any, executor._capabilities).request_model_context.return_value = models
    executor._resources.resolve = AsyncMock(
        return_value=SimpleNamespace(
            models=models,
            current_images=[],
            injected_tools=[],
            resource_manifest=(),
            web_sources=None,
            registry=None,
            image_budget=None,
            query_images=None,
            current_image_count=0,
        )
    )
    run = await executor.prepare_orchestrated_run(
        query="parent",
        workspaces=["test"],
        retrieval=RetrievalOptions(),
        filters=None,
        resources=None,
        pinned_image_descriptions=(),
        projected_history=PriorTurns(),
        model_profiles=profiles,
        resolved_mode="research",
        resource_scope="owner/run",
        pinned_models=pins,
        agent_effort=agent_effort,
    )
    return executor, run.orchestrator, provider, pins


@pytest.mark.parametrize("role", CHAT_MODEL_SELECTORS)
async def test_selected_binding_survives_durable_continuation_recovery_and_rejects_drift(
    monkeypatch, role
):
    executor, orchestrator, provider, pins = await _prepared_executor(monkeypatch)
    parent, child = SessionId.new(), SessionId.new()
    context = _context_snapshot(parent)
    repo = MemoryAgentSessionRepository[EffectHostUpdate]()
    row = {}

    async def persist(**kw):
        row.setdefault("plan", kw["plan"])
        row["model_role"] = kw["model_role"]

    async def run(key, objective):
        row.update(
            operation_key=key, operation_id=OperationId.deterministic(idempotency_key=key).value
        )
        return await run_child_session(
            orchestrator=orchestrator,
            repository=repo,
            session=cast(Any, _FakeSession(run_id=parent.value)),
            fetched_buffer=FetchedResourceBuffer(),
            child_id=child,
            request=ChildRequest(
                objective=objective, model_role=role, tools=("search_knowledge_base",)
            ),
            parent_call_id="call",
            parent_session_id=parent,
            context_snapshot=context,
            persist_child_runtime=persist,
            claim_child=AsyncMock(return_value=1),
            load_child=AsyncMock(side_effect=lambda **kw: dict(row)),
        )

    try:
        assert (await run("first", "Compose a sonnet")).status == "succeeded"
        assert row["model_role"] == role
        chosen = next(pin for pin in pins if pin.role == role)
        assert row["plan"]["model_identity"]["model"] == chosen.fingerprint.model
        assert row["plan"]["model_identity"]["reasoning_settings"] == chosen.reasoning_settings
        assert row["plan"]["model_profile"]["supports_images"] == (role == "vlm")
        assert (await run("continue", "Critique its rhythm")).status == "succeeded"
        call_count = provider.complete_tool_turn.await_count
        first_runtime = executor._models
        # Rebuild all process-local model/orchestration objects around the same durable repository.
        executor, orchestrator, recovered_provider, recovered_pins = await _prepared_executor(
            monkeypatch
        )
        assert recovered_pins == pins
        await first_runtime.aclose()
        assert (await run("continue", "Critique its rhythm")).status == "succeeded"
        assert provider.complete_tool_turn.await_count == call_count == 2
        recovered_provider.complete_tool_turn.assert_not_awaited()
        assert "a sonnet" in str(provider.complete_tool_turn.call_args.args[0])
        assert provider.complete_tool_turn.call_args.args[1] == chosen.fingerprint.model
        assert provider.complete_tool_turn.call_args.kwargs[
            "model_kwargs"
        ] == merge_reasoning_kwargs(
            {},
            resolve_reasoning(
                chosen.profile.reasoning, cast(Any, chosen.reasoning_settings)["agentic"]
            ),
        )
        assert {tool["definition"]["name"] for tool in row["plan"]["tools"]} <= {
            "search_knowledge_base",
            "ask_parent",
        }
        # A fresh operation after process restart still dispatches the selected model.
        assert (await run("after-restart", "Revise the sonnet")).status == "succeeded"
        recovered_provider.complete_tool_turn.assert_awaited_once()
        assert recovered_provider.complete_tool_turn.call_args.args[1] == chosen.fingerprint.model
        assert (
            recovered_provider.complete_tool_turn.call_args.kwargs["model_kwargs"]
            == (provider.complete_tool_turn.call_args.kwargs["model_kwargs"])
        )
        # A new runtime after configuration drift cannot reconstruct this child on a new endpoint.
        runtime = executor._models
        runtime._tool_models.clear()
        runtime._settings = replace(
            runtime._settings, model_roles=ModelRoleSettings(default=ModelSettings(model="drifted"))
        )
        with pytest.raises(IncompatibleActiveRunError, match="binding changed"):
            await run("third", "Do not reroute")
        assert provider.complete_tool_turn.await_count == call_count
        recovered_provider.complete_tool_turn.assert_awaited_once()
    finally:
        await executor._models.aclose()


@pytest.mark.parametrize("drift", ["default_identity", "default_reasoning", "incomplete_pins"])
def test_parent_recovery_fails_closed_on_default_or_reasoning_drift(drift):
    executor = _executor()
    roles = _roles()
    pins = _pins(roles)
    executor._model_fingerprint_for_role = lambda role: model_fingerprint(roles.resolve(role))
    executor._models.model_settings = roles.resolve
    request = SimpleNamespace(
        pinned_models=pins,
        context_policy_revision=CONTEXT_POLICY_REVISION,
        model_catalog_revision=current_model_catalog_revision(),
    )
    assert (
        executor.validate_pinned_model_profiles(cast(Any, request))["default"] == pins[-1].profile
    )
    if drift == "default_identity":
        executor._model_fingerprint_for_role = lambda role: (
            model_fingerprint(ModelSettings(model="changed"))
            if role == "default"
            else model_fingerprint(roles.resolve(role))
        )
    elif drift == "default_reasoning":
        executor._models.model_settings = lambda role: (
            ModelSettings(model="general", reasoning="max")
            if role == "default"
            else roles.resolve(role)
        )
    else:
        request.pinned_models = pins[:-1]
    with pytest.raises(IncompatibleActiveRunError):
        executor.validate_pinned_model_profiles(cast(Any, request))


async def test_nonvisual_selected_model_rejects_image_inputs_without_provider_io(monkeypatch):
    executor, orchestrator, provider, _ = await _prepared_executor(monkeypatch)
    import base64
    import io
    import json

    from PIL import Image

    from tests.unit.conftest import answer_image_policy

    pixels = io.BytesIO()
    Image.new("RGB", (32, 32), "blue").save(pixels, format="PNG")
    policy = answer_image_policy(max_images=1)
    image = policy.new_budget().add_base64(
        base64.b64encode(pixels.getvalue()).decode(), label="generated user image"
    )
    assert image is not None
    orchestrator._image_budget = policy.new_budget()
    parent = SessionId.new()
    context = replace(
        _context_snapshot(parent),
        messages_json=json.dumps([{"role": "user", "content": [image]}]),
    )
    try:
        with pytest.raises(ValueError, match="inherited images"):
            orchestrator.prepare_child_session(
                ChildRequest(objective="Inspect pixels", context="parent", model_role="default"),
                context_snapshot=context,
            )
        visual = orchestrator.prepare_child_session(
            ChildRequest(objective="Inspect pixels", context="parent", model_role="vlm"),
            context_snapshot=context,
        )
        assert visual.model_profile.supports_images
        assert orchestrator._image_budget.count == 1
        from dlightrag.engine.answer.errors import AnswerInputOverflowError

        with pytest.raises(AnswerInputOverflowError, match="image budget"):
            orchestrator.prepare_child_session(
                ChildRequest(objective="View again", context="parent", model_role="vlm"),
                context_snapshot=context,
            )
        with pytest.raises(ModelCapabilityError, match="image inputs"):
            await executor._models.tool_model("default")(
                messages=context.messages,
                tools=[],
                model_profile=ModelProfile(context_window_tokens=100_000),
            )
        provider.complete_tool_turn.assert_not_awaited()
    finally:
        await executor._models.aclose()


async def test_acceptance_and_execution_share_pinned_five_model_spawn_guidance(monkeypatch):
    from dlightrag.engine.answer.execution.input import AnswerRunInput
    from tests.unit.test_answer_service import _OWNER, _request, _service

    roles = _roles()
    pins = _pins(roles)
    service = _service()
    service._models.model_settings = roles.resolve
    service._model_fingerprint_for_role = lambda role: model_fingerprint(roles.resolve(role))
    cast(Any, service._capabilities).current_profiles = lambda: {
        pin.role: pin.profile for pin in pins
    }
    service._research_tool_supplements = lambda: subagent_tools(host=SubagentHost())
    await service.create(request=_request(mode="research"), owner_id=_OWNER)
    accepted = AnswerRunInput.from_request(cast(Any, service._store).created[0]["prepared_input"])
    assert accepted.pinned_models == pins
    assert accepted.agent_run_plan is not None
    accepted_spawn = next(
        tool for tool in accepted.agent_run_plan.tools if tool.name == "spawn_agent"
    )
    executor, orchestrator, _, _ = await _prepared_executor(monkeypatch)
    try:
        parent = orchestrator.prepare_run("parent")
        runtime_spawn = next(tool for tool in parent.tools if tool.name == "spawn_agent")
        assert accepted_spawn.definition == asdict(runtime_spawn.definition)
        assert accepted_spawn.input_schema_digest == runtime_spawn.input_schema_digest
        assert accepted_spawn.contract_version == runtime_spawn.contract_version == 5
    finally:
        await executor._models.aclose()
