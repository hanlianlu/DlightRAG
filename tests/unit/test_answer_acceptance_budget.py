# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""One budget rule for the model calls an Answer Run is accepted and executed under."""

from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from dlightrag.engine.agent.session.fold import PriorTurns
from dlightrag.engine.ai.capacity import ModelProfile
from dlightrag.engine.ai.settings import CHAT_MODEL_SELECTORS, ChatModelSelector
from dlightrag.engine.ai.tokens import estimate_messages_tokens
from dlightrag.engine.answer.capabilities import RequestModelContext
from dlightrag.engine.answer.errors import AnswerInputOverflowError, UnsupportedAnswerModeError
from dlightrag.engine.answer.execution import acceptance
from dlightrag.engine.answer.execution.acceptance import (
    FAST_GENERATION,
    FAST_PLANNER,
    RESEARCH_PLANNER,
    RESEARCH_SEED,
    ROUTER,
    AnswerHistoryBudget,
    ResearchSeed,
    accept_history,
    reserved_memory_text,
)
from dlightrag.engine.answer.execution.executor import _worst_case_recall_block
from dlightrag.engine.answer.execution.input import AnswerRunInput
from dlightrag.engine.answer.history import HistoryProjectionTarget
from dlightrag.engine.answer.memory import reserved_auto_recall_text
from dlightrag.engine.answer.mode import ModeResource
from dlightrag.engine.answer.router import AnswerModeRouter
from dlightrag.engine.rag.retrieval import RetrievalOptions
from dlightrag.engine.rag.retrieval.planner import RetrievalPlanner
from tests.unit.conftest import answer_image_policy
from tests.unit.test_answer_executor import _executor
from tests.unit.test_answer_service import _Capabilities, _request, _service, _Store

_EXTRACT = ModelProfile(context_window_tokens=64_000, max_output_tokens=8_000)
_QUERY = ModelProfile(context_window_tokens=200_000, max_output_tokens=16_000)
_KEYWORD = ModelProfile(context_window_tokens=48_000, max_output_tokens=4_000)
_SCHEMA = {"author": {"type": "string"}}
_HISTORY = [
    {"role": "user", "content": "What did the filing say about revenue?"},
    {"role": "assistant", "content": "Revenue grew eleven percent year over year."},
]


def _profiles(**overrides: ModelProfile) -> dict[ChatModelSelector, ModelProfile]:
    profiles: dict[ChatModelSelector, ModelProfile] = {
        role: ModelProfile(context_window_tokens=128_000) for role in CHAT_MODEL_SELECTORS
    }
    profiles.update({"extract": _EXTRACT, "query": _QUERY, "keyword": _KEYWORD})
    profiles.update(cast(dict[ChatModelSelector, ModelProfile], overrides))
    return profiles


class _Planner:
    """A real planner serializer that records how each call asked for it."""

    def __init__(self, profile: ModelProfile, calls: list[tuple[ModelProfile, dict[str, Any]]]):
        self._profile = profile
        self._calls = calls
        self._planner = RetrievalPlanner(model_profile=profile)

    def history_input_measure(self, query: str, **kwargs: Any) -> Any:
        self._calls.append((self._profile, {"query": query, **kwargs}))
        return self._planner.history_input_measure(query, **kwargs)


class _Planning:
    def __init__(self) -> None:
        self.calls: list[tuple[ModelProfile, dict[str, Any]]] = []

    def planner_for(self, model_profile: ModelProfile | None = None) -> Any:
        assert model_profile is not None
        return _Planner(model_profile, self.calls)

    async def schema_for(self, workspaces: Sequence[str]) -> dict[str, Any]:
        del workspaces
        return dict(_SCHEMA)

    def warm(self, workspaces: Sequence[str]) -> None:
        del workspaces


def _budget(
    planning: _Planning,
    *,
    profiles: Mapping[ChatModelSelector, ModelProfile] | None = None,
    memory_text: str = "",
) -> AnswerHistoryBudget:
    return AnswerHistoryBudget(
        query="Summarize the filing",
        profiles=profiles or _profiles(),
        planner_for=planning.planner_for,
        schema=dict(_SCHEMA),
        answer_image_policy=lambda _profile: answer_image_policy(),
        image_descriptions=("Image 1: a revenue chart",),
        memory_text=memory_text,
    )


def _signature(target: HistoryProjectionTarget) -> tuple[Any, ...]:
    return (
        target.name,
        target.profile,
        target.proactive_compaction,
        target.require_full_dynamic_reserve,
        target.measure_input([], ""),
        target.measure_input(_HISTORY, "Earlier conversation: the filing is for 2025."),
    )


def test_fast_calls_are_measured_on_their_own_models_with_the_full_reserve() -> None:
    planning = _Planning()

    planner, generation = _budget(planning, memory_text="remembered").fast_targets()

    assert (planner.name, planner.profile) == (FAST_PLANNER, _EXTRACT)
    assert (generation.name, generation.profile) == (FAST_GENERATION, _QUERY)
    assert all(
        target.proactive_compaction and target.require_full_dynamic_reserve
        for target in (planner, generation)
    )
    # Fast plans without preserving the query, over the schema retrieval uses.
    assert planning.calls == [
        (
            _EXTRACT,
            {
                "query": "Summarize the filing",
                "schema": _SCHEMA,
                "current_image_descriptions": ["Image 1: a revenue chart"],
                "preserve_query": None,
            },
        )
    ]
    # The reserved memory block is part of what generation sends.
    without_memory = _budget(_Planning()).fast_targets()[1]
    assert generation.measure_input([], "") > without_memory.measure_input([], "")


def test_research_plans_preserving_the_query_and_only_its_seed_compacts() -> None:
    planning = _Planning()
    seed = ResearchSeed(tools=(), query_images=None, resource_manifest=(), image_budget=None)

    planner, first_request = _budget(planning).research_targets(seed)

    assert (planner.name, planner.profile, planner.proactive_compaction) == (
        RESEARCH_PLANNER,
        _EXTRACT,
        False,
    )
    assert (first_request.name, first_request.profile) == (RESEARCH_SEED, _QUERY)
    assert first_request.proactive_compaction
    assert not first_request.require_full_dynamic_reserve
    assert planning.calls[0][1]["preserve_query"] is True


async def test_routing_is_measured_on_the_keyword_model_exactly_as_it_is_sent() -> None:
    sent: list[list[dict[str, Any]]] = []

    async def keyword_model(**kwargs: Any) -> str:
        sent.append(kwargs["messages"])
        return '{"mode":"fast"}'

    resources = (ModeResource(role="image"), ModeResource(role="document"))
    budget = _budget(_Planning())

    target = budget.router_target(
        resources=resources, valid_modes=("fast", "research"), web_search=True
    )
    await AnswerModeRouter(keyword_model).choose(
        query=budget.query,
        history=_HISTORY,
        resources=resources,
        web_search=True,
        valid_modes=("fast", "research"),
    )

    assert (target.name, target.profile) == (ROUTER, _KEYWORD)
    assert target.measure_input(_HISTORY, "") == estimate_messages_tokens(sent[0])
    assert "images: True" in sent[0][-1]["content"]
    assert "tools: search_knowledge_base,search_web" in sent[0][-1]["content"]


def test_a_routing_call_that_cannot_fit_makes_auto_unsupported() -> None:
    # The routing request alone is over a hundred tokens; the keyword model that
    # routes cannot take it, however large the answering model is.
    budget = _budget(
        _Planning(),
        profiles=_profiles(keyword=ModelProfile(context_window_tokens=64)),
    )

    with pytest.raises(UnsupportedAnswerModeError):
        accept_history(
            budget,
            history=_HISTORY,
            requested_mode="auto",
            allowed_modes=frozenset({"fast", "research"}),
            research=ResearchSeed(
                tools=(), query_images=None, resource_manifest=(), image_budget=None
            ),
            mode_resources=(),
            web_search=False,
        )


def test_an_explicit_fast_request_that_cannot_hold_its_reserve_names_the_planner() -> None:
    budget = _budget(
        _Planning(),
        profiles=_profiles(extract=ModelProfile(context_window_tokens=30_000)),
    )

    with pytest.raises(AnswerInputOverflowError, match=f"^{FAST_PLANNER} fixed input"):
        accept_history(
            budget,
            history=(),
            requested_mode="fast",
            allowed_modes=frozenset({"fast"}),
            research=None,
            mode_resources=(),
            web_search=False,
        )


def test_execution_reserves_the_memory_acceptance_recorded() -> None:
    reserved = reserved_auto_recall_text()

    assert reserved_memory_text(auth_mode="jwt", enabled=True) == reserved
    assert reserved_memory_text(auth_mode="jwt", enabled=False) == ""
    assert reserved_memory_text(auth_mode="simple", enabled=True) == ""
    assert _worst_case_recall_block(
        {"auth_mode": "jwt", "profile_memory_enabled": True}
    ) == reserved_memory_text(auth_mode="jwt", enabled=True)


class _DistinctProfileCapabilities(_Capabilities):
    def current_profiles(self) -> dict[ChatModelSelector, ModelProfile]:
        return _profiles()

    def answer_image_policy(self, profile: ModelProfile, /) -> Any:
        del profile
        return answer_image_policy()

    async def pinned_answer_context(self, models: RequestModelContext, /) -> tuple[Any, None]:
        return models, None


async def test_acceptance_and_execution_measure_a_fast_run_with_the_same_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built: list[tuple[HistoryProjectionTarget, ...]] = []
    fast_targets = AnswerHistoryBudget.fast_targets

    def recording_fast_targets(self: AnswerHistoryBudget) -> tuple[HistoryProjectionTarget, ...]:
        targets = fast_targets(self)
        built.append(targets)
        return targets

    monkeypatch.setattr(AnswerHistoryBudget, "fast_targets", recording_fast_targets)
    planning = _Planning()
    capabilities = _DistinctProfileCapabilities()
    store = _Store()

    async def memory_on(**_kwargs: Any) -> tuple[bool, int]:
        return True, 1

    service = _service(
        store=store,
        retrieval=planning,
        capabilities=capabilities,
        memory_capability=memory_on,
    )
    await service.create(request=_request(mode="fast"), owner_id="owner-1", auth_mode="jwt")
    prepared = store.created[0]["prepared_input"]
    accepted = AnswerRunInput.from_prepared_input(prepared)

    executor = _executor()
    executor._planning = planning
    executor._capabilities = cast(Any, capabilities)
    pinned: dict[ChatModelSelector, ModelProfile] = {
        cast(ChatModelSelector, pin.role): pin.profile for pin in accepted.pinned_models
    }
    executor._resources.resolve = AsyncMock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(
            models=capabilities.request_model_context(pinned),
            current_images=[],
            resource_manifest=(),
            web_sources=None,
            registry=None,
            image_budget=None,
            query_images=None,
            current_image_count=0,
        )
    )
    run = await executor.prepare_orchestrated_run(
        query=accepted.query,
        workspaces=list(accepted.workspaces),
        retrieval=RetrievalOptions(),
        filters=None,
        resources=None,
        pinned_image_descriptions=accepted.image_descriptions,
        worst_case_memory=_worst_case_recall_block(prepared),
        projected_history=PriorTurns(),
        model_profiles=pinned,
        resolved_mode="fast",
        resource_scope="owner-1\0run-1",
        pinned_models=accepted.pinned_models,
    )

    accepted_targets, executed_targets = built
    assert executed_targets is run.fast_history_targets
    assert [_signature(target) for target in executed_targets] == [
        _signature(target) for target in accepted_targets
    ]
    assert [target.name for target in executed_targets] == [FAST_PLANNER, FAST_GENERATION]
    # Both sides reserved the owner's standing memory block.
    assert executed_targets[1].measure_input([], "") > acceptance.AnswerSynthesizer(
        image_policy=answer_image_policy(), model_profile=_QUERY
    ).history_input_measure(accepted.query)([], "")
