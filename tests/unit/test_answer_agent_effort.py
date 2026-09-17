# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""One accepted run may re-level its own answering agent and nothing else."""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from dlightrag.adapters.http.rest.models import AnswerRequest
from dlightrag.engine.ai.capacity import ModelProfile
from dlightrag.engine.ai.reasoning import ReasoningLevel
from dlightrag.engine.ai.settings import ModelSettings
from dlightrag.engine.answer.client_contracts import (
    ANSWER_EFFORT_LEVELS,
    AnswerRequestContract,
    normalize_answer_effort,
)
from dlightrag.engine.answer.execution.input import AnswerRunRequest
from tests.unit.test_answer_model_runtime import _runtime
from tests.unit.test_answer_service import _Store
from tests.unit.test_child_model_roles import _prepared_executor, _roles


def test_the_three_levels_are_the_only_accepted_efforts():
    assert ANSWER_EFFORT_LEVELS == ("low", "high", "max")
    for level in ANSWER_EFFORT_LEVELS:
        assert normalize_answer_effort(level) == level
    assert normalize_answer_effort(" High ") == "high"
    # No choice at all is spelled as an absent or empty value, never as a level.
    assert normalize_answer_effort(None) is None
    assert normalize_answer_effort("") is None
    # An unsupported level is refused instead of being silently downgraded.
    for value in ["xhigh", "medium", "off", 3, ["low"]]:
        with pytest.raises(ValueError, match="effort"):
            normalize_answer_effort(value)


def test_every_public_transport_accepts_only_the_three_levels():
    for effort in ANSWER_EFFORT_LEVELS:
        assert (
            AnswerRequestContract.model_validate({"query": "q", "effort": effort}).effort == effort
        )
        assert AnswerRequest.model_validate({"query": "q", "effort": effort}).effort == effort
    with pytest.raises(ValidationError):
        AnswerRequest.model_validate({"query": "q", "effort": "xhigh"})


def test_one_run_carries_its_choice_through_the_accepted_input():
    request = AnswerRunRequest.from_request({"query": "q", "effort": "max"})
    assert request.effort == "max"
    assert request.as_request()["effort"] == "max"
    # An unsupported level never reaches an accepted run.
    with pytest.raises(ValueError, match="effort"):
        AnswerRunRequest.from_request({"query": "q", "effort": "medium"})


def test_re_leveling_the_answering_model_never_touches_another_role_or_run():
    runtime = _runtime()
    roles = _roles()
    runtime._settings = replace(runtime._settings, model_roles=roles)

    deployment = runtime.tool_model("query")
    re_leveled = runtime.tool_model("query", agentic_reasoning="max")

    assert deployment is not re_leveled
    assert deployment.settings.effective_agentic_reasoning == "high"
    assert re_leveled.settings.effective_agentic_reasoning == "max"
    # The same role at the deployment default is still the original wrapper.
    assert runtime.tool_model("query") is deployment
    # Children keep their own configured levels: no other role gained an override.
    assert set(runtime._tool_models) == {("query", None), ("query", "max")}
    assert all(role == "query" for role, _level in runtime._tool_models)


async def test_one_run_binds_the_re_leveled_model_while_children_keep_theirs(monkeypatch):
    executor, orchestrator, _provider, _pins = await _prepared_executor(
        monkeypatch, agent_effort="max"
    )

    runtime = executor._models
    bound = cast(Any, orchestrator)._model_func
    assert bound is runtime.tool_model("query", agentic_reasoning="max")
    assert bound.settings.effective_agentic_reasoning == "max"
    # Only the answering role of this run was re-leveled; every other role (and the
    # same role for any other run) still reads the deployment default.
    assert bound is not runtime.tool_model("query")
    assert all(role == "query" for role, _level in runtime._tool_models)
    assert runtime.tool_model("default").settings.effective_agentic_reasoning == "low"
    assert ("default", "max") not in runtime._tool_models


def test_the_offer_names_only_levels_the_answering_model_can_express():
    """A ladder that stops below a level never advertises it.

    Otherwise the control promises a level reasoning resolution would have to
    clamp, and the run reads as the caller's choice while running at another one.
    """
    from dlightrag.engine.ai.reasoning import ReasoningLevels, ReasoningProfile
    from dlightrag.engine.answer.client_contracts import offered_answer_efforts

    def profile(levels: ReasoningLevels) -> ModelProfile:
        return ModelProfile(
            context_window_tokens=100_000,
            max_output_tokens=8_000,
            reasoning=ReasoningProfile("openai", levels),
        )

    gemini_like = profile(
        ReasoningLevels(
            off="none", minimal=None, low="low", medium="medium", high="high", xhigh=None, max=None
        )
    )
    every_level = profile(
        ReasoningLevels(
            off="none",
            minimal="minimal",
            low="low",
            medium="medium",
            high="high",
            xhigh="xhigh",
            max="max",
        )
    )
    no_reasoning = ModelProfile(context_window_tokens=100_000, reasoning=None)

    # A best-effort (uncatalogued) endpoint maps every level, so it offers all three.
    assert offered_answer_efforts(every_level) == ("low", "high", "max")
    # A model whose ladder stops at `high` offers two, and never the level it cannot be.
    assert offered_answer_efforts(gemini_like) == ("low", "high")
    # A model that states no effort at all offers nothing: no picker, no valid choice.
    assert offered_answer_efforts(no_reasoning) == ()


def test_the_offer_states_the_levels_and_the_deployment_default():
    """One call states both, so a control cannot mark a default it does not offer."""
    from dlightrag.engine.ai.reasoning import ReasoningLevels, ReasoningProfile
    from tests.unit.test_answer_service import _service as answer_service

    every_level = ModelProfile(
        context_window_tokens=100_000,
        max_output_tokens=8_000,
        reasoning=ReasoningProfile(
            "openai",
            ReasoningLevels(
                off="none",
                minimal=None,
                low="low",
                medium=None,
                high="high",
                xhigh=None,
                max="max",
            ),
        ),
    )
    without_max = replace(
        every_level,
        reasoning=ReasoningProfile(
            "openai",
            ReasoningLevels(
                off="none", minimal=None, low="low", medium=None, high="high", xhigh=None, max=None
            ),
        ),
    )

    def offer(profile: ModelProfile, *, configured: ReasoningLevel | None = None, **settings: Any):
        service = cast(Any, answer_service())
        service.answering_model_profile = lambda: profile
        service._models = SimpleNamespace(
            model_settings=lambda role: ModelSettings(
                model="test", agentic_reasoning=configured, **settings
            )
        )
        return service.agent_effort_offer()

    # The deployment's own level is the default only where the model can express it.
    assert offer(every_level, configured="high").default == "high"
    assert offer(without_max, configured="max").default is None
    assert offer(every_level, configured="xhigh").default is None
    assert offer(every_level).default is None
    assert offer(without_max).levels == ("low", "high")

    # A role that owns its reasoning through raw provider fields applies no typed level
    # at all, so it offers none and marks no default — the control hides.
    # Configuration validation makes these two mutually exclusive, so a role in this
    # state necessarily configures no typed level.
    raw = offer(
        every_level,
        agentic_model_kwargs={"chat_template_kwargs": {"enable_thinking": True}},
    )
    assert raw.levels == ()
    assert raw.default is None


@pytest.mark.asyncio
async def test_a_raw_kwargs_role_admits_the_effort_and_ignores_it():
    """A caller's preference never fails a run over a deployment configuration fact."""
    from tests.unit.test_answer_service import _request as service_request
    from tests.unit.test_answer_service import _service as answer_service

    store = _Store()
    service = answer_service(
        store=store,
        models=MagicMock(
            model_settings=MagicMock(
                return_value=ModelSettings(
                    model="test",
                    agentic_model_kwargs={"chat_template_kwargs": {"enable_thinking": True}},
                )
            ),
        ),
    )

    creation = await service.create(
        request=service_request(mode="research", effort="max"),
        owner_id="owner-1",
        auth_mode="jwt",
    )

    assert creation.run.run_id
    # The choice stays auditable on the accepted input even though nothing applied it.
    assert store.created[0]["prepared_input"]["effort"] == "max"


def test_the_run_trace_states_what_was_chosen_and_what_actually_ran():
    """The stored effort alone would misreport every ending but the applied one.

    `max` below the model's ladder runs as `xhigh`; a role that owns its reasoning
    through raw kwargs takes no typed level; a model that names no level has nothing to
    clamp to; and a Fast answer enters no agent loop at all.
    """
    from dlightrag.engine.ai.reasoning import ReasoningLevels, ReasoningProfile
    from dlightrag.engine.answer.execution.executor import _agent_effort_trace

    clamped = ModelProfile(
        context_window_tokens=100_000,
        max_output_tokens=8_000,
        reasoning=ReasoningProfile(
            "openai",
            ReasoningLevels(
                off="none",
                minimal=None,
                low="low",
                medium=None,
                high="high",
                xhigh="xhigh",
                max=None,
            ),
        ),
    )
    no_levels = ModelProfile(
        context_window_tokens=100_000,
        max_output_tokens=8_000,
        reasoning=ReasoningProfile(
            "openai",
            ReasoningLevels(
                off="none",
                minimal=None,
                low=None,
                medium=None,
                high=None,
                xhigh=None,
                max=None,
            ),
        ),
    )

    assert _agent_effort_trace(None, "research", clamped) is None
    assert _agent_effort_trace("max", "research", clamped) == {
        "requested": "max",
        "effective": "xhigh",
    }
    assert _agent_effort_trace("high", "research", clamped) == {
        "requested": "high",
        "effective": "high",
    }
    assert _agent_effort_trace("max", "research", no_levels) == {
        "requested": "max",
        "effective": None,
        "ignored": "no_level",
    }
    assert _agent_effort_trace("max", "research", clamped, ("chat_template_kwargs",)) == {
        "requested": "max",
        "effective": None,
        "ignored": "raw_kwargs",
    }
    # Fast runs no agent turn, so it claims no effective agentic level.
    assert _agent_effort_trace("max", "fast", clamped) == {
        "requested": "max",
        "effective": None,
        "ignored": "fast",
    }
