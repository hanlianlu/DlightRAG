# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""One accepted run may re-level its own answering agent and nothing else."""

from collections.abc import Mapping
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from dlightrag.adapters.http.rest.models import AnswerRequest
from dlightrag.application.answer_runs.service import AnswerService
from dlightrag.application.settings import default_answer_effort
from dlightrag.engine.ai.capacity import ModelProfile
from dlightrag.engine.ai.reasoning import ReasoningLevel
from dlightrag.engine.ai.settings import ModelRoleOverrides, ModelSettings
from dlightrag.engine.answer.client_contracts import (
    ANSWER_EFFORT_LEVELS,
    AnswerRequestContract,
    normalize_answer_effort,
)
from dlightrag.engine.answer.execution.input import AnswerRunRequest
from tests.config_helpers import clone_config, replace_config
from tests.unit.test_answer_model_runtime import _runtime
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


def test_the_deployment_default_is_reported_only_when_it_is_one_of_the_three(test_config):
    def configured(level: ReasoningLevel | None):
        config = clone_config(test_config)
        replace_config(
            config,
            "models.chat.roles",
            ModelRoleOverrides(
                # A role override counts only with explicit auth; the default
                # endpoint's environment key never authorizes another model.
                query=ModelSettings(model="strongest", api_key="fake", agentic_reasoning=level),
            ),
        )
        return config

    assert default_answer_effort(configured("high")) == "high"
    # A level the three-level control cannot name is never mislabelled as one it can.
    assert default_answer_effort(configured("xhigh")) is None
    assert default_answer_effort(configured(None)) is None


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


def test_the_deployment_default_is_not_offered_when_the_model_cannot_express_it(test_config):
    from dlightrag.engine.ai.reasoning import ReasoningLevels, ReasoningProfile

    config = clone_config(test_config)
    replace_config(
        config,
        "models.chat.roles",
        ModelRoleOverrides(
            query=ModelSettings(model="strongest", api_key="fake", agentic_reasoning="max"),
        ),
    )
    without_max = ModelProfile(
        context_window_tokens=100_000,
        max_output_tokens=8_000,
        reasoning=ReasoningProfile(
            "openai",
            ReasoningLevels(
                off="none", minimal=None, low="low", medium=None, high="high", xhigh=None, max=None
            ),
        ),
    )
    with_max = ModelProfile(
        context_window_tokens=100_000,
        max_output_tokens=8_000,
        reasoning=ReasoningProfile(
            "openai",
            ReasoningLevels(
                off="none", minimal=None, low="low", medium=None, high="high", xhigh=None, max="max"
            ),
        ),
    )

    # The control never marks a default it is not also offering.
    assert default_answer_effort(config, without_max) is None
    assert default_answer_effort(config, with_max) == "max"
    assert default_answer_effort(config) == "max"


@pytest.mark.asyncio
async def test_an_effort_no_level_can_honor_is_refused_at_admission(test_config):
    """A model that names no level has nothing to clamp to, so the run is refused.

    Admitting it would either answer silently without the requested thinking or fail
    inside a provider call, reporting a configuration fact as a provider rejection.
    """
    from dlightrag.engine.answer.errors import UnsupportedAnswerEffortError

    service = cast(Any, _service_stub())
    service.answering_model_profile = lambda: ModelProfile(
        context_window_tokens=100_000, reasoning=None
    )
    request = AnswerRunRequest.from_request({"query": "q", "effort": "max"})

    with pytest.raises(UnsupportedAnswerEffortError) as refusal:
        service._reject_unhonorable_effort(request)

    assert "does not offer the 'max' agent effort" in refusal.value.public_message
    assert refusal.value.error_kind == "unsupported_effort"


def test_an_effort_is_refused_when_raw_kwargs_own_the_reasoning():
    """A role that states its own reasoning fields cannot also take a typed level.

    The single-owner rule rejects typed-beside-raw at configuration time, so applying
    a caller's effort would bypass that validation (`model_copy` re-validates nothing)
    and raise where the request is planned, which reaches the caller as a provider
    rejection on a run that already started.
    """
    from dlightrag.engine.answer.errors import UnsupportedAnswerEffortError

    service = cast(
        Any,
        _service_stub(agentic_model_kwargs={"chat_template_kwargs": {"enable_thinking": True}}),
    )
    request = AnswerRunRequest.from_request({"query": "q", "effort": "max"})

    with pytest.raises(UnsupportedAnswerEffortError) as refusal:
        service._reject_unhonorable_effort(request)

    assert "raw model kwargs" in refusal.value.public_message
    assert refusal.value.error_kind == "unsupported_effort"


@pytest.mark.asyncio
async def test_admission_refuses_the_effort_a_raw_kwargs_role_cannot_take():
    """The refusal lands at admission, before any run work, through the real seam."""
    from dlightrag.engine.answer.errors import UnsupportedAnswerEffortError
    from tests.unit.test_answer_service import _request as service_request
    from tests.unit.test_answer_service import _service as answer_service

    service = answer_service(
        models=MagicMock(
            model_settings=MagicMock(
                return_value=ModelSettings(
                    model="test",
                    agentic_model_kwargs={"chat_template_kwargs": {"enable_thinking": True}},
                )
            ),
        )
    )

    with pytest.raises(UnsupportedAnswerEffortError) as refusal:
        await service.create(
            request=service_request(mode="research", effort="max"),
            owner_id="owner-1",
            auth_mode="jwt",
        )

    assert "raw model kwargs" in refusal.value.public_message


def test_an_effort_a_below_top_model_can_clamp_is_still_admitted():
    """The documented clamp stands: only a model with no level at all is refused."""
    from dlightrag.engine.ai.reasoning import ReasoningLevels, ReasoningProfile

    service = cast(Any, _service_stub())
    service.answering_model_profile = lambda: ModelProfile(
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
                max=None,
            ),
        ),
    )
    service._reject_unhonorable_effort(
        AnswerRunRequest.from_request({"query": "q", "effort": "max"})
    )


def _service_stub(*, agentic_model_kwargs: Mapping[str, Any] | None = None) -> Any:
    """The two facts `_reject_unhonorable_effort` reads from its service."""

    class _Stub:
        answering_model_profile = staticmethod(lambda: None)
        _models = SimpleNamespace(
            model_settings=lambda role: ModelSettings(
                model="test", agentic_model_kwargs=agentic_model_kwargs or {}
            )
        )

    stub: Any = _Stub()
    stub._reject_unhonorable_effort = AnswerService._reject_unhonorable_effort.__get__(stub)
    return stub


def test_the_run_trace_states_what_was_chosen_and_what_actually_ran():
    """The stored effort alone would misreport a clamped run.

    `max` below the model's ladder runs as `xhigh`, a Fast answer never enters an
    agent loop at all, and no choice means no fact to state.
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

    assert _agent_effort_trace(None, "research", clamped) is None
    assert _agent_effort_trace("max", "research", clamped) == {
        "requested": "max",
        "effective": "xhigh",
    }
    assert _agent_effort_trace("high", "research", clamped) == {
        "requested": "high",
        "effective": "high",
    }
    # Fast runs no agent turn, so it claims no effective agentic level.
    assert _agent_effort_trace("max", "fast", clamped) == {"requested": "max", "effective": None}
