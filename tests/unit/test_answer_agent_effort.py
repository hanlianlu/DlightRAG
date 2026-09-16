# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""One accepted run may re-level its own answering agent and nothing else."""

from dataclasses import replace
from typing import Any, cast

import pytest
from pydantic import ValidationError

from dlightrag.adapters.http.rest.models import AnswerRequest
from dlightrag.application.settings import default_answer_effort
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
