# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""A Run's agents run at the effort asked for, and never above the ceiling pinned with the Run.

The model is the real tool model over a provider that records the request it is sent, so what is
asserted is the reasoning field a provider would receive: for the Run's own agent and for a child
on each model role, whatever the caller chose and whatever the role was configured with.
"""

from typing import Any, cast

import pytest

from dlightrag.engine.agent.session.ids import SessionId
from dlightrag.engine.ai.reasoning import ReasoningLevels, ReasoningProfile
from dlightrag.engine.ai.settings import ModelRoleOverrides, ModelRoleSettings, ModelSettings
from dlightrag.engine.answer.tools.subagents import ChildRequest
from tests.unit.test_child_model_roles import _prepared_executor
from tests.unit.test_subagents import _context_snapshot

_MESSAGES = [{"role": "user", "content": "hello"}]


def _effort(level: str) -> dict[str, Any]:
    """The field the OpenRouter dialect carries a reasoning level in."""
    return {"reasoning": {"effort": level}}


async def _sent(monkeypatch: pytest.MonkeyPatch, agent: str, **harness: Any) -> dict[str, Any]:
    """The model options the provider was sent for the Run's own agent (``root``) or for a
    child on the named model role."""
    executor, orchestrator, provider, pins = await _prepared_executor(monkeypatch, **harness)
    try:
        model: Any
        if agent == "root":
            model = cast(Any, orchestrator)._model_func
            profile = next(pin.profile for pin in pins if pin.role == "query")
        else:
            child = orchestrator.prepare_child_session(
                ChildRequest(objective="Look it up", model_role=agent),  # type: ignore[arg-type]
                context_snapshot=_context_snapshot(SessionId.new()),
            )
            model, profile = child.model_func, child.model_profile
        await model(messages=_MESSAGES, tools=[], model_profile=profile)
        return provider.complete_tool_turn.call_args.kwargs["model_kwargs"]
    finally:
        await executor._models.aclose()


# The roles configure the query role at `high`, default and keyword at `low`, extract at `medium`
# and vlm at `max`. Each row is one agent, the ceiling pinned with the Run, the caller's choice,
# and the level the provider was sent.
@pytest.mark.parametrize(
    ("agent", "ceiling", "picked", "sent"),
    [
        ("root", "max", None, "high"),
        ("root", "max", "max", "max"),
        ("vlm", "max", None, "max"),
        ("root", "low", None, "low"),
        ("root", "low", "max", "low"),
        ("root", "low", "low", "low"),
        ("query", "low", None, "low"),
        ("vlm", "low", None, "low"),
        ("extract", "low", None, "low"),
        ("default", "low", None, "low"),
        ("root", "high", "max", "high"),
        ("root", "high", "low", "low"),
        ("vlm", "high", None, "high"),
        ("extract", "high", None, "medium"),
    ],
)
async def test_every_agent_of_a_run_is_sent_a_level_no_higher_than_its_ceiling(
    monkeypatch, agent, ceiling, picked, sent
):
    assert await _sent(monkeypatch, agent, agent_effort=picked, effort_ceiling=ceiling) == (
        _effort(sent)
    )


async def test_a_model_that_offers_nothing_under_the_ceiling_is_sent_its_lowest_level(monkeypatch):
    high_and_max = ReasoningProfile(
        "openrouter",
        ReasoningLevels(
            off="disabled",
            minimal=None,
            low=None,
            medium=None,
            high="high",
            xhigh=None,
            max="max",
        ),
    )

    sent = await _sent(monkeypatch, "root", effort_ceiling="low", reasoning=high_and_max)

    assert sent == _effort("high")


async def test_a_role_that_owns_its_reasoning_is_sent_its_own_fields_whatever_is_chosen(
    monkeypatch,
):
    owns = {"chat_template_kwargs": {"enable_thinking": True}}
    roles = ModelRoleSettings(
        default=ModelSettings(model="general", api_key="fake"),
        roles=ModelRoleOverrides(
            query=ModelSettings(model="strongest", api_key="fake", agentic_model_kwargs=owns)
        ),
    )

    sent = await _sent(monkeypatch, "root", roles=roles, agent_effort="max", effort_ceiling="low")

    assert sent == owns, "neither the choice nor the ceiling reaches a role that owns its reasoning"


async def test_a_model_left_on_its_provider_default_is_held_to_the_ceiling(monkeypatch):
    roles = ModelRoleSettings(
        default=ModelSettings(model="general", api_key="fake"),
        roles=ModelRoleOverrides(query=ModelSettings(model="strongest", api_key="fake")),
    )

    unchanged = await _sent(monkeypatch, "root", roles=roles)
    held = await _sent(monkeypatch, "root", roles=roles, effort_ceiling="low")

    assert unchanged == {}, "no ceiling asks the provider for nothing it was not asked for before"
    assert held == _effort("low")
