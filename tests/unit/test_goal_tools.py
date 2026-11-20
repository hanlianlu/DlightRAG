# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The goal tools carry a turn's identity and round context; authority lives behind the service."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from dlightrag.engine.agent.tools.contracts import AgentTool, ToolResult
from dlightrag.engine.answer.goal import (
    BLOCKED_AFTER_ROUNDS,
    DEFAULT_MAX_ROUNDS,
    GOAL_INVALID_TRANSITION,
    GOAL_STALE_REVISION,
    Goal,
)
from dlightrag.engine.answer.goal_service import AnswerGoalService
from dlightrag.engine.answer.tools.composition import (
    CHILD_FORBIDDEN_TOOLS,
    research_tool_declarations,
)
from dlightrag.engine.answer.tools.goals import (
    GOAL_TOOL_NAMES,
    CreateGoalArgs,
    GetGoalArgs,
    GoalHost,
    UpdateGoalArgs,
    goal_tool_declarations,
    goal_tools,
)
from tests.support.goals import FakeGoalStore
from tests.support.tool_helpers import tool_runtime

_OWNER = "owner-1"
_SESSION = "session-1"
_RUN = "run-1"


def _service(*, goal_round: int | None = None) -> tuple[AnswerGoalService, FakeGoalStore]:
    store = FakeGoalStore()
    return AnswerGoalService(store=store), store


def _tools(
    service: AnswerGoalService,
    *,
    goal_round: int | None = None,
    completion_refusal: Any = None,
) -> dict[str, AgentTool]:
    host = GoalHost(
        service=service,
        owner_id=_OWNER,
        run_id=_RUN,
        session_id=_SESSION,
        goal_round=goal_round,
        completion_refusal=completion_refusal,
    )
    return {tool.name: tool for tool in goal_tools(host=host)}


async def _call(tool: AgentTool, args: BaseModel) -> ToolResult:
    return await tool.execute(args, tool_runtime(tool_name=tool.name))


async def _seed(service: AnswerGoalService, **kwargs: int) -> Goal:
    return await service.create(
        owner_id=_OWNER,
        session_id=_SESSION,
        run_id=_RUN,
        objective="finish the sealed Polestar model",
        max_rounds=kwargs.get("max_rounds", 4),
    )


def test_the_goal_tools_are_declared_with_their_replay_facts() -> None:
    tools = _tools(_service()[0])

    assert tuple(tools) == GOAL_TOOL_NAMES
    assert [declaration.name for declaration in goal_tool_declarations()] == list(GOAL_TOOL_NAMES)
    assert tools["create_goal"].replay_policy == "never"
    assert tools["update_goal"].replay_policy == "never"
    assert tools["get_goal"].replay_policy == "replayable"
    assert tools["get_goal"].read_only
    for tool in tools.values():
        assert tool.description


@pytest.mark.asyncio
async def test_complete_is_refused_while_a_sealed_skill_delivery_is_unattested() -> None:
    """A skipped read or unpassed verifier leaves the goal active; blocked is the honest exit."""
    service, _ = _service()
    goal = await _seed(service)
    tools = _tools(service, completion_refusal=lambda: "required read was skipped")
    result = await _call(
        tools["update_goal"],
        UpdateGoalArgs(goal_id=goal.id, revision=goal.revision, action="complete"),
    )

    assert result.is_error
    assert "goal_completion_unattested" in result.text_content
    assert "required read was skipped" in result.text_content
    assert "still active" in result.text_content
    stored = await service.get(owner_id=_OWNER, session_id=_SESSION)
    assert stored is not None and stored.phase.value == "active"


@pytest.mark.asyncio
async def test_the_completion_gate_is_not_consulted_for_other_actions() -> None:
    service, _ = _service()
    await _seed(service)
    tools = _tools(service, completion_refusal=lambda: "delivery is not attested")
    result = await _call(
        tools["get_goal"],
        GetGoalArgs(),
    )

    assert not result.is_error
    assert result.text_content.startswith("goal ")


@pytest.mark.asyncio
async def test_blocked_still_ends_a_goal_the_completion_gate_refused() -> None:
    service, _ = _service()
    goal = await _seed(service)
    tools = _tools(service, completion_refusal=lambda: "delivery is not attested")
    result = await _call(
        tools["update_goal"],
        UpdateGoalArgs(
            goal_id=goal.id,
            revision=goal.revision,
            action="blocked",
            blocked_reason="the sealed model cannot reconcile the supplier-finance balance",
        ),
    )

    assert not result.is_error
    stored = await service.get(owner_id=_OWNER, session_id=_SESSION)
    assert stored is not None and stored.phase.value == "blocked"


def test_a_goal_cap_has_no_product_ceiling() -> None:
    assert CreateGoalArgs(objective="one long task").max_goal_rounds == DEFAULT_MAX_ROUNDS
    assert CreateGoalArgs(objective="one long task", max_goal_rounds=1_000).max_goal_rounds == 1_000
    updated = UpdateGoalArgs(
        goal_id="goal-1",
        revision=1,
        action="edit",
        max_goal_rounds=1_000,
    )
    assert updated.max_goal_rounds == 1_000
    # The argument schema states the default the model must size above.
    cap = CreateGoalArgs.model_fields["max_goal_rounds"].description
    assert cap is not None
    assert f"default is {DEFAULT_MAX_ROUNDS}" in cap


@pytest.mark.asyncio
async def test_create_returns_the_goal_its_revision_and_its_rounds() -> None:
    service, store = _service()

    result = await _call(
        _tools(service)["create_goal"],
        CreateGoalArgs(objective="finish the sealed Polestar model"),
    )

    assert not result.is_error
    assert store.rows[(_OWNER, _SESSION)] is not None
    goal = store.rows[(_OWNER, _SESSION)]
    assert goal.id in result.text_content
    assert f"rounds 0/{DEFAULT_MAX_ROUNDS}" in result.text_content
    # The tool result tells the model what happens after this Run, so it need not infer it.
    assert "continues it automatically" in result.text_content
    assert (
        "if the process restarts, recovery takes the durable goal up again" in result.text_content
    )
    assert result.details is not None
    assert result.details["goal_action"] == "create"
    assert result.details["goal_id"] == goal.id


@pytest.mark.asyncio
async def test_get_says_when_the_conversation_carries_nothing() -> None:
    service, _ = _service()

    empty = await _call(_tools(service)["get_goal"], GetGoalArgs())
    assert not empty.is_error
    assert empty.text_content == "this conversation carries no goal"

    seeded = await _seed(service)
    found = await _call(_tools(service)["get_goal"], GetGoalArgs())
    assert seeded.id in found.text_content


@pytest.mark.asyncio
async def test_a_goal_round_can_complete_but_not_create_edit_pause_or_resume() -> None:
    service, store = _service()
    goal = await _seed(service)
    tools = _tools(service, goal_round=1)

    created = await _call(tools["create_goal"], CreateGoalArgs(objective="another"))
    assert created.is_error
    assert "direct human turn" in created.text_content

    for action in ("edit", "pause", "resume"):
        refused = await _call(
            tools["update_goal"],
            UpdateGoalArgs(goal_id=goal.id, revision=goal.revision, action=action, objective="new"),
        )
        assert refused.is_error, action
        assert "human" in refused.text_content

    completed = await _call(
        tools["update_goal"],
        UpdateGoalArgs(goal_id=goal.id, revision=goal.revision, action="complete"),
    )
    assert not completed.is_error
    assert store.rows[(_OWNER, _SESSION)].phase == "complete"


@pytest.mark.asyncio
async def test_a_goal_round_cannot_block_before_its_third_round() -> None:
    service, store = _service()
    goal = await _seed(service)

    refused = await _call(
        _tools(service, goal_round=1)["update_goal"],
        UpdateGoalArgs(
            goal_id=goal.id,
            revision=goal.revision,
            action="blocked",
            blocked_reason="the filings portal is down",
        ),
    )
    assert refused.is_error
    assert f"needs {BLOCKED_AFTER_ROUNDS} rounds started" in refused.text_content

    # Three started rounds are the reference threshold; the same call then moves the goal.
    mature = Goal(
        id=goal.id,
        revision=goal.revision,
        objective=goal.objective,
        phase=goal.phase,
        max_rounds=goal.max_rounds,
        rounds_started=BLOCKED_AFTER_ROUNDS,
    )
    store.rows[(_OWNER, _SESSION)] = mature
    blocked = await _call(
        _tools(service, goal_round=BLOCKED_AFTER_ROUNDS)["update_goal"],
        UpdateGoalArgs(
            goal_id=mature.id,
            revision=mature.revision,
            action="blocked",
            blocked_reason="the filings portal is down",
        ),
    )
    assert not blocked.is_error
    assert store.rows[(_OWNER, _SESSION)].blocked_reason == "the filings portal is down"


@pytest.mark.asyncio
async def test_a_human_turn_blocks_immediately_and_clears_on_resume() -> None:
    service, store = _service()
    goal = await _seed(service)
    tools = _tools(service)

    blocked = await _call(
        tools["update_goal"],
        UpdateGoalArgs(
            goal_id=goal.id,
            revision=goal.revision,
            action="blocked",
            blocked_reason="the filings portal is down",
        ),
    )
    assert not blocked.is_error
    assert "blocked: the filings portal is down" in blocked.text_content

    blocked_goal = store.rows[(_OWNER, _SESSION)]
    resumed = await _call(
        tools["update_goal"],
        UpdateGoalArgs(goal_id=blocked_goal.id, revision=blocked_goal.revision, action="resume"),
    )
    assert not resumed.is_error
    assert store.rows[(_OWNER, _SESSION)].phase == "active"
    assert store.rows[(_OWNER, _SESSION)].blocked_reason is None


@pytest.mark.asyncio
async def test_a_stale_revision_is_refused_with_its_code() -> None:
    service, _ = _service()
    goal = await _seed(service)

    refused = await _call(
        _tools(service)["update_goal"],
        UpdateGoalArgs(goal_id=goal.id, revision=2, action="complete"),
    )

    assert refused.is_error
    assert refused.details is not None
    assert refused.details["goal_error"] == GOAL_STALE_REVISION


@pytest.mark.asyncio
async def test_a_goal_round_cannot_create_a_second_goal_through_the_tool() -> None:
    service, _ = _service()
    await _seed(service)

    refused = await _call(
        _tools(service, goal_round=1)["create_goal"], CreateGoalArgs(objective="another")
    )

    assert refused.is_error
    assert refused.details is not None
    assert refused.details["goal_error"] == GOAL_INVALID_TRANSITION


@pytest.mark.asyncio
async def test_a_paused_goal_is_resumed_by_the_human_not_by_the_tool() -> None:
    """Pause is a human control; the model's tool path cannot undo a human pause."""
    service, store = _service()
    goal = await _seed(service)
    tools = _tools(service)

    paused = await _call(
        tools["update_goal"],
        UpdateGoalArgs(goal_id=goal.id, revision=goal.revision, action="pause"),
    )
    assert not paused.is_error
    paused_goal = store.rows[(_OWNER, _SESSION)]
    assert paused_goal.phase == "paused"

    refused = await _call(
        tools["update_goal"],
        UpdateGoalArgs(goal_id=goal.id, revision=paused_goal.revision, action="resume"),
    )
    assert refused.is_error
    assert "human" in refused.text_content

    resumed = await service.update(
        owner_id=_OWNER,
        session_id=_SESSION,
        run_id=_RUN,
        goal_id=goal.id,
        revision=paused_goal.revision,
        action="resume",
        human_control=True,
    )
    assert resumed.phase == "active"


@pytest.mark.asyncio
async def test_create_replaces_a_completed_goal_and_refuses_an_unfinished_one() -> None:
    """One goal per conversation: a finished goal yields to a new objective."""
    service, store = _service()
    first = await _seed(service)
    tools = _tools(service)

    unfinished = await _call(tools["create_goal"], CreateGoalArgs(objective="another objective"))
    assert unfinished.is_error
    assert "unfinished goal" in unfinished.text_content

    await _call(
        tools["update_goal"],
        UpdateGoalArgs(goal_id=first.id, revision=first.revision, action="complete"),
    )
    completed = await _call(tools["create_goal"], CreateGoalArgs(objective="another objective"))

    assert not completed.is_error
    replacement = store.rows[(_OWNER, _SESSION)]
    assert replacement.id != first.id
    assert replacement.objective == "another objective"
    assert replacement.phase == "active"
    assert replacement.rounds_started == 0


@pytest.mark.asyncio
async def test_a_terminal_round_result_is_a_full_closing_brief() -> None:
    """The last tool result tells the model what the reader's closing message must contain."""
    service, _ = _service()
    goal = await _seed(service)
    tools = _tools(service)

    complete = await _call(
        tools["update_goal"],
        UpdateGoalArgs(goal_id=goal.id, revision=goal.revision, action="complete"),
    )
    text = complete.text_content.lower()
    assert "closing message" in text
    assert "outcome" in text
    assert "evidence" in text
    assert "concrete results" in text
    assert "user directly" in text

    blocked_service, _ = _service()
    blocked_goal = await _seed(blocked_service)
    blocked = await _call(
        _tools(blocked_service)["update_goal"],
        UpdateGoalArgs(
            goal_id=blocked_goal.id,
            revision=blocked_goal.revision,
            action="blocked",
            blocked_reason="the filings portal is down",
        ),
    )
    blocked_text = blocked.text_content.lower()
    assert "closing message" in blocked_text
    assert "what remains" in blocked_text
    assert "blocking condition" in blocked_text
    assert "resume" in blocked_text


def test_goal_declarations_are_research_tools_a_child_never_holds() -> None:
    """The declaration surface matches the gate: opt-in for a Run, absent from every Child."""
    names = {declaration.name for declaration in research_tool_declarations(goals=True)}
    assert set(GOAL_TOOL_NAMES) <= names
    assert set(GOAL_TOOL_NAMES) <= CHILD_FORBIDDEN_TOOLS

    child = {declaration.name for declaration in research_tool_declarations(goals=True, child=True)}
    assert not set(GOAL_TOOL_NAMES) & child
