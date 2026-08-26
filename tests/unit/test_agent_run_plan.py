# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Canonical immutable Agent Run Plan contracts."""

from pydantic import BaseModel, ConfigDict, Field

from dlightrag.agent.session.plan import AgentRunPlan
from dlightrag.agent.tools import AgentTool, ToolResult


class SearchArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(description="One query.")


async def _execute(_args: BaseModel, _runtime: object) -> ToolResult:
    return ToolResult.text("ok")


def test_agent_run_plan_round_trips_one_canonical_payload() -> None:
    tools = (
        AgentTool("search", "Search docs.", SearchArgs, _execute, replay_policy="replayable"),
        AgentTool("mutate", "Mutate state.", SearchArgs, _execute),
    )

    plan = AgentRunPlan.from_tools(
        tools,
        model_role="query",
        context_policy_revision="agent-v3-reserves",
        model_identity={"provider": "openai", "model": "query"},
        model_profile={"context_window_tokens": 100000},
    )
    restored = AgentRunPlan.from_payload(plan.canonical_payload())

    assert restored == plan
    assert restored.digest == plan.digest
    assert (
        AgentRunPlan.from_tools(
            tuple(reversed(tools)),
            model_role="query",
            context_policy_revision="agent-v3-reserves",
            model_identity={"provider": "openai", "model": "query"},
            model_profile={"context_window_tokens": 100000},
        ).digest
        == plan.digest
    )
    policies = {tool.name: tool.replay_policy for tool in restored.tools}
    assert policies == {"mutate": "never", "search": "replayable"}
    assert restored.canonical_json() == plan.canonical_json()
    assert restored.canonical_payload()["model_identity"]["model"] == "query"
    assert restored.provider_attempt_limit == 2
    assert restored.compaction_attempt_limit == 3


def test_agent_run_plan_digest_pins_provider_definition_and_execution_contract() -> None:
    first = AgentRunPlan.from_tools(
        (AgentTool("search", "Search docs.", SearchArgs, _execute),),
        model_role="query",
        context_policy_revision="policy-1",
    )
    changed_description = AgentRunPlan.from_tools(
        (AgentTool("search", "Search trusted docs.", SearchArgs, _execute),),
        model_role="query",
        context_policy_revision="policy-1",
    )
    changed_replay = AgentRunPlan.from_tools(
        (
            AgentTool(
                "search",
                "Search docs.",
                SearchArgs,
                _execute,
                replay_policy="replayable",
            ),
        ),
        model_role="query",
        context_policy_revision="policy-1",
    )

    assert first.digest != changed_description.digest
    assert first.digest != changed_replay.digest
    assert first.tool_schema_tokens > 0
