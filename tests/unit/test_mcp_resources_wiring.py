# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""On-demand MCP resources: pinned declarations, measured guidance, one Run host."""

from collections.abc import Sequence
from typing import Any
from unittest.mock import AsyncMock

import pytest

from dlightrag.engine.agent.environment.toolchain import (
    SearchToolchain,
)
from dlightrag.engine.agent.session.fold import PriorTurns, WorkingContextProjection
from dlightrag.engine.agent.session.plan import AgentToolPlan
from dlightrag.engine.agent.tools.contracts import ToolResult, declaration_tokens
from dlightrag.engine.ai.capacity import CONTEXT_POLICY, ModelProfile
from dlightrag.engine.answer.evidence import EvidenceLedger
from dlightrag.engine.answer.execution.acceptance import research_history_input_measure
from dlightrag.engine.answer.execution.connection_binding import (
    ConnectionResourceOperations,
    ConnectionSurface,
    ResearchToolClaim,
)
from dlightrag.engine.answer.execution.mcp_resource_host import RunMcpResourceHost
from dlightrag.engine.answer.prompts.agent import (
    agent_control_prompt,
    external_connection_context,
)
from dlightrag.engine.answer.research.context import ContextAssembler
from dlightrag.engine.answer.tools.composition import (
    compose_research_tools,
    research_tool_declarations,
)
from tests.support.tool_helpers import tool_runtime

_RESOURCE_TOOLS = {"mcp_list_resources", "mcp_list_resource_templates", "mcp_read_resource"}
_QUERY_PROFILE = ModelProfile(context_window_tokens=200_000, max_output_tokens=16_000)


def _surface(*, instructions: str = "") -> tuple[ConnectionSurface, ...]:
    return (
        ConnectionSurface(
            connection_id="a" * 32,
            token="Notion",
            label="Notion",
            instructions=instructions,
        ),
    )


def _composed(
    declared: Sequence[Any], *, surface: tuple[ConnectionSurface, ...] | None
) -> list[Any]:
    host = (
        None
        if surface is None
        else RunMcpResourceHost(
            surface=surface,
            operations=AsyncMock(spec=ConnectionResourceOperations),
            claim=ResearchToolClaim("owner", "run", "worker", 1, AsyncMock()),
        )
    )
    return compose_research_tools(
        evidence=EvidenceLedger(),
        trace={},
        retrieve_knowledge_base=AsyncMock(),
        search_web=None,
        register_web_source=None,
        resource_reader=AsyncMock(),
        resource_viewer=AsyncMock(),
        admitted_bytes_reader=AsyncMock(),
        injected_tools=[],
        mcp_resource_host=host,
        search_toolchain=SearchToolchain(),
    )


def test_resource_tools_are_pinned_only_for_a_run_with_connections() -> None:
    declared = research_tool_declarations(
        mcp_resources=True, resource_read=True, resource_view=True
    )
    bound = _composed(declared, surface=_surface())

    assert _RESOURCE_TOOLS <= {tool.name for tool in declared}
    assert [AgentToolPlan.from_tool(tool) for tool in bound] == [
        AgentToolPlan.from_tool(tool) for tool in declared
    ]
    without = research_tool_declarations(mcp_resources=False)
    assert not (_RESOURCE_TOOLS & {tool.name for tool in without})
    # A Run with no enabled Connection composes none of the three, so the plan it pins
    # also declares none.
    assert not (_RESOURCE_TOOLS & {tool.name for tool in _composed(without, surface=None)})


def test_acceptance_measure_counts_the_pinned_connection_guidance() -> None:
    tools = research_tool_declarations(mcp_resources=True, resource_read=True, resource_view=True)
    surface = _surface(instructions="Use page bodies for evidence.")
    kwargs: dict[str, Any] = {
        "model_profile": _QUERY_PROFILE,
        "context_policy": CONTEXT_POLICY,
        "query": "Summarize the page",
        "query_images": None,
        "resource_manifest": (),
        "image_budget": None,
        "tools": tools,
    }
    without = research_history_input_measure(**kwargs)
    with_guidance = research_history_input_measure(**kwargs, connection_surface=surface)

    assert with_guidance([], "") > without([], "")
    external = external_connection_context(surface, include_resources=True)
    assert "Connections available for MCP resources: Notion." in external
    assert "Use page bodies for evidence." in external
    prompt = agent_control_prompt(mcp_resources=True)
    assert "mcp_list_resources" in prompt and "mcp_read_resource" in prompt

    # Acceptance measures the exact request execution assembles, external block included.
    context = ContextAssembler(
        model_profile=_QUERY_PROFILE,
        context_policy=CONTEXT_POLICY,
        query="Summarize the page",
        history=PriorTurns([]),
        query_images=None,
        resource_manifest=(),
        connection_surface=surface,
    )
    measured = context.measure_control_input(
        evidence=EvidenceLedger(), working=WorkingContextProjection()
    )
    assert measured + declaration_tokens(tool.definition for tool in tools) == with_guidance([], "")


def test_a_narrowed_child_is_not_told_about_resource_tools_it_lacks() -> None:
    surface = _surface(instructions="Use page bodies for evidence.")
    without_tools = research_tool_declarations(
        mcp_resources=False, resource_read=True, resource_view=True
    )
    external = external_connection_context(surface, include_resources=False)
    assert "Connections available for MCP resources" not in external
    # The server's instructions still ride with the Connection tools the Session has.
    assert "Use page bodies for evidence." in external and "Notion" in external

    kwargs: dict[str, Any] = {
        "model_profile": _QUERY_PROFILE,
        "context_policy": CONTEXT_POLICY,
        "query": "Summarize the page",
        "query_images": None,
        "resource_manifest": (),
        "image_budget": None,
        "tools": without_tools,
    }
    measure = research_history_input_measure(**kwargs, connection_surface=surface)
    context = ContextAssembler(
        model_profile=_QUERY_PROFILE,
        context_policy=CONTEXT_POLICY,
        query="Summarize the page",
        history=PriorTurns([]),
        query_images=None,
        resource_manifest=(),
        connection_surface=surface,
        mcp_resources=False,
    )
    measured = context.measure_control_input(
        evidence=EvidenceLedger(), working=WorkingContextProjection()
    )
    assert measured + declaration_tokens(tool.definition for tool in without_tools) == measure(
        [], ""
    )


async def test_run_resource_host_reads_only_pinned_connections_and_hides_failures() -> None:
    operations = AsyncMock(spec=ConnectionResourceOperations)
    operations.read_resource.return_value = ToolResult.text("page text")
    host = RunMcpResourceHost(
        surface=_surface(),
        operations=operations,
        claim=ResearchToolClaim("owner", "run", "worker", 1, AsyncMock()),
    )
    runtime = tool_runtime(tool_name="mcp_read_resource")

    result = await host.read_resource(connection="Notion", uri="notion://docs/a", runtime=runtime)
    assert (result.text_content, result.is_error) == ("page text", False)
    call = operations.read_resource.await_args
    assert call.kwargs["connection_id"] == "a" * 32
    assert call.kwargs["uri"] == "notion://docs/a"

    missing = await host.read_resource(connection="Notion ", uri="x", runtime=runtime)
    assert missing.is_error and "No enabled Connection" in missing.text_content

    operations.read_resource.side_effect = RuntimeError("carries a private token")
    failed = await host.read_resource(connection="Notion", uri="notion://docs/a", runtime=runtime)
    assert failed.is_error
    assert "private token" not in failed.text_content


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__])
