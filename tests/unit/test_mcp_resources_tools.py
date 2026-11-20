# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The MCP Resource tools' pinned contracts and their dispatch to one Run's host."""

from typing import Any

import pytest
from pydantic import ValidationError

from dlightrag.engine.agent.session.plan import AgentToolPlan
from dlightrag.engine.agent.tools.contracts import (
    AgentTool,
    ToolResult,
    ToolRuntime,
)
from dlightrag.engine.answer.tools.mcp_resources import (
    MCP_LIST_RESOURCE_TEMPLATES,
    MCP_LIST_RESOURCES,
    MCP_READ_RESOURCE,
    ListInput,
    McpResourceHost,
    ReadResourceInput,
    mcp_resource_declarations,
    mcp_resource_tools,
)
from tests.support.tool_helpers import tool_runtime

_NAMES = (MCP_LIST_RESOURCES, MCP_LIST_RESOURCE_TEMPLATES, MCP_READ_RESOURCE)


class _FakeHost:
    """A host that records each dispatch and answers with one prepared result."""

    def __init__(self, result: ToolResult) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    async def list_resources(
        self, *, connection: str, cursor: str | None, runtime: ToolRuntime
    ) -> ToolResult:
        self.calls.append(
            {
                "tool": "list_resources",
                "connection": connection,
                "cursor": cursor,
                "runtime": runtime,
            }
        )
        return self.result

    async def list_resource_templates(
        self, *, connection: str, cursor: str | None, runtime: ToolRuntime
    ) -> ToolResult:
        self.calls.append(
            {
                "tool": "list_resource_templates",
                "connection": connection,
                "cursor": cursor,
                "runtime": runtime,
            }
        )
        return self.result

    async def read_resource(self, *, connection: str, uri: str, runtime: ToolRuntime) -> ToolResult:
        self.calls.append(
            {"tool": "read_resource", "connection": connection, "uri": uri, "runtime": runtime}
        )
        return self.result


def _host(result: ToolResult | None = None) -> McpResourceHost:
    return _FakeHost(ToolResult.text("unused") if result is None else result)


def test_the_three_tools_are_offered_in_contract_order() -> None:
    tools = mcp_resource_tools(_host())

    assert tuple(tool.name for tool in tools) == _NAMES
    assert all(isinstance(tool, AgentTool) for tool in tools)
    assert [declaration.name for declaration in mcp_resource_declarations()] == list(_NAMES)


def test_the_listing_arguments_are_required_bounded_and_paginated() -> None:
    list_tools = mcp_resource_tools(_host())[:2]

    for tool in list_tools:
        parameters = tool.definition.parameters
        assert parameters["required"] == ["connection"]
        assert parameters["properties"]["connection"]["maxLength"] == 80
        cursor = parameters["properties"]["cursor"]
        assert cursor["default"] is None
        assert {"type": "string", "maxLength": 2048} in cursor["anyOf"]
        assert ListInput.model_fields["cursor"].default is None

    assert ListInput(connection="docs").cursor is None
    assert ListInput(connection="docs", cursor="next").cursor == "next"
    with pytest.raises(ValidationError):
        ListInput(connection="")
    with pytest.raises(ValidationError):
        ListInput(connection="c" * 81)
    with pytest.raises(ValidationError):
        ListInput(connection="docs", cursor="c" * 2049)


def test_the_read_arguments_require_a_bounded_uri() -> None:
    read = mcp_resource_tools(_host())[2]

    parameters = read.definition.parameters
    assert parameters["required"] == ["connection", "uri"]
    assert parameters["properties"]["uri"]["maxLength"] == 4096
    assert ReadResourceInput(connection="docs", uri="mcp://doc/1").uri == "mcp://doc/1"
    with pytest.raises(ValidationError):
        ReadResourceInput(connection="docs", uri="")
    with pytest.raises(ValidationError):
        ReadResourceInput(connection="docs", uri="u" * 4097)
    with pytest.raises(ValidationError):
        ReadResourceInput(connection="docs")  # type: ignore[call-arg]


def test_each_tool_replays_reads_only_and_keeps_its_declared_contract() -> None:
    tools = mcp_resource_tools(_host())

    assert [AgentToolPlan.from_tool(tool) for tool in tools] == [
        AgentToolPlan.from_tool(declaration) for declaration in mcp_resource_declarations()
    ]
    for tool in tools:
        assert tool.replay_policy == "replayable"
        assert tool.read_only is True
        assert tool.contract_version == 1


async def test_the_factory_passes_arguments_and_returns_the_host_result_unchanged() -> None:
    listed = ToolResult.text("documents", subject="docs")
    host = _FakeHost(listed)
    list_resources, list_templates, read_resource = mcp_resource_tools(host)
    runtime = tool_runtime(tool_name=MCP_LIST_RESOURCES)

    assert await list_resources.execute(ListInput(connection="docs"), runtime) is listed
    assert (
        await list_templates.execute(ListInput(connection="docs", cursor="page-2"), runtime)
        is listed
    )
    assert (
        await read_resource.execute(
            ReadResourceInput(connection="docs", uri="mcp://doc/1"), runtime
        )
        is listed
    )
    assert host.calls == [
        {
            "tool": "list_resources",
            "connection": "docs",
            "cursor": None,
            "runtime": runtime,
        },
        {
            "tool": "list_resource_templates",
            "connection": "docs",
            "cursor": "page-2",
            "runtime": runtime,
        },
        {"tool": "read_resource", "connection": "docs", "uri": "mcp://doc/1", "runtime": runtime},
    ]


async def test_a_host_failure_result_propagates_unchanged() -> None:
    failure = ToolResult.text("The server refused the read.", is_error=True)
    host = _FakeHost(failure)
    read_resource = mcp_resource_tools(host)[2]

    result = await read_resource.execute(
        ReadResourceInput(connection="docs", uri="mcp://missing"), tool_runtime()
    )

    assert result is failure
    assert result.is_error is True
    assert result.text_content == "The server refused the read."
