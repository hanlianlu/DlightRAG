# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Research tools for the documents a user's connected MCP servers expose as Resources.

A Connection publishes tools and, separately, Resources: documents addressed by URI.
These three tools read those documents without routing every read through one server
tool, and without putting each server's catalogue on the Run's tool surface. The
Connection transport owns what one reply may carry, so a tool returns the host's
result unchanged and never truncates or spills it.
"""

from __future__ import annotations

from typing import Annotated, Protocol

from pydantic import BaseModel, ConfigDict, Field

from dlightrag.engine.agent.tools.contracts import (
    AgentTool,
    ToolDeclaration,
    ToolResult,
    ToolRuntime,
)

MCP_LIST_RESOURCES = "mcp_list_resources"
MCP_LIST_RESOURCE_TEMPLATES = "mcp_list_resource_templates"
MCP_READ_RESOURCE = "mcp_read_resource"
#: The three shared tools as one membership fact: a prompt may offer them only when the
#: Run actually composes them.
RESOURCE_TOOL_NAMES = (MCP_LIST_RESOURCES, MCP_LIST_RESOURCE_TEMPLATES, MCP_READ_RESOURCE)

_ConnectionToken = Annotated[
    str,
    Field(
        min_length=1,
        max_length=80,
        description="The Connection token exactly as the Run prompt lists it.",
    ),
]

_LIST_RESOURCES_DESCRIPTION = (
    "List the documents the user's connected external MCP servers expose, by URI. "
    "A reply may carry a next cursor; pass it back to list the next page. Resources "
    "are read-only."
)

_LIST_TEMPLATES_DESCRIPTION = (
    "List the URI templates the user's connected external MCP servers publish. A "
    "template describes documents whose URI you fill in yourself; pass back a cursor "
    "from the reply to list the next page."
)

_READ_RESOURCE_DESCRIPTION = (
    "Read one document from a user's connected external MCP server by URI. Returns the "
    "document's text; binary resources are described rather than returned. Resources "
    "are read-only, and reading is how server guidance documents (such as a markdown "
    "specification) become available."
)


class ListInput(BaseModel):
    """The arguments of one Resource listing, for documents and for templates."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    connection: _ConnectionToken
    cursor: str | None = Field(
        default=None,
        max_length=2048,
        description="Pass the next cursor the previous reply returned.",
    )


class ReadResourceInput(BaseModel):
    """The arguments of one Resource read."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    connection: _ConnectionToken
    uri: str = Field(
        min_length=1,
        max_length=4096,
        description="A URI from mcp_list_resources or a template the prompt names.",
    )


def list_resources_declaration() -> ToolDeclaration:
    """The declaration of the tool that lists documents as URIs."""
    return ToolDeclaration(
        name=MCP_LIST_RESOURCES,
        description=_LIST_RESOURCES_DESCRIPTION,
        input_model=ListInput,
        replay_policy="replayable",
        read_only=True,
        contract_version=1,
    )


def list_resource_templates_declaration() -> ToolDeclaration:
    """The declaration of the tool that lists the servers' URI templates."""
    return ToolDeclaration(
        name=MCP_LIST_RESOURCE_TEMPLATES,
        description=_LIST_TEMPLATES_DESCRIPTION,
        input_model=ListInput,
        replay_policy="replayable",
        read_only=True,
        contract_version=1,
    )


def read_resource_declaration() -> ToolDeclaration:
    """The declaration of the tool that reads one document by URI."""
    return ToolDeclaration(
        name=MCP_READ_RESOURCE,
        description=_READ_RESOURCE_DESCRIPTION,
        input_model=ReadResourceInput,
        replay_policy="replayable",
        read_only=True,
        contract_version=1,
    )


def mcp_resource_declarations() -> tuple[ToolDeclaration, ToolDeclaration, ToolDeclaration]:
    """The three declarations, in the order the tools are offered."""
    return (
        list_resources_declaration(),
        list_resource_templates_declaration(),
        read_resource_declaration(),
    )


class McpResourceHost(Protocol):
    """One accepted Run's pinned Connections, reached through the Resource tools.

    The host answers under the Run's own authority: it resolves the Connection token to
    the generations the Run pinned, and its transport states what one reply may carry.
    """

    async def list_resources(
        self, *, connection: str, cursor: str | None, runtime: ToolRuntime
    ) -> ToolResult: ...

    async def list_resource_templates(
        self, *, connection: str, cursor: str | None, runtime: ToolRuntime
    ) -> ToolResult: ...

    async def read_resource(
        self, *, connection: str, uri: str, runtime: ToolRuntime
    ) -> ToolResult: ...


def mcp_resource_tools(
    host: McpResourceHost,
) -> tuple[AgentTool, AgentTool, AgentTool]:
    """Bind the three Resource declarations to one Run's pinned Connections."""

    async def list_resources(raw: BaseModel, runtime: ToolRuntime) -> ToolResult:
        args = raw if isinstance(raw, ListInput) else ListInput.model_validate(raw)
        return await host.list_resources(
            connection=args.connection,
            cursor=args.cursor,
            runtime=runtime,
        )

    async def list_resource_templates(raw: BaseModel, runtime: ToolRuntime) -> ToolResult:
        args = raw if isinstance(raw, ListInput) else ListInput.model_validate(raw)
        return await host.list_resource_templates(
            connection=args.connection,
            cursor=args.cursor,
            runtime=runtime,
        )

    async def read_resource(raw: BaseModel, runtime: ToolRuntime) -> ToolResult:
        args = raw if isinstance(raw, ReadResourceInput) else ReadResourceInput.model_validate(raw)
        return await host.read_resource(
            connection=args.connection,
            uri=args.uri,
            runtime=runtime,
        )

    return (
        list_resources_declaration().bind(list_resources),
        list_resource_templates_declaration().bind(list_resource_templates),
        read_resource_declaration().bind(read_resource),
    )


__all__ = [
    "MCP_LIST_RESOURCES",
    "RESOURCE_TOOL_NAMES",
    "MCP_LIST_RESOURCE_TEMPLATES",
    "MCP_READ_RESOURCE",
    "ListInput",
    "McpResourceHost",
    "ReadResourceInput",
    "list_resource_templates_declaration",
    "list_resources_declaration",
    "mcp_resource_declarations",
    "mcp_resource_tools",
    "read_resource_declaration",
]
