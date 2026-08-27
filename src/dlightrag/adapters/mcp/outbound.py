# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Thin outbound MCP client adapter for explicitly configured tools.

There is no discovery registry, marketplace, credential service, or OAuth
platform. Deployment config names each remote tool; one MCP SDK session is
opened for that foreground call and closed before returning.
"""

from __future__ import annotations

import json
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Literal, cast

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from pydantic import BaseModel, RootModel

from dlightrag.engine.agent.tools import AgentTool, ToolResult, ToolRuntime

_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True, slots=True)
class OutboundMcpServer:
    """One deployment-declared MCP endpoint and remote tool allowlist."""

    name: str
    transport: Literal["stdio", "streamable-http"]
    tools: tuple[str, ...]
    command: str | None = None
    args: tuple[str, ...] = ()
    url: str | None = None

    def __post_init__(self) -> None:
        if not _NAME.fullmatch(self.name):
            raise ValueError("outbound MCP server name must contain only letters, digits, _ or -")
        if not self.tools or any(not _NAME.fullmatch(tool) for tool in self.tools):
            raise ValueError("outbound MCP tools must be a non-empty tuple of simple names")
        if len(set(self.tools)) != len(self.tools):
            raise ValueError("outbound MCP tool names must be unique per server")
        if self.transport == "stdio" and (not self.command or self.url):
            raise ValueError("stdio MCP requires command and forbids url")
        if self.transport == "streamable-http" and (not self.url or self.command):
            raise ValueError("streamable-http MCP requires url and forbids command")


class McpToolArguments(RootModel[dict[str, Any]]):
    """Provider-neutral argument object forwarded to one declared MCP tool."""


def outbound_mcp_tools(servers: tuple[OutboundMcpServer, ...]) -> tuple[AgentTool, ...]:
    """Build prefixed Agent tools without contacting endpoints at startup."""
    tools: list[AgentTool] = []
    for server in servers:
        for remote_name in server.tools:
            tools.append(_proxy_tool(server, remote_name))
    return tuple(tools)


def _proxy_tool(server: OutboundMcpServer, remote_name: str) -> AgentTool:
    async def execute(raw: BaseModel, _runtime: ToolRuntime) -> ToolResult:
        args = cast(McpToolArguments, raw).root
        async with _session(server) as session:
            result = await session.call_tool(remote_name, arguments=args)
        content = _render_result(result)
        is_error = bool(getattr(result, "isError", False))
        return ToolResult.text(
            (f"Outbound MCP tool failed: {content}" if is_error else content),
            details={
                "mcp_server": server.name,
                "mcp_tool": remote_name,
                "is_error": is_error,
            },
            is_error=is_error,
        )

    return AgentTool(
        name=f"mcp_{server.name}_{remote_name}",
        description=f"Call configured outbound MCP tool {server.name}/{remote_name}.",
        input_model=McpToolArguments,
        execute=execute,
        replay_policy="never",
    )


@asynccontextmanager
async def _session(server: OutboundMcpServer):
    if server.transport == "stdio":
        parameters = StdioServerParameters(command=server.command or "", args=list(server.args))
        async with stdio_client(parameters) as streams:
            read, write = streams
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session
        return
    async with streamable_http_client(server.url or "") as streams:
        read, write = streams[0], streams[1]
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


def _render_result(result: object) -> str:
    structured = getattr(result, "structuredContent", None)
    pieces: list[str] = []
    for item in getattr(result, "content", ()) or ():
        text = getattr(item, "text", None)
        if isinstance(text, str):
            pieces.append(text)
        else:
            pieces.append(json.dumps(item.model_dump(mode="json"), ensure_ascii=False))
    if structured is not None:
        pieces.append(json.dumps(structured, ensure_ascii=False, sort_keys=True))
    return "\n".join(pieces) or "(empty MCP result)"


__all__ = ["McpToolArguments", "OutboundMcpServer", "outbound_mcp_tools"]
