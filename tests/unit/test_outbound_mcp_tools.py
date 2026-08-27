# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Thin configured outbound MCP tools."""

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from dlightrag.adapters.mcp.outbound import (
    McpToolArguments,
    OutboundMcpServer,
    outbound_mcp_tools,
)
from tests.tool_helpers import tool_runtime


def test_outbound_mcp_requires_explicit_endpoint_and_tool_names() -> None:
    with pytest.raises(ValueError, match="requires url"):
        OutboundMcpServer(name="docs", transport="streamable-http", tools=("search",))
    with pytest.raises(ValueError, match="non-empty"):
        OutboundMcpServer(
            name="docs",
            transport="streamable-http",
            tools=(),
            url="https://mcp.example.test",
        )


@pytest.mark.asyncio
async def test_declared_tool_calls_sdk_session_and_closes_before_return(monkeypatch) -> None:
    call_tool = AsyncMock(
        return_value=SimpleNamespace(
            content=(SimpleNamespace(text="remote result"),),
            structuredContent={"count": 1},
            isError=False,
        )
    )
    session = SimpleNamespace(call_tool=call_tool)
    closed = False

    @asynccontextmanager
    async def fake_session(_server):
        nonlocal closed
        try:
            yield session
        finally:
            closed = True

    monkeypatch.setattr("dlightrag.adapters.mcp.outbound._session", fake_session)
    (tool,) = outbound_mcp_tools(
        (
            OutboundMcpServer(
                name="docs",
                transport="streamable-http",
                tools=("search",),
                url="https://mcp.example.test",
            ),
        )
    )

    result = await tool.execute(McpToolArguments({"query": "agent"}), tool_runtime())

    assert tool.name == "mcp_docs_search"
    call_tool.assert_awaited_once_with("search", arguments={"query": "agent"})
    assert "remote result" in result.text_content
    assert '"count": 1' in result.text_content
    assert result.is_error is False
    assert closed


@pytest.mark.asyncio
async def test_declared_tool_preserves_remote_error_semantics(monkeypatch) -> None:
    call_tool = AsyncMock(
        return_value=SimpleNamespace(
            content=(SimpleNamespace(text="permission denied"),),
            structuredContent=None,
            isError=True,
        )
    )
    session = SimpleNamespace(call_tool=call_tool)

    @asynccontextmanager
    async def fake_session(_server):
        yield session

    monkeypatch.setattr("dlightrag.adapters.mcp.outbound._session", fake_session)
    (tool,) = outbound_mcp_tools(
        (
            OutboundMcpServer(
                name="docs",
                transport="streamable-http",
                tools=("search",),
                url="https://mcp.example.test",
            ),
        )
    )

    result = await tool.execute(McpToolArguments({"query": "agent"}), tool_runtime())

    assert result.is_error is True
    assert result.details == {
        "mcp_server": "docs",
        "mcp_tool": "search",
        "is_error": True,
    }
    assert result.text_content == "Outbound MCP tool failed: permission denied"
