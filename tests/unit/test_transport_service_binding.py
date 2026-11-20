# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Transport modules stay inert until their lifespan binds an Application."""

from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient
from mcp.types import CallToolResult, TextContent

from dlightrag.adapters.http import server as http_server
from dlightrag.adapters.mcp import server as mcp_server
from dlightrag.application.application import ApplicationClosedError
from dlightrag.application.config.sections import DlightragConfig
from tests.support.application_double import application_double


@pytest.mark.asyncio
async def test_http_route_fails_fast_without_a_lifespan_bound_application(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_application = AsyncMock()
    monkeypatch.setattr(http_server, "create_application", create_application)
    application = http_server.create_app()

    async with AsyncClient(
        transport=ASGITransport(app=application),
        base_url="http://test",
    ) as client:
        response = await client.get("/runs/0199a0a0-0000-7000-8000-0000000000aa")

    assert response.status_code == 503
    assert response.json()["detail"] == "Application is not bound to this HTTP transport"
    create_application.assert_not_awaited()


@pytest.mark.asyncio
async def test_mcp_lifespan_binds_and_closes_exactly_one_application(
    monkeypatch: pytest.MonkeyPatch, test_config: DlightragConfig
) -> None:
    application = application_double(test_config)
    create_application = AsyncMock(return_value=application)
    monkeypatch.setattr(mcp_server, "create_application", create_application)
    monkeypatch.setattr(mcp_server, "_application", None)

    async with mcp_server._mcp_lifespan(mcp_server.mcp_app):
        assert await mcp_server._ensure_application() is application

    create_application.assert_awaited_once_with()
    application.aclose.assert_awaited_once_with()
    assert mcp_server._application is None


@pytest.mark.asyncio
async def test_mcp_tool_fails_fast_without_a_lifespan_bound_application(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    create_application = AsyncMock()
    monkeypatch.setattr(mcp_server, "create_application", create_application)
    monkeypatch.setattr(mcp_server, "_application", None)

    with pytest.raises(ApplicationClosedError, match="not bound to this MCP transport"):
        await mcp_server._ensure_application()

    result = await mcp_server.mcp_app.call_tool("retrieve", {"query": "why"})

    assert isinstance(result, CallToolResult)
    assert result.is_error is True
    assert isinstance(result.content[0], TextContent)
    assert "Application is not bound to this MCP transport" in result.content[0].text
    create_application.assert_not_awaited()
