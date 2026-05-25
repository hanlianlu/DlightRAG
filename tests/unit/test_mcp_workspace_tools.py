# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for MCP workspace lifecycle tools."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from dlightrag.mcp import server as mcp_server


@pytest.fixture
def mock_mcp_manager(monkeypatch):
    manager = AsyncMock()
    manager.list_workspaces = AsyncMock(return_value=["default"])
    manager.acreate_workspace = AsyncMock()
    manager.areset = AsyncMock(return_value={"workspaces": {"old_ws": {}}, "total_errors": 0})
    monkeypatch.setattr(mcp_server, "_ensure_manager", AsyncMock(return_value=manager))
    return manager


async def test_mcp_lists_workspace_lifecycle_tools() -> None:
    tools = await mcp_server.list_tools()
    names = {tool.name for tool in tools}

    assert "create_workspace" in names
    assert "delete_workspace" in names


async def test_mcp_create_workspace_uses_manager_registry(mock_mcp_manager) -> None:
    result = await mcp_server.call_tool(
        "create_workspace",
        {"workspace": "New Workspace", "display_name": "New Workspace"},
    )

    body = json.loads(result[0].text)
    assert body == {
        "workspace": "new_workspace",
        "display_name": "New Workspace",
        "created": True,
    }
    mock_mcp_manager.acreate_workspace.assert_awaited_once_with(
        "new_workspace",
        display_name="New Workspace",
    )


async def test_mcp_delete_workspace_resets_workspace(mock_mcp_manager) -> None:
    result = await mcp_server.call_tool(
        "delete_workspace",
        {"workspace": "Old Workspace", "keep_files": True, "dry_run": True},
    )

    body = json.loads(result[0].text)
    assert body["workspace"] == "old_workspace"
    assert body["deleted"] is False
    mock_mcp_manager.areset.assert_awaited_once_with(
        workspace="old_workspace",
        keep_files=True,
        dry_run=True,
    )
