# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for MCP workspace lifecycle tools."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from dlightrag.citations.schemas import SourceReference
from dlightrag.core.retrieval.protocols import RetrievalResult
from dlightrag.mcp import server as mcp_server
from dlightrag.models.schemas import Reference


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
    answer_tool = next(tool for tool in tools if tool.name == "answer")
    answer_props = answer_tool.inputSchema["properties"]
    assert "conversation_history" in answer_props
    assert "query_images" in answer_props
    assert "session_id" in answer_props
    assert "referenced_image_ids" in answer_props
    assert "filters" in answer_props


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


async def test_mcp_answer_forwards_manager_answer_capabilities_and_sanitizes_contexts(
    mock_mcp_manager,
) -> None:
    mock_mcp_manager.aanswer = AsyncMock(
        return_value=RetrievalResult(
            answer="Answer [1-1].",
            contexts={
                "chunks": [
                    {
                        "chunk_id": "c1",
                        "reference_id": "1",
                        "file_path": "/private/report.pdf",
                        "content": "Evidence",
                        "image_data": "base64-payload",
                        "_workspace": "default",
                    }
                ]
            },
            references=[Reference(id="1", title="report.pdf")],
            sources=[SourceReference(id="1", title="report.pdf", path="/private/report.pdf")],
        )
    )

    result = await mcp_server.call_tool(
        "answer",
        {
            "query": "Follow up",
            "workspaces": ["default"],
            "top_k": 8,
            "answer_candidate_top_k": 12,
            "answer_context_top_k": 4,
            "conversation_history": [{"role": "user", "content": "Previous"}],
            "query_images": ["data:image/png;base64,abc"],
            "session_id": "session-1",
            "referenced_image_ids": ["img_1"],
            "filters": {"doc_title": "Manual"},
        },
    )

    body = json.loads(result[0].text)
    assert body["answer"] == "Answer [1-1]."
    assert body["contexts"]["chunks"][0]["image_url"] == "/images/default/c1?size=full"
    assert "image_data" not in body["contexts"]["chunks"][0]
    assert body["sources"][0]["id"] == "1"

    call_kwargs = mock_mcp_manager.aanswer.call_args.kwargs
    assert call_kwargs["workspaces"] == ["default"]
    assert call_kwargs["top_k"] == 8
    assert call_kwargs["answer_candidate_top_k"] == 12
    assert call_kwargs["answer_context_top_k"] == 4
    assert call_kwargs["conversation_history"] == [{"role": "user", "content": "Previous"}]
    assert call_kwargs["query_images"] == ["data:image/png;base64,abc"]
    assert call_kwargs["session_id"] == "session-1"
    assert call_kwargs["referenced_image_ids"] == ["img_1"]
    assert call_kwargs["filters"].doc_title == "Manual"
