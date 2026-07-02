# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for MCP workspace lifecycle tools."""

from __future__ import annotations

import inspect
import json
from typing import Any
from unittest.mock import AsyncMock

import pytest

import dlightrag
from dlightrag.citations.schemas import SourceReference
from dlightrag.config import DlightragConfig
from dlightrag.core.retrieval.protocols import RetrievalResult
from dlightrag.mcp import server as mcp_server
from dlightrag.models.schemas import Reference

_IMAGE_BLOCK = {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}


def _metadata_policy_enum(prop: dict) -> list[str]:
    if "enum" in prop:
        return prop["enum"]
    return next(item["enum"] for item in prop["anyOf"] if "enum" in item)


def _resolve_ref(schema: dict, item: dict) -> dict:
    ref = item.get("$ref")
    if not ref:
        return item
    name = ref.removeprefix("#/$defs/")
    return schema["$defs"][name]


def _query_image_schema(schema: dict, prop: dict) -> dict:
    return _resolve_ref(schema, prop["items"])


def _tool_content(result):
    return result[0] if isinstance(result, tuple) else result


def _tool_text(result) -> str:
    return _tool_content(result)[0].text


def _tool_json(result):
    return json.loads(_tool_text(result))


@pytest.fixture
def mock_mcp_manager(monkeypatch):
    manager = AsyncMock()
    manager.list_workspaces = AsyncMock(return_value=["default"])
    manager.acreate_workspace = AsyncMock()
    manager.areset = AsyncMock(return_value={"workspaces": {"old_ws": {}}, "total_errors": 0})
    manager.aretrieve = AsyncMock()
    manager.aanswer = AsyncMock()
    manager.aingest = AsyncMock()
    manager.astart_ingest_job = AsyncMock()
    monkeypatch.setattr(mcp_server, "_ensure_manager", AsyncMock(return_value=manager))
    return manager


def test_mcp_server_info_uses_dlightrag_version() -> None:
    assert mcp_server.server.version == dlightrag.__version__


async def test_mcp_success_payloads_keep_fastmcp_structured_output(mock_mcp_manager) -> None:
    mock_mcp_manager.get_ingest_job = AsyncMock(
        return_value={"job_id": "job-1", "status": "running"}
    )

    result = await mcp_server.mcp_app.call_tool("ingest_job_status", {"job_id": "job-1"})

    assert isinstance(result, tuple)
    assert _tool_json(result) == {"job_id": "job-1", "status": "running"}
    assert result[1] == {"job_id": "job-1", "status": "running"}


async def test_mcp_lists_workspace_lifecycle_tools() -> None:
    tools = await mcp_server.mcp_app.list_tools()
    names = {tool.name for tool in tools}

    assert hasattr(mcp_server, "mcp_app")
    assert "create_workspace" in names
    assert "delete_workspace" in names
    answer_tool = next(tool for tool in tools if tool.name == "answer")
    answer_props = answer_tool.inputSchema["properties"]
    assert "args" not in answer_props
    assert "query" in answer_props
    assert "conversation_history" in answer_props
    assert "query_images" in answer_props
    image_block_schema = _query_image_schema(answer_tool.inputSchema, answer_props["query_images"])
    assert image_block_schema["type"] == "object"
    assert image_block_schema["properties"]["type"]["const"] == "image_url"
    assert set(image_block_schema["required"]) == {"type", "image_url"}
    history_items = _resolve_ref(
        answer_tool.inputSchema,
        answer_props["conversation_history"]["anyOf"][0]["items"],
    )
    assert history_items["properties"]["role"]["enum"] == ["system", "user", "assistant"]
    assert "chunk_top_k" in answer_props
    assert "session_id" in answer_props
    assert "referenced_image_ids" in answer_props
    assert "filters" in answer_props
    retrieve_tool = next(tool for tool in tools if tool.name == "retrieve")
    retrieve_props = retrieve_tool.inputSchema["properties"]
    assert "chunk_top_k" not in retrieve_props
    retrieve_image_block_schema = _query_image_schema(
        retrieve_tool.inputSchema,
        retrieve_props["query_images"],
    )
    assert retrieve_image_block_schema["type"] == "object"
    assert retrieve_image_block_schema["properties"]["type"]["const"] == "image_url"
    ingest_tool = next(tool for tool in tools if tool.name == "ingest")
    ingest_props = ingest_tool.inputSchema["properties"]
    assert "title" in ingest_props
    assert "author" in ingest_props
    assert "metadata" in ingest_props
    assert _metadata_policy_enum(ingest_props["metadata_policy"]) == [
        "validate",
        "reject_unknown",
        "store_only",
    ]
    assert "ingest_job_status" in names


def test_mcp_streamable_http_uses_modern_transport_defaults() -> None:
    source = inspect.getsource(mcp_server.run_streamable_http)

    assert "StreamableHTTPSessionManager" in source
    assert "TransportSecuritySettings" in source
    assert "enable_dns_rebinding_protection=True" in source
    assert "json_response=True" in source
    assert "stateless=True" in source
    assert "MCPPathMiddleware" in source
    assert 'Mount("/mcp"' in source
    assert '"/sse"' not in source
    assert '"/messages"' not in source
    assert "StreamableHTTPServerTransport" not in source


def test_mcp_security_defaults_are_loopback_only() -> None:
    cfg = DlightragConfig()

    assert cfg.mcp_allowed_hosts == ["127.0.0.1:*", "localhost:*", "[::1]:*"]
    assert cfg.mcp_allowed_origins == [
        "http://127.0.0.1:*",
        "http://localhost:*",
        "http://[::1]:*",
    ]


async def test_mcp_rejects_unknown_mode_without_schema_wrapper(mock_mcp_manager) -> None:
    result = await mcp_server.mcp_app.call_tool("answer", {"query": "x", "mode": "mix"})

    content = _tool_content(result)
    assert content[0].type == "text"
    assert "Error:" in content[0].text
    assert "mode" in content[0].text
    mock_mcp_manager.aanswer.assert_not_awaited()


@pytest.mark.parametrize(
    ("tool_name", "payload", "manager_method", "error_fragment"),
    [
        (
            "retrieve",
            {
                "query": "x",
                "query_images": [_IMAGE_BLOCK, _IMAGE_BLOCK, _IMAGE_BLOCK, _IMAGE_BLOCK],
            },
            "aretrieve",
            "query_images",
        ),
        (
            "answer",
            {"query": "x", "query_images": [{"url": "data:image/png;base64,abc"}]},
            "aanswer",
            "image_url",
        ),
        (
            "retrieve",
            {"query": "x", "query_images": ["data:image/png;base64,abc"]},
            "aretrieve",
            "valid dictionary",
        ),
    ],
)
async def test_mcp_rejects_invalid_query_image_payloads(
    mock_mcp_manager,
    tool_name: str,
    payload: dict[str, Any],
    manager_method: str,
    error_fragment: str,
) -> None:
    result = await mcp_server.mcp_app.call_tool(
        tool_name,
        payload,
    )

    assert "Error:" in _tool_text(result)
    assert error_fragment in _tool_text(result)
    getattr(mock_mcp_manager, manager_method).assert_not_awaited()


async def test_mcp_rejects_invalid_metadata_policy(mock_mcp_manager) -> None:
    result = await mcp_server.mcp_app.call_tool(
        "ingest",
        {"source_type": "local", "metadata_policy": "loose"},
    )

    assert "Error:" in _tool_text(result)
    assert "metadata_policy" in _tool_text(result)
    mock_mcp_manager.aingest.assert_not_awaited()


async def test_mcp_rejects_mutually_exclusive_s3_key_and_prefix(mock_mcp_manager) -> None:
    result = await mcp_server.mcp_app.call_tool(
        "ingest",
        {"source_type": "s3", "bucket": "b", "key": "a.pdf", "prefix": "docs/"},
    )

    assert "Error:" in _tool_text(result)
    assert "mutually exclusive" in _tool_text(result)
    mock_mcp_manager.aingest.assert_not_awaited()


async def test_mcp_create_workspace_uses_manager_registry(mock_mcp_manager) -> None:
    result = await mcp_server.mcp_app.call_tool(
        "create_workspace",
        {"workspace": "New Workspace", "display_name": "New Workspace"},
    )

    body = _tool_json(result)
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
    result = await mcp_server.mcp_app.call_tool(
        "delete_workspace",
        {"workspace": "Old Workspace", "keep_files": True, "dry_run": True},
    )

    body = _tool_json(result)
    assert body["workspace"] == "old_workspace"
    assert body["deleted"] is False
    mock_mcp_manager.areset.assert_awaited_once_with(
        workspace="Old Workspace",
        keep_files=True,
        dry_run=True,
    )


async def test_mcp_ingest_forwards_document_metadata(
    mock_mcp_manager, test_config: DlightragConfig
) -> None:
    mock_mcp_manager.astart_ingest_job = AsyncMock(
        return_value={
            "job_id": "job-1",
            "workspace": "finance",
            "source_type": "local",
            "status": "queued",
        }
    )
    path = str(test_config.input_dir_path / "finance" / "report.pdf")

    result = await mcp_server.mcp_app.call_tool(
        "ingest",
        {
            "source_type": "local",
            "path": path,
            "workspace": "finance",
            "title": "Quarterly Report",
            "author": "Ada",
            "metadata": {"department": "Finance"},
            "metadata_policy": "reject_unknown",
        },
    )

    assert _tool_json(result)["job_id"] == "job-1"
    mock_mcp_manager.astart_ingest_job.assert_awaited_once_with(
        "finance",
        source_type="local",
        path=path,
        title="Quarterly Report",
        author="Ada",
        metadata={"department": "Finance"},
        metadata_policy="reject_unknown",
    )
    mock_mcp_manager.aingest.assert_not_awaited()


async def test_mcp_rejects_local_path_outside_input_dir(mock_mcp_manager) -> None:
    result = await mcp_server.mcp_app.call_tool(
        "ingest",
        {"source_type": "local", "path": "/tmp/report.pdf"},
    )

    assert "under input_dir" in _tool_text(result)
    mock_mcp_manager.astart_ingest_job.assert_not_awaited()
    mock_mcp_manager.aingest.assert_not_awaited()


async def test_mcp_rejects_local_path_outside_workspace_input_dir(
    mock_mcp_manager, test_config: DlightragConfig
) -> None:
    result = await mcp_server.mcp_app.call_tool(
        "ingest",
        {
            "source_type": "local",
            "path": str(test_config.input_dir_path / "default" / "report.pdf"),
            "workspace": "finance",
        },
    )

    assert "under input_dir" in _tool_text(result)
    mock_mcp_manager.astart_ingest_job.assert_not_awaited()
    mock_mcp_manager.aingest.assert_not_awaited()


async def test_mcp_remote_prefix_ingest_starts_background_job(mock_mcp_manager) -> None:
    mock_mcp_manager.astart_ingest_job = AsyncMock(
        return_value={
            "job_id": "job-1",
            "workspace": "default",
            "source_type": "s3",
            "status": "queued",
        }
    )

    result = await mcp_server.mcp_app.call_tool(
        "ingest",
        {
            "source_type": "s3",
            "bucket": "bucket",
            "prefix": "docs/",
            "workspace": "default",
        },
    )

    assert _tool_json(result) == {
        "job_id": "job-1",
        "workspace": "default",
        "source_type": "s3",
        "status": "queued",
    }
    mock_mcp_manager.astart_ingest_job.assert_awaited_once_with(
        "default",
        source_type="s3",
        bucket="bucket",
        prefix="docs/",
    )
    mock_mcp_manager.aingest.assert_not_awaited()


async def test_mcp_local_directory_ingest_starts_background_job(
    mock_mcp_manager, test_config: DlightragConfig
) -> None:
    docs_dir = test_config.input_dir_path / "default" / "docs"
    docs_dir.mkdir(parents=True)
    mock_mcp_manager.astart_ingest_job = AsyncMock(
        return_value={
            "job_id": "job-1",
            "workspace": "default",
            "source_type": "local",
            "status": "queued",
        }
    )

    result = await mcp_server.mcp_app.call_tool(
        "ingest",
        {
            "source_type": "local",
            "path": str(docs_dir),
            "workspace": "default",
        },
    )

    assert _tool_json(result)["job_id"] == "job-1"
    mock_mcp_manager.astart_ingest_job.assert_awaited_once_with(
        "default",
        source_type="local",
        path=str(docs_dir),
    )
    mock_mcp_manager.aingest.assert_not_awaited()


async def test_mcp_single_s3_key_defaults_to_background_job(mock_mcp_manager) -> None:
    mock_mcp_manager.astart_ingest_job = AsyncMock(
        return_value={
            "job_id": "job-1",
            "workspace": "default",
            "source_type": "s3",
            "status": "queued",
        }
    )

    result = await mcp_server.mcp_app.call_tool(
        "ingest",
        {
            "source_type": "s3",
            "bucket": "bucket",
            "key": "docs/file.pdf",
            "workspace": "default",
        },
    )

    assert _tool_json(result)["job_id"] == "job-1"
    mock_mcp_manager.astart_ingest_job.assert_awaited_once_with(
        "default",
        source_type="s3",
        bucket="bucket",
        key="docs/file.pdf",
    )
    mock_mcp_manager.aingest.assert_not_awaited()


async def test_mcp_ingest_job_status_reads_manager_job(mock_mcp_manager) -> None:
    mock_mcp_manager.get_ingest_job = AsyncMock(
        return_value={
            "job_id": "job-1",
            "status": "running",
            "processed_items": 64,
        }
    )

    result = await mcp_server.mcp_app.call_tool("ingest_job_status", {"job_id": "job-1"})

    assert _tool_json(result) == {
        "job_id": "job-1",
        "status": "running",
        "processed_items": 64,
    }
    mock_mcp_manager.get_ingest_job.assert_awaited_once_with("job-1")


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

    result = await mcp_server.mcp_app.call_tool(
        "answer",
        {
            "query": "Follow up",
            "workspaces": ["default"],
            "top_k": 8,
            "chunk_top_k": 12,
            "answer_context_top_k": 4,
            "conversation_history": [
                {"role": "user", "content": [{"type": "text", "text": "Previous"}]}
            ],
            "query_images": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
            ],
            "session_id": "session-1",
            "referenced_image_ids": ["img_1"],
            "filters": {"doc_title": "Manual"},
        },
    )

    body = _tool_json(result)
    assert body["answer"] == "Answer [1-1]."
    assert body["contexts"]["chunks"][0]["image_url"] == "/images/default/c1?size=full"
    assert "image_data" not in body["contexts"]["chunks"][0]
    assert body["sources"][0]["id"] == "1"

    call_kwargs = mock_mcp_manager.aanswer.call_args.kwargs
    assert call_kwargs["workspaces"] == ["default"]
    assert call_kwargs["top_k"] == 8
    assert call_kwargs["chunk_top_k"] == 12
    assert call_kwargs["answer_context_top_k"] == 4
    assert call_kwargs["conversation_history"] == [
        {"role": "user", "content": [{"type": "text", "text": "Previous"}]}
    ]
    assert call_kwargs["query_images"] == [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}
    ]
    assert call_kwargs["session_id"] == "session-1"
    assert call_kwargs["referenced_image_ids"] == ["img_1"]
    assert call_kwargs["filters"].doc_title == "Manual"
