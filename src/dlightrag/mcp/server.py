# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""MCP server for agent integration (stdio + streamable-http).

Entry point: dlightrag-mcp
Primarily used by DeerFlow and other MCP-compatible agents for
retrieve() + lightweight ingest().
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from dlightrag.config import DlightragConfig, get_config, load_config, set_config
from dlightrag.core.client_payloads import (
    answer_payload,
    metadata_filter_from_payload,
    retrieval_payload,
)
from dlightrag.core.ingest_policy import should_wait_for_ingest
from dlightrag.core.scope import RequestScope, current_request_scope, request_scope_context
from dlightrag.core.servicemanager import RAGServiceManager

logger = logging.getLogger(__name__)
_METADATA_POLICY_VALUES = ("validate", "reject_unknown", "store_only")
_QUERY_IMAGE_ITEMS_SCHEMA = {"anyOf": [{"type": "string"}, {"type": "object"}]}

server = Server(
    "dlightrag",
    version=__import__("dlightrag").__version__,
)


def _get_config() -> DlightragConfig:
    return get_config()


_manager: RAGServiceManager | None = None


async def _ensure_manager() -> RAGServiceManager:
    global _manager
    if _manager is None:
        _manager = await RAGServiceManager.create()
    return _manager


def _json_content(payload: dict[str, Any]) -> list[TextContent]:
    return [TextContent(type="text", text=json.dumps(payload, default=str, indent=2))]


def _normalize_workspace_argument(arguments: dict[str, Any]) -> tuple[str, str]:
    from dlightrag.utils import normalize_workspace, validate_workspace_name

    label = validate_workspace_name(str(arguments.get("workspace") or ""))
    display_name = validate_workspace_name(str(arguments.get("display_name") or label))
    return normalize_workspace(label), display_name


# ═══════════════════════════════════════════════════════════════════
# Tool definitions
# ═══════════════════════════════════════════════════════════════════


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools."""
    return [
        Tool(
            name="retrieve",
            description="Query the RAG knowledge base for relevant information. Supports structured metadata filters for precise document lookups.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top results to return",
                    },
                    "chunk_top_k": {
                        "type": "integer",
                        "description": "Vector chunk candidate count override",
                    },
                    "workspaces": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Workspace names to search. Omit to search default workspace.",
                    },
                    "filters": {
                        "type": "object",
                        "description": "Metadata filters for structured queries (filename, doc_author, etc.)",
                        "properties": {
                            "filename": {"type": "string"},
                            "filename_pattern": {
                                "type": "string",
                                "description": "Explicit SQL ILIKE pattern",
                            },
                            "file_extension": {"type": "string"},
                            "doc_title": {"type": "string"},
                            "doc_author": {"type": "string"},
                            "custom": {"type": "object"},
                        },
                    },
                    "query_images": {
                        "type": "array",
                        "maxItems": 3,
                        "items": _QUERY_IMAGE_ITEMS_SCHEMA,
                        "description": "User-attached image URLs or data URI blocks (max 3) for visual retrieval",
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Session id for image memory.",
                    },
                    "referenced_image_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Previously stored image ids to include in the query.",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="ingest",
            description="Ingest document(s) into the RAG knowledge base.",
            inputSchema={
                "type": "object",
                "properties": {
                    "source_type": {
                        "type": "string",
                        "enum": ["local", "azure_blob", "s3"],
                        "description": "Type of data source",
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory path (for local source)",
                    },
                    "container_name": {
                        "type": "string",
                        "description": "Azure Blob container name",
                    },
                    "blob_path": {
                        "type": "string",
                        "description": "Specific blob path (azure_blob)",
                    },
                    "bucket": {
                        "type": "string",
                        "description": "S3 bucket name (s3)",
                    },
                    "key": {
                        "type": "string",
                        "description": "S3 object key — single object or prefix (s3)",
                    },
                    "prefix": {
                        "type": "string",
                        "description": "Path/blob/key prefix filter (azure_blob, s3)",
                    },
                    "replace": {
                        "type": "boolean",
                        "default": False,
                        "description": "Replace existing documents",
                    },
                    "workspace": {
                        "type": "string",
                        "description": "Target workspace. Omit for default workspace.",
                    },
                    "title": {
                        "type": "string",
                        "description": "Optional document title metadata.",
                    },
                    "author": {
                        "type": "string",
                        "description": "Optional document author metadata.",
                    },
                    "metadata": {
                        "type": "object",
                        "description": "User metadata to attach to ingested documents.",
                    },
                    "metadata_policy": {
                        "type": "string",
                        "enum": list(_METADATA_POLICY_VALUES),
                        "description": "How undeclared user metadata fields are handled.",
                    },
                    "wait": {
                        "type": "boolean",
                        "description": "Wait for completion. Defaults to true for single files/objects and false for batch-shaped ingests.",
                    },
                },
                "required": ["source_type"],
            },
        ),
        Tool(
            name="ingest_job_status",
            description="Return the status of an ingest job.",
            inputSchema={
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "string",
                        "description": "Ingest job id returned by the ingest tool.",
                    }
                },
                "required": ["job_id"],
            },
        ),
        Tool(
            name="answer",
            description="Ask a question and get an LLM-generated answer backed by retrieved context from the knowledge base.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question to answer",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Retrieval candidate count override for this answer",
                    },
                    "chunk_top_k": {
                        "type": "integer",
                        "description": "Vector chunk candidate count override for this answer",
                    },
                    "answer_candidate_top_k": {
                        "type": "integer",
                        "description": "Answer retrieval candidates fetched before final prompt packing",
                    },
                    "answer_context_top_k": {
                        "type": "integer",
                        "description": "Maximum chunks included in the final answer prompt",
                    },
                    "workspaces": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Workspace names to search. Omit to search default workspace.",
                    },
                    "filters": {
                        "type": "object",
                        "description": "Metadata filters for structured queries (filename, doc_author, etc.)",
                        "properties": {
                            "filename": {"type": "string"},
                            "filename_pattern": {
                                "type": "string",
                                "description": "Explicit SQL ILIKE pattern",
                            },
                            "file_extension": {"type": "string"},
                            "doc_title": {"type": "string"},
                            "doc_author": {"type": "string"},
                            "custom": {"type": "object"},
                        },
                    },
                    "conversation_history": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Prior conversation turns as role/content objects.",
                    },
                    "query_images": {
                        "type": "array",
                        "maxItems": 3,
                        "items": _QUERY_IMAGE_ITEMS_SCHEMA,
                        "description": "User-attached image URLs or data URI blocks (max 3) for visual answer context.",
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Session id for image memory.",
                    },
                    "referenced_image_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Previously stored image ids to include in the answer.",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="list_workspaces",
            description="List all registered workspaces.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="create_workspace",
            description="Create an empty DlightRAG workspace.",
            inputSchema={
                "type": "object",
                "properties": {
                    "workspace": {
                        "type": "string",
                        "description": "Workspace name to create.",
                    },
                    "display_name": {
                        "type": "string",
                        "description": "Optional user-facing display name.",
                    },
                },
                "required": ["workspace"],
            },
        ),
        Tool(
            name="delete_workspace",
            description="Delete/reset one DlightRAG workspace.",
            inputSchema={
                "type": "object",
                "properties": {
                    "workspace": {
                        "type": "string",
                        "description": "Workspace name to delete.",
                    },
                    "keep_files": {
                        "type": "boolean",
                        "default": False,
                        "description": "Keep source files on disk.",
                    },
                    "dry_run": {
                        "type": "boolean",
                        "default": False,
                        "description": "Report what would be deleted without mutating storage.",
                    },
                },
                "required": ["workspace"],
            },
        ),
        Tool(
            name="list_files",
            description="List all documents ingested in the knowledge base.",
            inputSchema={
                "type": "object",
                "properties": {
                    "workspace": {
                        "type": "string",
                        "description": "Workspace to list files from. Omit for default workspace.",
                    },
                },
            },
        ),
        Tool(
            name="delete_files",
            description="Delete documents from the knowledge base.",
            inputSchema={
                "type": "object",
                "properties": {
                    "filenames": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of filenames to delete",
                    },
                    "file_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of file paths to delete",
                    },
                    "workspace": {
                        "type": "string",
                        "description": "Workspace to delete from. Omit for default workspace.",
                    },
                },
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle MCP tool calls."""
    try:
        if name == "retrieve":
            manager = await _ensure_manager()
            if "mode" in arguments:
                raise ValueError("MCP retrieve does not accept 'mode'; DlightRAG always uses mix")
            kwargs: dict[str, Any] = {}
            filters = metadata_filter_from_payload(arguments.get("filters"))
            if filters is not None:
                kwargs["filters"] = filters
            if arguments.get("query_images"):
                qi = arguments["query_images"]
                if len(qi) > 3:
                    raise ValueError("Maximum 3 query_images per request")
                kwargs["query_images"] = qi
            if arguments.get("session_id"):
                kwargs["session_id"] = arguments["session_id"]
            if arguments.get("referenced_image_ids"):
                kwargs["referenced_image_ids"] = arguments["referenced_image_ids"]
            scope = current_request_scope().for_workspaces(arguments.get("workspaces"))
            result = await manager.aretrieve(
                arguments["query"],
                workspaces=arguments.get("workspaces"),
                top_k=arguments.get("top_k"),
                chunk_top_k=arguments.get("chunk_top_k"),
                scope=scope,
                **kwargs,
            )
            return _json_content(retrieval_payload(result))

        if name == "answer":
            manager = await _ensure_manager()
            if "mode" in arguments:
                raise ValueError("MCP answer does not accept 'mode'; DlightRAG always uses mix")
            kwargs = {}
            filters = metadata_filter_from_payload(arguments.get("filters"))
            if filters is not None:
                kwargs["filters"] = filters
            if arguments.get("query_images"):
                qi = arguments["query_images"]
                if len(qi) > 3:
                    raise ValueError("Maximum 3 query_images per request")
                kwargs["query_images"] = qi
            if arguments.get("session_id"):
                kwargs["session_id"] = arguments["session_id"]
            if arguments.get("referenced_image_ids"):
                kwargs["referenced_image_ids"] = arguments["referenced_image_ids"]
            scope = current_request_scope().for_workspaces(arguments.get("workspaces"))
            result = await manager.aanswer(
                arguments["query"],
                conversation_history=arguments.get("conversation_history"),
                workspaces=arguments.get("workspaces"),
                top_k=arguments.get("top_k"),
                chunk_top_k=arguments.get("chunk_top_k"),
                answer_candidate_top_k=arguments.get("answer_candidate_top_k"),
                answer_context_top_k=arguments.get("answer_context_top_k"),
                scope=scope,
                **kwargs,
            )
            return _json_content(answer_payload(result))

        if name == "list_workspaces":
            manager = await _ensure_manager()
            records = await manager.list_workspace_records()
            return _json_content(
                {
                    "workspaces": [row["workspace"] for row in records],
                    "records": records,
                }
            )

        if name == "create_workspace":
            manager = await _ensure_manager()
            workspace, display_name = _normalize_workspace_argument(arguments)
            existing = await manager.list_workspaces()
            if workspace in existing:
                raise ValueError(f"Workspace '{display_name}' already exists")
            await manager.acreate_workspace(workspace, display_name=display_name)
            return _json_content(
                {
                    "workspace": workspace,
                    "display_name": display_name,
                    "created": True,
                }
            )

        if name == "delete_workspace":
            manager = await _ensure_manager()
            from dlightrag.utils import normalize_workspace, validate_workspace_name

            label = validate_workspace_name(str(arguments.get("workspace") or ""))
            workspace = normalize_workspace(label)
            dry_run = bool(arguments.get("dry_run", False))
            result = await manager.areset(
                workspace=label,
                keep_files=bool(arguments.get("keep_files", False)),
                dry_run=dry_run,
            )
            return _json_content(
                {
                    "workspace": workspace,
                    "deleted": not dry_run,
                    "result": result,
                }
            )

        if name == "ingest":
            manager = await _ensure_manager()
            ws = arguments.get("workspace") or _get_config().workspace
            source_type = arguments["source_type"]
            kwargs: dict[str, Any] = {}
            if source_type == "local":
                kwargs["path"] = arguments.get("path", ".")
            elif source_type == "azure_blob":
                if arguments.get("blob_path") and arguments.get("prefix") is not None:
                    raise ValueError("'blob_path' and 'prefix' are mutually exclusive")
                kwargs["container_name"] = arguments.get("container_name", "")
                if arguments.get("blob_path"):
                    kwargs["blob_path"] = arguments["blob_path"]
                if arguments.get("prefix") is not None:
                    kwargs["prefix"] = arguments["prefix"]
            elif source_type == "s3":
                if arguments.get("key") and arguments.get("prefix") is not None:
                    raise ValueError("'key' and 'prefix' are mutually exclusive")
                kwargs["bucket"] = arguments.get("bucket", "")
                if arguments.get("key"):
                    kwargs["key"] = arguments["key"]
                if arguments.get("prefix") is not None:
                    kwargs["prefix"] = arguments["prefix"]
            if arguments.get("replace") is not None:
                kwargs["replace"] = arguments["replace"]
            if arguments.get("title") is not None:
                kwargs["title"] = arguments["title"]
            if arguments.get("author") is not None:
                kwargs["author"] = arguments["author"]
            if arguments.get("metadata") is not None:
                kwargs["metadata"] = arguments["metadata"]
            if arguments.get("metadata_policy") is not None:
                metadata_policy = str(arguments["metadata_policy"])
                if metadata_policy not in _METADATA_POLICY_VALUES:
                    raise ValueError(f"Invalid metadata_policy: {metadata_policy}")
                kwargs["metadata_policy"] = metadata_policy
            wait = should_wait_for_ingest(
                source_type=source_type,
                path=kwargs.get("path"),
                prefix=kwargs.get("prefix"),
                wait=arguments.get("wait"),
            )
            if wait:
                result = await manager.aingest(ws, source_type=source_type, **kwargs)
            else:
                result = await manager.astart_ingest_job(ws, source_type=source_type, **kwargs)
            return _json_content(result)

        if name == "ingest_job_status":
            manager = await _ensure_manager()
            job_id = str(arguments.get("job_id") or "")
            if not job_id:
                raise ValueError("job_id is required")
            result = await manager.get_ingest_job(job_id)
            if result is None:
                raise ValueError(f"Ingest job not found: {job_id}")
            return _json_content(result)

        if name == "list_files":
            manager = await _ensure_manager()
            ws = arguments.get("workspace") or _get_config().workspace
            try:
                files = await manager.list_ingested_files(ws)
            except NotImplementedError:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {"error": "File listing is not supported in unified RAG mode"},
                        ),
                    )
                ]
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {"files": files, "count": len(files), "workspace": ws}, default=str
                    ),
                )
            ]

        if name == "delete_files":
            manager = await _ensure_manager()
            ws = arguments.get("workspace") or _get_config().workspace
            try:
                results = await manager.delete_files(
                    ws, filenames=arguments.get("filenames"), file_paths=arguments.get("file_paths")
                )
            except NotImplementedError:
                return [
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {"error": "File deletion is not supported in unified RAG mode"},
                        ),
                    )
                ]
            return [
                TextContent(
                    type="text", text=json.dumps({"results": results, "workspace": ws}, default=str)
                )
            ]

        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        logger.exception(f"MCP tool '{name}' failed: {e}")
        return [TextContent(type="text", text=f"Error: {e}")]


# ═══════════════════════════════════════════════════════════════════
# Server startup
# ═══════════════════════════════════════════════════════════════════


async def run_stdio() -> None:
    """Run MCP server over stdio transport."""
    await _ensure_manager()
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )
    finally:
        if _manager is not None:
            await _manager.close()


async def run_streamable_http(host: str, port: int) -> None:
    """Run MCP server over streamable-http transport.

    Bearer auth is enforced when ``auth_mode`` is not ``"none"``.
    Both simple (``DLIGHTRAG_API_AUTH_TOKEN``) and JWT (``DLIGHTRAG_JWT_SECRET``)
    modes are supported via the shared ``verify_bearer_token`` dispatcher.
    Without auth, the server runs open — caller is responsible for binding
    to loopback or trusted network only. We log a loud warning in that case.
    """
    import uvicorn
    from fastapi import HTTPException
    from mcp.server.streamable_http import StreamableHTTPServerTransport
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse
    from starlette.routing import Mount

    cfg = _get_config()
    has_auth = cfg.auth_mode != "none"

    if not has_auth:
        if host not in ("127.0.0.1", "localhost", "::1"):
            logger.warning(
                "=" * 72 + "\nMCP streamable-http on host=%s:%d WITHOUT auth (auth_mode='none').\n"
                "If this bind reaches a non-loopback network, ANY client can call\n"
                "ingest, delete_files, retrieve, answer against EVERY workspace.\n"
                "Safe configurations:\n"
                "  (a) Set DLIGHTRAG_AUTH_MODE=simple + DLIGHTRAG_API_AUTH_TOKEN\n"
                "      — bearer token guards MCP and REST (single secret).\n"
                "  (b) Set DLIGHTRAG_AUTH_MODE=jwt + DLIGHTRAG_JWT_SECRET\n"
                "      — JWT bearer auth guards MCP and REST (same secret).\n"
                "  (c) Bind to 127.0.0.1 (loopback only).\n"
                "  (d) Map host port to 127.0.0.1 only (compose: '127.0.0.1:8101:8101')\n"
                "      — safe even with container-internal 0.0.0.0.\n" + "=" * 72,
                host,
                port,
            )
        else:
            logger.info("MCP streamable-http on %s:%d (loopback, no auth required)", host, port)

    class BearerAuthMiddleware(BaseHTTPMiddleware):
        """Enforce Bearer auth on every request to the MCP transport.

        Delegates to ``verify_bearer_token`` for auth-mode dispatch.
        No-op when ``auth_mode='none'`` (operator opted out).
        """

        async def dispatch(self, request, call_next):
            if not has_auth:
                with request_scope_context(RequestScope.anonymous()):
                    return await call_next(request)
            header = request.headers.get("Authorization", "")
            if not header.startswith("Bearer "):
                return JSONResponse(
                    {"error": "Missing or invalid Authorization header"}, status_code=401
                )
            try:
                from dlightrag.api.auth import verify_bearer_token

                user = verify_bearer_token(
                    header[7:],
                    cfg,
                    default_user_id=request.headers.get("X-User-Id", "anonymous"),
                )
            except HTTPException as exc:
                return JSONResponse({"error": exc.detail}, status_code=exc.status_code)
            with request_scope_context(RequestScope.from_user(user)):
                return await call_next(request)

    await _ensure_manager()

    transport = StreamableHTTPServerTransport(mcp_session_id=None)

    starlette_app = Starlette(
        routes=[Mount("/mcp", app=transport.handle_request)],
        middleware=[Middleware(BearerAuthMiddleware)],
    )

    config = uvicorn.Config(
        starlette_app,
        host=host,
        port=port,
        log_level="info",
    )
    uv_server = uvicorn.Server(config)

    try:
        async with transport.connect() as (read_stream, write_stream):
            server_task = asyncio.create_task(
                server.run(
                    read_stream,
                    write_stream,
                    server.create_initialization_options(),
                )
            )
            await uv_server.serve()
            server_task.cancel()
    finally:
        if _manager is not None:
            await _manager.close()


def main() -> None:
    """Entry point for dlightrag-mcp."""
    import argparse

    parser = argparse.ArgumentParser(description="dlightrag MCP server")
    parser.add_argument("--env-file", help="Path to .env configuration file")
    args = parser.parse_args()

    if args.env_file:
        config = load_config(args.env_file)
        set_config(config)
    else:
        config = _get_config()

    logging.basicConfig(level=getattr(logging, config.log_level.upper(), logging.INFO))

    if config.mcp_transport == "streamable-http":
        logger.info(f"Starting MCP server (streamable-http) on {config.mcp_host}:{config.mcp_port}")
        asyncio.run(run_streamable_http(config.mcp_host, config.mcp_port))
    else:
        logger.info("Starting MCP server (stdio)")
        asyncio.run(run_stdio())


__all__ = ["main", "server"]
