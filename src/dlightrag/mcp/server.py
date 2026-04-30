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

from dlightrag.citations.processor import CitationProcessor
from dlightrag.citations.source_builder import build_sources
from dlightrag.config import DlightragConfig, get_config
from dlightrag.core.servicemanager import RAGServiceManager

logger = logging.getLogger(__name__)

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
                    "mode": {
                        "type": "string",
                        "enum": ["local", "global", "hybrid", "naive", "mix"],
                        "default": "mix",
                        "description": "Retrieval mode",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top results to return",
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
                                "description": "SQL ILIKE pattern",
                            },
                            "file_extension": {"type": "string"},
                            "doc_title": {"type": "string"},
                            "doc_author": {"type": "string"},
                            "custom": {"type": "object"},
                        },
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
                },
                "required": ["source_type"],
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
                    "mode": {
                        "type": "string",
                        "enum": ["local", "global", "hybrid", "naive", "mix"],
                        "default": "mix",
                        "description": "Retrieval mode",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top results to retrieve",
                    },
                    "workspaces": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Workspace names to search. Omit to search default workspace.",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="list_workspaces",
            description="List all available workspaces with ingested data.",
            inputSchema={"type": "object", "properties": {}},
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
            kwargs: dict[str, Any] = {}
            if arguments.get("filters"):
                from dlightrag.core.retrieval.models import MetadataFilter

                kwargs["filters"] = MetadataFilter(**arguments["filters"])
            result = await manager.aretrieve(
                arguments["query"],
                workspaces=arguments.get("workspaces"),
                mode=arguments.get("mode", "mix"),
                top_k=arguments.get("top_k"),
                **kwargs,
            )
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "answer": result.answer,
                            "contexts": result.contexts,
                            "sources": [s.model_dump() for s in build_sources(result.contexts)],
                        },
                        default=str,
                        indent=2,
                    ),
                )
            ]

        if name == "answer":
            manager = await _ensure_manager()
            result = await manager.aanswer(
                arguments["query"],
                workspaces=arguments.get("workspaces"),
                mode=arguments.get("mode", "mix"),
                top_k=arguments.get("top_k"),
            )
            # Build cited-only sources via CitationProcessor
            flat_contexts: list[dict[str, Any]] = []
            for items in result.contexts.values():
                if isinstance(items, list):
                    flat_contexts.extend(items)
            all_sources = build_sources(result.contexts)
            if result.answer and flat_contexts:
                processor = CitationProcessor(contexts=flat_contexts, available_sources=all_sources)
                cited = processor.process(result.answer)
                sources = cited.sources
            else:
                sources = []
            return [
                TextContent(
                    type="text",
                    text=json.dumps(
                        {
                            "answer": result.answer,
                            "contexts": result.contexts,
                            "references": [r.model_dump() for r in result.references]
                            if result.references
                            else [],
                            "sources": [s.model_dump() for s in sources],
                        },
                        default=str,
                        indent=2,
                    ),
                )
            ]

        if name == "list_workspaces":
            manager = await _ensure_manager()
            ws_list = await manager.list_workspaces()
            return [TextContent(type="text", text=json.dumps({"workspaces": ws_list}, indent=2))]

        if name == "ingest":
            manager = await _ensure_manager()
            ws = arguments.get("workspace") or _get_config().workspace
            source_type = arguments["source_type"]
            kwargs: dict[str, Any] = {}
            if source_type == "local":
                kwargs["path"] = arguments.get("path", ".")
            elif source_type == "azure_blob":
                kwargs["container_name"] = arguments.get("container_name", "")
                if arguments.get("blob_path"):
                    kwargs["blob_path"] = arguments["blob_path"]
                if arguments.get("prefix") is not None:
                    kwargs["prefix"] = arguments["prefix"]
            elif source_type == "s3":
                kwargs["bucket"] = arguments.get("bucket", "")
                kwargs["key"] = arguments.get("key", "")
                if arguments.get("prefix") is not None:
                    kwargs["prefix"] = arguments["prefix"]
            if arguments.get("replace") is not None:
                kwargs["replace"] = arguments["replace"]
            result = await manager.aingest(ws, source_type=source_type, **kwargs)
            return [TextContent(type="text", text=json.dumps(result, default=str))]

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

    Bearer-token auth is enforced when ``DLIGHTRAG_API_AUTH_TOKEN`` is set
    (the same secret as the REST API). Without a token, the server runs
    open — caller is responsible for binding to loopback or trusted network
    only. We log a loud warning in that case and refuse the unsafe combo
    of ``host=0.0.0.0`` + no token.
    """
    import secrets

    import uvicorn
    from mcp.server.streamable_http import StreamableHTTPServerTransport
    from starlette.applications import Starlette
    from starlette.middleware import Middleware
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import JSONResponse
    from starlette.routing import Mount

    cfg = _get_config()
    token = cfg.api_auth_token

    # Refuse the misconfiguration that lets anyone on a reachable network
    # delete every workspace's data.
    if not token and host not in ("127.0.0.1", "localhost", "::1"):
        raise RuntimeError(
            f"MCP streamable-http on host={host!r} requires DLIGHTRAG_API_AUTH_TOKEN. "
            "Without auth, any client reaching the bind address can call ingest, "
            "delete_files, retrieve, and answer. Either set the token or restrict "
            "the bind to loopback (127.0.0.1)."
        )
    if not token:
        logger.warning(
            "MCP streamable-http running on %s:%d without DLIGHTRAG_API_AUTH_TOKEN. "
            "Loopback-only is safe for local agents; exposing this to other hosts "
            "without auth is not.",
            host,
            port,
        )

    class BearerAuthMiddleware(BaseHTTPMiddleware):
        """Enforce Bearer auth on every request to the MCP transport.

        No-op when token is None (operator opted out). Constant-time
        comparison defends against timing side-channels.
        """

        async def dispatch(self, request, call_next):
            if not token:
                return await call_next(request)
            header = request.headers.get("Authorization", "")
            if not header.startswith("Bearer "):
                return JSONResponse(
                    {"error": "Missing or invalid Authorization header"}, status_code=401
                )
            if not secrets.compare_digest(header[7:], token):
                return JSONResponse({"error": "Invalid token"}, status_code=403)
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

    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(description="dlightrag MCP server")
    parser.add_argument("--env-file", help="Path to .env configuration file")
    args = parser.parse_args()

    if args.env_file:
        load_dotenv(args.env_file, override=True)

    config = _get_config()

    logging.basicConfig(level=getattr(logging, config.log_level.upper(), logging.INFO))

    if config.mcp_transport == "streamable-http":
        logger.info(f"Starting MCP server (streamable-http) on {config.mcp_host}:{config.mcp_port}")
        asyncio.run(run_streamable_http(config.mcp_host, config.mcp_port))
    else:
        logger.info("Starting MCP server (stdio)")
        asyncio.run(run_stdio())


__all__ = ["main", "server"]
