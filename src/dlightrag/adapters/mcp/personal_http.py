# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Bounded SDK Streamable HTTP discovery with per-request DNS address pinning."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable
from contextvars import ContextVar
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx2
from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client
from pydantic import SecretStr

from dlightrag.application.connections import ConnectionPolicy, ConnectionsError
from dlightrag.engine.agent.tools import ToolResult
from dlightrag.engine.network_admission import (
    _normalize_host_patterns,
    _resolve_public_target,
    validate_credential_free_query,
)

_PRIVATE_SESSION = ContextVar("private_mcp_session", default=False)


class _PrivateSdkLogs(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # SDK diagnostics may contain entire remote messages and OAuth values.
        # The owner module emits only redacted status, not SDK diagnostics.
        return not _PRIVATE_SESSION.get()


def _protect_sdk_logs() -> None:
    for name in tuple(logging.Logger.manager.loggerDict):
        if name.startswith(("mcp.", "httpx2", "httpcore")):
            logger = logging.getLogger(name)
            if not any(isinstance(item, _PrivateSdkLogs) for item in logger.filters):
                logger.addFilter(_PrivateSdkLogs())


class _BoundedStream(httpx2.AsyncByteStream):
    def __init__(self, stream: httpx2.AsyncByteStream, limit: int) -> None:
        self._stream = stream
        self._limit = limit

    async def __aiter__(self) -> AsyncIterator[bytes]:
        size = 0
        async for chunk in self._stream:
            size += len(chunk)
            if size > self._limit:
                raise ConnectionsError("MCP response exceeds policy")
            yield chunk

    async def aclose(self) -> None:
        await self._stream.aclose()


class _AdmittedTransport(httpx2.AsyncBaseTransport):
    def __init__(
        self,
        endpoint: str,
        bearer: SecretStr | None,
        policy: ConnectionPolicy,
        inner: httpx2.AsyncBaseTransport,
        *,
        foreground: bool = False,
        oauth: bool = False,
    ) -> None:
        self._oauth = oauth
        self._endpoint = httpx2.URL(endpoint)
        self._bearer = bearer
        self._policy = policy
        self._inner = inner
        self.authentication_failed = False
        self._foreground = foreground
        self._effect_sent = False
        self._post_ids: set[str] = set()

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        # The SDK can follow same-origin POST redirects and resume SSE. A
        # foreground session permits each JSON-RPC request once, regardless of
        # HTTP status/outcome. Mark BEFORE transport I/O, not after its response.
        if self._foreground:
            if request.method == "GET":
                return httpx2.Response(405)
            if request.method == "POST":
                body = json.loads(await request.aread())
                if "id" in body:
                    identity = str(body["id"])
                    if identity in self._post_ids:
                        raise ConnectionsError("MCP request replay blocked")
                    self._post_ids.add(identity)
                if body.get("method") == "tools/call":
                    if self._effect_sent:
                        raise ConnectionsError("MCP effect replay blocked")
                    self._effect_sent = True
        url = request.url
        if (
            (
                not self._oauth
                and (url.scheme, url.host, url.port)
                != (self._endpoint.scheme, self._endpoint.host, self._endpoint.port)
            )
            or url.userinfo
            or url.fragment
            or (
                (self._policy.require_https or self._endpoint.scheme == "https")
                and url.scheme != "https"
            )
        ):
            raise ConnectionsError("MCP target rejected by network policy")
        validate_credential_free_query(str(url))
        async with asyncio.timeout(self._policy.connect_timeout):
            target = await _resolve_public_target(
                str(url),
                allow_private_hosts=_normalize_host_patterns(self._policy.allow_private_hosts),
            )
        parts = urlsplit(str(url))
        address = target.addresses[0]
        host = f"[{address}]" if ":" in address else address
        pinned = urlunsplit((parts.scheme, f"{host}:{target.port}", parts.path, parts.query, ""))
        # Never inherit browser credentials, proxy headers, cookies, or a remote
        # argument-to-header annotation. Only MCP protocol headers are forwarded.
        allowed = {
            "accept",
            "content-type",
            "content-length",
            "mcp-protocol-version",
            "mcp-method",
            "mcp-name",
            "mcp-session-id",
            "last-event-id",
        }
        headers = {name: value for name, value in request.headers.items() if name in allowed}
        headers.update(
            {"host": url.netloc.decode(), "connection": "close", "accept-encoding": "identity"}
        )
        if self._oauth and "authorization" in request.headers:
            authorization = request.headers["authorization"]
            if authorization.startswith("Bearer ") and (url.scheme, url.host, url.port) != (
                self._endpoint.scheme,
                self._endpoint.host,
                self._endpoint.port,
            ):
                raise ConnectionsError("OAuth credential audience rejected")
            headers["authorization"] = authorization
        if self._bearer is not None:
            headers["authorization"] = "Bearer " + self._bearer.get_secret_value()
        extensions = {**request.extensions, "sni_hostname": target.host}
        pinned_request = httpx2.Request(
            request.method, pinned, headers=headers, stream=request.stream, extensions=extensions
        )
        response = await self._inner.handle_async_request(pinned_request)
        if response.status_code in {401, 403}:
            self.authentication_failed = True
        if response.is_stream_consumed and len(response.content) > self._policy.max_response_bytes:
            await response.aclose()
            raise ConnectionsError("MCP response exceeds policy")
        if response.headers.get("content-encoding", "identity") != "identity":
            await response.aclose()
            raise ConnectionsError("MCP compressed response rejected")
        if not isinstance(response.stream, httpx2.AsyncByteStream):
            await response.aclose()
            raise ConnectionsError("MCP response stream rejected")
        response.stream = _BoundedStream(response.stream, self._policy.max_response_bytes)
        return response

    async def aclose(self) -> None:
        await self._inner.aclose()


class PersonalMcpClient:
    def __init__(
        self, *, transport_factory: Callable[[], httpx2.AsyncBaseTransport] | None = None
    ) -> None:
        self._transport_factory = transport_factory or (
            lambda: httpx2.AsyncHTTPTransport(
                retries=0,
                http2=False,
                trust_env=False,
                limits=httpx2.Limits(max_keepalive_connections=0),
            )
        )

    async def call(
        self,
        *,
        endpoint: str,
        bearer: SecretStr | None,
        policy: ConnectionPolicy,
        name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        """One fresh SDK session, no OAuth flow, discovery, resumption or replay."""
        if len(json.dumps(arguments).encode()) > policy.max_call_argument_bytes:
            raise ConnectionsError("MCP arguments exceed policy")
        _protect_sdk_logs()
        token = _PRIVATE_SESSION.set(True)
        transport = _AdmittedTransport(
            endpoint, bearer, policy, self._transport_factory(), foreground=True
        )
        try:
            async with (
                asyncio.timeout(policy.call_timeout),
                httpx2.AsyncClient(
                    transport=transport,
                    trust_env=False,
                    follow_redirects=False,
                    timeout=httpx2.Timeout(policy.idle_timeout, connect=policy.connect_timeout),
                ) as client,
                streamable_http_client(endpoint, http_client=client) as streams,
                ClientSession(streams[0], streams[1]) as session,
            ):
                await session.initialize()
                # send_request avoids call_tool's discovery/cache and interactive
                # input negotiation. No automatic authorization or retry is allowed.
                result = await session.send_request(
                    types.CallToolRequest(
                        params=types.CallToolRequestParams(name=name, arguments=arguments)
                    ),
                    types.CallToolResult,
                )
                if len(result.content) > policy.max_result_parts:
                    raise ConnectionsError("MCP result part quota exceeded")
                text = []
                for part in result.content:
                    if not isinstance(part, types.TextContent):
                        raise ConnectionsError("MCP media result unsupported")
                    text.append(part.text)
                if result.structured_content is not None:
                    text.append(json.dumps(result.structured_content, ensure_ascii=False))
                content = "\n".join(text)
                if bearer is not None:
                    content = content.replace(bearer.get_secret_value(), "[redacted]")
                if len(content.encode()) > policy.max_result_bytes:
                    raise ConnectionsError("MCP result quota exceeded")
                if result.is_error:
                    # Remote errors may echo bearer values or private diagnostics.
                    raise ConnectionsError("MCP remote tool failed")
                return ToolResult.text(content)
        except asyncio.CancelledError:
            raise
        except Exception:
            if transport.authentication_failed:
                raise ConnectionsError("MCP authentication failed", 401) from None
            raise ConnectionsError("MCP call failed; outcome may be unknown") from None
        finally:
            _PRIVATE_SESSION.reset(token)

    async def discover(
        self, *, endpoint: str, bearer: SecretStr | None, policy: ConnectionPolicy
    ) -> list[dict[str, object]]:
        tools = await self.discover_authorized(endpoint=endpoint, bearer=bearer, policy=policy)
        if bearer is not None:
            encoded = json.dumps(tools).replace(
                json.dumps(bearer.get_secret_value())[1:-1], "[redacted]"
            )
            return json.loads(encoded)
        return tools

    async def discover_authorized(
        self,
        *,
        endpoint: str,
        bearer: SecretStr | None,
        policy: ConnectionPolicy,
        auth: httpx2.Auth | None = None,
    ) -> list[dict[str, Any]]:
        """Settings-only SDK negotiation; foreground never uses this path."""
        _protect_sdk_logs()
        token = _PRIVATE_SESSION.set(True)
        transport = None
        try:
            transport = _AdmittedTransport(
                endpoint, bearer, policy, self._transport_factory(), oauth=auth is not None
            )
            async with (
                asyncio.timeout(
                    policy.oauth_timeout if auth is not None else policy.discovery_timeout
                ),
                httpx2.AsyncClient(
                    transport=transport,
                    auth=auth,
                    trust_env=False,
                    follow_redirects=False,
                    # httpx2 counts metadata/registration/exchange responses in
                    # auth history too. SDK OAuth separately caps each redirect
                    # chain at five; bound the entire negotiation here.
                    max_redirects=32 if auth is not None else 5,
                    timeout=httpx2.Timeout(policy.idle_timeout, connect=policy.connect_timeout),
                ) as client,
            ):
                async with streamable_http_client(endpoint, http_client=client) as streams:
                    async with ClientSession(streams[0], streams[1]) as session:
                        await session.initialize()
                        tools: list[dict[str, object]] = []
                        seen: set[str] = set()
                        cursor = None
                        for _ in range(policy.max_pages):
                            # Use the SDK's public request seam, rather than its
                            # list_tools cache which silently drops invalid header tools.
                            page = await session.send_request(
                                types.ListToolsRequest(
                                    params=types.PaginatedRequestParams(cursor=cursor)
                                    if cursor
                                    else None
                                ),
                                types.ListToolsResult,
                            )
                            for tool in page.tools:
                                tools.append(
                                    {
                                        "name": tool.name,
                                        "description": tool.description or "",
                                        "input_schema": tool.input_schema,
                                    }
                                )
                            if len(tools) > policy.max_tools:
                                raise ConnectionsError("MCP tool quota exceeded")
                            cursor = page.next_cursor
                            if cursor is None:
                                return tools
                            if not cursor or len(cursor) > 2048 or cursor in seen:
                                raise ConnectionsError("MCP pagination rejected")
                            seen.add(cursor)
                        raise ConnectionsError("MCP page quota exceeded")
        except asyncio.CancelledError:
            raise
        except Exception:
            if transport is not None and transport.authentication_failed:
                raise ConnectionsError("MCP authentication failed", 401) from None
            raise ConnectionsError("MCP discovery failed") from None
        finally:
            _PRIVATE_SESSION.reset(token)
