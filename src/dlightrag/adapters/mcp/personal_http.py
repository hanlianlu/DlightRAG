# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Bounded SDK Streamable HTTP discovery with per-request DNS address pinning."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx2
from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client

from dlightrag.application.connections.credentials import (
    DEFAULT_CREDENTIAL_HEADER,
    DEFAULT_CREDENTIAL_PREFIX,
)
from dlightrag.application.connections.models import (
    ConnectionsError,
    CredentialAccess,
    McpDiscovery,
    UnsupportedResultError,
)
from dlightrag.application.connections.policy import ConnectionPolicy, StatedHeaders
from dlightrag.engine.agent.tools.contracts import ToolResult
from dlightrag.engine.network_admission import (
    _normalize_host_patterns,
    _resolve_public_target,
    validate_credential_free_query,
)

logger = logging.getLogger(__name__)

# How many exchanges one failure line may carry: enough for discovery, consent, exchange and retry.
_EXCHANGE_LOG_LIMIT = 12
# One page cursor and one resource URI: the same bound discovery puts on a cursor, and a URI a
# remote can state without turning a listing into a payload.
_MAX_CURSOR_CHARS = 2048
_MAX_RESOURCE_URI_CHARS = 4096

_PRIVATE_SESSION = ContextVar("private_mcp_session", default=False)


class _PrivateSdkLogs(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # SDK diagnostics may contain entire remote messages and OAuth values.
        # The owner module emits only redacted status, not SDK diagnostics.
        return not _PRIVATE_SESSION.get()


def _unsupported_result(exc: BaseException) -> UnsupportedResultError | None:
    """Find the unsupported-result signal an SDK task group may have wrapped in a group."""
    if isinstance(exc, UnsupportedResultError):
        return exc
    if isinstance(exc, BaseExceptionGroup):
        for inner in exc.exceptions:
            if (found := _unsupported_result(inner)) is not None:
                return found
    return None


def _leaf_failures(exc: BaseException) -> list[BaseException]:
    """Every failure inside a group, so a line names what actually went wrong.

    The SDK runs its session in a task group, which reports the reason wrapped in an
    ``ExceptionGroup``: the group's own name says nothing an operator can act on.
    """
    if isinstance(exc, BaseExceptionGroup):
        return [leaf for inner in exc.exceptions for leaf in _leaf_failures(inner)]
    return [exc]


def _failure_name(exc: BaseException) -> str:
    """Our own words, or a class name and an HTTP status: never a foreign message."""
    if isinstance(exc, ConnectionsError):
        return str(exc)
    status = getattr(getattr(exc, "response", None), "status_code", None)
    return f"{type(exc).__name__} {status}" if isinstance(status, int) else type(exc).__name__


def oauth_failure_category(exc: BaseException) -> str:
    """What a log line may say about a failed exchange: our own words, or a class name.

    Never the exception's message: an SDK failure can carry an endpoint, a code, or a token, and
    a log line is not the place for any of them.
    """
    described = dict.fromkeys(_failure_name(leaf) for leaf in _leaf_failures(exc))
    return ", ".join(described) or type(exc).__name__


def _protect_sdk_logs() -> None:
    for name in tuple(logging.Logger.manager.loggerDict):
        if name.startswith(("mcp.", "httpx2", "httpcore")):
            logger = logging.getLogger(name)
            if not any(isinstance(item, _PrivateSdkLogs) for item in logger.filters):
                logger.addFilter(_PrivateSdkLogs())


def _bounded_instructions(
    instructions: str | None, *, policy: ConnectionPolicy, access: CredentialAccess | None
) -> str | None:
    """The server's initialize guidance, bounded and stripped of this Connection's token.

    Instructions are remote-authored text and ride in a Run prompt as attributed data, so the
    credential never survives into them and the policy cap applies to their own UTF-8 bytes. A
    server that states only whitespace states nothing.
    """
    if instructions is None or not instructions.strip():
        return None
    if len(instructions.encode()) > policy.max_instruction_bytes:
        raise ConnectionsError("MCP instructions too large")
    if access is not None:
        return instructions.replace(access.token.get_secret_value(), "[redacted]")
    return instructions


def _cursor_params(cursor: str | None) -> types.PaginatedRequestParams | None:
    """One page cursor as the request's pagination, or no pagination at all."""
    if not cursor:
        return None
    if len(cursor) > _MAX_CURSOR_CHARS:
        raise ConnectionsError("MCP resource cursor rejected")
    return types.PaginatedRequestParams(cursor=cursor)


def _resource_line(resource: types.Resource) -> str:
    """One model-visible listing line: what to read, what it is called, what it holds."""
    line = resource.uri
    if resource.name:
        line += f" — {resource.name}"
    if resource.mime_type:
        line += f" ({resource.mime_type})"
    return line


def _template_line(template: types.ResourceTemplate) -> str:
    """One model-visible template line, with the server's own description when it states one."""
    line = template.uri_template
    if template.name:
        line += f" — {template.name}"
    if template.mime_type:
        line += f" ({template.mime_type})"
    if template.description:
        line += f": {template.description}"
    return line


def _render_listing(lines: list[str], *, next_cursor: str | None, empty: str) -> str:
    """A bounded page of listing lines, plus the cursor that asks the server for the next one."""
    if next_cursor is not None:
        if not next_cursor or len(next_cursor) > _MAX_CURSOR_CHARS:
            raise ConnectionsError("MCP resource cursor rejected")
        lines.append(f"next cursor: {next_cursor}")
    return "\n".join(lines) if lines else empty


def _render_resource_contents(contents: Sequence[Any]) -> str:
    """Text as the model reads it; binary content and unknown kinds as bounded diagnostics.

    A read result is not a tool result: a binary resource has no text form the model could use, so
    it is named (URI, MIME type, size) rather than dropped, and its payload is never echoed. When
    one read returns several contents, each text is prefixed with the URI it belongs to.
    """
    if not contents:
        return "Empty MCP resource."
    multiple = len(contents) > 1
    rendered: list[str] = []
    for content in contents:
        if isinstance(content, types.TextResourceContents):
            rendered.append(f"{content.uri}: {content.text}" if multiple else content.text)
        elif isinstance(content, types.BlobResourceContents):
            mime_type = content.mime_type or "unknown"
            rendered.append(
                f"[binary resource {content.uri}: {mime_type}, "
                f"{len(content.blob)} base64 characters]"
            )
        else:
            kind = getattr(content, "kind", None) or getattr(content, "type", None)
            rendered.append(f"[unsupported resource content: {kind or type(content).__name__}]")
    return "\n".join(rendered)


def _bounded_resource_result(
    content: str, *, policy: ConnectionPolicy, access: CredentialAccess | None
) -> ToolResult:
    """One resource result under the same token redaction and byte quota as a tool result."""
    if access is not None:
        content = content.replace(access.token.get_secret_value(), "[redacted]")
    if len(content.encode()) > policy.max_result_bytes:
        raise ConnectionsError("MCP result quota exceeded")
    return ToolResult.text(content)


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


def _exchange_suffix(transport: _AdmittedTransport | None) -> str:
    """What this session asked and what it was answered, for the one line a failure writes.

    A method, a host, a path and a status carry no query, no body and no credential, and they are
    what separates a token endpoint that refused the exchange from an endpoint that refused the
    token - or a step that never ran at all.
    """
    if transport is None or not transport.exchanges:
        return ""
    return " (requests: " + "; ".join(transport.exchanges) + ")"


class _AdmittedTransport(httpx2.AsyncBaseTransport):
    def __init__(
        self,
        endpoint: str,
        access: CredentialAccess | None,
        policy: ConnectionPolicy,
        inner: httpx2.AsyncBaseTransport,
        *,
        foreground: bool = False,
        oauth: bool = False,
        endpoint_headers: StatedHeaders = (),
    ) -> None:
        self._oauth = oauth
        self._endpoint = httpx2.URL(endpoint)
        self._policy = policy
        self._inner = inner
        self.authentication_failed = False
        self._foreground = foreground
        self._effect_sent = False
        self._post_ids: set[str] = set()
        # What this Connection states about its own endpoint, and the credential it sends. The
        # credential is applied last, and policy refuses a Connection whose stated headers would
        # name the same header. An unauthenticated Connection states its headers and no token.
        self._endpoint_headers = endpoint_headers
        self._token = access.token if access is not None else None
        self._credential_header = access.header if access is not None else DEFAULT_CREDENTIAL_HEADER
        self._credential_prefix = access.prefix if access is not None else DEFAULT_CREDENTIAL_PREFIX
        # Every request this session made and what it answered, for the one log line a failed
        # exchange writes: which step refused is the whole diagnosis, and the list carries no
        # query, no body and no credential.
        self.exchanges: list[str] = []

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
        same_origin = (url.scheme, url.host, url.port) == (
            self._endpoint.scheme,
            self._endpoint.host,
            self._endpoint.port,
        )
        if (
            (not self._oauth and not same_origin)
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
            if authorization.startswith("Bearer ") and not same_origin:
                raise ConnectionsError("OAuth credential audience rejected")
            headers["authorization"] = authorization
        # What this endpoint needs stated, and then the credential in the header it named. The
        # two are applied in that order because policy refuses a Connection whose stated headers
        # would be the credential's own: nothing here silently overwrites a credential.
        if same_origin:
            for name, value in self._endpoint_headers:
                headers[name.lower()] = value
        if self._token is not None:
            name = self._credential_header.lower()
            if any(stated.lower() == name for stated, _ in self._endpoint_headers):
                raise ConnectionsError("MCP credential header collision")
            headers[name] = self._credential_prefix + self._token.get_secret_value()
        extensions = {**request.extensions, "sni_hostname": target.host}
        pinned_request = httpx2.Request(
            request.method, pinned, headers=headers, stream=request.stream, extensions=extensions
        )
        response = await self._inner.handle_async_request(pinned_request)
        self.exchanges.append(
            f"{request.method} {url.netloc.decode()}{url.path} -> {response.status_code}"
        )
        del self.exchanges[:-_EXCHANGE_LOG_LIMIT]
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
        access: CredentialAccess | None,
        policy: ConnectionPolicy,
        name: str,
        arguments: dict[str, Any],
        endpoint_headers: StatedHeaders = (),
    ) -> ToolResult:
        """One fresh SDK session, no OAuth flow, discovery, resumption or replay."""
        if len(json.dumps(arguments).encode()) > policy.max_call_argument_bytes:
            raise ConnectionsError("MCP arguments exceed policy")
        _protect_sdk_logs()
        token = _PRIVATE_SESSION.set(True)
        transport = _AdmittedTransport(
            endpoint,
            access,
            policy,
            self._transport_factory(),
            foreground=True,
            endpoint_headers=endpoint_headers,
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
                if result.is_error:
                    # Remote errors may echo bearer values or private diagnostics.
                    raise ConnectionsError("MCP remote tool failed")
                text = [part.text for part in result.content if isinstance(part, types.TextContent)]
                if len(text) < len(result.content):
                    raise UnsupportedResultError(
                        tuple(
                            dict.fromkeys(
                                part.type
                                for part in result.content
                                if not isinstance(part, types.TextContent)
                            )
                        )
                    )
                if result.structured_content is not None:
                    text.append(json.dumps(result.structured_content, ensure_ascii=False))
                content = "\n".join(text)
                if access is not None:
                    content = content.replace(access.token.get_secret_value(), "[redacted]")
                if len(content.encode()) > policy.max_result_bytes:
                    raise ConnectionsError("MCP result quota exceeded")
                return ToolResult.text(content)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if (unsupported := _unsupported_result(exc)) is not None:
                raise unsupported from None
            if transport.authentication_failed:
                raise ConnectionsError("MCP authentication failed", 401) from None
            raise ConnectionsError("MCP call failed; outcome may be unknown") from None
        finally:
            _PRIVATE_SESSION.reset(token)

    @asynccontextmanager
    async def _resource_session(
        self,
        *,
        endpoint: str,
        access: CredentialAccess | None,
        policy: ConnectionPolicy,
        endpoint_headers: StatedHeaders,
    ) -> AsyncIterator[ClientSession]:
        """One fresh admitted SDK session for a read, with no flow, resumption or replay."""
        _protect_sdk_logs()
        token = _PRIVATE_SESSION.set(True)
        transport = _AdmittedTransport(
            endpoint,
            access,
            policy,
            self._transport_factory(),
            foreground=True,
            endpoint_headers=endpoint_headers,
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
                yield session
        except asyncio.CancelledError:
            raise
        except Exception:
            if transport.authentication_failed:
                raise ConnectionsError("MCP authentication failed", 401) from None
            raise ConnectionsError("MCP resource request failed") from None
        finally:
            _PRIVATE_SESSION.reset(token)

    async def list_resources(
        self,
        *,
        endpoint: str,
        access: CredentialAccess | None,
        policy: ConnectionPolicy,
        endpoint_headers: StatedHeaders = (),
        cursor: str | None = None,
    ) -> ToolResult:
        """One page of the server's known resources; the next cursor is the caller's to ask for."""
        params = _cursor_params(cursor)
        async with self._resource_session(
            endpoint=endpoint,
            access=access,
            policy=policy,
            endpoint_headers=endpoint_headers,
        ) as session:
            page = await session.send_request(
                types.ListResourcesRequest(params=params), types.ListResourcesResult
            )
        if len(page.resources) > policy.max_result_parts:
            raise ConnectionsError("MCP result part quota exceeded")
        content = _render_listing(
            [_resource_line(resource) for resource in page.resources],
            next_cursor=page.next_cursor,
            empty="No MCP resources.",
        )
        return _bounded_resource_result(content, policy=policy, access=access)

    async def list_resource_templates(
        self,
        *,
        endpoint: str,
        access: CredentialAccess | None,
        policy: ConnectionPolicy,
        endpoint_headers: StatedHeaders = (),
        cursor: str | None = None,
    ) -> ToolResult:
        """One page of the server's URI templates, rendered like its known resources."""
        params = _cursor_params(cursor)
        async with self._resource_session(
            endpoint=endpoint,
            access=access,
            policy=policy,
            endpoint_headers=endpoint_headers,
        ) as session:
            page = await session.send_request(
                types.ListResourceTemplatesRequest(params=params),
                types.ListResourceTemplatesResult,
            )
        if len(page.resource_templates) > policy.max_result_parts:
            raise ConnectionsError("MCP result part quota exceeded")
        content = _render_listing(
            [_template_line(template) for template in page.resource_templates],
            next_cursor=page.next_cursor,
            empty="No MCP resource templates.",
        )
        return _bounded_resource_result(content, policy=policy, access=access)

    async def read_resource(
        self,
        *,
        endpoint: str,
        access: CredentialAccess | None,
        policy: ConnectionPolicy,
        endpoint_headers: StatedHeaders = (),
        uri: str,
    ) -> ToolResult:
        """Read one resource; text reaches the model, a binary is named and bounded."""
        if (
            not uri
            or len(uri) > _MAX_RESOURCE_URI_CHARS
            or any(ord(char) < 32 or ord(char) == 127 for char in uri)
        ):
            raise ConnectionsError("MCP resource URI rejected")
        async with self._resource_session(
            endpoint=endpoint,
            access=access,
            policy=policy,
            endpoint_headers=endpoint_headers,
        ) as session:
            result = await session.send_request(
                types.ReadResourceRequest(params=types.ReadResourceRequestParams(uri=uri)),
                types.ReadResourceResult,
            )
        if len(result.contents) > policy.max_result_parts:
            raise ConnectionsError("MCP result part quota exceeded")
        content = _render_resource_contents(result.contents)
        return _bounded_resource_result(content, policy=policy, access=access)

    async def discover(
        self,
        *,
        endpoint: str,
        access: CredentialAccess | None,
        policy: ConnectionPolicy,
        endpoint_headers: StatedHeaders = (),
    ) -> McpDiscovery:
        discovered = await self.discover_authorized(
            endpoint=endpoint,
            access=access,
            policy=policy,
            endpoint_headers=endpoint_headers,
        )
        if access is not None:
            encoded = json.dumps(discovered.tools).replace(
                json.dumps(access.token.get_secret_value())[1:-1], "[redacted]"
            )
            return McpDiscovery(
                tools=tuple(json.loads(encoded)), instructions=discovered.instructions
            )
        return discovered

    async def discover_authorized(
        self,
        *,
        endpoint: str,
        access: CredentialAccess | None,
        policy: ConnectionPolicy,
        auth: httpx2.Auth | None = None,
        endpoint_headers: StatedHeaders = (),
    ) -> McpDiscovery:
        """Settings-only SDK negotiation; foreground never uses this path."""
        _protect_sdk_logs()
        token = _PRIVATE_SESSION.set(True)
        transport = None
        try:
            transport = _AdmittedTransport(
                endpoint,
                access,
                policy,
                self._transport_factory(),
                oauth=auth is not None,
                endpoint_headers=endpoint_headers,
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
                        initialized = await session.initialize()
                        # What the server says about using it is remote-authored context, never
                        # a log line and never a place this Connection's token may appear.
                        instructions = _bounded_instructions(
                            initialized.instructions, policy=policy, access=access
                        )
                        tools: list[dict[str, Any]] = []
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
                                return McpDiscovery(tools=tuple(tools), instructions=instructions)
                            if not cursor or len(cursor) > 2048 or cursor in seen:
                                raise ConnectionsError("MCP pagination rejected")
                            seen.add(cursor)
                        raise ConnectionsError("MCP page quota exceeded")
        except asyncio.CancelledError:
            # A cancelled authorization is a real ending, and the only trace of it is this line.
            logger.warning("Authorized MCP discovery cancelled%s", _exchange_suffix(transport))
            raise
        except Exception as exc:
            logger.warning(
                "Authorized MCP discovery failed: %s%s",
                oauth_failure_category(exc),
                _exchange_suffix(transport),
            )
            if transport is not None and transport.authentication_failed:
                raise ConnectionsError("MCP authentication failed", 401) from None
            # Our own bounded refusal (instructions, cursor, quota) is already the safe line
            # an operator can act on; the SDK's task group may have wrapped it, and only a
            # foreign failure becomes the generic discovery text.
            refusal = next(
                (leaf for leaf in _leaf_failures(exc) if isinstance(leaf, ConnectionsError)),
                None,
            )
            if refusal is not None:
                raise refusal from None
            raise ConnectionsError("MCP discovery failed") from None
        finally:
            _PRIVATE_SESSION.reset(token)
