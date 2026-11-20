# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Finite product ceilings; no Run, storage, or execution dependencies."""

import re
from collections.abc import Iterable
from typing import Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator

from dlightrag.engine.ai.settings import ServiceUrl

#: The Web route a provider sends the browser back to after an authorization.
OAUTH_CALLBACK_PATH = "/web/oauth/connections/mcp/callback"
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

#: How a Connection proves who is calling: nothing, a bearer, a header of its own, or OAuth.
Authentication = Literal["none", "bearer", "header", "oauth"]
#: What a Grant's credential is. A Grant is never "none": an unauthenticated Connection has none.
GrantKind = Literal["bearer", "header", "oauth"]
#: What a Connection states about its endpoint, in the order it states it: [name, value] pairs.
StatedHeaders = tuple[tuple[str, str], ...]

#: The limits one stated header obeys, and the numbers the Web route refuses on too.
MAX_HEADER_NAME_CHARS = 64
MAX_HEADER_PREFIX_CHARS = 64
MAX_HEADER_VALUE_CHARS = 1024
#: What one Connection may state about its endpoint. A handful of configuration headers, not a
#: place to put a payload: the pair list is stored as configuration and never encrypted.
MAX_ENDPOINT_HEADERS = 16
#: One header name, in the grammar RFC 9110 gives a field name.
_HEADER_NAME = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]{1," + str(MAX_HEADER_NAME_CHARS) + r"}")
#: Headers our own transport states, or must keep stating itself. A Connection cannot take them
#: over: they carry the framing, the session, and the address the request was admitted against.
_TRANSPORT_HEADERS = frozenset(
    {
        "accept",
        "accept-encoding",
        "connection",
        "content-encoding",
        "content-length",
        "content-type",
        "host",
        "mcp-method",
        "mcp-name",
        "mcp-protocol-version",
        "mcp-session-id",
        # The SDK resumes an SSE stream with this cursor; a Connection stating its own would
        # silently move that cursor for every request of the session.
        "last-event-id",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)
#: Headers a credential travels in. An endpoint header may not be one of these: a secret has
#: exactly one home -- the Credential Grant, sealed -- and this surface never becomes a second.
_CREDENTIAL_HEADERS = frozenset({"authorization", "cookie", "proxy-authorization", "set-cookie"})
#: Headers a credential may not travel in, because the transport itself states them.
_UNCREDENTIALED_HEADERS = _TRANSPORT_HEADERS | frozenset(
    {"cookie", "proxy-authorization", "set-cookie"}
)


def _printable(value: str, *, extra: str) -> bool:
    """Whether every character is one a header value may carry here."""
    return all(char in extra or 33 <= ord(char) <= 126 for char in value)


def endpoint_header_problem(name: object, value: object) -> str | None:
    """Why one endpoint header is refused, or None when it is a legal, non-secret statement.

    Never echoes the value: the reason is product copy, and a refused value may be anything.
    """
    if not isinstance(name, str) or not _HEADER_NAME.fullmatch(name):
        return "An endpoint header name is not allowed"
    if name.lower() in _CREDENTIAL_HEADERS:
        return "An endpoint header may not carry a credential"
    if name.lower() in _TRANSPORT_HEADERS:
        return "The transport states that endpoint header itself"
    if not isinstance(value, str) or len(value) > MAX_HEADER_VALUE_CHARS:
        return "An endpoint header value is too long"
    if not _printable(value, extra="\t "):
        return "An endpoint header value holds a control character"
    return None


def endpoint_headers_problem(
    headers: Iterable[tuple[str, str]], *, credential_header: str | None = None
) -> str | None:
    """Why a Connection's stated headers are refused, or None.

    ``credential_header`` is the header its credential travels in, when that is known: the
    credential is applied after what the endpoint states, so the two may not name one header.
    """
    pairs = list(headers)
    if len(pairs) > MAX_ENDPOINT_HEADERS:
        return "Too many endpoint headers"
    seen: set[str] = set()
    for name, value in pairs:
        problem = endpoint_header_problem(name, value)
        if problem is not None:
            return problem
        lowered = name.lower()
        if lowered in seen:
            return "An endpoint header name is stated twice"
        if credential_header is not None and lowered == credential_header.lower():
            return "An endpoint header may not be the credential's own header"
        seen.add(lowered)
    return None


def credential_header_problem(name: object, prefix: object) -> str | None:
    """Why a credential may not travel in this header, with this prefix, or None.

    A credential names the header it travels in, because not every endpoint takes
    ``Authorization``. It may name ``Authorization`` -- that is what ``bearer`` and ``oauth``
    are -- but never a header the transport states itself.
    """
    if not isinstance(name, str) or not _HEADER_NAME.fullmatch(name):
        return "A credential header name is not allowed"
    if name.lower() in _UNCREDENTIALED_HEADERS:
        return "The transport states that credential header itself"
    if (
        not isinstance(prefix, str)
        or len(prefix) > MAX_HEADER_PREFIX_CHARS
        or not _printable(prefix, extra=" ")
    ):
        return "A credential header prefix is not allowed"
    return None


def oauth_callback_problem(url: str) -> str | None:
    """Why ``url`` cannot be this deployment's OAuth callback, or None when it can.

    A provider sends the user's browser there; DlightRAG never calls it, so the
    outbound network policy does not apply, and plain HTTP is fine on loopback.
    """
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        return "the OAuth callback must be an absolute HTTP(S) URL"
    if parts.scheme != "https" and parts.hostname not in _LOOPBACK_HOSTS:
        return "OAuth needs HTTPS unless DlightRAG is reached on loopback"
    if parts.path != OAUTH_CALLBACK_PATH or parts.query or parts.fragment:
        return f"the OAuth callback must end in {OAUTH_CALLBACK_PATH}, with no query or fragment"
    return None


class ConnectionPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)
    oauth_callback_url: ServiceUrl | None = Field(default=None, max_length=2048)
    oauth_timeout: float = Field(default=300, ge=30, le=600)
    max_connections: int = Field(default=20, ge=1, le=100)
    max_enabled_tools: int = Field(default=256, ge=1, le=1024)
    max_tools: int = Field(default=128, ge=1, le=256)
    max_pages: int = Field(default=16, ge=1, le=32)
    # One tool's schema, not the budget: Notion's own MCP server ships an 83 KB one, and the
    # published catalogue as a whole is bounded by max_catalogue_bytes below.
    max_schema_bytes: int = Field(default=131072, ge=1, le=262144)
    # The server's own initialize instructions ride in the Run prompt as attributed data.
    # 32 KiB matches the harness's MCP client; its ceiling is what a practical server sends,
    # and a text over the bound rejects the catalogue candidate whole rather than truncating.
    max_instruction_bytes: int = Field(default=32768, ge=1024, le=131072)
    max_description_bytes: int = Field(default=8192, ge=1, le=16384)
    max_catalogue_bytes: int = Field(default=524288, ge=1, le=1048576)
    discovery_concurrency: int = Field(default=4, ge=1, le=16)
    refresh_seconds: float = Field(default=300, ge=1, le=3600)
    discovery_timeout: float = Field(default=30, ge=1, le=120)
    call_timeout: float = Field(default=60, ge=1, le=120)
    call_concurrency: int = Field(default=8, ge=1, le=32)
    max_result_bytes: int = Field(default=262144, ge=1, le=1048576)
    max_call_argument_bytes: int = Field(default=65536, ge=1, le=262144)
    max_result_parts: int = Field(default=32, ge=1, le=128)
    connect_timeout: float = Field(default=10, ge=1, le=30)
    idle_timeout: float = Field(default=15, ge=1, le=60)
    max_response_bytes: int = Field(default=1048576, ge=1, le=4194304)
    allow_private_hosts: tuple[str, ...] = ()
    require_https: bool = True

    @field_validator("oauth_callback_url")
    @classmethod
    def _callback_route(cls, value: str | None) -> str | None:
        """An override of the callback, which otherwise follows the address a request reached."""
        if value is not None and (problem := oauth_callback_problem(value)):
            raise ValueError(problem)
        return value
