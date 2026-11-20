# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The OAuth callback a deployment publishes is checked once, when configuration loads."""

import json
from collections.abc import Callable
from typing import Any, cast

import pytest
from pydantic import ValidationError

from dlightrag.application.connections.models import ConnectionsError
from dlightrag.application.connections.policy import (
    MAX_ENDPOINT_HEADERS,
    MAX_HEADER_VALUE_CHARS,
    ConnectionPolicy,
    credential_header_problem,
    endpoint_header_problem,
    endpoint_headers_problem,
)


@pytest.mark.parametrize(
    "callback",
    [
        "https://app.example/web/oauth/connections/mcp/callback",
        "http://localhost:8100/web/oauth/connections/mcp/callback",
        "http://127.0.0.1:8100/web/oauth/connections/mcp/callback",
        "http://[::1]:8100/web/oauth/connections/mcp/callback",
    ],
)
def test_a_public_https_or_loopback_callback_is_accepted(callback: str) -> None:
    assert ConnectionPolicy(oauth_callback_url=callback).oauth_callback_url == callback


@pytest.mark.parametrize(
    "callback",
    [
        "http://app.example/web/oauth/connections/mcp/callback",
        "https://app.example/web/oauth/callback",
        "https://app.example/web/oauth/connections/mcp/callback?next=1",
        "https://app.example/web/oauth/connections/mcp/callback#top",
        "/web/oauth/connections/mcp/callback",
        "ftp://app.example/web/oauth/connections/mcp/callback",
    ],
)
def test_any_other_callback_fails_configuration_without_echoing_it(callback: str) -> None:
    with pytest.raises(ValidationError) as raised:
        ConnectionPolicy(oauth_callback_url=callback)

    assert callback not in str(raised.value)


def _catalogue_validator(policy: ConnectionPolicy) -> Callable[[list[dict[str, Any]]], Any]:
    """The catalogue check alone: no store, no MCP port, and no cipher it ever reaches for."""
    from dlightrag.application.connections.service import Connections

    return Connections(
        store=cast(Any, object()),
        mcp=cast(Any, object()),
        cipher=cast(Any, object()),
        policy=policy,
    )._validate_catalogue


def test_a_real_provider_schema_fits_the_default_limits() -> None:
    """A starter preset must be publishable under the defaults this repository ships.

    Notion's own MCP server publishes a tool whose input schema is about 83 KB. A limit that
    rejects it rejects the preset DlightRAG offers first, so the per-tool bound stays above a
    real provider's schema, and the published catalogue as a whole is what `max_catalogue_bytes`
    bounds.
    """
    schema = {
        "type": "object",
        "properties": {
            f"property_{index}": {"type": "string", "description": "x" * 200}
            for index in range(400)
        },
    }
    catalogue = [
        {"name": "notion-create-pages", "description": "Create pages", "input_schema": schema}
    ]
    assert len(json.dumps(schema).encode()) > 83073, "the fixture must weigh what Notion's does"

    validated = _catalogue_validator(ConnectionPolicy())(catalogue)

    assert [tool.remote_name for tool in validated] == ["notion-create-pages"]


def test_a_pathological_schema_still_fails_the_default_limits() -> None:
    policy = ConnectionPolicy()
    schema = {
        "type": "object",
        "properties": {
            "blob": {"type": "string", "description": "x" * (policy.max_schema_bytes + 1)}
        },
    }
    validate = _catalogue_validator(policy)

    with pytest.raises(ConnectionsError, match="a schema is too large"):
        validate([{"name": "blob", "description": "Blob", "input_schema": schema}])


def test_a_stated_header_is_a_plain_non_secret_statement() -> None:
    """What an endpoint needs stated is configuration: it is sent verbatim and never sealed."""
    assert endpoint_header_problem("X-Lark-MCP-Allowed-Tools", "create-doc,fetch-doc") is None
    assert endpoint_header_problem("X-Api-Version", "2026-01-01") is None
    assert endpoint_header_problem("X-Flag", "") is None
    assert endpoint_headers_problem([("X-Api-Version", "1")], credential_header=None) is None


@pytest.mark.parametrize(
    ("name", "value"),
    [
        # The header a credential travels in. A secret has exactly one home, and this is not it.
        ("Authorization", "Bearer fixture"),
        ("authorization", "fixture"),
        ("Proxy-Authorization", "Basic fixture"),
        ("Cookie", "session=fixture"),
        # What our own transport states, or must keep stating itself.
        ("Host", "elsewhere.example"),
        ("Connection", "keep-alive"),
        ("Content-Type", "text/plain"),
        ("Content-Length", "1"),
        ("Accept-Encoding", "gzip"),
        ("Mcp-Session-Id", "fixture"),
        ("Last-Event-ID", "7"),
        ("Transfer-Encoding", "chunked"),
        # Not a field name at all.
        ("X Bad", "1"),
        ("X-Bad:", "1"),
        ("", "1"),
        ("X-Café", "1"),
    ],
)
def test_a_header_a_secret_could_hide_in_is_refused(name: str, value: str) -> None:
    assert endpoint_header_problem(name, value) is not None


def test_a_value_no_http_grammar_carries_is_refused() -> None:
    assert endpoint_header_problem("X-Fixture", "one\r\ntwo") is not None
    assert endpoint_header_problem("X-Fixture", "one\ntwo") is not None
    assert endpoint_header_problem("X-Fixture", "one\x00two") is not None
    assert endpoint_header_problem("X-Fixture", "x" * (MAX_HEADER_VALUE_CHARS + 1)) is not None
    assert endpoint_header_problem("X-Fixture", 1) is not None


def test_stated_headers_are_bounded_and_distinct() -> None:
    pairs = [(f"X-Fixture-{index}", "value") for index in range(MAX_ENDPOINT_HEADERS)]
    assert endpoint_headers_problem(pairs) is None
    assert endpoint_headers_problem([*pairs, ("X-One-More", "value")]) is not None
    assert endpoint_headers_problem([("X-Fixture", "1"), ("x-fixture", "2")]) is not None


def test_a_stated_header_may_not_be_the_one_the_credential_travels_in() -> None:
    """The credential is applied after what the endpoint states, so the two cannot be one header."""
    pairs = [("X-Lark-MCP-TAT", "fixture")]

    assert endpoint_headers_problem(pairs, credential_header="X-Lark-MCP-UAT") is None
    assert endpoint_headers_problem(pairs, credential_header="x-lark-mcp-tat") is not None


def test_a_credential_names_its_own_header_but_never_a_transport_one() -> None:
    """``Authorization: Bearer`` is one instance, not the rule: a vendor header is the other."""
    assert credential_header_problem("Authorization", "Bearer ") is None
    assert credential_header_problem("X-Lark-MCP-TAT", "") is None
    assert credential_header_problem("X-Lark-MCP-UAT", "Token ") is None
    assert credential_header_problem("Host", "") is not None
    assert credential_header_problem("Content-Type", "") is not None
    assert credential_header_problem("Cookie", "") is not None
    assert credential_header_problem("X Bad", "") is not None
    assert credential_header_problem("X-Fixture", "Bearer\r\nX-Evil: 1") is not None
    assert credential_header_problem("X-Fixture", "x" * 65) is not None


def test_a_refusal_never_echoes_what_it_refused() -> None:
    secret = "fixture-value-that-must-not-be-logged"

    for problem in (
        endpoint_header_problem("Authorization", secret),
        endpoint_header_problem("X-Fixture", "\r\n" + secret),
        endpoint_headers_problem([("X-Fixture", secret), ("X-Fixture", secret)]),
        credential_header_problem("Host", secret),
    ):
        assert problem is not None and secret not in problem
