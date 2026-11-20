# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Fake HTTP/SDK fixtures exercise the personal MCP transport interface."""

import json
import logging
import socket
from typing import Any

import httpx2
import pytest
from pydantic import SecretStr

from dlightrag.adapters.mcp.personal_http import PersonalMcpClient
from dlightrag.application.connections.models import (
    ConnectionsError,
    CredentialAccess,
    UnsupportedResultError,
)
from dlightrag.application.connections.policy import ConnectionPolicy
from tests.support.dns import public_dns


def access(token: str, header: str = "Authorization", prefix: str = "Bearer ") -> CredentialAccess:
    """The credential one request carries: its token, and the header it travels in."""
    return CredentialAccess(token=SecretStr(token), header=header, prefix=prefix)


INITIALIZE_RESULT: dict[str, Any] = {
    "protocolVersion": "2025-11-25",
    "capabilities": {"tools": {}, "resources": {}},
    "serverInfo": {"name": "fake", "version": "1"},
}


def json_rpc(
    responses: dict[str, Any],
    *,
    initialize: dict[str, Any] | None = None,
    requests: list[dict[str, Any]] | None = None,
):
    """One fake MCP endpoint: answer initialize, then each method from ``responses``."""

    def handler(request):
        if request.method == "DELETE":
            return httpx2.Response(200)
        body = json.loads(request.content)
        if "id" not in body:
            return httpx2.Response(202)
        if requests is not None:
            requests.append(body)
        if body["method"] == "initialize":
            result = {**INITIALIZE_RESULT, **(initialize or {})}
        else:
            result = responses[body["method"]]
        return httpx2.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})

    return handler


def test_public_dns_leaves_loopback_to_the_real_resolver(monkeypatch):
    """The fake is installed on the process-wide `socket.getaddrinfo`, so it must not hijack it.

    Every integration test that patches this helper also opens a real PostgreSQL connection on
    `localhost`; a fake that answers that lookup sends the connect to the public fixture address.
    """
    monkeypatch.setattr(socket, "getaddrinfo", public_dns)
    infos = socket.getaddrinfo("localhost", 5432, type=socket.SOCK_STREAM)
    assert infos, "localhost must still resolve"
    assert infos[0][4][0] in {"127.0.0.1", "::1"}, "localhost must not be faked to a public address"


@pytest.mark.asyncio
async def test_discovery_pins_dns_and_sends_only_connection_bearer(monkeypatch):
    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    requests = []

    def handler(request):
        requests.append(request)
        assert request.url.host == "93.184.216.34"
        assert request.headers["host"] == "fixture.example"
        assert request.extensions["sni_hostname"] == "fixture.example"
        assert request.headers["authorization"] == "Bearer fixture-secret"
        assert "cookie" not in request.headers
        body = json.loads(request.content)
        if "id" not in body:
            return httpx2.Response(202)
        if body["method"] == "initialize":
            result = {
                "protocolVersion": "2025-11-25",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake", "version": "1"},
            }
        else:
            result = {
                "tools": [
                    {
                        "name": "new_tool",
                        "description": "Fixture fixture-secret",
                        "inputSchema": {"type": "object"},
                    }
                ]
            }
        return httpx2.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})

    mcp = PersonalMcpClient(transport_factory=lambda: httpx2.MockTransport(handler))
    discovery = await mcp.discover(
        endpoint="https://fixture.example/mcp",
        access=access("fixture-secret"),
        policy=ConnectionPolicy(),
    )
    assert discovery.tools[0]["name"] == "new_tool"
    assert "fixture-secret" not in str(discovery)
    assert len(requests) == 3


@pytest.mark.asyncio
async def test_mixed_private_dns_rejected_before_any_http(monkeypatch):
    monkeypatch.setattr(
        "dlightrag.engine.network_admission.socket.getaddrinfo",
        lambda host, port, *args, **kwargs: (
            public_dns(host, port, *args, **kwargs)
            + [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))]
        ),
    )
    requests = []

    def handler(request):
        requests.append(request)
        return httpx2.Response(500)

    mcp = PersonalMcpClient(transport_factory=lambda: httpx2.MockTransport(handler))
    with pytest.raises(ConnectionsError):
        await mcp.discover(
            endpoint="https://fixture.example/mcp", access=None, policy=ConnectionPolicy()
        )
    assert requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("address", ["224.0.0.1", "ff0e::1", "169.254.169.254", "0.0.0.0", "::1"])
async def test_non_unicast_destinations_never_receive_bearer(address, monkeypatch):
    monkeypatch.setattr(
        "dlightrag.engine.network_admission.socket.getaddrinfo",
        lambda *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 443))],
    )
    requests = []

    def handler(request):
        requests.append(request)
        return httpx2.Response(500)

    mcp = PersonalMcpClient(transport_factory=lambda: httpx2.MockTransport(handler))
    with pytest.raises(ConnectionsError):
        await mcp.discover(
            endpoint="https://fixture.example/mcp",
            access=access("fixture-secret"),
            policy=ConnectionPolicy(),
        )
    assert requests == []


@pytest.mark.asyncio
async def test_cross_origin_redirect_and_sensitive_errors_never_leak_auth(monkeypatch, caplog):
    import logging

    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    requests = []

    def handler(request):
        requests.append(request)
        return httpx2.Response(
            307, headers={"Location": "https://evil.example/path?token=fixture-secret"}
        )

    caplog.set_level(logging.DEBUG)
    mcp = PersonalMcpClient(transport_factory=lambda: httpx2.MockTransport(handler))
    with pytest.raises(ConnectionsError) as error:
        await mcp.discover(
            endpoint="https://fixture.example/mcp",
            access=access("fixture-secret"),
            policy=ConnectionPolicy(),
        )
    assert len(requests) == 1
    assert "fixture-secret" not in str(error.value)
    assert "fixture-secret" not in caplog.text


@pytest.mark.asyncio
async def test_pagination_duplicate_cursor_rejects_the_whole_discovery(monkeypatch):
    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)

    def handler(request):
        body = json.loads(request.content)
        if "id" not in body:
            return httpx2.Response(202)
        result = (
            {
                "protocolVersion": "2025-11-25",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake", "version": "1"},
            }
            if body["method"] == "initialize"
            else {"tools": [], "nextCursor": "same"}
        )
        return httpx2.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})

    mcp = PersonalMcpClient(transport_factory=lambda: httpx2.MockTransport(handler))
    with pytest.raises(ConnectionsError):
        await mcp.discover(
            endpoint="https://fixture.example/mcp", access=None, policy=ConnectionPolicy()
        )


@pytest.mark.asyncio
async def test_authentication_fault_is_redacted_and_distinct(monkeypatch):
    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    mcp = PersonalMcpClient(
        transport_factory=lambda: httpx2.MockTransport(
            lambda request: httpx2.Response(401, text="fixture-sensitive-provider-error")
        )
    )
    with pytest.raises(ConnectionsError) as error:
        await mcp.discover(
            endpoint="https://fixture.example/mcp",
            access=access("fixture-secret"),
            policy=ConnectionPolicy(),
        )
    assert error.value.status == 401
    assert "fixture-sensitive-provider-error" not in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, "redirect", "unauthorized", "disconnect", "oversize"])
async def test_foreground_call_is_bounded_and_never_reposts_effect(monkeypatch, failure):
    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    calls = []

    def handler(request):
        body = json.loads(request.content)
        if "id" not in body:
            return httpx2.Response(202)
        if body["method"] == "initialize":
            result = {
                "protocolVersion": "2025-11-25",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake", "version": "1"},
            }
        else:
            assert body["method"] == "tools/call"  # no implicit list/discovery
            assert body["params"]["name"] == "write"
            assert body["params"]["arguments"] == {"path": "x"}
            calls.append(request)
            if failure == "redirect":
                return httpx2.Response(307, headers={"location": "https://fixture.example/other"})
            if failure == "unauthorized":
                return httpx2.Response(401)
            if failure == "disconnect":
                raise httpx2.ReadError("secret remote error")
            result = {
                "content": [
                    {"type": "text", "text": "x" * 10000 if failure == "oversize" else "written"}
                ]
            }
        return httpx2.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})

    mcp = PersonalMcpClient(transport_factory=lambda: httpx2.MockTransport(handler))
    kwargs: dict[str, Any] = dict(
        endpoint="https://fixture.example/mcp",
        access=None,
        policy=ConnectionPolicy(max_response_bytes=4096),
        name="write",
        arguments={"path": "x"},
    )
    if failure:
        with pytest.raises(ConnectionsError) as error:
            await mcp.call(**kwargs)
        assert "secret" not in str(error.value)
        if failure == "unauthorized":
            assert error.value.status == 401
    else:
        result = await mcp.call(**kwargs)
        assert result.text_content == "written"
    assert len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["sse", "input_required", "parts", "media", "timeout"])
async def test_foreground_faults_never_resume_or_renegotiate_effect(monkeypatch, failure):
    import asyncio

    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    methods = []
    closed = []

    class Transport(httpx2.MockTransport):
        async def aclose(self):
            closed.append(True)
            await super().aclose()

    async def handler(request):
        if request.method == "DELETE":
            return httpx2.Response(200)
        assert request.method == "POST"  # even SSE resumption sends zero GET I/O
        body = json.loads(request.content)
        if "id" not in body:
            return httpx2.Response(202)
        methods.append(body["method"])
        if body["method"] == "initialize":
            result = {
                "protocolVersion": "2025-11-25",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake", "version": "1"},
            }
        else:
            if failure == "sse":
                return httpx2.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=b"id: effect-possibly-written\nretry: 0\ndata:\n\n",
                )
            if failure == "input_required":
                return httpx2.Response(
                    400,
                    json={
                        "jsonrpc": "2.0",
                        "id": body["id"],
                        "error": {"code": -32022, "message": "Please retry write"},
                    },
                )
            if failure == "timeout":
                await asyncio.Event().wait()
            result = {
                "content": [{"type": "text", "text": "x"}] * 33
                if failure == "parts"
                else [{"type": "image", "data": "eA==", "mimeType": "image/png"}]
            }
        return httpx2.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})

    client = PersonalMcpClient(transport_factory=lambda: Transport(handler))
    with pytest.raises(ConnectionsError):
        await client.call(
            endpoint="https://fixture.example/mcp",
            access=None,
            name="write",
            arguments={},
            policy=ConnectionPolicy(call_timeout=1),
        )
    assert methods == ["initialize", "tools/call"]
    assert closed == [True]


@pytest.mark.asyncio
async def test_foreground_argument_quota_rejects_before_any_transport():
    calls = []

    def factory():
        calls.append(True)
        raise AssertionError("No transport expected")

    client = PersonalMcpClient(transport_factory=factory)
    with pytest.raises(ConnectionsError, match="arguments exceed"):
        await client.call(
            endpoint="https://fixture.example/mcp",
            access=None,
            name="write",
            arguments={"x": "x" * 100},
            policy=ConnectionPolicy(max_call_argument_bytes=10),
        )
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("is_error", [False, True])
async def test_a_result_with_media_is_unsupported_unless_the_remote_tool_failed(
    monkeypatch, is_error
):
    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)

    def handler(request):
        body = json.loads(request.content)
        if "id" not in body:
            return httpx2.Response(202)
        if body["method"] == "initialize":
            result = {
                "protocolVersion": "2025-11-25",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake", "version": "1"},
            }
        else:
            result = {
                "content": [
                    {"type": "text", "text": "saved"},
                    {"type": "image", "data": "eA==", "mimeType": "image/png"},
                    {"type": "image", "data": "eA==", "mimeType": "image/png"},
                    {"type": "audio", "data": "eA==", "mimeType": "audio/wav"},
                ],
                "isError": is_error,
            }
        return httpx2.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})

    mcp = PersonalMcpClient(transport_factory=lambda: httpx2.MockTransport(handler))
    with pytest.raises(ConnectionsError) as error:
        await mcp.call(
            endpoint="https://fixture.example/mcp",
            access=None,
            policy=ConnectionPolicy(call_timeout=5),
            name="write",
            arguments={},
        )

    if is_error:
        assert not isinstance(error.value, UnsupportedResultError)
    else:
        assert isinstance(error.value, UnsupportedResultError)
        assert error.value.kinds == ("image", "audio")


def test_an_unshowable_result_says_the_call_completed() -> None:
    from types import SimpleNamespace

    from dlightrag.application.connections.service import _unshowable_result

    result = _unshowable_result(
        SimpleNamespace(local_name="notion_search"),  # type: ignore[arg-type]
        ("image", "audio"),
    )

    assert result.is_error
    assert "notion_search completed" in result.text_content
    assert "image, audio" in result.text_content
    assert "unknown" not in result.text_content and "Do not retry" in result.text_content


def test_a_failed_exchange_names_what_is_inside_the_group() -> None:
    """The SDK reports its reason wrapped in a group, and the group's own name says nothing."""
    from dlightrag.adapters.mcp.personal_http import oauth_failure_category

    request = httpx2.Request("POST", "https://mcp.notion.com/token")
    refused = httpx2.HTTPStatusError(
        "client error", request=request, response=httpx2.Response(400, request=request)
    )
    group = ExceptionGroup(
        "unhandled errors in a TaskGroup",
        [refused, ConnectionsError("MCP target rejected by network policy", 422)],
    )

    described = oauth_failure_category(group)

    assert "HTTPStatusError 400" in described
    assert "MCP target rejected by network policy" in described


def test_a_failed_exchange_never_carries_a_foreign_message() -> None:
    """An SDK failure can hold an endpoint, a code, or a token; a log line holds none of them."""
    from dlightrag.adapters.mcp.personal_http import oauth_failure_category

    exc = RuntimeError("https://mcp.notion.com/token?code=secret-code")

    assert oauth_failure_category(exc) == "RuntimeError"


@pytest.mark.asyncio
async def test_a_refused_discovery_names_every_request_it_made(monkeypatch, caplog):
    """A refused exchange lists what it asked and what it was answered, in order.

    The list is what tells an operator which step refused - the metadata, the registration, the
    token endpoint, or the endpoint with the token. It carries a method, a host, a path and a
    status: the response body is remote text and stays out of it, and so does the credential.
    """
    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    mcp = PersonalMcpClient(
        transport_factory=lambda: httpx2.MockTransport(
            lambda request: httpx2.Response(400, text="fixture-sensitive-provider-error")
        )
    )
    with caplog.at_level(logging.WARNING, logger="dlightrag.adapters.mcp.personal_http"):
        with pytest.raises(ConnectionsError):
            await mcp.discover_authorized(
                endpoint="https://fixture.example/mcp",
                access=access("fixture-secret"),
                policy=ConnectionPolicy(),
            )

    assert "requests: POST fixture.example/mcp -> 400" in caplog.text
    assert "fixture-sensitive-provider-error" not in caplog.text
    assert "fixture-secret" not in caplog.text


@pytest.mark.asyncio
async def test_a_connection_states_its_headers_and_its_credential_header(monkeypatch):
    """Two facts about one request: what the endpoint needs stated, and where the token goes.

    The vendor header carries no secret and the credential carries nothing else; the result
    still redacts the token from a tool result, and the request states no ``Authorization``.
    """
    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    requests = []

    def handler(request):
        requests.append(request)
        assert request.headers["x-lark-mcp-allowed-tools"] == "create-doc,fetch-doc"
        assert request.headers["x-lark-mcp-tat"] == "t-fixture-token"
        assert "authorization" not in request.headers
        assert "cookie" not in request.headers
        body = json.loads(request.content)
        if "id" not in body:
            return httpx2.Response(202)
        result = (
            {
                "protocolVersion": "2025-11-25",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake", "version": "1"},
            }
            if body["method"] == "initialize"
            else {
                "tools": [
                    {
                        "name": "fetch-doc",
                        "description": "Fixture t-fixture-token",
                        "inputSchema": {"type": "object"},
                    }
                ]
            }
        )
        return httpx2.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})

    mcp = PersonalMcpClient(transport_factory=lambda: httpx2.MockTransport(handler))
    discovery = await mcp.discover(
        endpoint="https://fixture.example/mcp",
        access=access("t-fixture-token", "X-Lark-MCP-TAT", ""),
        policy=ConnectionPolicy(),
        endpoint_headers=(("X-Lark-MCP-Allowed-Tools", "create-doc,fetch-doc"),),
    )

    assert discovery.tools[0]["name"] == "fetch-doc"
    assert "t-fixture-token" not in str(discovery)
    assert len(requests) == 3


@pytest.mark.asyncio
async def test_a_stated_header_may_not_take_over_the_credential_header(monkeypatch, caplog):
    """Policy refuses the collision before a Connection is stored; the transport refuses it too.

    Discovery reports one bounded refusal to its caller, so the reason is the warning line: what
    matters here is that the request was never sent and the credential was never overwritten.
    """
    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    requests = []

    def handler(request):
        requests.append(request)
        return httpx2.Response(500)

    mcp = PersonalMcpClient(transport_factory=lambda: httpx2.MockTransport(handler))
    with caplog.at_level(logging.WARNING, logger="dlightrag.adapters.mcp.personal_http"):
        with pytest.raises(ConnectionsError):
            await mcp.discover(
                endpoint="https://fixture.example/mcp",
                access=access("t-fixture-token", "X-Lark-MCP-TAT", ""),
                policy=ConnectionPolicy(),
                endpoint_headers=(("X-Lark-MCP-TAT", "not-the-credential"),),
            )

    assert "MCP credential header collision" in caplog.text
    assert "t-fixture-token" not in caplog.text
    assert requests == []


@pytest.mark.asyncio
async def test_stated_headers_go_to_the_endpoint_and_not_to_another_origin(monkeypatch):
    """The headers describe this endpoint. An authorization server is not this endpoint."""
    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    seen = {}

    def handler(request):
        # The transport pins the address it admitted, so the host the request was made for is
        # the header it states, not the resolved address it was sent to.
        seen[request.headers["host"]] = dict(request.headers)
        body = json.loads(request.content)
        if "id" not in body:
            return httpx2.Response(202)
        result = (
            {
                "protocolVersion": "2025-11-25",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake", "version": "1"},
            }
            if body["method"] == "initialize"
            else {"tools": []}
        )
        return httpx2.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})

    mcp = PersonalMcpClient(transport_factory=lambda: httpx2.MockTransport(handler))
    await mcp.discover(
        endpoint="https://fixture.example/mcp",
        access=None,
        policy=ConnectionPolicy(),
        endpoint_headers=(("X-Api-Version", "2026-01-01"),),
    )

    assert seen["fixture.example"]["x-api-version"] == "2026-01-01"


# --- Discovery instructions ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_discovery_keeps_bounded_redacted_instructions(monkeypatch):
    """The server's own guidance survives discovery, bounded and stripped of this token."""
    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    handler = json_rpc(
        {"tools/list": {"tools": []}},
        initialize={"instructions": "Use read only. fixture-secret stays private."},
    )
    mcp = PersonalMcpClient(transport_factory=lambda: httpx2.MockTransport(handler))

    discovery = await mcp.discover(
        endpoint="https://fixture.example/mcp",
        access=access("fixture-secret"),
        policy=ConnectionPolicy(),
    )

    assert discovery.tools == ()
    assert discovery.instructions == "Use read only. [redacted] stays private."
    assert "fixture-secret" not in repr(discovery)


@pytest.mark.asyncio
async def test_a_server_without_instructions_states_none(monkeypatch):
    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    handler = json_rpc({"tools/list": {"tools": []}})
    mcp = PersonalMcpClient(transport_factory=lambda: httpx2.MockTransport(handler))

    discovery = await mcp.discover(
        endpoint="https://fixture.example/mcp", access=None, policy=ConnectionPolicy()
    )

    assert discovery.instructions is None


@pytest.mark.asyncio
@pytest.mark.parametrize("instructions", ["", "   \n\t "])
async def test_blank_instructions_state_nothing(monkeypatch, instructions):
    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    handler = json_rpc({"tools/list": {"tools": []}}, initialize={"instructions": instructions})
    mcp = PersonalMcpClient(transport_factory=lambda: httpx2.MockTransport(handler))

    discovery = await mcp.discover(
        endpoint="https://fixture.example/mcp", access=None, policy=ConnectionPolicy()
    )

    assert discovery.instructions is None


@pytest.mark.asyncio
async def test_oversized_instructions_are_refused_before_publication(monkeypatch):
    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    handler = json_rpc({"tools/list": {"tools": []}}, initialize={"instructions": "x" * 1100})
    mcp = PersonalMcpClient(transport_factory=lambda: httpx2.MockTransport(handler))

    with pytest.raises(ConnectionsError, match="instructions too large"):
        await mcp.discover(
            endpoint="https://fixture.example/mcp",
            access=None,
            policy=ConnectionPolicy(max_instruction_bytes=1024),
        )


# --- Resource listing ---------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resources_list_renders_one_line_per_item_and_the_next_cursor(monkeypatch):
    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    requests: list[dict[str, Any]] = []
    handler = json_rpc(
        {
            "resources/list": {
                "resources": [
                    {"uri": "file:///notes/a.md", "name": "A", "mimeType": "text/markdown"},
                    {"uri": "file:///notes/b.md", "name": "", "mimeType": ""},
                    {"uri": "file:///notes/c.md", "name": ""},
                ],
                "nextCursor": "page-2",
            }
        },
        requests=requests,
    )
    mcp = PersonalMcpClient(transport_factory=lambda: httpx2.MockTransport(handler))

    result = await mcp.list_resources(
        endpoint="https://fixture.example/mcp", access=None, policy=ConnectionPolicy()
    )

    assert result.text_content == (
        "file:///notes/a.md — A (text/markdown)\n"
        "file:///notes/b.md\n"
        "file:///notes/c.md\n"
        "next cursor: page-2"
    )
    assert [body["method"] for body in requests] == ["initialize", "resources/list"]
    assert "params" not in requests[-1]


@pytest.mark.asyncio
async def test_resources_list_passes_its_cursor_and_renders_an_empty_page(monkeypatch):
    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    requests: list[dict[str, Any]] = []
    handler = json_rpc({"resources/list": {"resources": []}}, requests=requests)
    mcp = PersonalMcpClient(transport_factory=lambda: httpx2.MockTransport(handler))

    result = await mcp.list_resources(
        endpoint="https://fixture.example/mcp",
        access=None,
        policy=ConnectionPolicy(),
        cursor="page-2",
    )

    assert result.text_content == "No MCP resources."
    assert requests[-1]["params"] == {"cursor": "page-2"}


@pytest.mark.asyncio
async def test_an_oversized_resource_cursor_never_opens_a_transport(monkeypatch):
    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)

    def factory():
        raise AssertionError("No transport expected")

    mcp = PersonalMcpClient(transport_factory=factory)
    with pytest.raises(ConnectionsError, match="cursor rejected"):
        await mcp.list_resources(
            endpoint="https://fixture.example/mcp",
            access=None,
            policy=ConnectionPolicy(),
            cursor="c" * 2049,
        )


@pytest.mark.asyncio
async def test_an_oversized_next_cursor_rejects_the_page(monkeypatch):
    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    handler = json_rpc({"resources/list": {"resources": [], "nextCursor": "n" * 2049}})
    mcp = PersonalMcpClient(transport_factory=lambda: httpx2.MockTransport(handler))

    with pytest.raises(ConnectionsError, match="cursor rejected"):
        await mcp.list_resources(
            endpoint="https://fixture.example/mcp", access=None, policy=ConnectionPolicy()
        )


@pytest.mark.asyncio
async def test_a_resource_page_respects_the_part_and_byte_quotas(monkeypatch):
    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    resources = [{"uri": f"file:///r{index}", "name": "R"} for index in range(2)]
    handler = json_rpc({"resources/list": {"resources": resources}})
    mcp = PersonalMcpClient(transport_factory=lambda: httpx2.MockTransport(handler))

    with pytest.raises(ConnectionsError, match="part quota"):
        await mcp.list_resources(
            endpoint="https://fixture.example/mcp",
            access=None,
            policy=ConnectionPolicy(max_result_parts=1),
        )
    with pytest.raises(ConnectionsError, match="result quota"):
        await mcp.list_resources(
            endpoint="https://fixture.example/mcp",
            access=None,
            policy=ConnectionPolicy(max_result_bytes=1),
        )


# --- Resource templates -------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resource_templates_render_the_description_and_an_empty_page(monkeypatch):
    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    requests: list[dict[str, Any]] = []
    handler = json_rpc(
        {
            "resources/templates/list": {
                "resourceTemplates": [
                    {
                        "uriTemplate": "file:///{path}",
                        "name": "Files",
                        "mimeType": "text/plain",
                        "description": "Read a file",
                    },
                    {
                        "uriTemplate": "db://{table}",
                        "name": "",
                        "description": "Rows",
                    },
                ]
            }
        },
        requests=requests,
    )
    mcp = PersonalMcpClient(transport_factory=lambda: httpx2.MockTransport(handler))

    result = await mcp.list_resource_templates(
        endpoint="https://fixture.example/mcp", access=None, policy=ConnectionPolicy()
    )

    assert result.text_content == (
        "file:///{path} — Files (text/plain): Read a file\ndb://{table}: Rows"
    )
    assert requests[-1]["method"] == "resources/templates/list"


@pytest.mark.asyncio
async def test_an_empty_template_page_says_so(monkeypatch):
    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    handler = json_rpc({"resources/templates/list": {"resourceTemplates": []}})
    mcp = PersonalMcpClient(transport_factory=lambda: httpx2.MockTransport(handler))

    result = await mcp.list_resource_templates(
        endpoint="https://fixture.example/mcp", access=None, policy=ConnectionPolicy()
    )

    assert result.text_content == "No MCP resource templates."


# --- Resource reads -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_resource_returns_text_and_prefixes_it_when_one_read_holds_several(monkeypatch):
    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    requests: list[dict[str, Any]] = []
    handler = json_rpc(
        {
            "resources/read": {
                "contents": [
                    {"uri": "file:///a", "text": "one"},
                    {"uri": "file:///b", "text": "two"},
                ]
            }
        },
        requests=requests,
    )
    mcp = PersonalMcpClient(transport_factory=lambda: httpx2.MockTransport(handler))

    result = await mcp.read_resource(
        endpoint="https://fixture.example/mcp",
        access=None,
        policy=ConnectionPolicy(),
        uri="file:///a",
    )

    assert result.text_content == "file:///a: one\nfile:///b: two"
    assert requests[-1]["method"] == "resources/read"
    assert requests[-1]["params"] == {"uri": "file:///a"}


@pytest.mark.asyncio
async def test_read_resource_returns_lone_text_without_a_prefix(monkeypatch):
    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    handler = json_rpc({"resources/read": {"contents": [{"uri": "file:///a", "text": "one"}]}})
    mcp = PersonalMcpClient(transport_factory=lambda: httpx2.MockTransport(handler))

    result = await mcp.read_resource(
        endpoint="https://fixture.example/mcp",
        access=None,
        policy=ConnectionPolicy(),
        uri="file:///a",
    )

    assert result.text_content == "one"


@pytest.mark.asyncio
async def test_read_resource_names_a_binary_without_echoing_its_payload(monkeypatch):
    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    handler = json_rpc(
        {
            "resources/read": {
                "contents": [{"uri": "file:///bin", "mimeType": "image/png", "blob": "c2VjcmV0"}]
            }
        }
    )
    mcp = PersonalMcpClient(transport_factory=lambda: httpx2.MockTransport(handler))

    result = await mcp.read_resource(
        endpoint="https://fixture.example/mcp",
        access=None,
        policy=ConnectionPolicy(),
        uri="file:///bin",
    )

    assert result.text_content == "[binary resource file:///bin: image/png, 8 base64 characters]"
    assert "c2VjcmV0" not in result.text_content


@pytest.mark.asyncio
async def test_read_resource_says_when_the_server_returned_nothing(monkeypatch):
    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    handler = json_rpc({"resources/read": {"contents": []}})
    mcp = PersonalMcpClient(transport_factory=lambda: httpx2.MockTransport(handler))

    result = await mcp.read_resource(
        endpoint="https://fixture.example/mcp",
        access=None,
        policy=ConnectionPolicy(),
        uri="file:///empty",
    )

    assert result.text_content == "Empty MCP resource."


@pytest.mark.asyncio
async def test_read_resource_redacts_the_connection_token(monkeypatch):
    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    handler = json_rpc(
        {"resources/read": {"contents": [{"uri": "file:///a", "text": "fixture-secret"}]}}
    )
    mcp = PersonalMcpClient(transport_factory=lambda: httpx2.MockTransport(handler))

    result = await mcp.read_resource(
        endpoint="https://fixture.example/mcp",
        access=access("fixture-secret"),
        policy=ConnectionPolicy(),
        uri="file:///a",
    )

    assert result.text_content == "[redacted]"


@pytest.mark.asyncio
@pytest.mark.parametrize("uri", ["", "a\x00b", "x" * 4097])
async def test_an_invalid_resource_uri_never_opens_a_transport(monkeypatch, uri):
    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)

    def factory():
        raise AssertionError("No transport expected")

    mcp = PersonalMcpClient(transport_factory=factory)
    with pytest.raises(ConnectionsError, match="URI rejected"):
        await mcp.read_resource(
            endpoint="https://fixture.example/mcp",
            access=None,
            policy=ConnectionPolicy(),
            uri=uri,
        )


@pytest.mark.asyncio
async def test_read_resource_respects_the_part_quota(monkeypatch):
    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)
    handler = json_rpc(
        {
            "resources/read": {
                "contents": [
                    {"uri": "file:///a", "text": "one"},
                    {"uri": "file:///b", "text": "two"},
                ]
            }
        }
    )
    mcp = PersonalMcpClient(transport_factory=lambda: httpx2.MockTransport(handler))

    with pytest.raises(ConnectionsError, match="part quota"):
        await mcp.read_resource(
            endpoint="https://fixture.example/mcp",
            access=None,
            policy=ConnectionPolicy(max_result_parts=1),
            uri="file:///a",
        )


def test_an_unknown_resource_content_kind_is_named_without_its_payload():
    """A future content variant is diagnostic text, never the bytes it carries."""
    from types import SimpleNamespace

    from dlightrag.adapters.mcp.personal_http import _render_resource_contents

    rendered = _render_resource_contents(
        [SimpleNamespace(kind="image", data="fixture-payload-secret")]
    )

    assert rendered == "[unsupported resource content: image]"
    assert "fixture-payload-secret" not in rendered


# --- Resource faults ----------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["fault", "unauthorized"])
async def test_resource_faults_are_sanitized_and_authentication_stays_distinct(
    monkeypatch, failure
):
    monkeypatch.setattr("dlightrag.engine.network_admission.socket.getaddrinfo", public_dns)

    def handler(request):
        body = json.loads(request.content)
        if "id" not in body:
            return httpx2.Response(202)
        if body["method"] == "initialize":
            return httpx2.Response(
                200, json={"jsonrpc": "2.0", "id": body["id"], "result": INITIALIZE_RESULT}
            )
        if failure == "unauthorized":
            return httpx2.Response(401, text="fixture-sensitive-provider-error")
        return httpx2.Response(500, text="fixture-sensitive-provider-error")

    mcp = PersonalMcpClient(transport_factory=lambda: httpx2.MockTransport(handler))
    with pytest.raises(ConnectionsError) as error:
        await mcp.list_resources(
            endpoint="https://fixture.example/mcp",
            access=access("fixture-secret"),
            policy=ConnectionPolicy(),
        )

    assert "fixture-sensitive-provider-error" not in str(error.value)
    assert "fixture-secret" not in str(error.value)
    if failure == "unauthorized":
        assert error.value.status == 401
    else:
        assert "MCP resource request failed" in str(error.value)
