# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Fake HTTP/SDK fixtures exercise the personal MCP transport interface."""

import json
import socket
from typing import Any

import httpx2
import pytest
from pydantic import SecretStr

from dlightrag.adapters.mcp.personal_http import PersonalMcpClient
from dlightrag.application.connections import ConnectionPolicy, ConnectionsError
from tests.support.dns import public_dns


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
    tools = await mcp.discover(
        endpoint="https://fixture.example/mcp",
        bearer=SecretStr("fixture-secret"),
        policy=ConnectionPolicy(),
    )
    assert tools[0]["name"] == "new_tool"
    assert "fixture-secret" not in str(tools)
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
            endpoint="https://fixture.example/mcp", bearer=None, policy=ConnectionPolicy()
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
            bearer=SecretStr("fixture-secret"),
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
            bearer=SecretStr("fixture-secret"),
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
            endpoint="https://fixture.example/mcp", bearer=None, policy=ConnectionPolicy()
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
            bearer=SecretStr("fixture-secret"),
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
        bearer=None,
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
            bearer=None,
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
            bearer=None,
            name="write",
            arguments={"x": "x" * 100},
            policy=ConnectionPolicy(max_call_argument_bytes=10),
        )
    assert calls == []
