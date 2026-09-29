# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Closed transient dependency classification for durable execution."""

import asyncio
import socket
import ssl
import warnings
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import anthropic
import httpx
import httpx2
import openai
import pytest

from dlightrag.application.errors import CorpusUnavailableError
from dlightrag.engine.dependencies import (
    ParserUnavailableError,
    ProviderUnavailableError,
    classify_transient_dependency,
    is_transient_request_failure,
)
from tests.support.loopback import (
    LoopbackCertificate,
    alerting_tls_server,
    loopback_certificate,
    loopback_server,
    reset_on_accept,
    tls_error,
)


def _http_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://provider.example")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError("provider rejected request", request=request, response=response)


@pytest.mark.parametrize("status", [401, 403, 400, 413, 422])
def test_auth_input_and_context_rejections_are_not_transient(status: int) -> None:
    assert classify_transient_dependency(_http_error(status)) is None


_RETRYABLE = [408, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 529]


@pytest.mark.parametrize("status", _RETRYABLE)
def test_known_provider_status_interruptions_are_transient(status: int) -> None:
    assert classify_transient_dependency(_http_error(status)) == "providers"


def test_unknown_exception_is_not_transient() -> None:
    assert classify_transient_dependency(RuntimeError("unknown")) is None


def test_explicit_typed_dependency_boundaries_are_transient() -> None:
    assert classify_transient_dependency(CorpusUnavailableError()) == "corpus_storage"
    assert classify_transient_dependency(ProviderUnavailableError()) == "providers"


def test_typed_parser_and_hinted_transport_failures_are_transient() -> None:
    assert classify_transient_dependency(ParserUnavailableError()) == "parser"
    assert classify_transient_dependency(ConnectionError("down")) is None
    assert (
        classify_transient_dependency(ConnectionError("down"), component_hint="corpus_storage")
        == "corpus_storage"
    )
    assert (
        classify_transient_dependency(TimeoutError(), component_hint="corpus_storage")
        == "corpus_storage"
    )


def test_authentication_cause_wins_over_a_transient_wrapper() -> None:
    try:
        raise RuntimeError("password authentication failed")
    except RuntimeError as auth:
        wrapper = CorpusUnavailableError()
        wrapper.__cause__ = auth

    assert classify_transient_dependency(wrapper) is None


_REQUEST = httpx.Request("POST", "https://dependency.example")


@pytest.mark.parametrize(
    "error",
    [
        httpx.ConnectError("refused", request=_REQUEST),
        httpx.ReadError("reset", request=_REQUEST),
        httpx.ConnectTimeout("slow", request=_REQUEST),
        httpx.PoolTimeout("saturated", request=_REQUEST),
        httpx.RemoteProtocolError("server disconnected", request=_REQUEST),
        httpx.ProxyError("proxy unavailable", request=_REQUEST),
    ],
)
def test_transient_transport_failures_agree_across_request_and_run_classification(
    error: httpx.TransportError,
) -> None:
    assert is_transient_request_failure(error) is True
    assert classify_transient_dependency(error) == "providers"
    assert classify_transient_dependency(error, component_hint="parser") == "parser"


@pytest.mark.parametrize(
    "error",
    [
        httpx.UnsupportedProtocol("missing scheme", request=_REQUEST),
        httpx.LocalProtocolError("illegal header", request=_REQUEST),
    ],
)
def test_transport_configuration_failures_are_not_transient(error: httpx.TransportError) -> None:
    assert is_transient_request_failure(error) is False
    assert classify_transient_dependency(error) is None


@pytest.mark.parametrize("status", _RETRYABLE)
def test_retryable_statuses_are_transient_requests(status: int) -> None:
    assert is_transient_request_failure(_http_error(status)) is True
    stamped = RuntimeError(f"parser upload failed: HTTP {status}")
    stamped.status_code = status  # type: ignore[attr-defined]
    assert is_transient_request_failure(stamped) is True


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 413, 422, 501, 505, 525, 526])
def test_rejected_and_unlisted_statuses_are_not_transient_requests(status: int) -> None:
    # 409 is a conflict with the target's state; 525/526 are an edge proxy's TLS
    # handshake with a misconfigured origin. Resending cannot resolve either.
    assert is_transient_request_failure(_http_error(status)) is False
    assert classify_transient_dependency(_http_error(status)) is None


def test_request_failure_needs_an_explicit_transient_surface() -> None:
    assert is_transient_request_failure(RuntimeError("temporarily unavailable")) is False
    assert is_transient_request_failure(TimeoutError()) is False


def test_non_retryable_marker_wins_over_a_transient_request_surface() -> None:
    try:
        raise httpx.ConnectError("refused", request=_REQUEST)
    except httpx.ConnectError as transport:
        wrapper = RuntimeError("invalid api key")
        wrapper.__cause__ = transport

    assert is_transient_request_failure(wrapper) is False


def test_a_named_parser_outage_is_not_attributed_to_the_providers() -> None:
    try:
        raise httpx.ConnectError("refused", request=_REQUEST)
    except httpx.ConnectError as transport:
        outage = ParserUnavailableError()
        outage.__cause__ = transport

    assert classify_transient_dependency(outage) == "parser"


def _caused[E: BaseException](error: E, cause: BaseException) -> E:
    error.__cause__ = cause
    return error


def _misconfigured_causes() -> list[BaseException]:
    return [
        ssl.SSLCertVerificationError(
            1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed"
        ),
        tls_error("WRONG_VERSION_NUMBER"),
    ]


@pytest.mark.parametrize(
    "cause",
    _misconfigured_causes(),
    ids=["tls-certificate", "https-to-plain-http"],
)
def test_a_misconfigured_endpoint_is_not_an_outage(cause: BaseException) -> None:
    # httpx and the SDKs' own client raise these connect failures with the TLS
    # error as the cause; resending the request cannot fix either.
    transport = _caused(httpx.ConnectError("connect failed", request=_REQUEST), cause)
    sdk_request = httpx2.Request("POST", "https://provider.example/v1")
    sdk = _caused(
        openai.APIConnectionError(request=sdk_request),
        _caused(httpx2.ConnectError("connect failed", request=sdk_request), cause),
    )

    assert is_transient_request_failure(transport) is False
    assert classify_transient_dependency(transport) is None
    assert classify_transient_dependency(transport, component_hint="providers") is None
    assert classify_transient_dependency(sdk, component_hint="providers") is None


def _reset_during_handshake() -> BaseException:
    # A peer resetting the TLS handshake leaves the SSLWantReadError the
    # handshake was waiting on as the reset's implicit context.
    try:
        raise ssl.SSLWantReadError(2, "The operation did not complete (read)")
    except ssl.SSLWantReadError:
        try:
            raise BrokenPipeError(32, "Broken pipe")
        except BrokenPipeError as reset:
            return reset


@pytest.mark.parametrize(
    "cause",
    [
        # macOS answers EAI_NONAME while offline, and Compose's DNS does for a
        # service that is stopped or restarting.
        socket.gaierror(socket.EAI_NONAME, "nodename nor servname provided, or not known"),
        socket.gaierror(socket.EAI_AGAIN, "Temporary failure in name resolution"),
        ssl.SSLEOFError(8, "EOF occurred in violation of protocol"),
        _reset_during_handshake(),
        ConnectionRefusedError(61, "Connection refused"),
    ],
    ids=[
        "dns-no-such-name",
        "dns-temporary-failure",
        "tls-dropped-mid-handshake",
        "tls-handshake-reset",
        "refused",
    ],
)
def test_a_temporarily_unreachable_endpoint_stays_transient(cause: BaseException) -> None:
    transport = _caused(httpx.ConnectError("connect failed", request=_REQUEST), cause)

    assert is_transient_request_failure(transport) is True
    assert classify_transient_dependency(transport) == "providers"


def test_a_typed_boundary_still_decides_for_a_misconfigured_cause() -> None:
    # Corpus storage wraps its own connection failures; scoping the rule to
    # client transports keeps that deferral (and startup degradation) intact.
    for cause in _misconfigured_causes():
        wrapper = _caused(CorpusUnavailableError(), cause)
        assert classify_transient_dependency(wrapper) == "corpus_storage"


@pytest.mark.parametrize(
    "error",
    [
        httpx2.ReadTimeout("stream stalled"),
        httpx2.ReadError("connection reset"),
        httpx2.RemoteProtocolError("peer closed connection without sending complete body"),
    ],
    ids=["read-timeout", "reset", "incomplete-body"],
)
def test_the_sdk_http_client_failing_mid_stream_is_a_provider_interruption(
    error: httpx2.TransportError,
) -> None:
    # The SDKs wrap a failed request, but an error while reading a streamed
    # response reaches the caller as their httpx fork's own exception.
    assert classify_transient_dependency(error, component_hint="providers") == "providers"
    assert classify_transient_dependency(error) == "providers"
    assert is_transient_request_failure(error) is True


def test_the_sdk_http_client_status_error_uses_the_shared_statuses() -> None:
    request = httpx2.Request("POST", "https://provider.example/v1")
    for status, verdict in ((503, "providers"), (409, None)):
        response = httpx2.Response(status, request=request)
        error = httpx2.HTTPStatusError("status", request=request, response=response)
        assert classify_transient_dependency(error) == verdict


def test_an_overloaded_anthropic_provider_is_transient() -> None:
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    overloaded = anthropic.OverloadedError(
        "Overloaded",
        response=httpx2.Response(529, request=request),
        body={"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
    )

    assert classify_transient_dependency(overloaded, component_hint="providers") == "providers"


async def _anthropic_stream_error(error_type: str) -> BaseException:
    """Raise an error event inside an Anthropic stream through the real SDK."""

    events = (
        "event: message_start\n"
        'data: {"type":"message_start","message":{"id":"msg","type":"message",'
        '"role":"assistant","content":[],"model":"claude","stop_reason":null,'
        '"stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":0}}}\n\n'
        "event: error\n"
        f'data: {{"type":"error","error":{{"type":"{error_type}","message":"stream failed"}}}}\n\n'
    )

    def endpoint(_request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, content=events.encode()
        )

    client = anthropic.AsyncAnthropic(
        api_key="test-key",
        max_retries=0,
        http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(endpoint)),
    )
    try:
        stream = await client.messages.create(
            model="claude",
            max_tokens=8,
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
        async for _event in stream:
            pass
    except anthropic.APIStatusError as exc:
        return exc
    finally:
        await client.close()
    raise AssertionError("the stream did not fail")


@pytest.mark.parametrize(
    ("error_type", "verdict"),
    [
        ("overloaded_error", "providers"),
        ("api_error", "providers"),
        ("rate_limit_error", "providers"),
        ("timeout_error", "providers"),
        ("invalid_request_error", None),
        ("authentication_error", None),
    ],
)
async def test_an_anthropic_stream_error_is_classified_by_its_error_type(
    error_type: str,
    verdict: str | None,
) -> None:
    # The error event arrives on the 200 streaming response, so its status says
    # nothing; Anthropic's documented error type is the only verdict.
    error = await _anthropic_stream_error(error_type)

    assert getattr(error, "status_code", None) == 200
    assert classify_transient_dependency(error, component_hint="providers") == verdict


def _stamped(message: str, status: int) -> RuntimeError:
    error = RuntimeError(message)
    error.status_code = status  # type: ignore[attr-defined]
    return error


def test_a_boundary_can_ignore_caller_controlled_text() -> None:
    # A parser operation label names the user's file; its words are not a verdict.
    outage = _stamped("Docling upload for 'db_schema.pdf' failed: HTTP 503 unsupported", 503)

    assert is_transient_request_failure(outage) is False
    assert is_transient_request_failure(outage, text_vetoes=False) is True


def test_ignoring_text_keeps_the_status_type_and_endpoint_vetoes() -> None:
    class AuthenticationError(Exception):
        pass

    misconfigured = _caused(
        httpx.ConnectError("connect failed", request=_REQUEST),
        ssl.SSLCertVerificationError(1, "[SSL: CERTIFICATE_VERIFY_FAILED]"),
    )
    for rejected in (
        _stamped("upload for 'report.pdf' failed: HTTP 401", 401),
        _stamped("upload for 'report.pdf' failed: HTTP 404", 404),
        _caused(AuthenticationError("denied"), _http_error(503)),
        misconfigured,
    ):
        assert is_transient_request_failure(rejected, text_vetoes=False) is False


# Real loopback endpoints, so the chains below are the ones httpcore and anyio
# build (implicit exception context included), not constructed stand-ins.


async def _plain_http(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    await reader.read(1024)
    writer.write(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
    await writer.drain()
    writer.close()


async def _hold_open(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    await reader.read(1024)
    writer.close()


async def _get_failure(
    client_module: ModuleType,
    url: str,
    *,
    verify: ssl.SSLContext | bool = True,
    timeout: object = 5,
) -> BaseException:
    async with client_module.AsyncClient(verify=verify, timeout=timeout) as client:
        try:
            await client.get(url)
        except Exception as exc:  # noqa: BLE001 - the failure is the subject
            return exc
    raise AssertionError("the request did not fail")


@pytest.mark.parametrize("client_module", [httpx, httpx2], ids=["httpx", "httpx2"])
async def test_a_real_handshake_reset_is_transient(client_module: ModuleType) -> None:
    async with loopback_server(reset_on_accept) as port:
        for _ in range(3):
            failure = await _get_failure(client_module, f"https://127.0.0.1:{port}/")

            assert is_transient_request_failure(failure) is True
            assert classify_transient_dependency(failure) == "providers"


async def test_a_real_handshake_reset_through_the_sdk_is_a_provider_interruption() -> None:
    async with loopback_server(reset_on_accept) as port:
        client = openai.AsyncOpenAI(
            api_key="test-key",
            base_url=f"https://127.0.0.1:{port}/v1",
            max_retries=0,
            http_client=httpx2.AsyncClient(timeout=5),
        )
        try:
            with pytest.raises(openai.APIConnectionError) as raised:
                await client.chat.completions.create(
                    model="model", messages=[{"role": "user", "content": "hi"}]
                )
        finally:
            await client.close()

    assert classify_transient_dependency(raised.value, component_hint="providers") == "providers"
    assert classify_transient_dependency(raised.value) == "providers"


@pytest.mark.parametrize("client_module", [httpx, httpx2], ids=["httpx", "httpx2"])
async def test_a_real_untrusted_certificate_is_misconfiguration(
    client_module: ModuleType,
    tmp_path: Path,
) -> None:
    server_tls = loopback_certificate(tmp_path).server_context()
    async with loopback_server(_hold_open, tls=server_tls) as port:
        failure = await _get_failure(client_module, f"https://127.0.0.1:{port}/")

    assert is_transient_request_failure(failure) is False
    assert classify_transient_dependency(failure, component_hint="providers") is None


@pytest.mark.parametrize("client_module", [httpx, httpx2], ids=["httpx", "httpx2"])
async def test_a_real_https_url_for_plain_http_is_misconfiguration(
    client_module: ModuleType,
) -> None:
    async with loopback_server(_plain_http) as port:
        failure = await _get_failure(client_module, f"https://127.0.0.1:{port}/")

    assert is_transient_request_failure(failure) is False
    assert classify_transient_dependency(failure, component_hint="providers") is None


@pytest.mark.parametrize("client_module", [httpx, httpx2], ids=["httpx", "httpx2"])
@pytest.mark.parametrize("stage", ["read", "handshake"])
async def test_a_real_https_timeout_is_transient(
    client_module: ModuleType,
    stage: str,
    tmp_path: Path,
) -> None:
    # A timed-out TLS read or handshake leaves the SSLWantReadError it was waiting
    # on at the end of the chain (Timeout <- TimeoutError <- CancelledError <-
    # SSLWantReadError), the widest chain a negative TLS rule once misread.
    certificate = loopback_certificate(tmp_path)
    release = asyncio.Event()

    async def stall(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.read(1024)
        await release.wait()
        writer.close()

    if stage == "read":
        server_tls: ssl.SSLContext | None = certificate.server_context()
        timeout = client_module.Timeout(5, read=0.2)
        expected: type[Exception] = client_module.ReadTimeout
    else:
        # A plain TCP peer never answers the ClientHello.
        server_tls = None
        timeout = client_module.Timeout(5, connect=0.2)
        expected = client_module.ConnectTimeout
    async with loopback_server(stall, tls=server_tls) as port:
        try:
            failure = await _get_failure(
                client_module,
                f"https://127.0.0.1:{port}/",
                verify=certificate.client_context(),
                timeout=timeout,
            )
        finally:
            release.set()

    assert isinstance(failure, expected)
    assert is_transient_request_failure(failure) is True
    assert classify_transient_dependency(failure) == "providers"


_V1_2 = ssl.TLSVersion.TLSv1_2
_V1_3 = ssl.TLSVersion.TLSv1_3
_LEGACY_CIPHERS = "DEFAULT:@SECLEVEL=0"


def _tls_versions_disagree(certificate: LoopbackCertificate) -> tuple[ssl.SSLContext, ...]:
    # The server alerts that the client's only version is one it does not speak.
    return certificate.server_context(maximum=_V1_2), certificate.client_context(minimum=_V1_3)


def _server_picks_an_older_version(certificate: LoopbackCertificate) -> tuple[ssl.SSLContext, ...]:
    # A legacy server answers with TLS 1.1, below the client's only version.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # enabling TLS 1.0/1.1
        legacy_server = certificate.server_context(
            minimum=ssl.TLSVersion.TLSv1,
            maximum=ssl.TLSVersion.TLSv1_1,
            ciphers=_LEGACY_CIPHERS,
        )
    return (
        legacy_server,
        certificate.client_context(minimum=_V1_2, maximum=_V1_2, ciphers=_LEGACY_CIPHERS),
    )


def _client_enables_no_version(certificate: LoopbackCertificate) -> tuple[ssl.SSLContext, ...]:
    return certificate.server_context(), certificate.client_context(minimum=_V1_3, maximum=_V1_2)


def _no_shared_cipher(certificate: LoopbackCertificate) -> tuple[ssl.SSLContext, ...]:
    return (
        certificate.server_context(maximum=_V1_2, ciphers="ECDHE-ECDSA-CHACHA20-POLY1305"),
        certificate.client_context(maximum=_V1_2, ciphers="ECDHE-ECDSA-AES128-GCM-SHA256"),
    )


def _tls_reasons(error: BaseException) -> list[object]:
    reasons: list[object] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ssl.SSLError):
            reasons.append(getattr(current, "reason", None))
        current = current.__cause__ or current.__context__
    return reasons


@pytest.mark.parametrize(
    ("contexts", "reason", "transient"),
    [
        (_tls_versions_disagree, "TLSV1_ALERT_PROTOCOL_VERSION", False),
        (_server_picks_an_older_version, "UNSUPPORTED_PROTOCOL", False),
        (_client_enables_no_version, "NO_PROTOCOLS_AVAILABLE", False),
        # A handshake the server refuses for another reason is not named
        # misconfiguration, so it stays transient.
        (_no_shared_cipher, "SSLV3_ALERT_HANDSHAKE_FAILURE", True),
    ],
    ids=["protocol-version-alert", "unsupported-protocol", "no-protocols", "handshake-failure"],
)
async def test_a_real_tls_protocol_mismatch_is_misconfiguration(
    contexts: Callable[[LoopbackCertificate], tuple[ssl.SSLContext, ...]],
    reason: str,
    transient: bool,
    tmp_path: Path,
) -> None:
    server_tls, client_tls = contexts(loopback_certificate(tmp_path))
    with alerting_tls_server(server_tls) as port:
        failure = await _get_failure(httpx, f"https://127.0.0.1:{port}/", verify=client_tls)

    assert reason in _tls_reasons(failure)
    assert is_transient_request_failure(failure) is transient
    assert (classify_transient_dependency(failure) == "providers") is transient
