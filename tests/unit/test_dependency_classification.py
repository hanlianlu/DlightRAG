# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Closed transient dependency classification for durable execution."""

import socket
import ssl

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


def _http_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://provider.example")
    response = httpx.Response(status, request=request)
    return httpx.HTTPStatusError("provider rejected request", request=request, response=response)


@pytest.mark.parametrize("status", [401, 403, 400, 413, 422])
def test_auth_input_and_context_rejections_are_not_transient(status: int) -> None:
    assert classify_transient_dependency(_http_error(status)) is None


@pytest.mark.parametrize("status", [408, 425, 429, 500, 502, 503, 504])
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


@pytest.mark.parametrize("status", [408, 425, 429, 500, 502, 503, 504])
def test_retryable_statuses_are_transient_requests(status: int) -> None:
    assert is_transient_request_failure(_http_error(status)) is True
    stamped = RuntimeError(f"parser upload failed: HTTP {status}")
    stamped.status_code = status  # type: ignore[attr-defined]
    assert is_transient_request_failure(stamped) is True


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 413, 422, 501, 505])
def test_rejected_and_unlisted_statuses_are_not_transient_requests(status: int) -> None:
    assert is_transient_request_failure(_http_error(status)) is False


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
        socket.gaierror(socket.EAI_NONAME, "nodename nor servname provided, or not known"),
        ssl.SSLCertVerificationError(
            1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed"
        ),
        ssl.SSLError(1, "[SSL: WRONG_VERSION_NUMBER] wrong version number"),
    ]


@pytest.mark.parametrize(
    "cause",
    _misconfigured_causes(),
    ids=["dns-no-such-name", "tls-certificate", "https-to-plain-http"],
)
def test_a_misconfigured_endpoint_is_not_an_outage(cause: BaseException) -> None:
    # httpx and the SDKs' own client raise these connect failures with the DNS
    # or TLS error as the cause; resending the request cannot fix either.
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


@pytest.mark.parametrize(
    "cause",
    [
        socket.gaierror(socket.EAI_AGAIN, "Temporary failure in name resolution"),
        ssl.SSLEOFError(8, "EOF occurred in violation of protocol"),
        ConnectionRefusedError(61, "Connection refused"),
    ],
    ids=["dns-temporary-failure", "tls-dropped-mid-handshake", "refused"],
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
