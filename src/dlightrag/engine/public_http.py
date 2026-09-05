# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Safe anonymous GET acquisition for public HTTP(S) resources.

The module is deliberately product-neutral. It validates every redirect hop,
forbids HTTPS-to-HTTP downgrade, bounds response bytes, and pins each connection
to an IP address returned by the validated DNS lookup. Connecting to that
validated address, rather than resolving the hostname again inside the
transport, closes the DNS rebinding/TOCTOU gap.
"""

from __future__ import annotations

import asyncio
import fnmatch
import ipaddress
import socket
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urljoin, urlparse, urlsplit, urlunsplit

import httpx

_MAX_REDIRECTS = 5
_PUBLIC_NETWORK_CONCURRENCY = 32
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_PUBLIC_NETWORK_ADMISSION = asyncio.Semaphore(_PUBLIC_NETWORK_CONCURRENCY)


class PublicHttpPolicyError(ValueError):
    """A URL or redirect violates the public anonymous-GET policy."""


_SENSITIVE_QUERY_NAMES = frozenset(
    {
        "access_token",
        "api-key",
        "api_key",
        "apikey",
        "auth",
        "authorization",
        "credential",
        "credentials",
        "jwt",
        "key",
        "password",
        "secret",
        "sig",
        "signature",
        "token",
        "x-amz-credential",
        "x-amz-security-token",
        "x-amz-signature",
        "x-goog-credential",
        "x-goog-signature",
    }
)


@dataclass(frozen=True, slots=True)
class PublicHttpPresentation:
    """Bounded representation preferences allowed on an Agent URL read."""

    user_agent: str | None = None
    accept: str | None = None
    accept_language: str | None = None

    def __post_init__(self) -> None:
        for name, value, limit in (
            ("user_agent", self.user_agent, 256),
            ("accept", self.accept, 512),
            ("accept_language", self.accept_language, 256),
        ):
            if value is None:
                continue
            if not value.strip() or len(value) > limit or "\r" in value or "\n" in value:
                raise ValueError(f"invalid public HTTP {name}")

    def headers(self) -> dict[str, str]:
        values = {
            "user-agent": self.user_agent,
            "accept": self.accept,
            "accept-language": self.accept_language,
        }
        return {name: value for name, value in values.items() if value is not None}


@dataclass(frozen=True, slots=True)
class PublicHttpFetch:
    """One complete bounded response and its canonical final URL."""

    content: bytes
    final_url: str
    media_type: str | None
    status_code: int


@dataclass(frozen=True, slots=True)
class PublicHttpDownload:
    """Metadata for one complete bounded response streamed to disk."""

    final_url: str
    media_type: str | None
    status_code: int
    size_bytes: int


@dataclass(frozen=True, slots=True)
class _ResolvedTarget:
    url: str
    host: str
    port: int
    addresses: tuple[str, ...]


async def fetch_public_http(
    url: str,
    *,
    max_bytes: int,
    timeout: float = 120.0,
    presentation: PublicHttpPresentation | None = None,
    allow_private_hosts: Sequence[str] = (),
    client: Any | None = None,
    agent_url: bool = False,
) -> PublicHttpFetch:
    """Fetch one public HTTP(S) URL under redirect, SSRF, and byte bounds."""
    limit = max(1, int(max_bytes))

    async def consume(response: Any) -> tuple[bytes, int]:
        chunks: list[bytes] = []
        written = 0
        async for chunk in response.aiter_bytes():
            if not chunk:
                continue
            written += len(chunk)
            if written > limit:
                raise ValueError(f"url fetch exceeds maximum size of {limit} bytes")
            chunks.append(chunk)
        return b"".join(chunks), written

    async with public_network_admission(), asyncio.timeout(timeout):
        (content, _), target, response = await _follow_and_consume(
            url,
            timeout=timeout,
            presentation=presentation or PublicHttpPresentation(),
            allow_private_hosts=_normalize_host_patterns(allow_private_hosts),
            client=client,
            agent_url=agent_url,
            consume=consume,
        )
    return PublicHttpFetch(
        content=content,
        final_url=normalize_public_http_url_identity(target.url),
        media_type=_media_type(response),
        status_code=int(response.status_code),
    )


async def download_public_http(
    url: str,
    destination: Path,
    *,
    max_bytes: int,
    timeout: float = 120.0,
    allow_private_hosts: Sequence[str] = (),
    client: Any | None = None,
) -> PublicHttpDownload:
    """Stream one public HTTP(S) response to *destination* under the same policy."""
    limit = max(1, int(max_bytes))

    async def consume(response: Any) -> tuple[None, int]:
        written = 0
        with destination.open("wb") as out:
            async for chunk in response.aiter_bytes():
                if not chunk:
                    continue
                written += len(chunk)
                if written > limit:
                    raise ValueError(f"url ingest exceeds maximum size of {limit} bytes")
                out.write(chunk)
        return None, written

    try:
        async with public_network_admission(), asyncio.timeout(timeout):
            (_, size), target, response = await _follow_and_consume(
                url,
                timeout=timeout,
                presentation=PublicHttpPresentation(),
                allow_private_hosts=_normalize_host_patterns(allow_private_hosts),
                client=client,
                agent_url=False,
                consume=consume,
            )
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return PublicHttpDownload(
        final_url=normalize_public_http_url_identity(target.url),
        media_type=_media_type(response),
        status_code=int(response.status_code),
        size_bytes=size,
    )


async def _follow_and_consume[T](
    url: str,
    *,
    timeout: float,
    presentation: PublicHttpPresentation,
    allow_private_hosts: frozenset[str],
    client: Any | None,
    agent_url: bool,
    consume: Callable[[Any], Awaitable[T]],
) -> tuple[T, _ResolvedTarget, Any]:
    current_url = url
    for _ in range(_MAX_REDIRECTS + 1):
        if agent_url:
            validate_agent_public_url(current_url)
        target = await _resolve_public_target(
            current_url,
            allow_private_hosts=allow_private_hosts,
        )
        async with _pinned_stream(
            target,
            timeout=timeout,
            presentation=presentation,
            client=client,
        ) as response:
            if response.status_code in _REDIRECT_STATUSES:
                current_url = _redirect_target(
                    target.url,
                    response,
                    current_scheme=_url_scheme(target.url),
                )
                continue
            response.raise_for_status()
            value = await consume(response)
            return value, target, response
    raise PublicHttpPolicyError("url fetch exceeded maximum redirects")


def _pinned_stream(
    target: _ResolvedTarget,
    *,
    timeout: float,
    presentation: PublicHttpPresentation,
    client: Any | None,
) -> Any:
    """Open one request against a validated address with original Host/TLS SNI."""
    address = target.addresses[0]
    parts = urlsplit(target.url)
    explicit_port = parts.port is not None
    address_host = f"[{address}]" if ":" in address else address
    netloc = f"{address_host}:{target.port}" if explicit_port else address_host
    pinned_url = urlunsplit((parts.scheme, netloc, parts.path, parts.query, ""))
    default_port = 80 if parts.scheme.lower() == "http" else 443
    original_host = f"[{target.host}]" if ":" in target.host else target.host
    authority = f"{original_host}:{target.port}" if target.port != default_port else original_host
    headers = {
        "host": authority,
        # Prevent a pooled IP-origin connection from crossing logical hostnames.
        "connection": "close",
        **presentation.headers(),
    }
    extensions: dict[str, Any] = {
        "timeout": httpx.Timeout(timeout).as_dict(),
    }
    if parts.scheme.lower() == "https":
        extensions["sni_hostname"] = target.host
    if isinstance(client, httpx.AsyncClient):
        request = httpx.Request(
            "GET",
            pinned_url,
            headers=headers,
            extensions=extensions,
        )
        return _BorrowedStream(client, request)
    active = client or httpx.AsyncClient(
        follow_redirects=False,
        timeout=httpx.Timeout(timeout),
        trust_env=False,
    )
    stream = active.stream(
        "GET",
        pinned_url,
        headers=headers,
        extensions=extensions,
    )
    if client is not None:
        return stream
    return _OwnedStream(active, stream)


class _BorrowedStream:
    """Stream a raw request without inheriting client auth, cookies, or headers."""

    def __init__(self, client: httpx.AsyncClient, request: httpx.Request) -> None:
        self._client = client
        self._request = request
        self._response: httpx.Response | None = None

    async def __aenter__(self) -> httpx.Response:
        self._response = await self._client.send(
            self._request,
            stream=True,
            auth=None,
            follow_redirects=False,
        )
        return self._response

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._response is not None:
            await self._response.aclose()


class _OwnedStream:
    def __init__(self, client: httpx.AsyncClient, stream: Any) -> None:
        self._client = client
        self._stream = stream
        self._response: Any = None

    async def __aenter__(self) -> Any:
        try:
            self._response = await self._stream.__aenter__()
            return self._response
        except BaseException:
            await self._client.aclose()
            raise

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        try:
            await self._stream.__aexit__(exc_type, exc, tb)
        finally:
            await self._client.aclose()


def validate_public_http_url(
    raw_url: str,
    *,
    resolve_host: bool = False,
    allow_private_hosts: Sequence[str] = (),
) -> str:
    """Apply HTTP(S) policy, with explicit private-host exceptions for ingestion."""
    patterns = _normalize_host_patterns(allow_private_hosts)
    pending = _static_url_checks(raw_url, patterns)
    if resolve_host and pending is not None:
        host, port = pending
        _resolve_and_validate(
            host,
            port,
            allow_private=_host_allowed_private(host, patterns),
        )
    return raw_url


@asynccontextmanager
async def public_network_admission() -> AsyncIterator[None]:
    """Bound process-wide public-network work across direct and hosted paths."""
    async with _PUBLIC_NETWORK_ADMISSION:
        yield


async def avalidate_public_http_url(
    raw_url: str,
    *,
    allow_private_hosts: Sequence[str] = (),
    timeout: float = 10.0,
) -> str:
    """Apply HTTP(S) policy with bounded DNS resolution off the event loop."""
    async with public_network_admission(), asyncio.timeout(timeout):
        await _resolve_public_target(
            raw_url,
            allow_private_hosts=_normalize_host_patterns(allow_private_hosts),
        )
    return raw_url


def validate_public_web_url(raw_url: str) -> str:
    """Validate a public HTTP(S) provenance URL for browser navigation."""
    return validate_public_http_url(raw_url)


def validate_agent_public_url(raw_url: str) -> str:
    """Reject credential-bearing/signed URLs before they become Agent resources."""
    validate_public_http_url(raw_url)
    for name, _value in parse_qsl(urlsplit(raw_url).query, keep_blank_values=True):
        normalized = name.lower().strip()
        if normalized in _SENSITIVE_QUERY_NAMES or normalized.endswith(("_token", "_secret")):
            raise PublicHttpPolicyError(
                "Agent URL reads do not accept credential or signed query parameters"
            )
    return raw_url


def normalize_public_http_url_identity(url: str) -> str:
    """Normalize scheme/authority and discard fragments that never reach the server."""
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    host = _normalize_host(parts.hostname or "")
    rendered_host = f"[{host}]" if ":" in host else host
    port = parts.port
    default_port = 80 if scheme == "http" else 443 if scheme == "https" else None
    netloc = rendered_host if port is None or port == default_port else f"{rendered_host}:{port}"
    return urlunsplit((scheme, netloc, parts.path, parts.query, ""))


async def _resolve_public_target(
    raw_url: str,
    *,
    allow_private_hosts: frozenset[str],
) -> _ResolvedTarget:
    pending = _static_url_checks(raw_url, allow_private_hosts)
    parsed = urlsplit(raw_url)
    host = _normalize_host(parsed.hostname or "")
    port = parsed.port or (80 if parsed.scheme.lower() == "http" else 443)
    if pending is None:
        addresses = (
            (str(ipaddress.ip_address(host)),)
            if _is_ip(host)
            else await asyncio.to_thread(
                _resolve_and_validate,
                host,
                port,
                allow_private=True,
            )
        )
    else:
        addresses = await asyncio.to_thread(
            _resolve_and_validate,
            pending[0],
            pending[1],
            allow_private=False,
        )
    if not addresses:
        raise PublicHttpPolicyError("url fetch requires a resolvable host")
    return _ResolvedTarget(raw_url, host, port, addresses)


def _static_url_checks(
    raw_url: str,
    allow_private_hosts: frozenset[str],
) -> tuple[str, int] | None:
    parsed = urlparse(raw_url)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise PublicHttpPolicyError("url fetch only accepts http or https URLs")
    if not parsed.hostname:
        raise PublicHttpPolicyError("url fetch requires a hostname")
    if parsed.username or parsed.password:
        raise PublicHttpPolicyError("url fetch does not accept credentials in URLs")
    host = _normalize_host(parsed.hostname)
    if _host_allowed_private(host, allow_private_hosts):
        return None
    if host == "localhost" or host.endswith(".localhost") or host.endswith(".local"):
        raise PublicHttpPolicyError("url fetch requires a public host")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return (host, parsed.port or (80 if scheme == "http" else 443))
    if not ip.is_global:
        raise PublicHttpPolicyError("url fetch requires a public host")
    return None


def _resolve_and_validate(host: str, port: int, *, allow_private: bool) -> tuple[str, ...]:
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except OSError as exc:
        raise PublicHttpPolicyError("url fetch requires a resolvable public host") from exc
    addresses: list[str] = []
    for info in infos:
        sockaddr = info[4]
        if not sockaddr:
            continue
        address = str(sockaddr[0])
        if not allow_private and not ipaddress.ip_address(address).is_global:
            raise PublicHttpPolicyError("url fetch requires a public host")
        if address not in addresses:
            addresses.append(address)
    return tuple(addresses)


def _redirect_target(current_url: str, response: Any, *, current_scheme: str) -> str:
    headers = getattr(response, "headers", {}) or {}
    location = headers.get("location") or headers.get("Location")
    if not location:
        raise PublicHttpPolicyError("url redirect is missing Location header")
    target = urljoin(current_url, str(location))
    if current_scheme == "https" and _url_scheme(target) == "http":
        raise PublicHttpPolicyError("url redirect cannot downgrade https to http")
    return target


def _media_type(response: Any) -> str | None:
    value = str((getattr(response, "headers", {}) or {}).get("content-type") or "").strip()
    return value or None


def _url_scheme(url: str) -> str:
    return urlparse(url).scheme.lower()


def _normalize_host_patterns(values: Sequence[str]) -> frozenset[str]:
    return frozenset(_normalize_host(value) for value in values if value)


def _normalize_host(value: str) -> str:
    return value.lower().strip("[]").rstrip(".")


def _host_allowed_private(host: str, patterns: frozenset[str]) -> bool:
    return any(fnmatch.fnmatchcase(host, pattern) for pattern in patterns)


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


__all__ = [
    "PublicHttpDownload",
    "PublicHttpFetch",
    "PublicHttpPresentation",
    "PublicHttpPolicyError",
    "avalidate_public_http_url",
    "download_public_http",
    "fetch_public_http",
    "normalize_public_http_url_identity",
    "public_network_admission",
    "validate_agent_public_url",
    "validate_public_http_url",
    "validate_public_web_url",
]
