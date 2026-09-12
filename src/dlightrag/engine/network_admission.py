# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared DNS/IP policy and address pinning for public GET and MCP transport."""

from __future__ import annotations

import asyncio
import fnmatch
import ipaddress
import socket
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, urljoin, urlparse, urlsplit, urlunsplit

_PUBLIC_NETWORK_ADMISSION = asyncio.Semaphore(32)


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
class _ResolvedTarget:
    url: str
    host: str
    port: int
    addresses: tuple[str, ...]


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
    validate_credential_free_query(raw_url)
    return raw_url


def validate_credential_free_query(raw_url: str) -> None:
    """Reject credentials in URL query data, including redirects and metadata URLs."""
    for name, _value in parse_qsl(urlsplit(raw_url).query, keep_blank_values=True):
        normalized = name.lower().strip()
        if normalized in _SENSITIVE_QUERY_NAMES or normalized.endswith(("_token", "_secret")):
            raise PublicHttpPolicyError(
                "Agent URL reads do not accept credential or signed query parameters"
            )


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
    if parsed.username is not None or parsed.password is not None:
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
    if not _public_unicast(ip):
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
        if not allow_private and not _public_unicast(ipaddress.ip_address(address)):
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


def _public_unicast(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return ip.is_global and not ip.is_multicast and not ip.is_reserved and not ip.is_unspecified
