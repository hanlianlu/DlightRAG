# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Edge-asserted Web identity: verify the front door's credential, not a login page.

The browser front door (Cloudflare Access, Azure Easy Auth, AWS Amplify/
CloudFront auth) has already authenticated the human. Each provider extracts
the edge credential from the proxied request, verifies it cryptographically
against the edge's published keys, and returns a transport-neutral
:class:`EdgeIdentity`; the Web middleware projects it into the same
``UserContext`` (and therefore the same owner) the rest of the product uses.

Verification is stateless: nothing here issues cookies or sessions, and a
missing or unverifiable credential is a rejection.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal, Protocol

import jwt
from starlette.requests import Request

from dlightrag.config import WebIdentitySettings

EdgeIdentityErrorKind = Literal[
    "missing_credential",
    "invalid_credential",
    "expired_credential",
    "misconfigured",
]

_EDGE_ERROR_STATUS: dict[EdgeIdentityErrorKind, int] = {
    "missing_credential": 401,
    "invalid_credential": 401,
    "expired_credential": 401,
    "misconfigured": 500,
}

_CLOUDFLARE_CERTS_PATH = "/cdn-cgi/access/certs"


@dataclass(frozen=True, slots=True)
class EdgeIdentity:
    """One verified edge-asserted caller."""

    issuer: str
    subject: str
    claims: dict[str, Any]


class EdgeIdentityError(RuntimeError):
    """The edge credential was absent, unverifiable, or the verifier is broken."""

    def __init__(self, kind: EdgeIdentityErrorKind, message: str) -> None:
        super().__init__(message)
        self.kind: EdgeIdentityErrorKind = kind

    @property
    def http_status(self) -> int:
        return _EDGE_ERROR_STATUS[self.kind]


class EdgeIdentityProvider(Protocol):
    """Resolve one request's edge-asserted identity or raise EdgeIdentityError."""

    def authenticate(self, request: Request) -> EdgeIdentity: ...


@lru_cache(maxsize=16)
def _jwks_client(url: str) -> jwt.PyJWKClient:
    return jwt.PyJWKClient(url)


def _decode_edge_jwt(
    raw_token: str,
    *,
    jwks_url: str,
    issuer: str,
    audience: str | list[str],
    algorithms: tuple[str, ...] = ("RS256",),
) -> dict[str, Any]:
    try:
        key = _jwks_client(jwks_url).get_signing_key_from_jwt(raw_token).key
    except jwt.PyJWKClientError:
        raise EdgeIdentityError("invalid_credential", "Invalid edge credential") from None
    try:
        return jwt.decode(
            raw_token,
            key,
            algorithms=list(algorithms),
            issuer=issuer,
            audience=audience,
        )
    except jwt.ExpiredSignatureError:
        raise EdgeIdentityError("expired_credential", "Edge credential expired") from None
    except jwt.InvalidTokenError:
        raise EdgeIdentityError("invalid_credential", "Invalid edge credential") from None


def _identity_from_claims(claims: dict[str, Any]) -> EdgeIdentity:
    subject = claims.get("sub")
    issuer = claims.get("iss")
    if not subject or not isinstance(subject, str) or not issuer or not isinstance(issuer, str):
        raise EdgeIdentityError("invalid_credential", "Edge credential missing 'sub' claim")
    return EdgeIdentity(issuer=issuer, subject=subject, claims=dict(claims))


class CloudflareAccessProvider:
    """Verify the Cloudflare Access JWT injected on every proxied request.

    Primary credential is the ``Cf-Access-Jwt-Assertion`` header; the
    ``CF_Authorization`` cookie is the fallback. The token is signed by the
    team's Access keys (``https://<team>.cloudflareaccess.com/cdn-cgi/access/certs``)
    and carries ``iss`` (the team domain), ``aud`` (the application AUD tag),
    ``sub``, ``exp``, and a session ``identity_nonce``.
    """

    def __init__(self, settings: WebIdentitySettings) -> None:
        issuer = (settings.issuer or "").rstrip("/")
        if not issuer:
            raise EdgeIdentityError("misconfigured", "Cloudflare Access issuer not configured")
        self._issuer = issuer
        self._audience = settings.audience or ""
        self._jwks_url = settings.jwks_url or f"{issuer}{_CLOUDFLARE_CERTS_PATH}"

    def authenticate(self, request: Request) -> EdgeIdentity:
        raw_token = request.headers.get("Cf-Access-Jwt-Assertion") or request.cookies.get(
            "CF_Authorization"
        )
        if not raw_token:
            raise EdgeIdentityError(
                "missing_credential",
                "Missing Cloudflare Access credential",
            )
        claims = _decode_edge_jwt(
            raw_token,
            jwks_url=self._jwks_url,
            issuer=self._issuer,
            audience=self._audience,
        )
        return _identity_from_claims(claims)


def edge_identity_provider(settings: WebIdentitySettings) -> EdgeIdentityProvider:
    """Build the configured edge provider, raising on unknown edges."""
    if settings.edge == "cloudflare":
        return CloudflareAccessProvider(settings)
    raise EdgeIdentityError(
        "misconfigured",
        f"Unsupported web identity edge: {settings.edge}",
    )


__all__ = [
    "CloudflareAccessProvider",
    "EdgeIdentity",
    "EdgeIdentityError",
    "EdgeIdentityErrorKind",
    "EdgeIdentityProvider",
    "edge_identity_provider",
]
