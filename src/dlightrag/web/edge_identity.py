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

_CLOUDFLARE_CERTS_PATH = "/cdn-cgi/access/certs"


@dataclass(frozen=True, slots=True)
class EdgeIdentity:
    """One verified edge-asserted caller."""

    issuer: str
    subject: str
    claims: dict[str, Any]
    display_claims: dict[str, Any] | None = None
    """Unsigned, platform-injected enrichment (e.g. Azure's principal header).

    Never merged into ``claims``: it is not cryptographically verified and
    must not influence authorization.
    """


class EdgeIdentityError(RuntimeError):
    """The edge credential was absent, unverifiable, or the verifier is broken."""

    def __init__(self, kind: EdgeIdentityErrorKind, message: str) -> None:
        super().__init__(message)
        self.kind: EdgeIdentityErrorKind = kind


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
    if settings.edge == "azure":
        return AzureEasyAuthProvider(settings)
    raise EdgeIdentityError(
        "misconfigured",
        f"Unsupported web identity edge: {settings.edge}",
    )


class AzureEasyAuthProvider:
    """Verify the AAD ID token Azure Easy Auth passes through.

    Easy Auth authenticates the browser and forwards the IdP token as
    ``X-MS-TOKEN-AAD-ID-TOKEN``. That token is a real AAD ID token: it is
    verified against the tenant discovery keys with the configured tenant
    issuer and the App Registration client id as audience. The unsigned
    ``X-MS-CLIENT-PRINCIPAL`` header is parsed only as display enrichment and
    never influences authorization — a request with a principal header but no
    verifiable ID token is rejected.
    """

    def __init__(self, settings: WebIdentitySettings) -> None:
        issuer = (settings.issuer or "").rstrip("/")
        if not issuer:
            raise EdgeIdentityError("misconfigured", "Azure issuer not configured")
        self._issuer = issuer
        self._audience = settings.audience or ""
        self._jwks_url = settings.jwks_url or _azure_discovery_url(issuer)

    def authenticate(self, request: Request) -> EdgeIdentity:
        raw_token = request.headers.get("X-MS-TOKEN-AAD-ID-TOKEN")
        if not raw_token:
            raise EdgeIdentityError(
                "missing_credential",
                "Missing Azure Easy Auth ID token",
            )
        claims = _decode_edge_jwt(
            raw_token,
            jwks_url=self._jwks_url,
            issuer=self._issuer,
            audience=self._audience,
        )
        identity = _identity_from_claims(claims)
        return EdgeIdentity(
            issuer=identity.issuer,
            subject=identity.subject,
            claims=identity.claims,
            display_claims=_parse_principal_header(request.headers.get("X-MS-CLIENT-PRINCIPAL")),
        )


def _azure_discovery_url(issuer: str) -> str:
    """Derive the AAD v2 discovery keys endpoint from a v2 issuer."""
    if issuer.endswith("/v2.0"):
        return f"{issuer[:-5]}/discovery/v2.0/keys"
    raise EdgeIdentityError(
        "misconfigured",
        "Azure issuer is not a v2 issuer; configure web_identity.jwks_url explicitly",
    )


def _parse_principal_header(value: str | None) -> dict[str, Any] | None:
    """Decode the unsigned principal header for display; never for trust."""
    if not value:
        return None
    import base64
    import json as _json

    try:
        padded = value + ("=" * (-len(value) % 4))
        payload = _json.loads(base64.b64decode(padded, validate=True).decode())
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


__all__ = [
    "AzureEasyAuthProvider",
    "CloudflareAccessProvider",
    "EdgeIdentity",
    "EdgeIdentityError",
    "EdgeIdentityErrorKind",
    "EdgeIdentityProvider",
    "edge_identity_provider",
]
