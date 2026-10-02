# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Edge-asserted Web identity: verify the front door's credential, not a login page.

The browser front door (Cloudflare Access, Azure Easy Auth, AWS Amplify/
CloudFront auth) has already authenticated the human and forwards a JWT. Edges
differ only in where that token rides: it is verified exactly like an API
bearer (issuer, audience, the issuer's published keys), so the Web caller is the
same owner REST and MCP see.

Verification is stateless: nothing here issues cookies or sessions, and a
missing or unverifiable credential is a rejection.
"""

from collections.abc import Callable
from typing import Literal

from starlette.requests import Request

from dlightrag.application.access import UserContext
from dlightrag.application.access.authentication import (
    AuthenticationError,
    AuthenticationSettings,
    authenticate_bearer_token,
)

type EdgeIdentityErrorKind = Literal[
    "missing_credential",
    "invalid_credential",
    "expired_credential",
    "misconfigured",
]


class EdgeIdentityError(RuntimeError):
    """The edge credential was absent, unverifiable, or the verifier is broken."""

    def __init__(self, kind: EdgeIdentityErrorKind, message: str) -> None:
        super().__init__(message)
        self.kind: EdgeIdentityErrorKind = kind


def _cloudflare_token(request: Request) -> str | None:
    """Cloudflare Access: the ``Cf-Access-Jwt-Assertion`` header, else the ``CF_Authorization`` cookie."""
    return request.headers.get("Cf-Access-Jwt-Assertion") or request.cookies.get("CF_Authorization")


def _azure_token(request: Request) -> str | None:
    """Azure Easy Auth forwards the AAD ID token; its unsigned principal header is never read."""
    return request.headers.get("X-MS-TOKEN-AAD-ID-TOKEN")


def _aws_token(request: Request) -> str | None:
    """Amplify Hosting auth and CloudFront Authorization@Edge forward the IdP JWT as a bearer."""
    header = request.headers.get("Authorization", "")
    return header.removeprefix("Bearer ") if header.startswith("Bearer ") else None


_EDGE_TOKENS: dict[str, Callable[[Request], str | None]] = {
    "cloudflare": _cloudflare_token,
    "azure": _azure_token,
    "aws": _aws_token,
}

_ERROR_KINDS: dict[str, EdgeIdentityErrorKind] = {
    "token_expired": "expired_credential",
    "verifier_misconfigured": "misconfigured",
}


def authenticate_edge(
    request: Request, *, edge: str, settings: AuthenticationSettings
) -> UserContext:
    """Verify the token the edge put on this request into the caller it names."""
    raw_token = _EDGE_TOKENS[edge](request)
    if not raw_token:
        raise EdgeIdentityError("missing_credential", "Missing edge credential")
    try:
        return authenticate_bearer_token(raw_token, settings)
    except AuthenticationError as exc:
        raise EdgeIdentityError(
            _ERROR_KINDS.get(exc.kind, "invalid_credential"), str(exc)
        ) from None


__all__ = [
    "EdgeIdentityError",
    "EdgeIdentityErrorKind",
    "authenticate_edge",
]
