# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Transport-neutral bearer authentication."""

import secrets
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Literal

import httpx
import jwt

from dlightrag.application.access.principal import UserContext

type AuthenticationErrorKind = Literal[
    "invalid_token",
    "token_expired",
    "token_subject_missing",
    "verifier_misconfigured",
]

_DISCOVERY_PATH = "/.well-known/openid-configuration"
_DISCOVERY_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True, slots=True)
class AuthenticationSettings:
    mode: Literal["none", "simple", "jwt"] = "none"
    api_token: str | None = field(default=None, repr=False)
    jwt_verification_key: str | None = field(default=None, repr=False)
    jwt_jwks_url: str | None = None
    jwt_issuer: str | None = None
    jwt_audience: str | tuple[str, ...] | None = None
    # Pins the signing algorithm. Unset, a published key names its own and a
    # static key is HS256.
    jwt_algorithm: str | None = None


class AuthenticationError(RuntimeError):
    """Bearer authentication failed with a transport-neutral reason."""

    def __init__(self, kind: AuthenticationErrorKind, message: str) -> None:
        super().__init__(message)
        self.kind = kind


@lru_cache(maxsize=16)
def _jwks_client(url: str) -> jwt.PyJWKClient:
    return jwt.PyJWKClient(url)


@lru_cache(maxsize=16)
def _issuer_keys_url(issuer: str) -> str:
    """The signing keys an issuer publishes, as its OpenID discovery document names them.

    The document must describe this very issuer, as OpenID Connect Discovery
    requires. A failure is not cached, so the next request asks again.
    """
    try:
        response = httpx.get(
            issuer.rstrip("/") + _DISCOVERY_PATH, timeout=_DISCOVERY_TIMEOUT_SECONDS
        )
        response.raise_for_status()
        document = response.json()
    except httpx.HTTPError, ValueError:
        raise AuthenticationError(
            "verifier_misconfigured", "Cannot discover the issuer's signing keys"
        ) from None
    keys_url = document.get("jwks_uri") if isinstance(document, dict) else None
    if not isinstance(keys_url, str) or not keys_url or document.get("issuer") != issuer:
        raise AuthenticationError(
            "verifier_misconfigured", "The issuer's discovery document does not name its keys"
        )
    return keys_url


def _verification_key(raw_token: str, settings: AuthenticationSettings) -> tuple[Any, str]:
    """The key that verifies this token and the algorithm it was signed with.

    An explicit key set wins, then a static key; otherwise the issuer's discovery
    document names the key set.
    """
    keys_url = settings.jwt_jwks_url
    if keys_url is None and settings.jwt_verification_key:
        return settings.jwt_verification_key, settings.jwt_algorithm or "HS256"
    if keys_url is None and settings.jwt_issuer:
        keys_url = _issuer_keys_url(settings.jwt_issuer)
    if keys_url is None:
        raise AuthenticationError("verifier_misconfigured", "JWT verification key not configured")
    try:
        key = _jwks_client(keys_url).get_signing_key_from_jwt(raw_token)
    except jwt.PyJWKClientError:
        raise AuthenticationError("invalid_token", "Invalid token") from None
    return key.key, settings.jwt_algorithm or key.algorithm_name


def _jwt_decode_kwargs(settings: AuthenticationSettings, algorithm: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"algorithms": [algorithm]}
    if settings.jwt_issuer:
        kwargs["issuer"] = settings.jwt_issuer
    if settings.jwt_audience:
        kwargs["audience"] = settings.jwt_audience
    else:
        kwargs["options"] = {"verify_aud": False}
    return kwargs


def authenticate_bearer_token(
    raw_token: str,
    settings: AuthenticationSettings,
    *,
    default_user_id: str = "anonymous",
) -> UserContext:
    """Authenticate one raw bearer token into transport-neutral caller facts."""
    if settings.mode == "none":
        return UserContext(user_id="anonymous", auth_mode="none")

    if settings.mode == "simple":
        if not settings.api_token or not secrets.compare_digest(raw_token, settings.api_token):
            raise AuthenticationError("invalid_token", "Invalid token")
        return UserContext(user_id=default_user_id, auth_mode="simple")

    try:
        key, algorithm = _verification_key(raw_token, settings)
        claims = jwt.decode(raw_token, key, **_jwt_decode_kwargs(settings, algorithm))
    except jwt.ExpiredSignatureError:
        raise AuthenticationError("token_expired", "Token expired") from None
    except jwt.InvalidTokenError:
        raise AuthenticationError("invalid_token", "Invalid token") from None
    subject = claims.get("sub")
    if not subject:
        raise AuthenticationError("token_subject_missing", "Token missing 'sub' claim")
    return UserContext(user_id=str(subject), auth_mode="jwt", claims=dict(claims))


__all__ = [
    "AuthenticationError",
    "AuthenticationErrorKind",
    "AuthenticationSettings",
    "authenticate_bearer_token",
]
