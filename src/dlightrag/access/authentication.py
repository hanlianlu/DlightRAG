# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Transport-neutral bearer authentication."""

import secrets
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

import jwt

from dlightrag.access.principal import UserContext

type AuthenticationErrorKind = Literal[
    "invalid_token",
    "token_expired",
    "token_subject_missing",
    "verifier_misconfigured",
]


@dataclass(frozen=True, slots=True)
class AuthenticationSettings:
    mode: Literal["none", "simple", "jwt"] = "none"
    api_token: str | None = None
    jwt_verification_key: str | None = None
    jwt_jwks_url: str | None = None
    jwt_issuer: str | None = None
    jwt_audience: str | tuple[str, ...] | None = None
    jwt_algorithm: str = "HS256"


class AuthenticationError(RuntimeError):
    """Bearer authentication failed with a transport-neutral reason."""

    def __init__(self, kind: AuthenticationErrorKind, message: str) -> None:
        super().__init__(message)
        self.kind = kind


@lru_cache(maxsize=16)
def _jwks_client(url: str) -> jwt.PyJWKClient:
    return jwt.PyJWKClient(url)


def _jwt_signing_key(raw_token: str, settings: AuthenticationSettings) -> Any:
    if settings.jwt_jwks_url:
        try:
            return _jwks_client(settings.jwt_jwks_url).get_signing_key_from_jwt(raw_token).key
        except jwt.PyJWKClientError:
            raise AuthenticationError("invalid_token", "Invalid token") from None
    if settings.jwt_verification_key:
        return settings.jwt_verification_key
    raise AuthenticationError("verifier_misconfigured", "JWT verification key not configured")


def _jwt_decode_kwargs(settings: AuthenticationSettings) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"algorithms": [settings.jwt_algorithm]}
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
        claims = jwt.decode(
            raw_token,
            _jwt_signing_key(raw_token, settings),
            **_jwt_decode_kwargs(settings),
        )
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
