# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Pluggable authentication for DlightRAG REST API.

Routes receive UserContext via FastAPI dependency injection.
They never know which auth strategy is active.
"""

from __future__ import annotations

import logging
import secrets
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import jwt
from fastapi import HTTPException, Request
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from dlightrag.config import DlightragConfig

logger = logging.getLogger(__name__)


class UserContext(BaseModel, frozen=True):
    """Identity returned by the auth strategy. Injected into all routes."""

    user_id: str
    auth_mode: str  # "none" | "simple" | "jwt"
    claims: dict[str, object] = Field(default_factory=dict)


def _extract_bearer_token(request: Request) -> str:
    """Extract Bearer token from Authorization header. Raises 401 if missing/malformed."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    return auth_header[7:]


@lru_cache(maxsize=16)
def _jwks_client(url: str) -> jwt.PyJWKClient:
    return jwt.PyJWKClient(url)


def _jwt_signing_key(raw_token: str, cfg: DlightragConfig) -> Any:
    if cfg.jwt_jwks_url:
        try:
            return _jwks_client(cfg.jwt_jwks_url).get_signing_key_from_jwt(raw_token).key
        except jwt.PyJWKClientError:
            raise HTTPException(status_code=401, detail="Invalid token") from None
    if cfg.jwt_verification_key:
        return cfg.jwt_verification_key
    raise HTTPException(status_code=500, detail="JWT verification key not configured")


def _jwt_decode_kwargs(cfg: DlightragConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"algorithms": [cfg.jwt_algorithm]}
    if cfg.jwt_issuer:
        kwargs["issuer"] = cfg.jwt_issuer
    if cfg.jwt_audience:
        kwargs["audience"] = cfg.jwt_audience
    else:
        kwargs["options"] = {"verify_aud": False}
    return kwargs


def verify_bearer_token(
    raw_token: str,
    cfg: DlightragConfig,
    default_user_id: str = "anonymous",
) -> UserContext:
    """Validate a Bearer token against the configured auth strategy.

    Accepts a raw bearer string (the part after ``Bearer ``) and a config
    object. Returns ``UserContext`` on success or raises ``HTTPException``.
    Callable from FastAPI dependencies, Starlette middleware, or tests.

    *default_user_id* is only used in simple mode -- FastAPI may override
    it with the ``X-User-Id`` header; MCP leaves it as "anonymous".
    """
    mode = cfg.auth_mode

    if mode == "none":
        return UserContext(user_id="anonymous", auth_mode="none")

    if mode == "simple":
        if not cfg.api_auth_token or not secrets.compare_digest(raw_token, cfg.api_auth_token):
            raise HTTPException(status_code=403, detail="Invalid token")
        return UserContext(user_id=default_user_id, auth_mode="simple")

    if mode == "jwt":
        try:
            claims = jwt.decode(
                raw_token,
                _jwt_signing_key(raw_token, cfg),
                **_jwt_decode_kwargs(cfg),
            )
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired") from None
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token") from None
        sub = claims.get("sub")
        if not sub:
            raise HTTPException(status_code=401, detail="Token missing 'sub' claim")
        return UserContext(user_id=sub, auth_mode="jwt", claims=dict(claims))

    raise HTTPException(status_code=500, detail=f"Unknown auth mode: {mode}")


async def get_current_user(request: Request) -> UserContext:
    """FastAPI dependency -- extract Bearer token and delegate to verify_bearer_token."""
    from dlightrag.config import get_config

    cfg = get_config()
    if cfg.auth_mode == "none":
        return UserContext(user_id="anonymous", auth_mode="none")

    raw_token = _extract_bearer_token(request)
    user_id = request.headers.get("X-User-Id", "anonymous")
    return verify_bearer_token(raw_token, cfg, default_user_id=user_id)
