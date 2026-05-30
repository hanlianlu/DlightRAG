# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Pluggable authentication for DlightRAG REST API.

Routes receive UserContext via FastAPI dependency injection.
They never know which auth strategy is active.
"""

from __future__ import annotations

import logging
import secrets

import jwt
from fastapi import HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class UserContext(BaseModel, frozen=True):
    """Identity returned by the auth strategy. Injected into all routes."""

    user_id: str
    auth_mode: str  # "none" | "simple" | "jwt"


def _extract_bearer_token(request: Request) -> str:
    """Extract Bearer token from Authorization header. Raises 401 if missing/malformed."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    return auth_header[7:]


def verify_bearer_token(
    raw_token: str,
    cfg: "DlightragConfig",
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
        if not cfg.jwt_secret:
            raise HTTPException(status_code=500, detail="JWT secret not configured")
        try:
            claims = jwt.decode(raw_token, cfg.jwt_secret, algorithms=[cfg.jwt_algorithm])
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired") from None
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token") from None
        sub = claims.get("sub")
        if not sub:
            raise HTTPException(status_code=401, detail="Token missing 'sub' claim")
        return UserContext(user_id=sub, auth_mode="jwt")

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
