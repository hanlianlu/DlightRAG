# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Pluggable authentication for DlightRAG REST API.

Routes receive UserContext via FastAPI dependency injection.
They never know which auth strategy is active.
"""

from __future__ import annotations

import logging

import jwt
from fastapi import HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class UserContext(BaseModel, frozen=True):
    """Identity returned by the auth strategy. Injected into all routes."""

    user_id: str
    auth_mode: str  # "none" | "simple" | "jwt"


def _get_auth_config() -> tuple[str, str | None, str | None, str]:
    """Read auth config from global singleton. Returns (mode, token, jwt_secret, jwt_alg)."""
    from dlightrag.config import get_config

    cfg = get_config()
    return cfg.auth_mode, cfg.api_auth_token, cfg.jwt_secret, cfg.jwt_algorithm


def _extract_bearer_token(request: Request) -> str:
    """Extract Bearer token from Authorization header. Raises 401 if missing/malformed."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid Authorization header")
    return auth_header[7:]


async def get_current_user(request: Request) -> UserContext:
    """FastAPI dependency -- dispatches to auth strategy based on config.

    Inject into routes: ``user: UserContext = Depends(get_current_user)``
    """
    mode, token, jwt_secret, jwt_algorithm = _get_auth_config()

    if mode == "none":
        return UserContext(user_id="anonymous", auth_mode="none")

    if mode == "simple":
        provided = _extract_bearer_token(request)
        if not token or provided != token:
            raise HTTPException(status_code=403, detail="Invalid token")
        user_id = request.headers.get("X-User-Id", "anonymous")
        return UserContext(user_id=user_id, auth_mode="simple")

    if mode == "jwt":
        if not jwt_secret:
            raise HTTPException(status_code=500, detail="JWT secret not configured")
        raw_token = _extract_bearer_token(request)
        try:
            claims = jwt.decode(raw_token, jwt_secret, algorithms=[jwt_algorithm])
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired") from None
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token") from None
        sub = claims.get("sub")
        if not sub:
            raise HTTPException(status_code=401, detail="Token missing 'sub' claim")
        return UserContext(user_id=sub, auth_mode="jwt")

    raise HTTPException(status_code=500, detail=f"Unknown auth mode: {mode}")
