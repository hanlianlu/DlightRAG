# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Transport-neutral caller identity and durable owner projection."""

import hashlib

from pydantic import BaseModel, Field


class UserContext(BaseModel, frozen=True):
    """Authenticated caller facts shared by every transport."""

    user_id: str
    auth_mode: str
    claims: dict[str, object] = Field(default_factory=dict)


def owner_id_from_principal(
    *,
    auth_mode: str,
    user_id: str,
    issuer: str | None = None,
) -> str:
    """Project an authenticated principal into a stable owner namespace."""
    if auth_mode == "none":
        namespace = "none\0deployment\0anonymous"
    elif auth_mode == "simple":
        namespace = "simple\0deployment\0shared"
    else:
        namespace = f"jwt\0{issuer or 'unscoped'}\0{user_id}"
    return hashlib.sha256(namespace.encode("utf-8")).hexdigest()


DEPLOYMENT_OWNER_ID = owner_id_from_principal(auth_mode="none", user_id="anonymous")


def owner_id_from_user(user: UserContext | None) -> str:
    """Return the owner that scopes this caller's runs, events, and artifacts."""
    if user is None:
        return DEPLOYMENT_OWNER_ID
    return owner_id_from_principal(
        auth_mode=user.auth_mode,
        user_id=user.user_id,
        issuer=str(user.claims.get("iss") or "") or None,
    )


__all__ = [
    "DEPLOYMENT_OWNER_ID",
    "UserContext",
    "owner_id_from_principal",
    "owner_id_from_user",
]
