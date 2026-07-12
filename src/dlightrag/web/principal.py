# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Server-owned principal identity for durable Web conversations."""

import hashlib

from dlightrag.api.auth import UserContext


def principal_id_from_user(user: UserContext | None) -> str:
    """Project an authenticated user into a stable conversation owner namespace."""
    if user is None or user.auth_mode == "none":
        namespace = "none\0deployment\0anonymous"
    elif user.auth_mode == "simple":
        namespace = "simple\0deployment\0shared"
    else:
        issuer = str(user.claims.get("iss") or "unscoped")
        namespace = f"jwt\0{issuer}\0{user.user_id}"
    return hashlib.sha256(namespace.encode("utf-8")).hexdigest()


__all__ = ["principal_id_from_user"]
