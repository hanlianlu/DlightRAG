# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""One transport-neutral owner identity for durable runs and conversations.

``auth_mode="none"`` and ``auth_mode="simple"`` deliberately collapse callers
into one deployment owner; ``auth_mode="jwt"`` is the tenant boundary. Every
transport projects its own caller object into these primitives, so core never
imports a transport's user model.
"""

import hashlib


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


#: Owner used by direct in-process manager calls that carry no authenticated user.
DEPLOYMENT_OWNER_ID = owner_id_from_principal(auth_mode="none", user_id="anonymous")


__all__ = ["DEPLOYMENT_OWNER_ID", "owner_id_from_principal"]
