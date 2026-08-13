# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Project an authenticated REST/Web user into the core owner namespace."""

from dlightrag.api.auth import UserContext
from dlightrag.core.principal import owner_id_from_principal


def owner_id_from_user(user: UserContext | None) -> str:
    """Return the owner that scopes this caller's runs, events, and artifacts."""
    if user is None:
        return owner_id_from_principal(auth_mode="none", user_id="anonymous")
    return owner_id_from_principal(
        auth_mode=user.auth_mode,
        user_id=user.user_id,
        issuer=str(user.claims.get("iss") or "") or None,
    )


__all__ = ["owner_id_from_user"]
