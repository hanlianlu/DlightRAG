# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Which authentication modes name a personal owner.

Profile Memory and Personal MCP Connections belong to one person, so both need
an owner identity that is that person's alone: a JWT subject, or the stable
single user of a deployment without authentication. ``simple`` is a shared
password bucket, not a personal identity (ADR 0012).
"""

PERSONAL_AUTH_MODES = frozenset({"jwt", "none"})


def personal_owner(auth_mode: str) -> bool:
    """Whether an owner authenticated this way is one person's identity."""
    return auth_mode in PERSONAL_AUTH_MODES


__all__ = ["PERSONAL_AUTH_MODES", "personal_owner"]
