# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Stable PostgreSQL advisory-lock identities."""

import hashlib


def advisory_lock_key(namespace: str, scope: str) -> int:
    """Return one stable signed 64-bit key for ``namespace`` and ``scope``."""
    digest = hashlib.blake2b(f"{namespace}:{scope}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big", signed=True)


__all__ = ["advisory_lock_key"]
