# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""SQL identifier validation for trusted dynamic PostgreSQL paths."""

import re

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def pg_identifier(identifier: str) -> str:
    """Validate a PostgreSQL identifier used in trusted dynamic SQL."""
    if not _IDENTIFIER_RE.fullmatch(identifier):
        raise ValueError(f"Unsafe PostgreSQL identifier: {identifier!r}")
    return identifier


def pg_qualified_identifier(identifier: str) -> str:
    """Validate a PostgreSQL identifier with optional schema qualification."""
    parts = identifier.split(".")
    if not parts or any(not part for part in parts):
        raise ValueError(f"Unsafe PostgreSQL identifier: {identifier!r}")
    return ".".join(pg_identifier(part) for part in parts)
