# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared utilities."""

import re

from dlightrag.utils.images import image_data_uri as image_data_uri

_WORKSPACE_FORBIDDEN_RE = re.compile(r'[/\\<>"\']')


def normalize_workspace(name: str) -> str:
    """Normalize workspace name to a safe, lowercase PG identifier.

    Replaces non-alphanumeric/underscore characters with ``_``, lowercases,
    and prepends ``_`` if the name starts with a digit (invalid PG schema).

    Examples::

        normalize_workspace("Ian Davis")   -> "ian_davis"
        normalize_workspace("legacy-workspace") -> "legacy_workspace"
        normalize_workspace("123abc")      -> "_123abc"
        normalize_workspace("café-art")    -> "caf__art"
    """
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", name.strip()).lower()
    if safe and safe[0].isdigit():
        safe = f"_{safe}"
    return safe


def validate_workspace_name(name: str, *, max_length: int = 64) -> str:
    """Validate and trim a user-facing workspace name.

    The returned value is still a display label. Call ``normalize_workspace``
    when an internal PostgreSQL-safe workspace identifier is needed.
    """
    label = name.strip()
    if not label:
        raise ValueError("Workspace name cannot be empty")
    if len(label) > max_length:
        raise ValueError(f"Workspace name too long (max {max_length} characters)")
    if _WORKSPACE_FORBIDDEN_RE.search(label):
        raise ValueError("Workspace name contains forbidden characters")
    return label
