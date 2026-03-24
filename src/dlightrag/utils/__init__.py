# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared utilities."""

import re


def normalize_workspace(name: str) -> str:
    """Normalize workspace name to a safe, lowercase PG identifier.

    Replaces non-alphanumeric/underscore characters with ``_``, lowercases,
    and prepends ``_`` if the name starts with a digit (invalid PG schema).

    Examples::

        normalize_workspace("Ian Davis")   -> "ian_davis"
        normalize_workspace("project-alpha") -> "project_alpha"
        normalize_workspace("123abc")      -> "_123abc"
        normalize_workspace("café-art")    -> "caf__art"
    """
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", name.strip()).lower()
    if safe and safe[0].isdigit():
        safe = f"_{safe}"
    return safe
