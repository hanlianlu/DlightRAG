# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""How a cut text says it was cut, once for every surface that repeats page or mail text.

A model that reads a partial fact as the whole one is how a working site comes to look
empty and a login comes to look unanswerable, so every cut carries the same marker and the
count of what it left out (ADR 0032). This module is a leaf: the adapter, the tool and the
mailbox all state the same rule without importing each other.
"""

from __future__ import annotations

#: What a text says where a cut left the rest of it out, and how much of it the model got.
ELISION = " … [{shown} of {total} characters]"


def elide(text: str, limit: int) -> str:
    """``text`` kept to ``limit`` characters, with the cut marked and counted."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}{ELISION.format(shown=limit, total=len(text))}"


__all__ = ["elide"]
