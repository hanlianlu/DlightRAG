# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Storage-neutral retrieval result records."""

from dataclasses import dataclass, field
from typing import Any

from dlightrag.rag.retrieval.models import ContextRow

RetrievalContexts = dict[str, list[ContextRow]]


@dataclass
class RetrievalResult:
    """Corpus retrieval output before Answer projection."""

    contexts: RetrievalContexts = field(
        default_factory=lambda: {"chunks": [], "entities": [], "relationships": []}
    )
    image_descriptions: list[str] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)


__all__ = ["RetrievalContexts", "RetrievalResult"]
