# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Data models for multi-path retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class MetadataFilter:
    """Structured filter for document metadata queries."""

    filename: str | None = None
    filename_pattern: str | None = None
    file_extension: str | None = None
    doc_title: str | None = None
    doc_author: str | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    rag_mode: str | None = None
    custom: dict[str, Any] | None = None

    def is_empty(self) -> bool:
        """Return True if no filter criteria are set."""
        return all(
            v is None
            for v in [
                self.filename,
                self.filename_pattern,
                self.file_extension,
                self.doc_title,
                self.doc_author,
                self.date_from,
                self.date_to,
                self.rag_mode,
                self.custom,
            ]
        )


@dataclass
class RetrievalPlan:
    """Output of QueryAnalyzer — describes which retrieval paths to activate.

    The ``query`` field is always the original user query, unchanged.
    QueryAnalyzer only extracts metadata filters — it never rewrites
    or decomposes the query itself (that's LightRAG's job).
    """

    query: str
    metadata_filters: MetadataFilter | None
    paths: list[str] = field(default_factory=lambda: ["kgvector"])
