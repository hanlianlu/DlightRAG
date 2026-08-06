# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Data models for multi-path retrieval."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator


class MetadataFilter(BaseModel):
    """Structured filter for document metadata queries."""

    # An unknown name would otherwise be dropped, turning a typo into a filter
    # that matches every document rather than an error.
    model_config = ConfigDict(extra="forbid")

    filename: str | None = None
    file_extension: str | None = None
    title: str | None = None
    author: str | None = None
    creation_date_from: datetime | None = None
    creation_date_to: datetime | None = None
    custom: dict[str, Any] | None = None

    @field_validator(
        "filename",
        "file_extension",
        "title",
        "author",
        mode="before",
    )
    @classmethod
    def _strip_text_filter(cls, value: Any) -> Any:
        """Normalize user/LLM filter strings without adding fuzzy semantics."""
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @field_validator("file_extension")
    @classmethod
    def _normalize_file_extension(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.lstrip(".").lower()

    @field_validator("creation_date_from", "creation_date_to")
    @classmethod
    def _as_utc(cls, value: datetime | None) -> datetime | None:
        """Store one instant regardless of how the caller wrote it.

        Offsets are accepted for compatibility and converted; a bare timestamp
        is read as UTC rather than as the server's local time. The column is
        naive so PostgreSQL cannot reinterpret the result against a session
        timezone.
        """
        if value is None:
            return None
        if value.tzinfo is None:
            return value
        return value.astimezone(UTC).replace(tzinfo=None)

    def is_empty(self) -> bool:
        """Return True if no filter criteria are set."""
        # An empty `custom` dict carries no criterion, and treating it as one
        # would drop every condition and return the whole workspace.
        return not any(self.model_dump().values())


@dataclass(frozen=True, slots=True)
class MetadataScope:
    """Documents a metadata filter selected, plus their chunk fan-out.

    Retrieval filters by ``full_doc_id`` rather than by chunk id: the filter is
    a document-level predicate, and one document can own thousands of chunks, so
    expanding it client-side would ship that fan-out to PostgreSQL and back on
    every vector and BM25 query. ``chunk_count`` is only what the exact-scan
    branch needs to bound its brute-force cost.
    """

    doc_ids: frozenset[str]
    chunk_count: int

    def __bool__(self) -> bool:
        return bool(self.doc_ids)

    def as_list(self) -> list[str]:
        return list(self.doc_ids)
