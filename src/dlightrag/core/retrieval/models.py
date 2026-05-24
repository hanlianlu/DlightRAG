# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Data models for multi-path retrieval."""

from __future__ import annotations

import warnings
from datetime import datetime
from typing import Any

from pydantic import BaseModel, field_validator


class MetadataFilter(BaseModel):
    """Structured filter for document metadata queries."""

    filename: str | None = None
    filename_stem: str | None = None
    filename_pattern: str | None = None
    file_extension: str | None = None
    doc_title: str | None = None
    doc_author: str | None = None
    date_from: datetime | None = None
    date_to: datetime | None = None
    custom: dict[str, Any] | None = None

    @field_validator(
        "filename",
        "filename_stem",
        "filename_pattern",
        "file_extension",
        "doc_title",
        "doc_author",
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

    def is_empty(self) -> bool:
        """Return True if no filter criteria are set."""
        return all(v is None for v in self.model_dump().values())


def _validate_filter_coverage() -> None:
    """Warn at import time if searchable fields are missing from MetadataFilter."""
    from dlightrag.core.retrieval.metadata_fields import searchable_field_ids

    filter_fields = set(MetadataFilter.model_fields.keys())
    missing = searchable_field_ids() - filter_fields
    if missing:
        warnings.warn(
            f"MetadataFilter missing searchable fields: {missing}",
            stacklevel=2,
        )


_validate_filter_coverage()
