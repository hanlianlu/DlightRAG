# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Data models for multi-path retrieval."""

from __future__ import annotations

import warnings
from datetime import datetime
from typing import Any

from pydantic import BaseModel


class MetadataFilter(BaseModel):
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
        return all(v is None for v in self.model_dump().values())


def _validate_filter_coverage() -> None:
    """Warn at import time if searchable fields are missing from MetadataFilter."""
    from dlightrag.storage.metadata_fields import searchable_field_ids

    filter_fields = set(MetadataFilter.model_fields.keys())
    missing = searchable_field_ids() - filter_fields
    if missing:
        warnings.warn(
            f"MetadataFilter missing searchable fields: {missing}",
            stacklevel=2,
        )


_validate_filter_coverage()
