# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Metadata field registry — single source of truth for document metadata columns."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class MetadataFieldDef:
    """Defines a metadata column in dlightrag_doc_metadata.

    Attributes:
        field_id: Column name in the metadata table.
        pg_type: PostgreSQL column type (e.g. ``VARCHAR(512)``, ``JSONB DEFAULT '{}'``).
        filterable: Whether this field can be used in query filters.
        searchable: Whether this field is eligible for query-time text matching.
        trigram: Use pg_trgm similarity() for fuzzy matching (requires gin_trgm index).
        index_type: PostgreSQL index type (``gin_trgm``, ``btree``, ``gin``, or None).
        filter_hint: Human-readable hint describing how to filter on this field.
    """

    field_id: str
    pg_type: str
    filterable: bool = False
    searchable: bool = False
    trigram: bool = False
    index_type: str | None = None
    filter_hint: str | None = None


METADATA_FIELDS: tuple[MetadataFieldDef, ...] = (
    MetadataFieldDef(
        "filename",
        "VARCHAR(512)",
        filterable=True,
        searchable=True,
        trigram=True,
        index_type="gin_trgm",
        filter_hint="Fuzzy match on the original file name (e.g. 'report.pdf').",
    ),
    MetadataFieldDef(
        "filename_stem",
        "VARCHAR(512)",
        filterable=True,
        searchable=True,
        trigram=True,
        index_type="gin_trgm",
        filter_hint="Fuzzy match on the file name without extension (e.g. 'report').",
    ),
    MetadataFieldDef("file_path", "TEXT"),
    MetadataFieldDef(
        "file_extension",
        "VARCHAR(32)",
        filterable=True,
        searchable=True,
        index_type="btree",
        filter_hint="Exact match on file extension (e.g. 'pdf', 'docx').",
    ),
    MetadataFieldDef(
        "doc_title",
        "TEXT",
        filterable=True,
        searchable=True,
        trigram=True,
        index_type="gin_trgm",
        filter_hint="Fuzzy match on the document title.",
    ),
    MetadataFieldDef(
        "doc_author",
        "VARCHAR(255)",
        filterable=True,
        searchable=True,
        index_type="btree",
        filter_hint="Exact match on the document author.",
    ),
    MetadataFieldDef(
        "creation_date",
        "TIMESTAMPTZ",
        filterable=True,
        index_type="btree",
        filter_hint="Date range filter (date_from / date_to).",
    ),
    MetadataFieldDef("original_format", "VARCHAR(32)"),
    MetadataFieldDef("page_count", "INTEGER"),
    MetadataFieldDef(
        "rag_mode",
        "VARCHAR(32)",
        filterable=True,
        filter_hint="Exact match on RAG processing mode (e.g. 'caption', 'native').",
    ),
    MetadataFieldDef("ingested_at", "TIMESTAMPTZ DEFAULT NOW()"),
    MetadataFieldDef(
        "custom_metadata",
        "JSONB DEFAULT '{}'",
        filterable=True,
        index_type="gin",
        filter_hint="JSONB containment query on user-defined key-value pairs.",
    ),
)


# ---------------------------------------------------------------------------
# Derived helpers — cached so they are computed only once
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def system_field_ids() -> frozenset[str]:
    """System field names (everything except ``custom_metadata``)."""
    return frozenset(f.field_id for f in METADATA_FIELDS if f.field_id != "custom_metadata")


@lru_cache(maxsize=1)
def searchable_field_ids() -> frozenset[str]:
    """Field IDs eligible for query-time text matching."""
    return frozenset(f.field_id for f in METADATA_FIELDS if f.searchable)


@lru_cache(maxsize=1)
def filterable_field_ids() -> frozenset[str]:
    """Field IDs that support query filters."""
    return frozenset(f.field_id for f in METADATA_FIELDS if f.filterable)


def field_by_id(field_id: str) -> MetadataFieldDef | None:
    """Look up a field definition by its ``field_id``, or return None."""
    return _FIELD_BY_ID.get(field_id)


@lru_cache(maxsize=1)
def build_filter_hints() -> dict[str, str]:
    """Map filterable field_id -> human-readable hint string.

    Used by QueryAnalyzer to build a context-aware LLM prompt.
    Only includes fields that are both filterable and have a hint.
    """
    return {
        f.field_id: f.filter_hint
        for f in METADATA_FIELDS
        if f.filterable and f.filter_hint is not None
    }


# Internal lookup table (private)
_FIELD_BY_ID: dict[str, MetadataFieldDef] = {f.field_id: f for f in METADATA_FIELDS}
