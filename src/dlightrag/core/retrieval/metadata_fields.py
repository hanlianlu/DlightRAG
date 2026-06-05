# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Metadata field registry — single source of truth for document metadata columns."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal


@dataclass(frozen=True)
class MetadataFieldDef:
    """Defines a metadata column in dlightrag_doc_metadata.

    Attributes:
        field_id: Column name in the metadata table.
        pg_type: PostgreSQL column type (e.g. ``VARCHAR(512)``, ``JSONB DEFAULT '{}'``).
        filterable: Whether this field can be used in query filters.
        searchable: Whether this field is eligible for query-time text matching.
        index_type: PostgreSQL index type (``btree``, ``gin``, or None).
        filter_hint: Human-readable hint describing how to filter on this field.
    """

    field_id: str
    pg_type: str
    filterable: bool = False
    searchable: bool = False
    index_type: str | None = None
    filter_hint: str | None = None


MetadataIngestPolicy = Literal["validate", "reject_unknown", "store_only"]


@dataclass(frozen=True)
class DeclaredMetadataField:
    field_id: str
    field_type: str = "string"
    normalizer: str = "identity"
    filter_ops: tuple[str, ...] = field(default_factory=tuple)

    @property
    def filterable(self) -> bool:
        return bool(self.filter_ops)

    @property
    def type(self) -> str:
        return self.field_type


@dataclass(frozen=True)
class NormalizedUserMetadata:
    filterable: dict[str, Any]
    raw_json: dict[str, Any]


class MetadataFieldRegistry:
    """Runtime registry for user-declared metadata filter fields."""

    def __init__(self, fields: Mapping[str, DeclaredMetadataField] | None = None) -> None:
        self._fields = dict(fields or {})

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> MetadataFieldRegistry:
        fields = {}
        for field_id, raw in (config or {}).items():
            fields[field_id] = DeclaredMetadataField(
                field_id=field_id,
                field_type=str(_field_option(raw, "type", "string") or "string"),
                normalizer=str(_field_option(raw, "normalizer", "identity") or "identity"),
                filter_ops=tuple(_field_option(raw, "filter_ops", ()) or ()),
            )
        return cls(fields)

    def get(self, field_id: str) -> DeclaredMetadataField | None:
        return self._fields.get(field_id)

    def filter_spec(self, field_id: str) -> DeclaredMetadataField | None:
        field_def = self._fields.get(field_id)
        if field_def is None or not field_def.filterable:
            return None
        return field_def


def _field_option(raw: Any, key: str, default: Any) -> Any:
    if isinstance(raw, Mapping):
        return raw.get(key, default)
    return getattr(raw, key, default)


def normalize_user_metadata(
    metadata: Mapping[str, Any] | None,
    registry: MetadataFieldRegistry,
    *,
    metadata_policy: MetadataIngestPolicy = "validate",
    allow_ad_hoc_json: bool = True,
) -> NormalizedUserMetadata:
    """Normalize user metadata into filterable fields and JSON enrichment."""
    if not metadata:
        return NormalizedUserMetadata(filterable={}, raw_json={})
    filterable: dict[str, Any] = {}
    raw_json: dict[str, Any] = {}
    for key, value in metadata.items():
        if key.startswith(("sys.", "lightrag.", "user.")):
            raise ValueError(f"Metadata key uses reserved namespace: {key}")
        field_def = registry.get(key)
        if field_def is None:
            if metadata_policy == "reject_unknown":
                raise ValueError(f"undeclared metadata field: {key}")
            if allow_ad_hoc_json:
                raw_json[key] = value
            continue
        if metadata_policy != "store_only" and field_def.filterable:
            filterable[key] = _normalize_value(value, field_def.normalizer)
        raw_json[key] = value
    return NormalizedUserMetadata(filterable=filterable, raw_json=raw_json)


def extract_system_metadata(path: str | Path, *, ingest_strategy: str) -> dict[str, Any]:
    """Build system metadata that DlightRAG owns."""
    file_path = Path(path)
    return {
        "filename": file_path.name,
        "filename_stem": file_path.stem,
        "file_path": str(file_path),
        "file_extension": file_path.suffix.lower().lstrip("."),
        "ingest_strategy": ingest_strategy,
    }


def _normalize_value(value: Any, normalizer: str) -> Any:
    if normalizer == "casefold_trim" and isinstance(value, str):
        return value.strip().casefold()
    if normalizer == "trim" and isinstance(value, str):
        return value.strip()
    return value


METADATA_FIELDS: tuple[MetadataFieldDef, ...] = (
    MetadataFieldDef(
        "filename",
        "VARCHAR(512)",
        filterable=True,
        searchable=True,
        index_type="btree",
        filter_hint="Exact match on the original file name (e.g. 'report.pdf').",
    ),
    MetadataFieldDef(
        "filename_stem",
        "VARCHAR(512)",
        filterable=True,
        searchable=True,
        index_type="btree",
        filter_hint="Exact match on the file name without extension (e.g. 'report').",
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
        index_type="btree",
        filter_hint="Exact match on the document title.",
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
    MetadataFieldDef("ingest_strategy", "VARCHAR(64)"),
    MetadataFieldDef("parse_engine", "VARCHAR(64)"),
    MetadataFieldDef("process_options", "JSONB DEFAULT '{}'"),
    MetadataFieldDef("ingested_at", "TIMESTAMPTZ DEFAULT NOW()"),
    MetadataFieldDef(
        "custom_metadata",
        "JSONB DEFAULT '{}'",
        filterable=True,
        index_type="gin",
        filter_hint="JSONB containment query on user-defined key-value pairs.",
    ),
    MetadataFieldDef(
        "metadata_json",
        "JSONB DEFAULT '{}'",
        index_type="gin",
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
