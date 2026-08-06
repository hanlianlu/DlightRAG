# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Metadata field registry — single source of truth for document metadata columns."""

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from dlightrag.contracts import MetadataPolicy


@dataclass(frozen=True)
class MetadataFieldDef:
    """Defines a metadata column in dlightrag_doc_metadata.

    Attributes:
        field_id: Column name in the metadata table.
        pg_type: PostgreSQL column type (e.g. ``VARCHAR(512)``, ``JSONB DEFAULT '{}'``).
        index_type: PostgreSQL index type (``btree``, ``gin``, or None).
    """

    field_id: str
    pg_type: str
    index_type: str | None = None


@dataclass(frozen=True)
class DeclaredMetadataField:
    field_id: str
    field_type: str = "string"
    normalizer: str = "identity"
    filterable: bool = False

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
            field_type = str(_field_option(raw, "type", "string") or "string")
            filterable = bool(_field_option(raw, "filterable", False))
            normalizer = _field_option(raw, "normalizer", None)
            fields[field_id] = DeclaredMetadataField(
                field_id=field_id,
                field_type=field_type,
                normalizer=str(normalizer or _default_normalizer(field_type, filterable)),
                filterable=filterable,
            )
        return cls(fields)

    def get(self, field_id: str) -> DeclaredMetadataField | None:
        return self._fields.get(field_id)

    def filter_spec(self, field_id: str) -> DeclaredMetadataField | None:
        field_def = self._fields.get(field_id)
        if field_def is None or not field_def.filterable:
            return None
        return field_def

    def normalize_filter(self, filters: Any) -> Any:
        custom = getattr(filters, "custom", None)
        if not custom:
            return filters

        normalized_custom: dict[str, Any] = {}
        changed = False
        for key, value in custom.items():
            field_def = self.filter_spec(key)
            normalized_value = (
                _normalize_value(value, field_def.normalizer) if field_def is not None else value
            )
            normalized_custom[key] = normalized_value
            changed = changed or normalized_value != value

        if not changed:
            return filters
        return filters.model_copy(update={"custom": normalized_custom})


def _field_option(raw: Any, key: str, default: Any) -> Any:
    if isinstance(raw, Mapping):
        return raw.get(key, default)
    return getattr(raw, key, default)


def _default_normalizer(field_type: str, filterable: bool) -> str:
    # Custom filtering is JSONB containment, so a filterable string only matches
    # when ingest and query normalize identically.
    if field_type == "string" and filterable:
        return "casefold_trim"
    return "identity"


def normalize_user_metadata(
    metadata: Mapping[str, Any] | None,
    registry: MetadataFieldRegistry,
    *,
    metadata_policy: MetadataPolicy = "validate",
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


def extract_system_metadata(
    path: str | Path,
    *,
    display_filename: str | None = None,
    source_uri: str,
    download_locator: str,
) -> dict[str, Any]:
    """Build system metadata that DlightRAG owns."""
    raw_path = str(path)
    if display_filename:
        filename = display_filename
    elif raw_path.startswith(("azure://", "s3://", "https://")):
        filename = PurePosixPath(raw_path.split("://", 1)[1]).name
    else:
        filename = Path(path).name
    file_name = Path(filename)
    return {
        "filename": file_name.name,
        "filename_stem": file_name.stem,
        "file_path": raw_path,
        "source_uri": source_uri,
        "download_locator": download_locator,
        "file_extension": file_name.suffix.lower().lstrip("."),
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
        index_type="btree",
    ),
    MetadataFieldDef(
        "filename_stem",
        "VARCHAR(512)",
        index_type="btree",
    ),
    MetadataFieldDef("file_path", "TEXT"),
    MetadataFieldDef("source_uri", "TEXT"),
    MetadataFieldDef("download_locator", "TEXT"),
    MetadataFieldDef(
        "file_extension",
        "VARCHAR(32)",
        index_type="btree",
    ),
    MetadataFieldDef(
        "doc_title",
        "TEXT",
        index_type="btree",
    ),
    MetadataFieldDef(
        "doc_author",
        "VARCHAR(255)",
        index_type="btree",
    ),
    MetadataFieldDef(
        # Naive on purpose: values are normalized to UTC before they are bound,
        # so no session timezone can reinterpret them on the way back out.
        "creation_date",
        "TIMESTAMP",
        index_type="btree",
    ),
    MetadataFieldDef("parse_engine", "VARCHAR(64)"),
    MetadataFieldDef("process_options", "JSONB DEFAULT '{}'"),
    MetadataFieldDef("ingested_at", "TIMESTAMPTZ DEFAULT NOW()"),
    MetadataFieldDef(
        "custom_metadata",
        "JSONB DEFAULT '{}'",
        index_type="gin",
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


def field_by_id(field_id: str) -> MetadataFieldDef | None:
    """Look up a field definition by its ``field_id``, or return None."""
    return _FIELD_BY_ID.get(field_id)


# Filter fields the planner may emit, mapped to the columns whose data backs
# them. Neither side is 1:1: a named file is matched against two columns, and
# one date column backs both ends of a range.
FILTER_FIELD_COLUMNS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "filename": ("filename", "filename_stem"),
        "file_extension": ("file_extension",),
        "doc_title": ("doc_title",),
        "doc_author": ("doc_author",),
        "date_from": ("creation_date",),
        "date_to": ("creation_date",),
    }
)


# Internal lookup table (private)
_FIELD_BY_ID: dict[str, MetadataFieldDef] = {f.field_id: f for f in METADATA_FIELDS}
