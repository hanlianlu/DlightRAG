# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Document metadata columns — single source of truth for the metadata table."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any


class MetadataValidationError(ValueError):
    """Caller-supplied metadata was rejected, as distinct from an internal ValueError."""


@dataclass(frozen=True)
class MetadataFieldDef:
    """Defines a metadata column in dlightrag_doc_metadata.

    Attributes:
        field_id: Column name in the metadata table.
        pg_type: PostgreSQL column type (e.g. ``VARCHAR(512)``, ``JSONB DEFAULT '{}'``).
        indexed: Whether the column carries an index for the canonical match.
    """

    field_id: str
    pg_type: str
    indexed: bool = False


@dataclass(frozen=True)
class NormalizedUserMetadata:
    custom_metadata: dict[str, Any]
    system: dict[str, Any] = field(default_factory=dict)


def _coerce_creation_date(value: Any) -> datetime:
    """Accept what a caller can serialize; store the instant the filter compares."""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError as exc:
            raise MetadataValidationError(
                f"creation_date must be an ISO 8601 date or timestamp, got {value!r}"
            ) from exc
    else:
        raise MetadataValidationError(
            f"creation_date must be an ISO 8601 date or timestamp, got {type(value).__name__}"
        )
    if parsed.tzinfo is None:
        return parsed
    return parsed.astimezone(UTC).replace(tzinfo=None)


# The one built-in column a caller may set through `metadata`. Everything else
# about a document is derived from the file, and title/author have their own
# ingest parameters.
_CALLER_SETTABLE_COLUMN = "creation_date"


def canonical_metadata_key(key: str) -> str:
    """Fold a metadata key the same way its value is matched, so both sides agree."""
    return key.strip().casefold()


def normalize_user_metadata(metadata: Mapping[str, Any] | None) -> NormalizedUserMetadata:
    """Route caller metadata to its own column, or verbatim into the JSONB column."""
    if not metadata:
        return NormalizedUserMetadata(custom_metadata={})
    custom: dict[str, Any] = {}
    system: dict[str, Any] = {}
    for raw_key, value in metadata.items():
        key = canonical_metadata_key(raw_key)
        if key == _CALLER_SETTABLE_COLUMN:
            system[key] = _coerce_creation_date(value)
            continue
        if key in _RESERVED_METADATA_KEYS:
            raise MetadataValidationError(
                f"{key} is a built-in metadata field and cannot be set through metadata"
            )
        custom[key] = value
    return NormalizedUserMetadata(custom_metadata=custom, system=system)


def extract_system_metadata(
    path: str | Path,
    *,
    display_filename: str | None = None,
    source_uri: str,
    download_locator: str,
) -> dict[str, Any]:
    """Build system metadata that DlightRAG owns."""
    file_name = Path(display_filename or Path(str(path)).name)
    return {
        "filename": file_name.name,
        "filename_stem": file_name.stem,
        "source_uri": source_uri,
        "download_locator": download_locator,
        "file_extension": file_name.suffix.lower().lstrip("."),
    }


METADATA_FIELDS: tuple[MetadataFieldDef, ...] = (
    MetadataFieldDef("filename", "VARCHAR(512)", indexed=True),
    MetadataFieldDef("filename_stem", "VARCHAR(512)", indexed=True),
    MetadataFieldDef("source_uri", "TEXT"),
    MetadataFieldDef("download_locator", "TEXT"),
    MetadataFieldDef("file_extension", "VARCHAR(32)", indexed=True),
    MetadataFieldDef("title", "TEXT", indexed=True),
    MetadataFieldDef("author", "VARCHAR(255)", indexed=True),
    # Naive on purpose: values are normalized to UTC before they are bound,
    # so no session timezone can reinterpret them on the way back out.
    MetadataFieldDef("creation_date", "TIMESTAMP", indexed=True),
    MetadataFieldDef("ingested_at", "TIMESTAMPTZ DEFAULT NOW()"),
    # Matched with LOWER(custom_metadata ->> key), which no index can serve.
    MetadataFieldDef("custom_metadata", "JSONB DEFAULT '{}'"),
)


# ---------------------------------------------------------------------------
# Derived helpers — cached so they are computed only once
# ---------------------------------------------------------------------------


# Filter fields the planner may emit, mapped to the columns whose data backs
# them. Neither side is 1:1: a named file is matched against two columns, and
# one date column backs both ends of a range.
FILTER_FIELD_COLUMNS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "filename": ("filename", "filename_stem"),
        "file_extension": ("file_extension",),
        "title": ("title",),
        "author": ("author",),
        "creation_date_from": ("creation_date",),
        "creation_date_to": ("creation_date",),
    }
)


# Names that already resolve to a column or a filter, plus the primary key that
# the table declares outside the registry. Accepting one as user metadata would
# store the value in JSONB where no filter ever reads it.
_RESERVED_METADATA_KEYS: frozenset[str] = (
    frozenset(FILTER_FIELD_COLUMNS)
    | {f.field_id for f in METADATA_FIELDS}
    | {"workspace", "doc_id"}
) - {_CALLER_SETTABLE_COLUMN}
