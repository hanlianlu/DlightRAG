# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Document metadata columns — single source of truth for the metadata table."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path, PurePosixPath
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
        index_type: PostgreSQL index type (``btree``, ``gin``, or None).
    """

    field_id: str
    pg_type: str
    index_type: str | None = None


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


# Built-in columns a caller may set through `metadata`. Everything else about a
# document is derived from the file, and title/author have their own ingest
# parameters, so this stays the one attribute with no other way in.
_CALLER_SETTABLE_COLUMNS: Mapping[str, Callable[[Any], Any]] = MappingProxyType(
    {"creation_date": _coerce_creation_date}
)


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
        if raw_key.startswith(("sys.", "lightrag.", "user.")):
            raise MetadataValidationError(f"Metadata key uses reserved namespace: {raw_key}")
        key = canonical_metadata_key(raw_key)
        coerce = _CALLER_SETTABLE_COLUMNS.get(key)
        if coerce is not None:
            # A typed column of its own, so it is neither JSONB nor re-declarable.
            system[key] = coerce(value)
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
        "source_uri": source_uri,
        "download_locator": download_locator,
        "file_extension": file_name.suffix.lower().lstrip("."),
    }


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
    MetadataFieldDef("source_uri", "TEXT"),
    MetadataFieldDef("download_locator", "TEXT"),
    MetadataFieldDef(
        "file_extension",
        "VARCHAR(32)",
        index_type="btree",
    ),
    MetadataFieldDef(
        "title",
        "TEXT",
        index_type="btree",
    ),
    MetadataFieldDef(
        "author",
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
    MetadataFieldDef("ingested_at", "TIMESTAMPTZ DEFAULT NOW()"),
    # Matched with LOWER(custom_metadata ->> key), which no GIN index can serve.
    MetadataFieldDef("custom_metadata", "JSONB DEFAULT '{}'"),
)


# ---------------------------------------------------------------------------
# Derived helpers — cached so they are computed only once
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def system_field_ids() -> frozenset[str]:
    """System field names (everything except ``custom_metadata``)."""
    return frozenset(f.field_id for f in METADATA_FIELDS if f.field_id != "custom_metadata")


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
) - frozenset(_CALLER_SETTABLE_COLUMNS)
