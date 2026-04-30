# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""JSON file-based metadata index — fallback for non-PostgreSQL backends.

Storage layout: ``{working_dir}/metadata_index/{workspace}.json``
Thread-safety: designed for single-process async usage (no file locking).
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from dlightrag.core.retrieval.metadata_fields import METADATA_FIELDS, system_field_ids
from dlightrag.core.retrieval.models import MetadataFilter

logger = logging.getLogger(__name__)

# Fields excluded from get_field_schema output — structural, not user-filterable
_SCHEMA_EXCLUDE = frozenset({"custom_metadata", "ingested_at", "workspace", "doc_id"})

# Trigram fields → case-insensitive substring match in JSON fallback
_TRIGRAM_FIELDS = frozenset(f.field_id for f in METADATA_FIELDS if f.trigram)

# Exact match fields (btree string columns) → case-insensitive exact match
_EXACT_FIELDS = frozenset(
    f.field_id
    for f in METADATA_FIELDS
    if f.filterable
    and not f.trigram
    and f.field_id not in {"creation_date", "rag_mode", "custom_metadata"}
)


class JsonMetadataIndex:
    """JSON file-based document metadata index.

    Drop-in replacement for PGMetadataIndex when PostgreSQL is unavailable.
    Implements :class:`~dlightrag.storage.protocols.MetadataIndexProtocol`.
    """

    def __init__(self, working_dir: Path | str, workspace: str = "default") -> None:
        self._working_dir = Path(working_dir)
        self._workspace = workspace
        self._data: dict[str, dict[str, Any]] = {}
        self._file: Path | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Load existing data from disk (or create empty store)."""
        directory = self._working_dir / "metadata_index"
        await asyncio.to_thread(directory.mkdir, parents=True, exist_ok=True)
        self._file = directory / f"{self._workspace}.json"
        if await asyncio.to_thread(self._file.exists):
            text = await asyncio.to_thread(self._file.read_text, encoding="utf-8")
            self._data = json.loads(text) if text.strip() else {}
        else:
            self._data = {}

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    async def upsert(self, doc_id: str, metadata: dict[str, Any]) -> None:
        """Insert or update metadata with COALESCE semantics.

        - System fields: new non-None values override; existing kept otherwise.
        - Non-system fields: merged into ``custom_metadata`` sub-dict.
        """
        existing = self._data.get(doc_id, {})
        _sys_ids = system_field_ids()

        # Partition incoming metadata
        system_vals: dict[str, Any] = {}
        custom_vals: dict[str, Any] = {}
        for key, value in metadata.items():
            if key in _sys_ids:
                system_vals[key] = value
            elif key != "ingested_at":
                custom_vals[key] = value

        # COALESCE for system fields: only overwrite if new value is not None
        for key, value in system_vals.items():
            if value is not None:
                existing[key] = value
            # If value is None and key not in existing, skip (leave absent)

        # Merge custom_metadata (JSONB || semantics)
        existing_custom: dict[str, Any] = existing.get("custom_metadata", {})
        existing_custom.update(custom_vals)
        if existing_custom:
            existing["custom_metadata"] = existing_custom

        self._data[doc_id] = existing
        await asyncio.to_thread(self._persist)

    async def get(self, doc_id: str) -> dict[str, Any] | None:
        """Return metadata dict for a single document, or None."""
        row = self._data.get(doc_id)
        if row is None:
            return None
        # Return a copy to avoid mutation
        return dict(row)

    async def delete(self, doc_id: str) -> None:
        """Remove metadata for a document."""
        if doc_id in self._data:
            del self._data[doc_id]
            await asyncio.to_thread(self._persist)

    async def clear(self) -> None:
        """Remove all metadata for this workspace."""
        self._data.clear()
        await asyncio.to_thread(self._persist)
        logger.info("JsonMetadataIndex cleared for workspace %s", self._workspace)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    async def query(self, filters: MetadataFilter) -> list[str]:
        """Return doc_ids matching the given filters.

        Match strategy mirrors PGMetadataIndex:
        - Trigram fields (filename, filename_stem, doc_title) → case-insensitive substring
        - Exact fields (file_extension, doc_author) → case-insensitive exact
        - filename_pattern → strip ``%`` wildcards, substring match
        - Date range → Python datetime comparison
        - rag_mode → exact match
        - custom → dict containment check
        - Empty filter → return all doc_ids
        """
        if filters.is_empty():
            return list(self._data.keys())

        matching: list[str] = []
        for doc_id, row in self._data.items():
            if self._matches(row, filters):
                matching.append(doc_id)
        return matching

    def _matches(self, row: dict[str, Any], filters: MetadataFilter) -> bool:  # noqa: C901
        """Check whether a single row satisfies all filter criteria."""
        # Trigram fields — case-insensitive substring
        for attr in ("filename", "filename_stem", "doc_title"):
            value = getattr(filters, attr, None)
            if value is None:
                continue
            stored = row.get(attr)
            if stored is None:
                return False
            if value.lower() not in stored.lower():
                return False

        # Exact match fields — case-insensitive exact
        for attr in ("file_extension", "doc_author"):
            value = getattr(filters, attr, None)
            if value is None:
                continue
            stored = row.get(attr)
            if stored is None:
                return False
            if stored.lower() != value.lower():
                return False

        # filename_pattern: strip % wildcards, substring match
        if filters.filename_pattern:
            pattern = filters.filename_pattern.strip("%")
            stored = row.get("filename")
            if stored is None:
                return False
            if pattern.lower() not in stored.lower():
                return False

        # Date range
        if filters.date_from or filters.date_to:
            stored_str = row.get("creation_date")
            if stored_str is None:
                return False
            stored_dt = datetime.fromisoformat(stored_str)
            if filters.date_from and stored_dt < filters.date_from:
                return False
            if filters.date_to and stored_dt > filters.date_to:
                return False

        # rag_mode — exact match
        if filters.rag_mode:
            if row.get("rag_mode") != filters.rag_mode:
                return False

        # custom — dict containment
        if filters.custom:
            stored_custom = row.get("custom_metadata", {})
            for key, value in filters.custom.items():
                if stored_custom.get(key) != value:
                    return False

        return True

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    async def find_by_filename(self, name: str) -> list[str]:
        """Find doc_ids by case-insensitive exact filename match."""
        name_lower = name.lower()
        return [
            doc_id
            for doc_id, row in self._data.items()
            if row.get("filename", "").lower() == name_lower
        ]

    async def get_field_schema(self) -> dict[str, Any]:
        """Return schema information compatible with PGMetadataIndex.get_field_schema.

        - ``columns``: from METADATA_FIELDS registry (excluding internal cols).
        - ``custom_keys``: union of all keys found in documents' custom_metadata.
        """
        columns = [
            {"name": f.field_id, "type": f.pg_type}
            for f in METADATA_FIELDS
            if f.field_id not in _SCHEMA_EXCLUDE
        ]

        # Scan all documents for custom_metadata keys
        all_keys: set[str] = set()
        for row in self._data.values():
            cm = row.get("custom_metadata")
            if isinstance(cm, dict):
                all_keys.update(cm.keys())

        return {"columns": columns, "custom_keys": sorted(all_keys)}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _persist(self) -> None:
        """Write in-memory data to disk."""
        assert self._file is not None, "initialize() must be called first"
        self._file.write_text(
            json.dumps(self._data, indent=2, default=str),
            encoding="utf-8",
        )
