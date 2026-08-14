# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for storage.metadata_fields — field registry."""

from datetime import UTC, datetime

import pytest
from dlightrag_rag.retrieval import MetadataFilter

from dlightrag.core.retrieval.metadata_fields import (
    FILTER_FIELD_COLUMNS,
    METADATA_FIELDS,
    NormalizedUserMetadata,
    extract_system_metadata,
    normalize_user_metadata,
)


def _writer_config():
    """Minimal writer config so the RAGService write guard passes."""
    from typing import Any, cast

    from dlightrag.config import DlightragConfig, EmbeddingConfig

    return cast(Any, DlightragConfig)(
        _env_file=None,
        embedding=EmbeddingConfig(
            provider="voyage", model="m", api_key="k", dim=8, startup_probe=False
        ),
    )


class TestMetadataFieldDef:
    """MetadataFieldDef frozen dataclass basics."""

    def test_frozen(self) -> None:
        from dlightrag.core.retrieval.metadata_fields import MetadataFieldDef

        f = MetadataFieldDef("x", "TEXT")
        with pytest.raises(AttributeError):
            f.field_id = "y"  # type: ignore[misc]


class TestMetadataFields:
    """METADATA_FIELDS tuple — the canonical field registry."""

    def test_has_filename(self) -> None:
        from dlightrag.core.retrieval.metadata_fields import METADATA_FIELDS

        ids = [f.field_id for f in METADATA_FIELDS]
        assert "filename" in ids

    def test_filename_is_indexed(self) -> None:
        from dlightrag.core.retrieval.metadata_fields import METADATA_FIELDS

        fn = next(f for f in METADATA_FIELDS if f.field_id == "filename")
        assert fn.indexed

    def test_all_fields_have_pg_type(self) -> None:
        from dlightrag.core.retrieval.metadata_fields import METADATA_FIELDS

        for f in METADATA_FIELDS:
            assert f.pg_type, f"{f.field_id} missing pg_type"

    def test_field_ids_unique(self) -> None:
        from dlightrag.core.retrieval.metadata_fields import METADATA_FIELDS

        ids = [f.field_id for f in METADATA_FIELDS]
        assert len(ids) == len(set(ids))


def test_metadata_registry_has_source_identity_and_download_locator() -> None:
    ids = {field.field_id for field in METADATA_FIELDS}

    assert {"source_uri", "download_locator"} <= ids


def test_extract_system_metadata_stores_distinct_source_and_download_fields() -> None:
    metadata = extract_system_metadata(
        "https://cdn.example.com/assets/1.pdf",
        display_filename="report.pdf",
        source_uri="bynder://asset/1",
        download_locator="https://cdn.example.com/assets/1.pdf",
    )

    assert metadata["source_uri"] == "bynder://asset/1"
    assert metadata["download_locator"] == "https://cdn.example.com/assets/1.pdf"


class TestDerivedFunctions:
    """Derived helper functions built from METADATA_FIELDS."""

    def test_filter_fields_map_to_real_columns(self) -> None:
        from dlightrag.core.retrieval.metadata_fields import FILTER_FIELD_COLUMNS

        columns = {f.field_id for f in METADATA_FIELDS}
        backing = {column for cols in FILTER_FIELD_COLUMNS.values() for column in cols}
        assert backing <= columns
        # Every filter the planner may emit resolves to a column.
        assert set(MetadataFilter.model_fields) - {"custom"} == set(FILTER_FIELD_COLUMNS)
        # A named file is matched against the stored name and its stem.
        assert FILTER_FIELD_COLUMNS["filename"] == ("filename", "filename_stem")
        # One column backs both ends of the range the planner emits.
        assert (
            FILTER_FIELD_COLUMNS["creation_date_from"] == FILTER_FIELD_COLUMNS["creation_date_to"]
        )


def test_user_metadata_is_stored_verbatim() -> None:
    """Case folding is applied by the SQL comparison, so storage stays lossless."""
    norm = normalize_user_metadata({"department": " Finance ", "sku": "AbC-123"})

    assert norm.custom_metadata == {"department": " Finance ", "sku": "AbC-123"}


def test_any_key_is_accepted_without_declaring_it() -> None:
    norm = normalize_user_metadata({"project": "Analytical Engine"})

    assert norm.custom_metadata == {"project": "Analytical Engine"}


def test_non_string_values_survive_untouched() -> None:
    norm = normalize_user_metadata({"pages": 42, "reviewed": True})

    assert norm.custom_metadata == {"pages": 42, "reviewed": True}


async def test_metadata_update_stores_without_reindexing() -> None:
    from unittest.mock import AsyncMock

    from dlightrag.core.service import RAGService

    service = object.__new__(RAGService)
    service.config = _writer_config()
    service._metadata_index = AsyncMock()
    service._lightrag = AsyncMock()

    await service.aupdate_metadata("doc-1", {"reviewer": " Ada Lovelace "})

    _, saved = service._metadata_index.merge_custom_metadata.await_args.args
    assert saved["custom_metadata"]["reviewer"] == " Ada Lovelace "
    service._lightrag.apipeline_enqueue_documents.assert_not_called()


async def test_metadata_update_reports_an_unknown_document() -> None:
    from unittest.mock import AsyncMock

    from dlightrag.core.service import RAGService

    service = object.__new__(RAGService)
    service.config = _writer_config()
    service._metadata_index = AsyncMock()
    service._metadata_index.merge_custom_metadata.return_value = False

    # Updating a document that was never ingested must not conjure one.
    with pytest.raises(KeyError):
        await service.aupdate_metadata("ghost", {"reviewer": "Ada"})


class TestCallerSettableColumns:
    """creation_date is the one filterable document attribute with no ingest parameter."""

    @staticmethod
    def _normalize(value: object) -> NormalizedUserMetadata:
        return normalize_user_metadata({"creation_date": value})

    def test_iso_date_reaches_the_column_not_jsonb(self) -> None:
        norm = self._normalize("2024-03-05")

        assert norm.system == {"creation_date": datetime(2024, 3, 5, 0, 0)}
        assert norm.custom_metadata == {}

    def test_offset_is_converted_to_the_same_instant_the_filter_uses(self) -> None:
        norm = self._normalize("2024-01-01T08:00:00+08:00")

        assert norm.system == {"creation_date": datetime(2024, 1, 1, 0, 0)}

    def test_bare_timestamp_is_read_as_utc(self) -> None:
        norm = self._normalize("2024-01-01T12:30:00")

        assert norm.system == {"creation_date": datetime(2024, 1, 1, 12, 30)}

    def test_datetime_object_is_accepted(self) -> None:
        norm = self._normalize(datetime(2024, 1, 1, 9, 0, tzinfo=UTC))

        assert norm.system == {"creation_date": datetime(2024, 1, 1, 9, 0)}

    @pytest.mark.parametrize("value", ["2024/01/01", "not-a-date", 1704067200])
    def test_unparseable_value_is_rejected_loudly(self, value: object) -> None:
        with pytest.raises(ValueError, match="creation_date must be an ISO 8601"):
            self._normalize(value)


class TestReservedMetadataKeys:
    """A filter name in `metadata` would be stored where no filter ever reads it."""

    @pytest.mark.parametrize("key", sorted(FILTER_FIELD_COLUMNS))
    def test_filter_names_are_rejected_rather_than_stored_in_jsonb(self, key: str) -> None:
        with pytest.raises(ValueError, match="built-in metadata field"):
            normalize_user_metadata({key: "x"})
