# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for storage.metadata_fields — field registry."""

from datetime import UTC, datetime

import pytest

from dlightrag.core.retrieval.metadata_fields import (
    METADATA_FIELDS,
    MetadataFieldRegistry,
    NormalizedUserMetadata,
    extract_system_metadata,
    normalize_user_metadata,
)
from dlightrag.core.retrieval.models import MetadataFilter


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

    def test_has_system_fields(self) -> None:
        from dlightrag.core.retrieval.metadata_fields import METADATA_FIELDS

        assert len(METADATA_FIELDS) >= 12

    def test_has_filename(self) -> None:
        from dlightrag.core.retrieval.metadata_fields import METADATA_FIELDS

        ids = [f.field_id for f in METADATA_FIELDS]
        assert "filename" in ids

    def test_filename_btree_indexed(self) -> None:
        from dlightrag.core.retrieval.metadata_fields import METADATA_FIELDS

        fn = next(f for f in METADATA_FIELDS if f.field_id == "filename")
        assert fn.index_type == "btree"

    def test_no_trigram_metadata_fields(self) -> None:
        from dlightrag.core.retrieval.metadata_fields import METADATA_FIELDS

        assert all(f.index_type != "gin_trgm" for f in METADATA_FIELDS)

    def test_custom_metadata_gin_indexed(self) -> None:
        from dlightrag.core.retrieval.metadata_fields import METADATA_FIELDS

        cm = next(f for f in METADATA_FIELDS if f.field_id == "custom_metadata")
        assert cm.index_type == "gin"

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

    def test_system_field_ids_excludes_custom(self) -> None:
        from dlightrag.core.retrieval.metadata_fields import system_field_ids

        ids = system_field_ids()
        assert isinstance(ids, frozenset)
        assert "custom_metadata" not in ids
        assert "filename" in ids

    def test_field_by_id_found(self) -> None:
        from dlightrag.core.retrieval.metadata_fields import field_by_id

        f = field_by_id("filename")
        assert f is not None
        assert f.field_id == "filename"

    def test_field_by_id_not_found(self) -> None:
        from dlightrag.core.retrieval.metadata_fields import field_by_id

        assert field_by_id("nonexistent") is None

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


def test_declared_metadata_field_is_normalized_for_exact_filtering() -> None:
    registry = MetadataFieldRegistry.from_config(
        {
            "author": {
                "type": "string",
                "normalizer": "casefold_trim",
                "filterable": True,
            }
        }
    )

    normalized = normalize_user_metadata({"author": " Ada Lovelace "}, registry)

    assert normalized.filterable["author"] == "ada lovelace"


def test_string_exact_metadata_defaults_to_casefold_trim() -> None:
    registry = MetadataFieldRegistry.from_config(
        {"department": {"type": "string", "filterable": True}}
    )

    normalized = normalize_user_metadata({"department": " Finance "}, registry)

    spec = registry.filter_spec("department")
    assert spec is not None
    assert spec.normalizer == "casefold_trim"
    assert normalized.filterable["department"] == "finance"


def test_identity_normalizer_can_preserve_exact_string_metadata() -> None:
    registry = MetadataFieldRegistry.from_config(
        {
            "sku": {
                "type": "string",
                "normalizer": "identity",
                "filterable": True,
            }
        }
    )

    normalized = normalize_user_metadata({"sku": " AbC-123 "}, registry)

    assert normalized.filterable["sku"] == " AbC-123 "


def test_custom_metadata_filter_is_normalized_with_registry() -> None:
    registry = MetadataFieldRegistry.from_config(
        {
            "department": {"type": "string", "filterable": True},
            "sku": {
                "type": "string",
                "normalizer": "identity",
                "filterable": True,
            },
        }
    )

    normalized = registry.normalize_filter(
        MetadataFilter(custom={"department": " Finance ", "sku": " AbC-123 ", "raw_note": " Raw "})
    )

    assert normalized.custom == {
        "department": "finance",
        "sku": " AbC-123 ",
        "raw_note": " Raw ",
    }


def test_unknown_metadata_is_stored_but_not_filterable() -> None:
    registry = MetadataFieldRegistry.from_config({})

    normalized = normalize_user_metadata(
        {"project": "Analytical Engine"},
        registry,
        metadata_policy="validate",
        allow_ad_hoc_json=True,
    )

    assert normalized.raw_json["project"] == "Analytical Engine"
    assert "project" not in normalized.filterable


def test_reject_unknown_metadata_policy_blocks_undeclared_key() -> None:
    registry = MetadataFieldRegistry.from_config({})

    with pytest.raises(ValueError, match="undeclared"):
        normalize_user_metadata(
            {"project": "Analytical Engine"},
            registry,
            metadata_policy="reject_unknown",
            allow_ad_hoc_json=True,
        )


def test_store_only_metadata_policy_never_promotes_declared_fields() -> None:
    registry = MetadataFieldRegistry.from_config(
        {
            "author": {
                "type": "string",
                "normalizer": "casefold_trim",
                "filterable": True,
            }
        }
    )

    normalized = normalize_user_metadata(
        {"author": " Ada Lovelace "},
        registry,
        metadata_policy="store_only",
        allow_ad_hoc_json=True,
    )

    assert normalized.raw_json["author"] == " Ada Lovelace "
    assert normalized.filterable == {}


@pytest.mark.parametrize("key", ["sys.filename", "lightrag.content_hash", "user.author"])
def test_reserved_namespaces_are_rejected_for_user_metadata(key: str) -> None:
    registry = MetadataFieldRegistry.from_config({})

    with pytest.raises(ValueError, match="reserved"):
        normalize_user_metadata({key: "x"}, registry)


def test_intent_detection_cannot_filter_unknown_metadata_field() -> None:
    registry = MetadataFieldRegistry.from_config({})

    assert registry.filter_spec("project") is None


def test_json_contains_requires_declared_metadata_json_field() -> None:
    registry = MetadataFieldRegistry.from_config(
        {"metadata_json": {"type": "json", "filterable": True}}
    )

    spec = registry.filter_spec("metadata_json")
    assert spec is not None
    assert spec.type == "json"
    # Only string fields get case folding; JSON values must match as written.
    assert spec.normalizer == "identity"


async def test_metadata_update_revalidates_without_reindexing() -> None:
    from unittest.mock import AsyncMock

    from dlightrag.core.service import RAGService

    service = object.__new__(RAGService)
    service.config = _writer_config()
    service._metadata_index = AsyncMock()
    service._metadata_registry = MetadataFieldRegistry.from_config(
        {
            "author": {
                "type": "string",
                "normalizer": "casefold_trim",
                "filterable": True,
            }
        }
    )
    service._allow_ad_hoc_metadata = True
    service._default_metadata_policy = "validate"
    service._lightrag = AsyncMock()

    await service.aupdate_metadata(
        "doc-1",
        {"author": " Ada Lovelace "},
        mode="merge",
        metadata_policy="validate",
    )

    _, saved = service._metadata_index.upsert.await_args.args
    assert saved["metadata_filterable"]["author"] == "ada lovelace"
    service._lightrag.apipeline_enqueue_documents.assert_not_called()


class TestCallerSettableColumns:
    """creation_date is the one filterable document attribute with no ingest parameter."""

    @staticmethod
    def _normalize(value: object) -> NormalizedUserMetadata:
        return normalize_user_metadata(
            {"creation_date": value}, MetadataFieldRegistry.from_config({})
        )

    def test_iso_date_reaches_the_column_not_jsonb(self) -> None:
        norm = self._normalize("2024-03-05")

        assert norm.system == {"creation_date": datetime(2024, 3, 5, 0, 0)}
        assert norm.raw_json == {}
        assert norm.filterable == {}

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

    def test_declaring_it_as_a_custom_field_does_not_divert_it(self) -> None:
        registry = MetadataFieldRegistry.from_config(
            {"creation_date": {"type": "string", "filterable": True}}
        )

        norm = normalize_user_metadata({"creation_date": "2024-03-05"}, registry)

        # The built-in column wins; it must not also land in custom_metadata.
        assert norm.system == {"creation_date": datetime(2024, 3, 5, 0, 0)}
        assert norm.filterable == {}
