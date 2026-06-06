# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for storage.metadata_fields — field registry."""

from __future__ import annotations

import pytest

from dlightrag.core.retrieval.metadata_fields import (
    MetadataFieldRegistry,
    normalize_user_metadata,
)
from dlightrag.core.retrieval.models import MetadataFilter


class TestMetadataFieldDef:
    """MetadataFieldDef frozen dataclass basics."""

    def test_frozen(self) -> None:
        from dlightrag.core.retrieval.metadata_fields import MetadataFieldDef

        f = MetadataFieldDef("x", "TEXT")
        with pytest.raises(AttributeError):
            f.field_id = "y"  # type: ignore[misc]

    def test_filter_hint_defaults_none(self) -> None:
        from dlightrag.core.retrieval.metadata_fields import MetadataFieldDef

        f = MetadataFieldDef("x", "TEXT")
        assert f.filter_hint is None

    def test_filter_hint_set(self) -> None:
        from dlightrag.core.retrieval.metadata_fields import MetadataFieldDef

        f = MetadataFieldDef("x", "TEXT", filter_hint="exact match on x")
        assert f.filter_hint == "exact match on x"


class TestMetadataFields:
    """METADATA_FIELDS tuple — the canonical field registry."""

    def test_is_tuple(self) -> None:
        from dlightrag.core.retrieval.metadata_fields import METADATA_FIELDS

        assert isinstance(METADATA_FIELDS, tuple)

    def test_has_system_fields(self) -> None:
        from dlightrag.core.retrieval.metadata_fields import METADATA_FIELDS

        assert len(METADATA_FIELDS) >= 12

    def test_has_filename(self) -> None:
        from dlightrag.core.retrieval.metadata_fields import METADATA_FIELDS

        ids = [f.field_id for f in METADATA_FIELDS]
        assert "filename" in ids

    def test_filename_filterable_btree(self) -> None:
        from dlightrag.core.retrieval.metadata_fields import METADATA_FIELDS

        fn = next(f for f in METADATA_FIELDS if f.field_id == "filename")
        assert fn.filterable is True
        assert fn.searchable is True
        assert fn.index_type == "btree"

    def test_no_trigram_metadata_fields(self) -> None:
        from dlightrag.core.retrieval.metadata_fields import METADATA_FIELDS

        assert all(f.index_type != "gin_trgm" for f in METADATA_FIELDS)

    def test_custom_metadata_gin_indexed(self) -> None:
        from dlightrag.core.retrieval.metadata_fields import METADATA_FIELDS

        cm = next(f for f in METADATA_FIELDS if f.field_id == "custom_metadata")
        assert cm.filterable is True
        assert cm.index_type == "gin"

    def test_page_count_not_filterable(self) -> None:
        from dlightrag.core.retrieval.metadata_fields import METADATA_FIELDS

        pc = next(f for f in METADATA_FIELDS if f.field_id == "page_count")
        assert pc.filterable is False

    def test_all_fields_have_pg_type(self) -> None:
        from dlightrag.core.retrieval.metadata_fields import METADATA_FIELDS

        for f in METADATA_FIELDS:
            assert f.pg_type, f"{f.field_id} missing pg_type"

    def test_field_ids_unique(self) -> None:
        from dlightrag.core.retrieval.metadata_fields import METADATA_FIELDS

        ids = [f.field_id for f in METADATA_FIELDS]
        assert len(ids) == len(set(ids))


class TestDerivedFunctions:
    """Derived helper functions built from METADATA_FIELDS."""

    def test_system_field_ids_excludes_custom(self) -> None:
        from dlightrag.core.retrieval.metadata_fields import system_field_ids

        ids = system_field_ids()
        assert isinstance(ids, frozenset)
        assert "custom_metadata" not in ids
        assert "filename" in ids

    def test_searchable_field_ids(self) -> None:
        from dlightrag.core.retrieval.metadata_fields import searchable_field_ids

        ids = searchable_field_ids()
        assert isinstance(ids, frozenset)
        assert "filename" in ids
        assert "filename_stem" in ids
        assert "doc_title" in ids
        # page_count is not searchable
        assert "page_count" not in ids

    def test_filterable_field_ids(self) -> None:
        from dlightrag.core.retrieval.metadata_fields import filterable_field_ids

        ids = filterable_field_ids()
        assert isinstance(ids, frozenset)
        assert "filename" in ids
        assert "custom_metadata" in ids
        # page_count is not filterable
        assert "page_count" not in ids

    def test_field_by_id_found(self) -> None:
        from dlightrag.core.retrieval.metadata_fields import field_by_id

        f = field_by_id("filename")
        assert f is not None
        assert f.field_id == "filename"

    def test_field_by_id_not_found(self) -> None:
        from dlightrag.core.retrieval.metadata_fields import field_by_id

        assert field_by_id("nonexistent") is None

    def test_build_filter_hints(self) -> None:
        from dlightrag.core.retrieval.metadata_fields import build_filter_hints

        hints = build_filter_hints()
        assert isinstance(hints, dict)
        # Only filterable fields should appear
        assert "filename" in hints
        assert "page_count" not in hints
        # custom_metadata should be present
        assert "custom_metadata" in hints


def test_declared_metadata_field_is_normalized_for_exact_filtering() -> None:
    registry = MetadataFieldRegistry.from_config(
        {
            "author": {
                "type": "string",
                "normalizer": "casefold_trim",
                "filter_ops": ["exact"],
            }
        }
    )

    normalized = normalize_user_metadata({"author": " Ada Lovelace "}, registry)

    assert normalized.filterable["author"] == "ada lovelace"


def test_string_exact_metadata_defaults_to_casefold_trim() -> None:
    registry = MetadataFieldRegistry.from_config(
        {"department": {"type": "string", "filter_ops": ["exact"]}}
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
                "filter_ops": ["exact"],
            }
        }
    )

    normalized = normalize_user_metadata({"sku": " AbC-123 "}, registry)

    assert normalized.filterable["sku"] == " AbC-123 "


def test_custom_metadata_filter_is_normalized_with_registry() -> None:
    registry = MetadataFieldRegistry.from_config(
        {
            "department": {"type": "string", "filter_ops": ["exact"]},
            "sku": {
                "type": "string",
                "normalizer": "identity",
                "filter_ops": ["exact"],
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
                "filter_ops": ["exact"],
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
        {"metadata_json": {"type": "json", "filter_ops": ["contains"]}}
    )

    spec = registry.filter_spec("metadata_json")
    assert spec is not None
    assert spec.type == "json"
    assert "contains" in spec.filter_ops


async def test_metadata_update_revalidates_without_reindexing() -> None:
    from unittest.mock import AsyncMock

    from dlightrag.core.service import RAGService

    service = object.__new__(RAGService)
    service._metadata_index = AsyncMock()
    service._metadata_registry = MetadataFieldRegistry.from_config(
        {
            "author": {
                "type": "string",
                "normalizer": "casefold_trim",
                "filter_ops": ["exact"],
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
