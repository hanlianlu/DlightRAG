# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for storage.metadata_fields — field registry."""

from __future__ import annotations

import pytest


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

    def test_has_12_fields(self) -> None:
        from dlightrag.core.retrieval.metadata_fields import METADATA_FIELDS

        assert len(METADATA_FIELDS) == 12

    def test_has_filename(self) -> None:
        from dlightrag.core.retrieval.metadata_fields import METADATA_FIELDS

        ids = [f.field_id for f in METADATA_FIELDS]
        assert "filename" in ids

    def test_filename_filterable_gin_trgm(self) -> None:
        from dlightrag.core.retrieval.metadata_fields import METADATA_FIELDS

        fn = next(f for f in METADATA_FIELDS if f.field_id == "filename")
        assert fn.filterable is True
        assert fn.searchable is True
        assert fn.trigram is True
        assert fn.index_type == "gin_trgm"

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
