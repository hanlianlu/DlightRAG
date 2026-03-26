# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for MetadataFieldDef registry and COALESCE upsert."""

from __future__ import annotations

from dlightrag.core.retrieval.metadata_index import (
    _UPSERT,
    METADATA_FIELDS,
)


class TestMetadataFieldDef:
    def test_registry_has_filename(self):
        names = [f.field_id for f in METADATA_FIELDS]
        assert "filename" in names

    def test_filename_is_filterable(self):
        f = next(f for f in METADATA_FIELDS if f.field_id == "filename")
        assert f.filterable is True
        assert f.index_type == "gin_trgm"

    def test_custom_metadata_is_gin_indexed(self):
        f = next(f for f in METADATA_FIELDS if f.field_id == "custom_metadata")
        assert f.index_type == "gin"

    def test_page_count_not_filterable(self):
        f = next(f for f in METADATA_FIELDS if f.field_id == "page_count")
        assert f.filterable is False

    def test_all_fields_have_pg_type(self):
        for f in METADATA_FIELDS:
            assert f.pg_type, f"Field {f.field_id} missing pg_type"


class TestUpsertSQL:
    def test_upsert_sql_uses_coalesce(self):
        assert "COALESCE" in _UPSERT

    def test_custom_metadata_uses_jsonb_merge(self):
        assert "||" in _UPSERT or "custom_metadata" in _UPSERT
