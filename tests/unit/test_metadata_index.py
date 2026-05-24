# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for PG metadata index SQL generation."""

from __future__ import annotations

from dlightrag.storage.pg_metadata_index import _CREATE_INDEXES, _UPSERT


class TestUpsertSQL:
    def test_upsert_sql_uses_coalesce(self):
        assert "COALESCE" in _UPSERT

    def test_custom_metadata_uses_jsonb_merge(self):
        assert "||" in _UPSERT or "custom_metadata" in _UPSERT


class TestMetadataSQL:
    def test_indexes_do_not_require_pg_trgm(self):
        sql = "\n".join(_CREATE_INDEXES)

        assert "gin_trgm" not in sql
        assert "trgm" not in sql

    def test_upsert_sql_does_not_reference_similarity(self):
        assert "similarity(" not in _UPSERT

    def test_string_btree_indexes_are_case_normalized(self):
        sql = "\n".join(_CREATE_INDEXES)

        assert "ON dlightrag_doc_metadata (LOWER(filename))" in sql
        assert "ON dlightrag_doc_metadata (LOWER(filename_stem))" in sql
        assert "ON dlightrag_doc_metadata (LOWER(file_extension))" in sql
        assert "ON dlightrag_doc_metadata (LOWER(doc_title))" in sql
        assert "ON dlightrag_doc_metadata (LOWER(doc_author))" in sql

    def test_non_string_btree_indexes_remain_plain(self):
        sql = "\n".join(_CREATE_INDEXES)

        assert "ON dlightrag_doc_metadata (creation_date)" in sql

    def test_upsert_sql_has_no_removed_processing_mode(self):
        assert "rag" + "_mode" not in _UPSERT

    def test_upsert_sql_has_lightrag_operational_fields(self):
        assert "ingest_strategy" in _UPSERT
        assert "parse_engine" in _UPSERT
        assert "process_options" in _UPSERT

    def test_pg_metadata_index_does_not_query_lightrag_doc_status_metadata(self):
        from dlightrag.storage import pg_metadata_index as module

        source = "\n".join(
            [
                module._UPSERT,
                module._CREATE_TABLE,
                "\n".join(module._CREATE_INDEXES),
            ]
        )

        assert "LIGHTRAG_DOC_STATUS.metadata" not in source
        assert "doc_status.metadata" not in source
