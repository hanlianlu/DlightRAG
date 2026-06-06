# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for PG metadata index SQL generation."""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

from dlightrag.core.retrieval.metadata_fields import METADATA_FIELDS
from dlightrag.storage import pg_metadata_index
from dlightrag.storage.pg_metadata_index import _CREATE_INDEXES, _SCHEMA_MIGRATIONS, _UPSERT


class _Tx:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *args: object) -> None:
        return None


class _Conn:
    def __init__(self) -> None:
        self.applied: set[tuple[str, str]] = set()
        self.executed: list[tuple[str, tuple[Any, ...]]] = []
        self.table_exists = True

    def transaction(self) -> _Tx:
        return _Tx()

    async def fetchval(self, query: str, *args: Any) -> Any:
        if "information_schema.tables" in query:
            return self.table_exists
        if "dlightrag_schema_migrations" in query and "version" in query:
            return 1 if (str(args[0]), str(args[1])) in self.applied else None
        return None

    async def execute(self, query: str, *args: Any) -> None:
        self.executed.append((query, args))
        if query.startswith("INSERT INTO dlightrag_schema_migrations"):
            self.applied.add((str(args[0]), str(args[1])))


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

    def test_upsert_fields_follow_metadata_registry(self):
        expected = tuple(f.field_id for f in METADATA_FIELDS if f.field_id != "ingested_at")

        assert pg_metadata_index._UPSERT_FIELD_IDS == expected

        insert_columns = _UPSERT.split("VALUES", 1)[0]
        for field_id in expected:
            assert field_id in insert_columns
        assert "ingested_at" not in insert_columns

    def test_upsert_placeholders_match_registry_fields(self):
        placeholders = {int(match) for match in re.findall(r"\$(\d+)", _UPSERT)}

        assert max(placeholders) == len(pg_metadata_index._UPSERT_FIELD_IDS) + 2

    def test_upsert_params_follow_registry_field_order(self):
        metadata = {
            "filename": "report.pdf",
            "filename_stem": "report",
            "file_path": "/tmp/report.pdf",
            "file_extension": "pdf",
            "doc_title": "Report",
            "doc_author": "Ada",
            "ingest_strategy": "lightrag_sidecar_unified",
            "parse_engine": "mineru-iteP",
            "process_options": {"chunker": "recursive"},
        }
        params = pg_metadata_index._build_upsert_params(
            workspace="default",
            doc_id="doc-1",
            system=metadata,
            custom={"department": "finance"},
            metadata_json={"department": "Finance"},
        )

        assert params[:2] == ["default", "doc-1"]
        field_values = dict(
            zip(pg_metadata_index._UPSERT_FIELD_IDS, params[2:], strict=True)
        )
        assert field_values["filename"] == "report.pdf"
        assert json.loads(field_values["process_options"]) == {"chunker": "recursive"}
        assert json.loads(field_values["custom_metadata"]) == {"department": "finance"}
        assert json.loads(field_values["metadata_json"]) == {"department": "Finance"}
        assert "ingested_at" not in field_values

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

    def test_metadata_schema_migrations_cover_registry_columns_and_indexes(self):
        versions = {migration.version for migration in _SCHEMA_MIGRATIONS}
        sql = "\n".join(stmt for migration in _SCHEMA_MIGRATIONS for stmt in migration.statements)

        assert "0001_base" in versions
        for field in METADATA_FIELDS:
            assert f"column_{field.field_id}" in versions
            assert f"ADD COLUMN IF NOT EXISTS {field.field_id}" in sql
            if field.index_type is not None:
                assert f"index_{field.field_id}" in versions


async def test_metadata_index_read_only_verifies_migrations_without_ddl() -> None:
    conn = _Conn()
    conn.applied.update(("doc_metadata", migration.version) for migration in _SCHEMA_MIGRATIONS)
    idx = pg_metadata_index.PGMetadataIndex(workspace="default")

    async def run(operation):  # noqa: ANN001, ANN202
        return await operation(conn)

    idx._run = run  # type: ignore[method-assign]

    await idx.initialize(read_only=True)

    assert conn.executed == []


async def test_metadata_index_read_only_rejects_missing_migration() -> None:
    idx = pg_metadata_index.PGMetadataIndex(workspace="default")

    async def run(operation):  # noqa: ANN001, ANN202
        return await operation(_Conn())

    idx._run = run  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="doc_metadata schema migration"):
        await idx.initialize(read_only=True)
