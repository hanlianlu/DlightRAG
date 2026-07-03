# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for PG metadata index SQL generation."""

from __future__ import annotations

import json
import re
from typing import Any

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
        field_values = dict(zip(pg_metadata_index._UPSERT_FIELD_IDS, params[2:], strict=True))
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


async def test_metadata_index_initializes_schema_with_migrations() -> None:
    conn = _Conn()
    idx = pg_metadata_index.PGMetadataIndex(workspace="default")

    async def run(operation):  # noqa: ANN001, ANN202
        return await operation(conn)

    idx._run = run  # type: ignore[method-assign]

    await idx.initialize()

    executed_sql = "\n".join(query for query, _ in conn.executed)
    assert "CREATE TABLE IF NOT EXISTS dlightrag_schema_migrations" in executed_sql
    assert "CREATE TABLE IF NOT EXISTS dlightrag_doc_metadata" in executed_sql
    assert conn.applied == {("doc_metadata", migration.version) for migration in _SCHEMA_MIGRATIONS}


async def test_metadata_index_finds_by_exact_file_path() -> None:
    idx = pg_metadata_index.PGMetadataIndex(workspace="default")
    seen: dict[str, Any] = {}

    class Conn:
        async def fetch(self, query: str, *args: Any) -> list[dict[str, str]]:
            seen["query"] = query
            seen["args"] = args
            return [{"doc_id": "doc-1"}]

    async def run(operation):  # noqa: ANN001, ANN202
        return await operation(Conn())

    idx._run = run  # type: ignore[method-assign]

    assert await idx.find_by_file_path("/inputs/default/a/report.pdf") == ["doc-1"]
    assert "file_path=$2" in seen["query"]
    assert seen["args"] == ("default", "/inputs/default/a/report.pdf")


async def test_metadata_index_get_many_fetches_doc_ids_in_one_query() -> None:
    idx = pg_metadata_index.PGMetadataIndex(workspace="default")
    seen: dict[str, Any] = {}

    class Conn:
        async def fetch(self, query: str, *args: Any) -> list[dict[str, str]]:
            seen["query"] = query
            seen["args"] = args
            return [
                {"doc_id": "doc-1", "department": "finance"},
                {"doc_id": "doc-2", "department": "legal"},
            ]

    async def run(operation):  # noqa: ANN001, ANN202
        return await operation(Conn())

    idx._run = run  # type: ignore[method-assign]

    assert await idx.get_many(["doc-1", "doc-2", "doc-1"]) == {
        "doc-1": {"doc_id": "doc-1", "department": "finance"},
        "doc-2": {"doc_id": "doc-2", "department": "legal"},
    }
    assert "doc_id = ANY($2::text[])" in seen["query"]
    assert seen["args"] == ("default", ["doc-1", "doc-2"])
