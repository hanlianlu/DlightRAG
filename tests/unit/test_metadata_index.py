# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for PG metadata index SQL generation."""

import json
import re
from typing import Any

from dlightrag.core.retrieval.metadata_fields import METADATA_FIELDS
from dlightrag.core.retrieval.models import MetadataFilter
from dlightrag.storage import pg_metadata_index
from dlightrag.storage.pg_metadata_index import _SCHEMA_MIGRATIONS, _UPSERT


def _index_sql() -> str:
    return "\n".join(
        stmt
        for migration in _SCHEMA_MIGRATIONS
        for stmt in migration.statements
        if stmt.startswith("CREATE INDEX")
    )


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
        return None

    async def fetch(self, query: str, *args: Any) -> list[dict[str, str]]:
        if "dlightrag_schema_migrations" in query and "version" in query:
            scope = str(args[0])
            versions = sorted(
                version for applied_scope, version in self.applied if applied_scope == scope
            )
            return [{"version": version} for version in versions]
        return []

    async def execute(self, query: str, *args: Any) -> None:
        self.executed.append((query, args))
        if query.startswith("INSERT INTO dlightrag_schema_migrations"):
            self.applied.add((str(args[0]), str(args[1])))


class TestUpsertSQL:
    def test_upsert_sql_uses_coalesce(self):
        assert "COALESCE" in _UPSERT


class TestFilenameResolution:
    """One `filename` field, resolved server-side against name, stem, then contains."""

    @staticmethod
    def _index(rows_by_sql: dict[str, list[dict[str, str]]]) -> Any:
        index = pg_metadata_index.PGMetadataIndex.__new__(pg_metadata_index.PGMetadataIndex)
        index._workspace = "default"  # type: ignore[attr-defined]
        executed: list[tuple[str, tuple[Any, ...]]] = []

        async def _run(operation: Any) -> Any:
            class _Conn:
                async def fetch(self, sql: str, *params: Any) -> list[dict[str, str]]:
                    executed.append((sql, params))
                    for fragment, rows in rows_by_sql.items():
                        if fragment in sql:
                            return rows
                    return []

            return await operation(_Conn())

        index._run = _run  # type: ignore[attr-defined]
        return index, executed

    async def test_exact_hit_never_widens(self) -> None:
        index, executed = self._index({"LOWER(filename)": [{"doc_id": "d1"}]})

        result = await index.query(MetadataFilter(filename="report.pdf"))

        assert result == ["d1"]
        assert len(executed) == 1
        assert "ILIKE" not in executed[0][0]

    async def test_exact_clause_covers_name_and_stem(self) -> None:
        index, executed = self._index({"LOWER(filename)": [{"doc_id": "d1"}]})

        await index.query(MetadataFilter(filename="report"))

        sql = executed[0][0]
        assert "LOWER(filename) = LOWER($2)" in sql
        assert "LOWER(filename_stem) = LOWER($2)" in sql

    async def test_miss_widens_to_contains(self) -> None:
        index, executed = self._index({"ILIKE": [{"doc_id": "d2"}]})

        result = await index.query(MetadataFilter(filename="Linear Algebra"))

        assert result == ["d2"]
        assert len(executed) == 2
        assert "ILIKE $2" in executed[1][0]
        assert executed[1][1][1] == "%Linear Algebra%"

    async def test_caller_wildcards_are_preserved(self) -> None:
        index, executed = self._index({"ILIKE": [{"doc_id": "d3"}]})

        await index.query(MetadataFilter(filename="%IMG%9551%"))

        assert executed[1][1][1] == "%IMG%9551%"

    async def test_widening_keeps_other_conditions(self) -> None:
        index, executed = self._index({"ILIKE": [{"doc_id": "d4"}]})

        await index.query(MetadataFilter(filename="report", file_extension="pdf"))

        widened = executed[1][0]
        assert "LOWER(file_extension) = LOWER($2)" in widened
        assert "filename ILIKE $3" in widened

    async def test_no_filename_never_runs_twice(self) -> None:
        index, executed = self._index({})

        assert await index.query(MetadataFilter(file_extension="pdf")) == []
        assert len(executed) == 1


class TestMetadataSQL:
    def test_indexes_do_not_require_pg_trgm(self):
        sql = _index_sql()

        assert "gin_trgm" not in sql
        assert "trgm" not in sql

    def test_upsert_sql_does_not_reference_similarity(self):
        assert "similarity(" not in _UPSERT

    def test_string_btree_indexes_are_case_normalized(self):
        sql = _index_sql()

        assert "ON dlightrag_doc_metadata (LOWER(filename))" in sql
        assert "ON dlightrag_doc_metadata (LOWER(filename_stem))" in sql
        assert "ON dlightrag_doc_metadata (LOWER(file_extension))" in sql
        assert "ON dlightrag_doc_metadata (LOWER(doc_title))" in sql
        assert "ON dlightrag_doc_metadata (LOWER(doc_author))" in sql

    def test_non_string_btree_indexes_remain_plain(self):
        sql = _index_sql()

        assert "ON dlightrag_doc_metadata (creation_date)" in sql

    def test_download_locator_has_workspace_scoped_exact_index(self) -> None:
        sql = _index_sql()

        assert "ON dlightrag_doc_metadata (workspace, download_locator)" in sql

    def test_upsert_sql_has_lightrag_operational_fields(self):
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

    def test_metadata_schema_migrations_cover_registry_columns_and_indexes(self):
        versions = {migration.version for migration in _SCHEMA_MIGRATIONS}
        sql = "\n".join(stmt for migration in _SCHEMA_MIGRATIONS for stmt in migration.statements)

        assert "0001_base" in versions
        for field in METADATA_FIELDS:
            assert f"column_{field.field_id}" in versions
            assert f"ADD COLUMN IF NOT EXISTS {field.field_id}" in sql
            if field.index_type is not None:
                assert f"index_{field.field_id}" in versions

    def test_metadata_migrations_backfill_new_source_columns(self) -> None:
        migrations = list(_SCHEMA_MIGRATIONS)
        versions = [migration.version for migration in migrations]
        backfill_index = versions.index("backfill_source_download_contract")

        assert versions.index("column_source_uri") < backfill_index
        assert versions.index("column_download_locator") < backfill_index
        statement = "\n".join(migrations[backfill_index].statements)
        assert "source_uri = COALESCE(source_uri, CASE" in statement
        assert "download_locator = COALESCE(download_locator, file_path)" in statement
        assert "WHERE source_uri IS NULL OR download_locator IS NULL" in statement


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


async def test_metadata_index_initialization_disables_prefix_only_validation() -> None:
    idx = pg_metadata_index.PGMetadataIndex(workspace="default")
    seen: dict[str, Any] = {}

    async def fake_apply_migrations(conn: Any, **kwargs: Any) -> None:
        seen["conn"] = conn
        seen.update(kwargs)

    async def run(operation):  # noqa: ANN001, ANN202
        return await operation(object())

    original = pg_metadata_index.apply_migrations
    pg_metadata_index.apply_migrations = fake_apply_migrations  # type: ignore[assignment]
    idx._run = run  # type: ignore[method-assign]
    try:
        await idx.initialize()
    finally:
        pg_metadata_index.apply_migrations = original  # type: ignore[assignment]

    assert seen["scope"] == "doc_metadata"
    assert seen["migrations"] == _SCHEMA_MIGRATIONS
    assert seen["require_applied_prefix"] is False


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


async def test_metadata_index_finds_by_exact_download_locator() -> None:
    idx = pg_metadata_index.PGMetadataIndex(workspace="finance")
    seen: dict[str, Any] = {}

    class Conn:
        async def fetch(self, query: str, *args: Any) -> list[dict[str, str]]:
            seen["query"] = query
            seen["args"] = args
            return [{"doc_id": "doc-1"}, {"doc_id": "doc-2"}]

    async def run(operation):  # noqa: ANN001, ANN202
        return await operation(Conn())

    idx._run = run  # type: ignore[method-assign]

    assert await idx.find_by_download_locator("s3://bucket/team/report.pdf") == [
        "doc-1",
        "doc-2",
    ]
    assert "download_locator=$2" in seen["query"]
    assert "LOWER" not in seen["query"]
    assert seen["args"] == ("finance", "s3://bucket/team/report.pdf")


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


async def test_metadata_field_schema_reports_only_populated_filters() -> None:
    idx = pg_metadata_index.PGMetadataIndex(workspace="default")

    class Conn:
        async def fetchrow(self, query: str, *args: Any) -> dict[str, Any]:
            assert "workspace = ANY($1::text[])" in query
            assert args == (["default"],)
            return {
                "filename": True,
                "filename_stem": True,
                "file_extension": True,
                "doc_title": False,
                "doc_author": False,
                "creation_date": False,
                "custom_keys": ["department"],
            }

    async def run(operation):  # noqa: ANN001, ANN202
        return await operation(Conn())

    idx._run = run  # type: ignore[method-assign]

    schema = await idx.get_field_schema()

    # An empty column would only invite the planner to filter on nothing.
    assert schema == {
        "filters": ["filename", "file_extension"],
        "custom_keys": ["department"],
    }


async def test_metadata_field_schema_offers_a_date_range_from_one_column() -> None:
    idx = pg_metadata_index.PGMetadataIndex(workspace="default")

    class Conn:
        async def fetchrow(self, query: str, *args: Any) -> dict[str, Any]:
            return dict.fromkeys(pg_metadata_index._FILTERABLE_COLUMNS, False) | {
                "creation_date": True,
                "custom_keys": None,
            }

    async def run(operation):  # noqa: ANN001, ANN202
        return await operation(Conn())

    idx._run = run  # type: ignore[method-assign]

    schema = await idx.get_field_schema()

    assert schema == {"filters": ["creation_date_from", "creation_date_to"], "custom_keys": []}


async def test_metadata_field_schema_reads_multiple_workspaces_in_one_operation() -> None:
    idx = pg_metadata_index.PGMetadataIndex(workspace="default")
    seen: list[tuple[str, tuple[Any, ...]]] = []

    class Conn:
        async def fetchrow(self, query: str, *args: Any) -> dict[str, Any]:
            seen.append((query, args))
            return dict.fromkeys(pg_metadata_index._FILTERABLE_COLUMNS, False) | {
                "custom_keys": ["department", "jurisdiction"],
            }

    async def run(operation):  # noqa: ANN001, ANN202
        return await operation(Conn())

    idx._run = run  # type: ignore[method-assign]

    schema = await idx.get_field_schema(workspaces=("reports", "legal"))

    assert schema["custom_keys"] == ["department", "jurisdiction"]
    assert len(seen) == 1
    assert "workspace = ANY($1::text[])" in seen[0][0]
    assert seen[0][1] == (["reports", "legal"],)
