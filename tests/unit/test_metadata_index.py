# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PG metadata index rules that hold before any statement runs.

Queries, visibility, field schema and the migrations themselves run against
PostgreSQL in tests/integration/test_metadata_scope_pg.py.
"""

from dlightrag.adapters.postgres.corpus import pg_metadata_index
from dlightrag.adapters.postgres.corpus.pg_metadata_index import _SCHEMA_MIGRATIONS
from dlightrag.engine.rag.retrieval import MetadataFilter, MetadataScope
from dlightrag.engine.rag.retrieval.metadata_fields import METADATA_FIELD_IDS


def test_contains_predicate_escapes_pattern_language() -> None:
    from dlightrag.adapters.postgres.corpus.pg_metadata_index import like_contains_pattern

    assert like_contains_pattern("Report.pdf") == "%Report.pdf%"
    assert like_contains_pattern("50%_off\\docs") == "%50\\%\\_off\\\\docs%"


def test_migrations_are_derived_not_recorded_history() -> None:
    """Every version maps to a metadata field or named foundation step declared today."""
    declared = set(METADATA_FIELD_IDS)
    allowed = (
        {
            "document_metadata",
            "column_finalization_complete",
            "partition_default_child",
            "function_canonical_custom_metadata",
            "column_custom_metadata_search",
            "backfill_custom_metadata_search",
            "index_workspace_download_locator",
            "index_custom_metadata_search_gin",
            "index_filename_trgm",
            "metadata_field_stats",
            "product_document_visibility",
            "publish_legacy_processed_documents",
            "source_retrieval_options",
            "parser_input_sha256",
        }
        | {f"column_{field_id}" for field_id in declared}
        | {f"index_{field_id}_canonical" for field_id in declared}
    )

    assert {migration.version for migration in _SCHEMA_MIGRATIONS} <= allowed


async def test_empty_visibility_subset_never_queries_storage() -> None:
    idx = pg_metadata_index.PGMetadataIndex(workspace="finance")

    async def run(_operation):  # noqa: ANN001, ANN202
        raise AssertionError("empty caller subset must not query PostgreSQL")

    idx._run = run  # type: ignore[method-assign]

    assert await idx.visible_subset([]) == frozenset()
    empty_scope = MetadataScope(
        filters=MetadataFilter(filename="missing.pdf"),
        filename_mode="exact",
        doc_exists=False,
        candidate_count=0,
        candidate_count_exact=True,
    )
    assert await idx.visible_subset(["doc-candidate"], scope=empty_scope) == frozenset()
