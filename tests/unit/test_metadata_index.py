# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""PG metadata index rules that hold before any statement runs.

Queries, visibility, field schema and the migrations themselves run against
PostgreSQL in tests/integration/test_metadata_scope_pg.py.
"""

from dlightrag.adapters.postgres.corpus import pg_metadata_index
from dlightrag.engine.rag.retrieval.models import MetadataFilter, MetadataScope


def test_contains_predicate_escapes_pattern_language() -> None:
    from dlightrag.adapters.postgres.corpus.pg_metadata_index import like_contains_pattern

    assert like_contains_pattern("Report.pdf") == "%Report.pdf%"
    assert like_contains_pattern("50%_off\\docs") == "%50\\%\\_off\\\\docs%"


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
