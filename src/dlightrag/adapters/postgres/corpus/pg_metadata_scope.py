# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Bounded PostgreSQL metadata scope preflight.

The retrieval runtime never materializes a document-id set. This module builds
the one statement that answers the only two routing questions it needs:

* does the metadata predicate match at least one document, and
* how many chunks do the matching documents own, capped at
  ``metadata_filter_exact_vector_threshold + 1``.

The chunk probe counts rows of a subquery that stops at the cap, so a
multi-million-chunk match costs at most ``threshold + 1`` chunk visits and no
document ids cross into Python. A probe at or below the threshold is the exact
chunk count; the cap itself is a lower-bound sentinel.
"""

from typing import Any

from dlightrag.adapters.postgres.corpus._corpus_schema import LIGHTRAG_CHUNKS_TABLE
from dlightrag.adapters.postgres.corpus.pg_metadata_index import (
    METADATA_TABLE,
    metadata_match_conditions,
)
from dlightrag.engine.rag.retrieval import MetadataFilter


def build_bounded_scope_probe(
    workspace: str,
    filters: MetadataFilter,
    *,
    filename_mode: str,
    threshold: int,
) -> tuple[str, list[Any]]:
    """Build one preflight statement for one attempted filename mode.

    The statement returns one row with ``doc_exists`` (EXISTS over the metadata
    predicate) and ``chunk_count`` (a count over a chunk subquery capped at
    ``threshold + 1``). The workspace is authenticated on every relation the
    statement touches: the EXISTS probe, the chunk source, and the inner
    metadata subquery each carry their own bound ``workspace = $n``.

    Parameter layout: the EXISTS conditions occupy ``$1..$n``, the chunk-side
    workspace follows, then the inner metadata conditions repeat with shifted
    numbering, and the probe cap binds last.
    """
    threshold_value = int(threshold)
    if threshold_value < 0:
        raise ValueError("metadata scope probe threshold cannot be negative")
    exists_conditions, exists_params = metadata_match_conditions(
        workspace,
        filters,
        filename_mode=filename_mode,
        start_index=1,
    )
    inner_conditions, inner_params = metadata_match_conditions(
        workspace,
        filters,
        filename_mode=filename_mode,
        start_index=len(exists_params) + 2,
        alias="m",
    )
    chunk_workspace_slot = len(exists_params) + 1
    limit_slot = len(exists_params) + 2 + len(inner_params)
    sql = (
        "SELECT "  # noqa: S608 - private table constants; only $-params are bound.
        f"EXISTS (SELECT 1 FROM {METADATA_TABLE} "
        f"WHERE {' AND '.join(exists_conditions)}) AS doc_exists, "
        "(SELECT count(*) FROM ("
        f"SELECT 1 FROM {LIGHTRAG_CHUNKS_TABLE} c "
        f"WHERE c.workspace = ${chunk_workspace_slot} "
        "AND EXISTS ("
        f"SELECT 1 FROM {METADATA_TABLE} m "
        "WHERE m.workspace = c.workspace AND m.doc_id = c.full_doc_id "
        f"AND {' AND '.join(inner_conditions)}"
        f") LIMIT ${limit_slot}"
        ") AS bounded_chunks) AS chunk_count"
    )
    params: list[Any] = [*exists_params, workspace, *inner_params, threshold_value + 1]
    return sql, params


__all__ = ["build_bounded_scope_probe"]
