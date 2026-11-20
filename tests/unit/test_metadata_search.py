# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Bounded metadata-search cursor, page requests, and refusals before any read.

The exact-then-contains traversal runs against PostgreSQL in
tests/integration/test_pg_storage.py and test_metadata_scope_pg.py.
"""

from typing import Any, cast

import pytest

from dlightrag.adapters.postgres.corpus.pg_metadata_index import (
    metadata_match_conditions,
)
from dlightrag.adapters.postgres.corpus.pg_metadata_search import PGMetadataSearchStore
from dlightrag.application.corpus_admin.file_panel import FilePanelCursor, FilePanelCursorCodec
from dlightrag.application.corpus_admin.metadata_search import (
    MetadataSearchCursor,
    MetadataSearchCursorCodec,
    MetadataSearchCursorError,
    MetadataSearchPageRequest,
)
from dlightrag.engine.rag.retrieval.models import MetadataFilter
from tests.support.pg_fakes import FakePool, RecordingConnection

# ---------------------------------------------------------------------------
# Cursor codec
# ---------------------------------------------------------------------------


def test_metadata_search_cursor_round_trips_both_modes_and_max_doc_id() -> None:
    codec = MetadataSearchCursorCodec(b"cursor-secret")

    for cursor in (
        MetadataSearchCursor(workspace="finance", after_doc_id="doc-1", mode="exact"),
        MetadataSearchCursor(workspace="finance", after_doc_id="d" * 255, mode="contains"),
    ):
        assert codec.decode(codec.encode(cursor)) == cursor


def test_metadata_search_cursor_rejects_tampered_foreign_and_invalid_tokens() -> None:
    secret = b"cursor-secret"
    codec = MetadataSearchCursorCodec(secret)
    token = codec.encode(
        MetadataSearchCursor(workspace="finance", after_doc_id="doc-1", mode="exact")
    )
    foreign = FilePanelCursorCodec(secret).encode(
        FilePanelCursor(workspace="finance", updated_at=None, doc_id="doc-1")
    )
    # Defense in depth: a sealed token is trusted for its shape, never its values.
    invalid_modes = [
        codec._envelope.encode({"after_doc_id": "doc-1", "mode": mode, "workspace": "finance"})
        for mode in ("regex", ["exact"])
    ]

    for value in (
        token + "x",
        ("A" if token[0] != "A" else "B") + token[1:],
        "not-a-token",
        foreign,
        *invalid_modes,
    ):
        with pytest.raises(MetadataSearchCursorError):
            codec.decode(value)


def test_metadata_search_page_validation_and_cursor_invariants() -> None:
    with pytest.raises(ValueError, match="between 1 and 100"):
        MetadataSearchPageRequest(limit=0)
    with pytest.raises(ValueError, match="between 1 and 100"):
        MetadataSearchPageRequest(limit=101)
    with pytest.raises(ValueError, match="integer"):
        MetadataSearchPageRequest(limit=True)
    with pytest.raises(ValueError, match="non-empty"):
        MetadataSearchCursor(workspace="finance", after_doc_id="", mode="exact")
    with pytest.raises(ValueError, match="exceeds the storage bound"):
        MetadataSearchCursor(workspace="finance", after_doc_id="d" * 256, mode="exact")
    with pytest.raises(ValueError, match="mode"):
        MetadataSearchCursor(workspace="finance", after_doc_id="doc", mode=cast(Any, "regex"))
    with pytest.raises(ValueError, match="canonical"):
        MetadataSearchCursor(workspace="Finance Reports", after_doc_id="doc", mode="exact")


# ---------------------------------------------------------------------------
# Shared condition builder
# ---------------------------------------------------------------------------


def test_match_conditions_reject_an_unknown_filename_mode() -> None:
    with pytest.raises(ValueError, match="mode"):
        metadata_match_conditions(
            "finance", MetadataFilter(filename="Quarterly Report"), filename_mode="regex"
        )


# ---------------------------------------------------------------------------
# Paged PostgreSQL adapter
# ---------------------------------------------------------------------------


async def test_page_store_rejects_cross_workspace_before_fetch() -> None:
    conn = RecordingConnection()
    store = PGMetadataSearchStore(pool=FakePool(conn))

    with pytest.raises(ValueError, match="another workspace"):
        await store.search_metadata_page(
            "finance",
            MetadataFilter(filename="Report"),
            page=MetadataSearchPageRequest(
                cursor=MetadataSearchCursor(
                    workspace="legal",
                    after_doc_id="doc",
                    mode="exact",
                )
            ),
        )

    assert conn.fetches == []
