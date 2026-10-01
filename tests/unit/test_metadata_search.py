# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Bounded metadata-search cursor, page requests, and refusals before any read.

The exact-then-contains traversal runs against PostgreSQL in
tests/integration/test_pg_storage.py and test_metadata_scope_pg.py.
"""

import base64
import hashlib
import hmac
import json
from typing import Any, cast

import pytest

from dlightrag.adapters.postgres.corpus.pg_metadata_index import (
    metadata_match_conditions,
)
from dlightrag.adapters.postgres.corpus.pg_metadata_search import PGMetadataSearchStore
from dlightrag.application.corpus_admin import (
    MetadataSearchCursor,
    MetadataSearchCursorCodec,
    MetadataSearchCursorError,
    MetadataSearchPageRequest,
)
from dlightrag.engine.rag.retrieval import MetadataFilter


class _Acquire:
    def __init__(self, conn: Any) -> None:
        self._conn = conn

    async def __aenter__(self) -> Any:
        return self._conn

    async def __aexit__(self, *_exc: object) -> bool:
        return False


class _Pool:
    def __init__(self, conn: Any) -> None:
        self._conn = conn

    def acquire(self) -> _Acquire:
        return _Acquire(self._conn)


class _Conn:
    """Records every read, so a refusal can show it reached no storage."""

    def __init__(self) -> None:
        self.fetches: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.fetches.append((query, args))
        return []


def _signed_token(secret: bytes, payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    mac = hmac.new(secret, b"metadata-match\0" + raw, hashlib.sha256).digest()[:16]

    def encode(value: bytes) -> str:
        return base64.urlsafe_b64encode(value).rstrip(b"=").decode()

    return f"{encode(raw)}.{encode(mac)}"


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


def test_metadata_search_cursor_rejects_tamper_malformed_scope_version_and_mode() -> None:
    secret = b"cursor-secret"
    codec = MetadataSearchCursorCodec(secret)
    token = codec.encode(
        MetadataSearchCursor(workspace="finance", after_doc_id="doc-1", mode="exact")
    )
    encoded, mac = token.split(".")

    invalid = [
        token + "x",
        "not-a-token",
        f"{encoded}=.{mac}",
        _signed_token(
            secret,
            {
                "after_doc_id": "doc-1",
                "mode": "exact",
                "scope": "file-panel",
                "v": 1,
                "workspace": "finance",
            },
        ),
        _signed_token(
            secret,
            {
                "after_doc_id": "doc-1",
                "mode": "exact",
                "scope": "metadata-match",
                "v": 2,
                "workspace": "finance",
            },
        ),
        _signed_token(
            secret,
            {
                "after_doc_id": "doc-1",
                "mode": "regex",
                "scope": "metadata-match",
                "v": 1,
                "workspace": "finance",
            },
        ),
        _signed_token(
            secret,
            {
                "after_doc_id": "doc-1",
                "mode": ["exact"],
                "scope": "metadata-match",
                "v": 1,
                "workspace": "finance",
            },
        ),
        _signed_token(
            secret,
            {
                "after_doc_id": "doc-1",
                "mode": "exact",
                "scope": "metadata-match",
                "v": 1,
                "workspace": "finance",
                "extra": True,
            },
        ),
    ]
    for value in invalid:
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
    conn = _Conn()
    store = PGMetadataSearchStore(pool=_Pool(conn))

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
