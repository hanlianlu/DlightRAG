# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for principal-scoped durable Web conversation storage."""

from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from dlightrag.core.request.attachments import (
    AttachmentCacheKey,
    AttachmentContextChunk,
    ParsedAttachmentBundle,
)
from dlightrag.storage.web_conversations import (
    WEB_CONVERSATION_MIGRATIONS,
    CommitTurnResult,
    ConversationSnapshot,
    PendingConversationImage,
    PGWebConversationStore,
    StoredConversationImage,
)


class FakeTransaction:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *_args: object) -> None:
        return None


class FakeConnection:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.transactions: list[dict[str, Any]] = []
        self.fetch_result: list[dict[str, Any]] = []
        self.fetchrow_result: dict[str, Any] | None = None
        self.fetchrow_results: list[dict[str, Any] | None] = []

    async def execute(self, query: str, *args: Any) -> str:
        self.calls.append((query, args))
        return "OK"

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.calls.append((query, args))
        return self.fetch_result

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        self.calls.append((query, args))
        if self.fetchrow_results:
            return self.fetchrow_results.pop(0)
        return self.fetchrow_result

    def transaction(self, **kwargs: Any) -> FakeTransaction:
        self.transactions.append(kwargs)
        return FakeTransaction()


class FakeAcquire:
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection

    async def __aenter__(self) -> FakeConnection:
        return self.connection

    async def __aexit__(self, *_args: object) -> None:
        return None


class FakePool:
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection

    def acquire(self) -> FakeAcquire:
        return FakeAcquire(self.connection)


class FakeProductionPool:
    def __init__(self, connection: FakeConnection) -> None:
        self.connection = connection
        self.retrying_calls = 0
        self.single_attempt_calls = 0

    async def run(self, operation):  # noqa: ANN001, ANN202
        self.retrying_calls += 1
        return await operation(self.connection)

    async def run_once(self, operation):  # noqa: ANN001, ANN202
        self.single_attempt_calls += 1
        return await operation(self.connection)


def make_store(connection: FakeConnection) -> PGWebConversationStore:
    store = PGWebConversationStore(pool=FakePool(connection))
    store._initialized = True
    return store


def _cached_chunk(
    *,
    cache_chunk_id: str = "cache-1",
    parser_signature: str = "parser-v1",
    chunk_signature: str = "chunker-v1",
    embedding_signature: str | None = "embed-v1",
    embedding_vector: list[float] | None = None,
) -> AttachmentContextChunk:
    return AttachmentContextChunk(
        chunk_id="citation-1",
        attachment_id="attachment-1",
        filename="report.pdf",
        chunk_index=1,
        content="alpha",
        cache_key=AttachmentCacheKey(
            content_sha256="content-v1",
            parser_signature=parser_signature,
            chunk_signature=chunk_signature,
            cache_chunk_id=cache_chunk_id,
        ),
        embedding_signature=embedding_signature,
        embedding_vector=embedding_vector,
    )


async def test_attachment_chunk_load_is_ttl_scoped_and_keeps_cache_identity_private() -> None:
    conn = FakeConnection()
    conn.fetch_result = [
        {
            "chunk_id": "stored-cache-id",
            "chunk_index": 1,
            "content": "alpha",
            "page_idx": None,
            "bbox": None,
            "sidecar_type": None,
            "image_bytes": None,
            "image_mime_type": None,
            "token_estimate": 2,
            "metadata": {},
            "parser_signature": "parser-v1",
            "chunk_signature": "chunker-v1",
        }
    ]
    store = make_store(conn)

    bundle = await store.load_attachment_chunks(
        "p1",
        "c1",
        "attachment-new",
        "report.pdf",
        content_sha256="content-v1",
        parser_signature="parser-v1",
        chunk_signature="chunker-v1",
        ttl_days=30,
    )

    assert bundle is not None
    chunk = bundle.chunks[0]
    assert chunk.chunk_id == "att:attachment-new:0001"
    assert chunk.cache_key == AttachmentCacheKey(
        "content-v1", "parser-v1", "chunker-v1", "stored-cache-id"
    )
    assert chunk.embedding_signature is None
    assert chunk.embedding_vector is None
    query, args = conn.calls[-1]
    assert "JOIN web_conversations" in query
    assert "updated_at >= NOW()" in query
    assert "embedding_vector" not in query
    assert args == ("p1", "c1", "content-v1", "parser-v1", "chunker-v1", 30)


async def test_materialize_attachment_chunks_persists_vectors_atomically_when_unexpired() -> None:
    conn = FakeConnection()
    conn.fetchrow_result = {"unexpired": 1}
    store = make_store(conn)
    bundle = ParsedAttachmentBundle(
        chunks=[_cached_chunk(embedding_vector=[0.25, 0.75])],
        parser_signature="parser-v1",
    )

    materialized = await store.materialize_attachment_chunks(
        "p1",
        "c1",
        content_sha256="content-v1",
        parser_signature="parser-v1",
        chunk_signature="chunker-v1",
        bundle=bundle,
        ttl_days=30,
    )

    assert materialized is True
    assert conn.transactions == [{}]
    guard_query, guard_args = conn.calls[0]
    assert "FOR UPDATE" in guard_query
    assert "updated_at >= NOW()" in guard_query
    assert guard_args == ("p1", "c1", 30)
    insert_query, insert_args = conn.calls[-1]
    assert "embedding_signature" in insert_query
    assert "embedding_vector" in insert_query
    assert insert_args[-2] == "embed-v1"
    assert insert_args[-1] == "[0.25, 0.75]"


async def test_expired_attachment_chunk_writes_are_noops() -> None:
    conn = FakeConnection()
    conn.fetchrow_result = None
    store = make_store(conn)
    bundle = ParsedAttachmentBundle(chunks=[_cached_chunk(embedding_vector=[1.0, 0.0])])

    assert (
        await store.materialize_attachment_chunks(
            "p1",
            "c1",
            content_sha256="content-v1",
            parser_signature="parser-v1",
            chunk_signature="chunker-v1",
            bundle=bundle,
            ttl_days=30,
        )
        is False
    )
    assert (
        await store.aupdate_attachment_chunk_vectors(
            "p1", "c1", [_cached_chunk(embedding_vector=[1.0, 0.0])], ttl_days=30
        )
        is False
    )
    assert not any(
        "DELETE FROM web_conversation_attachment_chunks" in query for query, _ in conn.calls
    )
    assert not any("UPDATE web_conversation_attachment_chunks" in query for query, _ in conn.calls)


async def test_vector_refresh_uses_the_complete_scoped_cache_key() -> None:
    conn = _UpdatingConnection()
    conn.fetchrow_result = {"unexpired": 1}
    store = make_store(conn)
    chunk = _cached_chunk(
        cache_chunk_id="same-id",
        parser_signature="parser-current",
        chunk_signature="chunker-current",
        embedding_vector=[1.0, 2.0],
    )

    assert await store.aupdate_attachment_chunk_vectors("p1", "c1", [chunk], ttl_days=30) is True

    update_query, update_args = conn.calls[-1]
    assert "principal_id = $1" in update_query
    assert "conversation_id = $2::text::uuid" in update_query
    assert "content_sha256 = $3" in update_query
    assert "parser_signature = $4" in update_query
    assert "chunk_signature = $5" in update_query
    assert "chunk_id = $6" in update_query
    assert update_args[:6] == (
        "p1",
        "c1",
        "content-v1",
        "parser-current",
        "chunker-current",
        "same-id",
    )


async def test_attachment_vector_iterator_preserves_duplicate_reference_ordinality() -> None:
    key = AttachmentCacheKey("content-v1", "parser-v1", "chunker-v1", "cache-1")
    conn = FakeConnection()
    conn.fetch_result = [
        {
            "global_order": 3,
            "content_sha256": key.content_sha256,
            "parser_signature": key.parser_signature,
            "chunk_signature": key.chunk_signature,
            "cache_chunk_id": key.cache_chunk_id,
            "embedding_signature": "embed-v1",
            "embedding_vector": [1.0, 0.0],
        },
        {
            "global_order": 8,
            "content_sha256": key.content_sha256,
            "parser_signature": key.parser_signature,
            "chunk_signature": key.chunk_signature,
            "cache_chunk_id": key.cache_chunk_id,
            "embedding_signature": "embed-v1",
            "embedding_vector": [1.0, 0.0],
        },
    ]
    store = make_store(conn)

    pages = [
        page
        async for page in store.aiter_attachment_vectors(
            "p1",
            "c1",
            [(8, key), (3, key)],
            ttl_days=30,
            page_size=10,
        )
    ]

    assert [[row.global_order for row in page] for page in pages] == [[3, 8]]
    assert [row.cache_key for row in pages[0]] == [key, key]
    query, args = conn.calls[-1]
    assert "WITH ORDINALITY" in query
    assert "JOIN web_conversations" in query
    assert "updated_at >= NOW()" in query
    assert args[2] == [3, 8]


async def test_list_image_catalog_returns_caption_fields() -> None:
    conn = FakeConnection()
    conn.fetch_result = [
        {"image_id": "img-1", "turn_number": 1, "ordinal": 0, "vlm_description": "a red square"},
    ]
    store = make_store(conn)

    catalog = await store.list_image_catalog("alice", "c1", max_turns=100, ttl_days=30)

    assert catalog == [
        {"image_id": "img-1", "turn_number": 1, "ordinal": 0, "vlm_description": "a red square"}
    ]


async def test_fetch_images_by_ids_carries_vlm_description() -> None:
    conn = FakeConnection()
    conn.fetch_result = [
        {
            "image_id": "img-1",
            "mime_type": "image/png",
            "image_bytes": b"PNG",
            "vlm_description": "a red square",
        },
    ]
    store = make_store(conn)

    images = await store.fetch_images_by_ids("alice", "c1", ["img-1"], ttl_days=30)

    assert len(images) == 1
    assert isinstance(images[0], StoredConversationImage)
    assert images[0].image_bytes == b"PNG"
    assert images[0].vlm_description == "a red square"


async def test_fetch_images_by_ids_empty_short_circuits() -> None:
    conn = FakeConnection()
    store = make_store(conn)

    assert await store.fetch_images_by_ids("alice", "c1", [], ttl_days=30) == []
    assert conn.calls == []  # never issues a query for an empty id list


async def test_outcome_sensitive_mutations_use_single_attempt_pool_path(monkeypatch) -> None:
    conn = FakeConnection()
    production_pool = FakeProductionPool(conn)
    monkeypatch.setattr("dlightrag.storage.pool.pg_pool", production_pool)
    store = PGWebConversationStore()
    store._initialized = True

    conn.fetchrow_result = {"conversation_id": "c1"}
    await store.create_conversation("p1")

    conn.fetchrow_result = {"conversation_id": "c1", "title": "Renamed"}
    await store.rename_conversation("p1", "c1", title="Renamed", ttl_days=30)

    conn.fetchrow_result = {"deleted": 1}
    assert await store.delete_conversation("p1", "c1", ttl_days=30) is True

    conn.fetchrow_results = [
        None,
        {"conversation_id": "c1", "title": "Renamed", "content_revision": 1},
        {"turn_id": "t1", "turn_number": 1},
    ]
    await store.commit_turn(
        principal_id="p1",
        conversation_id="c1",
        expected_revision=0,
        user_text="Question",
        assistant_text="Answer",
        answer_sources={},
        queried_workspaces=[],
        images=[],
        attachments=[],
        max_turns=100,
        submission_id="00000000-0000-4000-8000-000000000001",
        ttl_days=30,
    )

    conn.fetchrow_result = {"count": 2}
    assert await store.prune_expired(ttl_days=30) == 2

    assert production_pool.single_attempt_calls == 5
    assert production_pool.retrying_calls == 0


async def test_reads_keep_retrying_pool_path(monkeypatch) -> None:
    conn = FakeConnection()
    production_pool = FakeProductionPool(conn)
    monkeypatch.setattr("dlightrag.storage.pool.pg_pool", production_pool)
    store = PGWebConversationStore()
    store._initialized = True

    assert await store.snapshot("p1", "c1", ttl_days=30) is None

    assert production_pool.retrying_calls == 1
    assert production_pool.single_attempt_calls == 0


class _UpdatingConnection(FakeConnection):
    async def execute(self, query: str, *args: Any) -> str:
        self.calls.append((query, args))
        return "UPDATE 1"


async def test_update_turn_sources_writes_snapshot_by_submission_key() -> None:
    conn = _UpdatingConnection()
    store = make_store(conn)

    updated = await store.update_turn_sources(
        principal_id="p1",
        conversation_id="c1",
        submission_id="00000000-0000-4000-8000-000000000001",
        answer_sources={
            "sources": [{"id": "1", "chunks": [{"highlight_phrases": ["neural nets"]}]}],
            "answer_images": [],
        },
    )

    assert updated is True
    query, args = conn.calls[-1]
    assert "UPDATE web_conversation_turns" in query
    assert "answer_sources" in query
    assert args[0] == "p1"
    assert args[1] == "c1"
    assert args[2] == "00000000-0000-4000-8000-000000000001"
    assert "highlight_phrases" in args[3]


async def test_update_turn_sources_reports_missing_turn() -> None:
    conn = FakeConnection()  # execute() returns "OK" → no row matched the submission key
    store = make_store(conn)

    assert (
        await store.update_turn_sources(
            principal_id="p1",
            conversation_id="c1",
            submission_id="00000000-0000-4000-8000-000000000002",
            answer_sources={},
        )
        is False
    )


async def test_commit_reconciliation_lookup_uses_single_attempt_pool_path(monkeypatch) -> None:
    conn = FakeConnection()
    production_pool = FakeProductionPool(conn)
    monkeypatch.setattr("dlightrag.storage.pool.pg_pool", production_pool)
    store = PGWebConversationStore()
    store._initialized = True

    assert (
        await store.find_committed_turn(
            "p1",
            "c1",
            "00000000-0000-4000-8000-000000000001",
            ttl_days=30,
            retry=False,
        )
        is None
    )

    assert production_pool.single_attempt_calls == 1
    assert production_pool.retrying_calls == 0


async def test_list_prunes_expired_rows_before_selecting() -> None:
    conn = FakeConnection()
    conn.fetch_result = []
    store = make_store(conn)

    assert await store.list_conversations("principal-a", ttl_days=30) == []

    statements = "\n".join(query for query, _ in conn.calls)
    assert "DELETE FROM web_conversations" in statements
    assert "updated_at < NOW()" in statements
    assert "principal_id = $1" in statements
    assert conn.calls[0][1] == ("principal-a", 30)
    assert conn.calls[1][1] == ("principal-a", 30)


async def test_global_expiry_prune_is_batched_and_skip_locked() -> None:
    conn = FakeConnection()
    conn.fetchrow_result = {"count": 500}
    store = make_store(conn)

    deleted = await store.prune_expired(ttl_days=30, batch_size=500)

    assert deleted == 500
    query, args = conn.calls[-1]
    assert "LIMIT $2" in query
    assert "FOR UPDATE SKIP LOCKED" in query
    assert args == (30, 500)
    migration_sql = "\n".join(
        statement for migration in WEB_CONVERSATION_MIGRATIONS for statement in migration.statements
    )
    assert "idx_web_conversations_updated" in migration_sql


async def test_same_conversation_id_is_scoped_by_principal() -> None:
    conn = FakeConnection()
    conn.fetchrow_result = None
    store = make_store(conn)

    assert await store.snapshot("principal-a", "same-id", ttl_days=30) is None

    query, args = conn.calls[-1]
    assert "principal_id = $1" in query
    assert "conversation_id = $2" in query
    assert "updated_at >= NOW()" in query
    assert args[:2] == ("principal-a", "same-id")


async def test_snapshot_reads_revision_and_history_from_one_database_snapshot() -> None:
    import datetime

    now = datetime.datetime(2026, 7, 12, tzinfo=datetime.UTC)
    conn = FakeConnection()
    conn.fetchrow_result = {
        "principal_id": "p1",
        "conversation_id": "c1",
        "content_revision": 4,
        "title": "Conversation",
        "created_at": now,
        "updated_at": now,
    }
    store = make_store(conn)

    snapshot = await store.snapshot("p1", "c1", ttl_days=30)

    assert snapshot is not None
    assert snapshot.content_revision == 4
    assert snapshot.created_at == now
    assert snapshot.updated_at == now
    assert snapshot.history == ()
    assert conn.transactions == [{"isolation": "repeatable_read", "readonly": True}]
    metadata_query, _ = conn.calls[0]
    assert "created_at" in metadata_query
    assert "updated_at" in metadata_query


async def test_commit_turn_is_revision_guarded_and_trims_old_turns() -> None:
    conn = FakeConnection()
    conn.fetchrow_results = [
        None,
        {"conversation_id": "c1", "title": "First", "content_revision": 2},
        {"turn_id": "t1", "turn_number": 101},
    ]
    store = make_store(conn)

    result = await store.commit_turn(
        principal_id="p1",
        conversation_id="c1",
        expected_revision=1,
        user_text="Question",
        assistant_text="Answer",
        answer_sources={"sources": [], "answer_images": []},
        queried_workspaces=["default"],
        images=[],
        attachments=[],
        max_turns=100,
        submission_id="00000000-0000-4000-8000-000000000001",
        ttl_days=30,
    )

    assert result == CommitTurnResult(
        saved=True,
        reason=None,
        summary={"conversation_id": "c1", "title": "First", "content_revision": 2},
        turn_id="t1",
        assistant_text="Answer",
        answer_sources={"sources": [], "answer_images": []},
    )
    statements = "\n".join(query for query, _ in conn.calls)
    assert "content_revision = $3" in statements
    assert "DELETE FROM web_conversation_turns" in statements
    trim_query, trim_args = conn.calls[-1]
    assert "principal_id = $1" in trim_query
    assert "conversation_id = $2" in trim_query
    assert trim_args == ("p1", "c1", 1)


async def test_commit_turn_stops_when_revision_changed() -> None:
    conn = FakeConnection()
    conn.fetchrow_result = None
    store = make_store(conn)

    result = await store.commit_turn(
        principal_id="p1",
        conversation_id="c1",
        expected_revision=8,
        user_text="Question",
        assistant_text="Answer",
        answer_sources={},
        queried_workspaces=[],
        images=[],
        attachments=[],
        max_turns=100,
        submission_id="00000000-0000-4000-8000-000000000002",
        ttl_days=30,
    )

    assert result == CommitTurnResult(False, "conversation_changed", None, None)
    assert len(conn.calls) == 3
    query, args = conn.calls[1]
    assert "principal_id = $1" in query
    assert "conversation_id = $2" in query
    assert "content_revision = $3" in query
    assert args[:3] == ("p1", "c1", 8)


async def test_commit_turn_inserts_scoped_image_bytes() -> None:
    conn = FakeConnection()
    conn.fetchrow_results = [
        None,
        {"conversation_id": "c1", "title": "Question", "content_revision": 1},
        {"turn_id": "t1", "turn_number": 1},
    ]
    store = make_store(conn)
    image = PendingConversationImage(
        image_id="i1",
        ordinal=0,
        mime_type="image/png",
        image_bytes=b"png",
        content_sha256="abc123",
        vlm_description="A chart",
    )

    result = await store.commit_turn(
        principal_id="p1",
        conversation_id="c1",
        expected_revision=0,
        user_text="Question",
        assistant_text="Answer",
        answer_sources={},
        queried_workspaces=["research"],
        images=[image],
        attachments=[],
        max_turns=100,
        submission_id="00000000-0000-4000-8000-000000000003",
        ttl_days=30,
    )

    assert result.saved is True
    image_calls = [call for call in conn.calls if "INSERT INTO web_conversation_images" in call[0]]
    assert len(image_calls) == 1
    assert image_calls[0][1] == (
        "i1",
        "p1",
        "c1",
        "t1",
        0,
        "image/png",
        b"png",
        3,
        "abc123",
        "A chart",
    )


async def test_get_image_is_scoped_and_ttl_guarded() -> None:
    conn = FakeConnection()
    conn.fetchrow_result = {
        "image_id": "i1",
        "mime_type": "image/png",
        "image_bytes": b"png",
    }
    store = make_store(conn)

    assert await store.get_image("p1", "c1", "i1", ttl_days=30) == StoredConversationImage(
        image_id="i1",
        mime_type="image/png",
        image_bytes=b"png",
    )
    query, args = conn.calls[-1]
    assert "i.principal_id = $1" in query
    assert "i.conversation_id = $2" in query
    assert "i.image_id = $3" in query
    assert "c.updated_at >= NOW()" in query
    assert args == ("p1", "c1", "i1", 30)


def test_records_are_frozen_and_schema_is_exact() -> None:
    import datetime

    now = datetime.datetime(2026, 7, 12, tzinfo=datetime.UTC)
    snapshot = ConversationSnapshot("p1", "c1", 0, None, now, now, ())
    with pytest.raises(FrozenInstanceError):
        snapshot.title = "changed"  # type: ignore[misc]

    sql = "\n".join(
        statement for migration in WEB_CONVERSATION_MIGRATIONS for statement in migration.statements
    )
    assert "PRIMARY KEY (principal_id, conversation_id)" in sql
    assert "UNIQUE (principal_id, conversation_id, turn_number)" in sql
    assert "FOREIGN KEY (principal_id, conversation_id, turn_id)" in sql
    assert "ON DELETE CASCADE" in sql
    assert "BYTEA" in sql
    assert "submission_id UUID NOT NULL" in sql
    assert "principal_id, conversation_id, submission_id" in sql


async def test_same_submission_replay_returns_authoritative_turn_without_insert() -> None:
    conn = FakeConnection()
    conn.fetchrow_result = {
        "conversation_id": "c1",
        "title": "Question",
        "content_revision": 1,
        "created_at": "created",
        "updated_at": "updated",
        "turn_id": "t1",
        "assistant_text": "Stored answer",
        "answer_sources": {"sources": []},
        "images": [{"image_id": "i1", "ordinal": 1, "vlm_description": "chart"}],
    }
    store = make_store(conn)

    result = await store.commit_turn(
        principal_id="p1",
        conversation_id="c1",
        submission_id="00000000-0000-4000-8000-000000000001",
        expected_revision=99,
        user_text="Question",
        assistant_text="Different retry answer",
        answer_sources={},
        queried_workspaces=[],
        images=[],
        attachments=[],
        max_turns=100,
        ttl_days=30,
    )

    assert result.saved is True
    assert result.replayed is True
    assert result.assistant_text == "Stored answer"
    assert result.current_image_ids == ("i1",)
    assert not any("INSERT INTO web_conversation_turns" in query for query, _args in conn.calls)


def test_migrations_drop_only_legacy_conversation_storage() -> None:
    sql = "\n".join(
        statement for migration in WEB_CONVERSATION_MIGRATIONS for statement in migration.statements
    )
    assert "DROP TABLE IF EXISTS dlightrag_checkpoints" in sql
    assert "DELETE FROM dlightrag_schema_migrations WHERE scope = 'checkpoints'" in sql
    assert "dlightrag_doc_metadata" not in sql
    assert "lightrag_doc_chunks" not in sql
