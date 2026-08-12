# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for principal-scoped durable Web conversation storage.

Storage keeps Web answer attachments as raw resources only: there is no image
table, no parse/chunk/vector cache, and no stored VLM description. Every prior
derived-artifact API and dataclass has been removed.
"""

import datetime
import inspect
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

import dlightrag.storage.web_conversations as web_conversations_module
from dlightrag.storage.web_conversations import (
    WEB_CONVERSATION_MIGRATIONS,
    CommitTurnResult,
    ConversationSnapshot,
    PendingConversationAttachment,
    PGWebConversationStore,
    StoredConversationAttachment,
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
        self.applied: set[tuple[str, str]] = set()
        self.fetch_result: list[dict[str, Any]] = []
        self.fetchrow_result: dict[str, Any] | None = None
        self.fetchrow_results: list[dict[str, Any] | None] = []

    async def execute(self, query: str, *args: Any) -> str:
        self.calls.append((query, args))
        if query.startswith("INSERT INTO dlightrag_schema_migrations"):
            self.applied.add((str(args[0]), str(args[1])))
            return "INSERT 0 1"
        if query.startswith("SELECT pg_advisory_lock") or query.startswith(
            "SELECT pg_advisory_unlock"
        ):
            return "SELECT 1"
        return "OK"

    async def executemany(self, query: str, args_seq: Any) -> None:
        for args in args_seq:
            self.calls.append((query, tuple(args)))

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.calls.append((query, args))
        if "dlightrag_schema_migrations" in query and "version" in query:
            scope = str(args[0])
            versions = sorted(
                version for applied_scope, version in self.applied if applied_scope == scope
            )
            return [{"version": version} for version in versions]
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


class _UpdatingConnection(FakeConnection):
    async def execute(self, query: str, *args: Any) -> str:
        self.calls.append((query, args))
        return "UPDATE 1"


def make_store(connection: FakeConnection) -> PGWebConversationStore:
    store = PGWebConversationStore(pool=FakePool(connection))
    store._initialized = True
    return store


def _migration_sql() -> str:
    return "\n".join(
        statement for migration in WEB_CONVERSATION_MIGRATIONS for statement in migration.statements
    )


# --- Unified schema and reset migration -------------------------------------


def test_storage_module_has_no_derived_artifact_surface() -> None:
    """No image, parse-cache, vector, or VLM-description SQL/dataclass remains."""
    source = inspect.getsource(web_conversations_module)
    forbidden_tokens = (
        "parse_summary",
        "parse_catalog",
        "parser_signature",
        "chunk_signature",
        "embedding_signature",
        "embedding_vector",
        "vlm_description",
        "image_bytes",
        "PendingConversationImage",
        "StoredConversationImage",
        "AttachmentCacheKey",
        "AttachmentContextChunk",
        "AttachmentVectorPageRow",
        "ParsedAttachmentBundle",
    )
    present = [token for token in forbidden_tokens if token in source]
    assert present == [], f"forbidden storage surface still present: {present}"

    # The superseded tables may be named only to DROP them in the reset
    # migration; they are never created, queried, or written.
    for old_table in ("web_conversation_images", "web_conversation_attachment_chunks"):
        for line in source.splitlines():
            if old_table in line:
                assert f'"DROP TABLE IF EXISTS {old_table}"' in line, (
                    f"{old_table} referenced outside a DROP statement: {line.strip()}"
                )

    removed_methods = (
        "get_image",
        "fetch_images_by_ids",
        "list_image_catalog",
        "load_attachment_chunks",
        "materialize_attachment_chunks",
        "aupdate_attachment_chunk_vectors",
        "aiter_attachment_vectors",
        "fetch_documents_by_ids",
    )
    still_defined = [name for name in removed_methods if hasattr(PGWebConversationStore, name)]
    assert still_defined == [], f"removed storage methods still defined: {still_defined}"


def test_unified_attachment_table_stores_raw_resources_only() -> None:
    sql = _migration_sql()
    assert "CREATE TABLE IF NOT EXISTS web_conversation_attachments" in sql
    for column in (
        "attachment_id UUID NOT NULL",
        "principal_id TEXT NOT NULL",
        "conversation_id UUID NOT NULL",
        "turn_id UUID NOT NULL",
        "ordinal INTEGER NOT NULL",
        "filename TEXT NOT NULL",
        "mime_type TEXT NOT NULL",
        "suffix TEXT NOT NULL",
        "attachment_bytes BYTEA NOT NULL",
        "byte_size INTEGER NOT NULL",
        "content_sha256 TEXT NOT NULL",
        "created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()",
    ):
        assert column in sql, column
    assert "UNIQUE (principal_id, conversation_id, turn_id, ordinal)" in sql
    assert (
        "FOREIGN KEY (principal_id, conversation_id, turn_id)\n"
        "      REFERENCES web_conversation_turns (principal_id, conversation_id, turn_id)\n"
        "      ON DELETE CASCADE"
    ) in sql
    assert "idx_web_conversation_attachments_catalog" in sql


def test_attachment_migration_resets_rows_and_drops_superseded_tables() -> None:
    versions = [migration.version for migration in WEB_CONVERSATION_MIGRATIONS]
    assert versions == sorted(versions)
    assert len(versions) >= 2
    reset = next(
        migration
        for migration in WEB_CONVERSATION_MIGRATIONS
        if migration.version == "0002_unified_web_conversation_attachments"
    )
    statements = list(reset.statements)
    joined = "\n".join(statements)

    # Intentional reset so no committed turn can reference a dropped entity.
    assert any(
        statement.strip().startswith("DELETE FROM web_conversations") for statement in statements
    )
    # Drop the three superseded tables in FK-safe order (children before the
    # unified table is recreated).
    assert "DROP TABLE IF EXISTS web_conversation_attachment_chunks" in joined
    assert "DROP TABLE IF EXISTS web_conversation_images" in joined
    assert "DROP TABLE IF EXISTS web_conversation_attachments" in joined
    drop_old = joined.index("DROP TABLE IF EXISTS web_conversation_attachments")
    create_new = joined.index("CREATE TABLE IF NOT EXISTS web_conversation_attachments")
    assert drop_old < create_new
    # No compatibility view or renamed-column bridge.
    assert "CREATE VIEW" not in joined
    assert "CREATE OR REPLACE VIEW" not in joined


def test_canonical_answer_snapshot_migration_resets_incompatible_rows() -> None:
    migration = WEB_CONVERSATION_MIGRATIONS[-1]

    assert migration.version == "0003_canonical_answer_sources"
    assert migration.statements == ("DELETE FROM web_conversations",)


def test_migrations_touch_only_web_conversation_storage() -> None:
    sql = _migration_sql()
    for foreign in (
        "dlightrag_doc_metadata",
        "dlightrag_schema_migrations",
        "lightrag_doc_chunks",
        "lightrag_doc_full",
        "lightrag_graph_nodes",
        "lightrag_vdb_chunks",
        "ingest_jobs",
        "dlightrag_checkpoints",
    ):
        assert foreign not in sql, foreign
    # Every DDL object this scope owns is a web_conversation* entity.
    assert "web_conversations" in sql
    assert "web_conversation_turns" in sql
    assert "web_conversation_attachments" in sql


# --- Conversation lifecycle -------------------------------------------------


async def test_delete_all_conversations_is_principal_scoped() -> None:
    conn = FakeConnection()
    conn.fetchrow_result = {"deleted_count": 3}
    store = make_store(conn)

    deleted_count = await store.delete_all_conversations("alice")

    assert deleted_count == 3
    assert conn.calls[-1][1] == ("alice",)


async def test_list_filters_expired_rows_without_request_path_write() -> None:
    conn = FakeConnection()
    conn.fetch_result = []
    store = make_store(conn)

    assert await store.list_conversations("principal-a", ttl_days=30) == []

    assert len(conn.calls) == 1
    query, args = conn.calls[0]
    assert query.lstrip().startswith("SELECT")
    assert "updated_at >= NOW()" in query
    assert args == ("principal-a", 30)


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
    assert snapshot.history == ()
    assert conn.transactions == [{"isolation": "repeatable_read", "readonly": True}]


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


# --- Raw attachment insert / fetch ------------------------------------------


async def test_commit_turn_inserts_ordered_raw_attachments() -> None:
    conn = FakeConnection()
    conn.fetchrow_results = [
        None,
        {"conversation_id": "c1", "title": "Question", "content_revision": 1},
        {"turn_id": "t1", "turn_number": 1},
    ]
    store = make_store(conn)
    attachment = PendingConversationAttachment(
        attachment_id="a1",
        ordinal=0,
        filename="report.pdf",
        mime_type="application/pdf",
        suffix=".pdf",
        attachment_bytes=b"PDF-BYTES",
        content_sha256="abc123",
    )

    result = await store.commit_turn(
        principal_id="p1",
        conversation_id="c1",
        expected_revision=0,
        user_text="Question",
        assistant_text="Answer",
        answer_sources={},
        queried_workspaces=["research"],
        attachments=[attachment],
        max_turns=100,
        submission_id="00000000-0000-4000-8000-000000000003",
        ttl_days=30,
    )

    assert result.saved is True
    assert result.current_attachment_ids == ("a1",)
    inserts = [call for call in conn.calls if "INSERT INTO web_conversation_attachments" in call[0]]
    assert len(inserts) == 1
    assert inserts[0][1] == (
        "a1",
        "p1",
        "c1",
        "t1",
        0,
        "report.pdf",
        "application/pdf",
        ".pdf",
        b"PDF-BYTES",
        9,
        "abc123",
    )


async def test_get_attachment_is_scoped_and_ttl_guarded() -> None:
    conn = FakeConnection()
    conn.fetchrow_result = {
        "attachment_id": "a1",
        "filename": "report.pdf",
        "mime_type": "application/pdf",
        "suffix": ".pdf",
        "attachment_bytes": b"PDF",
        "content_sha256": "abc123",
    }
    store = make_store(conn)

    stored = await store.get_attachment("p1", "c1", "a1", ttl_days=30)

    assert stored == StoredConversationAttachment(
        attachment_id="a1",
        filename="report.pdf",
        mime_type="application/pdf",
        suffix=".pdf",
        attachment_bytes=b"PDF",
        content_sha256="abc123",
    )
    query, args = conn.calls[-1]
    assert "a.principal_id = $1" in query
    assert "a.conversation_id = $2" in query
    assert "a.attachment_id = $3" in query
    assert "c.updated_at >= NOW()" in query
    assert args == ("p1", "c1", "a1", 30)


async def test_fetch_attachments_by_ids_returns_owned_bytes() -> None:
    conn = FakeConnection()
    conn.fetch_result = [
        {
            "attachment_id": "a1",
            "filename": "report.pdf",
            "mime_type": "application/pdf",
            "suffix": ".pdf",
            "attachment_bytes": b"PDF",
            "content_sha256": "abc123",
        },
    ]
    store = make_store(conn)

    fetched = await store.fetch_attachments_by_ids("p1", "c1", ["a1"], ttl_days=30)

    assert len(fetched) == 1
    assert fetched[0].attachment_bytes == b"PDF"
    query, args = conn.calls[-1]
    assert "= ANY($3::uuid[])" in query
    assert args == ("p1", "c1", ["a1"], 30)


async def test_fetch_attachments_by_ids_empty_short_circuits() -> None:
    conn = FakeConnection()
    store = make_store(conn)

    assert await store.fetch_attachments_by_ids("p1", "c1", [], ttl_days=30) == []
    assert conn.calls == []


async def test_snapshot_history_carries_compact_attachment_manifest() -> None:
    now = datetime.datetime(2026, 7, 12, tzinfo=datetime.UTC)
    conn = FakeConnection()
    conn.fetchrow_result = {
        "principal_id": "p1",
        "conversation_id": "c1",
        "content_revision": 1,
        "title": "Conversation",
        "created_at": now,
        "updated_at": now,
    }
    conn.fetch_result = [
        {
            "turn_id": "t1",
            "turn_number": 1,
            "submission_id": "s1",
            "user_text": "Question",
            "assistant_text": "Answer",
            "answer_sources": {},
            "queried_workspaces": [],
            "created_at": now,
            "attachments": [
                {
                    "attachment_id": "a1",
                    "ordinal": 0,
                    "filename": "report.pdf",
                    "mime_type": "application/pdf",
                    "byte_size": 9,
                    "content_sha256": "abc123",
                }
            ],
        }
    ]
    store = make_store(conn)

    snapshot = await store.snapshot("p1", "c1", ttl_days=30)

    assert snapshot is not None
    assert len(snapshot.history) == 1
    manifest = snapshot.history[0]["attachments"]
    assert manifest == [
        {
            "attachment_id": "a1",
            "ordinal": 0,
            "filename": "report.pdf",
            "mime_type": "application/pdf",
            "byte_size": 9,
            "content_sha256": "abc123",
        }
    ]
    history_query = conn.calls[-1][0]
    assert "web_conversation_attachments" in history_query
    assert "parse_summary" not in history_query


# --- Idempotent committed-turn semantics ------------------------------------


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
        queried_workspaces=("default",),
    )
    statements = "\n".join(query for query, _ in conn.calls)
    assert "content_revision = $3" in statements
    assert "DELETE FROM web_conversation_turns" in statements
    trim_query, trim_args = conn.calls[-1]
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
        attachments=[],
        max_turns=100,
        submission_id="00000000-0000-4000-8000-000000000002",
        ttl_days=30,
    )

    assert result == CommitTurnResult(False, "conversation_changed", None, None)


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
        "attachments": [
            {
                "attachment_id": "a1",
                "ordinal": 0,
                "filename": "report.pdf",
                "mime_type": "application/pdf",
                "byte_size": 9,
                "content_sha256": "abc123",
            }
        ],
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
        attachments=[],
        max_turns=100,
        ttl_days=30,
    )

    assert result.saved is True
    assert result.replayed is True
    assert result.assistant_text == "Stored answer"
    assert result.current_attachment_ids == ("a1",)
    assert not any("INSERT INTO web_conversation_turns" in query for query, _args in conn.calls)


async def test_find_committed_turn_uses_single_attempt_pool_path(monkeypatch) -> None:
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


# --- Dataclass contracts ----------------------------------------------------


def test_records_are_frozen() -> None:
    now = datetime.datetime(2026, 7, 12, tzinfo=datetime.UTC)
    snapshot = ConversationSnapshot("p1", "c1", 0, None, now, now, ())
    with pytest.raises(FrozenInstanceError):
        snapshot.title = "changed"  # type: ignore[misc]

    attachment = PendingConversationAttachment(
        attachment_id="a1",
        ordinal=0,
        filename="report.pdf",
        mime_type="application/pdf",
        suffix=".pdf",
        attachment_bytes=b"PDF",
        content_sha256="abc123",
    )
    assert attachment.byte_size == 3
    with pytest.raises(FrozenInstanceError):
        attachment.filename = "other.pdf"  # type: ignore[misc]
