# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Integration tests for unified Web conversation attachment storage on PG 18.

Exercises the raw-attachment schema end-to-end against a live PostgreSQL 18
instance (asyncpg): the reset migration, ordered raw-attachment inserts through a
committed turn, the compact history JSONB manifest, principal/conversation/turn
scoped byte reads, idempotent committed-turn replay, and the ON DELETE CASCADE
lifecycle.

Every test runs inside a throwaway database created and dropped per test, so no
shared development data is ever mutated and the intentional reset migration
(``DELETE FROM web_conversations`` + dropping the superseded tables) can be
proven without touching the developer's ``dlightrag`` database.

Requires a running PostgreSQL instance (localhost:5432, dlightrag/dlightrag).
Skipped automatically if PostgreSQL is not available.
"""

import hashlib
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

import pytest

from dlightrag.storage.migrations import apply_migrations
from dlightrag.storage.web_conversations import (
    WEB_CONVERSATION_MIGRATIONS,
    PendingConversationAttachment,
    PGWebConversationStore,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
]

_PG_CONN_KWARGS: dict[str, Any] = dict(
    host="localhost",
    port=5432,
    user="dlightrag",
    password="dlightrag",
    database="dlightrag",
)

_TTL_DAYS = 30
_MAX_TURNS = 100


async def _pg_available() -> bool:
    try:
        import asyncpg

        conn = await asyncpg.connect(**_PG_CONN_KWARGS)
        await conn.fetchval("SELECT 1")
        await conn.close()
        return True
    except Exception:
        return False


@pytest.fixture
async def temp_pool() -> AsyncIterator[Any]:
    """Provision an isolated throwaway database and yield a pool bound to it."""
    if not await _pg_available():
        pytest.skip("PostgreSQL not available")

    import asyncpg

    db_name = f"dlightrag_t8_{uuid4().hex[:12]}"
    admin = await asyncpg.connect(**_PG_CONN_KWARGS)
    try:
        await admin.execute(f'CREATE DATABASE "{db_name}"')
    finally:
        await admin.close()

    pool = await asyncpg.create_pool(
        **{**_PG_CONN_KWARGS, "database": db_name}, min_size=1, max_size=2
    )
    try:
        yield pool
    finally:
        await pool.close()
        admin = await asyncpg.connect(**_PG_CONN_KWARGS)
        try:
            await admin.execute(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)')
        finally:
            await admin.close()


def _attachment(*, ordinal: int, filename: str, payload: bytes) -> PendingConversationAttachment:
    suffix = filename[filename.rfind(".") :] if "." in filename else ""
    return PendingConversationAttachment(
        attachment_id=str(uuid4()),
        ordinal=ordinal,
        filename=filename,
        mime_type="application/pdf",
        suffix=suffix,
        attachment_bytes=payload,
        content_sha256=hashlib.sha256(payload).hexdigest(),
    )


async def _table_exists(pool: Any, name: str) -> bool:
    async with pool.acquire() as conn:
        return bool(await conn.fetchval("SELECT to_regclass($1) IS NOT NULL", name))


async def _column_names(pool: Any, table: str) -> set[str]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT column_name FROM information_schema.columns WHERE table_name = $1",
            table,
        )
    return {row["column_name"] for row in rows}


async def test_reset_migration_yields_unified_schema_only(temp_pool: Any) -> None:
    store = PGWebConversationStore(pool=temp_pool)
    await store.initialize()

    assert await _table_exists(temp_pool, "web_conversations")
    assert await _table_exists(temp_pool, "web_conversation_turns")
    assert await _table_exists(temp_pool, "web_conversation_attachments")
    # Superseded tables never exist after the reset migration.
    assert not await _table_exists(temp_pool, "web_conversation_images")
    assert not await _table_exists(temp_pool, "web_conversation_attachment_chunks")

    columns = await _column_names(temp_pool, "web_conversation_attachments")
    assert columns == {
        "attachment_id",
        "principal_id",
        "conversation_id",
        "turn_id",
        "ordinal",
        "filename",
        "mime_type",
        "suffix",
        "attachment_bytes",
        "byte_size",
        "content_sha256",
        "created_at",
    }


async def test_reset_migration_drops_legacy_tables_and_wipes_rows(temp_pool: Any) -> None:
    """Simulate an existing deployment and prove the reset transition on live PG."""
    async with temp_pool.acquire() as conn:
        # Stand up the conversations/turns tables via migration 0001 only.
        await apply_migrations(
            conn,
            scope="web_conversations",
            migrations=WEB_CONVERSATION_MIGRATIONS[:1],
        )
        # Recreate the legacy child tables an older deployment would have.
        await conn.execute(
            """
            CREATE TABLE web_conversation_images (
                image_id UUID NOT NULL,
                principal_id TEXT NOT NULL,
                conversation_id UUID NOT NULL,
                turn_id UUID NOT NULL,
                PRIMARY KEY (principal_id, conversation_id, image_id)
            )
            """
        )
        await conn.execute(
            """
            CREATE TABLE web_conversation_attachment_chunks (
                principal_id TEXT NOT NULL,
                conversation_id UUID NOT NULL,
                chunk_id TEXT NOT NULL,
                PRIMARY KEY (principal_id, conversation_id, chunk_id)
            )
            """
        )
        conversation_id = str(uuid4())
        await conn.execute(
            "INSERT INTO web_conversations (principal_id, conversation_id) VALUES ($1, $2::uuid)",
            "legacy",
            conversation_id,
        )
        assert (
            await conn.fetchval(
                "SELECT COUNT(*) FROM web_conversations WHERE principal_id='legacy'"
            )
            == 1
        )

        # Apply the full ledger: only migration 0002 is new and must transition.
        await apply_migrations(
            conn,
            scope="web_conversations",
            migrations=WEB_CONVERSATION_MIGRATIONS,
        )

    assert not await _table_exists(temp_pool, "web_conversation_images")
    assert not await _table_exists(temp_pool, "web_conversation_attachment_chunks")
    assert await _table_exists(temp_pool, "web_conversation_attachments")
    async with temp_pool.acquire() as conn:
        # The reset deleted the pre-existing conversation row.
        assert await conn.fetchval("SELECT COUNT(*) FROM web_conversations") == 0


async def test_reset_migration_leaves_unrelated_tables_intact(temp_pool: Any) -> None:
    async with temp_pool.acquire() as conn:
        await conn.execute("CREATE TABLE workspace_documents (id TEXT PRIMARY KEY)")
        await conn.execute("INSERT INTO workspace_documents (id) VALUES ('doc-1')")

    store = PGWebConversationStore(pool=temp_pool)
    await store.initialize()

    assert await _table_exists(temp_pool, "workspace_documents")
    async with temp_pool.acquire() as conn:
        assert await conn.fetchval("SELECT COUNT(*) FROM workspace_documents") == 1


async def test_commit_turn_persists_ordered_raw_attachments(temp_pool: Any) -> None:
    store = PGWebConversationStore(pool=temp_pool)
    await store.initialize()
    created = await store.create_conversation("alice")
    conversation_id = str(created["conversation_id"])

    first = _attachment(ordinal=0, filename="a.pdf", payload=b"AAA")
    second = _attachment(ordinal=1, filename="b.pdf", payload=b"BBBB")
    result = await store.commit_turn(
        principal_id="alice",
        conversation_id=conversation_id,
        submission_id=str(uuid4()),
        expected_revision=0,
        user_text="What do these say?",
        assistant_text="They summarize quarterly results.",
        answer_sources={"sources": []},
        queried_workspaces=["research"],
        attachments=[first, second],
        max_turns=_MAX_TURNS,
        ttl_days=_TTL_DAYS,
    )

    assert result.saved is True
    assert result.current_attachment_ids == (first.attachment_id, second.attachment_id)

    snapshot = await store.snapshot("alice", conversation_id, ttl_days=_TTL_DAYS)
    assert snapshot is not None
    assert len(snapshot.history) == 1
    manifest = snapshot.history[0]["attachments"]
    assert [entry["ordinal"] for entry in manifest] == [0, 1]
    assert [entry["filename"] for entry in manifest] == ["a.pdf", "b.pdf"]
    assert [entry["byte_size"] for entry in manifest] == [3, 4]
    assert manifest[0]["content_sha256"] == first.content_sha256
    assert all("parse_summary" not in entry for entry in manifest)

    # Authorized byte reads: single and batch, both principal/conversation scoped.
    fetched = await store.get_attachment(
        "alice", conversation_id, first.attachment_id, ttl_days=_TTL_DAYS
    )
    assert fetched is not None
    assert fetched.attachment_bytes == b"AAA"
    assert fetched.suffix == ".pdf"

    batch = await store.fetch_attachments_by_ids(
        "alice",
        conversation_id,
        [first.attachment_id, second.attachment_id],
        ttl_days=_TTL_DAYS,
    )
    assert {item.attachment_bytes for item in batch} == {b"AAA", b"BBBB"}

    # A different principal cannot read another principal's attachment bytes.
    assert (
        await store.get_attachment(
            "mallory", conversation_id, first.attachment_id, ttl_days=_TTL_DAYS
        )
        is None
    )


async def test_committed_turn_replay_is_idempotent(temp_pool: Any) -> None:
    store = PGWebConversationStore(pool=temp_pool)
    await store.initialize()
    created = await store.create_conversation("alice")
    conversation_id = str(created["conversation_id"])
    submission_id = str(uuid4())
    attachment = _attachment(ordinal=0, filename="a.pdf", payload=b"AAA")

    async def _commit(assistant_text: str) -> Any:
        return await store.commit_turn(
            principal_id="alice",
            conversation_id=conversation_id,
            submission_id=submission_id,
            expected_revision=0,
            user_text="Question",
            assistant_text=assistant_text,
            answer_sources={"sources": []},
            queried_workspaces=[],
            attachments=[attachment],
            max_turns=_MAX_TURNS,
            ttl_days=_TTL_DAYS,
        )

    first = await _commit("First answer")
    assert first.saved is True and first.replayed is False

    replay = await _commit("Different retry answer")
    assert replay.saved is True
    assert replay.replayed is True
    assert replay.assistant_text == "First answer"
    assert replay.current_attachment_ids == (attachment.attachment_id,)

    found = await store.find_committed_turn(
        "alice", conversation_id, submission_id, ttl_days=_TTL_DAYS
    )
    assert found is not None
    assert found.assistant_text == "First answer"

    # Exactly one attachment row exists despite the replay.
    async with temp_pool.acquire() as conn:
        assert (
            await conn.fetchval(
                "SELECT COUNT(*) FROM web_conversation_attachments WHERE principal_id='alice'"
            )
            == 1
        )


async def test_deleting_conversation_cascades_attachments(temp_pool: Any) -> None:
    store = PGWebConversationStore(pool=temp_pool)
    await store.initialize()
    created = await store.create_conversation("alice")
    conversation_id = str(created["conversation_id"])
    attachment = _attachment(ordinal=0, filename="a.pdf", payload=b"AAA")
    await store.commit_turn(
        principal_id="alice",
        conversation_id=conversation_id,
        submission_id=str(uuid4()),
        expected_revision=0,
        user_text="Question",
        assistant_text="Answer",
        answer_sources={},
        queried_workspaces=[],
        attachments=[attachment],
        max_turns=_MAX_TURNS,
        ttl_days=_TTL_DAYS,
    )

    async with temp_pool.acquire() as conn:
        assert await conn.fetchval("SELECT COUNT(*) FROM web_conversation_attachments") == 1

    assert await store.delete_conversation("alice", conversation_id, ttl_days=_TTL_DAYS) is True

    async with temp_pool.acquire() as conn:
        assert await conn.fetchval("SELECT COUNT(*) FROM web_conversation_attachments") == 0
        assert await conn.fetchval("SELECT COUNT(*) FROM web_conversation_turns") == 0


async def test_all_migration_sql_prepares_on_live_pg(temp_pool: Any) -> None:
    """Every runtime query prepares (parse/plan) against real PG 18."""
    from dlightrag.storage import web_conversations as module

    store = PGWebConversationStore(pool=temp_pool)
    await store.initialize()

    queries = [
        value
        for name, value in vars(module).items()
        if name.startswith("_")
        and isinstance(value, str)
        and value.strip().upper().startswith(("SELECT", "INSERT", "UPDATE", "DELETE", "WITH"))
    ]
    assert queries, "expected runtime SQL constants to prepare"
    async with temp_pool.acquire() as conn:
        for sql in queries:
            stmt = await conn.prepare(sql)
            assert stmt is not None
