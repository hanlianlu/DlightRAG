# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Integration tests for Web conversation document-attachment storage.

Exercises the attachment SQL shipped in migrations 0003/0004 end-to-end against
a live PostgreSQL 18 instance (asyncpg): migration application, the attachment
insert, the history JSONB aggregation, the content-addressed parse cache, the
principal-scoped document reads, and the ON DELETE CASCADE lifecycle.

Requires a running PostgreSQL instance (localhost:5432, dlightrag/dlightrag).
Skipped automatically if PostgreSQL is not available.
"""

import hashlib
from dataclasses import replace
from typing import Any
from uuid import uuid4

import pytest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.asyncio,
]

_PG_CONN_KWARGS = dict(
    host="localhost",
    port=5432,
    user="dlightrag",
    password="dlightrag",
    database="dlightrag",
)

_TTL_DAYS = 30
_MAX_TURNS = 100


async def _pg_available() -> bool:
    """Check if PostgreSQL is available."""
    try:
        import asyncpg

        conn = await asyncpg.connect(
            host=str(_PG_CONN_KWARGS["host"]),
            port=int(_PG_CONN_KWARGS["port"]),
            user=str(_PG_CONN_KWARGS["user"]),
            password=str(_PG_CONN_KWARGS["password"]),
            database=str(_PG_CONN_KWARGS["database"]),
        )
        await conn.fetchval("SELECT 1")
        await conn.close()
        return True
    except Exception:
        return False


@pytest.fixture
async def pg_check() -> None:
    """Skip test if PostgreSQL is not available."""
    if not await _pg_available():
        pytest.skip("PostgreSQL not available")


async def _open_pool() -> Any:
    import asyncpg

    return await asyncpg.create_pool(
        host=str(_PG_CONN_KWARGS["host"]),
        port=int(_PG_CONN_KWARGS["port"]),
        user=str(_PG_CONN_KWARGS["user"]),
        password=str(_PG_CONN_KWARGS["password"]),
        database=str(_PG_CONN_KWARGS["database"]),
        min_size=1,
        max_size=2,
    )


async def _count_attachment_rows(pool: Any, principal_id: str) -> int:
    async with pool.acquire() as conn:
        return int(
            await conn.fetchval(
                "SELECT COUNT(*) FROM web_conversation_attachments WHERE principal_id = $1",
                principal_id,
            )
        )


async def _count_chunk_rows(pool: Any, principal_id: str) -> int:
    async with pool.acquire() as conn:
        return int(
            await conn.fetchval(
                "SELECT COUNT(*) FROM web_conversation_attachment_chunks WHERE principal_id = $1",
                principal_id,
            )
        )


async def _delete_principal(pool: Any, principal_id: str) -> None:
    async with pool.acquire() as conn:
        await conn.execute("DELETE FROM web_conversations WHERE principal_id = $1", principal_id)
        # Belt-and-suspenders in case a prior run left orphan cache rows.
        await conn.execute(
            "DELETE FROM web_conversation_attachment_chunks WHERE principal_id = $1",
            principal_id,
        )


@pytest.mark.usefixtures("pg_check")
async def test_web_conversation_attachment_storage_round_trip() -> None:
    """Attachment insert/select/cache/cascade round-trips against live pg18."""
    from dlightrag.core.request.attachments import (
        AttachmentCacheKey,
        AttachmentContextChunk,
        ParsedAttachmentBundle,
    )
    from dlightrag.storage.web_conversations import (
        PendingConversationAttachment,
        PGWebConversationStore,
    )

    principal_id = f"itest-attach-{uuid4()}"
    other_principal_id = f"itest-attach-other-{uuid4()}"

    pool = await _open_pool()
    try:
        # 1) Initialize store + apply migrations (0003 + 0004 must succeed).
        store = PGWebConversationStore(pool=pool)
        await store.initialize()

        await _delete_principal(pool, principal_id)
        await _delete_principal(pool, other_principal_id)

        # 2) Create a conversation and commit a turn with one attachment.
        conversation = await store.create_conversation(principal_id)
        conversation_id = str(conversation["conversation_id"])

        document_bytes = b"%PDF-1.7 integration attachment payload\n"
        content_sha256 = hashlib.sha256(document_bytes).hexdigest()
        attachment_id = str(uuid4())
        filename = "report.pdf"

        commit = await store.commit_turn(
            principal_id=principal_id,
            conversation_id=conversation_id,
            submission_id=str(uuid4()),
            expected_revision=0,
            user_text="Summarize the attached report.",
            assistant_text="Here is the summary of report.pdf.",
            answer_sources={},
            queried_workspaces=[],
            images=[],
            attachments=[
                PendingConversationAttachment(
                    attachment_id=attachment_id,
                    ordinal=0,
                    filename=filename,
                    mime_type="application/pdf",
                    suffix=".pdf",
                    attachment_bytes=document_bytes,
                    content_sha256=content_sha256,
                    parse_summary="1 page, 2 chunks",
                    parse_catalog={"pages": 1},
                )
            ],
            max_turns=_MAX_TURNS,
            ttl_days=_TTL_DAYS,
        )
        assert commit.saved is True
        assert commit.current_attachment_ids == (attachment_id,)

        # 3) History projects the attachment via the JSONB aggregation SQL.
        snapshot = await store.snapshot(
            principal_id, conversation_id, ttl_days=_TTL_DAYS, max_turns=_MAX_TURNS
        )
        assert snapshot is not None
        assert len(snapshot.history) == 1
        history_attachments = snapshot.history[0]["attachments"]
        assert len(history_attachments) == 1
        projected = history_attachments[0]
        assert projected["attachment_id"] == attachment_id
        assert projected["ordinal"] == 0
        assert projected["filename"] == filename
        assert projected["mime_type"] == "application/pdf"
        assert projected["byte_size"] == len(document_bytes)
        assert projected["content_sha256"] == content_sha256
        assert projected["parse_summary"] == "1 page, 2 chunks"

        # 4) Parse-cache round-trip (content-addressed): a text chunk and a
        #    visual chunk (with image bytes) materialized then loaded back.
        parser_signature = "legacy@v1"
        chunk_signature = "chunker@v1"
        image_bytes = b"\x89PNG\r\n\x1a\n visual-chunk-bytes"
        source_bundle = ParsedAttachmentBundle(
            parser_signature=parser_signature,
            chunks=[
                AttachmentContextChunk(
                    chunk_id="src-text-0",
                    attachment_id=attachment_id,
                    filename=filename,
                    chunk_index=0,
                    content="The quarterly results improved.",
                    token_estimate=7,
                    page_idx=0,
                    metadata={"kind": "text"},
                    embedding_signature="embed@v1",
                    embedding_vector=[1.0, 0.0],
                ),
                AttachmentContextChunk(
                    chunk_id="src-visual-1",
                    attachment_id=attachment_id,
                    filename=filename,
                    chunk_index=1,
                    content="Figure 1: revenue chart.",
                    token_estimate=5,
                    page_idx=0,
                    sidecar_type="image",
                    image_bytes=image_bytes,
                    image_mime_type="image/png",
                    metadata={"kind": "visual"},
                    embedding_signature="embed@v1",
                    embedding_vector=[0.0, 1.0],
                ),
            ],
        )
        assert (
            await store.materialize_attachment_chunks(
                principal_id,
                conversation_id,
                content_sha256=content_sha256,
                parser_signature=parser_signature,
                chunk_signature=chunk_signature,
                bundle=source_bundle,
                ttl_days=_TTL_DAYS,
            )
            is True
        )

        # Cache HIT: same content+parser+chunk signature, but a *new*
        # attachment id — chunk ids must be remapped to the requested id.
        reused_attachment_id = str(uuid4())
        loaded = await store.load_attachment_chunks(
            principal_id,
            conversation_id,
            reused_attachment_id,
            filename,
            content_sha256=content_sha256,
            parser_signature=parser_signature,
            chunk_signature=chunk_signature,
            ttl_days=_TTL_DAYS,
        )
        assert loaded is not None
        assert loaded.parser_signature == parser_signature
        assert len(loaded.chunks) == 2
        text_chunk, visual_chunk = loaded.chunks
        assert text_chunk.chunk_id == f"att:{reused_attachment_id}:0000"
        assert text_chunk.attachment_id == reused_attachment_id
        assert text_chunk.content == "The quarterly results improved."
        assert text_chunk.image_bytes is None
        assert text_chunk.cache_key == AttachmentCacheKey(
            content_sha256,
            parser_signature,
            chunk_signature,
            "src-text-0",
        )
        assert text_chunk.embedding_signature is None
        assert text_chunk.embedding_vector is None
        assert visual_chunk.chunk_id == f"att:{reused_attachment_id}:0001"
        assert visual_chunk.image_bytes == image_bytes
        assert visual_chunk.image_mime_type == "image/png"
        assert visual_chunk.sidecar_type == "image"

        # Vector reads use the stable full cache key, decode one bounded page,
        # and preserve duplicate references in caller global order.
        assert text_chunk.cache_key is not None
        duplicate_pages = [
            page
            async for page in store.aiter_attachment_vectors(
                principal_id,
                conversation_id,
                [(8, text_chunk.cache_key), (3, text_chunk.cache_key)],
                ttl_days=_TTL_DAYS,
                page_size=1,
            )
        ]
        assert [[row.global_order for row in page] for page in duplicate_pages] == [[3], [8]]
        assert [page[0].embedding_signature for page in duplicate_pages] == [
            "embed@v1",
            "embed@v1",
        ]
        assert [page[0].embedding_vector for page in duplicate_pages] == [
            [1.0, 0.0],
            [1.0, 0.0],
        ]

        # Cache MISS: a different chunk signature returns nothing.
        missed = await store.load_attachment_chunks(
            principal_id,
            conversation_id,
            attachment_id,
            filename,
            content_sha256=content_sha256,
            parser_signature=parser_signature,
            chunk_signature="chunker@v2",
            ttl_days=_TTL_DAYS,
        )
        assert missed is None

        # The same stored cache id can coexist under old signatures; refreshing
        # the current generation must not alter the stale generation.
        stale_parser_signature = "legacy@old"
        stale_chunk_signature = "chunker@old"
        stale_bundle = ParsedAttachmentBundle(
            parser_signature=stale_parser_signature,
            chunks=[
                replace(
                    source_bundle.chunks[0],
                    embedding_signature="embed@old",
                    embedding_vector=[0.5, 0.5],
                )
            ],
        )
        assert (
            await store.materialize_attachment_chunks(
                principal_id,
                conversation_id,
                content_sha256=content_sha256,
                parser_signature=stale_parser_signature,
                chunk_signature=stale_chunk_signature,
                bundle=stale_bundle,
                ttl_days=_TTL_DAYS,
            )
            is True
        )
        refreshed = replace(
            text_chunk,
            embedding_signature="embed@v2",
            embedding_vector=[0.25, 0.75],
        )
        assert (
            await store.aupdate_attachment_chunk_vectors(
                principal_id,
                conversation_id,
                [refreshed],
                ttl_days=_TTL_DAYS,
            )
            is True
        )
        stale_key = AttachmentCacheKey(
            content_sha256,
            stale_parser_signature,
            stale_chunk_signature,
            "src-text-0",
        )
        generation_pages = [
            page
            async for page in store.aiter_attachment_vectors(
                principal_id,
                conversation_id,
                [(0, stale_key), (1, text_chunk.cache_key)],
                ttl_days=_TTL_DAYS,
                page_size=2,
            )
        ]
        assert [row.embedding_signature for row in generation_pages[0]] == [
            "embed@old",
            "embed@v2",
        ]

        # Identical key material cannot be read through another principal or
        # another conversation owned by the same principal.
        other_conversation = await store.create_conversation(principal_id)
        other_conversation_id = str(other_conversation["conversation_id"])
        assert (
            await store.load_attachment_chunks(
                other_principal_id,
                conversation_id,
                attachment_id,
                filename,
                content_sha256=content_sha256,
                parser_signature=parser_signature,
                chunk_signature=chunk_signature,
                ttl_days=_TTL_DAYS,
            )
            is None
        )
        assert (
            await store.load_attachment_chunks(
                principal_id,
                other_conversation_id,
                attachment_id,
                filename,
                content_sha256=content_sha256,
                parser_signature=parser_signature,
                chunk_signature=chunk_signature,
                ttl_days=_TTL_DAYS,
            )
            is None
        )

        # 5) Principal-scoped document reads.
        fetched = await store.fetch_documents_by_ids(
            principal_id, conversation_id, [attachment_id], ttl_days=_TTL_DAYS
        )
        assert len(fetched) == 1
        assert fetched[0].attachment_id == attachment_id
        assert fetched[0].attachment_bytes == document_bytes

        owned = await store.get_attachment(
            principal_id, conversation_id, attachment_id, ttl_days=_TTL_DAYS
        )
        assert owned is not None
        assert owned.document_bytes == document_bytes
        assert owned.content_sha256 == content_sha256

        # Principal isolation: a different principal cannot read the bytes.
        foreign = await store.get_attachment(
            other_principal_id, conversation_id, attachment_id, ttl_days=_TTL_DAYS
        )
        assert foreign is None

        # 6) Cascade cleanup: deleting the conversation must reclaim BOTH the
        #    attachment rows and the parse-cache rows (migration 0004 FK).
        assert await _count_attachment_rows(pool, principal_id) == 1
        assert await _count_chunk_rows(pool, principal_id) == 3

        deleted = await store.delete_conversation(principal_id, conversation_id, ttl_days=_TTL_DAYS)
        assert deleted is True

        assert await _count_attachment_rows(pool, principal_id) == 0
        assert await _count_chunk_rows(pool, principal_id) == 0
    finally:
        # 7) Final cleanup: remove any residue for our unique test principals.
        await _delete_principal(pool, principal_id)
        await _delete_principal(pool, other_principal_id)
        await pool.close()


@pytest.mark.usefixtures("pg_check")
async def test_attachment_vectors_require_exact_principal_and_conversation_scope() -> None:
    from dlightrag.core.request.attachments import (
        AttachmentCacheKey,
        AttachmentContextChunk,
        ParsedAttachmentBundle,
    )
    from dlightrag.storage.web_conversations import PGWebConversationStore

    owner_principal_id = f"itest-vector-owner-{uuid4()}"
    foreign_principal_id = f"itest-vector-foreign-{uuid4()}"
    cache_key = AttachmentCacheKey(
        "shared-content",
        "shared-parser",
        "shared-chunker",
        "shared-cache-chunk",
    )
    owner_chunk = AttachmentContextChunk(
        chunk_id="owner-public-chunk",
        attachment_id="owner-attachment",
        filename="owner.pdf",
        chunk_index=0,
        content="owner scoped vector",
        cache_key=cache_key,
        embedding_signature="owner-embedding",
        embedding_vector=[1.0, 0.0],
    )
    foreign_chunk = replace(
        owner_chunk,
        chunk_id="foreign-public-chunk",
        attachment_id="foreign-attachment",
        filename="foreign.pdf",
        content="foreign scoped vector",
        embedding_signature="foreign-embedding",
        embedding_vector=[0.0, 1.0],
    )

    pool = await _open_pool()
    try:
        store = PGWebConversationStore(pool=pool)
        await store.initialize()
        await _delete_principal(pool, owner_principal_id)
        await _delete_principal(pool, foreign_principal_id)

        owner_conversation = await store.create_conversation(owner_principal_id)
        owner_conversation_id = str(owner_conversation["conversation_id"])
        wrong_conversation = await store.create_conversation(owner_principal_id)
        wrong_conversation_id = str(wrong_conversation["conversation_id"])
        foreign_conversation = await store.create_conversation(foreign_principal_id)
        foreign_conversation_id = str(foreign_conversation["conversation_id"])

        for principal_id, conversation_id, chunk in (
            (owner_principal_id, owner_conversation_id, owner_chunk),
            (foreign_principal_id, foreign_conversation_id, foreign_chunk),
        ):
            assert (
                await store.materialize_attachment_chunks(
                    principal_id,
                    conversation_id,
                    content_sha256=cache_key.content_sha256,
                    parser_signature=cache_key.parser_signature,
                    chunk_signature=cache_key.chunk_signature,
                    bundle=ParsedAttachmentBundle(
                        chunks=[chunk],
                        parser_signature=cache_key.parser_signature,
                    ),
                    ttl_days=_TTL_DAYS,
                )
                is True
            )

        async with pool.acquire() as conn:
            scoped_rows = await conn.fetch(
                "SELECT principal_id, conversation_id::text AS conversation_id, "
                "content_sha256, parser_signature, chunk_signature, chunk_id "
                "FROM web_conversation_attachment_chunks "
                "WHERE content_sha256 = $1 AND parser_signature = $2 "
                "AND chunk_signature = $3 AND chunk_id = $4 "
                "AND principal_id = ANY($5::text[]) "
                "ORDER BY principal_id",
                cache_key.content_sha256,
                cache_key.parser_signature,
                cache_key.chunk_signature,
                cache_key.cache_chunk_id,
                [owner_principal_id, foreign_principal_id],
            )
        assert {(str(row["principal_id"]), str(row["conversation_id"])) for row in scoped_rows} == {
            (owner_principal_id, owner_conversation_id),
            (foreign_principal_id, foreign_conversation_id),
        }
        assert {
            (
                str(row["content_sha256"]),
                str(row["parser_signature"]),
                str(row["chunk_signature"]),
                str(row["chunk_id"]),
            )
            for row in scoped_rows
        } == {
            (
                cache_key.content_sha256,
                cache_key.parser_signature,
                cache_key.chunk_signature,
                cache_key.cache_chunk_id,
            )
        }

        foreign_principal_pages = [
            page
            async for page in store.aiter_attachment_vectors(
                foreign_principal_id,
                owner_conversation_id,
                [(0, cache_key)],
                ttl_days=_TTL_DAYS,
                page_size=1,
            )
        ]
        wrong_conversation_pages = [
            page
            async for page in store.aiter_attachment_vectors(
                owner_principal_id,
                wrong_conversation_id,
                [(0, cache_key)],
                ttl_days=_TTL_DAYS,
                page_size=1,
            )
        ]
        assert foreign_principal_pages == [[]]
        assert wrong_conversation_pages == [[]]

        attempted_overwrite = replace(
            owner_chunk,
            embedding_signature="attacker-embedding",
            embedding_vector=[0.5, 0.5],
        )
        assert (
            await store.aupdate_attachment_chunk_vectors(
                foreign_principal_id,
                owner_conversation_id,
                [attempted_overwrite],
                ttl_days=_TTL_DAYS,
            )
            is False
        )
        assert (
            await store.aupdate_attachment_chunk_vectors(
                owner_principal_id,
                wrong_conversation_id,
                [attempted_overwrite],
                ttl_days=_TTL_DAYS,
            )
            is False
        )

        owner_pages = [
            page
            async for page in store.aiter_attachment_vectors(
                owner_principal_id,
                owner_conversation_id,
                [(0, cache_key)],
                ttl_days=_TTL_DAYS,
                page_size=1,
            )
        ]
        assert len(owner_pages) == 1
        assert len(owner_pages[0]) == 1
        assert owner_pages[0][0].embedding_signature == "owner-embedding"
        assert owner_pages[0][0].embedding_vector == [1.0, 0.0]
    finally:
        await _delete_principal(pool, owner_principal_id)
        await _delete_principal(pool, foreign_principal_id)
        await pool.close()


@pytest.mark.usefixtures("pg_check")
async def test_expired_attachment_vector_reads_and_writes_miss_before_prune() -> None:
    from dlightrag.core.request.attachments import (
        AttachmentCacheKey,
        AttachmentContextChunk,
        ParsedAttachmentBundle,
    )
    from dlightrag.storage.web_conversations import PGWebConversationStore

    principal_id = f"itest-expired-vector-{uuid4()}"
    pool = await _open_pool()
    try:
        store = PGWebConversationStore(pool=pool)
        await store.initialize()
        await _delete_principal(pool, principal_id)
        conversation = await store.create_conversation(principal_id)
        conversation_id = str(conversation["conversation_id"])
        cache_key = AttachmentCacheKey("content-expired", "parser-v1", "chunker-v1", "cache-1")
        chunk = AttachmentContextChunk(
            chunk_id=cache_key.cache_chunk_id,
            attachment_id="attachment-1",
            filename="expired.pdf",
            chunk_index=1,
            content="cached before expiry",
            cache_key=cache_key,
            embedding_signature="embed-v1",
            embedding_vector=[1.0, 0.0],
        )
        bundle = ParsedAttachmentBundle(chunks=[chunk], parser_signature=cache_key.parser_signature)
        assert (
            await store.materialize_attachment_chunks(
                principal_id,
                conversation_id,
                content_sha256=cache_key.content_sha256,
                parser_signature=cache_key.parser_signature,
                chunk_signature=cache_key.chunk_signature,
                bundle=bundle,
                ttl_days=_TTL_DAYS,
            )
            is True
        )

        async with pool.acquire() as conn:
            await conn.execute(
                "UPDATE web_conversations SET updated_at = NOW() - INTERVAL '31 days' "
                "WHERE principal_id = $1 AND conversation_id = $2::text::uuid",
                principal_id,
                conversation_id,
            )

        assert (
            await store.load_attachment_chunks(
                principal_id,
                conversation_id,
                "attachment-new",
                "expired.pdf",
                content_sha256=cache_key.content_sha256,
                parser_signature=cache_key.parser_signature,
                chunk_signature=cache_key.chunk_signature,
                ttl_days=_TTL_DAYS,
            )
            is None
        )
        assert [
            page
            async for page in store.aiter_attachment_vectors(
                principal_id,
                conversation_id,
                [(0, cache_key)],
                ttl_days=_TTL_DAYS,
                page_size=1,
            )
        ] == [[]]
        assert (
            await store.materialize_attachment_chunks(
                principal_id,
                conversation_id,
                content_sha256=cache_key.content_sha256,
                parser_signature=cache_key.parser_signature,
                chunk_signature=cache_key.chunk_signature,
                bundle=bundle,
                ttl_days=_TTL_DAYS,
            )
            is False
        )
        assert (
            await store.aupdate_attachment_chunk_vectors(
                principal_id,
                conversation_id,
                [replace(chunk, embedding_vector=[0.0, 1.0])],
                ttl_days=_TTL_DAYS,
            )
            is False
        )
        assert await _count_chunk_rows(pool, principal_id) == 1

        assert await store.prune_expired(ttl_days=_TTL_DAYS) >= 1
        assert await _count_chunk_rows(pool, principal_id) == 0
    finally:
        await _delete_principal(pool, principal_id)
        await pool.close()


@pytest.mark.usefixtures("pg_check")
async def test_max_turn_trimming_preserves_conversation_vector_cache() -> None:
    from dlightrag.core.request.attachments import AttachmentContextChunk, ParsedAttachmentBundle
    from dlightrag.storage.web_conversations import PGWebConversationStore

    principal_id = f"itest-trim-vector-{uuid4()}"
    pool = await _open_pool()
    try:
        store = PGWebConversationStore(pool=pool)
        await store.initialize()
        await _delete_principal(pool, principal_id)
        conversation = await store.create_conversation(principal_id)
        conversation_id = str(conversation["conversation_id"])
        bundle = ParsedAttachmentBundle(
            chunks=[
                AttachmentContextChunk(
                    chunk_id="cache-1",
                    attachment_id="attachment-1",
                    filename="trim.pdf",
                    chunk_index=1,
                    content="conversation-level cache",
                    embedding_signature="embed-v1",
                    embedding_vector=[1.0, 0.0],
                )
            ],
            parser_signature="parser-v1",
        )
        assert (
            await store.materialize_attachment_chunks(
                principal_id,
                conversation_id,
                content_sha256="content-trim",
                parser_signature="parser-v1",
                chunk_signature="chunker-v1",
                bundle=bundle,
                ttl_days=_TTL_DAYS,
            )
            is True
        )

        for revision in range(2):
            result = await store.commit_turn(
                principal_id=principal_id,
                conversation_id=conversation_id,
                submission_id=str(uuid4()),
                expected_revision=revision,
                user_text=f"Question {revision}",
                assistant_text=f"Answer {revision}",
                answer_sources={},
                queried_workspaces=[],
                images=[],
                attachments=[],
                max_turns=1,
                ttl_days=_TTL_DAYS,
            )
            assert result.saved is True

        async with pool.acquire() as conn:
            turn_count = await conn.fetchval(
                "SELECT COUNT(*) FROM web_conversation_turns "
                "WHERE principal_id = $1 AND conversation_id = $2::text::uuid",
                principal_id,
                conversation_id,
            )
        assert int(turn_count) == 1
        assert (
            await store.load_attachment_chunks(
                principal_id,
                conversation_id,
                "attachment-remapped",
                "trim.pdf",
                content_sha256="content-trim",
                parser_signature="parser-v1",
                chunk_signature="chunker-v1",
                ttl_days=_TTL_DAYS,
            )
            is not None
        )
        assert await _count_chunk_rows(pool, principal_id) == 1
    finally:
        await _delete_principal(pool, principal_id)
        await pool.close()
