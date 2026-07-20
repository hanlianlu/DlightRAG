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
    from dlightrag.core.query_attachments import (
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
                ),
            ],
        )
        await store.materialize_attachment_chunks(
            principal_id,
            conversation_id,
            content_sha256=content_sha256,
            parser_signature=parser_signature,
            chunk_signature=chunk_signature,
            bundle=source_bundle,
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
        )
        assert loaded is not None
        assert loaded.parser_signature == parser_signature
        assert len(loaded.chunks) == 2
        text_chunk, visual_chunk = loaded.chunks
        assert text_chunk.chunk_id == f"att:{reused_attachment_id}:0000"
        assert text_chunk.attachment_id == reused_attachment_id
        assert text_chunk.content == "The quarterly results improved."
        assert text_chunk.image_bytes is None
        assert visual_chunk.chunk_id == f"att:{reused_attachment_id}:0001"
        assert visual_chunk.image_bytes == image_bytes
        assert visual_chunk.image_mime_type == "image/png"
        assert visual_chunk.sidecar_type == "image"

        # Cache MISS: a different chunk signature returns nothing.
        missed = await store.load_attachment_chunks(
            principal_id,
            conversation_id,
            attachment_id,
            filename,
            content_sha256=content_sha256,
            parser_signature=parser_signature,
            chunk_signature="chunker@v2",
        )
        assert missed is None

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
        assert await _count_chunk_rows(pool, principal_id) == 2

        deleted = await store.delete_conversation(principal_id, conversation_id, ttl_days=_TTL_DAYS)
        assert deleted is True

        assert await _count_attachment_rows(pool, principal_id) == 0
        assert await _count_chunk_rows(pool, principal_id) == 0
    finally:
        # 7) Final cleanup: remove any residue for our unique test principals.
        await _delete_principal(pool, principal_id)
        await _delete_principal(pool, other_principal_id)
        await pool.close()
