# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Integration tests for Web conversation document-attachment storage.

Exercises the attachment SQL shipped in migrations 0003/0004 end-to-end against
a live PostgreSQL 18 instance (asyncpg): migration application, the attachment
insert, the history JSONB aggregation, the content-addressed parse cache, the
principal-scoped document reads, and the ON DELETE CASCADE lifecycle.

Workspace non-write is proved without brittle global row counts. A complete
``QueryAttachmentService`` cache miss runs through the parse-owner shim, strict
analysis proxy, real LightRAG chunk/MM rendering, robust embedding, and Web
materialization while write-failing workspace handles record any access. The
only permitted result store is the principal/conversation-scoped
``web_conversation_*`` tables.

Requires a running PostgreSQL instance (localhost:5432, dlightrag/dlightrag).
Skipped automatically if PostgreSQL is not available.
"""

import hashlib
import json
from dataclasses import replace
from typing import Any, Never, cast
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


_WORKSPACE_STORAGE_HANDLES = (
    "full_docs",
    "doc_status",
    "text_chunks",
    "chunks_vdb",
    "entities_vdb",
    "relationships_vdb",
    "chunk_entity_relation_graph",
    "llm_response_cache",
    "bm25",
)


class _WorkspaceStorageSentinel:
    def __init__(self, name: str, accesses: list[str]) -> None:
        self._name = name
        self._accesses = accesses

    def _fail(self, operation: str) -> Never:
        access = f"{self._name}.{operation}"
        self._accesses.append(access)
        raise AssertionError(f"Composer touched workspace storage: {access}")

    def __getattr__(self, name: str) -> Any:
        self._fail(name)

    def __bool__(self) -> bool:
        self._fail("__bool__")


class _StorageGuardedLightRAG:
    """Only the initialized parser/MM-renderer attributes Composer may borrow."""

    def __getattribute__(self, name: str) -> Any:
        value = object.__getattribute__(self, name)
        if name in _WORKSPACE_STORAGE_HANDLES:
            cast(_WorkspaceStorageSentinel, value)._fail("__getattribute__")
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        attributes = object.__getattribute__(self, "__dict__")
        if name in _WORKSPACE_STORAGE_HANDLES and name in attributes:
            cast(_WorkspaceStorageSentinel, attributes[name])._fail("__setattr__")
        object.__setattr__(self, name, value)

    def __init__(self, storage_accesses: list[str]) -> None:
        from lightrag.utils import TiktokenTokenizer

        self.tokenizer = TiktokenTokenizer()
        self.addon_params: dict[str, Any] = {}
        self.chunk_token_size = 1200
        self.embedding_func = None
        self.mm_renderer_calls = 0
        for name in _WORKSPACE_STORAGE_HANDLES:
            setattr(self, name, _WorkspaceStorageSentinel(name, storage_accesses))

    def _build_mm_chunks_from_sidecars(self, **kwargs: Any) -> list[dict[str, Any]]:
        from lightrag import LightRAG

        self.mm_renderer_calls += 1
        return LightRAG._build_mm_chunks_from_sidecars(cast(Any, self), **kwargs)


class _DeterministicComposerParser:
    async def parse(self, ctx: Any) -> Any:
        from lightrag.parser.base import ParseResult

        source_path = ctx.source_path("legacy")
        assert source_path.read_bytes() == b"Composer integration source"
        blocks_path = source_path.with_name("composer-contract.blocks.jsonl")
        content = "The report records quarterly revenue and operating margin."
        blocks_path.write_text(
            "\n".join(
                (
                    json.dumps({"type": "meta", "schema_version": 1}),
                    json.dumps(
                        {
                            "type": "content",
                            "blockid": "block-1",
                            "content": content,
                        }
                    ),
                )
            )
            + "\n",
            encoding="utf-8",
        )
        blocks_path.with_name("composer-contract.tables.json").write_text(
            json.dumps(
                {
                    "tables": {
                        "table-1": {
                            "content": (
                                "<table><tr><th>Quarter</th><th>Revenue</th></tr>"
                                "<tr><td>Q4</td><td>42</td></tr></table>"
                            ),
                            "format": "html",
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
        await ctx.rag._persist_parsed_full_docs(
            ctx.doc_id,
            {"sidecar_location": str(blocks_path)},
        )
        assert not any(hasattr(ctx.rag, name) for name in _WORKSPACE_STORAGE_HANDLES)
        return ParseResult(
            doc_id=ctx.doc_id,
            file_path=ctx.file_path,
            parse_format="composer-contract",
            content=content,
            blocks_path=str(blocks_path),
            parse_engine="legacy",
        )


class _DeterministicComposerModels:
    vlm_identity = {"provider": "test", "model": "unused-vlm"}
    extract_identity = {"provider": "test", "model": "deterministic-extract"}

    def __init__(self) -> None:
        self.extract_calls = 0

    async def vlm_func(self, *_args: Any, **_kwargs: Any) -> str:
        raise AssertionError("the table-only fixture must not call VLM")

    async def extract_func(self, *_args: Any, **_kwargs: Any) -> str:
        self.extract_calls += 1
        return json.dumps(
            {
                "name": "Quarterly revenue",
                "description": "A table showing Q4 revenue of 42.",
            }
        )


class _DeterministicEmbeddingProvider:
    asymmetric = False

    def __init__(self) -> None:
        self.document_batches: list[list[str]] = []

    async def embed_texts(self, texts: list[str], *, context: str) -> list[list[float]]:
        assert context == "document"
        self.document_batches.append(list(texts))
        return [[1.0, 0.5, 0.25] for _text in texts]

    async def embed_index_fused(self, _items: Any) -> list[list[float]]:
        raise AssertionError("the table-only fixture must use text embeddings")


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


@pytest.mark.usefixtures("pg_check")
async def test_composer_cache_miss_materializes_only_web_rows(
    monkeypatch: pytest.MonkeyPatch,
    test_config: Any,
) -> None:
    from lightrag import LightRAG

    from dlightrag.core.document_embedding import RobustDocumentEmbedder
    from dlightrag.core.request.attachments import QueryAttachmentService
    from dlightrag.storage.web_conversations import PGWebConversationStore

    principal_id = f"itest-composer-isolation-{uuid4()}"
    document_bytes = b"Composer integration source"
    content_sha256 = hashlib.sha256(document_bytes).hexdigest()
    storage_accesses: list[str] = []
    lightrag = _StorageGuardedLightRAG(storage_accesses)
    models = _DeterministicComposerModels()
    embedding_provider = _DeterministicEmbeddingProvider()
    embedder = RobustDocumentEmbedder(
        embedder=cast(Any, embedding_provider),
        image_enabled=False,
        dimension=3,
        min_image_pixel=1,
        batch_size=8,
        max_concurrency=1,
    )
    config = test_config.model_copy(
        update={
            "parser_sidecars": test_config.parser_sidecars.model_copy(
                update={"vlm": test_config.parser_sidecars.vlm.model_copy(update={"enabled": True})}
            )
        }
    )

    parser = _DeterministicComposerParser()
    monkeypatch.setattr("lightrag.parser.registry.get_parser", lambda _engine: parser)
    real_analyze_multimodal = LightRAG.analyze_multimodal
    analysis_proxies: list[Any] = []

    async def _assert_proxy_then_analyze(proxy: Any, *args: Any, **kwargs: Any) -> dict[str, Any]:
        analysis_proxies.append(proxy)
        assert proxy.llm_response_cache is None
        assert proxy._build_global_config()["llm_response_cache"] is None
        for name in _WORKSPACE_STORAGE_HANDLES:
            if name == "llm_response_cache":
                continue
            assert not hasattr(proxy, name)
        return await real_analyze_multimodal(proxy, *args, **kwargs)

    monkeypatch.setattr(LightRAG, "analyze_multimodal", _assert_proxy_then_analyze)

    pool = await _open_pool()
    try:
        store = PGWebConversationStore(pool=pool)
        await store.initialize()
        await _delete_principal(pool, principal_id)
        conversation = await store.create_conversation(principal_id)
        conversation_id = str(conversation["conversation_id"])
        service = QueryAttachmentService(
            lightrag=lightrag,
            store=store,
            parser_rules="",
            ttl_days=_TTL_DAYS,
            robust_document_embedder=embedder,
            direct_image_embedding_enabled=False,
            model_bundle=models,
            config=config,
            principal_id=principal_id,
            conversation_id=conversation_id,
        )

        bundle, trace = await service.achunks_for_attachment(
            principal_id=principal_id,
            conversation_id=conversation_id,
            attachment_id=str(uuid4()),
            filename="report.[-tP].txt",
            document_bytes=document_bytes,
            content_sha256=content_sha256,
        )

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT content, sidecar_type, embedding_signature, embedding_vector "
                "FROM web_conversation_attachment_chunks "
                "WHERE principal_id = $1 AND conversation_id = $2::text::uuid "
                "ORDER BY chunk_index",
                principal_id,
                conversation_id,
            )

        assert trace["attachment_parse_cache_hit"] is False
        assert trace["attachment_analysis_outcome"] == "success"
        assert trace["attachment_mm_chunk_count"] == 1
        assert trace["attachment_cache_write_status"] == "written"
        assert trace["attachment_cache_materialized"] is True
        assert trace["attachment_embedding_text"] == len(bundle.chunks)
        assert len(rows) == len(bundle.chunks) >= 2
        assert any(row["sidecar_type"] == "table" for row in rows)
        assert all(row["embedding_signature"] for row in rows)
        assert all(len(json.loads(row["embedding_vector"])) == 3 for row in rows)
        assert analysis_proxies
        assert models.extract_calls == 1
        assert lightrag.mm_renderer_calls == 1
        assert sum(len(batch) for batch in embedding_provider.document_batches) == len(
            bundle.chunks
        )
        assert storage_accesses == []
    finally:
        await _delete_principal(pool, principal_id)
        await pool.close()
