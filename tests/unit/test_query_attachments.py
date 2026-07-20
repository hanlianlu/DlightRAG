# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for Web query attachment parsing and budget packing."""

from pathlib import Path
from typing import Any

import pytest
from lightrag.utils import TiktokenTokenizer

from dlightrag.core import query_attachments
from dlightrag.core.query_attachments import (
    ATTACHMENT_TEXT_TOKEN_BUDGET,
    AttachmentContextChunk,
    ParsedAttachmentBundle,
    QueryAttachmentService,
    _ParseOwnerShim,
    parse_attachment_to_bundle,
    resolve_attachment_chunk_signature,
    resolve_attachment_parser_signature,
    select_attachment_context,
)


def text_chunk(chunk_id: str, tokens: int = 10) -> AttachmentContextChunk:
    return AttachmentContextChunk(
        chunk_id=chunk_id,
        attachment_id="att-1",
        filename="report.pdf",
        chunk_index=1,
        content="alpha",
        token_estimate=tokens,
    )


def visual_chunk(chunk_id: str) -> AttachmentContextChunk:
    return AttachmentContextChunk(
        chunk_id=chunk_id,
        attachment_id="att-1",
        filename="report.pdf",
        chunk_index=2,
        content="chart description",
        token_estimate=10,
        sidecar_type="drawing",
        image_bytes=b"png-bytes",
        image_mime_type="image/png",
    )


def test_attachment_context_chunk_emits_image_data_and_source_metadata() -> None:
    row = visual_chunk("v1").to_context_row()

    assert row["reference_id"] == "att-1"
    assert row["full_doc_id"] == "att-1"
    assert row["_workspace"] == "__web_attachment__"
    assert row["metadata"]["source_uri"] == "web-attachment://att-1"
    assert row["metadata"]["source_download_locator"] == "web-attachment://att-1"
    assert row["image_data"] == "cG5nLWJ5dGVz"
    assert row["image_mime_type"] == "image/png"
    # No pre-budget flag: the single AnswerContextPacker owns image budgeting.
    assert "_answer_image_sent" not in row


def test_select_attachment_context_passes_images_through_for_the_packer() -> None:
    bundle = ParsedAttachmentBundle(chunks=[text_chunk("t1"), visual_chunk("v1")])

    rows, trace = select_attachment_context(bundle, text_token_budget=ATTACHMENT_TEXT_TOKEN_BUDGET)

    assert [row["chunk_id"] for row in rows] == ["t1", "v1"]
    assert rows[1]["image_data"] == "cG5nLWJ5dGVz"
    assert trace["attachment_context_strategy"] == "full"


def test_select_attachment_context_respects_text_budget() -> None:
    bundle = ParsedAttachmentBundle(
        chunks=[text_chunk("a", 100), text_chunk("b", ATTACHMENT_TEXT_TOKEN_BUDGET)]
    )

    rows, trace = select_attachment_context(bundle, text_token_budget=ATTACHMENT_TEXT_TOKEN_BUDGET)

    assert [row["chunk_id"] for row in rows] == ["a"]
    assert trace["attachment_context_strategy"] == "budgeted"


def test_select_attachment_context_does_not_budget_images() -> None:
    # A visual chunk with a large text token_estimate is still dropped by the
    # text budget, but the reducer must not apply any separate image budget.
    bundle = ParsedAttachmentBundle(chunks=[visual_chunk("v1")])

    rows, trace = select_attachment_context(bundle, text_token_budget=1)

    assert rows == []
    assert trace["attachment_context_strategy"] == "budgeted"


class _FakeLightRAG:
    """A minimal object exposing only the read-only parse/chunk knobs.

    Deliberately carries NO storage handles: reusing it for parsing must never
    reach a workspace store.
    """

    def __init__(self) -> None:
        self.tokenizer = TiktokenTokenizer()
        self.addon_params: dict[str, Any] = {}
        self.chunk_token_size = 1200
        self.embedding_func = None

    def _build_mm_chunks_from_sidecars(self, **_kwargs: Any) -> list[dict[str, Any]]:
        return []


class _SpyStore:
    def __init__(self, cached: ParsedAttachmentBundle | None) -> None:
        self._cached = cached
        self.load_calls: list[dict[str, Any]] = []
        self.materialized: list[ParsedAttachmentBundle] = []

    async def load_attachment_chunks(
        self,
        principal_id: str,
        conversation_id: str,
        attachment_id: str,
        filename: str,
        *,
        content_sha256: str,
        parser_signature: str,
        chunk_signature: str,
    ) -> ParsedAttachmentBundle | None:
        self.load_calls.append(
            {
                "principal_id": principal_id,
                "conversation_id": conversation_id,
                "attachment_id": attachment_id,
                "content_sha256": content_sha256,
                "parser_signature": parser_signature,
                "chunk_signature": chunk_signature,
            }
        )
        return self._cached

    async def materialize_attachment_chunks(
        self,
        principal_id: str,
        conversation_id: str,
        *,
        content_sha256: str,
        parser_signature: str,
        chunk_signature: str,
        bundle: ParsedAttachmentBundle,
    ) -> None:
        self.materialized.append(bundle)


def test_parse_owner_shim_persist_is_a_noop_holding_no_store() -> None:
    shim = _ParseOwnerShim()

    # The shim owns no storage handles; parsing through it cannot write rows.
    assert not hasattr(shim, "full_docs")
    assert not hasattr(shim, "doc_status")
    assert not hasattr(shim, "chunks_vdb")
    assert shim._resolve_source_file_for_parser("/tmp/x.pdf") == "/tmp/x.pdf"


async def test_parse_owner_shim_persist_captures_sidecar_only() -> None:
    shim = _ParseOwnerShim()

    result = await shim._persist_parsed_full_docs(
        "att-doc", {"content": "body", "sidecar_location": "loc://x"}
    )

    # No-op returns nothing and only remembers the sidecar location.
    assert result is None
    assert shim.sidecar_location == "loc://x"


async def test_parse_attachment_to_bundle_native_path_writes_no_store(
    tmp_path: Path,
) -> None:
    # A .txt routes to the local legacy engine: no MinerU, no network.
    lightrag = _FakeLightRAG()
    text = "First paragraph about ships.\n\nSecond paragraph about the sea."

    bundle = await parse_attachment_to_bundle(
        lightrag=lightrag,
        attachment_id="att-1",
        filename="notes.txt",
        document_bytes=text.encode("utf-8"),
        parser_rules="",
    )

    assert bundle.chunks, "native parse produced no chunks"
    assert all(chunk.attachment_id == "att-1" for chunk in bundle.chunks)
    assert "ships" in " ".join(chunk.content for chunk in bundle.chunks)
    assert bundle.parser_signature.startswith("legacy:")


async def test_service_returns_cache_hit_without_parsing(monkeypatch: Any) -> None:
    cached = ParsedAttachmentBundle(chunks=[text_chunk("t1")], parser_signature="legacy:")
    store = _SpyStore(cached=cached)

    def _boom(*_args: Any, **_kwargs: Any) -> None:  # pragma: no cover - must not run
        raise AssertionError("parse must not run on a cache hit")

    monkeypatch.setattr(query_attachments, "parse_attachment_to_bundle", _boom)

    service = QueryAttachmentService(lightrag=_FakeLightRAG(), store=store, parser_rules="")
    bundle, trace = await service.achunks_for_attachment(
        principal_id="p1",
        conversation_id="c1",
        attachment_id="att-1",
        filename="report.txt",
        document_bytes=b"ignored",
        content_sha256="deadbeef",
    )

    assert trace["attachment_parse_cache_hit"] is True
    assert bundle is cached
    assert store.materialized == []


async def test_service_cache_miss_parses_and_materializes(monkeypatch: Any) -> None:
    store = _SpyStore(cached=None)
    parsed = ParsedAttachmentBundle(chunks=[text_chunk("t1")], parser_signature="legacy:")

    async def _fake_parse(**_kwargs: Any) -> ParsedAttachmentBundle:
        return parsed

    monkeypatch.setattr(query_attachments, "parse_attachment_to_bundle", _fake_parse)

    service = QueryAttachmentService(lightrag=_FakeLightRAG(), store=store, parser_rules="")
    bundle, trace = await service.achunks_for_attachment(
        principal_id="p1",
        conversation_id="c1",
        attachment_id="att-1",
        filename="report.txt",
        document_bytes=b"body",
        content_sha256="deadbeef",
    )

    assert trace["attachment_parse_cache_hit"] is False
    assert bundle is parsed
    assert store.materialized == [parsed]


async def test_service_parse_failure_is_attachment_scoped(monkeypatch: Any) -> None:
    store = _SpyStore(cached=None)

    async def _fail(**_kwargs: Any) -> ParsedAttachmentBundle:
        raise RuntimeError("mineru exploded")

    monkeypatch.setattr(query_attachments, "parse_attachment_to_bundle", _fail)

    service = QueryAttachmentService(lightrag=_FakeLightRAG(), store=store, parser_rules="")
    bundle, trace = await service.achunks_for_attachment(
        principal_id="p1",
        conversation_id="c1",
        attachment_id="att-1",
        filename="report.txt",
        document_bytes=b"body",
        content_sha256="deadbeef",
    )

    assert bundle.chunks == []
    assert trace["attachment_parse_error"] == "RuntimeError"
    assert trace["attachment_parse_cache_hit"] is False
    assert store.materialized == []


def test_parser_and_chunk_signatures_are_stable_strings() -> None:
    lightrag = _FakeLightRAG()
    parser_sig = resolve_attachment_parser_signature("report.txt", "")
    chunk_sig = resolve_attachment_chunk_signature(lightrag, "report.txt", "")

    assert parser_sig == resolve_attachment_parser_signature("report.txt", "")
    assert chunk_sig == resolve_attachment_chunk_signature(lightrag, "report.txt", "")
    assert parser_sig.startswith("legacy:")
    assert "tokenizer" in chunk_sig


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
