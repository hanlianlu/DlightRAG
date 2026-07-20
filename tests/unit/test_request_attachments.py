# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for Web query attachment parsing and budget packing."""

from pathlib import Path
from typing import Any

import pytest
from lightrag.utils import TiktokenTokenizer

from dlightrag.core.request import attachment_digest, attachments
from dlightrag.core.request.attachment_digest import (
    ATTACHMENT_PLANNER_DIGEST_TOKEN_BUDGET,
    build_attachment_planner_digests,
)
from dlightrag.core.request.attachments import (
    AttachmentContextChunk,
    ParsedAttachmentBundle,
    QueryAttachmentService,
    _ParseOwnerShim,
    parse_attachment_to_bundle,
    resolve_attachment_chunk_signature,
    resolve_attachment_parser_signature,
)
from dlightrag.utils.tokens import estimate_tokens


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


def planner_bundle(
    attachment_id: str,
    contents: list[str],
    *,
    sidecar_types: dict[int, str] | None = None,
) -> ParsedAttachmentBundle:
    return ParsedAttachmentBundle(
        chunks=[
            AttachmentContextChunk(
                chunk_id=f"{attachment_id}-{index}",
                attachment_id=attachment_id,
                filename=f"{attachment_id}.pdf",
                chunk_index=index,
                content=content,
                token_estimate=estimate_tokens(content),
                sidecar_type=(sidecar_types or {}).get(index),
            )
            for index, content in enumerate(contents, start=1)
        ]
    )


def test_attachment_planner_digest_keeps_short_document_in_full() -> None:
    bundle = planner_bundle(
        "short",
        [
            "Fractions Challenge Worksheet\nInstructions: simplify every fraction.",
            "Problem 20: Explain how the numerator and denominator change.",
        ],
    )

    digests, trace = build_attachment_planner_digests([("short", bundle)])

    assert "Fractions Challenge Worksheet" in digests["short"]
    assert "Problem 20" in digests["short"]
    assert trace["attachment_digest_strategy"] == "full"
    assert trace["attachment_digest_output_tokens"] <= ATTACHMENT_PLANNER_DIGEST_TOKEN_BUDGET


def test_attachment_planner_digest_preserves_structure_and_uniform_coverage() -> None:
    contents = [f"Section {index} " + (f"body-{index} " * 700) for index in range(24)]
    contents[0] = "START-MARKER Document title and executive introduction. " + contents[0]
    contents[8] = "[Table Name] Revenue by region\nColumns: Region, Revenue. " + contents[8]
    contents[12] = "MIDDLE-MARKER Central findings. " + contents[12]
    contents[-1] = "END-MARKER Final conclusion and recommendations. " + contents[-1]
    bundle = planner_bundle("long", contents, sidecar_types={9: "table"})

    digests, trace = build_attachment_planner_digests([("long", bundle)])
    digest = digests["long"]

    assert "START-MARKER" in digest
    assert "[Table Name] Revenue by region" in digest
    assert "MIDDLE-MARKER" in digest
    assert "END-MARKER" in digest
    assert trace["attachment_digest_strategy"] == "sampled"
    assert trace["attachment_digest_output_tokens"] <= ATTACHMENT_PLANNER_DIGEST_TOKEN_BUDGET


def test_attachment_planner_digest_shares_global_budget_across_documents() -> None:
    bundles = [
        (
            f"doc-{index}",
            planner_bundle(
                f"doc-{index}",
                [f"DOC-{index}-SECTION-{part} " + ("detail " * 700) for part in range(16)],
            ),
        )
        for index in range(3)
    ]

    digests, trace = build_attachment_planner_digests(bundles)

    assert set(digests) == {"doc-0", "doc-1", "doc-2"}
    assert all(digests.values())
    budgets = trace["attachment_digest_document_budgets"]
    assert sum(budgets.values()) == ATTACHMENT_PLANNER_DIGEST_TOKEN_BUDGET
    assert min(budgets.values()) >= 1_536
    assert max(budgets.values()) - min(budgets.values()) <= 1
    assert trace["attachment_digest_output_tokens"] <= ATTACHMENT_PLANNER_DIGEST_TOKEN_BUDGET

    _, reversed_trace = build_attachment_planner_digests(list(reversed(bundles)))
    assert reversed_trace["attachment_digest_document_budgets"] == budgets


def test_attachment_planner_token_samples_do_not_cluster_at_the_front() -> None:
    contents = [f"chunk-{index} " + ("x " * 200) for index in range(24)]

    selected = attachment_digest._uniform_token_indices(contents, list(range(24)), 7)

    assert selected[0] == 0
    assert selected[-1] == 23
    assert any(abs(index - 12) <= 1 for index in selected)
    assert max(right - left for left, right in zip(selected, selected[1:], strict=False)) <= 6


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


async def test_parse_attachment_to_bundle_docx_native_path(tmp_path: Path) -> None:
    # A .docx routes to the native engine, which imports python-docx AND
    # defusedxml. Pin that path end-to-end: a missing parser dependency used to
    # be swallowed as an attachment_parse_error (0 chunks, document silently
    # dropped) instead of failing a test.
    from docx import Document

    doc = Document()
    doc.add_paragraph("Fractions worksheet about numerators and denominators.")
    doc.add_paragraph("Add the numerators; keep the denominator.")
    path = tmp_path / "frac.docx"
    doc.save(str(path))

    bundle = await parse_attachment_to_bundle(
        lightrag=_FakeLightRAG(),
        attachment_id="att-docx",
        filename="frac.docx",
        document_bytes=path.read_bytes(),
        parser_rules="docx:native-iteP,*:mineru-iteP",
    )

    assert bundle.chunks, "native docx parse produced no chunks"
    assert all(chunk.attachment_id == "att-docx" for chunk in bundle.chunks)
    assert "numerators" in " ".join(chunk.content for chunk in bundle.chunks).lower()
    assert bundle.parser_signature.startswith("native")


async def test_service_returns_cache_hit_without_parsing(monkeypatch: Any) -> None:
    cached = ParsedAttachmentBundle(chunks=[text_chunk("t1")], parser_signature="legacy:")
    store = _SpyStore(cached=cached)

    def _boom(*_args: Any, **_kwargs: Any) -> None:  # pragma: no cover - must not run
        raise AssertionError("parse must not run on a cache hit")

    monkeypatch.setattr(attachments, "parse_attachment_to_bundle", _boom)

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

    monkeypatch.setattr(attachments, "parse_attachment_to_bundle", _fake_parse)

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

    monkeypatch.setattr(attachments, "parse_attachment_to_bundle", _fail)

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
