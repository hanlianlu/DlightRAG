# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for unified LightRAG sidecar ingestion engine."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from lightrag.utils import compute_mdhash_id
from lightrag.utils_pipeline import normalize_document_file_path

from dlightrag.core.ingestion.engine import UnifiedIngestionEngine
from dlightrag.core.retrieval.metadata_fields import MetadataFieldRegistry


def _make_engine(**overrides):
    lightrag = AsyncMock()
    lightrag.apipeline_enqueue_documents.return_value = "track-1"
    stores = AsyncMock()
    stores.get_doc_status.return_value = {"chunks_list": ["chunk-a"], "content_hash": "sha256:abc"}
    stores.get_full_doc.return_value = {
        "parse_engine": "mineru",
        "process_options": "iteP",
        "chunk_options": {"paragraph_semantic": {"chunk_token_size": 2000}},
        "sidecar_location": "file:///tmp/sample.parsed/",
    }
    defaults = {
        "lightrag": lightrag,
        "stores": stores,
        "metadata_index": AsyncMock(),
        "multimodal_embedder": AsyncMock(),
        "vlm_func": AsyncMock(return_value="visual description"),
        "workspace": "default",
        "parser_rules": "docx:native-iteP,*:mineru-iteP",
        "chunk_options": {},
    }
    defaults.update(overrides)
    return UnifiedIngestionEngine(**defaults), defaults


async def test_document_ingest_resolves_lightrag_parser_rules(tmp_path: Path) -> None:
    source = tmp_path / "sample[mineru-iteP].pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()

    await engine.aingest_file(source, replace=False)

    kwargs = deps["lightrag"].apipeline_enqueue_documents.await_args.kwargs
    assert kwargs["docs_format"] == "pending_parse"
    assert "ids" not in kwargs
    assert kwargs["parse_engine"] == "mineru"
    assert kwargs["process_options"] == "iteP"
    deps["lightrag"].apipeline_process_enqueue_documents.assert_awaited_once()
    deps["metadata_index"].upsert.assert_awaited_once()


async def test_document_ingest_uses_lightrag_canonical_doc_id(tmp_path: Path) -> None:
    source = tmp_path / "1912.09363v3.pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()

    result = await engine.aingest_file(source, replace=False)

    expected_doc_id = compute_mdhash_id(
        normalize_document_file_path(source),
        prefix="doc-",
    )
    assert result["doc_id"] == expected_doc_id
    deps["metadata_index"].upsert.assert_awaited_once()
    assert deps["metadata_index"].upsert.await_args.args[0] == expected_doc_id


async def test_document_ingest_rejects_builtin_fallback_parser(tmp_path: Path) -> None:
    """Files resolving to the built-in fallback ('legacy') must raise an error.

    The user must explicitly configure parser.rules to route every file type
    to a supported parser (mineru, native, or docling).  Silently accepting
    the legacy fallback would mask a broken or missing parser_rules config.
    """
    source = tmp_path / "notes.unsupported"
    source.write_text("plain text")
    engine, deps = _make_engine(parser_rules="docx:native-iteP")  # no wildcard → fallback

    with pytest.raises(ValueError, match="Configure parser.rules"):
        await engine.aingest_file(source, replace=False)

    deps["lightrag"].apipeline_enqueue_documents.assert_not_awaited()


async def test_document_ingest_accepts_explicit_user_metadata(tmp_path: Path) -> None:
    source = tmp_path / "sample[mineru-iteP].pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine(
        metadata_registry=MetadataFieldRegistry.from_config(
            {
                "author": {
                    "type": "string",
                    "normalizer": "casefold_trim",
                    "filter_ops": ["exact"],
                    "indexed": True,
                }
            }
        ),
        allow_ad_hoc_metadata=True,
        default_metadata_policy="validate",
    )

    await engine.aingest_file(
        source,
        replace=False,
        metadata={"author": " Ada Lovelace ", "project": "Analytical Engine"},
        metadata_policy="validate",
    )

    _, saved = deps["metadata_index"].upsert.await_args.args
    assert saved["user_metadata"]["author"] == " Ada Lovelace "
    assert saved["metadata_filterable"]["author"] == "ada lovelace"
    assert saved["metadata_json"]["project"] == "Analytical Engine"


async def test_native_image_ingest_adds_direct_vector_and_visual_semantic_doc(
    tmp_path: Path,
) -> None:
    from PIL import Image

    source = tmp_path / "image.png"
    Image.new("RGB", (1, 1), "white").save(source)
    embedder = AsyncMock()
    embedder.embed_index_images.return_value = [[0.1, 0.2, 0.3]]
    engine, deps = _make_engine(multimodal_embedder=embedder)

    result = await engine.aingest_file(source)

    assert result["source_kind"] == "image"
    deps["stores"].upsert_document_record.assert_awaited_once()
    deps["stores"].upsert_chunks_with_vectors.assert_awaited_once()
    kwargs = deps["lightrag"].apipeline_enqueue_documents.await_args.kwargs
    assert kwargs["docs_format"] == "raw"
    assert kwargs["process_options"] == "P"


async def test_document_ingest_cleans_up_partial_before_reingest(tmp_path: Path) -> None:
    """When a doc exists with status 'analyzing' (interrupted MinerU run),
    re-ingesting must clean up the partial record and proceed normally."""
    source = tmp_path / "sample[mineru-iteP].pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()
    doc_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")

    # Simulate a partial record from an interrupted ingest.
    deps["stores"].get_doc_status.return_value = {
        "chunks_list": ["old-chunk-1"],
        "content_hash": "sha256:deadbeef",
        "status": "analyzing",
    }
    deps["stores"].get_full_doc.return_value = {
        "parse_engine": "mineru",
        "process_options": "iteP",
        "chunk_options": {},
        "sidecar_location": "file:///tmp/nonexistent.parsed/",
    }

    result = await engine.aingest_file(source, replace=False)

    # Must have cleaned up the old partial record.
    deps["stores"].cleanup_doc.assert_awaited_once_with(doc_id)
    deps["metadata_index"].delete.assert_awaited_once_with(doc_id)

    # Must have proceeded with normal ingest.
    assert result["doc_id"] == doc_id
    deps["lightrag"].apipeline_enqueue_documents.assert_awaited_once()
    deps["lightrag"].apipeline_process_enqueue_documents.assert_awaited_once()


async def test_document_ingest_skips_cleanup_when_already_processed(tmp_path: Path) -> None:
    """When a doc exists with status 'processed', re-ingest must NOT
    clean up (it should go through normal duplicate detection instead)."""
    source = tmp_path / "sample[mineru-iteP].pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.return_value = {
        "chunks_list": ["chunk-1"],
        "content_hash": "sha256:abc",
        "status": "processed",
    }

    await engine.aingest_file(source, replace=False)

    # Cleanup must NOT have been called for a healthy doc.
    deps["stores"].cleanup_doc.assert_not_awaited()
    deps["metadata_index"].delete.assert_not_awaited()


async def test_document_ingest_first_time_no_cleanup(tmp_path: Path) -> None:
    """When no prior doc_status exists, ingest proceeds without cleanup."""
    source = tmp_path / "sample[mineru-iteP].pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.return_value = None

    result = await engine.aingest_file(source, replace=False)

    deps["stores"].cleanup_doc.assert_not_awaited()
    assert result["doc_id"] is not None


async def test_concurrent_ingest_of_same_doc_is_serialized(tmp_path: Path) -> None:
    """Two concurrent ingests of the same failed doc must NOT both clean up.
    The per-doc lock ensures the second sees the first's state changes."""
    import asyncio

    source = tmp_path / "sample[mineru-iteP].pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()

    status_iter = iter(
        [
            {"chunks_list": [], "content_hash": "sha256:dead", "status": "failed"},
            {"chunks_list": ["chunk-1"], "content_hash": "sha256:abc", "status": "processing"},
        ]
    )

    async def status_side_effect(doc_id_arg: str) -> dict | None:
        try:
            return next(status_iter)
        except StopIteration:
            return {"chunks_list": ["chunk-1"], "content_hash": "sha256:abc", "status": "processed"}

    deps["stores"].get_doc_status = AsyncMock(side_effect=status_side_effect)
    deps["stores"].get_full_doc.return_value = {
        "parse_engine": "mineru",
        "process_options": "iteP",
        "chunk_options": {},
        "sidecar_location": "file:///tmp/sample.parsed/",
    }

    async def slow_cleanup(doc_id_arg: str) -> int:
        await asyncio.sleep(0.03)
        return 1

    deps["stores"].cleanup_doc = AsyncMock(side_effect=slow_cleanup)

    async def ingest() -> dict:
        return await engine.aingest_file(source, replace=False)

    results = await asyncio.gather(ingest(), ingest())
    assert len(results) == 2
    # Cleanup must have been called exactly once (not twice).
    assert deps["stores"].cleanup_doc.await_count == 1
