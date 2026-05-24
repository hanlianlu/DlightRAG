# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for unified LightRAG sidecar ingestion engine."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

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
        "document_artifacts": AsyncMock(),
        "chunk_provenance": AsyncMock(),
        "multimodal_embedder": AsyncMock(),
        "vlm_func": AsyncMock(return_value="visual description"),
        "workspace": "default",
        "parser_rules": "*:native-iteP,*:mineru-iteP,*:legacy-R",
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
    assert kwargs["parse_engine"] == "mineru"
    assert kwargs["process_options"] == "iteP"
    deps["lightrag"].apipeline_process_enqueue_documents.assert_awaited_once()
    deps["document_artifacts"].upsert.assert_awaited_once()
    deps["chunk_provenance"].upsert_many.assert_awaited_once()


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
    deps["stores"].upsert_chunks_with_vectors.assert_awaited_once()
    kwargs = deps["lightrag"].apipeline_enqueue_documents.await_args.kwargs
    assert kwargs["docs_format"] == "raw"
    assert kwargs["process_options"] == "P"
