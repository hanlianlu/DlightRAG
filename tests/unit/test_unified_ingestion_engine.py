# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for unified LightRAG sidecar ingestion engine."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

from lightrag.utils import compute_mdhash_id
from lightrag.utils_pipeline import normalize_document_file_path
from PIL import Image

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


async def test_batch_document_ingest_uses_lightrag_staged_pipeline(tmp_path: Path) -> None:
    pdf = tmp_path / "b[mineru-iteP].pdf"
    docx = tmp_path / "a.docx"
    pdf.write_bytes(b"%PDF-1.4")
    docx.write_bytes(b"fake-docx")
    engine, deps = _make_engine()
    pdf_doc_id = compute_mdhash_id(normalize_document_file_path(pdf), prefix="doc-")
    docx_doc_id = compute_mdhash_id(normalize_document_file_path(docx), prefix="doc-")
    deps["stores"].get_doc_status.side_effect = [
        None,
        None,
        {"chunks_list": ["chunk-docx"], "content_hash": "sha256:docx", "status": "processed"},
        {"chunks_list": ["chunk-pdf"], "content_hash": "sha256:pdf", "status": "processed"},
    ]
    deps["stores"].get_full_doc.side_effect = [
        {
            "parse_engine": "native",
            "process_options": "iteP",
            "chunk_options": {},
            "sidecar_location": None,
        },
        {
            "parse_engine": "mineru",
            "process_options": "iteP",
            "chunk_options": {},
            "sidecar_location": None,
        },
    ]

    result = await engine.aingest_files([docx, pdf], replace=False)

    assert result["processed"] == 2
    assert [item["doc_id"] for item in result["results"]] == [docx_doc_id, pdf_doc_id]
    kwargs = deps["lightrag"].apipeline_enqueue_documents.await_args.kwargs
    assert kwargs["input"] == ["", ""]
    assert kwargs["file_paths"] == [str(docx), str(pdf)]
    assert kwargs["lightrag_document_paths"] == [str(docx), str(pdf)]
    assert kwargs["parse_engine"] == ["native", "mineru"]
    assert kwargs["process_options"] == ["iteP", "iteP"]
    deps["lightrag"].apipeline_process_enqueue_documents.assert_awaited_once()
    assert deps["metadata_index"].upsert.await_count == 2


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


async def test_document_ingest_delegates_lightrag_raw_parser_route(tmp_path: Path) -> None:
    """LightRAG routing is the ingestability boundary.

    If no DlightRAG parser rule matches, DlightRAG still enqueues the
    LightRAG-resolved parser route and simply skips sidecar vector overrides
    when no sidecar location exists.
    """
    source = tmp_path / "notes.txt"
    source.write_text("plain text")
    engine, deps = _make_engine(parser_rules="docx:native-iteP")  # no wildcard
    deps["stores"].get_full_doc.return_value = {
        "parse_engine": "legacy",
        "process_options": "",
        "chunk_options": {},
        "sidecar_location": None,
    }

    result = await engine.aingest_file(source, replace=False)

    assert result["doc_id"] is not None
    assert result["parse_engine"] == "legacy"
    assert result["chunks"] == ["chunk-a"]
    kwargs = deps["lightrag"].apipeline_enqueue_documents.await_args.kwargs
    assert kwargs["parse_engine"] == "legacy"
    assert kwargs["process_options"] == ""
    deps["stores"].overwrite_chunk_vectors.assert_not_awaited()


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


async def test_image_file_ingest_delegates_to_lightrag_parser(
    tmp_path: Path,
) -> None:
    from PIL import Image

    source = tmp_path / "image.png"
    Image.new("RGB", (1, 1), "white").save(source)
    embedder = AsyncMock()
    embedder.embed_index_images.return_value = [[0.1, 0.2, 0.3]]
    engine, deps = _make_engine(multimodal_embedder=embedder)

    result = await engine.aingest_file(source)

    assert result["source_kind"] == "document"
    deps["stores"].overwrite_chunk_vectors.assert_not_awaited()
    kwargs = deps["lightrag"].apipeline_enqueue_documents.await_args.kwargs
    assert kwargs["docs_format"] == "pending_parse"
    assert kwargs["parse_engine"] == "mineru"
    assert kwargs["process_options"] == "iteP"


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


async def test_parser_image_sidecar_overwrites_lightrag_mm_chunk_vector(
    tmp_path: Path,
) -> None:
    source = tmp_path / "sample[mineru-iteP].pdf"
    source.write_bytes(b"%PDF-1.4")
    doc_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    mm_chunk_id = f"{doc_id}-mm-drawing-000"
    artifact_dir = tmp_path / "sample.parsed"
    assets_dir = artifact_dir / "sample.blocks.assets"
    assets_dir.mkdir(parents=True)
    (artifact_dir / "sample.blocks.jsonl").write_text("", encoding="utf-8")
    image_path = assets_dir / "fig.png"
    Image.new("RGB", (128, 128), "white").save(image_path)
    (artifact_dir / "sample.drawings.json").write_text(
        """
        {
          "drawings": {
            "fig-1": {
              "id": "fig-1",
              "path": "sample.blocks.assets/fig.png",
              "llm_analyze_result": {
                "status": "success",
                "name": "Harness QR",
                "type": "QR code",
                "description": "hallucinated harness lifecycle description"
              }
            }
          }
        }
        """,
        encoding="utf-8",
    )
    embedder = AsyncMock()
    embedder.embed_index_images.return_value = [[0.1, 0.2, 0.3]]
    engine, deps = _make_engine(multimodal_embedder=embedder)
    deps["stores"].get_doc_status.side_effect = [
        None,
        None,
        {
            "chunks_list": ["chunk-a", mm_chunk_id],
            "content_hash": "sha256:parsed",
            "status": "processed",
        },
    ]
    deps["stores"].get_full_doc.return_value = {
        "parse_engine": "mineru",
        "process_options": "iteP",
        "chunk_options": {},
        "sidecar_location": artifact_dir.as_uri(),
    }

    result = await engine.aingest_file(source, replace=False)

    assert result["chunks"] == ["chunk-a", mm_chunk_id]
    deps["stores"].overwrite_chunk_vectors.assert_awaited_once()
    vectors = deps["stores"].overwrite_chunk_vectors.await_args.args[0]
    assert vectors == {mm_chunk_id: [0.1, 0.2, 0.3]}


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


async def test_reingest_skips_when_content_hash_matches(tmp_path: Path) -> None:
    """Re-ingesting a file with the same content_hash must skip and return early."""
    source = tmp_path / "sample[mineru-iteP].pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()
    doc_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")

    current_hash = _file_sha256_static(source)
    deps["stores"].get_doc_status.return_value = {
        "chunks_list": ["chunk-1", "chunk-2"],
        "content_hash": current_hash,
        "status": "processed",
    }

    result = await engine.aingest_file(source, replace=False)

    assert result["doc_id"] == doc_id
    assert result["source_kind"] == "skipped"
    assert result["reason"] == "content_hash_match"
    deps["lightrag"].apipeline_enqueue_documents.assert_not_awaited()


def _file_sha256_static(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


async def test_reingest_proceeds_when_content_hash_differs(tmp_path: Path) -> None:
    """Re-ingesting with different content_hash must proceed normally."""
    source = tmp_path / "sample[mineru-iteP].pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.return_value = {
        "chunks_list": ["chunk-1"],
        "content_hash": "sha256:different_hash",
        "status": "processed",
    }

    result = await engine.aingest_file(source, replace=False)

    assert result.get("source_kind") != "skipped"
    deps["lightrag"].apipeline_enqueue_documents.assert_awaited_once()


async def test_reingest_proceeds_when_not_processed(tmp_path: Path) -> None:
    """Re-ingesting a failed doc must proceed even if hash matches."""
    source = tmp_path / "sample[mineru-iteP].pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()

    current_hash = _file_sha256_static(source)
    deps["stores"].get_doc_status.return_value = {
        "chunks_list": [],
        "content_hash": current_hash,
        "status": "failed",
    }

    result = await engine.aingest_file(source, replace=False)

    assert result.get("source_kind") != "skipped"
    deps["lightrag"].apipeline_enqueue_documents.assert_awaited_once()


def test_sidecar_dir_from_location_handles_file_scheme() -> None:
    from pathlib import Path

    from dlightrag.core.sidecar_provenance import sidecar_dir_from_location

    result = sidecar_dir_from_location("file:///tmp/sample.parsed/")
    assert result == Path("/tmp/sample.parsed/")


def test_sidecar_dir_from_location_handles_file_scheme_with_encoding() -> None:
    from pathlib import Path

    from dlightrag.core.sidecar_provenance import sidecar_dir_from_location

    result = sidecar_dir_from_location("file:///tmp/path%20with%20spaces/")
    assert result == Path("/tmp/path with spaces/")


def test_sidecar_dir_from_location_rejects_non_file_scheme() -> None:
    from dlightrag.core.sidecar_provenance import sidecar_dir_from_location

    assert sidecar_dir_from_location("s3://bucket/key/parsed/") is None
    assert sidecar_dir_from_location("azure://container/path/") is None


def test_sidecar_dir_from_location_handles_bare_path() -> None:
    from pathlib import Path

    from dlightrag.core.sidecar_provenance import sidecar_dir_from_location

    result = sidecar_dir_from_location("/tmp/local/path")
    assert result == Path("/tmp/local/path")


def test_sidecar_dir_from_location_handles_none() -> None:
    from dlightrag.core.sidecar_provenance import sidecar_dir_from_location

    assert sidecar_dir_from_location(None) is None
    assert sidecar_dir_from_location("") is None
