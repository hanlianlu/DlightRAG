# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for unified LightRAG sidecar ingestion engine."""

import hashlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from lightrag.base import DocStatus
from lightrag.utils import compute_mdhash_id
from lightrag.utils_pipeline import normalize_document_file_path
from PIL import Image

from dlightrag.core.ingestion.engine import (
    PreparedIngestFile,
    UnifiedIngestionEngine,
    _prepare_ingest_item,
)
from dlightrag.core.retrieval.metadata_fields import MetadataFieldRegistry


def _sha256(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _make_engine(**overrides):
    lightrag = AsyncMock()
    lightrag.apipeline_enqueue_documents.return_value = "track-1"
    stores = AsyncMock()
    stores.get_doc_status.return_value = {
        "status": "processed",
        "chunks_list": ["chunk-a"],
        "content_hash": "sha256:abc",
    }
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


async def test_replace_false_keeps_idempotent_skip(tmp_path: Path) -> None:
    content = b"%PDF-1.4"
    source = tmp_path / "sample.pdf"
    source.write_bytes(content)
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.return_value = {
        "chunks_list": ["chunk-a"],
        "content_hash": _sha256(content),
        "status": "processed",
    }

    result = await engine.aingest_file(source, replace=False)

    assert result["source_kind"] == "skipped"
    deps["lightrag"].adelete_by_doc_id.assert_not_awaited()
    deps["lightrag"].apipeline_enqueue_documents.assert_not_awaited()


async def test_replace_true_bypasses_idempotent_skip(tmp_path: Path) -> None:
    content = b"%PDF-1.4"
    source = tmp_path / "sample.pdf"
    source.write_bytes(content)
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.side_effect = [
        {"chunks_list": ["old-chunk"], "content_hash": _sha256(content), "status": "processed"},
        None,
        {"chunks_list": ["new-chunk"], "content_hash": _sha256(content), "status": "processed"},
    ]

    result = await engine.aingest_file(source, replace=True)

    assert result["source_kind"] == "document"
    assert result["chunks"] == ["new-chunk"]
    deps["lightrag"].adelete_by_doc_id.assert_awaited_once()
    deps["lightrag"].apipeline_enqueue_documents.assert_awaited_once()


async def test_batch_replace_true_bypasses_idempotent_skip(tmp_path: Path) -> None:
    content = b"%PDF-1.4"
    source = tmp_path / "sample.pdf"
    source.write_bytes(content)
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.side_effect = [
        {"chunks_list": ["old-chunk"], "content_hash": _sha256(content), "status": "processed"},
        {"chunks_list": ["new-chunk"], "content_hash": _sha256(content), "status": "processed"},
    ]

    result = await engine.aingest_files([source], replace=True)

    assert result["processed"] == 1
    assert result["results"][0]["chunks"] == ["new-chunk"]
    deps["lightrag"].adelete_by_doc_id.assert_awaited_once()
    deps["lightrag"].apipeline_enqueue_documents.assert_awaited_once()


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
    assert deps["metadata_index"].upsert.await_count == 2


async def test_document_ingest_raises_when_pipeline_finishes_failed(tmp_path: Path) -> None:
    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.side_effect = [
        None,
        None,
        {
            "status": DocStatus.FAILED,
            "chunks_list": [],
            "content_hash": None,
            "content_summary": "PDF parser failed on page 3",
            "error_msg": None,
        },
    ]

    with pytest.raises(RuntimeError, match="PDF parser failed on page 3"):
        await engine.aingest_file(source, replace=False)

    assert deps["metadata_index"].upsert.await_count == 1
    deps["stores"].overwrite_chunk_vectors.assert_not_awaited()


async def test_document_ingest_preserves_lightrag_parser_engine_params(
    tmp_path: Path,
) -> None:
    source = tmp_path / "sample.[mineru(page_range=1-3)-iteP].pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()

    await engine.aingest_file(source, replace=False)

    kwargs = deps["lightrag"].apipeline_enqueue_documents.await_args.kwargs
    assert kwargs["parse_engine"] == "mineru(page_range=1-3)"


async def test_document_ingest_labels_bm25_chunk_languages(tmp_path: Path) -> None:
    source = tmp_path / "sample[mineru-iteP].pdf"
    source.write_bytes(b"%PDF-1.4")

    class FakeClassifier:
        def detect(self, content: str) -> str:
            return {"现金流 风险": "zh", "risk factors": "en"}.get(content, "simple")

    engine, deps = _make_engine(bm25_language_classifier=FakeClassifier())
    deps["stores"].get_doc_status.return_value = {
        "chunks_list": ["chunk-zh", "chunk-en"],
        "content_hash": "sha256:abc",
        "status": "processed",
    }
    deps["stores"].fetch_chunk_contents.return_value = [
        {"id": "chunk-zh", "content": "现金流 风险"},
        {"id": "chunk-en", "content": "risk factors"},
    ]

    await engine.aingest_file(source, replace=False)

    deps["stores"].fetch_chunk_contents.assert_awaited_once_with(["chunk-zh", "chunk-en"])
    deps["stores"].update_chunk_bm25_languages.assert_awaited_once_with(
        {"chunk-zh": "zh", "chunk-en": "en"}
    )


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
    assert kwargs["parse_engine"] == ["native", "mineru"]
    assert kwargs["process_options"] == ["iteP", "iteP"]
    deps["lightrag"].apipeline_process_enqueue_documents.assert_awaited_once()
    assert deps["metadata_index"].upsert.await_count == 4


async def test_batch_document_ingest_preserves_per_file_chunk_params(
    tmp_path: Path,
) -> None:
    pdf = tmp_path / "b.[mineru-iteP(chunk_ts=1234,drop_rf=true)].pdf"
    docx = tmp_path / "a.docx"
    pdf.write_bytes(b"%PDF-1.4")
    docx.write_bytes(b"fake-docx")
    engine, deps = _make_engine(
        parser_rules="docx:native-iteP,*:mineru-iteP",
        chunk_options={"paragraph_semantic": {"chunk_overlap_token_size": 99}},
    )
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

    await engine.aingest_files([docx, pdf], replace=False)

    kwargs = deps["lightrag"].apipeline_enqueue_documents.await_args.kwargs
    assert kwargs["chunk_options"] == [
        {"paragraph_semantic": {"chunk_overlap_token_size": 99}},
        {
            "paragraph_semantic": {
                "chunk_overlap_token_size": 99,
                "chunk_token_size": 1234,
                "drop_references": True,
            }
        },
    ]


async def test_prepared_batch_uses_explicit_download_locator(tmp_path: Path) -> None:
    parser_source = tmp_path / "report__s3_abcd1234.pdf"
    parser_source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.side_effect = [
        None,
        {"chunks_list": ["chunk-report"], "content_hash": "sha256:pdf", "status": "processed"},
    ]
    deps["stores"].get_full_doc.return_value = {
        "parse_engine": "mineru",
        "process_options": "iteP",
        "chunk_options": {},
        "sidecar_location": None,
    }

    result = await engine.aingest_files(
        [
            PreparedIngestFile(
                parser_path=parser_source,
                source_uri="s3://bucket/team-a/report.pdf",
                download_locator="s3://bucket/team-a/report.pdf",
                display_filename="report.pdf",
            )
        ],
        replace=False,
    )

    assert result["processed"] == 1
    kwargs = deps["lightrag"].apipeline_enqueue_documents.await_args.kwargs
    assert kwargs["file_paths"] == [str(parser_source)]
    _, saved = deps["metadata_index"].upsert.await_args.args
    assert saved["file_path"] == "s3://bucket/team-a/report.pdf"
    assert saved["source_uri"] == "s3://bucket/team-a/report.pdf"
    assert saved["download_locator"] == "s3://bucket/team-a/report.pdf"
    assert saved["filename"] == "report.pdf"
    assert saved["filename_stem"] == "report"
    assert saved["file_extension"] == "pdf"


def test_raw_path_preparation_uses_collision_safe_local_identity(tmp_path: Path) -> None:
    first = tmp_path / "team-a" / "report.pdf"
    second = tmp_path / "team-b" / "report.pdf"
    first.parent.mkdir()
    second.parent.mkdir()

    first_item = _prepare_ingest_item(first, workspace="Finance Team")
    second_item = _prepare_ingest_item(second, workspace="Finance Team")

    assert first_item.source_uri.startswith("local://finance_team/")
    assert second_item.source_uri.startswith("local://finance_team/")
    assert first_item.source_uri.endswith("/report.pdf")
    assert second_item.source_uri.endswith("/report.pdf")
    assert first_item.source_uri != second_item.source_uri
    assert str(tmp_path) not in first_item.source_uri
    assert str(tmp_path) not in second_item.source_uri
    assert first_item.download_locator == str(first)
    assert second_item.download_locator == str(second)


async def test_single_file_forwards_explicit_source_contract_to_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "sample.pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, _deps = _make_engine()
    prepare_metadata = MagicMock(wraps=engine._prepare_metadata_record)
    monkeypatch.setattr(engine, "_prepare_metadata_record", prepare_metadata)

    await engine.aingest_file(
        source,
        source_uri="local://default/docs/sample.pdf",
        download_locator=str(source),
    )

    assert prepare_metadata.call_args.kwargs["source_uri"] == ("local://default/docs/sample.pdf")
    assert prepare_metadata.call_args.kwargs["download_locator"] == str(source)


async def test_metadata_only_update_forwards_explicit_source_contract(
    tmp_path: Path, monkeypatch
) -> None:
    content = b"%PDF-1.4"
    source = tmp_path / "sample.pdf"
    source.write_bytes(content)
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.return_value = {
        "chunks_list": ["chunk-a"],
        "content_hash": _sha256(content),
        "status": "processed",
    }
    prepare_metadata = MagicMock(wraps=engine._prepare_metadata_record)
    monkeypatch.setattr(engine, "_prepare_metadata_record", prepare_metadata)

    result = await engine.aingest_file(
        source,
        source_uri="local://default/docs/sample.pdf",
        download_locator=str(source),
        title="Updated title",
    )

    assert result["source_kind"] == "metadata_updated"
    assert prepare_metadata.call_args.kwargs["source_uri"] == ("local://default/docs/sample.pdf")
    assert prepare_metadata.call_args.kwargs["download_locator"] == str(source)


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
    assert deps["metadata_index"].upsert.await_count == 2
    assert all(
        call.args[0] == expected_doc_id for call in deps["metadata_index"].upsert.await_args_list
    )


async def test_pending_metadata_is_persisted_before_parser_enqueue_failure(
    tmp_path: Path,
) -> None:
    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.return_value = None
    persisted: list[dict] = []

    async def save_metadata(_doc_id: str, metadata: dict) -> None:
        persisted.append(metadata)

    async def fail_enqueue(**_kwargs) -> None:
        assert persisted
        raise RuntimeError("parser enqueue failed")

    deps["metadata_index"].upsert = AsyncMock(side_effect=save_metadata)
    deps["lightrag"].apipeline_enqueue_documents = AsyncMock(side_effect=fail_enqueue)

    with pytest.raises(RuntimeError, match="parser enqueue failed"):
        await engine.aingest_file(
            source,
            source_uri="bynder://asset/1",
            download_locator="https://cdn.example.com/assets/1.pdf",
        )

    assert persisted == [
        {
            "filename": "1.pdf",
            "filename_stem": "1",
            "file_path": "https://cdn.example.com/assets/1.pdf",
            "source_uri": "bynder://asset/1",
            "download_locator": "https://cdn.example.com/assets/1.pdf",
            "file_extension": "pdf",
            "ingest_strategy": "lightrag_sidecar_unified",
            "user_metadata": {},
            "metadata_filterable": {},
            "metadata_json": {},
        }
    ]


async def test_batch_pending_metadata_is_persisted_before_parser_enqueue_failure(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    first.write_bytes(b"%PDF-1.4 first")
    second.write_bytes(b"%PDF-1.4 second")
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.side_effect = [None, None]
    persisted: list[tuple[str, dict]] = []

    async def save_metadata(doc_id: str, metadata: dict) -> None:
        persisted.append((doc_id, metadata))

    async def fail_enqueue(**_kwargs) -> None:
        assert len(persisted) == 2
        raise RuntimeError("batch parser enqueue failed")

    deps["metadata_index"].upsert = AsyncMock(side_effect=save_metadata)
    deps["lightrag"].apipeline_enqueue_documents = AsyncMock(side_effect=fail_enqueue)

    with pytest.raises(RuntimeError, match="batch parser enqueue failed"):
        await engine.aingest_files(
            [
                PreparedIngestFile(
                    parser_path=first,
                    source_uri="bynder://asset/1",
                    download_locator="https://cdn.example.com/assets/1.pdf",
                ),
                PreparedIngestFile(
                    parser_path=second,
                    source_uri="bynder://asset/2",
                    download_locator="s3://documents/assets/2.pdf",
                ),
            ]
        )

    assert [metadata["source_uri"] for _, metadata in persisted] == [
        "bynder://asset/1",
        "bynder://asset/2",
    ]
    assert [metadata["download_locator"] for _, metadata in persisted] == [
        "https://cdn.example.com/assets/1.pdf",
        "s3://documents/assets/2.pdf",
    ]


async def test_document_ingest_delegates_non_sidecar_parser_route(tmp_path: Path) -> None:
    """LightRAG routing is the ingestability boundary.

    DlightRAG enqueues the LightRAG-resolved parser route and skips sidecar
    vector overrides when that route does not produce a sidecar location.
    """
    source = tmp_path / "notes.docx"
    source.write_bytes(b"fake docx")
    engine, deps = _make_engine()
    deps["stores"].get_full_doc.return_value = {
        "parse_engine": "native",
        "process_options": "iteP",
        "chunk_options": {},
        "sidecar_location": None,
    }

    result = await engine.aingest_file(source, replace=False)

    assert result["doc_id"] is not None
    assert result["parse_engine"] == "native"
    assert result["process_options"] == "iteP"
    assert result["chunks"] == ["chunk-a"]
    kwargs = deps["lightrag"].apipeline_enqueue_documents.await_args.kwargs
    assert kwargs["parse_engine"] == "native"
    assert kwargs["process_options"] == "iteP"
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


async def test_prepared_file_metadata_overlays_batch_metadata(tmp_path: Path) -> None:
    source = tmp_path / "asset.pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine(
        metadata_registry=MetadataFieldRegistry.from_config(
            {
                "source_system": {"type": "string", "filter_ops": ["exact"]},
                "department": {"type": "string", "filter_ops": ["exact"]},
                "asset_id": {"type": "string", "filter_ops": ["exact"]},
            }
        ),
        allow_ad_hoc_metadata=True,
        default_metadata_policy="validate",
    )

    await engine.aingest_files(
        [
            PreparedIngestFile(
                parser_path=source,
                source_uri="local://default/asset.pdf",
                download_locator=str(source),
                metadata={"department": " Legal ", "asset_id": "A-123"},
            )
        ],
        replace=False,
        metadata={"source_system": "Bynder", "department": "Marketing"},
        metadata_policy="validate",
    )

    _, saved = deps["metadata_index"].upsert.await_args.args
    assert saved["user_metadata"] == {
        "source_system": "Bynder",
        "department": " Legal ",
        "asset_id": "A-123",
    }
    assert saved["metadata_filterable"] == {
        "source_system": "bynder",
        "department": "legal",
        "asset_id": "a-123",
    }


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
    artifact_dir = tmp_path / "old.parsed"
    artifact_dir.mkdir()
    events: list[str] = []

    # Simulate a partial record from an interrupted ingest.
    partial_status = {
        "chunks_list": ["old-chunk-1"],
        "content_hash": "sha256:deadbeef",
        "status": "analyzing",
    }
    deps["stores"].get_doc_status.side_effect = [
        partial_status,
        partial_status,
        {"chunks_list": ["chunk-a"], "content_hash": "sha256:abc", "status": "processed"},
    ]

    async def get_full_doc(doc_id_arg: str) -> dict | None:
        assert doc_id_arg == doc_id
        events.append("get_full_doc")
        if "adelete_by_doc_id" in events:
            return {
                "parse_engine": "mineru",
                "process_options": "iteP",
                "chunk_options": {},
                "sidecar_location": None,
            }
        return {
            "parse_engine": "mineru",
            "process_options": "iteP",
            "chunk_options": {},
            "sidecar_location": artifact_dir.as_uri(),
        }

    async def delete_doc(doc_id_arg: str, *, delete_llm_cache: bool) -> object:
        assert doc_id_arg == doc_id
        assert delete_llm_cache is True
        events.append("adelete_by_doc_id")
        return type("DeletionResult", (), {"status": "success"})()

    deps["stores"].get_full_doc = AsyncMock(side_effect=get_full_doc)
    deps["lightrag"].adelete_by_doc_id = AsyncMock(side_effect=delete_doc)

    result = await engine.aingest_file(source, replace=False)

    # Must have cleaned up the old partial record.
    deps["lightrag"].adelete_by_doc_id.assert_awaited_once_with(doc_id, delete_llm_cache=True)
    deps["metadata_index"].delete.assert_awaited_once_with(doc_id)
    assert events[:2] == ["get_full_doc", "adelete_by_doc_id"]
    assert not artifact_dir.exists()

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
    deps["lightrag"].adelete_by_doc_id.assert_not_awaited()
    deps["metadata_index"].delete.assert_not_awaited()


async def test_document_ingest_first_time_no_cleanup(tmp_path: Path) -> None:
    """When no prior doc_status exists, ingest proceeds without cleanup."""
    source = tmp_path / "sample[mineru-iteP].pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.side_effect = [
        None,
        None,
        {"chunks_list": ["chunk-a"], "content_hash": "sha256:abc", "status": "processed"},
    ]

    result = await engine.aingest_file(source, replace=False)

    deps["lightrag"].adelete_by_doc_id.assert_not_awaited()
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


async def test_parser_image_sidecar_skips_vector_overwrite_when_direct_embedding_disabled(
    tmp_path: Path,
) -> None:
    source = tmp_path / "sample[mineru-iteP].pdf"
    source.write_bytes(b"%PDF-1.4")
    doc_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    mm_chunk_id = f"{doc_id}-mm-drawing-000"
    artifact_dir = tmp_path / "sample.parsed"
    assets_dir = artifact_dir / "sample.blocks.assets"
    assets_dir.mkdir(parents=True)
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
                "description": "LightRAG semantic visual chunk"
              }
            }
          }
        }
        """,
        encoding="utf-8",
    )
    embedder = AsyncMock()
    embedder.embed_index_images.return_value = [[0.1, 0.2, 0.3]]
    engine, deps = _make_engine(
        multimodal_embedder=embedder,
        direct_image_embedding_enabled=False,
    )
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
    embedder.embed_index_images.assert_not_awaited()
    deps["stores"].overwrite_chunk_vectors.assert_not_awaited()


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
        assert doc_id_arg == compute_mdhash_id(
            normalize_document_file_path(source),
            prefix="doc-",
        )
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

    async def slow_delete(doc_id_arg: str, *, delete_llm_cache: bool) -> object:
        assert doc_id_arg == compute_mdhash_id(
            normalize_document_file_path(source),
            prefix="doc-",
        )
        assert delete_llm_cache is True
        await asyncio.sleep(0.03)
        return type("DeletionResult", (), {"status": "success"})()

    deps["lightrag"].adelete_by_doc_id = AsyncMock(side_effect=slow_delete)

    async def ingest() -> dict:
        return await engine.aingest_file(source, replace=False)

    results = await asyncio.gather(ingest(), ingest())
    assert len(results) == 2
    # Cleanup must have been called exactly once (not twice).
    assert deps["lightrag"].adelete_by_doc_id.await_count == 1


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


async def test_reingest_hash_check_runs_off_event_loop(tmp_path: Path, monkeypatch) -> None:
    import dlightrag.core.ingestion.engine as engine_module

    source = tmp_path / "sample[mineru-iteP].pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.return_value = {
        "chunks_list": ["chunk-1"],
        "content_hash": _file_sha256_static(source),
        "status": "processed",
    }
    calls = []

    async def fake_to_thread(func, *args, **kwargs):
        calls.append(func)
        return func(*args, **kwargs)

    monkeypatch.setattr(engine_module.asyncio, "to_thread", fake_to_thread)

    await engine.aingest_file(source, replace=False)

    assert engine_module._file_sha256 in calls


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
    failed_status = {
        "chunks_list": [],
        "content_hash": current_hash,
        "status": "failed",
    }
    deps["stores"].get_doc_status.side_effect = [
        failed_status,
        failed_status,
        {"chunks_list": ["chunk-a"], "content_hash": current_hash, "status": "processed"},
    ]

    result = await engine.aingest_file(source, replace=False)

    assert result.get("source_kind") != "skipped"
    deps["lightrag"].apipeline_enqueue_documents.assert_awaited_once()


async def test_sidecar_image_prework_runs_off_event_loop(tmp_path: Path, monkeypatch) -> None:
    import json

    import dlightrag.core.ingestion.engine as engine_module

    artifact_dir = tmp_path / "sample.parsed"
    artifact_dir.mkdir()
    (artifact_dir / "sample.blocks.jsonl").write_text("{}\n", encoding="utf-8")
    image_path = artifact_dir / "chart.png"
    Image.new("RGB", (128, 128), color=(255, 0, 0)).save(image_path)
    (artifact_dir / "sample.drawings.json").write_text(
        json.dumps(
            {
                "drawings": {
                    "im-1": {
                        "path": "chart.png",
                        "llm_analyze_result": {"status": "success"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    engine, deps = _make_engine()
    deps["stores"].overwrite_chunk_vectors = AsyncMock()
    deps["multimodal_embedder"].embed_index_images = AsyncMock(return_value=[[0.1, 0.2]])
    calls = []

    async def fake_to_thread(func, *args, **kwargs):
        calls.append(func.__name__)
        return func(*args, **kwargs)

    monkeypatch.setattr(engine_module.asyncio, "to_thread", fake_to_thread)

    await engine._overwrite_sidecar_image_vectors(
        doc_id="doc-1",
        sidecar_location=artifact_dir.as_uri(),
        chunk_ids={"doc-1-mm-drawing-000"},
    )

    assert "_image_dims" in calls
    assert "_open_rgb_image" in calls
    deps["stores"].overwrite_chunk_vectors.assert_awaited_once()


async def test_sidecar_image_embed_failure_is_non_fatal(tmp_path: Path, monkeypatch) -> None:
    import json

    import dlightrag.core.ingestion.engine as engine_module

    artifact_dir = tmp_path / "sample.parsed"
    artifact_dir.mkdir()
    (artifact_dir / "sample.blocks.jsonl").write_text("{}\n", encoding="utf-8")
    Image.new("RGB", (128, 128), color=(0, 128, 255)).save(artifact_dir / "chart.png")
    (artifact_dir / "sample.drawings.json").write_text(
        json.dumps(
            {
                "drawings": {
                    "im-1": {
                        "path": "chart.png",
                        "llm_analyze_result": {"status": "success"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    engine, deps = _make_engine()
    deps["stores"].overwrite_chunk_vectors = AsyncMock()
    deps["multimodal_embedder"].embed_index_images = AsyncMock(
        side_effect=RuntimeError("provider rejected oversized image")
    )

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(engine_module.asyncio, "to_thread", fake_to_thread)

    # A single unembeddable image must not raise or fail the whole document.
    await engine._overwrite_sidecar_image_vectors(
        doc_id="doc-1",
        sidecar_location=artifact_dir.as_uri(),
        chunk_ids={"doc-1-mm-drawing-000"},
    )

    deps["stores"].overwrite_chunk_vectors.assert_not_awaited()


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
