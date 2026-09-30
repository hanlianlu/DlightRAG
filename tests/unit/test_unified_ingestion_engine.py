# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for unified LightRAG sidecar ingestion engine."""

import asyncio
import hashlib
from collections.abc import Iterator, Mapping
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from lightrag.base import DocStatus
from lightrag.parser.routing import FilenameParserHintError
from lightrag.utils import compute_mdhash_id
from lightrag.utils_pipeline import (
    compute_text_content_hash,
    doc_status_parse_failure_fields,
    normalize_document_file_path,
)
from PIL import Image

from dlightrag.engine.dependencies import ParserUnavailableError, classify_transient_dependency
from dlightrag.engine.rag.corpus.ingestion.document_embedding import (
    DocumentEmbeddingInput,
    DocumentEmbeddingTrace,
    DocumentEmbeddingVector,
)
from dlightrag.engine.rag.corpus.ingestion.engine import (
    _FINALIZATION_COMPLETE_KEY,
    PreparedIngestFile,
    UnifiedIngestionEngine,
    _prepare_ingest_item,
    _raw_path_source_uri,
)
from dlightrag.engine.rag.corpus.ingestion.errors import ParserInputPlacementError
from dlightrag.engine.rag.retrieval.metadata_fields import PARSER_INPUT_SHA256_FIELD


def _sha256(content: bytes) -> str:
    """The parser-input digest ingestion records with a document."""
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _lightrag_content_hash(content: bytes) -> str:
    """LightRAG's own ``content_hash``: an MD5 of the parsed text, not of the file."""
    return compute_text_content_hash(content.decode("utf-8", "replace"))


def _recorded(content: bytes, **fields: Any) -> dict[str, Any]:
    """A finalized document's metadata row, recorded from these parser-input bytes."""
    return {_FINALIZATION_COMPLETE_KEY: True, PARSER_INPUT_SHA256_FIELD: _sha256(content), **fields}


_PARSER_INPUT_ROOTS: list[Path] = []


@pytest.fixture(autouse=True)
def parser_input_root(tmp_path: Path) -> Iterator[Path]:
    """The Workspace's directory in LightRAG's INPUT_DIR, where engines place parser inputs."""
    root = tmp_path / "corpus" / "default"
    _PARSER_INPUT_ROOTS.append(root)
    try:
        yield root
    finally:
        _PARSER_INPUT_ROOTS.remove(root)


def _one_file(
    path: Path,
    *,
    workspace: str = "default",
    source_uri: str | None = None,
    download_locator: str | None = None,
    display_filename: str | None = None,
    source_uri_explicit: bool | None = None,
    download_locator_explicit: bool | None = None,
    title: str | None = None,
    author: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> PreparedIngestFile:
    """One raw file; provenance a caller names counts as explicit unless told otherwise."""
    return PreparedIngestFile(
        parser_path=path,
        source_uri=source_uri or _raw_path_source_uri(path, workspace=workspace),
        download_locator=download_locator or str(path.resolve()),
        display_filename=display_filename,
        title=title,
        author=author,
        metadata=metadata,
        source_uri_explicit=(
            source_uri is not None if source_uri_explicit is None else source_uri_explicit
        ),
        download_locator_explicit=(
            download_locator is not None
            if download_locator_explicit is None
            else download_locator_explicit
        ),
        display_filename_explicit=display_filename is not None,
    )


async def _ingest_one(
    engine: UnifiedIngestionEngine, path: Path, *, replace: bool = False, **fields: Any
) -> dict[str, Any]:
    """Ingest one file as a one-document batch and return its document result."""
    batch = await engine.aingest_files(
        [_one_file(path, workspace=engine._workspace, **fields)], replace=replace
    )
    assert not batch["errors"], batch["errors"]
    (result,) = batch["results"]
    return result


def _make_engine(**overrides):
    lightrag = AsyncMock()
    lightrag.apipeline_enqueue_documents.return_value = "track-1"
    lightrag.adelete_by_doc_id.return_value = SimpleNamespace(status="success")
    stores = AsyncMock()
    stores.fetch_chunk_contents.return_value = []
    stores.get_doc_status.return_value = {
        "status": "processed",
        "chunks_list": ["chunk-a"],
        "content_hash": _lightrag_content_hash(b"abc"),
    }
    stores.get_full_doc_statuses.side_effect = lambda doc_ids: {
        doc_id: {
            "status": "processed",
            "chunks_list": ["chunk-a"],
            "content_hash": _lightrag_content_hash(b"abc"),
        }
        for doc_id in doc_ids
    }
    stores.get_full_doc.return_value = {
        "parse_engine": "mineru",
        "process_options": "iteP",
        "chunk_options": {"paragraph_semantic": {"chunk_token_size": 2000}},
        "sidecar_location": "file:///tmp/sample.parsed/",
    }
    document_embedder = AsyncMock()
    document_embedder.image_enabled = True
    document_embedder.dimension = 3
    document_embedder.aembed_documents.return_value = (
        [],
        DocumentEmbeddingTrace(fused=0, text=0, fused_to_text_fallback=0, failed=0),
    )
    defaults = {
        "lightrag": lightrag,
        "stores": stores,
        "metadata_index": AsyncMock(),
        "document_embedder": document_embedder,
        "workspace": "default",
        "input_root": _PARSER_INPUT_ROOTS[-1],
        "parser_rules": "docx:native-iteP,*:mineru-iteP",
        "chunk_options": {},
    }
    defaults.update(overrides)
    defaults["metadata_index"].get.return_value = None
    return UnifiedIngestionEngine(**defaults), defaults


async def test_replace_false_keeps_idempotent_skip(tmp_path: Path) -> None:
    content = b"%PDF-1.4"
    source = tmp_path / "sample.pdf"
    source.write_bytes(content)
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.return_value = {
        "chunks_list": ["chunk-a"],
        "content_hash": _lightrag_content_hash(content),
        "status": "processed",
    }
    deps["metadata_index"].get.return_value = _recorded(content)

    result = await _ingest_one(engine, source, replace=False)

    assert result["source_kind"] == "skipped"
    deps["lightrag"].adelete_by_doc_id.assert_not_awaited()
    deps["lightrag"].apipeline_enqueue_documents.assert_not_awaited()


async def test_replace_true_bypasses_idempotent_skip(tmp_path: Path) -> None:
    content = b"%PDF-1.4"
    source = tmp_path / "sample.pdf"
    source.write_bytes(content)
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.side_effect = [
        {
            "chunks_list": ["old-chunk"],
            "content_hash": _lightrag_content_hash(content),
            "status": "processed",
        },
        {
            "chunks_list": ["old-chunk"],
            "content_hash": _lightrag_content_hash(content),
            "status": "processed",
        },
        {
            "chunks_list": ["new-chunk"],
            "content_hash": _lightrag_content_hash(content),
            "status": "processed",
        },
    ]

    result = await _ingest_one(engine, source, replace=True)

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
        {
            "chunks_list": ["old-chunk"],
            "content_hash": _lightrag_content_hash(content),
            "status": "processed",
        },
        {
            "chunks_list": ["old-chunk"],
            "content_hash": _lightrag_content_hash(content),
            "status": "processed",
        },
        {
            "chunks_list": ["new-chunk"],
            "content_hash": _lightrag_content_hash(content),
            "status": "processed",
        },
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

    await _ingest_one(engine, source, replace=False)

    kwargs = deps["lightrag"].apipeline_enqueue_documents.await_args.kwargs
    assert kwargs["docs_format"] == "pending_parse"
    assert "ids" not in kwargs
    assert kwargs["parse_engine"] == ["mineru"]
    assert kwargs["process_options"] == ["iteP"]
    deps["lightrag"].apipeline_process_enqueue_documents.assert_awaited_once()
    assert deps["metadata_index"].upsert.await_count == 3
    assert deps["metadata_index"].upsert.await_args_list[0].args[1] == {
        _FINALIZATION_COMPLETE_KEY: False
    }


async def test_ingest_waits_when_processing_is_queued_behind_busy_owner(
    tmp_path: Path,
) -> None:
    source = tmp_path / "queued.pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()
    doc_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    statuses: dict[str, dict[str, object]] = {}
    processing_call_returned = asyncio.Event()

    async def enqueue(**_kwargs: object) -> str:
        statuses[doc_id] = {"status": "pending", "chunks_list": []}
        return "track-queued"

    async def queue_behind_busy_owner() -> None:
        processing_call_returned.set()

    deps["stores"].get_doc_status.side_effect = lambda current: statuses.get(current)
    deps["stores"].get_full_doc_statuses.side_effect = lambda doc_ids: {
        doc_id: statuses[doc_id] for doc_id in doc_ids if doc_id in statuses
    }
    deps["lightrag"].apipeline_enqueue_documents.side_effect = enqueue
    deps["lightrag"].apipeline_process_enqueue_documents.side_effect = queue_behind_busy_owner

    task = asyncio.create_task(engine.aingest_files([source]))
    await asyncio.wait_for(processing_call_returned.wait(), timeout=1)
    await asyncio.sleep(0)

    assert not task.done()

    statuses[doc_id] = {"status": "processed", "chunks_list": ["chunk-queued"]}
    result = await asyncio.wait_for(task, timeout=1)

    assert result["processed"] == 1
    assert result["errors"] == []
    assert result["results"][0]["chunks"] == ["chunk-queued"]


async def test_ingest_redrives_queue_after_busy_owner_exits_abnormally(
    tmp_path: Path,
) -> None:
    source = tmp_path / "retry-queued.pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()
    doc_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    statuses: dict[str, dict[str, object]] = {}
    processing_calls = 0

    async def enqueue(**_kwargs: object) -> str:
        statuses[doc_id] = {"status": "pending", "chunks_list": []}
        return "track-retry-queued"

    async def process_or_observe_busy_owner() -> None:
        nonlocal processing_calls
        processing_calls += 1
        if processing_calls == 2:
            statuses[doc_id] = {
                "status": "processed",
                "chunks_list": ["chunk-retry-queued"],
            }

    deps["stores"].get_doc_status.side_effect = lambda current: statuses.get(current)
    deps["stores"].get_full_doc_statuses.side_effect = lambda doc_ids: {
        doc_id: statuses[doc_id] for doc_id in doc_ids if doc_id in statuses
    }
    deps["lightrag"].apipeline_enqueue_documents.side_effect = enqueue
    deps["lightrag"].apipeline_process_enqueue_documents.side_effect = process_or_observe_busy_owner

    result = await asyncio.wait_for(engine.aingest_files([source]), timeout=1)

    assert processing_calls == 2
    assert result["processed"] == 1
    assert result["errors"] == []
    assert result["results"][0]["chunks"] == ["chunk-retry-queued"]


async def test_ingest_does_not_wait_forever_on_unknown_pipeline_status(
    tmp_path: Path,
) -> None:
    source = tmp_path / "unknown-status.pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()
    doc_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    unknown_status = {"status": "future-state", "chunks_list": []}

    deps["stores"].get_doc_status.return_value = unknown_status
    deps["stores"].get_full_doc_statuses.side_effect = lambda _doc_ids: {doc_id: unknown_status}

    result = await asyncio.wait_for(engine.aingest_files([source]), timeout=1)

    assert result["processed"] == 0
    assert result["errors"] == ["unknown-status.pdf: document processing failed"]


async def test_document_ingest_persists_lightrag_archived_source_locator(
    tmp_path: Path,
) -> None:
    source = tmp_path / "inputs" / "default" / "report.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"%PDF-1.4")
    archived = source.parent / "__parsed__" / source.name
    engine, deps = _make_engine()

    async def archive_source() -> None:
        archived.parent.mkdir()
        source.replace(archived)

    deps["lightrag"].apipeline_process_enqueue_documents.side_effect = archive_source

    await _ingest_one(engine, source, replace=False)

    _, saved = deps["metadata_index"].upsert.await_args.args
    assert saved["download_locator"] == str(archived.resolve())


async def test_document_ingest_reports_a_failed_pipeline_per_document(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.side_effect = [
        None,
        {
            "status": DocStatus.FAILED,
            "chunks_list": [],
            "content_hash": None,
            "content_summary": "PDF parser failed on page 3",
            "error_msg": None,
        },
    ]

    batch = await engine.aingest_files([_one_file(source)], replace=False)

    # A failed document is its own outcome, never an exception for its batch;
    # the parser's reason stays in the log.
    assert batch == {
        "processed": 0,
        "errors": ["report.pdf: document processing failed"],
        "results": [],
    }
    assert "PDF parser failed on page 3" in caplog.text
    assert deps["metadata_index"].upsert.await_count == 1
    deps["stores"].overwrite_chunk_vectors.assert_not_awaited()


async def test_a_parser_outage_ends_the_batch_once_every_document_has_settled(
    tmp_path: Path,
) -> None:
    """The typed outage leaves the batch; a document that parsed is published first.

    Finalization turns LightRAG's recorded verdict back into the typed parser error,
    so the caller can retry the batch later instead of failing a document that may
    well parse then. A document that failed for its own reason does not change
    that: it is retried with the batch.
    """
    names = ("parsed.pdf", "rejected.pdf", "stalled.pdf")
    paths = [tmp_path / name for name in names]
    for path in paths:
        path.write_bytes(b"%PDF-" + path.stem.encode())
    engine, deps = _make_engine()
    parsed_id, rejected_id, stalled_id = (
        compute_mdhash_id(normalize_document_file_path(path), prefix="doc-") for path in paths
    )
    outage_fields, _ = doc_status_parse_failure_fields(
        ParserUnavailableError(),
        status_doc={"content_summary": "", "metadata": {}},
        engine_hint="mineru",
    )
    statuses: dict[str, dict[str, Any]] = {}

    async def process() -> None:
        statuses[parsed_id] = {"status": DocStatus.PROCESSED, "chunks_list": ["chunk-parsed"]}
        statuses[rejected_id] = {
            "status": DocStatus.FAILED,
            "chunks_list": [],
            "error_msg": "MinerU rejected the document: HTTP 422",
        }
        statuses[stalled_id] = {"status": DocStatus.FAILED, "chunks_list": [], **outage_fields}

    deps["stores"].get_doc_status.side_effect = statuses.get
    deps["lightrag"].apipeline_process_enqueue_documents.side_effect = process

    with pytest.raises(ParserUnavailableError) as raised:
        await engine.aingest_files([_one_file(path) for path in paths], replace=False)

    assert classify_transient_dependency(raised.value) == "parser"
    published = {
        doc_id: row[_FINALIZATION_COMPLETE_KEY]
        for doc_id, row in (call.args for call in deps["metadata_index"].upsert.await_args_list)
        if row.get(_FINALIZATION_COMPLETE_KEY)
    }
    assert published == {parsed_id: True}


async def test_document_ingest_preserves_lightrag_parser_engine_params(
    tmp_path: Path,
) -> None:
    source = tmp_path / "sample.[mineru(page_range=1-3)-iteP].pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()

    await _ingest_one(engine, source, replace=False)

    kwargs = deps["lightrag"].apipeline_enqueue_documents.await_args.kwargs
    assert kwargs["parse_engine"] == ["mineru(page_range=1-3)"]


@pytest.mark.parametrize(
    "fault_phase",
    [
        pytest.param("visual", id="required-visual-fusion"),
        pytest.param("bm25", id="required-bm25-labels"),
        pytest.param("metadata_source_readiness", id="metadata-source-and-readiness-marker"),
    ],
)
async def test_required_product_document_finalizer_fault_replays_before_readiness(
    fault_phase: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed required finalizer cannot publish; retry replays the same document."""
    engine, deps = _make_engine()
    metadata = {
        "filename": "report.pdf",
        "source_uri": "local://default/report.pdf",
        "download_locator": "/shared/default/report.pdf",
        _FINALIZATION_COMPLETE_KEY: False,
    }
    visual = AsyncMock()
    bm25 = AsyncMock()
    monkeypatch.setattr(engine, "_overwrite_sidecar_image_vectors", visual)
    monkeypatch.setattr(engine, "_label_bm25_languages", bm25)
    if fault_phase == "visual":
        visual.side_effect = [RuntimeError("visual finalizer unavailable"), None]
    elif fault_phase == "bm25":
        bm25.side_effect = [RuntimeError("BM25 finalizer unavailable"), None]
    else:
        deps["metadata_index"].upsert.side_effect = [
            RuntimeError("metadata/source/readiness commit unavailable"),
            None,
        ]

    with pytest.raises(RuntimeError):
        await engine._finalize_ingested_document(
            doc_id="doc-report",
            metadata_record=metadata,
            parse_engine="mineru",
            process_options="iteP",
        )

    # No compensating write marks an unfinished document ready. The exact same
    # idempotent finalization can then complete without replaying LightRAG ingest.
    result = await engine._finalize_ingested_document(
        doc_id="doc-report",
        metadata_record=metadata,
        parse_engine="mineru",
        process_options="iteP",
    )

    assert result["doc_id"] == "doc-report"
    assert deps["lightrag"].apipeline_enqueue_documents.await_count == 0
    committed = deps["metadata_index"].upsert.await_args.args[1]
    assert committed[_FINALIZATION_COMPLETE_KEY] is True
    assert committed["source_uri"] == "local://default/report.pdf"


async def test_document_ingest_labels_bm25_chunk_languages(tmp_path: Path) -> None:
    source = tmp_path / "sample[mineru-iteP].pdf"
    source.write_bytes(b"%PDF-1.4")

    class FakeClassifier:
        def detect(self, content: str) -> str:
            return {"现金流 风险": "zh", "risk factors": "en"}.get(content, "simple")

    engine, deps = _make_engine(bm25_language_classifier=FakeClassifier())
    deps["stores"].get_doc_status.return_value = {
        "chunks_list": ["chunk-zh", "chunk-en"],
        "content_hash": _lightrag_content_hash(b"abc"),
        "status": "processed",
    }
    deps["stores"].fetch_chunk_contents.return_value = [
        {"id": "chunk-zh", "content": "现金流 风险"},
        {"id": "chunk-en", "content": "risk factors"},
    ]

    await _ingest_one(engine, source, replace=False)

    deps["stores"].fetch_chunk_contents.assert_awaited_once_with(["chunk-zh", "chunk-en"])
    deps["stores"].update_chunk_bm25_languages.assert_awaited_once_with(
        {"chunk-zh": "zh", "chunk-en": "en"}
    )


async def test_batch_document_ingest_uses_lightrag_staged_pipeline(
    tmp_path: Path, parser_input_root: Path
) -> None:
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
        {
            "chunks_list": ["chunk-docx"],
            "content_hash": _lightrag_content_hash(b"docx"),
            "status": "processed",
        },
        {
            "chunks_list": ["chunk-pdf"],
            "content_hash": _lightrag_content_hash(b"pdf"),
            "status": "processed",
        },
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
    assert kwargs["file_paths"] == [
        str(parser_input_root / docx.name),
        str(parser_input_root / pdf.name),
    ]
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
        {
            "chunks_list": ["chunk-docx"],
            "content_hash": _lightrag_content_hash(b"docx"),
            "status": "processed",
        },
        {
            "chunks_list": ["chunk-pdf"],
            "content_hash": _lightrag_content_hash(b"pdf"),
            "status": "processed",
        },
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


async def test_prepared_batch_uses_explicit_download_locator(
    tmp_path: Path, parser_input_root: Path
) -> None:
    parser_source = tmp_path / "report__s3_abcd1234.pdf"
    parser_source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.side_effect = [
        None,
        {
            "chunks_list": ["chunk-report"],
            "content_hash": _lightrag_content_hash(b"pdf"),
            "status": "processed",
        },
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
    assert kwargs["file_paths"] == [str(parser_input_root / parser_source.name)]
    _, saved = deps["metadata_index"].upsert.await_args.args
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

    first_item = _prepare_ingest_item(first, workspace="finance_team")
    second_item = _prepare_ingest_item(second, workspace="finance_team")

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

    await _ingest_one(
        engine,
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
        "content_hash": _lightrag_content_hash(content),
        "status": "processed",
    }
    deps["metadata_index"].get.return_value = _recorded(content)
    prepare_metadata = MagicMock(wraps=engine._prepare_metadata_record)
    monkeypatch.setattr(engine, "_prepare_metadata_record", prepare_metadata)

    result = await _ingest_one(
        engine,
        source,
        source_uri="local://default/docs/sample.pdf",
        download_locator=str(source),
        title="Updated title",
    )

    assert result["source_kind"] == "metadata_updated"
    assert prepare_metadata.call_args.kwargs["source_uri"] == ("local://default/docs/sample.pdf")
    assert prepare_metadata.call_args.kwargs["download_locator"] == str(source)


@pytest.mark.parametrize(
    ("title", "expected_source_kind"),
    [
        pytest.param(None, "skipped", id="unchanged_metadata"),
        pytest.param("Updated title", "metadata_updated", id="updated_metadata"),
    ],
)
async def test_single_hash_match_bypasses_parser_directives(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    title: str | None,
    expected_source_kind: str,
) -> None:
    content = b"%PDF-1.4"
    source = tmp_path / "sample.pdf"
    source.write_bytes(content)
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.return_value = {
        "chunks_list": ["chunk-a"],
        "content_hash": _lightrag_content_hash(content),
        "status": "processed",
    }
    deps["metadata_index"].get.return_value = _recorded(content)

    def fail_parser_directives(_path: Path) -> tuple[str, str, dict[str, object] | None]:
        raise AssertionError("parser directives should not be resolved for hash-match fast path")

    monkeypatch.setattr(engine, "_parser_directives_for", fail_parser_directives)

    result = await _ingest_one(engine, source, replace=False, title=title)

    assert result["source_kind"] == expected_source_kind


async def test_batch_hash_match_skip_does_not_resolve_invalid_parser_directives(
    tmp_path: Path,
) -> None:
    content = b"%PDF-1.4"
    source = tmp_path / "bad.[unknown-iteP].pdf"
    source.write_bytes(content)
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.return_value = {
        "chunks_list": ["chunk-a"],
        "content_hash": _lightrag_content_hash(content),
        "status": "processed",
    }
    deps["metadata_index"].get.return_value = _recorded(content)

    result = await engine.aingest_files([source], replace=False)
    single_result = await _ingest_one(engine, source, replace=False)

    assert result == {
        "processed": 1,
        "errors": [],
        "results": [
            {
                "doc_id": compute_mdhash_id(
                    normalize_document_file_path(source),
                    prefix="doc-",
                ),
                "source_kind": "skipped",
                "reason": "content_hash_match",
                "chunks": ["chunk-a"],
            }
        ],
    }
    assert result["results"][0] == single_result
    deps["metadata_index"].upsert.assert_not_awaited()
    deps["lightrag"].apipeline_enqueue_documents.assert_not_awaited()


async def test_batch_replace_validates_all_enqueue_candidates_before_cleanup(
    tmp_path: Path,
) -> None:
    good = tmp_path / "good.pdf"
    bad = tmp_path / "bad.[unknown-iteP].pdf"
    good.write_bytes(b"%PDF-1.4 good")
    bad.write_bytes(b"%PDF-1.4 bad")
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.side_effect = [
        {
            "chunks_list": ["chunk-good"],
            "content_hash": _lightrag_content_hash(b"good"),
            "status": "processed",
        },
        {
            "chunks_list": ["chunk-bad"],
            "content_hash": _lightrag_content_hash(b"bad"),
            "status": "processed",
        },
    ]

    with pytest.raises(FilenameParserHintError):
        await engine.aingest_files([good, bad], replace=True)

    deps["lightrag"].adelete_by_doc_id.assert_not_awaited()
    deps["metadata_index"].delete.assert_not_awaited()
    deps["metadata_index"].upsert.assert_not_awaited()
    deps["lightrag"].apipeline_enqueue_documents.assert_not_awaited()
    deps["lightrag"].apipeline_process_enqueue_documents.assert_not_awaited()


async def test_batch_hash_match_metadata_update_waits_for_enqueue_validation(
    tmp_path: Path,
) -> None:
    content = b"%PDF-1.4"
    first = tmp_path / "first.pdf"
    bad = tmp_path / "bad.[unknown-iteP].pdf"
    first.write_bytes(content)
    bad.write_bytes(b"%PDF-1.4 bad")
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.side_effect = [
        {
            "chunks_list": ["chunk-first"],
            "content_hash": _lightrag_content_hash(content),
            "status": "processed",
        },
        None,
    ]
    deps["metadata_index"].get.return_value = {
        "filename": "first.pdf",
        "filename_stem": "first",
        "file_path": str(first),
        "source_uri": "local://default/first.pdf",
        "download_locator": str(first),
        "file_extension": "pdf",
        "title": "Old title",
        "custom_metadata": {},
    }

    with pytest.raises(FilenameParserHintError):
        await engine.aingest_files(
            [
                PreparedIngestFile(
                    parser_path=first,
                    source_uri="local://default/first.pdf",
                    download_locator=str(first),
                    title="Updated title",
                ),
                bad,
            ],
            replace=False,
        )

    deps["metadata_index"].get.assert_awaited_once()
    deps["metadata_index"].upsert.assert_not_awaited()
    deps["lightrag"].apipeline_enqueue_documents.assert_not_awaited()
    deps["lightrag"].apipeline_process_enqueue_documents.assert_not_awaited()


async def test_failed_document_cleanup_requires_documented_delete_success(
    tmp_path: Path,
) -> None:
    source = tmp_path / "failed.pdf"
    source.write_bytes(b"%PDF-1.4 failed")
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.return_value = {
        "chunks_list": [],
        "content_hash": None,
        "status": "failed",
        "error_msg": "parser failed",
    }
    deps["stores"].get_full_doc.return_value = {"sidecar_location": None}
    deps["lightrag"].adelete_by_doc_id.return_value = None

    with pytest.raises(RuntimeError, match="deletion was not acknowledged"):
        await engine.aingest_files([source], replace=False)

    deps["metadata_index"].delete.assert_not_awaited()
    assert deps["metadata_index"].upsert.await_args.args[1] == {_FINALIZATION_COMPLETE_KEY: False}
    deps["lightrag"].apipeline_enqueue_documents.assert_not_awaited()


async def test_failed_document_ingest_fails_closed_when_status_snapshot_disappears(
    tmp_path: Path,
) -> None:
    source = tmp_path / "failed.pdf"
    source.write_bytes(b"%PDF-1.4 failed")
    engine, deps = _make_engine()
    failed_status = {"chunks_list": [], "content_hash": None, "status": "failed"}
    deps["stores"].get_doc_status.side_effect = [failed_status, None]

    with pytest.raises(RuntimeError, match="status snapshot is unavailable"):
        await engine.aingest_files([source], replace=False)

    deps["lightrag"].adelete_by_doc_id.assert_not_awaited()
    deps["metadata_index"].delete.assert_not_awaited()
    assert deps["metadata_index"].upsert.await_args.args[1] == {_FINALIZATION_COMPLETE_KEY: False}


async def test_failed_document_retry_cancellation_restores_discoverability(
    tmp_path: Path,
) -> None:
    source = tmp_path / "failed.pdf"
    source.write_bytes(b"%PDF-1.4 failed")
    engine, deps = _make_engine()
    original_status = {
        "chunks_list": [],
        "content_hash": None,
        "status": "failed",
        "error_msg": "parser failed",
    }
    original_metadata = {
        "filename": "failed.pdf",
        "source_uri": "local://default/failed.pdf",
        "download_locator": str(source),
        "title": "Preserved title",
        "custom_metadata": {"department": "finance"},
    }
    deps["stores"].get_doc_status.side_effect = [original_status, original_status, None]
    deps["stores"].get_full_doc.return_value = {"sidecar_location": None}
    deps["metadata_index"].get.return_value = original_metadata
    deps["lightrag"].adelete_by_doc_id.return_value = SimpleNamespace(status="success")
    deps["lightrag"].apipeline_enqueue_documents.side_effect = asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await engine.aingest_files([source], replace=False)

    doc_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    deps["stores"].doc_status.upsert.assert_not_awaited()
    first_metadata_write = deps["metadata_index"].upsert.await_args_list[0]
    assert first_metadata_write.args == (
        doc_id,
        {"_dlightrag_finalization_complete": False},
    )


async def test_concurrent_single_file_replacements_serialize_cleanup(
    tmp_path: Path,
) -> None:
    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()
    doc_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    statuses = {doc_id: {"status": "processed", "chunks_list": ["old"]}}
    deps["stores"].get_doc_status.side_effect = lambda current: statuses.get(current)
    deps["stores"].get_full_doc.return_value = {"sidecar_location": None}
    active_deletes = 0
    max_active_deletes = 0

    async def delete(current: str, **_kwargs: object) -> SimpleNamespace:
        nonlocal active_deletes, max_active_deletes
        active_deletes += 1
        max_active_deletes = max(max_active_deletes, active_deletes)
        await asyncio.sleep(0.01)
        statuses.pop(current, None)
        active_deletes -= 1
        return SimpleNamespace(status="success")

    async def process() -> None:
        statuses[doc_id] = {"status": "processed", "chunks_list": ["new"]}

    deps["lightrag"].adelete_by_doc_id.side_effect = delete
    deps["lightrag"].apipeline_process_enqueue_documents.side_effect = process

    first, second = await asyncio.gather(
        _ingest_one(engine, source, replace=True),
        _ingest_one(engine, source, replace=True),
    )

    assert first["doc_id"] == doc_id
    assert second["doc_id"] == doc_id
    assert max_active_deletes == 1
    assert deps["lightrag"].adelete_by_doc_id.await_count == 2


async def test_delete_time_cancellation_after_status_delete_does_not_restore_zombie(
    tmp_path: Path,
) -> None:
    source = tmp_path / "failed.pdf"
    source.write_bytes(b"%PDF-1.4 failed")
    engine, deps = _make_engine()
    doc_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    status = {"status": "failed", "chunks_list": ["stale"], "content_hash": None}
    statuses = {doc_id: dict(status)}
    deps["stores"].get_doc_status.side_effect = lambda current: statuses.get(current)
    deps["stores"].get_full_doc.return_value = {"sidecar_location": None}

    async def cancel_delete(current: str, **_kwargs: object) -> None:
        statuses.pop(current, None)
        raise asyncio.CancelledError

    async def upsert(rows: dict[str, dict[str, object]]) -> None:
        statuses.update({key: dict(value) for key, value in rows.items()})

    deps["lightrag"].adelete_by_doc_id.side_effect = cancel_delete
    deps["stores"].doc_status.upsert.side_effect = upsert

    with pytest.raises(asyncio.CancelledError):
        await engine.aingest_files([source])

    assert doc_id not in statuses
    deps["stores"].doc_status.upsert.assert_not_awaited()
    deps["lightrag"].apipeline_enqueue_documents.assert_not_awaited()


async def test_cancellation_after_processing_commit_keeps_upstream_processed_and_hidden(
    tmp_path: Path,
) -> None:
    source = tmp_path / "failed.pdf"
    source.write_bytes(b"%PDF-1.4 failed")
    engine, deps = _make_engine()
    doc_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    original = {"status": "failed", "chunks_list": [], "content_hash": None}
    statuses = {doc_id: dict(original)}
    deps["stores"].get_doc_status.side_effect = lambda current: statuses.get(current)
    deps["stores"].get_full_doc.return_value = {"sidecar_location": None}

    async def upsert_status(rows: dict[str, dict[str, object]]) -> None:
        statuses.update({key: dict(value) for key, value in rows.items()})

    deps["stores"].doc_status.upsert.side_effect = upsert_status

    async def delete(current: str, **_kwargs: object) -> SimpleNamespace:
        statuses.pop(current, None)
        return SimpleNamespace(status="success")

    async def process() -> None:
        statuses[doc_id] = {"status": "processed", "chunks_list": ["new-chunk"]}

    metadata_writes = 0

    async def upsert_metadata(*_args: object) -> None:
        nonlocal metadata_writes
        metadata_writes += 1
        if metadata_writes == 3:
            raise asyncio.CancelledError

    deps["lightrag"].adelete_by_doc_id.side_effect = delete
    deps["lightrag"].apipeline_process_enqueue_documents.side_effect = process
    deps["metadata_index"].upsert.side_effect = upsert_metadata

    with pytest.raises(asyncio.CancelledError):
        await engine.aingest_files([source])

    assert statuses[doc_id]["status"] == "processed"
    assert statuses[doc_id]["chunks_list"] == ["new-chunk"]
    assert "error_msg" not in statuses[doc_id]
    deps["stores"].doc_status.upsert.assert_not_awaited()
    assert (
        deps["metadata_index"].upsert.await_args_list[0].args[1][_FINALIZATION_COMPLETE_KEY]
        is False
    )


async def test_remote_locator_replacement_deletes_old_metadata_only_after_new_commit(
    tmp_path: Path,
) -> None:
    source = tmp_path / "renamed.pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()
    new_doc_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    old_doc_id = "doc-old-locator"
    statuses = {
        old_doc_id: {"status": "processed", "chunks_list": ["old-chunk"]},
    }
    metadata = {
        old_doc_id: {
            "download_locator": "s3://bucket/report.pdf",
            "source_uri": "s3://bucket/report.pdf",
        }
    }
    deps["stores"].get_doc_status.side_effect = lambda current: statuses.get(current)
    deps["stores"].get_full_doc.return_value = {"sidecar_location": None}
    deps["metadata_index"].get.side_effect = lambda current: metadata.get(current)

    async def delete(current: str, **_kwargs: object) -> SimpleNamespace:
        statuses.pop(current, None)
        return SimpleNamespace(status="success")

    async def process() -> None:
        statuses[new_doc_id] = {"status": "processed", "chunks_list": ["new-chunk"]}

    async def delete_metadata(current: str) -> None:
        assert statuses.get(new_doc_id, {}).get("status") == "processed"
        metadata.pop(current, None)

    deps["lightrag"].adelete_by_doc_id.side_effect = delete
    deps["lightrag"].apipeline_process_enqueue_documents.side_effect = process
    deps["metadata_index"].delete.side_effect = delete_metadata

    result = await engine.aingest_files(
        [
            PreparedIngestFile(
                parser_path=source,
                source_uri="s3://bucket/report.pdf",
                download_locator="s3://bucket/report.pdf",
                replacement_doc_ids=(old_doc_id,),
                replacement_ownership=(
                    (old_doc_id, "s3://bucket/report.pdf", "s3://bucket/report.pdf"),
                ),
            )
        ],
        replace=True,
    )

    assert result["processed"] == 1
    assert statuses[new_doc_id]["status"] == "processed"
    assert old_doc_id not in metadata
    deps["metadata_index"].delete.assert_awaited_once_with(old_doc_id)


async def test_batch_partial_cleanup_waits_for_enqueue_validation(
    tmp_path: Path,
) -> None:
    good = tmp_path / "good.pdf"
    bad = tmp_path / "bad.[unknown-iteP].pdf"
    good.write_bytes(b"%PDF-1.4 good")
    bad.write_bytes(b"%PDF-1.4 bad")
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.side_effect = [
        {
            "chunks_list": ["chunk-good"],
            "content_hash": _lightrag_content_hash(b"stale"),
            "status": "analyzing",
        },
        None,
    ]

    with pytest.raises(FilenameParserHintError):
        await engine.aingest_files([good, bad], replace=False)

    deps["lightrag"].adelete_by_doc_id.assert_not_awaited()
    deps["metadata_index"].delete.assert_not_awaited()
    deps["metadata_index"].upsert.assert_not_awaited()
    deps["lightrag"].apipeline_enqueue_documents.assert_not_awaited()
    deps["lightrag"].apipeline_process_enqueue_documents.assert_not_awaited()


async def test_single_hash_match_source_contract_change_updates_metadata(
    tmp_path: Path,
) -> None:
    content = b"%PDF-1.4"
    source = tmp_path / "sample.pdf"
    source.write_bytes(content)
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.return_value = {
        "chunks_list": ["chunk-a"],
        "content_hash": _lightrag_content_hash(content),
        "status": "processed",
    }
    deps["metadata_index"].get.return_value = {
        "filename": "sample.pdf",
        "filename_stem": "sample",
        "file_path": "https://cdn.example.com/old-sample.pdf",
        "source_uri": "bynder://asset/old",
        "download_locator": "https://cdn.example.com/old-sample.pdf",
        "file_extension": "pdf",
        "custom_metadata": {},
        _FINALIZATION_COMPLETE_KEY: True,
        PARSER_INPUT_SHA256_FIELD: _sha256(content),
    }

    result = await _ingest_one(
        engine,
        source,
        source_uri="bynder://asset/new",
        download_locator="https://cdn.example.com/new-sample.pdf",
        display_filename="renamed-sample.pdf",
        replace=False,
    )

    assert result == {
        "doc_id": compute_mdhash_id(normalize_document_file_path(source), prefix="doc-"),
        "source_kind": "metadata_updated",
        "reason": "content_hash_match",
        "chunks": ["chunk-a"],
    }
    deps["metadata_index"].upsert.assert_awaited_once()
    _, saved = deps["metadata_index"].upsert.await_args.args
    assert saved["filename"] == "renamed-sample.pdf"
    assert saved["filename_stem"] == "renamed-sample"
    assert saved["source_uri"] == "bynder://asset/new"
    assert saved["download_locator"] == "https://cdn.example.com/new-sample.pdf"
    deps["lightrag"].apipeline_enqueue_documents.assert_not_awaited()


async def test_single_hash_match_local_noop_checks_finalization_marker(
    tmp_path: Path,
) -> None:
    content = b"%PDF-1.4"
    source = tmp_path / "sample.pdf"
    source.write_bytes(content)
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.return_value = {
        "chunks_list": ["chunk-a"],
        "content_hash": _lightrag_content_hash(content),
        "status": "processed",
    }
    deps["metadata_index"].get.return_value = _recorded(content)

    result = await _ingest_one(engine, source, replace=False)

    assert result == {
        "doc_id": compute_mdhash_id(normalize_document_file_path(source), prefix="doc-"),
        "source_kind": "skipped",
        "reason": "content_hash_match",
        "chunks": ["chunk-a"],
    }
    deps["metadata_index"].get.assert_awaited_once()
    deps["metadata_index"].upsert.assert_not_awaited()
    deps["lightrag"].apipeline_enqueue_documents.assert_not_awaited()


async def test_single_hash_match_internal_local_contract_checks_finalization_marker(
    tmp_path: Path,
) -> None:
    content = b"%PDF-1.4"
    source = tmp_path / "sample.pdf"
    source.write_bytes(content)
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.return_value = {
        "chunks_list": ["chunk-a"],
        "content_hash": _lightrag_content_hash(content),
        "status": "processed",
    }
    deps["metadata_index"].get.return_value = _recorded(content)

    result = await _ingest_one(
        engine,
        source,
        source_uri=_raw_path_source_uri(source, workspace="default"),
        download_locator=str(source.resolve()),
        source_uri_explicit=False,
        download_locator_explicit=False,
        replace=False,
    )

    assert result == {
        "doc_id": compute_mdhash_id(normalize_document_file_path(source), prefix="doc-"),
        "source_kind": "skipped",
        "reason": "content_hash_match",
        "chunks": ["chunk-a"],
    }
    deps["metadata_index"].get.assert_awaited_once()
    deps["metadata_index"].upsert.assert_not_awaited()
    deps["lightrag"].apipeline_enqueue_documents.assert_not_awaited()


async def test_single_hash_match_explicit_default_source_contract_updates_metadata(
    tmp_path: Path,
) -> None:
    content = b"%PDF-1.4"
    source = tmp_path / "sample.pdf"
    source.write_bytes(content)
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.return_value = {
        "chunks_list": ["chunk-a"],
        "content_hash": _lightrag_content_hash(content),
        "status": "processed",
    }
    deps["metadata_index"].get.return_value = {
        "filename": "old-name.pdf",
        "filename_stem": "old-name",
        "file_path": "https://cdn.example.com/old-sample.pdf",
        "source_uri": "bynder://asset/old",
        "download_locator": "https://cdn.example.com/old-sample.pdf",
        "file_extension": "pdf",
        "custom_metadata": {},
        _FINALIZATION_COMPLETE_KEY: True,
        PARSER_INPUT_SHA256_FIELD: _sha256(content),
    }

    result = await _ingest_one(
        engine,
        source,
        source_uri=_raw_path_source_uri(source, workspace="default"),
        download_locator=str(source.resolve()),
        display_filename=source.name,
        replace=False,
    )

    assert result == {
        "doc_id": compute_mdhash_id(normalize_document_file_path(source), prefix="doc-"),
        "source_kind": "metadata_updated",
        "reason": "content_hash_match",
        "chunks": ["chunk-a"],
    }
    deps["metadata_index"].get.assert_awaited_once()
    deps["metadata_index"].upsert.assert_awaited_once()
    _, saved = deps["metadata_index"].upsert.await_args.args
    assert saved["filename"] == "sample.pdf"
    assert saved["filename_stem"] == "sample"
    assert saved["source_uri"] == _raw_path_source_uri(source, workspace="default")
    assert saved["download_locator"] == str(source.resolve())
    deps["lightrag"].apipeline_enqueue_documents.assert_not_awaited()


async def test_batch_metadata_only_update_preserves_source_contract_and_chunks(
    tmp_path: Path, monkeypatch
) -> None:
    content = b"%PDF-1.4"
    parser_source = tmp_path / "report__s3_abcd1234.pdf"
    parser_source.write_bytes(content)
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.return_value = {
        "chunks_list": ["chunk-report"],
        "content_hash": _lightrag_content_hash(content),
        "status": "processed",
    }
    deps["metadata_index"].get.return_value = _recorded(content)
    prepare_metadata = MagicMock(wraps=engine._prepare_metadata_record)
    monkeypatch.setattr(engine, "_prepare_metadata_record", prepare_metadata)

    result = await engine.aingest_files(
        [
            PreparedIngestFile(
                parser_path=parser_source,
                source_uri="bynder://asset/1",
                download_locator="https://cdn.example.com/assets/1.pdf",
                display_filename="report.pdf",
                title="Updated title",
                author="Updated author",
                metadata={"category": "finance"},
            )
        ],
        replace=False,
    )

    assert result["processed"] == 1
    assert result["errors"] == []
    assert result["results"] == [
        {
            "doc_id": compute_mdhash_id(
                normalize_document_file_path(parser_source),
                prefix="doc-",
            ),
            "source_kind": "metadata_updated",
            "reason": "content_hash_match",
            "chunks": ["chunk-report"],
        }
    ]
    assert prepare_metadata.call_args.kwargs["source_uri"] == "bynder://asset/1"
    assert prepare_metadata.call_args.kwargs["download_locator"] == (
        "https://cdn.example.com/assets/1.pdf"
    )
    _, saved = deps["metadata_index"].upsert.await_args.args
    assert saved["source_uri"] == "bynder://asset/1"
    assert saved["download_locator"] == "https://cdn.example.com/assets/1.pdf"
    deps["lightrag"].apipeline_enqueue_documents.assert_not_awaited()
    deps["lightrag"].apipeline_process_enqueue_documents.assert_not_awaited()
    deps["lightrag"].adelete_by_doc_id.assert_not_awaited()


async def test_document_ingest_uses_lightrag_canonical_doc_id(tmp_path: Path) -> None:
    source = tmp_path / "1912.09363v3.pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()

    result = await _ingest_one(engine, source, replace=False)

    expected_doc_id = compute_mdhash_id(
        normalize_document_file_path(source),
        prefix="doc-",
    )
    assert result["doc_id"] == expected_doc_id
    assert deps["metadata_index"].upsert.await_count == 3
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
        await _ingest_one(
            engine,
            source,
            source_uri="bynder://asset/1",
            download_locator="https://cdn.example.com/assets/1.pdf",
        )

    assert persisted == [
        {
            "filename": "1.pdf",
            "filename_stem": "1",
            "source_uri": "bynder://asset/1",
            "download_locator": "https://cdn.example.com/assets/1.pdf",
            "file_extension": "pdf",
            "custom_metadata": {},
            _FINALIZATION_COMPLETE_KEY: False,
            "_dlightrag_source_options": {},
            PARSER_INPUT_SHA256_FIELD: _sha256(b"%PDF-1.4"),
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


async def test_post_processing_failure_keeps_processed_status_hidden_for_retry(
    tmp_path: Path,
) -> None:
    source = tmp_path / "report.pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()
    processed = {"status": "processed", "chunks_list": ["chunk-a"], "content_hash": "hash"}
    deps["stores"].get_doc_status.side_effect = [None, processed]
    writes = 0

    async def metadata_write(*_args: object) -> None:
        nonlocal writes
        writes += 1
        if writes == 2:
            raise RuntimeError("metadata index unavailable")

    deps["metadata_index"].upsert.side_effect = metadata_write

    result = await engine.aingest_files([source])

    assert result["processed"] == 0
    assert result["errors"] == ["report.pdf: document processing failed"]
    assert processed["status"] == "processed"
    deps["stores"].doc_status.upsert.assert_not_awaited()
    assert (
        deps["metadata_index"].upsert.await_args_list[0].args[1][_FINALIZATION_COMPLETE_KEY]
        is False
    )


async def test_batch_finalization_aggregates_failed_and_processed_documents(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    first.write_bytes(b"%PDF-1.4 first")
    second.write_bytes(b"%PDF-1.4 second")
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.side_effect = [
        None,
        None,
        {
            "status": DocStatus.FAILED,
            "chunks_list": [],
            "error_msg": "private parser detail",
        },
        {
            "status": DocStatus.PROCESSED,
            "chunks_list": ["chunk-second"],
            "content_hash": _lightrag_content_hash(b"second"),
        },
    ]

    result = await engine.aingest_files(
        [
            PreparedIngestFile(
                parser_path=first,
                source_uri="bynder://asset/1",
                download_locator="https://cdn.example.com/assets/1.pdf",
                display_filename="first.pdf",
            ),
            PreparedIngestFile(
                parser_path=second,
                source_uri="bynder://asset/2",
                download_locator="https://cdn.example.com/assets/2.pdf",
                display_filename="second.pdf",
            ),
        ]
    )

    assert result["processed"] == 1
    assert result["errors"] == ["first.pdf: document processing failed"]
    assert len(result["results"]) == 1
    assert result["results"][0]["chunks"] == ["chunk-second"]
    assert deps["stores"].get_doc_status.await_count == 4
    assert deps["metadata_index"].upsert.await_count == 3


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

    result = await _ingest_one(engine, source, replace=False)

    assert result["doc_id"] is not None
    assert result["parse_engine"] == "native"
    assert result["process_options"] == "iteP"
    assert result["chunks"] == ["chunk-a"]
    kwargs = deps["lightrag"].apipeline_enqueue_documents.await_args.kwargs
    assert kwargs["parse_engine"] == ["native"]
    assert kwargs["process_options"] == ["iteP"]
    deps["stores"].overwrite_chunk_vectors.assert_not_awaited()


async def test_document_ingest_accepts_explicit_user_metadata(tmp_path: Path) -> None:
    source = tmp_path / "sample[mineru-iteP].pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()

    await _ingest_one(
        engine,
        source,
        replace=False,
        metadata={"reviewer": " Ada Lovelace ", "project": "Analytical Engine"},
    )

    _, saved = deps["metadata_index"].upsert.await_args.args
    assert saved["custom_metadata"] == {
        "reviewer": " Ada Lovelace ",
        "project": "Analytical Engine",
    }


async def test_prepared_file_metadata_overlays_batch_metadata(tmp_path: Path) -> None:
    source = tmp_path / "asset.pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()

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
    )

    _, saved = deps["metadata_index"].upsert.await_args.args
    assert saved["custom_metadata"] == {
        "source_system": "Bynder",
        "department": " Legal ",
        "asset_id": "A-123",
    }


async def test_image_file_ingest_delegates_to_lightrag_parser(
    tmp_path: Path,
) -> None:
    from PIL import Image

    source = tmp_path / "image.png"
    Image.new("RGB", (1, 1), "white").save(source)
    engine, deps = _make_engine()

    result = await _ingest_one(engine, source)

    assert result["source_kind"] == "document"
    deps["stores"].overwrite_chunk_vectors.assert_not_awaited()
    kwargs = deps["lightrag"].apipeline_enqueue_documents.await_args.kwargs
    assert kwargs["docs_format"] == "pending_parse"
    assert kwargs["parse_engine"] == ["mineru"]
    assert kwargs["process_options"] == ["iteP"]


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
        "content_hash": _lightrag_content_hash(b"deadbeef"),
        "status": "analyzing",
    }
    deps["stores"].get_doc_status.side_effect = [
        partial_status,
        partial_status,
        {
            "chunks_list": ["chunk-a"],
            "content_hash": _lightrag_content_hash(b"abc"),
            "status": "processed",
        },
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

    result = await _ingest_one(engine, source, replace=False)

    # Must have cleaned up the old partial record.
    deps["lightrag"].adelete_by_doc_id.assert_awaited_once_with(doc_id, delete_llm_cache=True)
    deps["metadata_index"].delete.assert_not_awaited()
    deps["metadata_index"].upsert.assert_awaited()
    assert events[:2] == ["get_full_doc", "adelete_by_doc_id"]
    assert not artifact_dir.exists()

    # Must have proceeded with normal ingest.
    assert result["doc_id"] == doc_id
    deps["lightrag"].apipeline_enqueue_documents.assert_awaited_once()
    deps["lightrag"].apipeline_process_enqueue_documents.assert_awaited_once()


async def test_document_ingest_replaces_processed_hash_mismatch(tmp_path: Path) -> None:
    """Pinned LightRAG rejects duplicate IDs, so changed content is cleaned first."""
    source = tmp_path / "sample[mineru-iteP].pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.return_value = {
        "chunks_list": ["chunk-1"],
        "content_hash": _lightrag_content_hash(b"abc"),
        "status": "processed",
    }

    await _ingest_one(engine, source, replace=False)

    deps["lightrag"].adelete_by_doc_id.assert_awaited_once()
    deps["metadata_index"].delete.assert_not_awaited()


async def test_document_ingest_first_time_no_cleanup(tmp_path: Path) -> None:
    """When no prior doc_status exists, ingest proceeds without cleanup."""
    source = tmp_path / "sample[mineru-iteP].pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.side_effect = [
        None,
        {
            "chunks_list": ["chunk-a"],
            "content_hash": _lightrag_content_hash(b"abc"),
            "status": "processed",
        },
    ]

    result = await _ingest_one(engine, source, replace=False)

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
    document_embedder = AsyncMock()
    document_embedder.image_enabled = True
    document_embedder.dimension = 3
    document_embedder.aembed_documents.return_value = (
        [DocumentEmbeddingVector(mm_chunk_id, [0.1, 0.2, 0.3], "fused")],
        DocumentEmbeddingTrace(fused=1, text=0, fused_to_text_fallback=0, failed=0),
    )
    engine, deps = _make_engine(document_embedder=document_embedder)
    deps["stores"].fetch_chunk_contents.return_value = [
        {"id": mm_chunk_id, "content": "public/private sector mapping chart"}
    ]
    deps["stores"].get_doc_status.side_effect = [
        None,
        {
            "chunks_list": ["chunk-a", mm_chunk_id],
            "content_hash": _lightrag_content_hash(b"parsed"),
            "status": "processed",
        },
    ]
    deps["stores"].get_full_doc.return_value = {
        "parse_engine": "mineru",
        "process_options": "iteP",
        "chunk_options": {},
        "sidecar_location": artifact_dir.as_uri(),
    }

    result = await _ingest_one(engine, source, replace=False)

    assert result["chunks"] == ["chunk-a", mm_chunk_id]
    deps["stores"].overwrite_chunk_vectors.assert_awaited_once()
    vectors = deps["stores"].overwrite_chunk_vectors.await_args.args[0]
    assert vectors == {mm_chunk_id: [0.1, 0.2, 0.3]}
    document_embedder.aembed_documents.assert_awaited_once_with(
        [
            DocumentEmbeddingInput(
                key=mm_chunk_id,
                text="public/private sector mapping chart",
                image_path=image_path,
            )
        ]
    )


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
    document_embedder = AsyncMock()
    document_embedder.image_enabled = False
    document_embedder.dimension = 3
    engine, deps = _make_engine(
        document_embedder=document_embedder,
    )
    deps["stores"].get_doc_status.side_effect = [
        None,
        {
            "chunks_list": ["chunk-a", mm_chunk_id],
            "content_hash": _lightrag_content_hash(b"parsed"),
            "status": "processed",
        },
    ]
    deps["stores"].get_full_doc.return_value = {
        "parse_engine": "mineru",
        "process_options": "iteP",
        "chunk_options": {},
        "sidecar_location": artifact_dir.as_uri(),
    }

    result = await _ingest_one(engine, source, replace=False)

    assert result["chunks"] == ["chunk-a", mm_chunk_id]
    document_embedder.aembed_documents.assert_not_awaited()
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
            {
                "chunks_list": [],
                "content_hash": _lightrag_content_hash(b"dead"),
                "status": "failed",
            },
            {
                "chunks_list": ["chunk-1"],
                "content_hash": _lightrag_content_hash(b"abc"),
                "status": "processing",
            },
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
            return {
                "chunks_list": ["chunk-1"],
                "content_hash": _lightrag_content_hash(b"%PDF-1.4"),
                "status": "processed",
            }

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
    # The second ingest reads what the first recorded: the same bytes, finalized.
    recorded: dict[str, dict[str, Any]] = {}

    async def upsert(doc_id: str, row: dict[str, Any]) -> None:
        recorded[doc_id] = {**recorded.get(doc_id, {}), **row}

    deps["metadata_index"].upsert.side_effect = upsert
    deps["metadata_index"].get.side_effect = lambda doc_id: recorded.get(doc_id)

    async def ingest() -> dict:
        return await _ingest_one(engine, source, replace=False)

    results = await asyncio.gather(ingest(), ingest())
    assert len(results) == 2
    # Cleanup must have been called exactly once (not twice).
    assert deps["lightrag"].adelete_by_doc_id.await_count == 1


async def test_reingest_skips_when_content_hash_matches(tmp_path: Path) -> None:
    """Re-ingesting finalized content with the same hash returns early."""
    source = tmp_path / "sample[mineru-iteP].pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()
    doc_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")

    deps["stores"].get_doc_status.return_value = {
        "chunks_list": ["chunk-1", "chunk-2"],
        "content_hash": _lightrag_content_hash(source.read_bytes()),
        "status": "processed",
    }
    deps["metadata_index"].get.return_value = _recorded(source.read_bytes())

    result = await _ingest_one(engine, source, replace=False)

    assert result["doc_id"] == doc_id
    assert result["source_kind"] == "skipped"
    assert result["reason"] == "content_hash_match"
    deps["lightrag"].apipeline_enqueue_documents.assert_not_awaited()


async def test_reingest_hash_check_runs_off_event_loop(tmp_path: Path, monkeypatch) -> None:
    import asyncio

    import dlightrag.engine.rag.corpus.ingestion.engine as engine_module

    source = tmp_path / "sample[mineru-iteP].pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.return_value = {
        "chunks_list": ["chunk-1"],
        "content_hash": _lightrag_content_hash(source.read_bytes()),
        "status": "processed",
    }
    calls = []

    async def fake_to_thread(func, *args, **kwargs):
        calls.append(func)
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    await _ingest_one(engine, source, replace=False)

    assert engine_module._file_sha256 in calls


async def test_reingest_proceeds_when_content_hash_differs(tmp_path: Path) -> None:
    """Re-ingesting bytes other than the recorded ones must proceed normally."""
    source = tmp_path / "sample[mineru-iteP].pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.return_value = {
        "chunks_list": ["chunk-1"],
        "content_hash": _lightrag_content_hash(b"different_hash"),
        "status": "processed",
    }
    deps["metadata_index"].get.return_value = _recorded(b"%PDF-1.3")

    result = await _ingest_one(engine, source, replace=False)

    assert result.get("source_kind") != "skipped"
    deps["lightrag"].apipeline_enqueue_documents.assert_awaited_once()


async def test_reingest_proceeds_when_not_processed(tmp_path: Path) -> None:
    """Re-ingesting a failed doc must proceed even if hash matches."""
    source = tmp_path / "sample[mineru-iteP].pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()

    failed_status = {"chunks_list": [], "content_hash": None, "status": "failed"}
    deps["stores"].get_doc_status.side_effect = [
        failed_status,
        failed_status,
        {
            "chunks_list": ["chunk-a"],
            "content_hash": _lightrag_content_hash(b"%PDF-1.4"),
            "status": "processed",
        },
    ]
    deps["metadata_index"].get.return_value = _recorded(b"%PDF-1.4")

    result = await _ingest_one(engine, source, replace=False)

    assert result.get("source_kind") != "skipped"
    deps["lightrag"].apipeline_enqueue_documents.assert_awaited_once()


async def test_sidecar_image_vectors_delegate_document_inputs(tmp_path: Path) -> None:
    import json

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
    deps["stores"].fetch_chunk_contents = AsyncMock(
        return_value=[{"id": "doc-1-mm-drawing-000", "content": "chart"}]
    )
    deps["document_embedder"].dimension = 2
    deps["document_embedder"].aembed_documents.return_value = (
        [DocumentEmbeddingVector("doc-1-mm-drawing-000", [0.1, 0.2], "fused")],
        DocumentEmbeddingTrace(fused=1, text=0, fused_to_text_fallback=0, failed=0),
    )

    await engine._overwrite_sidecar_image_vectors(
        doc_id="doc-1",
        sidecar_location=artifact_dir.as_uri(),
        chunk_ids={"doc-1-mm-drawing-000"},
    )

    deps["document_embedder"].aembed_documents.assert_awaited_once_with(
        [
            DocumentEmbeddingInput(
                key="doc-1-mm-drawing-000",
                text="chart",
                image_path=image_path,
            )
        ]
    )
    deps["stores"].overwrite_chunk_vectors.assert_awaited_once()


async def test_sidecar_image_embed_failure_fails_finalization(tmp_path: Path) -> None:
    import json

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
    deps["stores"].fetch_chunk_contents = AsyncMock(
        return_value=[{"id": "doc-1-mm-drawing-000", "content": "chart"}]
    )
    deps["document_embedder"].aembed_documents = AsyncMock(
        side_effect=RuntimeError("provider rejected oversized image")
    )

    with pytest.raises(RuntimeError, match="provider rejected"):
        await engine._overwrite_sidecar_image_vectors(
            doc_id="doc-1",
            sidecar_location=artifact_dir.as_uri(),
            chunk_ids={"doc-1-mm-drawing-000"},
        )

    deps["stores"].overwrite_chunk_vectors.assert_not_awaited()


async def test_sidecar_unreadable_image_fails_required_visual_fusion(tmp_path: Path) -> None:
    import json

    artifact_dir = tmp_path / "sample.parsed"
    artifact_dir.mkdir()
    (artifact_dir / "sample.blocks.jsonl").write_text("{}\n", encoding="utf-8")
    Image.new("RGB", (128, 128), color=(0, 200, 0)).save(artifact_dir / "good.png")
    (artifact_dir / "bad.png").write_bytes(b"not a real image")
    (artifact_dir / "sample.drawings.json").write_text(
        json.dumps(
            {
                "drawings": {
                    "im-1": {"path": "good.png", "llm_analyze_result": {"status": "success"}},
                    "im-2": {"path": "bad.png", "llm_analyze_result": {"status": "success"}},
                }
            }
        ),
        encoding="utf-8",
    )
    good_chunk = "doc-1-mm-drawing-000"
    bad_chunk = "doc-1-mm-drawing-001"
    engine, deps = _make_engine()
    deps["stores"].fetch_chunk_contents = AsyncMock(
        return_value=[
            {"id": good_chunk, "content": "keep"},
            {"id": bad_chunk, "content": "boom"},
        ]
    )
    deps["document_embedder"].dimension = 2
    deps["document_embedder"].aembed_documents.return_value = (
        [
            DocumentEmbeddingVector(good_chunk, [0.5, 0.6], "fused"),
            DocumentEmbeddingVector(bad_chunk, [0.7, 0.8], "text"),
        ],
        DocumentEmbeddingTrace(fused=1, text=1, fused_to_text_fallback=0, failed=0),
    )

    with pytest.raises(RuntimeError, match="required sidecar visual fusion"):
        await engine._overwrite_sidecar_image_vectors(
            doc_id="doc-1",
            sidecar_location=artifact_dir.as_uri(),
            chunk_ids={good_chunk, bad_chunk},
        )

    deps["stores"].overwrite_chunk_vectors.assert_not_awaited()


def test_resolve_sidecar_uri_handles_file_scheme() -> None:
    from pathlib import Path

    from lightrag.utils_pipeline import resolve_sidecar_uri

    assert resolve_sidecar_uri("file:///tmp/sample.parsed/") == Path("/tmp/sample.parsed")
    assert resolve_sidecar_uri("file:///tmp/path%20with%20spaces/") == Path("/tmp/path with spaces")


def test_resolve_sidecar_uri_rejects_everything_that_is_not_a_local_sidecar() -> None:
    """The unknown sentinel must never resolve: engine cleanup rmtree's the result."""
    from lightrag.utils_pipeline import SIDECAR_LOCATION_UNKNOWN, resolve_sidecar_uri

    assert resolve_sidecar_uri(SIDECAR_LOCATION_UNKNOWN) is None
    assert resolve_sidecar_uri("s3://bucket/key/parsed/") is None
    assert resolve_sidecar_uri("azure://container/path/") is None
    assert resolve_sidecar_uri("/tmp/local/path") is None
    assert resolve_sidecar_uri(None) is None
    assert resolve_sidecar_uri("") is None


async def test_a_later_document_with_a_name_already_in_the_batch_is_refused_alone(
    tmp_path: Path, parser_input_root: Path
) -> None:
    """One flat parser input serves each name, and LightRAG's id is the name's hash."""
    first = tmp_path / "a" / "report.pdf"
    second = tmp_path / "b" / "report.[mineru].pdf"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"one")
    second.write_bytes(b"two")
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.side_effect = [
        None,
        {
            "chunks_list": ["chunk-a"],
            "content_hash": _lightrag_content_hash(b"one"),
            "status": "processed",
        },
    ]
    enqueued: list[bytes] = []

    async def enqueue(**kwargs: Any) -> str:
        enqueued.extend(Path(path).read_bytes() for path in kwargs["file_paths"])
        return "track-1"

    deps["lightrag"].apipeline_enqueue_documents.side_effect = enqueue

    result = await engine.aingest_files([first, second])

    assert result["processed"] == 1
    assert result["errors"] == [
        "report._mineru_.pdf: another document in this batch has the same name"
    ]
    kwargs = deps["lightrag"].apipeline_enqueue_documents.await_args.kwargs
    assert kwargs["file_paths"] == [str(parser_input_root / "report.pdf")]
    assert enqueued == [b"one"]
    deps["lightrag"].adelete_by_doc_id.assert_not_awaited()


async def test_final_delete_failure_with_missing_status_never_restores_zombie(
    tmp_path: Path,
) -> None:
    source = tmp_path / "failed.pdf"
    source.write_bytes(b"failed")
    engine, deps = _make_engine()
    doc_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    statuses = {doc_id: {"status": "failed", "chunks_list": ["old"]}}
    deps["stores"].get_doc_status.side_effect = lambda current: statuses.get(current)
    deps["stores"].get_full_doc.return_value = {"sidecar_location": None}

    async def final_stage_failure(current: str, **_kwargs: object) -> SimpleNamespace:
        statuses.pop(current, None)
        return SimpleNamespace(status="fail")

    deps["lightrag"].adelete_by_doc_id.side_effect = final_stage_failure
    deps["lightrag"].apipeline_enqueue_documents.side_effect = RuntimeError("enqueue down")

    with pytest.raises(RuntimeError, match="enqueue down"):
        await engine.aingest_files([source])

    assert doc_id not in statuses
    deps["stores"].doc_status.upsert.assert_not_awaited()


async def test_metadata_only_remote_tombstone_recovers_and_retires_old_metadata(
    tmp_path: Path,
) -> None:
    source = tmp_path / "new.pdf"
    source.write_bytes(b"new")
    engine, deps = _make_engine()
    new_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    old_id = "doc-old"
    locator = "s3://bucket/item.pdf"
    source_uri = "bynder://asset/1"
    statuses: dict[str, dict[str, object]] = {}
    metadata = {old_id: {"download_locator": locator, "source_uri": source_uri}}
    deps["stores"].get_doc_status.side_effect = lambda current: statuses.get(current)
    deps["metadata_index"].get.side_effect = lambda current: metadata.get(current)

    async def process() -> None:
        statuses[new_id] = {"status": "processed", "chunks_list": ["new"]}

    async def delete_metadata(current: str) -> None:
        metadata.pop(current, None)

    deps["lightrag"].apipeline_process_enqueue_documents.side_effect = process
    deps["metadata_index"].delete.side_effect = delete_metadata
    item = PreparedIngestFile(
        source,
        source_uri,
        locator,
        replacement_doc_ids=(old_id,),
        replacement_ownership=((old_id, locator, source_uri),),
    )

    result = await engine.aingest_files([item], replace=True)

    assert result["processed"] == 1
    assert statuses[new_id]["status"] == "processed"
    assert old_id not in metadata
    deps["lightrag"].adelete_by_doc_id.assert_not_awaited()


async def test_failed_metadata_tombstone_replacement_keeps_only_new_failed_identity(
    tmp_path: Path,
) -> None:
    source = tmp_path / "new.pdf"
    source.write_bytes(b"new")
    engine, deps = _make_engine()
    new_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    old_id = "doc-old"
    locator = "s3://bucket/item.pdf"
    source_uri = "bynder://asset/1"
    statuses: dict[str, dict[str, object]] = {}
    metadata = {old_id: {"download_locator": locator, "source_uri": source_uri}}
    deps["stores"].get_doc_status.side_effect = lambda current: statuses.get(current)
    deps["metadata_index"].get.side_effect = lambda current: metadata.get(current)

    async def process() -> None:
        statuses[new_id] = {"status": "failed", "chunks_list": [], "error_msg": "parse"}

    async def delete_metadata(current: str) -> None:
        metadata.pop(current, None)

    deps["lightrag"].apipeline_process_enqueue_documents.side_effect = process
    deps["metadata_index"].delete.side_effect = delete_metadata
    item = PreparedIngestFile(
        source,
        source_uri,
        locator,
        replacement_doc_ids=(old_id,),
        replacement_ownership=((old_id, locator, source_uri),),
    )

    result = await engine.aingest_files([item], replace=True)

    assert result["processed"] == 0
    assert set(statuses) == {new_id}
    assert statuses[new_id]["status"] == "failed"
    assert old_id not in metadata


async def test_replacement_revalidates_locator_ownership_after_lock_wait(
    tmp_path: Path,
) -> None:
    source = tmp_path / "new.pdf"
    source.write_bytes(b"new")
    engine, deps = _make_engine()
    old_id = "doc-old"
    locator = "s3://bucket/item.pdf"
    source_uri = "bynder://asset/1"
    metadata = {old_id: {"download_locator": locator, "source_uri": source_uri}}
    deps["stores"].get_doc_status.return_value = None
    deps["metadata_index"].get.side_effect = lambda current: metadata.get(current)
    item = PreparedIngestFile(
        source,
        source_uri,
        locator,
        replacement_doc_ids=(old_id,),
        replacement_ownership=((old_id, locator, source_uri),),
    )
    lock = engine._get_ingest_lock(old_id)
    await lock.acquire()
    task = asyncio.create_task(engine.aingest_files([item], replace=True))
    await asyncio.sleep(0)
    metadata[old_id] = {
        "download_locator": "s3://other/unrelated.pdf",
        "source_uri": "bynder://asset/other",
    }
    lock.release()

    with pytest.raises(RuntimeError, match="ownership changed"):
        await task
    deps["lightrag"].adelete_by_doc_id.assert_not_awaited()
    deps["metadata_index"].delete.assert_not_awaited()


@pytest.mark.parametrize("boundary", ["vectors", "bm25"])
async def test_finalization_cancellation_keeps_upstream_processed_and_unpublished(
    tmp_path: Path, boundary: str
) -> None:
    source = tmp_path / f"{boundary}.pdf"
    source.write_bytes(b"content")
    engine, deps = _make_engine()
    doc_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    statuses: dict[str, dict[str, object]] = {}
    deps["stores"].get_doc_status.side_effect = lambda current: statuses.get(current)
    deps["stores"].get_full_doc.return_value = {"sidecar_location": None}

    async def process() -> None:
        statuses[doc_id] = {"status": "processed", "chunks_list": ["chunk"]}

    async def upsert_status(rows: dict[str, dict[str, object]]) -> None:
        statuses.update({key: dict(value) for key, value in rows.items()})

    deps["lightrag"].apipeline_process_enqueue_documents.side_effect = process
    deps["stores"].doc_status.upsert.side_effect = upsert_status
    if boundary == "vectors":
        engine._overwrite_sidecar_image_vectors = AsyncMock(  # type: ignore[method-assign]
            side_effect=asyncio.CancelledError
        )
    else:
        engine._label_bm25_languages = AsyncMock(  # type: ignore[method-assign]
            side_effect=asyncio.CancelledError
        )

    with pytest.raises(asyncio.CancelledError):
        await engine.aingest_files([source])

    assert statuses[doc_id]["status"] == "processed"
    assert "error_msg" not in statuses[doc_id]
    deps["stores"].doc_status.upsert.assert_not_awaited()
    assert (
        deps["metadata_index"].upsert.await_args_list[0].args[1][_FINALIZATION_COMPLETE_KEY]
        is False
    )


async def test_failed_old_to_new_replacement_restores_only_old_failed_identity(
    tmp_path: Path,
) -> None:
    source = tmp_path / "new.pdf"
    source.write_bytes(b"new")
    engine, deps = _make_engine()
    new_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    old_id = "doc-old"
    locator = "s3://bucket/item.pdf"
    source_uri = "bynder://asset/1"
    statuses = {old_id: {"status": "processed", "chunks_list": ["old"]}}
    metadata: dict[str, dict[str, object]] = {
        old_id: {
            "filename": "old.pdf",
            "download_locator": locator,
            "source_uri": source_uri,
        }
    }
    deps["stores"].get_doc_status.side_effect = lambda current: statuses.get(current)
    deps["stores"].get_full_doc.return_value = {"sidecar_location": None}
    deps["metadata_index"].get.side_effect = lambda current: metadata.get(current)

    async def delete(current: str, **_kwargs: object) -> SimpleNamespace:
        statuses.pop(current, None)
        return SimpleNamespace(status="success")

    async def process() -> None:
        statuses[new_id] = {"status": "failed", "chunks_list": [], "error_msg": "parse"}

    async def upsert_status(rows: dict[str, dict[str, object]]) -> None:
        statuses.update({key: dict(value) for key, value in rows.items()})

    async def upsert_metadata(current: str, row: dict[str, object]) -> None:
        metadata[current] = dict(row)

    async def delete_metadata(current: str) -> None:
        metadata.pop(current, None)

    deps["lightrag"].adelete_by_doc_id.side_effect = delete
    deps["lightrag"].apipeline_process_enqueue_documents.side_effect = process
    deps["stores"].doc_status.upsert.side_effect = upsert_status
    deps["metadata_index"].upsert.side_effect = upsert_metadata
    deps["metadata_index"].delete.side_effect = delete_metadata
    item = PreparedIngestFile(
        source,
        source_uri,
        locator,
        replacement_doc_ids=(old_id,),
        replacement_ownership=((old_id, locator, source_uri),),
    )

    result = await engine.aingest_files([item], replace=True)

    assert result["processed"] == 0
    assert statuses == {}
    assert set(metadata) == {old_id}
    assert metadata[old_id]["_dlightrag_finalization_complete"] is False
    assert deps["lightrag"].adelete_by_doc_id.await_count == 2


class _DeletionStatus(Enum):
    """A status reported as an enum member, whose str() is its name, not its value."""

    SUCCESS = "success"


@pytest.mark.parametrize("reported", [" success ", _DeletionStatus.SUCCESS], ids=["padded", "enum"])
async def test_failed_replacement_rollback_reads_a_padded_or_enum_deletion_as_success(
    tmp_path: Path, reported: object
) -> None:
    """The rollback's own deletion check goes through the status normalizer.

    The candidate's row stays visible here, so the reported status alone decides. Read
    raw, it looked like a failed deletion: the rollback kept the failed candidate as
    the retry identity and retired the old one, leaving a finished rollback to repair.
    """
    source = tmp_path / "new.pdf"
    source.write_bytes(b"new")
    engine, deps = _make_engine()
    new_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    old_id = "doc-old"
    locator = "s3://bucket/item.pdf"
    source_uri = "bynder://asset/1"
    statuses = {old_id: {"status": "processed", "chunks_list": ["old"]}}
    metadata: dict[str, dict[str, object]] = {
        old_id: {"filename": "old.pdf", "download_locator": locator, "source_uri": source_uri}
    }
    deps["stores"].get_doc_status.side_effect = lambda current: statuses.get(current)
    deps["stores"].get_full_doc.return_value = {"sidecar_location": None}
    deps["metadata_index"].get.side_effect = lambda current: metadata.get(current)

    async def delete(current: str, **_kwargs: object) -> SimpleNamespace:
        if current == old_id:
            statuses.pop(current, None)
            return SimpleNamespace(status="success")
        return SimpleNamespace(status=reported)

    async def process() -> None:
        statuses[new_id] = {"status": "failed", "chunks_list": [], "error_msg": "parse"}

    async def upsert_metadata(current: str, row: dict[str, object]) -> None:
        metadata[current] = dict(row)

    async def delete_metadata(current: str) -> None:
        metadata.pop(current, None)

    deps["lightrag"].adelete_by_doc_id.side_effect = delete
    deps["lightrag"].apipeline_process_enqueue_documents.side_effect = process
    deps["metadata_index"].upsert.side_effect = upsert_metadata
    deps["metadata_index"].delete.side_effect = delete_metadata
    item = PreparedIngestFile(
        source,
        source_uri,
        locator,
        replacement_doc_ids=(old_id,),
        replacement_ownership=((old_id, locator, source_uri),),
    )

    result = await engine.aingest_files([item], replace=True)

    assert result["processed"] == 0
    assert new_id in statuses
    # A successful deletion retires the candidate and keeps the old identity for retry.
    assert set(metadata) == {old_id}
    assert metadata[old_id]["_dlightrag_finalization_complete"] is False


@pytest.mark.parametrize("failure", [RuntimeError("enqueue failed"), asyncio.CancelledError()])
async def test_outer_enqueue_failure_settles_partial_candidate_and_old_anchor(
    tmp_path: Path, failure: BaseException
) -> None:
    source = tmp_path / "new.pdf"
    source.write_bytes(b"new")
    engine, deps = _make_engine()
    new_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    old_id = "doc-old"
    locator = "s3://bucket/item.pdf"
    source_uri = "bynder://asset/1"
    statuses = {old_id: {"status": "processed", "chunks_list": ["old"]}}
    metadata: dict[str, dict[str, object]] = {
        old_id: {"download_locator": locator, "source_uri": source_uri}
    }
    deps["stores"].get_doc_status.side_effect = lambda current: statuses.get(current)
    deps["stores"].get_full_doc.return_value = {"sidecar_location": None}
    deps["metadata_index"].get.side_effect = lambda current: metadata.get(current)

    async def delete(current: str, **_kwargs: object) -> SimpleNamespace:
        statuses.pop(current, None)
        return SimpleNamespace(status="success")

    async def enqueue(**_kwargs: object) -> None:
        statuses[new_id] = {"status": "pending", "chunks_list": []}
        raise failure

    async def upsert_status(rows: dict[str, dict[str, object]]) -> None:
        statuses.update({key: dict(value) for key, value in rows.items()})

    async def upsert_metadata(current: str, row: dict[str, object]) -> None:
        metadata[current] = dict(row)

    async def delete_metadata(current: str) -> None:
        metadata.pop(current, None)

    deps["lightrag"].adelete_by_doc_id.side_effect = delete
    deps["lightrag"].apipeline_enqueue_documents.side_effect = enqueue
    deps["stores"].doc_status.upsert.side_effect = upsert_status
    deps["metadata_index"].upsert.side_effect = upsert_metadata
    deps["metadata_index"].delete.side_effect = delete_metadata
    item = PreparedIngestFile(
        source,
        source_uri,
        locator,
        replacement_doc_ids=(old_id,),
        replacement_ownership=((old_id, locator, source_uri),),
    )

    with pytest.raises(type(failure)):
        await engine.aingest_files([item], replace=True)

    assert statuses == {}
    assert set(metadata) == {old_id}
    assert metadata[old_id]["_dlightrag_finalization_complete"] is False


async def test_old_metadata_retirement_failure_does_not_publish_deleted_candidate(
    tmp_path: Path,
) -> None:
    source = tmp_path / "new.pdf"
    source.write_bytes(b"new")
    engine, deps = _make_engine()
    new_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    old_id = "doc-old"
    locator = "s3://bucket/item.pdf"
    source_uri = "bynder://asset/1"
    statuses = {old_id: {"status": "processed", "chunks_list": ["old"]}}
    metadata: dict[str, dict[str, object]] = {
        old_id: {"download_locator": locator, "source_uri": source_uri}
    }
    deps["stores"].get_doc_status.side_effect = lambda current: statuses.get(current)
    deps["stores"].get_full_doc.return_value = {"sidecar_location": None}
    deps["metadata_index"].get.side_effect = lambda current: metadata.get(current)

    async def delete(current: str, **_kwargs: object) -> SimpleNamespace:
        statuses.pop(current, None)
        return SimpleNamespace(status="success")

    async def process() -> None:
        statuses[new_id] = {"status": "processed", "chunks_list": ["new"]}

    async def upsert_status(rows: dict[str, dict[str, object]]) -> None:
        statuses.update({key: dict(value) for key, value in rows.items()})

    async def upsert_metadata(current: str, row: dict[str, object]) -> None:
        metadata[current] = dict(row)

    async def delete_metadata(current: str) -> None:
        if current == old_id:
            raise RuntimeError("metadata retirement failed")
        metadata.pop(current, None)

    deps["lightrag"].adelete_by_doc_id.side_effect = delete
    deps["lightrag"].apipeline_process_enqueue_documents.side_effect = process
    deps["stores"].doc_status.upsert.side_effect = upsert_status
    deps["metadata_index"].upsert.side_effect = upsert_metadata
    deps["metadata_index"].delete.side_effect = delete_metadata
    item = PreparedIngestFile(
        source,
        source_uri,
        locator,
        replacement_doc_ids=(old_id,),
        replacement_ownership=((old_id, locator, source_uri),),
    )

    result = await engine.aingest_files([item], replace=True)

    assert result["processed"] == 0
    assert result["results"] == []
    assert result["errors"]
    assert statuses == {}
    assert set(metadata) == {old_id}
    assert metadata[old_id]["_dlightrag_finalization_complete"] is False


async def test_incomplete_finalization_marker_replays_without_reenqueue(
    tmp_path: Path,
) -> None:
    source = tmp_path / "report.pdf"
    content = b"content"
    source.write_bytes(content)
    engine, deps = _make_engine()
    doc_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    deps["stores"].get_doc_status.return_value = {
        "status": "processed",
        "chunks_list": ["chunk"],
        "content_hash": _lightrag_content_hash(content),
    }
    deps["metadata_index"].get.return_value = {
        "filename": "report.pdf",
        _FINALIZATION_COMPLETE_KEY: False,
        PARSER_INPUT_SHA256_FIELD: _sha256(content),
    }
    engine._overwrite_sidecar_image_vectors = AsyncMock()  # type: ignore[method-assign]
    engine._label_bm25_languages = AsyncMock()  # type: ignore[method-assign]

    result = await _ingest_one(engine, source)

    assert result["doc_id"] == doc_id
    assert result["source_kind"] == "document"
    deps["lightrag"].apipeline_enqueue_documents.assert_not_awaited()
    deps["lightrag"].apipeline_process_enqueue_documents.assert_not_awaited()
    engine._overwrite_sidecar_image_vectors.assert_awaited_once()  # type: ignore[attr-defined]
    engine._label_bm25_languages.assert_awaited_once_with(["chunk"])  # type: ignore[attr-defined]
    deps["stores"].doc_status.upsert.assert_not_awaited()
    _, completed = deps["metadata_index"].upsert.await_args.args
    assert completed[_FINALIZATION_COMPLETE_KEY] is True


@pytest.mark.parametrize("original", [RuntimeError("vectors failed"), asyncio.CancelledError()])
async def test_finalizer_failure_preserves_upstream_status_and_original_error(
    tmp_path: Path, original: BaseException, caplog: pytest.LogCaptureFixture
) -> None:

    source = tmp_path / "report.pdf"
    source.write_bytes(b"content")
    engine, deps = _make_engine()
    doc_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    statuses = {doc_id: {"status": "processed", "chunks_list": ["chunk"]}}
    deps["stores"].get_doc_status.side_effect = [None, statuses[doc_id]]
    deps["stores"].get_full_doc.return_value = {"sidecar_location": None}
    deps["lightrag"].apipeline_process_enqueue_documents.side_effect = lambda: None
    engine._overwrite_sidecar_image_vectors = AsyncMock(  # type: ignore[method-assign]
        side_effect=original
    )
    deps["stores"].doc_status.upsert.side_effect = RuntimeError("status store down")

    if isinstance(original, asyncio.CancelledError):
        with pytest.raises(asyncio.CancelledError) as raised:
            await engine.aingest_files([_one_file(source)])
        assert raised.value is original
    else:
        batch = await engine.aingest_files([_one_file(source)])
        assert batch["errors"] == ["report.pdf: document processing failed"]
        assert batch["results"] == []
        # The batch reports a fixed reason; the finalizer's own error is what
        # the log keeps, whatever settling the failure ran into afterwards.
        (reported,) = [
            record
            for record in caplog.records
            if record.getMessage() == "Document finalization failed for report.pdf"
        ]
        assert reported.exc_info is not None
        assert reported.exc_info[1] is original

    assert statuses[doc_id]["status"] == "processed"
    first_metadata = deps["metadata_index"].upsert.await_args_list[0].args[1]
    assert first_metadata[_FINALIZATION_COMPLETE_KEY] is False
    deps["stores"].doc_status.upsert.assert_not_awaited()


async def test_metadata_only_old_tombstone_enqueue_failure_removes_candidate_intent(
    tmp_path: Path,
) -> None:
    source = tmp_path / "new.pdf"
    source.write_bytes(b"new")
    engine, deps = _make_engine()
    new_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    old_id = "doc-old"
    locator = "s3://bucket/item.pdf"
    source_uri = "bynder://asset/1"
    metadata: dict[str, dict[str, object]] = {
        old_id: {"download_locator": locator, "source_uri": source_uri}
    }
    deps["stores"].get_doc_status.return_value = None
    deps["metadata_index"].get.side_effect = lambda current: metadata.get(current)

    async def upsert_metadata(current: str, row: dict[str, object]) -> None:
        metadata[current] = dict(row)

    async def delete_metadata(current: str) -> None:
        metadata.pop(current, None)

    deps["metadata_index"].upsert.side_effect = upsert_metadata
    deps["metadata_index"].delete.side_effect = delete_metadata
    deps["lightrag"].apipeline_enqueue_documents.side_effect = RuntimeError("enqueue failed")
    item = PreparedIngestFile(
        source,
        source_uri,
        locator,
        replacement_doc_ids=(old_id,),
        replacement_ownership=((old_id, locator, source_uri),),
    )

    with pytest.raises(RuntimeError, match="enqueue failed"):
        await engine.aingest_files([item], replace=True)

    assert set(metadata) == {old_id}
    assert new_id not in metadata


async def test_unrelated_preexisting_candidate_fails_before_replacement_cleanup(
    tmp_path: Path,
) -> None:
    source = tmp_path / "new.pdf"
    source.write_bytes(b"new")
    engine, deps = _make_engine()
    new_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    old_id = "doc-old"
    incoming_locator = "s3://bucket/incoming.pdf"
    incoming_source = "bynder://asset/incoming"
    candidate_metadata = {
        "download_locator": "s3://other/existing.pdf",
        "source_uri": "bynder://asset/existing",
        "filename": "existing.pdf",
    }
    metadata: dict[str, dict[str, object]] = {
        new_id: dict(candidate_metadata),
        old_id: {
            "download_locator": incoming_locator,
            "source_uri": incoming_source,
        },
    }
    statuses: dict[str, dict[str, object]] = {
        new_id: {"status": "processed", "chunks_list": ["old-chunk"]}
    }
    deps["stores"].get_doc_status.side_effect = lambda current: statuses.get(current)
    deps["stores"].get_full_doc.return_value = {"sidecar_location": None}
    deps["metadata_index"].get.side_effect = lambda current: metadata.get(current)

    item = PreparedIngestFile(
        source,
        incoming_source,
        incoming_locator,
        replacement_doc_ids=(old_id,),
        replacement_ownership=((old_id, incoming_locator, incoming_source),),
    )

    with pytest.raises(RuntimeError, match="candidate ownership changed"):
        await engine.aingest_files([item], replace=True)

    assert metadata[new_id] == candidate_metadata
    assert statuses[new_id]["status"] == "processed"
    assert metadata[old_id]["download_locator"] == incoming_locator
    deps["lightrag"].adelete_by_doc_id.assert_not_awaited()
    deps["lightrag"].apipeline_enqueue_documents.assert_not_awaited()
    deps["metadata_index"].upsert.assert_not_awaited()
    deps["metadata_index"].delete.assert_not_awaited()


async def test_multiple_metadata_tombstones_collapse_to_one_owner_on_enqueue_failure(
    tmp_path: Path,
) -> None:
    source = tmp_path / "new.pdf"
    source.write_bytes(b"new")
    engine, deps = _make_engine()
    new_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    old_ids = ("doc-old-a", "doc-old-b")
    locator = "s3://bucket/incoming.pdf"
    source_uri = "bynder://asset/incoming"
    metadata: dict[str, dict[str, object]] = {
        old_id: {"download_locator": locator, "source_uri": source_uri} for old_id in old_ids
    }
    deps["stores"].get_doc_status.return_value = None
    deps["metadata_index"].get.side_effect = lambda current: metadata.get(current)

    async def upsert_metadata(current: str, row: dict[str, object]) -> None:
        metadata[current] = dict(row)

    async def delete_metadata(current: str) -> None:
        metadata.pop(current, None)

    deps["metadata_index"].upsert.side_effect = upsert_metadata
    deps["metadata_index"].delete.side_effect = delete_metadata
    deps["lightrag"].apipeline_enqueue_documents.side_effect = RuntimeError("enqueue failed")
    item = PreparedIngestFile(
        source,
        source_uri,
        locator,
        replacement_doc_ids=old_ids,
        replacement_ownership=tuple((old_id, locator, source_uri) for old_id in old_ids),
    )

    with pytest.raises(RuntimeError, match="enqueue failed"):
        await engine.aingest_files([item], replace=True)

    assert set(metadata) == {old_ids[0]}
    assert new_id not in metadata


async def test_replacement_completion_marker_commits_after_old_metadata_retirement(
    tmp_path: Path,
) -> None:
    source = tmp_path / "new.pdf"
    source.write_bytes(b"new")
    engine, deps = _make_engine()
    new_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    old_id = "doc-old"
    locator = "s3://bucket/incoming.pdf"
    source_uri = "bynder://asset/incoming"
    statuses: dict[str, dict[str, object]] = {
        old_id: {"status": "processed", "chunks_list": ["old"]}
    }
    metadata: dict[str, dict[str, object]] = {
        old_id: {"download_locator": locator, "source_uri": source_uri}
    }
    events: list[tuple[str, str, object]] = []
    deps["stores"].get_doc_status.side_effect = lambda current: statuses.get(current)
    deps["stores"].get_full_doc.return_value = {"sidecar_location": None}
    deps["metadata_index"].get.side_effect = lambda current: metadata.get(current)

    async def delete_doc(current: str, **_kwargs: object) -> SimpleNamespace:
        statuses.pop(current, None)
        return SimpleNamespace(status="success")

    async def process() -> None:
        statuses[new_id] = {"status": "processed", "chunks_list": ["new"]}

    async def upsert_metadata(current: str, row: dict[str, object]) -> None:
        metadata[current] = dict(row)
        events.append(("upsert", current, row.get(_FINALIZATION_COMPLETE_KEY)))

    async def delete_metadata(current: str) -> None:
        events.append(("delete", current, None))
        metadata.pop(current, None)

    deps["lightrag"].adelete_by_doc_id.side_effect = delete_doc
    deps["lightrag"].apipeline_process_enqueue_documents.side_effect = process
    deps["metadata_index"].upsert.side_effect = upsert_metadata
    deps["metadata_index"].delete.side_effect = delete_metadata
    item = PreparedIngestFile(
        source,
        source_uri,
        locator,
        replacement_doc_ids=(old_id,),
        replacement_ownership=((old_id, locator, source_uri),),
    )

    result = await engine.aingest_files([item], replace=True)

    assert result["processed"] == 1
    retirement = events.index(("delete", old_id, None))
    completion = events.index(("upsert", new_id, True))
    assert retirement < completion
    assert all(
        marker is not True
        for operation, current, marker in events[:completion]
        if operation == "upsert" and current == new_id
    )
    assert metadata[new_id][_FINALIZATION_COMPLETE_KEY] is True
    assert old_id not in metadata


async def test_recovered_incomplete_replacement_retires_remaining_owner_before_commit(
    tmp_path: Path,
) -> None:
    content = b"new"
    source = tmp_path / "new.pdf"
    source.write_bytes(content)
    engine, deps = _make_engine()
    new_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    old_id = "doc-old"
    locator = "s3://bucket/incoming.pdf"
    source_uri = "bynder://asset/incoming"
    statuses: dict[str, dict[str, object]] = {
        new_id: {
            "status": "processed",
            "chunks_list": ["new"],
            "content_hash": _lightrag_content_hash(content),
        }
    }
    metadata: dict[str, dict[str, object]] = {
        new_id: {
            "download_locator": locator,
            "source_uri": source_uri,
            _FINALIZATION_COMPLETE_KEY: False,
            PARSER_INPUT_SHA256_FIELD: _sha256(content),
        },
        old_id: {"download_locator": locator, "source_uri": source_uri},
    }
    events: list[tuple[str, str, object]] = []
    deps["stores"].get_doc_status.side_effect = lambda current: statuses.get(current)
    deps["stores"].get_full_doc.return_value = {"sidecar_location": None}
    deps["metadata_index"].get.side_effect = lambda current: metadata.get(current)

    async def upsert_metadata(current: str, row: dict[str, object]) -> None:
        metadata[current] = dict(row)
        events.append(("upsert", current, row.get(_FINALIZATION_COMPLETE_KEY)))

    async def delete_metadata(current: str) -> None:
        events.append(("delete", current, None))
        metadata.pop(current, None)

    deps["metadata_index"].upsert.side_effect = upsert_metadata
    deps["metadata_index"].delete.side_effect = delete_metadata
    item = PreparedIngestFile(
        source,
        source_uri,
        locator,
        replacement_doc_ids=(old_id,),
        replacement_ownership=((old_id, locator, source_uri),),
    )

    result = await engine.aingest_files([item], replace=False)

    assert result["processed"] == 1
    assert events.index(("delete", old_id, None)) < events.index(("upsert", new_id, True))
    assert old_id not in metadata
    assert metadata[new_id][_FINALIZATION_COMPLETE_KEY] is True
    deps["lightrag"].apipeline_enqueue_documents.assert_not_awaited()
    deps["lightrag"].apipeline_process_enqueue_documents.assert_not_awaited()


async def test_replacement_marker_failure_after_retirement_leaves_retriable_candidate(
    tmp_path: Path,
) -> None:
    source = tmp_path / "new.pdf"
    source.write_bytes(b"new")
    engine, deps = _make_engine()
    new_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    old_id = "doc-old"
    locator = "s3://bucket/incoming.pdf"
    source_uri = "bynder://asset/incoming"
    statuses: dict[str, dict[str, object]] = {
        old_id: {"status": "processed", "chunks_list": ["old"]}
    }
    metadata: dict[str, dict[str, object]] = {
        old_id: {"download_locator": locator, "source_uri": source_uri}
    }
    deps["stores"].get_doc_status.side_effect = lambda current: statuses.get(current)
    deps["stores"].get_full_doc.return_value = {"sidecar_location": None}
    deps["metadata_index"].get.side_effect = lambda current: metadata.get(current)

    async def delete_doc(current: str, **_kwargs: object) -> SimpleNamespace:
        statuses.pop(current, None)
        return SimpleNamespace(status="success")

    async def process() -> None:
        statuses[new_id] = {"status": "processed", "chunks_list": ["new"]}

    async def upsert_status(rows: dict[str, dict[str, object]]) -> None:
        statuses.update(rows)

    async def upsert_metadata(current: str, row: dict[str, object]) -> None:
        if current == new_id and row.get(_FINALIZATION_COMPLETE_KEY) is True:
            raise RuntimeError("completion marker unavailable")
        metadata[current] = dict(row)

    async def delete_metadata(current: str) -> None:
        metadata.pop(current, None)

    deps["lightrag"].adelete_by_doc_id.side_effect = delete_doc
    deps["lightrag"].apipeline_process_enqueue_documents.side_effect = process
    deps["stores"].doc_status.upsert.side_effect = upsert_status
    deps["metadata_index"].upsert.side_effect = upsert_metadata
    deps["metadata_index"].delete.side_effect = delete_metadata
    item = PreparedIngestFile(
        source,
        source_uri,
        locator,
        replacement_doc_ids=(old_id,),
        replacement_ownership=((old_id, locator, source_uri),),
    )

    result = await engine.aingest_files([item], replace=True)

    assert result["processed"] == 0
    assert result["errors"] == ["incoming.pdf: document processing failed"]
    assert old_id not in metadata
    assert metadata[new_id][_FINALIZATION_COMPLETE_KEY] is False
    assert statuses[new_id]["status"] == "processed"
    assert "error_msg" not in statuses[new_id]
    deps["stores"].doc_status.upsert.assert_not_awaited()


async def test_unrelated_candidate_with_metadata_tombstones_collapses_only_safe_surplus(
    tmp_path: Path,
) -> None:
    source = tmp_path / "new.pdf"
    source.write_bytes(b"new")
    engine, deps = _make_engine()
    new_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    old_ids = ("doc-old-a", "doc-old-b")
    locator = "s3://bucket/incoming.pdf"
    source_uri = "bynder://asset/incoming"
    candidate = {
        "download_locator": "s3://other/existing.pdf",
        "source_uri": "bynder://asset/existing",
        "filename": "existing.pdf",
    }
    metadata: dict[str, dict[str, object]] = {
        new_id: dict(candidate),
        **{old_id: {"download_locator": locator, "source_uri": source_uri} for old_id in old_ids},
    }
    statuses = {new_id: {"status": "processed", "chunks_list": ["existing"]}}
    deps["stores"].get_doc_status.side_effect = lambda current: statuses.get(current)
    deps["metadata_index"].get.side_effect = lambda current: metadata.get(current)

    async def delete_metadata(current: str) -> None:
        metadata.pop(current, None)

    deps["metadata_index"].delete.side_effect = delete_metadata
    item = PreparedIngestFile(
        source,
        source_uri,
        locator,
        replacement_doc_ids=old_ids,
        replacement_ownership=tuple((old_id, locator, source_uri) for old_id in old_ids),
    )

    with pytest.raises(RuntimeError, match="candidate ownership changed"):
        await engine.aingest_files([item], replace=True)

    assert metadata[new_id] == candidate
    assert statuses[new_id] == {"status": "processed", "chunks_list": ["existing"]}
    assert set(metadata) == {new_id, old_ids[0]}
    deps["metadata_index"].delete.assert_awaited_once_with(old_ids[1])
    deps["lightrag"].adelete_by_doc_id.assert_not_awaited()
    deps["lightrag"].apipeline_enqueue_documents.assert_not_awaited()


async def test_unrelated_candidate_does_not_collapse_when_external_owner_has_status(
    tmp_path: Path,
) -> None:
    source = tmp_path / "new.pdf"
    source.write_bytes(b"new")
    engine, deps = _make_engine()
    new_id = compute_mdhash_id(normalize_document_file_path(source), prefix="doc-")
    old_ids = ("doc-old-a", "doc-old-b")
    locator = "s3://bucket/incoming.pdf"
    source_uri = "bynder://asset/incoming"
    candidate = {
        "download_locator": "s3://other/existing.pdf",
        "source_uri": "bynder://asset/existing",
    }
    metadata: dict[str, dict[str, object]] = {
        new_id: dict(candidate),
        **{old_id: {"download_locator": locator, "source_uri": source_uri} for old_id in old_ids},
    }
    statuses = {
        new_id: {"status": "processed", "chunks_list": ["existing"]},
        old_ids[1]: {"status": "failed", "chunks_list": []},
    }
    deps["stores"].get_doc_status.side_effect = lambda current: statuses.get(current)
    deps["metadata_index"].get.side_effect = lambda current: metadata.get(current)
    item = PreparedIngestFile(
        source,
        source_uri,
        locator,
        replacement_doc_ids=old_ids,
        replacement_ownership=tuple((old_id, locator, source_uri) for old_id in old_ids),
    )

    with pytest.raises(RuntimeError, match="candidate ownership changed"):
        await engine.aingest_files([item], replace=True)

    assert metadata == {
        new_id: candidate,
        old_ids[0]: {"download_locator": locator, "source_uri": source_uri},
        old_ids[1]: {"download_locator": locator, "source_uri": source_uri},
    }
    deps["metadata_index"].delete.assert_not_awaited()
    deps["lightrag"].adelete_by_doc_id.assert_not_awaited()
    deps["lightrag"].apipeline_enqueue_documents.assert_not_awaited()


def _record_enqueued(deps: dict[str, Any]) -> list[bytes]:
    """Read every parser input at enqueue time, as LightRAG's parse would."""
    enqueued: list[bytes] = []

    async def enqueue(**kwargs: Any) -> str:
        enqueued.extend(Path(path).read_bytes() for path in kwargs["file_paths"])
        return "track-1"

    deps["lightrag"].apipeline_enqueue_documents.side_effect = enqueue
    return enqueued


def _image_size(content: bytes) -> tuple[int, int]:
    import io

    from PIL import Image

    with Image.open(io.BytesIO(content)) as image:
        return image.size


async def test_image_ingest_enqueues_a_padded_parser_input(
    tmp_path: Path, parser_input_root: Path
) -> None:
    """An image source gets page context, while its provenance stays the file
    the caller supplied."""
    from PIL import Image

    source = tmp_path / "plate.jpg"
    Image.new("RGB", (400, 300), (5, 10, 15)).save(source)
    engine, deps = _make_engine()
    enqueued = _record_enqueued(deps)

    await _ingest_one(engine, source)

    kwargs = deps["lightrag"].apipeline_enqueue_documents.await_args.kwargs
    # Identity: LightRAG keys documents by the canonical basename.
    assert kwargs["file_paths"] == [str(parser_input_root / source.name)]
    (padded,) = enqueued
    assert _image_size(padded) > (400, 300)
    assert kwargs["parse_engine"] == ["mineru"]
    _, saved = deps["metadata_index"].upsert.await_args.args
    assert saved["download_locator"] == str(source.resolve())
    # An unchanged image is known by its own bytes, whatever margin it is parsed with.
    assert saved[PARSER_INPUT_SHA256_FIELD] == _sha256(source.read_bytes())
    assert source.exists()
    # The padded copy is not the document's source, so it goes once LightRAG settled it.
    assert not (parser_input_root / source.name).exists()


async def test_padded_parser_input_is_removed_after_the_batch(tmp_path: Path) -> None:
    from PIL import Image

    from dlightrag.engine.rag.corpus.ingestion.image_normalization import (
        PADDED_INPUT_DIR_NAME,
    )

    source = tmp_path / "plate.png"
    Image.new("RGB", (200, 200), (0, 0, 0)).save(source)
    engine, deps = _make_engine()

    await _ingest_one(engine, source)

    assert not (tmp_path / PADDED_INPUT_DIR_NAME).exists()
    assert source.exists()


async def test_zero_image_margin_enqueues_the_source_bytes(
    tmp_path: Path, parser_input_root: Path
) -> None:
    from PIL import Image

    from dlightrag.engine.rag.corpus.ingestion.image_normalization import (
        PADDED_INPUT_DIR_NAME,
    )

    source = tmp_path / "plate.png"
    Image.new("RGB", (200, 200), (0, 0, 0)).save(source)
    engine, deps = _make_engine(image_margin=0.0)
    enqueued = _record_enqueued(deps)

    await _ingest_one(engine, source)

    kwargs = deps["lightrag"].apipeline_enqueue_documents.await_args.kwargs
    assert kwargs["file_paths"] == [str(parser_input_root / source.name)]
    assert enqueued == [source.read_bytes()]
    assert not (tmp_path / PADDED_INPUT_DIR_NAME).exists()


async def test_a_local_image_is_parsed_as_supplied_and_kept_as_its_source(
    tmp_path: Path, parser_input_root: Path
) -> None:
    """LightRAG archives the flat input it parsed, which for a local source is its source."""
    from PIL import Image

    from dlightrag.engine.rag.corpus.ingestion.image_normalization import (
        PADDED_INPUT_DIR_NAME,
    )

    source = tmp_path / "stage" / "plate.png"
    source.parent.mkdir()
    Image.new("RGB", (200, 200), (0, 0, 0)).save(source)
    engine, deps = _make_engine()
    enqueued = _record_enqueued(deps)
    parser_input = parser_input_root / source.name

    await engine.aingest_files(
        [
            PreparedIngestFile(
                parser_path=source,
                source_uri="local://default/plate.png",
                download_locator=str(parser_input),
            )
        ]
    )

    assert enqueued == [source.read_bytes()]
    assert parser_input.read_bytes() == source.read_bytes()
    assert not (source.parent / PADDED_INPUT_DIR_NAME).exists()


async def test_image_padding_runs_off_the_event_loop(
    tmp_path: Path, parser_input_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    from PIL import Image

    from dlightrag.engine.rag.corpus.ingestion import image_normalization

    source = tmp_path / "plate.png"
    Image.new("RGB", (200, 200), (0, 0, 0)).save(source)
    pad = image_normalization.padded_parser_path
    threads: list[threading.Thread] = []

    def probe(path: Path, *, margin: float) -> Path | None:
        threads.append(threading.current_thread())
        return pad(path, margin=margin)

    monkeypatch.setattr(image_normalization, "padded_parser_path", probe)
    engine, deps = _make_engine()

    await engine.aingest_files([_prepare_ingest_item(source, workspace="default")])

    kwargs = deps["lightrag"].apipeline_enqueue_documents.await_args.kwargs
    assert kwargs["file_paths"] == [str(parser_input_root / source.name)]
    assert threads and threads[0] is not threading.current_thread()


async def test_duplicate_documents_discard_their_padded_inputs(tmp_path: Path) -> None:
    from PIL import Image

    from dlightrag.engine.rag.corpus.ingestion.image_normalization import (
        PADDED_INPUT_DIR_NAME,
    )

    first = tmp_path / "a" / "plate.png"
    second = tmp_path / "b" / "plate.png"
    for source in (first, second):
        source.parent.mkdir()
        Image.new("RGB", (50, 50), (0, 0, 0)).save(source)
    engine, _deps = _make_engine()

    result = await engine.aingest_files(
        [_prepare_ingest_item(path, workspace="default") for path in (first, second)]
    )

    assert result["errors"] == ["plate.png: another document in this batch has the same name"]
    assert not (first.parent / PADDED_INPUT_DIR_NAME).exists()
    assert not (second.parent / PADDED_INPUT_DIR_NAME).exists()


def _remote_item(source: Path) -> PreparedIngestFile:
    return PreparedIngestFile(
        parser_path=source,
        source_uri="s3://bucket/report.pdf",
        download_locator="s3://bucket/report.pdf",
        display_filename="report.pdf",
    )


async def test_a_placed_copy_that_is_not_the_documents_source_goes_once_lightrag_settles(
    tmp_path: Path, parser_input_root: Path
) -> None:
    source = tmp_path / "download" / "report__abc.pdf"
    source.parent.mkdir()
    source.write_bytes(b"remote")
    parser_input = parser_input_root / source.name
    archived = parser_input_root / "__parsed__" / source.name
    engine, deps = _make_engine()

    async def parse_and_archive() -> None:
        if parser_input.exists():
            archived.parent.mkdir(parents=True, exist_ok=True)
            parser_input.rename(archived)

    deps["lightrag"].apipeline_process_enqueue_documents.side_effect = parse_and_archive

    await engine.aingest_files([_remote_item(source)])

    assert not parser_input.exists()
    assert not archived.exists()
    assert source.read_bytes() == b"remote"


def _archiving_engine(
    input_root: Path, monkeypatch: pytest.MonkeyPatch, **overrides: Any
) -> tuple[UnifiedIngestionEngine, dict[str, Any], dict[str, Any]]:
    """An engine whose LightRAG looks up and archives parser inputs as LightRAG does.

    Each enqueued document is found with LightRAG's own resolver and archived with
    its own ``move_file_to_parsed_dir``. The returned state holds the doc_status
    rows (``status``), the metadata rows (``metadata``) and, per document, the
    bytes LightRAG parsed (``parsed``).
    """
    from lightrag.pipeline import _PipelineMixin
    from lightrag.utils import move_file_to_parsed_dir

    input_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("INPUT_DIR", str(input_root.parent))
    monkeypatch.chdir(input_root.parent)
    workspace = input_root.name
    engine, deps = _make_engine(input_root=input_root, workspace=workspace, **overrides)
    state: dict[str, Any] = {"status": {}, "metadata": {}, "parsed": {}}
    queue: list[str] = []

    async def enqueue(**kwargs: Any) -> str:
        for path in kwargs["file_paths"]:
            doc_id = compute_mdhash_id(normalize_document_file_path(path), prefix="doc-")
            state["status"][doc_id] = {"status": "pending", "chunks_list": []}
            queue.append(Path(path).name)
        return "track-1"

    async def process() -> None:
        while queue:
            name = queue.pop(0)
            doc_id = compute_mdhash_id(normalize_document_file_path(name), prefix="doc-")
            source = Path(
                _PipelineMixin._resolve_source_file_for_parser(
                    cast(Any, SimpleNamespace(workspace=workspace)),
                    normalize_document_file_path(name),
                    source_file=name,
                )
            )
            state["parsed"][doc_id] = source.read_bytes()
            await move_file_to_parsed_dir(source, skip_if_already_parsed=True)
            state["status"][doc_id] = {"status": "processed", "chunks_list": [f"chunk-{doc_id}"]}

    def delete(doc_id: str, **_kwargs: Any) -> SimpleNamespace:
        state["status"].pop(doc_id, None)
        return SimpleNamespace(status="success")

    def upsert(doc_id: str, record: Mapping[str, Any]) -> None:
        state["metadata"][doc_id] = {**state["metadata"].get(doc_id, {}), **record}

    deps["lightrag"].apipeline_enqueue_documents.side_effect = enqueue
    deps["lightrag"].apipeline_process_enqueue_documents.side_effect = process
    deps["lightrag"].adelete_by_doc_id.side_effect = delete
    deps["stores"].get_doc_status.side_effect = state["status"].get
    deps["stores"].get_full_doc_statuses.side_effect = lambda doc_ids: {
        doc_id: state["status"][doc_id] for doc_id in doc_ids if doc_id in state["status"]
    }
    deps["stores"].get_full_doc.return_value = {"sidecar_location": None}
    deps["metadata_index"].get.side_effect = state["metadata"].get
    deps["metadata_index"].upsert.side_effect = upsert
    deps["metadata_index"].delete.side_effect = lambda doc_id: state["metadata"].pop(doc_id, None)
    return engine, deps, state


def _staged_local_item(source: Path, input_root: Path) -> PreparedIngestFile:
    """A staged local source as WorkspaceRag describes it: its flat input is its copy."""
    return PreparedIngestFile(
        parser_path=source,
        source_uri=f"local://{input_root.name}/{source.name}",
        download_locator=str(input_root / source.name),
    )


def _staged_versions(tmp_path: Path, name: str, *versions: bytes) -> list[Path]:
    staged = []
    for index, content in enumerate(versions):
        path = tmp_path / "stage" / str(index) / name
        path.parent.mkdir(parents=True)
        path.write_bytes(content)
        staged.append(path)
    return staged


@pytest.mark.parametrize("replace", [True, False], ids=["replace", "changed-bytes"])
async def test_a_new_version_is_archived_where_its_locator_points(
    tmp_path: Path, parser_input_root: Path, monkeypatch: pytest.MonkeyPatch, replace: bool
) -> None:
    """LightRAG archives an input as ``<stem>_001<ext>`` while its name is taken.

    Deleting the old version's document leaves the old archive under that name, so
    the new version would be downloaded, and retried, as the old bytes.
    """
    engine, _deps, state = _archiving_engine(parser_input_root, monkeypatch)
    first, second = _staged_versions(tmp_path, "report.md", b"# one\n", b"# two\n")

    await engine.aingest_files([_staged_local_item(first, parser_input_root)])
    result = await engine.aingest_files(
        [_staged_local_item(second, parser_input_root)], replace=replace
    )

    assert not result["errors"]
    (row,) = state["metadata"].values()
    assert Path(row["download_locator"]).read_bytes() == b"# two\n"
    assert sorted(path.name for path in (parser_input_root / "__parsed__").iterdir()) == [
        "report.md"
    ]


async def test_a_replacement_that_fails_to_finalize_is_retried_from_its_own_bytes(
    tmp_path: Path, parser_input_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retry reads the document's archive, which must hold the version it replays."""
    from dlightrag.engine.rag.workspace.workspace_rag import WorkspaceRag

    engine, deps, state = _archiving_engine(
        parser_input_root,
        monkeypatch,
        bm25_language_classifier=SimpleNamespace(detect=lambda _text: "en"),
    )
    first, second = _staged_versions(tmp_path, "report.md", b"# one\n", b"# two\n")
    await engine.aingest_files([_staged_local_item(first, parser_input_root)])
    deps["stores"].fetch_chunk_contents.side_effect = RuntimeError("chunk store unavailable")

    result = await engine.aingest_files(
        [_staged_local_item(second, parser_input_root)], replace=True
    )

    assert result["errors"] == ["report.md: document processing failed"]
    (row,) = state["metadata"].values()
    retried = WorkspaceRag._retry_local_source_path(
        cast(Any, SimpleNamespace(_workspace_input_root=lambda: parser_input_root)),
        row["download_locator"],
    )
    assert retried.read_bytes() == b"# two\n"


async def test_a_retry_from_lightrags_archive_is_archived_back_in_its_place(
    tmp_path: Path, parser_input_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A document LightRAG archived is replayed from its archive, which stays its copy.

    The replay parses a flat copy of it as supplied (a local image is not padded),
    which LightRAG archives back under the document's name.
    """
    from PIL import Image

    archived = parser_input_root / "__parsed__" / "plate.png"
    archived.parent.mkdir(parents=True)
    Image.new("RGB", (40, 40), (1, 2, 3)).save(archived)
    original = archived.read_bytes()
    engine, _deps, state = _archiving_engine(parser_input_root, monkeypatch)

    result = await engine.aingest_files(
        [
            PreparedIngestFile(
                parser_path=archived,
                source_uri="local://default/plate.png",
                download_locator=str(archived),
            )
        ]
    )

    assert not result["errors"]
    assert list(state["parsed"].values()) == [original]
    assert sorted(path.name for path in archived.parent.iterdir()) == ["plate.png"]
    assert archived.read_bytes() == original
    (row,) = state["metadata"].values()
    assert row["download_locator"] == str(archived)


async def test_a_placed_copy_stays_while_lightrag_may_still_parse_it(
    tmp_path: Path, parser_input_root: Path
) -> None:
    source = tmp_path / "download" / "report__abc.pdf"
    source.parent.mkdir()
    source.write_bytes(b"remote")
    engine, deps = _make_engine()
    deps["lightrag"].apipeline_process_enqueue_documents.side_effect = RuntimeError("worker died")

    with pytest.raises(RuntimeError, match="worker died"):
        await engine.aingest_files([_remote_item(source)])

    assert (parser_input_root / source.name).read_bytes() == b"remote"


async def test_a_parser_input_that_cannot_be_placed_deletes_nothing(
    tmp_path: Path, parser_input_root: Path
) -> None:
    source = tmp_path / "stage" / "report.pdf"
    source.parent.mkdir()
    source.write_bytes(b"%PDF")
    # A folder where the flat copy belongs: the copy cannot replace it.
    (parser_input_root / "report.pdf").mkdir(parents=True)
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.return_value = {
        "status": "processed",
        "chunks_list": ["chunk-old"],
        "content_hash": _lightrag_content_hash(b"old"),
    }

    with pytest.raises(ParserInputPlacementError):
        await engine.aingest_files(
            [
                PreparedIngestFile(
                    parser_path=source,
                    source_uri="local://default/report.pdf",
                    download_locator=str(parser_input_root / "report.pdf"),
                )
            ],
            replace=True,
        )

    deps["lightrag"].adelete_by_doc_id.assert_not_awaited()
    deps["metadata_index"].upsert.assert_not_awaited()
    deps["lightrag"].apipeline_enqueue_documents.assert_not_awaited()


async def test_lightrag_parses_the_placed_input_before_any_same_named_decoy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LightRAG's own resolver finds our flat copy first while it processes the document.

    With the process running in the working directory, LightRAG's fallbacks after
    its input directory are a bare name there and ``inputs/<workspace>``, which is
    then operators' source folder.
    """
    from lightrag.pipeline import _PipelineMixin

    working_dir = tmp_path / "storage"
    corpus = working_dir / "corpus"
    input_root = corpus / "default"
    decoys = (
        working_dir / "report.pdf",
        working_dir / "inputs" / "default" / "report.pdf",
    )
    for decoy in decoys:
        decoy.parent.mkdir(parents=True, exist_ok=True)
        decoy.write_bytes(b"decoy")
    source = input_root / ".runs" / "run-1" / "sources" / "0" / "report.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"ours")
    monkeypatch.setenv("INPUT_DIR", str(corpus))
    monkeypatch.chdir(working_dir)
    engine, deps = _make_engine(input_root=input_root)
    enqueued: list[str] = []
    parsed: list[Path] = []

    async def enqueue(**kwargs: Any) -> str:
        enqueued.extend(kwargs["file_paths"])
        return "track-1"

    async def process() -> None:
        # What LightRAG keeps from the enqueue: the canonical basename as the
        # document's file_path, and the enqueued basename as its source_file.
        for path in enqueued:
            parsed.append(
                Path(
                    _PipelineMixin._resolve_source_file_for_parser(
                        cast(Any, SimpleNamespace(workspace="default")),
                        normalize_document_file_path(path),
                        source_file=Path(path).name,
                    )
                )
            )

    deps["lightrag"].apipeline_enqueue_documents.side_effect = enqueue
    deps["lightrag"].apipeline_process_enqueue_documents.side_effect = process

    await engine.aingest_files(
        [
            PreparedIngestFile(
                parser_path=source,
                source_uri="local://default/report.pdf",
                download_locator=str(input_root / "report.pdf"),
            )
        ]
    )

    assert parsed == [input_root / "report.pdf"]
    assert parsed[0].read_bytes() == b"ours"
    assert all(decoy.read_bytes() == b"decoy" for decoy in decoys)


async def test_partial_cleanup_removes_sidecars_off_the_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    from dlightrag.engine.rag.corpus.ingestion import engine as engine_module

    sidecar = tmp_path / "sample.parsed"
    sidecar.mkdir()
    (sidecar / "content.json").write_text("{}")
    engine, deps = _make_engine()
    deps["stores"].get_full_doc.return_value = {"sidecar_location": sidecar.as_uri()}
    remove = engine_module._remove_sidecar_dir
    threads: list[threading.Thread] = []

    def probe(path: Path) -> None:
        threads.append(threading.current_thread())
        remove(path)

    monkeypatch.setattr(engine_module, "_remove_sidecar_dir", probe)

    await engine._cleanup_partial_doc("doc-1")

    assert not sidecar.exists()
    assert threads and threads[0] is not threading.current_thread()


async def test_non_image_sources_are_not_normalized(
    tmp_path: Path, parser_input_root: Path
) -> None:
    source = tmp_path / "plain.pdf"
    source.write_bytes(b"%PDF-1.4")
    engine, deps = _make_engine()
    enqueued = _record_enqueued(deps)

    await engine.aingest_files([_prepare_ingest_item(source, workspace="default")])

    kwargs = deps["lightrag"].apipeline_enqueue_documents.await_args.kwargs
    assert kwargs["file_paths"] == [str(parser_input_root / source.name)]
    assert enqueued == [b"%PDF-1.4"]


async def test_source_options_survive_parser_failure_and_same_content_update(
    tmp_path: Path,
) -> None:
    from dlightrag.engine.rag.corpus.sources.factory import SourceRetrievalOptions
    from dlightrag.engine.rag.retrieval.metadata_fields import SOURCE_RETRIEVAL_OPTIONS_FIELD

    path = tmp_path / "report.pdf"
    content = b"%PDF-test"
    path.write_bytes(content)
    engine, deps = _make_engine()
    item = PreparedIngestFile(
        parser_path=path,
        source_uri="s3://bucket/report.pdf",
        download_locator="s3://bucket/report.pdf",
        source_options=SourceRetrievalOptions("eu-north-1"),
    )
    deps["stores"].get_doc_status.return_value = None
    deps["lightrag"].apipeline_enqueue_documents.side_effect = RuntimeError("parser failed")
    with pytest.raises(RuntimeError, match="parser failed"):
        await engine.aingest_files([item])
    pending = deps["metadata_index"].upsert.await_args.args[1]
    assert pending[SOURCE_RETRIEVAL_OPTIONS_FIELD] == {"s3_region": "eu-north-1"}
    assert pending[_FINALIZATION_COMPLETE_KEY] is False
    assert pending["custom_metadata"] == {}

    # Routing alone changes while bytes match: update metadata, do not reparse.
    deps["metadata_index"].get.return_value = {
        **pending,
        SOURCE_RETRIEVAL_OPTIONS_FIELD: {"s3_region": "us-east-1"},
        _FINALIZATION_COMPLETE_KEY: True,
    }
    deps["stores"].get_doc_status.return_value = {
        "chunks_list": ["chunk-a"],
        "content_hash": _lightrag_content_hash(content),
        "status": "processed",
    }
    deps["metadata_index"].upsert.reset_mock()
    deps["lightrag"].apipeline_enqueue_documents.reset_mock()
    result = await engine.aingest_files([item], replace=False)
    assert result["results"][0]["source_kind"] == "metadata_updated"
    saved = deps["metadata_index"].upsert.await_args.args[1]
    assert saved[SOURCE_RETRIEVAL_OPTIONS_FIELD] == {"s3_region": "eu-north-1"}
    deps["lightrag"].apipeline_enqueue_documents.assert_not_awaited()


@pytest.mark.parametrize(
    "new_locator,expected_options",
    [
        ("s3://bucket/report.pdf", {"s3_region": "eu-north-1"}),
        ("https://cdn.example.com/report.pdf", {}),
    ],
)
async def test_metadata_only_update_preserves_routing_for_same_locator(
    tmp_path: Path, new_locator: str, expected_options: dict
) -> None:
    from dlightrag.engine.rag.retrieval.metadata_fields import SOURCE_RETRIEVAL_OPTIONS_FIELD

    content = b"%PDF-routing"
    source = tmp_path / "report.pdf"
    source.write_bytes(content)
    engine, deps = _make_engine()
    deps["stores"].get_doc_status.return_value = {
        "chunks_list": ["chunk-a"],
        "content_hash": _lightrag_content_hash(content),
        "status": "processed",
    }
    deps["metadata_index"].get.return_value = {
        "filename": "report.pdf",
        "filename_stem": "report",
        "file_extension": "pdf",
        "source_uri": "s3://bucket/report.pdf",
        "download_locator": "s3://bucket/report.pdf",
        "custom_metadata": {},
        _FINALIZATION_COMPLETE_KEY: True,
        SOURCE_RETRIEVAL_OPTIONS_FIELD: {"s3_region": "eu-north-1"},
        PARSER_INPUT_SHA256_FIELD: _sha256(content),
    }
    result = await _ingest_one(
        engine,
        source,
        source_uri="s3://bucket/report.pdf",
        download_locator=new_locator,
        title="Updated title",
        replace=False,
    )
    assert result["source_kind"] == "metadata_updated"
    saved = deps["metadata_index"].upsert.await_args.args[1]
    assert saved[SOURCE_RETRIEVAL_OPTIONS_FIELD] == expected_options
    assert saved["title"] == "Updated title"
    deps["lightrag"].apipeline_enqueue_documents.assert_not_awaited()
