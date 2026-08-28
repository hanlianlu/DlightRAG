# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Focused tests for grouped, bounded LightRAG provenance hydration."""

import asyncio
import base64
import json
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from dlightrag.engine.rag.retrieval import provenance as provenance_module
from dlightrag.engine.rag.retrieval.provenance import (
    ProvenanceCache,
    hydrate_lightrag_chunk_provenance,
)


class _FakeStores:
    def __init__(
        self,
        raw_chunks: dict[str, dict[str, Any]],
        full_docs: dict[str, dict[str, Any]],
        *,
        fail_batch: bool = False,
    ) -> None:
        self.raw_chunks = raw_chunks
        self.full_docs = full_docs
        self.fail_batch = fail_batch
        self.text_chunk_calls: list[list[str]] = []
        self.full_doc_batch_calls: list[list[str]] = []
        self.full_doc_singular_calls: list[str] = []

    async def get_text_chunks(self, chunk_ids: list[str]) -> list[dict[str, Any] | None]:
        self.text_chunk_calls.append(chunk_ids)
        return [self.raw_chunks.get(chunk_id) for chunk_id in chunk_ids]

    async def get_full_docs(self, doc_ids: list[str]) -> list[dict[str, Any] | None]:
        self.full_doc_batch_calls.append(doc_ids)
        if self.fail_batch:
            raise RuntimeError("batch unavailable")
        return [self.full_docs.get(doc_id) for doc_id in doc_ids]

    async def get_full_doc(self, doc_id: str) -> dict[str, Any] | None:
        self.full_doc_singular_calls.append(doc_id)
        return self.full_docs.get(doc_id)


def _write_artifact_dir(root: Path, artifact_number: int) -> tuple[Path, dict[int, bytes]]:
    artifact_dir = root / f"doc-{artifact_number}.parsed"
    assets_dir = artifact_dir / f"doc-{artifact_number}.blocks.assets"
    assets_dir.mkdir(parents=True)

    block_rows = [
        {
            "blockid": f"drawing-block-{page}",
            "positions": [{"anchor": page}],
        }
        for page in (1, 2)
    ]
    block_rows.extend(
        [
            {"blockid": "table-block", "positions": [{"anchor": 3}]},
            {"blockid": "text-block", "positions": [{"anchor": 4}]},
        ]
    )
    (artifact_dir / "document.blocks.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in block_rows),
        encoding="utf-8",
    )
    (artifact_dir / "document.tables.json").write_text(
        json.dumps({"tables": {"tb-shared": {"blockid": "table-block"}}}),
        encoding="utf-8",
    )

    image_bytes: dict[int, bytes] = {}
    for page in (1, 2):
        payload = f"image-{artifact_number}-{page}".encode()
        image_bytes[page] = payload
        image_path = assets_dir / f"page-{page}.png"
        image_path.write_bytes(payload)
        (artifact_dir / f"page_{page}.drawings.json").write_text(
            json.dumps(
                {
                    "drawings": {
                        "im-shared": {
                            "blockid": f"drawing-block-{page}",
                            "img_path": f"{assets_dir.name}/{image_path.name}",
                        }
                    }
                }
            ),
            encoding="utf-8",
        )
    return artifact_dir, image_bytes


def _large_provenance_fixture(
    tmp_path: Path,
) -> tuple[list[dict[str, Any]], _FakeStores, dict[str, tuple[int, bytes]]]:
    chunks: list[dict[str, Any]] = []
    raw_chunks: dict[str, dict[str, Any]] = {}
    full_docs: dict[str, dict[str, Any]] = {}
    image_expectations: dict[str, tuple[int, bytes]] = {}

    for artifact_number in range(10):
        artifact_dir, image_bytes = _write_artifact_dir(tmp_path, artifact_number)
        for doc_offset in range(2):
            doc_number = artifact_number * 2 + doc_offset
            doc_id = f"doc-{doc_number}"
            full_docs[doc_id] = {
                "file_path": f"/documents/{doc_id}.pdf",
                "sidecar_location": artifact_dir.as_uri(),
            }

        for local_number in range(4):
            candidate_number = artifact_number * 4 + local_number
            chunk_id = f"chunk-{candidate_number:02d}"
            doc_id = f"doc-{candidate_number // 2}"
            if artifact_number < 6 and local_number < 2:
                page_number = local_number + 1
                sidecar = {
                    "type": "drawing",
                    "id": "im-shared",
                    "refs": [{"type": "drawing", "id": "im-shared"}],
                    "page_number": page_number,
                }
                asset_path = (
                    artifact_dir
                    / f"doc-{artifact_number}.blocks.assets"
                    / f"page-{page_number}.png"
                )
                file_path = str(asset_path)
                image_expectations[chunk_id] = (page_number, image_bytes[page_number])
            elif local_number < 3:
                sidecar = {
                    "type": "table",
                    "id": "tb-shared",
                    "refs": [{"type": "table", "id": "tb-shared"}],
                }
                file_path = f"/documents/{doc_id}.pdf"
            else:
                sidecar = {
                    "type": "block",
                    "id": "text-block",
                    "refs": [{"type": "block", "id": "text-block"}],
                }
                file_path = f"/documents/{doc_id}.pdf"
            raw_chunks[chunk_id] = {
                "file_path": file_path,
                "full_doc_id": doc_id,
                "sidecar": sidecar,
            }
            chunks.append({"chunk_id": chunk_id, "content": chunk_id, "file_path": ""})

    return chunks, _FakeStores(raw_chunks, full_docs), image_expectations


async def test_grouped_hydration_batches_docs_indexes_and_survivor_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunks, stores, image_expectations = _large_provenance_fixture(tmp_path)
    assert len(chunks) == 40
    assert len(image_expectations) == 12
    original_order = [chunk["chunk_id"] for chunk in chunks]
    original_objects = list(chunks)
    load_counts: Counter[Path] = Counter()
    image_reads: Counter[Path] = Counter()
    active_jobs = 0
    max_active_jobs = 0
    lock = threading.Lock()
    real_index_loader = provenance_module._load_sidecar_artifact_index
    real_image_loader = provenance_module._image_payload_from_path

    def _observe_job(counter: Counter[Path], path: Path, work):  # noqa: ANN001, ANN202
        nonlocal active_jobs, max_active_jobs
        with lock:
            counter[path.resolve()] += 1
            active_jobs += 1
            max_active_jobs = max(max_active_jobs, active_jobs)
        try:
            time.sleep(0.005)
            return work(path)
        finally:
            with lock:
                active_jobs -= 1

    def _load_index(path: Path):  # noqa: ANN202
        return _observe_job(load_counts, path, real_index_loader)

    def _load_image(path: Path):  # noqa: ANN202
        return _observe_job(image_reads, path, real_image_loader)

    monkeypatch.setattr(provenance_module, "_load_sidecar_artifact_index", _load_index)
    monkeypatch.setattr(provenance_module, "_image_payload_from_path", _load_image)

    cache = ProvenanceCache()
    await hydrate_lightrag_chunk_provenance(
        stores,
        chunks,
        include_image_data=False,
        cache=cache,
    )

    assert [chunk["chunk_id"] for chunk in chunks] == original_order
    assert all(
        actual is expected for actual, expected in zip(chunks, original_objects, strict=True)
    )
    assert len(stores.text_chunk_calls) == 1
    assert len(stores.full_doc_batch_calls) == 1
    assert stores.full_doc_batch_calls[0] == [f"doc-{number}" for number in range(20)]
    assert stores.full_doc_singular_calls == []
    assert len(load_counts) == 10
    assert set(load_counts.values()) == {1}
    assert image_reads == Counter()
    assert all("image_data" not in chunk for chunk in chunks)
    for chunk in chunks:
        if chunk["chunk_id"] in image_expectations:
            assert chunk["file_path"] == f"/documents/{chunk['full_doc_id']}.pdf"

    image_candidates = [chunk for chunk in chunks if chunk["chunk_id"] in image_expectations]
    image_survivors = [image_candidates[index] for index in (0, 1, 3, 4, 8, 11)]
    assert {image_expectations[chunk["chunk_id"]][0] for chunk in image_survivors} == {1, 2}
    non_image_survivors = [
        chunk for chunk in chunks if chunk["chunk_id"] not in image_expectations
    ][:4]
    survivors = image_survivors + non_image_survivors
    survivor_order = [chunk["chunk_id"] for chunk in survivors]
    survivor_objects = list(survivors)
    await hydrate_lightrag_chunk_provenance(stores, survivors, cache=cache)

    assert [chunk["chunk_id"] for chunk in survivors] == survivor_order
    assert all(
        actual is expected for actual, expected in zip(survivors, survivor_objects, strict=True)
    )
    assert len(stores.text_chunk_calls) == 1
    assert len(stores.full_doc_batch_calls) == 1
    assert stores.full_doc_singular_calls == []
    assert set(load_counts.values()) == {1}
    assert len(image_reads) == len(image_survivors)
    assert sum(image_reads.values()) == len(image_survivors)
    assert set(image_reads.values()) == {1}
    assert 1 <= max_active_jobs <= provenance_module._FILESYSTEM_CONCURRENCY

    for chunk in image_survivors:
        expected_page, expected_bytes = image_expectations[chunk["chunk_id"]]
        assert chunk["page_number"] == expected_page
        assert chunk["file_path"] == f"/documents/{chunk['full_doc_id']}.pdf"
        assert chunk["image_data"] == base64.b64encode(expected_bytes).decode("ascii")
        assert chunk["image_mime_type"] == "image/png"
    for chunk in chunks:
        if chunk["chunk_id"] not in {item["chunk_id"] for item in image_survivors}:
            assert not chunk.get("image_data")


async def test_malformed_sidecar_location_degrades_per_chunk() -> None:
    stores = _FakeStores(
        {
            "chunk-bad-uri": {
                "file_path": "/documents/report.pdf",
                "sidecar_location": "file://[",
                "sidecar": {"page_number": 7},
            }
        },
        {},
    )
    chunk = {"chunk_id": "chunk-bad-uri", "content": "usable", "file_path": ""}

    await hydrate_lightrag_chunk_provenance(stores, [chunk], include_image_data=False)

    assert chunk["page_number"] == 7
    assert chunk["file_path"] == "/documents/report.pdf"
    assert "image_data" not in chunk


async def test_full_doc_batch_failure_falls_back_best_effort(tmp_path: Path) -> None:
    artifact_dir, _ = _write_artifact_dir(tmp_path, 0)
    raw_chunks = {
        "chunk-a": {
            "full_doc_id": "doc-a",
            "file_path": str(artifact_dir / "doc-0.blocks.assets" / "page-1.png"),
            "sidecar": {"page_number": 1},
        },
        "chunk-b": {
            "full_doc_id": "doc-b",
            "file_path": "/documents/doc-b.pdf",
            "sidecar": {"page_number": 2},
        },
    }
    full_docs = {
        "doc-a": {"file_path": "/documents/doc-a.pdf"},
        "doc-b": {"file_path": "/documents/doc-b.pdf"},
    }
    stores = _FakeStores(raw_chunks, full_docs, fail_batch=True)
    chunks = [
        {"chunk_id": "chunk-a", "content": "a", "file_path": ""},
        {"chunk_id": "chunk-b", "content": "b", "file_path": ""},
    ]

    await hydrate_lightrag_chunk_provenance(stores, chunks, include_image_data=False)

    assert stores.full_doc_batch_calls == [["doc-a", "doc-b"]]
    assert sorted(stores.full_doc_singular_calls) == ["doc-a", "doc-b"]
    assert [chunk["page_number"] for chunk in chunks] == [1, 2]
    assert chunks[0]["file_path"] == "/documents/doc-a.pdf"


async def test_hydration_propagates_cancellation_during_image_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"image")
    stores = _FakeStores(
        {
            "chunk-a": {
                "file_path": str(image_path),
                "sidecar": {"path": str(image_path), "page_number": 1},
            }
        },
        {},
    )
    chunks = [{"chunk_id": "chunk-a", "content": "a", "file_path": ""}]
    loop = asyncio.get_running_loop()
    started = asyncio.Event()
    release = threading.Event()

    real_image_loader = provenance_module._image_payload_from_path

    def _blocking_with_real_result(path: Path) -> tuple[str, str] | None:
        loop.call_soon_threadsafe(started.set)
        release.wait(timeout=2)
        return real_image_loader(path)

    monkeypatch.setattr(provenance_module, "_image_payload_from_path", _blocking_with_real_result)
    task = asyncio.create_task(hydrate_lightrag_chunk_provenance(stores, chunks))
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
