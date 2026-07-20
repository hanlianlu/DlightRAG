# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for Composer-only attachment evidence selection."""

from typing import Any

from dlightrag.core.request.composer_evidence import ComposerEvidenceSelector


def _row(
    chunk_id: str,
    content: str,
    *,
    attachment_id: str = "current-doc",
    scope: str = "current",
    chunk_index: int = 1,
    sidecar_type: str | None = None,
) -> dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "reference_id": attachment_id,
        "full_doc_id": attachment_id,
        "file_path": f"{attachment_id}.pdf",
        "content": content,
        "metadata": {
            "source_type": "web_attachment",
            "attachment_scope": scope,
            "source_uri": f"web-attachment://{attachment_id}",
            "source_download_locator": f"web-attachment://{attachment_id}",
            "chunk_index": chunk_index,
            "sidecar_type": sidecar_type,
        },
        "_workspace": "__web_attachment__",
    }


async def test_short_composer_documents_pass_through_without_retrieval() -> None:
    async def _must_not_run(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("short documents must not call embedding or rerank")

    selector = ComposerEvidenceSelector(
        embedding_func=_must_not_run,
        rerank_func=_must_not_run,
    )
    current = [
        _row("c1", "Fractions Challenge Worksheet"),
        _row("c2", "Twenty fraction problems and an answer key", chunk_index=2),
    ]

    selected, trace = await selector.select(
        query="这说的啥",
        current_rows=current,
        history_rows=[],
    )

    assert selected == current
    assert trace["composer_evidence_strategy"] == "full"
    assert trace["composer_evidence_current_chunks"] == 2
    assert trace["composer_evidence_history_chunks"] == 0


async def test_empty_composer_lane_is_a_noop() -> None:
    selector = ComposerEvidenceSelector()

    selected, trace = await selector.select(query="q", current_rows=[], history_rows=[])

    assert selected == []
    assert trace["composer_evidence_strategy"] == "empty"


async def test_long_composer_documents_use_only_local_candidates_and_rerank() -> None:
    rerank_inputs: list[list[str]] = []

    async def _rerank(*, query: str, chunks: list[dict[str, Any]], top_k: int):
        assert query == "termination liability"
        assert top_k == len(chunks)
        rerank_inputs.append([str(chunk["chunk_id"]) for chunk in chunks])
        return sorted(
            chunks,
            key=lambda chunk: "termination liability" not in str(chunk["content"]),
        )

    selector = ComposerEvidenceSelector(
        rerank_func=_rerank,
        full_pass_tokens=8,
        current_target_tokens=80,
        history_target_tokens=40,
        total_tokens=100,
        candidate_limit=8,
    )
    current = [
        _row("a-1", "Agreement overview and parties", attachment_id="doc-a", chunk_index=1),
        _row(
            "a-2",
            "Termination liability and damages are capped",
            attachment_id="doc-a",
            chunk_index=2,
        ),
        _row("a-3", "Boilerplate notices and signatures", attachment_id="doc-a", chunk_index=3),
        _row(
            "b-1",
            "[Table Name] Commercial schedule",
            attachment_id="doc-b",
            chunk_index=1,
            sidecar_type="table",
        ),
        _row("b-2", "Payment timing and currency", attachment_id="doc-b", chunk_index=2),
        _row("b-3", "Final appendix", attachment_id="doc-b", chunk_index=3),
    ]

    selected, trace = await selector.select(
        query="termination liability",
        current_rows=current,
        history_rows=[],
    )

    selected_ids = {str(row["chunk_id"]) for row in selected}
    assert rerank_inputs
    assert all(chunk_id.startswith(("a-", "b-")) for chunk_id in rerank_inputs[0])
    assert "a-2" in selected_ids
    assert selected_ids & {"b-1", "b-2", "b-3"}  # doc-b keeps minimum evidence
    assert trace["composer_evidence_strategy"] == "retrieved"
    assert trace["composer_evidence_reranked"] is True


async def test_composer_rerank_failure_falls_back_to_local_fusion() -> None:
    async def _fail_rerank(**_kwargs: Any) -> Any:
        raise RuntimeError("reranker unavailable")

    selector = ComposerEvidenceSelector(
        rerank_func=_fail_rerank,
        full_pass_tokens=4,
        current_target_tokens=30,
        history_target_tokens=0,
        total_tokens=30,
        candidate_limit=4,
    )
    current = [
        _row("c-1", "unrelated introduction", chunk_index=1),
        _row("c-2", "specific indemnity obligation", chunk_index=2),
        _row("c-3", "unrelated appendix", chunk_index=3),
    ]

    selected, trace = await selector.select(
        query="indemnity obligation",
        current_rows=current,
        history_rows=[],
    )

    assert any(row["chunk_id"] == "c-2" for row in selected)
    assert trace["composer_evidence_strategy"] == "retrieved"
    assert trace["composer_evidence_reranked"] is False
    assert trace["composer_evidence_rerank_error"] == "RuntimeError"


async def test_composer_embedding_failure_falls_back_to_local_fusion() -> None:
    async def _fail_embedding(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("embedding unavailable")

    selector = ComposerEvidenceSelector(
        embedding_func=_fail_embedding,
        full_pass_tokens=4,
        current_target_tokens=80,
        history_target_tokens=0,
        total_tokens=80,
        candidate_limit=4,
    )
    rows = [
        _row("e-1", "general introduction " + ("detail " * 20), chunk_index=1),
        _row("e-2", "termination liability " + ("detail " * 20), chunk_index=2),
        _row("e-3", "appendix " + ("detail " * 20), chunk_index=3),
    ]

    selected, trace = await selector.select(
        query="termination liability",
        current_rows=rows,
        history_rows=[],
    )

    assert any(row["chunk_id"] == "e-2" for row in selected)
    assert trace["composer_evidence_embedding_error"] == "RuntimeError"


async def test_short_document_full_pass_survives_beside_long_document() -> None:
    selector = ComposerEvidenceSelector(
        full_pass_tokens=12,
        current_target_tokens=64,
        history_target_tokens=0,
        total_tokens=64,
        candidate_limit=4,
    )
    short = [
        _row("short-1", "short title", attachment_id="short", chunk_index=1),
        _row("short-2", "short body", attachment_id="short", chunk_index=2),
        _row("short-3", "short ending", attachment_id="short", chunk_index=3),
    ]
    long = [
        _row(
            f"long-{index}",
            f"long section {index} " + ("detail " * 20),
            attachment_id="long",
            chunk_index=index,
        )
        for index in range(1, 8)
    ]

    selected, trace = await selector.select(
        query="section 6",
        current_rows=[*short, *long],
        history_rows=[],
    )

    selected_ids = {str(row["chunk_id"]) for row in selected}
    assert {"short-1", "short-2", "short-3"} <= selected_ids
    assert any(chunk_id.startswith("long-") for chunk_id in selected_ids)
    assert trace["composer_evidence_strategy"] == "retrieved"


async def test_dense_ranking_embeds_all_long_document_rows_before_candidate_limit() -> None:
    embedded_documents: list[str] = []

    async def _embed(texts: list[str], *, context: str):
        if context == "query":
            return [[1.0, 0.0]]
        embedded_documents.extend(texts)
        return [[1.0, 0.0] if "semantic target" in text else [0.0, 1.0] for text in texts]

    selector = ComposerEvidenceSelector(
        embedding_func=_embed,
        full_pass_tokens=4,
        current_target_tokens=80,
        history_target_tokens=0,
        total_tokens=80,
        candidate_limit=3,
    )
    rows = [
        _row(
            f"chunk-{index}",
            ("semantic target " if index == 9 else "ordinary text ") + ("detail " * 20),
            chunk_index=index,
        )
        for index in range(10)
    ]

    selected, _ = await selector.select(
        query="conceptually related",
        current_rows=rows,
        history_rows=[],
    )

    assert len(embedded_documents) == len(rows)
    assert any(row["chunk_id"] == "chunk-9" for row in selected)


async def test_selector_closes_independent_embedding_and_rerank_resources() -> None:
    class _Closeable:
        def __init__(self) -> None:
            self.closed = 0

        async def aclose(self) -> None:
            self.closed += 1

    embedding_owner = _Closeable()
    rerank = _Closeable()
    selector = ComposerEvidenceSelector(
        embedding_owner=embedding_owner,
        rerank_func=rerank,
    )

    await selector.aclose()

    assert embedding_owner.closed == 1
    assert rerank.closed == 1
