# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for Composer-only attachment evidence selection."""

import logging
from typing import Any

import pytest

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
        raise AssertionError("short documents must not call rerank")

    selector = ComposerEvidenceSelector()
    current = [
        _row("c1", "Fractions Challenge Worksheet"),
        _row("c2", "Twenty fraction problems and an answer key", chunk_index=2),
    ]

    selected, trace = await selector.select(
        query="这说的啥",
        current_rows=current,
        history_rows=[],
        current_dense_rankings=current,
        history_dense_rankings=[],
        rerank_func=_must_not_run,
    )

    assert selected == current
    assert trace["composer_evidence_strategy"] == "full"
    assert trace["composer_evidence_current_chunks"] == 2
    assert trace["composer_evidence_history_chunks"] == 0


async def test_empty_composer_lane_is_a_noop() -> None:
    selector = ComposerEvidenceSelector()

    selected, trace = await selector.select(
        query="q",
        current_rows=[],
        history_rows=[],
        current_dense_rankings=[],
        history_dense_rankings=[],
        rerank_func=None,
    )

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
        current_dense_rankings=[],
        history_dense_rankings=[],
        rerank_func=_rerank,
    )

    selected_ids = {str(row["chunk_id"]) for row in selected}
    assert rerank_inputs
    assert all(chunk_id.startswith(("a-", "b-")) for chunk_id in rerank_inputs[0])
    assert "a-2" in selected_ids
    assert selected_ids & {"b-1", "b-2", "b-3"}  # doc-b keeps minimum evidence
    assert trace["composer_evidence_strategy"] == "retrieved"
    assert trace["composer_evidence_reranked"] is True


async def test_composer_rerank_failure_falls_back_with_one_shared_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    failure = RuntimeError("reranker unavailable")

    async def _fail_rerank(**_kwargs: Any) -> Any:
        raise failure

    selector = ComposerEvidenceSelector(
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

    with caplog.at_level(logging.WARNING):
        selected, trace = await selector.select(
            query="indemnity obligation",
            current_rows=current,
            history_rows=[],
            current_dense_rankings=[],
            history_dense_rankings=[],
            rerank_func=_fail_rerank,
        )

    assert any(row["chunk_id"] == "c-2" for row in selected)
    assert trace["composer_evidence_strategy"] == "retrieved"
    assert trace["composer_evidence_reranked"] is False
    assert trace["composer_evidence_rerank_error"] == "RuntimeError"
    records = [record for record in caplog.records if "rerank" in record.getMessage().casefold()]
    assert len(records) == 1
    assert records[0].name == "dlightrag.core.retrieval.rerank"
    assert records[0].exc_info is not None
    assert records[0].exc_info[1] is failure


async def test_composer_without_dense_rows_falls_back_to_local_fusion() -> None:
    selector = ComposerEvidenceSelector(
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
        current_dense_rankings=[],
        history_dense_rankings=[],
        rerank_func=None,
    )

    assert any(row["chunk_id"] == "e-2" for row in selected)
    assert "composer_evidence_embedding_error" not in trace


async def test_composer_bm25_failure_is_reported_in_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fail_bm25(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("BM25 unavailable")

    monkeypatch.setattr(
        "dlightrag.core.request.composer_evidence.rank_composer_bm25",
        _fail_bm25,
    )
    selector = ComposerEvidenceSelector(
        full_pass_tokens=4,
        current_target_tokens=80,
        history_target_tokens=0,
        total_tokens=80,
        candidate_limit=4,
    )
    rows = [
        _row("b-1", "general introduction " + ("detail " * 20), chunk_index=1),
        _row("b-2", "termination liability " + ("detail " * 20), chunk_index=2),
    ]

    selected, trace = await selector.select(
        query="termination liability",
        current_rows=rows,
        history_rows=[],
        current_dense_rankings=[],
        history_dense_rankings=[],
        rerank_func=None,
    )

    assert selected
    assert trace["composer_evidence_lexical_error"] == "RuntimeError"


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
        current_dense_rankings=[],
        history_dense_rankings=[],
        rerank_func=None,
    )

    selected_ids = {str(row["chunk_id"]) for row in selected}
    assert {"short-1", "short-2", "short-3"} <= selected_ids
    assert any(chunk_id.startswith("long-") for chunk_id in selected_ids)
    assert trace["composer_evidence_strategy"] == "retrieved"


async def test_default_candidate_limit_matches_rag_chunk_breadth() -> None:
    rerank_candidate_counts: list[int] = []

    async def _rerank(*, query: str, chunks: list[dict[str, Any]], top_k: int):
        assert query == "target"
        assert top_k == len(chunks)
        rerank_candidate_counts.append(len(chunks))
        return chunks

    selector = ComposerEvidenceSelector(
        full_pass_tokens=1,
        current_target_tokens=100_000,
        history_target_tokens=0,
        total_tokens=100_000,
    )
    rows = [
        _row(
            f"candidate-{index}",
            f"target section {index} " + ("detail " * 20),
            chunk_index=index,
        )
        for index in range(40)
    ]

    await selector.select(
        query="target",
        current_rows=rows,
        history_rows=[],
        current_dense_rankings=[],
        history_dense_rankings=[],
        rerank_func=_rerank,
    )

    assert rerank_candidate_counts == [30]


async def test_precomputed_dense_ranking_is_added_to_local_rrf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fused_rankings: list[list[dict[str, Any]]] = []

    def _capture(rankings: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        fused_rankings.extend(rankings)
        return [*rankings[-1], *rankings[0], *rankings[1], *rankings[2]]

    monkeypatch.setattr("dlightrag.core.request.composer_evidence.rrf_fuse", _capture)
    selector = ComposerEvidenceSelector(
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
        for index in range(11)
    ]
    dense = [rows[7]]

    selected, _ = await selector.select(
        query="conceptually related",
        current_rows=rows,
        history_rows=[],
        current_dense_rankings=dense,
        history_dense_rankings=[],
        rerank_func=None,
    )

    assert fused_rankings[3] == dense
    assert any(row["chunk_id"] == "chunk-7" for row in selected)


async def test_current_and_history_lanes_each_rerank_top_30() -> None:
    rerank_counts: list[int] = []

    async def _rerank(*, query: str, chunks: list[dict[str, Any]], top_k: int):
        assert query == "target"
        rerank_counts.append(top_k)
        return chunks

    selector = ComposerEvidenceSelector(
        full_pass_tokens=1,
        current_target_tokens=100_000,
        history_target_tokens=100_000,
        total_tokens=200_000,
    )
    current = [
        _row(
            f"current-{index}",
            f"target current {index} " + ("detail " * 20),
            chunk_index=index,
        )
        for index in range(40)
    ]
    history = [
        _row(
            f"history-{index}",
            f"target history {index} " + ("detail " * 20),
            attachment_id="history-doc",
            scope="history",
            chunk_index=index,
        )
        for index in range(40)
    ]

    await selector.select(
        query="target",
        current_rows=current,
        history_rows=history,
        current_dense_rankings=list(reversed(current)),
        history_dense_rankings=list(reversed(history)),
        rerank_func=_rerank,
    )

    assert rerank_counts == [30, 30]


def test_selector_has_no_client_ownership_or_legacy_embedding_signature() -> None:
    selector = ComposerEvidenceSelector()

    assert not hasattr(selector, "aclose")
    assert not hasattr(selector, "_embedding_func")
    assert not hasattr(selector, "_rerank_func")
    selector_factory: Any = ComposerEvidenceSelector
    with pytest.raises(TypeError):
        selector_factory(embedding_func=object())
