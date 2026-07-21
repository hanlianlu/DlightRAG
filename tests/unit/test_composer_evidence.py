# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for Composer-only attachment evidence selection."""

import importlib
import logging
from typing import Any

import pytest

from dlightrag.core.request.composer_evidence import (
    COMPOSER_ATTACHMENT_TOKEN_BUDGET,
    ComposerEvidenceSelector,
    partition_composer_rows_by_attachment_budget,
)
from dlightrag.utils.tokens import estimate_tokens


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


async def test_full_pass_attachments_do_not_share_one_allowance() -> None:
    async def _must_not_run(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("short documents must not call rerank")

    selector = ComposerEvidenceSelector(attachment_token_budget=4)
    current = [
        _row("c1", "A" * 16, attachment_id="doc-a"),
        _row("c2", "B" * 16, attachment_id="doc-b"),
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
    assert trace["composer_evidence_candidates"] == 0
    assert trace["composer_evidence_reranked"] is False


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
    assert trace["composer_evidence_candidates"] == 0
    assert trace["composer_evidence_reranked"] is False


async def test_attachment_at_exact_boundary_passes_without_retrieval() -> None:
    async def _must_not_run(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("boundary attachment must not call rerank")

    selector = ComposerEvidenceSelector(attachment_token_budget=4)
    boundary = [_row("boundary", "A" * 16)]

    selected, trace = await selector.select(
        query="boundary",
        current_rows=boundary,
        history_rows=[],
        current_dense_rankings=boundary,
        history_dense_rankings=[],
        rerank_func=_must_not_run,
    )

    assert selected == boundary
    assert trace["composer_evidence_strategy"] == "full"


async def test_attachment_one_token_over_boundary_uses_retrieval_without_overpacking() -> None:
    rerank_calls = 0

    async def _rerank(*, query: str, chunks: list[dict[str, Any]], top_k: int):
        nonlocal rerank_calls
        assert query == "boundary"
        assert top_k == len(chunks)
        rerank_calls += 1
        return chunks

    selector = ComposerEvidenceSelector(attachment_token_budget=4)
    oversized = [_row("oversized", "A" * 17)]

    selected, trace = await selector.select(
        query="boundary",
        current_rows=oversized,
        history_rows=[],
        current_dense_rankings=oversized,
        history_dense_rankings=[],
        rerank_func=_rerank,
    )

    assert estimate_tokens(str(oversized[0]["content"])) == 5
    assert selected == []
    assert rerank_calls == 1
    assert trace["composer_evidence_strategy"] == "retrieved"


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
        attachment_token_budget=15,
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


async def test_oversized_attachments_pack_with_independent_allowances() -> None:
    async def _rerank(*, query: str, chunks: list[dict[str, Any]], top_k: int):
        assert query == "evidence"
        assert top_k == len(chunks)
        return sorted(chunks, key=lambda row: str(row["chunk_id"]))

    selector = ComposerEvidenceSelector(attachment_token_budget=12)
    current = [
        _row(
            f"{attachment_id}-{index}",
            character * 20,
            attachment_id=attachment_id,
            chunk_index=index,
            sidecar_type="table" if index == 1 else None,
        )
        for attachment_id, character in (("doc-a", "A"), ("doc-b", "B"))
        for index in range(3)
    ]

    selected, trace = await selector.select(
        query="evidence",
        current_rows=current,
        history_rows=[],
        current_dense_rankings=current,
        history_dense_rankings=[],
        rerank_func=_rerank,
    )

    selected_references = {str(row["reference_id"]) for row in selected}
    assert selected_references == {"doc-a", "doc-b"}
    for reference_id in selected_references:
        reference_rows = [row for row in selected if row["reference_id"] == reference_id]
        assert len(reference_rows) >= 2
        tokens = sum(estimate_tokens(str(row["content"])) for row in reference_rows)
        assert tokens <= 12
    assert trace["composer_evidence_strategy"] == "retrieved"


async def test_packing_uses_a_later_representative_when_the_first_does_not_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [
        _row("best", "A" * 16, chunk_index=0),
        _row("ordinary", "B" * 16, chunk_index=1),
        _row("coverage-too-large", "C" * 20, chunk_index=2),
        _row("structural-fit", "D" * 12, chunk_index=3, sidecar_type="table"),
    ]

    def _limited_candidates(_rankings: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        return [rows[0], rows[2]]

    monkeypatch.setattr("dlightrag.core.request.composer_evidence.rrf_fuse", _limited_candidates)
    selector = ComposerEvidenceSelector(attachment_token_budget=8, candidate_limit=2)

    selected, _ = await selector.select(
        query="evidence",
        current_rows=rows,
        history_rows=[],
        current_dense_rankings=[],
        history_dense_rankings=[],
        rerank_func=None,
    )

    assert {row["chunk_id"] for row in selected} >= {"best", "structural-fit"}
    assert sum(estimate_tokens(str(row["content"])) for row in selected) <= 8


async def test_packing_prefers_highest_ranked_fitting_row_before_representative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ranked_rows = [
        _row("rank-1-oversized", "A" * 36, chunk_index=0),
        _row("rank-2-fitting", "B" * 20, chunk_index=1),
        _row(
            "rank-3-structural",
            "C" * 16,
            chunk_index=3,
            sidecar_type="table",
        ),
    ]
    rows = [
        ranked_rows[0],
        ranked_rows[1],
        _row("coverage-oversized", "D" * 36, chunk_index=2),
        ranked_rows[2],
        _row("ending-oversized", "E" * 36, chunk_index=4),
    ]

    monkeypatch.setattr(
        "dlightrag.core.request.composer_evidence.rrf_fuse",
        lambda _rankings: ranked_rows,
    )
    selector = ComposerEvidenceSelector(attachment_token_budget=8, candidate_limit=3)

    selected, _ = await selector.select(
        query="evidence",
        current_rows=rows,
        history_rows=[],
        current_dense_rankings=[],
        history_dense_rankings=[],
        rerank_func=None,
    )

    assert [row["chunk_id"] for row in selected] == ["rank-2-fitting"]
    assert sum(estimate_tokens(str(row["content"])) for row in selected) <= 8


async def test_composer_rerank_failure_falls_back_with_one_shared_traceback(
    caplog: pytest.LogCaptureFixture,
) -> None:
    failure = RuntimeError("reranker unavailable")

    async def _fail_rerank(**_kwargs: Any) -> Any:
        raise failure

    selector = ComposerEvidenceSelector(
        attachment_token_budget=12,
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
        attachment_token_budget=50,
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
        attachment_token_budget=50,
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
        attachment_token_budget=12,
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
            f"long section {index} detail detail",
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
        attachment_token_budget=100,
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


async def test_unified_rrf_merges_dense_rankings_for_oversized_chunks_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fused_calls: list[list[list[dict[str, Any]]]] = []

    def _capture(rankings: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        fused_calls.append(rankings)
        return [*rankings[-1], *rankings[0], *rankings[1], *rankings[2]]

    monkeypatch.setattr("dlightrag.core.request.composer_evidence.rrf_fuse", _capture)
    selector = ComposerEvidenceSelector(
        attachment_token_budget=12,
        candidate_limit=8,
    )
    current_oversized = [
        _row(
            f"current-{index}",
            "C" * 20,
            attachment_id="current-oversized",
            chunk_index=index,
        )
        for index in range(3)
    ]
    history_oversized = [
        _row(
            f"history-{index}",
            "H" * 20,
            attachment_id="history-oversized",
            scope="history",
            chunk_index=index,
        )
        for index in range(3)
    ]
    current_full = _row("current-full", "F" * 16, attachment_id="current-full")
    history_full = _row(
        "history-full",
        "P" * 16,
        attachment_id="history-full",
        scope="history",
    )
    foreign = _row("foreign", "outside selection")
    current_dense = [current_full, current_oversized[2], foreign]
    history_dense = [history_full, history_oversized[2]]

    selected, _ = await selector.select(
        query="conceptually related",
        current_rows=[current_full, *current_oversized],
        history_rows=[history_full, *history_oversized],
        current_dense_rankings=current_dense,
        history_dense_rankings=history_dense,
        rerank_func=None,
    )

    assert len(fused_calls) == 1
    assert fused_calls[0][3] == [current_oversized[2], history_oversized[2]]
    assert {row["chunk_id"] for row in selected} >= {
        "current-full",
        "current-2",
        "history-full",
        "history-2",
    }


async def test_current_and_history_share_one_top_30_rerank_call() -> None:
    rerank_counts: list[int] = []

    async def _rerank(*, query: str, chunks: list[dict[str, Any]], top_k: int):
        assert query == "target"
        rerank_counts.append(top_k)
        return list(reversed(chunks))

    selector = ComposerEvidenceSelector(
        attachment_token_budget=100,
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

    selected, trace = await selector.select(
        query="target",
        current_rows=current,
        history_rows=history,
        current_dense_rankings=list(reversed(current)),
        history_dense_rankings=list(reversed(history)),
        rerank_func=_rerank,
    )

    assert rerank_counts == [30]
    selected_ids = [str(row["chunk_id"]) for row in selected]
    input_ids = [str(row["chunk_id"]) for row in [*current, *history]]
    assert selected_ids == [chunk_id for chunk_id in input_ids if chunk_id in set(selected_ids)]
    assert any(chunk_id.startswith("current-") for chunk_id in selected_ids)
    assert any(chunk_id.startswith("history-") for chunk_id in selected_ids)
    selected_current = [row for row in selected if row["reference_id"] == "current-doc"]
    selected_history = [row for row in selected if row["reference_id"] == "history-doc"]
    assert trace["composer_evidence_current_chunks"] == len(selected_current)
    assert trace["composer_evidence_history_chunks"] == len(selected_history)
    assert trace["composer_evidence_tokens"] == sum(
        estimate_tokens(str(row["content"])) for row in selected
    )
    assert trace["composer_evidence_candidates"] == rerank_counts[0]
    assert trace["composer_evidence_reranked"] is True


def test_partition_helper_uses_the_exported_attachment_allowance() -> None:
    assert COMPOSER_ATTACHMENT_TOKEN_BUDGET == 24_576
    boundary = _row("boundary", "A" * 16, attachment_id="boundary")
    oversized = _row("oversized", "B" * 17, attachment_id="oversized")

    full_pass, retrieval = partition_composer_rows_by_attachment_budget(
        [boundary, oversized],
        attachment_token_budget=4,
    )

    assert full_pass == [boundary]
    assert retrieval == [oversized]


def test_partition_migration_alias_is_not_publicly_exported() -> None:
    composer_evidence = importlib.import_module("dlightrag.core.request.composer_evidence")

    assert "partition_composer_rows_by_document_size" not in composer_evidence.__all__


@pytest.mark.parametrize(
    "legacy_budget",
    ["full_pass_tokens", "current_target_tokens", "history_target_tokens", "total_tokens"],
)
def test_selector_rejects_legacy_budget_arguments(legacy_budget: str) -> None:
    selector_factory: Any = ComposerEvidenceSelector
    with pytest.raises(TypeError):
        selector_factory(**{legacy_budget: 1})


def test_selector_has_no_client_ownership_or_legacy_embedding_signature() -> None:
    selector = ComposerEvidenceSelector()

    assert not hasattr(selector, "aclose")
    assert not hasattr(selector, "_embedding_func")
    assert not hasattr(selector, "_rerank_func")
    selector_factory: Any = ComposerEvidenceSelector
    with pytest.raises(TypeError):
        selector_factory(embedding_func=object())
