# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for cross-workspace federated retrieval."""

from collections.abc import Mapping, Sequence
from typing import Any
from unittest.mock import AsyncMock

import pytest

from dlightrag.engine.rag.retrieval import RetrievalResult
from dlightrag.engine.rag.retrieval.federation import (
    FederationMergePolicy,
    federated_retrieve,
    merge_results,
)


def _make_result(
    chunks: list[dict] | None = None,
) -> RetrievalResult:
    """Helper to create a RetrievalResult with given data."""
    return RetrievalResult(
        contexts={
            "chunks": chunks or [],
            "entities": [],
            "relationships": [],
        },
    )


def _chunks(count: int, ws: str) -> list[dict]:
    """One workspace's ranked chunk list with per-chunk ids and chunk ids."""
    return [
        {"id": f"{ws}-{i}", "chunk_id": f"{ws}-chunk-{i}", "file_path": f"/{ws}/{i}.pdf"}
        for i in range(count)
    ]


def _service_map(workspaces: Sequence[str], *, chunk_count: int = 20) -> dict[str, AsyncMock]:
    """One AsyncMock workspace service per id, each returning a ranked chunk list."""
    services: dict[str, AsyncMock] = {}
    for workspace in workspaces:
        service = AsyncMock()
        service.aretrieve.return_value = _make_result(chunks=_chunks(chunk_count, workspace))
        service._metadata_index = None
        services[workspace] = service
    return services


def _get_service(services: Mapping[str, Any]):
    async def get_svc(workspace: str):
        return services[workspace]

    return get_svc


class TestMergeResults:
    """Test round-robin merge logic."""

    def test_round_robin_interleaves_chunks(self) -> None:
        r1 = _make_result(chunks=[{"id": "a1"}, {"id": "a2"}, {"id": "a3"}])
        r2 = _make_result(chunks=[{"id": "b1"}, {"id": "b2"}])

        merged = merge_results([r1, r2], ["ws-a", "ws-b"], policy=FederationMergePolicy())
        chunk_ids = [c["id"] for c in merged.contexts["chunks"]]

        assert chunk_ids == ["a1", "b1", "a2", "b2", "a3"]

    def test_chunks_tagged_with_workspace(self) -> None:
        r1 = _make_result(chunks=[{"id": "c1"}])
        r2 = _make_result(chunks=[{"id": "c2"}])

        merged = merge_results([r1, r2], ["legal", "finance"], policy=FederationMergePolicy())

        assert merged.contexts["chunks"][0]["_workspace"] == "legal"
        assert merged.contexts["chunks"][1]["_workspace"] == "finance"

    def test_same_chunk_id_in_different_workspaces_keeps_both_sources(self) -> None:
        r1 = _make_result(chunks=[{"chunk_id": "chunk-same", "file_path": "/legal/report.pdf"}])
        r2 = _make_result(chunks=[{"chunk_id": "chunk-same", "file_path": "/finance/report.pdf"}])

        merged = merge_results([r1, r2], ["legal", "finance"], policy=FederationMergePolicy())

        assert [
            (chunk["_workspace"], chunk["file_path"]) for chunk in merged.contexts["chunks"]
        ] == [
            ("legal", "/legal/report.pdf"),
            ("finance", "/finance/report.pdf"),
        ]

    def test_configured_chunk_budget_participates_in_the_formula(self) -> None:
        """The formula reuses the effective chunk_top_k as its budget term —
        no hardcoded 20: a configured 40 admits each workspace's full 20."""
        r1 = _make_result(chunks=_chunks(20, "a"))
        r2 = _make_result(chunks=_chunks(20, "b"))

        merged = merge_results(
            [r1, r2],
            ["ws-a", "ws-b"],
            policy=FederationMergePolicy(chunk_top_k=40),
        )

        # cap = max(7 * 2, 40) = 40: each workspace keeps its full 20.
        assert len(merged.contexts["chunks"]) == 40
        assert merged.trace["federation_cap"] == 40
        assert merged.trace["per_workspace_chunk_count"] == {"ws-a": 20, "ws-b": 20}

    def test_default_budget_floor_keeps_per_workspace_quality_frontier(self) -> None:
        """With the default budget of 20 the fairness floor never shrinks it,
        but at 5 the floor lifts it to 7 per contributing workspace."""
        r1 = _make_result(chunks=_chunks(10, "a"))
        r2 = _make_result(chunks=_chunks(10, "b"))

        merged = merge_results(
            [r1, r2],
            ["ws-a", "ws-b"],
            policy=FederationMergePolicy(chunk_top_k=5),
        )

        # cap = max(7 * 2, 5) = 14: each workspace keeps its top 7.
        assert len(merged.contexts["chunks"]) == 14
        assert merged.trace["federation_cap"] == 14
        assert merged.trace["per_workspace_chunk_count"] == {"ws-a": 7, "ws-b": 7}

    def test_default_budget_unchanged_for_two_workspaces(self) -> None:
        r1 = _make_result(chunks=_chunks(20, "a"))
        r2 = _make_result(chunks=_chunks(20, "b"))

        merged = merge_results(
            [r1, r2],
            ["ws-a", "ws-b"],
            policy=FederationMergePolicy(chunk_top_k=20),
        )

        # cap = max(14, 20) = 20: ten per workspace.
        assert len(merged.contexts["chunks"]) == 20
        assert merged.trace["federation_cap"] == 20
        assert merged.trace["per_workspace_chunk_count"] == {"ws-a": 10, "ws-b": 10}

    def test_fanout_scales_with_contributing_workspaces(self) -> None:
        results = [_make_result(chunks=_chunks(20, ws)) for ws in ("a", "b", "c", "d")]
        merged = merge_results(
            results,
            ["ws-a", "ws-b", "ws-c", "ws-d"],
            policy=FederationMergePolicy(chunk_top_k=20),
        )

        # cap = max(7 * 4, 20) = 28: seven per workspace.
        assert len(merged.contexts["chunks"]) == 28
        assert merged.trace["federation_cap"] == 28
        assert merged.trace["per_workspace_chunk_count"] == {
            "ws-a": 7,
            "ws-b": 7,
            "ws-c": 7,
            "ws-d": 7,
        }

    def test_three_workspaces_lift_the_default_budget(self) -> None:
        results = [_make_result(chunks=_chunks(20, ws)) for ws in ("a", "b", "c")]
        merged = merge_results(
            results,
            ["ws-a", "ws-b", "ws-c"],
            policy=FederationMergePolicy(chunk_top_k=20),
        )

        # cap = max(7 * 3, 20) = 21: seven per workspace, the first lift above
        # the default budget.
        assert len(merged.contexts["chunks"]) == 21
        assert merged.trace["federation_cap"] == 21
        assert merged.trace["per_workspace_chunk_count"] == {
            "ws-a": 7,
            "ws-b": 7,
            "ws-c": 7,
        }

    def test_empty_workspaces_do_not_count_toward_the_floor(self) -> None:
        r1 = _make_result(chunks=_chunks(20, "a"))
        r2 = _make_result(chunks=[])
        r3 = _make_result(chunks=_chunks(20, "c"))

        merged = merge_results(
            [r1, r2, r3],
            ["ws-a", "ws-empty", "ws-c"],
            policy=FederationMergePolicy(chunk_top_k=5),
        )

        # Only two contributing workspaces: cap = max(14, 5) = 14.
        assert len(merged.contexts["chunks"]) == 14
        assert merged.trace["federation_cap"] == 14
        assert merged.trace["per_workspace_chunk_count"] == {"ws-a": 7, "ws-c": 7}

    def test_zero_floor_disables_the_fairness_formula(self) -> None:
        r1 = _make_result(chunks=_chunks(10, "a"))
        r2 = _make_result(chunks=_chunks(10, "b"))

        merged = merge_results(
            [r1, r2],
            ["ws-a", "ws-b"],
            policy=FederationMergePolicy(chunk_top_k=5, min_chunks_per_workspace=0),
        )

        assert len(merged.contexts["chunks"]) == 5
        assert merged.trace["federation_cap"] == 5

    def test_empty_results(self) -> None:
        merged = merge_results([], [], policy=FederationMergePolicy())
        assert merged.contexts["chunks"] == []

    def test_canonicalizes_ref_ids_across_workspaces(self) -> None:
        """Two workspaces both ingesting different docs that happen to share a
        filename: each had reference_id=1 in its own answer, but post-merge
        they must have distinct reference_ids so [1-2] is unambiguous.
        """
        # Same filename in both workspaces (different actual docs).
        r1 = _make_result(
            chunks=[
                {"chunk_id": "a1", "file_path": "/report.pdf", "reference_id": "1"},
                {"chunk_id": "a2", "file_path": "/report.pdf", "reference_id": "1"},
            ]
        )
        r2 = _make_result(
            chunks=[
                {"chunk_id": "b1", "file_path": "/report.pdf", "reference_id": "1"},
            ]
        )

        merged = merge_results([r1, r2], ["ws-a", "ws-b"], policy=FederationMergePolicy())
        chunks = merged.contexts["chunks"]

        # ws-a chunks share one ref_id; ws-b chunks have a different one.
        ws_a_chunks = [c for c in chunks if c["_workspace"] == "ws-a"]
        ws_b_chunks = [c for c in chunks if c["_workspace"] == "ws-b"]
        assert len({c["reference_id"] for c in ws_a_chunks}) == 1
        assert len({c["reference_id"] for c in ws_b_chunks}) == 1
        assert ws_a_chunks[0]["reference_id"] != ws_b_chunks[0]["reference_id"]
        # file_path is preserved unchanged (sentinel proxy is internal).
        assert ws_a_chunks[0]["file_path"] == "/report.pdf"
        assert ws_b_chunks[0]["file_path"] == "/report.pdf"


class TestFederatedRetrieve:
    """Test federated_retrieve orchestration."""

    @pytest.mark.asyncio
    async def test_multi_workspace_parallel(self) -> None:
        svc_a = AsyncMock()
        svc_a.aretrieve.return_value = _make_result(chunks=[{"id": "a1"}])
        svc_a._metadata_index = None
        svc_b = AsyncMock()
        svc_b.aretrieve.return_value = _make_result(chunks=[{"id": "b1"}])
        svc_b._metadata_index = None

        services = {"ws-a": svc_a, "ws-b": svc_b}

        async def get_svc(ws: str):
            return services[ws]

        result = await federated_retrieve(
            "query",
            ["ws-a", "ws-b"],
            get_svc,
            policy=FederationMergePolicy(),
        )

        assert len(result.contexts["chunks"]) == 2
        assert result.contexts["chunks"][0]["_workspace"] == "ws-a"
        assert result.contexts["chunks"][1]["_workspace"] == "ws-b"

    @pytest.mark.asyncio
    async def test_failed_workspace_excluded(self) -> None:
        svc_ok = AsyncMock()
        svc_ok.aretrieve.return_value = _make_result(chunks=[{"id": "ok1"}])
        svc_ok._metadata_index = None

        svc_fail = AsyncMock()
        svc_fail.aretrieve.side_effect = RuntimeError("DB down")
        svc_fail._metadata_index = None

        services = {"ws-ok": svc_ok, "ws-fail": svc_fail}

        async def get_svc(ws: str):
            return services[ws]

        result = await federated_retrieve(
            "query",
            ["ws-ok", "ws-fail"],
            get_svc,
            policy=FederationMergePolicy(),
        )

        assert len(result.contexts["chunks"]) == 1
        assert result.contexts["chunks"][0]["_workspace"] == "ws-ok"
        assert result.trace["failed_workspaces"] == ["ws-fail"]

    @pytest.mark.asyncio
    async def test_all_workspaces_fail(self) -> None:
        svc = AsyncMock()
        svc.aretrieve.side_effect = RuntimeError("fail")
        svc._metadata_index = None

        async def get_svc(ws: str):
            return svc

        with pytest.raises(RuntimeError, match="fail"):
            await federated_retrieve(
                "query", ["ws-a", "ws-b"], get_svc, policy=FederationMergePolicy()
            )

    @pytest.mark.asyncio
    async def test_federated_rerank_selects_policy_budget_from_full_pool(self) -> None:
        services = _service_map(("ws-a", "ws-b"))

        reranker = AsyncMock()
        reranker.side_effect = lambda query, chunks, top_k: list(reversed(chunks))[:top_k]

        result = await federated_retrieve(
            "query",
            ["ws-a", "ws-b"],
            _get_service(services),
            policy=FederationMergePolicy(chunk_top_k=20),
            reranker=reranker,
        )

        # The full interleaved pool (40) was reranked; the final list is the
        # policy budget (20), no fairness truncation before the rerank.
        reranker.assert_awaited_once()
        assert reranker.await_args is not None
        pool = reranker.await_args.kwargs["chunks"]
        assert len(pool) == 40
        assert len(result.contexts["chunks"]) == 20
        assert result.trace["federated_rerank"] == {
            "pool_chunk_count": 40,
            "reranked": True,
            "output_chunk_count": 20,
        }
        assert result.trace["federation_cap"] == 20
        # The reranked order is the reversed interleave, so ws-b's top chunk leads.
        assert result.contexts["chunks"][0]["_workspace"] == "ws-b"

    @pytest.mark.asyncio
    async def test_federated_rerank_failure_degrades_to_capped_interleave(self) -> None:
        services = _service_map(("ws-a", "ws-b"))

        reranker = AsyncMock(side_effect=RuntimeError("rerank down"))

        result = await federated_retrieve(
            "query",
            ["ws-a", "ws-b"],
            _get_service(services),
            policy=FederationMergePolicy(chunk_top_k=20),
            reranker=reranker,
        )

        # Degrades to the default path result: interleave capped at max(7*2, 20).
        assert len(result.contexts["chunks"]) == 20
        assert result.trace["per_workspace_chunk_count"] == {"ws-a": 10, "ws-b": 10}
        assert result.trace["federated_rerank"]["reranked"] is False
        assert result.trace["federated_rerank"]["error_type"] == "RuntimeError"
        assert result.trace["federation_cap"] == 20

    @pytest.mark.asyncio
    async def test_federated_rerank_failure_fallback_keeps_the_fairness_lift(self) -> None:
        """At n=4 the default-path cap (28) exceeds the chunk budget (20); the
        fallback must keep the lifted result, not silently shrink to 20."""
        services = _service_map(("ws-a", "ws-b", "ws-c", "ws-d"))

        reranker = AsyncMock(side_effect=RuntimeError("rerank down"))

        result = await federated_retrieve(
            "query",
            ["ws-a", "ws-b", "ws-c", "ws-d"],
            _get_service(services),
            policy=FederationMergePolicy(chunk_top_k=20),
            reranker=reranker,
        )

        assert len(result.contexts["chunks"]) == 28
        assert result.trace["federation_cap"] == 28
        assert result.trace["per_workspace_chunk_count"] == {
            "ws-a": 7,
            "ws-b": 7,
            "ws-c": 7,
            "ws-d": 7,
        }
        assert result.trace["federated_rerank"]["reranked"] is False
        assert result.trace["federated_rerank"]["error_type"] == "RuntimeError"

    @pytest.mark.asyncio
    async def test_rerank_ignored_without_a_policy_budget(self) -> None:
        svc = AsyncMock()
        svc.aretrieve.return_value = _make_result(chunks=[{"id": "x1"}])
        svc._metadata_index = None

        async def get_svc(ws: str):
            return svc

        reranker = AsyncMock()

        result = await federated_retrieve(
            "query",
            ["ws-a", "ws-b"],
            get_svc,
            policy=FederationMergePolicy(chunk_top_k=None),
            reranker=reranker,
        )

        reranker.assert_not_awaited()
        assert len(result.contexts["chunks"]) == 2
        assert "federated_rerank" not in result.trace
