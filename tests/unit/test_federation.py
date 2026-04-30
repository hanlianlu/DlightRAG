# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for cross-workspace federated retrieval."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from dlightrag.core.federation import (
    federated_retrieve,
    merge_results,
)
from dlightrag.core.retrieval.protocols import RetrievalResult


def _make_result(
    chunks: list[dict] | None = None,
    answer: str | None = None,
) -> RetrievalResult:
    """Helper to create a RetrievalResult with given data."""
    return RetrievalResult(
        answer=answer,
        contexts={
            "chunks": chunks or [],
            "entities": [],
            "relationships": [],
        },
    )


class TestMergeResults:
    """Test round-robin merge logic."""

    def test_round_robin_interleaves_chunks(self) -> None:
        r1 = _make_result(chunks=[{"id": "a1"}, {"id": "a2"}, {"id": "a3"}])
        r2 = _make_result(chunks=[{"id": "b1"}, {"id": "b2"}])

        merged = merge_results([r1, r2], ["ws-a", "ws-b"])
        chunk_ids = [c["id"] for c in merged.contexts["chunks"]]

        assert chunk_ids == ["a1", "b1", "a2", "b2", "a3"]

    def test_chunks_tagged_with_workspace(self) -> None:
        r1 = _make_result(chunks=[{"id": "c1"}])
        r2 = _make_result(chunks=[{"id": "c2"}])

        merged = merge_results([r1, r2], ["legal", "finance"])

        assert merged.contexts["chunks"][0]["_workspace"] == "legal"
        assert merged.contexts["chunks"][1]["_workspace"] == "finance"

    def test_chunk_top_k_truncates(self) -> None:
        r1 = _make_result(chunks=[{"id": f"a{i}"} for i in range(10)])
        r2 = _make_result(chunks=[{"id": f"b{i}"} for i in range(10)])

        merged = merge_results([r1, r2], ["ws-a", "ws-b"], chunk_top_k=5)

        assert len(merged.contexts["chunks"]) == 5

    def test_answer_always_none(self) -> None:
        r1 = _make_result(answer="Answer from A")
        r2 = _make_result(answer="Answer from B")

        merged = merge_results([r1, r2], ["ws-a", "ws-b"])

        assert merged.answer is None

    def test_references_empty(self) -> None:
        r1 = _make_result(chunks=[{"id": "a1"}])
        r2 = _make_result(chunks=[{"id": "b1"}])

        merged = merge_results([r1, r2], ["ws-a", "ws-b"])

        assert merged.references == []

    def test_empty_results(self) -> None:
        merged = merge_results([], [])
        assert merged.contexts["chunks"] == []
        assert merged.answer is None

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

        merged = merge_results([r1, r2], ["ws-a", "ws-b"])
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
    async def test_single_workspace_no_federation(self) -> None:
        mock_svc = AsyncMock()
        mock_svc.aretrieve.return_value = _make_result(chunks=[{"id": "c1"}])

        async def get_svc(ws: str):
            return mock_svc

        result = await federated_retrieve("test query", ["ws-only"], get_svc)

        mock_svc.aretrieve.assert_awaited_once()
        assert result.contexts["chunks"][0]["_workspace"] == "ws-only"

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

        result = await federated_retrieve("query", ["ws-a", "ws-b"], get_svc)

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

        result = await federated_retrieve("query", ["ws-ok", "ws-fail"], get_svc)

        assert len(result.contexts["chunks"]) == 1
        assert result.contexts["chunks"][0]["_workspace"] == "ws-ok"

    @pytest.mark.asyncio
    async def test_all_workspaces_fail(self) -> None:
        svc = AsyncMock()
        svc.aretrieve.side_effect = RuntimeError("fail")
        svc._metadata_index = None

        async def get_svc(ws: str):
            return svc

        result = await federated_retrieve("query", ["ws-a", "ws-b"], get_svc)

        assert result.contexts["chunks"] == []

    @pytest.mark.asyncio
    async def test_workspace_filter_rbac(self) -> None:
        svc = AsyncMock()
        svc.aretrieve.return_value = _make_result(chunks=[{"id": "c1"}])

        async def get_svc(ws: str):
            return svc

        async def only_allow_a(requested: list[str]) -> list[str]:
            return [ws for ws in requested if ws == "ws-a"]

        result = await federated_retrieve(
            "query", ["ws-a", "ws-b"], get_svc, workspace_filter=only_allow_a
        )

        svc.aretrieve.assert_awaited_once()
        assert result.contexts["chunks"][0]["_workspace"] == "ws-a"

    @pytest.mark.asyncio
    async def test_workspace_filter_denies_all(self) -> None:
        svc = AsyncMock()

        async def get_svc(ws: str):
            return svc

        async def deny_all(requested: list[str]) -> list[str]:
            return []

        result = await federated_retrieve("query", ["ws-a"], get_svc, workspace_filter=deny_all)

        svc.aretrieve.assert_not_awaited()
        assert result.contexts["chunks"] == []

    @pytest.mark.asyncio
    async def test_empty_workspaces_list(self) -> None:
        async def get_svc(ws: str):
            raise AssertionError("Should not be called")

        result = await federated_retrieve("query", [], get_svc)

        assert result.contexts["chunks"] == []
