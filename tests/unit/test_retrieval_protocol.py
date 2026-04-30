# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for RetrievalBackend Protocol and RetrievalResult."""

from __future__ import annotations

from typing import Any

from dlightrag.core.retrieval.protocols import (
    RetrievalBackend,
    RetrievalContexts,
    RetrievalResult,
)


class TestRetrievalResult:
    def test_defaults(self) -> None:
        r = RetrievalResult()
        assert r.answer is None
        assert r.contexts == {"chunks": [], "entities": [], "relationships": []}

    def test_with_values(self) -> None:
        ctx: RetrievalContexts = {
            "chunks": [
                {
                    "chunk_id": "c1",
                    "reference_id": "r1",
                    "file_path": "/f",
                    "content": "hello",
                }
            ],
            "entities": [],
            "relationships": [],
        }
        r = RetrievalResult(answer="hello", contexts=ctx)
        assert r.answer == "hello"
        assert len(r.contexts["chunks"]) == 1
        assert r.contexts["chunks"][0]["chunk_id"] == "c1"

    def test_typeddict_structural_compat(self) -> None:
        """Plain dicts that match the TypedDict structure work fine."""
        plain_dict = {
            "chunks": [
                {
                    "chunk_id": "c1",
                    "reference_id": "r1",
                    "file_path": "/f",
                    "content": "text",
                }
            ],
            "entities": [],
            "relationships": [],
        }
        r = RetrievalResult(answer=None, contexts=plain_dict)
        assert r.contexts["chunks"][0]["content"] == "text"


class TestRetrievalResultReferences:
    def test_default_empty_references(self) -> None:
        result = RetrievalResult()
        assert result.references == []

    def test_references_populated(self) -> None:
        from dlightrag.models.schemas import Reference

        refs = [Reference(id=1, title="doc.pdf")]
        result = RetrievalResult(answer="text", references=refs)
        assert result.references[0].title == "doc.pdf"


class TestRetrievalBackendProtocol:
    def test_structural_subtyping(self) -> None:
        class FakeBackend:
            async def aretrieve(
                self,
                query: str,
                *,
                mode: str = "mix",
                top_k: int | None = None,
                chunk_top_k: int | None = None,
                **kwargs: Any,
            ) -> RetrievalResult:
                return RetrievalResult()

        backend: RetrievalBackend = FakeBackend()
        assert isinstance(backend, FakeBackend)


class TestCanonicalizeReferenceIds:
    """canonicalize_reference_ids fills empty reference_id slots and is
    idempotent on chunks already assigned by aquery_data."""

    def test_empty_input(self) -> None:
        from dlightrag.core.retrieval import canonicalize_reference_ids

        assert canonicalize_reference_ids([]) == []

    def test_assigns_ids_by_file_path_frequency(self) -> None:
        from dlightrag.core.retrieval import canonicalize_reference_ids

        chunks = [
            {"chunk_id": "c1", "file_path": "/A.pdf", "reference_id": ""},
            {"chunk_id": "c2", "file_path": "/B.pdf", "reference_id": ""},
            {"chunk_id": "c3", "file_path": "/A.pdf", "reference_id": ""},
        ]
        out = canonicalize_reference_ids(chunks)
        # A.pdf appears 2x → gets ref_id=1; B.pdf 1x → ref_id=2
        assert out[0]["reference_id"] == "1"
        assert out[2]["reference_id"] == "1"
        assert out[1]["reference_id"] == "2"

    def test_fills_missing_on_injected_chunks(self) -> None:
        from dlightrag.core.retrieval import canonicalize_reference_ids

        # Mix: chunks from aquery_data (with ref_id) + injected (empty ref_id)
        chunks = [
            {"chunk_id": "c1", "file_path": "/A.pdf", "reference_id": "1"},
            {"chunk_id": "c2", "file_path": "/B.pdf", "reference_id": "2"},
            {"chunk_id": "c3", "file_path": "/A.pdf", "reference_id": ""},  # injected, dup
            {"chunk_id": "c4", "file_path": "/C.pdf", "reference_id": ""},  # injected, new
        ]
        out = canonicalize_reference_ids(chunks)
        # Same file → same ref_id; new file → next sequential id
        assert out[0]["reference_id"] == out[2]["reference_id"]
        assert out[3]["reference_id"] not in {out[0]["reference_id"], out[1]["reference_id"]}
        assert out[3]["reference_id"] != ""

    def test_empty_file_path_keeps_empty_ref_id(self) -> None:
        from dlightrag.core.retrieval import canonicalize_reference_ids

        chunks = [
            {"chunk_id": "c1", "file_path": "/A.pdf", "reference_id": ""},
            {"chunk_id": "c2", "file_path": "", "reference_id": ""},
        ]
        out = canonicalize_reference_ids(chunks)
        assert out[0]["reference_id"] == "1"
        assert out[1]["reference_id"] == ""

    def test_does_not_mutate_input(self) -> None:
        from dlightrag.core.retrieval import canonicalize_reference_ids

        chunks = [{"chunk_id": "c1", "file_path": "/A.pdf", "reference_id": ""}]
        canonicalize_reference_ids(chunks)
        assert chunks[0]["reference_id"] == ""

    def test_federated_separates_same_filename_across_workspaces(self) -> None:
        """Two workspaces ingesting different docs with the same filename
        must end up with distinct reference_ids after federation merge."""
        from dlightrag.core.retrieval import canonicalize_reference_ids

        chunks = [
            # Workspace A's report.pdf (one doc)
            {"chunk_id": "a1", "file_path": "/report.pdf", "_workspace": "ws_a"},
            {"chunk_id": "a2", "file_path": "/report.pdf", "_workspace": "ws_a"},
            # Workspace B's report.pdf (a DIFFERENT doc, same filename)
            {"chunk_id": "b1", "file_path": "/report.pdf", "_workspace": "ws_b"},
        ]
        out = canonicalize_reference_ids(chunks, federated=True)
        ws_a_ref = out[0]["reference_id"]
        ws_b_ref = out[2]["reference_id"]
        assert ws_a_ref != ws_b_ref, "cross-workspace duplicates must split"
        # Same workspace + same file → same ref_id
        assert out[0]["reference_id"] == out[1]["reference_id"]
        # file_path on the chunk must be the original, not the \x00-prefixed proxy
        assert out[0]["file_path"] == "/report.pdf"
        assert out[2]["file_path"] == "/report.pdf"

    def test_federated_groups_same_workspace_same_file(self) -> None:
        from dlightrag.core.retrieval import canonicalize_reference_ids

        chunks = [
            {"chunk_id": "c1", "file_path": "/A.pdf", "_workspace": "ws"},
            {"chunk_id": "c2", "file_path": "/B.pdf", "_workspace": "ws"},
            {"chunk_id": "c3", "file_path": "/A.pdf", "_workspace": "ws"},
        ]
        out = canonicalize_reference_ids(chunks, federated=True)
        assert out[0]["reference_id"] == out[2]["reference_id"]
        assert out[0]["reference_id"] != out[1]["reference_id"]
