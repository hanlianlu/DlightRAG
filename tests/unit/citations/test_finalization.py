# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for shared answer finalization."""

from __future__ import annotations


class TestFinalizeAnswer:
    def test_cleans_answer_and_projects_cited_sources_with_public_source_contexts(self) -> None:
        from dlightrag.citations.finalization import finalize_answer

        full_contexts = {
            "chunks": [
                {
                    "chunk_id": "c1",
                    "reference_id": "1",
                    "file_path": "/private/report.pdf",
                    "content": "Evidence in the cited chunk.",
                    "image_data": "base64-image",
                    "_workspace": "default",
                }
            ],
            "entities": [],
            "relationships": [],
        }
        public_contexts = {
            "chunks": [
                {
                    "chunk_id": "c1",
                    "reference_id": "1",
                    "file_path": "/private/report.pdf",
                    "content": "Evidence in the cited chunk.",
                    "image_url": "/images/default/c1?size=full",
                    "thumbnail_url": "/images/default/c1?size=thumb",
                    "_workspace": "default",
                }
            ],
            "entities": [],
            "relationships": [],
        }

        result = finalize_answer(
            "Answer cites valid [1-1] and invalid [9-1].\n\nReferences\n[9] made up",
            full_contexts,
            source_contexts=public_contexts,
        )

        assert result.answer == "Answer cites valid [1-1] and invalid ."
        assert result.cited_chunks == {"1": ["c1"]}
        assert [source.id for source in result.sources] == ["1"]
        assert result.sources[0].chunks is not None
        assert result.sources[0].chunks[0].image_url == "/images/default/c1?size=full"
        assert result.sources[0].chunks[0].thumbnail_url == "/images/default/c1?size=thumb"

    def test_keeps_uncited_answer_when_contexts_are_empty(self) -> None:
        from dlightrag.citations.finalization import finalize_answer

        result = finalize_answer("Plain answer with no retrieved chunks.", {"chunks": []})

        assert result.answer == "Plain answer with no retrieved chunks."
        assert result.sources == []
        assert result.cited_chunks == {}
        assert result.flat_contexts == []

    def test_reuses_one_citation_index_for_sources_and_validation(self, monkeypatch) -> None:
        from dlightrag.citations import indexer as indexer_module
        from dlightrag.citations.finalization import finalize_answer

        calls = 0
        original = indexer_module.CitationIndexer.build_index

        def counted(self, contexts):  # noqa: ANN001, ANN202
            nonlocal calls
            calls += 1
            return original(self, contexts)

        monkeypatch.setattr(indexer_module.CitationIndexer, "build_index", counted)

        result = finalize_answer(
            "Answer cites [1-1].",
            {
                "chunks": [
                    {
                        "chunk_id": "c1",
                        "reference_id": "1",
                        "file_path": "/docs/report.pdf",
                        "content": "Evidence.",
                    }
                ]
            },
        )

        assert result.cited_chunks == {"1": ["c1"]}
        assert calls == 1
