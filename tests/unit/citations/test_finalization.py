# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for shared answer finalization."""


class TestFinalizeAnswer:
    def test_compact_attachment_reference_preserves_durable_identity(self) -> None:
        from dlightrag.citations.finalization import finalize_answer

        attachment_id = "98ec1e3a-1187-454b-8929-743bd5bc7d4b"
        result = finalize_answer(
            "The worksheet contains fraction problems [att-1-1].",
            {
                "chunks": [
                    {
                        "chunk_id": "composer-chunk-1",
                        "reference_id": "att-1",
                        "full_doc_id": attachment_id,
                        "file_path": "Fractions_Worksheet.docx",
                        "content": "Fractions Challenge Worksheet",
                        "_workspace": "__web_attachment__",
                        "metadata": {
                            "source_type": "web_attachment",
                            "source_uri": f"web-attachment://{attachment_id}",
                            "source_download_locator": f"web-attachment://{attachment_id}",
                        },
                    }
                ]
            },
        )

        assert result.cited_chunks == {"att-1": ["composer-chunk-1"]}
        (source,) = result.sources
        assert source.id == "att-1"
        assert source.document_id == attachment_id
        assert source.source_uri == f"web-attachment://{attachment_id}"

    def test_composer_document_level_reference_cites_all_attachment_chunks(self) -> None:
        from dlightrag.citations.finalization import finalize_answer

        attachment_id = "98ec1e3a-1187-454b-8929-743bd5bc7d4b"
        reference_id = "composer_98ec1e3a1187454b8929743bd5bc7d4b"
        chunks = [
            {
                "chunk_id": f"composer-chunk-{index}",
                "reference_id": reference_id,
                "full_doc_id": attachment_id,
                "file_path": "worksheet.docx",
                "content": f"Evidence {index}",
                "_workspace": "__web_attachment__",
                "metadata": {
                    "source_type": "web_attachment",
                    "source_uri": f"web-attachment://{attachment_id}",
                    "source_download_locator": f"web-attachment://{attachment_id}",
                },
            }
            for index in (1, 2)
        ]

        result = finalize_answer(f"This is a worksheet [{reference_id}].", {"chunks": chunks})

        assert result.cited_chunks == {reference_id: ["composer-chunk-1", "composer-chunk-2"]}

    def test_cleans_answer_and_builds_cited_sources_from_raw_contexts(self) -> None:
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
                    "metadata": {
                        "source_uri": "local://default/report.pdf",
                        "source_download_locator": "/private/report.pdf",
                    },
                }
            ],
            "entities": [],
            "relationships": [],
        }
        result = finalize_answer(
            "Answer cites valid [1-1] and invalid [9-1].\n\nReferences\n[9] made up",
            full_contexts,
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
                        "_workspace": "default",
                        "metadata": {
                            "source_uri": "local://default/report.pdf",
                            "source_download_locator": "/docs/report.pdf",
                        },
                    }
                ]
            },
        )

        assert result.cited_chunks == {"1": ["c1"]}
        assert calls == 1
