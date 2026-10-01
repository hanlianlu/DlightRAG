# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for dynamic contract chunk rendering in _build_excerpt_blocks."""

from dlightrag.engine.answer.citations.indexer import CitationIndexer
from dlightrag.engine.answer.excerpts import (
    build_excerpt_lane_blocks,
    build_image_label,
    format_chunk_metadata,
)
from dlightrag.engine.answer.tools.web_search import web_context_rows
from dlightrag.engine.answer.web_sources import WebSearchHit


def _heading_and_label(row: dict) -> tuple[str, str]:
    indexer = CitationIndexer()
    indexer.build_index([row])
    heading, passage = build_excerpt_lane_blocks([row], indexer=indexer)
    return heading["text"], passage["text"].split("\n", 1)[0]


def test_a_corpus_heading_describes_the_document_and_keeps_its_storage_from_the_model() -> None:
    """Live knowledge-base results printed `/app/dlightrag_storage/...` locators, the
    `local://` source uri and the stored file name in every heading. Citations need
    them; the model needs what describes the document."""
    row = {
        "chunk_id": "c1",
        "reference_id": "1",
        "content": "Body",
        "file_path": "Macroeconomics (2017, Pearson).pdf",
        "_workspace": "hlyu",
        "metadata": {
            "author": "Blanchard",
            "region": "EU",
            "source_uri": "local://hlyu/Macroeconomics%20%282017%2C%20Pearson%29.pdf",
            "source_download_locator": (
                "/app/dlightrag_storage/corpus/hlyu/__local_sources__/"
                "Macroeconomics (2017, Pearson).pdf"
            ),
            "source_file_name": "Macroeconomics__2017__Pearson_.pdf",
        },
    }

    assert _heading_and_label(row) == (
        "### Document [1] [workspace: hlyu]: Macroeconomics (2017, Pearson).pdf "
        "(author: Blanchard, region: EU)",
        "[1-1] Macroeconomics (2017, Pearson).pdf",
    )


def test_a_web_heading_names_the_page_by_its_whole_title_and_links_it() -> None:
    """A title is no path: `2026/9/21-2026/9/27` once left `27】...` as the page's name.
    The address stays, since an answer links it; acquisition bookkeeping, the page's
    remote picture and the request's own workspace do not."""
    (row,) = web_context_rows(
        [
            WebSearchHit(
                url="https://news.example/bonds",
                title="Bond weekly 2026/9/21-2026/9/27",
                text="Body",
                published_date="2026-09-28",
                image_url="https://news.example/logo.png",
            )
        ]
    )
    row["reference_id"] = "1"
    row["metadata"]["resource_id"] = "res-1"

    assert _heading_and_label(row) == (
        "### Document [1]: Bond weekly 2026/9/21-2026/9/27 [resource: res-1] "
        "(source uri: https://news.example/bonds, published date: 2026-09-28)",
        "[1-1] Bond weekly 2026/9/21-2026/9/27",
    )


class TestFormatChunkMetadata:
    def test_empty_when_no_extra_fields(self) -> None:
        chunk = {
            "chunk_id": "c1",
            "chunk_idx": 2,
            "reference_id": "r1",
            "content": "Hello",
            "file_path": "/docs/x.pdf",
        }
        assert format_chunk_metadata(chunk) == ""

    def test_skips_internal_keys(self) -> None:
        chunk = {
            "chunk_id": "c1",
            "reference_id": "r1",
            "content": "Hello",
            "image_data": "base64...",
            "file_path": "/docs/x.pdf",
            "metadata": {"title": "T"},
            "page_number": 3,
            "image_mime_type": "image/png",
            "image_url": "/images/default/c1?size=full",
            "thumbnail_url": "/images/default/c1?size=thumb",
            "relevance_score": 0.85,
            "rerank_score": 0.91,
            "score": 0.22,
            "distance": 0.78,
            "bm25_profile": "english",
            "full_doc_id": "doc-abc123",
            "_workspace": "default",
            "_answer_image_sent": True,
            "sidecar": {"type": "drawing", "id": "im-hash-1"},
            "sidecar_location": "file:///tmp/report.parsed",
            "pipeline_stage": "rerank",
        }
        result = format_chunk_metadata(chunk)
        assert result == ""
        assert "chunk_id" not in result
        assert "chunk_idx" not in result
        assert "image_data" not in result
        assert "_answer_image_sent" not in result
        assert "relevance_score" not in result
        assert "rerank_score" not in result
        assert "image_url" not in result
        assert "thumbnail_url" not in result
        assert "score" not in result
        assert "distance" not in result
        assert "bm25_profile" not in result
        assert "_workspace" not in result
        assert "full_doc_id" not in result
        assert "sidecar" not in result
        assert "sidecar_location" not in result
        assert "pipeline_stage" not in result

    def test_formats_simple_scalar_fields(self) -> None:
        chunk = {"section_title": "Risk Factors", "chunk_id": "c1"}
        result = format_chunk_metadata(chunk)
        assert "section_title=Risk Factors" in result
        assert "[meta:" in result

    def test_formats_dict_fields_with_dot_notation(self) -> None:
        chunk = {
            "attributes": {"department": "legal", "asset_id": "asset-123"},
            "chunk_id": "c1",
        }
        result = format_chunk_metadata(chunk)
        assert "attributes.department=legal" in result
        assert "attributes.asset_id=asset-123" in result

    def test_formats_list_fields_truncated(self) -> None:
        chunk = {
            "tags": ["a", "b", "c", "d", "e", "f"],
            "chunk_id": "c1",
        }
        result = format_chunk_metadata(chunk)
        assert "tags=[" in result
        assert "...(6 total)" in result

    def test_skips_none_and_empty_values(self) -> None:
        chunk = {
            "score": None,
            "sidecar": {},
            "empty_str": "",
            "chunk_id": "c1",
        }
        result = format_chunk_metadata(chunk)
        assert result == ""

    def test_truncates_long_string_values(self) -> None:
        chunk = {
            "long_field": "x" * 200,
            "chunk_id": "c1",
        }
        result = format_chunk_metadata(chunk)
        assert len(result) < 200
        assert "..." in result

    def test_handles_bool_values(self) -> None:
        chunk = {"is_visual": True, "chunk_id": "c1"}
        result = format_chunk_metadata(chunk)
        assert "is_visual=True" in result

    def test_skips_underscore_prefixed_keys(self) -> None:
        chunk = {"_private": "secret", "business": {"type": "contract"}, "chunk_id": "c1"}
        result = format_chunk_metadata(chunk)
        assert "_private" not in result
        assert "business.type=contract" in result


class TestBuildImageLabel:
    def test_basic_label_with_citation_and_page(self) -> None:
        chunk = {"page_number": 7, "metadata": {"title": "My Report"}}
        result = build_image_label(cite_tag="[1-2]", chunk=chunk, filename="report.pdf")
        assert '[1-2] "My Report" Page 7' == result

    def test_label_with_vlm_drawing_sidecar(self) -> None:
        chunk = {
            "page_number": 3,
            "metadata": {},
            "sidecar": {"type": "drawing", "id": "im-hash-abc123def456"},
        }
        result = build_image_label(cite_tag="[1-1]", chunk=chunk, filename="paper.pdf")
        assert "(VLM drawing: im-hash-abc123def456)" in result
        assert "[1-1]" in result

    def test_label_with_drawing_sidecar_no_id(self) -> None:
        chunk = {
            "metadata": {},
            "sidecar": {"type": "drawing"},
        }
        result = build_image_label(cite_tag="[1-1]", chunk=chunk, filename="paper.pdf")
        assert "(VLM-generated drawing)" in result

    def test_label_with_other_sidecar_type(self) -> None:
        chunk = {
            "metadata": {},
            "sidecar": {"type": "custom_visual", "path": "/img/x.png"},
        }
        result = build_image_label(cite_tag="[1-1]", chunk=chunk, filename="paper.pdf")
        assert "(sidecar: custom_visual)" in result

    def test_label_falls_back_to_filename(self) -> None:
        chunk = {"metadata": {}}
        result = build_image_label(cite_tag="", chunk=chunk, filename="photo.jpg")
        assert "photo.jpg" == result
