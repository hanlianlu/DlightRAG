"""Tests for build_sources() shared utility."""

from __future__ import annotations

from dlightrag.citations.source_builder import build_sources


def _chunk(
    chunk_id: str,
    reference_id: str,
    file_path: str = "/data/report.pdf",
    content: str = "text",
    image_data: str | None = None,
    page_idx: int | None = None,
) -> dict:
    return {
        "chunk_id": chunk_id,
        "reference_id": reference_id,
        "file_path": file_path,
        "content": content,
        "image_data": image_data,
        "page_idx": page_idx,
        "metadata": {"file_name": file_path.rsplit("/", 1)[-1]},
    }


class TestBuildSources:
    def test_groups_by_reference_id(self) -> None:
        contexts = {
            "chunks": [
                _chunk("c1", "ref-1", page_idx=1),
                _chunk("c2", "ref-1", page_idx=2),
                _chunk("c3", "ref-2", file_path="/data/chart.xlsx"),
            ],
            "entities": [],
            "relationships": [],
        }
        sources = build_sources(contexts)

        assert len(sources) == 2
        assert sources[0].id == "ref-1"
        assert len(sources[0].chunks) == 2
        assert sources[1].id == "ref-2"
        assert len(sources[1].chunks) == 1

    def test_chunks_sorted_by_page_idx(self) -> None:
        contexts = {
            "chunks": [
                _chunk("c2", "ref-1", page_idx=3),
                _chunk("c1", "ref-1", page_idx=1),
                _chunk("c3", "ref-1", page_idx=2),
            ],
        }
        sources = build_sources(contexts)

        page_order = [c.page_idx for c in sources[0].chunks]
        assert page_order == [1, 2, 3]

    def test_source_title_from_filename(self) -> None:
        contexts = {"chunks": [_chunk("c1", "ref-1", file_path="/long/path/report.pdf")]}
        sources = build_sources(contexts)

        assert sources[0].title == "report.pdf"
        assert sources[0].path == "/long/path/report.pdf"

    def test_preserves_image_data(self) -> None:
        contexts = {"chunks": [_chunk("c1", "ref-1", image_data="base64data", page_idx=1)]}
        sources = build_sources(contexts)

        assert sources[0].chunks[0].image_data == "base64data"

    def test_empty_contexts(self) -> None:
        assert build_sources({}) == []
        assert build_sources({"chunks": []}) == []

    def test_source_order_by_first_appearance(self) -> None:
        contexts = {
            "chunks": [
                _chunk("c1", "ref-2"),
                _chunk("c2", "ref-1"),
                _chunk("c3", "ref-2"),
            ],
        }
        sources = build_sources(contexts)

        assert sources[0].id == "ref-2"
        assert sources[1].id == "ref-1"

    def test_chunk_idx_assigned_sequentially(self) -> None:
        contexts = {
            "chunks": [
                _chunk("c1", "ref-1", page_idx=1),
                _chunk("c2", "ref-1", page_idx=2),
            ],
        }
        sources = build_sources(contexts)

        assert sources[0].chunks[0].chunk_idx == 0
        assert sources[0].chunks[1].chunk_idx == 1

    def test_none_page_idx_sorted_last(self) -> None:
        contexts = {
            "chunks": [
                _chunk("c1", "ref-1", page_idx=None),
                _chunk("c2", "ref-1", page_idx=1),
            ],
        }
        sources = build_sources(contexts)

        assert sources[0].chunks[0].page_idx == 1
        assert sources[0].chunks[1].page_idx is None

    def test_with_path_resolver(self) -> None:
        contexts = {"chunks": [_chunk("c1", "ref-1", file_path="/data/report.pdf")]}

        class FakeResolver:
            def resolve(self, path: str) -> str:
                return f"https://cdn.example.com/{path}"

        sources = build_sources(contexts, path_resolver=FakeResolver())

        assert sources[0].url == "https://cdn.example.com//data/report.pdf"

    def test_skips_chunks_without_reference_id(self) -> None:
        contexts = {
            "chunks": [
                _chunk("c1", ""),
                _chunk("c2", "ref-1"),
            ],
        }
        sources = build_sources(contexts)

        assert len(sources) == 1
        assert sources[0].id == "ref-1"
