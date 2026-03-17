from dlightrag.citations.schemas import ChunkSnippet, SourceReference


def test_chunk_snippet_minimal():
    cs = ChunkSnippet(chunk_id="abc123", content="some text")
    assert cs.chunk_id == "abc123"
    assert cs.chunk_idx is None
    assert cs.page_idx is None
    assert cs.highlight_phrases is None


def test_chunk_snippet_full():
    cs = ChunkSnippet(
        chunk_id="abc123",
        chunk_idx=2,
        page_idx=3,
        content="market growth reached 15%",
        highlight_phrases=["15%"],
    )
    assert cs.chunk_idx == 2
    assert cs.highlight_phrases == ["15%"]


def test_source_reference_minimal():
    sr = SourceReference(id="1", path="/docs/report.pdf")
    assert sr.id == "1"
    assert sr.title is None
    assert sr.chunks is None


def test_source_reference_with_chunks():
    chunk = ChunkSnippet(chunk_id="c1", chunk_idx=1, content="text")
    sr = SourceReference(
        id="1",
        path="/docs/report.pdf",
        title="Report",
        type="pdf",
        chunks=[chunk],
        cited_chunk_ids=["c1"],
    )
    assert len(sr.chunks) == 1
    assert sr.cited_chunk_ids == ["c1"]
