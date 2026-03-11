# tests/unit/citations/test_processor.py
from dlightrag.citations.processor import CitationProcessor, CitationResult
from dlightrag.citations.schemas import SourceReference


def _make_contexts():
    """Contexts simulating DlightRAG RetrievalResult.contexts flattened."""
    return [
        {
            "chunk_id": "c1",
            "reference_id": "1",
            "content": "Market growth reached 15% this year.",
            "metadata": {"file_name": "report.pdf", "page_idx": 3},
        },
        {
            "chunk_id": "c2",
            "reference_id": "1",
            "content": "Revenue exceeded expectations.",
            "metadata": {"file_name": "report.pdf", "page_idx": 5},
        },
        {
            "chunk_id": "c3",
            "reference_id": "2",
            "content": "Technical indicators show positive trend.",
            "metadata": {"file_name": "analysis.xlsx", "page_idx": 1},
        },
    ]


def _make_sources():
    return [
        SourceReference(id="1", path="/docs/report.pdf", title="Report", type="pdf"),
        SourceReference(id="2", path="/docs/analysis.xlsx", title="Analysis", type="xlsx"),
    ]


def test_process_basic():
    contexts = _make_contexts()
    sources = _make_sources()
    proc = CitationProcessor(contexts=contexts, available_sources=sources)
    result = proc.process("Growth was strong [1-1] and trends are positive [2-1].")

    assert isinstance(result, CitationResult)
    assert "[1-1]" in result.answer
    assert "[2-1]" in result.answer
    assert len(result.sources) == 2
    src1 = next(s for s in result.sources if s.id == "1")
    assert src1.chunks is not None
    assert src1.chunks[0].chunk_id == "c1"


def test_process_removes_invalid_citations():
    contexts = _make_contexts()
    sources = _make_sources()
    proc = CitationProcessor(contexts=contexts, available_sources=sources)
    result = proc.process("Valid [1-1] and hallucinated [3-1] and [1-99].")

    assert "[1-1]" in result.answer
    assert "[3-1]" not in result.answer
    assert "[1-99]" not in result.answer


def test_process_empty_answer():
    proc = CitationProcessor(contexts=[], available_sources=[])
    result = proc.process("")
    assert result.answer == ""
    assert result.sources == []


def test_cited_chunks_populated():
    contexts = _make_contexts()
    sources = _make_sources()
    proc = CitationProcessor(contexts=contexts, available_sources=sources)
    result = proc.process("See [1-1].")
    assert "1" in result.cited_chunks
    assert "c1" in result.cited_chunks["1"]
