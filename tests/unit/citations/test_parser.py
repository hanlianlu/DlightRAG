from dlightrag.citations.parser import (
    CITATION_PATTERN,
    extract_citation_keys,
    clean_invalid_citations,
    extract_cited_chunks,
)
from dlightrag.citations.indexer import CitationIndexer


def test_citation_pattern_matches_ref_chunk():
    m = CITATION_PATTERN.search("text [1-2] more")
    assert m is not None
    assert m.group(1) == "1"
    assert m.group(2) == "2"


def test_citation_pattern_no_match():
    m = CITATION_PATTERN.search("no citations here")
    assert m is None


def test_extract_citation_keys_basic():
    text = "Answer [1-2] and [2-1] here [1-2] again."
    keys = extract_citation_keys(text)
    assert keys == ["1-2", "2-1"]


def test_extract_citation_keys_empty():
    assert extract_citation_keys("no citations") == []


def test_clean_invalid_citations():
    indexer = CitationIndexer()
    indexer.build_index([
        {"chunk_id": "c1", "reference_id": "1", "content": "text"},
    ])
    text = "Valid [1-1] and invalid [1-99] and [2-1]."
    cleaned = clean_invalid_citations(indexer, text)
    assert "[1-1]" in cleaned
    assert "[1-99]" not in cleaned
    assert "[2-1]" not in cleaned


def test_extract_cited_chunks():
    indexer = CitationIndexer()
    indexer.build_index([
        {"chunk_id": "c1", "reference_id": "1", "content": "text1"},
        {"chunk_id": "c2", "reference_id": "1", "content": "text2"},
        {"chunk_id": "c3", "reference_id": "2", "content": "text3"},
    ])
    text = "See [1-1] and [2-1]."
    cited = extract_cited_chunks(indexer, text)
    assert "1" in cited
    assert "c1" in cited["1"]
    assert "2" in cited
    assert "c3" in cited["2"]
