from dlightrag.citations.indexer import CitationIndexer
from dlightrag.citations.parser import (
    CITATION_PATTERN,
    DOC_CITATION_PATTERN,
    clean_invalid_citations,
    extract_citation_keys,
    extract_cited_chunks,
)


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
    indexer.build_index(
        [
            {"chunk_id": "c1", "reference_id": "1", "content": "text"},
        ]
    )
    text = "Valid [1-1] and invalid [1-99] and [2-1]."
    cleaned = clean_invalid_citations(indexer, text)
    assert "[1-1]" in cleaned
    assert "[1-99]" not in cleaned
    assert "[2-1]" not in cleaned


def test_extract_cited_chunks():
    indexer = CitationIndexer()
    indexer.build_index(
        [
            {"chunk_id": "c1", "reference_id": "1", "content": "text1"},
            {"chunk_id": "c2", "reference_id": "1", "content": "text2"},
            {"chunk_id": "c3", "reference_id": "2", "content": "text3"},
        ]
    )
    text = "See [1-1] and [2-1]."
    cited = extract_cited_chunks(indexer, text)
    assert "1" in cited
    assert "c1" in cited["1"]
    assert "2" in cited
    assert "c3" in cited["2"]


class TestDocCitationPattern:
    """Test [n] doc-level citation pattern."""

    def test_matches_single_digit(self):
        assert DOC_CITATION_PATTERN.findall("See [1] for details") == ["1"]

    def test_matches_multi_digit(self):
        assert DOC_CITATION_PATTERN.findall("Sources [12] and [3]") == ["12", "3"]

    def test_no_false_positive_on_chunk_format(self):
        """[1-2] should NOT be matched by doc pattern."""
        assert DOC_CITATION_PATTERN.findall("See [1-2]") == []

    def test_no_match_on_empty_brackets(self):
        assert DOC_CITATION_PATTERN.findall("See [] here") == []


class TestExtractCitationKeysDocLevel:
    """Test extract_citation_keys with doc-level [n] format."""

    def test_extracts_doc_level(self):
        keys = extract_citation_keys("Answer based on [1] and [2].")
        assert keys == ["1", "2"]

    def test_extracts_mixed(self):
        keys = extract_citation_keys("From [1] and specifically [1-2].")
        assert keys == ["1", "1-2"]

    def test_deduplicates_doc_level(self):
        keys = extract_citation_keys("[1] agrees with [1].")
        assert keys == ["1"]


class TestExtractCitedChunksDocLevel:
    """Test extract_cited_chunks with doc-level citations."""

    def test_doc_level_maps_to_all_chunks(self):
        indexer = CitationIndexer()
        indexer.build_index([
            {"reference_id": "1", "chunk_id": "c1", "content": "text1"},
            {"reference_id": "1", "chunk_id": "c2", "content": "text2"},
            {"reference_id": "2", "chunk_id": "c3", "content": "text3"},
        ])
        result = extract_cited_chunks(indexer, "Based on [1].")
        assert "1" in result
        assert set(result["1"]) == {"c1", "c2"}


class TestCleanInvalidCitationsDocLevel:
    """Test removal of invalid doc-level citations."""

    def test_keeps_valid_doc_citation(self):
        indexer = CitationIndexer()
        indexer.build_index([
            {"reference_id": "1", "chunk_id": "c1", "content": "text"},
        ])
        assert "[1]" in clean_invalid_citations(indexer, "See [1].")

    def test_removes_invalid_doc_citation(self):
        indexer = CitationIndexer()
        indexer.build_index([
            {"reference_id": "1", "chunk_id": "c1", "content": "text"},
        ])
        cleaned = clean_invalid_citations(indexer, "See [99].")
        assert "[99]" not in cleaned
