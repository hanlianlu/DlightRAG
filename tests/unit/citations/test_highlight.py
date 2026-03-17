import pytest

from dlightrag.citations.highlight import (
    HighlightExtractor,
    HighlightPhrases,
    extract_all_citing_sentences,
)


def test_extract_citing_sentences_basic():
    text = "Market grew fast. Growth was 15% [1-1]. Tech improved too [2-1]."
    result = extract_all_citing_sentences(text)
    assert "1-1" in result
    assert "2-1" in result
    assert any("15%" in s for s in result["1-1"])


def test_extract_citing_sentences_chinese_punctuation():
    text = "市场增长了15%[1-1]。技术也提升了[2-1]。"
    result = extract_all_citing_sentences(text)
    assert "1-1" in result
    assert "2-1" in result


def test_extract_citing_sentences_multiple_citations_one_sentence():
    text = "Both growth [1-1] and tech [2-1] improved."
    result = extract_all_citing_sentences(text)
    assert "1-1" in result
    assert "2-1" in result


def test_extract_citing_sentences_doc_level():
    text = "Revenue grew 15% [1]. Tech improved [2]."
    result = extract_all_citing_sentences(text)
    assert "1" in result
    assert "2" in result
    assert any("15%" in s for s in result["1"])


def test_extract_citing_sentences_mixed_levels():
    text = "Revenue grew 15% [1-1]. Overall the report [1] is positive."
    result = extract_all_citing_sentences(text)
    assert "1-1" in result
    assert "1" in result


def test_extract_citing_sentences_empty():
    assert extract_all_citing_sentences("no citations here") == {}


def test_highlight_phrases_model():
    hp = HighlightPhrases(phrases=["market growth", "15%"], confidence=0.9)
    assert len(hp.phrases) == 2
    assert hp.confidence == 0.9


def test_highlight_phrases_defaults():
    hp = HighlightPhrases()
    assert hp.phrases == []
    assert hp.confidence == 1.0


class TestHighlightExtractor:
    @pytest.fixture()
    def mock_llm(self):
        async def llm_func(*, messages, **kwargs) -> str:
            return '{"phrases": ["market growth"], "confidence": 0.9}'

        return llm_func

    @pytest.fixture()
    def extractor(self, mock_llm):
        return HighlightExtractor(llm_func=mock_llm, cache_size=10)

    @pytest.mark.asyncio
    async def test_extract_highlights(self, extractor):
        result = await extractor.extract_highlights(
            citing_sentence="The market growth was impressive.",
            chunk_content="Reports show market growth reached 15% in 2025.",
            chunk_id="c1",
        )
        assert isinstance(result, HighlightPhrases)
        assert "market growth" in result.phrases

    @pytest.mark.asyncio
    async def test_cache_hit(self, extractor):
        r1 = await extractor.extract_highlights("sentence", "chunk content with data", "c1")
        r2 = await extractor.extract_highlights("sentence", "chunk content with data", "c1")
        assert r1.phrases == r2.phrases

    @pytest.mark.asyncio
    async def test_doc_level_citation_triggers_highlights(self, mock_llm):
        """Doc-level [n] citations should trigger highlights for all chunks of that source."""
        from dlightrag.citations.highlight import extract_highlights_for_sources
        from dlightrag.citations.schemas import ChunkSnippet, SourceReference

        sources = [
            SourceReference(
                id="1",
                path="/docs/report.pdf",
                title="report.pdf",
                chunks=[
                    ChunkSnippet(
                        chunk_id="c1",
                        chunk_idx=1,
                        content="Reports show market growth reached 15% in 2025.",
                    ),
                ],
            ),
        ]
        answer_text = "The market growth was impressive [1]."
        result = await extract_highlights_for_sources(sources, answer_text, mock_llm)
        # Doc-level [1] should match source id="1" and trigger highlights
        assert result[0].chunks[0].highlight_phrases is not None
        assert len(result[0].chunks[0].highlight_phrases) > 0

    @pytest.mark.asyncio
    async def test_invalid_phrases_filtered(self):
        async def bad_llm(*, messages, **kwargs) -> str:
            return '{"phrases": ["hallucinated phrase", "actual text"], "confidence": 0.8}'

        ext = HighlightExtractor(llm_func=bad_llm, cache_size=10)
        result = await ext.extract_highlights(
            citing_sentence="The actual text matters.",
            chunk_content="This has actual text in it.",
            chunk_id="c1",
        )
        assert "actual text" in result.phrases
        assert "hallucinated phrase" not in result.phrases
