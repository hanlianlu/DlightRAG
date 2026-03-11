"""Citation processing and semantic highlighting for DlightRAG."""

from .highlight import HighlightExtractor, extract_highlights_for_sources
from .indexer import CitationIndexer, build_citation_index
from .parser import extract_citation_keys, extract_cited_chunks
from .processor import CitationProcessor, CitationResult
from .schemas import ChunkSnippet, SourceReference

__all__ = [
    "ChunkSnippet",
    "CitationIndexer",
    "CitationProcessor",
    "CitationResult",
    "HighlightExtractor",
    "SourceReference",
    "build_citation_index",
    "extract_citation_keys",
    "extract_cited_chunks",
    "extract_highlights_for_sources",
]
