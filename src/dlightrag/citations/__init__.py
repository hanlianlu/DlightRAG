"""Citation processing and semantic highlighting for DlightRAG."""

from .finalization import FinalizedAnswer, finalize_answer, flatten_context_chunks
from .highlight import HighlightExtractor, extract_highlights_for_sources
from .indexer import CitationIndexer, build_citation_index
from .parser import extract_citation_keys, extract_cited_chunks
from .processor import CitationProcessor, CitationResult
from .schemas import ChunkSnippet, SourceReference
from .source_builder import build_sources, build_sources_from_chunks

__all__ = [
    "ChunkSnippet",
    "CitationIndexer",
    "CitationProcessor",
    "CitationResult",
    "FinalizedAnswer",
    "HighlightExtractor",
    "SourceReference",
    "build_citation_index",
    "build_sources",
    "build_sources_from_chunks",
    "extract_citation_keys",
    "extract_cited_chunks",
    "extract_highlights_for_sources",
    "finalize_answer",
    "flatten_context_chunks",
]
