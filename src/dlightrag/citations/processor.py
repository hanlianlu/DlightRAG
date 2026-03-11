"""Citation processor — orchestrates parse, validate, build sources.

Simplified from sandbox_agent: no VLM/tool citations, no MediaReference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .indexer import CitationIndexer
from .parser import clean_invalid_citations, extract_cited_chunks
from .schemas import ChunkSnippet, SourceReference
from .utils import filter_content_for_display

logger = logging.getLogger(__name__)


@dataclass
class CitationResult:
    """Output of citation processing."""

    answer: str
    sources: list[SourceReference]
    cited_chunks: dict[str, list[str]] = field(default_factory=dict)


class CitationProcessor:
    """Process raw LLM answer + retrieval contexts into enriched citations."""

    def __init__(
        self,
        contexts: list[dict[str, Any]],
        available_sources: list[SourceReference],
    ) -> None:
        self._available_sources = available_sources
        self._indexer = CitationIndexer()
        self._indexer.build_index(contexts)
        self._enriched_contexts = self._indexer.inject_chunk_idx(contexts)

    def process(self, answer_text: str) -> CitationResult:
        """Clean citations, extract chunks, build source references."""
        cleaned = clean_invalid_citations(self._indexer, answer_text)
        cited_chunks = extract_cited_chunks(self._indexer, cleaned)
        sources = self._build_sources(cited_chunks)

        return CitationResult(
            answer=cleaned,
            sources=sources,
            cited_chunks=cited_chunks,
        )

    def _build_sources(self, cited_chunks: dict[str, list[str]]) -> list[SourceReference]:
        """Build SourceReference list with ChunkSnippet details."""
        if not cited_chunks or not self._available_sources:
            return []

        source_lookup = {s.id: s for s in self._available_sources}
        chunk_lookup: dict[str, dict[str, Any]] = {}
        for ctx in self._enriched_contexts:
            cid = ctx.get("chunk_id")
            if cid:
                chunk_lookup[cid] = ctx

        result: list[SourceReference] = []
        for ref_id, chunk_ids in cited_chunks.items():
            src = source_lookup.get(ref_id)
            if src is None:
                logger.warning("Cited ref_id=%s not in available_sources", ref_id)
                continue

            chunks: list[ChunkSnippet] = []
            for cid in chunk_ids:
                ctx = chunk_lookup.get(cid)
                if ctx is None:
                    logger.warning("Cited chunk_id=%s not found in contexts", cid)
                    continue

                content = filter_content_for_display(ctx.get("content", ""))
                metadata = ctx.get("metadata", {})
                page_idx = ctx.get("page_idx") or metadata.get("page_idx")
                chunks.append(
                    ChunkSnippet(
                        chunk_id=cid,
                        chunk_idx=ctx.get("chunk_idx"),
                        page_idx=page_idx,
                        content=content,
                    )
                )

            result.append(
                SourceReference(
                    id=ref_id,
                    title=src.title,
                    path=src.path,
                    type=src.type,
                    url=src.url,
                    cited_chunk_ids=chunk_ids,
                    chunks=chunks if chunks else None,
                )
            )

        return result
