"""Citation pattern matching and extraction.

Supports two formats:
- [ref]     — doc-level citations (LightRAG document format)
- [ref-idx] — chunk-level citations (DlightRAG granular format)
"""

import logging
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

from .indexer import CitationIndexer
from .syntax import Citation, CitationDocument, parse_citations

logger = logging.getLogger(__name__)


def extract_cited_chunks(
    indexer: CitationIndexer,
    answer_text: str | CitationDocument,
    *,
    claimless_chunks: frozenset[str] = frozenset(),
) -> dict[str, list[str]]:
    """Extract cited chunk_ids grouped by ref_id.

    Handles both [n] (all chunks for ref) and [n-m] (specific chunk).
    """
    document = parse_citations(answer_text) if isinstance(answer_text, str) else answer_text
    result: defaultdict[str, list[str]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for original in document.citations:
        citation = _validated_citation(indexer, original, claimless_chunks)
        if citation is None:
            continue
        ref_id, chunk_idx = citation.ref_id, citation.chunk_idx
        if chunk_idx is not None:
            chunk_id = indexer.get_chunk_id(ref_id, chunk_idx)
            if chunk_id is None:
                logger.debug("Invalid citation [%s-%d]: no chunk found", ref_id, chunk_idx)
                continue
            if (ref_id, chunk_id) in seen:
                continue
            seen.add((ref_id, chunk_id))
            result[ref_id].append(chunk_id)
            continue

        max_idx = indexer.get_max_chunk_idx(ref_id)
        if max_idx == 0:
            logger.debug("Invalid citation [%s]: no chunks found", ref_id)
        for idx in range(1, max_idx + 1):
            chunk_id = indexer.get_chunk_id(ref_id, idx)
            if chunk_id and (ref_id, chunk_id) not in seen:
                seen.add((ref_id, chunk_id))
                result[ref_id].append(chunk_id)

    return result


_HEADING_LINE_RE = re.compile(r"^#{1,6}[ \t]")


def claimless_chunk_ids(contexts: Iterable[dict[str, Any]]) -> frozenset[str]:
    """Chunk ids whose text is nothing but Markdown headings.

    Such an excerpt states no fact, so a claim can never be drawn from it. A
    heading-only chunk that carries an image is excluded -- the image is evidence.
    """
    claimless: set[str] = set()
    for ctx in contexts:
        chunk_id = ctx.get("chunk_id")
        if not chunk_id or ctx.get("image_data"):
            continue
        lines = [ln.strip() for ln in str(ctx.get("content") or "").split("\n") if ln.strip()]
        if lines and all(_HEADING_LINE_RE.match(ln) for ln in lines):
            claimless.add(str(chunk_id))
    return frozenset(claimless)


def clean_invalid_citations(
    indexer: CitationIndexer,
    answer_text: str | CitationDocument,
    *,
    claimless_chunks: frozenset[str] = frozenset(),
) -> str:
    """Remove citations that reference non-existent chunks/docs.

    A marker resolving to a claimless excerpt is degraded to its document
    marker: the claim is supported by the document, just not by that excerpt.
    """

    document = parse_citations(answer_text) if isinstance(answer_text, str) else answer_text

    def replacement(original: Citation) -> str:
        citation = _validated_citation(indexer, original, claimless_chunks)
        return citation.marker if citation is not None else ""

    return document.rewrite(replacement)


def _validated_citation(
    indexer: CitationIndexer, citation: Citation, claimless_chunks: frozenset[str]
) -> Citation | None:
    if citation.chunk_idx is None:
        return citation if indexer.get_max_chunk_idx(citation.ref_id) > 0 else None
    chunk_id = indexer.get_chunk_id(citation.ref_id, citation.chunk_idx)
    if chunk_id is None:
        return None
    if chunk_id in claimless_chunks:
        return replace(citation, chunk_idx=None, marker=f"[{citation.ref_id}]")
    return citation
