"""Citation pattern matching and extraction.

Supports two formats:
- [n]       — doc-level citations (LightRAG's native format)
- [ref-idx] — chunk-level citations (DlightRAG granular format)
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .indexer import CitationIndexer

logger = logging.getLogger(__name__)

# Chunk-level: [ref_id-chunk_idx] e.g., [1-2], [abc-3]
CITATION_PATTERN = re.compile(r"\[(\w+)-(\d+)\]")

# Doc-level: [n] e.g., [1], [12] — must NOT match [1-2]
DOC_CITATION_PATTERN = re.compile(r"\[(\d+)\](?![\w-])")


def extract_citation_keys(answer_text: str) -> list[str]:
    """Extract unique citation keys in order of appearance.

    Returns both doc-level ("1") and chunk-level ("1-2") keys.
    """
    seen: set[str] = set()
    keys: list[str] = []

    # Find all citation positions for ordered extraction
    positions: list[tuple[int, str]] = []

    for m in DOC_CITATION_PATTERN.finditer(answer_text):
        key = m.group(1)
        positions.append((m.start(), key))

    for m in CITATION_PATTERN.finditer(answer_text):
        key = f"{m.group(1)}-{m.group(2)}"
        positions.append((m.start(), key))

    positions.sort(key=lambda x: x[0])
    for _, key in positions:
        if key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def extract_cited_chunks(indexer: CitationIndexer, answer_text: str) -> dict[str, list[str]]:
    """Extract cited chunk_ids grouped by ref_id.

    Handles both [n] (all chunks for ref) and [n-m] (specific chunk).
    """
    result: dict[str, list[str]] = {}
    seen: set[tuple[str, str]] = set()

    # Chunk-level citations: [ref_id-chunk_idx]
    for m in CITATION_PATTERN.finditer(answer_text):
        ref_id = m.group(1)
        chunk_idx = int(m.group(2))
        chunk_id = indexer.get_chunk_id(ref_id, chunk_idx)

        if chunk_id is None:
            logger.debug("Invalid citation [%s-%d]: no chunk found", ref_id, chunk_idx)
            continue

        if (ref_id, chunk_id) in seen:
            continue
        seen.add((ref_id, chunk_id))
        result.setdefault(ref_id, []).append(chunk_id)

    # Doc-level citations: [n]
    for m in DOC_CITATION_PATTERN.finditer(answer_text):
        ref_id = m.group(1)

        max_idx = indexer.get_max_chunk_idx(ref_id)
        if max_idx == 0:
            logger.debug("Invalid citation [%s]: no chunks found", ref_id)
            continue

        for idx in range(1, max_idx + 1):
            chunk_id = indexer.get_chunk_id(ref_id, idx)
            if chunk_id and (ref_id, chunk_id) not in seen:
                seen.add((ref_id, chunk_id))
                result.setdefault(ref_id, []).append(chunk_id)

    return result


def clean_invalid_citations(indexer: CitationIndexer, answer_text: str) -> str:
    """Remove citations that reference non-existent chunks/docs."""

    def _replace_chunk(m: re.Match) -> str:
        ref_id = m.group(1)
        chunk_idx = int(m.group(2))
        if indexer.get_chunk_id(ref_id, chunk_idx) is not None:
            return m.group(0)
        logger.debug("Removing invalid citation [%s-%d]", ref_id, chunk_idx)
        return ""

    def _replace_doc(m: re.Match) -> str:
        ref_id = m.group(1)
        if indexer.get_max_chunk_idx(ref_id) > 0:
            return m.group(0)
        logger.debug("Removing invalid citation [%s]", ref_id)
        return ""

    text = CITATION_PATTERN.sub(_replace_chunk, answer_text)
    text = DOC_CITATION_PATTERN.sub(_replace_doc, text)
    return text
