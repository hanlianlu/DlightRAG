"""Citation pattern matching and extraction — ported from sandbox_agent.

Simplified: only supports [ref_id-chunk_idx] format.
Dropped: VLM [vlm_N], tool [sandbox_*], [browser_*] patterns.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .indexer import CitationIndexer

logger = logging.getLogger(__name__)

CITATION_PATTERN = re.compile(r"\[(\w+)-(\d+)\]")


def extract_citation_keys(answer_text: str) -> list[str]:
    """Extract unique citation keys in order of appearance."""
    seen: set[str] = set()
    keys: list[str] = []
    for m in CITATION_PATTERN.finditer(answer_text):
        key = f"{m.group(1)}-{m.group(2)}"
        if key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def extract_cited_chunks(
    indexer: CitationIndexer, answer_text: str
) -> dict[str, list[str]]:
    """Extract cited chunk_ids grouped by ref_id."""
    result: dict[str, list[str]] = {}
    seen: set[tuple[str, str]] = set()

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

    return result


def clean_invalid_citations(
    indexer: CitationIndexer, answer_text: str
) -> str:
    """Remove citations that reference non-existent chunks."""

    def _replace(m: re.Match) -> str:
        ref_id = m.group(1)
        chunk_idx = int(m.group(2))
        if indexer.get_chunk_id(ref_id, chunk_idx) is not None:
            return m.group(0)
        logger.debug("Removing invalid citation [%s-%d]", ref_id, chunk_idx)
        return ""

    return CITATION_PATTERN.sub(_replace, answer_text)
