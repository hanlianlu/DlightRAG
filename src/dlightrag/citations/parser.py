"""Citation pattern matching and extraction.

Supports two formats:
- [ref]     — doc-level citations (LightRAG and attachment formats)
- [ref-idx] — chunk-level citations (DlightRAG granular format)
"""

import logging
import re
from collections import defaultdict

from .indexer import CitationIndexer

logger = logging.getLogger(__name__)

_CHUNK_INDEX = r"[0-9]{1,9}"
_NUMERIC_REF = r"[0-9]+"
_LEGACY_COMPOSER_REF = r"composer_[0-9a-f]+"
_ATTACHMENT_REF = r"att-[0-9]+"
# ``att-N`` is a reserved document namespace, so the generic branch excludes
# interpreting it as reference ``att`` plus chunk N.
_GENERIC_CHUNK_REF = rf"(?!att-{_CHUNK_INDEX}\])\w+"
_CHUNK_REFERENCE_ID = (
    rf"(?:{_ATTACHMENT_REF}|{_NUMERIC_REF}|{_LEGACY_COMPOSER_REF}|{_GENERIC_CHUNK_REF})"
)

# Keep the trailing guard ASCII-only so citations remain valid before Chinese text.
_CITATION_TRAILING_BOUNDARY = r"(?![A-Za-z0-9_-])"

# Chunk-level: [ref_id-chunk_idx] e.g., [1-2], [abc-3], [att-1-2].
CITATION_PATTERN = re.compile(
    rf"\[({_CHUNK_REFERENCE_ID})-({_CHUNK_INDEX})\]{_CITATION_TRAILING_BOUNDARY}"
)

# Doc-level: [n], [att-N], or [composer_<uuidhex>] — must NOT match chunk refs or
# adjacent identifier text. The Composer namespace is deliberately narrow so
# ordinary Markdown labels do not become citations.
DOC_CITATION_PATTERN = re.compile(
    rf"\[((?:{_NUMERIC_REF}|{_ATTACHMENT_REF}|{_LEGACY_COMPOSER_REF}))\]"
    rf"{_CITATION_TRAILING_BOUNDARY}"
)


def _parse_chunk_index(match: re.Match[str]) -> int | None:
    try:
        return int(match.group(2))
    except TypeError, ValueError:
        logger.debug("Ignoring citation with an invalid chunk index")
        return None


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
    positions: list[tuple[int, str, int | None]] = []
    for m in CITATION_PATTERN.finditer(answer_text):
        chunk_idx = _parse_chunk_index(m)
        if chunk_idx is not None:
            positions.append((m.start(), m.group(1), chunk_idx))
    for m in DOC_CITATION_PATTERN.finditer(answer_text):
        positions.append((m.start(), m.group(1), None))

    result: defaultdict[str, list[str]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    for _, ref_id, chunk_idx in sorted(positions, key=lambda item: item[0]):
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


def clean_invalid_citations(indexer: CitationIndexer, answer_text: str) -> str:
    """Remove citations that reference non-existent chunks/docs."""

    def _replace_chunk(m: re.Match) -> str:
        ref_id = m.group(1)
        chunk_idx = _parse_chunk_index(m)
        if chunk_idx is None:
            return m.group(0)
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


# Matches generated reference-section headings at a line boundary:
# # References, ## References, **References**, References:, References
_REFERENCES_HEADING_RE = re.compile(r"(?im)^\s{0,3}(?:#{1,6}\s*|\*{2})?references(?:\*{2})?[:\s]*$")


def strip_generated_references_section(answer_text: str) -> str:
    """Strip a model-generated trailing References section from answer text.

    DlightRAG builds cited sources deterministically from validated inline
    markers. A model-generated ``### References`` tail is therefore protocol
    noise, not a trusted data source.
    """
    match = None
    for candidate in _REFERENCES_HEADING_RE.finditer(answer_text):
        match = candidate
    if match is None:
        return answer_text
    return answer_text[: match.start()].rstrip()
