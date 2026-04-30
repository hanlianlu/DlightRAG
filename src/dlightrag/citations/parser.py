"""Citation pattern matching and extraction.

Supports two formats:
- [n]       — doc-level citations (LightRAG's native format)
- [ref-idx] — chunk-level citations (DlightRAG granular format)
"""

from __future__ import annotations

import json
import logging
import re

from dlightrag.models.schemas import Reference

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


# Heading regex used by parse_freetext_references
# Matches: # References, ## References, …, ###### References,
#           **References**, References:, References (bare)
_REFERENCES_HEADING_RE = re.compile(
    r"(?m)^(?:#{1,6}\s*|\*{2})?references(?:\*{2})?[:\s]*$",
    re.IGNORECASE,
)

# Ref line: [1] title, [1]: title, 【1】title
_REF_LINE_RE = re.compile(r"(?:\[(\d+)\]|【(\d+)】):?\s*(.+)")


def parse_freetext_references(raw: str) -> tuple[str, list[Reference]]:
    """Level-2 reference extractor: parse a ``### References`` markdown section.

    Splits *raw* on the references heading, then matches lines like
    ``[n] Title``, ``[n]: Title``, or ``【n】Title``. Returns
    ``(answer_text, references)`` with the references section stripped
    from the answer.

    Usually consumed via :func:`extract_references` (JSON first, this
    second). Exposed directly so tests can target the markdown path in
    isolation.
    """
    parts = _REFERENCES_HEADING_RE.split(raw)
    if len(parts) < 2:
        return raw, []

    answer_text = parts[0].rstrip()
    ref_section = parts[-1]

    refs: list[Reference] = []
    for match in _REF_LINE_RE.finditer(ref_section):
        ref_id = int(match.group(1) or match.group(2))
        title = match.group(3).strip().rstrip(".")
        refs.append(Reference(id=ref_id, title=title))

    logger.info(
        "[Parser] parse_freetext_references: found %d refs",
        len(refs),
    )
    return answer_text, refs


def extract_references(raw: str) -> tuple[str, list[Reference]]:
    """Extract references from LLM output with 3-level fallback.

    Returns (cleaned_answer_text, references).

    Level 1 — JSON block: find and parse JSON containing "references" array,
              strip JSON block from answer text.
    Level 2 — ### References section: parse [n] Title lines,
              strip references section from answer text.
    Level 3 — Empty: return original text + empty list.
    """
    # Level 1: JSON block extraction
    answer, refs, found = _try_json_extraction(raw)
    if found:
        logger.info("[RefExtract] Level 1 (JSON block): refs=%d", len(refs))
        return answer, refs

    # Level 2: ### References section
    answer, refs = parse_freetext_references(raw)
    if refs:
        logger.info("[RefExtract] Level 2 (### References): refs=%d", len(refs))
        return answer, refs

    # Level 3: empty
    logger.info("[RefExtract] Level 3: no references found in LLM output")
    return raw, []


def _try_json_extraction(raw: str) -> tuple[str, list[Reference], bool]:
    """Attempt to extract references from a JSON block in the text.

    Returns (answer, refs, json_found). The ``json_found`` flag is True
    when a JSON block was successfully parsed (even if refs is empty),
    so callers can distinguish "no JSON" from "JSON with no refs".
    """
    from dlightrag.utils.text import extract_json

    json_str = extract_json(raw)
    # If extract_json returned the original text unchanged, no JSON found
    if json_str == raw and not raw.lstrip().startswith("{"):
        return raw, [], False

    try:
        data = json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        return raw, [], False

    if not isinstance(data, dict):
        return raw, [], False

    # JSON parsed — determine answer text.
    # If there is preamble text before the JSON block, that freetext IS the
    # answer (the LLM wrote narrative first, then appended a JSON refs block).
    # Only fall back to data["answer"] when the JSON is the entire input.
    stripped = _strip_json_block(raw, json_str)
    if stripped:
        # Preamble exists — use it as the answer regardless of "answer" key
        answer = stripped
    elif "answer" in data:
        answer = data["answer"]
    else:
        answer = ""

    raw_refs = data.get("references")
    if not isinstance(raw_refs, list):
        return answer, [], True

    refs: list[Reference] = []
    for r in raw_refs:
        try:
            refs.append(Reference.model_validate(r))
        except Exception:
            pass

    return answer, refs, True


def _strip_json_block(raw: str, json_str: str) -> str:
    """Strip a JSON block (and surrounding code fences) from raw text."""
    # Try to strip a fenced block first: ```json\n...\n```
    fenced = re.compile(r"```(?:json)?\s*" + re.escape(json_str) + r"\s*```")
    cleaned = fenced.sub("", raw).strip()
    if cleaned != raw.strip():
        return cleaned

    # Try to find the exact JSON substring in raw
    idx = raw.find(json_str)
    if idx >= 0:
        before = raw[:idx].rstrip()
        after = raw[idx + len(json_str) :].lstrip()
        parts = [p for p in (before, after) if p]
        return "\n\n".join(parts) if parts else ""
    # Fallback: strip from first { onward
    start = raw.find("{")
    if start > 0:
        return raw[:start].rstrip()
    return ""
