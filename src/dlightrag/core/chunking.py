# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Docling HybridChunker wrapper conforming to LightRAG's chunking_func signature.

LightRAG accepts a ``chunking_func`` callback for splitting merged document text
into token-counted chunks.  This module provides an implementation that:

1. Reconstructs a ``DoclingDocument`` from the merged markdown/plain-text string.
2. Runs Docling's ``HybridChunker`` for structure-aware hierarchical chunking.
3. Returns chunk dicts compatible with LightRAG's internal pipeline.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from docling.chunking import HybridChunker
from docling_core.types.doc import DocItemLabel, DoclingDocument

logger = logging.getLogger(__name__)

# Regex matching markdown headings (# through ####).
_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$")


def _build_docling_document(content: str) -> DoclingDocument:
    """Parse a merged text string into a DoclingDocument.

    Detects markdown headings (``# ## ### ####``) and adds them via
    ``doc.add_heading``.  All other non-empty text blocks (separated by
    blank lines) are added as ``PARAGRAPH`` items.
    """
    doc = DoclingDocument(name="chunking_input")
    blocks = re.split(r"\n{2,}", content)

    for block in blocks:
        text = block.strip()
        if not text:
            continue

        heading_match = _HEADING_RE.match(text)
        if heading_match:
            level = len(heading_match.group(1))
            heading_text = heading_match.group(2).strip()
            doc.add_heading(text=heading_text, level=level)
        else:
            doc.add_text(text=text, label=DocItemLabel.PARAGRAPH)

    return doc


def docling_hybrid_chunking_func(
    tokenizer: Any,  # noqa: ARG001 – required by LightRAG signature
    content: str,
    split_by_character: str | None,  # noqa: ARG001
    split_by_character_only: bool,  # noqa: ARG001
    chunk_overlap_token_size: int,  # noqa: ARG001
    chunk_token_size: int,
) -> list[dict[str, Any]]:
    """LightRAG-compatible chunking function using Docling's HybridChunker.

    Parameters
    ----------
    tokenizer:
        Ignored.  HybridChunker uses its own tokenizer internally.
    content:
        Merged document text (plain or with markdown headings).
    split_by_character:
        Ignored (LightRAG default chunking param).
    split_by_character_only:
        Ignored (LightRAG default chunking param).
    chunk_overlap_token_size:
        Ignored (HybridChunker handles overlap via its own strategy).
    chunk_token_size:
        Maximum tokens per chunk, forwarded to ``HybridChunker(max_tokens=...)``.

    Returns
    -------
    list[dict[str, Any]]
        Each dict contains ``content``, ``tokens``, ``chunk_order_index``,
        and ``_raw_content``.
    """
    if not content or not content.strip():
        return []

    doc = _build_docling_document(content)
    chunker = HybridChunker(max_tokens=chunk_token_size)

    results: list[dict[str, Any]] = []
    for idx, chunk in enumerate(chunker.chunk(doc)):
        contextualized = chunker.contextualize(chunk)
        raw_text = chunk.text
        token_count = chunker.tokenizer.count_tokens(contextualized)

        results.append(
            {
                "content": contextualized,
                "tokens": token_count,
                "chunk_order_index": idx,
                "_raw_content": raw_text,
            }
        )

    return results
