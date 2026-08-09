# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Resource-neutral lexical focus ranking.

A single mixed-script tokenizer emits Unicode words plus CJK unigrams and
overlapping bigrams so Latin and CJK queries share one BM25 index. The ranking
core is source agnostic: it operates on tokenized documents and returns their
indices, so any caller (structural resource windows, Composer rows) can reuse it
without pulling in a domain shape.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import bm25s


@runtime_checkable
class StructuralWindow(Protocol):
    """A rankable resource window that exposes its readable text."""

    @property
    def text(self) -> str: ...


def _is_cjk_char(char: str) -> bool:
    code = ord(char)
    return (
        0x3400 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x20000 <= code <= 0x2FA1F
        or 0x30000 <= code <= 0x323AF
        or 0x3040 <= code <= 0x30FF
        or 0x31F0 <= code <= 0x31FF
        or 0xAC00 <= code <= 0xD7AF
    )


def mixed_script_terms(text: str) -> list[str]:
    """Tokenize Unicode words and emit CJK unigrams plus overlapping bigrams."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    terms: list[str] = []
    word: list[str] = []
    cjk: list[str] = []

    def flush_word() -> None:
        if word:
            terms.append("".join(word))
            word.clear()

    def flush_cjk() -> None:
        if not cjk:
            return
        terms.extend(cjk)
        terms.extend(first + second for first, second in zip(cjk, cjk[1:], strict=False))
        cjk.clear()

    for char in normalized:
        if _is_cjk_char(char):
            flush_word()
            cjk.append(char)
            continue
        flush_cjk()
        category = unicodedata.category(char)
        if category[0] in {"L", "N"} or char == "_":
            word.append(char)
        elif category[0] == "M" and word:
            word.append(char)
        elif char in {"\u200c", "\u200d"} and word:
            word.append(char)
        else:
            flush_word()
    flush_word()
    flush_cjk()
    return terms


def bm25_rank(
    query_terms: Sequence[str],
    documents: Sequence[Sequence[str]],
    *,
    limit: int,
) -> list[tuple[int, float]]:
    """Rank tokenized *documents* against *query_terms* with Lucene BM25.

    Returns ``(document_index, score)`` pairs, most relevant first, with every
    zero-score hit discarded. Empty documents are skipped but their original
    positions are preserved in the returned indices. An empty query, empty
    corpus, or non-positive ``limit`` yields no results.
    """
    if limit <= 0 or not query_terms:
        return []

    indexed: list[tuple[int, list[str]]] = []
    vocabulary: set[str] = set()
    for position, terms in enumerate(documents):
        token_list = list(terms)
        if not token_list:
            continue
        indexed.append((position, token_list))
        vocabulary.update(token_list)
    if not indexed:
        return []

    known_query_terms = [term for term in query_terms if term in vocabulary]
    if not known_query_terms:
        return []

    retriever = bm25s.BM25(
        method="lucene",
        k1=1.2,
        b=0.75,
        backend="numpy",
        csc_backend="numpy",
        auto_compile=False,
    )
    retriever.index(
        [terms for _, terms in indexed],
        create_empty_token=False,
        show_progress=False,
    )
    result_ids, result_scores = retriever.retrieve(
        [known_query_terms],
        k=min(limit, len(indexed)),
        show_progress=False,
    )

    ranked: list[tuple[int, float]] = []
    for raw_index, raw_score in zip(result_ids[0], result_scores[0], strict=True):
        score = float(raw_score)
        if score <= 0:
            continue
        ranked.append((indexed[int(raw_index)][0], score))
    return ranked


def rank_resource_windows[W: StructuralWindow](
    query: str,
    windows: Sequence[W],
    *,
    limit: int,
) -> list[W]:
    """Return the most query-relevant *windows*, discarding zero-score matches."""
    query_terms = mixed_script_terms(query)
    documents = [mixed_script_terms(window.text) for window in windows]
    ranked = bm25_rank(query_terms, documents, limit=limit)
    return [windows[index] for index, _ in ranked]


__all__ = ["StructuralWindow", "bm25_rank", "mixed_script_terms", "rank_resource_windows"]
