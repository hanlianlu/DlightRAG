# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""In-memory BM25 for Web Composer attachment chunks."""

from __future__ import annotations

import unicodedata

import bm25s

from dlightrag.core.retrieval.protocols import ContextRow


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


def rank_composer_bm25(
    query: str,
    rows: list[ContextRow],
    *,
    limit: int,
) -> list[ContextRow]:
    """Rank Composer rows with Lucene BM25 and discard every zero-score hit."""
    query_terms = mixed_script_terms(query)
    if not rows or limit <= 0 or not query_terms:
        return []

    indexed_rows: list[tuple[int, list[str]]] = []
    vocabulary: set[str] = set()
    for row_index, row in enumerate(rows):
        terms = mixed_script_terms(str(row.get("content") or ""))
        if not terms:
            continue
        indexed_rows.append((row_index, terms))
        vocabulary.update(terms)
    if not indexed_rows:
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
        [terms for _, terms in indexed_rows],
        create_empty_token=False,
        show_progress=False,
    )
    result_ids, result_scores = retriever.retrieve(
        [known_query_terms],
        k=min(limit, len(indexed_rows)),
        show_progress=False,
    )

    ranked: list[ContextRow] = []
    for raw_index, raw_score in zip(result_ids[0], result_scores[0], strict=True):
        score = float(raw_score)
        if score <= 0:
            continue
        source_index = indexed_rows[int(raw_index)][0]
        row = dict(rows[source_index])
        row["relevance_score"] = score
        ranked.append(row)
    return ranked


__all__ = ["mixed_script_terms", "rank_composer_bm25"]
