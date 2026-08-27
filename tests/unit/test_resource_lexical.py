# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for resource-neutral lexical focus ranking."""

from __future__ import annotations

from dlightrag.engine.answer.resources.lexical import (
    bm25_rank,
    mixed_script_terms,
)


def test_mixed_script_terms_emit_words_and_cjk_bigrams() -> None:
    terms = mixed_script_terms("Termination LIABILITY 分数练习题")

    assert "termination" in terms
    assert "liability" in terms
    assert {"分", "数", "分数", "练习", "习题"} <= set(terms)


def test_mixed_script_terms_preserve_unicode_words_and_plane_three_cjk() -> None:
    terms = mixed_script_terms("ＡＢＣ Straße हिन्दी مَرْحَبًا \U00030000\U00030001")

    assert "abc" in terms  # NFKC
    assert "strasse" in terms  # casefold expansion
    assert "हिन्दी" in terms  # combining marks stay inside the word
    assert "مَرْحَبًا" in terms
    assert {"\U00030000", "\U00030001", "\U00030000\U00030001"} <= set(terms)


def test_bm25_rank_prioritizes_chinese_and_latin_documents() -> None:
    documents = [
        mixed_script_terms("分数挑战练习题：分子相加，分母保持不变"),
        mixed_script_terms("货币、权力与国际政治经济学"),
        mixed_script_terms("Termination liability and damages cap"),
    ]

    chinese = bm25_rank(mixed_script_terms("分数练习题"), documents, limit=3)
    latin = bm25_rank(mixed_script_terms("termination liability"), documents, limit=3)

    assert chinese[0][0] == 0
    assert latin[0][0] == 2


def test_bm25_rank_ignores_unknown_terms_and_empty_documents() -> None:
    documents = [[], mixed_script_terms("fraction worksheet"), mixed_script_terms("contract")]

    ranked = bm25_rank(
        mixed_script_terms("fraction totallyunknownterm"),
        documents,
        limit=3,
    )

    assert [index for index, _ in ranked] == [1]
