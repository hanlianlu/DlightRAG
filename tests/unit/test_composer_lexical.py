# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for Composer-local in-memory BM25."""

from dlightrag.core.request.composer_lexical import mixed_script_terms, rank_composer_bm25


def _row(chunk_id: str, content: str) -> dict[str, object]:
    return {"chunk_id": chunk_id, "content": content}


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


def test_bm25_ranks_chinese_and_latin_queries() -> None:
    rows = [
        _row("fraction", "分数挑战练习题：分子相加，分母保持不变"),
        _row("politics", "货币、权力与国际政治经济学"),
        _row("contract", "Termination liability and damages cap"),
    ]

    chinese = rank_composer_bm25("分数练习题", rows, limit=3)
    latin = rank_composer_bm25("termination liability", rows, limit=3)

    assert chinese[0]["chunk_id"] == "fraction"
    assert latin[0]["chunk_id"] == "contract"
    assert all(float(row["relevance_score"]) > 0 for row in [*chinese, *latin])


def test_bm25_drops_all_zero_results() -> None:
    rows = [_row("a", "fraction worksheet"), _row("b", "contract liability")]

    ranked = rank_composer_bm25("这说的啥", rows, limit=2)

    assert ranked == []


def test_bm25_oov_query_does_not_rank_empty_visual_chunk() -> None:
    rows = [_row("visual", ""), _row("text", "fraction worksheet")]

    ranked = rank_composer_bm25("unknown-out-of-vocabulary", rows, limit=2)

    assert ranked == []


def test_bm25_all_empty_corpus_returns_no_results() -> None:
    rows = [_row("visual-1", ""), _row("visual-2", "   ")]

    ranked = rank_composer_bm25("anything", rows, limit=2)

    assert ranked == []


def test_bm25_mixed_known_and_oov_query_uses_only_known_terms() -> None:
    rows = [_row("fraction", "fraction worksheet"), _row("contract", "contract clause")]

    ranked = rank_composer_bm25("fraction totallyunknownterm", rows, limit=2)

    assert [row["chunk_id"] for row in ranked] == ["fraction"]
