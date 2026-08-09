# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for resource-neutral lexical focus ranking."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from dlightrag.core.resources.lexical import (
    mixed_script_terms,
    rank_resource_windows,
)


@dataclass(frozen=True)
class _Window:
    name: str
    text: str


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


def test_rank_resource_windows_ranks_chinese_and_latin_queries() -> None:
    windows = [
        _Window("fraction", "分数挑战练习题：分子相加，分母保持不变"),
        _Window("politics", "货币、权力与国际政治经济学"),
        _Window("contract", "Termination liability and damages cap"),
    ]

    chinese = rank_resource_windows("分数练习题", windows, limit=3)
    latin = rank_resource_windows("termination liability", windows, limit=3)

    assert chinese[0].name == "fraction"
    assert latin[0].name == "contract"


def test_rank_resource_windows_returns_original_objects() -> None:
    windows = [_Window("a", "fraction worksheet"), _Window("b", "contract clause")]

    ranked = rank_resource_windows("fraction", windows, limit=2)

    assert ranked[0] is windows[0]


@pytest.mark.parametrize(
    ("query", "windows"),
    [
        pytest.param(
            "这说的啥",
            [_Window("a", "fraction worksheet"), _Window("b", "contract liability")],
            id="all_zero_results",
        ),
        pytest.param(
            "unknown-out-of-vocabulary",
            [_Window("visual", ""), _Window("text", "fraction worksheet")],
            id="oov_query_skips_empty_window",
        ),
        pytest.param(
            "anything",
            [_Window("visual-1", ""), _Window("visual-2", "   ")],
            id="all_empty_corpus_returns_no_results",
        ),
    ],
)
def test_rank_resource_windows_returns_no_results(query: str, windows: list[_Window]) -> None:
    assert rank_resource_windows(query, windows, limit=2) == []


def test_rank_resource_windows_mixed_known_and_oov_uses_only_known_terms() -> None:
    windows = [_Window("fraction", "fraction worksheet"), _Window("contract", "contract clause")]

    ranked = rank_resource_windows("fraction totallyunknownterm", windows, limit=2)

    assert [window.name for window in ranked] == ["fraction"]


def test_rank_resource_windows_zero_limit_returns_empty() -> None:
    windows = [_Window("a", "fraction worksheet")]

    assert rank_resource_windows("fraction", windows, limit=0) == []
