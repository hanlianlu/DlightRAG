# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for dlightrag.ai.tokens."""

from dlightrag.ai.tokens import estimate_tokens


class TestEstimateTokens:
    """Token estimation across script families."""

    def test_ascii_only(self) -> None:
        # 11 ASCII chars / 4 ≈ 3 tokens
        assert estimate_tokens("hello world") == 3

    def test_cjk_chinese(self) -> None:
        # 4 CJK chars / 1.5 ≈ 3 tokens
        assert estimate_tokens("你好世界") == 3

    def test_japanese_kana(self) -> None:
        # 5 kana chars / 1.5 ≈ 4 tokens
        assert estimate_tokens("こんにちは") == 4

    def test_korean_hangul(self) -> None:
        # 5 hangul chars / 1.5 ≈ 4 tokens
        assert estimate_tokens("안녕하세요") == 4

    def test_latin_extended_german(self) -> None:
        # "Üntersuchung" — Ü is latin-ext (1 char / 3), rest is ASCII (11 chars / 4)
        result = estimate_tokens("Üntersuchung")
        assert result == 4  # ceil(11/4) + ceil(1/3) = 3 + 1

    def test_mixed_scripts(self) -> None:
        # "hello 你好 Grüße"
        # ASCII: "hello " + "e" = 7, Dense: "你好" = 2, LatinExt: "Grüß" = 4 (ü, ß are ext)
        result = estimate_tokens("hello 你好 Grüße")
        assert result > 0

    def test_empty_string(self) -> None:
        assert estimate_tokens("") == 0

    def test_french_accents(self) -> None:
        # "résumé" — r, s, u, m are ASCII (4), é×2 are latin-ext (2)
        result = estimate_tokens("résumé")
        assert result == 2  # ceil(4/4) + ceil(2/3) = 1 + 1

    def test_swedish(self) -> None:
        # "åäö" — all latin-ext
        result = estimate_tokens("åäö")
        assert result == 1  # ceil(3/3) = 1
