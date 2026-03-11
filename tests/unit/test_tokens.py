# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for dlightrag.utils.tokens."""

from __future__ import annotations

from dlightrag.utils.tokens import estimate_tokens, truncate_conversation_history


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


class TestTruncateConversationHistory:
    """Conversation history truncation."""

    def test_empty_history(self) -> None:
        assert truncate_conversation_history([], max_messages=10, max_tokens=1000) == []

    def test_within_limits(self) -> None:
        history = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = truncate_conversation_history(history, max_messages=10, max_tokens=1000)
        assert result == history

    def test_message_limit(self) -> None:
        history = [{"role": "user", "content": f"msg {i}"} for i in range(10)]
        result = truncate_conversation_history(history, max_messages=3, max_tokens=999999)
        assert len(result) == 3
        assert result[0]["content"] == "msg 7"

    def test_token_limit(self) -> None:
        # Each "x" * 400 ≈ 100 tokens (400 ASCII / 4)
        history = [{"role": "user", "content": "x" * 400} for _ in range(5)]
        # Budget 250 tokens → should keep last 2 messages (200 tokens)
        result = truncate_conversation_history(history, max_messages=50, max_tokens=250)
        assert len(result) == 2

    def test_token_limit_cjk(self) -> None:
        # Each "你" * 150 ≈ 100 tokens (150 CJK / 1.5)
        history = [{"role": "user", "content": "你" * 150} for _ in range(5)]
        # Budget 250 tokens → should keep last 2 messages
        result = truncate_conversation_history(history, max_messages=50, max_tokens=250)
        assert len(result) == 2

    def test_both_limits_message_tighter(self) -> None:
        history = [{"role": "user", "content": "short"} for _ in range(10)]
        result = truncate_conversation_history(history, max_messages=2, max_tokens=999999)
        assert len(result) == 2

    def test_both_limits_token_tighter(self) -> None:
        history = [{"role": "user", "content": "x" * 400} for _ in range(10)]
        # 10 messages ok, but only ~250 tokens budget → keeps 2
        result = truncate_conversation_history(history, max_messages=10, max_tokens=250)
        assert len(result) == 2
