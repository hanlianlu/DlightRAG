# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for BM25 language detection."""

from __future__ import annotations

from dlightrag.core.retrieval.language_detect import detect_bm25_config


class TestDetectBm25Config:
    async def test_english_detection(self) -> None:
        chunks = [{"content": "The quick brown fox jumps over the lazy dog. " * 3}] * 5
        config = await detect_bm25_config(chunks)
        assert config == "english"

    async def test_chinese_detection(self) -> None:
        chunks = [{"content": "自然语言处理是人工智能的一个重要分支领域。" * 3}] * 5
        config = await detect_bm25_config(chunks)
        assert config == "jiebacfg"

    async def test_french_detection(self) -> None:
        chunks = [
            {"content": "Le traitement du langage naturel est une branche importante. " * 3}
        ] * 5
        config = await detect_bm25_config(chunks)
        assert config == "french"

    async def test_empty_chunks_fallback(self) -> None:
        config = await detect_bm25_config([])
        assert config == "simple"

    async def test_no_text_content_fallback(self) -> None:
        chunks = [{"other_field": "value"}] * 5
        config = await detect_bm25_config(chunks)
        assert config == "simple"
