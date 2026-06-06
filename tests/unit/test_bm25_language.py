# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for BM25 language profile classification."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dlightrag.core.retrieval import bm25_language
from dlightrag.core.retrieval.bm25_language import BM25LanguageClassifier


def test_bm25_language_classifier_routes_plain_chinese() -> None:
    classifier = BM25LanguageClassifier(("zh", "en"))

    assert classifier.detect("现金流 风险 因素") == "zh"


def test_bm25_language_classifier_routes_plain_english() -> None:
    classifier = BM25LanguageClassifier(("zh", "en"))

    assert classifier.detect("risk factors revenue guidance") == "en"


def test_bm25_language_classifier_lets_lingua_handle_chinese(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class FakeDetector:
        def detect_language_of(self, query: str) -> SimpleNamespace:
            calls.append(query)
            return SimpleNamespace(name="CHINESE")

    monkeypatch.setattr(
        bm25_language,
        "_lingua_detector",
        lambda _language_codes: (FakeDetector(), {"CHINESE": "zh"}),
    )

    classifier = BM25LanguageClassifier(("zh", "en"))

    assert classifier.detect("现金流 风险 因素") == "zh"
    assert calls == ["现金流 风险 因素"]


def test_bm25_language_classifier_uses_configured_lingua_language_set() -> None:
    classifier = BM25LanguageClassifier(("de", "en"))

    assert classifier.detect("Wie hoch ist der Umsatz im letzten Quartal?") == "de"


def test_bm25_language_classifier_routes_ambiguous_query_to_simple() -> None:
    classifier = BM25LanguageClassifier(("fr", "de", "en"))

    assert classifier.detect("bonjour revenue Umsatz") == "simple"


def test_bm25_language_classifier_uses_cjk_fallback_only_after_lingua_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDetector:
        def detect_language_of(self, _query: str) -> None:
            return None

    monkeypatch.setattr(
        bm25_language,
        "_lingua_detector",
        lambda _language_codes: (FakeDetector(), {"CHINESE": "zh"}),
    )
    classifier = BM25LanguageClassifier(("zh", "en"))

    assert classifier.detect("现金流 风险 因素") == "zh"


def test_bm25_language_classifier_disables_cjk_fallback_when_ja_or_ko_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDetector:
        def detect_language_of(self, _query: str) -> None:
            return None

    monkeypatch.setattr(
        bm25_language,
        "_lingua_detector",
        lambda _language_codes: (FakeDetector(), {}),
    )
    classifier = BM25LanguageClassifier(("zh", "ja", "en"))

    assert classifier.detect("现金流 风险 因素") == "simple"
