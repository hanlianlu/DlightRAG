# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Language classification for BM25 profile routing and chunk labeling."""

import re
from functools import lru_cache
from typing import Any

BM25_LANGUAGE_COLUMN = "dlightrag_bm25_language"
BM25_FALLBACK_LANGUAGE = "simple"
_LINGUA_MIN_RELATIVE_DISTANCE = 0.08
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")


def normalize_language_code(language: str) -> str:
    value = str(language).strip().lower().replace("_", "-")
    if not value:
        return ""
    if value.startswith("zh") or value in {"cn", "chinese"}:
        return "zh"
    return value.split("-", maxsplit=1)[0]


def language_codes_for_detection(supported_languages: tuple[str, ...]) -> tuple[str, ...]:
    language_codes: list[str] = []
    seen: set[str] = set()
    for language in supported_languages:
        code = normalize_language_code(language)
        if not code or code in seen or code == BM25_FALLBACK_LANGUAGE:
            continue
        seen.add(code)
        language_codes.append(code)
    return tuple(language_codes)


def _lingua_language_for_code(code: str) -> Any | None:
    try:
        from lingua import IsoCode639_1, Language
    except Exception:
        return None
    try:
        iso_code = IsoCode639_1.from_str(code.upper())
        return Language.from_iso_code_639_1(iso_code)
    except Exception:
        try:
            return Language.from_str(code)
        except Exception:
            return None


@lru_cache(maxsize=32)
def _lingua_detector(language_codes: tuple[str, ...]) -> tuple[Any | None, dict[str, str]]:
    try:
        from lingua import LanguageDetectorBuilder
    except Exception:
        return None, {}

    lingua_languages: list[Any] = []
    code_by_lingua_name: dict[str, str] = {}
    for code in language_codes:
        lingua_language = _lingua_language_for_code(code)
        if lingua_language is None:
            continue
        lingua_name = lingua_language.name
        if lingua_name in code_by_lingua_name:
            continue
        code_by_lingua_name[lingua_name] = code
        lingua_languages.append(lingua_language)

    if not lingua_languages:
        return None, {}
    detector = (
        LanguageDetectorBuilder.from_languages(*lingua_languages)
        .with_minimum_relative_distance(_LINGUA_MIN_RELATIVE_DISTANCE)
        .build()
    )
    return detector, code_by_lingua_name


class BM25LanguageClassifier:
    """Shared classifier for query routing and ingest-time chunk labeling."""

    def __init__(self, supported_languages: tuple[str, ...]) -> None:
        self._language_codes = language_codes_for_detection(supported_languages)
        self._language_code_set = set(self._language_codes)

    @property
    def supported_languages(self) -> tuple[str, ...]:
        return self._language_codes

    def detect(self, text: str) -> str:
        value = str(text or "").strip()
        if not value:
            return BM25_FALLBACK_LANGUAGE

        detector, code_by_lingua_name = _lingua_detector(self._language_codes)
        if detector is not None:
            try:
                detected = detector.detect_language_of(value)
            except Exception:
                detected = None
            if detected:
                code = code_by_lingua_name.get(detected.name)
                if code:
                    return code

        if (
            _CJK_RE.search(value)
            and "zh" in self._language_code_set
            and self._language_code_set.isdisjoint({"ja", "ko"})
        ):
            return "zh"
        return BM25_FALLBACK_LANGUAGE
