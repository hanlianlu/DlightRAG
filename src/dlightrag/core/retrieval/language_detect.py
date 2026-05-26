# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Lightweight BM25 language detection via langdetect.

Maps detected language to PostgreSQL text search configurations for
language-appropriate tokenization and stemming in BM25 indexes.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_LANG_TO_PG_CONFIG: dict[str, str] = {
    "zh-cn": "jiebacfg",
    "zh-tw": "jiebacfg",
    "en": "english",
    "fr": "french",
    "de": "german",
    "es": "spanish",
    "it": "italian",
    "pt": "portuguese",
    "nl": "dutch",
    "ru": "russian",
    "sv": "swedish",
    "no": "norwegian",
    "da": "danish",
    "fi": "finnish",
    "hu": "hungarian",
    "tr": "turkish",
    "ro": "romanian",
}
_FALLBACK_CONFIG = "simple"
_SAMPLE_SIZE = 10
_MAX_CHAR_PER_CHUNK = 300


async def detect_bm25_config(
    chunks: list[dict[str, Any]],
    *,
    sample_size: int = _SAMPLE_SIZE,
) -> str:
    """Detect workspace language from sample chunks, return PG text config name.

    Falls back to ``"simple"`` if detection fails or the detected language
    has no PG text config mapping. Callers should verify the returned config
    exists on the PG instance before creating a BM25 index.
    """
    if not chunks:
        logger.info("BM25 auto-detect: no chunks available, using '%s'", _FALLBACK_CONFIG)
        return _FALLBACK_CONFIG

    texts: list[str] = []
    for chunk in chunks[:sample_size]:
        content = chunk.get("content", "")
        if content and isinstance(content, str):
            texts.append(content[:_MAX_CHAR_PER_CHUNK])

    if not texts:
        logger.info("BM25 auto-detect: no text content in chunks, using '%s'", _FALLBACK_CONFIG)
        return _FALLBACK_CONFIG

    combined = "\n".join(texts)
    try:
        import langdetect

        lang = langdetect.detect(combined)
    except Exception:
        logger.warning("BM25 auto-detect: langdetect failed, using '%s'", _FALLBACK_CONFIG)
        return _FALLBACK_CONFIG

    if not lang:
        return _FALLBACK_CONFIG

    lang_lower = lang.lower()
    pg_config = _LANG_TO_PG_CONFIG.get(lang_lower)
    if pg_config is None:
        logger.info(
            "BM25 auto-detect: no PG config for language '%s', using '%s'",
            lang_lower,
            _FALLBACK_CONFIG,
        )
        return _FALLBACK_CONFIG

    logger.info("BM25 auto-detect: language='%s' → pg_config='%s'", lang_lower, pg_config)
    return pg_config


async def verify_pg_config(pool: Any, config_name: str) -> str:
    """Verify a PG text config exists. Falls back to 'simple' if missing."""
    if config_name == _FALLBACK_CONFIG:
        return config_name
    try:
        async with pool.acquire() as conn:
            exists = await conn.fetchval(
                "SELECT 1 FROM pg_ts_config WHERE cfgname = $1", config_name
            )
        if exists:
            return config_name
        logger.warning(
            "PG text config '%s' not found, falling back to '%s'", config_name, _FALLBACK_CONFIG
        )
        return _FALLBACK_CONFIG
    except Exception:
        logger.warning(
            "Cannot verify PG config '%s', falling back to '%s'", config_name, _FALLBACK_CONFIG
        )
        return _FALLBACK_CONFIG


__all__ = ["detect_bm25_config", "verify_pg_config"]
