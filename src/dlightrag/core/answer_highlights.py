# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared semantic highlight enrichment for answer sources."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from dlightrag.citations.highlight import extract_highlights_for_sources
from dlightrag.citations.schemas import SourceReference

if TYPE_CHECKING:
    from dlightrag.config import DlightragConfig

logger = logging.getLogger(__name__)


async def enrich_semantic_highlights(
    sources: list[SourceReference],
    *,
    answer_text: str | None,
    config: DlightragConfig,
) -> list[SourceReference]:
    """Return sources with optional semantic highlight phrases."""
    highlight_cfg = config.citations.highlights
    if not highlight_cfg.enabled or not answer_text or not _has_text_chunks(sources):
        return sources

    try:
        from dlightrag.models.llm import get_keyword_model_func

        llm_func = get_keyword_model_func(config)
        return await asyncio.wait_for(
            extract_highlights_for_sources(
                sources=sources,
                answer_text=answer_text,
                llm_func=llm_func,
                max_concurrency=highlight_cfg.max_concurrency,
                batch_size=highlight_cfg.batch_size,
                max_input_chars=highlight_cfg.max_input_chars,
                cache_size=highlight_cfg.cache_size,
            ),
            timeout=highlight_cfg.timeout,
        )
    except TimeoutError:
        logger.warning(
            "Semantic highlight extraction timed out (%.1fs), skipping",
            highlight_cfg.timeout,
        )
    except Exception:
        logger.warning("Semantic highlight extraction failed", exc_info=True)
    return sources


def _has_text_chunks(sources: list[SourceReference]) -> bool:
    return any(chunk.content for source in sources if source.chunks for chunk in source.chunks)


__all__ = ["enrich_semantic_highlights"]
