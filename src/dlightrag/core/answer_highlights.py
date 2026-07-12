# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared semantic highlight enrichment for answer sources."""

import asyncio
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

from dlightrag.citations.highlight import extract_highlights_for_sources
from dlightrag.citations.schemas import HighlightSource

if TYPE_CHECKING:
    from dlightrag.config import DlightragConfig

logger = logging.getLogger(__name__)


async def enrich_semantic_highlights[SourceT: HighlightSource](
    sources: list[SourceT],
    *,
    answer_text: str | None,
    config: DlightragConfig,
) -> list[SourceT]:
    """Return sources with optional semantic highlight phrases."""
    highlight_cfg = config.citations.highlights
    text_chunk_count = _text_chunk_count(sources)
    if not highlight_cfg.enabled or not answer_text or text_chunk_count == 0:
        return sources

    try:
        from dlightrag.models.llm import get_keyword_model_func
        from dlightrag.observability import trace_observation

        llm_func = get_keyword_model_func(config)
        async with trace_observation(
            "semantic_highlights",
            as_type="chain",
            metadata={"source_count": len(sources), "text_chunk_count": text_chunk_count},
        ) as trace:
            highlighted = await asyncio.wait_for(
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
            trace.update(output=_highlight_output(highlighted))
            return highlighted
    except TimeoutError:
        logger.warning(
            "Semantic highlight extraction timed out (%.1fs), skipping",
            highlight_cfg.timeout,
        )
    except Exception:
        logger.warning("Semantic highlight extraction failed", exc_info=True)
    return sources


def _text_chunk_count(sources: Sequence[HighlightSource]) -> int:
    return sum(1 for source in sources if source.chunks for chunk in source.chunks if chunk.content)


def _highlight_output(sources: Sequence[HighlightSource]) -> dict[str, int]:
    highlighted_source_count = 0
    highlighted_chunk_count = 0
    for source in sources:
        source_has_highlight = False
        for chunk in source.chunks or []:
            if chunk.highlight_phrases:
                source_has_highlight = True
                highlighted_chunk_count += 1
        if source_has_highlight:
            highlighted_source_count += 1
    return {
        "highlighted_source_count": highlighted_source_count,
        "highlighted_chunk_count": highlighted_chunk_count,
    }


__all__ = ["enrich_semantic_highlights"]
