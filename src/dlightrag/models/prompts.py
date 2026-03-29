# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Custom VLM prompts for RAGAnything ingestion.

Provides smart content type detection (table vs image) with differentiated
analysis strategies. Injects custom prompts into RAGAnything's global PROMPTS dict.

Key design decisions:
- VLM first classifies content type (table vs image) - don't trust parser labels
- For tables: Output Markdown format + key data summary in detailed_description
- For images: Standard visual description in detailed_description
- All content goes in detailed_description (only field extracted by RAGAnything)
"""

from __future__ import annotations

import logging

from dlightrag.prompts import (
    SMART_IMAGE_ANALYSIS_PROMPT,
    SMART_IMAGE_ANALYSIS_PROMPT_WITH_CONTEXT,
    SMART_IMAGE_ANALYSIS_SYSTEM,
)

logger = logging.getLogger(__name__)


def inject_custom_prompts() -> None:
    """Inject custom prompts into RAGAnything's PROMPTS dict.

    Must be called BEFORE RAGAnything is instantiated, as processors
    read from PROMPTS during initialization.
    """
    from raganything.prompt import PROMPTS

    PROMPTS["IMAGE_ANALYSIS_SYSTEM"] = SMART_IMAGE_ANALYSIS_SYSTEM
    PROMPTS["vision_prompt"] = SMART_IMAGE_ANALYSIS_PROMPT
    PROMPTS["vision_prompt_with_context"] = SMART_IMAGE_ANALYSIS_PROMPT_WITH_CONTEXT

    logger.info("Injected custom VLM prompts into RAGAnything")


__all__ = [
    "inject_custom_prompts",
]
