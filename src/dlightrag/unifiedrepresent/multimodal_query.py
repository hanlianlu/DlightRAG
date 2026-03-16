# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Multimodal query enhancement for unified representational RAG.

Uses VLM to describe uploaded images and combines descriptions with
the original text query for enhanced text-path retrieval.
"""

from __future__ import annotations

import base64
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


async def enhance_query_with_images(
    query: str,
    images: list[bytes],
    vision_model_func: Callable[..., Any],
    conversation_context: str | None = None,
) -> str:
    """Enhance a text query with VLM descriptions of uploaded images.

    Calls vision_model_func for each image to generate a text description,
    then combines all descriptions with the original query and optional
    conversation context into an enhanced query string for text-path retrieval.

    This mirrors RAGAnything's ``_process_multimodal_query_content()`` logic
    but is an independent implementation with no dependency on RAGAnything.

    Note: Conversation context truncation (min(last 5 turns, 50K tokens) per spec)
    is the caller's responsibility. This function receives pre-truncated context.
    The web route and API server apply truncation before calling this function.

    Args:
        query: Original text query from user.
        images: List of raw image bytes (max 3).
        vision_model_func: Async VLM callable accepting ``messages=`` (OpenAI chat format).
        conversation_context: Optional pre-truncated conversation history string.

    Returns:
        Enhanced query string combining original query, context, and image descriptions.
    """
    if not images:
        return query

    descriptions: list[str] = []
    for img_bytes in images:
        try:
            b64 = base64.b64encode(img_bytes).decode()
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                        {
                            "type": "text",
                            "text": "Describe this image in detail for document retrieval.",
                        },
                    ],
                },
            ]
            desc = await vision_model_func(messages=messages)
            descriptions.append(str(desc))
        except Exception:
            logger.warning("VLM image description failed", exc_info=True)
            descriptions.append("[Image description unavailable]")

    parts: list[str] = [query]
    if conversation_context:
        parts.append(f"Conversation context: {conversation_context}")
    for i, desc in enumerate(descriptions):
        parts.append(f"Image {i + 1} content: {desc}")

    return "\n\n".join(parts)
