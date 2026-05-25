# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""VLM-assisted query image semantic enhancement."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from dlightrag.utils.images import image_url_block

logger = logging.getLogger(__name__)


@dataclass
class QueryImageEnhancement:
    """Result of query-image semantic enhancement."""

    query: str
    descriptions: list[str] = field(default_factory=list)


class QueryImageEnhancer:
    """Describe user query images for semantic/BM25/KG retrieval."""

    def __init__(
        self,
        *,
        vlm_func: Callable[..., Any] | None,
        enabled: bool = True,
        max_images: int = 3,
    ) -> None:
        self._vlm_func = vlm_func
        self._enabled = enabled
        self._max_images = max(0, int(max_images))

    async def enhance(
        self,
        query: str,
        images: list[str | dict[str, Any]] | None,
    ) -> QueryImageEnhancement:
        """Append concise visual descriptions to the retrieval query."""
        if not self._enabled or self._vlm_func is None or not images or self._max_images <= 0:
            return QueryImageEnhancement(query=query)

        descriptions: list[str] = []
        for idx, image in enumerate(images[: self._max_images], start=1):
            block = image_url_block(image)
            if block is None:
                continue
            try:
                response = await self._vlm_func(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                block,
                                {
                                    "type": "text",
                                    "text": (
                                        "Describe this query image for document retrieval. "
                                        "Be concise, concrete, and avoid speculation. "
                                        "Mention visible text, objects, layout, chart/table cues, "
                                        "and any domain-specific identifiers if present."
                                    ),
                                },
                            ],
                        }
                    ]
                )
            except Exception:
                logger.warning("Query image description failed", exc_info=True)
                continue
            if isinstance(response, str) and response.strip():
                descriptions.append(f"Image {idx}: {response.strip()}")

        if not descriptions:
            return QueryImageEnhancement(query=query)
        visual_context = "\n".join(descriptions)
        return QueryImageEnhancement(
            query=f"{query}\n\nVisual context from user-supplied images:\n{visual_context}",
            descriptions=descriptions,
        )


__all__ = ["QueryImageEnhancement", "QueryImageEnhancer"]
