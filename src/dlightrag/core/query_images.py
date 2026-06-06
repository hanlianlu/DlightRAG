# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""VLM-assisted query image semantic enhancement."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from dlightrag.utils.concurrency import bounded_map
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

    async def aclose(self) -> None:
        """Release VLM worker resources owned by this enhancer."""
        from dlightrag.utils.concurrency import shutdown_async_callable

        await shutdown_async_callable(self._vlm_func)

    async def enhance(
        self,
        query: str,
        images: list[str | dict[str, Any]] | None,
    ) -> QueryImageEnhancement:
        """Append concise visual descriptions to the retrieval query."""
        if not self._enabled or self._vlm_func is None or not images or self._max_images <= 0:
            return QueryImageEnhancement(query=query)
        vlm_func = self._vlm_func

        async def _describe(item: tuple[int, str | dict[str, Any]]) -> str | None:
            idx, image = item
            block = image_url_block(image)
            if block is None:
                return None
            try:
                response = await vlm_func(
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
                return None
            if isinstance(response, str) and response.strip():
                return f"Image {idx}: {response.strip()}"
            return None

        items = list(enumerate(images[: self._max_images], start=1))
        results = await bounded_map(
            items,
            _describe,
            max_concurrent=max(1, min(self._max_images, len(items))),
            task_name="query-image-description",
        )
        descriptions = [item for item in results if isinstance(item, str) and item.strip()]

        if not descriptions:
            return QueryImageEnhancement(query=query)
        visual_context = "\n".join(descriptions)
        return QueryImageEnhancement(
            query=f"{query}\n\nVisual context from user-supplied images:\n{visual_context}",
            descriptions=descriptions,
        )


__all__ = ["QueryImageEnhancement", "QueryImageEnhancer"]
