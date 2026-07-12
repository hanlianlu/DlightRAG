# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""VLM-assisted query image semantic enhancement."""

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
    descriptions: dict[str, str] = field(default_factory=dict)


@dataclass
class PreparedQueryImages:
    query: str
    answer_images: list[dict[str, Any]]
    multimodal_content: list[dict[str, Any]]
    descriptions: list[str]
    descriptions_by_ordinal: dict[str, str]


def images_to_multimodal_content(images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return direct visual retrieval content blocks for inline data images."""
    items: list[dict[str, Any]] = []
    for image in images:
        block = image_url_block(image)
        if block is None:
            continue
        url = block.get("image_url", {}).get("url")
        if isinstance(url, str) and url.strip().startswith("data:"):
            items.append(block)
    return items


class QueryImageEnhancer:
    """Describe user query images for semantic/BM25/KG retrieval."""

    def __init__(
        self,
        *,
        vlm_func: Callable[..., Any] | None,
        max_images: int = 3,
    ) -> None:
        self._vlm_func = vlm_func
        self._max_images = max(0, int(max_images))

    async def aclose(self) -> None:
        """Release VLM worker resources owned by this enhancer."""
        from dlightrag.utils.concurrency import shutdown_async_callable

        await shutdown_async_callable(self._vlm_func)

    async def enhance(
        self,
        query: str,
        images: list[dict[str, Any]] | None,
    ) -> QueryImageEnhancement:
        """Append concise visual descriptions to the retrieval query."""
        if self._vlm_func is None or not images or self._max_images <= 0:
            return QueryImageEnhancement(query=query)
        vlm_func = self._vlm_func

        async def _describe(item: tuple[int, dict[str, Any]]) -> tuple[str, str] | None:
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
                return str(idx), f"Image {idx}: {response.strip()}"
            return None

        items = list(enumerate(images[: self._max_images], start=1))
        results = await bounded_map(
            items,
            _describe,
            max_concurrent=max(1, min(self._max_images, len(items))),
            task_name="query-image-description",
        )
        descriptions: dict[str, str] = {}
        for item in results:
            if isinstance(item, tuple):
                key, text = item
                descriptions[key] = text

        if not descriptions:
            return QueryImageEnhancement(query=query)
        visual_context = "\n".join(descriptions.values())
        return QueryImageEnhancement(
            query=f"{query}\n\nVisual context from user-supplied images:\n{visual_context}",
            descriptions=descriptions,
        )


async def prepare_query_images(
    query: str,
    *,
    query_images: list[dict[str, Any]] | None,
    enhancer: Any,
) -> PreparedQueryImages:
    """Create semantic and direct-retrieval inputs from current request images."""
    current_images = list(query_images or [])
    enhanced = await enhancer.enhance(query, current_images)
    return PreparedQueryImages(
        query=enhanced.query,
        answer_images=current_images,
        multimodal_content=images_to_multimodal_content(current_images),
        descriptions=list(enhanced.descriptions.values()),
        descriptions_by_ordinal=enhanced.descriptions,
    )


__all__ = [
    "PreparedQueryImages",
    "QueryImageEnhancement",
    "QueryImageEnhancer",
    "images_to_multimodal_content",
    "prepare_query_images",
]
