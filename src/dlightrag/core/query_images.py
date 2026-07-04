# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""VLM-assisted query image semantic enhancement."""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from dlightrag.core.scope import RequestScope
from dlightrag.utils.concurrency import bounded_map
from dlightrag.utils.images import image_url_block

logger = logging.getLogger(__name__)


@dataclass
class QueryImageEnhancement:
    """Result of query-image semantic enhancement."""

    query: str
    descriptions: list[str] = field(default_factory=list)


@dataclass
class PreparedQueryImages:
    query: str
    answer_images: list[dict[str, Any]]
    multimodal_content: list[dict[str, Any]]
    descriptions: list[str]
    current_image_ids: list[str]


def image_blocks_from_strings(images: list[str]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for image in images:
        block = image_url_block(image)
        if block is not None:
            blocks.append(block)
    return blocks


def storable_image_strings(images: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for image in images:
        block = image_url_block(image)
        if block is None:
            continue
        url = block.get("image_url", {}).get("url")
        if isinstance(url, str) and url.strip().startswith("data:"):
            values.append(url)
    return values


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

        async def _describe(item: tuple[int, dict[str, Any]]) -> str | None:
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


async def prepare_query_images(
    query: str,
    *,
    query_images: list[dict[str, Any]] | None,
    session_id: str | None,
    referenced_image_ids: list[str] | None,
    store_current: bool,
    session_images: Any,
    enhancer: Any,
    scope: RequestScope | None = None,
) -> PreparedQueryImages:
    """Resolve session images and create semantic/direct image query inputs."""
    scoped_session_id = scope.session_key(session_id) if scope is not None else session_id
    current_images = list(query_images or [])
    current_storable = storable_image_strings(current_images)
    current_ids = (
        session_images.store(scoped_session_id, current_storable)
        if store_current and current_storable
        else []
    )
    historical = image_blocks_from_strings(
        session_images.get(scoped_session_id, referenced_image_ids)
    )
    answer_images = [*historical, *current_images]

    enhanced = await enhancer.enhance(query, answer_images)
    return PreparedQueryImages(
        query=enhanced.query,
        answer_images=answer_images,
        multimodal_content=images_to_multimodal_content(answer_images),
        descriptions=enhanced.descriptions,
        current_image_ids=current_ids,
    )


__all__ = [
    "PreparedQueryImages",
    "QueryImageEnhancement",
    "QueryImageEnhancer",
    "image_blocks_from_strings",
    "images_to_multimodal_content",
    "prepare_query_images",
    "storable_image_strings",
]
