# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""VLM-assisted query-image description for retrieval planning."""

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from dlightrag.ai.concurrency import bounded_map
from dlightrag.ai.media import image_url_block
from dlightrag.answer.images import AnswerImagePolicy

logger = logging.getLogger(__name__)
MAX_QUERY_IMAGES = 3


class QueryImageDescriber:
    """Describe user query images for semantic/BM25/KG retrieval planning."""

    def __init__(
        self,
        *,
        vlm_func: Callable[..., Any] | None,
        image_policy: AnswerImagePolicy,
        max_images: int = MAX_QUERY_IMAGES,
    ) -> None:
        self._vlm_func = vlm_func
        self._image_policy = image_policy
        self._max_images = max(0, int(max_images))

    async def describe(
        self,
        images: list[dict[str, Any]] | None,
    ) -> list[str]:
        """Return concise per-image visual descriptions in request order."""
        if self._vlm_func is None or not images or self._max_images <= 0:
            return []
        vlm_func = self._vlm_func

        async def _describe(item: tuple[int, dict[str, Any]]) -> tuple[str, str] | None:
            idx, image = item
            block = image_url_block(image)
            if block is None:
                return None
            budget = self._image_policy.new_budget()
            bounded_block = await asyncio.to_thread(
                budget.add_user_image,
                block,
                label=f"query_image_{idx}",
            )
            if bounded_block is None:
                return None
            try:
                response = await vlm_func(
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                bounded_block,
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
        return [item[1] for item in results if isinstance(item, tuple)]


async def prepare_query_images(
    *,
    query_images: list[dict[str, Any]] | None,
    describer: Any,
) -> list[str]:
    """Describe current-request images for image-aware retrieval planning."""
    return list(await describer.describe(list(query_images or [])))


__all__ = [
    "MAX_QUERY_IMAGES",
    "QueryImageDescriber",
    "prepare_query_images",
]
