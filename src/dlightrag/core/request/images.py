# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""VLM-assisted query-image description for retrieval planning."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from dlightrag.core.answer.images import AnswerImageBudget
from dlightrag.utils.concurrency import bounded_map
from dlightrag.utils.images import image_url_block

logger = logging.getLogger(__name__)


@dataclass
class PreparedQueryImages:
    """Current-request image inputs described for retrieval planning."""

    descriptions: list[str]
    descriptions_by_ordinal: dict[str, str]


class QueryImageDescriber:
    """Describe user query images for semantic/BM25/KG retrieval planning."""

    def __init__(
        self,
        *,
        vlm_func: Callable[..., Any] | None,
        max_images: int = 3,
        max_total_bytes: int,
        max_bytes_per_image: int,
        max_px: int,
        min_px: int,
        quality: int,
        min_quality: int,
    ) -> None:
        self._vlm_func = vlm_func
        self._max_images = max(0, int(max_images))
        self._max_total_bytes = max(1, int(max_total_bytes))
        self._max_bytes_per_image = max(1, int(max_bytes_per_image))
        self._max_px = max(1, int(max_px))
        self._min_px = max(1, int(min_px))
        self._quality = max(1, int(quality))
        self._min_quality = max(1, int(min_quality))

    async def aclose(self) -> None:
        """Release VLM worker resources owned by this describer."""
        from dlightrag.utils.concurrency import shutdown_async_callable

        await shutdown_async_callable(self._vlm_func)

    async def describe(
        self,
        images: list[dict[str, Any]] | None,
    ) -> dict[str, str]:
        """Return concise per-image visual descriptions keyed by 1-based ordinal."""
        if self._vlm_func is None or not images or self._max_images <= 0:
            return {}
        vlm_func = self._vlm_func

        async def _describe(item: tuple[int, dict[str, Any]]) -> tuple[str, str] | None:
            idx, image = item
            block = image_url_block(image)
            if block is None:
                return None
            budget = AnswerImageBudget(
                max_images=1,
                max_total_bytes=self._max_total_bytes,
                max_bytes_per_image=self._max_bytes_per_image,
                max_px=self._max_px,
                min_px=self._min_px,
                quality=self._quality,
                min_quality=self._min_quality,
            )
            bounded_block = budget.add_user_image(block, label=f"query_image_{idx}")
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
        descriptions: dict[str, str] = {}
        for item in results:
            if isinstance(item, tuple):
                key, text = item
                descriptions[key] = text
        return descriptions


async def prepare_query_images(
    *,
    query_images: list[dict[str, Any]] | None,
    describer: Any,
) -> PreparedQueryImages:
    """Describe current-request images for image-aware retrieval planning."""
    descriptions = await describer.describe(list(query_images or []))
    return PreparedQueryImages(
        descriptions=list(descriptions.values()),
        descriptions_by_ordinal=descriptions,
    )


__all__ = [
    "PreparedQueryImages",
    "QueryImageDescriber",
    "prepare_query_images",
]
