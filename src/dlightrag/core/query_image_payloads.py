# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Query image session and payload preparation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dlightrag.core.scope import RequestScope
from dlightrag.utils.images import image_url_block


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
    "image_blocks_from_strings",
    "images_to_multimodal_content",
    "prepare_query_images",
    "storable_image_strings",
]
