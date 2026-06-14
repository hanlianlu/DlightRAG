# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Transport-neutral client payload contracts."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ClientContractModel(BaseModel):
    """Base model for public client contracts."""

    model_config = ConfigDict(extra="forbid")


class ImageURL(ClientContractModel):
    url: str


class TextContentBlock(ClientContractModel):
    type: Literal["text"]
    text: str


class ImageURLContentBlock(ClientContractModel):
    type: Literal["image_url"]
    image_url: ImageURL


type ContentBlock = Annotated[
    TextContentBlock | ImageURLContentBlock,
    Field(discriminator="type"),
]
type QueryImage = ImageURLContentBlock


class ConversationMessage(ClientContractModel):
    role: Literal["system", "user", "assistant"]
    content: str | list[ContentBlock]


def model_dump_json_safe(value: Any) -> Any:
    """Return plain JSON-ready data from Pydantic models and containers."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [model_dump_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [model_dump_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): model_dump_json_safe(item) for key, item in value.items()}
    return value


def dump_optional_list(value: list[Any] | None) -> list[Any] | None:
    if value is None:
        return None
    return model_dump_json_safe(value)


__all__ = [
    "ClientContractModel",
    "ContentBlock",
    "ConversationMessage",
    "ImageURL",
    "ImageURLContentBlock",
    "QueryImage",
    "TextContentBlock",
    "dump_optional_list",
    "model_dump_json_safe",
]
