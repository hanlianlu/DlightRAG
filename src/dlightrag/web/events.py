# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Typed browser-facing SSE event payloads."""

from typing import Any, Literal

from pydantic import Field

from dlightrag.core.client_contracts import ClientContractModel


class AnswerMetaEvent(ClientContractModel):
    history_kept: int


class AnswerProgressEvent(ClientContractModel):
    phase: Literal["planning", "generating"]


class AnswerDoneEvent(ClientContractModel):
    html: str
    answer: str
    current_image_ids: list[str] = Field(default_factory=list)
    image_descriptions: list[str] | dict[str, str] = Field(default_factory=list)
    answer_images: list[dict[str, Any]] = Field(default_factory=list)
    answer_blocks: list[dict[str, Any]] = Field(default_factory=list)
    checkpoint_saved: bool = True


class AnswerTraceEvent(ClientContractModel):
    trace: dict[str, Any]


class AnswerImageMetaEvent(ClientContractModel):
    current_image_ids: list[str] = Field(default_factory=list)
    image_descriptions: list[str] | dict[str, str] = Field(default_factory=list)


class AnswerErrorEvent(ClientContractModel):
    message: str


__all__ = [
    "AnswerDoneEvent",
    "AnswerErrorEvent",
    "AnswerImageMetaEvent",
    "AnswerMetaEvent",
    "AnswerProgressEvent",
    "AnswerTraceEvent",
]
