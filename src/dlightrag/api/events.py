# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""REST API server-sent event payload contracts."""

import json
from typing import Any, Literal

from pydantic import Field

from dlightrag.citations.schemas import SourceReferencePayload
from dlightrag.core.client_contracts import ClientContractModel, model_dump_json_safe


class AnswerContextStreamEvent(ClientContractModel):
    type: Literal["context"] = "context"
    data: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)


class AnswerTokenStreamEvent(ClientContractModel):
    type: Literal["token"] = "token"
    content: str


class AnswerSourcesStreamEvent(ClientContractModel):
    type: Literal["sources"] = "sources"
    data: list[SourceReferencePayload] = Field(default_factory=list)


class AnswerTraceStreamEvent(ClientContractModel):
    type: Literal["trace"] = "trace"
    data: dict[str, Any]


class AnswerImageMetaStreamEvent(ClientContractModel):
    type: Literal["image_meta"] = "image_meta"
    image_descriptions: list[str] | dict[str, str] = Field(default_factory=list)


class AnswerDoneStreamEvent(ClientContractModel):
    type: Literal["done"] = "done"
    answer: str
    answer_images: list[dict[str, Any]] = Field(default_factory=list)
    answer_blocks: list[dict[str, Any]] = Field(default_factory=list)


class AnswerErrorStreamEvent(ClientContractModel):
    type: Literal["error"] = "error"
    message: str
    error_kind: str | None = None


type AnswerStreamEvent = (
    AnswerContextStreamEvent
    | AnswerTokenStreamEvent
    | AnswerSourcesStreamEvent
    | AnswerTraceStreamEvent
    | AnswerImageMetaStreamEvent
    | AnswerDoneStreamEvent
    | AnswerErrorStreamEvent
)


def sse_data_event(payload: AnswerStreamEvent) -> str:
    """Serialize one REST SSE data frame without changing the public wire shape."""
    data = json.dumps(model_dump_json_safe(payload), ensure_ascii=False)
    return f"data: {data}\n\n"


__all__ = [
    "AnswerContextStreamEvent",
    "AnswerDoneStreamEvent",
    "AnswerErrorStreamEvent",
    "AnswerImageMetaStreamEvent",
    "AnswerSourcesStreamEvent",
    "AnswerStreamEvent",
    "AnswerTokenStreamEvent",
    "AnswerTraceStreamEvent",
    "sse_data_event",
]
