# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Typed browser route request payloads."""

from uuid import UUID

from fastapi import HTTPException, Request
from pydantic import Field

from dlightrag.core.client_contracts import ClientContractModel

_WEB_ANSWER_JSON_OVERHEAD_BYTES = 64 * 1024


class WebAnswerRequest(ClientContractModel):
    query: str = ""
    images: list[str] = Field(default_factory=list)
    workspaces: list[str] | None = None
    conversation_id: UUID
    submission_id: UUID


async def read_limited_answer_body(
    request: Request, *, max_images: int, max_upload_bytes: int
) -> bytes:
    encoded_image_bytes = ((max_upload_bytes + 2) // 3) * 4
    max_body_bytes = _WEB_ANSWER_JSON_OVERHEAD_BYTES + max(0, max_images) * encoded_image_bytes
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > max_body_bytes:
                raise HTTPException(status_code=413, detail="Web answer request body is too large")
        except ValueError:
            # Malformed Content-Length header: ignore it and rely on the
            # streaming byte cap below to bound the request body size.
            pass
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > max_body_bytes:
            raise HTTPException(status_code=413, detail="Web answer request body is too large")
        body.extend(chunk)
    return bytes(body)


__all__ = ["WebAnswerRequest", "read_limited_answer_body"]
